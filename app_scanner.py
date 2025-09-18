# app_scanner.py — market scanner + auto-trader (TradersPost + Polygon)
# - Dynamic symbol universe (Polygon reference, or env override)
# - Multiple TFs; volume gate; sentiment gate; position sizing
# - Daily profit target / drawdown kill switch (+ optional flatten)
# - Max concurrent positions; per-minute order throttle
# - End-of-day flatten
# - Proper TradersPost TP/SL payload shape

import os, time, json, math, hashlib, requests
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

# ------------------------------
# ENV / CONFIG
# ------------------------------
TP_URL              = os.getenv("TP_WEBHOOK_URL", "")            # TradersPost strategy webhook
POLYGON_API_KEY     = os.getenv("POLYGON_API_KEY", "")           # Polygon key
POLL_SECONDS        = int(os.getenv("POLL_SECONDS", "10"))       # main loop sleep
RUN_ID              = datetime.now().astimezone().strftime("%Y-%m-%d")

# Universe / scan config
SCANNER_SYMBOLS           = os.getenv("SCANNER_SYMBOLS", "").strip()  # comma list to override universe (optional)
SCANNER_MAX_PAGES         = int(os.getenv("SCANNER_MAX_PAGES", "1"))  # pages of Polygon ref tickers (1000 per page)
SCANNER_MIN_TODAY_VOL     = int(os.getenv("SCANNER_MIN_TODAY_VOL", "500000"))  # today volume gate per symbol
TF_MIN_LIST               = [int(x) for x in os.getenv("TF_MIN_LIST", "1,2,3,5,10").split(",") if x.strip()]

# Risk / sizing (bit-for-bit with your global wrapper intent)
EQUITY_USD          = float(os.getenv("EQUITY_USD",  "100000"))
RISK_PCT            = float(os.getenv("RISK_PCT",    "0.01"))     # 1%
MAX_POS_PCT         = float(os.getenv("MAX_POS_PCT", "0.10"))     # 10% notional cap
MIN_QTY             = int(os.getenv("MIN_QTY", "1"))
ROUND_LOT           = int(os.getenv("ROUND_LOT","1"))

# --- Daily guard config ---
DAILY_GUARD_ENABLED   = os.getenv("DAILY_GUARD_ENABLED", "1").lower() in ("1","true","yes")
DAILY_TP_PCT          = float(os.getenv("DAILY_TP_PCT", "0.25"))   # +25% lock-in
DAILY_DD_PCT          = float(os.getenv("DAILY_DD_PCT", "0.05"))   # -5% stop day
DAILY_FLATTEN_ON_HIT  = os.getenv("DAILY_FLATTEN_ON_HIT", "1").lower() in ("1","true","yes")

# Use START_EQUITY if provided, otherwise default to your sizing equity
START_EQUITY = float(os.getenv("START_EQUITY", str(EQUITY_USD)))

# Daily state
DAY_STAMP     = datetime.now().astimezone().strftime("%Y-%m-%d")
HALT_TRADING  = False

# Engine guards
MAX_CONCURRENT_POSITIONS  = int(os.getenv("MAX_CONCURRENT_POSITIONS", "100"))  # hard cap
MAX_ORDERS_PER_MIN        = int(os.getenv("MAX_ORDERS_PER_MIN", "60"))         # throttle for TradersPost
MARKET_TZ                 = "America/New_York"
ALLOW_PREMARKET           = os.getenv("ALLOW_PREMARKET", "0").lower() in ("1","true","yes")
ALLOW_AFTERHOURS          = os.getenv("ALLOW_AFTERHOURS", "0").lower() in ("1","true","yes")

# Sentiment gate
SENTIMENT_LOOKBACK_MIN    = int(os.getenv("SENTIMENT_LOOKBACK_MIN", "60"))   # minutes on 1m bars
SENTIMENT_NEUTRAL_BAND    = float(os.getenv("SENTIMENT_NEUTRAL_BAND", "0.0015"))  # +/- drift band
SENTIMENT_ONLY_GATE       = os.getenv("SENTIMENT_ONLY_GATE","1").lower() in ("1","true","yes")  # gate orders by sentiment
SENTIMENT_SYMBOLS         = os.getenv("SENTIMENT_SYMBOLS", "SPY,QQQ").split(",")

# Daily portfolio guard
START_EQUITY              = float(os.getenv("START_EQUITY", "100000"))
DAILY_TP_PCT              = float(os.getenv("DAILY_TP_PCT", "0.03"))  # +3%
DAILY_DD_PCT              = float(os.getenv("DAILY_DD_PCT", "0.05"))  # -5%
DAILY_FLATTEN_ON_HIT      = os.getenv("DAILY_FLATTEN_ON_HIT","1").lower() in ("1","true","yes")
HALT_TRADING              = False
DAY_STAMP                 = datetime.now().astimezone().strftime("%Y-%m-%d")

# Diagnostics / replay
DRY_RUN                   = os.getenv("DRY_RUN","0") == "1"
REPLAY_ON_START           = os.getenv("REPLAY_ON_START","0") == "1"
REPLAY_SYMBOL             = os.getenv("REPLAY_SYMBOL","SPY")
REPLAY_TF                 = int(os.getenv("REPLAY_TF","5"))
REPLAY_HOURS              = int(os.getenv("REPLAY_HOURS","24"))
REPLAY_SEND_ORDERS        = os.getenv("REPLAY_SEND_ORDERS","0").lower() in ("1","true","yes")

# Counters / ledgers
COUNTS        = defaultdict(int)
COMBO_COUNTS  = defaultdict(int)     # combo::<metric>
PERF          = {}                   # combo -> rolling performance
OPEN_TRADES   = defaultdict(list)    # (symbol, tf) -> [LiveTrade]
_sent_keys    = set()                # de-dupe
_order_times  = deque()              # throttle timestamps (epoch seconds)

# ------------------------------
# Data models
# ------------------------------
class LiveTrade:
    def __init__(self, combo, symbol, tf_min, side, entry, tp, sl, qty, entry_time):
        self.combo = combo
        self.symbol = symbol
        self.tf_min = int(tf_min)
        self.side = side            # "buy" or "sell"
        self.entry = float(entry) if entry is not None else float("nan")
        self.tp = float(tp) if tp is not None else float("nan")
        self.sl = float(sl) if sl is not None else float("nan")
        self.qty = int(qty)
        self.entry_time = entry_time
        self.is_open = True
        self.exit = None
        self.exit_time = None
        self.reason = None

# ------------------------------
# Utils
# ------------------------------
def _now_et():
    return datetime.now(timezone.utc).astimezone()

def _is_rth(ts):
    # 9:30<=t<16:00 local exchange time
    h, m = ts.hour, ts.minute
    return ((h > 9) or (h == 9 and m >= 30)) and (h < 16)

def _in_session(ts):
    if _is_rth(ts):
        return True
    if ALLOW_PREMARKET and (4 <= ts.hour < 9 or (ts.hour == 9 and ts.minute < 30)):
        return True
    if ALLOW_AFTERHOURS and (16 <= ts.hour < 20):
        return True
    return False

def _resample(df1m: pd.DataFrame, tf_min: int) -> pd.DataFrame:
    if df1m is None or df1m.empty:
        return pd.DataFrame()
    if not isinstance(df1m.index, pd.DatetimeIndex):
        return pd.DataFrame()
    rule = f"{int(tf_min)}min"
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    try:
        bars = df1m.resample(rule, origin="start_day", label="right").agg(agg).dropna()
    except Exception:
        bars = pd.DataFrame()
    return bars

def _dedupe_key(strategy: str, symbol: str, tf: int, action: str, bar_time: str) -> str:
    raw = f"{strategy}|{symbol}|{tf}|{action}|{bar_time}"
    return hashlib.sha256(raw.encode()).hexdigest()

# Position sizing (your wrapper logic)
def _position_qty(entry_price: float, stop_price: float,
                  equity=EQUITY_USD, risk_pct=RISK_PCT, max_pos_pct=MAX_POS_PCT,
                  min_qty=MIN_QTY, round_lot=ROUND_LOT) -> int:
    if entry_price is None or stop_price is None:
        return 0
    risk_per_share = abs(float(entry_price) - float(stop_price))
    if risk_per_share <= 0:
        return 0
    qty_risk     = (equity * risk_pct) / risk_per_share
    qty_notional = (equity * max_pos_pct) / max(1e-9, float(entry_price))
    qty = math.floor(max(min(qty_risk, qty_notional), 0) / max(1, round_lot)) * max(1, round_lot)
    return int(max(qty, min_qty if qty > 0 else 0))

# ------------------------------
# Polygon fetchers
# ------------------------------
def fetch_polygon_1m(symbol: str, lookback_minutes: int = 2400) -> pd.DataFrame:
    """Return tz-aware ET 1m bars with o/h/l/c/volume."""
    if not POLYGON_API_KEY:
        return pd.DataFrame()
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=lookback_minutes)
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/"
        f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"
    )
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return pd.DataFrame()
        rows = r.json().get("results", [])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        df = df.set_index("ts").sort_index()
        df.index = df.index.tz_convert(MARKET_TZ)
        df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
        return df[["open","high","low","close","volume"]]
    except Exception:
        return pd.DataFrame()

def get_universe_symbols() -> list:
    """Universe: env override OR Polygon reference tickers (active US stocks)."""
    if SCANNER_SYMBOLS:
        return [s.strip().upper() for s in SCANNER_SYMBOLS.split(",") if s.strip()]
    if not POLYGON_API_KEY:
        return []
    out = []
    page_token = None
    pages = 0
    while pages < SCANNER_MAX_PAGES:
        params = {
            "market":"stocks",
            "active":"true",
            "limit":"1000",
            "apiKey": POLYGON_API_KEY
        }
        if page_token:
            params["cursor"] = page_token
        url = "https://api.polygon.io/v3/reference/tickers"
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            break
        j = r.json()
        results = j.get("results", [])
        out.extend([x["ticker"] for x in results if x.get("ticker")])
        page_token = j.get("next_url", None)
        if not page_token:
            break
        pages += 1
    # Basic hygiene: drop weird symbols
    out = [s for s in out if s.isalnum()]
    return out

# ------------------------------
# TradersPost I/O
# ------------------------------
def send_to_traderspost(payload: dict):
    """POST order payload to TradersPost or DRY-RUN."""
    try:
        if DRY_RUN:
            print(f"[DRY-RUN] {json.dumps(payload)[:500]}", flush=True)
            return True, "dry-run"
        if not TP_URL:
            print("[ERROR] TP_WEBHOOK_URL missing.", flush=True)
            return False, "no-webhook-url"

        # throttle: max N per minute
        now = time.time()
        while _order_times and now - _order_times[0] > 60:
            _order_times.popleft()
        if len(_order_times) >= MAX_ORDERS_PER_MIN:
            return False, f"throttled: {len(_order_times)}/{MAX_ORDERS_PER_MIN} in last 60s"
        _order_times.append(now)

        # annotate meta
        payload.setdefault("meta", {})
        payload["meta"]["environment"] = "paper"
        payload["meta"]["sentAt"] = datetime.now(timezone.utc).isoformat()

        r = requests.post(TP_URL, json=payload, timeout=12)
        ok = 200 <= r.status_code < 300
        info = f"{r.status_code} {r.text[:300]}"
        if not ok:
            print(f"[POST ERROR] {info}", flush=True)
        return ok, info
    except Exception as e:
        return False, f"exception: {e}"

def build_payload(symbol: str, sig: dict):
    """
    Accepts either:
      - sig['tp_abs'] / sig['sl_abs']  (absolute)
      - or legacy sig['takeProfit'] / sig['stopLoss'] (absolute)
    and shapes TradersPost nested fields.
    """
    action     = sig.get("action")
    order_type = sig.get("orderType","market")
    qty        = int(sig.get("quantity", 0))

    payload = {
        "ticker": symbol,
        "action": action,
        "orderType": order_type,
        "quantity": qty,
        "meta": {}
    }
    if isinstance(sig.get("meta"), dict):
        payload["meta"].update(sig["meta"])
    if sig.get("barTime"):
        payload["meta"]["barTime"] = sig["barTime"]

    # map limit if provided
    if order_type.lower() == "limit" and sig.get("price") is not None:
        payload["limitPrice"] = float(round(sig["price"], 2))

    tp_abs = sig.get("tp_abs", sig.get("takeProfit"))
    sl_abs = sig.get("sl_abs", sig.get("stopLoss"))

    if tp_abs is not None:
        payload["takeProfit"] = {"limitPrice": float(round(tp_abs, 2))}
    if sl_abs is not None:
        payload["stopLoss"] = {"type":"stop", "stopPrice": float(round(sl_abs, 2))}
    return payload

# ------------------------------
# Performance & ledger
# ------------------------------
def _combo_key(strategy: str, symbol: str, tf_min: int) -> str:
    return f"{strategy}|{symbol}|{int(tf_min)}"

def _perf_init(combo: str):
    if combo not in PERF:
        PERF[combo] = {
            "trades": 0, "wins": 0, "losses": 0,
            "gross_profit": 0.0, "gross_loss": 0.0,
            "net_pnl": 0.0, "max_dd": 0.0,
            "equity_curve": [0.0],
        }

def _perf_update(combo: str, pnl: float):
    _perf_init(combo)
    p = PERF[combo]
    p["trades"] += 1
    if pnl > 0:
        p["wins"] += 1
        p["gross_profit"] += pnl
    elif pnl < 0:
        p["losses"] += 1
        p["gross_loss"] += pnl
    p["net_pnl"] += pnl
    ec = p["equity_curve"]
    ec.append(ec[-1] + pnl)
    dd = min(0.0, ec[-1] - max(ec))
    p["max_dd"] = min(p["max_dd"], dd)

def _record_open_trade(strat_name: str, symbol: str, tf_min: int, sig: dict):
    combo = _combo_key(strat_name, symbol, tf_min)
    _perf_init(combo)
    tp = sig.get("tp_abs", sig.get("takeProfit"))
    sl = sig.get("sl_abs", sig.get("stopLoss"))
    t = LiveTrade(
        combo=combo,
        symbol=symbol,
        tf_min=int(tf_min),
        side=sig["action"],
        entry=float(sig.get("entry") or sig.get("price") or sig.get("lastClose") or 0.0),
        tp=float(tp) if tp is not None else float("nan"),
        sl=float(sl) if sl is not None else float("nan"),
        qty=int(sig["quantity"]),
        entry_time=sig.get("barTime") or datetime.now(timezone.utc).isoformat(),
    )
    OPEN_TRADES[(symbol, int(tf_min))].append(t)

def _maybe_close_on_bar(symbol: str, tf_min: int, ts, high: float, low: float, close: float):
    key = (symbol, int(tf_min))
    trades = OPEN_TRADES.get(key, [])
    for t in trades:
        if not t.is_open:
            continue
        hit_tp = (high >= t.tp) if t.side == "buy" else (low <= t.tp)
        hit_sl = (low <= t.sl) if t.side == "buy" else (high >= t.sl)
        if hit_tp or hit_sl:
            t.is_open = False
            t.exit_time = ts.tz_convert("UTC").isoformat() if hasattr(ts, "tzinfo") else str(ts)
            if hit_tp:
                t.exit = t.tp
                t.reason = "tp"
            else:
                t.exit = t.sl
                t.reason = "sl"
            pnl = (t.exit - t.entry) * t.qty if t.side == "buy" else (t.entry - t.exit) * t.qty
            _perf_update(t.combo, pnl)
            print(f"[CLOSE] {t.combo} {t.reason.upper()} qty={t.qty} entry={t.entry:.2f} exit={t.exit:.2f} pnl={pnl:+.2f}", flush=True)

# ------------------------------
# Sentiment gate (SPY/QQQ drift)
# ------------------------------
def compute_sentiment() -> str:
    """Return 'bull', 'bear', or 'neutral' from SPY/QQQ 1m drift over SENTIMENT_LOOKBACK_MIN."""
    rets = []
    for sym in [s.strip().upper() for s in SENTIMENT_SYMBOLS if s.strip()]:
        df = fetch_polygon_1m(sym, lookback_minutes=max(SENTIMENT_LOOKBACK_MIN+5, 90))
        if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
            continue
        c0 = float(df["close"].iloc[-SENTIMENT_LOOKBACK_MIN]) if len(df) > SENTIMENT_LOOKBACK_MIN else float(df["close"].iloc[0])
        c1 = float(df["close"].iloc[-1])
        drift = (c1 - c0) / max(1e-9, c0)
        rets.append(drift)
    if not rets:
        return "neutral"
    avg = float(np.mean(rets))
    if avg > SENTIMENT_NEUTRAL_BAND:
        return "bull"
    if avg < -SENTIMENT_NEUTRAL_BAND:
        return "bear"
    return "neutral"

# ------------------------------
# Strategy: ML Pattern (live adapter)
# ------------------------------
def signal_ml_pattern(symbol: str, df1m: pd.DataFrame, tf_min: int, conf_threshold=0.8, r_multiple=3.0):
    """
    Lightweight live adapter that mirrors your backtest intent:
    - Resample to tf; build simple features; train RF on historical slice; predict last bar.
    - Long-only by default. Volume gate handled by caller via today's volume check.
    NOTE: This keeps model in-process (no persistence) for simplicity.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        import pandas_ta as ta
    except Exception:
        # If packages missing, skip
        return None

    if df1m is None or df1m.empty or not isinstance(df1m.index, pd.DatetimeIndex):
        return None

    bars = _resample(df1m, tf_min)
    if bars.empty or len(bars) < 120:
        return None

    # Features
    bars = bars.copy()
    bars["return"]     = bars["close"].pct_change()
    bars["rsi"]        = ta.rsi(bars["close"], length=14)
    bars["volatility"] = bars["close"].rolling(20).std()
    bars.dropna(inplace=True)
    if len(bars) < 60:
        return None

    X = bars[["return","rsi","volatility"]].copy()
    y = (bars["close"].shift(-1) > bars["close"]).astype(int)
    X = X.iloc[:-1]   # leave last row for inference
    y = y.iloc[:-1]

    if len(X) < 50:
        return None

    cut = int(len(X) * 0.7)
    X_train, y_train = X.iloc[:cut], y.iloc[:cut]
    # (We don’t need X_test here; we only score the latest bar)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # --- FIX: preserve feature names & alignment ---
    x_live = X.iloc[[-1]]  # DataFrame with same columns
    # If sklearn stored feature order, enforce it:
    try:
        feat_cols = list(clf.feature_names_in_)
        x_live = x_live[feat_cols]
    except Exception:
        pass

    proba = clf.predict_proba(x_live)[0][1]   # probability of “up”
    pred  = int(proba >= 0.5)

    ts = bars.index[-1]
    if not _in_session(ts):
        return None

    # Entry rule (long-only here; sentiment gate will filter later)
    if pred == 1 and proba >= conf_threshold:
        price = float(bars["close"].iloc[-1])
        sl    = price * 0.99                         # 1% stop
        tp    = price * (1 + 0.01 * r_multiple)      # r-multiple on 1% unit
        qty   = _position_qty(price, sl)
        if qty <= 0:
            return None
        return {
            "action": "buy",
            "orderType": "market",
            "price": None,
            "takeProfit": tp,
            "stopLoss": sl,
            "barTime": ts.tz_convert("UTC").isoformat(),
            "entry": price,
            "quantity": int(qty),
            "meta": {"note": "ml_pattern"}
        }
    return None

# ------------------------------
# Daily guard
# ------------------------------
def _today_local_date_str():
    return datetime.now().astimezone().strftime("%Y-%m-%d")

def _realized_day_pnl() -> float:
    total = 0.0
    for _, p in PERF.items():
        total += float(p.get("net_pnl", 0.0))
    return total

def _opposite(action: str) -> str:
    return "sell" if action == "buy" else "buy"

def flatten_all_open_positions():
    posted = 0
    for (sym, tf), trades in list(OPEN_TRADES.items()):
        for t in trades:
            if not t.is_open or not t.qty or not t.symbol:
                continue
            exit_action = _opposite(t.side)
            payload = {
                "ticker": t.symbol,
                "action": exit_action,
                "orderType": "market",
                "quantity": int(t.qty),
                "meta": {
                    "note": "daily-guard-flatten",
                    "combo": t.combo,
                    "triggeredAt": datetime.now(timezone.utc).isoformat()
                }
            }
            ok, info = send_to_traderspost(payload)
            print(f"[DAILY-GUARD] Flatten {t.combo} -> ok={ok} info={info}", flush=True)
            posted += 1
            try:
                t.is_open = False
                t.exit_time = datetime.now(timezone.utc).isoformat()
                t.exit = t.entry
                t.reason = "daily_guard"
            except Exception:
                pass
    print(f"[DAILY-GUARD] Flatten requests posted: {posted}", flush=True)

def reset_daily_guard_if_new_day():
    global DAY_STAMP, HALT_TRADING
    today = _today_local_date_str()
    if today != DAY_STAMP:
        HALT_TRADING = False
        DAY_STAMP = today
        print(f"[DAILY-GUARD] New day -> reset HALT_TRADING. Day={DAY_STAMP}", flush=True)

def check_daily_guard_and_maybe_halt():
    global HALT_TRADING
    realized = _realized_day_pnl()
    equity   = START_EQUITY + realized
    up_lim   = START_EQUITY * (1.0 + DAILY_TP_PCT)
    dn_lim   = START_EQUITY * (1.0 - DAILY_DD_PCT)

    print(f"[DAILY-GUARD] eq={equity:.2f} start={START_EQUITY:.2f} realized={realized:+.2f} "
          f"targets +{DAILY_TP_PCT*100:.1f}%({up_lim:.2f}) / -{DAILY_DD_PCT*100:.1f}%({dn_lim:.2f})",
          flush=True)

    if HALT_TRADING:
        return

    if equity >= up_lim:
        HALT_TRADING = True
        print("[DAILY-GUARD] ✅ Daily TP hit. Halting entries.", flush=True)
        if DAILY_FLATTEN_ON_HIT:
            flatten_all_open_positions()
    elif equity <= dn_lim:
        HALT_TRADING = True
        print("[DAILY-GUARD] ⛔ Daily DD hit. Halting entries.", flush=True)
        if DAILY_FLATTEN_ON_HIT:
            flatten_all_open_positions()

# ------------------------------
# Router (scanner)
# ------------------------------
def compute_signal(strategy_name, symbol, tf_minutes, df1m=None):
    # fetch only if not provided or empty
    if df1m is None or getattr(df1m, "empty", True):
        df1m = fetch_polygon_1m(symbol, lookback_minutes=max(240, tf_minutes*240))
        if df1m is None or df1m.empty:
            return None

    # ensure tz is ET and index is datetime
    if not isinstance(df1m.index, pd.DatetimeIndex):
        try:
            df1m.index = pd.to_datetime(df1m.index, utc=True)
        except Exception:
            return None
    try:
        df1m.index = df1m.index.tz_convert(MARKET_TZ)
    except Exception:
        df1m.index = df1m.index.tz_localize("UTC").tz_convert(MARKET_TZ)

    # today's cumulative volume gate
    try:
        last_day = df1m.index[-1].date()
        todays = df1m.loc[df1m.index.date == last_day]
        todays_vol = float(todays["volume"].sum()) if not todays.empty else 0.0
    except Exception:
        todays_vol = 0.0
    if todays_vol < SCANNER_MIN_TODAY_VOL:
        return None

    if strategy_name == "ml_pattern":
        return signal_ml_pattern(symbol, df1m, tf_minutes)
    return None

# ------------------------------
# Replay (optional)
# ------------------------------
def replay_signals_once():
    if not REPLAY_ON_START:
        return
    df1m = fetch_polygon_1m(REPLAY_SYMBOL, lookback_minutes=REPLAY_HOURS*60)
    if df1m is None or df1m.empty:
        print("[REPLAY] No data.", flush=True)
        return
    try:
        df1m.index = df1m.index.tz_convert(MARKET_TZ)
    except Exception:
        df1m.index = df1m.index.tz_localize("UTC").tz_convert(MARKET_TZ)
    bars = _resample(df1m, REPLAY_TF)
    if bars.empty:
        print("[REPLAY] Bars empty.", flush=True)
        return
    hits = 0
    last_key = None
    for i in range(len(bars)):
        df_slice = df1m.loc[:bars.index[i]]
        sig = signal_ml_pattern(REPLAY_SYMBOL, df_slice, REPLAY_TF)
        if sig:
            if REPLAY_SEND_ORDERS:
                handle_signal("ml_pattern", REPLAY_SYMBOL, REPLAY_TF, sig)
            k = (sig.get("barTime",""), sig.get("action",""))
            if k != last_key:
                hits += 1
                last_key = k
    print(f"[REPLAY] ml_pattern {REPLAY_SYMBOL} {REPLAY_TF}m -> {hits} signals in {REPLAY_HOURS}h.", flush=True)

# ------------------------------
# Signal handling
# ------------------------------
def handle_signal(strat_name: str, symbol: str, tf_min: int, sig: dict):
    combo_key = _combo_key(strat_name, symbol, tf_min)
    COUNTS["signals"] += 1
    COMBO_COUNTS[f"{combo_key}::signals"] += 1

    meta = sig.get("meta", {})
    meta["combo"] = combo_key
    meta["timeframe"] = f"{int(tf_min)}m"
    sig["meta"] = meta

    # Record in ledger first
    _record_open_trade(strat_name, symbol, tf_min, sig)

    payload = build_payload(symbol, sig)
    ok, info = send_to_traderspost(payload)
    try:
        if ok:
            COUNTS["orders.ok"] += 1
            COMBO_COUNTS[f"{combo_key}::orders.ok"] += 1
        else:
            COUNTS["orders.err"] += 1
            COMBO_COUNTS[f"{combo_key}::orders.err"] += 1
    except Exception:
        pass
    print(f"[ORDER] {combo_key} -> qty={sig.get('quantity')} ok={ok} info={info}", flush=True)

# ------------------------------
# Main
# ------------------------------
def main():
    print("Scanner starting…", flush=True)

    # Universe
    symbols = get_universe_symbols()
    print(f"[UNIVERSE] symbols={len(symbols)}  TFs={TF_MIN_LIST}  vol_gate={SCANNER_MIN_TODAY_VOL}", flush=True)

    replay_signals_once()

    while True:
        loop_start = time.time()
        try:
            print("Tick…", flush=True)

            # Sentiment (cache once per loop)
            sentiment = compute_sentiment() if SENTIMENT_ONLY_GATE else "neutral"
            print(f"[SENTIMENT] {sentiment}", flush=True)

            # ---- Daily guard housekeeping ----
            reset_daily_guard_if_new_day()
            if DAILY_GUARD_ENABLED:
                check_daily_guard_and_maybe_halt()

            # Decide if we allow NEW entries this loop
            ALLOW_ENTRIES = not (DAILY_GUARD_ENABLED and HALT_TRADING)

            # ---- Close phase: evaluate exits on last bars for all open trades ----
            touched = set((k[0], k[1]) for k in OPEN_TRADES.keys())
            for (sym, tf) in touched:
                try:
                    df = fetch_polygon_1m(sym, lookback_minutes=max(60, tf * 12))
                    bars = _resample(df, tf)
                    if bars is not None and not bars.empty:
                        row = bars.iloc[-1]
                        ts  = bars.index[-1]
                        _maybe_close_on_bar(sym, tf, ts,
                                            float(row["high"]), float(row["low"]), float(row["close"]))
                except Exception as e:
                    print(f"[CLOSE-PHASE ERROR] {sym} {tf}m: {e}", flush=True)

            # If entries are halted, skip scanning for NEW signals
            if not ALLOW_ENTRIES:
                time.sleep(POLL_SECONDS)
                continue

            # Max concurrent positions guard
            open_positions = sum(1 for lst in OPEN_TRADES.values() for t in lst if t.is_open)
            if open_positions >= MAX_CONCURRENT_POSITIONS:
                print(f"[LIMIT] Max concurrent positions hit: {open_positions}/{MAX_CONCURRENT_POSITIONS}", flush=True)
                time.sleep(POLL_SECONDS)
                continue

            # ---- Scan symbols & TFs ----
            for sym in symbols:
                # Optional: per-symbol concurrent cap example (currently passive)
                # if sum(1 for t in OPEN_TRADES.get((sym, TF_MIN_LIST[0]), []) if t.is_open) > 0:
                #     pass

                df1m = fetch_polygon_1m(sym, lookback_minutes=max(240, max(TF_MIN_LIST) * 240))
                if df1m is None or df1m.empty or not isinstance(df1m.index, pd.DatetimeIndex):
                    continue
                try:
                    df1m.index = df1m.index.tz_convert(MARKET_TZ)
                except Exception:
                    df1m.index = df1m.index.tz_localize("UTC").tz_convert(MARKET_TZ)

                # Today volume gate (per symbol)
                today_mask = df1m.index.date == df1m.index[-1].date()
                todays_vol = float(df1m.loc[today_mask, "volume"].sum()) if today_mask.any() else 0.0
                if todays_vol < SCANNER_MIN_TODAY_VOL:
                    continue

                for tf in TF_MIN_LIST:
                    # Produce a candidate signal
                    sig = compute_signal("ml_pattern", sym, tf, df1m=df1m)
                    if not sig:
                        continue

                    # Sentiment gate
                    if SENTIMENT_ONLY_GATE:
                        if sentiment == "bull" and sig["action"] != "buy":
                            continue
                        if sentiment == "bear" and sig["action"] != "sell":
                            continue

                    # De-dupe by combo/action/time
                    k = _dedupe_key("ml_pattern", sym, tf, sig["action"], sig.get("barTime", ""))
                    if k in _sent_keys:
                        continue
                    _sent_keys.add(k)

                    # Just-in-time concurrent cap
                    open_positions = sum(1 for lst in OPEN_TRADES.values() for t in lst if t.is_open)
                    if open_positions >= MAX_CONCURRENT_POSITIONS:
                        print(f"[LIMIT] Max concurrent reached mid-loop.", flush=True)
                        break

                    # Record + send
                    handle_signal("ml_pattern", sym, tf, sig)

            # EOD auto-flatten window (16:00–16:10 local)
            now_et = _now_et()
            if now_et.hour == 16 and now_et.minute < 10:
                print("[EOD] Auto-flatten window.", flush=True)
                flatten_all_open_positions()

        except Exception as e:
            import traceback
            print("[LOOP ERROR]", e, traceback.format_exc(), flush=True)

        elapsed = time.time() - loop_start
        time.sleep(max(1, POLL_SECONDS - int(elapsed)))

if __name__ == "__main__":
    main()

