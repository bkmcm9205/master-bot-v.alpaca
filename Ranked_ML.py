# app_scanner.py — market scanner + auto-trader (TradersPost + Polygon)
# - Dynamic symbol universe (Polygon reference, or env override)
# - Multiple TFs; volume gate; sentiment gate; position sizing
# - Daily profit target / drawdown guard (+ optional flatten), uses START_EQUITY + realized + unrealized(MTM)
# - Max concurrent positions; per-minute order throttle
# - End-of-day flatten (16:00–16:10 ET) + TP/SL bar-based exit checks
# - Proper TradersPost TP/SL payload shape
# - Allowed exchanges filter + Min-price gate
# - Boot banner for Render logs
import os, time, json, math, hashlib, requests
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
# =============================
# ENV / CONFIG
# =============================
TP_URL = os.getenv("TP_WEBHOOK_URL", "") # TradersPost strategy webhook
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "") # Polygon key
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "10")) # main loop sleep
RUN_ID = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
# Universe / scan config
SCANNER_SYMBOLS = os.getenv("SCANNER_SYMBOLS", "").strip() # comma list to override universe (optional)
SCANNER_MAX_PAGES = int(os.getenv("SCANNER_MAX_PAGES", "1")) # pages of Polygon ref tickers (1000 per page)
SCANNER_MIN_TODAY_VOL = int(os.getenv("SCANNER_MIN_TODAY_VOL", "500000")) # today volume gate per symbol
SCANNER_CONF_THRESHOLD = float(os.getenv("SCANNER_CONF_THRESHOLD", "0.8")) # confidence threshold for ML signals
SCANNER_R_MULTIPLE = float(os.getenv("SCANNER_R_MULTIPLE", "3.0")) # risk-reward ratio for TP
TF_MIN_LIST = [int(x) for x in os.getenv("TF_MIN_LIST", "1,2,3,5,10").split(",") if x.strip()]
# Price / exchange filters
MIN_PRICE = float(os.getenv("MIN_PRICE", "5.0")) # last trade must be >= this
ALLOWED_EXCHANGES = set(
    x.strip().upper()
    for x in os.getenv("ALLOWED_EXCHANGES", "NASD,NASDAQ,NYSE,XNAS,XNYS").split(",")
    if x.strip()
)
# Risk / sizing
EQUITY_USD = float(os.getenv("EQUITY_USD", "100000"))
RISK_PCT = float(os.getenv("RISK_PCT", "0.01")) # 1%
MAX_POS_PCT = float(os.getenv("MAX_POS_PCT", "0.10")) # 10% notional cap
MIN_QTY = int(os.getenv("MIN_QTY", "1"))
ROUND_LOT = int(os.getenv("ROUND_LOT","1"))
# Daily portfolio guard (manual baseline each morning)
START_EQUITY = float(os.getenv("START_EQUITY", "100000")) # <-- YOU update this daily
DAILY_TP_PCT = float(os.getenv("DAILY_TP_PCT", "0.03")) # +3%
DAILY_DD_PCT = float(os.getenv("DAILY_DD_PCT", "0.05")) # -5%
DAILY_FLATTEN_ON_HIT = os.getenv("DAILY_FLATTEN_ON_HIT","1").lower() in ("1","true","yes")
DAILY_GUARD_ENABLED = os.getenv("DAILY_GUARD_ENABLED","1").lower() in ("1","true","yes")
HALT_TRADING = False
DAY_STAMP = datetime.now().astimezone().strftime("%Y-%m-%d")
# Engine guards
MAX_CONCURRENT_POSITIONS = int(os.getenv("MAX_CONCURRENT_POSITIONS", "100"))
MAX_ORDERS_PER_MIN = int(os.getenv("MAX_ORDERS_PER_MIN", "60"))
MARKET_TZ = "America/New_York"
ALLOW_PREMARKET = os.getenv("ALLOW_PREMARKET", "0").lower() in ("1","true","yes")
ALLOW_AFTERHOURS = os.getenv("ALLOW_AFTERHOURS", "0").lower() in ("1","true","yes")
# Sentiment gate
SENTIMENT_LOOKBACK_MIN = int(os.getenv("SENTIMENT_LOOKBACK_MIN", "60"))
SENTIMENT_NEUTRAL_BAND = float(os.getenv("SENTIMENT_NEUTRAL_BAND", "0.0015"))
SENTIMENT_ONLY_GATE = os.getenv("SENTIMENT_ONLY_GATE","1").lower() in ("1","true","yes")
SENTIMENT_SYMBOLS = [s.strip() for s in os.getenv("SENTIMENT_SYMBOLS", "SPY,QQQ").split(",") if s.strip()]
SENTIMENT_NEUTRAL_ACTION = os.getenv("SENTIMENT_NEUTRAL_ACTION", "both").lower()
# Diagnostics / replay
DRY_RUN = os.getenv("DRY_RUN","0") == "1"
# Counters / ledgers
COUNTS = defaultdict(int)
COMBO_COUNTS = defaultdict(int)
PERF = {} # combo -> rolling realized performance
OPEN_TRADES = defaultdict(list) # (symbol, tf) -> [LiveTrade]
_sent_keys = set() # de-dupe
_order_times = deque() # throttle timestamps (epoch seconds)
LAST_PRICE = {} # symbol -> last seen price for MTM
# Render boot info (for logs)
RENDER_GIT_COMMIT = os.getenv("RENDER_GIT_COMMIT", "unknown")[:12]
RENDER_GIT_BRANCH = os.getenv("RENDER_GIT_BRANCH", os.getenv("BRANCH", "unknown"))
# =============================
# Data models
# =============================
class LiveTrade:
    def __init__(self, combo, symbol, tf_min, side, entry, tp, sl, qty, entry_time):
        self.combo = combo
        self.symbol = symbol
        self.tf_min = int(tf_min)
        self.side = side # "buy" or "sell"
        self.entry = float(entry) if entry is not None else float("nan")
        self.tp = float(tp) if tp is not None else float("nan")
        self.sl = float(sl) if sl is not None else float("nan")
        self.qty = int(qty)
        self.entry_time = entry_time
        self.is_open = True
        self.exit = None
        self.exit_time = None
        self.reason = None
# =============================
# Time/session helpers
# =============================
def _now_et():
    return datetime.now(timezone.utc).astimezone(ZoneInfo(MARKET_TZ))
def _is_rth(ts):
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
# =============================
# Math / sizing
# =============================
def _position_qty(entry_price: float, stop_price: float,
                  equity=EQUITY_USD, risk_pct=RISK_PCT, max_pos_pct=MAX_POS_PCT,
                  min_qty=MIN_QTY, round_lot=ROUND_LOT) -> int:
    if entry_price is None or stop_price is None:
        return 0
    risk_per_share = abs(float(entry_price) - float(stop_price))
    if risk_per_share <= 0:
        return 0
    qty_risk = (equity * risk_pct) / risk_per_share
    qty_notional = (equity * max_pos_pct) / max(1e-9, float(entry_price))
    qty = math.floor(max(min(qty_risk, qty_notional), 0) / max(1, round_lot)) * max(1, round_lot)
    return int(max(qty, min_qty if qty > 0 else 0))
# =============================
# Polygon fetchers
# =============================
def fetch_polygon_1m(symbol: str, lookback_minutes: int = 2400) -> pd.DataFrame:
    """Return tz-aware ET 1m bars with o/h/l/c/volume."""
    if not POLYGON_API_KEY:
        print("[ERROR] POLYGON_API_KEY missing in fetch_polygon_1m", flush=True)
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
            print(f"[ERROR] Polygon API failed for {symbol}: {r.status_code}", flush=True)
            return pd.DataFrame()
        rows = r.json().get("results", [])
        if not rows:
            print(f"[WARN] No data from Polygon for {symbol}", flush=True)
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        df = df.set_index("ts").sort_index()
        df.index = df.index.tz_convert(MARKET_TZ)
        df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
        return df[["open","high","low","close","volume"]]
    except Exception as e:
        print(f"[ERROR] fetch_polygon_1m for {symbol}: {e}", flush=True)
        return pd.DataFrame()
def get_universe_symbols() -> list:
    """Env override OR Polygon reference tickers filtered by ALLOWED_EXCHANGES."""
    if SCANNER_SYMBOLS:
        return [s.strip().upper() for s in SCANNER_SYMBOLS.split(",") if s.strip()]
    if not POLYGON_API_KEY:
        print("[ERROR] POLYGON_API_KEY missing in get_universe_symbols", flush=True)
        return []
    out, cursor, pages = [], None, 0
    while pages < SCANNER_MAX_PAGES:
        params = {
            "market":"stocks", "active":"true", "limit":"1000",
            "apiKey": POLYGON_API_KEY
        }
        if cursor:
            params["cursor"] = cursor
        url = "https://api.polygon.io/v3/reference/tickers"
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            print(f"[ERROR] Polygon API tickers failed: {r.status_code}", flush=True)
            break
        j = r.json()
        for x in j.get("results", []):
            exch = (x.get("primary_exchange") or x.get("exchange") or "").upper()
            tkr = x.get("ticker")
            if not tkr:
                continue
            if ALLOWED_EXCHANGES and exch not in ALLOWED_EXCHANGES:
                continue
            out.append(tkr)
        cursor = j.get("next_url") or None
        if not cursor:
            break
        pages += 1
    return [s for s in out if s.isalnum()]
# =============================
# TradersPost webhook I/O
# =============================
def _throttle_ok():
    now = time.time()
    while _order_times and now - _order_times[0] > 60:
        _order_times.popleft()
    if len(_order_times) >= MAX_ORDERS_PER_MIN:
        return False, f"throttled: {len(_order_times)}/{MAX_ORDERS_PER_MIN} in last 60s"
    _order_times.append(now)
    return True, ""
def send_to_traderspost(payload: dict):
    try:
        if DRY_RUN:
            print(f"[DRY-RUN] {json.dumps(payload)[:500]}", flush=True)
            return True, "dry-run"
        if not TP_URL:
            print("[ERROR] TP_WEBHOOK_URL missing.", flush=True)
            return False, "no-webhook-url"
        ok, msg = _throttle_ok()
        if not ok:
            return False, msg
        payload.setdefault("meta", {})
        payload["meta"]["environment"] = "paper"
        payload["meta"]["sentAt"] = datetime.now(timezone.utc).isoformat()
        r = requests.post(TP_URL, json=payload, timeout=12)
        info = f"{r.status_code} {r.text[:300]}"
        return (200 <= r.status_code < 300), info
    except Exception as e:
        print(f"[ERROR] send_to_traderspost: {e}", flush=True)
        return False, f"exception: {e}"
def build_payload(symbol: str, sig: dict):
    """Normalize legacy and absolute TP/SL into TradersPost nested fields."""
    action = sig.get("action")
    order_type = sig.get("orderType","market")
    qty = int(sig.get("quantity", 0))
    payload = {"ticker": symbol, "action": action, "orderType": order_type, "quantity": qty, "meta": {}}
    if isinstance(sig.get("meta"), dict):
        payload["meta"].update(sig["meta"])
    if sig.get("barTime"):
        payload["meta"]["barTime"] = sig["barTime"]
    if order_type.lower() == "limit" and sig.get("price") is not None:
        payload["limitPrice"] = float(round(sig["price"], 2))
    tp_abs = sig.get("tp_abs", sig.get("takeProfit"))
    sl_abs = sig.get("sl_abs", sig.get("stopLoss"))
    if tp_abs is not None:
        payload["takeProfit"] = {"limitPrice": float(round(tp_abs, 2))}
    if sl_abs is not None:
        payload["stopLoss"] = {"type":"stop", "stopPrice": float(round(sl_abs, 2))}
    return payload
# =============================
# Performance tracking
# =============================
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
        if t.side == "buy":
            hit_tp = high >= t.tp
            hit_sl = low <= t.sl
        else:  # sell
            hit_tp = low <= t.tp
            hit_sl = high >= t.sl
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
# =============================
# Sentiment
# =============================
def compute_sentiment():
    """Simple intraday momentum on SENTIMENT_SYMBOLS within SENTIMENT_NEUTRAL_BAND."""
    look_min = max(5, SENTIMENT_LOOKBACK_MIN)
    tf_min = 1
    vals = []
    for s in SENTIMENT_SYMBOLS:
        df = fetch_polygon_1m(s, lookback_minutes=look_min*2)
        if df is None or df.empty:
            print(f"[WARN] No sentiment data for {s}", flush=True)
            continue
        try:
            df.index = df.index.tz_convert(MARKET_TZ)
        except Exception:
            df.index = df.index.tz_localize("UTC").tz_convert(MARKET_TZ)
        # simple momentum over last look_min minutes
        window = df.iloc[-look_min:]
        if len(window) < 2:
            print(f"[WARN] Insufficient data for {s} sentiment", flush=True)
            continue
        vals.append(float(window["close"].iloc[-1]) / float(window["close"].iloc[0]) - 1.0)
    if not vals:
        print("[WARN] No valid sentiment data, defaulting to neutral", flush=True)
        return "neutral"
    avg = sum(vals) / len(vals)
    if avg >= SENTIMENT_NEUTRAL_BAND:
        return "bull"
    if avg <= -SENTIMENT_NEUTRAL_BAND:
        return "bear"
    return "neutral"
# =============================
# Strategy: ML Pattern (live adapter)
# =============================
def _resample(df1m: pd.DataFrame, tf_min: int) -> pd.DataFrame:
    if df1m is None or df1m.empty or not isinstance(df1m.index, pd.DatetimeIndex):
        print("[WARN] Invalid DataFrame for resampling", flush=True)
        return pd.DataFrame()
    rule = f"{int(tf_min)}min"
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    try:
        bars = df1m.resample(rule, origin="start_day", label="right").agg(agg).dropna()
    except Exception as e:
        print(f"[ERROR] Resampling failed: {e}", flush=True)
        return pd.DataFrame()
    return bars
def signal_ml_pattern(symbol: str, df1m: pd.DataFrame, tf_min: int, sentiment: str, conf_threshold=SCANNER_CONF_THRESHOLD, r_multiple=SCANNER_R_MULTIPLE):
    try:
        from sklearn.ensemble import RandomForestClassifier
        import pandas_ta as ta
    except ImportError as e:
        print(f"[ERROR] Failed to import sklearn or pandas_ta: {e}", flush=True)
        return None
    if df1m is None or df1m.empty or not isinstance(df1m.index, pd.DatetimeIndex):
        print(f"[WARN] Invalid data for {symbol} {tf_min}m", flush=True)
        return None
    bars = _resample(df1m, tf_min)
    if bars.empty or len(bars) < 120:
        print(f"[WARN] Insufficient bars for {symbol} {tf_min}m: {len(bars)}", flush=True)
        return None
    bars = bars.copy()
    bars["return"] = bars["close"].pct_change()
    try:
        bars["rsi"] = ta.rsi(bars["close"], length=14)
    except Exception as e:
        print(f"[ERROR] RSI calculation failed for {symbol}: {e}", flush=True)
        delta = bars["close"].diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = -delta.clip(upper=0).rolling(14).mean()
        rs = up / (down.replace(0, np.nan))
        bars["rsi"] = 100 - (100 / (1 + rs))
    bars["volatility"] = bars["close"].rolling(20).std()
    bars.dropna(inplace=True)
    if len(bars) < 60:
        print(f"[WARN] Insufficient data after dropna for {symbol} {tf_min}m: {len(bars)}", flush=True)
        return None
    X = bars[["return","rsi","volatility"]].iloc[:-1]
    y = (bars["close"].shift(-1) > bars["close"]).astype(int).iloc[:-1]
    if len(X) < 50:
        print(f"[WARN] Insufficient training data for {symbol} {tf_min}m: {len(X)}", flush=True)
        return None
    cut = int(len(X) * 0.7)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X.iloc[:cut], y.iloc[:cut])
    x_live = X.iloc[[-1]]
    try:
        x_live = x_live[list(clf.feature_names_in_)]
    except Exception:
        pass
    proba_up = clf.predict_proba(x_live)[0][1]
    proba_down = 1 - proba_up
    ts = bars.index[-1]
    if not _in_session(ts):
        print(f"[INFO] {symbol} {tf_min}m outside trading session", flush=True)
        return None
    price = float(bars["close"].iloc[-1])
    action = None
    confidence = None
    tp = None
    sl = None
    if sentiment == "bull" and proba_up >= conf_threshold:
        action = "buy"
        confidence = proba_up
        sl = price * 0.99  # 1% below
        tp = price * (1 + 0.01 * r_multiple)  # above
    elif sentiment == "bear" and proba_down >= conf_threshold:
        action = "sell"
        confidence = proba_down
        sl = price * 1.01  # 1% above
        tp = price * (1 - 0.01 * r_multiple)  # below
    elif sentiment == "neutral":
        if SENTIMENT_NEUTRAL_ACTION == "both" or SENTIMENT_NEUTRAL_ACTION == "buy":
            if proba_up >= conf_threshold:
                action = "buy"
                confidence = proba_up
                sl = price * 0.99
                tp = price * (1 + 0.01 * r_multiple)
        if (SENTIMENT_NEUTRAL_ACTION == "both" or SENTIMENT_NEUTRAL_ACTION == "sell") and action is None:
            if proba_down >= conf_threshold:
                action = "sell"
                confidence = proba_down
                sl = price * 1.01
                tp = price * (1 - 0.01 * r_multiple)
    if action is None:
        print(f"[INFO] No signal for {symbol} {tf_min}m (proba_up={proba_up:.3f}, proba_down={proba_down:.3f}, sentiment={sentiment})", flush=True)
        return None
    qty = _position_qty(price, sl)
    if qty <= 0:
        print(f"[WARN] Invalid quantity for {symbol} {tf_min}m: qty={qty}", flush=True)
        return None
    print(f"[SIGNAL] {symbol} {tf_min}m {action.upper()} confidence={confidence:.3f}", flush=True)
    return {
        "action": action,
        "orderType": "market",
        "price": None,
        "takeProfit": tp,
        "stopLoss": sl,
        "barTime": ts.tz_convert("UTC").isoformat(),
        "entry": price,
        "quantity": int(qty),
        "meta": {"note": "ml_pattern"},
        "confidence": confidence
    }
# =============================
# Guard helpers (realized + unrealized MTM)
# =============================
def _today_local_date_str():
    return datetime.now().astimezone().strftime("%Y-%m-%d")
def _realized_day_pnl() -> float:
    return sum(float(p.get("net_pnl", 0.0)) for p in PERF.values())
def _unrealized_day_pnl() -> float:
    tot = 0.0
    for trades in OPEN_TRADES.values():
        for t in trades:
            if not t.is_open or not t.qty:
                continue
            px = LAST_PRICE.get(t.symbol)
            if px is None or not (px == px):
                continue
            if t.side == "buy":
                tot += (px - t.entry) * t.qty
            else:
                tot += (t.entry - px) * t.qty
    return float(tot)
def _current_equity() -> float:
    return START_EQUITY + _realized_day_pnl() + _unrealized_day_pnl()
def _opposite(action: str) -> str:
    return "sell" if action == "buy" else "buy"
def flatten_all_open_positions():
    posted = 0
    for (sym, tf), trades in list(OPEN_TRADES.items()):
        for t in trades:
            if not t.is_open or not t.qty or not t.symbol:
                continue
            payload = {
                "ticker": t.symbol,
                "action": _opposite(t.side),
                "orderType": "market",
                "quantity": int(t.qty),
                "meta": {"note": "daily-guard-flatten", "combo": t.combo, "triggeredAt": datetime.now(timezone.utc).isoformat()}
            }
            ok, info = send_to_traderspost(payload)
            print(f"[DAILY-GUARD] Flatten {t.combo} -> ok={ok} info={info}", flush=True)
            posted += 1
            # mark closed locally
            t.is_open = False
            t.exit_time = datetime.now(timezone.utc).isoformat()
            t.exit = t.entry
            t.reason = "daily_guard"
    print(f"[DAILY-GUARD] Flatten requests posted: {posted}", flush=True)
def reset_daily_guard_if_new_day():
    global DAY_STAMP, HALT_TRADING
    today = _today_local_date_str()
    if today != DAY_STAMP:
        HALT_TRADING = False
        DAY_STAMP = today
        print(f"[DAILY-GUARD] New day -> reset HALT_TRADING. START_EQUITY={START_EQUITY:.2f} Day={DAY_STAMP}", flush=True)
def check_daily_guard_and_maybe_halt():
    global HALT_TRADING
    equity = _current_equity()
    up_lim = START_EQUITY * (1.0 + DAILY_TP_PCT)
    dn_lim = START_EQUITY * (1.0 - DAILY_DD_PCT)
    print(f"[DAILY-GUARD] eq={equity:.2f} start={START_EQUITY:.2f} "
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
# =============================
# Routing & signal handling
# =============================
def compute_signal(strategy_name, symbol, tf_minutes, sentiment, df1m=None):
    if df1m is None or getattr(df1m, "empty", True):
        df1m = fetch_polygon_1m(symbol, lookback_minutes=max(240, tf_minutes*240))
        if df1m is None or df1m.empty:
            return None
    if not isinstance(df1m.index, pd.DatetimeIndex):
        try:
            df1m.index = pd.to_datetime(df1m.index, utc=True)
        except Exception:
            print(f"[ERROR] Invalid index for {symbol} {tf_minutes}m", flush=True)
            return None
    try:
        df1m.index = df1m.index.tz_convert(MARKET_TZ)
    except Exception:
        df1m.index = df1m.index.tz_localize("UTC").tz_convert(MARKET_TZ)
    # price & volume gates
    try:
        last_px = float(df1m["close"].iloc[-1])
        LAST_PRICE[symbol] = last_px # update MTM cache
        if last_px < MIN_PRICE:
            print(f"[INFO] {symbol} price {last_px:.2f} below MIN_PRICE {MIN_PRICE}", flush=True)
            return None
    except Exception as e:
        print(f"[ERROR] Price check failed for {symbol}: {e}", flush=True)
        return None
    today_mask = df1m.index.date == df1m.index[-1].date()
    todays_vol = float(df1m.loc[today_mask, "volume"].sum()) if today_mask.any() else 0.0
    if todays_vol < SCANNER_MIN_TODAY_VOL:
        print(f"[INFO] {symbol} volume {todays_vol:.0f} below SCANNER_MIN_TODAY_VOL {SCANNER_MIN_TODAY_VOL}", flush=True)
        return None
    if strategy_name == "ml_pattern":
        return signal_ml_pattern(symbol, df1m, tf_minutes, sentiment)
    return None
def handle_signal(strat_name: str, symbol: str, tf_min: int, sig: dict):
    combo_key = _combo_key(strat_name, symbol, tf_min)
    COUNTS["signals"] += 1
    COMBO_COUNTS[f"{combo_key}::signals"] += 1
    meta = sig.get("meta", {})
    meta["combo"] = combo_key
    meta["timeframe"] = f"{int(tf_min)}m"
    sig["meta"] = meta
    _record_open_trade(strat_name, symbol, tf_min, sig)
    payload = build_payload(symbol, sig)
    ok, info = send_to_traderspost(payload)
    try:
        (COMBO_COUNTS if ok else COMBO_COUNTS)[f"{combo_key}::orders.{'ok' if ok else 'err'}"] += 1
        COUNTS["orders.ok" if ok else "orders.err"] += 1
    except Exception:
        pass
    print(f"[ORDER] {combo_key} -> qty={sig.get('quantity')} ok={ok} info={info}", flush=True)
# =============================
# Main
# =============================
def main():
    # ---- Boot banner ----
    print(f"[BOOT] RUN_ID={RUN_ID} BRANCH={RENDER_GIT_BRANCH} COMMIT={RENDER_GIT_COMMIT} START_CMD=python app.py", flush=True)
    print(f"[BOOT] PAPER_MODE=paper POLL_SECONDS={POLL_SECONDS} TFs={TF_MIN_LIST} CONF_THRESHOLD={SCANNER_CONF_THRESHOLD} R_MULTIPLE={SCANNER_R_MULTIPLE}", flush=True)
    print(f"[BOOT] DAILY_GUARD_ENABLED={int(DAILY_GUARD_ENABLED)} UP={DAILY_TP_PCT:.0%} DOWN={DAILY_DD_PCT:.0%} FLATTEN={int(DAILY_FLATTEN_ON_HIT)} START_EQUITY={START_EQUITY:.2f}", flush=True)
    print(f"[BOOT] SENTIMENT_LOOKBACK_MIN={SENTIMENT_LOOKBACK_MIN} SENTIMENT_ONLY_GATE={int(SENTIMENT_ONLY_GATE)} NEUTRAL_ACTION={SENTIMENT_NEUTRAL_ACTION.upper()}", flush=True)
    if not POLYGON_API_KEY:
        print("[FATAL] POLYGON_API_KEY missing.", flush=True)
        return
    if not TP_URL and not DRY_RUN:
        print("[FATAL] TP_WEBHOOK_URL missing (or set DRY_RUN=1).", flush=True)
        return
    # Universe
    symbols = get_universe_symbols()
    print(f"[UNIVERSE] symbols={len(symbols)} TFs={TF_MIN_LIST} vol_gate={SCANNER_MIN_TODAY_VOL} "
          f"MIN_PRICE={MIN_PRICE} EXCH={sorted(ALLOWED_EXCHANGES)}", flush=True)
    while True:
        loop_start = time.time()
        try:
            reset_daily_guard_if_new_day()
            # Close phase: evaluate exits on last bars for all open trades
            touched = set((k[0], k[1]) for k in OPEN_TRADES.keys())
            for (sym, tf) in touched:
                try:
                    df = fetch_polygon_1m(sym, lookback_minutes=max(60, tf * 12))
                    if df is None or df.empty:
                        continue
                    try:
                        df.index = df.index.tz_convert(MARKET_TZ)
                    except Exception:
                        df.index = df.index.tz_localize("UTC").tz_convert(MARKET_TZ)
                    bars = _resample(df, tf)
                    if bars is None or bars.empty:
                        continue
                    row = bars.iloc[-1]
                    ts = bars.index[-1]
                    LAST_PRICE[sym] = float(row["close"]) # refresh MTM
                    _maybe_close_on_bar(sym, tf, ts, float(row["high"]), float(row["low"]), float(row["close"]))
                except Exception as e:
                    print(f"[CLOSE-PHASE ERROR] {sym} {tf}m: {e}", flush=True)
            # Guard check after exits & price refresh
            if DAILY_GUARD_ENABLED:
                check_daily_guard_and_maybe_halt()
            allow_entries = not (DAILY_GUARD_ENABLED and HALT_TRADING)
            if not allow_entries:
                print("[INFO] Trading halted by daily guard", flush=True)
                time.sleep(POLL_SECONDS)
                continue
            # Max concurrent positions guard
            open_positions = sum(1 for lst in OPEN_TRADES.values() for t in lst if t.is_open)
            if open_positions >= MAX_CONCURRENT_POSITIONS:
                print(f"[LIMIT] Max concurrent positions hit: {open_positions}/{MAX_CONCURRENT_POSITIONS}", flush=True)
                time.sleep(POLL_SECONDS)
                continue
            # Scan & signal phase
            sentiment = compute_sentiment()
            print(f"[SENTIMENT] {sentiment}", flush=True)
            if SENTIMENT_ONLY_GATE and sentiment == "neutral" and SENTIMENT_NEUTRAL_ACTION == "none":
                print("[SENTIMENT] Neutral with no actions allowed.", flush=True)
                time.sleep(POLL_SECONDS)
                continue
            signals_list = []
            for sym in symbols:
                df1m = fetch_polygon_1m(sym, lookback_minutes=max(240, max(TF_MIN_LIST) * 240))
                if df1m is None or df1m.empty:
                    continue
                try:
                    df1m.index = df1m.index.tz_convert(MARKET_TZ)
                except Exception:
                    df1m.index = df1m.index.tz_localize("UTC").tz_convert(MARKET_TZ)
                # gates + MTM cache update happen inside compute_signal
                for tf in TF_MIN_LIST:
                    sig = compute_signal("ml_pattern", sym, tf, sentiment, df1m=df1m)
                    if not sig:
                        continue
                    # De-dupe
                    k = hashlib.sha256(f"ml_pattern|{sym}|{tf}|{sig['action']}|{sig.get('barTime','')}".encode()).hexdigest()
                    if k in _sent_keys:
                        print(f"[INFO] Duplicate signal skipped for {sym} {tf}m", flush=True)
                        continue
                    _sent_keys.add(k)
                    signals_list.append((sym, tf, sig))
            # Sort signals by confidence descending and execute
            signals_list.sort(key=lambda x: x[2]['confidence'], reverse=True)
            for sym, tf, sig in signals_list:
                open_positions = sum(1 for lst in OPEN_TRADES.values() for t in lst if t.is_open)
                if open_positions >= MAX_CONCURRENT_POSITIONS:
                    print(f"[LIMIT] Max concurrent reached during execution: {open_positions}/{MAX_CONCURRENT_POSITIONS}", flush=True)
                    break
                handle_signal("ml_pattern", sym, tf, sig)
            # EOD auto-flatten window (16:00–16:10 ET)
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
