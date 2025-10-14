# Ranked_ML.py — market scanner + auto-trader (PLUMBING SWAP: Polygon/TP -> Alpaca)

import os, time, json, math, hashlib
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

# ---- Alpaca adapters (data + broker) ----
from adapters.data_alpaca import fetch_1m as _alpaca_fetch_1m, get_universe_symbols as _alpaca_universe
from common.signal_bridge import (
    send_to_broker,
    close_all_positions,
    list_positions,
    get_account_equity,
    cancel_all_orders,
    probe_alpaca_auth,
)

# =============================
# ENV / CONFIG
# =============================
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "10"))
RUN_ID = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")

# Universe / scan config
SCANNER_SYMBOLS = os.getenv("SCANNER_SYMBOLS", "").strip()
SCANNER_MAX_PAGES = int(os.getenv("SCANNER_MAX_PAGES", "20"))  # compat only
SCANNER_MIN_TODAY_VOL = int(os.getenv("SCANNER_MIN_TODAY_VOL", "20000"))
SCANNER_CONF_THRESHOLD = float(os.getenv("SCANNER_CONF_THRESHOLD", "0.7"))
SCANNER_R_MULTIPLE = float(os.getenv("SCANNER_R_MULTIPLE", "1.5"))
TF_MIN_LIST = [int(x) for x in os.getenv("TF_MIN_LIST", "1,2,3,5,10").split(",") if x.strip()]

# Price / exchange filters
MIN_PRICE = float(os.getenv("MIN_PRICE", "3.0"))
ALLOWED_EXCHANGES = set(
    x.strip().upper()
    for x in os.getenv("ALLOWED_EXCHANGES", "NASD,NASDAQ,NYSE,XNAS,XNYS").split(",")
    if x.strip()
)

# Risk / sizing
EQUITY_USD = float(os.getenv("EQUITY_USD", "100000"))
RISK_PCT = float(os.getenv("RISK_PCT", "0.01"))
MAX_POS_PCT = float(os.getenv("MAX_POS_PCT", "0.10"))
MIN_QTY = int(os.getenv("MIN_QTY", "1"))
ROUND_LOT = int(os.getenv("ROUND_LOT","1"))

# Daily guard (local MTM baseline)
START_EQUITY = float(os.getenv("START_EQUITY", "100000"))
DAILY_TP_PCT = float(os.getenv("DAILY_TP_PCT", "0.25"))
DAILY_DD_PCT = float(os.getenv("DAILY_DD_PCT", "0.05"))
DAILY_FLATTEN_ON_HIT = os.getenv("DAILY_FLATTEN_ON_HIT","1").lower() in ("1","true","yes")
DAILY_GUARD_ENABLED = os.getenv("DAILY_GUARD_ENABLED","1").lower() in ("1","true","yes")
HALT_TRADING = False
DAY_STAMP = datetime.now().astimezone().strftime("%Y-%m-%d")

# --- Broker-equity guard (optional; match other apps)
USE_BROKER_EQUITY_GUARD   = os.getenv("USE_BROKER_EQUITY_GUARD","0").lower() in ("1","true","yes")
SESSION_BASELINE_AT_0930  = os.getenv("SESSION_BASELINE_AT_0930","1").lower() in ("1","true","yes")
TRAIL_GUARD_ENABLED       = os.getenv("TRAIL_GUARD_ENABLED","1").lower() in ("1","true","yes")
TRAIL_DD_PCT              = float(os.getenv("TRAIL_DD_PCT","0.05"))
SESSION_START_EQUITY      = None
EQUITY_BASELINE_SET       = False
EQUITY_BASELINE_DATE      = None
EQUITY_HIGH_WATER         = None

# Engine guards
MAX_CONCURRENT_POSITIONS = int(os.getenv("MAX_CONCURRENT_POSITIONS", "200"))
MAX_ORDERS_PER_MIN = int(os.getenv("MAX_ORDERS_PER_MIN", "60"))
MARKET_TZ = "America/New_York"
ALLOW_PREMARKET = os.getenv("ALLOW_PREMARKET", "0").lower() in ("1","true","yes")
ALLOW_AFTERHOURS = os.getenv("ALLOW_AFTERHOURS", "0").lower() in ("1","true","yes")

# Sentiment gate
SENTIMENT_LOOKBACK_MIN = int(os.getenv("SENTIMENT_LOOKBACK_MIN", "5"))
SENTIMENT_NEUTRAL_BAND = float(os.getenv("SENTIMENT_NEUTRAL_BAND", "0.0015"))
SENTIMENT_ONLY_GATE = os.getenv("SENTIMENT_ONLY_GATE","1").lower() in ("1","true","yes")
SENTIMENT_SYMBOLS = [s.strip() for s in os.getenv("SENTIMENT_SYMBOLS", "SPY,QQQ").split(",") if s.strip()]
SENTIMENT_NEUTRAL_ACTION = os.getenv("SENTIMENT_NEUTRAL_ACTION", "both").lower()  # both|buy|sell|none

# Diagnostics
DRY_RUN = os.getenv("DRY_RUN","0") == "1"

# Counters / ledgers
COUNTS = defaultdict(int)
COMBO_COUNTS = defaultdict(int)
PERF = {}
OPEN_TRADES = defaultdict(list)
_sent_keys = set()
_order_times = deque()
LAST_PRICE = {}

# Render boot info (for logs)
RENDER_GIT_COMMIT = os.getenv("RENDER_GIT_COMMIT", "unknown")[:12]
RENDER_GIT_BRANCH = os.getenv("RENDER_GIT_BRANCH", os.getenv("BRANCH", "unknown"))

# =============================
# Data model
# =============================
class LiveTrade:
    def __init__(self, combo, symbol, tf_min, side, entry, tp, sl, qty, entry_time):
        self.combo = combo
        self.symbol = symbol
        self.tf_min = int(tf_min)
        self.side = side  # "buy" or "sell"
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
    if _is_rth(ts): return True
    if ALLOW_PREMARKET and (4 <= ts.hour < 9 or (ts.hour == 9 and ts.minute < 30)): return True
    if ALLOW_AFTERHOURS and (16 <= ts.hour < 20): return True
    return False

# =============================
# Math / sizing
# =============================
def _position_qty(entry_price: float, stop_price: float,
                  equity=EQUITY_USD, risk_pct=RISK_PCT, max_pos_pct=MAX_POS_PCT,
                  min_qty=MIN_QTY, round_lot=ROUND_LOT) -> int:
    if entry_price is None or stop_price is None:
        return 0
    rps = abs(float(entry_price) - float(stop_price))
    if rps <= 0:
        return 0
    qty_risk = (equity * risk_pct) / rps
    qty_notional = (equity * max_pos_pct) / max(1e-9, float(entry_price))
    qty = math.floor(max(min(qty_risk, qty_notional), 0) / max(1, round_lot)) * max(1, round_lot)
    return int(max(qty, min_qty if qty > 0 else 0))

# =============================
# Alpaca data wrappers
# =============================
def fetch_bars_1m(symbol: str, lookback_minutes: int = 2400) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=lookback_minutes)
    start_iso = start.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    end_iso = end.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    df = _alpaca_fetch_1m(symbol, start_iso=start_iso, end_iso=end_iso, limit=10000)
    if df is None or df.empty:
        return pd.DataFrame()
    try: df.index = df.index.tz_convert(MARKET_TZ)
    except Exception: df.index = df.index.tz_localize("UTC").tz_convert(MARKET_TZ)
    cols = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    return df[cols].copy() if cols else pd.DataFrame()

def get_universe_symbols() -> list:
    if SCANNER_SYMBOLS:
        return [s.strip().upper() for s in SCANNER_SYMBOLS.split(",") if s.strip()]
    syms = _alpaca_universe(limit=10000)
    return [s for s in syms if isinstance(s, str) and s]

# =============================
# Sentiment
# =============================
def compute_sentiment():
    look_min = max(5, SENTIMENT_LOOKBACK_MIN)
    vals = []
    for s in SENTIMENT_SYMBOLS:
        df = fetch_bars_1m(s, lookback_minutes=look_min*2)
        if df is None or df.empty: continue
        try: df.index = df.index.tz_convert(MARKET_TZ)
        except Exception: df.index = df.index.tz_localize("UTC").tz_convert(MARKET_TZ)
        win = df.iloc[-look_min:]
        if len(win) < 2: continue
        vals.append(float(win["close"].iloc[-1]) / float(win["close"].iloc[0]) - 1.0)
    if not vals: return "neutral"
    avg = sum(vals)/len(vals)
    if avg >= SENTIMENT_NEUTRAL_BAND: return "bull"
    if avg <= -SENTIMENT_NEUTRAL_BAND: return "bear"
    return "neutral"

# =============================
# Strategy
# =============================
def _resample(df1m: pd.DataFrame, tf_min: int) -> pd.DataFrame:
    if df1m is None or df1m.empty or not isinstance(df1m.index, pd.DatetimeIndex):
        return pd.DataFrame()
    rule = f"{int(tf_min)}min"
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    try:
        return df1m.resample(rule, origin="start_day", label="right").agg(agg).dropna()
    except Exception:
        return pd.DataFrame()

def signal_ml_pattern(symbol: str, df1m: pd.DataFrame, tf_min: int, sentiment: str,
                      conf_threshold=SCANNER_CONF_THRESHOLD, r_multiple=SCANNER_R_MULTIPLE):
    try:
        from sklearn.ensemble import RandomForestClassifier
        import pandas_ta as ta
    except Exception as e:
        print(f"[ERROR] ML libs missing: {e}", flush=True); return None

    if df1m is None or df1m.empty or not isinstance(df1m.index, pd.DatetimeIndex):
        return None
    bars = _resample(df1m, tf_min)
    if bars.empty or len(bars) < 120:
        return None

    df = bars.copy()
    df["return"] = df["close"].pct_change()
    try: df["rsi"] = ta.rsi(df["close"], length=14)
    except Exception:
        delta = df["close"].diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = -delta.clip(upper=0).rolling(14).mean()
        rs = up / down.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
    df["volatility"] = df["close"].rolling(20).std()
    df.dropna(inplace=True)
    if len(df) < 60: return None

    X = df[["return","rsi","volatility"]].iloc[:-1]
    y = (df["close"].shift(-1) > df["close"]).astype(int).iloc[:-1]
    if len(X) < 50: return None

    cut = int(len(X)*0.7)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X.iloc[:cut], y.iloc[:cut])
    x_live = X.iloc[[-1]]
    try: x_live = x_live[list(clf.feature_names_in_)]
    except Exception: pass
    proba_up = float(clf.predict_proba(x_live)[0][1]); proba_down = 1.0 - proba_up
    ts = df.index[-1]
    if not _in_session(ts): return None

    price = float(df["close"].iloc[-1])
    action = None; confidence = None; tp = None; sl = None
    if sentiment == "bull" and proba_up >= conf_threshold:
        action = "buy"; confidence = proba_up; sl = price*0.99; tp = price*(1+0.01*r_multiple)
    elif sentiment == "bear" and proba_down >= conf_threshold:
        action = "sell"; confidence = proba_down; sl = price*1.01; tp = price*(1-0.01*r_multiple)
    elif sentiment == "neutral":
        if SENTIMENT_NEUTRAL_ACTION in ("both","buy") and proba_up >= conf_threshold:
            action = "buy"; confidence = proba_up; sl = price*0.99; tp = price*(1+0.01*r_multiple)
        if action is None and SENTIMENT_NEUTRAL_ACTION in ("both","sell") and proba_down >= conf_threshold:
            action = "sell"; confidence = proba_down; sl = price*1.01; tp = price*(1-0.01*r_multiple)
    if action is None: return None

    qty = _position_qty(price, sl)
    if qty <= 0: return None

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
# Guard helpers (local MTM)
# =============================
def _today_local_date_str():
    return datetime.now().astimezone().strftime("%Y-%m-%d")

def _realized_day_pnl() -> float:
    return sum(float(p.get("net_pnl", 0.0)) for p in PERF.values())

def _unrealized_day_pnl() -> float:
    tot = 0.0
    for trades in OPEN_TRADES.values():
        for t in trades:
            if not t.is_open or not t.qty: continue
            px = LAST_PRICE.get(t.symbol)
            if px is None or not (px == px): continue
            tot += ((px - t.entry) if t.side == "buy" else (t.entry - px)) * t.qty
    return float(tot)

def _current_equity_local() -> float:
    return START_EQUITY + _realized_day_pnl() + _unrealized_day_pnl()

def flatten_all_open_positions_local_reason(reason="daily_guard"):
    try:
        close_all_positions()
    except Exception as e:
        print(f"[FLATTEN] close_all_positions error: {e}", flush=True)
    closed = 0
    for (sym, tf), trades in list(OPEN_TRADES.items()):
        for t in trades:
            if t.is_open:
                t.is_open = False
                t.exit_time = datetime.now(timezone.utc).isoformat()
                t.exit = t.entry
                t.reason = reason
                closed += 1
    print(f"[FLATTEN] Locally marked {closed} trade(s) closed ({reason}).", flush=True)

def reset_daily_guard_if_new_day():
    global DAY_STAMP, HALT_TRADING, EQUITY_BASELINE_SET, EQUITY_HIGH_WATER, SESSION_START_EQUITY, EQUITY_BASELINE_DATE
    today = _today_local_date_str()
    if today != DAY_STAMP:
        HALT_TRADING = False
        DAY_STAMP = today
        EQUITY_BASELINE_SET = False
        EQUITY_HIGH_WATER = None
        SESSION_START_EQUITY = None
        EQUITY_BASELINE_DATE = None
        print(f"[DAILY] New day -> reset state. START_EQUITY={START_EQUITY:.2f} Day={DAY_STAMP}", flush=True)

# ---- Broker-equity guard (optional)
def ensure_session_baseline():
    global EQUITY_BASELINE_SET, SESSION_START_EQUITY, EQUITY_HIGH_WATER, EQUITY_BASELINE_DATE
    if not USE_BROKER_EQUITY_GUARD:
        return
    today = _today_local_date_str()
    if EQUITY_BASELINE_DATE != today:
        EQUITY_BASELINE_DATE = today
        EQUITY_BASELINE_SET = False
        EQUITY_HIGH_WATER = None
        SESSION_START_EQUITY = None
    if EQUITY_BASELINE_SET:
        return
    if SESSION_BASELINE_AT_0930:
        ts = _now_et()
        if ts.hour < 9 or (ts.hour == 9 and ts.minute < 30):
            return
    base = get_account_equity(START_EQUITY)
    try: base = float(base)
    except Exception: base = START_EQUITY
    SESSION_START_EQUITY = base
    EQUITY_HIGH_WATER = base
    EQUITY_BASELINE_SET = True
    print(f"[SESSION] Baseline set: session_start_equity={SESSION_START_EQUITY:.2f}", flush=True)

def check_daily_and_trailing_guards():
    global HALT_TRADING, EQUITY_HIGH_WATER
    if not DAILY_GUARD_ENABLED:
        return
    if USE_BROKER_EQUITY_GUARD:
        ensure_session_baseline()
        base = SESSION_START_EQUITY if SESSION_START_EQUITY is not None else START_EQUITY
        eq_now = get_account_equity(base)
        try: eq_now = float(eq_now)
        except Exception: eq_now = base
        up_lim = base * (1.0 + DAILY_TP_PCT)
        dn_lim = base * (1.0 - DAILY_DD_PCT)
        EQUITY_HIGH_WATER = max(EQUITY_HIGH_WATER or eq_now, eq_now)
        mode = "broker"
    else:
        eq_now = _current_equity_local()
        up_lim = START_EQUITY * (1.0 + DAILY_TP_PCT)
        dn_lim = START_EQUITY * (1.0 - DAILY_DD_PCT)
        EQUITY_HIGH_WATER = max(EQUITY_HIGH_WATER or eq_now, eq_now)
        mode = "local"

    print(f"[GUARD:{mode}] eq={eq_now:.2f} targets +{DAILY_TP_PCT*100:.1f}%({up_lim:.2f}) / -{DAILY_DD_PCT*100:.1f}%({dn_lim:.2f}) high_water={EQUITY_HIGH_WATER:.2f}", flush=True)

    if HALT_TRADING:
        return

    if eq_now >= up_lim or eq_now <= dn_lim:
        HALT_TRADING = True
        print(f"[GUARD:{mode}] ⛔ Limit hit. Halting entries.", flush=True)
        if DAILY_FLATTEN_ON_HIT:
            flatten_all_open_positions_local_reason("daily_guard")

    if TRAIL_GUARD_ENABLED and EQUITY_HIGH_WATER:
        trail_floor = EQUITY_HIGH_WATER * (1.0 - TRAIL_DD_PCT)
        if eq_now <= trail_floor and not HALT_TRADING:
            HALT_TRADING = True
            print(f"[GUARD:{mode}] 🛑 Trailing DD hit ({TRAIL_DD_PCT:.0%} from peak).", flush=True)
            flatten_all_open_positions_local_reason("trail_guard")

# =============================
# Routing & signal handling
# =============================
def compute_signal(strategy_name, symbol, tf_minutes, sentiment, df1m=None):
    if df1m is None or getattr(df1m, "empty", True):
        df1m = fetch_bars_1m(symbol, lookback_minutes=max(240, tf_minutes*240))
        if df1m is None or df1m.empty: return None
    if not isinstance(df1m.index, pd.DatetimeIndex):
        try: df1m.index = pd.to_datetime(df1m.index, utc=True)
        except Exception: return None
    try: df1m.index = df1m.index.tz_convert(MARKET_TZ)
    except Exception: df1m.index = df1m.index.tz_localize("UTC").tz_convert(MARKET_TZ)

    try:
        last_px = float(df1m["close"].iloc[-1]); LAST_PRICE[symbol] = last_px
        if last_px < MIN_PRICE: return None
    except Exception: return None

    today_mask = df1m.index.date == df1m.index[-1].date()
    todays_vol = float(df1m.loc[today_mask, "volume"].sum()) if today_mask.any() else 0.0
    if todays_vol < SCANNER_MIN_TODAY_VOL: return None

    if strategy_name == "ml_pattern":
        return signal_ml_pattern(symbol, df1m, tf_minutes, sentiment)
    return None

def _combo_key(strategy: str, symbol: str, tf_min: int) -> str:
    return f"{strategy}|{symbol}|{int(tf_min)}"

def _perf_init(combo: str):
    if combo not in PERF:
        PERF[combo] = {"trades":0,"wins":0,"losses":0,"gross_profit":0.0,"gross_loss":0.0,"net_pnl":0.0,"max_dd":0.0,"equity_curve":[0.0]}

def _perf_update(combo: str, pnl: float):
    _perf_init(combo)
    p = PERF[combo]; p["trades"] += 1
    if pnl > 0: p["wins"] += 1; p["gross_profit"] += pnl
    elif pnl < 0: p["losses"] += 1; p["gross_loss"] += pnl
    p["net_pnl"] += pnl
    ec = p["equity_curve"]; ec.append(ec[-1] + pnl)
    p["max_dd"] = min(p["max_dd"], min(0.0, ec[-1] - max(ec)))

def _record_open_trade(strat_name: str, symbol: str, tf_min: int, sig: dict):
    combo = _combo_key(strat_name, symbol, tf_min)
    _perf_init(combo)
    tp = sig.get("tp_abs", sig.get("takeProfit"))
    sl = sig.get("sl_abs", sig.get("stopLoss"))
    side_norm = "sell" if sig.get("action") in ("sell","sell_short") else "buy"
    t = LiveTrade(
        combo=combo, symbol=symbol, tf_min=int(tf_min), side=side_norm,
        entry=float(sig.get("entry") or sig.get("price") or sig.get("lastClose") or 0.0),
        tp=float(tp) if tp is not None else float("nan"),
        sl=float(sl) if sl is not None else float("nan"),
        qty=int(sig["quantity"]),
        entry_time=sig.get("barTime") or datetime.now(timezone.utc).isoformat(),
    )
    OPEN_TRADES[(symbol, int(tf_min))].append(t)

def _maybe_close_on_bar(symbol: str, tf_min: int, ts, high: float, low: float, close: float):
    key = (symbol, int(tf_min))
    for t in OPEN_TRADES.get(key, []):
        if not t.is_open: continue
        hit_tp = (high >= t.tp) if t.side == "buy" else (low <= t.tp)
        hit_sl = (low <= t.sl) if t.side == "buy" else (high >= t.sl)
        if hit_tp or hit_sl:
            t.is_open = False
            t.exit_time = ts.tz_convert("UTC").isoformat() if hasattr(ts, "tzinfo") else str(ts)
            t.exit  = t.tp if hit_tp else t.sl
            t.reason= "tp" if hit_tp else "sl"
            pnl = (t.exit - t.entry) * t.qty if t.side == "buy" else (t.entry - t.exit) * t.qty
            _perf_update(t.combo, pnl)
            print(f"[CLOSE] {t.combo} {t.reason.upper()} qty={t.qty} entry={t.entry:.2f} exit={t.exit:.2f} pnl={pnl:+.2f}", flush=True)

def _throttle_ok():
    now = time.time()
    while _order_times and now - _order_times[0] > 60:
        _order_times.popleft()
    if len(_order_times) >= MAX_ORDERS_PER_MIN:
        return False, f"throttled: {len(_order_times)}/{MAX_ORDERS_PER_MIN} in last 60s"
    _order_times.append(now)
    return True, ""

def handle_signal(strat_name: str, symbol: str, tf_min: int, sig: dict):
    ok_throttle, why = _throttle_ok()
    if not ok_throttle:
        print(f"[THROTTLE] {why}", flush=True); return
    combo_key = _combo_key(strat_name, symbol, tf_min)
    COUNTS["signals"] += 1
    COMBO_COUNTS[f"{combo_key}::signals"] += 1
    sig = dict(sig)
    meta = sig.get("meta", {})
    meta["combo"] = combo_key; meta["timeframe"] = f"{int(tf_min)}m"
    sig["meta"] = meta

    _record_open_trade(strat_name, symbol, tf_min, sig)

    send_sig = dict(sig)
    if send_sig.get("action") == "sell":
        send_sig["action"] = "sell_short"

    ok, info = send_to_broker(symbol, send_sig, strategy_tag="ranked_ml")
    try:
        COMBO_COUNTS[f"{combo_key}::orders.{'ok' if ok else 'err'}"] += 1
        COUNTS["orders.ok" if ok else "orders.err"] += 1
    except Exception:
        pass
    print(f"[ORDER] {combo_key} -> qty={sig.get('quantity')} ok={ok} info={info}", flush=True)

# =============================
# Main
# =============================
def main():
    global HALT_TRADING
    print(f"[BOOT] RUN_ID={RUN_ID} BRANCH={RENDER_GIT_BRANCH} COMMIT={RENDER_GIT_COMMIT} START_CMD=python Ranked_ML.py", flush=True)
    print(f"[BOOT] POLL_SECONDS={POLL_SECONDS} TFs={TF_MIN_LIST} CONF_THRESHOLD={SCANNER_CONF_THRESHOLD} R_MULTIPLE={SCANNER_R_MULTIPLE}", flush=True)
    print(f"[BOOT] DAILY_GUARD_ENABLED={int(DAILY_GUARD_ENABLED)} UP={DAILY_TP_PCT:.0%} DOWN={DAILY_DD_PCT:.0%} FLATTEN={int(DAILY_FLATTEN_ON_HIT)} START_EQUITY={START_EQUITY:.2f}", flush=True)
    print(f"[BOOT] SENTIMENT_LOOKBACK_MIN={SENTIMENT_LOOKBACK_MIN} SENTIMENT_ONLY_GATE={int(SENTIMENT_ONLY_GATE)} NEUTRAL_ACTION={SENTIMENT_NEUTRAL_ACTION.upper()}", flush=True)
    print(f"[BOOT] BROKER_GUARD={int(USE_BROKER_EQUITY_GUARD)} BASELINE_AT_0930={int(SESSION_BASELINE_AT_0930)} TRAIL_DD={TRAIL_DD_PCT:.0%}", flush=True)

    probe_alpaca_auth()

    symbols = get_universe_symbols()
    print(f"[UNIVERSE] symbols={len(symbols)} TFs={TF_MIN_LIST} vol_gate={SCANNER_MIN_TODAY_VOL} MIN_PRICE={MIN_PRICE} EXCH={sorted(ALLOWED_EXCHANGES)}", flush=True)

    while True:
        loop_start = time.time()
        try:
            reset_daily_guard_if_new_day()

            # ---- Close phase: update prices, check TP/SL on open trades
            touched = set((k[0], k[1]) for k in OPEN_TRADES.keys())
            for (sym, tf) in touched:
                try:
                    df = fetch_bars_1m(sym, lookback_minutes=max(60, tf * 12))
                    if df is None or df.empty: continue
                    try: df.index = df.index.tz_convert(MARKET_TZ)
                    except Exception: df.index = df.index.tz_localize("UTC").tz_convert(MARKET_TZ)
                    bars = _resample(df, tf)
                    if bars is None or bars.empty: continue
                    row = bars.iloc[-1]; ts = bars.index[-1]
                    LAST_PRICE[sym] = float(row["close"])
                    _maybe_close_on_bar(sym, tf, ts, float(row["high"]), float(row["low"]), float(row["close"]))
                except Exception as e:
                    print(f"[CLOSE-PHASE ERROR] {sym} {tf}m: {e}", flush=True)

            # ---- Guards (daily + trailing)
            check_daily_and_trailing_guards()

            # ---- Hardened EOD manager (ALWAYS runs, even if market “closed”)
            now_et = _now_et()

            def _cancel_all_open_orders_safely():
                try:
                    ok, info = cancel_all_orders()
                    print(f"[EOD] Cancel all open orders -> ok={ok} info={info}", flush=True)
                except Exception as e:
                    import traceback
                    print("[EOD] Cancel orders exception:", e, traceback.format_exc(), flush=True)

            def _flatten_until_flat(max_iters=5, sleep_s=2):
                for i in range(max_iters):
                    try:
                        pos = list_positions() or []
                        if not pos:
                            print("[EOD] Positions already flat.", flush=True)
                            return True
                        print(f"[EOD] Flatten pass {i+1}: positions={len(pos)}", flush=True)
                        ok, info = close_all_positions()
                        print(f"[EOD] Flatten call -> ok={ok} info={info}", flush=True)
                    except Exception as e:
                        import traceback
                        print("[EOD] Flatten exception:", e, traceback.format_exc(), flush=True)
                    time.sleep(sleep_s)
                pos = list_positions() or []
                print(f"[EOD] Final position check -> {len(pos)} open", flush=True)
                return len(pos) == 0

            # 1) Stop new entries before the close
            if now_et.hour == 15 and now_et.minute >= 45:
                if not HALT_TRADING:
                    HALT_TRADING = True
                    print("[EOD] HALT_TRADING enabled at 3:45 ET (no new entries).", flush=True)

            # 2) Pre-close consolidation & flatten window (3:50–4:00 ET)
            if now_et.hour == 15 and now_et.minute >= 50:
                print("[EOD] Pre-close window (3:50–4:00 ET): cancel orders + flatten until flat.", flush=True)
                _cancel_all_open_orders_safely()
                _flatten_until_flat()

            # 3) Safety net right after bell (4:00–4:03 ET)
            if now_et.hour == 16 and now_et.minute < 3:
                print("[EOD] Post-bell safety net (4:00–4:03 ET).", flush=True)
                _cancel_all_open_orders_safely()
                _flatten_until_flat()

            # ---- Decide if we may place new entries *after* EOD logic
            allow_entries = not HALT_TRADING
            if not allow_entries:
                time.sleep(POLL_SECONDS)
                continue

            # ---- Only scan for entries during market session (but DO NOT skip EOD above)
            ts_now = _now_et()
            if not _in_session(ts_now):
                time.sleep(POLL_SECONDS)
                continue

            # ---- Scan, rank, execute
            sentiment = compute_sentiment()
            print(f"[SENTIMENT] {sentiment}", flush=True)
            if SENTIMENT_ONLY_GATE and sentiment == "neutral" and SENTIMENT_NEUTRAL_ACTION == "none":
                print("[SENTIMENT] Neutral/no-action policy; skipping.", flush=True)
                time.sleep(POLL_SECONDS)
                continue

            signals_list = []
            for sym in symbols:
                df1m = fetch_bars_1m(sym, lookback_minutes=max(240, max(TF_MIN_LIST)*240))
                if df1m is None or df1m.empty: continue
                try: df1m.index = df1m.index.tz_convert(MARKET_TZ)
                except Exception: df1m.index = df1m.index.tz_localize("UTC").tz_convert(MARKET_TZ)
                for tf in TF_MIN_LIST:
                    sig = compute_signal("ml_pattern", sym, tf, sentiment, df1m=df1m)
                    if not sig: continue
                    k = hashlib.sha256(f"ml_pattern|{sym}|{tf}|{sig['action']}|{sig.get('barTime','')}".encode()).hexdigest()
                    if k in _sent_keys: continue
                    _sent_keys.add(k)
                    signals_list.append((sym, tf, sig))

            signals_list.sort(key=lambda x: x[2]['confidence'], reverse=True)
            for sym, tf, sig in signals_list:
                open_positions = sum(1 for lst in OPEN_TRADES.values() for t in lst if t.is_open)
                if open_positions >= MAX_CONCURRENT_POSITIONS:
                    print(f"[LIMIT] Max concurrent reached during execution: {open_positions}/{MAX_CONCURRENT_POSITIONS}", flush=True)
                    break
                handle_signal("ml_pattern", sym, tf, sig)

        except Exception as e:
            import traceback
            print("[LOOP ERROR]", e, traceback.format_exc(), flush=True)

        elapsed = time.time() - loop_start
        time.sleep(max(1, POLL_SECONDS - int(elapsed)))

if __name__ == "__main__":
    main()
