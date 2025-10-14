# ml_merged_app.py — bi-directional ML scanner with sentiment regime
# (PLUMBING SWAP: Polygon/TP -> Alpaca adapters)
# - ML & sentiment logic unchanged
# - Data via adapters.data_alpaca
# - Orders via common.signal_bridge (Alpaca)
# - Hardened EOD: halt 3:45 ET, cancel orders, flatten in passes, post-bell safety net

import os, time, json, math, hashlib
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

# === Alpaca adapters (data + broker/plumbing) ===
from adapters.data_alpaca import (
    fetch_1m as _alpaca_fetch_1m,     # fetch_1m(symbol, start_iso, end_iso, limit)
    resample as _resample,            # resample(df1m, tf_min)
    get_universe_symbols as _alpaca_universe,  # universe w/ exchange filters inside adapter
)

from common.signal_bridge import (
    send_to_broker,              # send_to_broker(payload: dict) -> (ok: bool, info: str)
    close_all_positions,         # -> (ok: bool, info: str)
    list_positions,              # -> list
    get_account_equity,          # (default_equity: float) -> float
    cancel_all_orders,           # -> (ok: bool, info: str)
)

# ------------------------------------------------------------------------------------
# Backward-compatible fetch wrapper (so existing code calling fetch_bars_1m keeps working)
# ------------------------------------------------------------------------------------
def fetch_bars_1m(symbol: str, lookback_minutes: int = 2400) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=lookback_minutes)
    start_iso = start.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    end_iso   = end.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    df = _alpaca_fetch_1m(symbol, start_iso=start_iso, end_iso=end_iso, limit=10000)
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        df.index = df.index.tz_convert(MARKET_TZ)
    except Exception:
        df.index = df.index.tz_localize("UTC").tz_convert(MARKET_TZ)
    # Ensure expected columns
    cols = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    return df[cols].copy() if cols else pd.DataFrame()

# =============================
# ENV / CONFIG
# =============================
POLL_SECONDS    = int(os.getenv("POLL_SECONDS", "10"))
DRY_RUN         = os.getenv("DRY_RUN", "0").lower() in ("1","true","yes")
PAPER_MODE      = os.getenv("PAPER_MODE", "true").lower() != "false"
SCANNER_DEBUG   = os.getenv("SCANNER_DEBUG", "0").lower() in ("1","true","yes")

RUN_ID            = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
RENDER_GIT_COMMIT = os.getenv("RENDER_GIT_COMMIT", "unknown")[:12]
RENDER_GIT_BRANCH = os.getenv("RENDER_GIT_BRANCH", os.getenv("BRANCH", "unknown"))

# --- Universe
SCANNER_SYMBOLS = os.getenv("SCANNER_SYMBOLS", "").strip()
TF_MIN_LIST     = [int(x) for x in os.getenv("TF_MIN_LIST", "1,2,3,5,10").split(",") if x.strip()]
MARKET_TZ       = os.getenv("MARKET_TZ", "America/New_York")

# Pagination/batching
SCAN_BATCH_SIZE             = int(os.getenv("SCAN_BATCH_SIZE", "300"))

# Volume gates (today’s 1m sum; grouped-daily prefilter is a Polygon-only concept, skipped)
SCANNER_MIN_TODAY_VOL       = int(os.getenv("SCANNER_MIN_TODAY_VOL", "500000"))

# Price + exchange gates (enforced in-scan)
MIN_PRICE = float(os.getenv("MIN_PRICE", "3.0"))
ALLOWED_EXCHANGES = set(
    x.strip().upper()
    for x in os.getenv("ALLOWED_EXCHANGES", "NASD,NASDAQ,NYSE,XNAS,XNYS").split(",")
    if x.strip()
)

# --- Session
ALLOW_PREMARKET  = os.getenv("ALLOW_PREMARKET","0").lower() in ("1","true","yes")
ALLOW_AFTERHOURS = os.getenv("ALLOW_AFTERHOURS","0").lower() in ("1","true","yes")

# --- Sizing
EQUITY_USD  = float(os.getenv("EQUITY_USD",  "100000"))
RISK_PCT    = float(os.getenv("RISK_PCT",    "0.01"))
MAX_POS_PCT = float(os.getenv("MAX_POS_PCT", "0.10"))
MIN_QTY     = int(os.getenv("MIN_QTY", "1"))
ROUND_LOT   = int(os.getenv("ROUND_LOT","1"))

# --- Model / sentiment
CONF_THR       = float(os.getenv("CONF_THR", "0.70"))
R_MULT         = float(os.getenv("R_MULT", "1.5"))
SHORTS_ENABLED = os.getenv("SHORTS_ENABLED","0").lower() in ("1","true","yes")

SENTIMENT_LOOKBACK_MIN = int(os.getenv("SENTIMENT_LOOKBACK_MIN", "60"))
SENTIMENT_NEUTRAL_BAND = float(os.getenv("SENTIMENT_NEUTRAL_BAND", "0.0015"))
SENTIMENT_SYMBOLS      = [s.strip() for s in os.getenv("SENTIMENT_SYMBOLS","SPY,QQQ").split(",") if s.strip()]
USE_SENTIMENT_REGIME   = os.getenv("USE_SENTIMENT_REGIME","1").lower() in ("1","true","yes")

# --- Engine guards / pacing
MAX_CONCURRENT_POSITIONS = int(os.getenv("MAX_CONCURRENT_POSITIONS", "100"))
MAX_ORDERS_PER_MIN       = int(os.getenv("MAX_ORDERS_PER_MIN", "60"))

# --- Daily guard (local) and broker guard
DAILY_GUARD_ENABLED  = os.getenv("DAILY_GUARD_ENABLED","1").lower() in ("1","true","yes")
START_EQUITY         = float(os.getenv("START_EQUITY","100000"))
DAILY_TP_PCT         = float(os.getenv("DAILY_TP_PCT","0.10"))  # 10%
DAILY_DD_PCT         = float(os.getenv("DAILY_DD_PCT","0.05"))  # 5%
DAILY_FLATTEN_ON_HIT = os.getenv("DAILY_FLATTEN_ON_HIT","1").lower() in ("1","true","yes")

USE_BROKER_EQUITY_GUARD   = os.getenv("USE_BROKER_EQUITY_GUARD","0").lower() in ("1","true","yes")
SESSION_START_EQUITY_ENV  = os.getenv("SESSION_START_EQUITY","").strip()
SESSION_START_EQUITY      = float(SESSION_START_EQUITY_ENV) if SESSION_START_EQUITY_ENV else None
SESSION_BASELINE_AT_0930  = os.getenv("SESSION_BASELINE_AT_0930","1").lower() in ("1","true","yes")

# --- Trailing high-water
TRAIL_GUARD_ENABLED   = os.getenv("TRAIL_GUARD_ENABLED","1").lower() in ("1","true","yes")
TRAIL_DD_PCT          = float(os.getenv("TRAIL_DD_PCT","0.05"))
TRAIL_FLATTEN_ON_HIT  = os.getenv("TRAIL_FLATTEN_ON_HIT","1").lower() in ("1","true","yes")

# --- Kill switch
KILL_SWITCH       = os.getenv("KILL_SWITCH","OFF").lower() in ("1","true","yes","on")
KILL_SWITCH_MODE  = os.getenv("KILL_SWITCH_MODE","halt").lower()   # 'halt' or 'flatten'

# =============================
# State / ledgers
# =============================
COUNTS        = defaultdict(int)
COMBO_COUNTS  = defaultdict(int)
PERF          = {}
OPEN_TRADES   = defaultdict(list)
_sent_keys    = set()
_order_times  = deque()
LAST_PRICE    = {}

DAY_STAMP     = datetime.now().astimezone().strftime("%Y-%m-%d")
HALT_TRADING  = False
HALTED        = False

EQUITY_BASELINE_DATE = None
EQUITY_BASELINE_SET  = False
EQUITY_HIGH_WATER    = None

_rr_idx = 0

# =============================
# Models / classes
# =============================
class LiveTrade:
    def __init__(self, combo, symbol, tf_min, side, entry, tp, sl, qty, entry_time):
        self.combo = combo
        self.symbol = symbol
        self.tf_min = int(tf_min)
        self.side = side            # "buy" or "sell" (normalized)
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

def _after_0930_now():
    ts = _now_et()
    return (ts.hour > 9) or (ts.hour == 9 and ts.minute >= 30)

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
    qty_risk     = (equity * risk_pct) / risk_per_share
    qty_notional = (equity * max_pos_pct) / max(1e-9, float(entry_price))
    qty = math.floor(max(min(qty_risk, qty_notional), 0) / max(1, round_lot)) * max(1, round_lot)
    return int(max(qty, min_qty if qty > 0 else 0))

# =============================
# Universe (Alpaca)
# =============================
def get_universe_symbols() -> list:
    """SCANNER_SYMBOLS override, else Alpaca assets via adapter."""
    if SCANNER_SYMBOLS:
        return [s.strip().upper() for s in SCANNER_SYMBOLS.split(",") if s.strip()]
    return _alpaca_universe(limit=10000)

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
        p["wins"] += 1; p["gross_profit"] += pnl
    elif pnl < 0:
        p["losses"] += 1; p["gross_loss"] += pnl
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
    side_norm = "sell" if sig["action"] in ("sell", "sell_short") else "buy"
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
                t.exit = t.tp; t.reason = "tp"
            else:
                t.exit = t.sl; t.reason = "sl"
            pnl = (t.exit - t.entry) * t.qty if t.side == "buy" else (t.entry - t.exit) * t.qty
            _perf_update(t.combo, pnl)
            print(f"[CLOSE] {t.combo} {t.reason.upper()} qty={t.qty} entry={t.entry:.2f} exit={t.exit:.2f} pnl={pnl:+.2f}", flush=True)

# =============================
# Sentiment
# =============================
def compute_sentiment():
    look_min = max(5, SENTIMENT_LOOKBACK_MIN)
    vals = []
    for s in SENTIMENT_SYMBOLS:
        df = fetch_bars_1m(s, lookback_minutes=look_min*2)
        if df is None or df.empty:
            continue
        try:
            df.index = df.index.tz_convert(MARKET_TZ)
        except Exception:
            df.index = df.index.tz_localize("UTC").tz_convert(MARKET_TZ)
        win = df.iloc[-look_min:]
        if len(win) < 2:
            continue
        vals.append(float(win["close"].iloc[-1]) / float(win["close"].iloc[0]) - 1.0)
    if not vals:
        return "neutral"
    avg = sum(vals) / len(vals)
    if avg >= SENTIMENT_NEUTRAL_BAND:
        return "bull"
    if avg <= -SENTIMENT_NEUTRAL_BAND:
        return "bear"
    return "neutral"

# =============================
# ML adapter
# =============================
def _ml_features_and_pred(bars: pd.DataFrame):
    try:
        from sklearn.ensemble import RandomForestClassifier
        import pandas_ta as ta
    except Exception:
        return None, None, None
    if bars is None or bars.empty or len(bars) < 120:
        return None, None, None
    df = bars.copy()
    df["return"] = df["close"].pct_change()
    try:
        import pandas_ta as ta
        df["rsi"] = ta.rsi(df["close"], length=14)
    except Exception:
        delta = df["close"].diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = -delta.clip(upper=0).rolling(14).mean()
        rs = up / (down.replace(0, np.nan))
        df["rsi"] = 100 - (100 / (1 + rs))
    df["volatility"] = df["close"].rolling(20).std()
    df.dropna(inplace=True)
    if len(df) < 60:
        return None, None, None
    X = df[["return","rsi","volatility"]].iloc[:-1]
    y = (df["close"].shift(-1) > df["close"]).astype(int).iloc[:-1]
    if len(X) < 50:
        return None, None, None
    cut = int(len(X) * 0.7)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X.iloc[:cut], y.iloc[:cut])
    x_live = X.iloc[[-1]]
    try:
        x_live = x_live[list(clf.feature_names_in_)]
    except Exception:
        pass
    proba_up = float(clf.predict_proba(x_live)[0][1])
    pred_up  = int(proba_up >= 0.5)
    ts = df.index[-1]
    return ts, proba_up, pred_up

def signal_ml_pattern(symbol: str, df1m: pd.DataFrame, tf_min: int,
                      conf_threshold=CONF_THR, r_multiple=R_MULT,
                      sentiment="neutral", shorts_enabled=SHORTS_ENABLED):
    if df1m is None or df1m.empty or not isinstance(df1m.index, pd.DatetimeIndex):
        return None
    bars = _resample(df1m, tf_min)
    if bars.empty:
        return None
    ts, proba_up, pred_up = _ml_features_and_pred(bars)
    if ts is None:
        return None
    if not _in_session(ts):
        return None

    price = float(bars["close"].iloc[-1])
    long_sl  = price * 0.99
    long_tp  = price * (1 + 0.01 * r_multiple)
    short_sl = price * 1.01
    short_tp = price * (1 - 0.01 * r_multiple)

    want_long  = (proba_up >= conf_threshold)
    want_short = ((1.0 - proba_up) >= conf_threshold) and shorts_enabled

    if USE_SENTIMENT_REGIME:
        if sentiment == "bull":
            want_short = False
        elif sentiment == "bear":
            want_long = False

    # Build signals — use "sell" for short to align with Alpaca bridge
    if want_long:
        qty = _position_qty(price, long_sl)
        if qty > 0:
            return {
                "action":"buy","orderType":"market","price":None,
                "tp_abs": long_tp,"sl_abs": long_sl,
                "barTime": ts.tz_convert("UTC").isoformat(),"entry": price,
                "quantity": int(qty),
                "meta":{"note":"ml_pattern_long","proba_up": proba_up,"sentiment": sentiment}
            }
    if want_short:
        qty = _position_qty(price, short_sl)
        if qty > 0:
            return {
                "action":"sell","orderType":"market","price":None,
                "tp_abs": short_tp,"sl_abs": short_sl,
                "barTime": ts.tz_convert("UTC").isoformat(),"entry": price,
                "quantity": int(qty),
                "meta":{"note":"ml_pattern_short","proba_up": proba_up,"sentiment": sentiment}
            }
    return None

# =============================
# Equity / guards
# =============================
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

def _current_equity_local() -> float:
    return START_EQUITY + _realized_day_pnl() + _unrealized_day_pnl()

def check_kill_switch() -> bool:
    global HALTED, HALT_TRADING
    if KILL_SWITCH and not HALTED:
        HALTED = True
        HALT_TRADING = True
        print(f"[GUARD] Kill switch ON (mode={KILL_SWITCH_MODE}). Trading halted.", flush=True)
        if KILL_SWITCH_MODE == "flatten":
            _cancel_and_flatten_in_passes("KILL")
        return True
    return HALTED

def ensure_session_baseline():
    """
    Establish broker-equity session baseline once per day if using broker guard.
    Baseline optionally waits for 9:30 if SESSION_BASELINE_AT_0930=1.
    """
    global SESSION_START_EQUITY, EQUITY_BASELINE_DATE, EQUITY_BASELINE_SET, EQUITY_HIGH_WATER
    if not USE_BROKER_EQUITY_GUARD:
        return
    today = datetime.now(timezone.utc).astimezone(ZoneInfo(MARKET_TZ)).date().isoformat()
    if EQUITY_BASELINE_DATE != today:
        EQUITY_BASELINE_SET = False
        EQUITY_BASELINE_DATE = today
        EQUITY_HIGH_WATER = None
    if EQUITY_BASELINE_SET:
        return
    if SESSION_BASELINE_AT_0930 and not _after_0930_now():
        return
    if SESSION_START_EQUITY is None:
        SESSION_START_EQUITY = get_account_equity(EQUITY_USD)
    EQUITY_HIGH_WATER = SESSION_START_EQUITY
    EQUITY_BASELINE_SET = True
    print(f"[SESSION] Baseline set: session_start_equity={SESSION_START_EQUITY:.2f}", flush=True)

def _active_equity_and_limits():
    if USE_BROKER_EQUITY_GUARD:
        ensure_session_baseline()
        base = SESSION_START_EQUITY if SESSION_START_EQUITY is not None else EQUITY_USD
        eq_now = get_account_equity(base)
        up_lim = base * (1.0 + DAILY_TP_PCT)
        dn_lim = base * (1.0 - DAILY_DD_PCT)
        return eq_now, up_lim, dn_lim, "broker"
    else:
        eq_now = _current_equity_local()
        up_lim = START_EQUITY * (1.0 + DAILY_TP_PCT)
        dn_lim = START_EQUITY * (1.0 - DAILY_DD_PCT)
        return eq_now, up_lim, dn_lim, "local"

def check_daily_guards():
    global HALT_TRADING, HALTED, EQUITY_HIGH_WATER
    eq_now, up_lim, dn_lim, mode = _active_equity_and_limits()

    if EQUITY_HIGH_WATER is None:
        EQUITY_HIGH_WATER = eq_now
    else:
        EQUITY_HIGH_WATER = max(EQUITY_HIGH_WATER, eq_now)

    print(f"[GUARD:{mode}] eq={eq_now:.2f} "
          f"targets +{DAILY_TP_PCT*100:.1f}%({up_lim:.2f}) / -{DAILY_DD_PCT*100:.1f}%({dn_lim:.2f}) "
          f"high_water={EQUITY_HIGH_WATER:.2f}", flush=True)

    if DAILY_GUARD_ENABLED:
        if eq_now >= up_lim and not HALTED:
            HALT_TRADING = True; HALTED = True
            print(f"[GUARD:{mode}] ✅ Profit target hit. Halting entries.", flush=True)
            if DAILY_FLATTEN_ON_HIT:
                _cancel_and_flatten_in_passes("TP")
        elif eq_now <= dn_lim and not HALTED:
            HALT_TRADING = True; HALTED = True
            print(f"[GUARD:{mode}] ⛔ Drawdown limit hit. Halting entries.", flush=True)
            if DAILY_FLATTEN_ON_HIT:
                _cancel_and_flatten_in_passes("DD")

def reset_daily_state_if_new_day():
    global DAY_STAMP, HALT_TRADING, HALTED, EQUITY_HIGH_WATER, EQUITY_BASELINE_SET
    today = datetime.now().astimezone().strftime("%Y-%m-%d")
    if today != DAY_STAMP:
        DAY_STAMP = today
        HALT_TRADING = False
        HALTED = False
        EQUITY_HIGH_WATER = None
        EQUITY_BASELINE_SET = False
        print(f"[NEW DAY] State reset. START_EQUITY={START_EQUITY:.2f} DAY={DAY_STAMP}", flush=True)

# =============================
# EOD manager (HALT @ 3:45 ET, cancel+flatten in passes, safety net after bell)
# =============================
def _cancel_and_flatten_in_passes(tag: str, max_passes: int = 6, sleep_sec: float = 2.0):
    try:
        okc, infoc = cancel_all_orders()
        print(f"[EOD:{tag}] Cancel orders -> ok={okc} info={infoc}", flush=True)
    except Exception as e:
        print(f"[EOD:{tag}] Cancel orders -> exception: {e}", flush=True)

    for i in range(1, max_passes + 1):
        try:
            pos = list_positions() or []
        except Exception as e:
            print(f"[EOD:{tag}] list_positions exception: {e}", flush=True)
            pos = []
        if not pos:
            print(f"[EOD:{tag}] Flat confirmed after {i-1} pass(es).", flush=True)
            break
        try:
            okf, infof = close_all_positions()
            print(f"[EOD:{tag}] Flatten pass {i}/{max_passes} -> ok={okf} info={infof} (open={len(pos)})", flush=True)
        except Exception as e:
            print(f"[EOD:{tag}] Flatten pass {i}/{max_passes} -> exception: {e}", flush=True)
        # Re-cancel in case new/recreated orders appear
        try:
            okc2, infoc2 = cancel_all_orders()
            print(f"[EOD:{tag}] Re-cancel -> ok={okc2} info={infoc2}", flush=True)
        except Exception as e:
            print(f"[EOD:{tag}] Re-cancel -> exception: {e}", flush=True)
        time.sleep(sleep_sec)

    # Mark locals closed (best-effort)
    for (sym, tf), trades in list(OPEN_TRADES.items()):
        for t in trades:
            if t.is_open:
                t.is_open = False
                t.exit_time = datetime.now(timezone.utc).isoformat()
                t.exit = t.entry
                t.reason = f"eod_{tag.lower()}"

def eod_manager():
    global HALT_TRADING
    ts = _now_et()
    # Pre-close window: 3:45–4:00 ET
    if (ts.hour == 15 and ts.minute >= 45) and (ts.hour < 16):
        if not HALT_TRADING:
            print("[EOD] Pre-close: HALTING new entries at 3:45 ET and starting cancel+flatten passes.", flush=True)
        HALT_TRADING = True
        _cancel_and_flatten_in_passes("PRE")
    # Safety net: 4:00–4:05 ET
    elif ts.hour == 16 and ts.minute < 5:
        if not HALT_TRADING:
            print("[EOD] Post-bell safety: halting entries and ensuring flat (4:00–4:05 ET).", flush=True)
        HALT_TRADING = True
        _cancel_and_flatten_in_passes("POST")

# =============================
# Routing & signal handling (Alpaca bridge expects TradersPost-like payload)
# =============================
def _throttle_ok():
    now = time.time()
    while _order_times and now - _order_times[0] > 60:
        _order_times.popleft()
    if len(_order_times) >= MAX_ORDERS_PER_MIN:
        return False, f"throttled: {len(_order_times)}/{MAX_ORDERS_PER_MIN} in last 60s"
    _order_times.append(now)
    return True, ""

def _build_payload(symbol: str, sig: dict) -> dict:
    payload = {
        "ticker": symbol,
        "action": sig.get("action"),
        "orderType": sig.get("orderType", "market"),
        "quantity": int(sig.get("quantity", 0)),
        "meta": sig.get("meta", {})
    }
    # absolute TP/SL for bracket
    tp_abs = sig.get("tp_abs", sig.get("takeProfit"))
    sl_abs = sig.get("sl_abs", sig.get("stopLoss"))
    if tp_abs is not None:
        payload["takeProfit"] = {"limitPrice": float(round(tp_abs, 2))}
    if sl_abs is not None:
        payload["stopLoss"]   = {"stopPrice": float(round(sl_abs, 2))}
    # helpful audit
    if sig.get("barTime"): payload["meta"]["barTime"] = sig["barTime"]
    if sig.get("entry")   : payload["entry"] = float(sig["entry"])
    payload["meta"]["environment"] = "paper" if PAPER_MODE else "live"
    payload["meta"]["runId"] = RUN_ID
    return payload

def handle_signal(strat_name: str, symbol: str, tf_min: int, sig: dict):
    ok_throttle, why = _throttle_ok()
    if not ok_throttle:
        print(f"[THROTTLE] {why}", flush=True); return

    combo_key = _combo_key(strat_name, symbol, tf_min)
    COUNTS["signals"] += 1
    COMBO_COUNTS[f"{combo_key}::signals"] += 1
    meta = sig.get("meta", {})
    meta["combo"] = combo_key
    meta["timeframe"] = f"{int(tf_min)}m"
    sig["meta"] = meta

    _record_open_trade(strat_name, symbol, tf_min, sig)

    payload = _build_payload(symbol, sig)
    ok, info = send_to_broker(payload)
    try:
        COMBO_COUNTS[f"{combo_key}::orders.{'ok' if ok else 'err'}"] += 1
        COUNTS["orders.ok" if ok else "orders.err"] += 1
    except Exception:
        pass
    print(f"[ORDER] {combo_key} -> action={sig.get('action')} qty={sig.get('quantity')} ok={ok} info={info}", flush=True)

# =============================
# Batching helper
# =============================
def _batched_symbols(universe: list):
    global _rr_idx
    if not universe:
        return []
    N = len(universe)
    start = _rr_idx % max(1, N)
    end = min(N, start + SCAN_BATCH_SIZE)
    batch = universe[start:end]
    _rr_idx = 0 if end >= N else end
    return batch

# =============================
# Signal computation (unchanged)
# =============================
def compute_signal(strategy_name, symbol, tf_minutes, df1m=None, sentiment="neutral"):
    if df1m is None or getattr(df1m, "empty", True):
        df1m = fetch_bars_1m(symbol, lookback_minutes=max(240, tf_minutes*240))
        if df1m is None or df1m.empty:
            return None
    if not isinstance(df1m.index, pd.DatetimeIndex):
        try:
            df1m.index = pd.to_datetime(df1m.index, utc=True)
        except Exception:
            return None
    try:
        df1m.index = df1m.index.tz_convert(MARKET_TZ)
    except Exception:
        df1m.index = df1m.index.tz_localize("UTC").tz_convert(MARKET_TZ)

    try:
        last_px = float(df1m["close"].iloc[-1])
        LAST_PRICE[symbol] = last_px
        if last_px < MIN_PRICE:
            return None
    except Exception:
        return None

    today_mask = df1m.index.date == df1m.index[-1].date()
    todays_vol = float(df1m.loc[today_mask, "volume"].sum()) if today_mask.any() else 0.0
    if todays_vol < SCANNER_MIN_TODAY_VOL:
        return None

    if strategy_name == "ml_pattern":
        return signal_ml_pattern(
            symbol, df1m, tf_minutes,
            conf_threshold=CONF_THR, r_multiple=R_MULT,
            sentiment=sentiment, shorts_enabled=SHORTS_ENABLED
        )
    return None

# =============================
# Main
# =============================
def main():
    global HALT_TRADING

    print(f"[BOOT] RUN_ID={RUN_ID} BRANCH={RENDER_GIT_BRANCH} COMMIT={RENDER_GIT_COMMIT} START_CMD=python ml_merged_app.py", flush=True)
    print(f"[BOOT] PAPER_MODE={'paper' if PAPER_MODE else 'live'} POLL_SECONDS={POLL_SECONDS} TFs={TF_MIN_LIST}", flush=True)
    print(f"[BOOT] CONF_THR={CONF_THR} R_MULT={R_MULT} SHORTS_ENABLED={int(SHORTS_ENABLED)} USE_SENTIMENT_REGIME={int(USE_SENTIMENT_REGIME)}", flush=True)
    print(f"[BOOT] DAILY_GUARD_ENABLED={int(DAILY_GUARD_ENABLED)} UP={DAILY_TP_PCT:.0%} DOWN={DAILY_DD_PCT:.0%} FLATTEN={int(DAILY_FLATTEN_ON_HIT)} START_EQUITY={START_EQUITY:.2f}", flush=True)
    print(f"[BOOT] BROKER_GUARD={int(USE_BROKER_EQUITY_GUARD)} BASELINE_AT_0930={int(SESSION_BASELINE_AT_0930)} TRAIL_DD={TRAIL_DD_PCT:.0%}", flush=True)
     
    from common.signal_bridge import probe_alpaca_auth
    probe_alpaca_auth()

    # Universe
    symbols = get_universe_symbols()
    if SCANNER_SYMBOLS:
        print(f"[BOOT] SCANNER_SYMBOLS override detected ({len(symbols)} symbols).", flush=True)
    print(f"[UNIVERSE] symbols={len(symbols)} TFs={TF_MIN_LIST} batch={SCAN_BATCH_SIZE} "
          f"vol_gate(today)={SCANNER_MIN_TODAY_VOL} MIN_PRICE={MIN_PRICE} EXCH={sorted(ALLOWED_EXCHANGES)}", flush=True)

    # Initial baseline (if broker guard)
    if USE_BROKER_EQUITY_GUARD:
        ensure_session_baseline()

    while True:
        loop_start = time.time()
        try:
            reset_daily_state_if_new_day()

            # --- Close phase: update MTM and check TP/SL on last bar for open trades
            touched = set((k[0], k[1]) for k in OPEN_TRADES.keys())
            for (sym, tf) in touched:
                try:
                    df = fetch_bars_1m(sym, lookback_minutes=max(60, tf * 12))
                    if df is None or df.empty:
                        continue
                    try:
                        df.index = df.index.tz_convert(MARKET_TZ)
                    except Exception:
                        df.index = df.index.tz_localize("UTC").tz_convert(MARKET_TZ)
                    bars = _resample(df, tf)
                    if bars is not None and not bars.empty:
                        row = bars.iloc[-1]
                        ts  = bars.index[-1]
                        LAST_PRICE[sym] = float(row["close"])
                        _maybe_close_on_bar(sym, tf, ts, float(row["high"]), float(row["low"]), float(row["close"]))
                except Exception as e:
                    print(f"[CLOSE-PHASE ERROR] {sym} {tf}m: {e}", flush=True)

            # --- Guards (kill, daily, trail)
            if not check_kill_switch():
                check_daily_guards()

            # --- EOD manager (pre-close + safety net) BEFORE deciding to enter
            eod_manager()

            allow_entries = not (HALT_TRADING or HALTED)
            if not allow_entries:
                time.sleep(POLL_SECONDS)
                continue

            # --- Scan & signal phase
            sentiment = compute_sentiment() if USE_SENTIMENT_REGIME else "neutral"
            print(f"[SENTIMENT] {sentiment}", flush=True)

            batch = _batched_symbols(symbols)
            for sym in batch:
                df1m = fetch_bars_1m(sym, lookback_minutes=max(240, max(TF_MIN_LIST)*240))
                if df1m is None or df1m.empty:
                    continue
                try:
                    df1m.index = df1m.index.tz_convert(MARKET_TZ)
                except Exception:
                    df1m.index = df1m.index.tz_localize("UTC").tz_convert(MARKET_TZ)

                # price/vol gates here on 1m data
                try:
                    last_px = float(df1m["close"].iloc[-1])
                except Exception:
                    continue
                if last_px < MIN_PRICE:
                    continue
                today_mask = df1m.index.date == df1m.index[-1].date()
                todays_vol = float(df1m.loc[today_mask, "volume"].sum()) if today_mask.any() else 0.0
                if todays_vol < SCANNER_MIN_TODAY_VOL:
                    continue

                for tf in TF_MIN_LIST:
                    sig = compute_signal("ml_pattern", sym, tf, sentiment=sentiment, df1m=df1m)
                    if not sig:
                        continue

                    dk = hashlib.sha256(f"ml_pattern|{sym}|{tf}|{sig['action']}|{sig.get('barTime','')}".encode()).hexdigest()
                    if dk in _sent_keys:
                        continue
                    _sent_keys.add(dk)

                    open_positions = sum(1 for lst in OPEN_TRADES.values() for t in lst if t.is_open)
                    if open_positions >= MAX_CONCURRENT_POSITIONS:
                        print(f"[LIMIT] Max concurrent reached: {open_positions}/{MAX_CONCURRENT_POSITIONS}", flush=True)
                        break

                    handle_signal("ml_pattern", sym, tf, sig)

        except Exception as e:
            import traceback
            print("[LOOP ERROR]", e, traceback.format_exc(), flush=True)

        elapsed = time.time() - loop_start
        time.sleep(max(1, POLL_SECONDS - int(elapsed)))

# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
