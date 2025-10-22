# ====================================================================================
# app.py — ML v3 Dual-Side with Long Failsafe, Market Bias, and Audit (2025-10-22)
# ====================================================================================
# Key additions vs your last post:
# - AUDIT: live counters showing whether longs are dying at model vs gates.
# - FAILSAFE: automatic intraday relaxation of long threshold if zero are passing.
# - MARKET BIAS: on green RS_SYMBOL tape, prefer longs unless shorts beat by margin.
# - DEBUG: per-symbol/TF signal line via DEBUG_DUMP_SIG=1.
# - GC: light cache clear after-hours to avoid memory creep on Render.
#
# Drop-in file. No other files changed.
# ====================================================================================

import os, time, json, math, requests
import pandas as pd, numpy as np
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
from zoneinfo import ZoneInfo
from collections import OrderedDict

# ---- Alpaca adapters (data + broker) ----
from adapters.data_alpaca import (
    fetch_1m as _alpaca_fetch_1m,
    get_universe_symbols as _alpaca_universe,
)
from common.signal_bridge import (
    send_to_broker,
    close_all_positions,
    list_positions,
    get_account_equity,
    cancel_all_orders,
)

# ==============================
# ENV / CONFIG
# ==============================
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "10"))
DRY_RUN      = os.getenv("DRY_RUN", "0").lower() in ("1","true","yes")
PAPER_MODE   = os.getenv("PAPER_MODE", "true").lower() != "false"
SCANNER_DEBUG= os.getenv("SCANNER_DEBUG", "0").lower() in ("1","true","yes")
DEBUG_DUMP_SIG = os.getenv("DEBUG_DUMP_SIG", "0").lower() in ("1","true","yes")

EXTRA_TICKS = int(os.getenv("EXTRA_TICKS", "2"))  # extra pennies to clear Alpaca base_price drift

TF_MIN_LIST  = [int(x) for x in os.getenv("TF_MIN_LIST", "1,2,3,5,10").split(",") if x.strip()]
MAX_UNIVERSE_PAGES = int(os.getenv("MAX_UNIVERSE_PAGES", "3"))
SCAN_BATCH_SIZE    = int(os.getenv("SCAN_BATCH_SIZE", "150"))
SCANNER_MIN_AVG_VOL = int(os.getenv("SCANNER_MIN_AVG_VOL", "0"))

MIN_PRICE = float(os.getenv("MIN_PRICE", "3.0"))
MIN_SHORT_PRICE = float(os.getenv("MIN_SHORT_PRICE", "5.0"))  # skip shorting very low-priced name

# Risk / sizing
EQUITY_USD  = float(os.getenv("EQUITY_USD",  "100000"))
RISK_PCT    = float(os.getenv("RISK_PCT",    "0.01"))
MAX_POS_PCT = float(os.getenv("MAX_POS_PCT", "0.10"))
MIN_QTY     = int(os.getenv("MIN_QTY", "1"))
ROUND_LOT   = int(os.getenv("ROUND_LOT","1"))
USE_FIXED_NOTIONAL = os.getenv("USE_FIXED_NOTIONAL", "1").lower() in ("1","true","yes")
FIXED_NOTIONAL     = float(os.getenv("FIXED_NOTIONAL", "2500"))
NO_PYRAMIDING      = os.getenv("NO_PYRAMIDING", "1").lower() in ("1","true","yes")

# Model thresholds
CONF_THR    = float(os.getenv("CONF_THR", "0.70"))
R_MULT      = float(os.getenv("R_MULT", "1.50"))

# Market/session
MARKET_TZ   = os.getenv("MARKET_TZ", "America/New_York")
SCANNER_MARKET_HOURS_ONLY = os.getenv("SCANNER_MARKET_HOURS_ONLY","1").lower() in ("1","true","yes")
ALLOW_PREMARKET  = os.getenv("ALLOW_PREMARKET","0").lower() in ("1","true","yes")
ALLOW_AFTERHOURS = os.getenv("ALLOW_AFTERHOURS","0").lower() in ("1","true","yes")

# Daily portfolio guard
START_EQUITY         = float(os.getenv("START_EQUITY", "100000"))
DAILY_TP_PCT         = float(os.getenv("DAILY_TP_PCT", "0.10"))     # +10%
DAILY_DD_PCT         = float(os.getenv("DAILY_DD_PCT", "0.05"))     # -5%
DAILY_FLATTEN_ON_HIT = os.getenv("DAILY_FLATTEN_ON_HIT","1").lower() in ("1","true","yes")
DAILY_GUARD_ENABLED  = os.getenv("DAILY_GUARD_ENABLED","1").lower() in ("1","true","yes")
USE_BROKER_EQUITY_GUARD   = os.getenv("USE_BROKER_EQUITY_GUARD","0").lower() in ("1","true","yes")
SESSION_BASELINE_AT_0930  = os.getenv("SESSION_BASELINE_AT_0930","1").lower() in ("1","true","yes")
TRAIL_GUARD_ENABLED       = os.getenv("TRAIL_GUARD_ENABLED","1").lower() in ("1","true","yes")
TRAIL_DD_PCT              = float(os.getenv("TRAIL_DD_PCT","0.05"))

# --- Kill switch (immediate halt without redeploy)
KILL_SWITCH              = os.getenv("KILL_SWITCH","0").lower() in ("1","true","yes")

# --- Confidence / consensus / selection controls
PROBA_EMA_ALPHA          = float(os.getenv("PROBA_EMA_ALPHA","0.0"))  # 0 = off
MIN_PROBA_GAP            = float(os.getenv("MIN_PROBA_GAP","0.0"))    # require conf >= thr+gap
CONSENSUS_TF             = int(os.getenv("CONSENSUS_TF","0"))         # 0 = off
CONSENSUS_WEIGHT         = float(os.getenv("CONSENSUS_WEIGHT","0.5"))
BATCH_TOP_K              = int(os.getenv("BATCH_TOP_K","0"))          # 0 = take all

# --- Relative strength benchmark (for features & tilt)
RS_SYMBOL                = os.getenv("RS_SYMBOL","SPY").upper()

# --- Rolling confidence quantile gate (dynamic threshold)
CONF_ROLL_N              = int(os.getenv("CONF_ROLL_N","0"))          # 0 = off
CONF_Q                   = float(os.getenv("CONF_Q","0.8"))

# --- Advanced modeling toggles (default OFF) ---
# 1) Triple-barrier labeling & meta-labeling
TB_ENABLED               = os.getenv("TB_ENABLED","0").lower() in ("1","true","yes")
TB_ATR_K                 = float(os.getenv("TB_ATR_K","1.0"))
TB_TIMEOUT_H             = int(os.getenv("TB_TIMEOUT_H","5"))  # bars
META_ENABLED             = os.getenv("META_ENABLED","0").lower() in ("1","true","yes")
# 2) Signal persistence
PERSIST_BARS             = int(os.getenv("PERSIST_BARS","0"))  # 0=off, 2=two-bar confirm
# 3) Purged CV with embargo for calibration
PURGED_CV_FOLDS          = int(os.getenv("PURGED_CV_FOLDS","0"))  # 0=TimeSeriesSplit
PURGED_EMBARGO_FRAC      = float(os.getenv("PURGED_EMBARGO_FRAC","0.01"))
# 5) Mixed horizons ensemble
MULTI_H_ENABLED          = os.getenv("MULTI_H_ENABLED","0").lower() in ("1","true","yes")
MULTI_H_LIST             = [int(x) for x in os.getenv("MULTI_H_LIST","3,5,10").split(",") if x.strip()]
# 6) Cost-aware selection/training
COST_BPS                 = float(os.getenv("COST_BPS","2"))      # round-trip cost in bps
COST_WEIGHTING           = int(os.getenv("COST_WEIGHTING","0"))   # 0=off
# 7) Portfolio constraints
MAX_PER_SECTOR           = int(os.getenv("MAX_PER_SECTOR","0"))  # 0=off
MAX_CAND_CORR            = float(os.getenv("MAX_CAND_CORR","1.00"))
CORR_LOOKBACK            = int(os.getenv("CORR_LOOKBACK","40"))
# 8) Drift monitoring / auto-response
DRIFT_MONITOR            = os.getenv("DRIFT_MONITOR","0").lower() in ("1","true","yes")
AUTO_THR_STEP            = float(os.getenv("AUTO_THR_STEP","0.02"))
AUTO_THR_MAX             = float(os.getenv("AUTO_THR_MAX","0.95"))
MIN_BUCKET_WIN           = float(os.getenv("MIN_BUCKET_WIN","0.55"))

# ---- Shorting toggles ----
ENABLE_SHORTS            = os.getenv("ENABLE_SHORTS","1").lower() in ("1","true","yes")
SHORT_CONF_THR           = float(os.getenv("SHORT_CONF_THR", str(CONF_THR)))

# Regime gate
REGIME_ENABLED           = os.getenv("REGIME_ENABLED","0").lower() in ("1","true","yes")
REGIME_MA                = int(os.getenv("REGIME_MA","200"))
REGIME_MIN_RVOL          = float(os.getenv("REGIME_MIN_RVOL","0.005"))  # 0.5%
REGIME_MAX_RVOL          = float(os.getenv("REGIME_MAX_RVOL","0.05"))   # 5%

# --- Caching (OFF by default) ---
CACHE_ENABLED        = os.getenv("CACHE_ENABLED","0").lower() in ("1","true","yes")
CACHE_MAX_MINUTES    = int(os.getenv("CACHE_MAX_MINUTES","5000"))   # ~8–9 RTH days
CACHE_MAX_SYMBOLS    = int(os.getenv("CACHE_MAX_SYMBOLS","2000"))   # LRU cap
CACHE_PERSIST_PATH   = os.getenv("CACHE_PERSIST_PATH","")           # optional (unused here)

# Render boot info
RENDER_GIT_COMMIT = os.getenv("RENDER_GIT_COMMIT", "unknown")[:12]
RENDER_GIT_BRANCH = os.getenv("RENDER_GIT_BRANCH", os.getenv("BRANCH", "unknown"))
RUN_ID            = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")

# Runtime-adjustable threshold (for drift auto)
CONF_THR_RUNTIME = CONF_THR

COUNTS_STAGE = defaultdict(int)

# --- Shortable cache (for proactive short filtering) ---
SHORTABLE_SET = set()
SHORTABLE_LAST = None
_SHORTABLE_ONEOFF = {}

# ==============================
# STATE / LEDGERS
# ==============================
COUNTS        = defaultdict(int)
COMBO_COUNTS  = defaultdict(int)
PERF          = {}
OPEN_TRADES   = defaultdict(list)
_sent_keys    = set()
_order_times  = deque()
LAST_PRICE    = {}

DAY_STAMP     = datetime.now().astimezone().strftime("%Y-%m-%d")
HALT_TRADING  = False

SESSION_BASELINE_SET  = False
SESSION_START_EQUITY  = None
EQUITY_HIGH_WATER     = None

_round_robin = 0

# Extra runtime state
_ML_CACHE   = {}     # (symbol, tf, last_ts_iso, last_close) -> (ts, proba_up, pred_up)
_PROBA_EMA  = {}     # (symbol, tf) -> smoothed probability
_CONF_HIST  = {}     # (symbol, tf) -> deque of recent confidences
_REF_BARS_1M = {}    # symbol -> recent 1m bars for features / regime
_PERSIST_OK = {}     # (symbol, tf) -> consecutive over-threshold count
_RELIABILITY = {     # confidence buckets: (wins, total)
    (0.70,0.75): [0,0], (0.75,0.80): [0,0], (0.80,0.85): [0,0], (0.85,0.90): [0,0], (0.90,1.01): [0,0]
}
# Short-side smoothing & persistence state
_PROBA_EMA_SHORT = {}    # (symbol, tf) -> smoothed short probability
_PERSIST_OK_S    = {}    # (symbol, tf) -> consecutive bars over short threshold

# In-memory caches
_BARCACHE: "OrderedDict[str, pd.DataFrame]" = OrderedDict()   # 1m bars per symbol (tz-aware)
_RESAMPLE_CACHE: dict[tuple[str,int], tuple[pd.Timestamp, pd.DataFrame]] = {}

# per-batch stage counts (pipeline gates) and per-model reasons
from collections import defaultdict
COUNTS_STAGE = defaultdict(int)   # already used by SCAN-AUDIT; harmless to reassign
COUNTS_MODEL = defaultdict(int)   # used by MODEL-AUDIT inside _ml_features_and_pred_core

# --- Broker constraint helpers & runtime denylist ---
MIN_TICK = float(os.getenv("MIN_TICK", "0.01"))  # stock tick size
SHORT_DENY = set()  # symbols we learned are not shortable (lifetime: this process)

def _tick_round(px: float, tick: float = MIN_TICK) -> float:
    if not np.isfinite(px): return px
    q = round(px / tick) * tick
    return float(f"{q:.4f}")

def _apply_tick_rules(side: str, base: float, tp: float, sl: float,
                      tick: float = MIN_TICK, extra_ticks: int = EXTRA_TICKS):
    """
    Enforce Alpaca bracket constraints with directional rounding vs broker base_price.
    Long:  TP >= base + buf (ceil), SL <= base - buf (floor)
    Short: TP <= base - buf (floor), SL >= base + buf (ceil)
    """
    import math
    buf = tick * max(1, int(extra_ticks))

    def _ceil_to_tick(x: float, t: float) -> float:
        return math.ceil(x / t) * t

    def _floor_to_tick(x: float, t: float) -> float:
        return math.floor(x / t) * t

    if side == "long":
        tp = max(tp, base + buf); sl = min(sl, base - buf)
        tp = _ceil_to_tick(tp, tick); sl = _floor_to_tick(sl, tick)
    else:  # short
        tp = min(tp, base - buf); sl = max(sl, base + buf)
        tp = _floor_to_tick(tp, tick); sl = _ceil_to_tick(sl, tick)

    return float(tp), float(sl)

def _is_shortable(symbol: str) -> bool:
    """
    Hard override if FORCE_ALLOW_SHORTS=1. Otherwise, use cached/broker check.
    """
    if os.getenv("FORCE_ALLOW_SHORTS", "0").lower() in ("1", "true", "yes"):
        return True
    s = symbol.upper()
    if SHORTABLE_SET and s in SHORTABLE_SET: return True
    if s in _SHORTABLE_ONEOFF: return _SHORTABLE_ONEOFF[s]
    try:
        base = os.getenv("ALPACA_TRADE_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
        url = f"{base}/v2/assets/{s}"
        headers = {
            "APCA-API-KEY-ID": os.getenv("APCA_API_KEY_ID") or os.getenv("APCA-API-KEY-ID") or "",
            "APCA-API-SECRET-KEY": os.getenv("APCA_API_SECRET_KEY") or os.getenv("APCA-API-SECRET-KEY") or "",
        }
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            flag = bool(data.get("shortable", False)) and bool(data.get("tradable", False))
            _SHORTABLE_ONEOFF[s] = flag
            return flag
    except Exception:
        pass
    _SHORTABLE_ONEOFF[s] = False
    return False

# ==============================
# DATA MODEL
# ==============================
class LiveTrade:
    def __init__(self, combo, symbol, tf_min, side, entry, tp, sl, qty, entry_time, proba=None):
        self.combo = combo
        self.symbol = symbol
        self.tf_min = int(tf_min)
        self.side = side
        self.entry = float(entry) if entry is not None else float("nan")
        self.tp = float(tp) if tp is not None else float("nan")
        self.sl = float(sl) if sl is not None else float("nan")
        self.qty = int(qty)
        self.entry_time = entry_time
        self.is_open = True
        self.exit = None
        self.exit_time = None
        self.reason = None
        self.proba = proba

# ==============================
# TIME / SESSION HELPERS
# ==============================
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

def _market_session_now():
    ts = _now_et()
    return _in_session(ts)

# ==============================
# MATH / SIZING
# ==============================
def _position_qty(entry_price: float, stop_price: float,
                  equity=EQUITY_USD, risk_pct=RISK_PCT, max_pos_pct=MAX_POS_PCT,
                  min_qty=MIN_QTY, round_lot=ROUND_LOT) -> int:
    if entry_price is None: return 0
    if USE_FIXED_NOTIONAL:
        if entry_price <= 0: return 0
        raw = FIXED_NOTIONAL / max(1e-9, float(entry_price))
        qty = math.floor(raw / max(1, round_lot)) * max(1, round_lot)
        return int(max(qty, min_qty if qty > 0 else 0))
    if stop_price is None: return 0
    risk_per_share = abs(float(entry_price) - float(stop_price))
    if risk_per_share <= 0: return 0
    qty_risk     = (equity * risk_pct) / risk_per_share
    qty_notional = (equity * max_pos_pct) / max(1e-9, float(entry_price))
    qty = math.floor(max(min(qty_risk, qty_notional), 0) / max(1, round_lot)) * max(1, round_lot)
    return int(max(qty, min_qty if qty > 0 else 0))

# ==============================
# UNIVERSE + DATA (Alpaca)
# ==============================
def build_universe() -> list[str]:
    manual = os.getenv("SCANNER_SYMBOLS", "").strip()
    if manual:
        syms = [s.strip().upper() for s in manual.split(",") if s.strip()]
        print(f"[BOOT] SCANNER_SYMBOLS override detected ({len(syms)}).", flush=True)
        return syms
    syms = _alpaca_universe(limit=10000)
    print(f"[UNIVERSE] fetched {len(syms)} tickers via Alpaca assets.", flush=True)
    if SCANNER_MIN_AVG_VOL:
        print("[UNIVERSE] NOTE: no Polygon prefilter under Alpaca.", flush=True)
    return syms

def fetch_bars_1m(symbol: str, lookback_minutes: int = 2400) -> pd.DataFrame:
    """
    Cached wrapper around Alpaca minute bars.
    - When CACHE_ENABLED=1: incrementally updates the per-symbol 1m cache, then returns the requested window.
    - When off: direct API call (original behavior).
    Returns tz-aware ET ohlcv DataFrame (open/high/low/close/volume).
    """
    if not CACHE_ENABLED:
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=lookback_minutes)
        start_iso = start.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        end_iso   = end.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        df = _alpaca_fetch_1m(symbol, start_iso=start_iso, end_iso=end_iso, limit=10000)
        if df is None or df.empty: return pd.DataFrame()
        try:
            df.index = df.index.tz_convert(MARKET_TZ)
        except Exception:
            df.index = df.index.tz_localize("UTC").tz_convert(MARKET_TZ)
        try:
            last_px = float(df["close"].iloc[-1])
            if last_px < MIN_PRICE: return pd.DataFrame()
        except Exception:
            return pd.DataFrame()
        return df[["open","high","low","close","volume"]].copy()

    # ----- Cached path -----
    def _touch(sym: str):
        _BARCACHE.setdefault(sym, pd.DataFrame())
        _BARCACHE.move_to_end(sym, last=True)
        while len(_BARCACHE) > max(1, CACHE_MAX_SYMBOLS):
            _BARCACHE.popitem(last=False)

    cached = _BARCACHE.get(symbol)
    if cached is None or cached.empty:
        end = datetime.now(timezone.utc)
        need_minutes = min(max(1, lookback_minutes), CACHE_MAX_MINUTES)
        start = end - timedelta(minutes=need_minutes)
        start_iso = start.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        end_iso   = end.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        df_new = _alpaca_fetch_1m(symbol, start_iso=start_iso, end_iso=end_iso, limit=10000)
        if df_new is None or df_new.empty: return pd.DataFrame()
        try:
            df_new.index = df_new.index.tz_convert(MARKET_TZ)
        except Exception:
            df_new.index = df_new.index.tz_localize("UTC").tz_convert(MARKET_TZ)
        cached = df_new
    else:
        try:
            last_ts = cached.index[-1].tz_convert("UTC") if hasattr(cached.index[-1], "tzinfo") else cached.index[-1]
        except Exception:
            last_ts = None
        end = datetime.now(timezone.utc)
        start = (last_ts + timedelta(minutes=1)).replace(tzinfo=timezone.utc) if last_ts is not None else end - timedelta(minutes=min(lookback_minutes, CACHE_MAX_MINUTES))
        if start < end:
            start_iso = start.replace(microsecond=0).isoformat().replace("+00:00", "Z")
            end_iso   = end.replace(microsecond=0).isoformat().replace("+00:00", "Z")
            df_inc = _alpaca_fetch_1m(symbol, start_iso=start_iso, end_iso=end_iso, limit=10000)
            if df_inc is not None and not df_inc.empty:
                try:
                    df_inc.index = df_inc.index.tz_convert(MARKET_TZ)
                except Exception:
                    df_inc.index = df_inc.index.tz_localize("UTC").tz_convert(MARKET_TZ)
                cached = pd.concat([cached, df_inc], axis=0)
                cached = cached[~cached.index.duplicated(keep="last")].sort_index()

    if len(cached) > 0 and CACHE_MAX_MINUTES > 0:
        cutoff = cached.index[-1] - timedelta(minutes=CACHE_MAX_MINUTES)
        cached = cached[cached.index >= cutoff]

    _BARCACHE[symbol] = cached
    _touch(symbol)

    try:
        last_px = float(cached["close"].iloc[-1])
        if last_px < MIN_PRICE: return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    if lookback_minutes > 0 and len(cached) > 0:
        cutoff2 = cached.index[-1] - timedelta(minutes=lookback_minutes)
        out = cached[cached.index >= cutoff2]
    else:
        out = cached

    return out[["open","high","low","close","volume"]].copy()

def _resample(df1m: pd.DataFrame, tf_min: int) -> pd.DataFrame:
    """
    Cache-aware resample:
    - If we’ve already resampled this (symbol, tf) source slice and no new 1m rows are added, return cached.
    """
    if df1m is None or df1m.empty:
        return pd.DataFrame()
    src_last = df1m.index[-1]
    cache_key = (id(df1m), int(tf_min))
    cached = _RESAMPLE_CACHE.get(cache_key)
    if cached and cached[0] is not None and cached[0] == src_last:
        return cached[1]
    rule = f"{int(tf_min)}min"
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    bars = df1m.resample(rule, origin="start_day", label="right").agg(agg).dropna()
    try:
        bars.index = bars.index.tz_convert(MARKET_TZ)
    except Exception:
        bars.index = bars.index.tz_localize("UTC").tz_convert(MARKET_TZ)
    _RESAMPLE_CACHE[cache_key] = (src_last, bars)
    return bars

# ==============================
# PERF TRACK
# ==============================
def _combo_key(strategy: str, symbol: str, tf_min: int) -> str:
    return f"{strategy}|{symbol}|{int(tf_min)}"

def _perf_init(combo: str):
    if combo not in PERF:
        PERF[combo] = {"trades":0,"wins":0,"losses":0,"gross_profit":0.0,"gross_loss":0.0,"net_pnl":0.0,"max_dd":0.0,"equity_curve":[0.0]}

def _perf_update(combo: str, pnl: float):
    _perf_init(combo)
    p = PERF[combo]
    p["trades"] += 1
    if pnl > 0: p["wins"] += 1; p["gross_profit"] += pnl
    elif pnl < 0: p["losses"] += 1; p["gross_loss"] += pnl
    p["net_pnl"] += pnl
    ec = p["equity_curve"]; ec.append(ec[-1] + pnl)
    dd = min(0.0, ec[-1] - max(ec)); p["max_dd"] = min(p["max_dd"], dd)

def _record_open_trade(strat_name: str, symbol: str, tf_min: int, sig: dict):
    combo = _combo_key(strat_name, symbol, tf_min)
    _perf_init(combo)
    tp = sig.get("tp_abs", sig.get("takeProfit"))
    sl = sig.get("sl_abs", sig.get("stopLoss"))
    side_norm = "sell" if sig["action"] in ("sell","sell_short") else "buy"
    t = LiveTrade(
        combo=combo, symbol=symbol, tf_min=int(tf_min), side=side_norm,
        entry=float(sig.get("entry") or sig.get("price") or sig.get("lastClose") or 0.0),
        tp=float(tp) if tp is not None else float("nan"),
        sl=float(sl) if sl is not None else float("nan"),
        qty=int(sig["quantity"]),
        entry_time=sig.get("barTime") or datetime.now(timezone.utc).isoformat(),
        proba=float(sig.get("meta",{}).get("proba_up", np.nan))
    )
    OPEN_TRADES[(symbol, int(tf_min))].append(t)

def _maybe_close_on_bar(symbol: str, tf_min: int, ts, high: float, low: float, close: float):
    key = (symbol, int(tf_min))
    trades = OPEN_TRADES.get(key, [])
    for t in trades:
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
            try:
                if DRIFT_MONITOR and t.proba is not None and np.isfinite(t.proba):
                    p = float(t.proba)
                    for (lo, hi), agg in _RELIABILITY.items():
                        if lo <= p < hi:
                            agg[1] += 1
                            if hit_tp: agg[0] += 1
                            break
            except Exception:
                pass
            print(f"[CLOSE] {t.combo} {t.reason.upper()} qty={t.qty} entry={t.entry:.2f} exit={t.exit:.2f} pnl={pnl:+.2f}", flush=True)

# ==============================
# DE-DUPE
# ==============================
def _dedupe_key(symbol: str, tf: int, action: str, bar_time: str) -> str:
    raw = f"{symbol}|{tf}|{action}|{bar_time}|{RUN_ID}"
    import hashlib
    return hashlib.sha256(raw.encode()).hexdigest()

# ==============================
# ML STRATEGY (Upgraded v3)
# ==============================
_H = 5
_RET_THR = 0.0025
_MIN_SAMPLES = int(os.getenv("ML_MIN_SAMPLES", "240"))

_FEATURES = [
    "ret1","ret3","ret5","ret10",
    "vol10","vol20",
    "rsi14","stoch_k",
    "ema10_slope","ema20_slope",
    "atrp","body_tr","dvolz",
    "vwap_dist","gap_pct","rs"
]

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret1"]  = out["close"].pct_change(1)
    out["ret3"]  = out["close"].pct_change(3)
    out["ret5"]  = out["close"].pct_change(5)
    out["ret10"] = out["close"].pct_change(10)
    out["vol10"] = out["close"].pct_change().rolling(10).std()
    out["vol20"] = out["close"].pct_change().rolling(20).std()
    delta = out["close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = up / (down.replace(0, np.nan))
    out["rsi14"] = 100 - (100 / (1 + rs))
    hh = out["high"].rolling(14).max()
    ll = out["low"].rolling(14).min()
    out["stoch_k"] = 100 * (out["close"] - ll) / (hh - ll).replace(0, np.nan)
    out["stoch_k"] = out["stoch_k"].rolling(3).mean()
    ema10 = out["close"].ewm(span=10, adjust=False).mean()
    ema20 = out["close"].ewm(span=20, adjust=False).mean()
    out["ema10_slope"] = ema10.pct_change()
    out["ema20_slope"] = ema20.pct_change()
    tr = pd.concat([
        (out["high"] - out["low"]),
        (out["high"] - out["close"].shift()).abs(),
        (out["low"] - out["close"].shift()).abs()
    ], axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14).mean()
    out["atrp"]  = out["atr14"] / out["close"]
    body = (out["close"] - out["open"]).abs()
    rng  = (out["high"] - out["low"]).replace(0, np.nan)
    out["body_tr"] = (body / rng).clip(0, 5)
    if "volume" in out.columns:
        out["dvol20"] = (out["close"] * out["volume"]).rolling(20).mean()
        out["dvolz"]  = (out["dvol20"] / out["dvol20"].rolling(60).mean()) - 1.0
    else:
        out["dvolz"] = 0.0
    try:
        vwap = (out["close"] * out.get("volume", pd.Series(index=out.index))).cumsum() / out.get("volume", pd.Series(index=out.index)).replace(0,np.nan).cumsum()
        out["vwap_dist"] = (out["close"] / vwap) - 1.0
    except Exception:
        out["vwap_dist"] = 0.0
    out["gap_pct"] = out["open"] / out["close"].shift(1) - 1.0
    try:
        ref = _REF_BARS_1M.get(RS_SYMBOL)
        if ref is not None and not ref.empty:
            ref = ref.reindex(out.index, method="pad")
            out["rs"] = out["close"].pct_change(10, fill_method=None) - ref["close"].pct_change(10, fill_method=None)
        else:
            out["rs"] = 0.0
    except Exception:
        out["rs"] = 0.0
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out

def _make_labels_basic(out: pd.DataFrame, h=_H, thr=_RET_THR):
    fwd = out["close"].shift(-h) / out["close"] - 1.0
    y = (fwd > thr).astype(int)
    return y

def _make_labels_tb(out: pd.DataFrame, h=_H, atr_k=1.0):
    price = out["close"]
    tr = pd.concat([
        (out["high"] - out["low"]),
        (out["high"] - out["close"].shift()).abs(),
        (out["low"] - out["close"].shift()).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    fut_high = out["high"].shift(-1).rolling(h).max()
    fut_low  = out["low"].shift(-1).rolling(h).min()
    tp_lvl = price + atr_k * atr14
    sl_lvl = price - atr_k * atr14
    hit_tp = (fut_high >= tp_lvl)
    hit_sl = (fut_low  <= sl_lvl)
    y = (hit_tp & ~hit_sl).astype(int)
    return y

class PurgedKFold:
    """Simple purged K-fold with embargo for time series calibration."""
    def __init__(self, n_splits=3, embargo_frac=0.01):
        self.n_splits = max(2, int(n_splits))
        self.embargo_frac = float(embargo_frac)
    def split(self, X):
        n = len(X)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        indices = np.arange(n)
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            emb = int(self.embargo_frac * n)
            train_mask = np.ones(n, dtype=bool)
            train_mask[start:stop] = False
            post = min(n, stop + emb)
            train_mask[stop:post] = False
            train_idx = indices[train_mask]
            current = stop
            yield train_idx, test_idx

def _fit_base_model(X_trn, y_trn):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    n = len(X_trn)
    hl = max(8, int(n * 0.25))
    lam = np.log(2) / hl
    w = np.exp(lam * (np.arange(n) - n))
    w = w / w.mean()
    rf.fit(X_trn, y_trn, sample_weight=w)
    return rf

def _calibrate(clf, X_trn, y_trn):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit
    import numpy as np
    try:
        nuniq = int(y_trn.nunique()) if getattr(y_trn, "nunique", None) else len(np.unique(y_trn))
        if nuniq < 2:
            return clf
        if PURGED_CV_FOLDS and PURGED_CV_FOLDS >= 2:
            cv_splits = list(PurgedKFold(PURGED_CV_FOLDS, PURGED_EMBARGO_FRAC).split(X_trn))
        else:
            cv_splits = list(TimeSeriesSplit(n_splits=3).split(X_trn))
        valid = []
        for tr, te in cv_splits:
            y_tr, y_te = y_trn.iloc[tr], y_trn.iloc[te]
            if len(np.unique(y_tr)) >= 2 and len(np.unique(y_te)) >= 2:
                valid.append((tr, te))
        if not valid:
            return clf
        cal = CalibratedClassifierCV(clf, method="isotonic", cv=valid)
        cal.fit(X_trn, y_trn)
        return cal
    except Exception as e:
        print(f"[ML] Calibration fallback (using base model): {e}", flush=True)
        return clf

def _ml_features_and_pred_core(bars: pd.DataFrame, h: int):
    if bars is None or bars.empty or len(bars) < 300:
        return None, None, None
    df = _build_features(bars)
    if len(df) < _MIN_SAMPLES:
        return None, None, None
    X = df[_FEATURES].copy()
    y = _make_labels_tb(df, h, TB_ATR_K) if TB_ENABLED else _make_labels_basic(df, h, _RET_THR)
    valid = y.notna()
    X, y = X[valid], y[valid]
    if len(X) < _MIN_SAMPLES:
        return None, None, None
    x_live = X.iloc[[-1]]
    X_trn, y_trn = X.iloc[:-1], y.iloc[:-1]
    if len(X_trn) < _MIN_SAMPLES:
        return None, None, None
    base = _fit_base_model(X_trn, y_trn)
    try:
        classes_ = getattr(base, "classes_", None)
        if classes_ is not None and len(classes_) < 2:
            p_live = base.predict_proba(x_live)
            if p_live.shape[1] == 1:
                only_cls = int(classes_[0])
                proba_up = 1.0 if only_cls == 1 else 0.0
                pred_up  = int(proba_up >= 0.5)
                ts = df.index[-1]
                return ts, float(proba_up), int(pred_up)
    except Exception:
        pass
    clf = _calibrate(base, X_trn, y_trn)
    proba_mat = clf.predict_proba(x_live)
    if proba_mat.shape[1] == 1:
        classes_ = getattr(clf, "classes_", [1])
        only_cls = int(classes_[0]) if len(classes_) else 1
        proba_up = 1.0 if only_cls == 1 else 0.0
    else:
        proba_up = float(proba_mat[0, 1])
    pred_up  = int(proba_up >= 0.5)
    ts = df.index[-1]
    return ts, proba_up, pred_up

def _ml_features_and_pred(bars: pd.DataFrame):
    if MULTI_H_ENABLED:
        ps = []; last_r = None
        for h in MULTI_H_LIST:
            r = _ml_features_and_pred_core(bars, h)
            if r[0] is None: continue
            ps.append(r[1]); last_r = r
        if not ps: return None, None, None
        p = float(np.exp(np.mean(np.log(np.clip(ps, 1e-6, 1-1e-6)))))
        ts, _, _ = last_r
        return ts, p, int(p >= 0.5)
    else:
        return _ml_features_and_pred_core(bars, _H)

# ==============================
# AUDIT & FAILSAFE FOR LONGS
# ==============================
_AUDIT = {
    "scanned": 0,        # symbols * TFs examined
    "model_ok": 0,       # got p_up
    "p_up_ge_050": 0,    # p_up >= 0.50
    "p_up_ge_thr": 0,    # p_up >= current long threshold
    "long_ok": 0,        # passed all long gates (pre-selection)
    "last_print": 0.0,
}

def _audit_reset():
    for k in list(_AUDIT.keys()):
        if k != "last_print": _AUDIT[k] = 0

def _audit_note(p_up, long_ok, thr_long):
    _AUDIT["scanned"] += 1
    if p_up is not None:
        _AUDIT["model_ok"] += 1
        if p_up >= 0.50:
            _AUDIT["p_up_ge_050"] += 1
        if p_up >= thr_long:
            _AUDIT["p_up_ge_thr"] += 1
    if long_ok:
        _AUDIT["long_ok"] += 1

def _audit_maybe_print(thr_long):
    import time as _t
    now = _t.time()
    if now - _AUDIT["last_print"] < 15:    # print at most every 15s
        return
    _AUDIT["last_print"] = now
    S  = max(1, _AUDIT["scanned"])
    mo = _AUDIT["model_ok"]
    ge50 = _AUDIT["p_up_ge_050"]
    gethr= _AUDIT["p_up_ge_thr"]
    lok  = _AUDIT["long_ok"]
    print(f"[AUDIT] scanned={S} model_ok={mo} p>=.50={ge50} p>=thr={gethr} long_ok={lok} thr_long={thr_long:.3f}", flush=True)

def _failsafe_adjust_long_threshold(curr_thr, session_minutes):
    """
    If we’ve scanned a lot and long_ok is zero, temporarily relax the long threshold.
    Only activates if p_up >= .50 exists for >=1% of scans (i.e., model is alive).
    """
    S = _AUDIT["scanned"]; lok = _AUDIT["long_ok"]; ge50 = _AUDIT["p_up_ge_050"]
    if S < 200:  # wait for some mass
        return curr_thr
    if lok > 0:
        return curr_thr
    if ge50 < 0.01 * S:
        return curr_thr

    if session_minutes <= 30:
        return min(curr_thr, 0.58)
    elif session_minutes <= 90:
        return min(curr_thr, 0.62)
    else:
        return min(curr_thr, 0.66)

# ============================================================
# SHORT/LONG SIGNAL (dual-side wrapper) — with audit + market bias
# ============================================================
def signal_ml_pattern_dual(symbol: str, df1m: pd.DataFrame, tf_min: int,
                           conf_threshold=CONF_THR, r_multiple=R_MULT, atr_k=1.0):
    dbg = DEBUG_DUMP_SIG
    gates = []  # reasons for skips (debug only)

    if df1m is None or df1m.empty or not isinstance(df1m.index, pd.DatetimeIndex):
        COUNTS_STAGE["01_df_empty"] += 1
        if dbg: gates.append("df_empty")
        return None

    bars = _resample(df1m, tf_min)
    if bars is None or bars.empty:
        COUNTS_STAGE["02_resample_empty"] += 1
        if dbg: gates.append("resample_empty")
        return None

    # --- Per-bar model cache on this TF ---
    last_ts = bars.index[-1]
    last_ts_iso = last_ts.tz_convert("UTC").isoformat() if hasattr(last_ts, "tzinfo") else str(last_ts)
    cache_key = (symbol, int(tf_min), last_ts_iso, float(bars["close"].iloc[-1]))
    cached = _ML_CACHE.get(cache_key)
    if cached:
        ts, proba_up, pred_up = cached
    else:
        ts, proba_up, pred_up = _ml_features_and_pred(bars)
        if ts is not None:
            _ML_CACHE[cache_key] = (ts, proba_up, pred_up)

    if ts is None or proba_up is None:
        COUNTS_STAGE["03_model_none"] += 1
        if dbg: gates.append("model_none")
        return None
    if not _in_session(ts):
        COUNTS_STAGE["04_off_session"] += 1
        if dbg: gates.append("off_session")
        return None

    # === Market tilt based on RS_SYMBOL 60-bar return sign ===
    TILT_ON        = os.getenv("TILT_ON", "1").lower() in ("1","true","yes")
    TILT_RETMIN    = float(os.getenv("TILT_RET_MIN", "0.003"))  # 0.3%
    TILT_DELTA     = float(os.getenv("TILT_DELTA", "0.12"))
    TILT_LONG_BIAS = os.getenv("TILT_LONG_BIAS", "1").lower() in ("1","true","yes")
    TILT_SHORT_BIAS= os.getenv("TILT_SHORT_BIAS","1").lower() in ("1","true","yes")
    BIAS_DELTA     = float(os.getenv("BIAS_DELTA","0.05"))

    tilt = 0
    if TILT_ON:
        ref = _REF_BARS_1M.get(RS_SYMBOL, pd.DataFrame())
        if ref is not None and not ref.empty and len(ref) > 60:
            try:
                ret_60 = float(ref["close"].iloc[-1] / ref["close"].iloc[-60] - 1.0)
                if abs(ret_60) >= TILT_RETMIN:
                    tilt = 1 if ret_60 > 0 else -1
            except Exception:
                tilt = 0

    # --- helper smoothing ---
    def _smooth(store: dict, key, p, alpha: float):
        if alpha <= 0: return p
        p_prev = store.get(key, p)
        p_s = alpha * p + (1 - alpha) * p_prev
        store[key] = p_s
        return p_s

    key_pt = (symbol, int(tf_min))
    p_raw  = float(proba_up)

    # ---------------- LONG branch ----------------
    p_long = _smooth(_PROBA_EMA, key_pt, p_raw, PROBA_EMA_ALPHA)

    # Higher-TF consensus for LONGS (require p2 >= threshold if enabled)
    if CONSENSUS_TF and CONSENSUS_TF != tf_min:
        bars_hi = _resample(df1m, CONSENSUS_TF)
        if bars_hi is None or bars_hi.empty:
            COUNTS_STAGE["09_consensus_resample_empty_L"] += 1
            p_long = -1
            if dbg: gates.append("cons_long_empty")
        else:
            last_ts2 = bars_hi.index[-1]
            last_ts2_iso = last_ts2.tz_convert("UTC").isoformat() if hasattr(last_ts2, "tzinfo") else str(last_ts2)
            cache_key2 = (symbol, int(CONSENSUS_TF), last_ts2_iso, float(bars_hi["close"].iloc[-1]))
            cached2 = _ML_CACHE.get(cache_key2)
            if cached2:
                ts2, p2, _ = cached2
            else:
                ts2, p2, _ = _ml_features_and_pred(bars_hi)
                if ts2 is not None:
                    _ML_CACHE[cache_key2] = (ts2, p2, int(p2 >= 0.5))
            if ts2 is None or p2 is None:
                COUNTS_STAGE["10_consensus_model_none_L"] += 1
                p_long = -1
                if dbg: gates.append("cons_long_none")
            else:
                thr_cons_long = max(CONF_THR_RUNTIME, conf_threshold)
                if float(p2) < thr_cons_long:
                    COUNTS_STAGE["11_consensus_pred_down_L"] += 1
                    p_long = -1
                    if dbg: gates.append(f"cons_long_p2<{thr_cons_long:.2f}")
                else:
                    p_long = float((p_long ** (1 - CONSENSUS_WEIGHT)) * (float(p2) ** CONSENSUS_WEIGHT))

    # Rolling dynamic quantile gate for LONGS
    if p_long >= 0 and CONF_ROLL_N > 0:
        dq = _CONF_HIST.get(("L",) + key_pt)
        if dq is None:
            dq = deque(maxlen=CONF_ROLL_N)
            _CONF_HIST[("L",) + key_pt] = dq
        if len(dq) >= max(10, int(0.5 * CONF_ROLL_N)):
            qthr = float(np.quantile(np.array(dq), CONF_Q))
            if p_long < qthr:
                COUNTS_STAGE["12_below_dyn_quantile_L"] += 1
                p_long = -1
                if dbg: gates.append(f"long<q{CONF_Q:.2f}")
        if p_long >= 0:
            dq.append(p_long)

    # Regime gate for LONGS
    if REGIME_ENABLED and p_long >= 0:
        ref = _REF_BARS_1M.get(RS_SYMBOL, pd.DataFrame())
        if ref is None or ref.empty or len(ref) < max(REGIME_MA + 5, 60):
            COUNTS_STAGE["05_regime_ref_missing_L"] += 1
            p_long = -1
            if dbg: gates.append("regime_ref_missing_L")
        else:
            ema = ref["close"].ewm(span=REGIME_MA, adjust=False).mean().iloc[-1]
            above = float(ref["close"].iloc[-1]) > float(ema)
            rvol = ref["close"].pct_change().rolling(30).std().iloc[-1]
            if not (above and REGIME_MIN_RVOL <= float(rvol) <= REGIME_MAX_RVOL):
                COUNTS_STAGE["06_regime_fail_L"] += 1
                p_long = -1
                if dbg: gates.append("regime_fail_L")

    # ---------------- SHORT branch ----------------
    p_short = -1.0
    if ENABLE_SHORTS and (symbol not in SHORT_DENY):
        p_down_raw = 1.0 - p_raw
        p_short = _smooth(_PROBA_EMA_SHORT, key_pt, p_down_raw, PROBA_EMA_ALPHA)

        if CONSENSUS_TF and CONSENSUS_TF != tf_min:
            bars_hi = _resample(df1m, CONSENSUS_TF)
            if bars_hi is None or bars_hi.empty:
                p_short = -1
                if dbg: gates.append("cons_short_empty")
            else:
                last_ts2 = bars_hi.index[-1]
                last_ts2_iso = last_ts2.tz_convert("UTC").isoformat() if hasattr(last_ts2, "tzinfo") else str(last_ts2)
                cache_key2s = (symbol, int(CONSENSUS_TF), last_ts2_iso, float(bars_hi["close"].iloc[-1]))
                cached2s = _ML_CACHE.get(cache_key2s)
                if cached2s:
                    ts2s, p2s, _ = cached2s
                else:
                    ts2s, p2s, _ = _ml_features_and_pred(bars_hi)
                    if ts2s is not None:
                        _ML_CACHE[cache_key2s] = (ts2s, p2s, int(p2s >= 0.5))
                if (ts2s is None) or (p2s is None):
                    p_short = -1
                    if dbg: gates.append("cons_short_none")
                else:
                    thr_cons_short = max(CONF_THR_RUNTIME, SHORT_CONF_THR)
                    if (1.0 - float(p2s)) < thr_cons_short:
                        p_short = -1
                        if dbg: gates.append(f"cons_short_(1-p2)<{thr_cons_short:.2f}")
                    else:
                        p_short = float((p_short ** (1 - CONSENSUS_WEIGHT)) * ((1.0 - float(p2s)) ** CONSENSUS_WEIGHT))

        if p_short >= 0 and CONF_ROLL_N > 0:
            dqS = _CONF_HIST.get(("S",) + key_pt)
            if dqS is None:
                dqS = deque(maxlen=CONF_ROLL_N)
                _CONF_HIST[("S",) + key_pt] = dqS
            if len(dqS) >= max(10, int(0.5 * CONF_ROLL_N)):
                qthrS = float(np.quantile(np.array(dqS), CONF_Q))
                if p_short < qthrS:
                    p_short = -1
                    if dbg: gates.append(f"short<q{CONF_Q:.2f}")
            if p_short >= 0:
                dqS.append(p_short)

        if REGIME_ENABLED and p_short >= 0:
            ref = _REF_BARS_1M.get(RS_SYMBOL, pd.DataFrame())
            if ref is None or ref.empty or len(ref) < max(REGIME_MA + 5, 60):
                p_short = -1
                if dbg: gates.append("regime_ref_missing_S")
            else:
                ema = ref["close"].ewm(span=REGIME_MA, adjust=False).mean().iloc[-1]
                below = float(ref["close"].iloc[-1]) < float(ema)
                rvol = ref["close"].pct_change().rolling(30).std().iloc[-1]
                if not (below and REGIME_MIN_RVOL <= float(rvol) <= REGIME_MAX_RVOL):
                    p_short = -1
                    if dbg: gates.append("regime_fail_S")

    # ---------------- Thresholds (with market tilt) ----------------
    base_thr_long  = max(CONF_THR_RUNTIME, conf_threshold)
    base_thr_short = max(CONF_THR_RUNTIME, SHORT_CONF_THR)

    if TILT_ON and tilt == 1:
        thr_long  = max(0.50, base_thr_long  - TILT_DELTA)
        thr_short = min(0.99, base_thr_short + TILT_DELTA)
    elif TILT_ON and tilt == -1:
        thr_long  = min(0.99, base_thr_long  + TILT_DELTA)
        thr_short = max(0.50, base_thr_short - TILT_DELTA)
    else:
        thr_long, thr_short = base_thr_long, base_thr_short

    # --- FAILSAFE: relax long threshold if absurdly few longs pass early session ---
    try:
        sess_now = _now_et()
        session_minutes = (sess_now.hour - 9) * 60 + (sess_now.minute - 30)
        if 0 <= session_minutes <= 180:
            old_thr_long = thr_long
            thr_long = _failsafe_adjust_long_threshold(thr_long, session_minutes)
            if SCANNER_DEBUG and thr_long < old_thr_long:
                print(f"[FAILSAFE] thr_long relaxed {old_thr_long:.3f} -> {thr_long:.3f}", flush=True)
    except Exception:
        pass

    # ---------------- Persistence & gates ----------------
    long_ok = False
    if p_long >= 0 and p_long >= (thr_long + MIN_PROBA_GAP):
        if PERSIST_BARS >= 2:
            c = _PERSIST_OK.get(("L",) + key_pt, 0) + 1
            _PERSIST_OK[("L",) + key_pt] = c
            long_ok = (c >= PERSIST_BARS)
            if dbg and not long_ok: gates.append("persist_L")
        else:
            long_ok = True
    else:
        if p_long >= 0:
            COUNTS_STAGE["07_below_threshold_L"] += 1
            if dbg: gates.append("below_thr_L")
        _PERSIST_OK[("L",) + key_pt] = 0

    short_ok = False
    if ENABLE_SHORTS and p_short >= 0 and p_short >= (thr_short + MIN_PROBA_GAP):
        if PERSIST_BARS >= 2:
            cS = _PERSIST_OK_S.get(key_pt, 0) + 1
            _PERSIST_OK_S[key_pt] = cS
            short_ok = (cS >= PERSIST_BARS)
            if dbg and not short_ok: gates.append("persist_S")
        else:
            short_ok = True
    else:
        _PERSIST_OK_S[key_pt] = 0
        if ENABLE_SHORTS and p_short >= 0 and dbg:
            gates.append("below_thr_S")

    # --- AUDIT TAP (counts where longs are dying)
    try:
        _audit_note(p_up=(p_long if p_long >= 0 else None),
                    long_ok=bool(long_ok),
                    thr_long=thr_long)
    except Exception:
        pass

    # ---------------- Side selection with market bias ----------------
    side = None
    conf = -1.0
    lm = (p_long  - thr_long)  if long_ok  else -1e9
    sm = (p_short - thr_short) if short_ok else -1e9

    if (tilt == 1) and TILT_LONG_BIAS:
        if long_ok and (not short_ok or sm < lm + BIAS_DELTA):
            side, conf = ("long",  p_long)
        elif short_ok:
            side, conf = ("short", p_short)
    elif (tilt == -1) and TILT_SHORT_BIAS:
        if short_ok and (not long_ok or lm < sm + BIAS_DELTA):
            side, conf = ("short", p_short)
        elif long_ok:
            side, conf = ("long",  p_long)
    else:
        if long_ok and (not short_ok or lm >= sm):
            side, conf = ("long",  p_long)
        elif short_ok:
            side, conf = ("short", p_short)

    # If shorts are disabled, never pick short.
    if not ENABLE_SHORTS and side == "short":
        side = "long" if long_ok else None

    if side is None:
        COUNTS_STAGE["14_no_side_passed"] += 1
        if dbg:
            print(f"[SIG] {symbol} {tf_min}m none pL={p_long:.3f} thrL={thr_long:.3f} "
                  f"pS={p_short:.3f} thrS={thr_short:.3f} tilt={tilt} gates={';'.join(gates)}", flush=True)
        return None

    # ---------------- TP/SL & qty ----------------
    price = float(bars["close"].iloc[-1])
    tr = pd.concat([
        (bars["high"] - bars["low"]),
        (bars["high"] - bars["close"].shift()).abs(),
        (bars["low"] - bars["close"].shift()).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean().iloc[-1]
    if not np.isfinite(atr14) or atr14 <= 0:
        COUNTS_STAGE["13_no_atr"] += 1
        if dbg:
            print(f"[SIG] {symbol} {tf_min}m NO_ATR tilt={tilt} gates={';'.join(gates)}", flush=True)
        return None

    if side == "long":
        sl = price - atr_k * atr14
        tp = price + r_multiple * atr_k * atr14
        tp, sl = _apply_tick_rules("long", price, tp, sl)
        qty = _position_qty(price, sl); action = "buy"
    else:
        if symbol in SHORT_DENY:
            if dbg:
                print(f"[SIG] {symbol} {tf_min}m deny_short tilt={tilt} gates={';'.join(gates)}", flush=True)
            return None
        sl = price + atr_k * atr14
        tp = price - r_multiple * atr_k * atr14
        tp, sl = _apply_tick_rules("short", price, tp, sl)
        qty = _position_qty(price, sl); action = "sell_short"

    if qty <= 0:
        COUNTS_STAGE["15_qty_zero"] += 1
        if dbg:
            print(f"[SIG] {symbol} {tf_min}m qty_zero side={side} tilt={tilt} gates={';'.join(gates)}", flush=True)
        return None

    COUNTS_STAGE["16_signal_ok"] += 1
    p_adj = (conf - (COST_BPS / 10000.0)) if COST_WEIGHTING else conf

    if dbg:
        print(f"[SIG] {symbol} {tf_min}m side={side} pL={p_long:.3f} thrL={thr_long:.3f} "
              f"pS={p_short:.3f} thrS={thr_short:.3f} tilt={tilt} "
              f"margins(L={lm:.3f},S={sm:.3f}) qty={qty}", flush=True)

    return {
        "action": action,
        "orderType": "market",
        "price": None,
        "takeProfit": float(tp),
        "stopLoss": float(sl),
        "barTime": ts.tz_convert("UTC").isoformat(),
        "entry": price,
        "quantity": int(qty),
        "meta": {
            "note": "ml_pattern_v3_dual_tilt_bias",
            "proba": round(p_adj, 4),
            "direction": side,
            "thr_long": round(thr_long, 3),
            "thr_short": round(thr_short, 3),
            "tilt": int(tilt),
        },
    }

# ==============================
# DAILY GUARD (uses broker equity baseline)
# ==============================
def ensure_session_baseline():
    global SESSION_BASELINE_SET, SESSION_START_EQUITY, EQUITY_HIGH_WATER
    if SESSION_BASELINE_SET:
        return
    try:
        eq = get_account_equity(default_equity=START_EQUITY)
    except Exception:
        eq = START_EQUITY
    try:
        eq = float(eq)
    except Exception:
        eq = START_EQUITY
    SESSION_START_EQUITY = eq
    EQUITY_HIGH_WATER = eq
    SESSION_BASELINE_SET = True
    print(f"[SESSION] Baseline set: session_start_equity={SESSION_START_EQUITY:.2f}", flush=True)

def check_daily_guards():
    global HALT_TRADING, EQUITY_HIGH_WATER, CONF_THR_RUNTIME
    if not DAILY_GUARD_ENABLED:
        return
    ensure_session_baseline()
    try:
        eq_now = float(get_account_equity(default_equity=SESSION_START_EQUITY or START_EQUITY))
    except Exception:
        eq_now = SESSION_START_EQUITY or START_EQUITY

    if EQUITY_HIGH_WATER is None:
        EQUITY_HIGH_WATER = eq_now
    else:
        EQUITY_HIGH_WATER = max(EQUITY_HIGH_WATER, eq_now)

    up_lim = SESSION_START_EQUITY * (1.0 + DAILY_TP_PCT)
    dn_lim = SESSION_START_EQUITY * (1.0 - DAILY_DD_PCT)

    print(f"[DAILY-GUARD] eq={eq_now:.2f} start={SESSION_START_EQUITY:.2f} "
          f"targets +{DAILY_TP_PCT*100:.1f}%({up_lim:.2f}) / -{DAILY_DD_PCT*100:.1f}%({dn_lim:.2f})",
          flush=True)

    if HALT_TRADING:
        return

    hit_tp = eq_now >= up_lim
    hit_dd = eq_now <= dn_lim
    if hit_tp or hit_dd:
        HALT_TRADING = True
        reason = "Profit target" if hit_tp else "Drawdown limit"
        print(f"[DAILY-GUARD] ⛔ {reason} hit. Halting entries.", flush=True)
        if DAILY_FLATTEN_ON_HIT:
            ok, info = close_all_positions()
            print(f"[DAILY-GUARD] Flatten -> ok={ok} info={info}", flush=True)

    # Drift auto-response
    if DRIFT_MONITOR:
        bad = []
        for (lo, hi), (w, t) in _RELIABILITY.items():
            if t >= 10 and lo >= 0.80:
                wr = w / max(1, t)
                if wr < MIN_BUCKET_WIN:
                    bad.append((lo, hi, wr, t))
        if bad and CONF_THR_RUNTIME < AUTO_THR_MAX:
            old = CONF_THR_RUNTIME
            CONF_THR_RUNTIME = min(AUTO_THR_MAX, CONF_THR_RUNTIME + AUTO_THR_STEP)
            print(f"[DRIFT] Raising CONF_THR_RUNTIME {old:.2f} -> {CONF_THR_RUNTIME:.2f} due to weak buckets {bad}", flush=True)

def reset_daily_state_if_new_day():
    global DAY_STAMP, HALT_TRADING, SESSION_BASELINE_SET, EQUITY_HIGH_WATER, _ML_CACHE, _PROBA_EMA, _CONF_HIST, _PERSIST_OK, _RELIABILITY, CONF_THR_RUNTIME
    today = datetime.now().astimezone().strftime("%Y-%m-%d")
    if today != DAY_STAMP:
        DAY_STAMP = today
        HALT_TRADING = False
        SESSION_BASELINE_SET = False
        EQUITY_HIGH_WATER = None
        _ML_CACHE = {}
        _PROBA_EMA = {}
        _CONF_HIST = {}
        _PERSIST_OK = {}
        _RELIABILITY = { (0.70,0.75):[0,0], (0.75,0.80):[0,0], (0.80,0.85):[0,0], (0.85,0.90):[0,0], (0.90,1.01):[0,0] }
        CONF_THR_RUNTIME = CONF_THR
        print(f"[NEW DAY] State reset. START_EQUITY={START_EQUITY:.2f} DAY={DAY_STAMP}", flush=True)

# ==============================
# ROUTING & ORDER SEND
# ==============================
def _has_open_position(symbol: str) -> bool:
    for (sym, tf), trades in OPEN_TRADES.items():
        if sym != symbol:
            continue
        for t in trades:
            if t.is_open:
                return True
    return False

def handle_signal(strat_name: str, symbol: str, tf_min: int, sig: dict):
    combo_key = _combo_key(strat_name, symbol, tf_min)
    COUNTS["signals"] += 1
    COMBO_COUNTS[f"{combo_key}::signals"] += 1

    # --- pre-send short sanity gates ---
    if sig.get("action") in ("sell", "sell_short"):
        if 'SHORT_DENY' not in globals():
            globals()['SHORT_DENY'] = set()
        if symbol in SHORT_DENY:
            print(f"[ORDER-SKIP] {symbol} on denylist (not shortable).", flush=True)
            return
        last_px = float(sig.get("entry") or LAST_PRICE.get(symbol, 0.0) or 0.0)
        if last_px and last_px < MIN_SHORT_PRICE:
            print(f"[ORDER-SKIP] {symbol} short @ {last_px:.2f} < MIN_SHORT_PRICE={MIN_SHORT_PRICE:.2f}.", flush=True)
            return

    ok, info = send_to_broker(symbol, sig, strategy_tag="ml_pattern")

    info_str = str(info)
    if (not ok) and any(key in info_str for key in ("take_profit.limit_price", "stop_loss.stop_price")):
        try:
            broker_base = None
            try:
                s = info_str
                jstart = s.find("{")
                if jstart != -1:
                    j = json.loads(s[jstart:])
                    bp = j.get("base_price")
                    if bp is not None: broker_base = float(bp)
            except Exception:
                broker_base = None

            base = broker_base if (broker_base is not None) else float(sig.get("entry") or LAST_PRICE.get(symbol, 0.0) or 0.0)
            side = "long" if sig.get("action") == "buy" else "short"
            tp = float(sig["takeProfit"]); sl = float(sig["stopLoss"])
            tp2, sl2 = _apply_tick_rules(side, base, tp, sl, tick=MIN_TICK, extra_ticks=EXTRA_TICKS)

            if (abs(tp2 - tp) >= MIN_TICK/2) or (abs(sl2 - sl) >= MIN_TICK/2):
                sig2 = dict(sig); sig2["takeProfit"] = tp2; sig2["stopLoss"] = sl2
                ok, info = send_to_broker(symbol, sig2, strategy_tag="ml_pattern")
                if ok: sig = sig2
        except Exception:
            pass

    if (not ok) and ("cannot be sold short" in info_str):
        SHORT_DENY.add(symbol)

    try:
        COMBO_COUNTS[f"{combo_key}::orders.{'ok' if ok else 'err'}"] += 1
        COUNTS["orders.ok" if ok else "orders.err"] += 1
    except Exception:
        pass

    if ok:
        meta = sig.get("meta", {})
        meta["combo"] = combo_key
        meta["timeframe"] = f"{int(tf_min)}m"
        sig["meta"] = meta
        _record_open_trade(strat_name, symbol, tf_min, sig)

    print(f"[ORDER] {combo_key} -> action={sig.get('action')} qty={sig.get('quantity')} ok={ok} info={info}", flush=True)

# ==============================
# BATCHING
# ==============================
def _batched_symbols(universe: list):
    global _round_robin
    if not universe:
        return []
    N = len(universe)
    start = _round_robin % max(1, N)
    end = min(N, start + SCAN_BATCH_SIZE)
    batch = universe[start:end]
    _round_robin = 0 if end >= N else end
    return batch

def _get_sector(symbol: str) -> str:
    return "UNKNOWN"

def _prune_by_correlation(candidates, ret_series_map, max_corr=1.0):
    if max_corr >= 0.999:
        return candidates
    chosen = []
    for item in candidates:
        sym = item[1]
        r1 = ret_series_map.get(sym)
        if r1 is None or len(r1) < 5:
            chosen.append(item)
            continue
        ok = True
        for j in chosen:
            sym2 = j[1]
            r2 = ret_series_map.get(sym2)
            if r2 is None or len(r2) < 5:
                continue
            n = min(len(r1), len(r2))
            if n < 5:
                continue
            c = float(np.corrcoef(r1[-n:], r2[-n:])[0, 1])
            if np.isfinite(c) and abs(c) > max_corr:
                ok = False
                break
        if ok:
            chosen.append(item)
    return chosen

def _refresh_shortable_if_needed():
    global SHORTABLE_SET, SHORTABLE_LAST
    if not os.getenv("SKIP_NON_SHORTABLE", "0").lower() in ("1","true","yes"):
        return
    try:
        mins = int(os.getenv("SHORTABLE_REFRESH_MIN", "30"))
    except Exception:
        mins = 30
    now = datetime.now(timezone.utc)
    if SHORTABLE_LAST and (now - SHORTABLE_LAST) < timedelta(minutes=mins):
        return
    try:
        base = os.getenv("ALPACA_TRADE_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
        url = f"{base}/v2/assets?status=active&tradable=true&shortable=true"
        headers = {
            "APCA-API-KEY-ID": os.getenv("APCA_API_KEY_ID") or os.getenv("APCA-API-KEY-ID") or "",
            "APCA-API-SECRET-KEY": os.getenv("APCA_API_SECRET_KEY") or os.getenv("APCA-API-SECRET-KEY") or "",
        }
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code == 200:
            data = r.json()
            SHORTABLE_SET = {str(it.get("symbol","")).upper() for it in data if it.get("symbol")}
            SHORTABLE_LAST = now
            print(f"[SHORTABLE] cached {len(SHORTABLE_SET)} symbols (ttl ~{mins}m)", flush=True)
        else:
            print(f"[SHORTABLE] fetch failed {r.status_code} {r.text[:200]}", flush=True)
    except Exception as e:
        print(f"[SHORTABLE] exception {e}", flush=True)

# ==============================
# BROKER AUTH PROBE (boot banner)
# ==============================
def _get_alpaca_keys():
    key = (
        os.getenv("APCA_API_KEY_ID")
        or os.getenv("ALPACA_API_KEY_ID")
        or os.getenv("APCA-API-KEY-ID")
    )
    secret = (
        os.getenv("APCA_API_SECRET_KEY")
        or os.getenv("ALPACA_API_SECRET_KEY")
        or os.getenv("APCA-API-SECRET-KEY")
    )
    return key, secret

def _mask(s, keep=4):
    if not s:
        return "None"
    s = str(s)
    if len(s) <= keep*2:
        return "*" * len(s)
    return f"{s[:keep]}…{s[-keep:]}"

def probe_alpaca_auth():
    key, secret = _get_alpaca_keys()
    trade_base = os.getenv("ALPACA_TRADE_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
    data_base  = os.getenv("ALPACA_DATA_BASE_URL",  "https://data.alpaca.markets").rstrip("/")
    print(
        "[PROBE] Alpaca env "
        f"key={_mask(key)} secret={_mask(secret)} "
        f"trade_base={trade_base} data_base={data_base} PAPER_MODE={'1' if PAPER_MODE else '0'}",
        flush=True,
    )
    if not key or not secret:
        print("[PROBE] missing Alpaca API credentials (env not found)", flush=True)
        return
    try:
        headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        r = requests.get(f"{trade_base}/v2/account", headers=headers, timeout=15)
        if r.status_code == 200:
            acct = r.json()
            acct_num = acct.get("account_number", "unknown")
            status   = acct.get("status", "unknown")
            ccy      = acct.get("currency", "USD")
            print(f"[PROBE] GET /v2/account -> 200 account={acct_num} status={status} currency={ccy}", flush=True)
        else:
            print(f"[PROBE] GET /v2/account -> {r.status_code} body={r.text[:180]}", flush=True)
    except Exception as e:
        print(f"[PROBE] exception contacting Alpaca: {e}", flush=True)

# ==============================
# Maintenance / GC helpers
# ==============================
def _light_gc_afterhours():
    """
    When the market is closed AND SCANNER_MARKET_HOURS_ONLY=1,
    free big caches to avoid memory creep on Render.
    """
    if SCANNER_MARKET_HOURS_ONLY and not _market_session_now():
        _BARCACHE.clear(); _RESAMPLE_CACHE.clear(); _ML_CACHE.clear()

# ==============================
# MAIN LOOP
# ==============================
def main():
    global HALT_TRADING

    print(f"[BOOT] RUN_ID={RUN_ID} BRANCH={RENDER_GIT_BRANCH} COMMIT={RENDER_GIT_COMMIT} START_CMD=python app.py", flush=True)
    print(f"[BOOT] PAPER_MODE={'paper' if PAPER_MODE else 'live'} POLL_SECONDS={POLL_SECONDS} TFs={TF_MIN_LIST}", flush=True)
    print(f"[BOOT] CONF_THR={CONF_THR} (runtime={CONF_THR_RUNTIME}) R_MULT={R_MULT} FIXED_NOTIONAL={'ON' if USE_FIXED_NOTIONAL else 'OFF'} NOTIONAL={FIXED_NOTIONAL}")
    print(f"[BOOT] ADV: TB={int(TB_ENABLED)} META={int(META_ENABLED)} PERSIST_BARS={PERSIST_BARS} CONS_TF={CONSENSUS_TF} BATCH_TOP_K={BATCH_TOP_K} REGIME={int(REGIME_ENABLED)}", flush=True)
    print(f"[BOOT] DAILY_GUARD_ENABLED={int(DAILY_GUARD_ENABLED)} UP={DAILY_TP_PCT:.0%} DOWN={DAILY_DD_PCT:.0%} FLATTEN={int(DAILY_FLATTEN_ON_HIT)} START_EQUITY={START_EQUITY:.2f}", flush=True)

    probe_alpaca_auth()

    symbols = build_universe()
    print(f"[UNIVERSE] size={len(symbols)}  TFs={TF_MIN_LIST}  Batch={SCAN_BATCH_SIZE}", flush=True)

    while True:
        loop_start = time.time()
        try:
            reset_daily_state_if_new_day()

            # ---- Close phase: update TP/SL on open trades
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

            # ---- Daily guard
            check_daily_guards()

            # ---- EOD manager
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
                            print(f"[EOD] Flat after pass {i}", flush=True)
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

            if now_et.hour == 15 and now_et.minute >= 45:
                if not HALT_TRADING:
                    HALT_TRADING = True
                    print("[EOD] HALT_TRADING enabled at 3:45 ET (no new entries).", flush=True)
            if now_et.hour == 15 and now_et.minute >= 50:
                print("[EOD] Pre-close window (3:50–4:00 ET): cancel orders + flatten until flat.", flush=True)
                _cancel_all_open_orders_safely()
                _flatten_until_flat()
            if now_et.hour == 16 and now_et.minute < 3:
                print("[EOD] Post-bell safety net (4:00–4:03 ET).", flush=True)
                _cancel_all_open_orders_safely()
                _flatten_until_flat()

            # Kill switch
            if KILL_SWITCH:
                HALT_TRADING = True

            allow_entries = not HALT_TRADING

            # After-hours GC to keep memory in check
            _light_gc_afterhours()

            if not allow_entries:
                time.sleep(POLL_SECONDS)
                continue

            # ---- Scan gating
            if SCANNER_MARKET_HOURS_ONLY and not _market_session_now():
                if SCANNER_DEBUG:
                    print("[SCAN] Skipping — market session closed.", flush=True)
                time.sleep(POLL_SECONDS)
                continue

            # Update reference bars (RS + regime)
            try:
                _REF_BARS_1M[RS_SYMBOL] = fetch_bars_1m(RS_SYMBOL, lookback_minutes=300)
            except Exception:
                _REF_BARS_1M[RS_SYMBOL] = pd.DataFrame()

            # reset per-batch audit tallies
            COUNTS_STAGE.clear()
            COUNTS_MODEL.clear()
            _audit_reset()

            # (Optional) shortable prefetch
            try:
                _refresh_shortable_if_needed()
            except Exception:
                pass

            # ---- Build batch
            batch = _batched_symbols(symbols)
            if SCANNER_DEBUG:
                total = len(symbols)
                s = _round_robin if _round_robin else total
                start_idx = (s - len(batch)) if s else 0
                print(f"[SCAN] symbols {start_idx}:{start_idx+len(batch)} / {total}  (batch={len(batch)})", flush=True)

            candidates = []
            ret_series_map = {}  # for correlation pruning

            for sym in batch:
                # ---- fetch & normalize bars ----
                df1m = fetch_bars_1m(
                    sym,
                    lookback_minutes=max(
                        int(os.getenv("LOOKBACK_MINUTES_RTH", "2400")),
                        max(TF_MIN_LIST) * 300
                    )
                )
                if df1m is None or df1m.empty:
                    COUNTS_STAGE["02_no_data"] += 1
                    continue
                try:
                    df1m.index = df1m.index.tz_convert(MARKET_TZ)
                except Exception:
                    df1m.index = df1m.index.tz_localize("UTC").tz_convert(MARKET_TZ)

                # --- quick liquidity gate: require some intraday activity today (configurable) ---
                try:
                    today = _now_et().date()
                    today_mask = (df1m.index.date == today)
                    today_vol = int(df1m.loc[today_mask, "volume"].sum())
                    min_today = int(os.getenv("SCANNER_MIN_TODAY_VOL", "0"))
                    if today_vol < min_today:
                        COUNTS_STAGE["00_low_today_vol"] += 1
                        continue
                except Exception:
                    pass

                # --- optional ETF/ETN heuristic skip ---
                if os.getenv("SKIP_ETFS", "0").lower() in ("1", "true", "yes"):
                    s_up = sym.upper()
                    looks_etf = (
                        s_up.endswith("U") or
                        s_up.endswith("W") or
                        s_up in ("SPY", "QQQ", "IWM")
                    ) or (len(s_up) > 4)
                    if looks_etf:
                        COUNTS_STAGE["06_skip_etf"] += 1
                        if SCANNER_DEBUG:
                            print(f"[SKIP] {sym} ETF/ETN by heuristic.", flush=True)
                        continue

                # returns for corr pruning
                try:
                    ret_series_map[sym] = df1m["close"].pct_change().dropna().tail(CORR_LOOKBACK).to_numpy()
                except Exception:
                    pass

                # ---- per-timeframe signals ----
                for tf in TF_MIN_LIST:
                    sig = signal_ml_pattern_dual(sym, df1m, tf)

                    if not isinstance(sig, dict):
                        if SCANNER_DEBUG:
                            try:
                                bars_tf = _resample(df1m, tf)
                                if bars_tf is not None and not bars_tf.empty and len(bars_tf) > 300:
                                    ts_probe, p_up_probe, _ = _ml_features_and_pred(bars_tf)
                                    if ts_probe is not None and p_up_probe is not None:
                                        thrL = max(CONF_THR_RUNTIME, CONF_THR)
                                        thrS = max(CONF_THR_RUNTIME, float(os.getenv("SHORT_CONF_THR","0.7")))
                                        print(f"[PROBE] {sym} {tf}m p_up={p_up_probe:.3f} long_thr={thrL:.3f} short_thr={thrS:.3f}", flush=True)
                            except Exception:
                                pass
                        continue  # sig is None

                    action   = sig.get("action")
                    bar_time = sig.get("barTime") or sig.get("bar_time")
                    if not action or not bar_time:
                        if SCANNER_DEBUG:
                            print(f"[WARN] malformed signal {sym} {tf}m -> {sig}", flush=True)
                        continue

                    if NO_PYRAMIDING and _has_open_position(sym):
                        if SCANNER_DEBUG:
                            print(f"[PYRAMID-BLOCK] {sym} already has open exposure. Skipping.", flush=True)
                        continue

                    try:
                        k = _dedupe_key(sym, tf, action, bar_time)
                    except Exception as e:
                        if SCANNER_DEBUG:
                            print(f"[DEDUPE-ERROR] {sym} {tf}m -> {e}", flush=True)
                        continue

                    if k in _sent_keys:
                        continue

                    meta = sig.get("meta", {}) or {}
                    prob = meta.get("proba")
                    if prob is None:
                        prob = meta.get("proba_up", 0.0)
                    try:
                        prob = float(prob)
                    except Exception:
                        prob = 0.0

                    candidates.append((prob, sym, tf, sig, k, _get_sector(sym)))

            # -------- selection: sort, optional sector cap, corr prune, top-K, then send --------
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                print(f"[CAND-COUNT] raw={len(candidates)}", flush=True)
                top5_preview = ", ".join([f"{c[1]}:{c[0]:.3f}@{c[2]}m" for c in candidates[:5]])
                if top5_preview:
                    print(f"[SELECT] after-sort={len(candidates)} top5={top5_preview}", flush=True)

                if MAX_PER_SECTOR > 0:
                    sec_count = defaultdict(int)
                    kept = []
                    for item in candidates:
                        sec = item[5]
                        if sec_count[sec] < MAX_PER_SECTOR:
                            kept.append(item)
                            sec_count[sec] += 1
                    if SCANNER_DEBUG:
                        print(f"[SELECT] after-sector={len(kept)} (was {len(candidates)})", flush=True)
                    candidates = kept

                if MAX_CAND_CORR < 0.999:
                    prev = len(candidates)
                    candidates = _prune_by_correlation(candidates, ret_series_map, MAX_CAND_CORR)
                    if SCANNER_DEBUG:
                        print(f"[SELECT] after-corr={len(candidates)} (was {prev})", flush=True)
                else:
                    if SCANNER_DEBUG:
                        print(f"[SELECT] after-corr={len(candidates)} (was {len(candidates)})", flush=True)

                chosen = candidates[:BATCH_TOP_K] if BATCH_TOP_K else candidates
                print(f"[SELECT] chosen={len(chosen)}", flush=True)

                for prob, sym, tf, sig, k, _ in chosen:
                    _sent_keys.add(k)
                    handle_signal("ml_pattern", sym, tf, sig)
            else:
                if SCANNER_DEBUG:
                    print("[CAND-COUNT] raw=0", flush=True)

            # Print audit heartbeat
            try:
                _audit_maybe_print(thr_long=CONF_THR_RUNTIME)
            except Exception:
                pass

        except Exception as e:
            import traceback
            print("[LOOP ERROR]", e, traceback.format_exc(), flush=True)

        elapsed = time.time() - loop_start
        time.sleep(max(1, POLL_SECONDS - int(elapsed)))

if __name__ == "__main__":
    main()
