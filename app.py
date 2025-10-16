# ====================================================================================
# app.py — Upgrade Summary (v2025-10-15)
# ====================================================================================
# This version includes ALL changes discussed tonight, ready for long/short live use.
#
# CORE MODELING (ML v3)
# - Rich feature set: ret1, ret3, ret5, ret10, vol10, vol20, rsi14, stoch_k,
#   ema10_slope, ema20_slope, atrp, body_tr, dvolz, vwap_dist, gap_pct, rs.
# - Recency-weighted RandomForest fit (exponential time weights).
# - Probability calibration via isotonic with time-aware CV (or purged CV when enabled).
# - Optional triple-barrier labeling (TB_* vars) and meta-labeling (META_ENABLED).
# - Optional multi-horizon ensemble (MULTI_H_*), geometric-mean probability blend.
#
# LONG + SHORT SIGNAL ENGINE
# - Unified dual-side function: signal_ml_pattern_dual(...) (long and short).
# - Scan loop calls signal_ml_pattern_dual (replaces prior long-only call).
# - Both sides use ATR-based TP/SL; shorts are placed with action="sell_short".
# - Separate short controls: ENABLE_SHORTS, SHORT_CONF_THR.
# - Confidence smoothing & persistence tracked independently per side
#   (_PROBA_EMA / _PROBA_EMA_SHORT, _PERSIST_OK / _PERSIST_OK_S).
#
# CONFIDENCE GATES & SELECTION
# - Runtime threshold CONF_THR_RUNTIME (auto-raised by drift monitor when enabled).
# - MIN_PROBA_GAP enforces margin above threshold.
# - PERSIST_BARS requires N consecutive over-threshold bars before entry.
# - Higher-TF consensus (CONSENSUS_TF, CONSENSUS_WEIGHT) for both long & short.
# - Rolling quantile gate (CONF_ROLL_N, CONF_Q) using _CONF_HIST per (symbol, tf).
# - Cost-aware ranking (COST_BPS, COST_WEIGHTING).
# - Regime gate (REGIME_ENABLED): longs only when RS_SYMBOL > EMA(REGIME_MA) and
#   realized vol between REGIME_MIN_RVOL and REGIME_MAX_RVOL; shorts require below EMA.
#
# CANDIDATE PRUNING & PORTFOLIO CONSTRAINTS
# - Candidates sorted by adjusted confidence; optional BATCH_TOP_K cap.
# - Sector cap (MAX_PER_SECTOR) via _get_sector (stub returns "UNKNOWN" by default).
# - Correlation pruning (MAX_CAND_CORR, CORR_LOOKBACK) using recent returns.
#
# DRIFT MONITORING & AUTO-RESPONSE
# - Reliability buckets (_RELIABILITY) track win rate by confidence bucket.
# - If DRIFT_MONITOR=1 and high-confidence buckets underperform (< MIN_BUCKET_WIN),
#   CONF_THR_RUNTIME nudges up by AUTO_THR_STEP, capped at AUTO_THR_MAX.
#
# DATA PLUMBING & CACHING
# - Per-bar ML cache _ML_CACHE keyed by (symbol, tf, bar_end_iso, last_close).
# - Maintains _REF_BARS_1M[RS_SYMBOL] for RS/regime features each loop.
#
# DEDUPE & SAFETY
# - Dedupe key includes RUN_ID to avoid cross-run collisions.
# - KILL_SWITCH halts entries immediately without redeploy.
# - EOD manager: 3:45 halt new entries; 3:50 cancel+flatten; 4:00 safety net.
#
# ORDER ROUTING & BOOKKEEPING
# - handle_signal(...) unchanged in shape; receives long/short orders.
# - LiveTrade stores proba for drift stats; close logic updates reliability buckets.
#
# NEW/CHANGED ENV VARS (set in Render as needed)
# - Shorting:          ENABLE_SHORTS, SHORT_CONF_THR
# - Confidence:        PROBA_EMA_ALPHA, MIN_PROBA_GAP, CONSENSUS_TF, CONSENSUS_WEIGHT, BATCH_TOP_K
# - RS/benchmark:      RS_SYMBOL
# - Rolling quantile:  CONF_ROLL_N, CONF_Q
# - Triple-barrier:    TB_ENABLED, TB_ATR_K, TB_TIMEOUT_H
# - Meta-labeling:     META_ENABLED
# - Purged CV:         PURGED_CV_FOLDS, PURGED_EMBARGO_FRAC
# - Multi-horizon:     MULTI_H_ENABLED, MULTI_H_LIST
# - Cost-aware:        COST_BPS, COST_WEIGHTING
# - Portfolio:         MAX_PER_SECTOR, MAX_CAND_CORR, CORR_LOOKBACK
# - Drift auto:        DRIFT_MONITOR, AUTO_THR_STEP, AUTO_THR_MAX, MIN_BUCKET_WIN
# - Regime gate:       REGIME_ENABLED, REGIME_MA, REGIME_MIN_RVOL, REGIME_MAX_RVOL
# - Kill switch:       KILL_SWITCH
#
# QUICK TOGGLES (typical starting defaults)
# - ENABLE_SHORTS=1
# - SHORT_CONF_THR≈CONF_THR (e.g., 0.80 if you want shorts stricter)
# - PROBA_EMA_ALPHA=0.0 (set 0.3–0.5 if you want smoothing)
# - PERSIST_BARS=0 (set 2 for two-bar confirmation)
# - CONSENSUS_TF=0 (set 5 to require 5m confirmation of 1m)
# - CONF_ROLL_N=0 (set 100–300 to activate rolling-quantile gate)
# - DRIFT_MONITOR=0 (set 1 to auto-raise threshold if high-p buckets underperform)
# - MULTI_H_ENABLED=0 (set 1 with MULTI_H_LIST=3,5,10 for ensemble)
# - REGIME_ENABLED=0 (set 1 to restrict longs/shorts by market regime)
#
# NOTES
# - NO_PYRAMIDING prevents accumulating into a symbol across TFs.
# - FIXED_NOTIONAL sizing is the default (USE_FIXED_NOTIONAL=1). Set to 0 to use
#   risk-based sizing (EQUITY_USD, RISK_PCT, MAX_POS_PCT, ROUND_LOT).
# - Dedupe includes RUN_ID so each deployment has fresh signal keys.
# =============================================================================

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

TF_MIN_LIST  = [int(x) for x in os.getenv("TF_MIN_LIST", "1,2,3,5,10").split(",") if x.strip()]
MAX_UNIVERSE_PAGES = int(os.getenv("MAX_UNIVERSE_PAGES", "3"))
SCAN_BATCH_SIZE    = int(os.getenv("SCAN_BATCH_SIZE", "150"))
SCANNER_MIN_AVG_VOL = int(os.getenv("SCANNER_MIN_AVG_VOL", "0"))

MIN_PRICE = float(os.getenv("MIN_PRICE", "3.0"))

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

# --- Relative strength benchmark (for features)
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
    if _is_rth(ts):
        return True
    if ALLOW_PREMARKET and (4 <= ts.hour < 9 or (ts.hour == 9 and ts.minute < 30)):
        return True
    if ALLOW_AFTERHOURS and (16 <= ts.hour < 20):
        return True
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
    if entry_price is None:
        return 0
    if USE_FIXED_NOTIONAL:
        if entry_price <= 0:
            return 0
        raw = FIXED_NOTIONAL / max(1e-9, float(entry_price))
        qty = math.floor(raw / max(1, round_lot)) * max(1, round_lot)
        return int(max(qty, min_qty if qty > 0 else 0))
    if stop_price is None:
        return 0
    risk_per_share = abs(float(entry_price) - float(stop_price))
    if risk_per_share <= 0:
        return 0
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
    # Fallback to non-cached path
    if not CACHE_ENABLED:
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
        # price gate
        try:
            last_px = float(df["close"].iloc[-1])
            if last_px < MIN_PRICE:
                return pd.DataFrame()
        except Exception:
            return pd.DataFrame()
        return df[["open","high","low","close","volume"]].copy()

    # ----- Cached path -----
    # Helper: LRU admit/evict
    def _touch(sym: str):
        _BARCACHE.setdefault(sym, pd.DataFrame())
        _BARCACHE.move_to_end(sym, last=True)
        # LRU evict if needed
        while len(_BARCACHE) > max(1, CACHE_MAX_SYMBOLS):
            _BARCACHE.popitem(last=False)

    # Ensure cache entry exists
    cached = _BARCACHE.get(symbol)
    if cached is None or cached.empty:
        # First fill: backfill full requested window (capped by CACHE_MAX_MINUTES)
        end = datetime.now(timezone.utc)
        need_minutes = min(max(1, lookback_minutes), CACHE_MAX_MINUTES)
        start = end - timedelta(minutes=need_minutes)
        start_iso = start.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        end_iso   = end.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        df_new = _alpaca_fetch_1m(symbol, start_iso=start_iso, end_iso=end_iso, limit=10000)
        if df_new is None or df_new.empty:
            return pd.DataFrame()
        try:
            df_new.index = df_new.index.tz_convert(MARKET_TZ)
        except Exception:
            df_new.index = df_new.index.tz_localize("UTC").tz_convert(MARKET_TZ)
        cached = df_new
    else:
        # Incremental update: fetch bars strictly after last cached index
        try:
            last_ts = cached.index[-1].tz_convert("UTC") if hasattr(cached.index[-1], "tzinfo") else cached.index[-1]
        except Exception:
            last_ts = None
        end = datetime.now(timezone.utc)
        # Only fetch if we’re beyond the last cached minute
        if last_ts is not None:
            start = (last_ts + timedelta(minutes=1)).replace(tzinfo=timezone.utc)
        else:
            start = end - timedelta(minutes=min(lookback_minutes, CACHE_MAX_MINUTES))
        if start < end:
            start_iso = start.replace(microsecond=0).isoformat().replace("+00:00", "Z")
            end_iso   = end.replace(microsecond=0).isoformat().replace("+00:00", "Z")
            df_inc = _alpaca_fetch_1m(symbol, start_iso=start_iso, end_iso=end_iso, limit=10000)
            if df_inc is not None and not df_inc.empty:
                try:
                    df_inc.index = df_inc.index.tz_convert(MARKET_TZ)
                except Exception:
                    df_inc.index = df_inc.index.tz_localize("UTC").tz_convert(MARKET_TZ)
                # Merge, sort, dedupe
                cached = pd.concat([cached, df_inc], axis=0)
                cached = cached[~cached.index.duplicated(keep="last")].sort_index()

    # Trim cache to CACHE_MAX_MINUTES (LRU keeps symbol count in check)
    if len(cached) > 0 and CACHE_MAX_MINUTES > 0:
        cutoff = cached.index[-1] - timedelta(minutes=CACHE_MAX_MINUTES)
        cached = cached[cached.index >= cutoff]

    # Store back & LRU-touch
    _BARCACHE[symbol] = cached
    _touch(symbol)

    # price gate on latest
    try:
        last_px = float(cached["close"].iloc[-1])
        if last_px < MIN_PRICE:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    # Return only requested window (but served from cache)
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
    - If new rows exist, recompute resample (simple, robust).
    Note: We keep it simple (full recompute on new data) — big win still comes from 1m caching.
    """
    if df1m is None or df1m.empty:
        return pd.DataFrame()

    # Identify this source by its last timestamp (cheap & robust)
    src_last = df1m.index[-1]

    # Try cache hit if caller provided a symbol hint via index name
    # (Many pandas readers leave index.name None; so we don’t rely on the symbol key here.)
    cache_key = (id(df1m), int(tf_min))  # buffer-identity-based; safe per-call

    cached = _RESAMPLE_CACHE.get(cache_key)
    if cached and cached[0] is not None and cached[0] == src_last:
        return cached[1]

    # Otherwise (new data), perform resample
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
            # Drift monitor bucket update
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
# Tunables for ML/labels (base)
_H = 5
_RET_THR = 0.0025
_MIN_SAMPLES = 240

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
    # Basic returns
    out["ret1"]  = out["close"].pct_change(1)
    out["ret3"]  = out["close"].pct_change(3)
    out["ret5"]  = out["close"].pct_change(5)
    out["ret10"] = out["close"].pct_change(10)
    # Rolling vol
    out["vol10"] = out["close"].pct_change().rolling(10).std()
    out["vol20"] = out["close"].pct_change().rolling(20).std()
    # RSI(14)
    delta = out["close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = up / (down.replace(0, np.nan))
    out["rsi14"] = 100 - (100 / (1 + rs))
    # Stoch %K(14,3)
    hh = out["high"].rolling(14).max()
    ll = out["low"].rolling(14).min()
    out["stoch_k"] = 100 * (out["close"] - ll) / (hh - ll).replace(0, np.nan)
    out["stoch_k"] = out["stoch_k"].rolling(3).mean()
    # EMAs + slopes
    ema10 = out["close"].ewm(span=10, adjust=False).mean()
    ema20 = out["close"].ewm(span=20, adjust=False).mean()
    out["ema10_slope"] = ema10.pct_change()
    out["ema20_slope"] = ema20.pct_change()
    # ATR(14) & ATR%
    tr = pd.concat([
        (out["high"] - out["low"]),
        (out["high"] - out["close"].shift()).abs(),
        (out["low"] - out["close"].shift()).abs()
    ], axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14).mean()
    out["atrp"]  = out["atr14"] / out["close"]
    # Candle structure
    body = (out["close"] - out["open"]).abs()
    rng  = (out["high"] - out["low"]).replace(0, np.nan)
    out["body_tr"] = (body / rng).clip(0, 5)
    # Liquidity proxy
    if "volume" in out.columns:
        out["dvol20"] = (out["close"] * out["volume"]).rolling(20).mean()
        out["dvolz"]  = (out["dvol20"] / out["dvol20"].rolling(60).mean()) - 1.0
    else:
        out["dvolz"] = 0.0
    # VWAP distance
    try:
        vwap = (out["close"] * out.get("volume", pd.Series(index=out.index))).cumsum() / out.get("volume", pd.Series(index=out.index)).replace(0,np.nan).cumsum()
        out["vwap_dist"] = (out["close"] / vwap) - 1.0
    except Exception:
        out["vwap_dist"] = 0.0
    # Gap %
    out["gap_pct"] = out["open"] / out["close"].shift(1) - 1.0
    # Relative strength vs RS_SYMBOL (10-bar)
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
    # Triple-barrier: TP/SL via ATR*k within h bars ahead
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
    # Recency weights
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

    # If there aren't at least two classes in the training set,
    # return the base model (no calibration possible).
    try:
        if getattr(y_trn, "nunique", None):
            nuniq = int(y_trn.nunique())
        else:
            nuniq = len(np.unique(y_trn))
        if nuniq < 2:
            return clf

        # Choose CV
        if PURGED_CV_FOLDS and PURGED_CV_FOLDS >= 2:
            cv_splits = list(PurgedKFold(PURGED_CV_FOLDS, PURGED_EMBARGO_FRAC).split(X_trn))
        else:
            cv_splits = list(TimeSeriesSplit(n_splits=3).split(X_trn))

        # Keep only folds where BOTH train and test contain two classes
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
    # If the base model only saw one class, skip calibration and set proba logic accordingly.
    try:
        classes_ = getattr(base, "classes_", None)
        if classes_ is not None and len(classes_) < 2:
            # Predict_proba will be shape (n,1). If the single class is 1, prob_up=1, else 0.
            p_live = base.predict_proba(x_live)
            if p_live.shape[1] == 1:
                only_cls = int(classes_[0])
                proba_up = 1.0 if only_cls == 1 else 0.0
                pred_up  = int(proba_up >= 0.5)
                ts = df.index[-1]
                return ts, float(proba_up), int(pred_up)
    except Exception:
        pass

    # Otherwise proceed with calibration (now robust)
    clf = _calibrate(base, X_trn, y_trn)

    proba_mat = clf.predict_proba(x_live)
    if proba_mat.shape[1] == 1:
        # Calibrator/base returned single class unexpectedly; map it safely.
        classes_ = getattr(clf, "classes_", [1])
        only_cls = int(classes_[0]) if len(classes_) else 1
        proba_up = 1.0 if only_cls == 1 else 0.0
    else:
        proba_up = float(proba_mat[0, 1])

    pred_up  = int(proba_up >= 0.5)
    ts = df.index[-1]
    return ts, proba_up, pred_up

    # Meta-labeling (optional): stack base proba + a few features
    if TB_ENABLED and META_ENABLED:
        from sklearn.linear_model import LogisticRegression
        p_trn = clf.predict_proba(X_trn)[:,1]
        meta_X = np.column_stack([
            p_trn,
            X_trn["rsi14"].to_numpy(),
            X_trn["atrp"].to_numpy(),
            X_trn["vwap_dist"].to_numpy(),
        ])
        meta_y = y_trn.to_numpy().astype(int)
        meta = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None)
        meta.fit(meta_X, meta_y)
        p_live_base = float(clf.predict_proba(x_live)[0,1])
        x_meta_live = np.array([[p_live_base,
                                 float(x_live["rsi14"].iloc[0]),
                                 float(x_live["atrp"].iloc[0]),
                                 float(x_live["vwap_dist"].iloc[0])]])
        proba_up = float(meta.predict_proba(x_meta_live)[0,1])
    else:
        proba_up = float(clf.predict_proba(x_live)[0,1])

    pred_up  = int(proba_up >= 0.5)
    ts = df.index[-1]
    return ts, proba_up, pred_up

def _ml_features_and_pred(bars: pd.DataFrame):
    if MULTI_H_ENABLED:
        ps = []
        last_r = None
        for h in MULTI_H_LIST:
            r = _ml_features_and_pred_core(bars, h)
            if r[0] is None:
                continue
            ps.append(r[1]); last_r = r
        if not ps:
            return None, None, None
        # geometric mean across horizons
        p = float(np.exp(np.mean(np.log(np.clip(ps, 1e-6, 1-1e-6)))))
        ts, _, _ = last_r
        return ts, p, int(p >= 0.5)
    else:
        return _ml_features_and_pred_core(bars, _H)

# ==============================
# SHORT/LONG SIGNAL (dual-side wrapper)
# ==============================
def signal_ml_pattern_dual(symbol: str, df1m: pd.DataFrame, tf_min: int,
                           conf_threshold=CONF_THR, r_multiple=R_MULT, atr_k=1.0):
    if df1m is None or df1m.empty or not isinstance(df1m.index, pd.DatetimeIndex):
        return None
    bars = _resample(df1m, tf_min)
    if bars is None or bars.empty:
        return None

    # === Per-bar model cache ===
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
        return None
    if not _in_session(ts):
        return None

    # helper smoothing
    def _smooth(store: dict, key, p, alpha: float):
        if alpha <= 0: return p
        p_prev = store.get(key, p)
        p_s = alpha * p + (1 - alpha) * p_prev
        store[key] = p_s
        return p_s

    key_pt = (symbol, int(tf_min))
    p_raw  = float(proba_up)

    # ---------- LONG branch ----------
    p_long = _smooth(_PROBA_EMA, key_pt, p_raw, PROBA_EMA_ALPHA)

    # higher-TF consensus (long)
    if CONSENSUS_TF and CONSENSUS_TF != tf_min:
        bars_hi = _resample(df1m, CONSENSUS_TF)
        if bars_hi is None or bars_hi.empty:
            return None
        last_ts2 = bars_hi.index[-1]
        last_ts2_iso = last_ts2.tz_convert("UTC").isoformat() if hasattr(last_ts2, "tzinfo") else str(last_ts2)
        cache_key2 = (symbol, int(CONSENSUS_TF), last_ts2_iso, float(bars_hi["close"].iloc[-1]))
        cached2 = _ML_CACHE.get(cache_key2)
        if cached2:
            ts2, p2, pred2 = cached2
        else:
            ts2, p2, pred2 = _ml_features_and_pred(bars_hi)
            if ts2 is not None:
                _ML_CACHE[cache_key2] = (ts2, p2, pred2)
        if ts2 is None or p2 is None or pred2 != 1:
            p_long = -1
        else:
            p_long = float((p_long ** (1 - CONSENSUS_WEIGHT)) * (float(p2) ** CONSENSUS_WEIGHT))

    # dynamic quantile (long)
    if p_long >= 0 and CONF_ROLL_N > 0:
        dq = _CONF_HIST.get(key_pt)
        if dq is None:
            dq = deque(maxlen=CONF_ROLL_N)
            _CONF_HIST[key_pt] = dq
        if len(dq) >= max(10, int(0.5*CONF_ROLL_N)):
            qthr = float(np.quantile(np.array(dq), CONF_Q))
            if p_long < qthr:
                p_long = -1
        if p_long >= 0:
            dq.append(p_long)

    # regime gate (long when SPY > EMA)
    if REGIME_ENABLED and p_long >= 0:
        ref = _REF_BARS_1M.get(RS_SYMBOL, pd.DataFrame())
        if ref is None or ref.empty or len(ref) < max(REGIME_MA+5, 60):
            p_long = -1
        else:
            ema = ref["close"].ewm(span=REGIME_MA, adjust=False).mean().iloc[-1]
            above = float(ref["close"].iloc[-1]) > float(ema)
            rvol = ref["close"].pct_change().rolling(30).std().iloc[-1]
            if not (above and REGIME_MIN_RVOL <= float(rvol) <= REGIME_MAX_RVOL):
                p_long = -1

    # persistence (long)
    long_ok = False
    thr_long = max(CONF_THR_RUNTIME, conf_threshold)
    if p_long >= 0 and p_long >= thr_long + MIN_PROBA_GAP:
        if PERSIST_BARS >= 2:
            c = _PERSIST_OK.get(key_pt, 0) + 1
            _PERSIST_OK[key_pt] = c
            long_ok = (c >= PERSIST_BARS)
        else:
            long_ok = True
    else:
        _PERSIST_OK[key_pt] = 0

    # ---------- SHORT branch ----------
    short_ok = False
    p_short = -1.0
    if ENABLE_SHORTS:
        p_down_raw = 1.0 - p_raw
        p_short = _smooth(_PROBA_EMA_SHORT, key_pt, p_down_raw, PROBA_EMA_ALPHA)

        # higher-TF consensus (short)
        if CONSENSUS_TF and CONSENSUS_TF != tf_min:
            bars_hi = _resample(df1m, CONSENSUS_TF)
            if bars_hi is None or bars_hi.empty:
                p_short = -1
            else:
                last_ts2 = bars_hi.index[-1]
                last_ts2_iso = last_ts2.tz_convert("UTC").isoformat() if hasattr(last_ts2, "tzinfo") else str(last_ts2)
                cache_key2s = (symbol, int(CONSENSUS_TF), last_ts2_iso, float(bars_hi["close"].iloc[-1]))
                cached2s = _ML_CACHE.get(cache_key2s)
                if cached2s:
                    ts2s, p2s, pred2s = cached2s
                else:
                    ts2s, p2s, pred2s = _ml_features_and_pred(bars_hi)
                    if ts2s is not None:
                        _ML_CACHE[cache_key2s] = (ts2s, p2s, pred2s)
                if (ts2s is None) or (p2s is None) or (pred2s != 0):
                    p_short = -1
                else:
                    p_short = float((p_short ** (1 - CONSENSUS_WEIGHT)) * ((1.0 - float(p2s)) ** CONSENSUS_WEIGHT))

        # dynamic quantile (short): use 1 - dq to mirror long space
        if p_short >= 0 and CONF_ROLL_N > 0:
            dq = _CONF_HIST.get(key_pt)
            if dq is None:
                dq = deque(maxlen=CONF_ROLL_N)
                _CONF_HIST[key_pt] = dq
            if len(dq) >= max(10, int(0.5*CONF_ROLL_N)):
                qthr = float(np.quantile(1.0 - np.array(dq), CONF_Q))
                if p_short < qthr:
                    p_short = -1
            if p_short >= 0:
                dq.append(1.0 - p_short)

        # regime gate (short when SPY < EMA)
        if REGIME_ENABLED and p_short >= 0:
            ref = _REF_BARS_1M.get(RS_SYMBOL, pd.DataFrame())
            if ref is None or ref.empty or len(ref) < max(REGIME_MA+5, 60):
                p_short = -1
            else:
                ema = ref["close"].ewm(span=REGIME_MA, adjust=False).mean().iloc[-1]
                below = float(ref["close"].iloc[-1]) < float(ema)
                rvol = ref["close"].pct_change().rolling(30).std().iloc[-1]
                if not (below and REGIME_MIN_RVOL <= float(rvol) <= REGIME_MAX_RVOL):
                    p_short = -1

        # persistence (short)
        thr_short = max(CONF_THR_RUNTIME, SHORT_CONF_THR)
        if p_short >= 0 and p_short >= thr_short + MIN_PROBA_GAP:
            if PERSIST_BARS >= 2:
                c = _PERSIST_OK_S.get(key_pt, 0) + 1
                _PERSIST_OK_S[key_pt] = c
                short_ok = (c >= PERSIST_BARS)
            else:
                short_ok = True
        else:
            _PERSIST_OK_S[key_pt] = 0

    # choose side
    side = None
    conf = -1.0
    if long_ok:
        side, conf = ("long", p_long)
    if short_ok and (not long_ok or (p_short - max(CONF_THR_RUNTIME, SHORT_CONF_THR) >
                                     p_long - max(CONF_THR_RUNTIME, conf_threshold))):
        side, conf = ("short", p_short)

    if side is None:
        return None

    price = float(bars["close"].iloc[-1])
    tr = pd.concat([
        (bars["high"] - bars["low"]),
        (bars["high"] - bars["close"].shift()).abs(),
        (bars["low"] - bars["close"].shift()).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean().iloc[-1]
    if not np.isfinite(atr14) or atr14 <= 0:
        return None

    if side == "long":
        sl = price - atr_k * atr14
        tp = price + r_multiple * atr_k * atr14
        qty = _position_qty(price, sl); action = "buy"
    else:  # short
        sl = price + atr_k * atr14
        tp = price - r_multiple * atr_k * atr14
        qty = _position_qty(price, sl); action = "sell_short"

    if qty <= 0:
        return None

    p_adj = conf - (COST_BPS/10000.0 if COST_WEIGHTING else 0.0)
    return {
        "action": action,
        "orderType": "market",
        "price": None,
        "takeProfit": float(tp),
        "stopLoss": float(sl),
        "barTime": ts.tz_convert("UTC").isoformat(),
        "entry": price,
        "quantity": int(qty),
        "meta": {"note": "ml_pattern_v3", "proba_up": round(p_adj, 4), "direction": side},
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

    # Drift auto-response: if enabled, gently raise threshold when high-p buckets underperform
    if DRIFT_MONITOR:
        bad = []
        for (lo, hi), (w, t) in _RELIABILITY.items():
            if t >= 10 and lo >= 0.80:  # only high-conf buckets
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
    meta = sig.get("meta", {})
    meta["combo"] = combo_key
    meta["timeframe"] = f"{int(tf_min)}m"
    sig["meta"] = meta
    _record_open_trade(strat_name, symbol, tf_min, sig)

    ok, info = send_to_broker(symbol, sig, strategy_tag="ml_pattern")
    try:
        COMBO_COUNTS[f"{combo_key}::orders.{'ok' if ok else 'err'}"] += 1
        COUNTS["orders.ok" if ok else "orders.err"] += 1
    except Exception:
        pass
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

# Optional sector lookup (kept simple; returns 'UNKNOWN' unless you wire a map)
def _get_sector(symbol: str) -> str:
    return "UNKNOWN"

def _prune_by_correlation(candidates, ret_series_map, max_corr=1.0):
    if max_corr >= 0.999:  # effectively off
        return candidates
    chosen = []
    for item in candidates:
        ok = True
        _, sym, _, _, _ = item
        r1 = ret_series_map.get(sym)
        if r1 is None or len(r1) < 5:
            chosen.append(item); continue
        for j in chosen:
            sym2 = j[1]
            r2 = ret_series_map.get(sym2)
            if r2 is None: continue
            n = min(len(r1), len(r2))
            if n < 5: continue
            c = float(np.corrcoef(r1[-n:], r2[-n:])[0,1])
            if np.isfinite(c) and abs(c) > max_corr:
                ok = False
                break
        if ok:
            chosen.append(item)
    return chosen

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

    from common.signal_bridge import probe_alpaca_auth
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
                        row = bars.iloc[-1]; ts  = bars.index[-1]
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
            if not allow_entries:
                time.sleep(POLL_SECONDS)
                continue

            # ---- Scan & signal phase
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

            batch = _batched_symbols(symbols)
            if SCANNER_DEBUG:
                total = len(symbols)
                s = _round_robin if _round_robin else total
                start_idx = (s - len(batch)) if s else 0
                print(f"[SCAN] symbols {start_idx}:{start_idx+len(batch)} / {total}  (batch={len(batch)})", flush=True)

            candidates = []
            ret_series_map = {}  # for correlation pruning
            for sym in batch:
                df1m = fetch_bars_1m(
                    sym,
                    lookback_minutes=max(
                        int(os.getenv("LOOKBACK_MINUTES_RTH", "2400")),
                        max(TF_MIN_LIST) * 300
                    )
                )
                if df1m is None or df1m.empty:
                    continue
                try:
                    df1m.index = df1m.index.tz_convert(MARKET_TZ)
                except Exception:
                    df1m.index = df1m.index.tz_localize("UTC").tz_convert(MARKET_TZ)

                # store returns for corr pruning
                try:
                    ret_series_map[sym] = df1m["close"].pct_change().dropna().tail(CORR_LOOKBACK).to_numpy()
                except Exception:
                    pass

                for tf in TF_MIN_LIST:
                    sig = signal_ml_pattern_dual(sym, df1m, tf)
                    if not sig:
                        continue
                    if NO_PYRAMIDING and _has_open_position(sym):
                        if SCANNER_DEBUG:
                            print(f"[PYRAMID-BLOCK] {sym} already has open exposure. Skipping.", flush=True)
                        continue
                    k = _dedupe_key(sym, tf, sig["action"], sig.get("barTime",""))
                    if k in _sent_keys:
                        continue
                    prob = float(sig.get("meta",{}).get("proba_up", 0.0))
                    # Stage candidate for selection
                    candidates.append((prob, sym, tf, sig, k, _get_sector(sym)))

            # selection: sort, optional sector cap, corr prune, top-K
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)

                # sector cap
                if MAX_PER_SECTOR > 0:
                    sec_count = defaultdict(int)
                    kept = []
                    for item in candidates:
                        sec = item[5]
                        if sec_count[sec] < MAX_PER_SECTOR:
                            kept.append(item)
                            sec_count[sec] += 1
                    candidates = kept

                # correlation prune
                if MAX_CAND_CORR < 0.999:
                    candidates = _prune_by_correlation(candidates, ret_series_map, MAX_CAND_CORR)

                chosen = candidates[:BATCH_TOP_K] if BATCH_TOP_K else candidates
                for prob, sym, tf, sig, k, _ in chosen:
                    _sent_keys.add(k)
                    handle_signal("ml_pattern", sym, tf, sig)

        except Exception as e:
            import traceback
            print("[LOOP ERROR]", e, traceback.format_exc(), flush=True)

        elapsed = time.time() - loop_start
        time.sleep(max(1, POLL_SECONDS - int(elapsed)))

if __name__ == "__main__":
    main()
