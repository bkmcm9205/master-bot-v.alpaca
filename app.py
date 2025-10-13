# app.py — dynamic market scanner (PLUMBING SWAP: Polygon/TP -> Alpaca)
# Requires: pandas, numpy, requests, pandas_ta, scikit-learn
#
# ✅ Models/ML logic unchanged. Only data + broker I/O rewired to Alpaca.

import os, time, json, math, requests
import pandas as pd, numpy as np
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
from zoneinfo import ZoneInfo

# ---- Alpaca adapters (data + broker) ----
from adapters.data_alpaca import (
    fetch_1m as _alpaca_fetch_1m,
    get_universe_symbols as _alpaca_universe,
)
from common.signal_bridge import (
    send_to_broker,           # usage: send_to_broker(symbol, sig, strategy_tag="ml_pattern")
    close_all_positions,
    list_positions,
    get_account_equity,
)

# ==============================
# ENV / CONFIG (unchanged where possible)
# ==============================
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "10"))
DRY_RUN = os.getenv("DRY_RUN", "0").lower() in ("1","true","yes")
PAPER_MODE = os.getenv("PAPER_MODE", "true").lower() != "false"
SCANNER_DEBUG = os.getenv("SCANNER_DEBUG", "0").lower() in ("1","true","yes")

# Timeframes list (comma-separated)
TF_MIN_LIST = [int(x) for x in os.getenv("TF_MIN_LIST", "1,2,3,5,10").split(",") if x.strip()]

# Universe paging/size (Polygon-era knobs; keep but no-op for Alpaca paging)
MAX_UNIVERSE_PAGES = int(os.getenv("MAX_UNIVERSE_PAGES", "3"))
SCAN_BATCH_SIZE = int(os.getenv("SCAN_BATCH_SIZE", "150"))

# Liquidity filter (kept for compatibility; grouped-daily prefilter is skipped under Alpaca)
SCANNER_MIN_AVG_VOL = int(os.getenv("SCANNER_MIN_AVG_VOL", "0"))   # not used under Alpaca, kept for parity

# Sizing / gates
EQUITY_USD  = float(os.getenv("EQUITY_USD",  "100000"))
RISK_PCT    = float(os.getenv("RISK_PCT",    "0.01"))
MAX_POS_PCT = float(os.getenv("MAX_POS_PCT", "0.10"))
MIN_QTY     = int(os.getenv("MIN_QTY", "1"))
ROUND_LOT   = int(os.getenv("ROUND_LOT","1"))
MIN_PRICE   = float(os.getenv("MIN_PRICE", "3.0"))

# Model thresholds (env-driven to match your other workers)
CONF_THR = float(os.getenv("CONF_THR", "0.70"))
R_MULT   = float(os.getenv("R_MULT",   "1.50"))

# Session controls
SCANNER_MARKET_HOURS_ONLY = os.getenv("SCANNER_MARKET_HOURS_ONLY","1").lower() in ("1","true","yes")
ALLOW_PREMARKET  = os.getenv("ALLOW_PREMARKET","0").lower() in ("1","true","yes")
ALLOW_AFTERHOURS = os.getenv("ALLOW_AFTERHOURS","0").lower() in ("1","true","yes")
MARKET_TZ = os.getenv("MARKET_TZ", "America/New_York")

# --- Daily guard (uses broker equity via Alpaca) ---
START_EQUITY         = float(os.getenv("START_EQUITY", "100000"))
DAILY_TP_PCT         = float(os.getenv("DAILY_TP_PCT", "0.10"))   # +10%
DAILY_DD_PCT         = float(os.getenv("DAILY_DD_PCT", "0.05"))   # -5%
DAILY_FLATTEN_ON_HIT = os.getenv("DAILY_FLATTEN_ON_HIT","1").lower() in ("1","true","yes")
DAILY_GUARD_ENABLED  = os.getenv("DAILY_GUARD_ENABLED","1").lower() in ("1","true","yes")

# Establish baseline once per trading day (at/after 09:30 ET by default)
USE_BROKER_EQUITY_GUARD = True
SESSION_BASELINE_AT_0930 = os.getenv("SESSION_BASELINE_AT_0930","1").lower() in ("1","true","yes")
SESSION_START_EQUITY_ENV = os.getenv("SESSION_START_EQUITY","").strip()  # optional fixed baseline
SESSION_START_EQUITY = float(SESSION_START_EQUITY_ENV) if SESSION_START_EQUITY_ENV else None

# Render boot info (for logs)
RENDER_GIT_COMMIT = os.getenv("RENDER_GIT_COMMIT", "unknown")[:12]
RENDER_GIT_BRANCH = os.getenv("RENDER_GIT_BRANCH", os.getenv("BRANCH", "unknown"))
RUN_ID = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")

# ==============================
# State
# ==============================
COUNTS = defaultdict(int)
COMBO_COUNTS = defaultdict(int)
_sent = set()
_round_robin = 0
HALT_TRADING = False

DAY_STAMP = datetime.now().astimezone().strftime("%Y-%m-%d")
EQUITY_BASELINE_DATE = None
EQUITY_BASELINE_SET  = False
EQUITY_HIGH_WATER    = None

# ==============================
# Session helpers
# ==============================
def _now_et():
    return datetime.now(timezone.utc).astimezone(ZoneInfo(MARKET_TZ))

def _after_0930_now():
    ts = _now_et()
    return (ts.hour > 9) or (ts.hour == 9 and ts.minute >= 30)

def _market_session_now():
    now_et = _now_et()
    t = now_et.time()
    rth_start = (9,30); rth_end = (16,0)
    pre_start = (4,0);  pre_end = (9,30)
    ah_start  = (16,0); ah_end  = (20,0)

    def within(start, end):
        (sh, sm), (eh, em) = start, end
        ts = datetime(1,1,1, t.hour, t.minute).time()
        return (ts >= datetime(1,1,1,sh,sm).time()) and (ts < datetime(1,1,1,eh,em).time())

    in_rth = within(rth_start, rth_end)
    in_pre = ALLOW_PREMARKET  and within(pre_start, pre_end)
    in_ah  = ALLOW_AFTERHOURS and within(ah_start, ah_end)
    return in_rth or in_pre or in_ah

# ==============================
# Sizing helper (unchanged)
# ==============================
def _position_qty(entry_price: float, stop_price: float,
                  equity=EQUITY_USD, risk_pct=RISK_PCT, max_pos_pct=MAX_POS_PCT,
                  min_qty=MIN_QTY, round_lot=ROUND_LOT) -> int:
    if entry_price is None or stop_price is None:
        return 0
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share <= 0:
        return 0
    qty_risk     = (equity * risk_pct) / risk_per_share
    qty_notional = (equity * max_pos_pct) / max(1e-9, entry_price)
    qty = math.floor(max(min(qty_risk, qty_notional), 0) / max(1, round_lot)) * max(1, round_lot)
    return int(max(qty, min_qty if qty > 0 else 0))

# ==============================
# Data fetchers (Alpaca replacement for Polygon)
# ==============================
def fetch_bars_1m(symbol: str, lookback_minutes: int = 2400) -> pd.DataFrame:
    """
    Backward-compatible wrapper around Alpaca minute bars.
    Returns tz-aware ET ohlcv DataFrame (open/high/low/close/volume).
    """
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
    cols = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    out = df[cols].copy() if cols else pd.DataFrame()
    # gate on min price (consistent with your other workers)
    try:
        if not out.empty and float(out["close"].iloc[-1]) < MIN_PRICE:
            return pd.DataFrame()
    except Exception:
        pass
    return out

def _resample(df1m: pd.DataFrame, tf_min: int) -> pd.DataFrame:
    if df1m is None or df1m.empty:
        return pd.DataFrame()
    rule = f"{int(tf_min)}min"
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    bars = df1m.resample(rule, origin="start_day", label="right").agg(agg).dropna()
    try:
        bars.index = bars.index.tz_convert(MARKET_TZ)
    except Exception:
        bars.index = bars.index.tz_localize("UTC").tz_convert(MARKET_TZ)
    return bars

def build_universe() -> list[str]:
    """
    Env override OR Alpaca assets universe.
    Polygon grouped-daily prefilter has no 1:1 in Alpaca; skipped.
    """
    manual = os.getenv("SCANNER_SYMBOLS", "").strip()
    if manual:
        syms = [s.strip().upper() for s in manual.split(",") if s.strip()]
        print(f"[BOOT] SCANNER_SYMBOLS override detected ({len(syms)} symbols).", flush=True)
        return syms
    syms = _alpaca_universe(limit=10000)
    if SCANNER_DEBUG:
        print(f"[UNIVERSE] fetched {len(syms)} tickers via Alpaca assets.", flush=True)
        if SCANNER_MIN_AVG_VOL:
            print("[UNIVERSE] NOTE: Polygon grouped-daily prefilter skipped under Alpaca.", flush=True)
    return syms

# ==============================
# ML strategy adapter (unchanged – only plumbing)
# ==============================
def signal_ml_pattern(symbol: str, df1m: pd.DataFrame, tf_min: int,
                      conf_threshold: float = CONF_THR,
                      n_estimators: int = 100,
                      r_multiple: float = R_MULT,
                      min_volume_mult: float = 0.0):
    """
    Live-time adapter mimicking your backtest logic:
      - train simple RF on features
      - if last bar predicts 1 with prob>threshold AND volume > min_volume_mult * rolling avg
      - go long with 1% SL and TP = r_multiple * 1% above entry
    Returns a dict signal; order routing handled by Alpaca bridge.
    """
    try:
        import pandas_ta as ta  # noqa: F401
        from sklearn.ensemble import RandomForestClassifier
    except Exception as e:
        if SCANNER_DEBUG:
            print(f"[ML IMPORT WARN] {symbol} tf={tf_min}: {e}", flush=True)
        return None

    bars = _resample(df1m, tf_min)
    if bars is None or bars.empty or len(bars) < 120:
        return None

    bars = bars.copy()
    bars["return"] = bars["close"].pct_change()
    try:
        import pandas_ta as ta
        bars["rsi"] = ta.rsi(bars["close"], length=14)
    except Exception:
        delta = bars["close"].diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = -delta.clip(upper=0).rolling(14).mean()
        rs = up / (down.replace(0, np.nan))
        bars["rsi"] = 100 - (100 / (1 + rs))
    bars["volatility"] = bars["close"].rolling(20).std()
    bars.dropna(inplace=True)
    if len(bars) < 100:
        return None

    X = bars[["return", "rsi", "volatility"]]
    y = (bars["close"].shift(-1) > bars["close"]).astype(int)

    cut = int(len(X) * 0.7)
    if cut < 50:
        return None
    X_train, y_train = X.iloc[:cut], y.iloc[:cut]
    X_live = X.iloc[cut:]
    if X_live.empty:
        return None

    try:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
    except Exception as e:
        if SCANNER_DEBUG:
            print(f"[ML TRAIN ERR] {symbol} tf={tf_min}: {e}", flush=True)
        return None

    prob = model.predict_proba(X_live)[:, 1]
    preds = (prob > 0.5).astype(int)

    bars_live = bars.iloc[cut:].copy()
    bars_live["prediction"] = preds
    bars_live["confidence"] = prob

    avg_volume = bars_live["volume"].rolling(50).mean().fillna(bars_live["volume"].mean())
    last = bars_live.iloc[-1]
    ts = bars_live.index[-1]

    if (last["prediction"] == 1) and (last["confidence"] >= conf_threshold):
        if min_volume_mult > 0.0:
            try:
                i = len(bars_live) - 1
                if not (last["volume"] > min_volume_mult * avg_volume.iloc[i]):
                    return None
            except Exception:
                pass

        entry = float(last["close"])
        sl = entry * 0.99
        tp = entry * (1.0 + 0.01 * r_multiple)
        if entry < MIN_PRICE:
            return None
        qty = _position_qty(entry, sl)
        if qty <= 0:
            return None

        return {
            "action": "buy",
            "orderType": "market",
            "quantity": int(qty),
            "entry": entry,
            "tp_abs": float(tp),
            "sl_abs": float(sl),
            "barTime": ts.tz_convert("UTC").isoformat(),
            "meta": {"strategy": "ml_pattern", "timeframe": f"{int(tf_min)}m"}
        }

    return None

# ==============================
# Guard helpers (broker-equity guard via Alpaca)
# ==============================
def reset_daily_guard_if_new_day():
    global DAY_STAMP, HALT_TRADING, EQUITY_BASELINE_DATE, EQUITY_BASELINE_SET, EQUITY_HIGH_WATER
    today = datetime.now().astimezone().strftime("%Y-%m-%d")
    if today != DAY_STAMP:
        DAY_STAMP = today
        HALT_TRADING = False
        EQUITY_BASELINE_DATE = None
        EQUITY_BASELINE_SET  = False
        EQUITY_HIGH_WATER    = None
        print(f"[DAILY-GUARD] New day -> reset. START_EQUITY={START_EQUITY:.2f} Day={DAY_STAMP}", flush=True)

def ensure_session_baseline():
    """
    Establish session baseline once/day.
    If SESSION_START_EQUITY provided, use that; else pull live from Alpaca via get_account_equity().
    If SESSION_BASELINE_AT_0930=1, wait until >= 09:30 ET.
    """
    global SESSION_START_EQUITY, EQUITY_BASELINE_DATE, EQUITY_BASELINE_SET, EQUITY_HIGH_WATER
    if not DAILY_GUARD_ENABLED or not USE_BROKER_EQUITY_GUARD:
        return
    today = _now_et().date().isoformat()
    if EQUITY_BASELINE_DATE != today:
        EQUITY_BASELINE_DATE = today
        EQUITY_BASELINE_SET = False
        EQUITY_HIGH_WATER = None
    if EQUITY_BASELINE_SET:
        return
    if SESSION_BASELINE_AT_0930 and not _after_0930_now():
        return
    if SESSION_START_EQUITY is None:
        try:
            SESSION_START_EQUITY = float(get_account_equity() or START_EQUITY)
        except Exception:
            SESSION_START_EQUITY = START_EQUITY
    EQUITY_HIGH_WATER = SESSION_START_EQUITY
    EQUITY_BASELINE_SET = True
    print(f"[SESSION] Baseline set: session_start_equity={SESSION_START_EQUITY:.2f}", flush=True)

def _active_equity_and_limits():
    """
    Returns (equity_now, up_limit, down_limit, mode_str)
    Uses broker equity (Alpaca) as the source of truth.
    """
    base = SESSION_START_EQUITY if (SESSION_START_EQUITY is not None) else START_EQUITY
    try:
        eq_now = float(get_account_equity() or base)
    except Exception:
        eq_now = base
    up_lim = base * (1.0 + DAILY_TP_PCT)
    dn_lim = base * (1.0 - DAILY_DD_PCT)
    return eq_now, up_lim, dn_lim, "broker"

def check_daily_guard_and_maybe_halt():
    """
    Enforces profit target / drawdown.
    On hit: sets HALT_TRADING and optionally flattens positions via Alpaca.
    """
    global HALT_TRADING, EQUITY_HIGH_WATER
    if not DAILY_GUARD_ENABLED:
        return
    ensure_session_baseline()
    eq_now, up_lim, dn_lim, mode = _active_equity_and_limits()

    # trailing high-water (placeholder if you want trailing DD later)
    if EQUITY_HIGH_WATER is None:
        EQUITY_HIGH_WATER = eq_now
    else:
        EQUITY_HIGH_WATER = max(EQUITY_HIGH_WATER, eq_now)

    print(f"[DAILY-GUARD:{mode}] eq={eq_now:.2f} "
          f"targets +{DAILY_TP_PCT*100:.1f}%({up_lim:.2f}) / -{DAILY_DD_PCT*100:.1f}%({dn_lim:.2f}) "
          f"high_water={EQUITY_HIGH_WATER:.2f}", flush=True)

    if HALT_TRADING:
        return

    if eq_now >= up_lim:
        HALT_TRADING = True
        print(f"[DAILY-GUARD:{mode}] ✅ Profit target hit. Halting entries.", flush=True)
        if DAILY_FLATTEN_ON_HIT:
            ok, info = close_all_positions()
            print(f"[DAILY-GUARD:{mode}] Flatten -> ok={ok} info={info}", flush=True)
    elif eq_now <= dn_lim:
        HALT_TRADING = True
        print(f"[DAILY-GUARD:{mode}] ⛔ Drawdown limit hit. Halting entries.", flush=True)
        if DAILY_FLATTEN_ON_HIT:
            ok, info = close_all_positions()
            print(f"[DAILY-GUARD:{mode}] Flatten -> ok={ok} info={info}", flush=True)

# ==============================
# Dedupe helper
# ==============================
def _dedupe_key(symbol: str, tf: int, action: str, bar_time: str) -> str:
    raw = f"{symbol}|{tf}|{action}|{bar_time}"
    import hashlib
    return hashlib.sha256(raw.encode()).hexdigest()

# ==============================
# Scanner loop
# ==============================
def scan_once(universe: list[str]):
    global _round_robin, HALT_TRADING

    # Respect EOD halt
    if HALT_TRADING:
        if SCANNER_DEBUG:
            print("[SCAN] HALT_TRADING active — skipping new entries.", flush=True)
        return

    if SCANNER_MARKET_HOURS_ONLY and not _market_session_now():
        if SCANNER_DEBUG:
            print("[SCAN] Skipping — market session closed.", flush=True)
        return

    if not universe:
        return

    N = len(universe)
    start = _round_robin % max(1, N)
    end = min(N, start + SCAN_BATCH_SIZE)
    batch = universe[start:end]
    _round_robin = end if end < N else 0

    if SCANNER_DEBUG:
        print(f"[SCAN] symbols {start}:{end} / {N}  (batch={len(batch)})", flush=True)

    for sym in batch:
        try:
            df1m = fetch_bars_1m(sym, lookback_minutes=max(240, max(TF_MIN_LIST)*240))
            if df1m is None or df1m.empty:
                continue
            try:
                df1m.index = df1m.index.tz_convert(MARKET_TZ)
            except Exception:
                df1m.index = df1m.index.tz_localize("UTC").tz_convert(MARKET_TZ)
        except Exception as e:
            if SCANNER_DEBUG:
                print(f"[FETCH ERR] {sym}: {e}", flush=True)
            continue

        for tf in TF_MIN_LIST:
            try:
                if HALT_TRADING:
                    break

                sig = signal_ml_pattern(
                    sym, df1m, tf_min=tf,
                    conf_threshold=CONF_THR,
                    n_estimators=100,
                    r_multiple=R_MULT,
                    min_volume_mult=0.0
                )
                if not sig:
                    continue

                k = _dedupe_key(sym, tf, sig["action"], sig.get("barTime",""))
                if k in _sent:
                    continue
                _sent.add(k)

                if SCANNER_MARKET_HOURS_ONLY and not _market_session_now():
                    continue  # session flipped while scanning

                # ---- Alpaca bridge send (plumbing swap) ----
                ok, info = send_to_broker(sym, sig, strategy_tag="ml_pattern")

                COUNTS["signals"] += 1
                stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{stamp}] ml_pattern {sym} {tf}m -> {sig['action']} qty={sig['quantity']} | {info}", flush=True)
                if ok:
                    COMBO_COUNTS[f"{sym}|{tf}::orders.ok"] += 1
                else:
                    COMBO_COUNTS[f"{sym}|{tf}::orders.err"] += 1

            except Exception as e:
                if SCANNER_DEBUG:
                    import traceback
                    print(f"[SCAN ERR] {sym} {tf}m: {e}\n{traceback.format_exc()}", flush=True)
                continue

# ==============================
# Main
# ==============================
def main():
    # ---- Boot banner ----
    print(f"[BOOT] RUN_ID={RUN_ID} BRANCH={RENDER_GIT_BRANCH} COMMIT={RENDER_GIT_COMMIT} START_CMD=python app.py", flush=True)
    print(f"[BOOT] PAPER_MODE={'paper' if PAPER_MODE else 'live'} POLL_SECONDS={POLL_SECONDS} TFs={TF_MIN_LIST}", flush=True)
    print(f"[BOOT] CONF_THR={CONF_THR} R_MULT={R_MULT}", flush=True)

    universe = build_universe()
    up_lim = START_EQUITY * (1.0 + DAILY_TP_PCT)
    dn_lim = START_EQUITY * (1.0 - DAILY_DD_PCT)
    print(
        f"[BOOT] DAILY_GUARD_ENABLED={int(DAILY_GUARD_ENABLED)} "
        f"UP={DAILY_TP_PCT:.0%}({up_lim:.2f}) DOWN={DAILY_DD_PCT:.0%}({dn_lim:.2f}) "
        f"FLATTEN={int(DAILY_FLATTEN_ON_HIT)} START_EQUITY={START_EQUITY:.2f}",
        flush=True
    )
    print(f"[UNIVERSE] size={len(universe)}  TFs={TF_MIN_LIST}  Batch={SCAN_BATCH_SIZE}", flush=True)

    while True:
        loop_start = time.time()
        try:
            # ---- daily baseline & guard enforcement ----
            reset_daily_guard_if_new_day()
            check_daily_guard_and_maybe_halt()

            # Respect halt
            if HALT_TRADING:
                time.sleep(POLL_SECONDS)
                continue

            # ---- main scan tick ----
            scan_once(universe)

            # ---- EOD management (PRE-CLOSE + SAFETY NET) ----
            now_et = _now_et()

            # Pre-close auto-flatten window (3:50–4:00 ET)
            if now_et.hour == 15 and now_et.minute >= 50:
                print("[EOD] Pre-close flatten window (3:50–4:00 ET).", flush=True)
                ok, info = close_all_positions()
                print(f"[EOD] Alpaca flatten -> ok={ok} info={info}", flush=True)
                # Halt new entries for the remainder of the day
                global HALT_TRADING
                HALT_TRADING = True

            # Safety net just after the bell (4:00–4:02 ET)
            elif now_et.hour == 16 and now_et.minute < 3:
                pos = list_positions()
                if pos:
                    print(f"[EOD] Safety net: positions still open ({len(pos)}). Flattening…", flush=True)
                    ok, info = close_all_positions()
                    print(f"[EOD] Safety flatten -> ok={ok} info={info}", flush=True)
                    # keep HALT_TRADING = True

        except Exception as e:
            import traceback
            print("[LOOP ERROR]", e, traceback.format_exc(), flush=True)

        elapsed = time.time() - loop_start
        time.sleep(max(1, POLL_SECONDS - int(elapsed)))

if __name__ == "__main__":
    main()
