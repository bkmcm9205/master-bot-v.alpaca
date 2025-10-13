# scanner_alpaca.py — dynamic market scanner (PLUMBING SWAP: Polygon/TP -> Alpaca)
# Requires: pandas, numpy, requests, pandas_ta, scikit-learn
#
# ✅ Models/ML logic unchanged. Only data + broker I/O rewired to Alpaca.

import os, time, json, math, requests
import pandas as pd, numpy as np
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
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
TF_MIN_LIST = [int(x) for x in os.getenv("TF_MIN_LIST", "5").split(",")]

# Universe paging/size (Polygon-era knobs; keep but no-op for Alpaca paging)
MAX_UNIVERSE_PAGES = int(os.getenv("MAX_UNIVERSE_PAGES", "3"))
SCAN_BATCH_SIZE = int(os.getenv("SCAN_BATCH_SIZE", "150"))

# Liquidity filter (kept for compatibility; grouped-daily prefilter is skipped under Alpaca)
SCANNER_MIN_AVG_VOL = int(os.getenv("SCANNER_MIN_AVG_VOL", "1000000"))

RUN_ID = datetime.now().astimezone().strftime("%Y-%m-%d")
COUNTS = defaultdict(int)
COMBO_COUNTS = defaultdict(int)
_sent = set()
_round_robin = 0

# ---- Global position sizing ----
EQUITY_USD  = float(os.getenv("EQUITY_USD",  "100000"))
RISK_PCT    = float(os.getenv("RISK_PCT",    "0.01"))
MAX_POS_PCT = float(os.getenv("MAX_POS_PCT", "0.10"))
MIN_QTY     = int(os.getenv("MIN_QTY", "1"))
ROUND_LOT   = int(os.getenv("ROUND_LOT","1"))

SCANNER_MARKET_HOURS_ONLY = os.getenv("SCANNER_MARKET_HOURS_ONLY","1").lower() in ("1","true","yes")
ALLOW_PREMARKET  = os.getenv("ALLOW_PREMARKET","0").lower() in ("1","true","yes")
ALLOW_AFTERHOURS = os.getenv("ALLOW_AFTERHOURS","0").lower() in ("1","true","yes")

# ==============================
# Session helpers (unchanged)
# ==============================
def _market_session_now():
    now_et = datetime.now(ZoneInfo("America/New_York"))
    t = now_et.time()
    rth_start = (9,30); rth_end = (16,0)
    pre_start = (4,0);  pre_end = (9,30)
    ah_start  = (16,0); ah_end  = (20,0)

    def within(start, end):
        (sh, sm), (eh, em) = start, end
        return (t >= datetime(1,1,1,sh,sm).time()) and (t < datetime(1,1,1,eh,em).time())

    in_rth = within(rth_start, rth_end)
    in_pre = ALLOW_PREMARKET  and within(pre_start, pre_end)
    in_ah  = ALLOW_AFTERHOURS and within(ah_start, ah_end)
    return in_rth or in_pre or in_ah

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
    end_iso = end.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    df = _alpaca_fetch_1m(symbol, start_iso=start_iso, end_iso=end_iso, limit=10000)
    if df is None or df.empty:
        return pd.DataFrame()
    # ensure ET index and expected columns
    try:
        df.index = df.index.tz_convert("America/New_York")
    except Exception:
        df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
    # adapters.data_alpaca should already yield ohlcv; keep a safe select:
    cols = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    return df[cols].copy() if cols else pd.DataFrame()

def _resample(df1m: pd.DataFrame, tf_min: int) -> pd.DataFrame:
    if df1m is None or df1m.empty:
        return pd.DataFrame()
    rule = f"{int(tf_min)}min"
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    bars = df1m.resample(rule, origin="start_day", label="right").agg(agg).dropna()
    try:
        bars.index = bars.index.tz_convert("America/New_York")
    except Exception:
        bars.index = bars.index.tz_localize("UTC").tz_convert("America/New_York")
    return bars

def build_universe() -> list[str]:
    """
    Env override OR Alpaca assets universe. Polygon grouped-daily volume prefilter
    has no 1:1 in Alpaca; we skip it (log once if debug).
    """
    manual = os.getenv("SCANNER_SYMBOLS", "").strip()
    if manual:
        return [s.strip().upper() for s in manual.split(",") if s.strip()]
    syms = _alpaca_universe(limit=10000)
    if SCANNER_DEBUG:
        print(f"[UNIVERSE] fetched {len(syms)} tickers via Alpaca assets.", flush=True)
        if SCANNER_MIN_AVG_VOL:
            print("[UNIVERSE] NOTE: Polygon grouped-daily prefilter skipped under Alpaca.", flush=True)
    return syms

# ==============================
# ML strategy adapter (unchanged)
# ==============================
def signal_ml_pattern(symbol: str, df1m: pd.DataFrame, tf_min: int,
                      conf_threshold: float = 0.8,
                      n_estimators: int = 100,
                      r_multiple: float = 3.0,
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

    train_size = int(len(X) * 0.7)
    if train_size < 50:
        return None
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_test = X.iloc[train_size:]
    if X_test.empty:
        return None

    try:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
    except Exception as e:
        if SCANNER_DEBUG:
            print(f"[ML TRAIN ERR] {symbol} tf={tf_min}: {e}", flush=True)
        return None

    prob = model.predict_proba(X_test)[:, 1]
    preds = (prob > 0.5).astype(int)

    bars_live = bars.iloc[train_size:].copy()
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
# Dedupe helper (unchanged)
# ==============================
def _dedupe_key(symbol: str, tf: int, action: str, bar_time: str) -> str:
    raw = f"{symbol}|{tf}|{action}|{bar_time}"
    import hashlib
    return hashlib.sha256(raw.encode()).hexdigest()

# ==============================
# Scanner loop (unchanged logic; Alpaca plumbing)
# ==============================
def scan_once(universe: list[str]):
    global _round_robin

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
            df1m = fetch_bars_1m(sym, lookback_minutes=2400)  # ~40h
            if df1m is None or df1m.empty:
                continue
            try:
                df1m.index = df1m.index.tz_convert("America/New_York")
            except Exception:
                df1m.index = df1m.index.tz_localize("UTC").tz_convert("America/New_York")
        except Exception as e:
            if SCANNER_DEBUG:
                print(f"[FETCH ERR] {sym}: {e}", flush=True)
            continue

        for tf in TF_MIN_LIST:
            try:
                sig = signal_ml_pattern(
                    sym, df1m, tf_min=tf,
                    conf_threshold=0.8,
                    n_estimators=100,
                    r_multiple=3.0,
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

def main():
    print("Scanner starting…", flush=True)

    universe = build_universe()
    print(f"[READY] Universe size: {len(universe)}  TFs: {TF_MIN_LIST}  Batch: {SCAN_BATCH_SIZE}", flush=True)

    while True:
        loop_start = time.time()
        try:
            scan_once(universe)
        except Exception as e:
            import traceback
            print("[LOOP ERROR]", e, traceback.format_exc(), flush=True)

        elapsed = time.time() - loop_start
        time.sleep(max(1, POLL_SECONDS - int(elapsed)))

if __name__ == "__main__":
    main()
