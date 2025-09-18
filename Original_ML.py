# app_scanner.py — dynamic market scanner -> TradersPost (paper/live)
# Requires: pandas, numpy, requests, pandas_ta, scikit-learn

import os, time, json, math, requests
import pandas as pd, numpy as np
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
from zoneinfo import ZoneInfo

# ==============================
# ENV / CONFIG
# ==============================
TP_URL = os.getenv("TP_WEBHOOK_URL", "")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")

# Poll cadence & diagnostics
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "10"))
DRY_RUN = os.getenv("DRY_RUN", "0").lower() in ("1","true","yes")
PAPER_MODE = os.getenv("PAPER_MODE", "true").lower() != "false"
SCANNER_DEBUG = os.getenv("SCANNER_DEBUG", "0").lower() in ("1","true","yes")

# Timeframes list (comma-separated)
TF_MIN_LIST = [int(x) for x in os.getenv("TF_MIN_LIST", "5").split(",")]

# Universe paging/size
MAX_UNIVERSE_PAGES = int(os.getenv("MAX_UNIVERSE_PAGES", "3"))   # pages of /v3/reference/tickers (1000 per page)
SCAN_BATCH_SIZE = int(os.getenv("SCAN_BATCH_SIZE", "150"))        # how many tickers per loop iteration

# Liquidity filter (daily volume)
SCANNER_MIN_AVG_VOL = int(os.getenv("SCANNER_MIN_AVG_VOL", "1000000"))  # default 1,000,000 shares

RUN_ID = datetime.now().astimezone().strftime("%Y-%m-%d")
COUNTS = defaultdict(int)        # global counters
COMBO_COUNTS = defaultdict(int)  # per symbol|tf
_sent = set()                    # dedupe key set
_round_robin = 0                 # rotate universe

# ---- Global position sizing (aligned with your Colab wrapper semantics) ----
EQUITY_USD  = float(os.getenv("EQUITY_USD",  "100000"))
RISK_PCT    = float(os.getenv("RISK_PCT",    "0.01"))
MAX_POS_PCT = float(os.getenv("MAX_POS_PCT", "0.10"))
MIN_QTY     = int(os.getenv("MIN_QTY", "1"))
ROUND_LOT   = int(os.getenv("ROUND_LOT","1"))

SCANNER_MARKET_HOURS_ONLY = os.getenv("SCANNER_MARKET_HOURS_ONLY","1").lower() in ("1","true","yes")
ALLOW_PREMARKET  = os.getenv("ALLOW_PREMARKET","0").lower() in ("1","true","yes")
ALLOW_AFTERHOURS = os.getenv("ALLOW_AFTERHOURS","0").lower() in ("1","true","yes")

def _market_session_now():
    now_et = datetime.now(ZoneInfo("America/New_York"))
    t = now_et.time()
    # Sessions (ET)
    rth_start = (9,30)
    rth_end   = (16,0)
    pre_start = (4,0)
    pre_end   = (9,30)
    ah_start  = (16,0)
    ah_end    = (20,0)

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
# Data fetchers (Polygon)
# ==============================
def _get(url, params=None, timeout=15):
    params = params or {}
    if POLYGON_API_KEY:
        params["apiKey"] = POLYGON_API_KEY
    r = requests.get(url, params=params, timeout=timeout)
    if r.status_code != 200:
        if SCANNER_DEBUG:
            print(f"[HTTP {r.status_code}] {url} -> {r.text[:200]}", flush=True)
        return None
    return r.json()

def fetch_polygon_1m(symbol: str, lookback_minutes: int = 2400) -> pd.DataFrame:
    """Fetch recent 1m bars as tz-aware ET ohlcv DataFrame."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=lookback_minutes)
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
    js = _get(url, {"adjusted":"true","sort":"asc","limit":"50000"})
    if not js or not js.get("results"):
        return pd.DataFrame()
    df = pd.DataFrame(js["results"])
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    df.index = df.index.tz_convert("America/New_York")
    df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
    return df[["open","high","low","close","volume"]]

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

def fetch_polygon_universe(max_pages: int = 3) -> list[str]:
    """Pull active US stock tickers via /v3/reference/tickers (paginated)."""
    out = []
    url = "https://api.polygon.io/v3/reference/tickers"
    page_token = None
    pages = 0
    while pages < max_pages:
        params = {
            "market": "stocks",
            "active": "true",
            "limit": 1000
        }
        if page_token:
            params["page_token"] = page_token
        js = _get(url, params)
        if not js or not js.get("results"):
            break
        for row in js["results"]:
            sym = row.get("ticker")
            if sym and sym.isalpha():  # simple guard, ignore funny symbols
                out.append(sym)
        page_token = js.get("next_url", None)
        if page_token and "page_token=" in page_token:
            page_token = page_token.split("page_token=")[-1]
        pages += 1
        if not page_token:
            break
    if SCANNER_DEBUG:
        print(f"[UNIVERSE] fetched {len(out)} tickers across {pages} page(s).", flush=True)
    return out

def filter_by_daily_volume(tickers: list[str], min_vol: int) -> list[str]:
    """Use grouped daily to get today's volumes quickly, then filter."""
    if not tickers:
        return []
    # today's date ET
    today_et = datetime.now(timezone.utc).astimezone().date().strftime("%Y-%m-%d")
    url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{today_et}"
    js = _get(url, {"adjusted":"true"})
    if not js or not js.get("results"):
        return tickers  # fallback: no filter
    vol_map = {row["T"]: row.get("v", 0) for row in js["results"] if "T" in row}
    filtered = [t for t in tickers if vol_map.get(t, 0) >= min_vol]
    if SCANNER_DEBUG:
        print(f"[VOL FILTER] {len(tickers)} -> {len(filtered)} tickers with v≥{min_vol}", flush=True)
    return filtered

# ==============================
# ML strategy adapter (live signal)
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
    Returns a dict signal compatible with TradersPost payload builder.
    """
    # Lazy imports (so worker still boots if packages missing)
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

    # features
    bars = bars.copy()
    bars["return"] = bars["close"].pct_change()
    # RSI via pandas-ta
    try:
        import pandas_ta as ta
        bars["rsi"] = ta.rsi(bars["close"], length=14)
    except Exception:
        # fallback simple RSI-ish
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

    # train/test split
    train_size = int(len(X) * 0.7)
    if train_size < 50:
        return None
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_test = X.iloc[train_size:]
    if X_test.empty:
        return None

    # train RF
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
    except Exception as e:
        if SCANNER_DEBUG:
            print(f"[ML TRAIN ERR] {symbol} tf={tf_min}: {e}", flush=True)
        return None

    # infer last row
    prob = model.predict_proba(X_test)[:, 1]
    preds = (prob > 0.5).astype(int)

    bars_live = bars.iloc[train_size:].copy()
    bars_live["prediction"] = preds
    bars_live["confidence"] = prob

    # volume filter proxy (rolling mean over available test bars)
    avg_volume = bars_live["volume"].rolling(50).mean().fillna(bars_live["volume"].mean())

    last = bars_live.iloc[-1]
    ts = bars_live.index[-1]
    if (last["prediction"] == 1) and (last["confidence"] >= conf_threshold):
        if min_volume_mult > 0.0:
            try:
                # guard if rolling mean NaN very early
                i = len(bars_live) - 1
                if not (last["volume"] > min_volume_mult * avg_volume.iloc[i]):
                    return None
            except Exception:
                pass

        entry = float(last["close"])
        sl = entry * 0.99                            # 1% stop
        tp = entry * (1.0 + 0.01 * r_multiple)       # r_multiple * 1% take profit
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
# TradersPost helpers
# ==============================
def _dedupe_key(symbol: str, tf: int, action: str, bar_time: str) -> str:
    raw = f"{symbol}|{tf}|{action}|{bar_time}"
    import hashlib
    return hashlib.sha256(raw.encode()).hexdigest()

def build_payload(symbol: str, sig: dict) -> dict:
    """
    TradersPost requires absolute TP/SL as nested objects.
    """
    payload = {
        "ticker": symbol,
        "action": sig["action"],
        "orderType": sig.get("orderType", "market"),
        "quantity": int(sig["quantity"]),
        "meta": sig.get("meta", {})
    }

    # absolute prices
    tp_abs = sig.get("tp_abs")
    sl_abs = sig.get("sl_abs")
    if tp_abs is not None:
        payload["takeProfit"] = {"limitPrice": float(round(tp_abs, 2))}
    if sl_abs is not None:
        payload["stopLoss"] = {"type": "stop", "stopPrice": float(round(sl_abs, 2))}

    # helpful audit fields
    payload["meta"]["environment"] = "paper" if PAPER_MODE else "live"
    if sig.get("barTime"):
        payload["meta"]["barTime"] = sig["barTime"]
    payload["meta"]["runId"] = RUN_ID
    return payload

def send_to_traderspost(payload: dict):
    if DRY_RUN:
        print(f"[DRY-RUN] {json.dumps(payload)[:500]}", flush=True)
        return True, "dry-run"
    if not TP_URL:
        print("[ERROR] TP_WEBHOOK_URL is missing/empty.", flush=True)
        return False, "no-webhook"
    try:
        r = requests.post(TP_URL, json=payload, timeout=12)
        ok = 200 <= r.status_code < 300
        info = f"{r.status_code} {r.text[:300]}"
        if not ok:
            print(f"[POST ERROR] {info}", flush=True)
        return ok, info
    except Exception as e:
        import traceback
        print("[POST EXCEPTION]", e, traceback.format_exc(), flush=True)
        return False, f"exception: {e}"

# ==============================
# Scanner loop
# ==============================
def build_universe():
    # 1) fetch active stocks (paged)
    universe = fetch_polygon_universe(MAX_UNIVERSE_PAGES)
    if not universe:
        print("[UNIVERSE] Empty; check POLYGON_API_KEY / permissions.", flush=True)
        return []

    # 2) volume filter using grouped daily
    liquid = filter_by_daily_volume(universe, SCANNER_MIN_AVG_VOL)
    if not liquid:
        print("[UNIVERSE] Volume filter removed everything; lowering threshold?", flush=True)
        return universe  # fallback to full set if filter too strict
    return liquid

def scan_once(universe: list[str]):
    global _round_robin

    if SCANNER_MARKET_HOURS_ONLY and not _market_session_now():
        if SCANNER_DEBUG:
            print("[SCAN] Skipping — market session closed.", flush=True)
        return

    if not universe:
        return

    # rotate through universe in batches to control request rate
    N = len(universe)
    start = _round_robin % max(1, N)
    end = min(N, start + SCAN_BATCH_SIZE)
    batch = universe[start:end]
    _round_robin = end if end < N else 0

    if SCANNER_DEBUG:
        print(f"[SCAN] symbols {start}:{end} / {N}  (batch={len(batch)})", flush=True)

    for sym in batch:
        # pull once and reuse for all TFs to save API calls
        try:
            df1m = fetch_polygon_1m(sym, lookback_minutes=2400)  # ~ 40 hours
            if df1m is None or df1m.empty:
                continue
            # ensure tz ET
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
                    min_volume_mult=0.0  # you said you only care that it's tradable; set >0 to enforce volume spike
                )
                if not sig:
                    continue

                k = _dedupe_key(sym, tf, sig["action"], sig.get("barTime",""))
                if k in _sent:
                    continue
                _sent.add(k)

                if SCANNER_MARKET_HOURS_ONLY and not _market_session_now():
                    continue  # if the session flipped while scanning, skip sends

                payload = build_payload(sym, sig)
                ok, info = send_to_traderspost(payload)
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
    if not POLYGON_API_KEY:
        print("[FATAL] POLYGON_API_KEY missing.", flush=True)
        return
    if not TP_URL and not DRY_RUN:
        print("[FATAL] TP_WEBHOOK_URL missing (or set DRY_RUN=1).", flush=True)
        return

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
