# app_scanner.py — Market-wide scanner + ML strategy → TradersPost
# Scans US stocks by intraday volume (Polygon), runs ML signal, sizes qty, posts orders.

import os, time, json, math, hashlib, requests, traceback
from datetime import datetime, timezone, timedelta
from collections import defaultdict

import pandas as pd
import numpy as np

# -------- Optional deps (don’t crash if missing; we’ll log and skip ML) --------
try:
    import pandas_ta as ta
except Exception:
    ta = None
try:
    from sklearn.ensemble import RandomForestClassifier
except Exception:
    RandomForestClassifier = None

# ==============================
# ENV / CONFIG
# ==============================
TP_URL = os.getenv("TP_WEBHOOK_URL", "")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")

PAPER_MODE = os.getenv("PAPER_MODE", "true").lower() in ("1", "true", "yes")
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "12"))

# Universe scan
MIN_DAY_VOL = float(os.getenv("MIN_DAY_VOL", "500000"))
EXCLUDE_OTC = os.getenv("EXCLUDE_OTC", "true").lower() in ("1","true","yes")
EXCLUDE_ETF = os.getenv("EXCLUDE_ETF", "true").lower() in ("1","true","yes")
UNIVERSE_MAX = int(os.getenv("UNIVERSE_MAX", "400"))
UNIVERSE_REFRESH_MIN = int(os.getenv("UNIVERSE_REFRESH_MIN", "20"))

# Strategy + sizing
SCAN_TF = [int(x.strip()) for x in os.getenv("SCAN_TF", "1,3,5,15").split(",") if x.strip()]
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.8"))
R_MULTIPLE = float(os.getenv("R_MULTIPLE", "3.0"))

# Global position sizing (same spirit as your Colab wrapper)
EQUITY_USD  = float(os.getenv("EQUITY_USD",  "100000"))
RISK_PCT    = float(os.getenv("RISK_PCT",    "0.01"))     # 1% risk
MAX_POS_PCT = float(os.getenv("MAX_POS_PCT", "0.10"))     # 10% notional cap
MIN_QTY     = int(os.getenv("MIN_QTY", "1"))
ROUND_LOT   = int(os.getenv("ROUND_LOT","1"))

RUN_ID = datetime.now().astimezone().strftime("%Y-%m-%d")

# ==============================
# Globals / Counters
# ==============================
_sent = set()  # de-dupe by (symbol, tf, action, barTime)
COUNTS = defaultdict(int)

# ==============================
# Helpers
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

def _dedupe_key(symbol: str, tf: int, action: str, bar_time: str) -> str:
    raw = f"{symbol}|{tf}|{action}|{bar_time}"
    return hashlib.sha256(raw.encode()).hexdigest()

def _is_rth(ts):
    h, m = ts.hour, ts.minute
    return ((h > 9) or (h == 9 and m >= 30)) and (h < 16)

# ==============================
# TradersPost
# ==============================
def build_payload(symbol: str, sig: dict) -> dict:
    """
    Accepts keys:
      action: 'buy'|'sell'
      orderType: 'market' (default)
      quantity: int
      entry: float (optional, for audit)
      tp_abs: float | takeProfit: float
      sl_abs: float | stopLoss: float
      barTime: ISO string
    """
    action     = sig.get("action")
    order_type = sig.get("orderType", "market")
    qty        = int(sig.get("quantity", 0))

    payload = {
        "ticker": symbol,
        "action": action,
        "orderType": order_type,
        "quantity": qty,
        "meta": {
            "environment": "paper" if PAPER_MODE else "live",
            "runId": RUN_ID,
        }
    }

    if sig.get("barTime"):
        payload["meta"]["barTime"] = sig["barTime"]

    # Normalize TP/SL to absolute
    tp_abs = sig.get("tp_abs")
    sl_abs = sig.get("sl_abs")
    if tp_abs is None and sig.get("takeProfit") is not None:
        tp_abs = float(sig["takeProfit"])
    if sl_abs is None and sig.get("stopLoss") is not None:
        sl_abs = float(sig["stopLoss"])

    if tp_abs is not None:
        payload["takeProfit"] = {"limitPrice": float(round(tp_abs, 2))}
    if sl_abs is not None:
        payload["stopLoss"] = {"type": "stop", "stopPrice": float(round(sl_abs, 2))}

    return payload

def send_to_traderspost(payload: dict):
    try:
        if DRY_RUN:
            print(f"[DRY-RUN] Would POST: {json.dumps(payload)[:400]}", flush=True)
            return True, "dry-run"

        if not TP_URL:
            print("[POST ERROR] TP_WEBHOOK_URL missing.", flush=True)
            return False, "no-webhook-url"

        r = requests.post(TP_URL, json=payload, timeout=12)
        ok = 200 <= r.status_code < 300
        info = f"{r.status_code} {r.text[:280]}"
        if not ok:
            print(f"[POST ERROR] {info}", flush=True)
        return ok, info
    except Exception as e:
        print("[POST EXCEPTION]", e, traceback.format_exc(), flush=True)
        return False, f"exception: {e}"

# ==============================
# Polygon data
# ==============================
def build_universe_from_polygon():
    """
    Returns a list of tickers filtered by intraday volume (today's cumulative),
    excluding OTC and ETFs (configurable). Sorted by volume desc, top UNIVERSE_MAX.
    """
    if not POLYGON_API_KEY:
        print("[UNIVERSE] No POLYGON_API_KEY.", flush=True)
        return []

    url = "https://api.polygon.io/v3/reference/tickers"
    params = {
        "market": "stocks",
        "active": "true",
        "limit": 1000,
        "include": "market_data"
    }

    out = []
    next_url = None

    while True:
        if next_url:
            u = next_url + f"&apiKey={POLYGON_API_KEY}"
            resp = requests.get(u, timeout=15)
        else:
            resp = requests.get(url, params={**params, "apiKey": POLYGON_API_KEY}, timeout=15)

        if resp.status_code != 200:
            print("[UNIVERSE] Error:", resp.status_code, resp.text[:200], flush=True)
            break

        js = resp.json()
        results = js.get("results", []) or []
        for item in results:
            sym = item.get("ticker")
            if not sym:
                continue

            # Filter: OTC / ETF if requested
            # Polygon reference type codes vary; skip if obvious ETF or non-NMS.
            if EXCLUDE_ETF and (item.get("type") == "ETF" or str(item.get("name","")).upper().endswith(" ETF")):
                continue
            if EXCLUDE_OTC and sym.startswith("OTC:"):
                continue

            dayv = ((item.get("day") or {}).get("v"))
            if dayv is None:
                continue
            try:
                v = float(dayv)
            except Exception:
                continue
            if v < MIN_DAY_VOL:
                continue

            out.append((sym, v))

        next_url = js.get("next_url")
        if not next_url:
            break
        # Keep the universe build reasonably quick
        if len(out) >= UNIVERSE_MAX * 2:
            break

    # Sort by volume desc, take top UNIVERSE_MAX
    out.sort(key=lambda t: t[1], reverse=True)
    tickers = [t for t, _ in out[:UNIVERSE_MAX]]
    print(f"[UNIVERSE] {len(tickers)} symbols (min_day_vol={MIN_DAY_VOL}, exclude_otc={EXCLUDE_OTC}, exclude_etf={EXCLUDE_ETF})", flush=True)
    return tickers

def fetch_polygon_1m(symbol: str, lookback_minutes: int = 2400) -> pd.DataFrame:
    """Fetch recent 1m bars as tz-aware ET index with o/h/l/c/volume."""
    if not POLYGON_API_KEY:
        return pd.DataFrame()
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=lookback_minutes)
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/"
        f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"
    )
    r = requests.get(url, timeout=15)
    if r.status_code != 200:
        return pd.DataFrame()
    rows = r.json().get("results", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    try:
        df.index = df.index.tz_convert("America/New_York")
    except Exception:
        df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
    df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
    return df[["open","high","low","close","volume"]]

def _resample(df1m: pd.DataFrame, tf_min: int) -> pd.DataFrame:
    rule = f"{int(tf_min)}min"
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    return df1m.resample(rule, origin="start_day", label="right").agg(agg).dropna()

# ==============================
# ML Strategy (signal on last bar)
# ==============================
def signal_ml_pattern(symbol: str, df1m: pd.DataFrame, tf_min: int,
                      conf_threshold: float = CONF_THRESHOLD,
                      n_estimators: int = 100,
                      r_multiple: float = R_MULTIPLE,
                      min_volume_mult: float = 0.0):
    """
    Convert your backtest_ml_pattern into a real-time "should I trade this bar?" signal.
    Long-only: if model predicts up with prob > threshold and volume condition passes.
    """
    if df1m is None or df1m.empty or ta is None or RandomForestClassifier is None:
        # Log missing deps once
        if ta is None:
            print("[ML] pandas-ta not installed; skipping.", flush=True)
        if RandomForestClassifier is None:
            print("[ML] scikit-learn not installed; skipping.", flush=True)
        return None

    bars = _resample(df1m, tf_min)
    if bars.empty or len(bars) < 120:  # need some history to train
        return None

    # Features
    bars = bars.copy()
    bars["return"] = bars["close"].pct_change()
    bars["rsi"] = ta.rsi(bars["close"], length=14)
    bars["volatility"] = bars["close"].rolling(20).std()
    bars = bars.dropna()
    if len(bars) < 80:
        return None

    features = bars[["return", "rsi", "volatility"]]
    target = (bars["close"].shift(-1) > bars["close"]).astype(int)

    # Train/test split
    split = int(len(features) * 0.7)
    if split <= 10 or len(features) - split <= 5:
        return None

    X_train = features.iloc[:split]
    y_train = target.iloc[:split]
    X_test  = features.iloc[split:]
    y_test  = target.iloc[split:]  # not used, but kept for parity

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"[ML] Fit error {symbol}: {e}", flush=True)
        return None

    prob = model.predict_proba([features.iloc[-1]])[0][1]  # prob up on last bar
    pred = 1 if prob > 0.5 else 0

    # Volume guard (rolling on the same resampled TF)
    avg_vol = bars["volume"].rolling(50).mean().fillna(bars["volume"].mean())
    vol_ok = float(bars["volume"].iloc[-1]) > min_volume_mult * float(avg_vol.iloc[-1])

    ts = bars.index[-1]
    price = float(bars["close"].iloc[-1])

    if pred == 1 and prob >= conf_threshold and vol_ok and _is_rth(ts):
        # Simple SL 1% below, TP = r_multiple * 1% above
        sl = price * 0.99
        tp = price * (1.0 + 0.01 * r_multiple)
        qty = _position_qty(price, sl)
        if qty <= 0:
            return None
        return {
            "action": "buy",
            "orderType": "market",
            "quantity": int(qty),
            "entry": price,
            "tp_abs": float(tp),
            "sl_abs": float(sl),
            "barTime": ts.tz_convert("UTC").isoformat(),
        }

    return None

# ==============================
# Main scanner loop
# ==============================
def main():
    print("Scanner starting…", flush=True)
    if not POLYGON_API_KEY:
        print("[BOOT] Missing POLYGON_API_KEY.", flush=True)
    if not TP_URL and not DRY_RUN:
        print("[BOOT] Missing TP_WEBHOOK_URL (set DRY_RUN=1 to test locally).", flush=True)

    last_universe_at = None
    universe = []

    while True:
        loop_t0 = time.time()
        try:
            # Refresh universe on schedule
            now = datetime.now(timezone.utc)
            if (last_universe_at is None) or (now - last_universe_at > timedelta(minutes=UNIVERSE_REFRESH_MIN)):
                print("[UNIVERSE] Refreshing…", flush=True)
                universe = build_universe_from_polygon()
                last_universe_at = now
                print(f"[UNIVERSE] {len(universe)} symbols ready.", flush=True)

            if not universe:
                print("[WARN] Universe empty. Sleeping…", flush=True)
                time.sleep(POLL_SECONDS)
                continue

            # Scan each symbol x timeframe
            for sym in universe:
                # Fetch once per symbol (reuse across TFs)
                df1m = fetch_polygon_1m(sym, lookback_minutes=2400)
                if df1m is None or df1m.empty:
                    continue

                for tf in SCAN_TF:
                    sig = signal_ml_pattern(sym, df1m, tf_min=tf,
                                            conf_threshold=CONF_THRESHOLD,
                                            r_multiple=R_MULTIPLE,
                                            min_volume_mult=0.0)
                    if not sig:
                        continue

                    k = _dedupe_key(sym, tf, sig["action"], sig.get("barTime",""))
                    if k in _sent:
                        continue
                    _sent.add(k)

                    payload = build_payload(sym, sig)
                    ok, info = send_to_traderspost(payload)
                    COUNTS["signals"] += 1
                    if ok:
                        COUNTS["orders.ok"] += 1
                    else:
                        COUNTS["orders.err"] += 1

                    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{stamp}] ML {sym} {tf}m -> {sig['action']} qty={sig['quantity']} | {info}", flush=True)

        except Exception as e:
            print("[LOOP ERROR]", e, traceback.format_exc(), flush=True)

        elapsed = time.time() - loop_t0
        time.sleep(max(1, POLL_SECONDS - int(elapsed)))

if __name__ == "__main__":
    main()
