# app_scanner.py — market scanner + sentiment-gated auto-trader for ML strategy

import os, time, json, math, hashlib, requests
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

import pandas as pd
import numpy as np

# ==============================
# ENV / CONFIG
# ==============================
TP_URL              = os.getenv("TP_WEBHOOK_URL", "").strip()
POLYGON_API_KEY     = os.getenv("POLYGON_API_KEY", "").strip()

POLL_SECONDS        = int(os.getenv("POLL_SECONDS", "10"))

# Universe
SCAN_TICKERS        = os.getenv("SCAN_TICKERS", "").strip()  # optional override (CSV)
UNIVERSE_MAX        = int(os.getenv("UNIVERSE_MAX", "2000"))
UNIVERSE_CACHE_MIN  = int(os.getenv("UNIVERSE_CACHE_MIN", "30"))  # minutes

# Liquidity gates (absolute)
MIN_AVG_DAILY_VOL   = int(os.getenv("MIN_AVG_DAILY_VOL", "0"))
MIN_TODAY_VOL       = int(os.getenv("MIN_TODAY_VOL", "0"))

# Timeframes to scan
TF_MIN_LIST         = [int(x) for x in os.getenv("TF_MIN_LIST", "1,2,3,5,10").split(",") if x.strip()]

# Risk / sizing (Colab parity)
EQUITY_USD          = float(os.getenv("EQUITY_USD",  "100000"))
RISK_PCT            = float(os.getenv("RISK_PCT",    "0.01"))
MAX_POS_PCT         = float(os.getenv("MAX_POS_PCT", "0.10"))
MIN_QTY             = int(os.getenv("MIN_QTY", "1"))
ROUND_LOT           = int(os.getenv("ROUND_LOT", "1"))

# Run behavior
MAX_OPEN_POS        = int(os.getenv("MAX_OPEN_POS", "50"))
MAX_POSTS_PER_MIN   = int(os.getenv("MAX_POSTS_PER_MIN", "40"))

# Market hours (US RTH) + auto-flatten
AUTO_FLATTEN_MIN_BEFORE_CLOSE = int(os.getenv("AUTO_FLATTEN_MIN_BEFORE_CLOSE", "5"))
TRADE_ONLY_RTH                = os.getenv("TRADE_ONLY_RTH", "1") in ("1","true","yes")

# Sentiment gate
SENTIMENT_ON            = os.getenv("SENTIMENT_ON", "1") in ("1","true","yes")
SENTIMENT_SYMBOL        = os.getenv("SENTIMENT_SYMBOL", "SPY").strip()
SENTIMENT_TF            = int(os.getenv("SENTIMENT_TF", "5"))
SENTIMENT_CACHE_SEC     = int(os.getenv("SENTIMENT_CACHE_SEC", "60"))
SENTIMENT_NEUTRAL_ACTION= os.getenv("SENTIMENT_NEUTRAL_ACTION", "flat").lower()  # flat|long|short|both
SHORTS_ENABLED          = os.getenv("SHORTS_ENABLED", "0") in ("1","true","yes")

# Diagnostics
RUN_ID              = datetime.now().astimezone().strftime("%Y-%m-%d")
TP_SL_SAME_BAR      = os.getenv("TP_SL_SAME_BAR", "tp").lower()  # "tp" | "sl"

# Optional replay
REPLAY_ON_START     = os.getenv("REPLAY_ON_START", "0") in ("1","true","yes")
REPLAY_SYMBOL       = os.getenv("REPLAY_SYMBOL", "AAPL").strip()
REPLAY_TF           = int(os.getenv("REPLAY_TF", "3"))
REPLAY_HOURS        = int(os.getenv("REPLAY_HOURS", "24"))

# ==============================
# Globals
# ==============================
COUNTS        = defaultdict(int)
COMBO_COUNTS  = defaultdict(int)
_sent         = set()                # (symbol|tf|action|barTime) hash
OPEN_TRADES   = defaultdict(list)    # (symbol, tf) -> [LiveTrade]
POST_TIMES    = deque(maxlen=200)    # timestamps for throttle
_last_uni_ts  = 0
_uni_cache    = []

_last_sentiment = ("neutral", 0.0)   # (label, timestamp)

# ==============================
# Data classes
# ==============================
@dataclass
class LiveTrade:
    strategy: str
    symbol: str
    tf_min: int
    side: str         # "buy" | "sell"
    entry: float
    tp: float
    sl: float
    qty: int
    entry_time: str
    is_open: bool = True
    exit: float = None
    exit_time: str = None
    reason: str = None  # "tp" | "sl" | "flatten"

# ==============================
# Helpers: session, sizing, dedupe
# ==============================
def _is_rth(ts):
    h, m = ts.hour, ts.minute
    return ((h > 9) or (h == 9 and m >= 30)) and (h < 16)

def _near_close(ts, minutes=5):
    return (ts.hour == 15 and ts.minute >= (60 - minutes)) or (ts.hour == 16 and ts.minute == 0)

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

def _open_positions_count() -> int:
    return sum(t.is_open for lst in OPEN_TRADES.values() for t in lst)

# ==============================
# Polygon fetchers
# ==============================
def fetch_polygon_1m(symbol: str, lookback_minutes: int = 2400) -> pd.DataFrame:
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

def _poly_reference_symbols(limit=1000):
    if not POLYGON_API_KEY:
        return []
    out = []
    url = f"https://api.polygon.io/v3/reference/tickers?market=stocks&active=true&limit=1000&apiKey={POLYGON_API_KEY}"
    while url and len(out) < limit:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            break
        data = r.json()
        for row in data.get("results", []):
            t = row.get("ticker")
            primary = row.get("primary_exchange", "")
            _type = row.get("type", "")
            if t and _type in ("CS", "ADRC") and primary in ("XNYS", "XNAS", "ARCX", "BATS"):
                out.append(t)
                if len(out) >= limit:
                    break
        url = data.get("next_url")
        if url and "apiKey=" not in url:
            url = url + ("&" if "?" in url else "?") + f"apiKey={POLYGON_API_KEY}"
    return out

def _poly_daily_30d(symbol: str) -> pd.DataFrame:
    if not POLYGON_API_KEY:
        return pd.DataFrame()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=45)
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/"
        f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        f"?adjusted=true&sort=desc&limit=50&apiKey={POLYGON_API_KEY}"
    )
    r = requests.get(url, timeout=15)
    if r.status_code != 200:
        return pd.DataFrame()
    rows = r.json().get("results", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index(ascending=False)
    df = df.rename(columns={"v":"volume"})
    return df[["volume"]].head(30)

def _get_today_volume(symbol: str) -> int:
    df1m = fetch_polygon_1m(symbol, lookback_minutes=9*60)
    if df1m is None or df1m.empty:
        return 0
    try:
        df1m.index = df1m.index.tz_convert("America/New_York")
    except Exception:
        df1m.index = df1m.index.tz_localize("UTC").tz_convert("America/New_York")
    today_et = datetime.now(timezone.utc).astimezone().date()
    df_today = df1m[df1m.index.date == today_et]
    if df_today.empty:
        return 0
    return int(df_today["volume"].sum())

def volume_ok(symbol: str) -> bool:
    try:
        if MIN_AVG_DAILY_VOL > 0:
            d = _poly_daily_30d(symbol)
            if d.empty:
                return False
            adv30 = float(d["volume"].mean())
            if adv30 < MIN_AVG_DAILY_VOL:
                return False
        if MIN_TODAY_VOL > 0:
            if _get_today_volume(symbol) < MIN_TODAY_VOL:
                return False
        return True
    except Exception as e:
        print(f"[VOLUME CHECK ERROR] {symbol}: {e}", flush=True)
        return False

# ==============================
# Bars helpers
# ==============================
def _resample(df1m: pd.DataFrame, tf_min: int) -> pd.DataFrame:
    rule = f"{int(tf_min)}min"
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    return df1m.resample(rule, origin="start_day", label="right").agg(agg).dropna()

# ==============================
# Sentiment engine
# ==============================
def _compute_sentiment():
    """
    EMA(50/200) + price vs EMA50 on SENTIMENT_SYMBOL at SENTIMENT_TF.
    Returns: "bull" | "bear" | "neutral"
    """
    global _last_sentiment
    now = time.time()
    label, ts_cached = _last_sentiment
    if (now - ts_cached) < SENTIMENT_CACHE_SEC:
        return label

    df1m = fetch_polygon_1m(SENTIMENT_SYMBOL, lookback_minutes=max(240, SENTIMENT_TF*300))
    if df1m is None or df1m.empty:
        _last_sentiment = ("neutral", now); return "neutral"

    bars = _resample(df1m, SENTIMENT_TF)
    if bars is None or bars.empty or len(bars) < 220:
        _last_sentiment = ("neutral", now); return "neutral"

    ema50  = bars["close"].ewm(span=50, adjust=False).mean()
    ema200 = bars["close"].ewm(span=200, adjust=False).mean()
    c = bars["close"].iloc[-1]
    e50 = ema50.iloc[-1]
    e200= ema200.iloc[-1]

    if e50 > e200 and c > e50:
        label = "bull"
    elif e50 < e200 and c < e50:
        label = "bear"
    else:
        label = "neutral"

    _last_sentiment = (label, now)
    return label

def _sentiment_allows(action: str):
    """
    Decide whether a new entry 'buy' or 'sell' is allowed based on sentiment and settings.
    """
    if not SENTIMENT_ON:
        return True

    label = _compute_sentiment()
    if label == "bull":
        return action == "buy"
    if label == "bear":
        return SHORTS_ENABLED and action == "sell"

    # neutral
    if SENTIMENT_NEUTRAL_ACTION == "flat":
        return False
    if SENTIMENT_NEUTRAL_ACTION == "long":
        return action == "buy"
    if SENTIMENT_NEUTRAL_ACTION == "short":
        return SHORTS_ENABLED and action == "sell"
    if SENTIMENT_NEUTRAL_ACTION == "both":
        return action in ("buy", "sell") and (SHORTS_ENABLED or action == "buy")
    return False

# ==============================
# Ledger close on bar
# ==============================
def _record_open_trade(strategy: str, symbol: str, tf_min: int, sig: dict):
    tp = sig.get("tp_abs") if sig.get("tp_abs") is not None else sig.get("takeProfit")
    sl = sig.get("sl_abs") if sig.get("sl_abs") is not None else sig.get("stopLoss")

    t = LiveTrade(
        strategy=strategy,
        symbol=symbol,
        tf_min=int(tf_min),
        side=sig["action"],
        entry=float(sig.get("entry") or sig.get("price") or 0.0) if (sig.get("entry") is not None or sig.get("price") is not None) else float("nan"),
        tp=float(tp) if tp is not None else float("nan"),
        sl=float(sl) if sl is not None else float("nan"),
        qty=int(sig["quantity"]),
        entry_time=sig.get("barTime") or datetime.now(timezone.utc).isoformat(),
    )
    OPEN_TRADES[(symbol, int(tf_min))].append(t)

def _maybe_close_on_bar(symbol: str, tf_min: int, ts, high: float, low: float, close: float):
    key = (symbol, int(tf_min))
    if key not in OPEN_TRADES:
        return
    for t in OPEN_TRADES[key]:
        if not t.is_open:
            continue

        hit_tp = (high >= t.tp) if t.side == "buy" else (low <= t.tp)
        hit_sl = (low <= t.sl) if t.side == "buy" else (high >= t.sl)

        if hit_tp and hit_sl:
            if TP_SL_SAME_BAR == "sl":
                hit_tp, hit_sl = False, True
            else:
                hit_tp, hit_sl = True, False

        # auto-flatten near close
        flatten_now = TRADE_ONLY_RTH and _near_close(ts, AUTO_FLATTEN_MIN_BEFORE_CLOSE)

        if hit_tp or hit_sl or flatten_now:
            t.is_open = False
            t.exit_time = ts.tz_convert("UTC").isoformat() if hasattr(ts, "tzinfo") else str(ts)
            if hit_tp:
                t.exit = t.tp; t.reason = "tp"
            elif hit_sl:
                t.exit = t.sl; t.reason = "sl"
            else:
                t.exit = close; t.reason = "flatten"

            pnl = (t.exit - t.entry) * t.qty if t.side == "buy" else (t.entry - t.exit) * t.qty
            print(f"[CLOSE] {t.strategy}|{t.symbol}|{t.tf_min} {t.reason.upper()} qty={t.qty} entry={t.entry:.2f} exit={t.exit:.2f} pnl={pnl:+.2f}", flush=True)

# ==============================
# TradersPost payload / post / throttle
# ==============================
def build_payload(symbol: str, sig: dict):
    payload = {
        "ticker": symbol,
        "action": sig["action"],
        "orderType": sig.get("orderType", "market"),
        "quantity": int(sig["quantity"]),
        "meta": {}
    }
    if isinstance(sig.get("meta"), dict):
        payload["meta"].update(sig["meta"])
    if sig.get("barTime"):
        payload["meta"]["barTime"] = sig["barTime"]

    if payload["orderType"].lower() == "limit" and sig.get("price") is not None:
        payload["limitPrice"] = float(round(sig["price"], 2))

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

POST_TIMES = deque(maxlen=200)
def _throttle_ok():
    now = time.time()
    while POST_TIMES and now - POST_TIMES[0] > 60:
        POST_TIMES.popleft()
    return len(POST_TIMES) < MAX_POSTS_PER_MIN

def send_to_traderspost(payload: dict):
    if not TP_URL:
        print("[POST] Missing TP_WEBHOOK_URL", flush=True)
        return False, "missing-webhook-url"
    if not _throttle_ok():
        return False, "throttled"

    try:
        payload.setdefault("meta", {})
        payload["meta"]["environment"] = "paper"
        payload["meta"]["sentAt"] = datetime.now(timezone.utc).isoformat()
        r = requests.post(TP_URL, json=payload, timeout=12)
        ok = 200 <= r.status_code < 300
        info = f"{r.status_code} {r.text[:300]}"
        if ok:
            POST_TIMES.append(time.time())
        else:
            print(f"[POST ERROR] {info}", flush=True)
        return ok, info
    except Exception as e:
        return False, f"exception: {e}"

# ==============================
# Strategy adapter: ML pattern (long + optional short)
# ==============================
def signal_ml_pattern(symbol: str, df1m: pd.DataFrame, tf_min: int = 5,
                      conf_threshold: float = 0.8, n_estimators: int = 100,
                      r_multiple: float = 3.0, allow_shorts: bool = False):
    """
    Returns a single-bar-close signal dict or None.
    Long when model positive-prob >= threshold.
    Short (if allow_shorts) when negative-prob >= threshold.
    """
    if df1m is None or df1m.empty:
        return None

    try:
        import pandas_ta as ta
        from sklearn.ensemble import RandomForestClassifier
    except Exception as e:
        print(f"[STRAT IMPORT ERROR] {e}", flush=True)
        return None

    bars = _resample(df1m, tf_min)
    if bars is None or bars.empty or len(bars) < 80:
        return None

    bars["return"] = bars["close"].pct_change()
    bars["rsi"] = ta.rsi(bars["close"], length=14)
    bars["volatility"] = bars["close"].rolling(20).std()
    bars = bars.dropna()
    if len(bars) < 60:
        return None

    features = bars[["return","rsi","volatility"]]
    target = (bars["close"].shift(-1) > bars["close"]).astype(int)

    train_size = int(len(features) * 0.7)
    X_train, y_train = features.iloc[:train_size], target.iloc[:train_size]
    X_test  = features.iloc[train_size:]
    if X_test.empty:
        return None

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"[MODEL FIT ERROR] {symbol} {tf_min}m: {e}", flush=True)
        return None

    bars_test = bars.iloc[train_size:].copy()
    try:
        proba = model.predict_proba(X_test)
    except Exception as e:
        print(f"[MODEL PRED ERROR] {symbol} {tf_min}m: {e}", flush=True)
        return None

    pos_conf = proba[:,1]  # P(y=1)
    neg_conf = proba[:,0]  # P(y=0)

    bars_test["pos_conf"] = pos_conf
    bars_test["neg_conf"] = neg_conf

    ts = bars_test.index[-1]
    row = bars_test.iloc[-1]
    if TRADE_ONLY_RTH and (not _is_rth(ts)):
        return None

    price = float(row["close"])

    # ----- Long path -----
    if row["pos_conf"] >= conf_threshold:
        entry = price
        sl    = entry * 0.99
        tp    = entry * (1.0 + r_multiple * 0.01)
        qty   = _position_qty(entry, sl)
        if qty > 0:
            return {
                "action": "buy",
                "orderType": "market",
                "quantity": int(qty),
                "barTime": ts.tz_convert("UTC").isoformat(),
                "entry": entry,
                "tp_abs": float(tp),
                "sl_abs": float(sl),
                "meta": {"strategy": "ml_pattern", "tf": f"{tf_min}m"}
            }

    # ----- Short path (optional) -----
    if allow_shorts and row["neg_conf"] >= conf_threshold:
        entry = price
        sl    = entry * 1.01
        tp    = entry * (1.0 - r_multiple * 0.01)
        # position sizing uses entry vs stop
        qty   = _position_qty(entry, sl)
        if qty > 0:
            return {
                "action": "sell",
                "orderType": "market",
                "quantity": int(qty),
                "barTime": ts.tz_convert("UTC").isoformat(),
                "entry": entry,
                "tp_abs": float(tp),
                "sl_abs": float(sl),
                "meta": {"strategy": "ml_pattern", "tf": f"{tf_min}m"}
            }

    return None

# ==============================
# Universe & scanning
# ==============================
def get_universe():
    global _last_uni_ts, _uni_cache
    now = time.time()
    if _uni_cache and (now - _last_uni_ts) < (UNIVERSE_CACHE_MIN * 60):
        return _uni_cache

    if SCAN_TICKERS:
        syms = [x.strip().upper() for x in SCAN_TICKERS.split(",") if x.strip()]
        _uni_cache = syms; _last_uni_ts = now
        return syms

    syms = _poly_reference_symbols(limit=UNIVERSE_MAX)
    _uni_cache = syms; _last_uni_ts = now
    print(f"[UNIVERSE] {len(syms)} symbols discovered.", flush=True)
    return syms

def compute_signal(symbol: str, tf: int):
    df1m = fetch_polygon_1m(symbol, lookback_minutes=max(240, tf*80))
    if df1m is None or df1m.empty:
        return None
    allow_shorts = SHORTS_ENABLED
    return signal_ml_pattern(symbol, df1m, tf, allow_shorts=allow_shorts)

# ==============================
# Replay (optional)
# ==============================
def replay_once():
    if not REPLAY_ON_START:
        return
    df1m = fetch_polygon_1m(REPLAY_SYMBOL, lookback_minutes=REPLAY_HOURS*60)
    if df1m is None or df1m.empty:
        print("[REPLAY] no data", flush=True); return
    try:
        df1m.index = df1m.index.tz_convert("America/New_York")
    except Exception:
        df1m.index = df1m.index.tz_localize("UTC").tz_convert("America/New_York")
    bars = _resample(df1m, REPLAY_TF)
    hits = 0; last_key = None
    for i in range(len(bars)):
        df_slice = df1m.loc[:bars.index[i]]
        sig = signal_ml_pattern(REPLAY_SYMBOL, df_slice, REPLAY_TF, allow_shorts=SHORTS_ENABLED)
        if sig:
            k = (sig.get("barTime",""), sig.get("action",""))
            if k != last_key:
                hits += 1; last_key = k
    print(f"[REPLAY] {REPLAY_SYMBOL} {REPLAY_TF}m -> {hits} signal(s) in last {REPLAY_HOURS}h.", flush=True)

# ==============================
# Main
# ==============================
def main():
    print("Scanner starting…", flush=True)
    if REPLAY_ON_START:
        replay_once()

    while True:
        loop_start = time.time()
        try:
            now_et = datetime.now(timezone.utc).astimezone()
            universe = get_universe()

            # Compute sentiment once per loop (cached for SENTIMENT_CACHE_SEC)
            current_sent = _compute_sentiment() if SENTIMENT_ON else "off"
            if SENTIMENT_ON:
                print(f"[SENTIMENT] {SENTIMENT_SYMBOL} {SENTIMENT_TF}m -> {current_sent}", flush=True)

            for sym in universe:
                # absolute liquidity
                if (MIN_AVG_DAILY_VOL > 0) or (MIN_TODAY_VOL > 0):
                    if not volume_ok(sym):
                        continue

                for tf in TF_MIN_LIST:
                    # Close phase first
                    try:
                        df1m_latest = fetch_polygon_1m(sym, lookback_minutes=max(90, tf*12))
                        if df1m_latest is not None and not df1m_latest.empty:
                            bars_latest = _resample(df1m_latest, tf)
                            if bars_latest is not None and not bars_latest.empty:
                                last_row = bars_latest.iloc[-1]
                                last_ts  = bars_latest.index[-1]
                                _maybe_close_on_bar(
                                    sym, tf,
                                    last_ts,
                                    float(last_row["high"]),
                                    float(last_row["low"]),
                                    float(last_row["close"])
                                )
                    except Exception as e:
                        print(f"[CLOSE-PHASE ERROR] {sym} {tf}m: {e}", flush=True)

                    # Only enter during RTH
                    if TRADE_ONLY_RTH and not _is_rth(now_et):
                        continue

                    # Cap open positions
                    if _open_positions_count() >= MAX_OPEN_POS:
                        continue

                    sig = compute_signal(sym, tf)
                    if not sig:
                        continue

                    # Sentiment gate
                    if not _sentiment_allows(sig.get("action","")):
                        continue

                    # Fill entry for ledger math if missing
                    if "entry" not in sig:
                        try:
                            sig["entry"] = float(last_row["close"])
                        except Exception:
                            pass

                    # Re-check liquidity immediately before posting
                    if (MIN_AVG_DAILY_VOL > 0) or (MIN_TODAY_VOL > 0):
                        if not volume_ok(sym):
                            continue

                    dk = _dedupe_key(sym, tf, sig.get("action",""), sig.get("barTime",""))
                    if dk in _sent:
                        continue
                    _sent.add(dk)

                    try:
                        _record_open_trade("ml_pattern", sym, tf, sig)
                    except Exception as e:
                        print(f"[LEDGER OPEN ERROR] {sym} {tf}m: {e}", flush=True)

                    payload = build_payload(sym, sig)
                    ok, info = send_to_traderspost(payload)
                    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{stamp}] ml_pattern {sym} {tf}m -> {sig.get('action')} qty={sig.get('quantity')} | {info}", flush=True)

                    COUNTS["signals"] += 1
                    if ok: COUNTS["orders.ok"] += 1
                    else:  COUNTS["orders.err"] += 1

        except Exception as e:
            import traceback
            print("[LOOP ERROR]", e, traceback.format_exc(), flush=True)

        elapsed = time.time() - loop_start
        time.sleep(max(1, POLL_SECONDS - int(elapsed)))

if __name__ == "__main__":
    main()
