# ml_merged_app.py — bi-directional ML scanner with sentiment regime
# + REST equity/positions (TradersPost), kill switch, grouped-daily prefilter & batch scan,
# + local realized/unrealized MTM guard, + broker-equity guard, + trailing high-water guard,
# + EOD auto-flatten, throttle, max-concurrent, price/exchange/volume gates.
#
# Actions (your internal model → TradersPost webhook)
#   Long open:   "buy"         -> buy
#   Long close:  "sell"        -> sell
#   Short open:  "sell_short"  -> sell        (translated)
#   Short close: "buy_to_cover"-> buy         (translated)
#   Generic flatten:            -> exit
#
# Requirements: pandas, numpy, requests, scikit-learn, (optional) pandas_ta

import os, time, json, math, hashlib, requests
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# =============================
# ENV / CONFIG
# =============================
# --- Core connectors
TP_URL          = os.getenv("TP_WEBHOOK_URL", "").strip()
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "").strip()

# Optional TradersPost REST (equity/positions/flatten)
TP_BASE_URL     = os.getenv("TP_BASE_URL", "").rstrip("/")
TP_REST_TOKEN   = os.getenv("TP_REST_TOKEN", "")
TP_ACCOUNT_ID   = os.getenv("TP_ACCOUNT_ID", "")

# --- Diagnostics / runtime
POLL_SECONDS    = int(os.getenv("POLL_SECONDS", "10"))
DRY_RUN         = os.getenv("DRY_RUN", "0").lower() in ("1","true","yes")
PAPER_MODE      = os.getenv("PAPER_MODE", "true").lower() != "false"
SCANNER_DEBUG   = os.getenv("SCANNER_DEBUG", "0").lower() in ("1","true","yes")

RUN_ID            = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
RENDER_GIT_COMMIT = os.getenv("RENDER_GIT_COMMIT", "unknown")[:12]
RENDER_GIT_BRANCH = os.getenv("RENDER_GIT_BRANCH", os.getenv("BRANCH", "unknown"))

# --- Universe
SCANNER_SYMBOLS = os.getenv("SCANNER_SYMBOLS", "").strip()         # manual override, comma-separated
TF_MIN_LIST     = [int(x) for x in os.getenv("TF_MIN_LIST", "1,2,3,5,10").split(",") if x.strip()]
MARKET_TZ       = os.getenv("MARKET_TZ", "America/New_York")

# Polygon reference (for prefilter)
USE_GROUPED_DAILY_PREFILTER = os.getenv("USE_GROUPED_DAILY_PREFILTER","1").lower() in ("1","true","yes")
MAX_UNIVERSE_PAGES          = int(os.getenv("MAX_UNIVERSE_PAGES", "2"))   # /v3/reference/tickers, 1000/page
SCAN_BATCH_SIZE             = int(os.getenv("SCAN_BATCH_SIZE", "300"))
SCANNER_MIN_AVG_VOL         = int(os.getenv("SCANNER_MIN_AVG_VOL", "1000000"))  # grouped daily filter
# Per-symbol intraday volume gate (today)
SCANNER_MIN_TODAY_VOL       = int(os.getenv("SCANNER_MIN_TODAY_VOL", "500000"))

# Price + exchange gates
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
CONF_THR       = float(os.getenv("CONF_THR", "0.80"))
R_MULT         = float(os.getenv("R_MULT", "3.0"))
SHORTS_ENABLED = os.getenv("SHORTS_ENABLED","0").lower() in ("1","true","yes")

SENTIMENT_LOOKBACK_MIN = int(os.getenv("SENTIMENT_LOOKBACK_MIN", "60"))
SENTIMENT_NEUTRAL_BAND = float(os.getenv("SENTIMENT_NEUTRAL_BAND", "0.0015"))
SENTIMENT_SYMBOLS      = [s.strip() for s in os.getenv("SENTIMENT_SYMBOLS","SPY,QQQ").split(",") if s.strip()]
USE_SENTIMENT_REGIME   = os.getenv("USE_SENTIMENT_REGIME","1").lower() in ("1","true","yes")

# --- Engine guards / pacing
MAX_CONCURRENT_POSITIONS = int(os.getenv("MAX_CONCURRENT_POSITIONS", "100"))
MAX_ORDERS_PER_MIN       = int(os.getenv("MAX_ORDERS_PER_MIN", "60"))

# --- Local daily guard (uses START_EQUITY + realized + unrealized)
DAILY_GUARD_ENABLED  = os.getenv("DAILY_GUARD_ENABLED","1").lower() in ("1","true","yes")
START_EQUITY         = float(os.getenv("START_EQUITY","100000"))   # set this at ~9:30 each morning
DAILY_TP_PCT         = float(os.getenv("DAILY_TP_PCT","0.03"))     # +3%
DAILY_DD_PCT         = float(os.getenv("DAILY_DD_PCT","0.05"))     # -5%
DAILY_FLATTEN_ON_HIT = os.getenv("DAILY_FLATTEN_ON_HIT","1").lower() in ("1","true","yes")

# --- Broker-equity guard (uses REST equity vs session baseline)
USE_BROKER_EQUITY_GUARD   = os.getenv("USE_BROKER_EQUITY_GUARD","0").lower() in ("1","true","yes")
SESSION_START_EQUITY_ENV  = os.getenv("SESSION_START_EQUITY","").strip()  # optional fixed baseline
SESSION_START_EQUITY      = float(SESSION_START_EQUITY_ENV) if SESSION_START_EQUITY_ENV else None
SESSION_BASELINE_AT_0930  = os.getenv("SESSION_BASELINE_AT_0930","1").lower() in ("1","true","yes")

# --- Trailing high-water guard (applies to whichever equity mode you use)
TRAIL_GUARD_ENABLED   = os.getenv("TRAIL_GUARD_ENABLED","1").lower() in ("1","true","yes")
TRAIL_DD_PCT          = float(os.getenv("TRAIL_DD_PCT","0.05"))  # 5% from intraday peak
TRAIL_FLATTEN_ON_HIT  = os.getenv("TRAIL_FLATTEN_ON_HIT","1").lower() in ("1","true","yes")

# --- Kill switch
KILL_SWITCH       = os.getenv("KILL_SWITCH","OFF").lower() in ("1","true","yes","on")
KILL_SWITCH_MODE  = os.getenv("KILL_SWITCH_MODE","halt").lower()   # 'halt' or 'flatten'

# =============================
# State / ledgers
# =============================
COUNTS        = defaultdict(int)
COMBO_COUNTS  = defaultdict(int)
PERF          = {}                   # combo -> realized stats
OPEN_TRADES   = defaultdict(list)    # (symbol, tf) -> [LiveTrade]
_sent_keys    = set()                # de-dupe
_order_times  = deque()              # for throttle
LAST_PRICE    = {}                   # symbol -> last px (for MTM)

DAY_STAMP     = datetime.now().astimezone().strftime("%Y-%m-%d")
HALT_TRADING  = False                # local guard halts entries
HALTED        = False                # kill-switch / broker guard halts entries

EQUITY_BASELINE_DATE = None
EQUITY_BASELINE_SET  = False
EQUITY_HIGH_WATER    = None          # trailing peak equity (mode-dependent)

_rr_idx = 0                          # round-robin pointer for batch scanning

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
# TradersPost REST helpers
# =============================
def _tp_headers():
    return {"Authorization": f"Bearer {TP_REST_TOKEN}", "Content-Type": "application/json"}

def _tp_rest_ok():
    return bool(TP_BASE_URL and TP_REST_TOKEN and TP_ACCOUNT_ID)

def get_equity_from_tp(default_equity: float) -> float:
    if not _tp_rest_ok():
        return default_equity
    try:
        url = f"{TP_BASE_URL}/api/v2/accounts/{TP_ACCOUNT_ID}"
        r = requests.get(url, headers=_tp_headers(), timeout=12)
        if r.status_code >= 300:
            if SCANNER_DEBUG:
                print(f"[TP EQUITY] HTTP {r.status_code}: {r.text[:200]}", flush=True)
            return default_equity
        js = r.json()
        eq = js.get("equity") or js.get("cash") or js.get("netLiquidation") or default_equity
        return float(eq)
    except Exception as e:
        if SCANNER_DEBUG:
            import traceback
            print("[TP EQUITY EXC]", e, traceback.format_exc(), flush=True)
        return default_equity

def list_open_positions_tp():
    if not _tp_rest_ok():
        return []
    try:
        url = f"{TP_BASE_URL}/api/v2/accounts/{TP_ACCOUNT_ID}/positions"
        r = requests.get(url, headers=_tp_headers(), timeout=15)
        if r.status_code >= 300:
            if SCANNER_DEBUG:
                print(f"[TP POS] HTTP {r.status_code}: {r.text[:200]}", flush=True)
            return []
        js = r.json()
        return js.get("data", js if isinstance(js, list) else [])
    except Exception as e:
        if SCANNER_DEBUG:
            import traceback
            print("[TP POS EXC]", e, traceback.format_exc(), flush=True)
        return []

def close_position_tp(symbol: str, qty: float | int | None = None):
    if _tp_rest_ok():
        try:
            url = f"{TP_BASE_URL}/api/v2/accounts/{TP_ACCOUNT_ID}/orders"
            payload = {"symbol": symbol, "side": "sell", "type": "market"}
            if qty and qty > 0:
                payload["quantity"] = qty
            else:
                payload["closePosition"] = True  # if supported
            r = requests.post(url, headers=_tp_headers(), json=payload, timeout=15)
            return 200 <= r.status_code < 300
        except Exception as e:
            if SCANNER_DEBUG:
                import traceback
                print(f"[TP CLOSE EXC {symbol}]", e, traceback.format_exc(), flush=True)
    # Webhook fallback (best-effort)
    if TP_URL and not DRY_RUN:
        payload = {"ticker": symbol, "action": "exit", "orderType": "market",
                   "meta": {"environment": "paper" if PAPER_MODE else "live", "runId": RUN_ID, "intent": "flatten"}}
        try:
            r = requests.post(TP_URL, json=payload, timeout=12)
            return 200 <= r.status_code < 300
        except Exception:
            return False
    return False

def flatten_all_positions_tp():
    pos = list_open_positions_tp()
    if not pos:
        print("[FLATTEN] No broker positions found or REST not enabled.", flush=True)
        return
    print(f"[FLATTEN] Attempting REST close for {len(pos)} positions…", flush=True)
    closed = 0
    for p in pos:
        sym = p.get("symbol") or p.get("ticker") or ""
        qty = p.get("quantity") or p.get("qty") or None
        if not sym:
            continue
        ok = close_position_tp(sym, qty)
        closed += 1 if ok else 0
        time.sleep(0.2)
    print(f"[FLATTEN] REST close requested for {closed}/{len(pos)} positions.", flush=True)

# =============================
# HTTP helper for Polygon
# =============================
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

# =============================
# Polygon data fetchers
# =============================
def fetch_polygon_1m(symbol: str, lookback_minutes: int = 2400) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=lookback_minutes)
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
    js = _get(url, {"adjusted":"true","sort":"asc","limit":"50000"})
    if not js or not js.get("results"):
        return pd.DataFrame()
    df = pd.DataFrame(js["results"])
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    try:
        df.index = df.index.tz_convert(MARKET_TZ)
    except Exception:
        df.index = df.index.tz_localize("UTC").tz_convert(MARKET_TZ)
    df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
    return df[["open","high","low","close","volume"]]

def _resample(df1m: pd.DataFrame, tf_min: int) -> pd.DataFrame:
    if df1m is None or df1m.empty or not isinstance(df1m.index, pd.DatetimeIndex):
        return pd.DataFrame()
    rule = f"{int(tf_min)}min"
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    try:
        bars = df1m.resample(rule, origin="start_day", label="right").agg(agg).dropna()
    except Exception:
        bars = pd.DataFrame()
    return bars

def fetch_polygon_universe(max_pages: int = 2) -> list:
    out = []
    url = "https://api.polygon.io/v3/reference/tickers"
    page_token = None
    pages = 0
    while pages < max_pages:
        params = {"market":"stocks","active":"true","limit":1000}
        if page_token:
            params["page_token"] = page_token
        js = _get(url, params)
        if not js or not js.get("results"):
            break
        for row in js["results"]:
            sym = row.get("ticker")
            exch = (row.get("primary_exchange") or row.get("exchange") or "").upper()
            if sym and sym.isalnum() and (not ALLOWED_EXCHANGES or exch in ALLOWED_EXCHANGES):
                out.append(sym)
        nxt = js.get("next_url", None)
        if nxt and "page_token=" in nxt:
            page_token = nxt.split("page_token=")[-1]
        else:
            page_token = None
        pages += 1
        if not page_token:
            break
    if SCANNER_DEBUG:
        print(f"[UNIVERSE] reference tickers fetched={len(out)} pages={pages}", flush=True)
    return out

def filter_by_grouped_daily_volume(tickers: list, min_vol: int) -> list:
    if not tickers:
        return []
    today_et = datetime.now(timezone.utc).astimezone(ZoneInfo(MARKET_TZ)).date().strftime("%Y-%m-%d")
    url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{today_et}"
    js = _get(url, {"adjusted":"true"})
    if not js or not js.get("results"):
        return tickers
    vol_map = {row["T"]: row.get("v", 0) for row in js["results"] if "T" in row}
    filtered = [t for t in tickers if vol_map.get(t, 0) >= min_vol]
    if SCANNER_DEBUG:
        print(f"[VOL FILTER] {len(tickers)} -> {len(filtered)} by grouped-daily v≥{min_vol}", flush=True)
    return filtered

def get_universe_symbols() -> list:
    if SCANNER_SYMBOLS:
        return [s.strip().upper() for s in SCANNER_SYMBOLS.split(",") if s.strip()]
    base = fetch_polygon_universe(MAX_UNIVERSE_PAGES)
    if USE_GROUPED_DAILY_PREFILTER and base:
        base = filter_by_grouped_daily_volume(base, SCANNER_MIN_AVG_VOL)
    return base

# =============================
# Webhook I/O to TradersPost
# =============================
def _throttle_ok():
    now = time.time()
    while _order_times and now - _order_times[0] > 60:
        _order_times.popleft()
    if len(_order_times) >= MAX_ORDERS_PER_MIN:
        return False, f"throttled: {len(_order_times)}/{MAX_ORDERS_PER_MIN} in last 60s"
    _order_times.append(now)
    return True, ""

def _tp_translate_action(action: str) -> str:
    """
    Normalize internal actions to TradersPost-accepted actions.
    - buy, sell are passthrough
    - sell_short -> sell
    - buy_to_cover -> buy
    - exit/flatten/close -> exit
    """
    a = (action or "").lower()
    if a in ("sell_short", "short", "sell_to_open"):
        return "sell"
    if a in ("buy_to_cover", "cover", "buy_to_close"):
        return "buy"
    if a in ("close", "flatten", "exit"):
        return "exit"
    return a

def build_payload(symbol: str, sig: dict):
    raw_action = sig.get("action")
    action     = _tp_translate_action(raw_action)
    order_type = sig.get("orderType","market")
    qty        = int(sig.get("quantity", 0))
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
        payload["meta"]["environment"] = "paper" if PAPER_MODE else "live"
        payload["meta"]["sentAt"] = datetime.now(timezone.utc).isoformat()
        r = requests.post(TP_URL, json=payload, timeout=12)
        info = f"{r.status_code} {r.text[:300]}"
        return (200 <= r.status_code < 300), info
    except Exception as e:
        return False, f"exception: {e}"

# =============================
# Performance tracking (local)
# =============================
def _combo_key(strategy: str, symbol: str, tf_min: int) -> str:
    return f"{strategy}|{symbol}|{int(tf_min)}"

def _perf_init(combo: str):
    if combo not in PERF:
        PERF[combo] = {"trades":0,"wins":0,"losses":0,"gross_profit":0.0,"gross_loss":0.0,"net_pnl":0.0,"max_dd":0.0,"equity_curve":[0.0]}

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
    # Normalize side to "buy" or "sell" (shorts treated as "sell")
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
        df = fetch_polygon_1m(s, lookback_minutes=look_min*2)
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
# ML adapter with sentiment regime
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
        # neutral -> allow both

    if want_long:
        qty = _position_qty(price, long_sl)
        if qty > 0:
            return {"action":"buy","orderType":"market","price":None,
                    "takeProfit": long_tp,"stopLoss": long_sl,
                    "barTime": ts.tz_convert("UTC").isoformat(),"entry": price,
                    "quantity": int(qty),
                    "meta":{"note":"ml_pattern_long","proba_up": proba_up,"sentiment": sentiment}}
    if want_short:
        qty = _position_qty(price, short_sl)
        if qty > 0:
            # keep your internal intent as "sell_short" (translator will map to "sell")
            return {"action":"sell_short","orderType":"market","price":None,
                    "takeProfit": short_tp,"stopLoss": short_sl,
                    "barTime": ts.tz_convert("UTC").isoformat(),"entry": price,
                    "quantity": int(qty),
                    "meta":{"note":"ml_pattern_short","proba_up": proba_up,"sentiment": sentiment}}
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

def _opposite(action: str) -> str:
    a = (action or "").lower()
    if a == "buy":
        return "sell"
    if a == "sell":
        return "buy"
    return "exit"

def flatten_all_open_positions_webhook():
    posted = 0
    for (sym, tf), trades in list(OPEN_TRADES.items()):
        for t in trades:
            if not t.is_open or not t.qty or not t.symbol:
                continue
            payload = {
                "ticker": t.symbol,
                "action": "exit",  # generic close on TP
                "orderType": "market",
                "meta": {"note": "auto-flatten", "combo": t.combo, "triggeredAt": datetime.now(timezone.utc).isoformat()}
            }
            ok, info = send_to_traderspost(payload)
            print(f"[FLATTEN-LOCAL] {t.combo} -> ok={ok} info={info}", flush=True)
            posted += 1
            t.is_open = False; t.exit_time = datetime.now(timezone.utc).isoformat(); t.exit = t.entry; t.reason = "flatten"
    print(f"[FLATTEN-LOCAL] Requests posted: {posted}", flush=True)

def ensure_session_baseline():
    """
    Establish broker-equity session baseline once per day if using broker guard.
    If SESSION_START_EQUITY env provided, prefer that; else pull from TP.
    If SESSION_BASELINE_AT_0930=1, waits until >= 09:30 ET.
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
        SESSION_START_EQUITY = get_equity_from_tp(EQUITY_USD)
    EQUITY_HIGH_WATER = SESSION_START_EQUITY
    EQUITY_BASELINE_SET = True
    print(f"[SESSION] Baseline set: session_start_equity={SESSION_START_EQUITY:.2f}", flush=True)

def _active_equity_and_limits():
    """
    Returns (equity_now, up_limit, down_limit, mode_str) for guard evaluation.
    """
    if USE_BROKER_EQUITY_GUARD:
        ensure_session_baseline()
        base = SESSION_START_EQUITY if SESSION_START_EQUITY is not None else EQUITY_USD
        eq_now = get_equity_from_tp(base)
        up_lim = base * (1.0 + DAILY_TP_PCT)
        dn_lim = base * (1.0 - DAILY_DD_PCT)
        return eq_now, up_lim, dn_lim, "broker"
    else:
        eq_now = _current_equity_local()
        up_lim = START_EQUITY * (1.0 + DAILY_TP_PCT)
        dn_lim = START_EQUITY * (1.0 - DAILY_DD_PCT)
        return eq_now, up_lim, dn_lim, "local"

def check_kill_switch() -> bool:
    global HALTED
    if KILL_SWITCH and not HALTED:
        HALTED = True
        print(f"[GUARD] Kill switch ON (mode={KILL_SWITCH_MODE}). Trading halted.", flush=True)
        if KILL_SWITCH_MODE == "flatten":
            flatten_all_open_positions_webhook()
            if _tp_rest_ok():
                flatten_all_positions_tp()
        return True
    return HALTED

def check_daily_guards():
    """
    Evaluates profit target / drawdown and trailing DD on the selected equity mode.
    Sets HALT_TRADING/HALTED and optionally flattens.
    """
    global HALT_TRADING, HALTED, EQUITY_HIGH_WATER
    eq_now, up_lim, dn_lim, mode = _active_equity_and_limits()

    # Update high-water
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
                flatten_all_open_positions_webhook()
                if _tp_rest_ok():
                    flatten_all_positions_tp()
        elif eq_now <= dn_lim and not HALTED:
            HALT_TRADING = True; HALTED = True
            print(f"[GUARD:{mode}] ⛔ Drawdown limit hit. Halting entries.", flush=True)
            if DAILY_FLATTEN_ON_HIT:
                flatten_all_open_positions_webhook()
                if _tp_rest_ok():
                    flatten_all_positions_tp()

    if TRAIL_GUARD_ENABLED and EQUITY_HIGH_WATER and not HALTED:
        trail_floor = EQUITY_HIGH_WATER * (1.0 - TRAIL_DD_PCT)
        if eq_now <= trail_floor:
            HALT_TRADING = True; HALTED = True
            print(f"[GUARD:{mode}] 🛑 Trailing DD hit ({TRAIL_DD_PCT:.0%} from peak).", flush=True)
            if TRAIL_FLATTEN_ON_HIT:
                flatten_all_open_positions_webhook()
                if _tp_rest_ok():
                    flatten_all_positions_tp()

def reset_daily_state_if_new_day():
    global DAY_STAMP, HALT_TRADING, HALTED, EQUITY_HIGH_WATER
    today = datetime.now().astimezone().strftime("%Y-%m-%d")
    if today != DAY_STAMP:
        DAY_STAMP = today
        HALT_TRADING = False
        HALTED = False
        EQUITY_HIGH_WATER = None
        print(f"[NEW DAY] State reset. START_EQUITY={START_EQUITY:.2f} DAY={DAY_STAMP}", flush=True)

# =============================
# Routing & signal handling
# =============================
def compute_signal(strategy_name, symbol, tf_minutes, df1m=None, sentiment="neutral"):
    if df1m is None or getattr(df1m, "empty", True):
        df1m = fetch_polygon_1m(symbol, lookback_minutes=max(240, tf_minutes*240))
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

    # price & volume gates (+ MTM cache)
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
    print(f"[ORDER] {combo_key} -> action={sig.get('action')} qty={sig.get('quantity')} ok={ok} info={info}", flush=True)

# =============================
# Universe scan loop (batching)
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
# Main
# =============================
def main():
    print(f"[BOOT] RUN_ID={RUN_ID} BRANCH={RENDER_GIT_BRANCH} COMMIT={RENDER_GIT_COMMIT} START_CMD=python app_scanner_merged.py", flush=True)
    print(f"[BOOT] PAPER_MODE={'paper' if PAPER_MODE else 'live'} POLL_SECONDS={POLL_SECONDS} TFs={TF_MIN_LIST}", flush=True)
    print(f"[BOOT] CONF_THR={CONF_THR} R_MULT={R_MULT} SHORTS_ENABLED={int(SHORTS_ENABLED)} USE_SENTIMENT_REGIME={int(USE_SENTIMENT_REGIME)}", flush=True)
    print(f"[BOOT] DAILY_GUARD_ENABLED={int(DAILY_GUARD_ENABLED)} UP={DAILY_TP_PCT:.0%} DOWN={DAILY_DD_PCT:.0%} FLATTEN={int(DAILY_FLATTEN_ON_HIT)}", flush=True)
    print(f"[BOOT] BROKER_GUARD={int(USE_BROKER_EQUITY_GUARD)} BASELINE_AT_0930={int(SESSION_BASELINE_AT_0930)} TRAIL_DD={TRAIL_DD_PCT:.0%}", flush=True)

    if not POLYGON_API_KEY:
        print("[FATAL] POLYGON_API_KEY missing.", flush=True); return
    if not TP_URL and not DRY_RUN:
        print("[FATAL] TP_WEBHOOK_URL missing (or set DRY_RUN=1).", flush=True); return

    # Universe
    symbols = get_universe_symbols()
    print(f"[UNIVERSE] symbols={len(symbols)} TFs={TF_MIN_LIST} batch={SCAN_BATCH_SIZE} "
          f"vol_gate(today)={SCANNER_MIN_TODAY_VOL} MIN_PRICE={MIN_PRICE} EXCH={sorted(ALLOWED_EXCHANGES)}", flush=True)

    # Initial baseline (if broker guard)
    if USE_BROKER_EQUITY_GUARD:
        ensure_session_baseline()

    while True:
        loop_start = time.time()
        try:
            reset_daily_state_if_new_day()

            # --- Close phase: update prices, check TP/SL on open trades
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
                    if bars is not None and not bars.empty:
                        row = bars.iloc[-1]
                        ts  = bars.index[-1]
                        LAST_PRICE[sym] = float(row["close"])
                        _maybe_close_on_bar(sym, tf, ts, float(row["high"]), float(row["low"]), float(row["close"]))
                except Exception as e:
                    print(f"[CLOSE-PHASE ERROR] {sym} {tf}m: {e}", flush=True)

            # --- Guards (kill switch, daily, trailing)
            if check_kill_switch():
                pass
            else:
                check_daily_guards()

            allow_entries = not (HALT_TRADING or HALTED)
            if not allow_entries:
                time.sleep(POLL_SECONDS); continue

            # --- Scan & signal phase
            sentiment = compute_sentiment() if USE_SENTIMENT_REGIME else "neutral"
            print(f"[SENTIMENT] {sentiment}", flush=True)

            batch = _batched_symbols(symbols) if USE_GROUPED_DAILY_PREFILTER else symbols
            for sym in batch:
                df1m = fetch_polygon_1m(sym, lookback_minutes=max(240, max(TF_MIN_LIST)*240))
                if df1m is None or df1m.empty:
                    continue
                try:
                    df1m.index = df1m.index.tz_convert(MARKET_TZ)
                except Exception:
                    df1m.index = df1m.index.tz_localize("UTC").tz_convert(MARKET_TZ)

                for tf in TF_MIN_LIST:
                    sig = compute_signal("ml_pattern", sym, tf, df1m=df1m, sentiment=sentiment)
                    if not sig:
                        continue

                    # De-dupe per (strategy|symbol|tf|action|barTime)
                    dk = hashlib.sha256(f"ml_pattern|{sym}|{tf}|{sig['action']}|{sig.get('barTime','')}".encode()).hexdigest()
                    if dk in _sent_keys:
                        continue
                    _sent_keys.add(dk)

                    # Concurrent cap (just-in-time)
                    open_positions = sum(1 for lst in OPEN_TRADES.values() for t in lst if t.is_open)
                    if open_positions >= MAX_CONCURRENT_POSITIONS:
                        print(f"[LIMIT] Max concurrent reached: {open_positions}/{MAX_CONCURRENT_POSITIONS}", flush=True)
                        break

                    handle_signal("ml_pattern", sym, tf, sig)

            # --- EOD auto-flatten window (16:00–16:10 ET)
            now_et = _now_et()
            if now_et.hour == 16 and now_et.minute < 10:
                print("[EOD] Auto-flatten window.", flush=True)
                flatten_all_open_positions_webhook()
                if _tp_rest_ok():
                    flatten_all_positions_tp()

        except Exception as e:
            import traceback
            print("[LOOP ERROR]", e, traceback.format_exc(), flush=True)

        elapsed = time.time() - loop_start
        time.sleep(max(1, POLL_SECONDS - int(elapsed)))

if __name__ == "__main__":
    main()
