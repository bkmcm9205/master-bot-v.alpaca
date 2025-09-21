# Original_ML.py — dynamic market scanner -> TradeStation (direct API)
# Requires: pandas, numpy, requests, pandas_ta, scikit-learn

import os, time, json, math, requests, hashlib, logging
import pandas as pd, numpy as np
from datetime import datetime, timezone, timedelta, time as dtime
from collections import defaultdict
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any, List, Tuple

# ==============================
# LOGGING
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ==============================
# ENV / CONFIG (your variable names preserved)
# ==============================
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY","")

# Cadence & diagnostics
POLL_SECONDS = int(os.getenv("POLL_SECONDS","10"))
DRY_RUN = os.getenv("DRY_RUN","0").lower() in ("1","true","yes")
SCANNER_DEBUG = os.getenv("SCANNER_DEBUG","0").lower() in ("1","true","yes")

# Timeframes list
TF_MIN_LIST = [int(x) for x in os.getenv("TF_MIN_LIST","5").split(",")]

# Universe paging/size
MAX_UNIVERSE_PAGES = int(os.getenv("MAX_UNIVERSE_PAGES", os.getenv("SCANNER_MAX_PAGES","3")))
SCAN_BATCH_SIZE    = int(os.getenv("SCAN_BATCH_SIZE","150"))

# Liquidity + price filters (existing names + new MIN_LAST_PRICE)
SCANNER_MIN_AVG_VOL   = int(os.getenv("SCANNER_MIN_AVG_VOL", os.getenv("MIN_AVG_DAILY_VOL","1000000")))
SCANNER_MIN_TODAY_VOL = int(os.getenv("SCANNER_MIN_TODAY_VOL", os.getenv("MIN_TODAY_VOL","0")))
MIN_LAST_PRICE        = float(os.getenv("MIN_LAST_PRICE","3.0"))  # NEW: price floor

# Exchanges: NYSE + NASDAQ by default
ALLOW_EXCHANGES = set([s.strip().upper() for s in os.getenv("ALLOW_EXCHANGES","XNYS,XNAS").split(",") if s.strip()])

# Position sizing (your names)
EQUITY_USD  = float(os.getenv("EQUITY_USD","100000"))
RISK_PCT    = float(os.getenv("RISK_PCT","0.01"))
MAX_POS_PCT = float(os.getenv("MAX_POS_PCT","0.10"))
MIN_QTY     = int(os.getenv("MIN_QTY","1"))
ROUND_LOT   = int(os.getenv("ROUND_LOT","1"))

# Sessions
SCANNER_MARKET_HOURS_ONLY = os.getenv("SCANNER_MARKET_HOURS_ONLY","1").lower() in ("1","true","yes")
ALLOW_PREMARKET  = os.getenv("ALLOW_PREMARKET","0").lower() in ("1","true","yes")
ALLOW_AFTERHOURS = os.getenv("ALLOW_AFTERHOURS","0").lower() in ("1","true","yes")

# Sentiment: SHORTS_ENABLED controls neutrality/bull mode (your behavior)
SHORTS_ENABLED = os.getenv("SHORTS_ENABLED","1") in ("1","true","yes")

# Daily guard (your names)
DAILY_GUARD_ENABLED = os.getenv("DAILY_GUARD_ENABLED","0") in ("1","true","yes")
DAILY_DD_PCT = float(os.getenv("DAILY_DD_PCT","0.05"))   # 0.05 = 5% down
DAILY_TP_PCT = float(os.getenv("DAILY_TP_PCT","0.25"))   # 0.25 = 25% up

# Kill switch (your name)
KILL_SWITCH = os.getenv("KILL_SWITCH","OFF").strip().upper() == "ON"

# EOD flatten (optional, can add envs if you want)
EOD_FLATTEN = os.getenv("EOD_FLATTEN","true").lower() in ("1","true","yes")
EOD_FLATTEN_TIME = os.getenv("EOD_FLATTEN_TIME","15:59:30")

# Rate limiting — use your MAX_ORDERS_PER_MIN
MAX_ORDERS_PER_MIN = max(1, int(os.getenv("MAX_ORDERS_PER_MIN","60")))

RUN_ID = datetime.now().astimezone().strftime("%Y-%m-%d")
COUNTS = defaultdict(int); COMBO_COUNTS = defaultdict(int)
_sent = set(); _round_robin = 0
NY = ZoneInfo("America/New_York")

# ==============================
# TRADESTATION CREDS + token bootstrap from env
# ==============================
TS_AUTH_BASE = "https://signin.tradestation.com"
TS_API_BASE  = "https://api.tradestation.com/v3"
TS_ACCOUNT_ID = os.getenv("TS_ACCOUNT_ID","")
TS_TOKEN_PATH = os.getenv("TS_TOKEN_PATH","/app/ts_tokens.json")

def _maybe_write_ts_token_from_env():
    """Write TS_TOKEN_JSON env to TS_TOKEN_PATH (create folder if needed)."""
    j = os.getenv("TS_TOKEN_JSON","").strip()
    if not j:
        return
    folder = os.path.dirname(TS_TOKEN_PATH)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    with open(TS_TOKEN_PATH,"w") as f:
        f.write(j)
    logging.info(f"[BOOT] wrote TradeStation token file -> {TS_TOKEN_PATH}")

# ==============================
# Sessions, sizing, sentiment, dedupe, EOD
# ==============================
def _market_session_now():
    now_et = datetime.now(NY).time()
    def within(a,b): return a <= now_et < b
    in_rth = within(dtime(9,30), dtime(16,0))
    in_pre = ALLOW_PREMARKET and within(dtime(4,0), dtime(9,30))
    in_ah  = ALLOW_AFTERHOURS and within(dtime(16,0), dtime(20,0))
    return in_rth or in_pre or in_ah

def _position_qty(entry_price: float, stop_price: float) -> int:
    if entry_price is None or stop_price is None: return 0
    rps = abs(entry_price - stop_price)
    if rps <= 0: return 0
    qty_risk     = (EQUITY_USD * RISK_PCT) / rps
    qty_notional = (EQUITY_USD * MAX_POS_PCT) / max(1e-9, entry_price)
    qty = math.floor(max(min(qty_risk, qty_notional), 0) / max(1, ROUND_LOT)) * max(1, ROUND_LOT)
    return int(max(qty, MIN_QTY if qty > 0 else 0))

def _sentiment_allows(side: str) -> bool:
    # SHORTS_ENABLED=1 -> allow both; 0 -> block sells/shorts (bull-only)
    if SHORTS_ENABLED: return True
    return side.lower() in ("buy","long")

def _dedupe_key(symbol: str, tf: int, action: str, bar_time: str) -> str:
    return hashlib.sha256(f"{symbol}|{tf}|{action}|{bar_time}".encode()).hexdigest()

def _is_eod_now() -> bool:
    if not EOD_FLATTEN: return False
    try: tgt = dtime.fromisoformat(EOD_FLATTEN_TIME)
    except: tgt = dtime(15,59,30)
    return datetime.now(NY).time().replace(microsecond=0) >= tgt

# ==============================
# HTTP helper
# ==============================
def _get(url, params=None, timeout=15):
    params = params or {}
    if POLYGON_API_KEY: params["apiKey"] = POLYGON_API_KEY
    r = requests.get(url, params=params, timeout=timeout)
    if r.status_code != 200:
        if SCANNER_DEBUG: logging.warning("[HTTP %s] %s -> %s", r.status_code, url, r.text[:200])
        return None
    return r.json()

# ==============================
# Polygon data
# ==============================
def fetch_polygon_1m(symbol: str, lookback_minutes: int = 2400) -> pd.DataFrame:
    end = datetime.now(timezone.utc); start = end - timedelta(minutes=lookback_minutes)
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
    js = _get(url, {"adjusted":"true","sort":"asc","limit":"50000"})
    if not js or not js.get("results"): return pd.DataFrame()
    df = pd.DataFrame(js["results"])
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    df.index = df.index.tz_convert(NY)
    df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
    return df[["open","high","low","close","volume"]]

def _resample(df1m: pd.DataFrame, tf_min: int) -> pd.DataFrame:
    if df1m is None or df1m.empty: return pd.DataFrame()
    rule = f"{int(tf_min)}min"
    bars = df1m.resample(rule, origin="start_day", label="right").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    ).dropna()
    try: bars.index = bars.index.tz_convert(NY)
    except: bars.index = bars.index.tz_localize("UTC").tz_convert(NY)
    return bars

def fetch_polygon_universe(max_pages: int = 3) -> list[str]:
    """Active US stocks via /v3/reference/tickers; NYSE+NASDAQ only."""
    out, url, page_token, pages = [], "https://api.polygon.io/v3/reference/tickers", None, 0
    while pages < max_pages:
        params = {"market":"stocks","active":"true","limit":1000}
        if page_token: params["page_token"] = page_token
        js = _get(url, params); 
        if not js or not js.get("results"): break
        for row in js["results"]:
            sym = row.get("ticker")
            px  = (row.get("primary_exchange") or "").upper()
            # Some rows won't have last price; we enforce price again later via grouped daily and per-trade
            if sym and sym.isalpha() and px in ALLOW_EXCHANGES:
                out.append(sym)
        page_token = js.get("next_url", None)
        if page_token and "page_token=" in page_token:
            page_token = page_token.split("page_token=")[-1]
        pages += 1
        if not page_token: break
    if SCANNER_DEBUG:
        logging.info("[UNIVERSE] fetched %d tickers across %d page(s) exchanges=%s",
                     len(out), pages, sorted(list(ALLOW_EXCHANGES)))
    return out

def filter_by_daily_volume_and_price(tickers: list[str], min_avg_vol: int, min_today_vol: int, min_last_price: float) -> list[str]:
    """Fast daily filter using grouped daily; then price floor."""
    if not tickers: return []
    today = datetime.now(timezone.utc).astimezone().date().strftime("%Y-%m-%d")
    url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{today}"
    js = _get(url, {"adjusted":"true"}); 
    if not js or not js.get("results"):
        return tickers  # fallback
    # Map: symbol -> (today_volume, close)
    vol_map = {row["T"]: (row.get("v",0), row.get("c", None)) for row in js["results"] if "T" in row}
    out = []
    for t in tickers:
        v,c = vol_map.get(t, (0, None))
        if min_today_vol and v < min_today_vol: 
            continue
        if (c is not None) and (c < min_last_price):
            continue
        out.append(t)
    if SCANNER_DEBUG:
        logging.info("[VOL/PRICE FILTER] %d -> %d (min_today_vol=%s, min_last_price=%.2f)", len(tickers), len(out), min_today_vol, min_last_price)
    return out

# ==============================
# ML strategy (long entries; same logic)
# ==============================
def signal_ml_pattern(symbol: str, df1m: pd.DataFrame, tf_min: int,
                      conf_threshold: float = 0.8, n_estimators: int = 100,
                      r_multiple: float = 3.0, min_volume_mult: float = 0.0):
    try:
        import pandas_ta as ta  # noqa
        from sklearn.ensemble import RandomForestClassifier
    except Exception as e:
        if SCANNER_DEBUG: logging.warning("[ML IMPORT] %s %sm: %s", symbol, tf_min, e)
        return None

    bars = _resample(df1m, tf_min)
    if bars is None or bars.empty or len(bars) < 120: return None

    bars = bars.copy()
    bars["return"] = bars["close"].pct_change()
    try:
        import pandas_ta as ta
        bars["rsi"] = ta.rsi(bars["close"], length=14)
    except Exception:
        d = bars["close"].diff(); up = d.clip(lower=0).rolling(14).mean(); dn = -d.clip(upper=0).rolling(14).mean()
        rs = up / (dn.replace(0, np.nan)); bars["rsi"] = 100 - (100 / (1 + rs))
    bars["volatility"] = bars["close"].rolling(20).std()
    bars.dropna(inplace=True)
    if len(bars) < 100: return None

    X = bars[["return","rsi","volatility"]]
    y = (bars["close"].shift(-1) > bars["close"]).astype(int)
    tsz = int(len(X)*0.7)
    if tsz < 50: return None
    X_train, y_train = X.iloc[:tsz], y.iloc[:tsz]
    X_test = X.iloc[tsz:]; 
    if X_test.empty: return None

    try:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42); model.fit(X_train, y_train)
    except Exception as e:
        if SCANNER_DEBUG: logging.warning("[ML TRAIN] %s %sm: %s", symbol, tf_min, e)
        return None

    prob = model.predict_proba(X_test)[:,1]; preds = (prob > 0.5).astype(int)
    bars_live = bars.iloc[tsz:].copy(); bars_live["prediction"]=preds; bars_live["confidence"]=prob
    avgv = bars_live["volume"].rolling(50).mean().fillna(bars_live["volume"].mean())

    last = bars_live.iloc[-1]; ts = bars_live.index[-1]
    if (last["prediction"] == 1) and (last["confidence"] >= conf_threshold):
        if min_volume_mult > 0.0:
            i = len(bars_live)-1
            if not (last["volume"] > min_volume_mult * avgv.iloc[i]): return None
        entry = float(last["close"]); sl = entry*0.99; tp = entry*(1.0+0.01*r_multiple)
        qty = _position_qty(entry, sl); 
        if qty <= 0: return None
        return {"action":"buy","orderType":"market","quantity":int(qty),"entry":entry,
                "tp_abs":float(tp),"sl_abs":float(sl),"barTime": ts.tz_convert("UTC").isoformat(),
                "meta":{"strategy":"ml_pattern","timeframe":f"{int(tf_min)}m"}}
    return None

# ==============================
# TradeStation Broker (direct) + rate limiter
# ==============================
class RateLimiter:
    def __init__(self, per_min:int):
        self.capacity = max(1, per_min)
        self.tokens = float(self.capacity)
        self.refill_per_sec = self.capacity / 60.0
        self.last = time.time()
    def acquire(self):
        now = time.time()
        self.tokens = min(self.capacity, self.tokens + (now-self.last)*self.refill_per_sec)
        self.last = now
        if self.tokens < 1:
            sleep_for = (1 - self.tokens)/self.refill_per_sec
            time.sleep(max(0.05, sleep_for))
            self.tokens = 0
        else:
            self.tokens -= 1

class TradeStationBroker:
    def __init__(self):
        self.s = requests.Session()
        self.limiter = RateLimiter(MAX_ORDERS_PER_MIN)
        self.client_id = os.environ.get("TS_CLIENT_ID","")
        self.client_secret = os.getenv("TS_CLIENT_SECRET","")
        self.redirect = os.environ.get("TS_REDIRECT_URI","")
        self.account_id = TS_ACCOUNT_ID
        self.paper = os.getenv("TS_PAPER","true").lower() == "true"
        self.token_store = os.getenv("TS_TOKEN_STORE","file")
        self.token_path  = TS_TOKEN_PATH

    def _load_token(self) -> Dict[str,Any]:
        with open(self.token_path,"r") as f: return json.load(f)

    def _save_token(self, data: Dict[str,Any]):
        with open(self.token_path,"w") as f: json.dump(data, f)

    def _ensure_token(self) -> Dict[str,Any]:
        tok = self._load_token(); now = time.time()
        if now >= tok.get("expires_at",0) - 30:
            payload = {"grant_type":"refresh_token","client_id":self.client_id,"refresh_token":tok["refresh_token"]}
            if self.client_secret: payload["client_secret"] = self.client_secret
            r = requests.post(f"{TS_AUTH_BASE}/oauth/token", data=payload,
                              headers={"content-type":"application/x-www-form-urlencoded"}, timeout=15)
            r.raise_for_status()
            j = r.json()
            new_tok = {"access_token": j["access_token"],
                       "refresh_token": j.get("refresh_token", tok["refresh_token"]),
                       "token_type": j.get("token_type","Bearer"),
                       "scope": j.get("scope",""),
                       "expires_at": now + j.get("expires_in", 900)}
            self._save_token(new_tok); return new_tok
        return tok

    def _headers(self) -> Dict[str,str]:
        tok = self._ensure_token(); return {"Authorization": f"Bearer {tok['access_token']}"}
    def _acct_qs(self) -> str:
        return f"accountID={self.account_id}&env={'Sim' if self.paper else 'Live'}"

    def _post(self, path: str, body: Dict[str,Any]) -> Dict[str,Any]:
        self.limiter.acquire()
        url = f"{TS_API_BASE}{path}"
        r = self.s.post(url, json=body, headers=self._headers(), timeout=15)
        if r.status_code == 429:
            logging.warning("TS 429: backing off and retrying...")
            time.sleep(1.0); self.limiter.acquire()
            r = self.s.post(url, json=body, headers=self._headers(), timeout=15)
        r.raise_for_status(); return r.json()

    def place_market(self, symbol: str, side: str, qty: int, client_order_id: str) -> Dict[str,Any]:
        side_map = {"buy":"Buy","long":"Buy","sell":"Sell","short":"Sell"}
        body = {"AccountID": self.account_id, "Symbol": symbol, "OrderType":"Market",
                "Quantity": qty, "TradeAction": side_map[side.lower()],
                "TimeInForce":"DAY", "Route":"Intelligent", "ClientOrderID": client_order_id}
        return self._post(f"/orders?{self._acct_qs()}", body)

    def place_limit(self, symbol:str, side:str, qty:int, limit_price:float, client_order_id:str) -> Dict[str,Any]:
        side_map = {"buy":"Buy","long":"Buy","sell":"Sell","short":"Sell"}
        body = {"AccountID": self.account_id, "Symbol": symbol, "OrderType":"Limit",
                "LimitPrice": float(limit_price), "Quantity": qty,
                "TradeAction": side_map[side.lower()], "TimeInForce":"DAY", "ClientOrderID": client_order_id}
        return self._post(f"/orders?{self._acct_qs()}", body)

    def place_stop(self, symbol:str, side:str, qty:int, stop_price:float, client_order_id:str) -> Dict[str,Any]:
        side_map = {"buy":"Buy","long":"Buy","sell":"Sell","short":"Sell"}
        body = {"AccountID": self.account_id, "Symbol": symbol, "OrderType":"StopMarket",
                "StopPrice": float(stop_price), "Quantity": qty,
                "TradeAction": side_map[side.lower()], "TimeInForce":"DAY", "ClientOrderID": client_order_id}
        return self._post(f"/orders?{self._acct_qs()}", body)

    def get_equity(self) -> float:
        self.limiter.acquire()
        r = self.s.get(f"{TS_API_BASE}/balances?{self._acct_qs()}", headers=self._headers(), timeout=15)
        r.raise_for_status()
        bal = r.json().get("Balances",[{}])[0]
        return float(bal.get("NetLiquidatingValue") or bal.get("AccountValue") or 0.0)

    def list_positions(self) -> List[Dict[str,Any]]:
        self.limiter.acquire()
        r = self.s.get(f"{TS_API_BASE}/positions?{self._acct_qs()}", headers=self._headers(), timeout=15)
        r.raise_for_status(); return r.json().get("Positions", [])

    def close_all(self):
        pos = self.list_positions()
        for p in pos:
            sym = p.get("Symbol"); qty = abs(int(p.get("Quantity",0)))
            if qty <= 0: continue
            side = "sell" if p.get("LongShort") == "Long" else "buy"
            self.place_market(sym, side, qty, client_order_id=f"flatten-{sym}-{int(time.time())}")

BROKER = None
def broker() -> TradeStationBroker:
    global BROKER
    if BROKER is None: BROKER = TradeStationBroker()
    return BROKER

# ==============================
# Guards: daily equity + kill switch
# ==============================
_guard_blocked_for_day = False
_day_open_equity: Optional[float] = None

def ensure_day_open_equity():
    global _day_open_equity
    if _day_open_equity is None:
        try:
            eq = broker().get_equity()
            if eq > 0: _day_open_equity = float(eq); logging.info("[GUARD] opened_equity=%.2f", eq)
        except Exception as e:
            logging.warning("[GUARD] cannot fetch equity yet: %s", e)

def _daily_guard_check_and_act():
    """
    If DAILY_GUARD_ENABLED and PnL >= take OR <= drawdown:
      - flatten all positions
      - set _guard_blocked_for_day=True to block new entries until day changes
    """
    global _guard_blocked_for_day
    if not DAILY_GUARD_ENABLED or _day_open_equity is None or _guard_blocked_for_day:
        return
    try:
        eq_now = broker().get_equity()
    except Exception as e:
        logging.warning("[GUARD] equity fetch error: %s", e); return
    if _day_open_equity <= 0: return
    pnl_pct = (eq_now - _day_open_equity)/_day_open_equity
    if pnl_pct >= DAILY_TP_PCT or pnl_pct <= -abs(DAILY_DD_PCT):
        logging.warning("[GUARD] TRIPPED (PnL=%.2f%%). Flattening & blocking new entries.", pnl_pct*100.0)
        try:
            broker().close_all()
        finally:
            _guard_blocked_for_day = True
            # Mirror your kill switch convention too
            os.environ["KILL_SWITCH"] = "ON"

def entries_allowed_now(side: str) -> bool:
    if os.getenv("KILL_SWITCH","OFF").upper()=="ON": 
        return False
    if _guard_blocked_for_day: 
        return False
    if not _sentiment_allows(side): 
        return False
    return True

# ==============================
# Bracket helper
# ==============================
def place_bracket(symbol: str, side: str, qty: int, stop_price: float, take_price: float, client_tag: str):
    if DRY_RUN:
        logging.info("(DRY) ENTRY %s %s qty=%s sl=%.2f tp=%.2f", symbol, side, qty, stop_price, take_price)
        return {"id": f"dry-{symbol}-{int(time.time())}"}
    entry = broker().place_market(symbol, side, qty, client_order_id=f"{client_tag}-entry")
    exit_side = "sell" if side.lower() in ("buy","long") else "buy"
    sl = broker().place_stop(symbol, exit_side, qty, stop_price, client_order_id=f"{client_tag}-sl")
    tp = broker().place_limit(symbol, exit_side, qty, take_price, client_order_id=f"{client_tag}-tp")
    return {"entry": entry, "sl": sl, "tp": tp}

# ==============================
# Scanner loop
# ==============================
def build_universe():
    base = fetch_polygon_universe(MAX_UNIVERSE_PAGES)
    if not base:
        logging.error("[UNIVERSE] Empty; check POLYGON_API_KEY / permissions.")
        return []
    # daily price + volume filter (today)
    filtered = filter_by_daily_volume_and_price(base, SCANNER_MIN_AVG_VOL, SCANNER_MIN_TODAY_VOL, MIN_LAST_PRICE)
    return filtered if filtered else base

def scan_once(universe: list[str]):
    global _round_robin

    # EOD flatten
    if _is_eod_now():
        logging.warning("[EOD] Flatten trigger")
        try: broker().close_all()
        except Exception as e: logging.error("[EOD] flatten error: %s", e)

    # Market hours check
    if SCANNER_MARKET_HOURS_ONLY and not _market_session_now():
        if SCANNER_DEBUG: logging.info("[SCAN] Skipping — market session closed.")
        return
    if not universe: return

    # Equity baseline & guard
    ensure_day_open_equity()
    _daily_guard_check_and_act()

    # Rotate batch
    N = len(universe)
    start = _round_robin % max(1,N); end = min(N, start + SCAN_BATCH_SIZE)
    batch = universe[start:end]; _round_robin = end if end < N else 0
    if SCANNER_DEBUG: logging.info("[SCAN] symbols %s:%s / %s (batch=%s)", start, end, N, len(batch))

    for sym in batch:
        try:
            df1m = fetch_polygon_1m(sym, lookback_minutes=2400)
            if df1m is None or df1m.empty: continue
            last_price = float(df1m["close"].iloc[-1]) if not df1m.empty else None
            if (last_price is not None) and (last_price < MIN_LAST_PRICE): 
                continue
        except Exception as e:
            if SCANNER_DEBUG: logging.warning("[FETCH ERR] %s: %s", sym, e)
            continue

        for tf in TF_MIN_LIST:
            try:
                sig = signal_ml_pattern(
                    sym, df1m, tf_min=tf,
                    conf_threshold=float(os.getenv("SCANNER_CONF_THRESHOLD","0.8")),
                    n_estimators=100,
                    r_multiple=float(os.getenv("SCANNER_R_MULTIPLE","3.0")),
                    min_volume_mult=0.0
                )
                if not sig: continue

                k = _dedupe_key(sym, tf, sig["action"], sig.get("barTime",""))
                if k in _sent: continue
                _sent.add(k)

                if not entries_allowed_now(sig["action"]): 
                    continue
                # final guard check just before sending
                _daily_guard_check_and_act()
                if _guard_blocked_for_day or os.getenv("KILL_SWITCH","OFF").upper()=="ON":
                    continue

                entry = float(sig["entry"]); sl = float(sig["sl_abs"]); tp = float(sig["tp_abs"]); qty = int(sig["quantity"])
                if qty <= 0: continue

                stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if DRY_RUN:
                    logging.info("[%s] (DRY) %s %sm -> buy qty=%s tp=%.2f sl=%.2f", stamp, sym, tf, qty, tp, sl)
                else:
                    place_bracket(sym, "buy", qty, stop_price=sl, take_price=tp, client_tag=f"mlp-{sym}-{tf}")
                    logging.info("[%s] SENT %s %sm -> buy qty=%s tp=%.2f sl=%.2f", stamp, sym, tf, qty, tp, sl)

                COUNTS["signals"] += 1
                COMBO_COUNTS[f"{sym}|{tf}::orders.ok"] += 1
            except Exception as e:
                if SCANNER_DEBUG:
                    import traceback
                    logging.error("[SCAN ERR] %s %sm: %s\n%s", sym, tf, e, traceback.format_exc())
                COMBO_COUNTS[f"{sym}|{tf}::orders.err"] += 1
                continue

def main():
    logging.info("Scanner starting…")
    _maybe_write_ts_token_from_env()

    if not POLYGON_API_KEY:
        logging.error("[FATAL] POLYGON_API_KEY missing."); return

    # Soft-check TS token unless DRY_RUN
    if not DRY_RUN and TS_ACCOUNT_ID:
        try: 
            _ = broker().get_equity()
        except FileNotFoundError:
            logging.error("[FATAL] TS_TOKEN_PATH not found. Ensure TS_TOKEN_JSON is set so we can write it."); return
        except Exception as e:
            logging.warning("[WARN] Could not fetch TS equity yet: %s", e)

    universe = build_universe()
    logging.info("[READY] Universe size: %d  TFs: %s  Batch: %d  MIN_LAST_PRICE: %.2f",
                 len(universe), TF_MIN_LIST, SCAN_BATCH_SIZE, MIN_LAST_PRICE)

    while True:
        loop_start = time.time()
        try:
            scan_once(universe)
        except Exception as e:
            import traceback
            logging.error("[LOOP ERROR] %s\n%s", e, traceback.format_exc())
        elapsed = time.time() - loop_start
        time.sleep(max(1, POLL_SECONDS - int(elapsed)))

if __name__ == "__main__":
    main()
