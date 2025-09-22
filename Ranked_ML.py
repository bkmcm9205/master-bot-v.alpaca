# Ranked_ML.py — TradersPost ML scanner with confidence ranking + diagnostics
# - Confidence-ranked dispatch (highest confidence first)
# - Unlimited sends by default (MAX_ENTRIES_PER_CYCLE=0)
# - Sentiment gate preserved
# - Detailed per-loop diagnostics (candidate summary + gate counters)
# - BYPASS_SESSION env to allow after-hours/premarket testing without posting
# - Dedupe at send-time so candidates aren’t “burned” early

import os, time, json, math, hashlib, requests
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytz

# ==============================
# BOOT TAG (so you know this file is running)
# ==============================
APP_TAG = "Ranked_ML v1.2 (diagnostics + bypass_session)"
print(f"[BOOT] {APP_TAG}", flush=True)

# ==============================
# ENV / CONFIG
# ==============================
TP_URL              = os.getenv("TP_WEBHOOK_URL", "")
POLYGON_API_KEY     = os.getenv("POLYGON_API_KEY", "")
POLL_SECONDS        = int(os.getenv("POLL_SECONDS", "10"))
RUN_ID              = datetime.now().astimezone().strftime("%Y-%m-%d")

# Ranking & control
SCANNER_CONF_THRESHOLD = float(os.getenv("SCANNER_CONF_THRESHOLD", "0.80"))
SCANNER_R_MULTIPLE     = float(os.getenv("SCANNER_R_MULTIPLE", "3.0"))
MAX_ENTRIES_PER_CYCLE  = int(os.getenv("MAX_ENTRIES_PER_CYCLE", "0"))    # 0/neg => unlimited
ALLOW_ENTRIES          = os.getenv("ALLOW_ENTRIES","1").lower() in ("1","true","yes")

# Universe / scan config
SCANNER_SYMBOLS       = os.getenv("SCANNER_SYMBOLS", "").strip()
SCANNER_MAX_PAGES     = int(os.getenv("SCANNER_MAX_PAGES", "1"))
SCANNER_MIN_TODAY_VOL = int(os.getenv("SCANNER_MIN_TODAY_VOL", "100000"))
TF_MIN_LIST           = [int(x) for x in os.getenv("TF_MIN_LIST", "1,2,3,5,10").split(",") if x.strip()]

# Risk / sizing
EQUITY_USD  = float(os.getenv("EQUITY_USD",  "100000"))
RISK_PCT    = float(os.getenv("RISK_PCT",    "0.01"))
MAX_POS_PCT = float(os.getenv("MAX_POS_PCT", "0.10"))
MIN_QTY     = int(os.getenv("MIN_QTY", "1"))
ROUND_LOT   = int(os.getenv("ROUND_LOT","1"))

# Engine guards
MAX_CONCURRENT_POSITIONS  = int(os.getenv("MAX_CONCURRENT_POSITIONS", "100"))
MAX_ORDERS_PER_MIN        = int(os.getenv("MAX_ORDERS_PER_MIN", "60"))
MARKET_TZ                 = "America/New_York"
ALLOW_PREMARKET           = os.getenv("ALLOW_PREMARKET", "0").lower() in ("1","true","yes")
ALLOW_AFTERHOURS          = os.getenv("ALLOW_AFTERHOURS", "0").lower() in ("1","true","yes")
BYPASS_SESSION            = os.getenv("BYPASS_SESSION","0").lower() in ("1","true","yes")  # test-only switch

# Sentiment gate
SENTIMENT_ONLY_GATE   = os.getenv("SENTIMENT_ONLY_GATE","1").lower() in ("1","true","yes")

# Daily guard (realized PnL)
START_EQUITY         = float(os.getenv("START_EQUITY", "100000"))
DAILY_TP_PCT         = float(os.getenv("DAILY_TP_PCT", "0.25"))
DAILY_DD_PCT         = float(os.getenv("DAILY_DD_PCT", "0.05"))
DAILY_FLATTEN_ON_HIT = os.getenv("DAILY_FLATTEN_ON_HIT","1").lower() in ("1","true","yes")
DAILY_GUARD_ENABLED  = os.getenv("DAILY_GUARD_ENABLED","0").lower() in ("1","true","yes")

# Diagnostics / replay
DRY_RUN            = os.getenv("DRY_RUN","0") == "1"
REPLAY_ON_START    = os.getenv("REPLAY_ON_START","0") == "1"
REPLAY_SYMBOL      = os.getenv("REPLAY_SYMBOL","SPY")
REPLAY_TF          = int(os.getenv("REPLAY_TF","5"))
REPLAY_HOURS       = int(os.getenv("REPLAY_HOURS","24"))
REPLAY_SEND_ORDERS = os.getenv("REPLAY_SEND_ORDERS","0").lower() in ("1","true","yes")

# Ledgers
COUNTS        = defaultdict(int)
COMBO_COUNTS  = defaultdict(int)
PERF          = {}
OPEN_TRADES   = defaultdict(list)
_sent_keys    = set()
_order_times  = deque()

# ==============================
# Models & utils
# ==============================
class LiveTrade:
    def __init__(self, combo, symbol, tf_min, side, entry, tp, sl, qty, entry_time):
        self.combo = combo; self.symbol = symbol; self.tf_min = int(tf_min)
        self.side = side; self.entry = float(entry) if entry is not None else float("nan")
        self.tp = float(tp) if tp is not None else float("nan")
        self.sl = float(sl) if sl is not None else float("nan")
        self.qty = int(qty); self.entry_time = entry_time
        self.is_open = True; self.exit = None; self.exit_time=None; self.reason=None

def _now_et(): return datetime.now(timezone.utc).astimezone(ZoneInfo(MARKET_TZ))
def _is_rth(ts): h,m=ts.hour,ts.minute; return ((h>9) or (h==9 and m>=30)) and (h<16)
def _in_session(ts):
    if BYPASS_SESSION:
        return True
    if _is_rth(ts): return True
    if ALLOW_PREMARKET and (4 <= ts.hour < 9 or (ts.hour == 9 and ts.minute < 30)): return True
    if ALLOW_AFTERHOURS and (16 <= ts.hour < 20): return True
    return False

def _resample(df1m: pd.DataFrame, tf_min: int) -> pd.DataFrame:
    if df1m is None or df1m.empty or not isinstance(df1m.index, pd.DatetimeIndex): return pd.DataFrame()
    rule=f"{int(tf_min)}min"; agg={"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    try: return df1m.resample(rule, origin="start_day", label="right").agg(agg).dropna()
    except: return pd.DataFrame()

def _dedupe_key(strategy: str, symbol: str, tf: int, action: str, bar_time: str) -> str:
    return hashlib.sha256(f"{strategy}|{symbol}|{tf}|{action}|{bar_time}".encode()).hexdigest()

def _position_qty(entry_price: float, stop_price: float) -> int:
    if entry_price is None or stop_price is None: return 0
    rps = abs(float(entry_price) - float(stop_price))
    if rps <= 0: return 0
    qty_risk     = (EQUITY_USD * RISK_PCT) / rps
    qty_notional = (EQUITY_USD * MAX_POS_PCT) / max(1e-9, float(entry_price))
    qty = math.floor(max(min(qty_risk, qty_notional), 0) / max(1, ROUND_LOT)) * max(1, ROUND_LOT)
    return int(max(qty, MIN_QTY if qty > 0 else 0))

# ==============================
# Polygon fetchers
# ==============================
def fetch_polygon_1m(symbol: str, lookback_minutes: int = 2400) -> pd.DataFrame:
    if not POLYGON_API_KEY: return pd.DataFrame()
    end = datetime.now(timezone.utc); start = end - timedelta(minutes=lookback_minutes)
    url = (f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/"
           f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
           f"?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}")
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200: return pd.DataFrame()
        rows = r.json().get("results", [])
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        df = df.set_index("ts").sort_index(); df.index = df.index.tz_convert(MARKET_TZ)
        return df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})[["open","high","low","close","volume"]]
    except: return pd.DataFrame()

def get_universe_symbols() -> list:
    if SCANNER_SYMBOLS:
        return [s.strip().upper() for s in SCANNER_SYMBOLS.split(",") if s.strip()]
    if not POLYGON_API_KEY: return []
    out=[]; page_token=None; pages=0
    while pages < SCANNER_MAX_PAGES:
        params={"market":"stocks","active":"true","limit":"1000","apiKey":POLYGON_API_KEY}
        if page_token: params["cursor"]=page_token
        url="https://api.polygon.io/v3/reference/tickers"
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200: break
        j=r.json(); results=j.get("results", [])
        out.extend([x["ticker"] for x in results if x.get("ticker")])
        page_token=j.get("next_url", None)
        if not page_token: break
        pages+=1
    return [s for s in out if s.isalnum()]

# ==============================
# TradersPost I/O
# ==============================
def send_to_traderspost(payload: dict):
    try:
        if DRY_RUN or not ALLOW_ENTRIES:
            print(f"[DRY/PAUSED] {json.dumps(payload)[:500]}", flush=True); return True, "dry/paused"
        if not TP_URL:
            print("[ERROR] TP_WEBHOOK_URL missing.", flush=True); return False, "no-webhook-url"
        # throttle per minute
        now=time.time()
        while _order_times and now - _order_times[0] > 60: _order_times.popleft()
        if len(_order_times) >= MAX_ORDERS_PER_MIN:
            return False, f"throttled: {len(_order_times)}/{MAX_ORDERS_PER_MIN} in last 60s"
        _order_times.append(now)
        payload.setdefault("meta",{}); payload["meta"]["environment"]="paper"; payload["meta"]["sentAt"]=datetime.now(timezone.utc).isoformat()
        r=requests.post(TP_URL, json=payload, timeout=12)
        ok = 200 <= r.status_code < 300; info=f"{r.status_code} {r.text[:300]}"
        if not ok: print(f"[POST ERROR] {info}", flush=True)
        return ok, info
    except Exception as e:
        return False, f"exception: {e}"

def build_payload(symbol: str, sig: dict):
    action=sig.get("action"); order_type=sig.get("orderType","market"); qty=int(sig.get("quantity",0))
    payload={"ticker":symbol,"action":action,"orderType":order_type,"quantity":qty,"meta":{}}
    if isinstance(sig.get("meta"), dict): payload["meta"].update(sig["meta"])
    if sig.get("barTime"): payload["meta"]["barTime"]=sig["barTime"]
    if order_type.lower()=="limit" and sig.get("price") is not None: payload["limitPrice"]=float(round(sig["price"],2))
    tp_abs=sig.get("tp_abs", sig.get("takeProfit")); sl_abs=sig.get("sl_abs", sig.get("stopLoss"))
    if tp_abs is not None: payload["takeProfit"]={"limitPrice": float(round(tp_abs,2))}
    if sl_abs is not None: payload["stopLoss"]={"type":"stop","stopPrice": float(round(sl_abs,2))}
    return payload

# ==============================
# Perf & ledger
# ==============================
def _combo_key(strategy: str, symbol: str, tf_min: int) -> str: return f"{strategy}|{symbol}|{int(tf_min)}"
def _perf_init(combo: str):
    if combo not in PERF: PERF[combo]={"trades":0,"wins":0,"losses":0,"gross_profit":0.0,"gross_loss":0.0,"net_pnl":0.0,"max_dd":0.0,"equity_curve":[0.0]}
def _perf_update(combo: str, pnl: float):
    _perf_init(combo); p=PERF[combo]; p["trades"]+=1
    if pnl>0: p["wins"]+=1; p["gross_profit"]+=pnl
    elif pnl<0: p["losses"]+=1; p["gross_loss"]+=pnl
    p["net_pnl"]+=pnl; ec=p["equity_curve"]; ec.append(ec[-1]+pnl); p["max_dd"]=min(p["max_dd"], min(0.0, ec[-1]-max(ec)))

def _record_open_trade(strat_name: str, symbol: str, tf_min: int, sig: dict):
    combo=_combo_key(strat_name, symbol, tf_min); _perf_init(combo)
    tp=sig.get("tp_abs", sig.get("takeProfit")); sl=sig.get("sl_abs", sig.get("stopLoss"))
    t=LiveTrade(combo, symbol, int(tf_min), sig["action"], float(sig.get("entry") or sig.get("price") or sig.get("lastClose") or 0.0),
                float(tp) if tp is not None else float("nan"), float(sl) if sl is not None else float("nan"),
                int(sig["quantity"]), sig.get("barTime") or datetime.now(timezone.utc).isoformat())
    OPEN_TRADES[(symbol, int(tf_min))].append(t)

def _maybe_close_on_bar(symbol: str, tf_min: int, ts, high: float, low: float, close: float):
    key=(symbol, int(tf_min))
    for t in OPEN_TRADES.get(key, []):
        if not t.is_open: continue
        hit_tp=(high>=t.tp) if t.side=="buy" else (low<=t.tp)
        hit_sl=(low<=t.sl)  if t.side=="buy" else (high>=t.sl)
        if hit_tp or hit_sl:
            t.is_open=False; t.exit_time=ts.tz_convert("UTC").isoformat() if hasattr(ts, "tzinfo") else str(ts)
            t.exit=t.tp if hit_tp else t.sl; t.reason="tp" if hit_tp else "sl"
            pnl=(t.exit - t.entry)*t.qty if t.side=="buy" else (t.entry - t.exit)*t.qty
            _perf_update(t.combo, pnl)
            print(f"[CLOSE] {t.combo} {t.reason.upper()} qty={t.qty} entry={t.entry:.2f} exit={t.exit:.2f} pnl={pnl:+.2f}", flush=True)

# ==============================
# Sentiment (same structure as your file)
# ==============================
def _now_et_tz(): return datetime.now(timezone.utc).astimezone(pytz.timezone("America/New_York"))
def _is_rth_tz(ts): h,m=ts.hour,ts.minute; return ((h>9) or (h==9 and m>=30)) and (h<16)

def compute_sentiment():
    mode=os.getenv("SENTIMENT_MODE","rth_only").lower()
    symbols=[s.strip() for s in os.getenv("SENTS_SYMBOLS","SPY,QQQ").split(",") if s.strip()]
    look_min=int(os.getenv("SENTS_LOOKBACK_MIN","30")); tf_min=int(os.getenv("SENTS_TF","1"))
    gap_bp=float(os.getenv("SENTS_GAP_THRESHOLD_BP","20"))/10000.0
    now_et=_now_et_tz(); is_rth=_is_rth_tz(now_et)

    def _intraday_momentum(sym):
        df=fetch_polygon_1m(sym, lookback_minutes=max(look_min, tf_min*look_min))
        if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex): return 0.0
        try: df.index=df.index.tz_convert("America/New_York")
        except: df.index=df.index.tz_localize("UTC").tz_convert("America/New_York")
        bars=_resample(df, tf_min); 
        if bars is None or bars.empty: return 0.0
        window=bars.iloc[-min(len(bars), look_min):]
        if len(window)<2: return 0.0
        return (float(window["close"].iloc[-1])/float(window["close"].iloc[0]))-1.0

    def _premarket_gap(sym):
        df=fetch_polygon_1m(sym, lookback_minutes=24*60)
        if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex): return 0.0
        try: df.index=df.index.tz_convert("America/New_York")
        except: df.index=df.index.tz_localize("UTC").tz_convert("America/New_York")
        dates=sorted({d.date() for d in df.index}); 
        if len(dates)<2: return 0.0
        yday=dates[-2]; y_rth=df[(df.index.date==yday) & (df.index.map(_is_rth_tz))]
        if y_rth.empty: return 0.0
        y_close=float(y_rth["close"].iloc[-1]); last_px=float(df["close"].iloc[-1])
        return (last_px/y_close)-1.0

    vals=[]
    for s in symbols:
        if mode=="intraday_momentum": vals.append(_intraday_momentum(s))
        elif mode=="premarket_gap": vals.append(_intraday_momentum(s) if is_rth else _premarket_gap(s))
        else: vals.append(_intraday_momentum(s) if is_rth else 0.0)

    if not vals: print("[SENTIMENT] no data; neutral", flush=True); return "neutral"
    avg=sum(vals)/len(vals)
    if mode=="premarket_gap" and not is_rth:
        out="bull" if avg>=gap_bp else "bear" if avg<=-gap_bp else "neutral"
        print(f"[SENTIMENT] premarket gap avg={avg:+.4%} → {out}", flush=True); return out
    up_th=float(os.getenv("SENTS_UP_THRESH","0.0015")); dn_th=float(os.getenv("SENTS_DN_THRESH","-0.0015"))
    out="bull" if avg>=up_th else "bear" if avg<=dn_th else "neutral"
    print(f"[SENTIMENT] avg_momentum={avg:+.4%} → {out}", flush=True); return out

# ==============================
# ML strategy with confidence
# ==============================
def signal_ml_pattern(symbol: str, df1m: pd.DataFrame, tf_min: int, conf_threshold=None, r_multiple=None):
    try:
        from sklearn.ensemble import RandomForestClassifier
        import pandas_ta as ta
    except Exception:
        return None
    conf_threshold=float(conf_threshold if conf_threshold is not None else SCANNER_CONF_THRESHOLD)
    r_multiple=float(r_multiple if r_multiple is not None else SCANNER_R_MULTIPLE)
    if df1m is None or df1m.empty or not isinstance(df1m.index, pd.DatetimeIndex): return None
    bars=_resample(df1m, tf_min)
    if bars.empty or len(bars)<120: return None
    bars=bars.copy()
    bars["return"]=bars["close"].pct_change()
    try: bars["rsi"]=ta.rsi(bars["close"], length=14)
    except Exception:
        d=bars["close"].diff(); up=d.clip(lower=0).rolling(14).mean(); dn=-d.clip(upper=0).rolling(14).mean()
        rs=up/(dn.replace(0,np.nan)); bars["rsi"]=100-(100/(1+rs))
    bars["volatility"]=bars["close"].rolling(20).std()
    bars.dropna(inplace=True)
    if len(bars)<60: return None
    X=bars[["return","rsi","volatility"]].copy()
    y=(bars["close"].shift(-1) > bars["close"]).astype(int)
    X=X.iloc[:-1]; y=y.iloc[:-1]
    if len(X)<50: return None
    cut=int(len(X)*0.7); X_train,y_train=X.iloc[:cut],y.iloc[:cut]
    clf=RandomForestClassifier(n_estimators=100, random_state=42); clf.fit(X_train, y_train)
    x_live=X.iloc[[-1]]
    try: x_live=x_live[list(clf.feature_names_in_)]
    except Exception: pass
    proba=float(clf.predict_proba(x_live)[0][1]); pred=int(proba>=0.5)
    ts=bars.index[-1]
    if not _in_session(ts):
        return None
    
    if pred == 1:
        price = float(bars["close"].iloc[-1])
        sl    = price * 0.99
        tp    = price * (1 + 0.01 * r_multiple)
        qty   = _position_qty(price, sl)
        if qty <= 0:
            return None
        return {
            "action": "buy",
            "orderType": "market",
            "price": None,
            "takeProfit": tp,
            "stopLoss": sl,
            "barTime": ts.tz_convert("UTC").isoformat(),
            "entry": price,
            "quantity": int(qty),
            "confidence": float(proba),
            "score": float(proba),
            "meta": {"note": "ml_pattern", "confidence": float(proba)}
        }

    return None

# ==============================
# Daily guard (realized only)
# ==============================
DAY_STAMP = datetime.now().astimezone().strftime("%Y-%m-%d"); HALT_TRADING=False
def _today_local_date_str(): return datetime.now().astimezone().strftime("%Y-%m-%d")
def _realized_day_pnl()->float: return sum(float(p.get("net_pnl",0.0)) for p in PERF.values())
def _opposite(a:str)->str: return "sell" if a=="buy" else "buy"

def flatten_all_open_positions():
    posted=0
    for (sym, tf), trades in list(OPEN_TRADES.items()):
        for t in trades:
            if not t.is_open or not t.qty or not t.symbol: continue
            payload={"ticker":t.symbol,"action":_opposite(t.side),"orderType":"market","quantity":int(t.qty),
                     "meta":{"note":"daily-guard-flatten","combo":t.combo,"triggeredAt":datetime.now(timezone.utc).isoformat()}}
            ok,info=send_to_traderspost(payload); print(f"[DAILY-GUARD] Flatten {t.combo} -> ok={ok} info={info}", flush=True)
            t.is_open=False; t.exit_time=datetime.now(timezone.utc).isoformat(); t.exit=t.entry; t.reason="daily_guard"; posted+=1
    print(f"[DAILY-GUARD] Flatten requests posted: {posted}", flush=True)

def reset_daily_guard_if_new_day():
    global DAY_STAMP, HALT_TRADING
    today=_today_local_date_str()
    if today!=DAY_STAMP:
        HALT_TRADING=False; DAY_STAMP=today
        print(f"[DAILY-GUARD] New day -> reset HALT_TRADING. Day={DAY_STAMP}", flush=True)

def check_daily_guard_and_maybe_halt():
    global HALT_TRADING
    if not DAILY_GUARD_ENABLED: return
    realized=_realized_day_pnl(); equity=START_EQUITY + realized
    up_lim=START_EQUITY*(1.0+DAILY_TP_PCT); dn_lim=START_EQUITY*(1.0-DAILY_DD_PCT)
    print(f"[DAILY-GUARD] eq={equity:.2f} start={START_EQUITY:.2f} realized={realized:+.2f} targets +{DAILY_TP_PCT*100:.1f}%({up_lim:.2f}) / -{DAILY_DD_PCT*100:.1f}%({dn_lim:.2f})", flush=True)
    if HALT_TRADING: return
    if equity>=up_lim:
        HALT_TRADING=True; print("[DAILY-GUARD] ✅ Daily TP hit. Halting entries.", flush=True)
        if DAILY_FLATTEN_ON_HIT: flatten_all_open_positions()
    elif equity<=dn_lim:
        HALT_TRADING=True; print("[DAILY-GUARD] ⛔ Daily DD hit. Halting entries.", flush=True)
        if DAILY_FLATTEN_ON_HIT: flatten_all_open_positions()

# ==============================
# Diagnostics helpers
# ==============================
def _log_candidate_summary(cands):
    if not cands:
        print("[SCAN] candidates=0", flush=True); return
    top = sorted(cands, key=lambda x: x[0], reverse=True)[:5]
    view = ", ".join([f"{sym} {tf}m {score:.3f}" for score, sym, tf, _ in top])
    print(f"[SCAN] candidates={len(cands)} | top5: {view}", flush=True)

def _print_gate_counts(gc):
    # Always print the same keys for readability
    keys = ["no_data","vol_gate","no_signal","conf_gate","sentiment_block","dedupe"]
    parts=[f"{k}={gc.get(k,0)}" for k in keys]
    print("[GATES] " + " | ".join(parts), flush=True)

# ==============================
# Router (single signal)
# ==============================
def compute_signal(strategy_name, symbol, tf_minutes, df1m=None):
    if df1m is None or getattr(df1m, "empty", True):
        df1m = fetch_polygon_1m(symbol, lookback_minutes=max(240, tf_minutes*240))
        if df1m is None or df1m.empty: return None
    if not isinstance(df1m.index, pd.DatetimeIndex):
        try: df1m.index = pd.to_datetime(df1m.index, utc=True)
        except Exception: return None
    try: df1m.index = df1m.index.tz_convert(MARKET_TZ)
    except Exception: df1m.index = df1m.index.tz_localize("UTC").tz_convert(MARKET_TZ)

    # today volume gate
    try:
        last_day=df1m.index[-1].date(); todays=df1m.loc[df1m.index.date==last_day]
        todays_vol=float(todays["volume"].sum()) if not todays.empty else 0.0
    except Exception: todays_vol=0.0
    if todays_vol < SCANNER_MIN_TODAY_VOL: return None

    if strategy_name=="ml_pattern":
        return signal_ml_pattern(symbol, df1m, tf_minutes, SCANNER_CONF_THRESHOLD, SCANNER_R_MULTIPLE)
    return None

# ==============================
# Main loop
# ==============================
def main():
    print("Scanner starting…", flush=True)
    symbols = get_universe_symbols()
    print(f"[UNIVERSE] symbols={len(symbols)}  TFs={TF_MIN_LIST}  vol_gate={SCANNER_MIN_TODAY_VOL}", flush=True)

    if REPLAY_ON_START:
        df1m = fetch_polygon_1m(REPLAY_SYMBOL, lookback_minutes=REPLAY_HOURS*60)
        if df1m is not None and not df1m.empty:
            bars=_resample(df1m, REPLAY_TF)
            hits=0; last_key=None
            for i in range(len(bars)):
                df_slice=df1m.loc[:bars.index[i]]
                sig=signal_ml_pattern(REPLAY_SYMBOL, df_slice, REPLAY_TF, SCANNER_CONF_THRESHOLD, SCANNER_R_MULTIPLE)
                if sig:
                    if REPLAY_SEND_ORDERS:
                        payload=build_payload(REPLAY_SYMBOL, sig); print("[REPLAY-ORDER]", json.dumps(payload)[:200], flush=True)
                    k=(sig.get("barTime",""), sig.get("action",""))
                    if k != last_key: hits+=1; last_key=k
            print(f"[REPLAY] ml_pattern {REPLAY_SYMBOL} {REPLAY_TF}m -> {hits} signals in {REPLAY_HOURS}h.", flush=True)

    while True:
        loop_start=time.time()
        try:
            print("Tick…", flush=True)

            # Sentiment snapshot
            sentiment = compute_sentiment() if SENTIMENT_ONLY_GATE else "neutral"
            print(f"[SENTIMENT] {sentiment}", flush=True)

            # Daily guard housekeeping
            reset_daily_guard_if_new_day()
            if DAILY_GUARD_ENABLED: check_daily_guard_and_maybe_halt()

            allow_new_entries = (not (DAILY_GUARD_ENABLED and HALT_TRADING))

            # Close phase on open trades
            touched=set((k[0],k[1]) for k in OPEN_TRADES.keys())
            for (sym, tf) in touched:
                try:
                    df=fetch_polygon_1m(sym, lookback_minutes=max(60, tf*12))
                    bars=_resample(df, tf)
                    if bars is not None and not bars.empty:
                        row=bars.iloc[-1]; ts=bars.index[-1]
                        _maybe_close_on_bar(sym, tf, ts, float(row["high"]), float(row["low"]), float(row["close"]))
                except Exception as e:
                    print(f"[CLOSE-PHASE ERROR] {sym} {tf}m: {e}", flush=True)

            if not allow_new_entries:
                time.sleep(POLL_SECONDS); continue

            # Concurrent cap
            open_positions=sum(1 for lst in OPEN_TRADES.values() for t in lst if t.is_open)
            if open_positions >= MAX_CONCURRENT_POSITIONS:
                print(f"[LIMIT] Max concurrent positions hit: {open_positions}/{MAX_CONCURRENT_POSITIONS}", flush=True)
                time.sleep(POLL_SECONDS); continue

            # -------- SCAN (collect + reason codes) --------
            candidates=[]  # (score, symbol, tf, sig)
            gate_counts=defaultdict(int)

            for sym in symbols:
                df1m=fetch_polygon_1m(sym, lookback_minutes=max(240, max(TF_MIN_LIST)*240))
                if df1m is None or df1m.empty or not isinstance(df1m.index, pd.DatetimeIndex):
                    gate_counts["no_data"]+=1; continue
                try: df1m.index=df1m.index.tz_convert(MARKET_TZ)
                except Exception: df1m.index=df1m.index.tz_localize("UTC").tz_convert(MARKET_TZ)

                # Volume gate
                today_mask = df1m.index.date == df1m.index[-1].date()
                todays_vol = float(df1m.loc[today_mask, "volume"].sum()) if today_mask.any() else 0.0
                if todays_vol < SCANNER_MIN_TODAY_VOL:
                    gate_counts["vol_gate"]+=1; continue

                for tf in TF_MIN_LIST:
                    sig = compute_signal("ml_pattern", sym, tf, df1m=df1m)
                    if not sig:
                        gate_counts["no_signal"]+=1; continue

                    # Sentiment gate (long-only strategy)
                    if SENTIMENT_ONLY_GATE:
                        if sentiment == "bull" and sig["action"] != "buy":
                            gate_counts["sentiment_block"]+=1; continue
                        if sentiment == "bear" and sig["action"] != "sell":
                            gate_counts["sentiment_block"]+=1; continue
                        # neutral allows both

                    # Pre-check dedupe (don’t burn yet)
                    k = _dedupe_key("ml_pattern", sym, tf, sig["action"], sig.get("barTime",""))
                    if k in _sent_keys:
                        gate_counts["dedupe"]+=1; continue

                    score = float(sig.get("score", sig.get("confidence", 0.0)))
                    if score < SCANNER_CONF_THRESHOLD:
                        gate_counts["conf_gate"]+=1; continue

                    candidates.append((score, sym, tf, sig))

            _log_candidate_summary(candidates)
            _print_gate_counts(gate_counts)

            # -------- DISPATCH (ranked) --------
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                sent=0
                for score, sym, tf, sig in candidates:
                    # concurrent cap
                    open_positions=sum(1 for lst in OPEN_TRADES.values() for t in lst if t.is_open)
                    if open_positions >= MAX_CONCURRENT_POSITIONS:
                        print(f"[LIMIT] Max concurrent reached mid-loop.", flush=True); break
                    if MAX_ENTRIES_PER_CYCLE > 0 and sent >= MAX_ENTRIES_PER_CYCLE: break

                    # final dedupe burn
                    k = _dedupe_key("ml_pattern", sym, tf, sig["action"], sig.get("barTime",""))
                    if k in _sent_keys:
                        continue
                    _sent_keys.add(k)

                    # Record + send
                    combo_key = _combo_key("ml_pattern", sym, tf)
                    sig.setdefault("meta", {}); sig["meta"]["confidence"]=float(sig.get("confidence", score))
                    _record_open_trade("ml_pattern", sym, tf, sig)
                    payload = build_payload(sym, sig)
                    ok, info = send_to_traderspost(payload)
                    if ok:
                        COUNTS["orders.ok"]+=1; COMBO_COUNTS[f"{combo_key}::orders.ok"]+=1; sent+=1
                    else:
                        COUNTS["orders.err"]+=1; COMBO_COUNTS[f"{combo_key}::orders.err"]+=1
                    print(f"[ORDER] {combo_key} -> qty={sig.get('quantity')} conf={score:.3f} ok={ok} info={info}", flush=True)

            # EOD flatten (16:00–16:10 ET)
            now_et=_now_et()
            if now_et.hour==16 and now_et.minute<10:
                print("[EOD] Auto-flatten window.", flush=True); flatten_all_open_positions()

        except Exception as e:
            import traceback
            print("[LOOP ERROR]", e, traceback.format_exc(), flush=True)

        elapsed=time.time()-loop_start
        time.sleep(max(1, POLL_SECONDS - int(elapsed)))

if __name__ == "__main__":
    main()
