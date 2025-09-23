# Ranked_ML.py — ML scanner with confidence ranking, universe batching, two-sided trades, guards, and diagnostics
# v2.5: True market-wide ranking via round-robin batches + explicit rejection breakdown per loop.

import os, time, json, math, hashlib, requests, traceback, random
from collections import defaultdict, deque, Counter
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

APP_TAG = "Ranked_ML v2.5 (batch+reasons)"
print(f"[BOOT] {APP_TAG}", flush=True)

# -------- ENV HELPERS --------
def _env_bool(key, default="0"):
    return os.getenv(key, default).lower() in ("1","true","yes")

TP_URL              = os.getenv("TP_WEBHOOK_URL", "")
POLYGON_API_KEY     = os.getenv("POLYGON_API_KEY", "")
POLL_SECONDS        = int(os.getenv("POLL_SECONDS", "10"))
RUN_ID              = datetime.now().astimezone().strftime("%Y-%m-%d")
MARKET_TZ           = "America/New_York"

# Ranking & control
SCANNER_CONF_THRESHOLD = float(os.getenv("SCANNER_CONF_THRESHOLD", os.getenv("CONF_THR","0.80")))
SCANNER_R_MULTIPLE     = float(os.getenv("SCANNER_R_MULTIPLE", os.getenv("R_MULT","3.0")))
MAX_ENTRIES_PER_CYCLE  = int(os.getenv("MAX_ENTRIES_PER_CYCLE", "0"))  # 0/neg => unlimited
ALLOW_ENTRIES          = _env_bool("ALLOW_ENTRIES","1")                # gates DISPATCH only

# Universe
SCANNER_SYMBOLS       = os.getenv("SCANNER_SYMBOLS", "").strip()
SCANNER_MAX_PAGES     = int(os.getenv("SCANNER_MAX_PAGES", "11"))      # enough to pull ~8k tickers
SCANNER_MIN_TODAY_VOL = int(os.getenv("SCANNER_MIN_TODAY_VOL", os.getenv("MIN_TODAY_VOL","100000")))
TF_MIN_LIST           = [int(x) for x in os.getenv("TF_MIN_LIST", "1,2,3,5,10").split(",") if x.strip()]

# NEW: batch scanning controls
SCAN_BATCH_SIZE       = int(os.getenv("SCAN_BATCH_SIZE","250"))        # symbols per loop
BATCH_SHUFFLE         = _env_bool("BATCH_SHUFFLE","1")                 # shuffle tickers once at boot

# Price/exchange filters
MIN_PRICE = float(os.getenv("MIN_PRICE","5.0"))
ALLOWED_EXCHANGES = [s.strip().upper() for s in os.getenv(
    "ALLOWED_EXCHANGES","XNYS,XNAS,NYSE,NASDAQ,NASD").split(",") if s.strip()]

# Risk / sizing
EQUITY_USD  = float(os.getenv("EQUITY_USD",  "100000"))
RISK_PCT    = float(os.getenv("RISK_PCT",    "0.01"))
MAX_POS_PCT = float(os.getenv("MAX_POS_PCT", "0.10"))
MIN_QTY     = int(os.getenv("MIN_QTY", "1"))
ROUND_LOT   = int(os.getenv("ROUND_LOT","1"))

# Guards & session
MAX_CONCURRENT_POSITIONS = int(os.getenv("MAX_CONCURRENT_POSITIONS", "100"))
MAX_ORDERS_PER_MIN       = int(os.getenv("MAX_ORDERS_PER_MIN", "60"))
ALLOW_PREMARKET          = _env_bool("ALLOW_PREMARKET","0")
ALLOW_AFTERHOURS         = _env_bool("ALLOW_AFTERHOURS","0")
BYPASS_SESSION           = _env_bool("BYPASS_SESSION","0")

# Sentiment
SENTIMENT_ONLY_GATE      = _env_bool("SENTIMENT_ONLY_GATE","1")

# Two-sided
SHORTS_ENABLED           = _env_bool("SHORTS_ENABLED","1")

# Daily guard
START_EQUITY         = float(os.getenv("START_EQUITY", str(EQUITY_USD)))
DAILY_TP_PCT         = float(os.getenv("DAILY_TP_PCT", "0.25"))
DAILY_DD_PCT         = float(os.getenv("DAILY_DD_PCT", "0.05"))
DAILY_FLATTEN_ON_HIT = _env_bool("DAILY_FLATTEN_ON_HIT","1")
DAILY_GUARD_ENABLED  = _env_bool("DAILY_GUARD_ENABLED","1")

# Diagnostics
INTROSPECT         = _env_bool("INTROSPECT","0")
INTROSPECT_TOP_K   = int(os.getenv("INTROSPECT_TOP_K","10"))
LOG_REJECTIONS_N   = int(os.getenv("LOG_REJ_N", os.getenv("LOG_REJECTIONS_N","50")))  # max detailed NOSIG lines per loop
DIAG_ON            = _env_bool("DIAG_ON","0")
DIAG_SYMBOLS       = [s.strip().upper() for s in os.getenv("DIAG_SYMBOLS","SPY,QQQ,AAPL,NVDA").split(",") if s.strip()]
DIAG_SIG_DETAILS   = _env_bool("DIAG_SIG_DETAILS","0")
DIAG_MODEL_TRACE   = _env_bool("DIAG_MODEL_TRACE","0")

print("[CONFIG] "
      f"ALLOW_ENTRIES={ALLOW_ENTRIES} DRY_RUN={os.getenv('DRY_RUN','0')=='1'} BYPASS_SESSION={BYPASS_SESSION} "
      f"ALLOW_PREMARKET={ALLOW_PREMARKET} ALLOW_AFTERHOURS={ALLOW_AFTERHOURS} "
      f"SHORTS_ENABLED={SHORTS_ENABLED} SENTIMENT_ONLY_GATE={SENTIMENT_ONLY_GATE} "
      f"CONF_THR={SCANNER_CONF_THRESHOLD} R_MULT={SCANNER_R_MULTIPLE} "
      f"MIN_TODAY_VOL={SCANNER_MIN_TODAY_VOL} TFs={TF_MIN_LIST} "
      f"MIN_PRICE={MIN_PRICE} ALLOWED_EXCHANGES={ALLOWED_EXCHANGES} "
      f"INTROSPECT={INTROSPECT} DIAG_ON={DIAG_ON} DIAG_SIG_DETAILS={DIAG_SIG_DETAILS} DIAG_MODEL_TRACE={DIAG_MODEL_TRACE} "
      f"LOG_REJ_N={LOG_REJECTIONS_N} SCAN_BATCH_SIZE={SCAN_BATCH_SIZE} BATCH_SHUFFLE={BATCH_SHUFFLE}",
      flush=True)

# -------- STATE --------
COUNTS        = defaultdict(int)
COMBO_COUNTS  = defaultdict(int)
PERF          = {}
OPEN_TRADES   = defaultdict(list)
_sent_keys    = set()
_order_times  = deque()

DAY_STAMP     = datetime.now().astimezone().strftime("%Y-%m-%d")
HALT_TRADING  = False

# batching
_UNIVERSE      = []
_BATCH_INDEX   = 0

# -------- TIME --------
def _now_et(): return datetime.now(timezone.utc).astimezone(ZoneInfo(MARKET_TZ))
def _is_rth(ts): h,m=ts.hour,ts.minute; return ((h>9) or (h==9 and m>=30)) and (h<16)
def _in_session(ts):
    if BYPASS_SESSION: return True
    if _is_rth(ts): return True
    if ALLOW_PREMARKET and (4 <= ts.hour < 9 or (ts.hour == 9 and ts.minute < 30)): return True
    if ALLOW_AFTERHOURS and (16 <= ts.hour < 20): return True
    return False

# -------- RESAMPLE / SIZING --------
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

# -------- POLYGON --------
def _http_get(url, params=None, timeout=15):
    params = params.copy() if params else {}
    if POLYGON_API_KEY: params["apiKey"] = POLYGON_API_KEY
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code != 200:
            return None, f"{r.status_code} {r.text[:160]}"
        return r.json(), None
    except Exception as e:
        return None, f"exception {e}"

def fetch_polygon_1m(symbol: str, lookback_minutes: int = 2400) -> pd.DataFrame:
    if not POLYGON_API_KEY: return pd.DataFrame()
    end = datetime.now(timezone.utc); start = end - timedelta(minutes=lookback_minutes)
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
    js, err = _http_get(url, {"adjusted":"true","sort":"asc","limit":"50000"})
    if err or not js or not js.get("results"): return pd.DataFrame()
    try:
        df = pd.DataFrame(js["results"])
        for col in ("o","h","l","c","v","t"):
            if col not in df.columns: return pd.DataFrame()
        df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        df = df.set_index("ts").sort_index(); df.index = df.index.tz_convert(MARKET_TZ)
        df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
        return df[["open","high","low","close","volume"]]
    except Exception:
        return pd.DataFrame()

def _passes_price_exchange(symbol: str) -> bool:
    # Optional: quick reference hit for exchange + price filter
    url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
    js, err = _http_get(url)
    if err or not js or not js.get("results"): return True  # be permissive if ref call fails
    res = js["results"]
    exch = str(res.get("primary_exchange", "") or res.get("listing_exchange","")).upper()
    if exch and ALLOWED_EXCHANGES and exch not in ALLOWED_EXCHANGES: return False
    # last quote end-of-day price
    try:
        last = float(res.get("last_updated_price") or 0.0)
        if last and last < MIN_PRICE: return False
    except Exception:
        pass
    return True

def get_universe_symbols() -> list:
    if SCANNER_SYMBOLS:
        return [s.strip().upper() for s in SCANNER_SYMBOLS.split(",") if s.strip()]
    if not POLYGON_API_KEY: return []
    out=[]; cursor=None; pages=0
    while pages < SCANNER_MAX_PAGES:
        params={"market":"stocks","active":"true","limit":"1000"}
        if cursor: params["cursor"]=cursor
        js, err = _http_get("https://api.polygon.io/v3/reference/tickers", params, timeout=20)
        if err or not js: break
        results = js.get("results", [])
        for x in results:
            t=x.get("ticker")
            if not t or not t.isalnum(): continue
            out.append(t)
        cursor = js.get("next_url", None)
        pages += 1
        if not cursor: break
    return out

# -------- SENTIMENT --------
def compute_sentiment():
    def _is_rth_tz(ts): h,m=ts.hour,ts.minute; return ((h>9) or (h==9 and m>=30)) and (h<16)
    mode=os.getenv("SENTIMENT_MODE","rth_only").lower()
    symbols=[s.strip() for s in os.getenv("SENTS_SYMBOLS","SPY,QQQ").split(",") if s.strip()]
    look_min=int(os.getenv("SENTS_LOOKBACK_MIN","30")); tf_min=int(os.getenv("SENTS_TF","1"))
    gap_bp=float(os.getenv("SENTS_GAP_THRESHOLD_BP","20"))/10000.0
    now_et=_now_et(); is_rth=_is_rth_tz(now_et)

    def _intraday(sym):
        df=fetch_polygon_1m(sym, lookback_minutes=max(look_min, tf_min*look_min))
        if df is None or df.empty: return 0.0
        bars=_resample(df, tf_min)
        if bars is None or bars.empty: return 0.0
        w=bars.iloc[-min(len(bars), look_min):]
        if len(w)<2: return 0.0
        return (float(w["close"].iloc[-1])/float(w["close"].iloc[0]))-1.0

    def _premarket_gap(sym):
        df=fetch_polygon_1m(sym, lookback_minutes=24*60)
        if df is None or df.empty: return 0.0
        try: df.index=df.index.tz_convert(MARKET_TZ)
        except: df.index=df.index.tz_localize("UTC").tz_convert(MARKET_TZ)
        dates=sorted({d.date() for d in df.index})
        if len(dates)<2: return 0.0
        yday=dates[-2]; y_rth=df[(df.index.date==yday) & (df.index.map(_is_rth_tz))]
        if y_rth.empty: return 0.0
        y_close=float(y_rth["close"].iloc[-1]); last_px=float(df["close"].iloc[-1])
        return (last_px/y_close)-1.0

    vals=[]
    for s in symbols:
        if mode=="intraday_momentum": vals.append(_intraday(s))
        elif mode=="premarket_gap": vals.append(_intraday(s) if is_rth else _premarket_gap(s))
        else: vals.append(_intraday(s) if is_rth else 0.0)

    if not vals: print("[SENTIMENT] no data; neutral", flush=True); return "neutral"
    avg=sum(vals)/len(vals)
    if mode=="premarket_gap" and not is_rth:
        out="bull" if avg>=gap_bp else "bear" if avg<=-gap_bp else "neutral"
        print(f"[SENTIMENT] premarket gap avg={avg:+.4%} → {out}", flush=True); return out
    up_th=float(os.getenv("SENTS_UP_THRESH","0.0015")); dn_th=float(os.getenv("SENTS_DN_THRESH","-0.0015"))
    out="bull" if avg>=up_th else "bear" if avg<=dn_th else "neutral"
    print(f"[SENTIMENT] avg_momentum={avg:+.4%} → {out}", flush=True); return out

# -------- PERF / LEDGER --------
def _combo_key(strategy: str, symbol: str, tf_min: int) -> str: return f"{strategy}|{symbol}|{int(tf_min)}"

def _perf_init(combo: str):
    if combo not in PERF:
        PERF[combo] = {"trades":0,"wins":0,"losses":0,"gross_profit":0.0,"gross_loss":0.0,
                       "net_pnl":0.0,"max_dd":0.0,"equity_curve":[0.0]}

def _perf_update(combo: str, pnl: float):
    _perf_init(combo)
    p = PERF[combo]; p["trades"] += 1
    if pnl > 0: p["wins"] += 1; p["gross_profit"] += pnl
    elif pnl < 0: p["losses"] += 1; p["gross_loss"] += pnl
    p["net_pnl"] += pnl
    ec = p["equity_curve"]; ec.append(ec[-1] + pnl)
    dd = ec[-1] - max(ec); p["max_dd"] = min(p["max_dd"], dd)

class LiveTrade:
    def __init__(self, combo, symbol, tf_min, side, entry, tp, sl, qty, entry_time):
        self.combo = combo; self.symbol = symbol; self.tf_min = int(tf_min)
        self.side = side; self.entry = float(entry); self.tp = float(tp); self.sl = float(sl)
        self.qty = int(qty); self.entry_time = entry_time
        self.is_open=True; self.exit=None; self.exit_time=None; self.reason=None

def _record_open_trade(strat_name: str, symbol: str, tf_min: int, sig: dict):
    combo=_combo_key(strat_name, symbol, tf_min); _perf_init(combo)
    tp=sig.get("tp_abs", sig.get("takeProfit")); sl=sig.get("sl_abs", sig.get("stopLoss"))
    t=LiveTrade(combo, symbol, tf_min, sig["action"], sig.get("entry") or sig.get("price") or 0.0,
                float(tp) if tp is not None else 0.0, float(sl) if sl is not None else 0.0,
                int(sig["quantity"]), sig.get("barTime"))
    OPEN_TRADES[(symbol, tf_min)].append(t)

def _maybe_close_on_bar(symbol: str, tf_min: int, ts, high: float, low: float, close: float):
    key=(symbol, int(tf_min))
    for t in OPEN_TRADES.get(key, []):
        if not t.is_open: continue
        hit_tp=(high>=t.tp) if t.side=="buy" else (low<=t.tp)
        hit_sl=(low<=t.sl)  if t.side=="buy" else (high>=t.sl)
        if hit_tp or hit_sl:
            t.is_open=False; t.exit_time=ts.tz_convert("UTC").isoformat() if hasattr(ts,"tzinfo") else str(ts)
            t.exit=t.tp if hit_tp else t.sl; t.reason="tp" if hit_tp else "sl"
            pnl=(t.exit - t.entry)*t.qty if t.side=="buy" else (t.entry - t.exit)*t.qty
            _perf_update(t.combo, pnl)
            print(f"[CLOSE] {t.combo} {t.reason.upper()} qty={t.qty} entry={t.entry:.2f} exit={t.exit:.2f} pnl={pnl:+.2f}", flush=True)

# -------- DAILY GUARD --------
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
    print(f"[DAILY-GUARD] eq={equity:.2f} start={START_EQUITY:.2f} realized={realized:+.2f} "
          f"targets +{DAILY_TP_PCT*100:.1f}%({up_lim:.2f}) / -{DAILY_DD_PCT*100:.1f}%({dn_lim:.2f})", flush=True)
    if HALT_TRADING: return
    if equity>=up_lim:
        HALT_TRADING=True; print("[DAILY-GUARD] ✅ Daily TP hit. Halting entries.", flush=True)
        if DAILY_FLATTEN_ON_HIT: flatten_all_open_positions()
    elif equity<=dn_lim:
        HALT_TRADING=True; print("[DAILY-GUARD] ⛔ Daily DD hit. Halting entries.", flush=True)
        if DAILY_FLATTEN_ON_HIT: flatten_all_open_positions()

# -------- ML (two-sided, robust) --------
def _ml_features_from_resampled(bars: pd.DataFrame):
    out = bars.copy()
    out["return"] = out["close"].pct_change()
    try:
        import pandas_ta as ta
        out["rsi"] = ta.rsi(out["close"], length=14)
    except Exception:
        d=out["close"].diff(); up=d.clip(lower=0).rolling(14).mean(); dn=-d.clip(upper=0).rolling(14).mean()
        rs=up/(dn.replace(0,np.nan)); out["rsi"]=100-(100/(1+rs))
    out["volatility"] = out["close"].rolling(20).std()
    out.dropna(inplace=True); return out

def _fallback_proba_from_momentum(feats: pd.DataFrame, look=20):
    if feats is None or feats.empty or len(feats) < max(2, look+1): return 0.5
    ret = (float(feats["close"].iloc[-1]) / float(feats["close"].iloc[-look])) - 1.0
    return 0.5 + float(np.tanh(ret * 50.0)) / 2.0

def ml_score(symbol: str, df1m: pd.DataFrame, tf_min: int):
    """Return (proba_up, timestamp, reason). Never hard-fail: any error -> fallback momentum."""
    try:
        if df1m is None or df1m.empty or not isinstance(df1m.index, pd.DatetimeIndex):
            return None, None, "no_data"
        bars=_resample(df1m, tf_min)
        if bars.empty or len(bars)<60:
            return None, None, "bars_short"
        feats=_ml_features_from_resampled(bars)
        if feats.empty or len(feats)<50:
            return None, None, "nan_features"
        X=feats[["return","rsi","volatility"]].copy()
        y=(feats["close"].shift(-1) > feats["close"]).astype(int)
        X=X.iloc[:-1]; y=y.iloc[:-1]
        if len(X)<50:
            return None, None, "bars_short"

        from sklearn.ensemble import RandomForestClassifier
        cut=int(len(X)*0.7)
        X_train, y_train = X.iloc[:cut], y.iloc[:cut]

        if getattr(y_train, "nunique", None) and y_train.nunique() < 2:
            proba_up = _fallback_proba_from_momentum(feats); ts = feats.index[-1]
            if not _in_session(ts): return proba_up, ts, "not_in_session"
            return proba_up, ts, "fallback_single_class"

        clf=RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        x_live=X.iloc[[-1]]
        try: x_live=x_live[list(clf.feature_names_in_)]
        except Exception: pass

        proba_up=float(clf.predict_proba(x_live)[0][1])
        ts=feats.index[-1]
        if not _in_session(ts): return proba_up, ts, "not_in_session"
        return proba_up, ts, None

    except Exception as e:
        try:
            feats = feats if 'feats' in locals() else _ml_features_from_resampled(bars if 'bars' in locals() else pd.DataFrame())
            if feats is None or feats.empty:
                if DIAG_SIG_DETAILS:
                    print(f"[NOSIG] {symbol} {tf_min}m reason=model_error msg={str(e)[:120]}", flush=True)
                if DIAG_MODEL_TRACE: print("[TRACE]", traceback.format_exc(), flush=True)
                return None, None, "model_error"
            proba_up = _fallback_proba_from_momentum(feats); ts = feats.index[-1]
            if DIAG_SIG_DETAILS:
                print(f"[FALLBACK] {symbol} {tf_min}m reason=fallback_model_error proba_up={proba_up:.3f} msg={str(e)[:120]}", flush=True)
            if DIAG_MODEL_TRACE: print("[TRACE]", traceback.format_exc(), flush=True)
            if not _in_session(ts): return proba_up, ts, "not_in_session"
            return proba_up, ts, "fallback_model_error"
        except Exception as e2:
            if DIAG_SIG_DETAILS:
                print(f"[NOSIG] {symbol} {tf_min}m reason=model_error_fallback_failed msg={str(e2)[:120]}", flush=True)
            if DIAG_MODEL_TRACE:
                print("[TRACE-FALLBACK]", traceback.format_exc(), flush=True)
            return None, None, "model_error"

def signal_ml_pattern(symbol: str, df1m: pd.DataFrame, tf_min: int, r_multiple=None, reasons=None):
    if reasons is None: reasons = Counter()
    r_multiple=float(r_multiple if r_multiple is not None else SCANNER_R_MULTIPLE)
    proba_up, ts, reason=ml_score(symbol, df1m, tf_min)

    if proba_up is None:
        reasons[reason or "no_score"] += 1
        if DIAG_SIG_DETAILS and len(reasons) <= LOG_REJECTIONS_N:
            print(f"[NOSIG] {symbol} {tf_min}m reason={reason}", flush=True)
        return None
    if reason == "not_in_session":
        reasons["not_in_session"] += 1
        if DIAG_SIG_DETAILS and len(reasons) <= LOG_REJECTIONS_N:
            print(f"[NOSIG] {symbol} {tf_min}m reason=not_in_session proba_up={proba_up:.3f}", flush=True)
        return None

    bars=_resample(df1m, tf_min)
    if bars is None or bars.empty:
        reasons["bars_empty_after_score"] += 1
        if DIAG_SIG_DETAILS and len(reasons) <= LOG_REJECTIONS_N:
            print(f"[NOSIG] {symbol} {tf_min}m reason=bars_empty_after_score", flush=True)
        return None

    price=float(bars["close"].iloc[-1])

    # price / exchange guard (quick)
    # (Skip heavy ref call each time; rely on MIN_PRICE quickly)
    if price < MIN_PRICE:
        reasons["min_price"] += 1
        return None

    if proba_up >= 0.5:
        side="buy"; score=proba_up; sl=price*0.99; tp=price*(1+0.01*r_multiple)
    else:
        if not SHORTS_ENABLED:
            reasons["shorts_disabled"] += 1
            return None
        side="sell"; score=1.0-proba_up; sl=price*1.01; tp=price*(1-0.01*r_multiple)

    if score < SCANNER_CONF_THRESHOLD:
        reasons["conf_gate"] += 1
        return None

    qty=_position_qty(price, sl)
    if qty<=0:
        reasons["qty0"] += 1
        if DIAG_SIG_DETAILS and len(reasons) <= LOG_REJECTIONS_N:
            print(f"[NOSIG] {symbol} {tf_min}m reason=qty0 price={price:.2f} sl={sl:.2f} score={score:.3f} r_mult={r_multiple}", flush=True)
        return None

    meta_note = "ml_pattern"
    if reason in ("fallback_single_class", "fallback_model_error"):
        meta_note += f" ({reason})"

    return {
        "action":side,"orderType":"market","price":None,
        "takeProfit":tp,"stopLoss":sl,"barTime":ts.tz_convert("UTC").isoformat(),
        "entry":price,"quantity":int(qty),
        "confidence":score,"score":score,
        "meta":{"note":meta_note,"confidence":score}
    }

# -------- TP I/O --------
DRY_RUN = os.getenv("DRY_RUN","0") == "1"

def send_to_traderspost(payload: dict):
    try:
        if DRY_RUN or not ALLOW_ENTRIES:
            print(f"[DRY/PAUSED] {json.dumps(payload)[:300]}", flush=True); return True, "dry/paused"
        if not TP_URL:
            print("[ERROR] TP_WEBHOOK_URL missing.", flush=True); return False, "no-webhook-url"
        now=time.time()
        while _order_times and now - _order_times[0] > 60: _order_times.popleft()
        if len(_order_times) >= MAX_ORDERS_PER_MIN: return False, f"throttled: {len(_order_times)}/{MAX_ORDERS_PER_MIN} in last 60s"
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

# -------- LOOP HELPERS --------
def _log_candidate_summary(cands, reasons_counter):
    if not cands:
        top_rej = ", ".join([f"{k}:{v}" for k,v in reasons_counter.most_common(8)])
        print(f"[SCAN] candidates=0 | rejections {{{top_rej}}}", flush=True); return
    top = sorted(cands, key=lambda x: x[0], reverse=True)[:5]
    view = ", ".join([f"{sym} {tf}m {score:.3f}" for score, sym, tf, _ in top])
    print(f"[SCAN] candidates={len(cands)} | top5: {view}", flush=True)

def compute_signal(strategy_name, symbol, tf_minutes, df1m=None, reasons=None):
    if reasons is None: reasons = Counter()
    if df1m is None or getattr(df1m, "empty", True):
        df1m = fetch_polygon_1m(symbol, lookback_minutes=max(240, tf_minutes*240))
        if df1m is None or df1m.empty:
            reasons["df_empty"] += 1
            if DIAG_SIG_DETAILS and sum(reasons.values()) <= LOG_REJECTIONS_N:
                print(f"[NOSIG] {symbol} {tf_minutes}m reason=df_empty", flush=True)
            return None

    if not isinstance(df1m.index, pd.DatetimeIndex):
        try: df1m.index = pd.to_datetime(df1m.index, utc=True)
        except Exception:
            reasons["index_not_datetime"] += 1
            return None
    try: df1m.index = df1m.index.tz_convert(MARKET_TZ)
    except Exception: df1m.index = df1m.index.tz_localize("UTC").tz_convert(MARKET_TZ)

    today_mask = df1m.index.date == df1m.index[-1].date()
    todays_vol = float(df1m.loc[today_mask, "volume"].sum()) if today_mask.any() else 0.0
    if todays_vol < SCANNER_MIN_TODAY_VOL:
        reasons["vol_gate"] += 1
        return None

    if strategy_name=="ml_pattern":
        sig = signal_ml_pattern(symbol, df1m, tf_minutes, SCANNER_R_MULTIPLE, reasons)
        return sig
    return None

def _next_batch(symbols):
    global _BATCH_INDEX
    n=len(symbols)
    if n==0: return []
    start=_BATCH_INDEX
    end=min(n, start + max(1, SCAN_BATCH_SIZE))
    batch=symbols[start:end]
    _BATCH_INDEX = 0 if end>=n else end
    return batch

# -------- MAIN --------
def main():
    global _UNIVERSE

    print("Scanner starting…", flush=True)
    symbols = get_universe_symbols()
    if not symbols:
        print("[UNIVERSE] Empty; check POLYGON_API_KEY or SCANNER_SYMBOLS.", flush=True)
        symbols=[]
    # quick sanity log
    fl_hist=Counter([s[0] for s in symbols if s])
    preview=", ".join(symbols[:40])
    print(f"[UNIVERSE] symbols={len(symbols)}  TFs={TF_MIN_LIST}  vol_gate={SCANNER_MIN_TODAY_VOL}", flush=True)
    print(f"[UNIVERSE-DEBUG] first-letter hist => " + ", ".join([f"{k}:{fl_hist[k]}" for k in sorted(fl_hist)]), flush=True)
    if preview: print(f"[UNIVERSE-DEBUG] first 40 => {preview}", flush=True)
    _UNIVERSE = symbols[:]
    if BATCH_SHUFFLE: random.shuffle(_UNIVERSE)

    while True:
        loop_start=time.time()
        try:
            print("Tick…", flush=True)

            sentiment = compute_sentiment() if SENTIMENT_ONLY_GATE else "neutral"
            print(f"[SENTIMENT] {sentiment}", flush=True)

            reset_daily_guard_if_new_day()
            if DAILY_GUARD_ENABLED: check_daily_guard_and_maybe_halt()
            allow_new_entries = (not (DAILY_GUARD_ENABLED and HALT_TRADING)) and ALLOW_ENTRIES

            # EOD window flatten (16:00–16:10 ET)
            now_et=_now_et()
            if now_et.hour==16 and now_et.minute<10:
                print("[EOD] Auto-flatten window.", flush=True)
                flatten_all_open_positions()

            # ---- CLOSE PHASE on last bars for open trades ----
            touched = set((k[0], k[1]) for k in OPEN_TRADES.keys())
            for (sym, tf) in touched:
                try:
                    df = fetch_polygon_1m(sym, lookback_minutes=max(60, tf * 12))
                    bars = _resample(df, tf)
                    if bars is not None and not bars.empty:
                        row = bars.iloc[-1]
                        ts  = bars.index[-1]
                        _maybe_close_on_bar(sym, tf, ts, float(row["high"]), float(row["low"]), float(row["close"]))
                except Exception as e:
                    print(f"[CLOSE-PHASE ERROR] {sym} {tf}m: {e}", flush=True)

            # ---- ENTRY SCAN ----
            candidates=[]  # (score,symbol,tf,sig)
            reasons_counter=Counter()

            batch = _next_batch(_UNIVERSE)
            if not batch:
                batch = _next_batch(_UNIVERSE)  # defensive
            # fetch once per symbol
            for sym in batch:
                try:
                    df1m=fetch_polygon_1m(sym, lookback_minutes=max(240, max(TF_MIN_LIST)*240))
                    if df1m is None or df1m.empty or not isinstance(df1m.index, pd.DatetimeIndex):
                        reasons_counter["df_empty"] += 1
                        if DIAG_SIG_DETAILS and reasons_counter["df_empty"] <= LOG_REJECTIONS_N:
                            print(f"[NOSIG] {sym} reason=df_empty", flush=True)
                        continue
                    try: df1m.index=df1m.index.tz_convert(MARKET_TZ)
                    except Exception: df1m.index=df1m.index.tz_localize("UTC").tz_convert(MARKET_TZ)

                    # Today volume gate
                    today_mask = df1m.index.date == df1m.index[-1].date()
                    todays_vol = float(df1m.loc[today_mask, "volume"].sum()) if today_mask.any() else 0.0
                    if todays_vol < SCANNER_MIN_TODAY_VOL:
                        reasons_counter["vol_gate"] += 1
                        continue

                    # quick price gate
                    last_px=float(df1m["close"].iloc[-1])
                    if last_px < MIN_PRICE:
                        reasons_counter["min_price"] += 1
                        continue

                    for tf in TF_MIN_LIST:
                        sig = compute_signal("ml_pattern", sym, tf, df1m=df1m, reasons=reasons_counter)
                        if not sig: continue

                        # sentiment gate (optional directional filter)
                        if SENTIMENT_ONLY_GATE:
                            if (sentiment=="bull" and sig["action"]!="buy") or (sentiment=="bear" and sig["action"]!="sell"):
                                reasons_counter["filtered_by_sentiment"] += 1
                                continue

                        # dedupe
                        k = _dedupe_key("ml_pattern", sym, tf, sig["action"], sig.get("barTime",""))
                        if k in _sent_keys:
                            reasons_counter["dedupe"] += 1
                            continue
                        _sent_keys.add(k)

                        score = float(sig.get("score", sig.get("confidence", 0.0)))
                        candidates.append((score, sym, tf, sig))

                except Exception as e:
                    reasons_counter["fetch_or_loop_err"] += 1
                    if DIAG_SIG_DETAILS and reasons_counter["fetch_or_loop_err"] <= LOG_REJECTIONS_N:
                        print(f"[NOSIG] {sym} reason=fetch_or_loop_err msg={str(e)[:120]}", flush=True)
                    continue

            _log_candidate_summary(candidates, reasons_counter)

            # -------- DISPATCH (ranked) --------
            if allow_new_entries and candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                sent_ct=0
                for score, sym, tf, sig in candidates:
                    if MAX_ENTRIES_PER_CYCLE>0 and sent_ct>=MAX_ENTRIES_PER_CYCLE: break
                    open_positions=sum(1 for lst in OPEN_TRADES.values() for t in lst if t.is_open)
                    if open_positions >= MAX_CONCURRENT_POSITIONS:
                        print(f"[LIMIT] Max concurrent reached mid-loop.", flush=True); break
                    payload = build_payload(sym, sig)
                    ok, info = send_to_traderspost(payload)
                    if ok:
                        sent_ct+=1
                        _record_open_trade("ml_pattern", sym, tf, sig)
                    print(f"[ORDER] {sym} {tf}m {sig['action']} qty={sig['quantity']} conf={score:.3f} ok={ok} info={info}", flush=True)

        except Exception as e:
            print("[LOOP ERROR]", e, traceback.format_exc(), flush=True)

        elapsed=time.time()-loop_start
        time.sleep(max(1, POLL_SECONDS - int(elapsed)))

if __name__ == "__main__":
    main()
