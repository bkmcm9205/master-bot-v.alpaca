# Ranked_ML.py — ML scanner with confidence ranking, two-sided trades, guards, and diagnostics
# v2.3: Proper v3 paging (follow next_url), exchange filter in-pager, optional shuffle,
#       boot-time universe histogram, retains EOD flatten + neutral sentiment allows both sides.

import os, time, json, math, hashlib, requests, traceback, random
from collections import defaultdict, deque, Counter
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

APP_TAG = "Ranked_ML v2.3 (universe paging fix + exch filter + shuffle)"
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
SCANNER_CONF_THRESHOLD = float(os.getenv("SCANNER_CONF_THRESHOLD", "0.80"))
SCANNER_R_MULTIPLE     = float(os.getenv("SCANNER_R_MULTIPLE", "3.0"))
MAX_ENTRIES_PER_CYCLE  = int(os.getenv("MAX_ENTRIES_PER_CYCLE", "0"))  # 0/neg => unlimited
ALLOW_ENTRIES          = _env_bool("ALLOW_ENTRIES","1")                # gates DISPATCH only

# Universe
SCANNER_SYMBOLS       = os.getenv("SCANNER_SYMBOLS", "").strip()
SCANNER_MAX_PAGES     = int(os.getenv("SCANNER_MAX_PAGES", "3"))   # each page ~1000
SCANNER_MIN_TODAY_VOL = int(os.getenv("SCANNER_MIN_TODAY_VOL", "100000"))
TF_MIN_LIST           = [int(x) for x in os.getenv("TF_MIN_LIST", "1,2,3,5,10").split(",") if x.strip()]
UNIVERSE_SHUFFLE      = _env_bool("UNIVERSE_SHUFFLE", "1")

# Symbol filters
MIN_PRICE = float(os.getenv("MIN_PRICE", "5.0"))   # skip stocks below this
ALLOWED_EXCHANGES = [s.strip().upper() for s in os.getenv(
    "ALLOWED_EXCHANGES", "XNYS,XNAS,NYSE,NASDAQ,NASD").split(",") if s.strip()]

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
START_EQUITY         = float(os.getenv("START_EQUITY", "100000"))
DAILY_TP_PCT         = float(os.getenv("DAILY_TP_PCT", "0.25"))
DAILY_DD_PCT         = float(os.getenv("DAILY_DD_PCT", "0.05"))
DAILY_FLATTEN_ON_HIT = _env_bool("DAILY_FLATTEN_ON_HIT","1")
DAILY_GUARD_ENABLED  = _env_bool("DAILY_GUARD_ENABLED","0")

# Replay / debug
DRY_RUN            = os.getenv("DRY_RUN","0") == "1"
REPLAY_ON_START    = os.getenv("REPLAY_ON_START","0") == "1"
REPLAY_SYMBOL      = os.getenv("REPLAY_SYMBOL","SPY")
REPLAY_TF          = int(os.getenv("REPLAY_TF","5"))
REPLAY_HOURS       = int(os.getenv("REPLAY_HOURS","24"))
REPLAY_SEND_ORDERS = _env_bool("REPLAY_SEND_ORDERS","0")

# Diagnostics
INTROSPECT         = _env_bool("INTROSPECT","0")
INTROSPECT_TOP_K   = int(os.getenv("INTROSPECT_TOP_K","10"))
LOG_REJECTIONS_N   = int(os.getenv("LOG_REJECTIONS_N","0"))  # 0=off
DIAG_ON            = _env_bool("DIAG_ON","0")
DIAG_SYMBOLS       = [s.strip().upper() for s in os.getenv("DIAG_SYMBOLS","SPY,QQQ,AAPL,NVDA").split(",") if s.strip()]
DIAG_SIG_DETAILS   = _env_bool("DIAG_SIG_DETAILS","0")
DIAG_MODEL_TRACE   = _env_bool("DIAG_MODEL_TRACE","0")

print("[CONFIG] "
      f"ALLOW_ENTRIES={ALLOW_ENTRIES} DRY_RUN={DRY_RUN} BYPASS_SESSION={BYPASS_SESSION} "
      f"ALLOW_PREMARKET={ALLOW_PREMARKET} ALLOW_AFTERHOURS={ALLOW_AFTERHOURS} "
      f"SHORTS_ENABLED={SHORTS_ENABLED} SENTIMENT_ONLY_GATE={SENTIMENT_ONLY_GATE} "
      f"CONF_THR={SCANNER_CONF_THRESHOLD} R_MULT={SCANNER_R_MULTIPLE} "
      f"MIN_TODAY_VOL={SCANNER_MIN_TODAY_VOL} TFs={TF_MIN_LIST} "
      f"MIN_PRICE={MIN_PRICE} ALLOWED_EXCHANGES={ALLOWED_EXCHANGES} "
      f"INTROSPECT={INTROSPECT} DIAG_ON={DIAG_ON} DIAG_SIG_DETAILS={DIAG_SIG_DETAILS} DIAG_MODEL_TRACE={DIAG_MODEL_TRACE} "
      f"DIAG_SYMBOLS={DIAG_SYMBOLS} LOG_REJ_N={LOG_REJECTIONS_N}",
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

# -------- UTILS --------
def _now_et(): return datetime.now(timezone.utc).astimezone(ZoneInfo(MARKET_TZ))
def _is_rth(ts): h,m=ts.hour,ts.minute; return ((h>9) or (h==9 and m>=30)) and (h<16)
def _in_session(ts):
    if BYPASS_SESSION: return True
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

def _debug_universe_print(symbols: list, max_show: int = 40):
    first = [s[0] for s in symbols if s]
    hist = Counter(first)
    top = ", ".join([f"{k}:{v}" for k, v in sorted(hist.items())])
    sample = ", ".join(symbols[:max_show])
    print(f"[UNIVERSE-DEBUG] first-letter hist => {top}", flush=True)
    print(f"[UNIVERSE-DEBUG] first {max_show} => {sample}", flush=True)

# -------- POLYGON --------
def fetch_polygon_1m(symbol: str, lookback_minutes: int = 2400) -> pd.DataFrame:
    if not POLYGON_API_KEY: return pd.DataFrame()
    end = datetime.now(timezone.utc); start = end - timedelta(minutes=lookback_minutes)
    url = (f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/"
           f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
           f"?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}")
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200: return pd.DataFrame()
        js = r.json()
        rows = js.get("results", [])
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows)
        for col in ("o","h","l","c","v","t"):
            if col not in df.columns: return pd.DataFrame()
        df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        df = df.set_index("ts").sortindex().sort_index()
        df.index = df.index.tz_convert(MARKET_TZ)
        df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
        return df[["open","high","low","close","volume"]]
    except Exception:
        return pd.DataFrame()

def get_universe_symbols() -> list:
    """
    Universe: env override OR Polygon reference tickers (active US stocks), multi-page.
    - Correctly follows v3 `next_url` directly (already contains cursor).
    - Filters by exchange (MIC) against ALLOWED_EXCHANGES.
    - Optional shuffle to avoid alpha bias (UNIVERSE_SHUFFLE=1).
    """
    if SCANNER_SYMBOLS:
        syms = [s.strip().upper() for s in SCANNER_SYMBOLS.split(",") if s.strip()]
        return [s for s in syms if s.isalnum()]

    if not POLYGON_API_KEY: return []

    base_url = "https://api.polygon.io/v3/reference/tickers?market=stocks&active=true&limit=1000&apiKey=" + POLYGON_API_KEY
    url = base_url
    out, pages = [], 0

    while pages < SCANNER_MAX_PAGES and url:
        try:
            r = requests.get(url, timeout=20)
            if r.status_code != 200:
                print(f"[UNIVERSE] HTTP {r.status_code}: {r.text[:200]}", flush=True)
                break
            j = r.json()
            results = j.get("results", [])
            for x in results:
                sym = (x.get("ticker") or "").upper()
                exch = (x.get("primary_exchange") or x.get("primary_exchange_iex") or "").upper()
                if sym and sym.isalnum() and (not ALLOWED_EXCHANGES or exch in ALLOWED_EXCHANGES):
                    out.append(sym)
            next_url = j.get("next_url")
            if not next_url:
                break
            url = next_url  # absolute URL from Polygon; already has cursor/apiKey
            pages += 1
        except Exception as e:
            print(f"[UNIVERSE] paging error: {e}", flush=True)
            break

    if UNIVERSE_SHUFFLE:
        random.shuffle(out)

    return out

# -------- SENTIMENT --------
def compute_sentiment():
    def _is_rth_tz(ts): h,m=ts.hour,ts.minute; return ((h>9) or (h==9 and m>=30)) and (h<16)
    mode=os.getenv("SENTIMENT_MODE","rth_only").lower()
    symbols=[s.strip() for s in os.getenv("SENTS_SYMBOLS","SPY,QQQ").split(",") if s.strip()]
    look_min=int(os.getenv("SENTS_LOOKBACK_MIN","30")); tf_min=int(os.getenv("SENTS_TF","1"))
    gap_bp=float(os.getenv("SENTS_GAP_THRESHOLD_BP","20"))/10000.0
    now_et=_now_et(); is_rth=_is_rth_tz(now_et)

    def _intraday_momentum(sym):
        df=fetch_polygon_1m(sym, lookback_minutes=max(look_min, tf_min*look_min))
        if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex): return 0.0
        try: df.index=df.index.tz_convert("America/New_York")
        except: df.index=df.index.tz_localize("UTC").tz_convert("America/New_York")
        bars=_resample(df, tf_min)
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
    dd = ec[-1] - max(ec)
    p["max_dd"] = min(p["max_dd"], dd)

class LiveTrade:
    def __init__(self, combo, symbol, tf_min, side, entry, tp, sl, qty, entry_time):
        self.combo = combo; self.symbol = symbol; self.tf_min = int(tf_min)
        self.side = side; self.entry = float(entry); self.tp = float(tp); self.sl = float(sl)
        self.qty = int(qty); self.entry_time = entry_time
        self.is_open=True; self.exit=None; self.exit_time=None; self.reason=None

def _record_open_trade(strat_name: str, symbol: str, tf_min: int, sig: dict):
    combo=_combo_key(strat_name, symbol, tf_min); _perf_init(combo)
    tp=sig.get("tp_abs", sig.get("takeProfit")); sl=sig.get("sl_abs", sig.get("stopLoss"))
    t=LiveTrade(combo, symbol, tf_min, sig["action"], sig["entry"], float(tp), float(sl),
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

# -------- ML (two-sided) --------
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
    if len(feats) < max(2, look+1): return 0.5
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

        # Single-class? → fallback but still usable
        if getattr(y_train, "nunique", None) and y_train.nunique() < 2:
            proba_up = _fallback_proba_from_momentum(feats)
            ts = feats.index[-1]
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
                if DIAG_MODEL_TRACE:
                    print("[TRACE]", traceback.format_exc(), flush=True)
                return None, None, "model_error"
            proba_up = _fallback_proba_from_momentum(feats)
            ts = feats.index[-1]
            if DIAG_SIG_DETAILS:
                print(f"[FALLBACK] {symbol} {tf_min}m reason=fallback_model_error proba_up={proba_up:.3f} msg={str(e)[:120]}", flush=True)
            if DIAG_MODEL_TRACE:
                print("[TRACE]", traceback.format_exc(), flush=True)
            if not _in_session(ts): return proba_up, ts, "not_in_session"
            return proba_up, ts, "fallback_model_error"
        except Exception as e2:
            if DIAG_SIG_DETAILS:
                print(f"[NOSIG] {symbol} {tf_min}m reason=model_error_fallback_failed msg={str(e2)[:120]}", flush=True)
            if DIAG_MODEL_TRACE:
                print("[TRACE-FALLBACK]", traceback.format_exc(), flush=True)
            return None, None, "model_error"

def signal_ml_pattern(symbol: str, df1m: pd.DataFrame, tf_min: int, r_multiple=None):
    r_multiple=float(r_multiple if r_multiple is not None else SCANNER_R_MULTIPLE)
    proba_up, ts, reason=ml_score(symbol, df1m, tf_min)

    if proba_up is None:
        if DIAG_SIG_DETAILS:
            print(f"[NOSIG] {symbol} {tf_min}m reason={reason}", flush=True)
        return None
    if reason == "not_in_session":
        if DIAG_SIG_DETAILS:
            print(f"[NOSIG] {symbol} {tf_min}m reason=not_in_session proba_up={proba_up:.3f}", flush=True)
        return None

    bars=_resample(df1m, tf_min)
    if bars is None or bars.empty:
        if DIAG_SIG_DETAILS:
            print(f"[NOSIG] {symbol} {tf_min}m reason=bars_empty_after_score", flush=True)
        return None

    price=float(bars["close"].iloc[-1])

    # Price floor
    if price < MIN_PRICE:
        if DIAG_SIG_DETAILS:
            print(f"[NOSIG] {symbol} {tf_min}m reason=price_floor {price:.2f}<{MIN_PRICE}", flush=True)
        return None

    if proba_up >= 0.5:
        side="buy"; score=proba_up; sl=price*0.99; tp=price*(1+0.01*r_multiple)
    else:
        if not SHORTS_ENABLED:
            if DIAG_SIG_DETAILS:
                print(f"[NOSIG] {symbol} {tf_min}m reason=shorts_disabled proba_up={proba_up:.3f}", flush=True)
            return None
        side="sell"; score=1.0-proba_up; sl=price*1.01; tp=price*(1-0.01*r_multiple)

    qty=_position_qty(price, sl)
    if qty<=0:
        if DIAG_SIG_DETAILS:
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

# -------- ROUTER / DIAG --------
def _log_candidate_summary(cands):
    if not cands: print("[SCAN] candidates=0", flush=True); return
    top = sorted(cands, key=lambda x: x[0], reverse=True)[:5]
    view = ", ".join([f"{sym} {tf}m {score:.3f}" for score, sym, tf, _ in top])
    print(f"[SCAN] candidates={len(cands)} | top5: {view}", flush=True)

def compute_signal(strategy_name, symbol, tf_minutes, df1m=None):
    if df1m is None or getattr(df1m, "empty", True):
        df1m = fetch_polygon_1m(symbol, lookback_minutes=max(240, tf_minutes*240))
        if df1m is None or df1m.empty: 
            if DIAG_SIG_DETAILS: print(f"[NOSIG] {symbol} {tf_minutes}m reason=df_empty", flush=True)
            return None

    if not isinstance(df1m.index, pd.DatetimeIndex):
        try: df1m.index = pd.to_datetime(df1m.index, utc=True)
        except Exception:
            if DIAG_SIG_DETAILS: print(f"[NOSIG] {symbol} {tf_minutes}m reason=index_not_datetime", flush=True)
            return None
    try: df1m.index = df1m.index.tz_convert(MARKET_TZ)
    except Exception: df1m.index = df1m.index.tz_localize("UTC").tz_convert(MARKET_TZ)

    # today's volume gate
    today_mask = df1m.index.date == df1m.index[-1].date()
    todays_vol = float(df1m.loc[today_mask, "volume"].sum()) if today_mask.any() else 0.0
    if todays_vol < SCANNER_MIN_TODAY_VOL:
        if DIAG_SIG_DETAILS: print(f"[NOSIG] {symbol} {tf_minutes}m reason=vol_gate vol={int(todays_vol)}<min={SCANNER_MIN_TODAY_VOL}", flush=True)
        return None

    # exchange filter already applied in universe fetch; price floor checked in signal_ml_pattern
    if strategy_name=="ml_pattern":
        sig = signal_ml_pattern(symbol, df1m, tf_minutes, SCANNER_R_MULTIPLE)
        return sig
    return None

# -------- MAIN --------
def main():
    print("Scanner starting…", flush=True)
    symbols = get_universe_symbols()
    print(f"[UNIVERSE] symbols={len(symbols)}  TFs={TF_MIN_LIST}  vol_gate={SCANNER_MIN_TODAY_VOL}", flush=True)
    _debug_universe_print(symbols)

    if REPLAY_ON_START:
        try:
            df1m=fetch_polygon_1m(REPLAY_SYMBOL, lookback_minutes=REPLAY_HOURS*60)
            if df1m is not None and not df1m.empty:
                bars=_resample(df1m, REPLAY_TF); hits=0; last_key=None
                for i in range(len(bars)):
                    df_slice=df1m.loc[:bars.index[i]]
                    sig=signal_ml_pattern(REPLAY_SYMBOL, df_slice, REPLAY_TF, SCANNER_R_MULTIPLE)
                    if sig:
                        if REPLAY_SEND_ORDERS:
                            payload=build_payload(REPLAY_SYMBOL, sig); print("[REPLAY-ORDER]", json.dumps(payload)[:200], flush=True)
                        k=(sig.get("barTime",""), sig.get("action",""))
                        if k!=last_key: hits+=1; last_key=k
                print(f"[REPLAY] ml_pattern {REPLAY_SYMBOL} {REPLAY_TF}m -> {hits} signals in {REPLAY_HOURS}h.", flush=True)
        except Exception as e:
            print("[REPLAY ERROR]", e, flush=True)

    while True:
        loop_start=time.time()
        try:
            print("Tick…", flush=True)

            sentiment = compute_sentiment() if SENTIMENT_ONLY_GATE else "neutral"
            print(f"[SENTIMENT] {sentiment}", flush=True)

            reset_daily_guard_if_new_day()
            if DAILY_GUARD_ENABLED: check_daily_guard_and_maybe_halt()
            allow_new_entries = (not (DAILY_GUARD_ENABLED and HALT_TRADING)) and ALLOW_ENTRIES

            # --- Close phase: update exits on latest bars for any open trades
            touched = list(OPEN_TRADES.keys())
            for (sym, tf) in touched:
                try:
                    df = fetch_polygon_1m(sym, lookback_minutes=max(60, tf * 12))
                    bars = _resample(df, tf)
                    if bars is not None and not bars.empty:
                        row = bars.iloc[-1]; ts  = bars.index[-1]
                        _maybe_close_on_bar(sym, tf, ts, float(row["high"]), float(row["low"]), float(row["close"]))
                except Exception as e:
                    print(f"[CLOSE-PHASE ERROR] {sym} {tf}m: {e}", flush=True)

            # EOD window flatten (16:00–16:10 ET)
            now_et=_now_et()
            if now_et.hour==16 and now_et.minute<10:
                print("[EOD] Auto-flatten window.", flush=True)
                flatten_all_open_positions()

            candidates=[]  # (score,symbol,tf,sig)

            # If entries are halted, skip scanning for NEW signals
            if not allow_new_entries:
                time.sleep(POLL_SECONDS); continue

            # Max concurrent positions guard
            open_positions = sum(1 for lst in OPEN_TRADES.values() for t in lst if t.is_open)
            if open_positions >= MAX_CONCURRENT_POSITIONS:
                print(f"[LIMIT] Max concurrent positions hit: {open_positions}/{MAX_CONCURRENT_POSITIONS}", flush=True)
                time.sleep(POLL_SECONDS); continue

            for sym in symbols:
                df1m=fetch_polygon_1m(sym, lookback_minutes=max(240, max(TF_MIN_LIST)*240))
                if df1m is None or df1m.empty or not isinstance(df1m.index, pd.DatetimeIndex):
                    continue
                try: df1m.index=df1m.index.tz_convert(MARKET_TZ)
                except Exception: df1m.index=df1m.index.tz_localize("UTC").tz_convert(MARKET_TZ)

                today_mask = df1m.index.date == df1m.index[-1].date()
                todays_vol = float(df1m.loc[today_mask, "volume"].sum()) if today_mask.any() else 0.0
                if todays_vol < SCANNER_MIN_TODAY_VOL:
                    continue

                for tf in TF_MIN_LIST:
                    sig = compute_signal("ml_pattern", sym, tf, df1m=df1m)
                    if not sig:
                        continue

                    # sentiment gate (neutral allows both sides)
                    if SENTIMENT_ONLY_GATE:
                        if (sentiment=="bull" and sig["action"]!="buy") or (sentiment=="bear" and sig["action"]!="sell"):
                            continue

                    # dedupe
                    k = _dedupe_key("ml_pattern", sym, tf, sig["action"], sig.get("barTime",""))
                    if k in _sent_keys: continue
                    _sent_keys.add(k)

                    score = float(sig.get("score", sig.get("confidence", 0.0)))
                    if score < SCANNER_CONF_THRESHOLD: continue

                    candidates.append((score, sym, tf, sig))

            _log_candidate_summary(candidates)

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
                        # ledger open
                        _record_open_trade("ml_pattern", sym, tf, sig)
                        sent_ct+=1
                    print(f"[ORDER] {sym} {tf}m {sig['action']} qty={sig['quantity']} conf={score:.3f} ok={ok} info={info}", flush=True)

        except Exception as e:
            print("[LOOP ERROR]", e, traceback.format_exc(), flush=True)

        elapsed=time.time()-loop_start
        time.sleep(max(1, POLL_SECONDS - int(elapsed)))

if __name__ == "__main__":
    main()
