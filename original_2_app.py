# app.py — dynamic market scanner (PLUMBING SWAP: Polygon/TP -> Alpaca)
# Requires: pandas, numpy, requests, pandas_ta, scikit-learn
# NOTE: Models/ML logic unchanged. Only data + broker I/O rewired to Alpaca.

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
    send_to_broker,
    close_all_positions,
    list_positions,
    get_account_equity,
    cancel_all_orders,
)

# ==============================
# ENV / CONFIG (keep names stable)
# ==============================
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "10"))
DRY_RUN      = os.getenv("DRY_RUN", "0").lower() in ("1","true","yes")
PAPER_MODE   = os.getenv("PAPER_MODE", "true").lower() != "false"
SCANNER_DEBUG= os.getenv("SCANNER_DEBUG", "0").lower() in ("1","true","yes")

# Timeframes list (comma-separated)
TF_MIN_LIST  = [int(x) for x in os.getenv("TF_MIN_LIST", "1,2,3,5,10").split(",") if x.strip()]

# Universe paging/size (legacy knobs; kept for compatibility)
MAX_UNIVERSE_PAGES = int(os.getenv("MAX_UNIVERSE_PAGES", "3"))
SCAN_BATCH_SIZE    = int(os.getenv("SCAN_BATCH_SIZE", "150"))

# Liquidity filter (legacy; we don’t prefilter via grouped-daily under Alpaca)
SCANNER_MIN_AVG_VOL = int(os.getenv("SCANNER_MIN_AVG_VOL", "0"))

# Price gate
MIN_PRICE = float(os.getenv("MIN_PRICE", "3.0"))

# Risk / sizing
EQUITY_USD  = float(os.getenv("EQUITY_USD",  "100000"))
RISK_PCT    = float(os.getenv("RISK_PCT",    "0.01"))
MAX_POS_PCT = float(os.getenv("MAX_POS_PCT", "0.10"))
MIN_QTY     = int(os.getenv("MIN_QTY", "1"))
ROUND_LOT   = int(os.getenv("ROUND_LOT","1"))

# Model thresholds
CONF_THR    = float(os.getenv("CONF_THR", "0.70"))
R_MULT      = float(os.getenv("R_MULT", "1.50"))

# Market/session
MARKET_TZ   = os.getenv("MARKET_TZ", "America/New_York")
SCANNER_MARKET_HOURS_ONLY = os.getenv("SCANNER_MARKET_HOURS_ONLY","1").lower() in ("1","true","yes")
ALLOW_PREMARKET  = os.getenv("ALLOW_PREMARKET","0").lower() in ("1","true","yes")
ALLOW_AFTERHOURS = os.getenv("ALLOW_AFTERHOURS","0").lower() in ("1","true","yes")

# Daily portfolio guard (uses broker equity baseline)
START_EQUITY         = float(os.getenv("START_EQUITY", "100000"))
DAILY_TP_PCT         = float(os.getenv("DAILY_TP_PCT", "0.10"))     # +10%
DAILY_DD_PCT         = float(os.getenv("DAILY_DD_PCT", "0.05"))     # -5%
DAILY_FLATTEN_ON_HIT = os.getenv("DAILY_FLATTEN_ON_HIT","1").lower() in ("1","true","yes")
DAILY_GUARD_ENABLED  = os.getenv("DAILY_GUARD_ENABLED","1").lower() in ("1","true","yes")

# --- Broker-equity guard (optional; pulls actual Alpaca account equity)
USE_BROKER_EQUITY_GUARD   = os.getenv("USE_BROKER_EQUITY_GUARD","0").lower() in ("1","true","yes")
SESSION_BASELINE_AT_0930  = os.getenv("SESSION_BASELINE_AT_0930","1").lower() in ("1","true","yes")
TRAIL_GUARD_ENABLED       = os.getenv("TRAIL_GUARD_ENABLED","1").lower() in ("1","true","yes")
TRAIL_DD_PCT              = float(os.getenv("TRAIL_DD_PCT","0.05"))

# Render boot info (for logs)
RENDER_GIT_COMMIT = os.getenv("RENDER_GIT_COMMIT", "unknown")[:12]
RENDER_GIT_BRANCH = os.getenv("RENDER_GIT_BRANCH", os.getenv("BRANCH", "unknown"))
RUN_ID            = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")

# ==============================
# STATE / LEDGERS
# ==============================
COUNTS        = defaultdict(int)
COMBO_COUNTS  = defaultdict(int)
PERF          = {}                   # combo -> realized stats (kept for parity)
OPEN_TRADES   = defaultdict(list)    # (symbol, tf) -> [LiveTrade]
_sent_keys    = set()                # de-dupe across loops
_order_times  = deque()              # throttle timestamps (epoch seconds)
LAST_PRICE    = {}                   # symbol -> last px (for MTM)

DAY_STAMP     = datetime.now().astimezone().strftime("%Y-%m-%d")
HALT_TRADING  = False                # local guard halts entries

SESSION_BASELINE_SET  = False
SESSION_START_EQUITY  = None
EQUITY_HIGH_WATER     = None

_round_robin = 0                     # round-robin pointer for batch scanning

# ==============================
# DATA MODEL (kept)
# ==============================
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

# ==============================
# TIME / SESSION HELPERS
# ==============================
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

def _market_session_now():
    ts = _now_et()
    return _in_session(ts)

# ==============================
# MATH / SIZING (kept)
# ==============================
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

# ==============================
# UNIVERSE + DATA (Alpaca plumbing)
# ==============================
def build_universe() -> list[str]:
    manual = os.getenv("SCANNER_SYMBOLS", "").strip()
    if manual:
        syms = [s.strip().upper() for s in manual.split(",") if s.strip()]
        print(f"[BOOT] SCANNER_SYMBOLS override detected ({len(syms)} symbols).", flush=True)
        return syms
    syms = _alpaca_universe(limit=10000)  # adapters handles assets/v2 paging
    print(f"[UNIVERSE] fetched {len(syms)} tickers via Alpaca assets.", flush=True)
    if SCANNER_MIN_AVG_VOL:
        print("[UNIVERSE] NOTE: Polygon grouped-daily prefilter not applied under Alpaca.", flush=True)
    return syms

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
    # price gate
    try:
        last_px = float(df["close"].iloc[-1])
        if last_px < MIN_PRICE:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()
    return df[["open","high","low","close","volume"]].copy()

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

# ==============================
# PERF TRACK (kept)
# ==============================
def _combo_key(strategy: str, symbol: str, tf_min: int) -> str:
    return f"{strategy}|{symbol}|{int(tf_min)}"

def _perf_init(combo: str):
    if combo not in PERF:
        PERF[combo] = {"trades":0,"wins":0,"losses":0,"gross_profit":0.0,"gross_loss":0.0,"net_pnl":0.0,"max_dd":0.0,"equity_curve":[0.0]}

def _perf_update(combo: str, pnl: float):
    _perf_init(combo)
    p = PERF[combo]
    p["trades"] += 1
    if pnl > 0: p["wins"] += 1; p["gross_profit"] += pnl
    elif pnl < 0: p["losses"] += 1; p["gross_loss"] += pnl
    p["net_pnl"] += pnl
    ec = p["equity_curve"]; ec.append(ec[-1] + pnl)
    dd = min(0.0, ec[-1] - max(ec)); p["max_dd"] = min(p["max_dd"], dd)

def _record_open_trade(strat_name: str, symbol: str, tf_min: int, sig: dict):
    combo = _combo_key(strat_name, symbol, tf_min)
    _perf_init(combo)
    tp = sig.get("tp_abs", sig.get("takeProfit"))
    sl = sig.get("sl_abs", sig.get("stopLoss"))
    side_norm = "sell" if sig["action"] in ("sell","sell_short") else "buy"
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
        if not t.is_open: continue
        hit_tp = (high >= t.tp) if t.side == "buy" else (low <= t.tp)
        hit_sl = (low <= t.sl) if t.side == "buy" else (high >= t.sl)
        if hit_tp or hit_sl:
            t.is_open = False
            t.exit_time = ts.tz_convert("UTC").isoformat() if hasattr(ts, "tzinfo") else str(ts)
            t.exit  = t.tp if hit_tp else t.sl
            t.reason= "tp" if hit_tp else "sl"
            pnl = (t.exit - t.entry) * t.qty if t.side == "buy" else (t.entry - t.exit) * t.qty
            _perf_update(t.combo, pnl)
            print(f"[CLOSE] {t.combo} {t.reason.upper()} qty={t.qty} entry={t.entry:.2f} exit={t.exit:.2f} pnl={pnl:+.2f}", flush=True)

# ==============================
# DE-DUPE
# ==============================
def _dedupe_key(symbol: str, tf: int, action: str, bar_time: str) -> str:
    raw = f"{symbol}|{tf}|{action}|{bar_time}"
    import hashlib
    return hashlib.sha256(raw.encode()).hexdigest()

# ==============================
# ML STRATEGY (unchanged)
# ==============================
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
    from sklearn.ensemble import RandomForestClassifier
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
                      conf_threshold=CONF_THR, r_multiple=R_MULT):
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

    want_long = (proba_up is not None) and (proba_up >= conf_threshold)
    if not want_long:
        return None

    qty = _position_qty(price, long_sl)
    if qty <= 0:
        return None

    return {
        "action": "buy",
        "orderType": "market",
        "price": None,
        "takeProfit": long_tp,
        "stopLoss": long_sl,
        "barTime": ts.tz_convert("UTC").isoformat(),
        "entry": price,
        "quantity": int(qty),
        "meta": {"note": "ml_pattern", "proba_up": proba_up},
    }

# ==============================
# DAILY GUARD (broker equity baseline)
# ==============================
def ensure_session_baseline():
    global SESSION_BASELINE_SET, SESSION_START_EQUITY, EQUITY_HIGH_WATER
    if SESSION_BASELINE_SET:
        return
    try:
        eq = get_account_equity(default_equity=START_EQUITY)
    except TypeError:
        eq = START_EQUITY
    except Exception:
        eq = START_EQUITY
    try:
        eq = float(eq)
    except Exception:
        eq = START_EQUITY
    SESSION_START_EQUITY = eq
    EQUITY_HIGH_WATER = eq
    SESSION_BASELINE_SET = True
    print(f"[SESSION] Baseline set: session_start_equity={SESSION_START_EQUITY:.2f}", flush=True)

def check_daily_guards():
    global HALT_TRADING, EQUITY_HIGH_WATER
    if not DAILY_GUARD_ENABLED:
        return
    ensure_session_baseline()
    try:
        eq_now = get_account_equity(default_equity=SESSION_START_EQUITY or START_EQUITY)
    except TypeError:
        eq_now = SESSION_START_EQUITY or START_EQUITY
    except Exception:
        eq_now = SESSION_START_EQUITY or START_EQUITY
    try:
        eq_now = float(eq_now)
    except Exception:
        eq_now = SESSION_START_EQUITY or START_EQUITY

    if EQUITY_HIGH_WATER is None:
        EQUITY_HIGH_WATER = eq_now
    else:
        EQUITY_HIGH_WATER = max(EQUITY_HIGH_WATER, eq_now)

    up_lim = SESSION_START_EQUITY * (1.0 + DAILY_TP_PCT)
    dn_lim = SESSION_START_EQUITY * (1.0 - DAILY_DD_PCT)

    print(f"[DAILY-GUARD] eq={eq_now:.2f} start={SESSION_START_EQUITY:.2f} "
          f"targets +{DAILY_TP_PCT*100:.1f}%({up_lim:.2f}) / -{DAILY_DD_PCT*100:.1f}%({dn_lim:.2f})",
          flush=True)

    if HALT_TRADING:
        return

    hit_tp = eq_now >= up_lim
    hit_dd = eq_now <= dn_lim
    if hit_tp or hit_dd:
        HALT_TRADING = True
        reason = "Profit target" if hit_tp else "Drawdown limit"
        print(f"[DAILY-GUARD] ⛔ {reason} hit. Halting entries.", flush=True)
        if DAILY_FLATTEN_ON_HIT:
            ok, info = close_all_positions()
            print(f"[DAILY-GUARD] Flatten -> ok={ok} info={info}", flush=True)

def reset_daily_state_if_new_day():
    global DAY_STAMP, HALT_TRADING, SESSION_BASELINE_SET, EQUITY_HIGH_WATER
    today = datetime.now().astimezone().strftime("%Y-%m-%d")
    if today != DAY_STAMP:
        DAY_STAMP = today
        HALT_TRADING = False
        SESSION_BASELINE_SET = False
        EQUITY_HIGH_WATER = None
        print(f"[NEW DAY] State reset. START_EQUITY={START_EQUITY:.2f} DAY={DAY_STAMP}", flush=True)

# ==============================
# ROUTING & ORDER SEND
# ==============================
def handle_signal(strat_name: str, symbol: str, tf_min: int, sig: dict):
    combo_key = _combo_key(strat_name, symbol, tf_min)
    COUNTS["signals"] += 1
    COMBO_COUNTS[f"{combo_key}::signals"] += 1
    meta = sig.get("meta", {})
    meta["combo"] = combo_key
    meta["timeframe"] = f"{int(tf_min)}m"
    sig["meta"] = meta
    _record_open_trade(strat_name, symbol, tf_min, sig)

    ok, info = send_to_broker(symbol, sig, strategy_tag="ml_pattern")
    try:
        COMBO_COUNTS[f"{combo_key}::orders.{'ok' if ok else 'err'}"] += 1
        COUNTS["orders.ok" if ok else "orders.err"] += 1
    except Exception:
        pass
    print(f"[ORDER] {combo_key} -> action={sig.get('action')} qty={sig.get('quantity')} ok={ok} info={info}", flush=True)

# ==============================
# BATCHING
# ==============================
def _batched_symbols(universe: list):
    global _round_robin
    if not universe:
        return []
    N = len(universe)
    start = _round_robin % max(1, N)
    end = min(N, start + SCAN_BATCH_SIZE)
    batch = universe[start:end]
    _round_robin = 0 if end >= N else end
    return batch

# ==============================
# MAIN LOOP
# ==============================
def main():
    global HALT_TRADING

    # ---- Boot banner ----
    print(f"[BOOT] RUN_ID={RUN_ID} BRANCH={RENDER_GIT_BRANCH} COMMIT={RENDER_GIT_COMMIT} START_CMD=python app.py", flush=True)
    print(f"[BOOT] PAPER_MODE={'paper' if PAPER_MODE else 'live'} POLL_SECONDS={POLL_SECONDS} TFs={TF_MIN_LIST}", flush=True)
    print(f"[BOOT] CONF_THR={CONF_THR} R_MULT={R_MULT}", flush=True)
    up = START_EQUITY*(1+DAILY_TP_PCT); dn = START_EQUITY*(1-DAILY_DD_PCT)
    print(f"[BOOT] DAILY_GUARD_ENABLED={int(DAILY_GUARD_ENABLED)} UP={DAILY_TP_PCT:.0%} DOWN={DAILY_DD_PCT:.0%} "
          f"FLATTEN={int(DAILY_FLATTEN_ON_HIT)} START_EQUITY={START_EQUITY:.2f}", flush=True)
    print(f"[BOOT] BROKER_GUARD={int(USE_BROKER_EQUITY_GUARD)} BASELINE_AT_0930={int(SESSION_BASELINE_AT_0930)} TRAIL_DD={TRAIL_DD_PCT:.0%}", flush=True)

    from common.signal_bridge import probe_alpaca_auth
    probe_alpaca_auth()

    # Universe
    symbols = build_universe()
    print(f"[UNIVERSE] size={len(symbols)}  TFs={TF_MIN_LIST}  Batch={SCAN_BATCH_SIZE}", flush=True)

    while True:
        loop_start = time.time()
        try:
            reset_daily_state_if_new_day()

            # ---- Close phase: update prices, check TP/SL on open trades
            touched = set((k[0], k[1]) for k in OPEN_TRADES.keys())
            for (sym, tf) in touched:
                try:
                    df = fetch_bars_1m(sym, lookback_minutes=max(60, tf * 12))
                    if df is None or df.empty:
                        continue
                    try:
                        df.index = df.index.tz_convert(MARKET_TZ)
                    except Exception:
                        df.index = df.index.tz_localize("UTC").tz_convert(MARKET_TZ)
                    bars = _resample(df, tf)
                    if bars is not None and not bars.empty:
                        row = bars.iloc[-1]; ts  = bars.index[-1]
                        LAST_PRICE[sym] = float(row["close"])
                        _maybe_close_on_bar(sym, tf, ts, float(row["high"]), float(row["low"]), float(row["close"]))
                except Exception as e:
                    print(f"[CLOSE-PHASE ERROR] {sym} {tf}m: {e}", flush=True)

            # ---- Daily guard (broker equity)
            check_daily_guards()

            # ---- Hardened EOD manager (runs BEFORE deciding allow_entries)
            now_et = _now_et()

            def _cancel_all_open_orders_safely():
                try:
                    ok, info = cancel_all_orders()
                    print(f"[EOD] Cancel all open orders -> ok={ok} info={info}", flush=True)
                except Exception as e:
                    import traceback
                    print("[EOD] Cancel orders exception:", e, traceback.format_exc(), flush=True)

            def _flatten_until_flat(max_iters=5, sleep_s=2):
                for i in range(max_iters):
                    try:
                        pos = list_positions() or []
                        if not pos:
                            print(f"[EOD] Flat after pass {i}", flush=True)
                            return True
                        print(f"[EOD] Flatten pass {i+1}: positions={len(pos)}", flush=True)
                        ok, info = close_all_positions()
                        print(f"[EOD] Flatten call -> ok={ok} info={info}", flush=True)
                    except Exception as e:
                        import traceback
                        print("[EOD] Flatten exception:", e, traceback.format_exc(), flush=True)
                    time.sleep(sleep_s)
                pos = list_positions() or []
                print(f"[EOD] Final position check -> {len(pos)} open", flush=True)
                return len(pos) == 0

            # 1) Stop new entries before the close
            if now_et.hour == 15 and now_et.minute >= 45:
                if not HALT_TRADING:
                    HALT_TRADING = True
                    print("[EOD] HALT_TRADING enabled at 3:45 ET (no new entries).", flush=True)

            # 2) Pre-close consolidation & flatten window (3:50–4:00 ET)
            if now_et.hour == 15 and now_et.minute >= 50:
                print("[EOD] Pre-close window (3:50–4:00 ET): cancel orders + flatten until flat.", flush=True)
                _cancel_all_open_orders_safely()
                _flatten_until_flat()

            # 3) Safety net right after bell (4:00–4:03 ET)
            if now_et.hour == 16 and now_et.minute < 3:
                print("[EOD] Post-bell safety net (4:00–4:03 ET).", flush=True)
                _cancel_all_open_orders_safely()
                _flatten_until_flat()

            # ---- Decide if we may place new entries *after* EOD logic ran
            allow_entries = not HALT_TRADING
            if not allow_entries:
                time.sleep(POLL_SECONDS)
                continue

            # ---- Scan & signal phase
            if SCANNER_MARKET_HOURS_ONLY and not _market_session_now():
                if SCANNER_DEBUG:
                    print("[SCAN] Skipping — market session closed.", flush=True)
                time.sleep(POLL_SECONDS)
                continue

            # Round-robin batching to control API rate
            batch = _batched_symbols(symbols)
            if SCANNER_DEBUG:
                total = len(symbols)
                s = _round_robin if _round_robin else total
                start_idx = (s - len(batch)) if s else 0
                print(f"[SCAN] symbols {start_idx}:{start_idx+len(batch)} / {total}  (batch={len(batch)})", flush=True)

            for sym in batch:
                df1m = fetch_bars_1m(sym, lookback_minutes=max(240, max(TF_MIN_LIST)*240))
                if df1m is None or df1m.empty:
                    continue
                try:
                    df1m.index = df1m.index.tz_convert(MARKET_TZ)
                except Exception:
                    df1m.index = df1m.index.tz_localize("UTC").tz_convert(MARKET_TZ)

                for tf in TF_MIN_LIST:
                    sig = signal_ml_pattern(sym, df1m, tf)
                    if not sig:
                        continue
                    k = _dedupe_key(sym, tf, sig["action"], sig.get("barTime",""))
                    if k in _sent_keys:
                        continue
                    _sent_keys.add(k)

                    handle_signal("ml_pattern", sym, tf, sig)

        except Exception as e:
            import traceback
            print("[LOOP ERROR]", e, traceback.format_exc(), flush=True)

        elapsed = time.time() - loop_start
        time.sleep(max(1, POLL_SECONDS - int(elapsed)))

if __name__ == "__main__":
    main()
