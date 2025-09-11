# app.py — paper/live trading loop for your allow-listed (strategy, symbol, timeframe) combos
# Uses TradersPost webhook + Polygon 1m data (swap fetcher if you prefer your DB).

import os, time, json, hashlib, requests, math
import pandas as pd, numpy as np
import csv
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass

# ==============================
# ENV / CONFIG
# ==============================
TP_URL = os.getenv("TP_WEBHOOK_URL")            # TradersPost Strategy Webhook URL (Paper)
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")  # If you keep using Polygon here
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "10"))
PAPER_MODE = os.getenv("PAPER_MODE", "true").lower() != "false"  # default paper
STARTUP_TPSL_TEST = os.getenv("STARTUP_TPSL_TEST", "0").lower()  # "1"/"true"/"yes" to run test on boot
POLY_LIVE_TEST = os.getenv("POLY_LIVE_TEST", "0").lower()   # "1"/"true"/"yes" to run freshness check
POLY_TEST_SYMBOL = os.getenv("POLY_TEST_SYMBOL", "SPY")     # symbol to test freshness on
RUN_ID = datetime.now().astimezone().strftime("%Y-%m-%d")
COUNTS = defaultdict(int)        # e.g., COUNTS["orders.ok"] += 1
COMBO_COUNTS = defaultdict(int)  # keys like "poc|AAPL|5::signals"
# If both TP and SL are inside the same bar, which one do we assume hit first?
# "tp" (optimistic) or "sl" (conservative). You can override via env var.
TP_SL_SAME_BAR = os.getenv("TP_SL_SAME_BAR", "tp").lower()  # "tp" | "sl"
# Open trades keyed by (symbol, tf_min)
OPEN_TRADES = defaultdict(list)  # (symbol, tf) -> list[LiveTrade]

@dataclass
class LiveTrade:
    combo: str            # e.g., "poc|AAPL|5"
    symbol: str
    tf_min: int
    side: str             # "buy" or "sell"
    entry: float
    tp: float
    sl: float
    qty: int
    entry_time: str       # ISO string
    is_open: bool = True
    exit: float = None
    exit_time: str = None
    reason: str = None    # "tp" | "sl"  

# Per-combo performance
PERF = {
    # combo -> dict with rolling stats
    # "poc|AAPL|5": {
    #   "trades": 0, "wins": 0, "losses": 0,
    #   "gross_profit": 0.0, "gross_loss": 0.0,
    #   "net_pnl": 0.0, "max_dd": 0.0,
    #   "equity_curve": [0.0, ...]  # cumulative
    # }
}

# ===== Diagnostics =====
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"               # don't POST; just log payload
SEND_TEST_ORDER = os.getenv("SEND_TEST_ORDER", "0") == "1"  # fire one paper test order on boot
REPLAY_ON_START = os.getenv("REPLAY_ON_START", "0") == "1"  # run a quick historical replay then continue
REPLAY_STRAT = os.getenv("REPLAY_STRAT", "")              # e.g. "ict_bos_fvg"
REPLAY_SYMBOL = os.getenv("REPLAY_SYMBOL", "")            # e.g. "META"
REPLAY_TF = int(os.getenv("REPLAY_TF", "5"))              # minutes
REPLAY_HOURS = int(os.getenv("REPLAY_HOURS", "24"))       # how far back to replay
DEBUG_COMBO = os.getenv("DEBUG_COMBO", "")                # "strategy,symbol,tf" (e.g. "poc,AVGO,1")

# ---- Global position sizing (bit-for-bit with your Colab) ----
EQUITY_USD  = float(os.getenv("EQUITY_USD",  "100000"))  # $100k
RISK_PCT    = float(os.getenv("RISK_PCT",    "0.01"))    # 1% as 0.01 (same as your notebook)
MAX_POS_PCT = float(os.getenv("MAX_POS_PCT", "0.10"))    # 10% max notional per position
MIN_QTY     = int(os.getenv("MIN_QTY", "1"))
ROUND_LOT   = int(os.getenv("ROUND_LOT","1"))

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
# Allow-list (your 15 combos) — NO quantity here
# ==============================
# (strategy_name, symbol, timeframe_minutes)
COMBOS = [
    ("ict_bos_fvg_ob", "META", 3),
    ("ict_bos_fvg",    "IONQ", 1),
    ("ict_bos_fvg",    "AVGO", 2),
    ("ict_bos_fvg",    "MU",   2),
    ("ict_bos_fvg_ob", "META", 5),
    ("ict_bos_fvg",    "QQQ",  2),
    ("poc_pinescript", "CAN",  3),
    ("poc_pinescript", "WOLF", 3),
    ("ict_bos_fvg_ob", "QQQ",  1),
    ("ict_bos_fvg",    "SPY",  1),
    ("poc",            "AVGO", 1),
    ("ict_bos_fvg",    "META", 2),
    ("ict_bos_fvg",    "GOOGL",1),
    ("ict_bos_fvg",    "AVGO", 15),
    ("ict_bos_fvg",    "WOLF", 3),
]

# ==============================
# Utilities
# ==============================
_sent = set()  # dedupe: (strategy, symbol, tf, action, barTime)

def _dedupe_key(strategy: str, symbol: str, tf: int, action: str, bar_time: str) -> str:
    raw = f"{strategy}|{symbol}|{tf}|{action}|{bar_time}"
    return hashlib.sha256(raw.encode()).hexdigest()

def send_to_traderspost(payload: dict):
    try:
        if DRY_RUN:
            print(f"[DRY-RUN] Would POST to TradersPost: {json.dumps(payload)[:500]}", flush=True)
            return True, "dry-run"
        if not TP_URL:
            print("[ERROR] TP_WEBHOOK_URL is empty or missing.", flush=True)
            return False, "no-webhook-url"

        payload.setdefault("meta", {})
        payload["meta"]["environment"] = "paper" if PAPER_MODE else "live"
        payload["meta"]["sentAt"] = datetime.now(timezone.utc).isoformat()

        print(f"[POST] TradersPost URL present. Sending payload...", flush=True)
        r = requests.post(TP_URL, json=payload, timeout=12)
        ok = 200 <= r.status_code < 300
        info = f"{r.status_code} {r.text[:300]}"
        if not ok:
            print(f"[POST ERROR] {info}", flush=True)
        return ok, info
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[POST EXCEPTION] {e}\n{tb}", flush=True)
        return False, f"exception: {e}"
    try:
        combo = (payload.get("meta") or {}).get("combo", "UNKNOWN")
        if ok:
            COUNTS["orders.ok"] += 1
            COMBO_COUNTS[f"{combo}::orders.ok"] += 1
        else:
            COUNTS["orders.err"] += 1
            COMBO_COUNTS[f"{combo}::orders.err"] += 1
    except Exception:
        pass

def build_payload(symbol: str, sig: dict):
    """
    Build TradersPost payload with ABSOLUTE TP/SL formatted correctly.
    """
    payload = {
        "ticker": symbol,
        "action": sig["action"],          # "buy" or "sell"
        "orderType": sig.get("orderType", "market"),
        "quantity": int(sig["quantity"]),
        "meta": sig.get("meta", {})
    }

    if "meta" not in payload:
        payload["meta"] = {}

    payload["meta"].update({
        "runId": RUN_ID,
        "sentAt": datetime.now(timezone.utc).isoformat()
    })

    combo = (sig.get("meta") or {}).get("combo")
    if combo:
        payload["meta"]["combo"] = combo  # e.g., "poc|AAPL|5"

    tp_abs = sig.get("tp_abs")
    sl_abs = sig.get("sl_abs")

    if tp_abs is not None:
        payload["takeProfit"] = {"limitPrice": float(round(tp_abs, 2))}

    if sl_abs is not None:
        payload["stopLoss"] = {"type": "stop", "stopPrice": float(round(sl_abs, 2))}

    return payload

# ==============================
# Data fetch — Polygon 1m (replace with your DB if you like)
# ==============================
def fetch_polygon_1m(symbol: str, lookback_minutes: int = 2_400) -> pd.DataFrame:
    """Fetch recent 1m bars; returns tz-aware America/New_York index with o/h/l/c/volume."""
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
    df.index = df.index.tz_convert("America/New_York")
    df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
    return df[["open","high","low","close","volume"]]

# ==============================
# Shared helpers
# ==============================
def _is_rth(ts):
    h, m = ts.hour, ts.minute
    return ((h > 9) or (h == 9 and m >= 30)) and (h < 16)

def _resample(df1m: pd.DataFrame, tf_min: int) -> pd.DataFrame:
    rule = f"{int(tf_min)}min"
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    return df1m.resample(rule, origin="start_day", label="right").agg(agg).dropna()

def handle_signal(strat_name: str, symbol: str, tf_min: int, sig: dict):
    """
    Increments counters, attaches combo meta, builds payload and sends.
    """
    # Build a stable combo key like "poc|AAPL|5"
    combo_key = f"{strat_name}|{symbol}|{int(tf_min)}"

    # Count the signal
    COUNTS["signals"] += 1
    COMBO_COUNTS[f"{combo_key}::signals"] += 1

    # Attach combo and timeframe to meta so it flows into payload
    meta = sig.get("meta", {})
    meta["combo"] = combo_key
    meta["timeframe"] = f"{int(tf_min)}m"
    sig["meta"] = meta

    # Send

    _record_open_trade(strat_name, symbol, tf_min, sig)
  
    payload = build_payload(symbol, sig)
    ok, info = send_to_traderspost(payload)

    # Count accepted / rejected (safety if you didn't already in send_to_traderspost)
    try:
        if ok:
            COUNTS["orders.ok"] += 1
            COMBO_COUNTS[f"{combo_key}::orders.ok"] += 1
        else:
            COUNTS["orders.err"] += 1
            COMBO_COUNTS[f"{combo_key}::orders.err"] += 1
    except Exception:
        pass

    print(f"[ORDER SENT] {combo_key} -> ok={ok} info={info}", flush=True)

def _combo_key(strat_name: str, symbol: str, tf_min: int) -> str:
    return f"{strat_name}|{symbol}|{int(tf_min)}"

def _perf_init(combo: str):
    if combo not in PERF:
        PERF[combo] = {
            "trades": 0, "wins": 0, "losses": 0,
            "gross_profit": 0.0, "gross_loss": 0.0,
            "net_pnl": 0.0, "max_dd": 0.0,
            "equity_curve": [0.0],
        }

def _perf_update(combo: str, pnl: float):
    _perf_init(combo)
    p = PERF[combo]
    p["trades"] += 1
    if pnl > 0:
        p["wins"] += 1
        p["gross_profit"] += pnl
    elif pnl < 0:
        p["losses"] += 1
        p["gross_loss"] += pnl
    p["net_pnl"] += pnl
    # update equity curve + drawdown
    ec = p["equity_curve"]
    ec.append(ec[-1] + pnl)
    dd = min(0.0, ec[-1] - max(ec))  # negative number
    p["max_dd"] = min(p["max_dd"], dd)

def _record_open_trade(strat_name: str, symbol: str, tf_min: int, sig: dict):
    combo = _combo_key(strat_name, symbol, tf_min)
    _perf_init(combo)

    # Support both your legacy and new keys
    tp = sig.get("tp_abs", sig.get("takeProfit"))
    sl = sig.get("sl_abs", sig.get("stopLoss"))

    t = LiveTrade(
        combo=combo,
        symbol=symbol,
        tf_min=int(tf_min),
        side=sig["action"],  # "buy"/"sell"
        entry=float(sig.get("entry") or sig.get("price") or 0.0) if sig.get("entry") is not None or sig.get("price") is not None else float("nan"),
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
    trades = OPEN_TRADES[key]
    for t in trades:
        if not t.is_open:
            continue

        hit_tp = (high >= t.tp) if t.side == "buy" else (low <= t.tp)
        hit_sl = (low <= t.sl) if t.side == "buy" else (high >= t.sl)

        if hit_tp and hit_sl:
            # both inside the same bar — choose policy
            pick = TP_SL_SAME_BAR
            if pick == "sl":
                hit_tp, hit_sl = False, True
            else:
                hit_tp, hit_sl = True, False

        if hit_tp or hit_sl:
            t.is_open = False
            t.exit_time = ts.tz_convert("UTC").isoformat() if hasattr(ts, "tzinfo") else str(ts)
            if hit_tp:
                t.exit = t.tp
                t.reason = "tp"
            else:
                t.exit = t.sl
                t.reason = "sl"

            # PnL in dollars
            if t.side == "buy":
                pnl = (t.exit - t.entry) * t.qty
            else:
                pnl = (t.entry - t.exit) * t.qty

            _perf_update(t.combo, pnl)
            print(f"[CLOSE] {t.combo} {t.reason.upper()} qty={t.qty} entry={t.entry:.2f} exit={t.exit:.2f} pnl={pnl:+.2f}", flush=True)

# ==============================
# Signal adapters (return dict with quantity)
# ==============================

def signal_poc(
    symbol: str,
    df1m: pd.DataFrame,
    tf_min: int = 5,
    poc_bin: float = 0.05,
    retest_tol: float = 0.5,
    r_multiple: float = 2.0,
    stop_buf: float = 0.3,
    ema_fast: int = 8,
    ema_slow: int = 21,
    am_window=((9, 30), (11, 0)),
    pm_window=((15, 0), (16, 0)),
    trade_am: bool = True,
    trade_pm: bool = True,
):
    if df1m is None or df1m.empty:
        return None

    bars = _resample(df1m, tf_min)
    if bars.empty or len(bars) < max(ema_fast, ema_slow) + 5:
        return None

    # prev-day POC (volume bins)
    df1m_et = df1m.copy()
    df1m_et["date_et"] = df1m_et.index.date
    poc_by_date = {}
    for date, day_df in df1m_et.groupby("date_et", sort=True):
        rth = day_df[day_df.index.map(_is_rth)]
        if rth.empty:
            continue
        prices = rth["close"].to_numpy()
        vols   = rth["volume"].to_numpy()
        bins = np.floor(prices / poc_bin) * poc_bin
        vol_by_bin = {}
        for b, v in zip(bins, vols):
            vol_by_bin[b] = vol_by_bin.get(b, 0.0) + float(v)
        if vol_by_bin:
            poc_by_date[date] = float(max(vol_by_bin.items(), key=lambda kv: kv[1])[0])

    dates = sorted(poc_by_date.keys())
    if len(dates) < 2:
        return None
    prev_poc_map = {dates[i]: poc_by_date[dates[i - 1]] for i in range(1, len(dates))}
    bars = bars.copy()
    bars["date_et"] = bars.index.date
    bars["prevPOC"] = bars["date_et"].map(prev_poc_map)

    # EMAs
    emaF = bars["close"].ewm(span=ema_fast, adjust=False).mean()
    emaS = bars["close"].ewm(span=ema_slow, adjust=False).mean()
    up = (emaF > emaS).fillna(False)
    dn = (emaF < emaS).fillna(False)

    # session guard
    (am_sH, am_sM), (am_eH, am_eM) = am_window
    (pm_sH, pm_sM), (pm_eH, pm_eM) = pm_window
    def in_session(ts):
        hh, mm = ts.hour, ts.minute
        in_am = trade_am and ((hh > am_sH or (hh == am_sH and mm >= am_sM)) and (hh < am_eH or (hh == am_eH and mm <= am_eM)))
        in_pm = trade_pm and ((hh > pm_sH or (hh == pm_sH and mm >= pm_sM)) and (hh < pm_eH or (hh == pm_eH and mm <= pm_eM)))
        return in_am or in_pm

    ts = bars.index[-1]
    if not in_session(ts):
        return None

    row = bars.iloc[-1]
    poc = row["prevPOC"]
    if pd.isna(poc):
        return None

    near_long  = (row["low"]  <= poc + retest_tol) and (row["close"] >  poc)
    near_short = (row["high"] >= poc - retest_tol) and (row["close"] <  poc)
    long_trend  = bool(up.iloc[-1])
    short_trend = bool(dn.iloc[-1])

    if long_trend and near_long:
        entry = float(row["close"])
        sl    = float(poc - stop_buf)
        tp    = float(entry + r_multiple * max(entry - sl, 0.1))
        qty   = _position_qty(entry, sl)
        return {"action":"buy","orderType":"market","price":None,
                "takeProfit":tp,"stopLoss":sl,"barTime":ts.tz_convert("UTC").isoformat(),
                "quantity":int(qty)}

    if short_trend and near_short:
        entry = float(row["close"])
        sl    = float(poc + stop_buf)
        tp    = float(entry - r_multiple * max(sl - entry, 0.1))
        qty   = _position_qty(entry, sl)
        return {"action":"sell","orderType":"market","price":None,
                "takeProfit":tp,"stopLoss":sl,"barTime":ts.tz_convert("UTC").isoformat(),
                "quantity":int(qty)}
  
    return None


def signal_ict_bos_fvg(
    symbol: str,
    df1m: pd.DataFrame,
    tf_min: int = 5,
    bos_lookback: int = 5,
    min_fvg_size: float = 0.05,
    fvg_validity_bars: int = 50,
    entry_buffer: float = 0.02,
    r_multiple: float = 2.0,
    atr_period: int = 14,
    atr_stop_multiplier: float = 2.0,
    am_window=((9,30),(11,0)),
    pm_window=((15,0),(16,0)),
    trade_am=True,
    trade_pm=True,
):
    if df1m is None or df1m.empty:
        return None
    bars = _resample(df1m, tf_min)
    if len(bars) < max(atr_period, bos_lookback) + 10:
        return None

    # ATR
    hl = bars["high"] - bars["low"]
    hc = (bars["high"] - bars["close"].shift(1)).abs()
    lc = (bars["low"] - bars["close"].shift(1)).abs()
    atr = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(atr_period).mean()

    # BOS
    bos_bull = pd.Series(False, index=bars.index)
    bos_bear = pd.Series(False, index=bars.index)
    for i in range(bos_lookback, len(bars)):
        prev_high = bars.iloc[i-bos_lookback:i]["high"].max()
        prev_low  = bars.iloc[i-bos_lookback:i]["low"].min()
        if bars.iloc[i]["high"] > prev_high: bos_bull.iloc[i] = True
        if bars.iloc[i]["low"]  < prev_low:  bos_bear.iloc[i] = True

    # FVGs
    bull_top = pd.Series(np.nan, index=bars.index); bull_bot = pd.Series(np.nan, index=bars.index)
    bear_top = pd.Series(np.nan, index=bars.index); bear_bot = pd.Series(np.nan, index=bars.index)
    for i in range(2, len(bars)):
        gap_bottom = bars.iloc[i-2]["high"]; gap_top = bars.iloc[i]["low"]
        if gap_top > gap_bottom + min_fvg_size:
            bull_bot.iloc[i] = gap_bottom; bull_top.iloc[i] = gap_top
        gap_top2 = bars.iloc[i-2]["low"]; gap_bot2 = bars.iloc[i]["high"]
        if gap_top2 > gap_bot2 + min_fvg_size:
            bear_top.iloc[i] = gap_top2; bear_bot.iloc[i] = gap_bot2

    # Active FVG windows
    def _active(top, bot, validity):
        a_top = pd.Series(np.nan, index=bars.index); a_bot = pd.Series(np.nan, index=bars.index)
        cur_t = np.nan; cur_b = np.nan; age=0
        for i in range(len(bars)):
            if not np.isnan(top.iloc[i]): cur_t,cur_b,age = top.iloc[i], bot.iloc[i], 0
            else:
                age += 1
                if age > validity: cur_t,cur_b = np.nan, np.nan
            a_top.iloc[i]=cur_t; a_bot.iloc[i]=cur_b
        return a_top, a_bot

    a_bull_top, a_bull_bot = _active(bull_top, bull_bot, fvg_validity_bars)
    a_bear_top, a_bear_bot = _active(bear_top, bear_bot, fvg_validity_bars)

    # session guard
    (am_sH, am_sM), (am_eH, am_eM) = am_window
    (pm_sH, pm_sM), (pm_eH, pm_eM) = pm_window
    def in_session(ts):
        hh, mm = ts.hour, ts.minute
        in_am = trade_am and ((hh > am_sH or (hh == am_sH and mm >= am_sM)) and (hh < am_eH or (hh == am_eH and mm <= am_eM)))
        in_pm = trade_pm and ((hh > pm_sH or (hh == pm_sH and mm >= pm_sM)) and (hh < pm_eH or (hh == pm_eH and mm <= pm_eM)))
        return in_am or in_pm

    ts = bars.index[-1]
    if not in_session(ts): return None
    row = bars.iloc[-1]
    atr_now = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 1.0

    # Long
    if bool(bos_bull.iloc[-1]) and not np.isnan(a_bull_top.iloc[-1]) and not np.isnan(a_bull_bot.iloc[-1]):
        top_, bot_ = float(a_bull_top.iloc[-1]), float(a_bull_bot.iloc[-1])
        if (row["low"] <= top_ + entry_buffer) and (row["high"] >= bot_ - entry_buffer):
            entry = float(row["close"])
            sl    = bot_ - atr_now * atr_stop_multiplier
            tp    = entry + r_multiple * abs(entry - sl)
            qty   = _position_qty(entry, sl)
            return {"action":"buy","orderType":"market","price":None,
                    "takeProfit":tp,"stopLoss":sl,"barTime":ts.tz_convert("UTC").isoformat(),
                    "quantity":int(qty)}

    # Short
    if bool(bos_bear.iloc[-1]) and not np.isnan(a_bear_top.iloc[-1]) and not np.isnan(a_bear_bot.iloc[-1]):
        top_, bot_ = float(a_bear_top.iloc[-1]), float(a_bear_bot.iloc[-1])
        if (row["high"] >= bot_ - entry_buffer) and (row["low"] <= top_ + entry_buffer):
            entry = float(row["close"])
            sl    = top_ + atr_now * atr_stop_multiplier
            tp    = entry - r_multiple * abs(sl - entry)
            qty   = _position_qty(entry, sl)
            return {"action":"sell","orderType":"market","price":None,
                    "takeProfit":tp,"stopLoss":sl,"barTime":ts.tz_convert("UTC").isoformat(),
                    "quantity":int(qty)}

    return None


def signal_ict_bos_fvg_ob(
    symbol: str,
    df1m: pd.DataFrame,
    tf_min: int = 5,
    swing_length: int = 5,
    fvg_min_gap: float = 0.02,
    fvg_max_age_bars: int = 20,
    ob_lookback: int = 10,
    r_multiple: float = 2.0,
    atr_period: int = 14,
    min_stop_distance: float = 0.5,
    max_stop_distance: float = 2.0,
    am_window=((9,30),(11,0)),
    pm_window=((15,0),(16,0)),
    trade_am=True,
    trade_pm=True,
):
    if df1m is None or df1m.empty:
        return None
    bars = _resample(df1m, tf_min)
    if len(bars) < max(atr_period, swing_length*2+1) + 10:
        return None

    # ATR (for context)
    hl = bars['high'] - bars['low']
    hc = (bars['high'] - bars['close'].shift(1)).abs()
    lc = (bars['low'] - bars['close'].shift(1)).abs()
    atr = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(atr_period).mean()

    # FVG flags & zones (forward-fill with age limit)
    bull_fvg = (bars['low'] > bars['high'].shift(2) + fvg_min_gap)
    bear_fvg = (bars['high'] < bars['low'].shift(2) - fvg_min_gap)
    bars['bull_fvg_top'] = np.where(bull_fvg, bars['low'], np.nan)
    bars['bull_fvg_bot'] = np.where(bull_fvg, bars['high'].shift(2), np.nan)
    bars['bear_fvg_top'] = np.where(bear_fvg, bars['low'].shift(2), np.nan)
    bars['bear_fvg_bot'] = np.where(bear_fvg, bars['high'], np.nan)
    for col in ['bull_fvg_top','bull_fvg_bot','bear_fvg_top','bear_fvg_bot']:
        bars[col] = bars[col].ffill(limit=fvg_max_age_bars)

    # Simple OB proxy: highest volume bar in last N
    vol_slice = bars['volume'].iloc[-ob_lookback:]
    if vol_slice.empty: return None
    ob_idx = vol_slice.idxmax(); ob_bar = bars.loc[ob_idx]
    bull_ob_high = float(ob_bar['high']); bull_ob_low = float(ob_bar['low'])
    bear_ob_high = bull_ob_high;          bear_ob_low = bull_ob_low

    # session guard
    (am_sH, am_sM), (am_eH, am_eM) = am_window
    (pm_sH, pm_sM), (pm_eH, pm_eM) = pm_window
    def in_session(ts):
        hh, mm = ts.hour, ts.minute
        in_am = trade_am and ((hh > am_sH or (hh == am_sH and mm >= am_sM)) and (hh < am_eH or (hh == am_eH and mm <= am_eM)))
        in_pm = trade_pm and ((hh > pm_sH or (hh == pm_sH and mm >= pm_sM)) and (hh < pm_eH or (hh == pm_eH and mm <= pm_eM)))
        return in_am or in_pm

    ts = bars.index[-1]
    if not in_session(ts): return None
    row = bars.iloc[-1]

    # recent BOS (tight)
    i = len(bars)-1
    prev_high = bars.iloc[max(0,i-20):i]["high"].max()
    prev_low  = bars.iloc[max(0,i-20):i]["low"].min()
    bos_bull = row['high'] > prev_high + 0.1
    bos_bear = row['low']  < prev_low  - 0.1

    # Long: inside bull FVG AND within OB range
    in_bull_fvg = (row.get('bull_fvg_top',np.nan) <= row['low']) and (row.get('bull_fvg_bot',np.nan) >= row['high'])
    in_bull_ob  = (row['low'] <= bull_ob_high) and (row['high'] >= bull_ob_low)
    if bos_bull and not np.isnan(row.get('bull_fvg_top',np.nan)) and not np.isnan(row.get('bull_fvg_bot',np.nan)) and in_bull_fvg and in_bull_ob:
        entry = float(row['close'])
        atr_now = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 1.0
        # clamp stop distance
        raw_stop = bull_ob_low - max(0.5, atr_now*0.5)
        dist = max(min_stop_distance, min(abs(entry - raw_stop), max_stop_distance))
        sl = entry - dist
        tp = entry + r_multiple * dist
        qty = _position_qty(entry, sl)
        return {"action":"buy","orderType":"market","price":None,
                "takeProfit":tp,"stopLoss":sl,"barTime":ts.tz_convert("UTC").isoformat(),
                "quantity":int(qty)}

    # Short: inside bear FVG AND within OB range
    in_bear_fvg = (row.get('bear_fvg_top',np.nan) >= row['low']) and (row.get('bear_fvg_bot',np.nan) <= row['high'])
    in_bear_ob  = (row['low'] <= bear_ob_high) and (row['high'] >= bear_ob_low)
    if bos_bear and not np.isnan(row.get('bear_fvg_top',np.nan)) and not np.isnan(row.get('bear_fvg_bot',np.nan)) and in_bear_fvg and in_bear_ob:
        entry = float(row['close'])
        atr_now = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 1.0
        raw_stop = bear_ob_high + max(0.5, atr_now*0.5)
        dist = max(min_stop_distance, min(abs(raw_stop - entry), max_stop_distance))
        sl = entry + dist
        tp = entry - r_multiple * dist
        qty = _position_qty(entry, sl)
        return {"action":"sell","orderType":"market","price":None,
                "takeProfit":tp,"stopLoss":sl,"barTime":ts.tz_convert("UTC").isoformat(),
                "quantity":int(qty)}

    return None


def signal_poc_pinescript(
    symbol: str,
    df1m: pd.DataFrame,
    tf_min: int = 5,
    tp_pct: float = 0.010,
    sl_pct: float = 0.005,
    poc_bin: float = 0.05,
    use_vwap_poc: bool = True,
    am_window=((9,30),(11,0)),
    pm_window=((15,0),(16,0)),
    trade_am=True,
    trade_pm=True,
):
    if df1m is None or df1m.empty:
        return None
    bars = _resample(df1m, tf_min)
    if len(bars) < 3: return None

    # prev-day POC (VWAP or profile)
    df_grouped = df1m.groupby(df1m.index.date)
    prev_poc_by_date = {}
    for date, day_df in df_grouped:
        rth = day_df[day_df.index.map(_is_rth)]
        if rth.empty:
            prev_poc_by_date[date] = math.nan
            continue
        if use_vwap_poc:
            typical = (rth["high"]+rth["low"]+rth["close"])/3.0
            vol = rth["volume"]
            poc = (typical*vol).sum()/vol.sum() if vol.sum()>0 else math.nan
        else:
            prices = rth["close"].to_numpy(); vols = rth["volume"].to_numpy()
            bins = np.floor(prices/poc_bin)*poc_bin
            vol_by_bin = {}
            for b,v in zip(bins,vols):
                vol_by_bin[b]=vol_by_bin.get(b,0.0)+float(v)
            poc = max(vol_by_bin.items(), key=lambda kv: kv[1])[0] if vol_by_bin else math.nan
        prev_poc_by_date[date]=poc

    dates = sorted(prev_poc_by_date.keys())
    if len(dates) < 2: return None
    prev_for_today = {dates[i]:prev_poc_by_date[dates[i-1]] for i in range(1,len(dates))}
    bars = bars.copy(); bars["date_et"]=bars.index.date; bars["prevPOC"]=bars["date_et"].map(prev_for_today)

    # session guard
    (am_sH, am_sM), (am_eH, am_eM) = am_window
    (pm_sH, pm_sM), (pm_eH, pm_eM) = pm_window
    def in_session(ts):
        hh, mm = ts.hour, ts.minute
        in_am = trade_am and ((hh > am_sH or (hh == am_sH and mm >= am_sM)) and (hh < am_eH or (hh == am_eH and mm <= am_eM)))
        in_pm = trade_pm and ((hh > pm_sH or (hh == pm_sH and mm >= pm_sM)) and (hh < pm_eH or (hh == pm_eH and mm <= pm_eM)))
        return in_am or in_pm

    i = len(bars)-1; ts = bars.index[i]
    if not in_session(ts): return None

    price = float(bars.iloc[i]["close"])
    prev_close = float(bars.iloc[i-1]["close"]) if i>0 else math.nan
    poc = bars.iloc[i]["prevPOC"]
    if pd.isna(poc) or math.isnan(prev_close):
        return None

    cross_up   = (prev_close < poc) and (price >= poc)
    cross_down = (prev_close > poc) and (price <= poc)

    if cross_up:
        tp = price * (1.0 + tp_pct)
        sl = price * (1.0 - sl_pct)
        qty = _position_qty(price, sl)
        return {"action":"buy","orderType":"market","price":None,
                "takeProfit":tp,"stopLoss":sl,"barTime":ts.tz_convert("UTC").isoformat(),
                "quantity":int(qty)}

    if cross_down:
        tp = price * (1.0 - tp_pct)
        sl = price * (1.0 + sl_pct)
        qty = _position_qty(price, sl)
        return {"action":"sell","orderType":"market","price":None,
                "takeProfit":tp,"stopLoss":sl,"barTime":ts.tz_convert("UTC").isoformat(),
                "quantity":int(qty)}

    return None

# ==============================
# Router
# ==============================
def compute_signal(strategy_name, symbol, tf_minutes):
    df1m = fetch_polygon_1m(symbol, lookback_minutes=2400)  # ~40 hours
    if df1m is None or df1m.empty:
        return None

    # ensure tz is ET
    try:
        df1m.index = df1m.index.tz_convert("America/New_York")
    except Exception:
        df1m.index = df1m.index.tz_localize("UTC").tz_convert("America/New_York")

    if strategy_name == "poc":             return signal_poc(symbol, df1m, tf_minutes)
    if strategy_name == "ict_bos_fvg":     return signal_ict_bos_fvg(symbol, df1m, tf_minutes)
    if strategy_name == "ict_bos_fvg_ob":  return signal_ict_bos_fvg_ob(symbol, df1m, tf_minutes)
    if strategy_name == "poc_pinescript":  return signal_poc_pinescript(symbol, df1m, tf_minutes)
    return None

def send_startup_test_order():
    """Prove webhook/DRY-RUN path works on boot."""
    print(
        f"[STARTUP] Flags: SEND_TEST_ORDER={SEND_TEST_ORDER}  DRY_RUN={DRY_RUN}  PAPER_MODE={PAPER_MODE}  "
        f"TP_URL_SET={'yes' if bool(TP_URL) else 'no'}",
        flush=True
    )

    if not SEND_TEST_ORDER:
        print("[STARTUP] SEND_TEST_ORDER=0 → skipping startup test order.", flush=True)
        return

    test_payload = {
        "ticker": "AAPL",
        "action": "buy",
        "orderType": "market",
        "quantity": 1,
        "meta": {"note": "startup-test-order", "environment": "paper" if PAPER_MODE else "live"}
    }
    print("[STARTUP] Attempting startup test order (AAPL, qty=1)...", flush=True)
    ok, info = send_to_traderspost(test_payload)
    print(f"[TEST ORDER RESULT] ok={ok}  info={info}", flush=True)

def replay_signals_once():
    """Replay a single strategy on recent history and print how many signals you'd have had."""
    if not REPLAY_ON_START or not (REPLAY_STRAT and REPLAY_SYMBOL):
        return
    lookback = max(REPLAY_HOURS * 60, 120)  # minutes
    df1m = fetch_polygon_1m(REPLAY_SYMBOL, lookback_minutes=lookback)
    if df1m is None or df1m.empty:
        print("[REPLAY] No data fetched.", flush=True)
        return

    # Normalize tz
    try:
        df1m.index = df1m.index.tz_convert("America/New_York")
    except Exception:
        df1m.index = df1m.index.tz_localize("UTC").tz_convert("America/New_York")

    # Resample to TF and iterate bar-by-bar, calling the right adapter each time
    tf = REPLAY_TF
    bars = _resample(df1m, tf)
    hits = 0
    last_key = None
    for i in range(len(bars)):
        # Slice up to i to emulate "bar close" at that point
        df_slice = df1m.loc[:bars.index[i]]
        if REPLAY_STRAT == "poc":
            sig = signal_poc(REPLAY_SYMBOL, df_slice, tf)
        elif REPLAY_STRAT == "ict_bos_fvg":
            sig = signal_ict_bos_fvg(REPLAY_SYMBOL, df_slice, tf)
        elif REPLAY_STRAT == "ict_bos_fvg_ob":
            sig = signal_ict_bos_fvg_ob(REPLAY_SYMBOL, df_slice, tf)
        elif REPLAY_STRAT == "poc_pinescript":
            sig = signal_poc_pinescript(REPLAY_SYMBOL, df_slice, tf)
        else:
            print(f"[REPLAY] Unknown strategy '{REPLAY_STRAT}'", flush=True)
            return

        if sig:
            # OPTIONAL: actually run through payload pipeline during replay
            if os.getenv("REPLAY_SEND_ORDERS", "0").lower() in ("1", "true", "yes"):
                handle_signal(REPLAY_STRAT, REPLAY_SYMBOL, tf, sig)

            # de-dupe by bar time + action
            k = (sig.get("barTime", ""), sig.get("action", ""))
            if k != last_key:
                hits += 1
                last_key = k

    print(f"[REPLAY] {REPLAY_STRAT} {REPLAY_SYMBOL} {tf}m -> {hits} signal(s) in last {REPLAY_HOURS}h.", flush=True)

def debug_snapshot_one_combo(name, sym, tf):
    target = f"{name},{sym},{tf}"
    if DEBUG_COMBO != target:
        return

    df1m = fetch_polygon_1m(sym, lookback_minutes=600)
    if df1m is None or df1m.empty:
        print(f"[DEBUG] No data for {target}", flush=True)
        return

    try:
        df1m.index = df1m.index.tz_convert("America/New_York")
    except Exception:
        df1m.index = df1m.index.tz_localize("UTC").tz_convert("America/New_York")

    bars = _resample(df1m, tf)
    if bars.empty:
        print(f"[DEBUG] No bars after resample for {target}", flush=True)
        return

    row = bars.iloc[-1]; ts = bars.index[-1]
    print(f"[DEBUG] Latest {target} bar={ts} O={row['open']} H={row['high']} L={row['low']} C={row['close']} V={row['volume']}", flush=True)

    # Compute a signal on the full slice (bar-close behavior)
    if name == "poc":
        sig = signal_poc(sym, df1m, tf)
    elif name == "ict_bos_fvg":
        sig = signal_ict_bos_fvg(sym, df1m, tf)
    elif name == "ict_bos_fvg_ob":
        sig = signal_ict_bos_fvg_ob(sym, df1m, tf)
    elif name == "poc_pinescript":
        sig = signal_poc_pinescript(sym, df1m, tf)
    else:
        sig = None
    if sig:
        handle_signal(name, sym, tf, sig)
      
    print(f"[DEBUG] Signal -> {sig}", flush=True)

def replay_signals_once():
    """Replay a single strategy on recent history and print how many signals you'd have had."""
    try:
        if not REPLAY_ON_START or not (REPLAY_STRAT and REPLAY_SYMBOL):
            print(f"[REPLAY] Skipped. REPLAY_ON_START={REPLAY_ON_START} STRAT='{REPLAY_STRAT}' SYMBOL='{REPLAY_SYMBOL}'", flush=True)
            return

        lookback = max(REPLAY_HOURS * 60, 120)  # minutes
        print(f"[REPLAY] Starting: strat={REPLAY_STRAT} symbol={REPLAY_SYMBOL} tf={REPLAY_TF}m hours={REPLAY_HOURS}", flush=True)

        df1m = fetch_polygon_1m(REPLAY_SYMBOL, lookback_minutes=lookback)
        if df1m is None or df1m.empty:
            print("[REPLAY] No data fetched.", flush=True)
            return

        try:
            df1m.index = df1m.index.tz_convert("America/New_York")
        except Exception:
            df1m.index = df1m.index.tz_localize("UTC").tz_convert("America/New_York")

        tf = REPLAY_TF
        bars = _resample(df1m, tf)
        if bars.empty:
            print("[REPLAY] Resampled bars empty.", flush=True)
            return

        hits = 0
        last_key = None
        for i in range(len(bars)):
            df_slice = df1m.loc[:bars.index[i]]

        if REPLAY_STRAT == "poc":
            sig = signal_poc(REPLAY_SYMBOL, df_slice, tf)
        elif REPLAY_STRAT == "ict_bos_fvg":
            sig = signal_ict_bos_fvg(REPLAY_SYMBOL, df_slice, tf)
        elif REPLAY_STRAT == "ict_bos_fvg_ob":
            sig = signal_ict_bos_fvg_ob(REPLAY_SYMBOL, df_slice, tf)
        elif REPLAY_STRAT == "poc_pinescript":
            sig = signal_poc_pinescript(REPLAY_SYMBOL, df_slice, tf)
        else:
            print(f"[REPLAY] Unknown strategy '{REPLAY_STRAT}'", flush=True)
            return

        # OPTIONALLY send orders during replay (off by default)
        if sig and os.getenv("REPLAY_SEND_ORDERS", "0").lower() in ("1", "true", "yes"):
            handle_signal(REPLAY_STRAT, REPLAY_SYMBOL, tf, sig)

        # Your original hit-count logic (unchanged)
        if sig:
            k = (sig.get("barTime", ""), sig.get("action", ""))
            if k != last_key:
                hits += 1
                last_key = k

        print(f"[REPLAY] {REPLAY_STRAT} {REPLAY_SYMBOL} {tf}m -> {hits} signal(s) in last {REPLAY_HOURS}h.", flush=True)

    except Exception as e:
        import traceback
        print("[REPLAY ERROR]", e, traceback.format_exc(), flush=True)

def _print_polygon_freshness():
    try:
        # Use your unified fetch if you have it; otherwise call your polygon 1m fetcher directly.
        fetch_fn = globals().get("fetch_bars_1m") or globals().get("fetch_polygon_1m")
        if not fetch_fn:
            print("[FRESHNESS] No fetch function found (fetch_bars_1m/fetch_polygon_1m).", flush=True)
            return

        df = fetch_fn(POLY_TEST_SYMBOL, lookback_minutes=30)
        if df is None or df.empty:
            print(f"[FRESHNESS] No bars for {POLY_TEST_SYMBOL}.", flush=True)
            return

        last_ts = df.index[-1]                   # tz-aware ET
        now_utc = datetime.now(timezone.utc)
        now_et  = now_utc.astimezone(last_ts.tzinfo)
        age = (now_et - last_ts)
        age_sec = int(age.total_seconds())

        print(f"[FRESHNESS] {POLY_TEST_SYMBOL} last bar={last_ts.isoformat()}  now={now_et.isoformat()}  age={age_sec}s", flush=True)

        # Quick interpretation to make it obvious in logs:
        if age <= timedelta(minutes=1, seconds=30):
            print("[FRESHNESS] ✅ Looks REAL-TIME (≤ ~90s old).", flush=True)
        elif age <= timedelta(minutes=16):
            print("[FRESHNESS] ⚠️ Looks like ~15-minute delay.", flush=True)
        else:
            print("[FRESHNESS] ⏸ Market likely closed or no recent bars.", flush=True)
    except Exception as e:
        print(f"[FRESHNESS] Error: {e}", flush=True)

def print_eod_report():
    print("\n========== EOD REPORT ==========", flush=True)
    print(f"RUN_ID: {RUN_ID}", flush=True)
    print(f"Totals: signals={COUNTS['signals']}  orders.ok={COUNTS['orders.ok']}  orders.err={COUNTS['orders.err']}", flush=True)
    print("Per-Combo:", flush=True)
    for k in sorted(COMBO_COUNTS.keys()):
        if k.endswith("::signals"):
            base = k[:-9]
            sigs = COMBO_COUNTS[k]
            oks  = COMBO_COUNTS.get(f"{base}::orders.ok", 0)
            ers  = COMBO_COUNTS.get(f"{base}::orders.err", 0)
            print(f"  {base}  -> signals={sigs}  orders.ok={oks}  orders.err={ers}", flush=True)
    print("================================\n", flush=True)

# CSV for Tableau/Excel
    out_path = f"/opt/render/project/src/eod_perf_{RUN_ID}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["combo", "trades", "wins", "losses", "win_rate", "profit_factor", "net_pnl", "max_drawdown"])
        for combo in sorted(PERF.keys()):
            p = PERF[combo]
            trades = p["trades"]
            if trades == 0:
                continue
            wins = p["wins"]; losses = p["losses"]
            wr = (100.0 * wins / trades) if trades else 0.0
            gp = p["gross_profit"]; gl = p["gross_loss"]
            pf = (gp / abs(gl)) if gl < 0 else float("inf") if gp > 0 else 0.0
            w.writerow([combo, trades, wins, losses, round(wr,2), round(pf,3), round(p["net_pnl"],2), round(p["max_dd"],2)])

    print(f"[EOD] Wrote summary CSV → {out_path}", flush=True)

# ==============================
# Main loop — bar-close polling
# ==============================
def main():
    print("Bot starting…", flush=True)

    if POLY_LIVE_TEST in ("1", "true", "yes"):
        _print_polygon_freshness()


      # --- TEMP: send a one-time TP/SL test order on startup ---
    if STARTUP_TPSL_TEST in ("1", "true", "yes"):
        try:
            test_sig = {
                "action": "buy",
                "orderType": "market",
                "quantity": 1,
                # absolute TP/SL to validate payload shape
                "tp_abs": 102.00,
                "sl_abs": 99.00,
                "meta": {"note": "tp-sl-format-test"}
            }
            payload = build_payload("AAPL", test_sig)
            print("[STARTUP TEST PAYLOAD]", payload, flush=True)
            ok, info = send_to_traderspost(payload)
            print("[STARTUP TEST RESULT]", ok, info, flush=True)
        except Exception as e:
            print("[STARTUP TEST ERROR]", e, flush=True)

    # ---- startup diagnostics (runs once) ----
    try:
        send_startup_test_order()
        print("[STARTUP] Calling replay_signals_once()…", flush=True)
        replay_signals_once()
    except Exception as e:
        import traceback
        print("[STARTUP DIAGS ERROR]", e, traceback.format_exc(), flush=True)

# ---- main loop ----
while True:
    loop_start = time.time()
    try:
        print("Tick…", flush=True)

        for (name, sym, tf) in COMBOS:
            # Optional: one-combo debug snapshot (if you added that helper)
            if DEBUG_COMBO:
                try:
                    debug_snapshot_one_combo(name, sym, tf)
                except Exception as e:
                    print(f"[DEBUG SNAPSHOT ERROR] {name},{sym},{tf}: {e}", flush=True)

            # --- CLOSE FIRST: get the latest closed bar and close any open trade on TP/SL ---
            last_row_close = None
            try:
                # small, fast lookback just to get the last bar at this TF
                lookback = max(90, int(tf) * 10)  # minutes
                df1m_latest = fetch_polygon_1m(sym, lookback_minutes=lookback)
                bars_latest = _resample(df1m_latest, tf)
                if bars_latest is not None and not bars_latest.empty:
                    last_row = bars_latest.iloc[-1]
                    last_ts  = bars_latest.index[-1]
                    last_row_close = float(last_row["close"])
                    _maybe_close_on_bar(
                        sym, tf, last_ts,
                        float(last_row["high"]),
                        float(last_row["low"]),
                        last_row_close
                    )
            except Exception as e:
                print(f"[CLOSE-PHASE ERROR] {name} {sym} {tf}m: {e}", flush=True)

            # --- THEN compute a fresh signal on the full slice ---
            sig = compute_signal(name, sym, tf)
            if not sig:
                continue

            # give ledger an explicit entry if missing
            if "entry" not in sig and last_row_close is not None:
                sig["entry"] = last_row_close

            # de-dupe
            k = _dedupe_key(name, sym, tf, sig["action"], sig.get("barTime", ""))
            if k in _sent:
                continue
            _sent.add(k)

            # record open trade BEFORE posting
            try:
                _record_open_trade(name, sym, tf, sig)
            except Exception as e:
                print(f"[LEDGER OPEN ERROR] {name} {sym} {tf}m: {e}", flush=True)

            payload = build_payload(sym, sig)
            ok, info = send_to_traderspost(payload)
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{stamp}] {name} {sym} {tf}m -> {sig['action']} qty={sig['quantity']} | {info}", flush=True)

    except Exception as e:
        import traceback
        print("[LOOP ERROR]", e, traceback.format_exc(), flush=True)

    elapsed = time.time() - loop_start
    time.sleep(max(1, POLL_SECONDS - int(elapsed)))

def build_payload(symbol: str, sig: dict):
    """
    Build a TradersPost payload, accepting either:
      - legacy numeric:  takeProfit=<float>, stopLoss=<float>
      - new absolute:    tp_abs=<float>,    sl_abs=<float>
    and converting them to the required nested objects.
    """
    action     = sig.get("action")                     # "buy" | "sell"
    order_type = sig.get("orderType", "market")        # "market" | "limit"
    qty        = int(sig.get("quantity", 0))

    payload = {
        "ticker": symbol,
        "action": action,
        "orderType": order_type,
        "quantity": qty,
        "meta": {}
    }

    # pass-through optional meta
    if isinstance(sig.get("meta"), dict):
        payload["meta"].update(sig["meta"])

    # include barTime (useful for audit)
    if sig.get("barTime"):
        payload["meta"]["barTime"] = sig["barTime"]

    # limit orders: map price -> limitPrice
    if order_type.lower() == "limit":
        if sig.get("price") is not None:
            payload["limitPrice"] = float(round(sig["price"], 2))

    # ---- unify TP/SL sources ----
    tp_abs = sig.get("tp_abs")
    sl_abs = sig.get("sl_abs")

    # support your current numeric fields too
    if tp_abs is None and sig.get("takeProfit") is not None:
        tp_abs = float(sig["takeProfit"])
    if sl_abs is None and sig.get("stopLoss") is not None:
        sl_abs = float(sig["stopLoss"])

    # TradersPost requires nested objects:
    # takeProfit: {"limitPrice": <abs price>}
    # stopLoss  : {"type": "stop", "stopPrice": <abs price>}
    if tp_abs is not None:
        payload["takeProfit"] = {"limitPrice": float(round(tp_abs, 2))}
    if sl_abs is not None:
        payload["stopLoss"]   = {"type": "stop", "stopPrice": float(round(sl_abs, 2))}

    return payload

if __name__ == "__main__":
    main()
