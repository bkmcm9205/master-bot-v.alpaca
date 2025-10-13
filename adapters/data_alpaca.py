import os, requests
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ALPACA_DATA_BASE_URL = os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets").rstrip("/")
MARKET_TZ = os.getenv("MARKET_TZ", "America/New_York")
ALLOWED_EXCHANGES = set(x.strip().upper() for x in os.getenv("ALLOWED_EXCHANGES", "NASD,NASDAQ,NYSE,XNAS,XNYS").split(",") if x.strip())
MIN_PRICE = float(os.getenv("MIN_PRICE", "3.0"))

def _headers():
    return {
        "APCA-API-KEY-ID": os.getenv("ALPACA_KEY_ID", ""),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY", ""),
    }

def fetch_1m(symbol: str, start_iso: str | None = None, end_iso: str | None = None, limit: int = 10000) -> pd.DataFrame:
    """Return tz-aware ET 1m bars with columns: open, high, low, close, volume"""
    tf = "1Min"
    if end_iso is None:
        end_iso = datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
    if start_iso is None:
        start_iso = (datetime.now(timezone.utc).replace(microsecond=0) - pd.Timedelta(days=7)).isoformat().replace("+00:00","Z")
    url = f"{ALPACA_DATA_BASE_URL}/v2/stocks/{symbol}/bars"
    params = {"timeframe": tf, "start": start_iso, "end": end_iso, "limit": str(limit), "adjustment": "raw"}
    r = requests.get(url, headers=_headers(), params=params, timeout=20)
    if r.status_code != 200:
        return pd.DataFrame()
    js = r.json()
    bars = js.get("bars", [])
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    df["ts"] = pd.to_datetime(df["t"], utc=True)
    df = df.set_index("ts").sort_index()
    df.index = df.index.tz_convert(ZoneInfo(MARKET_TZ))
    df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
    return df[["open","high","low","close","volume"]]

def resample(df1m: pd.DataFrame, tf_min: int) -> pd.DataFrame:
    if df1m is None or df1m.empty or not isinstance(df1m.index, pd.DatetimeIndex):
        return pd.DataFrame()
    rule = f"{int(tf_min)}min"
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    return df1m.resample(rule, origin="start_day", label="right").agg(agg).dropna()

def get_universe_symbols(limit: int = 10000) -> list[str]:
    """Uses Alpaca /v2/assets (active, US equities), filters by ALLOWED_EXCHANGES and MIN_PRICE gate"""
    base = os.getenv("ALPACA_TRADE_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
    url = f"{base}/v2/assets"
    params = {"status":"active", "asset_class":"us_equity"}
    r = requests.get(url, headers=_headers(), params=params, timeout=30)
    if r.status_code != 200:
        return []
    out = []
    for row in r.json():
        sym = row.get("symbol", "")
        exch = (row.get("exchange", "") or "").upper()
        if not sym or not sym.isalnum():
            continue
        if ALLOWED_EXCHANGES and exch not in ALLOWED_EXCHANGES:
            continue
        out.append(sym)
        if len(out) >= limit:
            break
    return out
