# adapters/data_alpaca.py  — HARDENED (replace the whole file if you want)

import os, pandas as pd, requests
from datetime import datetime
from requests.adapters import HTTPAdapter, Retry

ALP_DATA_BASE = os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets").rstrip("/")
ALP_KEY  = os.getenv("ALPACA_API_KEY_ID", "")
ALP_SEC  = os.getenv("ALPACA_SECRET_KEY", "")

# Tunables (env overrideable)
DATA_TIMEOUT_SEC = float(os.getenv("ALPACA_DATA_TIMEOUT", "20"))
DATA_RETRIES     = int(os.getenv("ALPACA_DATA_RETRIES", "3"))
DATA_BACKOFF     = float(os.getenv("ALPACA_DATA_BACKOFF", "0.5"))

def _headers():
    return {
        "APCA-API-KEY-ID": ALP_KEY,
        "APCA-API-SECRET-KEY": ALP_SEC,
    }

# one shared resilient session
_session = None
def _get_session():
    global _session
    if _session is not None:
        return _session
    _session = requests.Session()
    retry = Retry(
        total=DATA_RETRIES,
        connect=DATA_RETRIES,
        read=DATA_RETRIES,
        backoff_factor=DATA_BACKOFF,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"])
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)
    _session.mount("https://", adapter)
    _session.mount("http://", adapter)
    return _session

def fetch_1m(symbol: str, start_iso: str, end_iso: str, limit: int = 10000, adjustment: str = "raw") -> pd.DataFrame:
    """
    Single-symbol 1m bars, resilient to transient timeouts and 5xx.
    Returns a tz-aware index DataFrame with columns: open, high, low, close, volume
    On any fatal error -> empty DataFrame (caller already handles empty).
    """
    try:
        url = f"{ALP_DATA_BASE}/v2/stocks/{symbol}/bars"
        params = {
            "timeframe": "1Min",
            "start": start_iso,   # e.g. '2025-10-12T23:00:31Z'
            "end":   end_iso,     # e.g. '2025-10-14T15:00:31Z'
            "limit": min(10000, int(limit)),
            "adjustment": adjustment
        }
        s = _get_session()
        r = s.get(url, headers=_headers(), params=params, timeout=DATA_TIMEOUT_SEC)
        if r.status_code != 200:
            # Soft-fail: return empty df; upstream will skip this symbol
            return pd.DataFrame()

        js = r.json()
        bars = js.get("bars") or []
        if not bars:
            return pd.DataFrame()

        df = pd.DataFrame(bars)
        # Normalize column names to your expected schema
        # Alpaca fields: t(Open time ISO), o, h, l, c, v, n, vw
        if "t" not in df.columns:
            return pd.DataFrame()
        df["ts"] = pd.to_datetime(df["t"], utc=True)
        df = df.set_index("ts").sort_index()
        rename = {"o":"open","h":"high","l":"low","c":"close","v":"volume"}
        df = df.rename(columns=rename)
        keep = [c for c in ["open","high","low","close","volume"] if c in df.columns]
        return df[keep]
    except Exception:
        # Any network/JSON error -> empty; your app already logs per-symbol fetch errors when SCANNER_DEBUG=1
        return pd.DataFrame()

# Optional: batched multi-symbol fetch (for later optimization)
# Alpaca supports /v2/stocks/bars?symbols=AAPL,MSFT&timeframe=1Min...
# You can add a fetch_1m_batch(symbols, start_iso, end_iso, ...) here when you want to
# reduce HTTP calls further. Keeping the single-symbol API for drop-in safety today.
