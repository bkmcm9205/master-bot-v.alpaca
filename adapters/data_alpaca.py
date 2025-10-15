# adapters/data_alpaca.py
# Minimal Alpaca data adapter with hardened HTTP + universe fetch
import os, time, math
from typing import List, Optional, Tuple
from datetime import datetime
import pandas as pd
import requests

import re
import pandas as pd
import numpy as np

_TS_ISO_DOT = re.compile(r".*\.\d+Z$")   # 2025-10-14T15:32:01.123Z
_TS_ISO_NODOT = re.compile(r".*Z$")      # 2025-10-14T15:32:01Z

def _parse_alpaca_ts(series: pd.Series) -> pd.Series:
    """Parse Alpaca 't' timestamps (ns epoch OR RFC3339) without inference warnings."""
    if np.issubdtype(series.dtype, np.integer) or np.issubdtype(series.dtype, np.floating):
        # ns epoch (rare in v2 stocks bars, but handle it)
        return pd.to_datetime(series, unit="ns", utc=True)
    s0 = series.iloc[0]
    if isinstance(s0, str):
        s0 = s0.strip()
        if _TS_ISO_DOT.match(s0):
            # e.g. 2025-10-14T15:32:01.123Z
            return pd.to_datetime(series, format="%Y-%m-%dT%H:%M:%S.%fZ", utc=True)
        if _TS_ISO_NODOT.match(s0):
            # e.g. 2025-10-14T15:32:01Z
            return pd.to_datetime(series, format="%Y-%m-%dT%H:%M:%SZ", utc=True)
    # Fallback (last resort): allow mixed/odd strings
    return pd.to_datetime(series, utc=True, errors="coerce")

DATA_BASE = os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets").rstrip("/")
ALP_KEY   = os.getenv("ALPACA_API_KEY_ID", "")
ALP_SEC   = os.getenv("ALPACA_SECRET_KEY", "")

# Optional knobs
REQ_TIMEOUT      = int(os.getenv("ALPACA_DATA_TIMEOUT", "30"))
PAGE_LIMIT       = int(os.getenv("ALPACA_ASSETS_PAGE_LIMIT", "250"))  # /v2/assets page size
ALLOWED_EXCH_ENV = os.getenv("ALLOWED_EXCHANGES", "NASD,NASDAQ,NYSE,XNAS,XNYS")
MIN_PRICE_ENV    = os.getenv("MIN_PRICE", "3.0")

def _headers():
    return {
        "APCA-API-KEY-ID": ALP_KEY,
        "APCA-API-SECRET-KEY": ALP_SEC,
        "Content-Type": "application/json",
    }

def _should_retry(resp: Optional[requests.Response], exc: Optional[Exception]) -> bool:
    if exc is not None:
        return True
    if resp is None:
        return True
    # retry 429 and 5xx
    return resp.status_code == 429 or 500 <= resp.status_code < 600

def _get_json(url: str, params: dict, timeout: int = REQ_TIMEOUT,
              max_retries: int = 4, backoff_base: float = 0.75):
    """GET with retry/backoff. Returns (ok, json or None, resp or None, err or None)."""
    last_err = None
    resp = None
    for i in range(max_retries + 1):
        try:
            resp = requests.get(url, headers=_headers(), params=params, timeout=timeout)
            if not _should_retry(resp, None):
                try:
                    return (True, resp.json(), resp, None)
                except Exception as e:
                    return (False, None, resp, e)
        except Exception as e:
            last_err = e
        # backoff
        sleep_s = backoff_base * (2 ** i)
        time.sleep(min(6.0, sleep_s))
    # final failure
    return (False, None, resp, last_err)

# ---------------------------
# Bars (1 minute) -> DataFrame
# ---------------------------
def fetch_1m(symbol: str, start_iso: str, end_iso: str, limit: int = 10000,
             adjustment: str = "raw") -> pd.DataFrame:
    """
    Fetch 1-minute bars via /v2/stocks/{symbol}/bars
    Returns tz-aware UTC index; columns: open, high, low, close, volume.
    start_iso / end_iso must be ISO8601 with Z (UTC), e.g. 2025-10-14T13:30:00Z
    """
    url = f"{DATA_BASE}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": "1Min",
        "start": start_iso,
        "end": end_iso,
        "limit": str(limit),
        "adjustment": adjustment,  # raw | split | all
    }
    ok, js, resp, err = _get_json(url, params)
    if not ok or not js:
        return pd.DataFrame()

    rows = js.get("bars") or []
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Expected keys: t (ISO), o,h,l,c,v
    # Normalize to milliseconds epoch or ISO -> pandas datetime
    try:
        df["ts"] = pd.to_datetime(df["t"], utc=True)
    except Exception:
        # Fallback: if "t" is epoch ns
        df["ts"] = pd.to_datetime(df["t"], unit="ns", utc=True)
    df = df.set_index("ts").sort_index()
    # Standardize column names
    rename_map = {"o":"open","h":"high","l":"low","c":"close","v":"volume"}
    df = df.rename(columns=rename_map)
    cols = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    return df[cols].copy()

# ---------------------------
# Resample helper (used by apps)
# ---------------------------
def resample(df1m: pd.DataFrame, tf_min: int) -> pd.DataFrame:
    if df1m is None or df1m.empty or not isinstance(df1m.index, pd.DatetimeIndex):
        return pd.DataFrame()
    rule = f"{int(tf_min)}min"
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    try:
        bars = df1m.resample(rule, origin="start_day", label="right").agg(agg).dropna()
    except Exception:
        return pd.DataFrame()
    return bars

# ---------------------------
# Universe fetch via /v2/assets
# ---------------------------
def get_universe_symbols(limit: int = 10000) -> List[str]:
    """
    Returns a list of active, tradable US equity tickers.
    Applies optional env filters:
      - ALLOWED_EXCHANGES (comma-separated, e.g. NASDAQ,NYSE,XNAS,XNYS)
      - MIN_PRICE        (float; filters by 'price' if present in payload, else ignored)
    NOTE: /v2/assets does not return live prices—MIN_PRICE is best-effort:
      we only use 'min_price' when 'price' exists; otherwise no price filter is applied here.
    """
    allowed_exch = set(x.strip().upper() for x in ALLOWED_EXCH_ENV.split(",") if x.strip())
    try:
        min_price = float(MIN_PRICE_ENV)
    except Exception:
        min_price = 0.0

    url = f"{DATA_BASE.replace('data.', 'paper-api.')}/v2/assets"  # asset list lives on trade API host
    # In case base swap is undesirable in your env, fall back:
    if "paper-api.alpaca.markets" not in url and "api.alpaca.markets" not in url:
        url = "https://paper-api.alpaca.markets/v2/assets"

    out: List[str] = []
    page_token = None

    while True:
        params = {
            "status": "active",
            "asset_class": "us_equity",
            "tradable": "true",
            "page_size": str(PAGE_LIMIT),
        }
        if page_token:
            params["page_token"] = page_token

        ok, js, resp, err = _get_json(url, params)
        if not ok or not js:
            break

        items = js if isinstance(js, list) else js.get("assets") or js.get("content") or []
        if not isinstance(items, list):
            # Some deployments return a list directly at top-level
            break

        for row in items:
            sym = (row.get("symbol") or "").upper()
            exch = (row.get("primary_exchange") or row.get("exchange") or "").upper()
            tradable = bool(row.get("tradable", True))
            if not sym or not sym.isalnum():
                continue
            if allowed_exch and exch and exch not in allowed_exch:
                continue
            if not tradable:
                continue
            # Price filtering is best-effort (assets payload usually lacks a 'price' field)
            px = None
            try:
                px = float(row.get("price")) if row.get("price") is not None else None
            except Exception:
                px = None
            if px is not None and px < min_price:
                continue

            out.append(sym)
            if len(out) >= limit:
                return out

        # Pagination: V2 assets typically uses 'next_page_token'
        page_token = None
        if isinstance(js, dict):
            page_token = js.get("next_page_token") or js.get("next_page")
        if not page_token:
            break

    return out

# ---------------------------
# Optional: quick data probe
# ---------------------------
def probe_data_auth():
    """
    Lightweight check that keys + data host are OK.
    """
    url = f"{DATA_BASE}/v2/stocks/SPY/bars"
    params = {"timeframe":"1Min", "limit":"1"}
    try:
        r = requests.get(url, headers=_headers(), params=params, timeout=min(REQ_TIMEOUT, 10))
        code = r.status_code
        where = DATA_BASE
        key_hint = f"{ALP_KEY[:3]}…{ALP_KEY[-3:]}" if ALP_KEY else "<empty>"
        print(f"[DATA PROBE] GET /v2/stocks/SPY/bars -> {code} key={key_hint} base={where}", flush=True)
        if code != 200:
            print(f"[DATA PROBE] body: {r.text[:300]}", flush=True)
    except Exception as e:
        print(f"[DATA PROBE] exception: {type(e).__name__}: {e}", flush=True)
