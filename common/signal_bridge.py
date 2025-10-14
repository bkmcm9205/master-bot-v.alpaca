# common/signal_bridge.py  — Alpaca plumbing (backward-compatible)
import os, json, requests
from datetime import datetime, timezone

ALP_BASE = os.getenv("ALPACA_TRADE_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
ALP_KEY  = os.getenv("ALPACA_API_KEY_ID", "")
ALP_SEC  = os.getenv("ALPACA_SECRET_KEY", "")

def _alp_headers():
    return {
        "APCA-API-KEY-ID": ALP_KEY,
        "APCA-API-SECRET-KEY": ALP_SEC,
        "Content-Type": "application/json",
    }

def _dbg(msg: str):
    if os.getenv("SCANNER_DEBUG", "0").lower() in ("1","true","yes"):
        print(msg, flush=True)

# ------------------------
# Auth probe (optional)
# ------------------------
def probe_alpaca_auth():
    try:
        url = f"{ALP_BASE}/v2/account"
        r = requests.get(url, headers=_alp_headers(), timeout=10)
        kid = (ALP_KEY or "")[:6] + "..." if ALP_KEY else "MISSING"
        print(f"[PROBE] ALP_BASE={ALP_BASE} KEY_ID~={kid} -> {r.status_code}", flush=True)
        if r.status_code == 200:
            js = r.json()
            print(f"[PROBE] account_number={js.get('account_number')} status={js.get('status')}", flush=True)
        else:
            print(f"[PROBE] body: {r.text[:300]}", flush=True)
    except Exception as e:
        print(f"[PROBE] exception: {e}", flush=True)

# ------------------------
# Payload normalization
# ------------------------
def _build_alpaca_payload_from_sig(symbol: str, sig: dict, strategy_tag: str | None = None) -> dict:
    """
    Convert your internal signal dict into an Alpaca order payload.
    Supports market or limit, and optional bracket TP/SL.
    """
    action = (sig.get("action") or "").lower()
    side = "buy"
    if action in ("sell", "sell_short", "short", "sell to open", "sell_to_open"):
        side = "sell"

    otype = (sig.get("orderType") or "market").lower()
    qty   = int(sig.get("quantity", 0))
    payload = {
        "symbol": symbol,
        "side": side,
        "type": "market" if otype == "market" else "limit",
        "time_in_force": "day",
        "qty": qty,
    }

    # Limit price (if any)
    lp = sig.get("limitPrice", sig.get("price"))
    if payload["type"] == "limit" and lp is not None:
        payload["limit_price"] = float(round(lp, 2))

    # TP/SL (absolute)
    tp = sig.get("tp_abs", sig.get("takeProfit"))
    sl = sig.get("sl_abs", sig.get("stopLoss"))

    # Enforce Alpaca min tick distance (>= $0.01 away from entry)
    base_price = sig.get("entry") or sig.get("price")
    if base_price is not None:
        base_price = float(base_price)
        if tp is not None and (abs(float(tp) - base_price) < 0.01):
            tp = round(base_price + (0.01 if side == "buy" else -0.01), 2)
        if sl is not None:
            if side == "buy" and (base_price - float(sl) < 0.01):
                sl = round(base_price - 0.01, 2)
            if side == "sell" and (float(sl) - base_price < 0.01):
                sl = round(base_price + 0.01, 2)

    if tp is not None or sl is not None:
        payload["order_class"] = "bracket"
        if tp is not None:
            payload["take_profit"] = {"limit_price": float(round(tp, 2))}
        if sl is not None:
            payload["stop_loss"] = {"stop_price": float(round(sl, 2))}

    return payload

def _normalize_send_args(*args, **kwargs) -> dict:
    """
    Accept either:
      1) send_to_broker(payload_dict)
      2) send_to_broker(symbol, sig_dict, strategy_tag="name")
    Returns a normalized Alpaca order payload (dict).
    """
    # Style 1: send_to_broker(payload_dict)
    if len(args) == 1 and isinstance(args[0], dict) and ("ticker" in args[0] or "symbol" in args[0]):
        p = args[0]
        # If 'ticker' used (legacy), map to 'symbol' and bracket fields
        symbol = p.get("symbol") or p.get("ticker")
        # Map TradersPost-style nested fields if present
        tp_obj = p.get("takeProfit") or {}
        sl_obj = p.get("stopLoss") or {}
        sig_like = {
            "action": p.get("action"),
            "orderType": p.get("orderType"),
            "quantity": p.get("quantity"),
            "price": p.get("limitPrice", p.get("price")),
            "takeProfit": (tp_obj or {}).get("limitPrice"),
            "stopLoss": (sl_obj or {}).get("stopPrice"),
            "entry": p.get("entry"),
        }
        return _build_alpaca_payload_from_sig(symbol, sig_like, strategy_tag=None)

    # Style 2: send_to_broker(symbol, sig_dict, strategy_tag="name")
    if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], dict):
        symbol = args[0]
        sig    = args[1]
        strategy_tag = kwargs.get("strategy_tag")
        return _build_alpaca_payload_from_sig(symbol, sig, strategy_tag=strategy_tag)

    raise TypeError("send_to_broker expects payload dict OR (symbol, sig_dict, strategy_tag=...)")

# ------------------------
# Public broker APIs
# ------------------------
def send_to_broker(*args, **kwargs):
    """
    Posts an order to Alpaca. Supports BOTH call styles:
      - send_to_broker(payload_dict)
      - send_to_broker(symbol, sig_dict, strategy_tag="name")
    Returns (ok: bool, info: str)
    """
    try:
        order = _normalize_send_args(*args, **kwargs)
        if not order.get("symbol") or not order.get("qty"):
            return False, "invalid-order: missing symbol/qty"

        kid = (ALP_KEY or "")[:6] + "..." if ALP_KEY else "MISSING"
        _dbg(f"[ORDER DBG] POST {ALP_BASE}/v2/orders as KEY_ID~={kid}")
        r = requests.post(f"{ALP_BASE}/v2/orders", headers=_alp_headers(), json=order, timeout=15)
        info = f"{r.status_code} {r.text[:300]}"
        return (200 <= r.status_code < 300), info
    except Exception as e:
        return False, f"exception: {e}"

def list_positions():
    try:
        r = requests.get(f"{ALP_BASE}/v2/positions", headers=_alp_headers(), timeout=15)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []

def close_all_positions():
    try:
        r = requests.delete(f"{ALP_BASE}/v2/positions", headers=_alp_headers(), timeout=30)
        return 200 <= r.status_code < 300, f"{r.status_code} {r.text[:300]}"
    except Exception as e:
        return False, f"exception: {e}"

def cancel_all_orders():
    try:
        r = requests.delete(f"{ALP_BASE}/v2/orders", headers=_alp_headers(), timeout=20)
        return 200 <= r.status_code < 300, f"{r.status_code} {r.text[:300]}"
    except Exception as e:
        return False, f"exception: {e}"

def get_account_equity(default_equity: float | None = None) -> float:
    """
    Return current Alpaca account equity. On error, fall back to default_equity (if provided),
    else 0.0 so guards can still compute.
    """
    try:
        r = requests.get(f"{ALP_BASE}/v2/account", headers=_alp_headers(), timeout=12)
        if r.status_code != 200:
            return float(default_equity) if default_equity is not None else 0.0
        js = r.json()
        return float(js.get("equity", default_equity if default_equity is not None else 0.0))
    except Exception:
        return float(default_equity) if default_equity is not None else 0.0

def probe_alpaca_auth():
    try:
        import json, os, requests
        base = os.getenv("ALPACA_TRADE_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
        r = requests.get(f"{base}/v2/account", headers=_alp_headers(), timeout=10)
        key_id = (os.getenv("ALPACA_API_KEY_ID","") or "")
        key_hint = f"{key_id[:3]}…{key_id[-3:]}" if len(key_id) >= 7 else "(short)"
        print(f"[PROBE] GET /v2/account -> {r.status_code} key={key_hint} base={base}", flush=True)
        if r.status_code == 200:
            acct = r.json()
            print(f"[PROBE] account_number={acct.get('account_number')} status={acct.get('status')} currency={acct.get('currency')}", flush=True)
        else:
            print(f"[PROBE] body: {r.text[:400]}", flush=True)
    except Exception as e:
        print(f"[PROBE] exception: {type(e).__name__}: {e}", flush=True)
