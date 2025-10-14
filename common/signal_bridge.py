# common/signal_bridge.py  — Alpaca plumbing (robust TP/SL normalization)

import os
import json
import requests
from datetime import datetime, timezone

ALP_TRADE_BASE = os.getenv("ALPACA_TRADE_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
ALP_KEY        = os.getenv("ALPACA_API_KEY_ID", "")
ALP_SEC        = os.getenv("ALPACA_SECRET_KEY", "")

def _alp_headers():
    return {
        "APCA-API-KEY-ID": ALP_KEY,
        "APCA-API-SECRET-KEY": ALP_SEC,
        "Content-Type": "application/json",
    }

# ---------------------------
# Helpers
# ---------------------------
def _to_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x))
    except Exception:
        return None

def _extract_tp(payload: dict):
    """
    Accepts any of:
      - payload["takeProfit"] = 123.45
      - payload["takeProfit"] = {"limitPrice": 123.45}
      - payload["tp_abs"] = 123.45
    Returns a float or None.
    """
    if "takeProfit" in payload:
        tp = payload["takeProfit"]
        if isinstance(tp, dict):
            return _to_float(tp.get("limitPrice") or tp.get("price"))
        return _to_float(tp)
    if "tp_abs" in payload:
        return _to_float(payload["tp_abs"])
    return None

def _extract_sl(payload: dict):
    """
    Accepts any of:
      - payload["stopLoss"] = 120.10
      - payload["stopLoss"] = {"stopPrice": 120.10}
      - payload["sl_abs"] = 120.10
    Returns a float or None.
    """
    if "stopLoss" in payload:
        sl = payload["stopLoss"]
        if isinstance(sl, dict):
            return _to_float(sl.get("stopPrice") or sl.get("price"))
        return _to_float(sl)
    if "sl_abs" in payload:
        return _to_float(payload["sl_abs"])
    return None

def _extract_limit(payload: dict):
    """
    Limit entry price if orderType == 'limit'.
    Accepts payload["limitPrice"] or payload["price"] (number/string).
    """
    lp = payload.get("limitPrice", payload.get("price"))
    return _to_float(lp)

def _side_from_action(action: str):
    a = (action or "").lower()
    if a in ("buy", "buy_to_cover"):
        return "buy"
    # normalize both "sell" and "sell_short" to "sell" for entry (Alpaca uses qty + side)
    return "sell"

def _payload_from_args(symbol_or_payload, sig=None, strategy_tag=None):
    """
    Supports both call styles:
      - send_to_broker(symbol, sig, strategy_tag=None)
      - send_to_broker(payload_dict)
    Returns unified payload dict with keys: ticker, action, orderType, quantity, entry, takeProfit/stopLoss (numeric or dict ok).
    """
    if isinstance(symbol_or_payload, dict) and sig is None:
        # Already a full payload
        payload = dict(symbol_or_payload)  # shallow copy
        if "ticker" not in payload and "symbol" in payload:
            payload["ticker"] = payload["symbol"]
        return payload

    # Build payload from (symbol, sig)
    symbol = symbol_or_payload
    sig = sig or {}
    payload = {
        "ticker": symbol,
        "action": sig.get("action"),
        "orderType": sig.get("orderType", "market"),
        "quantity": int(sig.get("quantity", 0)),
        "entry": _to_float(sig.get("entry") or sig.get("price") or sig.get("lastClose")),
        # Preserve both absolute and nested variants if provided
        "takeProfit": sig.get("takeProfit") if "takeProfit" in sig else sig.get("tp_abs"),
        "stopLoss":   sig.get("stopLoss")   if "stopLoss"   in sig else sig.get("sl_abs"),
        "limitPrice": sig.get("limitPrice") or sig.get("price"),
        "meta": {},
    }
    if isinstance(sig.get("meta"), dict):
        payload["meta"].update(sig["meta"])
    if strategy_tag:
        payload["meta"]["strategy"] = strategy_tag
    if sig.get("barTime"):
        payload["meta"]["barTime"] = sig["barTime"]
    return payload

# ---------------------------
# Public functions (used by apps)
# ---------------------------
def send_to_broker(symbol_or_payload, sig=None, strategy_tag=None):
    """
    Places an order via Alpaca REST.
    - Accepts both calling styles (see _payload_from_args).
    - Normalizes TP/SL whether numeric or dict.
    - Enforces Alpaca 1¢ min distance from base price.
    """
    try:
        payload = _payload_from_args(symbol_or_payload, sig, strategy_tag)

        symbol = payload["ticker"]
        action = payload.get("action")
        side   = _side_from_action(action)
        otype  = (payload.get("orderType") or "market").lower()
        qty    = int(payload.get("quantity", 0))

        order = {
            "symbol": symbol,
            "side": side,
            "type": "market" if otype == "market" else "limit",
            "time_in_force": "day",
            "qty": qty,
        }

        if order["type"] == "limit":
            lp = _extract_limit(payload)
            if lp is not None:
                order["limit_price"] = float(round(lp, 2))

        # Normalize TP/SL regardless of shape
        tp = _extract_tp(payload)
        sl = _extract_sl(payload)

        # Enforce 1¢ distance if we know base price (entry or limit)
        base_price = _to_float(payload.get("entry") or payload.get("price") or order.get("limit_price"))
        if base_price is not None:
            if tp is not None and (tp - base_price) < 0.01:
                tp = round(base_price + 0.01, 2)
            if sl is not None:
                if side == "buy" and (base_price - sl) < 0.01:
                    sl = round(base_price - 0.01, 2)
                if side == "sell" and (sl - base_price) < 0.01:
                    sl = round(base_price + 0.01, 2)

        if tp is not None or sl is not None:
            order["order_class"] = "bracket"
            if tp is not None:
                order["take_profit"] = {"limit_price": float(round(tp, 2))}
            if sl is not None:
                order["stop_loss"] = {"stop_price": float(round(sl, 2))}

        r = requests.post(f"{ALP_TRADE_BASE}/v2/orders", headers=_alp_headers(), json=order, timeout=15)
        return (200 <= r.status_code < 300), f"{r.status_code} {r.text[:300]}"

    except Exception as e:
        return False, f"exception: {e}"

def list_positions():
    r = requests.get(f"{ALP_TRADE_BASE}/v2/positions", headers=_alp_headers(), timeout=15)
    return r.json() if r.status_code == 200 else []

def close_all_positions():
    r = requests.delete(f"{ALP_TRADE_BASE}/v2/positions", headers=_alp_headers(), timeout=25)
    ok = 200 <= r.status_code < 300
    info = f"{r.status_code} {r.text[:300]}"
    return ok, info

def cancel_all_orders():
    r = requests.delete(f"{ALP_TRADE_BASE}/v2/orders", headers=_alp_headers(), timeout=15)
    ok = 200 <= r.status_code < 300
    info = f"{r.status_code} {r.text[:300]}"
    return ok, info

def get_account_equity(default_equity: float):
    r = requests.get(f"{ALP_TRADE_BASE}/v2/account", headers=_alp_headers(), timeout=12)
    if r.status_code != 200:
        return default_equity
    try:
        return float(r.json().get("equity", default_equity))
    except Exception:
        return default_equity

def probe_alpaca_auth():
    """Optional boot probe for keys + trade host."""
    try:
        key_hint = f"{ALP_KEY[:3]}…{ALP_KEY[-3:]}" if ALP_KEY else "<empty>"
        r = requests.get(f"{ALP_TRADE_BASE}/v2/account", headers=_alp_headers(), timeout=10)
        print(f"[PROBE] GET /v2/account -> {r.status_code} key={key_hint} base={ALP_TRADE_BASE}", flush=True)
        if r.status_code == 200:
            js = r.json()
            acct = js.get("account_number")
            status = js.get("status")
            ccy = js.get("currency")
            print(f"[PROBE] account_number={acct} status={status} currency={ccy}", flush=True)
        else:
            print(f"[PROBE] body: {r.text[:500]}", flush=True)
    except Exception as e:
        print(f"[PROBE] exception: {type(e).__name__}: {e}", flush=True)
