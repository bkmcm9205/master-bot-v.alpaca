# common/signal_bridge.py  (PLUMBING ONLY)
import os, requests
from datetime import datetime, timezone

# Base + creds (kept compatible with your env var names)
ALP_BASE = os.getenv("ALPACA_TRADE_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
ALP_KEY  = os.getenv("ALPACA_API_KEY_ID", "").strip()
ALP_SEC  = os.getenv("ALPACA_SECRET_KEY", "").strip()

def _alp_headers():
    return {
        "APCA-API-KEY-ID": ALP_KEY,
        "APCA-API-SECRET-KEY": ALP_SEC,
        "Content-Type": "application/json",
    }

def send_to_broker(payload: dict):
    """
    Input (your existing internal/TP-style):
      ticker, action, orderType, quantity,
      takeProfit.limitPrice, stopLoss.stopPrice, (optional) limitPrice/price, entry
    Output: Alpaca order; attaches bracket legs if provided.
    """
    try:
        symbol = payload["ticker"]
        act    = (payload.get("action") or "").lower()
        side   = "buy" if act in ("buy","buy_to_cover") else "sell"
        otype  = (payload.get("orderType") or "market").lower()
        qty    = int(payload.get("quantity", 0))

        if qty <= 0:
            return False, "invalid-qty"

        order = {
            "symbol": symbol,
            "side": side,
            "type": "market" if otype == "market" else "limit",
            "time_in_force": "day",
            "qty": qty,
        }

        if order["type"] == "limit":
            lp = payload.get("limitPrice", payload.get("price"))
            if lp is not None:
                order["limit_price"] = float(round(lp, 2))

        # Bracket legs
        tp_obj = payload.get("takeProfit", {}) or {}
        sl_obj = payload.get("stopLoss",   {}) or {}
        tp = tp_obj.get("limitPrice")
        sl = sl_obj.get("stopPrice")

        # Enforce Alpaca’s 1¢ distance rule if we have a base price
        base_price = payload.get("entry") or payload.get("price")
        if base_price is not None:
            base_price = float(base_price)
            if tp is not None and (tp - base_price) < 0.01:
                tp = round(base_price + 0.01, 2)
            if sl is not None:
                if side == "buy"  and (base_price - sl) < 0.01:
                    sl = round(base_price - 0.01, 2)
                if side == "sell" and (sl - base_price) < 0.01:
                    sl = round(base_price + 0.01, 2)

        if tp is not None or sl is not None:
            order["order_class"] = "bracket"
            if tp is not None:
                order["take_profit"] = {"limit_price": float(round(tp, 2))}
            if sl is not None:
                order["stop_loss"]   = {"stop_price":  float(round(sl, 2))}

        r = requests.post(f"{ALP_BASE}/v2/orders", headers=_alp_headers(), json=order, timeout=15)
        return (200 <= r.status_code < 300), f"{r.status_code} {r.text[:300]}"

    except Exception as e:
        return False, f"exception: {e}"

def cancel_all_orders(timeout: int = 15):
    """
    Cancel ALL open orders at Alpaca.
    Returns (ok: bool, info: str).
    """
    try:
        r = requests.delete(f"{ALP_BASE}/v2/orders", headers=_alp_headers(), timeout=timeout)
        ok = 200 <= r.status_code < 300
        return ok, f"{r.status_code} {r.text[:300]}"
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
        r = requests.delete(f"{ALP_BASE}/v2/positions", headers=_alp_headers(), timeout=25)
        return 200 <= r.status_code < 300, f"{r.status_code} {r.text[:300]}"
    except Exception as e:
        return False, f"exception: {e}"

def get_account_equity(default_equity: float):
    try:
        r = requests.get(f"{ALP_BASE}/v2/account", headers=_alp_headers(), timeout=12)
        if r.status_code != 200:
            return default_equity
        return float(r.json().get("equity", default_equity))
    except Exception:
        return default_equity
