import os, time, requests, uuid

ALPACA_TRADE_BASE_URL = os.getenv("ALPACA_TRADE_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")

def _headers():
    return {
        "APCA-API-KEY-ID": os.getenv("ALPACA_KEY_ID", ""),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY", ""),
        "Content-Type": "application/json",
    }

def get_account_equity(default_equity: float) -> float:
    try:
        r = requests.get(f"{ALPACA_TRADE_BASE_URL}/v2/account", headers=_headers(), timeout=12)
        if r.status_code != 200:
            return default_equity
        js = r.json()
        eq = js.get("equity") or js.get("cash") or default_equity
        return float(eq)
    except Exception:
        return default_equity

def list_positions():
    try:
        r = requests.get(f"{ALPACA_TRADE_BASE_URL}/v2/positions", headers=_headers(), timeout=15)
        if r.status_code != 200:
            return []
        return r.json()
    except Exception:
        return []

def close_position(symbol: str, qty: float | int | None = None):
    try:
        r = requests.delete(f"{ALPACA_TRADE_BASE_URL}/v2/positions/{symbol}", headers=_headers(), timeout=15)
        return 200 <= r.status_code < 300
    except Exception:
        return False

def close_all_positions():
    try:
        r = requests.delete(f"{ALPACA_TRADE_BASE_URL}/v2/positions", headers=_headers(), timeout=30)
        return 200 <= r.status_code < 300
    except Exception:
        return False

def _alpaca_side(action: str) -> str:
    a = (action or "").lower()
    if a in ("sell_short", "short", "sell_to_open"):
        return "sell"
    if a in ("buy_to_cover", "cover", "buy_to_close"):
        return "buy"
    if a in ("exit", "flatten", "close"):
        return ""
    return a

def _tif():
    return os.getenv("ALPACA_TIME_IN_FORCE", "day")

def send_signal(symbol: str, sig: dict, strategy_tag: str = "") -> tuple[bool, str]:
    action = sig.get("action", "")
    side = _alpaca_side(action)
    if not side:
        return True, "noop-exit"
    qty = int(sig.get("quantity", 0))
    if qty <= 0:
        return False, "qty<=0"

    order_type = (sig.get("orderType", "market") or "market").lower()
    client_id = sig.get("client_order_id") or f"{strategy_tag}.{symbol}.{int(time.time())}.{uuid.uuid4().hex[:6]}"

    body = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": order_type,
        "time_in_force": _tif(),
        "client_order_id": client_id
    }

    tp_abs = sig.get("tp_abs") or sig.get("takeProfit")
    sl_abs = sig.get("sl_abs") or sig.get("stopLoss")
    if tp_abs or sl_abs:
        body["order_class"] = "bracket"
        if tp_abs:
            body["take_profit"] = { "limit_price": round(float(tp_abs), 2) }
        if sl_abs:
            body["stop_loss"] = { "stop_price": round(float(sl_abs), 2) }

    if order_type == "limit" and sig.get("price") is not None:
        body["limit_price"] = round(float(sig["price"]), 2)

    try:
        r = requests.post(f"{ALPACA_TRADE_BASE_URL}/v2/orders", headers=_headers(), json=body, timeout=12)
        ok = 200 <= r.status_code < 300
        return ok, f"{r.status_code} {r.text[:300]}"
    except Exception as e:
        return False, f"exception: {e}"
