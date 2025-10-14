# common/signal_bridge.py
import os, json, requests
from datetime import datetime, timezone

# ---- ENV / BASE ----
ALP_BASE = (os.getenv("ALPACA_TRADE_BASE_URL", "https://paper-api.alpaca.markets") or "").rstrip("/")
ALP_KEY  = (os.getenv("ALPACA_API_KEY_ID", "") or "").strip()
ALP_SEC  = (os.getenv("ALPACA_SECRET_KEY", "") or "").strip()

def _alp_headers():
    return {
        "APCA-API-KEY-ID": ALP_KEY,
        "APCA-API-SECRET-KEY": ALP_SEC,
        "Content-Type": "application/json",
    }

def _safe_head(s: str, n=4):
    s = s or ""
    return (s[:n] + "…" + s[-3:]) if len(s) > (n + 3) else s

# --------------------------------------------------------------------
# AUTH PROBE (handy in boot logs)
# --------------------------------------------------------------------
def probe_alpaca_auth():
    try:
        url = f"{ALP_BASE}/v2/account"
        r = requests.get(url, headers=_alp_headers(), timeout=12)
        key_hint = _safe_head(ALP_KEY, 3)
        print(f"[PROBE] GET /v2/account -> {r.status_code} key={key_hint} base={ALP_BASE}", flush=True)
        if r.status_code == 200:
            js = r.json()
            acct = js.get("account_number", "?")
            status = js.get("status", "?")
            ccy = js.get("currency", "?")
            print(f"[PROBE] account_number={acct} status={status} currency={ccy}", flush=True)
        else:
            print(f"[PROBE] body: {(r.text or '')[:400]}", flush=True)
    except Exception as e:
        print(f"[PROBE] exception: {type(e).__name__}: {e}", flush=True)

# --------------------------------------------------------------------
# CORE SEND (accepts payload dict *or* (symbol, sig, strategy_tag))
# --------------------------------------------------------------------
def send_to_broker(symbol_or_payload, maybe_sig=None, strategy_tag=None):
    """
    Compatible with both call signatures:
      - send_to_broker(symbol, sig, strategy_tag=?)
      - send_to_broker(payload_dict)

    Expected fields (we normalize internally):
      ticker, action, orderType, quantity, price (optional),
      takeProfit / tp_abs, stopLoss / sl_abs, entry (optional), meta
    """
    # Normalize payload
    if isinstance(symbol_or_payload, dict):
        payload_in = dict(symbol_or_payload)
    else:
        symbol = symbol_or_payload
        sig = maybe_sig or {}
        payload_in = {
            "ticker": symbol,
            "action": sig.get("action"),
            "orderType": sig.get("orderType", "market"),
            "quantity": int(sig.get("quantity", 0)),
            "price": sig.get("limitPrice", sig.get("price")),
            "takeProfit": sig.get("tp_abs", sig.get("takeProfit")),
            "stopLoss": sig.get("sl_abs", sig.get("stopLoss")),
            "entry": sig.get("entry"),
            "meta": sig.get("meta", {}),
        }
        if strategy_tag:
            payload_in.setdefault("meta", {})
            payload_in["meta"]["strategy"] = strategy_tag

    try:
        symbol = payload_in["ticker"]
        act    = (payload_in.get("action") or "").lower()
        side   = "buy" if act in ("buy", "buy_to_cover") else "sell"
        otype  = (payload_in.get("orderType") or "market").lower()
        qty    = int(payload_in.get("quantity", 0))
        entry  = payload_in.get("entry")
        tp     = payload_in.get("takeProfit")
        sl     = payload_in.get("stopLoss")

        order = {
            "symbol": symbol,
            "side": side,
            "type": "market" if otype == "market" else "limit",
            "time_in_force": "day",
            "extended_hours": False,
            "qty": qty,
        }

        # limit entry price if provided
        if order["type"] == "limit" and payload_in.get("price") is not None:
            order["limit_price"] = float(round(float(payload_in["price"]), 2))

        # Enforce >= $0.01 spacing when we know entry
        if entry is not None:
            try:
                base = float(entry)
                if tp is not None:
                    tp = float(tp)
                    # move TP away by 0.01 in the favorable direction if needed
                    if side == "buy"  and (tp - base) < 0.01:
                        tp = round(base + 0.01, 2)
                    if side == "sell" and (base - tp) < 0.01:
                        tp = round(base - 0.01, 2)
                if sl is not None:
                    sl = float(sl)
                    if side == "buy"  and (base - sl) < 0.01:
                        sl = round(base - 0.01, 2)
                    if side == "sell" and (sl - base) < 0.01:
                        sl = round(base + 0.01, 2)
            except Exception:
                pass

        # Bracket (both) vs OTO (single exit)
        if tp is not None and sl is not None:
            order["order_class"] = "bracket"
            order["take_profit"] = {"limit_price": float(round(tp, 2))}
            order["stop_loss"]   = {"stop_price":  float(round(sl, 2))}
        elif tp is not None or sl is not None:
            order["order_class"] = "oto"
            if tp is not None:
                order["take_profit"] = {"limit_price": float(round(tp, 2))}
            if sl is not None:
                order["stop_loss"]   = {"stop_price":  float(round(sl, 2))}

        # POST
        url = f"{ALP_BASE}/v2/orders"
        r = requests.post(url, headers=_alp_headers(), json=order, timeout=15)

        if 200 <= r.status_code < 300:
            return True, f"{r.status_code} {r.text[:300]}"

        # 401 diagnostics
        if r.status_code == 401:
            key_hint = _safe_head(ALP_KEY, 4)
            diag = {
                "where": "POST /v2/orders",
                "status": r.status_code,
                "base": ALP_BASE,
                "key_prefix": key_hint,
                "symbol": symbol,
                "side": side,
                "type": order.get("type"),
                "order_class": order.get("order_class", "simple"),
                "has_tp": "take_profit" in order,
                "has_sl": "stop_loss" in order,
                "body": (r.text or "")[:400],
            }
            print(f"[AUTH-401] {json.dumps(diag, separators=(',',':'))}", flush=True)
        else:
            print(f"[ORDER-ERR] {r.status_code} {(r.text or '')[:400]}", flush=True)

        return False, f"{r.status_code} {r.text[:300]}"

    except Exception as e:
        return False, f"exception: {e}"

# --------------------------------------------------------------------
# POSITIONS / FLATTEN / CANCELS
# --------------------------------------------------------------------
def list_positions():
    try:
        r = requests.get(f"{ALP_BASE}/v2/positions", headers=_alp_headers(), timeout=15)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []

def close_all_positions():
    """
    Flatten all positions. Returns (ok: bool, info: str).
    (If callers assign only to `ok`, it's still truthy in prints.)
    """
    try:
        r = requests.delete(f"{ALP_BASE}/v2/positions", headers=_alp_headers(), timeout=30)
        ok = 200 <= r.status_code < 300
        return ok, f"{r.status_code} {(r.text or '')[:300]}"
    except Exception as e:
        return False, f"exception: {e}"

def cancel_all_orders():
    """
    Cancel all open orders. Returns (ok: bool, info: str).
    """
    try:
        r = requests.delete(f"{ALP_BASE}/v2/orders", headers=_alp_headers(), timeout=20)
        ok = 200 <= r.status_code < 300
        return ok, f"{r.status_code} {(r.text or '')[:300]}"
    except Exception as e:
        return False, f"exception: {e}"

# --------------------------------------------------------------------
# ACCOUNT EQUITY (used by broker guard)
# --------------------------------------------------------------------
def get_account_equity(default_equity: float):
    try:
        r = requests.get(f"{ALP_BASE}/v2/account", headers=_alp_headers(), timeout=12)
        if r.status_code != 200:
            return default_equity
        js = r.json()
        return float(js.get("equity", default_equity))
    except Exception:
        return default_equity
