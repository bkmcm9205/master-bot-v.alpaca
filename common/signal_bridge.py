# common/signal_bridge.py  (drop-in replacement for send_to_broker)
import os, requests, json
from datetime import datetime, timezone

ALP_BASE = os.getenv("ALPACA_TRADE_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
ALP_KEY  = (os.getenv("ALPACA_API_KEY_ID", "") or "").strip()
ALP_SEC  = (os.getenv("ALPACA_SECRET_KEY", "") or "").strip()

def _alp_headers():
    return {
        "APCA-API-KEY-ID": ALP_KEY,
        "APCA-API-SECRET-KEY": ALP_SEC,
        "Content-Type": "application/json",
    }

def _safe_head(s: str, n=3):
    return (s[:n] + "…" + s[-3:]) if s and len(s) > (n+3) else s

def send_to_broker(symbol_or_payload, maybe_sig=None, strategy_tag=None):
    """
    Compatible with both call styles:
      - send_to_broker(symbol, sig, strategy_tag=?)
      - send_to_broker(payload_dict)

    Payload shape expected (normalized inside):
      {
        "ticker": "AAPL",
        "action": "buy" | "sell" | "sell_short",
        "orderType": "market" | "limit",
        "quantity": int,
        "price": Optional[float],      # for limit entries
        "takeProfit": Optional[float], # absolute TP price
        "stopLoss": Optional[float],   # absolute SL stop price
        "entry": Optional[float],      # used to enforce 0.01 leg spacing
        "meta": {...}
      }
    """
    # ---- normalize input to a payload dict
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
            payload_in["meta"] = payload_in.get("meta", {})
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

        # limit entry (if specified)
        if order["type"] == "limit" and payload_in.get("price") is not None:
            order["limit_price"] = float(round(float(payload_in["price"]), 2))

        # Enforce Alpaca’s >= $0.01 distances when we have a base entry price
        if entry is not None:
            try:
                base = float(entry)
                if tp is not None and (float(tp) - base) < 0.01:
                    tp = round(base + 0.01, 2) if side == "buy" else round(base - 0.01, 2)
                if sl is not None:
                    if side == "buy" and (base - float(sl)) < 0.01:
                        sl = round(base - 0.01, 2)
                    if side == "sell" and (float(sl) - base) < 0.01:
                        sl = round(base + 0.01, 2)
            except Exception:
                pass

        # Order class: bracket (both exits), OTO (single child)
        if tp is not None and sl is not None:
            order["order_class"] = "bracket"
            order["take_profit"] = {"limit_price": float(round(float(tp), 2))}
            order["stop_loss"]   = {"stop_price":  float(round(float(sl), 2))}
        elif tp is not None or sl is not None:
            # One leg → use OTO. (Alpaca accepts OTO with a single child.)
            order["order_class"] = "oto"
            if tp is not None:
                order["take_profit"] = {"limit_price": float(round(float(tp), 2))}
            if sl is not None:
                order["stop_loss"] = {"stop_price": float(round(float(sl), 2))}

        # Fire
        url = f"{ALP_BASE}/v2/orders"
        r = requests.post(url, headers=_alp_headers(), json=order, timeout=15)

        # Success path
        if 200 <= r.status_code < 300:
            return True, f"{r.status_code} {r.text[:300]}"

        # Detailed diagnostics on 401
        if r.status_code == 401:
            key_hint = _safe_head(ALP_KEY, n=4)
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
            # Log compact failure info for other 4xx/5xx
            print(f"[ORDER-ERR] {r.status_code} {r.text[:400]}", flush=True)

        return False, f"{r.status_code} {r.text[:300]}"

    except Exception as e:
        return False, f"exception: {e}"
