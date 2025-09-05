# app.py
# requirements: requests, python-dotenv (optional), psycopg2-binary (optional if you read from Postgres)
import os, time, json, hashlib, requests, random
from datetime import datetime, timezone

# ========= 1) BASIC CONFIG =========
TRADERSPOST_WEBHOOK_URL = os.getenv("TP_WEBHOOK_URL")  # paste your Strategy Webhook URL in Render env vars
PAPER_MODE = True          # keep True during paper
POLL_SECONDS = 10          # how often to check for signals

# List your winners here: (strategy_name, symbol, timeframe_minutes, quantity)
COMBOS = [
    ("liquidity_sweep", "AAPL", 3, 50),
    ("poc_pinescript_version", "MSFT", 5, 40),
    ("ict_bos_fvg", "NVDA", 1, 25),
    # add more...
]

# ========= 2) YOUR SIGNAL LOGIC =========
def compute_signal(strategy_name, symbol, tf_minutes):
    """
    Return a dict when a signal appears, or None.
    Expected keys:
      action: 'buy' | 'add' | 'sell' | 'exit'
      orderType: 'market' | 'limit' | 'stop' (start with 'market' if unsure)
      price: optional float for limit/stop entries
      takeProfit: optional float (TP level)
      stopLoss: optional float (SL level)
    Replace this stub with YOUR real logic. For now, it does nothing.
    """
    # TODO: hook into your actual strategy functions here.
    # Example shape:
    # return {
    #   "action": "buy",
    #   "orderType": "market",
    #   "price": None,
    #   "takeProfit": 123.45,
    #   "stopLoss": 117.80
    # }
    return None

# ========= 3) SENDING HELPERS =========
_seen = set()

def _dedupe(payload: dict) -> str:
    js = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(js.encode()).hexdigest()

def build_payload(symbol: str, qty: int, sig: dict):
    """
    Minimal webhook body for TradersPost with TP/SL.
    You can add timeInForce, reduceOnly, clientTag, etc. later.
    """
    p = {
        "ticker": symbol,
        "action": sig["action"],        # 'buy'|'add'|'sell'|'exit'
        "orderType": sig.get("orderType", "market"),
        "quantity": qty
    }
    if sig.get("price") is not None:
        p["price"] = float(sig["price"])
    if sig.get("takeProfit") is not None:
        p["takeProfit"] = float(sig["takeProfit"])
    if sig.get("stopLoss") is not None:
        p["stopLoss"] = float(sig["stopLoss"])

    # Let the signal override strategy defaults in TradersPost (if enabled in TP):
    # p["timeInForce"] = "day"  # or 'gtc'
    # p["reduceOnly"] = False
    # p["clientTag"] = "paper-test"

    # Optional metadata for your own audit
    p["meta"] = {
        "environment": "paper" if PAPER_MODE else "live",
        "sentAt": datetime.now(timezone.utc).isoformat()
    }
    return p

def send_webhook(payload: dict, max_retries: int = 3, backoff_seconds: int = 2):
    if not TRADERSPOST_WEBHOOK_URL:
        raise RuntimeError("Missing TP_WEBHOOK_URL environment variable.")

    key = _dedupe(payload)
    if key in _seen:
        return False, "duplicate"

    tries = 0
    last = None
    while tries <= max_retries:
        tries += 1
        try:
            r = requests.post(TRADERSPOST_WEBHOOK_URL, json=payload, timeout=10)
            if 200 <= r.status_code < 300:
                _seen.add(key)
                return True, f"{r.status_code} {r.text[:200]}"
            last = f"{r.status_code} {r.text[:200]}"
        except Exception as e:
            last = f"error {type(e).__name__}: {e}"

        time.sleep(backoff_seconds * tries)  # simple backoff
    return False, last or "unknown error"

# ========= 4) MAIN LOOP =========
def main():
    print("▶ Running Python → TradersPost (paper) loop...")
    while True:
        loop_started = time.time()

        for (strategy_name, symbol, tf_min, qty) in COMBOS:
            sig = compute_signal(strategy_name, symbol, tf_min)
            if not sig:
                continue  # no signal right now

            payload = build_payload(symbol, qty, sig)
            ok, info = send_webhook(payload)
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{stamp}] {strategy_name} {symbol} {tf_min}m -> {sig['action']} | {info}")

        elapsed = time.time() - loop_started
        time.sleep(max(1, POLL_SECONDS - int(elapsed)))

if __name__ == "__main__":
    main()
