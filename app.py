# app.py  — live loop for paper trading via TradersPost
import os, time, json, hashlib, requests
from datetime import datetime, timezone

TP_URL = os.getenv("TP_WEBHOOK_URL")
PAPER_MODE = True
POLL_SECONDS = 10  # how often to check signals

# ====== YOUR TOP COMBOS ======
COMBOS = [
    # (strategy_name, symbol, timeframe_minutes, quantity)
    ("liquidity_sweep", "AAPL", 3, 50),
    # add the rest of your winners…
]

# ====== SIGNAL LOGIC: RETURN dict OR None ======
# Fill this in to call your real strategy functions
def compute_signal(strategy_name, symbol, tf_minutes):
    """
    Return a dict like:
      {
        "action": "buy"|"sell"|"exit"|"add",
        "orderType": "market"|"limit"|"stop",
        "price": None or float,
        "takeProfit": None or float,
        "stopLoss": None or float,
        "clientTag": "optional-string",
        "barTime": "2025-09-05T13:59:00Z"  # optional, for dedupe
      }
    or return None if no signal right now.
    """
    # TODO: call your strategy code here and build the dict above.
    return None

# ====== SENDER + DEDUPE ======
_seen = set()

def _dedupe_key(payload: dict) -> str:
    # include barTime/clientTag if you set them to ensure once-per-bar
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()

def build_payload(symbol: str, qty: int, sig: dict):
    p = {
        "ticker": symbol,
        "action": sig["action"],
        "orderType": sig.get("orderType", "market"),
        "quantity": qty,
        "meta": {
            "environment": "paper" if PAPER_MODE else "live",
            "sentAt": datetime.now(timezone.utc).isoformat()
        }
    }
    if sig.get("price") is not None:      p["price"] = float(sig["price"])
    if sig.get("takeProfit") is not None: p["takeProfit"] = float(sig["takeProfit"])
    if sig.get("stopLoss") is not None:   p["stopLoss"] = float(sig["stopLoss"])
    if sig.get("clientTag"):              p["clientTag"] = sig["clientTag"]
    if sig.get("timeInForce"):            p["timeInForce"] = sig["timeInForce"]
    return p

def send(payload: dict):
    if not TP_URL:
        raise RuntimeError("TP_WEBHOOK_URL is not set in Render → Environment.")

    key = _dedupe_key(payload)
    if key in _seen:
        return False, "duplicate"

    r = requests.post(TP_URL, json=payload, timeout=10)
    ok = 200 <= r.status_code < 300
    if ok: _seen.add(key)
    return ok, f"{r.status_code} {r.text[:200]}"

# ====== MAIN LOOP ======
def main():
    print("Bot starting…", flush=True)
    while True:
        loop_start = time.time()
        print("Bot loop tick…", flush=True)

        for (name, sym, tf, qty) in COMBOS:
            sig = compute_signal(name, sym, tf)
            if not sig: 
                continue
            payload = build_payload(sym, qty, sig)
            ok, info = send(payload)
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{stamp}] {name} {sym} {tf}m -> {sig['action']} | {info}", flush=True)

        time.sleep(max(1, POLL_SECONDS - int(time.time() - loop_start)))

if __name__ == "__main__":
    main()
