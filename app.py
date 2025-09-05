# TEMP: one-time TradersPost test to verify wiring end-to-end
import os, json, requests
from datetime import datetime, timezone
import sys

TP_URL = os.getenv("TP_WEBHOOK_URL")

def redacted(u):
    if not u: return "(missing)"
    # keep domain, hide token-like tail
    return u[:40] + "…"

print("Starting one-time TP test…", flush=True)
print("Webhook URL (redacted):", redacted(TP_URL), flush=True)

if not TP_URL:
    print("ERROR: TP_WEBHOOK_URL is not set in Render → Environment.", flush=True)
    sys.exit(1)

payload = {
    "ticker": "MSFT",
    "action": "buy",          # enter long
    "orderType": "market",
    "quantity": 1,            # tiny paper test
    # optional bracket fields (uncomment to test)
    # "takeProfit": 9999.0,
    # "stopLoss":   0.01,
    "meta": {
        "environment": "paper",
        "sentAt": datetime.now(timezone.utc).isoformat(),
        "note": "render-one-time-test"
    }
}

print("POSTing payload:", json.dumps(payload), flush=True)

try:
    r = requests.post(TP_URL, json=payload, timeout=10)
    print("Response:", r.status_code, r.text[:400], flush=True)
except Exception as e:
    print("Request failed:", type(e).__name__, str(e), flush=True)

print("Done. Replace this file with your real loop when verified.", flush=True)
