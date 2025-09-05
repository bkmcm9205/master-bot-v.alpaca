# Trading Bot Runbook

## Components
- **GitHub** → stores code (`app.py`, etc.)
- **Render (Background Worker)** → runs `python app.py` 24/7
- **Polygon** → market data (API key in env vars)
- **TradersPost** → sends orders to broker (Paper now, Live later)

---

## Stop Immediately
- **Suspend worker**: Render → Service → Settings → Suspend (or Scale to 0).  
- **Flip DRY_RUN**: Render → Environment → `DRY_RUN=1` → Save → Deploy.  
- **Disable strategy**: TradersPost → Strategy → set Inactive.  

*(Fastest: Suspend worker OR disable strategy.)*

---

## Pause / Resume
- **Pause**: set `DRY_RUN=1` OR disable strategy in TradersPost.  
- **Resume**: set `DRY_RUN=0` (or unset) AND ensure strategy is Active in TradersPost.

---

## Cold Start Checklist
1. **TradersPost**
   - Copy **Paper webhook URL**.  
   - Make sure strategy is Active.  

2. **Polygon**
   - Confirm valid API key.  

3. **GitHub**
   - Latest code committed to the branch Render uses.  

4. **Render**
   - Start command: `python app.py`  
   - Env vars:  
     - `TP_WEBHOOK_URL` = TradersPost webhook  
     - `POLYGON_API_KEY` = your key  
     - `DRY_RUN=0` for real paper orders (1 = dry run)  
     - `SEND_TEST_ORDER=0` (set 1 only for startup test)  
     - `REPLAY_ON_START=0` (use only for replay testing)  
   - Manual Deploy → Deploy latest.  
   - Logs should show `Bot starting…` then `Tick…`.

---

## Go Live (Real Money)
1. TradersPost → Live strategy → copy **Live webhook**.  
2. Render → Env → update `TP_WEBHOOK_URL`.  
3. Confirm `DRY_RUN=0`.  
4. Manual Deploy.  
5. TradersPost → strategy Active.  
6. Monitor first fills.

---

## Daily Safety
- All strategies **flatten at session end (16:00 ET)**.  
- To block tomorrow’s orders, set `DRY_RUN=1` after close or disable strategy in TradersPost.

---

## Troubleshooting
- **Only “Tick…”** → likely off-hours or no signals. Use `DEBUG_COMBO` for one combo.  
- **Webhook test** → set `SEND_TEST_ORDER=1` to fire a test order (`200` = success).  
- **Env change ignored** → always Manual Deploy after editing env vars.  
- **Emergency stop** → Suspend worker or disable strategy.
