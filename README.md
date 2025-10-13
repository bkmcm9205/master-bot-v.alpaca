# master-bot-v.alpaca (paper)
Plumbing only swap to Alpaca. Models/guards unchanged.

Data: adapters/data_alpaca.py
Orders/Equity/Flatten: common/signal_bridge.py -> adapters/broker_alpaca.py

Edits done:
- fetch_polygon_1m -> fetch_bars_1m
- send_to_traderspost -> send_to_broker
- tp_flatten_all -> close_all_positions
- tp_get_equity -> get_account_equity
- tp_list_positions -> list_positions
