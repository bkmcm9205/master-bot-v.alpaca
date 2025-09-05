# --- TEMP TEST: send one buy once, then never again ---
_test_sent = False

def compute_signal(strategy_name, symbol, tf_minutes):
    global _test_sent
    if _test_sent:
        return None  # no more signals after the first test

    # fire exactly once to verify end-to-end (adjust symbol/qty as you like)
    if strategy_name == "liquidity_sweep" and symbol == "AAPL":
        _test_sent = True
        return {
            "action": "buy",
            "orderType": "market",
            "price": None,
            "takeProfit": None,  # you can put a number here to test TP
            "stopLoss":  None    # or here to test SL
        }
    return None
