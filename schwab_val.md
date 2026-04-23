# Schwab Live Validation Guide

Follow these steps to authenticate with the Schwab API and execute your test trade.

### 1. Complete the OAuth Handshake
Run this command in your terminal to authenticate with Schwab:
```bash
python3 -m side_by_side_backtest.schwab_broker --auth
```
*A browser window will open. Log in, approve access, and wait for the success message in the terminal.*

### 2. Retrieve Your Account Hash
If you haven't added the `SCHWAB_ACCOUNT_HASH` to your `.env`, pull it by running:
```bash
python3 -m side_by_side_backtest.schwab_broker --get-account-hash
```
*Copy the resulting hash and ensure it is saved in your `.env` file.*

### 3. Flip to Live Mode
Open `side_by_side_backtest/autonomous_config.py` and change:
```python
paper_mode:        bool = False
```

### 4. Place the Test Order
To buy 1 share of QQQ, save and run a temporary scratch file with this snippet:
```python
from side_by_side_backtest.schwab_broker import SchwabBroker
from side_by_side_backtest.autonomous_config import CONFIG

# Double check paper mode is disabled
CONFIG.paper_mode = False

broker = SchwabBroker(CONFIG)
res = broker.place_order(ticker="QQQ", side="buy", quantity=1)

print(f"Status: {res.status}")
print(f"Order ID: {res.order_id}")
```
