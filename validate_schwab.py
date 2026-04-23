"""
Schwab API Validation Script
=============================
Places a live order (buy or sell) for 1 share of QQQ to confirm the Schwab
API is wired up correctly.

Usage:
  python3 validate_schwab.py              # defaults to sell (close the QQQ bought earlier)
  python3 validate_schwab.py --side buy
  python3 validate_schwab.py --side sell
  python3 validate_schwab.py --ticker SPY --side buy --qty 1

Prerequisites (see schwab_val.md for full setup steps):
  1. Run OAuth handshake:
       python3 -m side_by_side_backtest.schwab_broker --auth
  2. Confirm SCHWAB_ACCOUNT_HASH is set in .env:
       python3 -m side_by_side_backtest.schwab_broker --get-account-hash
  3. Run this script:
       python3 validate_schwab.py
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# Load .env before importing broker so env vars are present for pre-flight checks
_ENV_PATH = Path(".env")
if _ENV_PATH.exists():
    for _line in _ENV_PATH.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

from side_by_side_backtest.schwab_broker import SchwabBroker
from side_by_side_backtest.autonomous_config import CONFIG

# ── CLI args ──────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="Schwab API validation — place a test order")
_parser.add_argument("--side",   default="sell", choices=["buy", "sell"],
                     help="Order direction (default: sell)")
_parser.add_argument("--ticker", default="QQQ",
                     help="Ticker symbol (default: QQQ)")
_parser.add_argument("--qty",    default=1, type=int,
                     help="Number of shares (default: 1)")
_args = _parser.parse_args()

SIDE   = _args.side
TICKER = _args.ticker.upper()
QTY    = _args.qty

# ── Safety check ──────────────────────────────────────────────────────────────
print("=" * 60)
print(f"  Schwab API Validation — {SIDE.upper()} {QTY}×{TICKER} (live order)")
print("=" * 60)
print()

# ── Pre-flight: check token file is readable/writable by this process ─────────
_TOKEN_FILE = Path("side_by_side_backtest/.schwab_tokens.json")
if _TOKEN_FILE.exists():
    if not os.access(_TOKEN_FILE, os.R_OK | os.W_OK):
        stat = _TOKEN_FILE.stat()
        import pwd, grp
        owner = pwd.getpwuid(stat.st_uid).pw_name
        print(f"❌  Permission error on token file: {_TOKEN_FILE}")
        print(f"    Owner: {owner}  |  Mode: {oct(stat.st_mode)[-3:]}")
        print()
        print("Fix with:")
        print(f'    sudo chown "$USER" "{_TOKEN_FILE}" && chmod 600 "{_TOKEN_FILE}"')
        print()
        sys.exit(1)
else:
    print(f"⚠️  Token file not found at {_TOKEN_FILE}")
    print("    Run the OAuth handshake first:")
    print("    python3 -m side_by_side_backtest.schwab_broker --auth")
    print()
    sys.exit(1)

# ── Pre-flight: SCHWAB_ACCOUNT_HASH ──────────────────────────────────────────
_account_hash = os.environ.get("SCHWAB_ACCOUNT_HASH", "").strip()
if not _account_hash:
    print("❌  SCHWAB_ACCOUNT_HASH is not set in your .env file.")
    print()
    print("Retrieve it with:")
    print("    python3 -m side_by_side_backtest.schwab_broker --get-account-hash")
    print("Then add to .env:")
    print("    SCHWAB_ACCOUNT_HASH=<hash from above>")
    print()
    sys.exit(1)

# ── Pre-flight: required credential vars ─────────────────────────────────────
for _var in ("SCHWAB_CLIENT_ID", "SCHWAB_CLIENT_SECRET"):
    if not os.environ.get(_var, "").strip():
        print(f"❌  {_var} is not set in your .env file.")
        sys.exit(1)

# Force live mode for this validation run
CONFIG.paper_mode = False

if CONFIG.paper_mode:
    # Should never reach here, but guard anyway
    print("ERROR: paper_mode is still True — aborting.")
    sys.exit(1)

print(f"⚠️  paper_mode = False  →  THIS WILL PLACE A REAL {SIDE.upper()} ORDER for {QTY}×{TICKER}")
confirm = input("Type 'yes' to continue, anything else to abort: ").strip().lower()
if confirm != "yes":
    print("Aborted.")
    sys.exit(0)

print()
print("Connecting to Schwab…")

# ── Place the order ───────────────────────────────────────────────────────────
broker = SchwabBroker(CONFIG)
res = broker.place_order(ticker=TICKER, side=SIDE, quantity=QTY)

# ── Report ────────────────────────────────────────────────────────────────────
print()
print("─" * 40)
print(f"  Side     : {SIDE.upper()}")
print(f"  Ticker   : {TICKER}")
print(f"  Qty      : {QTY}")
print(f"  Status   : {res.status}")
print(f"  Order ID : {res.order_id or '(not returned)'}")
if res.fill_price:
    print(f"  Fill $   : {res.fill_price:.4f}")
if res.message:
    print(f"  Message  : {res.message}")
print("─" * 40)
print()

if res.status in ("pending", "filled"):
    print("✅  Order accepted by Schwab — API validation PASSED.")
elif res.status == "error":
    print("❌  Order returned an error — check credentials and account hash.")
    sys.exit(1)
else:
    print(f"⚠️  Unexpected status '{res.status}' — review manually.")
