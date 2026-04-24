import pandas as pd
from datetime import datetime
from simulator import simulate_entry
from models import WatchlistEntry, SessionType

# Create dummy data
# Bar 0: open=105, close=101 (touches support=100)
# Bar 1: open=90, close=91 (massive gap down)
# Bar 2: open=91, low=90, high=92, close=91
dates = pd.date_range("2026-04-20 14:00", periods=3, freq="5min", tz="UTC")
bars = pd.DataFrame({
    "open": [105, 90, 91],
    "high": [105, 91, 92],
    "low":  [100, 90, 90],
    "close":[101, 91, 91]
}, index=dates)

entry = WatchlistEntry(
    ticker="TEST",
    post_timestamp=dates[0],
    session_type=SessionType.MARKET_OPEN,
    support_level=100.0,
    resistance_level=110.0
)

# Mock pattern match to force entry
pm = {dates[0]: type("Pattern", (), {"confidence_score": 1.0, "pattern_type": "strict"})()}
pidx = {dates[0]}

res = simulate_entry(
    entry, bars,
    precomputed_support=100.0,
    precomputed_support_source="computed",
    precomputed_pattern_map=pm,
    precomputed_pattern_indices=pidx
)

for r in res:
    print(f"Outcome: {r.outcome}, Entry: {r.entry_price}, Exit: {r.exit_price}, PnL: {r.pnl_pct}%, Support OK: {r.support_respected}")

