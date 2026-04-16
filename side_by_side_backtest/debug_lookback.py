"""Debug: test different pattern_lookback values for NO_ENTRY tickers."""
import sys
sys.path.insert(0, '.')
from side_by_side_backtest.db import WatchlistDB
from side_by_side_backtest.data_fetcher import load_30day_bars
from side_by_side_backtest.simulator import _resolve_support, simulate_entry

db = WatchlistDB().connect()
entries = db.load_entries()

check = ['AIXI', 'TURB', 'UCAR', 'DEFT', 'EONR', 'ARAI', 'SNAL', 'ONDS']

seen = {}
for e in entries:
    if e.ticker not in check:
        continue
    if e.ticker not in seen:
        seen[e.ticker] = e

print(f"{'Ticker':8s} | {'lb=3':>8} | {'lb=5':>8} | {'lb=7':>8} | {'lb=10':>8} | {'lb=15':>8}")
print("-" * 60)

for ticker in sorted(seen):
    e = seen[ticker]
    bars = load_30day_bars(ticker)
    if bars is None or bars.empty:
        continue
    support, src = _resolve_support(e, bars, silent=True)
    if not support:
        continue
    if e.post_timestamp:
        bars_w = bars[bars.index >= e.post_timestamp]
    else:
        bars_w = bars
    if bars_w.empty:
        continue

    row = [ticker]
    for lb in [3, 5, 7, 10, 15]:
        trades = simulate_entry(
            e, bars_w,
            profit_target_pct=5.0, stop_loss_pct=1.0, max_loss_pct=5.0,
            pattern_lookback=lb,
            max_entry_attempts=0,
            precomputed_support=support,
            precomputed_support_source=src,
        )
        wins = sum(1 for t in trades if t.outcome == 'win')
        losses = sum(1 for t in trades if t.outcome == 'loss')
        total_pnl = sum(t.pnl_pct for t in trades)
        row.append(f"{len(trades)}t({wins}W/{losses}L) {total_pnl:+.0f}%")

    print(f"{row[0]:8s} | {row[1]:>8} | {row[2]:>8} | {row[3]:>8} | {row[4]:>8} | {row[5]:>8}")

db.close()
