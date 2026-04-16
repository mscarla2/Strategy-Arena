"""Debug: count support touches and attempts for NO_ENTRY tickers."""
import sys
sys.path.insert(0, '.')
from side_by_side_backtest.db import WatchlistDB
from side_by_side_backtest.data_fetcher import load_30day_bars
from side_by_side_backtest.simulator import _resolve_support, _build_pattern_map, simulate_entry
from side_by_side_backtest.models import WatchlistEntry

db = WatchlistDB().connect()
entries = db.load_entries()

check = ['ARAI', 'TURB', 'UCAR', 'AIXI', 'ONDS', 'DEFT', 'EONR']

seen = {}
for e in entries:
    if e.ticker not in check:
        continue
    if e.ticker not in seen:
        seen[e.ticker] = e

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

    # Count body touches manually (same logic as Phase A)
    touches = 0
    for idx in range(len(bars_w)):
        row = bars_w.iloc[idx]
        body_low = min(float(row['open']), float(row['close']))
        if body_low <= support * 1.005:
            touches += 1

    # Run with unlimited attempts to see how many trades would fire
    trades_unlimited = simulate_entry(
        e, bars_w,
        profit_target_pct=5.0, stop_loss_pct=1.0, max_loss_pct=5.0,
        pattern_lookback=5,
        max_entry_attempts=0,  # unlimited
        precomputed_support=support,
        precomputed_support_source=src,
    )

    trades_capped = simulate_entry(
        e, bars_w,
        profit_target_pct=5.0, stop_loss_pct=1.0, max_loss_pct=5.0,
        pattern_lookback=5,
        max_entry_attempts=10,  # current default
        precomputed_support=support,
        precomputed_support_source=src,
    )

    print(f"{ticker:8s}: touches={touches}  trades_unlimited={len(trades_unlimited)}  trades_capped10={len(trades_capped)}  support={support:.3f}")
    if trades_unlimited:
        wins = sum(1 for t in trades_unlimited if t.outcome == 'win')
        losses = sum(1 for t in trades_unlimited if t.outcome == 'loss')
        total_pnl = sum(t.pnl_pct for t in trades_unlimited)
        print(f"         unlimited: W={wins} L={losses}  total_pnl={total_pnl:+.1f}%")

db.close()
