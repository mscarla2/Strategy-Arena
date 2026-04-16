"""Debug: why do known tickers show NO_ENTRY in the new simulation?"""
import sys
sys.path.insert(0, '.')
from side_by_side_backtest.db import WatchlistDB
from side_by_side_backtest.data_fetcher import load_30day_bars
from side_by_side_backtest.simulator import _resolve_support, _build_pattern_map

db = WatchlistDB().connect()
entries = db.load_entries()

check_tickers = ['ARAI', 'SAFX', 'FUSE', 'TURB', 'STAK', 'GAME', 'UCAR', 'AIXI', 'DEFT', 'EONR', 'SNAL', 'ONDS', 'ATPC']

# Gather one entry per ticker (first occurrence)
seen = {}
for e in entries:
    if e.ticker not in check_tickers:
        continue
    if e.ticker not in seen:
        seen[e.ticker] = e

for ticker in sorted(seen):
    e = seen[ticker]
    bars = load_30day_bars(ticker)
    if bars is None or bars.empty:
        print(f"{ticker:8s}: NO BARS IN CACHE")
        continue

    support, src = _resolve_support(e, bars, silent=True)

    if e.post_timestamp:
        bars_w = bars[bars.index >= e.post_timestamp]
    else:
        bars_w = bars

    if bars_w.empty:
        print(f"{ticker:8s}: EMPTY POST-TS WINDOW (post_ts={e.post_timestamp})")
        continue

    current = float(bars_w['close'].iloc[-1])
    n_bars = len(bars_w)

    if not support or support <= 0:
        print(f"{ticker:8s}: NO SUPPORT RESOLVED  current={current:.3f}  wl_sup={e.support_level}")
        continue

    body_lows = bars_w[['open', 'close']].min(axis=1)
    touched = (body_lows <= support * 1.005).any()

    # Risk filter check
    risk_pct = (current - support) / current * 100 if current > 0 else 999

    # Pattern check
    pm, pi = _build_pattern_map(bars_w, support)

    # Sub-dollar check
    median_open = float(bars_w['open'].median())

    print(
        f"{ticker:8s}: sup={support:.3f}({src[:3]})  cur={current:.3f}  "
        f"risk={risk_pct:+.1f}%  touched={touched}  "
        f"patterns={len(pi)}  bars={n_bars}  median_open={median_open:.3f}  "
        f"wl_sup={e.support_level}"
    )

db.close()
