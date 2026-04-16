"""
Analyse how often bar closes fall below the computed support level for each ticker.

For each ticker with bars in the 30d cache:
  - Resolve the support level (watchlist or computed)
  - For bars where body_low touches support (within 0.5%), measure close vs support
  - Report the median close-below-support % and how many touches were invalidated
    by Phase B at each tolerance (0.2%, 1%, 2%, 3%)

Output: sorted table showing tickers with the most "support zone closing" behaviour.
"""
import sys
import statistics
sys.path.insert(0, '.')

from side_by_side_backtest.db import WatchlistDB
from side_by_side_backtest.data_fetcher import load_30day_bars
from side_by_side_backtest.simulator import _resolve_support

db = WatchlistDB().connect()
entries = db.load_entries()

# One entry per ticker
seen: dict = {}
for e in entries:
    if e.ticker not in seen:
        seen[e.ticker] = e

results = []

for ticker, e in seen.items():
    bars = load_30day_bars(ticker)
    if bars is None or bars.empty:
        continue

    support, src = _resolve_support(e, bars, silent=True)
    if not support or support <= 0:
        continue

    if e.post_timestamp:
        bars_w = bars[bars.index >= e.post_timestamp]
    else:
        bars_w = bars
    if len(bars_w) < 10:
        continue

    # Find bars that body-touch support (Phase A would trigger)
    body_lows = bars_w[['open', 'close']].min(axis=1)
    touch_mask = body_lows <= support * 1.005
    touch_bars = bars_w[touch_mask]

    if touch_bars.empty:
        continue

    # For each touch bar, measure close vs support
    closes = touch_bars['close'].values
    close_pcts = [(c - support) / support * 100 for c in closes]  # negative = below support

    n_touches = len(closes)
    n_below = sum(1 for p in close_pcts if p < 0)       # any close below
    n_below_02 = sum(1 for p in close_pcts if p < -0.2)  # below 0.2% (old Phase B threshold)
    n_below_1  = sum(1 for p in close_pcts if p < -1.0)  # below 1%
    n_below_2  = sum(1 for p in close_pcts if p < -2.0)  # below 2%
    n_below_3  = sum(1 for p in close_pcts if p < -3.0)  # below 3%

    if n_touches == 0:
        continue

    median_pct = statistics.median(close_pcts)
    min_pct = min(close_pcts)

    results.append({
        'ticker': ticker,
        'src': src[:3],
        'support': round(support, 3),
        'n_touches': n_touches,
        'n_below': n_below,
        'pct_below': round(n_below / n_touches * 100),
        'killed_02': n_below_02,   # would be killed by old 0.2% Phase B
        'killed_1': n_below_1,
        'killed_2': n_below_2,
        'killed_3': n_below_3,
        'median_close_pct': round(median_pct, 2),
        'worst_close_pct': round(min_pct, 2),
    })

# Sort by % of touches where close is below support
results.sort(key=lambda x: -x['pct_below'])

print(f"\n{'Ticker':8s} {'Src':3s} {'Sup':>7} {'Touches':>7} {'Below%':>6} "
      f"{'Kill@0.2%':>9} {'Kill@1%':>7} {'Kill@2%':>7} {'Kill@3%':>7} "
      f"{'Med%':>6} {'Worst%':>7}")
print("-" * 90)

for r in results:
    if r['n_touches'] < 5:
        continue
    print(
        f"{r['ticker']:8s} {r['src']:3s} {r['support']:>7.3f} {r['n_touches']:>7} "
        f"{r['pct_below']:>5}% "
        f"{r['killed_02']:>8} ({r['killed_02']/r['n_touches']*100:>4.0f}%) "
        f"{r['killed_1']:>6} ({r['killed_1']/r['n_touches']*100:>3.0f}%) "
        f"{r['killed_2']:>6} ({r['killed_2']/r['n_touches']*100:>3.0f}%) "
        f"{r['killed_3']:>6} ({r['killed_3']/r['n_touches']*100:>3.0f}%) "
        f"{r['median_close_pct']:>+6.2f}% {r['worst_close_pct']:>+7.2f}%"
    )

print(f"\nTotal tickers analysed: {len(results)}")
print(f"Tickers where >50% of touches close below support: "
      f"{sum(1 for r in results if r['pct_below'] > 50)}")

# Summary: what Phase B threshold would keep 80% of touches valid?
for thresh_pct in [0.2, 0.5, 1.0, 1.5, 2.0, 3.0]:
    total_touches = sum(r['n_touches'] for r in results)
    if thresh_pct == 0.2:
        killed = sum(r['killed_02'] for r in results)
    elif thresh_pct == 1.0:
        killed = sum(r['killed_1'] for r in results)
    elif thresh_pct == 2.0:
        killed = sum(r['killed_2'] for r in results)
    elif thresh_pct == 3.0:
        killed = sum(r['killed_3'] for r in results)
    else:
        # approximate
        killed = sum(
            sum(1 for c in [load_30day_bars(r['ticker'])['close'].values]
                if False)  # skip approximation
            for r in results
        )
        continue
    kept = total_touches - killed
    print(f"Phase B @{thresh_pct:.1f}% below support: {kept}/{total_touches} touches kept "
          f"({kept/total_touches*100:.0f}%),  {killed} killed ({killed/total_touches*100:.0f}%)")

db.close()
