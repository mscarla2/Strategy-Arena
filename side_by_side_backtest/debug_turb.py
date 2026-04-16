"""Debug TURB specifically — where are the patterns vs where is the post_timestamp window."""
import sys
sys.path.insert(0, '.')
from side_by_side_backtest.db import WatchlistDB
from side_by_side_backtest.data_fetcher import load_30day_bars
from side_by_side_backtest.simulator import _resolve_support, _build_pattern_map

db = WatchlistDB().connect()
entries = db.load_entries()

# Get all TURB entries
turb_entries = [e for e in entries if e.ticker == 'TURB']
print(f"TURB entries: {len(turb_entries)}")
for e in turb_entries[:5]:
    print(f"  post_ts={e.post_timestamp}  support={e.support_level}  resistance={e.resistance_level}")

bars = load_30day_bars('TURB')
print(f"\nFull bars window: {bars.index[0]} → {bars.index[-1]}  ({len(bars)} bars)")

# Check what patterns look like on full bars
support, src = _resolve_support(turb_entries[0], bars, silent=True)
print(f"Resolved support={support} src={src}")

pm_full, pi_full = _build_pattern_map(bars, support)
print(f"Patterns on FULL bars: {len(pi_full)}")
if pi_full:
    sorted_ts = sorted(pi_full)
    print(f"  Pattern timestamps: {sorted_ts[0]} → {sorted_ts[-1]}")

# Check each entry's post_timestamp window
for e in turb_entries[:3]:
    if e.post_timestamp:
        bars_w = bars[bars.index >= e.post_timestamp]
        print(f"\nEntry post_ts={e.post_timestamp}:")
        print(f"  bars_w window: {bars_w.index[0] if not bars_w.empty else 'EMPTY'} → {bars_w.index[-1] if not bars_w.empty else ''} ({len(bars_w)} bars)")
        
        # Count how many pattern timestamps are in this window
        patterns_in_window = sum(1 for ts in pi_full if ts in bars_w.index)
        print(f"  Pattern timestamps in this window: {patterns_in_window} / {len(pi_full)}")
        
        # Count support touches in window
        if not bars_w.empty:
            body_lows = bars_w[['open','close']].min(axis=1)
            touches = (body_lows <= support * 1.005).sum()
            print(f"  Support touches in window: {touches}")

db.close()
