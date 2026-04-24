import sys
from pathlib import Path
from datetime import datetime, timezone

_PKG = Path(__file__).parent
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from side_by_side_backtest.db import WatchlistDB
from side_by_side_backtest.setup_scorer import score_setup
from side_by_side_backtest.data_fetcher import load_30day_bars
from side_by_side_backtest.live_scanner import _load_entries

def main():
    with WatchlistDB() as db:
        # Load all live trades
        trades = db.load_actual_trades(source="live")
        if not trades:
            print("No live trades found in DB.")
            return
            
        print(f"Analyzing {len(trades)} live trades...")
        
        # Load entries from JSON
        json_path = _ROOT / "scraped_watchlists.json"
        entries = _load_entries(str(json_path), today_only=False)
        print(f"Loaded {len(entries)} watchlist entries from JSON.")
        
        passed_trades = []
        filtered_trades = []
        
        for t in trades:
            ticker = t["ticker"]
            entry_ts = datetime.fromisoformat(t["entry_ts"].replace(" ", "T"))
            
            # Load bars up to entry time
            bars = load_30day_bars(ticker)
            if bars.empty:
                print(f"  {ticker}: No bars found.")
                continue
                
            # Filter bars up to entry_ts (completed bars)
            if bars.index.tzinfo is None:
                bars = bars.copy()
                bars.index = bars.index.tz_localize("UTC")
            
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.replace(tzinfo=timezone.utc)
            
            historical_bars = bars[bars.index < entry_ts]
            if len(historical_bars) < 2:
                print(f"  {ticker}: Insufficient historical bars.")
                continue
                
            # Find relevant watchlist entry
            ticker_entries = [e for e in entries if e.ticker == ticker]
            if not ticker_entries:
                print(f"  {ticker}: No watchlist entry found for this ticker.")
                continue
            
            # Pick the entry with most recent post_timestamp before entry_ts
            valid_entries = []
            for e in ticker_entries:
                if e.post_timestamp:
                    pts = e.post_timestamp
                    if isinstance(pts, str):
                        pts = datetime.fromisoformat(pts.replace(" ", "T").replace("+0000", "+00:00"))
                    elif isinstance(pts, datetime):
                        pass
                    else:
                        continue
                    if pts.tzinfo is None:
                        pts = pts.replace(tzinfo=timezone.utc)
                    if pts < entry_ts:
                        valid_entries.append((pts, e))
            
            if not valid_entries:
                entry = ticker_entries[0]
            else:
                valid_entries.sort(key=lambda x: x[0])
                entry = valid_entries[-1][1]
                
            # Run setup scorer
            try:
                sc = score_setup(entry, historical_bars)
            except Exception as exc:
                print(f"  {ticker}: Scoring error: {exc}")
                continue
                
            # Apply FIX logic
            support_ok = getattr(sc, "support_ok", True)
            
            bar_N = historical_bars.iloc[-2]
            bar_Nplus1 = historical_bars.iloc[-1]
            
            body_low_N = min(float(bar_N["open"]), float(bar_N["close"]))
            close_Nplus1 = float(bar_Nplus1["close"])
            
            support = getattr(sc, "support", 0.0)
            
            touch_ok = True
            if support and support > 0:
                if not (body_low_N <= support * 1.02):
                    touch_ok = False
                if not (close_Nplus1 >= support * 0.90):
                    touch_ok = False
                    
            pattern_ok = getattr(sc, "pattern_score", 0.0) > 0
            
            if support_ok and touch_ok and pattern_ok:
                passed_trades.append(t)
            else:
                filtered_trades.append(t)
                
        # Calculate metrics
        print("\n=== Comparison Results ===")
        print(f"Total Actual Trades: {len(trades)}")
        print(f"Passed Trades (with fix): {len(passed_trades)}")
        print(f"Filtered Trades (blocked by fix): {len(filtered_trades)}")
        
        live_pnl = sum(t["pnl_dollar"] for t in trades)
        mitigated_pnl = sum(t["pnl_dollar"] for t in passed_trades)
        
        live_wins = [t for t in trades if t["outcome"] == "win"]
        mitigated_wins = [t for t in passed_trades if t["outcome"] == "win"]
        
        print(f"\nBaseline Live PnL: ${live_pnl:.2f} (WR: {len(live_wins)/len(trades):.1%})")
        wr_mitigated = len(mitigated_wins)/len(passed_trades) if passed_trades else 0.0
        print(f"Mitigated Live PnL: ${mitigated_pnl:.2f} (WR: {wr_mitigated:.1%})")
        print(f"Saved Losses: ${sum(t['pnl_dollar'] for t in filtered_trades if t['outcome'] == 'loss'):.2f}")

if __name__ == "__main__":
    main()
