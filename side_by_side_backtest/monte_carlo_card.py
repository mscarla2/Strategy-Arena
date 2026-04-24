import sys
import random
import json
from pathlib import Path
from datetime import datetime, date, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import zoneinfo

_PKG = Path(__file__).parent
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from side_by_side_backtest.models import RawWatchlist
from side_by_side_backtest.parser import parse_watchlist_post
from side_by_side_backtest.data_fetcher import load_30day_bars, is_banned, fetch_bars_for_entry, refresh_today
from side_by_side_backtest.card_strategy_simulator import simulate_card_strategy

_WL_PATH = _ROOT / "scraped_watchlists.json"

try:
    _ET = zoneinfo.ZoneInfo("America/New_York")
except Exception:
    _ET = timezone(timedelta(hours=-5))

def _load_posts() -> list[dict]:
    if not _WL_PATH.exists():
        print(f"ERROR: {_WL_PATH} not found", file=sys.stderr)
        sys.exit(1)
    return json.loads(_WL_PATH.read_text())

def _load_bars(entry, fetch: bool):
    import pandas as pd
    ticker = entry.ticker
    bars = load_30day_bars(ticker)
    if bars.empty or fetch:
        try:
            if fetch:
                bars = refresh_today(ticker)
            if bars.empty:
                bars = fetch_bars_for_entry(entry) or pd.DataFrame()
        except Exception:
            pass
    return bars

def main():
    print("Loading posts...")
    posts = _load_posts()
    
    all_dates = []
    seen_dates = set()
    
    for p in posts:
        ts_raw = p.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            d  = ts.astimezone(_ET).date()
            if d not in seen_dates:
                seen_dates.add(d)
                all_dates.append(d)
        except Exception:
            continue
            
    all_dates.sort()
    
    all_entries = []
    for p in posts:
        for e in parse_watchlist_post(RawWatchlist(**p)):
            all_entries.append(e)
            
    seen_t = set()
    unique_entries = []
    for e in all_entries:
        if e.ticker != "SPY" and e.ticker not in seen_t and not is_banned(e.ticker):
            seen_t.add(e.ticker)
            unique_entries.append(e)
            
    bars_map = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = {pool.submit(_load_bars, e, False): e for e in unique_entries}
        for fut in as_completed(futs):
            e = futs[fut]
            try:
                b = fut.result()
                if b is not None and not b.empty:
                    if b.index.tzinfo is None:
                        b = b.copy()
                        b.index = b.index.tz_localize("UTC")
                    bars_map[e.ticker] = b
            except Exception:
                pass
                
    all_trades = []
    
    print("Simulating Card Strategy across all dates...")
    for sim_date in all_dates:
        day_matched = []
        for p in posts:
            ts_raw = p.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                if ts.astimezone(_ET).date() == sim_date:
                    day_matched.append(p)
            except Exception:
                continue
                
        if not day_matched:
            continue
            
        day_entries = []
        for p in day_matched:
            day_entries.extend(parse_watchlist_post(RawWatchlist(**p)))
            
        day_seen = set()
        day_unique = []
        for e in day_entries:
            if e.ticker != "SPY" and e.ticker not in day_seen and not is_banned(e.ticker):
                day_seen.add(e.ticker)
                day_unique.append(e)
                
        day_bars = {e.ticker: bars_map[e.ticker] for e in day_unique if e.ticker in bars_map}
        if not day_bars:
            continue
            
        day_trades = simulate_card_strategy(
            day_unique, day_bars,
            min_score=4.3,
            budget_total=5000.0,
            trade_size=500.0,
            max_concurrent=10,
            daily_loss_halt=300.0,
            verbose=False,
            target_date=sim_date,
        )
        all_trades.extend(day_trades)
        
    if not all_trades:
        print("No trades simulated.")
        return
        
    print(f"\nTotal Trades Simulated: {len(all_trades)}")
    
    n_iterations = 1000
    trade_size = 500.0
    
    results = []
    
    for _ in range(n_iterations):
        iter_pnl = 0.0
        iter_wins = 0
        
        for t in all_trades:
            if not t.entry_price:
                continue
                
            slip_entry = random.uniform(0.001, 0.005)
            slip_exit = random.uniform(0.001, 0.005)
            
            effective_entry = t.entry_price * (1 + slip_entry)
            effective_exit = t.exit_price * (1 - slip_exit)
            
            shares = trade_size / effective_entry
            pnl = shares * (effective_exit - effective_entry)
            
            iter_pnl += pnl
            if effective_exit > effective_entry:
                iter_wins += 1
                
        results.append((iter_pnl, iter_wins / len(all_trades)))
        
    pnls = [r[0] for r in results]
    wrs = [r[1] for r in results]
    
    pnls.sort()
    wrs.sort()
    
    mean_pnl = sum(pnls) / n_iterations
    median_pnl = pnls[n_iterations // 2]
    p5_pnl = pnls[int(n_iterations * 0.05)]
    p95_pnl = pnls[int(n_iterations * 0.95)]
    
    mean_wr = sum(wrs) / n_iterations
    median_wr = wrs[n_iterations // 2]
    
    print("\n=== Global Monte Carlo Results for CARD Strategy ===")
    base_pnl = sum((trade_size/t.entry_price)*(t.exit_price - t.entry_price) for t in all_trades if t.entry_price)
    print(f"Base PnL: ${base_pnl:.2f}")
    print(f"Mean PnL: ${mean_pnl:.2f}")
    print(f"Median PnL: ${median_pnl:.2f}")
    print(f"5th Percentile (Worst Case): ${p5_pnl:.2f}")
    print(f"95th Percentile (Best Case): ${p95_pnl:.2f}")
    print(f"\nMean Win Rate: {mean_wr:.1%}")
    print(f"Median Win Rate: {median_wr:.1%}")
    
    loss_count = sum(1 for p in pnls if p < 0)
    print(f"\nProbability of Loss: {loss_count / n_iterations:.1%}")

if __name__ == "__main__":
    main()
