"""
Quick timing script — run with:
    python side_by_side_backtest/_time_brief.py

Measures:
  1. Disk-only path (load_30day_bars + score_setup) — what the first render now does.
  2. Background refresh wall-clock (fires in parallel, non-blocking for the UI).
  3. Blocking refresh wall-clock (what subsequent fragment ticks do).
"""
import json, sys, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from side_by_side_backtest.parser import parse_watchlist_post
from side_by_side_backtest.models import RawWatchlist
from side_by_side_backtest.data_fetcher import load_30day_bars, refresh_today
from side_by_side_backtest.setup_scorer import score_setup

data = json.loads(open("scraped_watchlists.json").read())

def _ts(p):
    return p.get("timestamp") or ""

latest_ts = max(_ts(p) for p in data)
latest_posts = [p for p in data if _ts(p) == latest_ts]

all_entries = []
for post in latest_posts:
    rw = RawWatchlist(**post)
    all_entries.extend(parse_watchlist_post(rw))

unique = list({e.ticker for e in all_entries})

print(f"\nLatest post: {latest_ts}")
print(f"Entries: {len(all_entries)}  |  Unique tickers: {len(unique)}")

# ── 1. Disk-only (first-render path after fix) ────────────────────────────────
print(f"\n{'─'*64}")
print("FIRST RENDER (disk-only, no network wait):")
print(f"{'Ticker':8s}  {'load_30d':>10s}  {'score_setup':>12s}  {'total':>8s}  {'bars':>6s}")
print("-" * 64)

grand_t0 = time.perf_counter()
for entry in all_entries:
    t0 = time.perf_counter()
    bars = load_30day_bars(entry.ticker)
    t1 = time.perf_counter()
    sc = score_setup(entry, bars)
    t2 = time.perf_counter()
    print(f"{entry.ticker:8s}  {1000*(t1-t0):8.1f}ms  {1000*(t2-t1):10.1f}ms  {1000*(t2-t0):6.1f}ms  {len(bars):6d}")

disk_total = 1000 * (time.perf_counter() - grand_t0)
print(f"\n✅ First render total (sequential): {disk_total:.0f}ms")

# ── 2. Background refresh (fire-and-forget, runs while UI renders) ────────────
print(f"\n{'─'*64}")
print("BACKGROUND REFRESH (daemon thread, non-blocking):")
bg_done = threading.Event()
bg_start = time.perf_counter()

def _bg():
    with ThreadPoolExecutor(max_workers=15) as pool:
        futs = [pool.submit(refresh_today, t, "schwab_data") for t in unique]
        for f in as_completed(futs):
            try: f.result()
            except Exception: pass
    bg_done.set()

t = threading.Thread(target=_bg, daemon=True)
t.start()
print(f"  Thread started — UI is unblocked immediately.")
print(f"  Waiting for background refresh to finish...")
bg_done.wait(timeout=30)
bg_elapsed = 1000 * (time.perf_counter() - bg_start)
print(f"  Background refresh completed in: {bg_elapsed:.0f}ms (invisible to user on first render)")

# ── 3. Blocking refresh (subsequent fragment ticks) ───────────────────────────
print(f"\n{'─'*64}")
print("SUBSEQUENT TICK (blocking refresh + re-score):")
t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=15) as pool:
    futs = [pool.submit(refresh_today, t, "schwab_data") for t in unique]
    for f in as_completed(futs):
        try: f.result()
        except Exception: pass
t1 = time.perf_counter()
refresh_ms = 1000 * (t1 - t0)

# Score again after refresh
t2 = time.perf_counter()
for entry in all_entries:
    bars = load_30day_bars(entry.ticker)
    score_setup(entry, bars)
t3 = time.perf_counter()
score_ms = 1000 * (t3 - t2)

print(f"  Blocking refresh ({len(unique)} tickers parallel): {refresh_ms:.0f}ms")
print(f"  Scoring (disk, parallel-ready):                  {score_ms:.0f}ms")
print(f"  Total subsequent tick:                           {refresh_ms + score_ms:.0f}ms")

print(f"\n{'═'*64}")
print(f"SUMMARY")
print(f"  First render (was blocking, now instant):  ~{disk_total:.0f}ms  (was ~{bg_elapsed:.0f}ms+)")
print(f"  Background refresh completes after:        ~{bg_elapsed:.0f}ms  (invisible to user)")
print(f"  Subsequent 60s ticks:                      ~{refresh_ms + score_ms:.0f}ms")
