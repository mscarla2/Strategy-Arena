"""
Refresh Cache — Daily 30-Day Rolling OHLCV Maintenance
=======================================================
Reads all tickers from scraped_watchlists.json (or a supplied list),
then for each ticker:

  • First run (no 30d parquet exists) → fetches full 30-day window
  • Subsequent runs                   → fetches today only, merges, prunes oldest day

Run this once before market open each morning (after the scraper).

Usage
-----
    # Seed/refresh all watchlist tickers (reads scraped_watchlists.json)
    python -m side_by_side_backtest.refresh_cache

    # Refresh specific tickers only
    python -m side_by_side_backtest.refresh_cache --tickers UGRO ANNA TURB

    # Force full 30-day re-fetch for all (e.g. after long holiday gap)
    python -m side_by_side_backtest.refresh_cache --full

    # Use Alpaca instead of yfinance
    python -m side_by_side_backtest.refresh_cache --provider alpaca
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

_PKG = Path(__file__).parent
_DEFAULT_JSON = _PKG.parent / "scraped_watchlists.json"


def _tickers_from_json(json_path: str) -> List[str]:
    """Extract unique tickers from scraped_watchlists.json via the parser."""
    from .models import RawWatchlist
    from .parser import parse_watchlist_post

    try:
        raw = json.loads(Path(json_path).read_text())
    except Exception as exc:
        print(f"[refresh_cache] Cannot load {json_path}: {exc}")
        return []

    seen: set[str] = set()
    for post in raw:
        try:
            entries = parse_watchlist_post(RawWatchlist(**post))
            for e in entries:
                seen.add(e.ticker)
        except Exception:
            pass
    return sorted(seen)


def refresh_all(
    tickers: List[str],
    provider: str = "yfinance",
    full: bool = False,
    delay: float = 0.4,
    max_workers: int = 6,
) -> None:
    """
    Refresh 30-day cache for each ticker in *tickers* using a thread pool.

    Parameters
    ----------
    max_workers : int
        Concurrent download threads (default 6).  yfinance is tolerant of
        modest concurrency; keep ≤ 8 to avoid transient 429s.
    delay : float
        Per-worker polite sleep after each fetch (seconds).
    """
    from .data_fetcher import (
        fetch_30day_bars,
        refresh_today,
        _30d_path,
        is_banned,
    )

    total = len(tickers)
    print(f"[refresh_cache] {'Full re-fetch' if full else 'Daily refresh'} "
          f"for {total} ticker(s) via {provider} "
          f"({min(max_workers, total)} workers)")

    # Partition into banned (skip immediately) and active work
    to_process: List[str] = []
    skipped = 0
    for ticker in tickers:
        if is_banned(ticker):
            print(f"  {ticker:8} BANNED — skip")
            skipped += 1
        else:
            to_process.append(ticker)

    ok = errors = 0
    _lock = __import__("threading").Lock()  # guard print ordering

    def _refresh_one(ticker: str) -> tuple[str, int, str]:
        """Fetch one ticker; return (ticker, bar_count, status)."""
        needs_full = full or not _30d_path(ticker).exists()
        try:
            if needs_full:
                df = fetch_30day_bars(ticker, provider=provider)
            else:
                df = refresh_today(ticker, provider=provider)
            if delay > 0:
                time.sleep(delay)
            return ticker, len(df), "ok"
        except Exception as exc:
            if delay > 0:
                time.sleep(delay)
            return ticker, 0, f"ERROR: {exc}"

    n_workers = min(max_workers, len(to_process)) if to_process else 1
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_refresh_one, t): t for t in to_process}
        completed = 0
        for fut in as_completed(futures):
            ticker, bars, status = fut.result()
            completed += 1
            mode_tag = "full 30d" if (full or not _30d_path(ticker).exists()) else "today"
            with _lock:
                if status == "ok":
                    print(f"  [{completed}/{len(to_process)}] {ticker:8} → {mode_tag}  {bars} bars  ✓")
                    ok += 1
                else:
                    print(f"  [{completed}/{len(to_process)}] {ticker:8} → {mode_tag}  {status}")
                    errors += 1

    print(f"\n[refresh_cache] Done — {ok} OK, {skipped} skipped, {errors} errors.")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="refresh_cache",
        description="Seed/refresh 30-day rolling OHLCV cache for watchlist tickers.",
    )
    p.add_argument("--watchlist", default=str(_DEFAULT_JSON),
                   help="Path to scraped_watchlists.json")
    p.add_argument("--tickers", nargs="+", metavar="TICKER",
                   help="Override ticker list (skip JSON parsing)")
    p.add_argument("--provider", default="yfinance",
                   choices=["yfinance", "alpaca"])
    p.add_argument("--full", action="store_true",
                   help="Force full 30-day re-fetch even if parquet exists")
    p.add_argument("--delay", type=float, default=0.4,
                   help="Per-worker sleep after each fetch (default 0.4s)")
    p.add_argument("--workers", type=int, default=6,
                   help="Parallel download threads (default 6)")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.tickers:
        tickers = [t.upper().lstrip("$") for t in args.tickers]
    else:
        tickers = _tickers_from_json(args.watchlist)

    if not tickers:
        print("[refresh_cache] No tickers found. Exiting.")
        sys.exit(1)

    refresh_all(tickers, provider=args.provider, full=args.full,
                delay=args.delay, max_workers=args.workers)
