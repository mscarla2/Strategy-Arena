"""
Live Scanner — Real-Time Entry Alerts
======================================
Polls every POLL_INTERVAL seconds for each ticker in today's watchlist entries.
When price touches the support level AND the Side-by-Side pattern fires,
an alert is emitted via:
  • Terminal  — coloured banner (ANSI)
  • Desktop   — macOS osascript notification
  • Sound     — system bell (\\a)

Usage
-----
    # Scan watchlist from default JSON, poll every 5 minutes
    python -m side_by_side_backtest.live_scanner

    # Custom path and interval
    python -m side_by_side_backtest.live_scanner \\
        --watchlist scraped_watchlists.json --interval 300
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

_PKG = Path(__file__).parent
_DEFAULT_JSON  = _PKG.parent / "scraped_watchlists.json"
_POLL_INTERVAL = 300          # seconds between full scans
_TOUCH_BAND    = 0.005        # body within 0.5% of support = "touched" (body-based, not wick)
_PATTERN_BARS  = 30           # last N bars to check for the pattern

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_RED    = "\033[91m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _banner(sc) -> None:
    """Print a coloured ANSI alert banner to stdout."""
    now = datetime.now(tz=timezone.utc).strftime("%H:%M ET")
    next_bar = datetime.now(tz=timezone.utc).strftime("%H:%M ET")  # approximate
    sep = "═" * 42
    print(f"\n{_RED}{_BOLD}╔{sep}╗{_RESET}")
    print(f"{_RED}{_BOLD}║  🚨 ENTRY SIGNAL FIRED — {now:<16}║{_RESET}")
    print(f"{_RED}{_BOLD}║  {sc.ticker:<6}|  Support: ${sc.support:<10.2f}        ║{_RESET}")
    print(f"{_RED}{_BOLD}║  Pattern: Side-by-Side White Lines       ║{_RESET}")
    print(f"{_RED}{_BOLD}║  Score: {sc.score:.1f}/10  |  R/R: {sc.rr_ratio:.1f}:1{' '*14}║{_RESET}")
    print(f"{_RED}{_BOLD}║  → Buy at next bar open (~{next_bar})  ║{_RESET}")
    print(f"{_RED}{_BOLD}╚{sep}╝{_RESET}\n")
    sys.stdout.flush()


def _desktop_notify(sc) -> None:
    """Fire a macOS notification via osascript (no-op on non-macOS)."""
    if sys.platform != "darwin":
        return
    msg  = f"Score {sc.score}/10 | R/R {sc.rr_ratio:.1f}:1 | Support ${sc.support:.2f}"
    title = f"🚨 ENTRY SIGNAL — {sc.ticker}"
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{msg}" with title "{title}" sound name "Glass"'],
            check=False, timeout=5,
        )
    except Exception:
        pass


def _sound() -> None:
    """Play effect.mp3 via afplay (macOS), fallback to system bell."""
    _mp3 = Path(__file__).parent / "effect.mp3"
    if sys.platform == "darwin" and _mp3.exists():
        try:
            subprocess.run(["afplay", str(_mp3)], check=False, timeout=5)
            return
        except Exception:
            pass
    print("\a", end="", flush=True)


# ---------------------------------------------------------------------------
# Watchlist loading
# ---------------------------------------------------------------------------

def _load_entries(json_path: str) -> list:
    """Parse today's watchlist entries from the JSON file."""
    from .models import RawWatchlist
    from .parser import parse_watchlist_post

    try:
        raw = json.loads(Path(json_path).read_text())
    except Exception as exc:
        print(f"[scanner] Cannot load {json_path}: {exc}")
        return []

    entries = []
    for post in raw:
        try:
            entries.extend(parse_watchlist_post(RawWatchlist(**post)))
        except Exception:
            pass
    return entries


# ---------------------------------------------------------------------------
# Per-ticker check
# ---------------------------------------------------------------------------

def _check_ticker(entry) -> Optional[object]:
    """
    Fetch fresh 5-min bars, check support touch + pattern.
    Returns a SetupScore if an entry signal fires, else None.
    """
    from .data_fetcher import fetch_bars_for_entry
    from .pattern_engine import detect_side_by_side
    from .setup_scorer import score_setup

    support = entry.support_level
    if not support or support <= 0:
        return None

    try:
        bars: pd.DataFrame = fetch_bars_for_entry(entry) or pd.DataFrame()
    except Exception:
        return None

    if bars.empty or len(bars) < 3:
        return None

    # Touch check: last bar's BODY low ≤ support × (1 + band)
    # Trader rule: "it's the bodies not the wicks"
    last_open  = float(bars["open"].iloc[-1])
    last_close = float(bars["close"].iloc[-1])
    body_low   = min(last_open, last_close)
    if body_low > support * (1 + _TOUCH_BAND):
        return None   # candle body hasn't reached support yet

    # Pattern check on last N bars
    recent = bars.iloc[-_PATTERN_BARS:]
    patterns = detect_side_by_side(recent)
    if not patterns:
        return None

    # Signal confirmed — score the full setup
    return score_setup(entry, bars)


# ---------------------------------------------------------------------------
# Scan loop
# ---------------------------------------------------------------------------

# Maximum concurrent ticker checks per scan pass.
# Each check does one yfinance HTTP request + pattern/score work.
_SCAN_WORKERS = 8


def scan_once(json_path: str, max_workers: int = _SCAN_WORKERS) -> int:
    """
    Run one full scan pass in parallel. Returns number of alerts fired.

    Parameters
    ----------
    max_workers : int
        Concurrent ticker-check threads (default 8).
    """
    entries = _load_entries(json_path)
    if not entries:
        print("[scanner] No entries to scan.")
        return 0

    alerts = []

    with ThreadPoolExecutor(max_workers=min(max_workers, len(entries))) as pool:
        futures = {pool.submit(_check_ticker, entry): entry for entry in entries}
        for fut in as_completed(futures):
            try:
                sc = fut.result()
            except Exception:
                sc = None
            if sc is not None:
                alerts.append(sc)

    # Fire alerts sequentially so sounds/notifications don't overlap
    for sc in alerts:
        _banner(sc)
        _desktop_notify(sc)
        _sound()

    ts = datetime.now(tz=timezone.utc).strftime("%H:%M:%S UTC")
    print(f"[scanner] {ts} — scanned {len(entries)} tickers, {len(alerts)} alert(s).")
    return len(alerts)


def run_loop(json_path: str, interval: int) -> None:
    """Poll indefinitely, sleeping *interval* seconds between scans."""
    print(f"[scanner] Starting live scan — interval {interval}s — Ctrl+C to stop")
    try:
        while True:
            scan_once(json_path)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[scanner] Stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="live_scanner",
        description="Real-time entry alert scanner for watchlist tickers.",
    )
    p.add_argument(
        "--watchlist", default=str(_DEFAULT_JSON),
        help="Path to scraped_watchlists.json",
    )
    p.add_argument(
        "--interval", type=int, default=_POLL_INTERVAL,
        help="Seconds between scan passes (default 300)",
    )
    p.add_argument(
        "--once", action="store_true",
        help="Run a single scan pass and exit (no loop)",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    if args.once:
        scan_once(args.watchlist)
    else:
        run_loop(args.watchlist, args.interval)
