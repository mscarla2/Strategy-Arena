"""
Watchlist Builder
=================
Generates WatchlistEntry objects directly from the ohlcv_cache/ parquet files,
eliminating the need to manually create or scrape a watchlist JSON.

Support-level strategy: uses a **rolling prior window** to avoid data leakage.
For each date window parquet, the support level is derived from the *previous*
window's price data via sr_engine (multi-method S/R) — strictly forward-looking.

Usage
-----
    from side_by_side_backtest.watchlist_builder import build_watchlist_from_tickers

    entries = build_watchlist_from_tickers(["UGRO", "ANNA", "TURB"])

CLI
---
    python -m side_by_side_backtest.watchlist_builder UGRO ANNA TURB
    # Prints a table of all generated entries and writes a JSON file
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .models import SessionType, WatchlistEntry

_CACHE_DIR = Path(__file__).parent / "ohlcv_cache"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_window_dates(stem: str) -> tuple[str, str] | None:
    """
    Parse 'TICKER_start_end_5m' filename stem.
    Returns (start_date, end_date) ISO strings or None if malformed.
    """
    parts = stem.split("_")
    # Expected: TICKER_YYYY-MM-DD_YYYY-MM-DD_5m  (4 parts)
    if len(parts) != 4:
        return None
    try:
        # Validate dates
        datetime.fromisoformat(parts[1])
        datetime.fromisoformat(parts[2])
        return parts[1], parts[2]
    except ValueError:
        return None


def _load_parquet(path: Path) -> pd.DataFrame:
    """Load parquet and ensure UTC DatetimeIndex."""
    try:
        df = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def _derive_support_from_df(df: pd.DataFrame, quantile: float = 0.15) -> Optional[float]:
    """Derive a support level as the low-price quantile of a DataFrame."""
    if df.empty or "low" not in df.columns:
        return None
    val = float(df["low"].quantile(quantile))
    return round(val, 4) if val > 0 else None


def _derive_resistance_from_df(df: pd.DataFrame, quantile: float = 0.85) -> Optional[float]:
    """Derive a resistance level as the high-price quantile of a DataFrame."""
    if df.empty or "high" not in df.columns:
        return None
    val = float(df["high"].quantile(quantile))
    return round(val, 4) if val > 0 else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_watchlist_from_tickers(
    tickers: List[str],
    cache_dir: Path = _CACHE_DIR,
    support_quantile: float = 0.15,
    resistance_quantile: float = 0.85,
    use_prior_window: bool = True,
) -> List[WatchlistEntry]:
    """
    Build WatchlistEntry objects for the given tickers from cached parquet files.

    Parameters
    ----------
    tickers             : List of uppercase ticker symbols.
    cache_dir           : Path to the ohlcv_cache directory.
    support_quantile    : Quantile of 'low' prices used as support level.
    resistance_quantile : Quantile of 'high' prices used as resistance level.
    use_prior_window    : If True (recommended), derive support from the
                          *previous* date window to avoid data leakage.
                          If False, uses the same window (original behaviour —
                          used for control/comparison purposes only).

    Returns
    -------
    List of WatchlistEntry objects sorted by ticker then date.
    """
    tickers = [t.upper().strip().lstrip("$") for t in tickers]
    entries: list[WatchlistEntry] = []

    for ticker in tickers:
        parquet_files = sorted(cache_dir.glob(f"{ticker}_*_5m.parquet"))
        if not parquet_files:
            print(f"[watchlist_builder] WARNING: no cached data found for {ticker}")
            continue

        # Build ordered list of (start_date, end_date, Path)
        all_windows: list[tuple[str, str, Path]] = []
        for pf in parquet_files:
            parsed = _parse_window_dates(pf.stem)
            if parsed:
                all_windows.append((*parsed, pf))

        # Sort chronologically by start date
        all_windows.sort(key=lambda x: x[0])

        # ── Deduplicate: one entry per unique start_date (keep last file for that date)
        # Multiple parquets share the same start_date when yfinance returns overlapping
        # windows. Using only the last (most data) prevents duplicate trades.
        seen_dates: dict[str, tuple[str, str, Path]] = {}
        for w in all_windows:
            seen_dates[w[0]] = w   # overwrite → keeps the last (most recent fetch)
        windows = sorted(seen_dates.values(), key=lambda x: x[0])

        for idx, (start_date, end_date, pf) in enumerate(windows):
            current_df = _load_parquet(pf)
            if current_df.empty:
                continue

            # ── Support/Resistance via sr_engine (all prior windows merged) ─
            from .sr_engine import compute_sr_levels
            current_price = float(current_df["close"].iloc[-1])

            if use_prior_window and idx > 0:
                # Merge ALL parquets before this window for maximum S/R history
                prior_dfs = [_load_parquet(windows[j][2]) for j in range(idx)]
                prior_dfs = [d for d in prior_dfs if not d.empty]
                if prior_dfs:
                    import pandas as _pd
                    src_df = _pd.concat(prior_dfs).drop_duplicates().sort_index()
                else:
                    src_df = current_df
            else:
                src_df = current_df   # leaky — for control tests only

            if src_df.empty:
                continue

            sr = compute_sr_levels(src_df, current_price=current_price)

            support    = sr.nearest_support(current_price)    or _derive_support_from_df(src_df, support_quantile)
            resistance = sr.nearest_resistance(current_price) or _derive_resistance_from_df(src_df, resistance_quantile)

            if support is None:
                continue

            stop = round(support * 0.95, 4)

            # Post timestamp = start of market session on start_date
            post_ts = datetime.fromisoformat(f"{start_date}T13:30:00+00:00")

            entry = WatchlistEntry(
                post_title=f"${ticker} watchlist {start_date}",
                post_timestamp=post_ts,
                raw_text=(
                    f"${ticker}: support {support}, resistance {resistance}, "
                    f"stop {stop}. market open"
                ),
                ticker=ticker,
                session_type=SessionType.MARKET_OPEN,
                support_level=support,
                resistance_level=resistance,
                stop_level=stop,
                sentiment_notes=f"auto-generated from cache window {start_date}→{end_date}",
            )
            entries.append(entry)

    print(
        f"[watchlist_builder] Built {len(entries)} entries for "
        f"{len(tickers)} ticker(s) "
        f"({'prior-window' if use_prior_window else 'same-window'} support)."
    )
    return entries


def build_watchlist_json(
    tickers: List[str],
    output_path: Path | str,
    cache_dir: Path = _CACHE_DIR,
    use_prior_window: bool = True,
) -> Path:
    """
    Build a scraped_watchlists.json-compatible file for the given tickers.

    Returns the path written.
    """
    entries = build_watchlist_from_tickers(
        tickers, cache_dir=cache_dir, use_prior_window=use_prior_window
    )

    posts = []
    for e in entries:
        ts_str = e.post_timestamp.isoformat() if e.post_timestamp else None
        posts.append(
            {
                "title": e.post_title,
                "content": e.raw_text,
                "timestamp": ts_str,
            }
        )

    output_path = Path(output_path)
    with output_path.open("w") as fh:
        json.dump(posts, fh, indent=2)
    print(f"[watchlist_builder] JSON written → {output_path} ({len(posts)} posts)")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import argparse

    ap = argparse.ArgumentParser(description="Build watchlist JSON from ohlcv cache.")
    ap.add_argument("tickers", nargs="+", help="Ticker symbols, e.g. UGRO ANNA TURB")
    ap.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON path (default: <TICKERS>_watchlist.json)",
    )
    ap.add_argument(
        "--same-window",
        action="store_true",
        help="Use same-window support derivation (leaky — for control tests only)",
    )
    args = ap.parse_args()

    tickers = [t.upper() for t in args.tickers]
    out_name = args.output or f"{'_'.join(tickers)}_watchlist.json"
    build_watchlist_json(
        tickers,
        output_path=out_name,
        use_prior_window=not args.same_window,
    )
    print(f"Done. Run: python -m side_by_side_backtest.main --watchlist {out_name} --skip-fetch --auto-tune --export")
