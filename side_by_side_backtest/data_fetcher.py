"""
Phase 1 — Data Fetcher
Retrieves 5-minute OHLCV bars for a given ticker and date range.

Primary provider: yfinance  (no API key needed)
Optional: Alpaca Markets (set ALPACA_API_KEY / ALPACA_SECRET_KEY env vars)

All returned data is a pandas DataFrame with a DatetimeTZAware index (UTC)
and columns: open, high, low, close, volume  (all lowercase).
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Optional heavy imports — deferred so the rest of the package still loads
# ---------------------------------------------------------------------------

def _yf():
    try:
        import yfinance as yf  # type: ignore
        return yf
    except ImportError as exc:
        raise ImportError("yfinance is required: pip install yfinance") from exc


import contextlib as _contextlib
import io as _io
import logging as _logging


@_contextlib.contextmanager
def _silence_yfinance():
    """Context manager that suppresses yfinance's 'possibly delisted' and other
    noisy log/stderr output.  yfinance emits these through both its own logger
    (``yfinance``) and directly to sys.stderr depending on the version."""
    import sys
    yf_logger = _logging.getLogger("yfinance")
    old_level  = yf_logger.level
    yf_logger.setLevel(_logging.CRITICAL)   # swallow WARNING / ERROR noise
    old_stderr = sys.stderr
    sys.stderr  = _io.StringIO()            # capture any direct stderr writes
    try:
        yield
    finally:
        yf_logger.setLevel(old_level)
        sys.stderr = old_stderr



def _alpaca_client():
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        return None
    try:
        from alpaca.data.historical import StockHistoricalDataClient  # type: ignore
        return StockHistoricalDataClient(key, secret)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Banned-ticker list (persistent — skip 404/delisted symbols forever)
# ---------------------------------------------------------------------------

_DISK_CACHE_DIR  = Path(__file__).parent / "ohlcv_cache"
_BANNED_FILE     = Path(__file__).parent / "banned_tickers.json"
_BANNED_TICKERS: set[str] = set()


def _load_banned() -> None:
    """Load the persistent banned-ticker set from disk."""
    global _BANNED_TICKERS
    if _BANNED_FILE.exists():
        try:
            import json as _json
            _BANNED_TICKERS = set(_json.loads(_BANNED_FILE.read_text()))
        except Exception:
            _BANNED_TICKERS = set()


def _save_banned() -> None:
    """Persist the banned-ticker set to disk."""
    try:
        import json as _json
        _BANNED_FILE.write_text(_json.dumps(sorted(_BANNED_TICKERS), indent=2))
    except Exception:
        pass


# Minimum ticker length and character requirements for a valid symbol.
# Prevents English words / parser noise from entering the ban list.
_MIN_TICKER_LEN = 2
_MAX_TICKER_LEN = 6


def _is_valid_ticker_symbol(ticker: str) -> bool:
    """Return True only if *ticker* looks like a real equity symbol."""
    t = ticker.upper().strip()
    if not (t.isalpha() and _MIN_TICKER_LEN <= len(t) <= _MAX_TICKER_LEN):
        return False
    # Reject obvious English words that the parser sometimes emits even
    # though they are in _TICKER_BLACKLIST — belt-and-suspenders guard.
    _COMMON_WORDS = {
        "DO", "IN", "OF", "TO", "ON", "AT", "BY", "UP", "AS", "IF",
        "OR", "AN", "IS", "IT", "WE", "MY", "ME", "HE", "SHE", "HIM",
        "ONLY", "COULD", "WOULD", "SHOULD", "MIGHT", "MUST", "MAY",
        "CAN", "LOSE", "GAIN", "STOP", "HOLD", "PLAY", "LOOK", "LIKE",
        "SELL", "BULL", "BEAR", "FLAT", "CASH", "COST", "FLOW", "GAP",
    }
    return t not in _COMMON_WORDS


def ban_ticker(ticker: str, reason: str = "") -> None:
    """Add a ticker to the persistent ban list and save immediately.

    Only accepts symbols that pass *_is_valid_ticker_symbol* so that parser
    noise (English words, numbers, etc.) never enters the ban list.
    """
    t = ticker.upper().strip()
    if not _is_valid_ticker_symbol(t):
        return  # silently ignore non-symbol strings
    if t not in _BANNED_TICKERS:
        print(f"[data_fetcher] BANNED: {t} — {reason or 'no data / delisted'}")
        _BANNED_TICKERS.add(t)
        _save_banned()


def is_banned(ticker: str) -> bool:
    t = ticker.upper().strip()
    if t == "SPY":
        return True
    return t in _BANNED_TICKERS


# Load ban list at import time
_load_banned()


# ---------------------------------------------------------------------------
# Cache (in-memory + optional disk)
# ---------------------------------------------------------------------------

_MEM_CACHE: Dict[Tuple[str, str, str], pd.DataFrame] = {}
_MEM_CACHE_LOCK = threading.Lock()   # guards concurrent reads/writes from parallel fetchers


def _cache_key(ticker: str, start: str, end: str) -> Tuple[str, str, str]:
    return (ticker.upper(), start, end)


def _disk_path(ticker: str, start: str, end: str) -> Path:
    _DISK_CACHE_DIR.mkdir(exist_ok=True)
    return _DISK_CACHE_DIR / f"{ticker}_{start}_{end}_5m.parquet"


def _read_disk(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    p = _disk_path(ticker, start, end)
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            p.unlink(missing_ok=True)
    return None


def _write_disk(df: pd.DataFrame, ticker: str, start: str, end: str) -> None:
    try:
        df.to_parquet(_disk_path(ticker, start, end))
    except Exception:
        pass  # disk cache is best-effort


# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------

def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns, ensure UTC timezone-aware DatetimeIndex."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Map yfinance-specific column names to standard names
    rename_map = {"adj close": "close", "date": "timestamp"}
    df.rename(columns=rename_map, inplace=True)

    # Keep only OHLCV
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            df[col] = float("nan")

    df = df[["open", "high", "low", "close", "volume"]].copy()

    # Ensure timezone-aware index
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.sort_index(inplace=True)
    df.dropna(subset=["open", "close"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# yfinance backend
# ---------------------------------------------------------------------------

def _fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    yf = _yf()

    # Primary path: Ticker.history() — single-ticker endpoint, avoids the
    # MultiIndex dividend-adjustment reindex bug in yfinance 1.1.x + pandas 2.3.
    df = pd.DataFrame()
    try:
        with _silence_yfinance():
            t = yf.Ticker(ticker)
            df = t.history(
                start=start, end=end, interval="5m",
                auto_adjust=True, prepost=True,
            )
    except Exception:
        df = pd.DataFrame()

    # Fallback: yf.download() — faster batch path but can hit the
    # "Reindexing only valid with uniquely valued Index objects" bug for
    # certain tickers when yfinance's internal auto-adjust logic encounters
    # duplicate timestamps.  Only use this if .history() returned nothing.
    if df.empty:
        try:
            with _silence_yfinance():
                df = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    interval="5m",
                    auto_adjust=True,
                    prepost=True,
                    progress=False,
                    threads=False,
                )
            # Robustly handle yfinance MultiIndex: new versions return ('Close', 'TICKER')
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            # Deduplicate index before any further processing to avoid
            # pandas 2.3 InvalidIndexError in downstream reindex calls.
            if not df.empty and df.index.duplicated().any():
                df = df[~df.index.duplicated(keep="last")]
        except Exception:
            df = pd.DataFrame()

    if df.empty:
        raise ValueError(f"yfinance returned no data for {ticker} {start}→{end}")

    return _normalise(df)


# ---------------------------------------------------------------------------
# Alpaca backend
# ---------------------------------------------------------------------------

def _fetch_alpaca(ticker: str, start: str, end: str) -> pd.DataFrame:
    client = _alpaca_client()
    if client is None:
        raise EnvironmentError("Alpaca env vars not set or alpaca-py not installed.")

    from alpaca.data.requests import StockBarsRequest  # type: ignore
    from alpaca.data.timeframe import TimeFrame  # type: ignore

    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Minute * 5,
        start=datetime.fromisoformat(start),
        end=datetime.fromisoformat(end),
    )
    bars = client.get_stock_bars(req)
    df = bars.df
    if df.empty:
        raise ValueError(f"Alpaca returned no data for {ticker} {start}→{end}")
    # Alpaca returns MultiIndex (symbol, timestamp)
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(ticker, level=0)
    return _normalise(df)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_5min_bars(
    ticker: str,
    start: str,
    end: str,
    provider: str = "yfinance",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch 5-minute OHLCV bars for *ticker* between *start* and *end* (ISO date strings).

    Parameters
    ----------
    ticker   : e.g. "AAPL"
    start    : "2024-03-01"
    end      : "2024-03-30"  (inclusive end; adds 1 day internally)
    provider : "yfinance" | "alpaca"
    use_cache: Whether to use in-memory + disk cache.

    Returns
    -------
    pd.DataFrame with UTC DatetimeIndex and columns [open, high, low, close, volume].
    """
    # Strip any accidental $ prefix or whitespace from ticker symbols
    ticker = ticker.strip().lstrip("$").upper()

    # Skip permanently banned tickers immediately (404/delisted)
    if is_banned(ticker):
        return pd.DataFrame()

    # yfinance 5m data is only available for the last 60 days; clamp BEFORE
    # computing the cache key so that the key is stable across runs.
    if provider == "yfinance":
        yf_limit = (datetime.now() - timedelta(days=59)).strftime("%Y-%m-%d")
        if start < yf_limit:
            start = yf_limit
        # After clamping, if the entire window is outside the 60-day horizon
        # (i.e. end is also before the yfinance limit), skip the request entirely
        # to avoid yfinance's "start date cannot be after end date" error.
        if end <= start:
            return pd.DataFrame()

    key = _cache_key(ticker, start, end)

    if use_cache:
        with _MEM_CACHE_LOCK:
            hit = _MEM_CACHE.get(key)
        if hit is not None:
            return hit
        disk = _read_disk(ticker, start, end)
        if disk is not None:
            with _MEM_CACHE_LOCK:
                _MEM_CACHE[key] = disk
            return disk

    # Extend end by 1 day so that end is inclusive
    end_dt = (datetime.fromisoformat(end) + timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        if provider == "alpaca":
            df = _fetch_alpaca(ticker, start, end_dt)
        else:
            df = _fetch_yfinance(ticker, start, end_dt)
    except Exception as exc:
        err = str(exc).lower()
        # Only permanently ban on unambiguous "symbol not found / delisted" errors.
        # Transient failures (rate-limits, network issues, yfinance internal bugs,
        # pandas version incompatibilities) must NOT result in a permanent ban.
        is_transient = isinstance(exc, (TypeError, AttributeError, KeyError, IndexError))
        is_permanent_miss = (
            not is_transient
            and ("404" in err or "no ticker" in err or "delisted" in err
                 or "no data found for" in err)
            # Extra guard: never ban well-known liquid symbols on a single failure
            and ticker not in {"SPY", "QQQ", "IWM", "DIA", "VOO", "VTI",
                                "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"}
        )
        if is_permanent_miss:
            ban_ticker(ticker, reason=str(exc)[:120])
        raise

    if use_cache:
        with _MEM_CACHE_LOCK:
            _MEM_CACHE[key] = df
        _write_disk(df, ticker, start, end)

    return df


def fetch_bars_for_entry(
    entry,
    lookback_days: int = 3,
    forward_days: int = 2,
    provider: str = "yfinance",
) -> pd.DataFrame:
    """
    Convenience wrapper: given a WatchlistEntry, fetch bars around the post date.

    Returns empty DataFrame if the entry has no timestamp or data is unavailable.
    """
    if entry.post_timestamp is None:
        return pd.DataFrame()

    post_date: date = entry.post_timestamp.date()
    start = (post_date - timedelta(days=lookback_days)).isoformat()
    end = (post_date + timedelta(days=forward_days)).isoformat()

    try:
        return fetch_5min_bars(entry.ticker, start, end, provider=provider)
    except Exception as exc:
        err = str(exc).lower()
        if "404" in err or "not found" in err or "no data" in err or "delisted" in err:
            ban_ticker(entry.ticker, reason=str(exc)[:120])
        print(f"[data_fetcher] WARNING: {entry.ticker} {start}→{end}: {exc}")
        return pd.DataFrame()


def fetch_bars_batch(
    entries: list,
    lookback_days: int = 3,
    forward_days: int = 2,
    provider: str = "yfinance",
    delay_seconds: float = 0.5,
    max_workers: int = 8,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch bars for multiple WatchlistEntry objects in parallel.
    Returns {ticker: DataFrame} (one DataFrame per unique ticker across all entries).
    Deduplicates requests so each (ticker, date-range) is only fetched once.

    Parameters
    ----------
    max_workers : int
        Maximum concurrent fetch threads (default 8).  Keep ≤ 8 to stay within
        yfinance's informal rate-limit.  Each thread sleeps *delay_seconds* after
        its own fetch to spread the load.
    """
    # Build the deduplicated work list first
    seen: set[tuple] = set()
    work: list[tuple[str, str, str]] = []   # (ticker, start, end)

    for entry in entries:
        if entry.post_timestamp is None:
            continue
        post_date = entry.post_timestamp.date()
        start = (post_date - timedelta(days=lookback_days)).isoformat()
        end   = (post_date + timedelta(days=forward_days)).isoformat()
        key   = (entry.ticker, start, end)
        if key in seen:
            continue
        seen.add(key)
        work.append(key)

    if not work:
        return {}

    def _fetch_one(ticker: str, start: str, end: str) -> tuple[str, pd.DataFrame]:
        df = pd.DataFrame()
        try:
            df = fetch_5min_bars(ticker, start, end, provider=provider)
        except Exception as exc:
            print(f"[data_fetcher] WARNING: {ticker}: {exc}")
        # Polite delay per worker to avoid hammering the provider
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        return ticker, df

    raw: Dict[str, list[pd.DataFrame]] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(work))) as pool:
        futures = {pool.submit(_fetch_one, t, s, e): (t, s, e) for t, s, e in work}
        for fut in as_completed(futures):
            ticker, df = fut.result()
            if not df.empty:
                raw.setdefault(ticker, []).append(df)

    # Merge multiple date-range slices for the same ticker.
    # Each frame must be deduplicated BEFORE concat — pd.concat raises
    # InvalidIndexError if any input frame itself has a non-unique index.
    def _dedup(df: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicate timestamps in-place (keep last = freshest data)."""
        # Deduplicate row index (timestamps)
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep="last")]
        # Deduplicate column index — pd.concat raises InvalidIndexError
        # if *any* participating frame has non-unique column labels.
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep="last")]
        return df

    result: Dict[str, pd.DataFrame] = {}
    for ticker, frames in raw.items():
        clean = [_dedup(f) for f in frames]
        if len(clean) == 1:
            result[ticker] = clean[0].sort_index()
        else:
            combined = pd.concat(clean, axis=0)
            result[ticker] = _dedup(combined).sort_index()

    print(f"[data_fetcher] Fetched bars for {len(result)} unique tickers "
          f"({len(work)} requests, {min(max_workers, len(work))} workers).")
    return result


# ---------------------------------------------------------------------------
# 30-Day Rolling Cache  (one canonical file per ticker)
# ---------------------------------------------------------------------------

_30D_SUFFIX = "_30d_5m.parquet"


def _30d_path(ticker: str) -> Path:
    """Canonical path for a ticker's 30-day rolling parquet."""
    _DISK_CACHE_DIR.mkdir(exist_ok=True)
    return _DISK_CACHE_DIR / f"{ticker.upper()}{_30D_SUFFIX}"


def fetch_30day_bars(
    ticker: str,
    provider: str = "yfinance",
    window_days: int = 30,
) -> pd.DataFrame:
    """
    Fetch a full *window_days* (default 30) rolling window of 5-min bars for
    *ticker*, write to the canonical ``{ticker}_30d_5m.parquet`` file, and
    return the DataFrame.

    yfinance only holds 60 days of 5-min data; this is clamped automatically.
    All sessions (pre-market, regular, after-hours) are included.
    """
    ticker = ticker.strip().lstrip("$").upper()
    if is_banned(ticker):
        return pd.DataFrame()

    end   = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    start = (datetime.now(tz=timezone.utc) - timedelta(days=window_days)).strftime("%Y-%m-%d")

    # Load any existing bars so we don't re-download data we already have
    existing = load_30day_bars(ticker)

    try:
        fresh = fetch_5min_bars(ticker, start, end, provider=provider, use_cache=False)
    except Exception as exc:
        print(f"[30d_cache] ERROR {ticker}: {exc}")
        return existing  # return what we have rather than empty

    if fresh.empty:
        return existing

    if not existing.empty:
        if existing.index.duplicated().any():
            existing = existing[~existing.index.duplicated(keep="last")]
        if fresh.index.duplicated().any():
            fresh = fresh[~fresh.index.duplicated(keep="last")]
        combined = pd.concat([existing, fresh])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = fresh

    _prune_and_save(ticker, combined, window_days)
    return combined


def refresh_today(
    ticker: str,
    provider: str = "yfinance",
    window_days: int = 30,
) -> pd.DataFrame:
    """
    Fetch any missing bars since the last cached bar, merge into the 30-day
    parquet, prune bars older than *window_days*, and save back.

    Gap-aware: if you were away 3 days, it fetches all 3 missing days at once
    rather than only today.  Falls back to a full 30-day re-fetch if the
    parquet doesn't exist yet or the gap exceeds the window.
    """
    ticker = ticker.strip().lstrip("$").upper()
    if is_banned(ticker):
        return pd.DataFrame()

    today = datetime.now(tz=timezone.utc).date()
    existing = load_30day_bars(ticker)

    if existing.empty:
        # No parquet yet — do a full seed
        return fetch_30day_bars(ticker, provider=provider, window_days=window_days)

    # Determine gap in minutes from last bar to now
    last_bar_ts  = existing.index[-1].tz_convert("UTC")
    now_ts       = datetime.now(tz=timezone.utc)
    gap_minutes  = (now_ts - last_bar_ts).total_seconds() / 60
    last_bar_date = last_bar_ts.date()
    gap_days      = (today - last_bar_date).days

    if gap_minutes < 6:
        # Last bar is less than 6 minutes old — already current (one 5-min bar)
        return existing

    if gap_days >= window_days:
        # Gap is so large that a full re-fetch is cleaner
        print(f"[30d_cache] {ticker}: gap={gap_days}d ≥ window — full re-fetch")
        return fetch_30day_bars(ticker, provider=provider, window_days=window_days)

    # Fetch the missing slice: start from last_bar_date (yfinance returns bars AFTER last cached)
    # fetch_start = last_bar_date so same-day intra-session gaps are filled.
    # fetch_end = today + 1 day because yfinance's end is EXCLUSIVE —
    # fetching start=today, end=today returns nothing.
    fetch_start = last_bar_date.isoformat()
    fetch_end   = (today + timedelta(days=1)).isoformat()
    gap_desc    = f"{gap_days}d" if gap_days >= 1 else f"{gap_minutes:.0f}min"
    print(f"[30d_cache] {ticker}: gap={gap_desc} — fetching {fetch_start} → {fetch_end}")
    try:
        fresh = fetch_5min_bars(ticker, fetch_start, fetch_end,
                                provider=provider, use_cache=False)
    except Exception as exc:
        print(f"[30d_cache] WARNING {ticker}: {exc}")
        fresh = pd.DataFrame()

    if not fresh.empty:
        _cols = ["open", "high", "low", "close", "volume"]
        # Force fresh copies with clean column indexes to avoid InvalidIndexError
        # when called concurrently from multiple Streamlit fragments
        def _prep(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            cols = [c for c in _cols if c in df.columns]
            out = df[cols].copy()
            if out.index.tzinfo is None:
                out.index = out.index.tz_localize("UTC")
            else:
                out.index = out.index.tz_convert("UTC")
            out = out[~out.index.duplicated(keep="last")]
            return out

        try:
            combined = pd.concat([_prep(existing), _prep(fresh)])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        except Exception:
            # Concurrent write/read race — return existing data without update
            combined = existing
    else:
        combined = existing

    if combined.empty:
        return combined

    _prune_and_save(ticker, combined, window_days)
    return combined


def _prune_and_save(ticker: str, df: pd.DataFrame, window_days: int) -> None:
    """Drop bars older than *window_days* and write to canonical parquet."""
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=window_days)
    df = df[df.index >= cutoff]
    try:
        df.to_parquet(_30d_path(ticker))
    except Exception as exc:
        print(f"[30d_cache] Could not save {ticker}: {exc}")


def load_30day_bars(ticker: str) -> pd.DataFrame:
    """
    Load the canonical 30-day parquet for *ticker*.
    Returns an empty DataFrame if it doesn't exist yet.
    """
    ticker = ticker.strip().lstrip("$").upper()
    p = _30d_path(ticker)
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(p)
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df
    except Exception:
        p.unlink(missing_ok=True)
        return pd.DataFrame()
