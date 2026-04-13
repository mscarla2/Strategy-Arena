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
import time
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


def ban_ticker(ticker: str, reason: str = "") -> None:
    """Add a ticker to the persistent ban list and save immediately."""
    t = ticker.upper().strip()
    if t not in _BANNED_TICKERS:
        print(f"[data_fetcher] BANNED: {t} — {reason or 'no data / delisted'}")
        _BANNED_TICKERS.add(t)
        _save_banned()


def is_banned(ticker: str) -> bool:
    return ticker.upper().strip() in _BANNED_TICKERS


# Load ban list at import time
_load_banned()


# ---------------------------------------------------------------------------
# Cache (in-memory + optional disk)
# ---------------------------------------------------------------------------

_MEM_CACHE: Dict[Tuple[str, str, str], pd.DataFrame] = {}


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
    # Primary: yf.download() — fast, but fails on some OTC/pink-sheet tickers
    # because it uses the quoteSummary endpoint which returns 404 for those.
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="5m",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    # Robustly handle yfinance MultiIndex: new versions return ('Close', 'TICKER')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Fallback: yf.Ticker.history() uses a different endpoint that works for OTC tickers
    if df.empty:
        try:
            t = yf.Ticker(ticker)
            df = t.history(start=start, end=end, interval="5m", auto_adjust=True)
        except Exception:
            pass

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

    key = _cache_key(ticker, start, end)

    if use_cache:
        if key in _MEM_CACHE:
            return _MEM_CACHE[key]
        disk = _read_disk(ticker, start, end)
        if disk is not None:
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
        if "404" in err or "not found" in err or "no data" in err or "delisted" in err:
            ban_ticker(ticker, reason=str(exc)[:120])
        raise

    if use_cache:
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
) -> Dict[str, pd.DataFrame]:
    """
    Fetch bars for multiple WatchlistEntry objects.
    Returns {ticker: DataFrame} (one DataFrame per unique ticker across all entries).
    Deduplicates requests so each (ticker, date-range) is only fetched once.
    """
    seen: set[tuple] = set()
    result: Dict[str, pd.DataFrame] = {}

    for entry in entries:
        if entry.post_timestamp is None:
            continue
        post_date = entry.post_timestamp.date()
        start = (post_date - timedelta(days=lookback_days)).isoformat()
        end = (post_date + timedelta(days=forward_days)).isoformat()
        key = (entry.ticker, start, end)
        if key in seen:
            continue
        seen.add(key)

        df = pd.DataFrame()
        try:
            df = fetch_5min_bars(entry.ticker, start, end, provider=provider)
        except Exception as exc:
            print(f"[data_fetcher] WARNING: {entry.ticker}: {exc}")

        # Merge into any existing data for the same ticker
        if entry.ticker in result and not df.empty:
            result[entry.ticker] = pd.concat([result[entry.ticker], df]).drop_duplicates().sort_index()
        elif not df.empty:
            result[entry.ticker] = df

        time.sleep(delay_seconds)

    print(f"[data_fetcher] Fetched bars for {len(result)} unique tickers.")
    return result
