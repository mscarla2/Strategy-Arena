"""
Phase 3 — Event-Driven Simulator (The "Kernel")
Replays 5-minute bars bar-by-bar for each WatchlistEntry, implementing the
"Wait and Stay" entry logic and the multi-exit trade management rules.

Entry Logic ("Wait and Stay"):
    1. Price touches (trades down to) the support_level during a 5m bar.
    2. Wait for that bar to *close*.
    3. If Close >= support_level AND a Side-by-Side pattern is present
       at or just before that bar → execute Buy at the NEXT bar's Open.

Exit Logic (first condition hit):
    A. Take-Profit:  price >= entry * (1 + profit_target_pct / 100)
    B. Stop-Loss:    price <= entry * (1 - stop_loss_pct / 100)
       (also invalidated if Close < support_level after entry)
    C. Time-Stop:    forced exit at session close (4:00 PM ET = 21:00 UTC)
                     or at the last available bar for PM / AH sessions.
"""
from __future__ import annotations

from datetime import datetime, time, timezone
from typing import List, Optional, Tuple

import pandas as pd

from .models import PatternMatch, SessionType, TradeResult, WatchlistEntry
from .pattern_engine import detect_side_by_side, pattern_near_support

# ---------------------------------------------------------------------------
# Session boundary constants (UTC)
# ---------------------------------------------------------------------------
# Pre-market:    09:00 – 13:30 UTC  (04:00 – 09:30 ET)
# Market Open:   13:30 – 20:00 UTC  (09:30 – 16:00 ET)
# After Hours:   20:00 – 24:00 UTC  (16:00 – 20:00 ET)

# All times are naive UTC — _bar_time_utc() returns naive time() so comparisons work.
_SESSION_CLOSE_UTC: dict[SessionType, time] = {
    SessionType.PRE_MARKET:   time(13, 25),   # 09:25 ET
    SessionType.MARKET_OPEN:  time(20, 0),    # 16:00 ET
    SessionType.AFTER_HOURS:  time(23, 55),   # 19:55 ET
    SessionType.UNKNOWN:      time(20, 0),
}

_SESSION_START_UTC: dict[SessionType, time] = {
    SessionType.PRE_MARKET:   time(9,  0),    # 04:00 ET
    SessionType.MARKET_OPEN:  time(13, 30),   # 09:30 ET
    SessionType.AFTER_HOURS:  time(20, 0),    # 16:00 ET
    SessionType.UNKNOWN:      time(13, 30),
}

# How many 5-min bars to look at for "support respected in first 60 min"
_SUPPORT_CHECK_BARS = 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar_time_utc(ts: pd.Timestamp) -> time:
    """Extract time component of a tz-aware Timestamp as UTC."""
    utc = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    return utc.time()


def _in_session(ts: pd.Timestamp, session: SessionType) -> bool:
    t = _bar_time_utc(ts)
    start = _SESSION_START_UTC.get(session, time(0))
    end = _SESSION_CLOSE_UTC.get(session, time(23, 59))
    # Handle midnight crossing (not common for US equities, included for safety)
    if start <= end:
        return start <= t <= end
    return t >= start or t <= end


def _after_session_close(ts: pd.Timestamp, session: SessionType) -> bool:
    t = _bar_time_utc(ts)
    close = _SESSION_CLOSE_UTC.get(session, time(20, 0))
    return t >= close


# ---------------------------------------------------------------------------
# Single-entry simulation
# ---------------------------------------------------------------------------

def simulate_entry(
    entry: WatchlistEntry,
    bars: pd.DataFrame,
    profit_target_pct: float = 2.0,
    stop_loss_pct: float = 1.0,
    pattern_tolerance_pct: float = 0.01,
    pattern_lookback: int = 5,
    strict_pattern_proximity: bool = False,
    pattern_proximity_pct: float = 0.005,
) -> Optional[TradeResult]:
    """
    Simulate one WatchlistEntry against its corresponding OHLCV bars.

    Parameters
    ----------
    entry                   : Parsed watchlist entry (must have support_level).
    bars                    : 5-min OHLCV DataFrame (UTC DatetimeIndex).
    profit_target_pct       : X% above entry price for take-profit.
    stop_loss_pct           : Y% below entry price for hard stop.
    pattern_tolerance_pct   : Body-midpoint tolerance for Side-by-Side detection.
    pattern_lookback        : How many bars back to look for the pattern.
    strict_pattern_proximity: If True, only accept patterns whose body-low is
                              within `pattern_proximity_pct` of the support level.
                              Eliminates loose pattern matches far from support.
    pattern_proximity_pct   : Fraction tolerance for strict proximity check
                              (default 0.005 = 0.5% from support).

    Returns
    -------
    TradeResult or None if no valid entry trigger was found.
    """
    if bars.empty or entry.support_level is None:
        return None

    support = entry.support_level
    session = entry.session_type
    ticker = entry.ticker

    # Filter bars to post-timestamp + session window
    if entry.post_timestamp is not None:
        post_utc = entry.post_timestamp
        bars = bars[bars.index >= post_utc]
    if bars.empty:
        return None

    # Pre-compute patterns on the full slice
    bars.attrs["ticker"] = ticker
    if strict_pattern_proximity and support is not None:
        # Strict mode: only accept patterns whose body is within proximity_pct of support
        patterns: List[PatternMatch] = pattern_near_support(
            bars,
            support=support,
            proximity_pct=pattern_proximity_pct,
            tolerance_pct=pattern_tolerance_pct,
            require_downtrend=True,
        )
    else:
        patterns = detect_side_by_side(
            bars,
            tolerance_pct=pattern_tolerance_pct,
            require_downtrend=True,
        )
    pattern_bar_indices: set[int] = {p.bar_index for p in patterns}

    # State machine
    waiting_for_close = False   # True once price has touched support
    touch_bar_idx: Optional[int] = None

    for i, (ts, row) in enumerate(bars.iterrows()):

        # ---- Time-stop gate ----
        if _after_session_close(ts, session):
            break

        # ---- Phase A: Scan for support touch ----
        if not waiting_for_close:
            # Bar low dipped to or below support
            if row["low"] <= support * 1.001:  # within 0.1% of support
                waiting_for_close = True
                touch_bar_idx = i
            continue

        # ---- Phase B: Wait for bar CLOSE ----
        # We're now examining the bar that closed after the touch

        # Check close above support (condition for "Wait and Stay")
        if row["close"] < support * 0.998:
            # Close significantly below support → support broken, reset
            waiting_for_close = False
            touch_bar_idx = None
            continue

        # Check if Side-by-Side pattern is present in the recent lookback window
        recent_range = range(max(0, i - pattern_lookback), i + 1)
        pattern_present = any(idx in pattern_bar_indices for idx in recent_range)

        if not pattern_present:
            # Keep waiting; pattern hasn't appeared yet
            continue

        # ---- Entry: execute at NEXT bar's open ----
        next_idx = i + 1
        if next_idx >= len(bars):
            break  # No next bar available

        next_ts = bars.index[next_idx]
        if _after_session_close(next_ts, session):
            break

        entry_price = bars.iloc[next_idx]["open"]
        if entry_price <= 0:
            break

        take_profit = entry_price * (1 + profit_target_pct / 100)
        stop_price  = min(
            entry_price * (1 - stop_loss_pct / 100),
            support * 0.999,  # stop just under support
        )

        # ---- Simulate forward from entry ----
        outcome = "timeout"
        exit_price: Optional[float] = None
        exit_ts: Optional[datetime] = None
        hold_bars = 0
        support_respected = True

        for j in range(next_idx + 1, len(bars)):
            fwd_ts = bars.index[j]
            fwd_row = bars.iloc[j]
            hold_bars += 1

            # Support respect check: first 12 bars (60 min)
            if j - next_idx <= _SUPPORT_CHECK_BARS:
                if fwd_row["low"] < support * 0.995:
                    support_respected = False

            # Time-stop
            if _after_session_close(fwd_ts, session):
                outcome = "timeout"
                exit_price = fwd_row["open"]  # exit at open of closing bar
                exit_ts = fwd_ts
                break

            # Take-profit: high touched TP
            if fwd_row["high"] >= take_profit:
                outcome = "win"
                exit_price = take_profit
                exit_ts = fwd_ts
                break

            # Stop-loss: low pierced stop
            if fwd_row["low"] <= stop_price:
                outcome = "loss"
                exit_price = stop_price
                exit_ts = fwd_ts
                break

            # Still in trade at last bar
            if j == len(bars) - 1:
                outcome = "timeout"
                exit_price = fwd_row["close"]
                exit_ts = fwd_ts

        if exit_price is None:
            exit_price = entry_price
            exit_ts = next_ts

        pnl_pct = (exit_price - entry_price) / entry_price * 100

        return TradeResult(
            ticker=ticker,
            entry_ts=next_ts,
            entry_price=entry_price,
            exit_ts=exit_ts,
            exit_price=exit_price,
            profit_target_pct=profit_target_pct,
            stop_loss_pct=stop_loss_pct,
            session_type=session,
            outcome=outcome,
            pnl_pct=round(pnl_pct, 4),
            hold_bars=hold_bars,
            support_respected=support_respected,
        )

    return None  # No entry trigger found


# ---------------------------------------------------------------------------
# Batch simulation
# ---------------------------------------------------------------------------

def simulate_all(
    entries: List[WatchlistEntry],
    bars_map: dict,   # {ticker: pd.DataFrame}
    profit_target_pct: float = 2.0,
    stop_loss_pct: float = 1.0,
    pattern_tolerance_pct: float = 0.01,
    strict_pattern_proximity: bool = False,
    pattern_proximity_pct: float = 0.005,
    verbose: bool = False,
) -> List[TradeResult]:
    """
    Run simulate_entry for every watchlist entry that has OHLCV data available.

    Parameters
    ----------
    entries                 : List of parsed WatchlistEntry objects.
    bars_map                : {ticker: DataFrame} from data_fetcher.fetch_bars_batch.
    profit_target_pct       : X% take-profit.
    stop_loss_pct           : Y% hard stop.
    pattern_tolerance_pct   : Passed to pattern engine.
    strict_pattern_proximity: Require pattern body within proximity_pct of support.
    pattern_proximity_pct   : Proximity tolerance for strict mode (default 0.5%).
    verbose                 : Print per-entry status.

    Returns
    -------
    List of TradeResult (one per entry that triggered an entry signal).
    """
    results: list[TradeResult] = []
    skipped = 0
    seen_entries: set[tuple] = set()  # deduplicate by (ticker, entry_ts)

    for entry in entries:
        bars = bars_map.get(entry.ticker)
        if bars is None or bars.empty:
            skipped += 1
            continue

        result = simulate_entry(
            entry,
            bars,
            profit_target_pct=profit_target_pct,
            stop_loss_pct=stop_loss_pct,
            pattern_tolerance_pct=pattern_tolerance_pct,
            strict_pattern_proximity=strict_pattern_proximity,
            pattern_proximity_pct=pattern_proximity_pct,
        )
        if result is not None:
            dedup_key = (result.ticker, result.entry_ts)
            if dedup_key in seen_entries:
                skipped += 1
                continue
            seen_entries.add(dedup_key)
            results.append(result)
            if verbose:
                print(
                    f"  [{result.outcome.upper():7s}] {result.ticker:6s} "
                    f"entry={result.entry_price:.2f}  "
                    f"exit={result.exit_price:.2f}  "
                    f"pnl={result.pnl_pct:+.2f}%  "
                    f"bars={result.hold_bars}"
                )
        elif verbose:
            print(f"  [NO_ENTRY] {entry.ticker:6s} — no trigger found")

    if verbose:
        print(
            f"\n[simulator] {len(results)} trades from {len(entries)} entries "
            f"({skipped} skipped — no data)."
        )
    return results
