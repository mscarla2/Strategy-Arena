"""
Phase 3 — Event-Driven Simulator (The "Kernel")
Replays 5-minute bars bar-by-bar for each WatchlistEntry, implementing the
"Wait and Stay" entry logic and the multi-exit trade management rules.

Entry Logic ("Wait and Stay"):
    1. Price touches (trades down to) the support_level during a 5m bar.
    2. Wait for that bar to *close*.
    3. If Close >= support_level AND a Side-by-Side pattern is present
       at or just before that bar → execute Buy at the NEXT bar's Open.
    4. After each completed trade (or failed touch), the state machine
       resets and continues scanning for the NEXT touch in the same session.
       All trades from a single entry object are returned as a list.

Exit Logic (first condition hit):
    A. Take-Profit:  price >= entry * (1 + profit_target_pct / 100)
    B. Stop-Loss:    price <= entry * (1 - stop_loss_pct / 100)
       (also invalidated if Close < support_level after entry)
    C. Time-Stop:    forced exit at session close (4:00 PM ET = 21:00 UTC)
                     or at the last available bar for PM / AH sessions.

TradeResult analysis tags (new):
    support_source    — "watchlist" | "computed"
    pattern_type      — "strict" | "exhaustion" | "absorption" | "none"
    bars_since_pattern — bars between pattern bar and entry bar
    entry_attempt     — 1-based touch counter within the session
"""
from __future__ import annotations

import logging
from datetime import datetime, time, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from .models import PatternMatch, SessionType, TradeResult, WatchlistEntry
from .pattern_engine import detect_side_by_side, pattern_near_support

# ---------------------------------------------------------------------------
# Simulation logger — writes to both stdout (via print) and an optional file
# ---------------------------------------------------------------------------

_LOG_DIR = Path(__file__).parent / "logs"


def _get_sim_logger(log_path: Optional[str] = None) -> logging.Logger:
    """
    Return a Logger that writes to *log_path* (appending).
    If log_path is None, returns a no-op logger.
    Each call with the same path reuses the same handler.
    """
    name = f"sim_log_{log_path or 'null'}"
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # don't double-print to root logger

    if log_path:
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(p), mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s",
                                          datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)
    else:
        logger.addHandler(logging.NullHandler())

    return logger

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
# Multi-entry simulation
# ---------------------------------------------------------------------------

def _resolve_support(
    entry: WatchlistEntry,
    bars: pd.DataFrame,
    silent: bool = False,
) -> Tuple[Optional[float], str]:
    """
    Resolve the support level for an entry, falling back to computed S/R
    when the watchlist level is missing or stale (>10% from current price).

    Returns (support_level, source) where source is "watchlist" or "computed".
    Extracted from simulate_entry so simulate_all can pre-compute once per entry
    and cache the result — avoiding repeated compute_sr_levels calls during
    auto-tune (which calls simulate_all 100+ times on the same data).
    """
    support = entry.support_level
    ticker  = entry.ticker

    current_price = float(bars["close"].iloc[-1]) if not bars.empty else 0.0

    def _level_stale(level: Optional[float], price: float, threshold: float = 0.10) -> bool:
        if not level or level <= 0 or price <= 0:
            return True
        return abs(level - price) / price > threshold

    if not _level_stale(support, current_price):
        return support, "watchlist"

    from .sr_engine import compute_sr_levels
    sr = compute_sr_levels(bars, current_price=current_price, price_range_pct=0.10)
    computed = sr.nearest_support(current_price)
    if computed:
        if not silent:
            print(f"[simulator] {ticker}: watchlist support {support} stale "
                  f"(>{10:.0f}% from {current_price:.3f}) → using computed {computed:.3f}")
        return computed, "computed"

    return None, "computed"


def _build_pattern_map(
    bars: pd.DataFrame,
    support: float,
    pattern_tolerance_pct: float = 0.01,
    strict_pattern_proximity: bool = False,
    pattern_proximity_pct: float = 0.005,
) -> Tuple[dict, set]:
    """
    Run all pattern detectors over *bars* and return (pattern_map, pattern_timestamps).

    Keys are pd.Timestamp (bar index timestamps), NOT positional integers.
    This makes the map slice-independent: when simulate_entry slices bars to the
    post-timestamp window, the timestamps remain valid regardless of position.

    Returns
    -------
    (pattern_map: {pd.Timestamp -> PatternMatch}, pattern_timestamps: set[pd.Timestamp])
    """
    from .pattern_engine import (
        detect_exhaustion_side_by_side,
        detect_support_absorption,
    )

    if strict_pattern_proximity and support is not None:
        _raw_patterns: List[PatternMatch] = pattern_near_support(
            bars, support=support,
            proximity_pct=pattern_proximity_pct,
            tolerance_pct=pattern_tolerance_pct,
            require_downtrend=True,
        )
    else:
        _raw_patterns = detect_side_by_side(
            bars, tolerance_pct=pattern_tolerance_pct, require_downtrend=True,
        )

    _raw_patterns = list(_raw_patterns)
    _raw_patterns += detect_exhaustion_side_by_side(bars)
    _raw_patterns += detect_support_absorption(bars, support=support, proximity_pct=0.015)

    # Key by timestamp — survives any subsequent DataFrame slicing
    pattern_map: dict = {}  # pd.Timestamp -> PatternMatch
    for pm in _raw_patterns:
        # Resolve the timestamp for this bar_index in the bars DataFrame
        if pm.bar_index < len(bars):
            ts_key = bars.index[pm.bar_index]
        else:
            continue  # out-of-range index — skip
        existing = pattern_map.get(ts_key)
        if existing is None or pm.confidence_score > existing.confidence_score:
            pattern_map[ts_key] = pm
    return pattern_map, set(pattern_map.keys())


def simulate_entry(
    entry: WatchlistEntry,
    bars: pd.DataFrame,
    profit_target_pct: float = 5.0,
    stop_loss_pct: float = 1.0,
    max_loss_pct: float = 5.0,
    pattern_tolerance_pct: float = 0.01,
    pattern_lookback: int = 5,
    strict_pattern_proximity: bool = False,
    pattern_proximity_pct: float = 0.005,
    trailing_stop_pct: Optional[float] = None,
    trail_activation_pct: float = 1.0,
    max_entry_attempts: int = 10,
    eqh_breakout_mode: bool = False,
    # ── Performance: pre-computed caches (passed by simulate_all) ────────────
    precomputed_support: Optional[float] = None,
    precomputed_support_source: Optional[str] = None,
    precomputed_pattern_map: Optional[dict] = None,
    precomputed_pattern_indices: Optional[set] = None,
) -> List[TradeResult]:
    """
    Simulate all trades for one WatchlistEntry within its session.

    The state machine now continues after each trade exits (non-timeout),
    scanning for the next support touch and re-entering when conditions align.
    Returns a list (possibly empty, possibly multiple trades per session).

    Parameters
    ----------
    entry                       : Parsed watchlist entry (must have support_level).
    bars                        : 5-min OHLCV DataFrame (UTC DatetimeIndex).
    profit_target_pct           : X% above entry price for take-profit (default 5%).
    stop_loss_pct               : Y% below entry price for hard stop.
    max_loss_pct                : Absolute maximum loss cap per trade (default 5%).
                                  Prevents gap-downs from blowing past the hard stop.
    pattern_tolerance_pct       : Body-midpoint tolerance for Side-by-Side detection.
    pattern_lookback            : How many bars back to look for the pattern (default 10).
    strict_pattern_proximity    : Require pattern body within proximity_pct of support.
    pattern_proximity_pct       : Proximity tolerance (default 0.5%).
    trailing_stop_pct           : Enable trailing stop (e.g. 1.5 = trail 1.5% below peak).
    trail_activation_pct        : Activate trail once +X% in profit (default 1.0%).
    max_entry_attempts          : Max support touches per session (0 = unlimited).
    precomputed_support         : Pre-resolved support level (skip _resolve_support).
    precomputed_support_source  : "watchlist" | "computed" for pre-resolved support.
    precomputed_pattern_map     : Pre-built bar_index→PatternMatch map (skip detection).
    precomputed_pattern_indices : Pre-built set of pattern bar indices.

    Returns
    -------
    List[TradeResult] — all trades triggered within the session (may be empty).
    """
    if bars.empty:
        return []

    # ── EQH breakout mode — delegate to dedicated simulator ──────────────────
    if eqh_breakout_mode:
        return simulate_eqh_breakout(
            entry, bars,
            profit_target_pct=profit_target_pct,
            stop_loss_pct=stop_loss_pct,
            max_loss_pct=max_loss_pct,
            max_forward_days=2,   # same default as --forward-days; prevents 30d-cache blowup
        )

    session = entry.session_type
    ticker  = entry.ticker

    # ── S/R resolution — use pre-computed if available (fast path) ────────────
    if precomputed_support is not None:
        support        = precomputed_support
        support_source = precomputed_support_source or "computed"
    else:
        support, support_source = _resolve_support(entry, bars, silent=False)

    if not support or support <= 0:
        return []  # no usable support level

    # Filter bars to post-timestamp + session window
    if entry.post_timestamp is not None:
        post_utc = entry.post_timestamp
        bars = bars[bars.index >= post_utc]
    if bars.empty:
        return []

    # ── Stale-support exclusion ───────────────────────────────────────────────
    # If the median price in the post-timestamp window is more than 15% above
    # the resolved support, the support level is stale (price has rallied far
    # past it). These tickers show 99–100% closes below support in analysis —
    # the level is no longer representative and entries would never pass Phase B.
    _median_price = float(bars[["open", "close"]].stack().median())
    if _median_price > 0 and support > 0:
        _support_staleness = (_median_price - support) / _median_price * 100
        if _support_staleness > 15.0:
            return []  # support too stale — skip this entry entirely

    # ── Pattern map — use pre-computed if available (fast path) ───────────────
    bars.attrs["ticker"] = ticker
    if precomputed_pattern_map is not None:
        pattern_map         = precomputed_pattern_map
        pattern_bar_indices = precomputed_pattern_indices if precomputed_pattern_indices is not None else set(pattern_map.keys())
    else:
        pattern_map, pattern_bar_indices = _build_pattern_map(
            bars, support,
            pattern_tolerance_pct=pattern_tolerance_pct,
            strict_pattern_proximity=strict_pattern_proximity,
            pattern_proximity_pct=pattern_proximity_pct,
        )

    # ── Multi-entry state machine ─────────────────────────────────────────────
    # Uses a manual while-loop index so we can restart from `scan_from_bar`
    # after each completed trade without Python for-loop restart limitations.
    results: List[TradeResult] = []
    waiting_for_close = False
    touch_bar_idx: Optional[int] = None
    entry_attempt = 0          # incremented at each support touch
    consecutive_losses = 0     # Fix 4: reset on win/timeout, breaks after MAX
    _MAX_CONSECUTIVE_LOSSES = 5
    _LOSS_COOLDOWN_BARS = 6    # Fix 4: 30-min cooldown after a loss
    cooldown_until = 0         # Fix 4: bar index below which re-entry is blocked
    i = 0                      # current bar index; advanced manually

    while i < len(bars):
        ts  = bars.index[i]
        row = bars.iloc[i]

        # ---- Time-stop gate ----
        if _after_session_close(ts, session):
            break

        # ---- Fix 6: max entry attempts cap ----
        if max_entry_attempts > 0 and entry_attempt >= max_entry_attempts:
            break

        # ---- Fix 4: post-loss cooldown ----
        if i < cooldown_until:
            i += 1
            continue

        # ---- Phase A: Scan for support touch (BODY-based, not wick) ----
        if not waiting_for_close:
            body_low = min(float(row["open"]), float(row["close"]))
            if body_low <= support * 1.005:
                waiting_for_close = True
                touch_bar_idx = i
                entry_attempt += 1
            i += 1
            continue

        # ---- Phase B: Wait for bar CLOSE ----
        # Allow close up to 3% below support — micro-cap stocks routinely close
        # slightly below a computed support level (which sits mid-zone) before
        # bouncing. Analysis showed 94% of touches close below at 0.2% tolerance;
        # 3% recovers the legitimate support-zone traders without letting in
        # tickers where support is genuinely broken.
        if row["close"] < support * 0.97:
            waiting_for_close = False
            touch_bar_idx = None
            i += 1
            continue

        # Check if a pattern is present within the lookback window
        # Use timestamp-based lookup (slice-safe) — compare bar timestamps
        # against the pattern_map keys which are also timestamps.
        lookback_start = max(0, i - pattern_lookback)
        recent_ts = bars.index[lookback_start: i + 1]
        pattern_present = any(ts in pattern_bar_indices for ts in recent_ts)

        if not pattern_present:
            i += 1
            continue   # keep waiting for pattern

        # ---- Entry: fire at NEXT bar's open ----
        next_idx = i + 1
        if next_idx >= len(bars):
            break

        next_ts = bars.index[next_idx]
        if _after_session_close(next_ts, session):
            break

        entry_price = float(bars.iloc[next_idx]["open"])
        if entry_price <= 0:
            break

        # ---- Fix 2a: penny-stock gate (below $0.10 = too illiquid to trade) ----
        _MIN_ENTRY_PRICE = 0.10
        if entry_price < _MIN_ENTRY_PRICE:
            waiting_for_close = False
            touch_bar_idx = None
            i += 1
            continue

        # ---- Fix 2b: pre-trade risk filter — skip if support too far below entry ----
        risk_pct = (entry_price - support) / entry_price * 100
        if risk_pct > max_loss_pct:
            waiting_for_close = False
            touch_bar_idx = None
            i += 1
            continue

        # Identify which pattern bar triggered this entry and tag it (timestamp-based)
        triggering_ts = max(
            (ts for ts in recent_ts if ts in pattern_bar_indices),
            default=None,
        )
        if triggering_ts is not None:
            trig_pm        = pattern_map[triggering_ts]
            p_type         = trig_pm.pattern_type
            # bars_since_pat: distance in bars between pattern ts and current bar i
            try:
                trig_pos   = bars.index.get_loc(triggering_ts)
                bars_since_pat = i - trig_pos
            except KeyError:
                bars_since_pat = 0
        else:
            p_type        = "none"
            bars_since_pat = 0

        # TP: use watchlist resistance level when available and > entry_price.
        # Fix 5: cap resistance-level override at 1.5× the configured profit target
        #        so trades don't stay open chasing an unreachable level.
        configured_tp = entry_price * (1 + profit_target_pct / 100)
        resistance_level = entry.resistance_level if entry.resistance_level else None
        if resistance_level and resistance_level > entry_price * 1.005:
            take_profit = min(float(resistance_level), configured_tp * 1.5)
        else:
            take_profit = configured_tp

        # Fix 2c: SL is a hard ceiling — support can only tighten, never widen it.
        # Tighten to support * 0.999 only when that level is ABOVE the configured SL.
        configured_stop = entry_price * (1 - stop_loss_pct / 100)
        support_stop    = support * 0.999
        stop_price = max(configured_stop, support_stop) if support_stop > configured_stop else configured_stop

        # Absolute max-loss floor (Fix 3) — always tighter than user SL when
        # the SL is smaller; prevents gap-downs from blowing past the cap.
        hard_loss_floor = entry_price * (1 - max_loss_pct / 100)

        # ---- Simulate forward from entry ----
        outcome = "timeout"
        exit_price: Optional[float] = None
        exit_ts: Optional[datetime] = None
        hold_bars = 0
        support_respected = True
        highest_high = entry_price
        trail_stop   = stop_price
        trail_active = False
        exit_bar_idx = next_idx  # track where the trade ended for scan resume

        for j in range(next_idx + 1, len(bars)):
            fwd_ts  = bars.index[j]
            fwd_row = bars.iloc[j]
            hold_bars += 1
            exit_bar_idx = j

            if j - next_idx <= _SUPPORT_CHECK_BARS:
                # Use CLOSE (not LOW) to distinguish real breakdowns from wick hunts.
                # Micro-cap stocks routinely wick 0.5–1% below support then close above —
                # only a BAR CLOSE below support * 0.99 signals genuine level failure.
                if fwd_row["close"] < support * 0.99:
                    support_respected = False

            if trailing_stop_pct is not None:
                bar_high = float(fwd_row["high"])
                if bar_high > highest_high:
                    highest_high = bar_high
                if not trail_active:
                    if (highest_high - entry_price) / entry_price * 100 >= trail_activation_pct:
                        trail_active = True
                if trail_active:
                    trail_stop = max(trail_stop,
                                     highest_high * (1 - trailing_stop_pct / 100))

            effective_stop = max(stop_price, trail_stop)

            # ── Fix 1 + Fix 4: TP and SL are checked BEFORE session-close ──
            # This ensures a position can never ride a gap-down into a TIMEOUT
            # without first being stopped out, and that intrabar TP hits are
            # captured even on the first bar after session close.

            # Fix 3: Hard max-loss cap — checked first (widest protection net)
            if fwd_row["low"] <= hard_loss_floor:
                outcome    = "loss"
                # Exit at the worse of: actual open (gap) or the floor price
                exit_price = min(float(fwd_row["open"]), hard_loss_floor)
                exit_ts    = fwd_ts
                break

            # Fix 4: Take-profit — intrabar high touches TP
            if fwd_row["high"] >= take_profit:
                outcome    = "win"
                exit_price = take_profit
                exit_ts    = fwd_ts
                break

            # Fix 1: Regular SL / trailing-stop — intrabar low touches stop
            if fwd_row["low"] <= effective_stop:
                outcome    = "win" if effective_stop > entry_price else "loss"
                exit_price = effective_stop
                exit_ts    = fwd_ts
                break

            # Session-close timeout — only fires if neither TP nor SL was hit
            if _after_session_close(fwd_ts, session):
                outcome    = "timeout"
                exit_price = float(fwd_row["close"])
                exit_ts    = fwd_ts
                break

            if j == len(bars) - 1:
                outcome    = "timeout"
                exit_price = float(fwd_row["close"])
                exit_ts    = fwd_ts

        if exit_price is None:
            exit_price = entry_price
            exit_ts    = next_ts

        pnl_pct = (exit_price - entry_price) / entry_price * 100

        results.append(TradeResult(
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
            support_source=support_source,
            pattern_type=p_type,
            bars_since_pattern=bars_since_pat,
            entry_attempt=entry_attempt,
        ))

        # ── Fix 4: consecutive-loss tracking and cooldown ─────────────────
        if outcome == "loss":
            consecutive_losses += 1
            cooldown_until = exit_bar_idx + _LOSS_COOLDOWN_BARS
            if consecutive_losses >= _MAX_CONSECUTIVE_LOSSES:
                break  # thesis broken — stop retrying this entry
        else:
            consecutive_losses = 0  # win or timeout resets streak

        # ── Resume scan after trade exits ────────────────────────────────
        # Time-stop means the session is over — nothing left to scan.
        if outcome == "timeout" or (exit_ts and _after_session_close(exit_ts, session)):
            break

        # Non-timeout exit: resume scanning from the bar after the exit bar.
        # Reset touch state so the next support touch starts a fresh cycle.
        i = exit_bar_idx + 1
        waiting_for_close = False
        touch_bar_idx     = None
        continue  # jump back to while condition with updated i

    return results


# ---------------------------------------------------------------------------
# EQH Breakout Simulator — runs when eqh_breakout_mode=True
# ---------------------------------------------------------------------------

def simulate_eqh_breakout(
    entry: WatchlistEntry,
    bars: pd.DataFrame,
    profit_target_pct: float = 5.0,
    stop_loss_pct: float = 1.0,
    max_loss_pct: float = 5.0,
    eqh_open_tolerance_pct: float = 0.02,
    body_clear_pct: float = 0.003,
    max_bars_after_pair: int = 5,
    max_forward_days: int = 2,
) -> List[TradeResult]:
    """
    Simulate the EQH Breakout strategy independently of the bearish S×S mode.

    Entry Logic ("EQH Breakout"):
      1. Scan bars for an Equal Highs pair (mixed-color, matching opens).
      2. In the next *max_bars_after_pair* bars, wait for a candle whose
         BODY closes ABOVE the EQH ceiling by at least body_clear_pct.
      3. Enter long at the NEXT bar's open.

    Exit Logic (first condition hit):
      A. Take-Profit:  price >= entry * (1 + profit_target_pct / 100)
      B. Stop-Loss:    price closes BELOW the EQH ceiling (thesis dead)
         OR price <= entry * (1 - stop_loss_pct / 100)
      C. Time-Stop:    forced exit at session close

    Parameters
    ----------
    max_forward_days : int
        Maximum calendar days of bars to scan after the post timestamp.
        Defaults to 2 (same window as the S×S simulator's --forward-days).
        This prevents the simulator from scanning weeks of 30d-cache data
        when the user passes the rolling parquet without --skip-fetch.

    Returns a list of TradeResult objects (typically 0 or 1 per session).
    """
    from .pattern_engine import detect_equal_highs_pair, detect_eqh_signal
    from datetime import timedelta

    if bars.empty:
        return []

    session = entry.session_type
    ticker  = entry.ticker
    bars.attrs["ticker"] = ticker

    # Filter to post-timestamp window (lower bound)
    if entry.post_timestamp is not None:
        post_utc = entry.post_timestamp
        bars = bars[bars.index >= post_utc]
        # Upper bound: clip to max_forward_days after the post date
        # This ensures the simulator only looks at the 1–2 trading days that
        # are relevant to the watchlist setup, not weeks of stale 30d-cache data.
        if max_forward_days > 0:
            cutoff = post_utc + timedelta(days=max_forward_days)
            bars = bars[bars.index <= cutoff]
    if bars.empty:
        return []

    # Detect EQH pairs and breakout signals
    eqh_pairs   = detect_equal_highs_pair(bars, open_tolerance_pct=eqh_open_tolerance_pct)
    eqh_signals = detect_eqh_signal(bars, eqh_pairs,
                                    body_clear_pct=body_clear_pct,
                                    max_bars_after=max_bars_after_pair)

    results: List[TradeResult] = []

    for sig in eqh_signals:
        if sig.pattern_type != "eqh_breakout":
            continue

        eqh_ceiling = sig.eqh_level
        if eqh_ceiling <= 0:
            continue

        # Entry: next bar's open after the breakout signal bar
        entry_idx = sig.bar_index + 1
        if entry_idx >= len(bars):
            continue

        entry_ts = bars.index[entry_idx]
        if _after_session_close(entry_ts, session):
            continue

        entry_price = float(bars.iloc[entry_idx]["open"])
        if entry_price <= 0 or entry_price < 0.10:
            continue

        # TP and SL
        take_profit  = entry_price * (1 + profit_target_pct / 100)
        stop_price   = max(
            entry_price * (1 - stop_loss_pct / 100),
            eqh_ceiling * 0.998,   # SL just below EQH ceiling (thesis dead if closed below)
        )
        hard_loss_floor = entry_price * (1 - max_loss_pct / 100)

        outcome      = "timeout"
        exit_price: Optional[float] = None
        exit_ts: Optional[datetime] = None
        hold_bars    = 0
        exit_bar_idx = entry_idx

        for j in range(entry_idx + 1, len(bars)):
            fwd_ts  = bars.index[j]
            fwd_row = bars.iloc[j]
            hold_bars += 1
            exit_bar_idx = j

            if _after_session_close(fwd_ts, session):
                outcome    = "timeout"
                exit_price = float(fwd_row["open"])
                exit_ts    = fwd_ts
                break

            # Hard max-loss cap
            if fwd_row["low"] <= hard_loss_floor:
                outcome    = "loss"
                exit_price = min(float(fwd_row["open"]), hard_loss_floor)
                exit_ts    = fwd_ts
                break

            # Take-profit
            if fwd_row["high"] >= take_profit:
                outcome    = "win"
                exit_price = take_profit
                exit_ts    = fwd_ts
                break

            # Stop-loss: price or close below EQH ceiling (thesis invalidated)
            if fwd_row["low"] <= stop_price:
                outcome    = "loss"
                exit_price = min(float(fwd_row["open"]), stop_price)
                exit_ts    = fwd_ts
                break

            if float(fwd_row["close"]) < eqh_ceiling:
                # Closed back below EQH ceiling — breakout failed
                outcome    = "loss"
                exit_price = float(fwd_row["close"])
                exit_ts    = fwd_ts
                break

        if exit_price is None:
            exit_price = float(bars.iloc[-1]["close"])
            exit_ts    = bars.index[-1]

        pnl = (exit_price - entry_price) / entry_price * 100

        results.append(TradeResult(
            ticker=ticker,
            entry_ts=entry_ts.to_pydatetime() if hasattr(entry_ts, "to_pydatetime") else entry_ts,
            entry_price=round(entry_price, 6),
            exit_ts=exit_ts.to_pydatetime() if hasattr(exit_ts, "to_pydatetime") else exit_ts,
            exit_price=round(exit_price, 6),
            profit_target_pct=profit_target_pct,
            stop_loss_pct=stop_loss_pct,
            session_type=session,
            outcome=outcome,
            pnl_pct=round(pnl, 4),
            hold_bars=hold_bars,
            support_respected=True,
            support_source="computed",
            pattern_type="eqh_breakout",
            bars_since_pattern=0,
            entry_attempt=1,
        ))

    return results


# ---------------------------------------------------------------------------
# Pre-computation cache (hoisted out of simulate_all for auto-tune reuse)
# ---------------------------------------------------------------------------

class SimulationCaches:
    """
    Immutable pre-computed data for a given (entries, bars_map, tolerance) combination.
    Build once with build_simulation_caches() and pass into every simulate_all() call
    during auto-tuning so the 62s of S/R + pattern pre-computation only runs ONCE
    regardless of how many trials are evaluated.

    Attributes
    ----------
    support_cache   : {ticker: (support_level, source_str)}
    pattern_cache   : {ticker: (pattern_map, pattern_bar_indices)}
    """
    __slots__ = ("support_cache", "pattern_cache")

    def __init__(
        self,
        support_cache: dict,
        pattern_cache: dict,
    ) -> None:
        self.support_cache = support_cache
        self.pattern_cache = pattern_cache


def build_simulation_caches(
    entries: List[WatchlistEntry],
    bars_map: dict,
    pattern_tolerance_pct: float = 0.01,
    strict_pattern_proximity: bool = False,
    pattern_proximity_pct: float = 0.005,
    verbose: bool = False,
    max_workers: int = 8,
) -> SimulationCaches:
    """
    Pre-compute S/R levels and pattern maps for every ticker in bars_map.

    This is the expensive step (≈62s serial for 349 tickers, ~10s parallel).
    Call it ONCE before starting auto-tune trials and pass the returned
    SimulationCaches object to every simulate_all() call via `caches=`.

    Parameters
    ----------
    entries                 : WatchlistEntry list (used to find best watchlist support).
    bars_map                : {ticker: DataFrame}.
    pattern_tolerance_pct   : Passed to pattern detectors.
    strict_pattern_proximity: Passed to pattern detectors.
    pattern_proximity_pct   : Passed to pattern detectors.
    verbose                 : Print progress while building.
    max_workers             : Thread-pool size for parallel per-ticker computation.

    Returns
    -------
    SimulationCaches
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .sr_engine import compute_sr_levels

    # Group watchlist support levels by ticker for fast lookup
    wl_supports: dict[str, list[Optional[float]]] = {}
    for e in entries:
        wl_supports.setdefault(e.ticker, []).append(e.support_level)

    unique_tickers = [t for t in bars_map if not bars_map[t].empty]
    if verbose:
        print(f"[sim_cache] Building caches for {len(unique_tickers)} tickers "
              f"(workers={max_workers}) …")

    def _build_one(ticker: str):
        tbars = bars_map[ticker]
        current_price = float(tbars["close"].iloc[-1])

        def _stale(level, price, thr=0.10):
            if not level or level <= 0 or price <= 0:
                return True
            return abs(level - price) / price > thr

        # Prefer a valid watchlist level
        sup: Optional[float] = None
        src: str = "computed"
        for wl_level in wl_supports.get(ticker, []):
            if not _stale(wl_level, current_price):
                sup = wl_level
                src = "watchlist"
                break

        if sup is None:
            sr = compute_sr_levels(tbars, current_price=current_price, price_range_pct=0.10)
            computed = sr.nearest_support(current_price)
            if computed:
                sup = computed
                src = "computed"

        pat_map: dict = {}
        pat_idx: set  = set()
        if sup and sup > 0:
            pat_map, pat_idx = _build_pattern_map(
                tbars, sup,
                pattern_tolerance_pct=pattern_tolerance_pct,
                strict_pattern_proximity=strict_pattern_proximity,
                pattern_proximity_pct=pattern_proximity_pct,
            )

        return ticker, sup, src, pat_map, pat_idx

    support_cache: dict[str, Tuple[Optional[float], str]] = {}
    pattern_cache: dict[str, Tuple[dict, set]]            = {}

    with ThreadPoolExecutor(max_workers=min(max_workers, len(unique_tickers))) as pool:
        futs = {pool.submit(_build_one, t): t for t in unique_tickers}
        for fut in as_completed(futs):
            try:
                ticker, sup, src, pat_map, pat_idx = fut.result()
                support_cache[ticker] = (sup, src)
                if pat_map:
                    pattern_cache[ticker] = (pat_map, pat_idx)
            except Exception as exc:
                print(f"[sim_cache] WARNING: {futs[fut]}: {exc}")

    if verbose:
        print(f"[sim_cache] Done — {len(support_cache)} support levels, "
              f"{len(pattern_cache)} pattern maps built.")

    return SimulationCaches(
        support_cache=support_cache,
        pattern_cache=pattern_cache,
    )


# ---------------------------------------------------------------------------
# Batch simulation
# ---------------------------------------------------------------------------

def _fetch_spy_benchmark(start_date: str, end_date: str) -> Optional[float]:
    """
    Fetch SPY daily bars and return the buy-and-hold return % from start to end.
    Returns None if data is unavailable.

    Guarantees at least a 5-calendar-day window so weekend-only ranges don't
    produce empty results.  yfinance noise is suppressed via _silence_yfinance().
    """
    from datetime import datetime as _dt, timedelta as _td
    from side_by_side_backtest.data_fetcher import _silence_yfinance

    try:
        import yfinance as yf

        # Ensure the window spans at least 5 calendar days (covers Mon–Fri).
        start_dt = _dt.fromisoformat(start_date)
        end_dt   = _dt.fromisoformat(end_date)
        if (end_dt - start_dt).days < 5:
            end_dt = start_dt + _td(days=5)

        with _silence_yfinance():
            spy = yf.download(
                "SPY",
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=False,
            )

        if spy.empty or len(spy) < 2:
            return None
        close = spy["Close"] if "Close" in spy.columns else spy.iloc[:, 0]
        # flatten MultiIndex columns if present
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        start_price = float(close.iloc[0])
        end_price   = float(close.iloc[-1])
        if start_price <= 0:
            return None
        return (end_price / start_price - 1) * 100
    except Exception as exc:
        print(f"[simulator] WARNING: SPY benchmark fetch failed: {exc}")
        return None


def simulate_all(
    entries: List[WatchlistEntry],
    bars_map: dict,   # {ticker: pd.DataFrame}
    profit_target_pct: float = 5.0,
    stop_loss_pct: float = 1.0,
    max_loss_pct: float = 5.0,
    pattern_tolerance_pct: float = 0.01,
    strict_pattern_proximity: bool = False,
    pattern_proximity_pct: float = 0.005,
    require_support_ok: bool = True,
    max_entry_attempts: int = 10,
    verbose: bool = False,
    log_path: Optional[str] = None,
    caches: Optional["SimulationCaches"] = None,
    eqh_breakout_mode: bool = False,
) -> List[TradeResult]:
    """
    Run simulate_entry for every watchlist entry that has OHLCV data available.

    Parameters
    ----------
    entries                 : List of parsed WatchlistEntry objects.
    bars_map                : {ticker: DataFrame} from data_fetcher.fetch_bars_batch.
    profit_target_pct       : X% take-profit (default 5%).
    stop_loss_pct           : Y% hard stop.
    max_loss_pct            : Absolute max loss cap per trade — prevents gap-through
                              blowouts beyond this % below entry (default 5%).
    pattern_tolerance_pct   : Passed to pattern engine.
    strict_pattern_proximity: Require pattern body within proximity_pct of support.
    pattern_proximity_pct   : Proximity tolerance for strict mode (default 0.5%).
    require_support_ok      : When True (default), filter out ALL trades where support
                              was not respected, regardless of source. Pass False to
                              allow support_ok=False trades through (legacy behaviour).
    max_entry_attempts      : Max support touches per session (default 10).
                              Prevents runaway re-entry on tickers like ATPC/AZI.
    verbose                 : Print per-entry status to stdout.
    log_path                : Optional path to append simulation log (e.g.
                              "side_by_side_backtest/logs/sim_2024-01-02.txt").
                              Defaults to side_by_side_backtest/logs/simulation.log
                              when verbose=True and log_path is not supplied.

    Returns
    -------
    List of TradeResult (one per entry that triggered an entry signal).
    """
    # Auto-assign default log path when verbose is on and no path given
    if verbose and log_path is None:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = str(_LOG_DIR / "simulation.log")

    logger = _get_sim_logger(log_path)

    def _log(msg: str) -> None:
        """Print to stdout (if verbose) and always write to log file."""
        if verbose:
            print(msg)
        logger.info(msg)

    results: list[TradeResult] = []
    skipped  = 0
    filtered = 0
    seen_entries: set[tuple] = set()

    # Feature B: determine date range across all entries for SPY benchmark
    entry_dates = [
        e.post_timestamp.date()
        for e in entries
        if e.post_timestamp is not None
    ]
    spy_bh_pct: Optional[float] = None
    spy_date_range: Optional[str] = None
    if entry_dates:
        spy_start = min(entry_dates).isoformat()
        # Add one extra day so yfinance includes the last trading date's close
        from datetime import timedelta
        spy_end   = (max(entry_dates) + timedelta(days=1)).isoformat()
        spy_date_range = f"{spy_start} → {spy_end}"
        spy_bh_pct = _fetch_spy_benchmark(spy_start, spy_end)

    _log(f"[simulator] START — {len(entries)} entries  "
         f"TP={profit_target_pct}%  SL={stop_loss_pct}%  "
         f"MAX_LOSS={max_loss_pct}%  MAX_ATTEMPTS={max_entry_attempts or '∞'}")

    # ── Perf 2+3: Use pre-built caches if provided (fast path for auto-tune) ──
    # When caches=None (normal single-pass), build them inline once.
    # When caches is a SimulationCaches object (auto-tune), skip the 62s build
    # entirely — the caller already built them once before starting trials.
    if caches is not None:
        _support_cache = caches.support_cache
        _pattern_cache = caches.pattern_cache
    else:
        built = build_simulation_caches(
            entries, bars_map,
            pattern_tolerance_pct=pattern_tolerance_pct,
            strict_pattern_proximity=strict_pattern_proximity,
            pattern_proximity_pct=pattern_proximity_pct,
            verbose=False,
        )
        _support_cache = built.support_cache
        _pattern_cache = built.pattern_cache

    # Feature A: compounded cumulative PnL multiplier
    cum_multiplier: float = 1.0

    for entry in entries:
        bars = bars_map.get(entry.ticker)
        if bars is None or bars.empty:
            skipped += 1
            continue

        # Use pre-computed support and patterns for this ticker
        ticker = entry.ticker
        entry_support, entry_src = _support_cache.get(ticker, (None, "computed"))
        pat_map, pat_idx = _pattern_cache.get(ticker, ({}, set()))

        trade_list = simulate_entry(
            entry, bars,
            profit_target_pct=profit_target_pct,
            stop_loss_pct=stop_loss_pct,
            max_loss_pct=max_loss_pct,
            pattern_tolerance_pct=pattern_tolerance_pct,
            strict_pattern_proximity=strict_pattern_proximity,
            pattern_proximity_pct=pattern_proximity_pct,
            max_entry_attempts=max_entry_attempts,
            eqh_breakout_mode=eqh_breakout_mode,
            precomputed_support=entry_support,
            precomputed_support_source=entry_src,
            precomputed_pattern_map=pat_map,
            precomputed_pattern_indices=pat_idx,
        )

        if not trade_list:
            _log(f"  [NO_ENTRY] {entry.ticker:6s} — no trigger found")
            continue

        for result in trade_list:
            dedup_key = (result.ticker, result.entry_ts)
            if dedup_key in seen_entries:
                skipped += 1
                continue
            seen_entries.add(dedup_key)

            # Fix 5: filter out computed-source trades where support was not respected
            # Fix 1: filter ALL sources (not just computed) when support not respected
            if (require_support_ok
                    and not result.support_respected):
                filtered += 1
                _log(
                    f"  [FILTERED] {result.ticker:6s} "
                    f"attempt={result.entry_attempt}  "
                    f"src={result.support_source}  "
                    f"support_ok=False — skipped (--require-support-ok)"
                )
                continue

            # Feature A: update compounded cumulative PnL
            cum_multiplier *= (1 + result.pnl_pct / 100)
            cum_pnl_pct = (cum_multiplier - 1) * 100

            results.append(result)
            _log(
                f"  [{result.outcome.upper():7s}] {result.ticker:6s} "
                f"attempt={result.entry_attempt}  "
                f"src={result.support_source}  "
                f"pat={result.pattern_type}(+{result.bars_since_pattern}b)  "
                f"entry={result.entry_price:.4f}  "
                f"exit={result.exit_price:.4f}  "
                f"pnl={result.pnl_pct:+.2f}%  "
                f"bars={result.hold_bars}  "
                f"support_ok={result.support_respected}  "
                f"cum={cum_pnl_pct:+.2f}%"
            )

    # Feature A: final cumulative PnL
    final_cum_pnl = (cum_multiplier - 1) * 100

    # Feature B: build SPY benchmark string
    spy_str = ""
    if spy_bh_pct is not None:
        spy_str = f"  spy_bh={spy_bh_pct:+.2f}%  ({spy_date_range})"
    elif spy_date_range:
        spy_str = f"  spy_bh=N/A  ({spy_date_range})"

    summary = (
        f"[simulator] END — {len(results)} trades / "
        f"{len(entries)} entries "
        f"({skipped} skipped/deduped, {filtered} filtered)  "
        f"cum_pnl={final_cum_pnl:+.2f}%"
        f"{spy_str}"
    )
    _log(summary)
    if log_path:
        _log(f"[simulator] Log written → {log_path}")
    return results
