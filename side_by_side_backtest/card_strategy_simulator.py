"""
Card Strategy Simulator
=======================
Backtests the "Card Strategy" across the full historical bars window by
replaying every trading session in the data.

For every (ticker, trading-date, session) combination:
  1. Slice bars UP TO the session-open bar (no look-ahead).
  2. Call score_setup() exactly as the Morning Brief does.
  3. If score >= min_score → enter at the open of the first session bar.
  4. Exit using the CARD's own TP/SL levels (SetupScore.resistance for TP,
     SetupScore.stop for SL) — not a fixed percentage.  This matches the
     real trading workflow where each card specifies its own levels.
  5. Budget / concurrent-position rules mirror the autonomous config:
       budget_total=$5,000 · trade_size=$500 · max_concurrent=10
       daily_loss_halt=$300 (per-strategy circuit breaker)

The result is a realistic replay of "what would have happened if you
traded every qualifying morning-brief card for the last 30 days."
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, time, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .models import SessionType, TradeResult, WatchlistEntry
from .sr_engine import SRLevels

logger = logging.getLogger(__name__)

# Truncate history passed to score_setup (matches setup_scorer._SCORE_BARS)
_SCORE_BARS = 500

# Session boundaries (UTC) — mirrors simulator.py
_SESSION_CLOSE_UTC: dict = {
    SessionType.PRE_MARKET:   time(13, 25),
    SessionType.MARKET_OPEN:  time(20, 0),
    SessionType.AFTER_HOURS:  time(23, 55),
    SessionType.UNKNOWN:      time(20, 0),
}
_SESSION_START_UTC: dict = {
    SessionType.PRE_MARKET:   time(9,  0),
    SessionType.MARKET_OPEN:  time(13, 30),
    SessionType.AFTER_HOURS:  time(20, 0),
    SessionType.UNKNOWN:      time(13, 30),
}

# Autonomous-config mirrors (kept here so simulator is self-contained)
_BUDGET_TOTAL     = 5_000.0   # total ring-fenced capital
_TRADE_SIZE       = 500.0     # $ per position
_MAX_CONCURRENT   = 100        # max open positions at once
_DAILY_LOSS_HALT  = 300.0     # halt day if daily PnL drops below -this

# Slippage applied to SL fills (Bug #6 fix).
# 0.05% is conservative for liquid small-caps; raise for illiquid stocks.
_SL_SLIPPAGE_PCT  = 0.0005    # 0.05%


def _bar_time_utc(ts: pd.Timestamp) -> time:
    utc = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    return utc.time()


def _ts_utc(ts) -> pd.Timestamp:
    """Return a UTC-aware pd.Timestamp regardless of input type/tz."""
    if not isinstance(ts, pd.Timestamp):
        ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _in_session(ts: pd.Timestamp, session: SessionType) -> bool:
    t = _bar_time_utc(ts)
    s = _SESSION_START_UTC.get(session, time(0))
    e = _SESSION_CLOSE_UTC.get(session, time(23, 59))
    if s <= e:
        return s <= t <= e
    return t >= s or t <= e


def _classify_session_regime(bars: pd.DataFrame, lookback_minutes: int = 60) -> str:
    """
    Classify the session regime as "trending" or "choppy" based on early price action.
    Counts VWAP crossings in the first N minutes of the session.
    """
    if bars.empty or len(bars) < 2:
        return "trending"
        
    n_bars = lookback_minutes // 5
    early_bars = bars.iloc[:n_bars]
    
    if early_bars.empty:
        return "trending"
        
    typical_price = (early_bars["close"] + early_bars["high"] + early_bars["low"]) / 3.0
    pv = typical_price * early_bars["volume"]
    cum_pv = pv.cumsum()
    cum_vol = early_bars["volume"].cumsum()
    vwap = cum_pv / cum_vol
    
    diff = early_bars["close"] - vwap
    crossings = ((diff.shift(1) * diff) < 0).sum()
    
    if crossings >= 4:
        return "choppy"
    return "trending"


# ---------------------------------------------------------------------------
# Day-by-day scoring: score EACH session in the full bars history
# ---------------------------------------------------------------------------

def _score_all_sessions(
    entry: WatchlistEntry,
    bars: pd.DataFrame,
    min_score: float,
    rescore_stride: int = 1,
    disable_sr_cache: bool = True,
    use_atr: bool = False,
    tp_atr_mult: float = 3.0,
    sl_atr_mult: float = 1.5,
    target_date: Optional[date] = None,
) -> List[Tuple[float, float, float, float, pd.Timestamp]]:
    """
    Replay every trading session present in *bars* and score at session open.

    For each calendar date that has bars in the target session window:
      - Build a history slice ending at the session-open bar (no look-ahead).
      - Call score_setup() exactly as Morning Brief does.
      - If score >= min_score, record (score, entry_price, tp_price, sl_price, entry_ts).

    Returns list of (score, entry_price, tp_price, sl_price, entry_ts) tuples,
    one per qualifying session.  TP/SL come from the card (SetupScore.resistance
    and SetupScore.stop) — not fixed percentages.

    Accuracy fixes:
      Bug #8 — Default fallback SL is now 2% below entry (was 0.124%).
      Bug #2 — score_setup() called with disable_sr_cache=True so each date
               gets S/R levels anchored to that date's price, not today's.
      Bug #1 — score_setup() receives cutoff_date so history_score only
               counts trades that existed at that point in time.
    """
    from .setup_scorer import score_setup

    session = entry.session_type

    if bars.empty:
        return []

    # Ensure UTC-aware index for consistent time comparisons
    if bars.index.tzinfo is None:
        bars = bars.copy()
        bars.index = bars.index.tz_localize("UTC")
    else:
        bars = bars.copy()
        bars.index = bars.index.tz_convert("UTC")

    from .pattern_engine import _atr
    atr_series = _atr(bars, period=14)

    # ── Find all unique calendar dates present in the bars ───────────────────
    # We iterate every date in the full 30-day window.  For each date we:
    #   a) Look for a "session open" bar using the session-window filter so
    #      we enter at the right time-of-day (realistic).
    #   b) If the session filter yields nothing (e.g. session_type=UNKNOWN on a
    #      ticker whose bars happen to fall outside 13:30–20:00 UTC), fall back
    #      to using ALL bars for that date so we still get an entry bar.
    #   c) Score using ALL bars up to (but not including) that date's first bar
    #      — same as Morning Brief which never session-filters the scoring slice.
    all_dates = sorted({ts.date() for ts in bars.index})
    if target_date is not None:
        all_dates = [d for d in all_dates if d == target_date]

    results: List[Tuple[float, float, float, float, pd.Timestamp]] = []

    for trading_date in all_dates:
        # All bars on this calendar date
        day_all_bars = bars[[ts.date() == trading_date for ts in bars.index]]
        if day_all_bars.empty:
            continue

        # Session-windowed bars for entry (fall back to all-day if no match)
        day_session_bars = day_all_bars[
            [_in_session(ts, session) for ts in day_all_bars.index]
        ]
        if day_session_bars.empty:
            day_session_bars = day_all_bars

        if len(day_session_bars) < 2:
            continue
            
        regime = _classify_session_regime(day_session_bars)
        print(f"[card_sim] {entry.ticker} {trading_date} regime: {regime}")

        # History before this day (no look-ahead)
        first_bar_ts = day_all_bars.index[0]
        pre_day_history = bars[bars.index < first_bar_ts]
        if len(pre_day_history) < 30:
            continue

        # ── Session entry cutoffs ─────────────────────────────────────────────
        # First entries: first 3 hours of regular session (9:30–12:30 ET = 13:30–17:00 UTC).
        # Re-entries: allowed up to 1:30 PM ET (18:30 UTC).
        _ENTRY_CUTOFF_UTC   = time(17, 0)    # 12:00 PM ET
        _REENTRY_CUTOFF_UTC = time(18, 30)   # 1:30 PM ET

        # ── Pre-compute day-level S/R once from pre-day history ──────────────
        # S/R levels don't change dramatically intraday for these small-cap
        # setups — computing them once per day is ~100× faster than per bar.
        try:
            from .sr_engine import compute_sr_levels as _csl_day
            _day_sr = _csl_day(
                pre_day_history.iloc[-200:] if len(pre_day_history) > 200 else pre_day_history,
                current_price=float(day_session_bars["open"].iloc[0]),
                price_range_pct=0.20,
                use_kmeans=False,
            )
        except Exception:
            _day_sr = None

        # ── Score every rescore_stride bars in the session ───────────────────
        # stride=1 → session open only (fast, for --all)
        # stride=4 → every 20 min (enables re-entry detection)
        for bar_idx in range(0, len(day_session_bars), max(1, rescore_stride)):
            bar_ts    = day_session_bars.index[bar_idx]
            intraday  = day_all_bars[day_all_bars.index < bar_ts]
            is_reentry = bar_idx > 0

            # ── Session cutoff filter ────────────────────────────────────────
            cutoff = _REENTRY_CUTOFF_UTC if is_reentry else _ENTRY_CUTOFF_UTC
            if _bar_time_utc(bar_ts) >= cutoff:
                break  # no more entries after cutoff

            if intraday.empty:
                history_for_scoring = pre_day_history
            else:
                combined = pd.concat([pre_day_history, intraday])
                history_for_scoring = combined.iloc[-_SCORE_BARS:] if len(combined) > _SCORE_BARS else combined

            # ── VWAP filter (re-entries only) ────────────────────────────────
            # Skip re-entry if price is below the session VWAP so far.
            # Only applies on re-entries (bar_idx > 0) — at bar 0 there are no
            # intraday bars yet so VWAP is undefined.
            entry_open = float(day_session_bars["open"].iloc[bar_idx])
            if not intraday.empty and "close" in intraday.columns and "volume" in intraday.columns:
                try:
                    vol_sum = intraday["volume"].sum()
                    if vol_sum > 0:
                        typical = (intraday["high"] + intraday["low"] + intraday["close"]) / 3
                        vwap = (typical * intraday["volume"]).sum() / vol_sum
                        if entry_open < vwap:
                            logger.debug(
                                "[card_sim] SKIP vwap: %s %s bar%d open=%.4f vwap=%.4f",
                                entry.ticker, trading_date, bar_idx, entry_open, vwap,
                            )
                            continue
                except Exception:
                    pass  # VWAP failure is non-fatal — allow the entry

            # ── EMA deceleration filter (re-entries only) ────────────────────
            # Block re-entry when the intraday EMA(8) is falling AND still
            # accelerating downward (no sign of reversal).
            if len(intraday) >= 4 and "close" in intraday.columns:
                ema8 = intraday["close"].ewm(span=8, adjust=False).mean()
                if len(ema8) >= 3:
                    slope_now  = float(ema8.iloc[-1] - ema8.iloc[-2])
                    slope_prev = float(ema8.iloc[-2] - ema8.iloc[-3])
                    if slope_now < 0 and slope_prev < 0 and slope_now < slope_prev:
                        logger.debug(
                            "[card_sim] SKIP ema-decel: %s %s bar%d",
                            entry.ticker, trading_date, bar_idx,
                        )
                        continue

            # ── Momentum Gate (all entries) ─────────────────────────────────
            if not intraday.empty and "close" in intraday.columns:
                ema8 = intraday["close"].ewm(span=8, adjust=False).mean()
                if float(day_session_bars["open"].iloc[bar_idx]) < float(ema8.iloc[-1]):
                    logger.debug(
                        "[card_sim] SKIP momentum gate: %s %s bar%d",
                        entry.ticker, trading_date, bar_idx,
                    )
                    continue

            try:
                sc = score_setup(
                    entry,
                    history_for_scoring,
                    None,
                    disable_sr_cache=disable_sr_cache,
                    cutoff_date=trading_date,
                )
            except Exception as exc:
                logger.debug(f"[card_sim] {entry.ticker} {trading_date} bar{bar_idx}: {exc}")
                continue

            if sc is None or sc.score < min_score:
                continue

            entry_price = float(day_session_bars["open"].iloc[bar_idx])
            entry_ts_   = bar_ts

            # ── TP: card resistance → cached day S/R (wide band) → 5% fallback
            # For first entries: always get a TP (5% fallback if S/R absent).
            # For re-entries: skip if no real resistance level found.
            # Resolve ATR for this bar
            # Always resolve current_atr for position sizing (Phase 1)
            try:
                current_atr = float(atr_series.loc[bar_ts])
            except KeyError:
                current_atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0

            if use_atr and current_atr > 0:
                tp_price = entry_price + current_atr * tp_atr_mult
                sl_price = entry_price - current_atr * sl_atr_mult
            else:
                tp_price = sc.resistance if (sc.resistance and sc.resistance > entry_price) else None
                if tp_price is None and _day_sr is not None:
                    tp_candidate = _day_sr.nearest_resistance(entry_price)
                    if tp_candidate and tp_candidate > entry_price:
                        tp_price = tp_candidate
                if tp_price is None:
                    if is_reentry:
                        logger.debug(
                            "[card_sim] SKIP no-real-tp (reentry): %s %s bar%d entry=%.4f",
                            entry.ticker, trading_date, bar_idx, entry_price,
                        )
                        continue
                    else:
                        tp_price = entry_price * (1.0 + _DEFAULT_TP_PCT)

                # ── SL: card stop → cached day support − 2.5% SL-hunt buffer ─────
                # Use cached _day_sr (fast) rather than recomputing per bar.
                sl_price = sc.stop if (sc.stop and sc.stop < entry_price) else None
                if is_reentry or sl_price is None:
                    computed_support = _day_sr.nearest_support(entry_price) if _day_sr else None
                    if computed_support and 0 < computed_support < entry_price:
                        sl_price = computed_support * (1.0 - 0.025)
                        sl_price = max(sl_price, entry_price * 0.95)
                    else:
                        default_pct = 0.03 if is_reentry else _DEFAULT_SL_PCT
                        sl_price = entry_price * (1.0 - default_pct)

            if regime == "choppy":
                # Tighten stop to max 2% on choppy days
                sl_price = max(sl_price, entry_price * 0.98)

            results.append((sc.score, entry_price, tp_price, sl_price, entry_ts_, sc.sr_levels, regime, current_atr))

    return results


# ---------------------------------------------------------------------------
# Per-session exit simulation (uses card's dollar TP/SL levels)
# ---------------------------------------------------------------------------

# Default TP fallback when no resistance level is found in the card or S/R engine.
# 5% matches typical small-cap intraday move targets — much more realistic than 1.5%.
_DEFAULT_TP_PCT = 0.08   # 8%
_DEFAULT_SL_PCT = 0.04   # 4%


def _simulate_from_entry(
    entry: WatchlistEntry,
    bars: pd.DataFrame,
    entry_price: float,
    tp_price: float,
    sl_price: float,
    entry_ts: pd.Timestamp,
    sr_levels: Optional[SRLevels] = None,
    regime: str = "trending",
    atr: float = 0.0,
) -> TradeResult:
    """
    Simulate exit from a confirmed entry using the card's own TP/SL dollar levels.

    Scans bars after entry_ts WITHIN THE SAME SESSION AND SAME CALENDAR DAY for:
      A. SL hit          : low  <= sl_price  → loss (fill with slippage)
      B. TP hit          : high >= tp_price  → win
      C. Session close time-stop → timeout (mark-to-market at last bar close)

    Both high and low are checked per bar so intrabar TP/SL are not missed.
    When both are hit on the same bar, SL takes priority (conservative).
    No arbitrary max-hold timer — position stays open until TP, SL, or session end,
    mirroring how the card strategy is actually traded.
    """
    session    = entry.session_type
    close_time = _SESSION_CLOSE_UTC.get(session, time(20, 0))

    # Derive pct equivalents for TradeResult fields (informational only)
    pt_pct = round((tp_price - entry_price) / entry_price * 100, 4) if entry_price > 0 else 0.0
    sl_pct = round((entry_price - sl_price) / entry_price * 100, 4) if entry_price > 0 else 0.0

    # Calculate Linda Raschke 3/10 Oscillator (SMA based)
    close_series = bars["close"]
    sma3 = close_series.rolling(window=3).mean()
    sma10 = close_series.rolling(window=10).mean()
    macd_line = sma3 - sma10
    signal_line = macd_line.rolling(window=16).mean()
    macd_hist = macd_line - signal_line
    macd_diff = macd_hist.diff()

    # ── Bug #3 fix: constrain exit scan to same calendar day + same session ──
    # The entry_ts is UTC-aware; derive the calendar date (UTC) once.
    entry_utc_date = entry_ts.tz_convert("UTC").date()

    # Filter post-entry bars to:
    #   1. After the entry bar
    #   2. Same UTC calendar date as the entry
    #   3. Within the same session window (same session type)
    # This prevents the exit loop from crossing into the next day's pre-market
    # or any future session — fixing the "free overnight hold" bug.
    all_post = bars[bars.index > entry_ts]
    if all_post.empty:
        # No bars at all after entry — timeout at entry price
        return TradeResult(
            ticker=entry.ticker,
            entry_ts=entry_ts.to_pydatetime(),
            entry_price=entry_price,
            exit_ts=entry_ts.to_pydatetime(),
            exit_price=round(entry_price, 4),
            profit_target_pct=pt_pct,
            stop_loss_pct=sl_pct,
            session_type=session,
            outcome="timeout",
            pnl_pct=0.0,
            hold_bars=0,
            support_respected=1,
            support_source="card",
            pattern_type="card_strategy",
            bars_since_pattern=0,
            entry_attempt=1,
            atr=atr,
        )

    # Vectorised date filter.
    # IMPORTANT: use a plain Python list (or numpy array) as the boolean mask —
    # pd.array(..., dtype=bool) returns an ExtensionArray which pandas interprets
    # as column labels rather than a row selector, causing the
    # "None of [Index([True, True, ...])] are in the [columns]" error.
    try:
        bar_utc_dates = all_post.index.tz_convert("UTC").date
    except AttributeError:
        bar_utc_dates = pd.DatetimeIndex(all_post.index).tz_convert("UTC").date

    same_day_mask = [d == entry_utc_date for d in bar_utc_dates]
    same_day_bars = all_post.loc[same_day_mask]

    # Further restrict to same session window
    session_mask = [_in_session(ts, session) for ts in same_day_bars.index]
    post_bars = same_day_bars.loc[session_mask]

    def _mk_result(ts, exit_price: float, outcome: str, hold: int) -> TradeResult:
        return TradeResult(
            ticker=entry.ticker,
            entry_ts=entry_ts.to_pydatetime(),
            entry_price=entry_price,
            exit_ts=ts.to_pydatetime(),
            exit_price=round(exit_price, 4),
            profit_target_pct=pt_pct,
            stop_loss_pct=sl_pct,
            session_type=session,
            outcome=outcome,
            pnl_pct=round((exit_price - entry_price) / entry_price * 100, 4),
            hold_bars=hold,
            support_respected=0 if outcome == "loss" else 1,
            support_source="card",
            pattern_type="card_strategy",
            bars_since_pattern=0,
            entry_attempt=1,
            atr=atr,
        )

    highest_high = entry_price
    trail_active = False
    trail_activation_pct = 2.0
    trailing_stop_pct = 1.0

    for i, (ts, bar) in enumerate(post_bars.iterrows()):
        high  = float(bar.get("high",  bar.get("close", entry_price)))
        low   = float(bar.get("low",   bar.get("close", entry_price)))
        close = float(bar.get("close", entry_price))

        # Check MACD momentum (Linda Raschke 3/10)
        last_diffs = macd_diff.loc[:ts].iloc[-3:]
        if len(last_diffs) >= 3 and (last_diffs < 0).all():
            # Momentum weakening! Tighten stop to max 1% below highest_high
            trail_stop = highest_high * 0.99
            if trail_stop > sl_price:
                sl_price = trail_stop
                
            # Tighten TP if position is in profit
            if close > entry_price:
                tp_price = min(tp_price, close * 1.005)

        if high > highest_high:
            highest_high = high
            if not trail_active and (highest_high - entry_price) / entry_price * 100 >= trail_activation_pct:
                trail_active = True
                
        if trail_active:
            trail_stop = highest_high * (1 - trailing_stop_pct / 100)
            if low <= trail_stop:
                return _mk_result(ts, trail_stop, "win", i + 1)

        if low <= sl_price:
            raw_fill   = min(sl_price, open)
            fill_price = raw_fill * (1.0 - _SL_SLIPPAGE_PCT)
            return _mk_result(ts, fill_price, "loss", i + 1)

        if high >= tp_price:
            if regime == "choppy":
                return _mk_result(ts, tp_price, "win", i + 1)
            else:
                trail_active = True

        if _bar_time_utc(ts) >= close_time:
            return _mk_result(ts, close, "timeout", i + 1)

        # Stall Exit: exit if trade is underwater after 30 minutes (6 bars)
        if i >= 5 and close < entry_price:
            return _mk_result(ts, close, "loss", i + 1)
        if i >= 5 and close < entry_price:
            return _mk_result(ts, close, "loss", i + 1)

    # End of same-session bars for this day — time-stop at last available bar.
    if not post_bars.empty:
        last_ts    = post_bars.index[-1]
        last_close = float(post_bars["close"].iloc[-1])
    elif not same_day_bars.empty:
        last_ts    = same_day_bars.index[-1]
        last_close = float(same_day_bars["close"].iloc[-1])
    else:
        last_ts    = entry_ts
        last_close = entry_price

    return TradeResult(
        ticker=entry.ticker,
        entry_ts=entry_ts.to_pydatetime(),
        entry_price=entry_price,
        exit_ts=last_ts.to_pydatetime(),
        exit_price=round(last_close, 4),
        profit_target_pct=pt_pct,
        stop_loss_pct=sl_pct,
        session_type=session,
        outcome="timeout",
        pnl_pct=round((last_close - entry_price) / entry_price * 100, 4),
        hold_bars=len(post_bars),
        support_respected=1,
        support_source="card",
        pattern_type="card_strategy",
        bars_since_pattern=0,
        entry_attempt=1,
        atr=atr,
    )


def _score_ticker_top(
    entry: WatchlistEntry,
    bars: Optional[pd.DataFrame],
    min_score: float,
    rescore_stride: int,
    disable_sr_cache: bool,
    use_atr: bool,
    tp_atr_mult: float,
    sl_atr_mult: float,
    target_date: Optional[date] = None,
) -> List[Tuple[WatchlistEntry, float, float, float, float, pd.Timestamp, Optional[SRLevels], str, float]]:
    if bars is None or bars.empty:
        return []
    sessions = _score_all_sessions(entry, bars, min_score,
                                   rescore_stride=rescore_stride,
                                   disable_sr_cache=disable_sr_cache,
                                   use_atr=use_atr,
                                   tp_atr_mult=tp_atr_mult,
                                   sl_atr_mult=sl_atr_mult,
                                   target_date=target_date)
    return [(entry, sc, ep, tp, sl, ets, srl, reg, atr) for sc, ep, tp, sl, ets, srl, reg, atr in sessions]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate_card_strategy(
    entries: List[WatchlistEntry],
    bars_map: Dict[str, pd.DataFrame],
    min_score: float = 4.3,
    profit_target_pct: float = 1.5,   # kept for API compat; ignored — card levels used
    stop_loss_pct: float = 0.124,      # kept for API compat; ignored — card levels used
    db=None,
    max_workers: int = 8,
    verbose: bool = False,
    budget_total: float = _BUDGET_TOTAL,
    trade_size: float = _TRADE_SIZE,
    max_concurrent: int = _MAX_CONCURRENT,
    daily_loss_halt: float = _DAILY_LOSS_HALT,
    reentry_min_score: float | None = None,   # None = min_score + 0.5
    rescore_stride: int = 4,   # bars between re-scores; 1 = session open only (fast)
    disable_sr_cache: bool = True,   # False = reuse S/R cache across days (faster batch)
    use_atr: bool = False,
    tp_atr_mult: float = 3.0,
    sl_atr_mult: float = 1.5,
    target_date: Optional[date] = None,
) -> List[TradeResult]:
    """
    Replay the card strategy across the FULL historical bars window.

    For every ticker in bars_map this function:
      1. Iterates every trading session present in the bars (day-by-day).
      2. Scores at session open using ONLY bars available up to that point
         (strict no look-ahead — mirrors the Morning Brief workflow).
      3. If score >= min_score, enters at the open of the first session bar.
      4. Exits using the card's own TP/SL DOLLAR levels (SetupScore.resistance
         for TP, SetupScore.stop for SL).
      5. Applies autonomous-config budget rules:
           - max_concurrent open positions at once
           - trade_size $ per position
           - daily_loss_halt $ circuit breaker per calendar day

    The result is a realistic replay of "what would have happened if you
    traded every qualifying morning-brief card every day for the last 30 days."
    """
    # db param kept for API compat with callers that pass db=_db, but score_setup
    # uses db_path=None (default path) directly — no DB connection needed here.
    # Resolve re-entry threshold: default = min_score + 0.5
    _reentry_threshold = reentry_min_score if reentry_min_score is not None else min_score + 0.5

    try:
        # De-duplicate entries by ticker (keep one per ticker — levels come
        # from the watchlist card, but the backtest ignores those fixed levels
        # in favour of computed S/R, so duplicates would just produce identical
        # sessions twice)
        seen_tickers: set = set()
        unique_entries: List[WatchlistEntry] = []
        for e in entries:
            if e.ticker not in seen_tickers:
                seen_tickers.add(e.ticker)
                unique_entries.append(e)

        if verbose:
            print(f"[card_sim] {len(unique_entries)} unique tickers, "
                  f"min_score={min_score}, reentry≥{_reentry_threshold}, workers={max_workers}")

        # ── Step 1: score all sessions for every ticker in parallel ──────────
        # Each worker returns a list of (entry, score, ep, tp, sl, ets) tuples.
        all_candidates: List[Tuple[WatchlistEntry, float, float, float, float, pd.Timestamp]] = []

        n_workers = min(max_workers, max(1, len(unique_entries)))
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = {
                pool.submit(
                    _score_ticker_top,
                    e,
                    bars_map.get(e.ticker),
                    min_score,
                    rescore_stride,
                    disable_sr_cache,
                    use_atr,
                    tp_atr_mult,
                    sl_atr_mult,
                    target_date,
                ): e
                for e in unique_entries
            }
            for fut in as_completed(futs):
                try:
                    results = fut.result()
                    if results:
                        all_candidates.extend(results)
                except Exception as exc:
                    logger.debug(f"[card_sim] scoring error: {exc}")

        if verbose:
            print(f"[card_sim] {len(all_candidates)} qualifying session entries. "
                  f"Simulating exits with budget rules …")

        # ── Step 2: sort candidates chronologically then simulate exits ───────
        # Apply budget / concurrent-position rules in time order so the
        # simulation mirrors real autonomous trading constraints.
        all_candidates.sort(key=lambda x: x[5])   # sort by entry_ts

        # Re-entry cooldown: bars to wait after a trade exits before re-entering
        # the same ticker.  Base = 20 min; extended to 40 min after a loss so
        # we don't chase a falling ticker.
        _REENTRY_COOLDOWN      = pd.Timedelta(minutes=20)
        _REENTRY_COOLDOWN_LOSS = pd.Timedelta(minutes=40)

        trades: List[TradeResult] = []
        seen_dedup: set = set()

        # Budget state
        open_positions: List[dict] = []   # {ticker, entry_ts, exit_ts}
        daily_pnl: dict = {}              # {date: cumulative $ PnL that day}
        last_exit_ts: dict = {}           # {ticker: (last_exit_UTC, outcome)}
        banned_same_day: dict = {}        # {(ticker, date): True} — tickers banned after a loss

        _dbg_n = 0   # count candidates examined for first-pass debug
        for entry, score, ep, tp, sl, ets, sr_levels, regime, atr in all_candidates:
            dedup_key = (entry.ticker, str(ets))
            if dedup_key in seen_dedup:
                continue
            seen_dedup.add(dedup_key)

            # ── Re-entry cooldown ─────────────────────────────────────────────
            # Allow re-entry into the same ticker only after _REENTRY_COOLDOWN
            # has elapsed since the previous exit.  20-min after win/stall/timeout,
            # 40-min after a loss so we don't re-enter a falling ticker.
            ets_utc_ts = _ts_utc(ets)
            prev_exit_info = last_exit_ts.get(entry.ticker)   # (ts, outcome) or None
            if prev_exit_info is not None:
                prev_ts, prev_outcome = prev_exit_info
                cooldown = (_REENTRY_COOLDOWN_LOSS
                            if prev_outcome == "loss"
                            else _REENTRY_COOLDOWN)
                if ets_utc_ts < prev_ts + cooldown:
                    logger.debug(
                        "[card_sim] SKIP cooldown (%s): %s ets=%s prev_exit=%s",
                        prev_outcome, entry.ticker, ets, prev_ts,
                    )
                    continue

                # Re-entry requires a higher score threshold to avoid chasing
                # a ticker that just proved it can move against us.
                if score < _reentry_threshold:
                    logger.debug(
                        "[card_sim] SKIP reentry-score: %s score=%.2f < %.2f",
                        entry.ticker, score, _reentry_threshold,
                    )
                    continue

            # ── Same-day loss ban ─────────────────────────────────────────────
            # After a loss on ticker X on a given calendar day, ban that ticker
            # for the rest of that day.  ENVB losing 3× on Apr 20 is the
            # canonical example this prevents.
            trade_date_check = ets.date() if hasattr(ets, "date") else ets.to_pydatetime().date()
            if banned_same_day.get((entry.ticker, trade_date_check)):
                logger.debug(
                    "[card_sim] SKIP same-day-ban: %s date=%s",
                    entry.ticker, trade_date_check,
                )
                continue

            bars = bars_map.get(entry.ticker)
            if bars is None or bars.empty:
                if _dbg_n < 5:
                    logger.warning("[card_sim] SKIP no-bars: %s ets=%s", entry.ticker, ets)
                _dbg_n += 1
                continue
            # Normalise to UTC-aware so index comparisons with ets (UTC) never raise.
            if bars.index.tzinfo is None:
                bars = bars.copy()
                bars.index = bars.index.tz_localize("UTC")
            elif str(bars.index.tzinfo) != "UTC":
                bars = bars.copy()
                bars.index = bars.index.tz_convert("UTC")

            trade_date = ets.date() if hasattr(ets, "date") else ets.to_pydatetime().date()

            # ── Circuit breaker: daily loss halt ─────────────────────────────
            day_pnl_dollars = daily_pnl.get(trade_date, 0.0)
            if day_pnl_dollars <= -daily_loss_halt:
                if _dbg_n < 5:
                    logger.warning("[card_sim] SKIP daily-halt: %s date=%s pnl=$%.2f",
                                   entry.ticker, trade_date, day_pnl_dollars)
                _dbg_n += 1
                continue

            # ── Max concurrent positions ──────────────────────────────────────
            # Expire positions that have already closed before this entry.
            # Ensure tz-aware comparison: make exit_ts UTC-aware if needed.
            ets_utc = ets if getattr(ets, "tzinfo", None) else pd.Timestamp(ets, tz="UTC")
            open_positions = [
                p for p in open_positions
                if _ts_utc(p["exit_ts"]) > ets_utc
            ]
            if len(open_positions) >= max_concurrent:
                if _dbg_n < 5:
                    logger.warning("[card_sim] SKIP max-concurrent: %s ets=%s open=%d",
                                   entry.ticker, ets, len(open_positions))
                _dbg_n += 1
                continue

            # ── Simulate exit ─────────────────────────────────────────────────
            try:
                trade = _simulate_from_entry(entry, bars, ep, tp, sl, ets, sr_levels, regime, atr)
            except Exception as exc:
                logger.warning("[card_sim] SKIP exit-error %s ets=%s: %s",
                                entry.ticker, ets, exc)
                _dbg_n += 1
                continue

            # ── Dollar PnL and position tracking ─────────────────────────────
            # Score-driven position sizing
            dynamic_trade_size = trade_size + 200 * (score - 4.3)
            dynamic_trade_size = max(100.0, min(1000.0, dynamic_trade_size))
            shares = dynamic_trade_size / ep if ep > 0 else 0.0
            dollar_pnl = shares * (trade.exit_price - ep)
            daily_pnl[trade_date] = daily_pnl.get(trade_date, 0.0) + dollar_pnl

            exit_ts_dt = _ts_utc(trade.exit_ts)
            open_positions.append({
                "ticker":   entry.ticker,
                "entry_ts": ets,
                "exit_ts":  exit_ts_dt,
            })

            # Record exit time + outcome for re-entry cooldown tracking
            last_exit_ts[entry.ticker] = (exit_ts_dt, trade.outcome)

            # Same-day loss ban: no more entries for this ticker today after a loss
            if trade.outcome == "loss":
                ban_date = trade_date if isinstance(trade_date, type(trade_date)) else trade_date
                banned_same_day[(entry.ticker, ban_date)] = True

            trades.append(trade)

        if verbose:
            wins = sum(1 for t in trades if t.outcome == "win")
            wr   = wins / len(trades) if trades else 0
            print(f"[card_sim] Done — {len(trades)} trades, WR={100*wr:.1f}%")

        return trades

    finally:
        pass  # no DB connection opened here — score_setup uses its own default path
