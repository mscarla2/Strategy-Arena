"""
Setup Scorer — Core Signal Engine
==================================
Scores a WatchlistEntry (0–10) using five component signals, each worth 0–2 pts:

  1. pattern_score  — Side-by-Side White Lines near support
  2. adx_score      — Trending market strength (ADX)
  3. rr_score       — Risk/Reward ratio (support + resistance → ratio)
  4. confluence_score — How many S/R methods agree on the support level
  5. history_score  — Historical win-rate for this ticker from the trade DB

Public API
----------
    from side_by_side_backtest.setup_scorer import score_setup, SetupScore

    score = score_setup(entry, bars_df, db)
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

import time as _time

import pandas as pd
from .sr_engine import SRLevels

# ---------------------------------------------------------------------------
# Performance constants
# ---------------------------------------------------------------------------
_SCORE_BARS = 500        # use last N bars for all scorer computations (3500 → 500)
_SR_CACHE_TTL = 300      # seconds to cache compute_sr_levels() per ticker
_sr_cache: dict = {}     # {ticker: (timestamp, SRLevels)}
_sr_cache_lock = threading.Lock()   # guards concurrent reads/writes from outer pool

# History cache — all trades loaded from SQLite once per TTL, shared across all tickers.
# The previous approach opened a fresh connection per ticker per scoring cycle, which
# could cost 85ms per call (SQLite cold-open latency × N tickers).
_HISTORY_CACHE_TTL = 60   # seconds — refreshes on each 1-min live-scan cycle
_history_cache: dict = {}  # {"_all": (timestamp, {ticker: (score, win_rate, n_trades)})}
_history_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class SetupScore:
    """Composite score for a single watchlist setup. Each component is 0–2 pts."""

    ticker: str
    score: float                 # 0.0 – 10.0 total
    signal: str                  # "🟢 STRONG" | "🟡 WATCH" | "🔴 SKIP"

    # Component scores
    pattern_score:      float = 0.0   # 0 = none, 1 = near support, 2 = confirmed
    adx_score:          float = 0.0   # 0 = ADX<15, 1 = 15–25, 2 = >25
    rr_score:           float = 0.0   # 0 = no levels, 1 = R/R<1.5, 2 = R/R≥2.0
    confluence_score:   float = 0.0   # 0 = none, 1 = 1 method, 2 = 2+ methods agree
    history_score:      float = 0.0   # 0 = no data, 1 = WR<50%, 2 = WR≥60%

    # Extended history fields (enhancement #1 and #2)
    history_avg_win:    float = 0.0   # avg pnl% of winning trades for this ticker
    history_avg_loss:   float = 0.0   # avg pnl% of losing trades for this ticker
    history_expectancy: float = 0.0   # wr*avg_win + (1-wr)*avg_loss
    history_streak:     str   = ""    # last-5 outcomes e.g. "✅✅❌✅✅"
    role_reversal_score: float = 0.0  # 0 = no flip, 1 = possible, 2 = confirmed R→S flip
    rejection_score:    float = 0.0   # 0 = 0 rejections, 1 = 1, 2 = 2+ wick rejections
    rel_vol_score:      float = 0.0   # 0 = <1.5×, 1 = 1.5-2×, 2 = ≥2× same-TOD volume
    macd_score:         float = 0.0   # 0 = hist falling, 1 = rising 1-bar, 2 = rising 3-bar
    rsi_div_score:      float = 0.0   # 0 = none, 1 = partial, 2 = confirmed divergence
    regime_score:       float = 0.0   # 0 = counter-trend, 1 = flat, 2 = aligned down
    eqh_score:          float = 0.0   # 0 = no EQH pair, 1 = pair found nearby, 2 = approaching/signal fired
    sr_levels:          Optional[SRLevels] = None

    # Raw data for the card display
    entry_price:      float = 0.0
    support:          float = 0.0
    resistance:       Optional[float] = None
    stop:             float = 0.0
    rr_ratio:         float = 0.0
    adx:              float = 0.0
    pattern_found:    bool  = False
    pattern_type:     str   = ""      # "strict" | "exhaustion" | "absorption" | ""
    role_reversal:    bool  = False   # support was formerly resistance
    rejection_count:  int   = 0       # wick rejections off support
    support_broken:   bool  = False   # current price is below the watchlist support level
    support_ok:       bool  = True    # recent closes hold above support (aligns with simulator gate)
    watchlist_note:   str   = ""      # original text from the post
    atr:              float = 0.0     # ATR at entry for sizing

    # EQH (Equal Highs / Liquidity Ceiling) fields
    eqh_level:        float = 0.0    # EQH ceiling price (0.0 if none found)
    eqh_signal:       str   = ""     # "" | "approaching" | "eqh_breakout" | "eqh_rejection"

    def signal_label(self) -> str:
        """Return coloured signal string based on total score."""
        if self.score >= 7.0:
            return "🟢 STRONG"
        if self.score >= 4.0:
            return "🟡 WATCH"
        return "🔴 SKIP"


# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------

def _score_pattern(bars: pd.DataFrame, support: Optional[float]) -> tuple[float, bool, str]:
    """
    Pattern score (0–2), weighted by confidence_score on PatternMatch.
    Returns (score, pattern_found, pattern_type).
    """
    from .pattern_engine import (
        detect_side_by_side,
        pattern_near_support,
        detect_exhaustion_side_by_side,
        detect_support_absorption,
        _body_low,
    )

    if bars.empty:
        return 0.0, False, ""

    recent = bars.iloc[-30:] if len(bars) >= 30 else bars

    # Pre-compute ADX once — detectors skip recomputation if column exists
    from .pattern_engine import _adx as _pe_adx
    recent = recent.copy()
    if "adx" not in recent.columns:
        recent["adx"] = _pe_adx(recent, 14)
    best_score = 0.0
    found = False
    best_type = ""

    if support and support > 0:
        for m in pattern_near_support(recent, support, proximity_pct=0.005):
            s = 2.0 * m.confidence_score
            if s > best_score:
                best_score, found, best_type = s, True, m.pattern_type

        for m in detect_exhaustion_side_by_side(recent):
            c3_o = m.candle3_open if m.candle3_open != 0.0 else m.candle2_open
            c3_c = m.candle3_close if m.candle3_open != 0.0 else m.candle2_close
            c3_low = _body_low(c3_o, c3_c)
            if abs(c3_low - support) / support <= 0.008:
                s = 2.0 * m.confidence_score
                if s > best_score:
                    best_score, found, best_type = s, True, m.pattern_type

        for m in detect_support_absorption(recent, support=support, proximity_pct=0.01):
            s = 2.0 * m.confidence_score
            if s > best_score:
                best_score, found, best_type = s, True, m.pattern_type

    if best_score > 0:
        return round(min(best_score, 2.0), 2), found, best_type

    # Anywhere in window (not at support)
    for m in detect_side_by_side(recent):
        s = 1.0 * m.confidence_score
        if s > best_score:
            best_score, found, best_type = s, True, m.pattern_type
    for m in detect_exhaustion_side_by_side(recent):
        s = 1.0 * m.confidence_score
        if s > best_score:
            best_score, found, best_type = s, True, m.pattern_type

    return round(min(best_score, 2.0), 2), found, best_type


def _score_adx(bars: pd.DataFrame) -> tuple[float, float]:
    """
    ADX score (0–2) using the last bar's ADX value.
      2 = ADX > 25
      1 = ADX 15–25
      0 = ADX < 15 or unavailable
    Returns (score, adx_value).
    """
    from .pattern_engine import _adx  # reuse existing helper

    if bars.empty or len(bars) < 30:
        return 0.0, 0.0

    adx_series = _adx(bars, period=14)
    adx_val = float(adx_series.iloc[-1]) if not adx_series.empty else 0.0

    if pd.isna(adx_val):
        return 0.0, 0.0
    if adx_val > 25:
        return 2.0, adx_val
    if adx_val >= 15:
        return 1.0, adx_val
    return 0.0, adx_val


def _score_rr(
    support: Optional[float],
    resistance: Optional[float],
    stop: Optional[float],
    entry: float,
) -> tuple[float, float, float]:
    """
    R/R score (0–2).
      2 = R/R ≥ 2.0
      1 = R/R 1.0–1.99
      0 = missing levels or R/R < 1.0
    Returns (score, rr_ratio, stop_price).
    """
    if not support or not resistance or support <= 0 or resistance <= entry:
        # No valid TP — R/R cannot be computed. stop_price falls back to the
        # already-computed SL (entry * 0.98) that score_setup sets before calling us,
        # but since we don't have it here, return entry * 0.98 as the best default.
        return 0.0, 0.0, entry * 0.98

    # Stop = explicit stop if given; otherwise 1% below entry (matches simulator SL default).
    # Previously used support * 0.98 — but the simulator now treats SL as entry * (1 - sl%),
    # not as a support-relative value.  Use entry * 0.99 so R/R on the card is consistent.
    stop_price = stop if (stop and stop > 0) else entry * 0.99
    risk   = entry - stop_price
    reward = resistance - entry

    if risk <= 0:
        return 0.0, 0.0, stop_price

    rr = min(reward / risk, 10.0)   # cap at 10 — anything higher is suspect
    if rr >= 2.0:
        return 2.0, round(rr, 2), stop_price
    if rr >= 1.0:
        return 1.0, round(rr, 2), stop_price
    return 0.0, round(rr, 2), stop_price


def _score_confluence(bars: pd.DataFrame, support: Optional[float], ticker: str = "") -> tuple[float, str]:
    """
    Confluence score (0–2): S/R methods agreeing on the support level.
    Optimised: results cached per ticker for _SR_CACHE_TTL seconds.
    K-Means disabled for speed (biggest single speedup).
    """
    from .sr_engine import compute_sr_levels

    if not support or support <= 0 or bars.empty:
        return 0.0, "no support level"

    # Cache lookup — lock protects concurrent dict access from the outer per-ticker pool
    now_t = _time.monotonic()
    with _sr_cache_lock:
        cached = _sr_cache.get(ticker)
    if cached and (now_t - cached[0]) < _SR_CACHE_TTL:
        levels = cached[1]
    else:
        levels = compute_sr_levels(
            bars, current_price=support, price_range_pct=0.05,
            use_kmeans=False,  # K-Means is slowest method; skip for live scoring
        )
        if ticker:
            with _sr_cache_lock:
                _sr_cache[ticker] = (now_t, levels)
    threshold = support * 0.003     # 0.3% band

    agreeing_methods: list[str] = []
    for lv in levels.all_levels:
        if abs(lv.price - support) <= threshold:
            agreeing_methods.append(lv.method)

    # VPOC proximity bonus: support near POC = institutional memory
    vpoc_bonus = False
    poc_levels = [lv for lv in levels.all_levels if lv.method == "poc"]
    for poc in poc_levels:
        if abs(poc.price - support) / support <= 0.005:   # within 0.5%
            vpoc_bonus = True
            if "poc" not in agreeing_methods:
                agreeing_methods.append("poc_bonus")
            break

    n = len(agreeing_methods)
    desc = ", ".join(agreeing_methods[:4]) or "none"
    if vpoc_bonus:
        desc += " 🏦VPOC"
    if n >= 2:
        return 2.0, f"{n} methods agree ({desc})"
    if n == 1:
        return 1.0, f"1 method ({desc})"
    return 0.0, "no confluence"


def _load_all_history(
    db_path: Optional[str] = None,
    cutoff_date=None,
) -> dict:
    """
    Load trades from the DB and return a {ticker: (score, win_rate, n, ...)} lookup.

    Results are cached in _history_cache for _HISTORY_CACHE_TTL seconds so the
    expensive SQLite open only happens once per live-scan interval.

    cutoff_date : Optional[date]
        When provided (backtest context), only trades with entry_ts BEFORE this
        date are counted.  This prevents future trades from leaking into the
        history score during historical replay (Bug #1 fix).
        When None (live scoring), all trades are included as before.
    """
    from pathlib import Path

    now_t = _time.monotonic()

    # Cache key: use "_all" for live (no cutoff) or the ISO date string for backtest.
    # This ensures that live scoring still hits the fast path, while each historical
    # date gets its own isolated snapshot without polluting the live cache.
    cache_key = "_all" if cutoff_date is None else str(cutoff_date)

    with _history_cache_lock:
        cached = _history_cache.get(cache_key)
    if cached and (now_t - cached[0]) < _HISTORY_CACHE_TTL:
        return cached[1]

    # Cache miss — open DB and build the lookup
    try:
        from .db import WatchlistDB
    except (ImportError, KeyError):
        try:
            from side_by_side_backtest.db import WatchlistDB
        except Exception:
            return {}

    path = db_path or str(Path(__file__).parent / "watchlist_backtest.db")
    try:
        with WatchlistDB(path) as db:
            all_trades = db.load_trades()
    except Exception:
        return {}

    # Bug #1 fix: when a cutoff_date is provided, drop trades on/after that date.
    # This stops future-trade data from inflating the history_score during backtest.
    if cutoff_date is not None:
        from datetime import date as _date
        filtered = []
        for t in all_trades:
            try:
                ts = t.entry_ts
                # entry_ts can be a datetime or an ISO string
                if isinstance(ts, str):
                    from datetime import datetime as _dt
                    ts = _dt.fromisoformat(ts)
                trade_date = ts.date() if hasattr(ts, "date") else ts
                if trade_date < cutoff_date:
                    filtered.append(t)
            except Exception:
                filtered.append(t)   # parse error → include (safe default)
        all_trades = filtered

    # Build per-ticker summary
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for t in all_trades:
        groups[t.ticker.upper()].append(t)

    lookup: dict = {}
    for tkr, trades in groups.items():
        total = len(trades)

        # Sort chronologically for streak calculation
        sorted_trades = sorted(trades, key=lambda t: getattr(t, "entry_ts", ""))

        if total < 3:
            lookup[tkr] = (0.0, 0.0, total, 0.0, 0.0, 0.0, "")
            continue

        wins       = [t for t in sorted_trades if t.outcome == "win"]
        losses     = [t for t in sorted_trades if t.outcome == "loss"]
        n_wins     = len(wins)
        wr         = n_wins / total
        avg_win    = sum(t.pnl_pct for t in wins)  / n_wins  if wins   else 0.0
        avg_loss   = sum(t.pnl_pct for t in losses)/ len(losses) if losses else 0.0
        expectancy = wr * avg_win + (1 - wr) * avg_loss

        # Last-5 streak: "✅✅❌✅✅" using the 5 most recent trades
        last5 = sorted_trades[-5:]
        streak_str = "".join("✅" if t.outcome == "win" else "❌" for t in last5)

        if wr >= 0.60:
            score = 2.0
        elif wr >= 0.30:
            score = 1.0
        else:
            score = 0.0

        lookup[tkr] = (
            score,
            round(wr, 3),
            total,
            round(avg_win, 3),
            round(avg_loss, 3),
            round(expectancy, 3),
            streak_str,
        )

    with _history_cache_lock:
        _history_cache[cache_key] = (now_t, lookup)
    return lookup


def _score_history(
    ticker: str,
    db_path: Optional[str] = None,
    cutoff_date=None,
) -> tuple:
    """
    History score (0–2) from the trade DB win-rate for this ticker.
      2 = win-rate ≥ 60%
      1 = win-rate 30–59%
      0 = fewer than 3 trades or win-rate < 30%
    Returns (score, win_rate, total_trades, avg_win, avg_loss, expectancy, streak_str).

    cutoff_date : Optional[date]
        When provided, only trades before this date are counted (Bug #1 fix).

    Optimised: loads all trades once per _HISTORY_CACHE_TTL (60s) and caches
    the per-ticker lookup dict — eliminates the N × SQLite-open overhead.
    """
    lookup = _load_all_history(db_path, cutoff_date=cutoff_date)
    result = lookup.get(ticker.upper())
    if result is None:
        return 0.0, 0.0, 0, 0.0, 0.0, 0.0, ""
    # Unpack: supports both old 3-tuple (legacy) and new 7-tuple
    if len(result) == 7:
        return result
    score, wr, total = result[:3]
    return score, wr, total, 0.0, 0.0, 0.0, ""


def _score_role_reversal(bars: pd.DataFrame, support: Optional[float]) -> tuple[float, bool]:
    """
    Role-Reversal score (0–2).
      2 = confirmed flip: support was previously resistance (≥3 closes above it,
          then price broke down and is now retesting from above)
      1 = possible flip: fewer closes above, or borderline retest
      0 = no evidence of role reversal
    Returns (score, is_role_reversal).
    """
    from .sr_engine import detect_role_reversals

    if not support or support <= 0 or bars.empty:
        return 0.0, False

    confirmed = detect_role_reversals(bars, support, min_closes_above=3)
    if confirmed:
        return 2.0, True

    # Softer check — at least 1 close above in recent 50 bars
    soft = detect_role_reversals(bars, support,
                                 min_closes_above=1, lookback_bars=50)
    if soft:
        return 1.0, False

    return 0.0, False


def _score_rejections(bars: pd.DataFrame, support: Optional[float]) -> tuple[float, int]:
    """
    Rejection score (0–2): multiple wick rejections off support = strong base.
      2 = 2+ rejection candles (low touched support, closed above)
      1 = exactly 1 rejection
      0 = no rejections
    Returns (score, count).
    """
    from .sr_engine import count_rejections

    if not support or support <= 0 or bars.empty:
        return 0.0, 0

    n = count_rejections(bars, support, lookback_bars=60)
    if n >= 2:
        return 2.0, n
    if n == 1:
        return 1.0, n
    return 0.0, 0


def _score_relative_volume(bars: pd.DataFrame) -> tuple[float, float]:
    """
    Relative volume score (0–2) on C1 (the momentum candle).
    Compares C1's volume to the median volume of bars at the same time-of-day
    over the prior N sessions.
      2 = C1 volume ≥ 2× median same-time volume
      1 = C1 volume 1.5–1.99×
      0 = below 1.5× or insufficient data
    Returns (score, relative_volume_ratio).
    """
    if bars.empty or "volume" not in bars.columns or len(bars) < 20:
        return 0.0, 0.0

    # Use the most recent C1 candidate = bar with largest body in last 15 bars
    recent = bars.iloc[-15:].copy()
    recent["body"] = (recent["close"] - recent["open"]).abs()
    if recent.empty:
        return 0.0, 0.0
    # Use integer position (not label) to avoid Timestamp-to-int error
    c1_pos_in_recent = int(recent["body"].values.argmax())
    c1_pos_in_bars   = len(bars) - 15 + c1_pos_in_recent

    c1_vol  = float(bars.iloc[c1_pos_in_bars]["volume"])
    if c1_vol <= 0:
        return 0.0, 0.0

    # ── Vectorised same-TOD comparison (replaces Python list comprehension) ────
    # Old approach iterated bar-by-bar with .time() calls → O(N) Python overhead.
    # New approach: use pandas .dt accessor to compute minute-of-day as an integer
    # Series, then boolean-mask for the historical window in one vectorised op.
    try:
        c1_ts = bars.index[c1_pos_in_bars]
        c1_min = c1_ts.hour * 60 + c1_ts.minute   # scalar int

        hist = bars.iloc[: len(bars) - 15]          # historical window (excludes last 15)
        if hist.empty:
            return 0.0, 0.0

        # Vectorised minute-of-day for all historical bars
        hist_idx = hist.index
        if hasattr(hist_idx, "hour"):
            bar_mins = hist_idx.hour * 60 + hist_idx.minute  # DatetimeIndex vectorised
        else:
            bar_mins = pd.DatetimeIndex(hist_idx).hour * 60 + pd.DatetimeIndex(hist_idx).minute

        mask = (bar_mins - c1_min).abs() <= 5
        tod_vols = hist["volume"].values[mask]
    except Exception:
        return 0.0, 0.0

    if len(tod_vols) < 3:
        return 0.0, 0.0

    import numpy as np
    median_vol = float(np.median(tod_vols))
    if median_vol <= 0:
        return 0.0, 0.0

    ratio = c1_vol / median_vol
    if ratio >= 2.0:
        return 2.0, round(ratio, 2)
    if ratio >= 1.5:
        return 1.0, round(ratio, 2)
    return 0.0, round(ratio, 2)


def _score_macd_slope(bars: pd.DataFrame) -> tuple[float, float]:
    """
    MACD histogram slope score (0–2).
    A narrowing (less negative) MACD histogram on C2→C3 indicates weakening
    selling pressure — the momentum for the downtrend is exhausting.
      2 = histogram rising (less negative) across last 3 bars
      1 = histogram rising across last 2 bars only
      0 = histogram still falling or insufficient data
    Returns (score, slope_value).
    """
    if bars.empty or len(bars) < 35:
        return 0.0, 0.0

    from .pattern_engine import _ema

    closes = bars["close"]
    ema12  = _ema(closes, 12)
    ema26  = _ema(closes, 26)
    macd   = ema12 - ema26
    signal = _ema(macd, 9)
    hist   = macd - signal

    if len(hist) < 3 or hist.isna().any():
        return 0.0, 0.0

    h0 = float(hist.iloc[-3])
    h1 = float(hist.iloc[-2])
    h2 = float(hist.iloc[-1])
    slope_3 = h2 - h0   # overall 3-bar slope
    slope_2 = h2 - h1   # last 1-bar change

    if slope_3 > 0:
        return 2.0, round(slope_3, 6)
    if slope_2 > 0:
        return 1.0, round(slope_2, 6)
    return 0.0, round(slope_3, 6)


def _score_rsi_divergence(bars: pd.DataFrame) -> tuple[float, bool]:
    """
    RSI divergence score (0–2).
    Bearish price action (lower low on C3) with RSI printing a higher low
    = momentum weakening = absorption signal.
      2 = confirmed divergence (price lower low, RSI higher low, lookback ≤ 20 bars)
      1 = partial divergence (RSI flat while price makes lower low)
      0 = no divergence
    Returns (score, divergence_found).
    """
    if bars.empty or len(bars) < 20:
        return 0.0, False

    closes  = bars["close"]
    n       = 14
    delta   = closes.diff()
    gain    = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss    = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs      = gain / loss.replace(0, float("nan"))
    rsi     = 100 - 100 / (1 + rs)

    # Look for prior swing low in last 20 bars (excluding last 3)
    recent_close = closes.iloc[-3:]
    recent_rsi   = rsi.iloc[-3:]
    prior_close  = closes.iloc[-20:-3]
    prior_rsi    = rsi.iloc[-20:-3]

    if prior_close.empty or recent_close.empty:
        return 0.0, False

    curr_low_price = float(recent_close.min())
    curr_low_rsi   = float(recent_rsi.iloc[recent_close.values.argmin()])
    prior_low_price = float(prior_close.min())
    prior_low_rsi   = float(prior_rsi.iloc[prior_close.values.argmin()])

    price_lower_low = curr_low_price < prior_low_price
    rsi_higher_low  = curr_low_rsi   > prior_low_rsi + 1.0   # at least 1 RSI point higher

    if price_lower_low and rsi_higher_low:
        return 2.0, True
    if price_lower_low and abs(curr_low_rsi - prior_low_rsi) < 1.0:
        return 1.0, False   # flat RSI on lower low = partial
    return 0.0, False


def _score_regime(bars: pd.DataFrame) -> tuple[float, str]:
    """
    Higher-timeframe regime score (0–2) as a soft gate.
    Computes a 30-min EMA trend using 6-bar (30-min) resampled closes.
      2 = 30-min EMA slope clearly down (aligned with bearish S×S thesis)
      1 = 30-min EMA flat (neutral)
      0 = 30-min EMA up (counter-trend — reduces conviction)
    Returns (score, description).
    """
    if bars.empty or len(bars) < 40:
        return 1.0, "insufficient data (neutral)"   # neutral default

    try:
        # Resample 5-min bars to 30-min
        bars_30 = bars["close"].resample("30min").last().dropna()
        if len(bars_30) < 10:
            return 1.0, "insufficient 30min bars"

        from .pattern_engine import _ema as _ema_fn
        ema_30 = _ema_fn(bars_30, 8)
        if len(ema_30) < 2:
            return 1.0, "ema insufficient"

        slope = float(ema_30.iloc[-1]) - float(ema_30.iloc[-4]) if len(ema_30) >= 4 else 0.0
        pct_slope = slope / float(bars_30.iloc[-1]) if float(bars_30.iloc[-1]) > 0 else 0.0

        if pct_slope < -0.005:    # 30-min trend clearly down
            return 2.0, f"30min EMA trending down ({pct_slope:.3%})"
        if pct_slope < 0.005:     # flat
            return 1.0, f"30min EMA flat ({pct_slope:.3%})"
        return 0.0, f"30min EMA up — counter-trend ({pct_slope:.3%})"
    except Exception:
        return 1.0, "regime calc error (neutral)"


# ---------------------------------------------------------------------------
# Support-OK indicator (aligns card with simulator filter)
# ---------------------------------------------------------------------------

def _compute_support_ok(bars: pd.DataFrame, support: Optional[float]) -> bool:
    """
    Return True if recent bar closes consistently hold above support * 0.99.
    Uses the last 12 bars (60 min) — same window as simulator _SUPPORT_CHECK_BARS.
    A ticker where >50% of recent closes fall below support * 0.99 is flagged
    support_ok=False on the card, signalling the level is not holding.

    This mirrors the simulator's support_respected check (close < support * 0.99).
    """
    if not support or support <= 0 or bars.empty:
        return True  # no data → assume OK (don't penalise)

    recent = bars.iloc[-12:]  # last 12 × 5-min bars = 60 min
    if recent.empty:
        return True

    threshold = support * 0.99
    n_below = (recent["close"] < threshold).sum()
    return n_below <= len(recent) * 0.5   # OK if ≤50% of closes are below


# ---------------------------------------------------------------------------
# EQH (Equal Highs / Liquidity Ceiling) scorer
# ---------------------------------------------------------------------------

def _score_eqh(
    bars: pd.DataFrame,
    current_price: float,
    eqh_lookback: int = 20,
    approach_pct: float = 0.005,   # within 0.5% of ceiling = "approaching"
    nearby_pct:   float = 0.02,    # within 2% = "nearby"
) -> tuple[float, float, str]:
    """
    EQH score (0–2).

    Scans the last *eqh_lookback* bars for Equal Highs pairs using
    detect_equal_highs_pair(), then checks where the current price sits
    relative to each detected ceiling.

      2 = EQH pair found AND current price is within *approach_pct* of ceiling
          OR an eqh_breakout/eqh_rejection signal fired in the lookback window
      1 = EQH pair found within *nearby_pct* of current price (approaching)
      0 = No EQH pair found within the lookback window

    Returns (score, eqh_level, eqh_signal).
      eqh_level  : ceiling price of the most relevant EQH pair (0.0 if none)
      eqh_signal : "" | "approaching" | "eqh_breakout" | "eqh_rejection"
    """
    from .pattern_engine import detect_equal_highs_pair, detect_eqh_signal

    if bars.empty or current_price <= 0:
        return 0.0, 0.0, ""

    recent = bars.iloc[-eqh_lookback:] if len(bars) >= eqh_lookback else bars
    eqh_pairs = detect_equal_highs_pair(recent)
    if not eqh_pairs:
        return 0.0, 0.0, ""

    # Check for confirmed breakout/rejection signals first
    signals = detect_eqh_signal(recent, eqh_pairs)
    for sig in reversed(signals):  # most recent first
        if sig.pattern_type in ("eqh_breakout", "eqh_rejection"):
            return 2.0, sig.eqh_level, sig.pattern_type

    # No confirmed signal — pick the EQH pair closest to current price
    best_pair = min(eqh_pairs, key=lambda p: abs(p.eqh_level - current_price))
    eqh_ceiling = best_pair.eqh_level
    if eqh_ceiling <= 0:
        return 0.0, 0.0, ""

    dist_pct = abs(eqh_ceiling - current_price) / current_price

    if dist_pct <= approach_pct:
        return 2.0, eqh_ceiling, "approaching"
    if dist_pct <= nearby_pct:
        return 1.0, eqh_ceiling, "approaching"
    return 0.0, eqh_ceiling, ""


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def score_setup(
    entry,                              # WatchlistEntry
    bars: pd.DataFrame,
    db_path: Optional[str] = None,
    *,
    disable_sr_cache: bool = False,
    cutoff_date=None,
) -> "SetupScore":
    """
    Compute a composite SetupScore for *entry* given its OHLCV bars.

    Parameters
    ----------
    entry            : WatchlistEntry — parsed ticker + support/resistance/stop levels.
    bars             : pd.DataFrame  — 5-min OHLCV bars (UTC DatetimeIndex).
    db_path          : path to SQLite trade DB; defaults to the package default.
    disable_sr_cache : bool (default False)
        When True, bypass the per-ticker S/R cache so that S/R levels are
        recomputed from the bars slice rather than returned from a stale cache
        entry that was computed at a different price.  Used by the card
        simulator so each historical date gets its own fresh levels (Bug #2 fix).
    cutoff_date      : Optional[date]
        When provided, the history_score component only counts trades with
        entry_ts < cutoff_date.  Prevents future trades from leaking into the
        score during historical replay (Bug #1 fix).
        Pass None (default) for live scoring — no change to existing behaviour.
    """
    # ── Smart S/R resolution ──────────────────────────────────────────────────
    # 1. Use watchlist levels if present AND within 30% of current price.
    # 2. Otherwise fall back to computed S/R from sr_engine.
    # This prevents stale/far-away watchlist levels from producing crazy R/R.

    current_price = float(bars["close"].iloc[-1]) if not bars.empty else 0.0

    # Calculate ATR for position sizing (Phase 6)
    from .pattern_engine import _atr
    atr_series = _atr(bars, period=14)
    current_atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0

    def _is_stale(level: Optional[float], price: float, threshold: float = 0.30) -> bool:
        """Return True if level is absent or >threshold% away from current price."""
        if not level or level <= 0 or price <= 0:
            return True
        return abs(level - price) / price > threshold

    support    = entry.support_level
    resistance = entry.resistance_level
    stop       = entry.stop_level

    # Relevance gate: 15% band around current price — aligned with simulator stale-support
    # exclusion threshold.  Watchlist levels >15% away are treated as stale and replaced
    # with computed S/R.  Tightened from 20% to match the simulator's 15% exclusion so
    # the card and the backtest agree on which support levels are usable.
    # CRITICAL: do NOT fall back to the stale watchlist value when computed S/R finds
    # nothing — that would reinstate an unreachable swing target (e.g. BTOG $8.30 when
    # price is $2.50).  Set resistance=None instead so _score_rr returns 0 cleanly.
    _RELEVANCE_GATE = 0.15   # 15% — matches simulator stale-support exclusion
    _resistance_stale = _is_stale(resistance, current_price, threshold=_RELEVANCE_GATE)
    _support_stale    = _is_stale(support,    current_price, threshold=_RELEVANCE_GATE)
    sr = None

    if not bars.empty and (_support_stale or _resistance_stale):
        now_t2 = _time.monotonic()
        _fb_key = f"_fallback_{entry.ticker}"
        # Bug #2 fix: when disable_sr_cache=True (backtest context), always
        # recompute S/R from the current bars slice rather than reading back
        # a cached result that was anchored to a different (more recent) price.
        cached2 = None
        if not disable_sr_cache:
            with _sr_cache_lock:
                cached2 = _sr_cache.get(_fb_key)
        if cached2 and (now_t2 - cached2[0]) < _SR_CACHE_TTL:
            sr = cached2[1]
        else:
            from .sr_engine import compute_sr_levels
            sr = compute_sr_levels(bars.iloc[-_SCORE_BARS:] if len(bars) > _SCORE_BARS else bars,
                                   current_price=current_price, price_range_pct=_RELEVANCE_GATE,
                                   use_kmeans=False)
            if not disable_sr_cache:
                with _sr_cache_lock:
                    _sr_cache[_fb_key] = (now_t2, sr)

        if _support_stale:
            support = sr.nearest_support(current_price) or support

        if _resistance_stale:
            # Use computed nearest resistance above current price.
            # If none found within the narrow band, widen to 40% so the card always
            # shows a meaningful TP rather than "—".
            computed_res = sr.nearest_resistance(current_price)
            if computed_res and computed_res > current_price:
                resistance = computed_res
            else:
                # Retry with a wider band (40%) — we'd rather show a real level
                # than leave TP blank on the card.
                from .sr_engine import compute_sr_levels as _csl_wide
                sr_wide = _csl_wide(
                    bars.iloc[-_SCORE_BARS:] if len(bars) > _SCORE_BARS else bars,
                    current_price=current_price, price_range_pct=0.40,
                    use_kmeans=False,
                )
                wide_res = sr_wide.nearest_resistance(current_price)
                resistance = wide_res if (wide_res and wide_res > current_price) else None

    # If resistance is still None after all fallbacks, make one last attempt with a
    # very wide band (60%) — always better to show a computed TP than "—".
    # BUG FIX: use a distinct cache key (_fallback_wide_) so this path never reads
    # back the narrow 15%-band SR object that was cached under _fallback_{ticker}.
    def _wide_resistance(price: float) -> Optional[float]:
        """Always compute fresh 60%-band SR and return nearest resistance above price."""
        from .sr_engine import compute_sr_levels as _csl_w
        sr_w = _csl_w(
            bars.iloc[-_SCORE_BARS:] if len(bars) > _SCORE_BARS else bars,
            current_price=price, price_range_pct=0.60,
            use_kmeans=False,
        )
        res = sr_w.nearest_resistance(price)
        return res if (res and res > price) else None

    if resistance is None and not bars.empty and current_price > 0:
        resistance = _wide_resistance(current_price)

    # If current price is already below support, the support level was broken.
    # Re-anchor: entry = current_price, find nearest computed S/R around current price.
    support_broken = (support and support > 0 and current_price > 0
                      and current_price < support * 0.99)
    if support_broken and not bars.empty:
        # Always re-fetch computed S/R anchored to current_price so that
        # both support and resistance are on the correct side of the entry.
        # Bug #2 fix: bypass cache when disable_sr_cache=True (backtest context).
        cached3 = None
        if not disable_sr_cache:
            with _sr_cache_lock:
                cached3 = _sr_cache.get(f"_fallback_{entry.ticker}")
        sr3 = cached3[1] if cached3 else None
        if sr3 is None:
            # Cache miss — compute now
            from .sr_engine import compute_sr_levels as _csl
            sr3 = _csl(
                bars.iloc[-_SCORE_BARS:] if len(bars) > _SCORE_BARS else bars,
                current_price=current_price, price_range_pct=_RELEVANCE_GATE,
                use_kmeans=False,
            )
            if not disable_sr_cache:
                with _sr_cache_lock:
                    _sr_cache[f"_fallback_{entry.ticker}"] = (_time.monotonic(), sr3)

        new_sup = sr3.nearest_support(current_price)
        new_res = sr3.nearest_resistance(current_price)
        # Only accept levels that make geometric sense:
        #   support must be below current_price
        #   resistance must be above current_price
        if new_sup and new_sup < current_price:
            support = new_sup
        else:
            support = current_price * 0.97   # 3% synthetic floor

        if new_res and new_res > current_price:
            resistance = new_res
        else:
            # BUG FIX: the wide-band retry must happen AFTER support_broken re-anchor,
            # not before it.  Previous code set resistance=None here with a comment
            # "will be filled by wide-band retry below" — but there was no retry below.
            resistance = _wide_resistance(current_price)

    # Entry is always the current price when trading; never use a stale watchlist
    # support level as the entry (that's where you'd set a limit order, not where
    # you enter now if price has already moved).
    # • support_broken  → price already below watchlist level → enter at current price
    # • normal case     → watchlist support IS the entry target (limit order zone)
    entry_price = current_price if support_broken else (support or current_price or 0.0)

    # SL fallback: if watchlist didn't provide an explicit stop, default to 2% below
    # entry (per README Step 5 — "Stop-Loss 2% below support").
    if not stop or stop <= 0:
        stop = entry_price * 0.98

    # Slice to last _SCORE_BARS for all component scorers (reduces compute 7×)
    sb = bars.iloc[-_SCORE_BARS:] if len(bars) > _SCORE_BARS else bars

    # ── Sequential component scoring ─────────────────────────────────────────
    # Components run sequentially within a single ticker.  The outer per-ticker
    # loop in morning_brief/_score_entries_raw() already parallelises across
    # tickers via ThreadPoolExecutor — nesting another pool here causes
    # thread-count explosion (8 outer × 11 inner = 88 threads) that thrashes
    # the GIL and makes overall throughput *worse* for CPU-bound pandas work.
    pat_s,   pat_found, pat_type    = _score_pattern(sb, support)
    adx_s,   adx_val                = _score_adx(sb)
    rr_s,    rr_ratio, stop_price   = _score_rr(support, resistance, stop, entry_price)
    # Bug #2 fix: pass disable_cache=True in backtest context so confluence
    # S/R levels are computed from this historical slice, not today's cache.
    conf_s,  conf_desc              = _score_confluence(
        sb, support,
        ticker=("" if disable_sr_cache else entry.ticker),
    )
    # Bug #1 fix: pass cutoff_date so history_score only uses trades that
    # existed at this point in time during historical replay.
    hist_s, win_rate, n_trades, avg_win, avg_loss, hist_expectancy, hist_streak = _score_history(
        entry.ticker, db_path, cutoff_date=cutoff_date,
    )
    rr_rev_s, is_role_rev           = _score_role_reversal(sb, support)
    rej_s,   rej_count              = _score_rejections(sb, support)
    rv_s,    rv_ratio               = _score_relative_volume(sb)
    macd_s,  macd_slope             = _score_macd_slope(sb)
    rsi_s,   rsi_div                = _score_rsi_divergence(sb)
    regime_s, regime_desc           = _score_regime(sb)
    eqh_s,   eqh_lvl, eqh_sig      = _score_eqh(sb, current_price)

    # Max score is now 24 (12 components × 2); normalise to 0–10
    # New Formula: Base * PA_Mult * Conviction
    Base = adx_s + macd_s + regime_s
    PA_Mult = 1.0 + (pat_s * 0.1) + (conf_s * 0.1) + (rr_rev_s * 0.1) + (rej_s * 0.05) + (eqh_s * 0.1) + (rsi_s * 0.05)
    Conviction = 1.0 + (rv_s * 0.1) + (hist_s * 0.1)
    
    raw_total = Base * PA_Mult * Conviction
    
    if eqh_s > 0 and conf_s > 0 and rr_rev_s > 0:
        raw_total *= 1.5
        
    total = round(raw_total / 16.8 * 10.0, 2)

    sig = "🟢 STRONG" if total >= 7.0 else ("🟡 WATCH" if total >= 4.0 else "🔴 SKIP")

    return SetupScore(
        ticker=entry.ticker,
        score=total,
        signal=sig,
        pattern_score=pat_s,
        adx_score=adx_s,
        rr_score=rr_s,
        confluence_score=conf_s,
        history_score=hist_s,
        history_avg_win=avg_win,
        history_avg_loss=avg_loss,
        history_expectancy=hist_expectancy,
        history_streak=hist_streak,
        role_reversal_score=rr_rev_s,
        rejection_score=rej_s,
        rel_vol_score=rv_s,
        macd_score=macd_s,
        rsi_div_score=rsi_s,
        regime_score=regime_s,
        eqh_score=eqh_s,
        sr_levels=sr,
        entry_price=round(entry_price, 4),
        support=round(support, 4) if support else 0.0,
        atr=current_atr,
        resistance=round(resistance, 4) if resistance else None,
        stop=round(stop_price, 4),
        rr_ratio=rr_ratio,
        adx=round(adx_val, 1),
        pattern_found=pat_found,
        pattern_type=pat_type,
        role_reversal=is_role_rev,
        rejection_count=rej_count,
        support_broken=support_broken,
        support_ok=_compute_support_ok(sb, support),
        watchlist_note=entry.raw_text or entry.sentiment_notes,
        eqh_level=round(eqh_lvl, 4) if eqh_lvl else 0.0,
        eqh_signal=eqh_sig,
    )
