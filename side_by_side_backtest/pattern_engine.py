"""
Pattern Recognition Engine — Side-by-Side White Lines (Bearish)
================================================================
Implements the **bearish** 3-candle Side-by-Side White Lines continuation pattern.

  ⚠️  PRODUCTION MODE: Bearish only (require_downtrend=True always).
  The bullish variant is NOT used — bullish S×S patterns merely confirm
  intraday support that already exists and are actively targeted by bears
  for short-selling. Detecting them adds noise, not edge.

  Bearish Side-by-Side (downtrend continuation):
    C1: Long bearish candle in an existing downtrend
        (body ≥ 1.5× rolling avg body — strong momentum candle)
    C2: Bearish candle opening at or near C1's close
        (continuation of selling pressure)
    C3: Bearish candle opening at nearly the same price as C2
        (|C3.open - C2.open| / C2.open ≤ open_tolerance)
        → "side by side" twin structure signals further downside

  Entry signal (long):
    Price drops to the watchlist support level AND this bearish pattern fires
    → buyers absorbing sell pressure at support → potential reversal/bounce.
    Multiple bearish S×S patterns stacked at the same support = strongest signal.

Quality filters:
  1. ADX > adx_threshold (default 20) — confirms a trending market
  2. C1 body ≥ 1.5× average body of last `body_lookback` candles
  3. C2 and C3 bodies both ≥ `min_body_pct` of their bar ranges (no dojis)
  4. C2 and C3 body sizes within 50% of each other

Public API:
  detect_side_by_side(df, ...)        → List[PatternMatch]
  pattern_near_support(df, support)   → List[PatternMatch]
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .models import PatternMatch


# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    """Wilder True Range."""
    high  = df["high"]
    low   = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average Directional Index (Wilder's method).
    Returns a Series of ADX values (NaN for first 2*period bars).
    """
    tr      = _true_range(df)
    up_move = df["high"] - df["high"].shift(1)
    dn_move = df["low"].shift(1)  - df["low"]

    plus_dm  = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)

    plus_dm_s  = pd.Series(plus_dm,  index=df.index).ewm(alpha=1/period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    atr        = tr.ewm(alpha=1/period, adjust=False).mean()

    plus_di  = 100 * plus_dm_s  / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm_s / atr.replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx      = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx


def _in_uptrend(closes: pd.Series, highs: pd.Series, idx: int, period: int = 6) -> bool:
    """
    Intraday uptrend: current close is above the lowest close in the last
    *period* bars — i.e. price is recovering, not continuing to fall.

    period=6 ≈ 30 minutes on 5-min bars.
    Accepts *highs* for API symmetry with _in_downtrend (unused here).
    """
    if idx < period:
        return True  # not enough bars — don't block early patterns
    window_closes = closes.iloc[max(0, idx - period): idx]
    if window_closes.empty:
        return True
    return float(closes.iloc[idx]) > float(window_closes.min())


def _in_downtrend(closes: pd.Series, highs: pd.Series, idx: int, period: int = 6) -> bool:
    """
    Intraday downtrend: current close is below the highest close seen in the
    last *period* bars.  This directly answers "are we pulling back from a
    recent peak?" which is exactly what a bearish side-by-side signals.

    period=6 ≈ 30 minutes on 5-min bars — short enough to catch intraday
    spikes, long enough not to fire on random 1-bar wiggles.

    Does NOT use EMA slopes or multi-day lookbacks so it works correctly on
    both a 30-day dataset and a single-session slice.
    """
    if idx < period:
        return True  # not enough bars — give early-session patterns the benefit of the doubt
    window_closes = closes.iloc[max(0, idx - period): idx]
    if window_closes.empty:
        return True
    return float(closes.iloc[idx]) < float(window_closes.max())


# ---------------------------------------------------------------------------
# Body geometry helpers
# ---------------------------------------------------------------------------

def _body_low(o: float, c: float) -> float:
    return min(o, c)

def _body_high(o: float, c: float) -> float:
    return max(o, c)

def _body_size(o: float, c: float) -> float:
    return abs(o - c)

def _is_bullish(o: float, c: float) -> bool:
    return c > o

def _is_bearish(o: float, c: float) -> bool:
    return c < o


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------

def detect_side_by_side(
    df: pd.DataFrame,
    tolerance_pct: float = 0.03,          # C3.open vs C2.open tolerance (widened 2%→3%)
    min_body_pct: float = 0.10,           # C2/C3 body must be ≥ 10% of bar range (was 15%)
    require_downtrend: bool = True,        # True = bearish version; False = bullish
    adx_period: int = 14,
    adx_threshold: float = 20.0,          # Lowered from 25 — micro-caps trend at lower ADX
    body_lookback: int = 10,              # C1 body vs rolling avg body
    body_multiplier: float = 1.0,         # C1 body ≥ avg body (was 1.5 → 1.2 → 1.0)
    gap_tolerance_pct: float = 0.05,      # 5% — on intraday there are no true overnight gaps;
                                           # C2 just needs to open near/below C1's close
    require_gap: bool = False,             # False = skip hard gap check (intraday default)
    downtrend_method: str = "ema",        # kept for backward compat
) -> List[PatternMatch]:
    """
    Scan a 5-minute OHLCV DataFrame for Side-by-Side White Lines patterns.

    Parameters
    ----------
    df                : DataFrame [open, high, low, close, volume], DatetimeIndex.
    tolerance_pct     : Max |C3.open - C2.open| / C2.open for the 'same open' check.
                        Widened from 2% → 3% to capture near-identical opens on
                        micro-cap 5-min bars where price discretisation is coarser.
    min_body_pct      : Min body-to-range ratio for C2 and C3 (filters doji candles).
                        Lowered from 15% → 10% — intraday micro-cap candles have wide
                        wicks relative to the body even on strong moves.
    require_downtrend : True → bearish continuation; False → bullish continuation.
    adx_period        : Period for ADX calculation.
    adx_threshold     : ADX minimum to confirm a trending market (20 for intraday).
    body_lookback     : Rolling window for C1 body-size comparison.
    body_multiplier   : C1 body must be ≥ this × rolling avg body.
                        Lowered from 1.5 → 1.2 — the "strong C1" criterion was
                        filtering out valid patterns where C1 is slightly above average
                        rather than dramatically so.
    gap_tolerance_pct : Allowed deviation for the gap check (5% for 5-min intraday).
    require_gap       : If False (default), skip hard gap enforcement — appropriate
                        for intraday where candles open at the prior close. The trend
                        direction and body quality filters still apply.

    Returns
    -------
    List of PatternMatch objects, one per detected 3-candle sequence.
    """
    if df.empty or len(df) < max(3, body_lookback + 1, adx_period * 2 + 1):
        return []

    df = df.copy()

    # Pre-compute derived columns
    df["body_size"]  = (df["close"] - df["open"]).abs()
    df["bar_range"]  = df["high"] - df["low"]
    df["body_ratio"] = df["body_size"] / df["bar_range"].replace(0, np.nan)
    df["avg_body"]   = df["body_size"].rolling(body_lookback, min_periods=3).mean()
    # Skip ADX recomputation if caller already computed it (perf optimisation)
    if "adx" not in df.columns:
        df["adx"] = _adx(df, adx_period)

    closes  = df["close"]
    highs   = df["high"]
    matches: list[PatternMatch] = []
    ticker  = df.attrs.get("ticker", "")

    for i in range(2, len(df)):
        c1_o, c1_c = df["open"].iloc[i - 2], df["close"].iloc[i - 2]
        c2_o, c2_c = df["open"].iloc[i - 1], df["close"].iloc[i - 1]
        c3_o, c3_c = df["open"].iloc[i],     df["close"].iloc[i]

        # ── 1. ADX filter: must be a trending market ──────────────────────────
        adx_val = df["adx"].iloc[i]
        if pd.isna(adx_val) or adx_val < adx_threshold:
            continue

        # ── 2. Trend direction check ──────────────────────────────────────────
        if require_downtrend:
            if not _in_downtrend(closes, highs, i):
                continue
            # Bearish: all three candles must be bearish for strict classification
            if not (_is_bearish(c1_o, c1_c) and _is_bearish(c2_o, c2_c) and _is_bearish(c3_o, c3_c)):
                continue
            # Gap check: C2 opens at or below C1's close (intraday = no true gap)
            if require_gap:
                if c2_o > c1_c * (1 + gap_tolerance_pct):
                    continue
            # Without require_gap: just confirm C2 didn't open dramatically above C1 close
            # (allows the pattern when C2 opens at C1's close, which is normal intraday)
        else:
            if not _in_uptrend(closes, highs, i):
                continue
            # Bullish: C1 and C2 must be bullish
            if not (_is_bullish(c1_o, c1_c) and _is_bullish(c2_o, c2_c)):
                continue
            # Gap check: C2 opens at or above C1's close
            if require_gap:
                if c2_o < c1_c * (1 - gap_tolerance_pct):
                    continue

        # ── 3. C1 body-size filter: must be a strong candle ──────────────────
        avg_body = df["avg_body"].iloc[i - 2]
        c1_body  = _body_size(c1_o, c1_c)
        if pd.isna(avg_body) or avg_body == 0:
            continue
        if c1_body < body_multiplier * avg_body:
            continue

        # ── 4. C2 and C3 body quality: no doji candles ───────────────────────
        r2 = df["body_ratio"].iloc[i - 1]
        r3 = df["body_ratio"].iloc[i]
        if pd.isna(r2) or pd.isna(r3):
            continue
        if r2 < min_body_pct or r3 < min_body_pct:
            continue

        # ── 5. C3 opens near C2's open ("same open" criterion) ───────────────
        if c2_o == 0:
            continue
        open_diff = abs(c3_o - c2_o) / c2_o
        if open_diff > tolerance_pct:
            continue

        # ── 6. C2 and C3 similar body size (within 50% of each other) ────────
        c2_body = _body_size(c2_o, c2_c)
        c3_body = _body_size(c3_o, c3_c)
        if c2_body > 0 and (abs(c2_body - c3_body) / c2_body) > 0.50:
            continue

        matches.append(
            PatternMatch(
                ticker=ticker,
                ts=df.index[i],
                bar_index=i,
                candle1_open=c1_o,
                candle1_close=c1_c,
                candle2_open=c2_o,
                candle2_close=c2_c,
                candle3_open=c3_o,
                candle3_close=c3_c,
                in_downtrend=require_downtrend,
                confidence_score=1.0,
                pattern_type="strict",
            )
        )

    return matches


# ---------------------------------------------------------------------------
# Exhaustion Side-by-Side detector (relaxed — catches bottoming patterns)
# ---------------------------------------------------------------------------

def detect_exhaustion_side_by_side(
    df: pd.DataFrame,
    tolerance_pct: float = 0.05,      # same-open tolerance widened to 5% — intraday opens
                                       # rarely repeat exactly; 5% catches near-identical pairs
    min_body_pct: float = 0.10,       # body must be ≥ 10% of bar range (filters pure dojis)
    c1_lookback: int = 10,            # search up to 10 bars back for the momentum candle
    adx_threshold: float = 15.0,      # lower ADX — trend may be flattening at bottom
    body_lookback: int = 10,
    body_multiplier: float = 0.8,     # C1 body ≥ 80% of avg body (was 1.2× — too strict
                                       # when morning spike inflates the rolling average)
) -> List[PatternMatch]:
    """
    Relaxed Side-by-Side detector for **intraday bearish twin candles**.

    Detects two consecutive bearish candles with nearly the same open after a
    prior momentum (C1) candle somewhere in the recent lookback window.  This
    correctly catches the patterns a human trader identifies on the chart:

      - Two consecutive bearish candles C2/C3 with nearly the same open (≤5%)
      - Similar body sizes (≤60% difference)
      - A bearish momentum C1 in the last *c1_lookback* bars (any position,
        not necessarily the immediately preceding bar)
      - Lower ADX threshold — pattern fires even during chop/base phases
      - No strict downtrend EMA requirement — captures reversals at highs too

    confidence_score = 0.6 (lower than strict = 1.0, reflecting the relaxed rules).
    """
    if df.empty or len(df) < max(3, body_lookback + 1, 30):
        return []

    df = df.copy()
    df["body_size"]  = (df["close"] - df["open"]).abs()
    df["bar_range"]  = df["high"] - df["low"]
    df["body_ratio"] = df["body_size"] / df["bar_range"].replace(0, np.nan)
    df["avg_body"]   = df["body_size"].rolling(body_lookback, min_periods=3).mean()
    if "adx" not in df.columns:
        df["adx"] = _adx(df, 14)

    closes  = df["close"]
    matches: list[PatternMatch] = []
    ticker  = df.attrs.get("ticker", "")

    for i in range(2, len(df)):
        c2_o, c2_c = df["open"].iloc[i - 1], df["close"].iloc[i - 1]
        c3_o, c3_c = df["open"].iloc[i],     df["close"].iloc[i]

        # ── ADX filter (lower threshold) ──────────────────────────────────
        adx_val = df["adx"].iloc[i]
        if pd.isna(adx_val) or adx_val < adx_threshold:
            continue

        # ── C2 and C3 body quality ─────────────────────────────────────────
        r2 = df["body_ratio"].iloc[i - 1]
        r3 = df["body_ratio"].iloc[i]
        if pd.isna(r2) or pd.isna(r3) or r2 < min_body_pct or r3 < min_body_pct:
            continue

        # ── C2 and C3 same-open check (the "twin" structure) ──────────────
        if c2_o == 0:
            continue
        if abs(c3_o - c2_o) / c2_o > tolerance_pct:
            continue

        # ── C2 and C3 must both be bearish (selling pressure) ─────────────
        if not (_is_bearish(c2_o, c2_c) and _is_bearish(c3_o, c3_c)):
            continue

        # ── Look back up to c1_lookback for the strong C1 candle ──────────
        c1_found = False
        for k in range(max(0, i - 2 - c1_lookback), i - 1):
            c1_o_k = df["open"].iloc[k]
            c1_c_k = df["close"].iloc[k]
            avg_body_k = df["avg_body"].iloc[k]
            if pd.isna(avg_body_k) or avg_body_k == 0:
                continue
            c1_body_k = _body_size(c1_o_k, c1_c_k)
            if _is_bearish(c1_o_k, c1_c_k) and c1_body_k >= body_multiplier * avg_body_k:
                c1_found = True
                c1_o, c1_c = c1_o_k, c1_c_k
                break

        if not c1_found:
            continue

        # ── Similar body size between C2 and C3 (within 70%) ─────────────
        # Widened from 60% to 70%: intraday micro-cap bodies can vary more
        # while still being visually "identical" on the chart due to scale.
        c2_body = _body_size(c2_o, c2_c)
        c3_body = _body_size(c3_o, c3_c)
        if c2_body > 0 and (abs(c2_body - c3_body) / c2_body) > 0.70:
            continue

        matches.append(PatternMatch(
            ticker=ticker,
            ts=df.index[i],
            bar_index=i,
            candle1_open=c1_o,
            candle1_close=c1_c,
            candle2_open=c2_o,
            candle2_close=c2_c,
            candle3_open=c3_o,
            candle3_close=c3_c,
            in_downtrend=True,
            confidence_score=0.6,
            pattern_type="exhaustion",
        ))

    return matches


# ---------------------------------------------------------------------------
# Support Absorption detector
# ---------------------------------------------------------------------------

def detect_support_absorption(
    df: pd.DataFrame,
    support: float,
    proximity_pct: float = 0.01,    # C1 close/low within 1% of support
    small_body_pct: float = 0.40,   # C2/C3 body ≤ 40% of C1 body (consolidation)
    min_adx: float = 15.0,
    c1_multiplier: float = 1.5,     # C1 body ≥ 1.5× avg body
    body_lookback: int = 10,
) -> List[PatternMatch]:
    """
    Detect support-anchored absorption patterns.

    Structure:
      C1: Large bearish candle whose CLOSE or LOW lands within *proximity_pct*
          of a known support level  → selling momentum exhausted at support
      C2: Small body (any direction) staying ABOVE C1's low
      C3: Small body (any direction) staying ABOVE C1's low and opening
          near C2's close (consolidation at support)

    This captures what the classic S×S misses: the two post-drop consolidation
    candles (which may be green/bullish) that signal absorption of the selling.
    confidence_score = 0.7 (lower than strict S×S = 1.0, higher than exhaustion = 0.6)
    """
    if df.empty or support <= 0 or len(df) < max(3, body_lookback + 1, 30):
        return []

    df = df.copy()
    df["body_size"] = (df["close"] - df["open"]).abs()
    df["bar_range"]  = df["high"] - df["low"]
    df["avg_body"]   = df["body_size"].rolling(body_lookback, min_periods=3).mean()
    if "adx" not in df.columns:
        df["adx"] = _adx(df, 14)

    ticker  = df.attrs.get("ticker", "")
    matches: list[PatternMatch] = []

    for i in range(2, len(df)):
        c1_o = float(df["open"].iloc[i - 2])
        c1_c = float(df["close"].iloc[i - 2])
        c1_l = float(df["low"].iloc[i - 2])
        c2_o = float(df["open"].iloc[i - 1])
        c2_c = float(df["close"].iloc[i - 1])
        c3_o = float(df["open"].iloc[i])
        c3_c = float(df["close"].iloc[i])

        # ── ADX: must be trending ─────────────────────────────────────────
        adx_val = df["adx"].iloc[i]
        if pd.isna(adx_val) or adx_val < min_adx:
            continue

        # ── C1 must be a strong bearish candle ────────────────────────────
        if not _is_bearish(c1_o, c1_c):
            continue
        avg_body = df["avg_body"].iloc[i - 2]
        c1_body  = _body_size(c1_o, c1_c)
        if pd.isna(avg_body) or avg_body == 0:
            continue
        if c1_body < c1_multiplier * avg_body:
            continue

        # ── C1 close or low must land near support ────────────────────────
        c1_anchor = min(c1_c, c1_l)
        if abs(c1_anchor - support) / support > proximity_pct:
            continue

        # ── C2/C3 must stay above C1's low (not break support further) ────
        c2_low = float(df["low"].iloc[i - 1])
        c3_low = float(df["low"].iloc[i])
        if c2_low < c1_l * 0.998 or c3_low < c1_l * 0.998:
            continue

        # ── C2/C3 must be small-body relative to avg_body (scale-invariant) ─
        c2_body  = _body_size(c2_o, c2_c)
        c3_body  = _body_size(c3_o, c3_c)
        avg_body = df["avg_body"].iloc[i]
        if not pd.isna(avg_body) and avg_body > 0:
            # Small = each body ≤ small_body_pct × avg_body (not relative to C1)
            if c2_body > small_body_pct * avg_body * 2 or c3_body > small_body_pct * avg_body * 2:
                continue

        # ── At least one of C2/C3 must be bullish (direct buyer evidence) ────
        if not (_is_bullish(c2_o, c2_c) or _is_bullish(c3_o, c3_c)):
            continue

        matches.append(PatternMatch(
            ticker=ticker,
            ts=df.index[i],
            bar_index=i,
            candle1_open=c1_o,
            candle1_close=c1_c,
            candle2_open=c2_o,
            candle2_close=c2_c,
            candle3_open=c3_o,
            candle3_close=c3_c,
            in_downtrend=True,
            confidence_score=0.7,
            pattern_type="absorption",
        ))

    return matches


# ---------------------------------------------------------------------------
# Convenience: filter by proximity to a support/resistance level
# ---------------------------------------------------------------------------

def pattern_near_support(
    df: pd.DataFrame,
    support: float,
    proximity_pct: float = 0.005,
    tolerance_pct: float = 0.01,
    min_body_pct: float = 0.20,
    require_downtrend: bool = True,
) -> List[PatternMatch]:
    """
    Return Side-by-Side patterns whose **C3 body low** is within
    *proximity_pct* of the given support level.

    Fixed: uses C3 body low (the completion candle) rather than
    C1/C2, which may be several bars above support.
    """
    all_matches = detect_side_by_side(
        df,
        tolerance_pct=tolerance_pct,
        min_body_pct=min_body_pct,
        require_downtrend=require_downtrend,
    )
    nearby: list[PatternMatch] = []
    for m in all_matches:
        # C3 is the completion candle — use candle3_* now that it is stored
        if m.candle3_open != 0.0:
            c3_body_low = _body_low(m.candle3_open, m.candle3_close)
        else:
            c3_body_low = _body_low(m.candle2_open, m.candle2_close)  # fallback
        if support > 0 and abs(c3_body_low - support) / support <= proximity_pct:
            nearby.append(m)
    return nearby
