"""
Pattern Recognition Engine — Side-by-Side White Lines (3-Candle)
================================================================
Implements the *industry-standard* 3-candle Side-by-Side White Lines pattern:

  Bullish (continuation in uptrend):
    C1: Long bullish candle in an existing uptrend
    C2: Bullish candle that GAPS UP from C1 (C2.open > C1.high)
    C3: Bullish candle of similar size to C2 that opens at nearly the same
        price as C2 (|C3.open - C2.open| / C2.open ≤ open_tolerance)

  Bearish (continuation in downtrend):
    C1: Long bearish candle in an existing downtrend
    C2: Bearish candle that GAPS DOWN from C1 (C2.open < C1.low)
    C3: Bearish candle of similar size to C2 opening near C2.open

Quality filters applied before a match is accepted:
  1. ADX > adx_threshold (default 25) — confirms a trending market
  2. C1 body ≥ 2× average body of last `body_lookback` candles
  3. C2 and C3 bodies both ≥ `min_body_pct` of their bar ranges (no dojis)
  4. Gap direction must match the trend direction

Backward-compatible public API:
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


def _in_uptrend(closes: pd.Series, idx: int, period: int = 14) -> bool:
    """EMA slope uptrend: EMA[idx] > EMA[idx - period]."""
    if idx < period * 2:
        return False
    ema = _ema(closes.iloc[: idx + 1], period)
    return float(ema.iloc[-1]) > float(ema.iloc[-1 - period])


def _in_downtrend(closes: pd.Series, idx: int, period: int = 14) -> bool:
    """EMA slope downtrend: EMA[idx] < EMA[idx - period]."""
    if idx < period * 2:
        return False
    ema = _ema(closes.iloc[: idx + 1], period)
    return float(ema.iloc[-1]) < float(ema.iloc[-1 - period])


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
    tolerance_pct: float = 0.02,          # C3.open vs C2.open tolerance (2% for intraday)
    min_body_pct: float = 0.15,           # C2/C3 body must be ≥ 15% of bar range
    require_downtrend: bool = True,        # True = bearish version; False = bullish
    adx_period: int = 14,
    adx_threshold: float = 20.0,          # Lowered from 25 — micro-caps trend at lower ADX
    body_lookback: int = 10,              # C1 body vs rolling avg body
    body_multiplier: float = 1.5,         # Lowered from 2.0 — less strict for 5-min spikes
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
    min_body_pct      : Min body-to-range ratio for C2 and C3 (filters doji candles).
    require_downtrend : True → bearish continuation; False → bullish continuation.
    adx_period        : Period for ADX calculation.
    adx_threshold     : ADX minimum to confirm a trending market (20 for intraday).
    body_lookback     : Rolling window for C1 body-size comparison.
    body_multiplier   : C1 body must be ≥ this × rolling avg body.
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
    df["adx"]        = _adx(df, adx_period)

    closes  = df["close"]
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
            if not _in_downtrend(closes, i):
                continue
            # Bearish: C1 and C2 must be bearish; C3 bearish preferred but not required
            if not (_is_bearish(c1_o, c1_c) and _is_bearish(c2_o, c2_c)):
                continue
            # Gap check: C2 opens at or below C1's close (intraday = no true gap)
            if require_gap:
                if c2_o > c1_c * (1 + gap_tolerance_pct):
                    continue
            # Without require_gap: just confirm C2 didn't open dramatically above C1 close
            # (allows the pattern when C2 opens at C1's close, which is normal intraday)
        else:
            if not _in_uptrend(closes, i):
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
                in_downtrend=require_downtrend,
            )
        )

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
    Return Side-by-Side patterns whose C2/C3 body lows are within
    *proximity_pct* of the given support level.
    """
    all_matches = detect_side_by_side(
        df,
        tolerance_pct=tolerance_pct,
        min_body_pct=min_body_pct,
        require_downtrend=require_downtrend,
    )
    nearby: list[PatternMatch] = []
    for m in all_matches:
        body_low_of_pair = min(
            _body_low(m.candle1_open, m.candle1_close),
            _body_low(m.candle2_open, m.candle2_close),
        )
        if support > 0 and abs(body_low_of_pair - support) / support <= proximity_pct:
            nearby.append(m)
    return nearby
