"""
Support & Resistance Engine
============================
Calculates multiple S/R levels from historical 5-minute OHLCV data using
four complementary methods:

  1. Pivot Points  — Floor Trader / Classic method (daily P, R1/R2, S1/S2)
  2. Local Extrema — Fractal highs/lows with rolling-window detection,
                     clustered into zones to remove noise
  3. Volume Profile — Point of Control (POC) and Value Area edges (VAH/VAL)
  4. K-Means       — ML clustering of all local highs/lows (optional,
                     requires sklearn; gracefully skipped if not installed)

Public API
----------
    from side_by_side_backtest.sr_engine import compute_sr_levels

    levels = compute_sr_levels(df)
    # Returns SRLevels dataclass with .supports and .resistances (sorted lists)
    # and .all_levels (combined, with metadata)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SRLevel:
    price: float
    method: str          # "pivot_r1" | "pivot_s1" | "local_high" | "local_low" | "poc" | "vah" | "val" | "kmeans"
    strength: float = 1.0   # higher = stronger (touch-count or volume weight)
    is_support: bool = True


@dataclass
class SRLevels:
    supports:    List[float] = field(default_factory=list)
    resistances: List[float] = field(default_factory=list)
    all_levels:  List[SRLevel] = field(default_factory=list)

    def nearest_support(self, price: float) -> Optional[float]:
        """Return the closest support level below `price`."""
        below = [s for s in self.supports if s < price]
        return max(below) if below else None

    def nearest_resistance(self, price: float) -> Optional[float]:
        """Return the closest resistance level above `price`."""
        above = [r for r in self.resistances if r > price]
        return min(above) if above else None


# ---------------------------------------------------------------------------
# 1. Pivot Points (Floor Trader / Classic daily)
# ---------------------------------------------------------------------------

def _pivot_points(df: pd.DataFrame) -> List[SRLevel]:
    """
    Compute classic pivot points from the most recent completed day's OHLCV.
    Uses the last calendar day's high, low, and close found in the DataFrame.
    """
    if df.empty:
        return []

    # Group by date and get last full day
    df_tz = df.copy()
    if df_tz.index.tzinfo is None:
        df_tz.index = df_tz.index.tz_localize("UTC")
    df_tz = df_tz.tz_convert("America/New_York")
    df_tz["date"] = df_tz.index.date

    days = sorted(df_tz["date"].unique())
    if len(days) < 2:
        return []

    prev_day = days[-2]
    prev = df_tz[df_tz["date"] == prev_day]

    H = float(prev["high"].max())
    L = float(prev["low"].min())
    C = float(prev["close"].iloc[-1])

    P  = (H + L + C) / 3
    R1 = 2 * P - L
    S1 = 2 * P - H
    R2 = P + (H - L)
    S2 = P - (H - L)

    levels = [
        SRLevel(price=round(P,  4), method="pivot_p",  is_support=False, strength=2.0),
        SRLevel(price=round(R1, 4), method="pivot_r1", is_support=False, strength=1.5),
        SRLevel(price=round(R2, 4), method="pivot_r2", is_support=False, strength=1.2),
        SRLevel(price=round(S1, 4), method="pivot_s1", is_support=True,  strength=1.5),
        SRLevel(price=round(S2, 4), method="pivot_s2", is_support=True,  strength=1.2),
    ]
    return [lv for lv in levels if lv.price > 0]


# ---------------------------------------------------------------------------
# 2. Local Extrema — Fractal detection + cluster merging + time-decay weight
# ---------------------------------------------------------------------------

def _local_extrema(
    df: pd.DataFrame,
    window: int = 5,
    cluster_pct: float = 0.003,
) -> List[SRLevel]:
    """
    Find fractal highs/lows with TIME-DECAY weighting:
      strength = sum(1 / max(days_ago, 0.5)) across cluster members.
    A level formed 2 days ago has strength ~0.5; one from 20 days ago ~0.05.
    This ensures recent levels dominate older ones of similar touch count.
    """
    if len(df) < 2 * window + 1:
        return []

    # Compute "days ago" for each bar relative to the last bar
    last_ts  = df.index[-1]
    highs    = df["high"].values
    lows     = df["low"].values
    idx_vals = df.index

    # Store (price, days_ago) tuples
    local_highs: list[tuple[float, float]] = []
    local_lows:  list[tuple[float, float]] = []

    for i in range(window, len(df) - window):
        segment_h = highs[i - window : i + window + 1]
        segment_l = lows[i  - window : i + window + 1]
        days_ago = max((last_ts - idx_vals[i]).total_seconds() / 86400, 0.5)
        if highs[i] == np.max(segment_h):
            local_highs.append((float(highs[i]), days_ago))
        if lows[i] == np.min(segment_l):
            local_lows.append((float(lows[i]), days_ago))

    def _cluster(pts_with_age: list[tuple[float, float]], is_support: bool) -> List[SRLevel]:
        if not pts_with_age:
            return []
        pts_with_age.sort(key=lambda x: x[0])
        clusters: list[list[tuple[float, float]]] = [[pts_with_age[0]]]
        for p, d in pts_with_age[1:]:
            ref = clusters[-1][-1][0]
            if ref > 0 and abs(p - ref) / ref <= cluster_pct:
                clusters[-1].append((p, d))
            else:
                clusters.append([(p, d)])
        result = []
        for cl in clusters:
            centroid  = float(np.mean([x[0] for x in cl]))
            # Time-decay strength: recent touches count more
            strength  = float(sum(1.0 / max(d, 0.5) for _, d in cl))
            result.append(SRLevel(
                price=round(centroid, 4),
                method="local_low" if is_support else "local_high",
                is_support=is_support,
                strength=round(strength, 4),
            ))
        return result

    return _cluster(local_highs, is_support=False) + _cluster(local_lows, is_support=True)


# ---------------------------------------------------------------------------
# 3. Daily Anchor Levels — Multi-Timeframe Hard S/R
# ---------------------------------------------------------------------------

def _daily_anchors(
    df: pd.DataFrame,
    n_days: int = 5,
) -> List[SRLevel]:
    """
    Pull the High and Low from the last `n_days` completed daily candles.
    These are the "hard" anchor levels professional traders use as primary S/R.
    A level is assigned:
      strength = n_days - day_rank  (most recent day = highest strength)
    If a 5-min cluster (from _local_extrema) sits within 0.3% of a daily H/L,
    its strength will be doubled in compute_sr_levels (confluence bonus).
    """
    if df.empty:
        return []

    df_tz = df.copy()
    if df_tz.index.tzinfo is None:
        df_tz.index = df_tz.index.tz_localize("UTC")
    df_tz = df_tz.tz_convert("America/New_York")
    df_tz["_date"] = df_tz.index.date

    days = sorted(df_tz["_date"].unique())
    # Take up to the last n_days completed days (exclude the current/last partial day)
    completed = days[:-1] if len(days) > 1 else days
    target_days = completed[-n_days:]

    levels: List[SRLevel] = []
    for rank, day in enumerate(reversed(target_days)):
        day_df = df_tz[df_tz["_date"] == day]
        if day_df.empty:
            continue
        strength = float(n_days - rank)    # most recent = n_days, oldest = 1
        H = float(day_df["high"].max())
        L = float(day_df["low"].min())
        levels.append(SRLevel(price=round(H, 4), method="daily_high", is_support=False, strength=strength))
        levels.append(SRLevel(price=round(L, 4), method="daily_low",  is_support=True,  strength=strength))

    return [lv for lv in levels if lv.price > 0]


# ---------------------------------------------------------------------------
# 4. Volume Profile — POC + Value Area (with optional visible-range filter)
# ---------------------------------------------------------------------------

def _volume_profile(
    df: pd.DataFrame,
    n_bins: int = 100,
    value_area_pct: float = 0.70,
    visible_range: Optional[tuple] = None,
) -> List[SRLevel]:
    """
    Compute the Volume Profile with optional visible-range filtering.

    visible_range : Optional (start_ts, end_ts) tuple. If provided, only bars
                    in that range are used (Visible Range Volume Profile — VRVP).
                    As the user scrolls/zooms the chart, pass the visible range
                    to update the levels dynamically.
    """
    if visible_range is not None:
        start_ts, end_ts = visible_range
        df = df[(df.index >= start_ts) & (df.index <= end_ts)]

    if df.empty or "volume" not in df.columns:
        return []

    price_min = float(df["low"].min())
    price_max = float(df["high"].max())
    if price_min >= price_max:
        return []

    bins = np.linspace(price_min, price_max, n_bins + 1)
    bin_vol = np.zeros(n_bins)

    for _, row in df.iterrows():
        lo, hi, vol = row["low"], row["high"], row["volume"]
        if vol <= 0 or lo >= hi:
            continue
        # Distribute bar's volume uniformly across its price range
        lo_idx = max(0, np.searchsorted(bins, lo, side="left") - 1)
        hi_idx = min(n_bins - 1, np.searchsorted(bins, hi, side="right") - 1)
        if lo_idx > hi_idx:
            continue
        span = hi_idx - lo_idx + 1
        bin_vol[lo_idx : hi_idx + 1] += vol / span

    bin_prices = (bins[:-1] + bins[1:]) / 2
    poc_idx    = int(np.argmax(bin_vol))
    poc_price  = float(bin_prices[poc_idx])

    # Value Area: start from POC, expand outward until 70% of total volume
    total_vol  = bin_vol.sum()
    target_vol = total_vol * value_area_pct
    lo_idx, hi_idx = poc_idx, poc_idx
    accumulated = bin_vol[poc_idx]

    while accumulated < target_vol:
        expand_down = lo_idx > 0
        expand_up   = hi_idx < n_bins - 1
        if not expand_down and not expand_up:
            break
        vol_down = bin_vol[lo_idx - 1] if expand_down else -1
        vol_up   = bin_vol[hi_idx + 1] if expand_up   else -1
        if vol_down >= vol_up and expand_down:
            lo_idx      -= 1
            accumulated += bin_vol[lo_idx]
        elif expand_up:
            hi_idx      += 1
            accumulated += bin_vol[hi_idx]
        else:
            break

    vah = float(bin_prices[hi_idx])
    val = float(bin_prices[lo_idx])

    return [
        SRLevel(price=round(poc_price, 4), method="poc", is_support=False, strength=3.0),
        SRLevel(price=round(vah, 4),       method="vah", is_support=False, strength=2.0),
        SRLevel(price=round(val, 4),       method="val", is_support=True,  strength=2.0),
    ]


# ---------------------------------------------------------------------------
# 4. K-Means clustering (optional — requires scikit-learn)
# ---------------------------------------------------------------------------

def _kmeans_levels(
    df: pd.DataFrame,
    n_clusters: int = 6,
    window: int = 5,
) -> List[SRLevel]:
    """
    Extract local highs/lows then cluster them with K-Means.
    Gracefully returns [] if sklearn is not installed.
    """
    try:
        from sklearn.cluster import KMeans  # type: ignore
    except ImportError:
        return []

    highs = df["high"].values
    lows  = df["low"].values
    points: list[float] = []

    for i in range(window, len(df) - window):
        if highs[i] == np.max(highs[i - window : i + window + 1]):
            points.append(float(highs[i]))
        if lows[i] == np.min(lows[i - window : i + window + 1]):
            points.append(float(lows[i]))

    if len(points) < n_clusters:
        return []

    X = np.array(points).reshape(-1, 1)
    km = KMeans(n_clusters=min(n_clusters, len(points)), random_state=42, n_init=10)
    km.fit(X)
    centroids = sorted(float(c[0]) for c in km.cluster_centers_)
    median_price = float(np.median(points))

    levels = []
    for c in centroids:
        levels.append(SRLevel(
            price=round(c, 4),
            method="kmeans",
            is_support=c <= median_price,
            strength=1.5,
        ))
    return levels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_sr_levels(
    df: pd.DataFrame,
    use_pivots:        bool = True,
    use_extrema:       bool = True,
    use_vprofile:      bool = True,
    use_kmeans:        bool = True,
    use_daily_anchors: bool = True,
    extrema_window:    int  = 5,
    cluster_pct:       float = 0.003,
    min_strength:      float = 0.5,
    current_price:     Optional[float] = None,
    price_range_pct:   float = 0.30,
    n_anchor_days:     int = 5,
    visible_range:     Optional[tuple] = None,
    confluence_pct:    float = 0.003,
) -> SRLevels:
    """
    Compute multi-method, time-decay-weighted S/R levels.

    Parameters
    ----------
    df               : 5-min OHLCV DataFrame with UTC DatetimeIndex.
    use_pivots       : Include Floor-Trader pivot points.
    use_extrema      : Include fractal local-high/low clusters (time-decay weighted).
    use_vprofile     : Include Volume Profile (POC, VAH, VAL).
    use_kmeans       : Include K-Means clustered levels (requires sklearn).
    use_daily_anchors: Include daily H/L anchors from last n_anchor_days sessions.
    extrema_window   : Lookback/lookahead bars for fractal detection.
    cluster_pct      : Merge nearby extrema within this % of each other.
    min_strength     : Discard levels with strength below this (default 0.5).
    current_price    : If set, only return levels within ±price_range_pct.
    price_range_pct  : Filter band (default ±30%).
    n_anchor_days    : Number of daily H/L anchor sessions to use.
    visible_range    : Optional (start_ts, end_ts) for Visible Range Volume Profile.
    confluence_pct   : If an intraday level sits within this % of a daily anchor,
                       double its strength (confluence bonus).
    """
    if df.empty:
        return SRLevels()

    raw: List[SRLevel] = []

    if use_pivots:
        raw.extend(_pivot_points(df))
    if use_extrema:
        raw.extend(_local_extrema(df, window=extrema_window, cluster_pct=cluster_pct))
    if use_vprofile:
        raw.extend(_volume_profile(df, visible_range=visible_range))
    if use_kmeans:
        raw.extend(_kmeans_levels(df))

    # Daily anchors (computed separately so we can apply confluence bonus)
    anchor_levels: List[SRLevel] = []
    if use_daily_anchors:
        anchor_levels = _daily_anchors(df, n_days=n_anchor_days)
        raw.extend(anchor_levels)

    # ── Confluence bonus: if an intraday level sits on top of a daily anchor,
    #    double its strength — this is the "hard level" signal
    anchor_prices = [lv.price for lv in anchor_levels]
    for lv in raw:
        if lv.method in ("daily_high", "daily_low", "pivot_p",
                         "pivot_r1", "pivot_r2", "pivot_s1", "pivot_s2"):
            continue  # don't double-boost anchors themselves
        for ap in anchor_prices:
            if ap > 0 and abs(lv.price - ap) / ap <= confluence_pct:
                lv.strength *= 2.0
                break

    # Filter by strength
    raw = [lv for lv in raw if lv.strength >= min_strength and lv.price > 0]

    # Filter by proximity to current price
    if current_price and current_price > 0:
        lo = current_price * (1 - price_range_pct)
        hi = current_price * (1 + price_range_pct)
        raw = [lv for lv in raw if lo <= lv.price <= hi]

    # Deduplicate: merge levels within 0.3% (keep highest strength)
    raw.sort(key=lambda lv: lv.price)
    deduped: List[SRLevel] = []
    for lv in raw:
        if deduped and lv.price > 0:
            prev = deduped[-1]
            if abs(lv.price - prev.price) / prev.price < 0.003:
                if lv.strength > prev.strength:
                    deduped[-1] = lv
                continue
        deduped.append(lv)

    supports    = sorted([lv.price for lv in deduped if lv.is_support])
    resistances = sorted([lv.price for lv in deduped if not lv.is_support])

    return SRLevels(
        supports=supports,
        resistances=resistances,
        all_levels=deduped,
    )


# ---------------------------------------------------------------------------
# Role-Reversal Detection
# ---------------------------------------------------------------------------

def detect_role_reversals(
    df: pd.DataFrame,
    level: float,
    band_pct: float = 0.003,
    lookback_bars: int = 100,
    min_closes_above: int = 3,
) -> bool:
    """
    Determine if *level* is a **role-reversal** support — i.e. price previously
    treated it as resistance (multiple closes above it), then broke through and
    is now retesting it from above as support.

    Algorithm
    ---------
    1. Look at the last `lookback_bars` bars.
    2. Split into two phases:
       a. 'Above phase'  — bars where close > level × (1 + band_pct)  ← old resistance
       b. 'Retest phase' — most-recent bars where close is within band of level
    3. Returns True if:
       - There were ≥ min_closes_above closes above the level (resistance was tested)
       - AND the most recent close is within band_pct of the level (retesting)
       - AND there was a downward cross back to the level after the above phase

    Parameters
    ----------
    df             : 5-min OHLCV DataFrame (UTC DatetimeIndex).
    level          : The price level to test for role reversal.
    band_pct       : Proximity band (default 0.3%).
    lookback_bars  : How many recent bars to examine.
    min_closes_above: Minimum number of closes that must have been above the level.
    """
    if df.empty or level <= 0:
        return False

    recent = df.iloc[-lookback_bars:] if len(df) >= lookback_bars else df
    closes = recent["close"].values

    band_hi = level * (1 + band_pct)
    band_lo = level * (1 - band_pct)

    n_above     = int((closes > band_hi).sum())
    last_close  = float(closes[-1])
    near_level  = band_lo <= last_close <= band_hi * 1.01   # within 0.3% on retest

    if n_above < min_closes_above:
        return False

    # There must be a descent back to the level: find last bar above, then a bar near level
    last_above_idx = -1
    for i in range(len(closes) - 1, -1, -1):
        if closes[i] > band_hi:
            last_above_idx = i
            break

    if last_above_idx < 0:
        return False

    # At least one bar after the last-above must be at or below the level
    post_above = closes[last_above_idx + 1:]
    descended  = any(c <= band_hi for c in post_above)

    return descended and near_level


def count_rejections(
    df: pd.DataFrame,
    level: float,
    band_pct: float = 0.003,
    lookback_bars: int = 60,
) -> int:
    """
    Count the number of times price touched *level* (low ≤ level × (1+band))
    but the bar **closed above** it — i.e. rejection wicks off support.

    Used by the scorer to measure downtrend setup quality:
    multiple rejections at a support = stronger base for a long entry.
    """
    if df.empty or level <= 0:
        return 0

    recent = df.iloc[-lookback_bars:] if len(df) >= lookback_bars else df
    touch_band = level * (1 + band_pct)
    count = 0
    for _, row in recent.iterrows():
        touched  = float(row["low"])  <= touch_band
        rejected = float(row["close"]) > level
        if touched and rejected:
            count += 1
    return count
