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

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


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
    pattern_score:    float = 0.0   # 0 = none, 1 = near support, 2 = confirmed
    adx_score:        float = 0.0   # 0 = ADX<15, 1 = 15–25, 2 = >25
    rr_score:         float = 0.0   # 0 = no levels, 1 = R/R<1.5, 2 = R/R≥2.0
    confluence_score: float = 0.0   # 0 = none, 1 = 1 method, 2 = 2+ methods agree
    history_score:    float = 0.0   # 0 = no data, 1 = WR<50%, 2 = WR≥60%

    # Raw data for the card display
    entry_price:    float = 0.0
    support:        float = 0.0
    resistance:     Optional[float] = None
    stop:           float = 0.0
    rr_ratio:       float = 0.0
    adx:            float = 0.0
    pattern_found:  bool  = False
    watchlist_note: str   = ""      # original text from the post

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

def _score_pattern(bars: pd.DataFrame, support: Optional[float]) -> tuple[float, bool]:
    """
    Pattern score (0–2).
      2 = Side-by-Side confirmed within 0.5% of support
      1 = pattern exists anywhere in last 30 bars
      0 = no pattern
    Returns (score, pattern_found).
    """
    from .pattern_engine import detect_side_by_side, pattern_near_support

    if bars.empty:
        return 0.0, False

    recent = bars.iloc[-30:] if len(bars) >= 30 else bars

    if support and support > 0:
        near = pattern_near_support(recent, support, proximity_pct=0.005)
        if near:
            return 2.0, True

    anywhere = detect_side_by_side(recent)
    if anywhere:
        return 1.0, True

    return 0.0, False


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
        return 0.0, 0.0, support or 0.0

    # Stop = support − 0.5% if not explicitly given
    stop_price = stop if (stop and stop > 0) else support * 0.995
    risk   = entry - stop_price
    reward = resistance - entry

    if risk <= 0:
        return 0.0, 0.0, stop_price

    rr = reward / risk
    if rr >= 2.0:
        return 2.0, round(rr, 2), stop_price
    if rr >= 1.0:
        return 1.0, round(rr, 2), stop_price
    return 0.0, round(rr, 2), stop_price


def _score_confluence(bars: pd.DataFrame, support: Optional[float]) -> tuple[float, str]:
    """
    Confluence score (0–2): how many independent S/R methods agree on the support.
      2 = 2+ methods within 0.3% of support
      1 = exactly 1 method
      0 = support missing or no methods agree
    Returns (score, description_string).
    """
    from .sr_engine import compute_sr_levels

    if not support or support <= 0 or bars.empty:
        return 0.0, "no support level"

    levels = compute_sr_levels(bars, current_price=support, price_range_pct=0.05)
    threshold = support * 0.003     # 0.3% band

    agreeing_methods: list[str] = []
    for lv in levels.all_levels:
        if abs(lv.price - support) <= threshold:
            agreeing_methods.append(lv.method)

    n = len(agreeing_methods)
    desc = ", ".join(agreeing_methods[:4]) or "none"
    if n >= 2:
        return 2.0, f"{n} methods agree ({desc})"
    if n == 1:
        return 1.0, f"1 method ({desc})"
    return 0.0, "no confluence"


def _score_history(ticker: str, db_path: Optional[str] = None) -> tuple[float, float, int]:
    """
    History score (0–2) from the trade DB win-rate for this ticker.
      2 = win-rate ≥ 60%
      1 = win-rate 30–59%
      0 = fewer than 3 trades or win-rate < 30%
    Returns (score, win_rate, total_trades).
    """
    from pathlib import Path
    from .db import WatchlistDB

    path = db_path or str(Path(__file__).parent / "watchlist_backtest.db")
    try:
        with WatchlistDB(path) as db:
            trades = db.load_trades()
    except Exception:
        return 0.0, 0.0, 0

    ticker_trades = [t for t in trades if t.ticker == ticker.upper()]
    total = len(ticker_trades)
    if total < 3:
        return 0.0, 0.0, total

    wins = sum(1 for t in ticker_trades if t.outcome == "win")
    wr   = wins / total
    if wr >= 0.60:
        return 2.0, round(wr, 3), total
    if wr >= 0.30:
        return 1.0, round(wr, 3), total
    return 0.0, round(wr, 3), total


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def score_setup(
    entry,                          # WatchlistEntry
    bars: pd.DataFrame,
    db_path: Optional[str] = None,
) -> "SetupScore":
    """
    Compute a composite SetupScore for *entry* given its OHLCV bars.

    Parameters
    ----------
    entry    : WatchlistEntry — parsed ticker + support/resistance/stop levels.
    bars     : pd.DataFrame  — 5-min OHLCV bars (UTC DatetimeIndex).
    db_path  : path to SQLite trade DB; defaults to the package default.
    """
    support    = entry.support_level
    resistance = entry.resistance_level
    stop       = entry.stop_level
    entry_price = support or (float(bars["close"].iloc[-1]) if not bars.empty else 0.0)

    pat_s,  pat_found            = _score_pattern(bars, support)
    adx_s,  adx_val              = _score_adx(bars)
    rr_s,   rr_ratio, stop_price = _score_rr(support, resistance, stop, entry_price)
    conf_s, conf_desc            = _score_confluence(bars, support)
    hist_s, win_rate, n_trades   = _score_history(entry.ticker, db_path)

    total = pat_s + adx_s + rr_s + conf_s + hist_s

    sig = "🟢 STRONG" if total >= 7.0 else ("🟡 WATCH" if total >= 4.0 else "🔴 SKIP")

    return SetupScore(
        ticker=entry.ticker,
        score=round(total, 2),
        signal=sig,
        pattern_score=pat_s,
        adx_score=adx_s,
        rr_score=rr_s,
        confluence_score=conf_s,
        history_score=hist_s,
        entry_price=round(entry_price, 4),
        support=round(support, 4) if support else 0.0,
        resistance=round(resistance, 4) if resistance else None,
        stop=round(stop_price, 4),
        rr_ratio=rr_ratio,
        adx=round(adx_val, 1),
        pattern_found=pat_found,
        watchlist_note=entry.raw_text or entry.sentiment_notes,
    )
