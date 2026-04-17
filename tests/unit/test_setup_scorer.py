"""
Unit tests for side_by_side_backtest/setup_scorer.py

Covers:
  - _score_adx       — ADX threshold tiers
  - _score_rr        — Risk/Reward ratio tiers
  - score_setup      — normalisation (raw/24 * 10), EQH field population
  - SetupScore       — field defaults, signal_label
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(n: int = 60, price: float = 10.0, adx_val: float = 30.0,
               trending: bool = True) -> pd.DataFrame:
    """
    Synthetic OHLCV DataFrame.

    When trending=True (default), generates a smooth downtrend so that the
    Wilder ADX calculation converges to a high value (>25) after ≥28 bars.
    When trending=False, generates a sideways choppy series (low ADX).

    adx_val is ignored — the actual ADX is computed from the OHLC series
    by _score_adx (it does not read the 'adx' column).
    """
    import numpy as np

    base_ts = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    idx = pd.DatetimeIndex(
        [base_ts + timedelta(minutes=5 * i) for i in range(n)],
        tz="UTC",
    )

    if trending:
        # Consistent downtrend: price drops 0.5% per bar → very high ADX
        prices = [price * (1 - 0.005 * i) for i in range(n)]
        opens  = prices
        highs  = [p * 1.002 for p in prices]
        lows   = [p * 0.998 for p in prices]
        closes = prices
    else:
        # Choppy sideways: alternates +/- 0.1% → very low ADX
        closes = [price * (1 + 0.001 * (1 if i % 2 == 0 else -1)) for i in range(n)]
        opens  = closes
        highs  = [p * 1.001 for p in closes]
        lows   = [p * 0.999 for p in closes]

    df = pd.DataFrame({
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": [500_000] * n,
    }, index=idx)
    return df


def _make_entry(
    ticker: str = "TEST",
    support: float = 9.8,
    resistance: float = 11.0,
    stop: float = 9.5,
):
    from side_by_side_backtest.models import SessionType, WatchlistEntry
    return WatchlistEntry(
        post_title=f"{ticker} test",
        post_timestamp=datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc),
        ticker=ticker,
        session_type=SessionType.MARKET_OPEN,
        support_level=support,
        resistance_level=resistance,
        stop_level=stop,
    )


# ---------------------------------------------------------------------------
# _score_adx
# ---------------------------------------------------------------------------

class TestScoreAdx:
    """
    _score_adx() recomputes ADX from OHLC bars using Wilder's method.
    We use controlled price series to drive ADX into the desired range.
    """

    def test_strong_trend_scores_2(self):
        """Consistent downtrend (0.5%/bar × 60 bars) → ADX > 25 → score 2."""
        from side_by_side_backtest.setup_scorer import _score_adx
        bars = _make_bars(n=60, price=10.0, trending=True)
        score, val = _score_adx(bars)
        # A strong sustained downtrend produces high ADX; score should be 2
        assert score == 2.0, (
            f"Strong downtrend should score 2.0, got {score} (ADX={val:.1f}). "
            "Check that _make_bars(trending=True) generates a steep enough trend."
        )
        assert val > 25, f"Expected ADX > 25, got {val:.1f}"

    def test_sideways_scores_0(self):
        """Choppy sideways price (±0.1%/bar) → ADX < 15 → score 0."""
        from side_by_side_backtest.setup_scorer import _score_adx
        bars = _make_bars(n=60, price=10.0, trending=False)
        score, val = _score_adx(bars)
        assert score == 0.0, (
            f"Sideways chop should score 0.0, got {score} (ADX={val:.1f})."
        )

    def test_few_bars_returns_zero(self):
        """Fewer than 30 bars → score must be 0 regardless."""
        from side_by_side_backtest.setup_scorer import _score_adx
        bars = _make_bars(n=5, trending=True)
        score, val = _score_adx(bars)
        assert score == 0.0, "Fewer than 30 bars → score must be 0"


# ---------------------------------------------------------------------------
# _score_rr
# ---------------------------------------------------------------------------

class TestScoreRR:

    def _score(self, support, resistance, stop, entry):
        from side_by_side_backtest.setup_scorer import _score_rr
        return _score_rr(support, resistance, stop, entry)

    def test_rr_2_or_higher_scores_2(self):
        # entry 10, resistance 12, stop 9.9 → risk 0.1, reward 2.0 → R/R 20
        score, rr, stop = self._score(10.0, 12.0, 9.9, 10.0)
        assert score == 2.0, f"R/R ≥ 2 should score 2.0, got {score}"
        assert rr >= 2.0

    def test_rr_1_to_2_scores_1(self):
        # entry 10, resistance 10.5, stop 9.7 → risk 0.3, reward 0.5 → R/R ~1.67
        score, rr, stop = self._score(10.0, 10.5, 9.7, 10.0)
        assert score == 1.0, f"R/R 1-2 should score 1.0, got {score} (rr={rr})"

    def test_rr_below_1_scores_0(self):
        # entry 10, resistance 10.2, stop 9.5 → risk 0.5, reward 0.2 → R/R 0.4
        score, rr, stop = self._score(10.0, 10.2, 9.5, 10.0)
        assert score == 0.0, f"R/R < 1 should score 0.0, got {score} (rr={rr})"

    def test_missing_resistance_scores_0(self):
        score, rr, stop = self._score(10.0, None, 9.9, 10.0)
        assert score == 0.0

    def test_stop_defaults_to_1pct_below_entry(self):
        # No explicit stop → default to entry * 0.99
        score, rr, stop = self._score(10.0, 12.0, None, 10.0)
        assert stop == pytest.approx(10.0 * 0.99, abs=0.01)


# ---------------------------------------------------------------------------
# SetupScore — signal_label
# ---------------------------------------------------------------------------

class TestSetupScoreSignalLabel:

    def _make_score(self, total: float):
        from side_by_side_backtest.setup_scorer import SetupScore
        return SetupScore(ticker="TEST", score=total, signal="")

    def test_strong_signal(self):
        sc = self._make_score(7.5)
        assert "STRONG" in sc.signal_label()

    def test_watch_signal(self):
        sc = self._make_score(5.0)
        assert "WATCH" in sc.signal_label()

    def test_skip_signal(self):
        sc = self._make_score(2.0)
        assert "SKIP" in sc.signal_label()


# ---------------------------------------------------------------------------
# score_setup — normalisation and EQH field
# ---------------------------------------------------------------------------

class TestScoreSetupNormalisation:

    def test_score_is_0_to_10(self):
        """score_setup must return a score in [0, 10] regardless of inputs."""
        from side_by_side_backtest.setup_scorer import score_setup
        entry = _make_entry()
        bars  = _make_bars(n=60, price=10.0, adx_val=30.0)
        sc = score_setup(entry, bars)
        assert 0.0 <= sc.score <= 10.0, f"Score out of range: {sc.score}"

    def test_eqh_fields_present(self):
        """SetupScore must have eqh_level and eqh_signal attributes."""
        from side_by_side_backtest.setup_scorer import score_setup
        entry = _make_entry()
        bars  = _make_bars(n=60, price=10.0, adx_val=25.0)
        sc = score_setup(entry, bars)
        assert hasattr(sc, "eqh_level"), "SetupScore missing eqh_level"
        assert hasattr(sc, "eqh_signal"), "SetupScore missing eqh_signal"
        assert isinstance(sc.eqh_level, float)
        assert isinstance(sc.eqh_signal, str)

    def test_eqh_score_between_0_and_2(self):
        """eqh_score component must be in [0, 2]."""
        from side_by_side_backtest.setup_scorer import score_setup
        entry = _make_entry()
        bars  = _make_bars(n=60, price=10.0, adx_val=25.0)
        sc = score_setup(entry, bars)
        assert 0.0 <= sc.eqh_score <= 2.0, f"eqh_score out of range: {sc.eqh_score}"

    def test_max_possible_score_does_not_exceed_10(self):
        """Even if every component returns 2.0, final score must be ≤ 10.0."""
        from side_by_side_backtest.setup_scorer import SetupScore
        # Manually construct a max-component score
        sc = SetupScore(
            ticker="TEST", score=0.0, signal="",
            pattern_score=2.0, adx_score=2.0, rr_score=2.0,
            confluence_score=2.0, history_score=2.0,
            role_reversal_score=2.0, rejection_score=2.0,
            rel_vol_score=2.0, macd_score=2.0, rsi_div_score=2.0,
            regime_score=2.0, eqh_score=2.0,
        )
        # Recompute as score_setup would
        raw = sum([
            sc.pattern_score, sc.adx_score, sc.rr_score, sc.confluence_score,
            sc.history_score, sc.role_reversal_score, sc.rejection_score,
            sc.rel_vol_score, sc.macd_score, sc.rsi_div_score, sc.regime_score,
            sc.eqh_score,
        ])
        total = round(raw / 24.0 * 10.0, 2)
        assert total == 10.0, f"Max raw score (24) should normalise to 10.0, got {total}"
