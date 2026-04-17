"""
Unit tests for side_by_side_backtest/pattern_engine.py

Covers:
  - detect_side_by_side (strict bearish)
  - detect_equal_highs_pair (new EQH mixed-color detector)
  - detect_eqh_signal (breakout / rejection)
"""
from __future__ import annotations

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(rows: list[dict], ticker: str = "TEST",
               prepend_context: int = 12) -> pd.DataFrame:
    """
    Build a minimal 5-min OHLCV DataFrame from a list of dicts with keys
    open, high, low, close, volume.  Index is a UTC DatetimeIndex.

    prepend_context: add N neutral bars before the actual rows so detectors
    have enough history to satisfy their minimum-bar requirements (default 12
    satisfies body_lookback=10 + 2 for detect_equal_highs_pair).
    """
    import pandas as pd
    from datetime import datetime, timezone, timedelta

    base_ts = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)  # 9:30 ET

    # Context bars: neutral candles well away from the pattern price
    context = [
        {"open": 12.0, "close": 12.0, "high": 12.05, "low": 11.95, "volume": 100_000}
        for _ in range(prepend_context)
    ]
    all_rows = context + list(rows)

    records = []
    for r in all_rows:
        records.append({
            "open":   r.get("open",  r.get("close", 10.0)),
            "high":   r.get("high",  r.get("close", 10.0) * 1.002),
            "low":    r.get("low",   r.get("close", 10.0) * 0.998),
            "close":  r.get("close", 10.0),
            "volume": r.get("volume", 100_000),
        })
    df = pd.DataFrame(records,
                      index=pd.DatetimeIndex(
                          [base_ts + timedelta(minutes=5*i) for i in range(len(records))],
                          tz="UTC"
                      ))
    df.attrs["ticker"] = ticker
    return df


# ---------------------------------------------------------------------------
# detect_equal_highs_pair — core EQH tests
# ---------------------------------------------------------------------------

class TestDetectEqualHighsPair:
    """Tests for the new mixed-color EQH detector."""

    def _import(self):
        from side_by_side_backtest.pattern_engine import detect_equal_highs_pair
        return detect_equal_highs_pair

    def test_detects_bull_bear_same_open(self):
        """Bullish C2 + bearish C3, same open → EQH pair detected."""
        detect = self._import()
        # C2: bull candle opening at 10.00, closing at 10.20
        # C3: bear candle opening at 10.01 (within 2%), closing at 9.80
        rows = [
            {"open": 9.5,  "close": 9.6,  "high": 9.65, "low": 9.45},   # context
            {"open": 10.0, "close": 10.2, "high": 10.25, "low": 9.95},  # C2 bull
            {"open": 10.01,"close": 9.80, "high": 10.05, "low": 9.75},  # C3 bear
        ]
        df = _make_bars(rows)
        matches = detect(df)
        assert len(matches) >= 1
        m = matches[0]
        assert m.pattern_type == "eqh_pair"
        assert abs(m.eqh_level - 10.01) < 0.05   # ceiling = max(10.0, 10.01)

    def test_detects_bear_bull_same_open(self):
        """Bearish C2 + bullish C3, same open → EQH pair detected."""
        detect = self._import()
        rows = [
            {"open": 10.5, "close": 10.4, "high": 10.55, "low": 10.35},
            {"open": 10.0, "close": 9.80, "high": 10.05, "low": 9.75},  # C2 bear
            {"open": 10.01,"close": 10.2, "high": 10.25, "low": 9.95},  # C3 bull
        ]
        df = _make_bars(rows)
        matches = detect(df)
        assert len(matches) >= 1
        assert matches[0].pattern_type == "eqh_pair"

    def test_rejects_same_color_pair(self):
        """Two bearish candles with same open must NOT be detected as EQH pair."""
        detect = self._import()
        rows = [
            {"open": 10.5, "close": 10.4, "high": 10.55, "low": 10.35},
            {"open": 10.0, "close": 9.80, "high": 10.05, "low": 9.75},  # bear
            {"open": 10.01,"close": 9.75, "high": 10.05, "low": 9.70},  # bear
        ]
        df = _make_bars(rows)
        matches = detect(df)
        assert len(matches) == 0, "Same-color pair should not be detected as EQH"

    def test_rejects_opens_too_far_apart(self):
        """Candles with opens > 2% apart must not match."""
        detect = self._import()
        rows = [
            {"open": 10.5, "close": 10.4, "high": 10.55, "low": 10.35},
            {"open": 10.0, "close": 10.2, "high": 10.25, "low": 9.95},
            {"open": 10.25,"close": 9.80, "high": 10.30, "low": 9.75},  # >2% away
        ]
        df = _make_bars(rows)
        matches = detect(df, open_tolerance_pct=0.02)
        assert len(matches) == 0

    def test_empty_df_returns_empty(self):
        detect = self._import()
        df = pd.DataFrame(columns=["open","high","low","close","volume"])
        assert detect(df) == []


class TestDetectEqhSignal:
    """Tests for detect_eqh_signal — breakout and rejection."""

    def _import(self):
        from side_by_side_backtest.pattern_engine import (
            detect_equal_highs_pair, detect_eqh_signal
        )
        return detect_equal_highs_pair, detect_eqh_signal

    def test_detects_breakout(self):
        """Bar closing body above EQH ceiling → eqh_breakout."""
        detect_pair, detect_sig = self._import()
        # Build a 4-bar sequence:
        #   bars 0-1: context
        #   bars 1-2: EQH pair (bull+bear at 10.00)
        #   bar 3: breakout bar closes above 10.00
        rows = [
            {"open": 9.8,  "close": 9.9,  "high": 9.95, "low": 9.75},
            {"open": 10.0, "close": 10.2, "high": 10.25, "low": 9.95},   # C2 bull
            {"open": 10.01,"close": 9.80, "high": 10.05, "low": 9.75},   # C3 bear
            {"open": 10.05,"close": 10.15,"high": 10.20, "low": 10.00},  # breakout body above 10.01
        ]
        df = _make_bars(rows)
        pairs   = detect_pair(df)
        signals = detect_sig(df, pairs, body_clear_pct=0.001)
        breakouts = [s for s in signals if s.pattern_type == "eqh_breakout"]
        assert len(breakouts) >= 1, f"Expected breakout signal, got: {[s.pattern_type for s in signals]}"

    def test_detects_rejection(self):
        """Bar closing body below pair low → eqh_rejection."""
        detect_pair, detect_sig = self._import()
        rows = [
            {"open": 9.8,  "close": 9.9,  "high": 9.95, "low": 9.75},
            {"open": 10.0, "close": 10.2, "high": 10.25, "low": 9.95},   # C2 bull
            {"open": 10.01,"close": 9.80, "high": 10.05, "low": 9.75},   # C3 bear (pair_low ~9.80)
            {"open": 9.78, "close": 9.70, "high": 9.82, "low": 9.65},    # rejection close below pair_low
        ]
        df = _make_bars(rows)
        pairs   = detect_pair(df)
        signals = detect_sig(df, pairs, body_clear_pct=0.001)
        rejections = [s for s in signals if s.pattern_type == "eqh_rejection"]
        assert len(rejections) >= 1, f"Expected rejection signal, got: {[s.pattern_type for s in signals]}"

    def test_no_signal_when_flat(self):
        """Bar closes inside the EQH zone → no signal."""
        detect_pair, detect_sig = self._import()
        rows = [
            {"open": 9.8,  "close": 9.9,  "high": 9.95,  "low": 9.75},
            {"open": 10.0, "close": 10.2, "high": 10.25, "low": 9.95},
            {"open": 10.01,"close": 9.80, "high": 10.05, "low": 9.75},
            {"open": 9.85, "close": 9.90, "high": 9.95,  "low": 9.82},   # inside zone
        ]
        df = _make_bars(rows)
        pairs   = detect_pair(df)
        signals = detect_sig(df, pairs)
        assert len(signals) == 0, "No signal expected when price stays inside EQH zone"
