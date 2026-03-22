"""
Unit tests for evolution/gp_fitness.py

Covers FitnessResult, calculate_fitness(), calculate_fitness_v2(),
_recency_weighted_mean(), and _get_drawdown_thresholds().
"""
import numpy as np
import pytest

from evolution.gp_fitness import (
    FitnessResult,
    calculate_fitness,
    calculate_fitness_v2,
    _recency_weighted_mean,
    _get_drawdown_thresholds,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _period(ret=0.05, sharpe=0.8, dd=0.12, turnover=0.3, n_days=63):
    return dict(total_return=ret, sharpe_ratio=sharpe, max_drawdown=dd,
                turnover=turnover, n_days=n_days)


def _bench(ret=0.03, sharpe=0.5, dd=0.10):
    return dict(total_return=ret, sharpe_ratio=sharpe, max_drawdown=dd)


def _n_periods(n, **kw):
    return [_period(**kw) for _ in range(n)]


def _n_benchmarks(n, **kw):
    return [_bench(**kw) for _ in range(n)]


# ─── _recency_weighted_mean ───────────────────────────────────────────────────

class TestRecencyWeightedMean:
    def test_empty_returns_zero(self):
        assert _recency_weighted_mean([]) == 0.0

    def test_single_value_identity(self):
        assert _recency_weighted_mean([0.75]) == pytest.approx(0.75)

    def test_equal_values_return_same_value(self):
        result = _recency_weighted_mean([0.3, 0.3, 0.3, 0.3], half_life_periods=4)
        assert result == pytest.approx(0.3, abs=1e-10)

    def test_recent_weight_dominates_with_short_half_life(self):
        # values = [0, 0, 0, 0, 1.0]  — only last element is non-zero
        values = [0.0, 0.0, 0.0, 0.0, 1.0]
        short = _recency_weighted_mean(values, half_life_periods=2)
        long_ = _recency_weighted_mean(values, half_life_periods=20)
        assert short > long_, "Short HL should give last element more weight"
        assert short > 0.2, "Short HL should give last element more than simple mean"

    def test_old_value_dominates_with_long_half_life(self):
        # values = [1.0, 0, 0, 0, 0]  — only first element is non-zero
        values = [1.0, 0.0, 0.0, 0.0, 0.0]
        short = _recency_weighted_mean(values, half_life_periods=1)
        long_ = _recency_weighted_mean(values, half_life_periods=10)
        assert short < long_, "Short HL should devalue old data"

    def test_result_is_between_min_and_max(self):
        values = [0.1, 0.4, 0.7, 0.2, 0.9]
        result = _recency_weighted_mean(values)
        assert min(values) <= result <= max(values)


# ─── _get_drawdown_thresholds ─────────────────────────────────────────────────

class TestDrawdownThresholds:
    def test_general_thresholds(self):
        t = _get_drawdown_thresholds("general")
        assert t["severe"] == pytest.approx(0.35)
        assert t["moderate"] == pytest.approx(0.25)
        assert t["mild"] == pytest.approx(0.15)

    def test_oil_microcap_wider_than_general(self):
        g = _get_drawdown_thresholds("general")
        o = _get_drawdown_thresholds("oil_microcap")
        assert o["severe"] > g["severe"]
        assert o["moderate"] > g["moderate"]
        assert o["mild"] > g["mild"]

    def test_oil_microcap_specific_values(self):
        t = _get_drawdown_thresholds("oil_microcap")
        assert t["severe"] == pytest.approx(0.50)
        assert t["moderate"] == pytest.approx(0.35)
        assert t["mild"] == pytest.approx(0.20)

    def test_oil_largecap_between_general_and_microcap(self):
        g = _get_drawdown_thresholds("general")
        l = _get_drawdown_thresholds("oil_largecap")
        m = _get_drawdown_thresholds("oil_microcap")
        assert g["severe"] <= l["severe"] <= m["severe"]

    def test_unknown_universe_falls_back_to_general(self):
        t = _get_drawdown_thresholds("unknown_type_xyz")
        g = _get_drawdown_thresholds("general")
        assert t["severe"] == g["severe"]

    def test_all_thresholds_have_required_keys(self):
        for ut in ["general", "oil_microcap", "oil_largecap"]:
            t = _get_drawdown_thresholds(ut)
            assert "severe" in t
            assert "moderate" in t
            assert "mild" in t


# ─── calculate_fitness (v1) ───────────────────────────────────────────────────

class TestCalculateFitness:
    def test_returns_fitness_result_type(self):
        r = calculate_fitness(_n_periods(5), _n_benchmarks(5))
        assert isinstance(r, FitnessResult)

    def test_empty_returns_zero(self):
        r = calculate_fitness([], [])
        assert r.total == 0
        assert r.n_periods == 0

    def test_total_bounded(self):
        for _ in range(20):
            np.random.seed(np.random.randint(0, 9999))
            periods = [_period(
                ret=np.random.uniform(-0.2, 0.3),
                sharpe=np.random.uniform(-1, 2),
                dd=np.random.uniform(0, 0.6),
                turnover=np.random.uniform(0, 1),
            ) for _ in range(5)]
            r = calculate_fitness(periods, _n_benchmarks(5))
            assert -1 <= r.total <= 1, f"total={r.total} out of bounds"

    def test_strong_strategy_beats_zero(self):
        """A strategy with good Sharpe & low DD & beating benchmark should be > 0."""
        r = calculate_fitness(
            _n_periods(6, ret=0.20, sharpe=1.5, dd=0.10, turnover=0.2),
            _n_benchmarks(6, ret=0.05, sharpe=0.5, dd=0.08),
        )
        assert r.total > 0

    def test_terrible_strategy_below_zero(self):
        """A strategy with negative returns and huge drawdown should be < 0."""
        r = calculate_fitness(
            _n_periods(6, ret=-0.20, sharpe=-1.5, dd=0.50, turnover=1.0),
            _n_benchmarks(6, ret=0.05, sharpe=0.5, dd=0.08),
        )
        assert r.total < 0

    def test_n_periods_correct(self):
        r = calculate_fitness(_n_periods(4), _n_benchmarks(4))
        assert r.n_periods == 4

    def test_avg_return_computed(self):
        r = calculate_fitness(_n_periods(5, ret=0.10), _n_benchmarks(5))
        assert abs(r.avg_return - 0.10) < 1e-9

    def test_avg_sharpe_computed(self):
        r = calculate_fitness(_n_periods(5, sharpe=1.2), _n_benchmarks(5))
        assert abs(r.avg_sharpe - 1.2) < 1e-9

    def test_win_rate_all_wins(self):
        """If strategy always beats benchmark return, win_rate = 1.0."""
        periods = _n_periods(5, ret=0.15)
        benchmarks = _n_benchmarks(5, ret=0.05)
        r = calculate_fitness(periods, benchmarks)
        assert r.win_rate == 1.0

    def test_win_rate_no_wins(self):
        periods = _n_periods(5, ret=0.02)
        benchmarks = _n_benchmarks(5, ret=0.10)
        r = calculate_fitness(periods, benchmarks)
        assert r.win_rate == 0.0

    def test_fewer_than_4_periods_reduces_total(self):
        r3 = calculate_fitness(_n_periods(3), _n_benchmarks(3))
        r4 = calculate_fitness(_n_periods(4), _n_benchmarks(4))
        # With identical per-period metrics, having only 3 periods should not
        # outperform 4 periods (due to the count adjustment).
        assert r3.total <= r4.total * 1.05  # allow tiny float error

    def test_high_turnover_penalised(self):
        low_to = calculate_fitness(
            _n_periods(5, turnover=0.05),
            _n_benchmarks(5),
        )
        high_to = calculate_fitness(
            _n_periods(5, turnover=1.0),
            _n_benchmarks(5),
        )
        assert low_to.total >= high_to.total


# ─── calculate_fitness_v2 ─────────────────────────────────────────────────────

class TestCalculateFitnessV2:
    def test_returns_fitness_result_type(self):
        np.random.seed(42)
        r = calculate_fitness_v2(_n_periods(5), _n_benchmarks(5))
        assert isinstance(r, FitnessResult)

    def test_empty_returns_zero(self):
        r = calculate_fitness_v2([], [])
        assert r.total == 0
        assert r.n_periods == 0

    def test_total_bounded(self):
        np.random.seed(0)
        r = calculate_fitness_v2(_n_periods(6), _n_benchmarks(6))
        assert -1 <= r.total <= 1

    def test_has_expected_attributes(self):
        r = calculate_fitness_v2(_n_periods(5), _n_benchmarks(5))
        assert r.sharpe_component is not None
        assert r.return_component is not None
        assert r.stability_component is not None

    def test_oil_microcap_less_penalised_for_high_dd(self):
        """Same high-drawdown strategy should score better under oil_microcap."""
        np.random.seed(42)
        periods = _n_periods(6, dd=0.38)
        benchmarks = _n_benchmarks(6)
        general = calculate_fitness_v2(periods, benchmarks, universe_type="general")
        oil = calculate_fitness_v2(periods, benchmarks, universe_type="oil_microcap")
        assert oil.total >= general.total

    def test_recency_short_hl_favours_recent_improvement(self):
        """Short half-life should weight recent strong periods more."""
        np.random.seed(42)
        periods = _n_periods(8, ret=-0.05, sharpe=-0.3)
        periods[-1]["total_return"] = 0.15
        periods[-1]["sharpe_ratio"] = 1.5
        periods[-2]["total_return"] = 0.12
        periods[-2]["sharpe_ratio"] = 1.2
        benchmarks = _n_benchmarks(8)
        short_hl = calculate_fitness_v2(periods, benchmarks, recency_half_life=2)
        long_hl = calculate_fitness_v2(periods, benchmarks, recency_half_life=20)
        assert short_hl.total > long_hl.total
