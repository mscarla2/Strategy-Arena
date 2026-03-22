"""
Integration tests for evolution/walkforward.py (WalkForwardEvaluator)

Tests the full evaluate_strategy() pipeline with synthetic data,
parameter serialization, and all fitness modes.
"""
import random
import numpy as np
import pandas as pd
import pytest

from evolution.features import FeatureLibrary
from evolution.tree_ops import TreeGenerator
from evolution.strategy import GPStrategy
from evolution.walkforward import WalkForwardEvaluator
from evolution.gp_fitness import FitnessResult


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def prices():
    """600 business days so execution_lag lookback clears the 504-day max_lookback."""
    np.random.seed(7)
    dates = pd.bdate_range("2021-01-04", periods=600)
    tickers = [f"T{i}" for i in range(6)]
    returns = np.random.normal(0.0005, 0.02, (600, 6))
    return pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=tickers)


@pytest.fixture(scope="module")
def volume(prices):
    np.random.seed(7)
    data = np.random.randint(100_000, 1_000_000, size=prices.shape).astype(float)
    return pd.DataFrame(data, index=prices.index, columns=prices.columns)


@pytest.fixture(scope="module")
def strategy():
    random.seed(42)
    lib = FeatureLibrary()
    gen = TreeGenerator(lib.feature_names)
    return GPStrategy(tree=gen.random_tree(max_depth=3), top_pct=50)


# Dummy benchmark results — calculate_fitness requires at least one entry.
_DUMMY_BENCH = [{"total_return": 0.03, "sharpe_ratio": 0.5, "max_drawdown": 0.10}] * 10


@pytest.fixture(scope="module")
def one_period(prices):
    """Test window starts at d[510] so effective_idx >= max_lookback (504)."""
    d = prices.index
    return [(d[0].strftime("%Y-%m-%d"),   d[509].strftime("%Y-%m-%d"),
             d[510].strftime("%Y-%m-%d"), d[-1].strftime("%Y-%m-%d"))]


@pytest.fixture(scope="module")
def two_periods(prices):
    d = prices.index
    return [
        (d[0].strftime("%Y-%m-%d"),   d[509].strftime("%Y-%m-%d"),
         d[510].strftime("%Y-%m-%d"), d[554].strftime("%Y-%m-%d")),
        (d[50].strftime("%Y-%m-%d"),  d[554].strftime("%Y-%m-%d"),
         d[555].strftime("%Y-%m-%d"), d[-1].strftime("%Y-%m-%d")),
    ]


# ─── Basic evaluation ─────────────────────────────────────────────────────────

class TestBasicEvaluation:
    def test_evaluate_returns_fitness_result(self, prices, strategy, one_period):
        ev = WalkForwardEvaluator(prices=prices, periods=one_period)
        result = ev.evaluate_strategy(strategy)
        assert isinstance(result, FitnessResult)

    def test_evaluate_produces_at_least_one_period(self, prices, strategy, one_period):
        ev = WalkForwardEvaluator(prices=prices, periods=one_period)
        result = ev.evaluate_strategy(strategy)
        assert result.n_periods >= 1

    def test_fitness_bounded(self, prices, strategy, two_periods):
        ev = WalkForwardEvaluator(prices=prices, periods=two_periods)
        result = ev.evaluate_strategy(strategy)
        assert -1 <= result.total <= 1

    def test_no_error_with_volume(self, prices, volume, strategy, one_period):
        ev = WalkForwardEvaluator(prices=prices, periods=one_period, volume=volume)
        result = ev.evaluate_strategy(strategy)
        assert isinstance(result, FitnessResult)

    def test_empty_periods_returns_zero_fitness(self, prices, strategy):
        ev = WalkForwardEvaluator(prices=prices, periods=[])
        result = ev.evaluate_strategy(strategy)
        assert result.total == 0
        assert result.n_periods == 0

    def test_multiple_strategies_produce_different_fitness(self, prices, two_periods):
        """GP diversity: multiple random strategies should not all get identical fitness."""
        random.seed(0)
        lib = FeatureLibrary()
        gen = TreeGenerator(lib.feature_names)
        ev = WalkForwardEvaluator(prices=prices, periods=two_periods)

        fitnesses = []
        for _ in range(8):
            s = GPStrategy(tree=gen.random_tree(max_depth=4), top_pct=50)
            fitnesses.append(ev.evaluate_strategy(s).total)

        unique = len(set(round(f, 4) for f in fitnesses))
        assert unique > 1, "All 8 strategies produced identical fitness"


# ─── tradeable_tickers filter ─────────────────────────────────────────────────

class TestTradeableTickers:
    def test_restricts_to_tradeable_subset(self, prices, strategy, one_period):
        tradeable = ["T0", "T1", "T2"]
        ev = WalkForwardEvaluator(
            prices=prices, periods=one_period, tradeable_tickers=tradeable
        )
        result = ev.evaluate_strategy(strategy)
        assert isinstance(result, FitnessResult)
        assert result.n_periods >= 1

    def test_missing_tradeable_returns_penalty(self, prices, strategy, one_period):
        """Tickers not in prices should be gracefully handled."""
        tradeable = ["NONEXISTENT1", "NONEXISTENT2"]
        ev = WalkForwardEvaluator(
            prices=prices, periods=one_period, tradeable_tickers=tradeable
        )
        result = ev.evaluate_strategy(strategy)
        # No exception — may return 0 periods or penalty
        assert isinstance(result, FitnessResult)


# ─── Fitness modes ────────────────────────────────────────────────────────────

class TestFitnessModes:
    def test_fitness_v2_accepted(self, prices, strategy, two_periods):
        ev = WalkForwardEvaluator(
            prices=prices, periods=two_periods,
            use_fitness_v2=True, universe_type="general",
        )
        result = ev.evaluate_strategy(strategy)
        assert -1 <= result.total <= 1

    def test_fitness_v2_oil_microcap(self, prices, strategy, two_periods):
        ev = WalkForwardEvaluator(
            prices=prices, periods=two_periods,
            use_fitness_v2=True, universe_type="oil_microcap",
        )
        result = ev.evaluate_strategy(strategy)
        assert isinstance(result, FitnessResult)

    def test_recency_half_life_stored(self, prices, two_periods):
        ev = WalkForwardEvaluator(
            prices=prices, periods=two_periods, recency_half_life=3
        )
        assert ev.recency_half_life == 3

    def test_universe_type_stored(self, prices, two_periods):
        ev = WalkForwardEvaluator(
            prices=prices, periods=two_periods, universe_type="oil_microcap"
        )
        assert ev.universe_type == "oil_microcap"


# ─── Serialisation / to_config & from_config ─────────────────────────────────

class TestSerialization:
    def test_to_config_is_dict(self, prices, one_period):
        ev = WalkForwardEvaluator(prices=prices, periods=one_period)
        config = ev.to_config()
        assert isinstance(config, dict)

    def test_from_config_roundtrip(self, prices, one_period):
        ev = WalkForwardEvaluator(
            prices=prices,
            periods=one_period,
            transaction_cost=0.003,
            rebalance_frequency=10,
            universe_type="oil_microcap",
            recency_half_life=3,
            use_fitness_v2=True,
            tradeable_tickers=["T0", "T1"],
        )
        ev2 = WalkForwardEvaluator.from_config(ev.to_config())
        assert ev2.transaction_cost == ev.transaction_cost
        assert ev2.rebalance_frequency == ev.rebalance_frequency
        assert ev2.universe_type == ev.universe_type
        assert ev2.recency_half_life == ev.recency_half_life
        assert ev2.use_fitness_v2 == ev.use_fitness_v2
        assert ev2.tradeable_tickers == ev.tradeable_tickers
        assert len(ev2.periods) == len(ev.periods)

    def test_expanding_window_serialized(self, prices, one_period):
        ev = WalkForwardEvaluator(prices=prices, periods=one_period, expanding_window=True)
        config = ev.to_config()
        assert config["expanding_window"] is True
        ev2 = WalkForwardEvaluator.from_config(config)
        assert ev2.expanding_window is True

    def test_expanding_window_default_false(self, prices):
        ev = WalkForwardEvaluator(prices=prices, periods=[])
        assert ev.expanding_window is False

    def test_deserialized_evaluator_produces_same_fitness(self, prices, strategy, two_periods):
        ev = WalkForwardEvaluator(prices=prices, periods=two_periods)
        r1 = ev.evaluate_strategy(strategy)
        ev2 = WalkForwardEvaluator.from_config(ev.to_config())
        r2 = ev2.evaluate_strategy(strategy)
        assert abs(r1.total - r2.total) < 1e-9


# ─── Rebalance frequency ──────────────────────────────────────────────────────

class TestRebalanceFrequency:
    @pytest.mark.parametrize("freq", [1, 5, 21])
    def test_different_frequencies_all_complete(self, prices, strategy, one_period, freq):
        ev = WalkForwardEvaluator(prices=prices, periods=one_period, rebalance_frequency=freq)
        result = ev.evaluate_strategy(strategy)
        assert isinstance(result, FitnessResult)
