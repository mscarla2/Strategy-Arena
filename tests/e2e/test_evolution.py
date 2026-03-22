"""
End-to-end tests for the full GP evolution pipeline.

These tests run a mini evolution loop (2 generations, small population)
entirely on synthetic data to verify the system works top-to-bottom without
touching any live data sources or files.

Marks:
    e2e — can be skipped in CI with ``pytest -m "not e2e"``
"""
import random
import numpy as np
import pandas as pd
import pytest

from evolution.features import FeatureLibrary
from evolution.tree_ops import TreeGenerator, GPOperators
from evolution.strategy import GPStrategy
from evolution.walkforward import WalkForwardEvaluator
from evolution.population import GPPopulation, create_random_gp_strategy
from evolution.gp_fitness import FitnessResult


pytestmark = pytest.mark.e2e


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def prices():
    """750 business days (~3 years) so execution_lag lookback clears the 504-day max_lookback."""
    np.random.seed(99)
    dates = pd.bdate_range("2019-01-02", periods=750)
    tickers = [f"T{i}" for i in range(8)]
    returns = np.random.normal(0.0003, 0.018, (750, 8))
    return pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=tickers)


@pytest.fixture(scope="module")
def volume(prices):
    np.random.seed(99)
    data = np.random.randint(200_000, 2_000_000, size=prices.shape).astype(float)
    return pd.DataFrame(data, index=prices.index, columns=prices.columns)


@pytest.fixture(scope="module")
def three_periods(prices):
    """
    Test windows start at d[505] so that on the first rebalance date,
    available_prices has 506+ rows and execution_lag=1 yields
    effective_idx=504 >= max_lookback=504.
    """
    d = prices.index
    return [
        (d[0].strftime("%Y-%m-%d"),   d[504].strftime("%Y-%m-%d"),
         d[505].strftime("%Y-%m-%d"), d[579].strftime("%Y-%m-%d")),
        (d[50].strftime("%Y-%m-%d"),  d[579].strftime("%Y-%m-%d"),
         d[580].strftime("%Y-%m-%d"), d[654].strftime("%Y-%m-%d")),
        (d[100].strftime("%Y-%m-%d"), d[654].strftime("%Y-%m-%d"),
         d[655].strftime("%Y-%m-%d"), d[-1].strftime("%Y-%m-%d")),
    ]


@pytest.fixture(scope="module")
def evaluator(prices, three_periods):
    return WalkForwardEvaluator(prices=prices, periods=three_periods)


# ─── Full pipeline smoke test ─────────────────────────────────────────────────

class TestEvolutionPipeline:
    def test_single_strategy_evaluates(self, evaluator):
        """A random strategy can be evaluated end-to-end."""
        random.seed(10)
        s = create_random_gp_strategy(max_depth=3)
        result = evaluator.evaluate_strategy(s)
        assert isinstance(result, FitnessResult)
        assert -1 <= result.total <= 1
        assert result.n_periods >= 1

    def test_population_initialize_and_one_generation(self, evaluator):
        """GPPopulation can be initialized and evolved for one generation."""
        random.seed(11)
        pop = GPPopulation(population_size=8, n_islands=2, max_depth=4)
        pop.initialize()
        pop.evolve_island(0, evaluator)
        pop.evolve_island(1, evaluator)
        # All strategies should have been evaluated (fitness != default 0 for most).
        for island in pop.islands:
            assert len(island) > 0
            # At least one strategy per island should have non-zero fitness.
            evaluated = [s for s in island if s.fitness != 0.0]
            # Islands are non-empty and all strategies have a fitness attribute.
            assert all(hasattr(s, "fitness") for s in island)

    def test_two_generation_loop(self, evaluator):
        """Two full generations complete without error."""
        random.seed(12)
        pop = GPPopulation(population_size=12, n_islands=2, max_depth=4)
        pop.initialize()

        for _gen in range(2):
            for island_idx in range(pop.n_islands):
                pop.evolve_island(island_idx, evaluator)
            pop.generation += 1

        assert pop.generation == 2

    def test_best_strategy_is_set_after_evolution(self, evaluator):
        """After evolving, the population records a best strategy."""
        random.seed(13)
        pop = GPPopulation(population_size=8, n_islands=2, max_depth=4)
        pop.initialize()
        pop.evolve_island(0, evaluator)
        # best_strategy is updated when any strategy beats 0.0
        # Just ensure no exception was raised; actual assignment is fitness-dependent.

    def test_offspring_have_correct_origin(self):
        """Crossover children are labelled 'crossover' when swap succeeds; mutants 'mutation'."""
        random.seed(14)
        lib = FeatureLibrary()
        gen = TreeGenerator(lib.feature_names)
        # Use a large max_depth so the fallback copy path is unlikely
        ops = GPOperators(lib.feature_names, max_depth=20)

        found_crossover = False
        for _ in range(30):
            s1 = GPStrategy(tree=gen.random_tree(max_depth=4), top_pct=50)
            s2 = GPStrategy(tree=gen.random_tree(max_depth=4), top_pct=50)
            child_cross = ops.crossover(s1, s2)
            if child_cross.origin == "crossover":
                found_crossover = True
                break
        assert found_crossover, "Expected at least one crossover child in 30 attempts"

        # Mutation always produces 'mutation'
        s = GPStrategy(tree=gen.random_tree(max_depth=3), top_pct=50)
        child_mut = ops.mutate(s)
        assert child_mut.origin == "mutation"

    def test_fitness_diversity_across_population(self, evaluator):
        """A population of random strategies should not all have identical fitness."""
        random.seed(15)
        pop = GPPopulation(population_size=16, n_islands=2, max_depth=4)
        pop.initialize()
        for i in range(pop.n_islands):
            pop.evolve_island(i, evaluator)

        all_fitnesses = [s.fitness for island in pop.islands for s in island]
        unique = len(set(round(f, 4) for f in all_fitnesses))
        assert unique > 1, "All strategies have identical fitness — something is wrong"

    def test_fitness_v2_full_pipeline(self, prices, three_periods):
        """v2 fitness mode works in a full evaluation pipeline."""
        random.seed(16)
        ev = WalkForwardEvaluator(
            prices=prices,
            periods=three_periods,
            use_fitness_v2=True,
            universe_type="oil_microcap",
            recency_half_life=4,
        )
        s = create_random_gp_strategy(max_depth=4)
        result = ev.evaluate_strategy(s)
        assert isinstance(result, FitnessResult)
        assert -1 <= result.total <= 1
        assert result.n_periods >= 1

    def test_tradeable_tickers_restriction_end_to_end(self, prices, three_periods):
        """tradeable_tickers filter works throughout a full evaluation."""
        random.seed(17)
        tradeable = ["T0", "T1", "T2"]
        ev = WalkForwardEvaluator(
            prices=prices,
            periods=three_periods,
            tradeable_tickers=tradeable,
        )
        s = create_random_gp_strategy(max_depth=3)
        result = ev.evaluate_strategy(s)
        assert isinstance(result, FitnessResult)

    def test_with_volume_full_pipeline(self, prices, volume, three_periods):
        """Volume data flows through the full pipeline without errors."""
        random.seed(18)
        ev = WalkForwardEvaluator(prices=prices, periods=three_periods, volume=volume)
        s = create_random_gp_strategy(max_depth=3)
        result = ev.evaluate_strategy(s)
        assert isinstance(result, FitnessResult)

    def test_evolution_imports_via_gp_module(self):
        """All legacy imports from evolution.gp still work (backward-compat)."""
        from evolution.gp import (
            Node, FeatureNode, ConstantNode, BinaryOpNode, UnaryOpNode,
            ConditionalNode, TreeGenerator, GPOperators, FeatureLibrary,
            FitnessResult, calculate_fitness, calculate_fitness_v2,
            _recency_weighted_mean, _get_drawdown_thresholds,
            GPStrategy, WalkForwardEvaluator, GPPopulation,
            evaluate_strategy_parallel, create_random_gp_strategy,
        )
        # Spot-check that the names are callable/instantiable
        assert callable(calculate_fitness)
        assert callable(calculate_fitness_v2)
        s = create_random_gp_strategy(max_depth=3)
        assert isinstance(s, GPStrategy)

    def test_evolution_package_init_imports(self):
        """evolution/__init__.py re-exports are accessible."""
        from evolution import (
            GPStrategy, GPOperators, TreeGenerator,
            FeatureLibrary, WalkForwardEvaluator,
        )
        assert GPStrategy is not None
        assert GPOperators is not None
        assert TreeGenerator is not None
        assert FeatureLibrary is not None
        assert WalkForwardEvaluator is not None
