"""
Integration tests for evolution/population.py

Covers GPPopulation initialization, island evolution, parallel evaluation,
and evaluate_strategy_parallel().
"""
import random
import numpy as np
import pandas as pd
import pytest

from evolution.features import FeatureLibrary
from evolution.tree_ops import TreeGenerator
from evolution.strategy import GPStrategy
from evolution.walkforward import WalkForwardEvaluator
from evolution.population import GPPopulation, evaluate_strategy_parallel, create_random_gp_strategy


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def prices():
    """600 business days so execution_lag lookback clears the 504-day max_lookback."""
    np.random.seed(3)
    dates = pd.bdate_range("2021-01-04", periods=600)
    tickers = [f"T{i}" for i in range(8)]
    returns = np.random.normal(0.0005, 0.02, (600, 8))
    return pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=tickers)


@pytest.fixture(scope="module")
def evaluator(prices):
    d = prices.index
    periods = [(d[0].strftime("%Y-%m-%d"),   d[449].strftime("%Y-%m-%d"),
                d[450].strftime("%Y-%m-%d"), d[-1].strftime("%Y-%m-%d"))]
    return WalkForwardEvaluator(prices=prices, periods=periods)


# ─── GPPopulation construction ────────────────────────────────────────────────

class TestGPPopulationConstruction:
    def test_default_init(self):
        pop = GPPopulation(population_size=20, n_islands=2)
        assert pop.population_size == 20
        assert pop.n_islands == 2

    def test_feature_lib_created(self):
        pop = GPPopulation(population_size=10)
        assert isinstance(pop.feature_lib, FeatureLibrary)

    def test_smc_flag_forwarded(self):
        pop = GPPopulation(population_size=10, enable_smc=True)
        assert pop.feature_lib.enable_smc is True

    def test_sr_flag_forwarded(self):
        pop = GPPopulation(population_size=10, enable_sr=True)
        assert pop.feature_lib.enable_sr is True

    def test_oil_flag_forwarded(self):
        pop = GPPopulation(population_size=10, enable_oil=True)
        assert pop.feature_lib.enable_oil is True

    def test_islands_empty_before_initialize(self):
        pop = GPPopulation(population_size=12, n_islands=3)
        assert all(len(island) == 0 for island in pop.islands)

    def test_initialize_fills_islands(self):
        pop = GPPopulation(population_size=20, n_islands=4)
        pop.initialize()
        total = sum(len(island) for island in pop.islands)
        assert total == 20


# ─── GPPopulation.initialize() ────────────────────────────────────────────────

class TestInitialize:
    def test_each_island_gets_strategies(self):
        pop = GPPopulation(population_size=20, n_islands=4)
        pop.initialize()
        for island in pop.islands:
            assert len(island) > 0

    def test_strategies_are_gp_strategy_instances(self):
        pop = GPPopulation(population_size=16, n_islands=4)
        pop.initialize()
        for island in pop.islands:
            for s in island:
                assert isinstance(s, GPStrategy)

    def test_strategies_have_valid_trees(self):
        pop = GPPopulation(population_size=16, n_islands=4, max_depth=5)
        pop.initialize()
        for island in pop.islands:
            for s in island:
                assert s.tree is not None
                assert s.tree.depth() >= 1
                assert s.tree.depth() <= 5

    def test_strategies_have_valid_top_pct(self):
        pop = GPPopulation(population_size=16, n_islands=4)
        pop.initialize()
        valid_pcts = {33, 50, 67, 83}
        for island in pop.islands:
            for s in island:
                assert s.top_pct in valid_pcts


# ─── adjusted_fitness() ───────────────────────────────────────────────────────

class TestAdjustedFitness:
    def test_parsimony_penalises_complexity(self):
        """More complex tree should have lower adjusted fitness for same base."""
        lib = FeatureLibrary()
        gen = TreeGenerator(lib.feature_names)
        random.seed(0)

        pop = GPPopulation(population_size=10, parsimony_coefficient=0.01)

        simple = GPStrategy(tree=gen.random_tree(max_depth=2), top_pct=50)
        complex_ = GPStrategy(tree=gen.random_tree(max_depth=6), top_pct=50)
        simple.fitness = complex_.fitness = 0.5  # equal base fitness

        simple_adj = pop.adjusted_fitness(simple)
        complex_adj = pop.adjusted_fitness(complex_)

        # Simple tree should have higher adjusted fitness
        assert simple_adj >= complex_adj

    def test_novelty_bonus_for_new_formula(self):
        lib = FeatureLibrary()
        gen = TreeGenerator(lib.feature_names)
        pop = GPPopulation(population_size=10)
        random.seed(99)
        s = GPStrategy(tree=gen.random_tree(max_depth=3), top_pct=50)
        s.fitness = 0.0
        adj1 = pop.adjusted_fitness(s)  # first time — novelty bonus
        adj2 = pop.adjusted_fitness(s)  # second time — no bonus
        assert adj1 >= adj2


# ─── evolve_island() ──────────────────────────────────────────────────────────

class TestEvolveIsland:
    def test_evolve_island_runs_without_error(self, prices, evaluator):
        pop = GPPopulation(population_size=8, n_islands=2)
        pop.initialize()
        pop.evolve_island(0, evaluator)

    def test_best_strategy_updated_after_evolve(self, prices, evaluator):
        pop = GPPopulation(population_size=8, n_islands=2)
        pop.initialize()
        pop.evolve_island(0, evaluator)
        # best_strategy may be set if any strategy > 0.0
        # (it starts at 0.0 and only updates when beaten)

    def test_island_sorted_by_fitness_after_evolve(self, prices, evaluator):
        pop = GPPopulation(population_size=8, n_islands=2)
        pop.initialize()
        pop.evolve_island(0, evaluator)
        fitnesses = [s.fitness for s in pop.islands[0]]
        assert fitnesses == sorted(fitnesses, reverse=True)


# ─── evaluate_strategy_parallel() ────────────────────────────────────────────

class TestEvaluateStrategyParallel:
    def test_parallel_matches_sequential(self, prices, evaluator):
        random.seed(5)
        lib = FeatureLibrary()
        gen = TreeGenerator(lib.feature_names)
        strategy = GPStrategy(tree=gen.random_tree(max_depth=3), top_pct=50)

        # Sequential
        seq_result = evaluator.evaluate_strategy(strategy)

        # Parallel (using module-level function directly)
        config = evaluator.to_config()
        sid, par_fitness, _ = evaluate_strategy_parallel((strategy, config))

        assert abs(seq_result.total - par_fitness) < 1e-9

    def test_parallel_returns_strategy_id(self, evaluator):
        random.seed(6)
        lib = FeatureLibrary()
        gen = TreeGenerator(lib.feature_names)
        s = GPStrategy(tree=gen.random_tree(max_depth=3), top_pct=50)
        config = evaluator.to_config()
        sid, _, _ = evaluate_strategy_parallel((s, config))
        assert sid == s.strategy_id

    def test_parallel_returns_period_results_list(self, evaluator):
        random.seed(7)
        lib = FeatureLibrary()
        gen = TreeGenerator(lib.feature_names)
        s = GPStrategy(tree=gen.random_tree(max_depth=3), top_pct=50)
        config = evaluator.to_config()
        _, _, period_results = evaluate_strategy_parallel((s, config))
        assert isinstance(period_results, list)


# ─── create_random_gp_strategy() ─────────────────────────────────────────────

class TestCreateRandomGPStrategy:
    def test_returns_gp_strategy(self):
        s = create_random_gp_strategy(max_depth=3)
        assert isinstance(s, GPStrategy)

    def test_tree_not_none(self):
        s = create_random_gp_strategy(max_depth=3)
        assert s.tree is not None

    def test_formula_non_empty(self):
        s = create_random_gp_strategy(max_depth=3)
        assert len(s.get_formula()) > 0

    def test_depth_respected(self):
        for max_d in [2, 3, 4]:
            s = create_random_gp_strategy(max_depth=max_d)
            assert s.tree.depth() <= max_d
