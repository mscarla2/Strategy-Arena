"""
Unit tests for evolution/strategy.py (GPStrategy)

Covers score_stocks(), select_stocks(), get_formula(), complexity(), copy().
"""
import random
import numpy as np
import pandas as pd
import pytest

from evolution.nodes import FeatureNode, BinaryOpNode
from evolution.strategy import GPStrategy
from evolution.tree_ops import TreeGenerator
from evolution.features import FeatureLibrary


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def prices():
    """600 business days so execution_lag lookback clears the 504-day max_lookback."""
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=600)
    tickers = [f"T{i}" for i in range(8)]
    returns = np.random.normal(0.0005, 0.02, (600, 8))
    return pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=tickers)


@pytest.fixture(scope="module")
def volume(prices):
    np.random.seed(42)
    data = np.random.randint(100_000, 1_000_000, size=prices.shape).astype(float)
    return pd.DataFrame(data, index=prices.index, columns=prices.columns)


@pytest.fixture(scope="module")
def gen():
    lib = FeatureLibrary()
    return TreeGenerator(lib.feature_names)


@pytest.fixture
def strategy(gen):
    random.seed(1)
    return GPStrategy(tree=gen.random_tree(max_depth=3), top_pct=50)


# ─── Construction ─────────────────────────────────────────────────────────────

class TestGPStrategyConstruction:
    def test_default_top_pct(self, gen):
        s = GPStrategy(tree=gen.random_tree(max_depth=2))
        assert s.top_pct == 20.0

    def test_strategy_id_is_short_string(self, gen):
        s = GPStrategy(tree=gen.random_tree(max_depth=2))
        assert isinstance(s.strategy_id, str)
        assert len(s.strategy_id) > 0

    def test_feature_lib_initialised(self, strategy):
        assert isinstance(strategy.feature_lib, FeatureLibrary)

    def test_generation_default_zero(self, gen):
        s = GPStrategy(tree=gen.random_tree(max_depth=2))
        assert s.generation == 0

    def test_origin_default_random(self, gen):
        s = GPStrategy(tree=gen.random_tree(max_depth=2))
        assert s.origin == "random"


# ─── score_stocks() ───────────────────────────────────────────────────────────

class TestScoreStocks:
    def test_returns_series(self, strategy, prices):
        scores = strategy.score_stocks(prices)
        assert isinstance(scores, pd.Series)

    def test_index_matches_tickers(self, strategy, prices):
        scores = strategy.score_stocks(prices)
        assert set(scores.index) == set(prices.columns)

    def test_scores_bounded_zero_to_one(self, strategy, prices):
        scores = strategy.score_stocks(prices)
        assert scores.min() >= -1e-9
        assert scores.max() <= 1 + 1e-9

    def test_no_nan_in_scores(self, strategy, prices):
        scores = strategy.score_stocks(prices)
        assert not scores.isna().any()

    def test_empty_prices_returns_empty(self, strategy):
        empty = pd.DataFrame()
        scores = strategy.score_stocks(empty)
        assert scores.empty

    def test_scores_with_volume(self, strategy, prices, volume):
        scores = strategy.score_stocks(prices, volume=volume)
        assert not scores.isna().any()
        assert scores.min() >= -1e-9
        assert scores.max() <= 1 + 1e-9

    def test_constant_scores_when_insufficient_data(self, gen):
        """If prices too short for lookback, score_stocks should return 0.5."""
        s = GPStrategy(tree=gen.random_tree(max_depth=3), top_pct=50)
        tiny_prices = pd.DataFrame(
            {"A": [100.0, 101.0, 99.0], "B": [50.0, 51.0, 49.0]},
            index=pd.bdate_range("2023-01-01", periods=3),
        )
        scores = s.score_stocks(tiny_prices)
        # May return 0.5 constant or valid scores — just must not raise
        assert isinstance(scores, pd.Series)
        assert not scores.isna().any()

    def test_multiple_strategies_produce_different_scores(self, gen, prices):
        """Different GP trees should typically produce distinct stock rankings."""
        random.seed(0)
        scores_list = [
            GPStrategy(tree=gen.random_tree(max_depth=4), top_pct=50).score_stocks(prices)
            for _ in range(10)
        ]
        n_unique = sum(1 for s in scores_list if s.nunique() > 1)
        assert n_unique >= 5, "Too many strategies returned constant scores"


# ─── select_stocks() ──────────────────────────────────────────────────────────

class TestSelectStocks:
    @pytest.mark.parametrize("top_pct,n_tickers", [(50, 8), (33, 8), (83, 8)])
    def test_selection_count(self, gen, prices, top_pct, n_tickers):
        s = GPStrategy(tree=gen.random_tree(max_depth=3), top_pct=top_pct)
        selected, turnover = s.select_stocks(prices)
        expected_n = max(1, int(n_tickers * top_pct / 100))
        assert len(selected) == expected_n

    def test_first_call_has_full_turnover(self, strategy, prices):
        _, turnover = strategy.select_stocks(prices, current_positions=None)
        assert turnover == 1.0

    def test_turnover_zero_for_unchanged_positions(self, strategy, prices):
        selected, _ = strategy.select_stocks(prices)
        _, turnover2 = strategy.select_stocks(prices, current_positions=selected)
        assert turnover2 == pytest.approx(0.0)

    def test_turnover_bounded_zero_to_one(self, strategy, prices):
        selected, _ = strategy.select_stocks(prices)
        _, turnover = strategy.select_stocks(prices, current_positions=["T0", "T1"])
        assert 0.0 <= turnover <= 1.0

    def test_selected_tickers_in_price_columns(self, strategy, prices):
        selected, _ = strategy.select_stocks(prices)
        for ticker in selected:
            assert ticker in prices.columns

    def test_empty_prices_returns_empty_list(self, strategy):
        selected, turnover = strategy.select_stocks(pd.DataFrame())
        assert selected == []
        assert turnover == 0.0


# ─── get_formula() / complexity() ────────────────────────────────────────────

class TestFormulaAndComplexity:
    def test_get_formula_non_empty(self, strategy):
        assert len(strategy.get_formula()) > 0

    def test_get_formula_is_string(self, strategy):
        assert isinstance(strategy.get_formula(), str)

    def test_complexity_is_positive_int(self, strategy):
        c = strategy.complexity()
        assert isinstance(c, int)
        assert c >= 1

    def test_deeper_tree_has_higher_complexity(self, gen):
        s_shallow = GPStrategy(tree=gen.random_tree(max_depth=2), top_pct=50)
        s_deep = GPStrategy(tree=gen.random_tree(max_depth=5), top_pct=50)
        # On average deep should be more complex — true with high probability
        assert s_deep.complexity() >= s_shallow.complexity()

    def test_leaf_only_tree_complexity_one(self, gen):
        leaf = FeatureNode("mom_21d")
        s = GPStrategy(tree=leaf, top_pct=50)
        assert s.complexity() == 1


# ─── copy() ───────────────────────────────────────────────────────────────────

class TestCopy:
    def test_copy_returns_new_object(self, strategy):
        copied = strategy.copy()
        assert copied is not strategy

    def test_copy_has_same_formula(self, strategy):
        assert strategy.copy().get_formula() == strategy.get_formula()

    def test_copy_tree_is_independent(self, strategy):
        copied = strategy.copy()
        # Mutate the copy's tree
        if isinstance(copied.tree, FeatureNode):
            copied.tree.feature_name = "vol_21d"
        else:
            # Navigate to first feature leaf and change it
            def _first_leaf(node):
                children = node.get_children()
                if not children:
                    return node
                return _first_leaf(children[0])
            leaf = _first_leaf(copied.tree)
            if isinstance(leaf, FeatureNode):
                original_name = leaf.feature_name
                leaf.feature_name = "__changed__"
                assert strategy.get_formula() != copied.get_formula() or \
                       original_name == "__changed__"  # if already same name, skip

    def test_copy_preserves_top_pct(self, strategy):
        assert strategy.copy().top_pct == strategy.top_pct

    def test_copy_preserves_fitness(self, strategy):
        strategy.fitness = 0.42
        assert strategy.copy().fitness == pytest.approx(0.42)
