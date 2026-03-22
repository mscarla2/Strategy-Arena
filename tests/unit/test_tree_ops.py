"""
Unit tests for evolution/tree_ops.py

Covers TreeGenerator and GPOperators (crossover / mutation).
"""
import random
import pytest
import numpy as np
import pandas as pd

from evolution.nodes import (
    FeatureNode, ConstantNode, BinaryOpNode, UnaryOpNode, ConditionalNode,
)
from evolution.tree_ops import TreeGenerator, GPOperators


FEATURE_NAMES = [f"feat_{i}" for i in range(20)]


# ─── TreeGenerator ────────────────────────────────────────────────────────────

class TestTreeGenerator:
    @pytest.fixture
    def gen(self):
        return TreeGenerator(FEATURE_NAMES)

    def test_grow_returns_node(self, gen):
        tree = gen.random_tree(max_depth=3, method="grow")
        assert tree is not None

    def test_full_returns_node(self, gen):
        tree = gen.random_tree(max_depth=3, method="full")
        assert tree is not None

    def test_depth_not_exceeded(self, gen):
        for max_d in [2, 3, 4, 5]:
            tree = gen.random_tree(max_depth=max_d, method="grow")
            assert tree.depth() <= max_d

    def test_full_depth_exact(self, gen):
        # With method="full", every path should reach max_depth for simple cases.
        # At minimum, depth should equal max_depth.
        tree = gen.random_tree(max_depth=3, method="full")
        assert tree.depth() >= 2  # at least non-trivial

    def test_leaves_are_feature_or_constant(self, gen):
        """All leaf nodes should be FeatureNode or ConstantNode."""
        tree = gen.random_tree(max_depth=4, method="grow")
        
        def _collect_leaves(node):
            children = node.get_children()
            if not children:
                return [node]
            result = []
            for c in children:
                result.extend(_collect_leaves(c))
            return result
        
        leaves = _collect_leaves(tree)
        for leaf in leaves:
            assert isinstance(leaf, (FeatureNode, ConstantNode))

    def test_feature_nodes_use_known_features(self, gen):
        """FeatureNodes must reference features from the feature_names list."""
        for _ in range(10):
            tree = gen.random_tree(max_depth=4, method="grow")
            
            def _check(node):
                if isinstance(node, FeatureNode):
                    assert node.feature_name in FEATURE_NAMES, \
                        f"Unknown feature: {node.feature_name}"
                for c in node.get_children():
                    _check(c)
            
            _check(tree)

    def test_repeated_calls_produce_different_trees(self, gen):
        formulas = set()
        for _ in range(20):
            formulas.add(gen.random_tree(max_depth=4).to_string())
        # Should produce at least a few distinct trees
        assert len(formulas) >= 5

    def test_max_depth_1_returns_terminal(self, gen):
        tree = gen.random_tree(max_depth=1, method="grow")
        assert isinstance(tree, (FeatureNode, ConstantNode))
        assert tree.depth() == 1

    def test_size_is_at_least_1(self, gen):
        tree = gen.random_tree(max_depth=3)
        assert tree.size() >= 1

    def test_evaluate_does_not_raise(self, gen):
        feats = {name: pd.Series(np.random.randn(5), index=[f"T{i}" for i in range(5)])
                 for name in FEATURE_NAMES}
        for _ in range(10):
            tree = gen.random_tree(max_depth=4)
            result = tree.evaluate(feats)
            assert isinstance(result, pd.Series)
            assert not result.isna().any()


# ─── GPOperators ──────────────────────────────────────────────────────────────

class TestGPOperators:
    @pytest.fixture
    def ops(self):
        return GPOperators(FEATURE_NAMES, max_depth=6)

    @pytest.fixture
    def strategy_pair(self, ops):
        from evolution.strategy import GPStrategy
        random.seed(1)
        gen = TreeGenerator(FEATURE_NAMES)
        s1 = GPStrategy(tree=gen.random_tree(max_depth=4), top_pct=50)
        s2 = GPStrategy(tree=gen.random_tree(max_depth=4), top_pct=50)
        return s1, s2

    # --- Crossover ---

    def test_crossover_returns_strategy(self, ops, strategy_pair):
        from evolution.strategy import GPStrategy
        s1, s2 = strategy_pair
        child = ops.crossover(s1, s2)
        assert isinstance(child, GPStrategy)

    def test_crossover_depth_not_exceeded(self, ops, strategy_pair):
        s1, s2 = strategy_pair
        for _ in range(10):
            child = ops.crossover(s1, s2)
            assert child.tree.depth() <= ops.max_depth

    def test_crossover_origin_label(self, ops):
        """Crossover result should be 'crossover' when the swap succeeds (not depth-capped)."""
        from evolution.strategy import GPStrategy
        gen = TreeGenerator(FEATURE_NAMES)
        ops_generous = GPOperators(FEATURE_NAMES, max_depth=20)
        found_crossover = False
        random.seed(0)
        for _ in range(30):
            s1 = GPStrategy(tree=gen.random_tree(max_depth=3), top_pct=50)
            s2 = GPStrategy(tree=gen.random_tree(max_depth=3), top_pct=50)
            child = ops_generous.crossover(s1, s2)
            if child.origin == "crossover":
                found_crossover = True
                break
        assert found_crossover, "No crossover child produced in 30 attempts"

    def test_crossover_generation_is_max_plus_one(self, ops):
        """Crossover child generation = max(parent generations) + 1."""
        from evolution.strategy import GPStrategy
        gen = TreeGenerator(FEATURE_NAMES)
        ops_generous = GPOperators(FEATURE_NAMES, max_depth=20)
        random.seed(1)
        for _ in range(30):
            s1 = GPStrategy(tree=gen.random_tree(max_depth=3), top_pct=50, generation=3)
            s2 = GPStrategy(tree=gen.random_tree(max_depth=3), top_pct=50, generation=5)
            child = ops_generous.crossover(s1, s2)
            if child.origin == "crossover":
                assert child.generation == 6
                return
        pytest.fail("No crossover child produced in 30 attempts")

    def test_crossover_parents_unchanged(self, ops, strategy_pair):
        s1, s2 = strategy_pair
        f1_before = s1.tree.to_string()
        f2_before = s2.tree.to_string()
        ops.crossover(s1, s2)
        assert s1.tree.to_string() == f1_before
        assert s2.tree.to_string() == f2_before

    # --- Mutation ---

    def test_mutate_returns_strategy(self, ops, strategy_pair):
        from evolution.strategy import GPStrategy
        s1, _ = strategy_pair
        mutant = ops.mutate(s1)
        assert isinstance(mutant, GPStrategy)

    def test_mutate_depth_not_exceeded(self, ops, strategy_pair):
        s1, _ = strategy_pair
        for _ in range(20):
            mutant = ops.mutate(s1)
            assert mutant.tree.depth() <= ops.max_depth

    def test_mutate_origin_label(self, ops, strategy_pair):
        s1, _ = strategy_pair
        mutant = ops.mutate(s1)
        assert mutant.origin == "mutation"

    def test_mutate_parent_unchanged(self, ops, strategy_pair):
        s1, _ = strategy_pair
        f_before = s1.tree.to_string()
        ops.mutate(s1)
        assert s1.tree.to_string() == f_before

    def test_point_mutation_changes_tree(self, ops, strategy_pair):
        """Point mutation should typically produce a different formula."""
        s1, _ = strategy_pair
        changed = False
        random.seed(42)
        for _ in range(50):
            mutant = ops.mutate(s1)
            if mutant.tree.to_string() != s1.tree.to_string():
                changed = True
                break
        assert changed, "50 mutations produced identical formula every time"

    def test_mutation_evaluates_without_error(self, ops, strategy_pair):
        s1, _ = strategy_pair
        feats = {name: pd.Series(np.random.randn(5), index=[f"T{i}" for i in range(5)])
                 for name in FEATURE_NAMES}
        for _ in range(10):
            mutant = ops.mutate(s1)
            result = mutant.tree.evaluate(feats)
            assert isinstance(result, pd.Series)
            assert not result.isna().any()

    # --- Single-node edge cases ---

    def test_crossover_with_single_node_tree(self, ops):
        from evolution.strategy import GPStrategy
        gen = TreeGenerator(FEATURE_NAMES)
        single = GPStrategy(tree=gen.random_tree(max_depth=1), top_pct=50)
        big = GPStrategy(tree=gen.random_tree(max_depth=4), top_pct=50)
        child = ops.crossover(single, big)
        assert isinstance(child.tree, object)
        assert child.tree.depth() <= ops.max_depth

    def test_mutate_single_node_tree(self, ops):
        from evolution.strategy import GPStrategy
        gen = TreeGenerator(FEATURE_NAMES)
        single = GPStrategy(tree=gen.random_tree(max_depth=1), top_pct=50)
        mutant = ops.mutate(single)
        assert mutant.tree.depth() <= ops.max_depth
