"""
Unit tests for evolution/nodes.py

Covers every node class and all operations for full behavior coverage.
"""
import numpy as np
import pandas as pd
import pytest

from evolution.nodes import (
    Node,
    FeatureNode,
    ConstantNode,
    BinaryOpNode,
    UnaryOpNode,
    ConditionalNode,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _series(values, tickers=None):
    idx = tickers or [f"T{i}" for i in range(len(values))]
    return pd.Series(values, index=idx, dtype=float)


def _features(n=4):
    return {
        "a": _series([0.1, 0.5, 0.9, 0.3]),
        "b": _series([0.8, 0.2, 0.4, 0.6]),
    }


# ─── FeatureNode ─────────────────────────────────────────────────────────────

class TestFeatureNode:
    def test_evaluate_known_feature(self):
        feats = _features()
        node = FeatureNode("a")
        result = node.evaluate(feats)
        pd.testing.assert_series_equal(result, feats["a"])

    def test_evaluate_unknown_feature_returns_zeros(self):
        feats = _features()
        node = FeatureNode("MISSING")
        result = node.evaluate(feats)
        assert (result == 0.0).all()

    def test_to_string(self):
        assert FeatureNode("momentum_21d").to_string() == "momentum_21d"

    def test_copy_is_independent(self):
        node = FeatureNode("a")
        copied = node.copy()
        copied.feature_name = "b"
        assert node.feature_name == "a"

    def test_size_is_one(self):
        assert FeatureNode("x").size() == 1

    def test_depth_is_one(self):
        assert FeatureNode("x").depth() == 1

    def test_get_children_empty(self):
        assert FeatureNode("x").get_children() == []

    def test_set_children_no_op(self):
        node = FeatureNode("x")
        node.set_children([ConstantNode(1.0)])  # should not raise


# ─── ConstantNode ─────────────────────────────────────────────────────────────

class TestConstantNode:
    def test_evaluate_broadcasts_to_index(self):
        feats = _features()
        node = ConstantNode(3.14)
        result = node.evaluate(feats)
        assert (result == 3.14).all()
        assert len(result) == len(feats["a"])

    def test_to_string_precision(self):
        s = ConstantNode(1.5).to_string()
        assert "1.5" in s

    def test_copy_is_independent(self):
        node = ConstantNode(7.0)
        copied = node.copy()
        copied.value = 99.0
        assert node.value == 7.0

    def test_size_and_depth(self):
        node = ConstantNode(0.0)
        assert node.size() == 1
        assert node.depth() == 1


# ─── BinaryOpNode ─────────────────────────────────────────────────────────────

class TestBinaryOpNode:
    @pytest.fixture
    def feats(self):
        return {
            "a": _series([1.0, 2.0, 3.0, 4.0]),
            "b": _series([4.0, 3.0, 2.0, 1.0]),
        }

    def _binary(self, op, fa="a", fb="b"):
        return BinaryOpNode(op, FeatureNode(fa), FeatureNode(fb))

    def test_add(self, feats):
        result = self._binary("add").evaluate(feats)
        expected = feats["a"] + feats["b"]
        pd.testing.assert_series_equal(result, expected)

    def test_sub(self, feats):
        result = self._binary("sub").evaluate(feats)
        expected = feats["a"] - feats["b"]
        pd.testing.assert_series_equal(result, expected)

    def test_mul(self, feats):
        result = self._binary("mul").evaluate(feats)
        expected = feats["a"] * feats["b"]
        pd.testing.assert_series_equal(result, expected)

    def test_div_by_zero_returns_zeros_not_inf(self, feats):
        feats_zero = {"a": _series([1.0, 2.0, 3.0, 4.0]), "b": _series([0.0, 0.0, 0.0, 0.0])}
        result = self._binary("div").evaluate(feats_zero)
        assert not result.isin([np.inf, -np.inf]).any()
        assert not result.isna().any()

    def test_max(self, feats):
        result = self._binary("max").evaluate(feats)
        expected = np.maximum(feats["a"], feats["b"])
        np.testing.assert_array_almost_equal(result.values, expected)

    def test_min(self, feats):
        result = self._binary("min").evaluate(feats)
        expected = np.minimum(feats["a"], feats["b"])
        np.testing.assert_array_almost_equal(result.values, expected)

    def test_avg(self, feats):
        result = self._binary("avg").evaluate(feats)
        expected = (feats["a"] + feats["b"]) / 2
        pd.testing.assert_series_equal(result, expected)

    def test_to_string_format(self, feats):
        node = self._binary("add")
        s = node.to_string()
        assert "add" in s
        assert "a" in s and "b" in s

    def test_size_counts_all_nodes(self, feats):
        node = self._binary("add")
        assert node.size() == 3  # 1 binary + 2 leaves

    def test_depth(self, feats):
        node = self._binary("mul")
        assert node.depth() == 2

    def test_copy_deep(self):
        node = BinaryOpNode("add", FeatureNode("a"), FeatureNode("b"))
        copied = node.copy()
        copied.op = "sub"
        copied.left.feature_name = "x"
        assert node.op == "add"
        assert node.left.feature_name == "a"

    def test_set_children(self):
        node = BinaryOpNode("add", FeatureNode("a"), FeatureNode("b"))
        node.set_children([FeatureNode("c"), FeatureNode("d")])
        assert node.left.feature_name == "c"
        assert node.right.feature_name == "d"


# ─── UnaryOpNode ──────────────────────────────────────────────────────────────

class TestUnaryOpNode:
    @pytest.fixture
    def feats(self):
        return {"a": _series([-2.0, -1.0, 0.0, 1.0, 2.0])}

    def test_neg(self, feats):
        node = UnaryOpNode("neg", FeatureNode("a"))
        result = node.evaluate(feats)
        pd.testing.assert_series_equal(result, -feats["a"])

    def test_abs(self, feats):
        node = UnaryOpNode("abs", FeatureNode("a"))
        result = node.evaluate(feats)
        assert (result >= 0).all()

    def test_sign(self, feats):
        node = UnaryOpNode("sign", FeatureNode("a"))
        result = node.evaluate(feats)
        assert set(result.unique()).issubset({-1.0, 0.0, 1.0})

    def test_sqrt_of_positive(self):
        feats = {"a": _series([1.0, 4.0, 9.0, 16.0])}
        node = UnaryOpNode("sqrt", FeatureNode("a"))
        result = node.evaluate(feats)
        np.testing.assert_array_almost_equal(result.values, [1.0, 2.0, 3.0, 4.0])

    def test_sqrt_of_negative_uses_abs(self):
        feats = {"a": _series([-4.0, -9.0])}
        node = UnaryOpNode("sqrt", FeatureNode("a"))
        result = node.evaluate(feats)
        assert (result >= 0).all()
        assert not result.isna().any()

    def test_square(self, feats):
        node = UnaryOpNode("square", FeatureNode("a"))
        result = node.evaluate(feats)
        expected = feats["a"] ** 2
        pd.testing.assert_series_equal(result, expected)

    def test_inv_no_nan_inf(self, feats):
        node = UnaryOpNode("inv", FeatureNode("a"))
        result = node.evaluate(feats)
        assert not result.isin([np.inf, -np.inf]).any()
        assert not result.isna().any()

    def test_log_no_nan(self):
        feats = {"a": _series([1.0, 10.0, 100.0])}
        node = UnaryOpNode("log", FeatureNode("a"))
        result = node.evaluate(feats)
        assert not result.isna().any()

    def test_sigmoid_bounds(self, feats):
        node = UnaryOpNode("sigmoid", FeatureNode("a"))
        result = node.evaluate(feats)
        assert (result >= 0).all() and (result <= 1).all()

    def test_tanh_bounds(self, feats):
        node = UnaryOpNode("tanh", FeatureNode("a"))
        result = node.evaluate(feats)
        assert (result >= -1).all() and (result <= 1).all()

    def test_rank_pct(self):
        feats = {"a": _series([10.0, 30.0, 20.0, 40.0])}
        node = UnaryOpNode("rank", FeatureNode("a"))
        result = node.evaluate(feats)
        assert result.min() > 0
        assert result.max() <= 1.0

    def test_zscore_no_nan(self):
        feats = {"a": _series([1.0, 2.0, 3.0, 4.0, 5.0])}
        node = UnaryOpNode("zscore", FeatureNode("a"))
        result = node.evaluate(feats)
        assert not result.isna().any()

    def test_size_and_depth(self, feats):
        node = UnaryOpNode("neg", FeatureNode("a"))
        assert node.size() == 2
        assert node.depth() == 2


# ─── ConditionalNode ──────────────────────────────────────────────────────────

class TestConditionalNode:
    @pytest.fixture
    def feats(self):
        return {
            "cond": _series([1.0, -1.0, 1.0, -1.0]),
            "yes":  _series([10.0, 10.0, 10.0, 10.0]),
            "no":   _series([20.0, 20.0, 20.0, 20.0]),
        }

    def test_correct_branch_selection(self, feats):
        node = ConditionalNode(FeatureNode("cond"), FeatureNode("yes"), FeatureNode("no"))
        result = node.evaluate(feats)
        # cond > 0 → yes (10); cond <= 0 → no (20)
        assert result.iloc[0] == 10.0
        assert result.iloc[1] == 20.0
        assert result.iloc[2] == 10.0
        assert result.iloc[3] == 20.0

    def test_no_nan_in_result(self, feats):
        node = ConditionalNode(FeatureNode("cond"), FeatureNode("yes"), FeatureNode("no"))
        result = node.evaluate(feats)
        assert not result.isna().any()

    def test_to_string_contains_if(self, feats):
        node = ConditionalNode(FeatureNode("cond"), FeatureNode("yes"), FeatureNode("no"))
        s = node.to_string()
        assert "if" in s or ">" in s

    def test_size(self, feats):
        node = ConditionalNode(FeatureNode("cond"), FeatureNode("yes"), FeatureNode("no"))
        assert node.size() == 4

    def test_depth(self, feats):
        node = ConditionalNode(FeatureNode("cond"), FeatureNode("yes"), FeatureNode("no"))
        assert node.depth() == 2

    def test_copy_independent(self, feats):
        node = ConditionalNode(FeatureNode("cond"), FeatureNode("yes"), FeatureNode("no"))
        copied = node.copy()
        copied.condition.feature_name = "CHANGED"
        assert node.condition.feature_name == "cond"

    def test_set_children(self, feats):
        node = ConditionalNode(FeatureNode("a"), FeatureNode("b"), FeatureNode("c"))
        node.set_children([FeatureNode("x"), FeatureNode("y"), FeatureNode("z")])
        assert node.condition.feature_name == "x"
        assert node.if_true.feature_name == "y"
        assert node.if_false.feature_name == "z"


# ─── Nested tree ──────────────────────────────────────────────────────────────

class TestNestedTree:
    """Test depth, size and evaluation on a manually constructed nested tree."""

    def test_nested_size_and_depth(self):
        # (a + b) * c  → size=5, depth=3
        tree = BinaryOpNode(
            "mul",
            BinaryOpNode("add", FeatureNode("a"), FeatureNode("b")),
            FeatureNode("c"),
        )
        assert tree.size() == 5
        assert tree.depth() == 3

    def test_nested_evaluate(self):
        feats = {
            "a": _series([1.0, 2.0]),
            "b": _series([3.0, 4.0]),
            "c": _series([2.0, 3.0]),
        }
        tree = BinaryOpNode(
            "mul",
            BinaryOpNode("add", FeatureNode("a"), FeatureNode("b")),
            FeatureNode("c"),
        )
        result = tree.evaluate(feats)
        expected = (feats["a"] + feats["b"]) * feats["c"]
        pd.testing.assert_series_equal(result, expected)
