"""
Shared pytest fixtures for the Strategy Arena test suite.
All fixtures use synthetic data — no network calls, no disk I/O.
"""
import numpy as np
import pandas as pd
import pytest


# ─── Synthetic market data ────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def synthetic_prices():
    """300 business days × 10 tickers of synthetic price data."""
    np.random.seed(42)
    dates = pd.bdate_range(start="2023-01-01", periods=300)
    tickers = [f"T{i}" for i in range(10)]
    returns = np.random.normal(0.0005, 0.02, (300, 10))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


@pytest.fixture(scope="session")
def synthetic_volume(synthetic_prices):
    """Matching synthetic volume DataFrame."""
    np.random.seed(42)
    vol = np.random.randint(100_000, 1_000_000, size=synthetic_prices.shape).astype(float)
    return pd.DataFrame(
        vol, index=synthetic_prices.index, columns=synthetic_prices.columns
    )


@pytest.fixture(scope="session")
def small_prices():
    """100 business days × 5 tickers — minimal data for fast unit tests."""
    np.random.seed(7)
    dates = pd.bdate_range(start="2023-01-01", periods=100)
    tickers = [f"S{i}" for i in range(5)]
    returns = np.random.normal(0.0003, 0.015, (100, 5))
    prices = 50 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


# ─── Evolution fixtures ───────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def feature_lib():
    """Default FeatureLibrary (no optional feature sets enabled)."""
    from evolution.features import FeatureLibrary
    return FeatureLibrary()


@pytest.fixture(scope="session")
def tree_generator(feature_lib):
    """TreeGenerator seeded from the default feature library."""
    from evolution.tree_ops import TreeGenerator
    return TreeGenerator(feature_lib.feature_names)


@pytest.fixture
def simple_strategy(tree_generator):
    """A single deterministic GPStrategy for reuse in tests."""
    import random
    from evolution.strategy import GPStrategy
    random.seed(0)
    return GPStrategy(tree=tree_generator.random_tree(max_depth=3), top_pct=50)


# ─── Walk-forward period helpers ──────────────────────────────────────────────

@pytest.fixture
def one_wf_period():
    """A single (train_start, train_end, test_start, test_end) tuple list."""
    return [("2023-01-02", "2023-06-30", "2023-07-03", "2023-09-29")]


@pytest.fixture
def two_wf_periods():
    return [
        ("2023-01-02", "2023-06-30", "2023-07-03", "2023-09-29"),
        ("2023-04-03", "2023-09-29", "2023-10-02", "2023-12-29"),
    ]
