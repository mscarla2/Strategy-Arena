# tests/test_decomposition.py
"""
Tests for gp.py decomposition, new features, and expanding window.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    """Verify all imports from evolution.gp still work after decomposition."""
    
    def test_import_node_classes(self):
        """Node classes importable from evolution.gp."""
        from evolution.gp import Node, FeatureNode, ConstantNode, BinaryOpNode, UnaryOpNode, ConditionalNode
        assert Node is not None
        assert FeatureNode is not None
        assert ConstantNode is not None
        assert BinaryOpNode is not None
        assert UnaryOpNode is not None
        assert ConditionalNode is not None
    
    def test_import_tree_ops(self):
        """Tree ops importable from evolution.gp."""
        from evolution.gp import TreeGenerator, GPOperators
        assert TreeGenerator is not None
        assert GPOperators is not None
    
    def test_import_feature_library(self):
        """FeatureLibrary importable from evolution.gp."""
        from evolution.gp import FeatureLibrary
        lib = FeatureLibrary()
        assert len(lib.feature_names) > 90  # Should have ~140 features now
    
    def test_import_fitness(self):
        """Fitness functions importable from evolution.gp."""
        from evolution.gp import FitnessResult, calculate_fitness, calculate_fitness_v2
        from evolution.gp import _recency_weighted_mean, _get_drawdown_thresholds
        assert FitnessResult is not None
        assert callable(calculate_fitness)
        assert callable(calculate_fitness_v2)
        assert callable(_recency_weighted_mean)
        assert callable(_get_drawdown_thresholds)
    
    def test_import_strategy(self):
        """GPStrategy importable from evolution.gp."""
        from evolution.gp import GPStrategy
        assert GPStrategy is not None
    
    def test_import_walkforward(self):
        """WalkForwardEvaluator importable from evolution.gp."""
        from evolution.gp import WalkForwardEvaluator
        assert WalkForwardEvaluator is not None
    
    def test_import_population(self):
        """GPPopulation importable from evolution.gp."""
        from evolution.gp import GPPopulation, evaluate_strategy_parallel, create_random_gp_strategy
        assert GPPopulation is not None
        assert callable(evaluate_strategy_parallel)
        assert callable(create_random_gp_strategy)
    
    def test_evolution_init_imports(self):
        """evolution/__init__.py imports still work."""
        from evolution import GPStrategy, GPOperators, TreeGenerator, FeatureLibrary, WalkForwardEvaluator
        assert GPStrategy is not None
        assert GPOperators is not None
        assert TreeGenerator is not None
        assert FeatureLibrary is not None
        assert WalkForwardEvaluator is not None
    
    def test_direct_module_imports(self):
        """Direct imports from decomposed modules work."""
        from evolution.nodes import Node, FeatureNode
        from evolution.tree_ops import TreeGenerator, GPOperators
        from evolution.features import FeatureLibrary
        from evolution.gp_fitness import FitnessResult, calculate_fitness
        from evolution.strategy import GPStrategy
        from evolution.walkforward import WalkForwardEvaluator
        from evolution.population import GPPopulation
        assert all(x is not None for x in [
            Node, FeatureNode, TreeGenerator, GPOperators,
            FeatureLibrary, FitnessResult, calculate_fitness,
            GPStrategy, WalkForwardEvaluator, GPPopulation
        ])
    
    def test_create_random_strategy(self):
        """create_random_gp_strategy works from both import paths."""
        from evolution.gp import create_random_gp_strategy
        strategy = create_random_gp_strategy(max_depth=3)
        assert strategy is not None
        assert strategy.tree is not None
        assert strategy.get_formula() != ""


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: Create synthetic price data
# ═══════════════════════════════════════════════════════════════════════════════

def _make_prices(n_days=300, n_tickers=10, seed=42):
    """Create synthetic price DataFrame for testing."""
    np.random.seed(seed)
    dates = pd.bdate_range(start='2023-01-01', periods=n_days)
    tickers = [f'T{i}' for i in range(n_tickers)]
    
    # Random walk prices starting at 100
    returns = np.random.normal(0.0005, 0.02, (n_days, n_tickers))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    
    return pd.DataFrame(prices, index=dates, columns=tickers)


def _make_volume(prices, seed=42):
    """Create synthetic volume DataFrame matching prices."""
    np.random.seed(seed)
    volume = np.random.randint(100000, 1000000, size=prices.shape)
    return pd.DataFrame(volume, index=prices.index, columns=prices.columns, dtype=float)


# ═══════════════════════════════════════════════════════════════════════════════
# NEW FEATURE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarketRelativeFeatures:
    """Test market-relative feature computations."""
    
    @pytest.fixture
    def lib(self):
        return __import__('evolution.features', fromlist=['FeatureLibrary']).FeatureLibrary()
    
    @pytest.fixture
    def prices(self):
        return _make_prices(n_days=300, n_tickers=10)
    
    @pytest.fixture
    def volume(self, prices):
        return _make_volume(prices)
    
    def test_market_beta_computation(self, lib, prices):
        """Beta = ~1.0 for equal-weight market proxy, varies for individual stocks."""
        spec = ('market_beta', 63)
        result = lib._compute_feature(prices, None, spec)
        assert isinstance(result, pd.Series)
        assert len(result) == len(prices.columns)
        # Average beta across all stocks should be close to 1.0
        # (since market proxy is equal-weight of all stocks)
        assert abs(result.mean() - 1.0) < 0.5
        # No NaN values
        assert not result.isna().any()
    
    def test_beta_instability(self, lib, prices):
        """Beta instability should be non-negative."""
        spec = ('beta_instability', 63)
        result = lib._compute_feature(prices, None, spec)
        assert isinstance(result, pd.Series)
        assert (result >= 0).all()
    
    def test_idiosyncratic_vol(self, lib, prices):
        """Idiosyncratic vol should be less than total vol for most tickers."""
        idio_spec = ('idiosyncratic_vol', 63)
        total_spec = ('volatility', 63)
        
        idio_vol = lib._compute_feature(prices, None, idio_spec)
        total_vol = lib._compute_feature(prices, None, total_spec)
        
        # Idio vol should be <= total vol for most tickers
        assert (idio_vol <= total_vol * 1.1).sum() >= len(prices.columns) * 0.7
    
    def test_sector_correlation(self, lib, prices):
        """Sector correlation should be between -1 and 1."""
        spec = ('sector_correlation', 63)
        result = lib._compute_feature(prices, None, spec)
        assert (result >= -1.01).all()
        assert (result <= 1.01).all()
    
    def test_universe_dispersion(self, lib, prices):
        """Universe dispersion should be same value for all tickers."""
        spec = ('universe_dispersion', 21)
        result = lib._compute_feature(prices, None, spec)
        # All values should be the same (market-level feature)
        assert result.nunique() == 1
        assert result.iloc[0] > 0


class TestMicrostructureFeatures:
    """Test microstructure feature computations."""
    
    @pytest.fixture
    def lib(self):
        return __import__('evolution.features', fromlist=['FeatureLibrary']).FeatureLibrary()
    
    @pytest.fixture
    def prices(self):
        return _make_prices(n_days=300, n_tickers=10)
    
    @pytest.fixture
    def volume(self, prices):
        return _make_volume(prices)
    
    def test_amihud_illiquidity(self, lib, prices, volume):
        """Amihud illiquidity should be non-negative."""
        spec = ('amihud_illiquidity', 21)
        result = lib._compute_feature(prices, volume, spec)
        assert isinstance(result, pd.Series)
        assert (result >= 0).all()
    
    def test_amihud_no_volume(self, lib, prices):
        """Amihud returns zeros when no volume data."""
        spec = ('amihud_illiquidity', 21)
        result = lib._compute_feature(prices, None, spec)
        assert (result == 0).all()
    
    def test_roll_spread(self, lib, prices):
        """Roll spread should be non-negative."""
        spec = ('roll_spread', 21)
        result = lib._compute_feature(prices, None, spec)
        assert isinstance(result, pd.Series)
        assert (result >= 0).all()
    
    def test_zero_return_fraction(self, lib, prices):
        """Zero return fraction should be between 0 and 1."""
        spec = ('zero_return_fraction', 21)
        result = lib._compute_feature(prices, None, spec)
        assert (result >= 0).all()
        assert (result <= 1).all()
    
    def test_turnover_rate(self, lib, prices, volume):
        """Turnover rate should be non-negative."""
        spec = ('turnover_rate', 21)
        result = lib._compute_feature(prices, volume, spec)
        assert (result >= 0).all()


class TestRegimeFeatures:
    """Test regime feature computations."""
    
    @pytest.fixture
    def lib(self):
        return __import__('evolution.features', fromlist=['FeatureLibrary']).FeatureLibrary()
    
    @pytest.fixture
    def prices(self):
        return _make_prices(n_days=300, n_tickers=10)
    
    def test_volatility_regime(self, lib, prices):
        """Volatility regime should be positive."""
        spec = ('volatility_regime', 63)
        result = lib._compute_feature(prices, None, spec)
        assert isinstance(result, pd.Series)
        assert (result > 0).all()
    
    def test_adx_proxy(self, lib, prices):
        """ADX proxy should be between 0 and 1."""
        spec = ('adx_proxy', 21)
        result = lib._compute_feature(prices, None, spec)
        assert (result >= 0).all()
        assert (result <= 1).all()
    
    def test_market_breadth(self, lib, prices):
        """Market breadth should be between 0 and 1."""
        spec = ('market_breadth', 50)
        result = lib._compute_feature(prices, None, spec)
        # All values should be the same (market-level feature)
        assert result.nunique() == 1
        val = result.iloc[0]
        assert 0 <= val <= 1


class TestEngineeredFeatures:
    """Test engineered feature computations."""
    
    @pytest.fixture
    def lib(self):
        return __import__('evolution.features', fromlist=['FeatureLibrary']).FeatureLibrary()
    
    @pytest.fixture
    def prices(self):
        return _make_prices(n_days=300, n_tickers=10)
    
    def test_momentum_sharpe(self, lib, prices):
        """Momentum Sharpe should be bounded."""
        spec = ('momentum_sharpe', 21)
        result = lib._compute_feature(prices, None, spec)
        assert isinstance(result, pd.Series)
        assert (result >= -3).all()
        assert (result <= 3).all()
    
    def test_efficiency_ratio_bounds(self, lib, prices):
        """Efficiency ratio should be between 0 and 1."""
        spec = ('efficiency_ratio', 21)
        result = lib._compute_feature(prices, None, spec)
        assert (result >= 0).all()
        assert (result <= 1).all()
    
    def test_vol_adjusted_momentum(self, lib, prices):
        """Vol-adjusted momentum should be bounded."""
        spec = ('vol_adjusted_momentum', 63)
        result = lib._compute_feature(prices, None, spec)
        assert isinstance(result, pd.Series)
        assert (result >= -5).all()
        assert (result <= 5).all()
    
    def test_fractal_dimension(self, lib, prices):
        """Fractal dimension should be between 1 and 2."""
        spec = ('fractal_dimension', 63)
        result = lib._compute_feature(prices, None, spec)
        assert (result >= 1.0).all()
        assert (result <= 2.0).all()
    
    def test_drawdown_adjusted_momentum(self, lib, prices):
        """Drawdown-adjusted momentum should be bounded."""
        spec = ('drawdown_adjusted_momentum', 63)
        result = lib._compute_feature(prices, None, spec)
        assert isinstance(result, pd.Series)
        assert (result >= -5).all()
        assert (result <= 5).all()


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE COUNT TEST
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureCount:
    """Verify feature library has the expected number of features."""
    
    def test_feature_count_base(self):
        """Base feature count should be ~120+ (without optional SMC/SR/Oil)."""
        from evolution.features import FeatureLibrary
        lib = FeatureLibrary()
        # Original ~90 + 9 market-relative + 7 microstructure + 6 regime + 8 engineered = ~120
        assert len(lib.feature_names) >= 118, f"Expected >= 118 features, got {len(lib.feature_names)}"
    
    def test_new_feature_specs_present(self):
        """All new feature specs should be present."""
        from evolution.features import FeatureLibrary
        lib = FeatureLibrary()
        
        new_features = [
            # Market-relative
            'market_beta_21d', 'market_beta_63d', 'market_beta_126d',
            'beta_instability_63d',
            'idio_vol_21d', 'idio_vol_63d',
            'sector_correlation_21d', 'sector_correlation_63d',
            'universe_dispersion_21d',
            # Microstructure
            'amihud_21d', 'amihud_63d',
            'roll_spread_21d', 'kyle_lambda_21d',
            'zero_return_days_21d', 'zero_return_days_63d',
            'turnover_rate_21d',
            # Regime
            'vol_regime', 'trend_strength_21d', 'trend_strength_63d',
            'correlation_spike_21d', 'breadth_50d', 'breadth_200d',
            # Engineered
            'mom_sharpe_21d', 'mom_sharpe_63d',
            'dd_adj_mom_63d',
            'vol_adj_mom_21d', 'vol_adj_mom_63d',
            'efficiency_ratio_21d', 'efficiency_ratio_63d',
            'fractal_dim_63d',
        ]
        
        for feat in new_features:
            assert feat in lib.feature_specs, f"Missing feature: {feat}"
    
    def test_category_map_has_new_categories(self):
        """Category map should include new categories."""
        from evolution.features import FeatureLibrary
        lib = FeatureLibrary()
        
        assert 'market_relative' in lib.feature_categories
        assert 'microstructure' in lib.feature_categories
        assert 'regime' in lib.feature_categories
        assert 'engineered' in lib.feature_categories
        
        assert len(lib.feature_categories['market_relative']) >= 9
        assert len(lib.feature_categories['microstructure']) >= 7
        assert len(lib.feature_categories['regime']) >= 5
        assert len(lib.feature_categories['engineered']) >= 7


# ═══════════════════════════════════════════════════════════════════════════════
# EXPANDING WINDOW TEST
# ═══════════════════════════════════════════════════════════════════════════════

class TestExpandingWindow:
    """Test expanding window option in WalkForwardEvaluator."""
    
    def test_expanding_window_param(self):
        """WalkForwardEvaluator accepts expanding_window parameter."""
        from evolution.walkforward import WalkForwardEvaluator
        prices = _make_prices(n_days=300, n_tickers=5)
        
        evaluator = WalkForwardEvaluator(
            prices=prices,
            periods=[('2023-01-01', '2023-06-30', '2023-07-01', '2023-09-30')],
            expanding_window=True,
        )
        assert evaluator.expanding_window is True
    
    def test_expanding_window_default_false(self):
        """expanding_window defaults to False."""
        from evolution.walkforward import WalkForwardEvaluator
        prices = _make_prices(n_days=300, n_tickers=5)
        
        evaluator = WalkForwardEvaluator(
            prices=prices,
            periods=[],
        )
        assert evaluator.expanding_window is False
    
    def test_expanding_window_in_config(self):
        """expanding_window is serialized in to_config/from_config."""
        from evolution.walkforward import WalkForwardEvaluator
        prices = _make_prices(n_days=300, n_tickers=5)
        
        evaluator = WalkForwardEvaluator(
            prices=prices,
            periods=[],
            expanding_window=True,
        )
        
        config = evaluator.to_config()
        assert config['expanding_window'] is True
        
        restored = WalkForwardEvaluator.from_config(config)
        assert restored.expanding_window is True


# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE ALL INTEGRATION TEST
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeAllIntegration:
    """Test that compute_all works with all features including new ones."""
    
    def test_compute_all_no_errors(self):
        """compute_all should not raise for any feature."""
        from evolution.features import FeatureLibrary
        lib = FeatureLibrary()
        prices = _make_prices(n_days=300, n_tickers=5)
        volume = _make_volume(prices)
        
        features = lib.compute_all(prices, volume=volume, lag=1, rank_transform=True)
        
        assert isinstance(features, dict)
        assert len(features) == len(lib.feature_names)
        
        # No feature should be all NaN
        for name, series in features.items():
            assert isinstance(series, pd.Series), f"{name} is not a Series"
            assert not series.isna().all(), f"{name} is all NaN"
