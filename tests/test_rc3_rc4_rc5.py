"""
Tests for RC-3 (Fitness v2), RC-4 (Oil Universe), RC-5 (Feature Cache)

Covers:
- RC-3: Recency weighting, universe-adaptive drawdown thresholds, calculate_fitness_v2
- RC-4: Oil reference panel, expanded universe, tradeable tickers
- RC-5: FeaturePrecomputeCache
"""

import pytest
import numpy as np
import pandas as pd
from typing import List, Dict


# ═══════════════════════════════════════════════════════════════════════════════
# RC-3: FITNESS FUNCTION REDESIGN TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestRecencyWeightedMean:
    """Test _recency_weighted_mean helper."""
    
    def test_empty_list(self):
        from evolution.gp import _recency_weighted_mean
        assert _recency_weighted_mean([]) == 0.0
    
    def test_single_value(self):
        from evolution.gp import _recency_weighted_mean
        assert _recency_weighted_mean([0.5]) == 0.5
    
    def test_recent_values_weighted_more(self):
        """Most recent value should have highest weight."""
        from evolution.gp import _recency_weighted_mean
        
        # All zeros except last value is 1.0
        values = [0.0, 0.0, 0.0, 0.0, 1.0]
        result_short = _recency_weighted_mean(values, half_life_periods=2)
        result_long = _recency_weighted_mean(values, half_life_periods=10)
        
        # With shorter half-life, the last value should get more weight
        assert result_short > result_long, \
            f"Short HL ({result_short}) should be > Long HL ({result_long})"
        # Simple mean would be 0.2; recency weighting should give more than that
        assert result_short > 0.2, f"Expected > 0.2 (simple mean), got {result_short}"
    
    def test_equal_values_returns_same(self):
        """If all values are equal, result should be that value."""
        from evolution.gp import _recency_weighted_mean
        
        values = [0.3, 0.3, 0.3, 0.3]
        result = _recency_weighted_mean(values, half_life_periods=4)
        assert abs(result - 0.3) < 1e-10
    
    def test_half_life_effect(self):
        """Shorter half-life should give more weight to recent values."""
        from evolution.gp import _recency_weighted_mean
        
        values = [1.0, 0.0, 0.0, 0.0, 0.0]  # Old value is high, recent are low
        
        short_hl = _recency_weighted_mean(values, half_life_periods=1)
        long_hl = _recency_weighted_mean(values, half_life_periods=10)
        
        # Short half-life should give less weight to old high value
        assert short_hl < long_hl, f"Short HL ({short_hl}) should be < Long HL ({long_hl})"


class TestDrawdownThresholds:
    """Test _get_drawdown_thresholds helper."""
    
    def test_general_thresholds(self):
        from evolution.gp import _get_drawdown_thresholds
        
        thresholds = _get_drawdown_thresholds('general')
        assert thresholds['severe'] == 0.35
        assert thresholds['moderate'] == 0.25
        assert thresholds['mild'] == 0.15
    
    def test_oil_microcap_thresholds_wider(self):
        """Oil microcap thresholds should be wider than general."""
        from evolution.gp import _get_drawdown_thresholds
        
        general = _get_drawdown_thresholds('general')
        oil = _get_drawdown_thresholds('oil_microcap')
        
        assert oil['severe'] > general['severe']
        assert oil['moderate'] > general['moderate']
        assert oil['mild'] > general['mild']
    
    def test_oil_microcap_specific_values(self):
        from evolution.gp import _get_drawdown_thresholds
        
        thresholds = _get_drawdown_thresholds('oil_microcap')
        assert thresholds['severe'] == 0.50
        assert thresholds['moderate'] == 0.35
        assert thresholds['mild'] == 0.20
    
    def test_oil_largecap_between_general_and_microcap(self):
        from evolution.gp import _get_drawdown_thresholds
        
        general = _get_drawdown_thresholds('general')
        largecap = _get_drawdown_thresholds('oil_largecap')
        microcap = _get_drawdown_thresholds('oil_microcap')
        
        assert general['severe'] <= largecap['severe'] <= microcap['severe']


class TestFitnessV2:
    """Test calculate_fitness_v2."""
    
    @staticmethod
    def _make_period_results(n=5, base_return=0.05, base_sharpe=0.8, base_dd=0.15):
        """Helper to create mock period results."""
        results = []
        for i in range(n):
            results.append({
                'total_return': base_return + np.random.normal(0, 0.02),
                'sharpe_ratio': base_sharpe + np.random.normal(0, 0.1),
                'max_drawdown': base_dd + abs(np.random.normal(0, 0.03)),
                'turnover': 0.3 + np.random.normal(0, 0.05),
                'n_days': 63,
            })
        return results
    
    @staticmethod
    def _make_benchmark_results(n=5, base_return=0.03, base_sharpe=0.5, base_dd=0.12):
        """Helper to create mock benchmark results."""
        results = []
        for i in range(n):
            results.append({
                'total_return': base_return + np.random.normal(0, 0.01),
                'sharpe_ratio': base_sharpe + np.random.normal(0, 0.05),
                'max_drawdown': base_dd + abs(np.random.normal(0, 0.02)),
            })
        return results
    
    def test_returns_fitness_result(self):
        from evolution.gp import calculate_fitness_v2, FitnessResult
        
        np.random.seed(42)
        period_results = self._make_period_results()
        benchmark_results = self._make_benchmark_results()
        
        result = calculate_fitness_v2(period_results, benchmark_results)
        assert isinstance(result, FitnessResult)
        assert -1 <= result.total <= 1
    
    def test_empty_results(self):
        from evolution.gp import calculate_fitness_v2
        
        result = calculate_fitness_v2([], [])
        assert result.total == 0
        assert result.n_periods == 0
    
    def test_oil_microcap_less_penalized_for_drawdown(self):
        """Oil microcap should be less penalized for same drawdown."""
        from evolution.gp import calculate_fitness_v2
        
        np.random.seed(42)
        # Create results with 35% drawdown (severe for general, moderate for oil)
        period_results = self._make_period_results(n=6, base_dd=0.36)
        benchmark_results = self._make_benchmark_results(n=6)
        
        general_fitness = calculate_fitness_v2(
            period_results, benchmark_results, universe_type='general'
        )
        oil_fitness = calculate_fitness_v2(
            period_results, benchmark_results, universe_type='oil_microcap'
        )
        
        # Oil microcap should have higher fitness (less penalty for same DD)
        assert oil_fitness.total >= general_fitness.total, \
            f"Oil ({oil_fitness.total:.3f}) should be >= General ({general_fitness.total:.3f})"
    
    def test_recency_weighting_affects_result(self):
        """Different recency half-lives should produce different results."""
        from evolution.gp import calculate_fitness_v2
        
        np.random.seed(42)
        # Create results where recent periods are much better
        period_results = self._make_period_results(n=8, base_return=-0.05)
        # Make last 2 periods much better
        period_results[-1]['total_return'] = 0.15
        period_results[-1]['sharpe_ratio'] = 1.5
        period_results[-2]['total_return'] = 0.12
        period_results[-2]['sharpe_ratio'] = 1.2
        
        benchmark_results = self._make_benchmark_results(n=8)
        
        short_hl = calculate_fitness_v2(
            period_results, benchmark_results, recency_half_life=2
        )
        long_hl = calculate_fitness_v2(
            period_results, benchmark_results, recency_half_life=20
        )
        
        # Short half-life should favor the recent good periods more
        assert short_hl.total > long_hl.total, \
            f"Short HL ({short_hl.total:.3f}) should be > Long HL ({long_hl.total:.3f})"
    
    def test_fitness_v2_has_calmar_component(self):
        """Fitness v2 should blend Sharpe and Calmar."""
        from evolution.gp import calculate_fitness_v2
        
        np.random.seed(42)
        period_results = self._make_period_results()
        benchmark_results = self._make_benchmark_results()
        
        result = calculate_fitness_v2(period_results, benchmark_results)
        
        # The total should be a blend of components
        assert result.sharpe_component is not None
        assert result.return_component is not None
        assert result.stability_component is not None


# ═══════════════════════════════════════════════════════════════════════════════
# RC-4: OIL UNIVERSE SPECIALIZATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestOilUniverse:
    """Test expanded oil universe with reference panel."""
    
    def test_oil_reference_panel_has_16_tickers(self):
        from data.universe import OIL_REFERENCE_PANEL
        assert len(OIL_REFERENCE_PANEL) == 16
    
    def test_oil_reference_panel_includes_majors(self):
        from data.universe import OIL_REFERENCE_PANEL
        for ticker in ["XOM", "CVX", "COP", "OXY", "EOG"]:
            assert ticker in OIL_REFERENCE_PANEL, f"{ticker} missing from reference panel"
    
    def test_oil_reference_panel_includes_etfs(self):
        from data.universe import OIL_REFERENCE_PANEL
        assert "XLE" in OIL_REFERENCE_PANEL
        assert "XOP" in OIL_REFERENCE_PANEL
    
    def test_oil_tradeable_universe_is_microcaps(self):
        from data.universe import OIL_TRADEABLE_UNIVERSE, OIL_MICROCAP_STOCKS
        assert OIL_TRADEABLE_UNIVERSE == OIL_MICROCAP_STOCKS
    
    def test_oil_benchmarks_include_xle(self):
        from data.universe import OIL_BENCHMARKS
        assert "XLE" in OIL_BENCHMARKS
        assert "XOP" in OIL_BENCHMARKS
        assert "USO" in OIL_BENCHMARKS
        assert "BNO" in OIL_BENCHMARKS
    
    def test_full_download_universe_is_superset(self):
        """Full download should contain all tradeable + reference + benchmarks."""
        from data.universe import (
            OIL_FULL_DOWNLOAD_UNIVERSE, OIL_TRADEABLE_UNIVERSE,
            OIL_REFERENCE_PANEL, OIL_BENCHMARKS
        )
        full_set = set(OIL_FULL_DOWNLOAD_UNIVERSE)
        
        for ticker in OIL_TRADEABLE_UNIVERSE:
            assert ticker in full_set, f"Tradeable {ticker} missing from full download"
        for ticker in OIL_REFERENCE_PANEL:
            assert ticker in full_set, f"Reference {ticker} missing from full download"
        for ticker in OIL_BENCHMARKS:
            assert ticker in full_set, f"Benchmark {ticker} missing from full download"
    
    def test_no_overlap_tradeable_and_reference(self):
        """Tradeable tickers should not overlap with reference panel."""
        from data.universe import OIL_TRADEABLE_UNIVERSE, OIL_REFERENCE_PANEL
        overlap = set(OIL_TRADEABLE_UNIVERSE) & set(OIL_REFERENCE_PANEL)
        assert len(overlap) == 0, f"Overlap between tradeable and reference: {overlap}"
    
    def test_get_oil_universe_expanded(self):
        from data.universe import get_oil_universe, OIL_FULL_DOWNLOAD_UNIVERSE
        result = get_oil_universe(expanded=True)
        assert set(result) == set(OIL_FULL_DOWNLOAD_UNIVERSE)
    
    def test_get_oil_universe_legacy(self):
        from data.universe import get_oil_universe, OIL_FOCUSED_UNIVERSE
        result = get_oil_universe(expanded=False)
        assert result == OIL_FOCUSED_UNIVERSE
    
    def test_get_oil_tradeable_tickers(self):
        from data.universe import get_oil_tradeable_tickers, OIL_TRADEABLE_UNIVERSE
        assert get_oil_tradeable_tickers() == OIL_TRADEABLE_UNIVERSE
    
    def test_get_oil_reference_panel(self):
        from data.universe import get_oil_reference_panel, OIL_REFERENCE_PANEL
        assert get_oil_reference_panel() == OIL_REFERENCE_PANEL
    
    def test_get_oil_benchmarks(self):
        from data.universe import get_oil_benchmarks, OIL_BENCHMARKS
        assert get_oil_benchmarks() == OIL_BENCHMARKS
    
    def test_full_download_has_enough_tickers_for_cross_sectional(self):
        """Need 20+ tickers for meaningful cross-sectional features."""
        from data.universe import OIL_FULL_DOWNLOAD_UNIVERSE
        assert len(OIL_FULL_DOWNLOAD_UNIVERSE) >= 20, \
            f"Only {len(OIL_FULL_DOWNLOAD_UNIVERSE)} tickers, need 20+ for cross-sectional"


# ═══════════════════════════════════════════════════════════════════════════════
# RC-5: FEATURE PRE-COMPUTATION CACHE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeaturePrecomputeCache:
    """Test FeaturePrecomputeCache."""
    
    @staticmethod
    def _make_mock_prices(n_days=500, n_tickers=5):
        """Create mock price data."""
        dates = pd.bdate_range('2023-01-01', periods=n_days)
        tickers = [f"TICK{i}" for i in range(n_tickers)]
        
        np.random.seed(42)
        data = {}
        for ticker in tickers:
            # Random walk prices starting at 100
            returns = np.random.normal(0.0005, 0.02, n_days)
            prices = 100 * np.cumprod(1 + returns)
            data[ticker] = prices
        
        return pd.DataFrame(data, index=dates)
    
    @staticmethod
    def _make_mock_periods(prices):
        """Create mock walk-forward periods from price data."""
        dates = prices.index
        mid = len(dates) // 2
        
        return [
            (
                dates[0].strftime("%Y-%m-%d"),
                dates[mid].strftime("%Y-%m-%d"),
                dates[mid].strftime("%Y-%m-%d"),
                dates[-1].strftime("%Y-%m-%d"),
            )
        ]
    
    def test_cache_creation(self):
        """Cache should be created without errors."""
        from evolution.feature_cache import FeaturePrecomputeCache
        
        prices = self._make_mock_prices()
        periods = self._make_mock_periods(prices)
        
        # Create a minimal mock feature library
        class MockFeatureLib:
            feature_names = ['mom_5d', 'vol_21d']
            max_lookback = 252
            
            def compute_all(self, prices, volume=None, lag=1, rank_transform=True):
                result = {}
                for name in self.feature_names:
                    result[name] = pd.Series(
                        np.random.randn(len(prices.columns)),
                        index=prices.columns
                    )
                return result
        
        cache = FeaturePrecomputeCache(
            feature_lib=MockFeatureLib(),
            prices=prices,
            volume=None,
            periods=periods,
            rebalance_frequency=21,
        )
        
        assert len(cache.cache) > 0
        assert cache.stats['cached_dates'] > 0
    
    def test_cache_hit_rate(self):
        """Cache should have hits when querying cached dates."""
        from evolution.feature_cache import FeaturePrecomputeCache
        
        prices = self._make_mock_prices()
        periods = self._make_mock_periods(prices)
        
        class MockFeatureLib:
            feature_names = ['mom_5d']
            max_lookback = 252
            
            def compute_all(self, prices, volume=None, lag=1, rank_transform=True):
                return {'mom_5d': pd.Series(0.5, index=prices.columns)}
        
        cache = FeaturePrecomputeCache(
            feature_lib=MockFeatureLib(),
            prices=prices,
            volume=None,
            periods=periods,
            rebalance_frequency=21,
        )
        
        # Query a cached date
        if cache.cache:
            first_date = list(cache.cache.keys())[0]
            result = cache.get_features(first_date)
            assert result is not None
            assert cache._hits == 1
    
    def test_cache_miss(self):
        """Cache should report misses for uncached dates."""
        from evolution.feature_cache import FeaturePrecomputeCache
        
        prices = self._make_mock_prices()
        periods = self._make_mock_periods(prices)
        
        class MockFeatureLib:
            feature_names = ['mom_5d']
            max_lookback = 252
            
            def compute_all(self, prices, volume=None, lag=1, rank_transform=True):
                return {'mom_5d': pd.Series(0.5, index=prices.columns)}
        
        cache = FeaturePrecomputeCache(
            feature_lib=MockFeatureLib(),
            prices=prices,
            volume=None,
            periods=periods,
            rebalance_frequency=21,
        )
        
        result = cache.get_features("1999-01-01")
        assert result is None
        assert cache._misses == 1
    
    def test_cache_stats(self):
        """Cache stats should be accurate."""
        from evolution.feature_cache import FeaturePrecomputeCache
        
        prices = self._make_mock_prices()
        periods = self._make_mock_periods(prices)
        
        class MockFeatureLib:
            feature_names = ['mom_5d', 'vol_21d']
            max_lookback = 252
            
            def compute_all(self, prices, volume=None, lag=1, rank_transform=True):
                return {
                    'mom_5d': pd.Series(0.5, index=prices.columns),
                    'vol_21d': pd.Series(0.2, index=prices.columns),
                }
        
        cache = FeaturePrecomputeCache(
            feature_lib=MockFeatureLib(),
            prices=prices,
            volume=None,
            periods=periods,
            rebalance_frequency=21,
        )
        
        stats = cache.stats
        assert stats['features_per_date'] == 2
        assert stats['cached_dates'] > 0
        assert stats['hits'] == 0
        assert stats['misses'] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfig:
    """Test new config settings."""
    
    def test_universe_type_default(self):
        from config import UNIVERSE_TYPE
        assert UNIVERSE_TYPE == "general"
    
    def test_recency_half_life_default(self):
        from config import RECENCY_HALF_LIFE
        assert RECENCY_HALF_LIFE == 4
    
    def test_oil_primary_benchmark(self):
        from config import OIL_PRIMARY_BENCHMARK
        assert OIL_PRIMARY_BENCHMARK == "XLE"
    
    def test_feature_cache_enabled(self):
        from config import ENABLE_FEATURE_CACHE
        assert ENABLE_FEATURE_CACHE is True
    
    def test_expanding_window_default(self):
        from config import USE_EXPANDING_WINDOW
        assert USE_EXPANDING_WINDOW is False


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration tests for RC-3/4/5 working together."""
    
    def test_fitness_v2_importable(self):
        """calculate_fitness_v2 should be importable from evolution.gp."""
        from evolution.gp import calculate_fitness_v2
        assert callable(calculate_fitness_v2)
    
    def test_recency_helpers_importable(self):
        """Helper functions should be importable."""
        from evolution.gp import _recency_weighted_mean, _get_drawdown_thresholds
        assert callable(_recency_weighted_mean)
        assert callable(_get_drawdown_thresholds)
    
    def test_feature_cache_importable(self):
        """FeaturePrecomputeCache should be importable."""
        from evolution.feature_cache import FeaturePrecomputeCache
        assert FeaturePrecomputeCache is not None
    
    def test_oil_universe_importable_from_data(self):
        """Oil universe should be importable from data module."""
        from data import (
            OIL_REFERENCE_PANEL, OIL_TRADEABLE_UNIVERSE,
            OIL_FULL_DOWNLOAD_UNIVERSE, OIL_BENCHMARKS,
            get_oil_universe, get_oil_tradeable_tickers,
            get_oil_reference_panel, get_oil_benchmarks,
        )
        assert len(OIL_REFERENCE_PANEL) > 0
        assert len(OIL_TRADEABLE_UNIVERSE) > 0
    
    def test_walkforward_evaluator_accepts_new_params(self):
        """WalkForwardEvaluator should accept RC-3/4 parameters."""
        from evolution.gp import WalkForwardEvaluator
        
        # Create minimal mock data
        dates = pd.bdate_range('2024-01-01', periods=100)
        prices = pd.DataFrame(
            {'A': np.random.randn(100).cumsum() + 100},
            index=dates
        )
        
        evaluator = WalkForwardEvaluator(
            prices=prices,
            periods=[],
            benchmark_results=[],
            universe_type='oil_microcap',
            recency_half_life=4,
            use_fitness_v2=True,
            tradeable_tickers=['A'],
        )
        
        assert evaluator.universe_type == 'oil_microcap'
        assert evaluator.recency_half_life == 4
        assert evaluator.use_fitness_v2 is True
        assert evaluator.tradeable_tickers == ['A']
    
    def test_walkforward_to_config_includes_new_params(self):
        """to_config should serialize new RC-3/4 parameters."""
        from evolution.gp import WalkForwardEvaluator
        
        dates = pd.bdate_range('2024-01-01', periods=100)
        prices = pd.DataFrame(
            {'A': np.random.randn(100).cumsum() + 100},
            index=dates
        )
        
        evaluator = WalkForwardEvaluator(
            prices=prices,
            periods=[],
            benchmark_results=[],
            universe_type='oil_microcap',
            recency_half_life=3,
            use_fitness_v2=True,
            tradeable_tickers=['A', 'B'],
        )
        
        config = evaluator.to_config()
        assert config['universe_type'] == 'oil_microcap'
        assert config['recency_half_life'] == 3
        assert config['use_fitness_v2'] is True
        assert config['tradeable_tickers'] == ['A', 'B']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
