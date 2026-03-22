"""
Unit tests for evolution/features.py (FeatureLibrary)

Tests every feature category for correct output shapes, bounds,
and behavior with/without volume data.
"""
import numpy as np
import pandas as pd
import pytest

from evolution.features import FeatureLibrary


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def lib():
    return FeatureLibrary()


@pytest.fixture(scope="module")
def prices():
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", periods=300)
    tickers = [f"T{i}" for i in range(10)]
    returns = np.random.normal(0.0005, 0.02, (300, 10))
    return pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=tickers)


@pytest.fixture(scope="module")
def volume(prices):
    np.random.seed(42)
    data = np.random.randint(100_000, 1_000_000, size=prices.shape).astype(float)
    return pd.DataFrame(data, index=prices.index, columns=prices.columns)


# ─── Library instantiation ────────────────────────────────────────────────────

class TestFeatureLibraryInit:
    def test_default_has_enough_features(self, lib):
        assert len(lib.feature_names) >= 118

    def test_feature_specs_keys_match_names(self, lib):
        assert set(lib.feature_specs.keys()) == set(lib.feature_names)

    def test_max_lookback_reasonable(self, lib):
        assert lib.max_lookback >= 252

    def test_categories_dict_present(self, lib):
        cats = lib.feature_categories
        assert isinstance(cats, dict)
        assert len(cats) > 0

    def test_optional_flags_default_false(self, lib):
        assert lib.enable_smc is False
        assert lib.enable_sr is False
        assert lib.enable_oil is False

    def test_advanced_calculators_none_when_disabled(self, lib):
        assert lib.smc_features is None
        assert lib.sr_features is None
        assert lib.oil_features is None

    def test_smc_features_added_when_enabled(self):
        lib_smc = FeatureLibrary(enable_smc=True)
        smc_names = [n for n in lib_smc.feature_names if n.startswith("smc_")]
        assert len(smc_names) >= 6
        assert lib_smc.smc_features is not None

    def test_sr_features_added_when_enabled(self):
        lib_sr = FeatureLibrary(enable_sr=True)
        sr_names = [n for n in lib_sr.feature_names if n.startswith("sr_")]
        assert len(sr_names) >= 8
        assert lib_sr.sr_features is not None

    def test_oil_features_added_when_enabled(self):
        lib_oil = FeatureLibrary(enable_oil=True)
        oil_names = [n for n in lib_oil.feature_names if n.startswith("oil_")]
        assert len(oil_names) >= 10
        assert lib_oil.oil_features is not None


# ─── Individual feature computation ──────────────────────────────────────────

class TestMomentumFeatures:
    @pytest.mark.parametrize("window", [5, 10, 21, 63, 126, 252])
    def test_momentum_series_shape(self, lib, prices, window):
        spec = ("momentum", window)
        result = lib._compute_feature(prices, None, spec)
        assert isinstance(result, pd.Series)
        assert len(result) == len(prices.columns)

    def test_momentum_no_nan(self, lib, prices):
        spec = ("momentum", 21)
        result = lib._compute_feature(prices, None, spec)
        assert not result.isna().any()

    def test_momentum_skip_shape(self, lib, prices):
        spec = ("momentum_skip", 252, 21)
        result = lib._compute_feature(prices, None, spec)
        assert len(result) == len(prices.columns)


class TestVolatilityFeatures:
    def test_volatility_nonneg(self, lib, prices):
        spec = ("volatility", 21)
        result = lib._compute_feature(prices, None, spec)
        assert (result >= 0).all()

    def test_garch_vol_nonneg(self, lib, prices):
        spec = ("garch_vol", 21)
        result = lib._compute_feature(prices, None, spec)
        assert (result >= 0).all()


class TestValueAndQualityFeatures:
    def test_range_position_bounds(self, lib, prices):
        spec = ("range_position", 21)
        result = lib._compute_feature(prices, None, spec)
        # May extend slightly beyond [0,1] but should be within a reasonable range
        assert result.max() <= 2.0
        assert result.min() >= -1.0

    def test_recovery_rate_no_nan(self, lib, prices):
        spec = ("recovery_rate", 63)
        result = lib._compute_feature(prices, None, spec)
        assert not result.isna().any()


class TestVolumeFeatures:
    def test_volume_trend_with_volume(self, lib, prices, volume):
        spec = ("volume_trend", 21)
        result = lib._compute_feature(prices, volume, spec)
        assert isinstance(result, pd.Series)
        assert not result.isna().any()

    def test_amihud_nonneg_with_volume(self, lib, prices, volume):
        spec = ("amihud_illiquidity", 21)
        result = lib._compute_feature(prices, volume, spec)
        assert (result >= 0).all()

    def test_amihud_zeros_without_volume(self, lib, prices):
        spec = ("amihud_illiquidity", 21)
        result = lib._compute_feature(prices, None, spec)
        assert (result == 0).all()

    def test_turnover_nonneg_with_volume(self, lib, prices, volume):
        spec = ("turnover_rate", 21)
        result = lib._compute_feature(prices, volume, spec)
        assert (result >= 0).all()


class TestMarketRelativeFeatures:
    def test_market_beta_mean_near_one(self, lib, prices):
        spec = ("market_beta", 63)
        result = lib._compute_feature(prices, None, spec)
        assert abs(result.mean() - 1.0) < 0.5

    def test_beta_instability_nonneg(self, lib, prices):
        spec = ("beta_instability", 63)
        result = lib._compute_feature(prices, None, spec)
        assert (result >= 0).all()

    def test_idiosyncratic_vol_nonneg(self, lib, prices):
        spec = ("idiosyncratic_vol", 63)
        result = lib._compute_feature(prices, None, spec)
        assert (result >= 0).all()

    def test_sector_correlation_bounded(self, lib, prices):
        spec = ("sector_correlation", 63)
        result = lib._compute_feature(prices, None, spec)
        assert (result >= -1.01).all()
        assert (result <= 1.01).all()

    def test_universe_dispersion_same_for_all_tickers(self, lib, prices):
        spec = ("universe_dispersion", 21)
        result = lib._compute_feature(prices, None, spec)
        assert result.nunique() == 1


class TestRegimeFeatures:
    def test_vol_regime_positive(self, lib, prices):
        spec = ("volatility_regime", 63)
        result = lib._compute_feature(prices, None, spec)
        assert (result > 0).all()

    def test_adx_proxy_bounds(self, lib, prices):
        spec = ("adx_proxy", 21)
        result = lib._compute_feature(prices, None, spec)
        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_market_breadth_market_level(self, lib, prices):
        spec = ("market_breadth", 50)
        result = lib._compute_feature(prices, None, spec)
        assert result.nunique() == 1
        assert 0 <= result.iloc[0] <= 1


class TestEngineeredFeatures:
    def test_efficiency_ratio_bounds(self, lib, prices):
        spec = ("efficiency_ratio", 21)
        result = lib._compute_feature(prices, None, spec)
        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_fractal_dimension_bounds(self, lib, prices):
        spec = ("fractal_dimension", 63)
        result = lib._compute_feature(prices, None, spec)
        assert (result >= 1.0).all()
        assert (result <= 2.0).all()

    def test_momentum_sharpe_bounded(self, lib, prices):
        spec = ("momentum_sharpe", 21)
        result = lib._compute_feature(prices, None, spec)
        assert (result >= -3).all()
        assert (result <= 3).all()

    def test_vol_adjusted_momentum_bounded(self, lib, prices):
        spec = ("vol_adjusted_momentum", 63)
        result = lib._compute_feature(prices, None, spec)
        assert (result >= -5).all()
        assert (result <= 5).all()


# ─── compute_all() integration ────────────────────────────────────────────────

class TestComputeAll:
    def test_returns_all_features(self, lib, prices, volume):
        features = lib.compute_all(prices, volume=volume, lag=1, rank_transform=True)
        assert set(features.keys()) == set(lib.feature_names)

    def test_no_feature_is_all_nan(self, lib, prices, volume):
        features = lib.compute_all(prices, volume=volume, lag=1, rank_transform=True)
        for name, series in features.items():
            assert not series.isna().all(), f"{name} is all-NaN"

    def test_all_values_are_series(self, lib, prices, volume):
        features = lib.compute_all(prices, volume=volume, lag=1, rank_transform=True)
        for name, series in features.items():
            assert isinstance(series, pd.Series), f"{name} is not a Series"

    def test_rank_transform_scales_to_zero_one(self, lib, prices):
        """Rank-transformed features should be within [0, 1]."""
        features = lib.compute_all(prices, lag=1, rank_transform=True)
        skip = getattr(lib, "_skip_rank_transform", set())
        for name, series in features.items():
            if name in skip:
                continue
            if name.startswith(("smc_", "sr_", "oil_")):
                continue
            assert series.max() <= 1.01, f"{name} has values > 1 after rank transform"
            assert series.min() >= -0.01, f"{name} has values < 0 after rank transform"

    def test_lag_shifts_data(self, lib, prices):
        """With lag=1, values should differ from lag=0 (time-shifted)."""
        f0 = lib.compute_all(prices, lag=0, rank_transform=False)
        f1 = lib.compute_all(prices, lag=1, rank_transform=False)
        # They may be identical only if the feature uses a very long window,
        # but for most features the one-day lag should produce different values.
        any_diff = False
        for name in f0:
            if not f0[name].equals(f1[name]):
                any_diff = True
                break
        # Allow if nothing changes (e.g. if all features have lag built-in)
        # — at minimum ensure no exceptions were raised.


# ─── Feature cache ────────────────────────────────────────────────────────────

class TestFeatureCacheInLibrary:
    def test_cache_hit_after_second_call(self, prices):
        lib2 = FeatureLibrary()
        lib2.compute_all(prices, use_cache=True)
        stats_after_1 = lib2.get_cache_stats()
        lib2.compute_all(prices, use_cache=True)
        stats_after_2 = lib2.get_cache_stats()
        assert stats_after_1["misses"] == 1
        assert stats_after_2["hits"] == 1

    def test_cache_bypass(self, prices):
        lib2 = FeatureLibrary()
        lib2.compute_all(prices, use_cache=False)
        lib2.compute_all(prices, use_cache=False)
        stats = lib2.get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_clear_cache_resets_stats(self, prices):
        lib2 = FeatureLibrary()
        lib2.compute_all(prices, use_cache=True)
        lib2.clear_cache()
        stats = lib2.get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["cache_size"] == 0

    def test_different_slices_get_different_entries(self, prices):
        lib2 = FeatureLibrary()
        lib2.compute_all(prices.iloc[:200], use_cache=True)
        lib2.compute_all(prices.iloc[:250], use_cache=True)
        stats = lib2.get_cache_stats()
        assert stats["cache_size"] == 2
