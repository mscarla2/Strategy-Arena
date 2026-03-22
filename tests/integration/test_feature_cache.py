"""
Integration tests for evolution/feature_cache.py (FeaturePrecomputeCache)

Tests in-memory cache, disk persistence, hit/miss tracking, and
multi-universe isolation.
"""
import numpy as np
import pandas as pd
import pytest

from evolution.feature_cache import FeaturePrecomputeCache


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_prices(n_days=500, n_tickers=5, seed=42):
    np.random.seed(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    tickers = [f"TICK{i}" for i in range(n_tickers)]
    data = {t: 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, n_days))
            for t in tickers}
    return pd.DataFrame(data, index=dates)


def _make_periods(prices):
    dates = prices.index
    mid = len(dates) // 2
    return [(
        dates[0].strftime("%Y-%m-%d"),
        dates[mid].strftime("%Y-%m-%d"),
        dates[mid].strftime("%Y-%m-%d"),
        dates[-1].strftime("%Y-%m-%d"),
    )]


class _MockLib:
    """Minimal feature library substitute for cache tests."""
    feature_names = ["feat_a", "feat_b"]
    max_lookback = 252
    call_count = 0

    def compute_all(self, prices, volume=None, lag=1, rank_transform=True):
        _MockLib.call_count += 1
        return {
            "feat_a": pd.Series(0.5, index=prices.columns),
            "feat_b": pd.Series(0.3, index=prices.columns),
        }


# ─── In-memory cache ─────────────────────────────────────────────────────────

class TestInMemoryCache:
    def test_cache_populates_on_create(self):
        prices = _make_prices()
        periods = _make_periods(prices)
        lib = _MockLib()
        cache = FeaturePrecomputeCache(
            feature_lib=lib, prices=prices, volume=None,
            periods=periods, rebalance_frequency=21, persist_to_disk=False,
        )
        assert len(cache.cache) > 0
        assert cache.stats["cached_dates"] > 0

    def test_get_features_hit(self):
        prices = _make_prices()
        periods = _make_periods(prices)
        cache = FeaturePrecomputeCache(
            feature_lib=_MockLib(), prices=prices, volume=None,
            periods=periods, rebalance_frequency=21, persist_to_disk=False,
        )
        first_date = list(cache.cache.keys())[0]
        result = cache.get_features(first_date)
        assert result is not None
        assert "feat_a" in result
        assert cache._hits == 1

    def test_get_features_miss(self):
        prices = _make_prices()
        periods = _make_periods(prices)
        cache = FeaturePrecomputeCache(
            feature_lib=_MockLib(), prices=prices, volume=None,
            periods=periods, rebalance_frequency=21, persist_to_disk=False,
        )
        result = cache.get_features("1900-01-01")
        assert result is None
        assert cache._misses == 1

    def test_stats_accurate(self):
        prices = _make_prices()
        periods = _make_periods(prices)
        cache = FeaturePrecomputeCache(
            feature_lib=_MockLib(), prices=prices, volume=None,
            periods=periods, rebalance_frequency=21, persist_to_disk=False,
        )
        stats = cache.stats
        assert stats["features_per_date"] == 2
        assert stats["cached_dates"] > 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_hit_rate_after_queries(self):
        prices = _make_prices()
        periods = _make_periods(prices)
        cache = FeaturePrecomputeCache(
            feature_lib=_MockLib(), prices=prices, volume=None,
            periods=periods, rebalance_frequency=21, persist_to_disk=False,
        )
        first_date = list(cache.cache.keys())[0]
        cache.get_features(first_date)    # hit
        cache.get_features("2099-01-01")  # miss
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_no_disk_saves_when_persist_false(self, tmp_path):
        prices = _make_prices()
        periods = _make_periods(prices)
        cache = FeaturePrecomputeCache(
            feature_lib=_MockLib(), prices=prices, volume=None,
            periods=periods, rebalance_frequency=21,
            cache_dir=tmp_path, persist_to_disk=False,
        )
        assert len(list(tmp_path.iterdir())) == 0
        assert cache.stats["disk_saves"] == 0


# ─── Disk persistence ─────────────────────────────────────────────────────────

class TestDiskPersistence:
    def test_parquet_files_created(self, tmp_path):
        prices = _make_prices()
        periods = _make_periods(prices)
        lib = _MockLib()
        _MockLib.call_count = 0
        cache = FeaturePrecomputeCache(
            feature_lib=lib, prices=prices, volume=None,
            periods=periods, rebalance_frequency=21,
            cache_dir=tmp_path, persist_to_disk=True,
        )
        assert cache.stats["disk_saves"] > 0
        universe_dir = next(tmp_path.iterdir())
        parquet_files = list(universe_dir.glob("*.parquet"))
        assert len(parquet_files) == cache.stats["cached_dates"]

    def test_second_run_loads_from_disk(self, tmp_path):
        prices = _make_prices()
        periods = _make_periods(prices)

        lib1 = _MockLib()
        _MockLib.call_count = 0
        cache1 = FeaturePrecomputeCache(
            feature_lib=lib1, prices=prices, volume=None,
            periods=periods, rebalance_frequency=21,
            cache_dir=tmp_path, persist_to_disk=True,
        )
        n_dates = cache1.stats["cached_dates"]

        lib2 = _MockLib()
        _MockLib.call_count = 0
        cache2 = FeaturePrecomputeCache(
            feature_lib=lib2, prices=prices, volume=None,
            periods=periods, rebalance_frequency=21,
            cache_dir=tmp_path, persist_to_disk=True,
        )
        assert cache2.stats["disk_loads"] == n_dates
        assert cache2.stats["disk_saves"] == 0
        assert _MockLib.call_count == 0, "compute_all should NOT be called on second run"

    def test_disk_data_matches_original(self, tmp_path):
        prices = _make_prices()
        periods = _make_periods(prices)

        cache1 = FeaturePrecomputeCache(
            feature_lib=_MockLib(), prices=prices, volume=None,
            periods=periods, rebalance_frequency=21,
            cache_dir=tmp_path, persist_to_disk=True,
        )
        cache2 = FeaturePrecomputeCache(
            feature_lib=_MockLib(), prices=prices, volume=None,
            periods=periods, rebalance_frequency=21,
            cache_dir=tmp_path, persist_to_disk=True,
        )
        for date_key in cache1.cache:
            assert date_key in cache2.cache
            for feat_name in cache1.cache[date_key]:
                pd.testing.assert_series_equal(
                    cache1.cache[date_key][feat_name],
                    cache2.cache[date_key][feat_name],
                    check_names=False,
                )

    def test_clear_disk_cache_removes_files(self, tmp_path):
        prices = _make_prices()
        periods = _make_periods(prices)
        cache = FeaturePrecomputeCache(
            feature_lib=_MockLib(), prices=prices, volume=None,
            periods=periods, rebalance_frequency=21,
            cache_dir=tmp_path, persist_to_disk=True,
        )
        universe_dir = next(tmp_path.iterdir())
        assert len(list(universe_dir.glob("*.parquet"))) > 0

        cache.clear_disk_cache()
        assert len(list(universe_dir.glob("*.parquet"))) == 0

    def test_different_universes_get_separate_dirs(self, tmp_path):
        prices_a = _make_prices(n_tickers=3, seed=1)
        prices_b = _make_prices(n_tickers=7, seed=2)

        FeaturePrecomputeCache(
            feature_lib=_MockLib(), prices=prices_a, volume=None,
            periods=_make_periods(prices_a), rebalance_frequency=21,
            cache_dir=tmp_path, persist_to_disk=True,
        )
        FeaturePrecomputeCache(
            feature_lib=_MockLib(), prices=prices_b, volume=None,
            periods=_make_periods(prices_b), rebalance_frequency=21,
            cache_dir=tmp_path, persist_to_disk=True,
        )
        subdirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(subdirs) == 2
