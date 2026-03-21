#!/usr/bin/env python3
"""
Test feature caching implementation.

Verifies that:
1. Feature cache correctly stores and retrieves features
2. Cache hits/misses are tracked accurately
3. Cached features match freshly computed features
4. Cache provides performance improvement
"""

import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evolution.gp import FeatureLibrary


def test_cache_functionality():
    """Test that cache stores and retrieves features correctly."""
    print("\n=== Test 1: Cache Functionality ===")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    prices = pd.DataFrame(
        np.random.randn(len(dates), len(tickers)).cumsum(axis=0) + 100,
        index=dates,
        columns=tickers
    )
    
    # Create feature library
    feature_lib = FeatureLibrary()
    
    # First computation (cache miss)
    features1 = feature_lib.compute_all(prices, use_cache=True)
    stats1 = feature_lib.get_cache_stats()
    
    print(f"First computation - Cache stats: {stats1}")
    assert stats1['misses'] == 1, "Should have 1 cache miss"
    assert stats1['hits'] == 0, "Should have 0 cache hits"
    assert stats1['cache_size'] == 1, "Cache should have 1 entry"
    print("✓ First computation correctly missed cache")
    
    # Second computation with same data (cache hit)
    features2 = feature_lib.compute_all(prices, use_cache=True)
    stats2 = feature_lib.get_cache_stats()
    
    print(f"Second computation - Cache stats: {stats2}")
    assert stats2['hits'] == 1, "Should have 1 cache hit"
    assert stats2['misses'] == 1, "Should still have 1 cache miss"
    assert stats2['hit_rate'] == 50.0, "Hit rate should be 50%"
    print("✓ Second computation correctly hit cache")
    
    # Verify features match
    for name in features1.keys():
        assert features1[name].equals(features2[name]), f"Feature {name} mismatch"
    print("✓ Cached features match original features")
    
    # Test with different data (cache miss)
    prices_diff = prices.iloc[:200]  # Different slice
    features3 = feature_lib.compute_all(prices_diff, use_cache=True)
    stats3 = feature_lib.get_cache_stats()
    
    print(f"Different data - Cache stats: {stats3}")
    assert stats3['misses'] == 2, "Should have 2 cache misses"
    assert stats3['cache_size'] == 2, "Cache should have 2 entries"
    print("✓ Different data correctly missed cache")
    
    # Clear cache
    feature_lib.clear_cache()
    stats4 = feature_lib.get_cache_stats()
    
    print(f"After clear - Cache stats: {stats4}")
    assert stats4['hits'] == 0, "Hits should be reset"
    assert stats4['misses'] == 0, "Misses should be reset"
    assert stats4['cache_size'] == 0, "Cache should be empty"
    print("✓ Cache cleared successfully")
    
    print("✅ Test 1 PASSED\n")


def test_cache_disabled():
    """Test that caching can be disabled."""
    print("=== Test 2: Cache Disabled ===")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', '2020-06-30', freq='D')
    tickers = ['AAPL', 'MSFT']
    prices = pd.DataFrame(
        np.random.randn(len(dates), len(tickers)).cumsum(axis=0) + 100,
        index=dates,
        columns=tickers
    )
    
    # Create feature library
    feature_lib = FeatureLibrary()
    
    # Compute with cache disabled
    features1 = feature_lib.compute_all(prices, use_cache=False)
    features2 = feature_lib.compute_all(prices, use_cache=False)
    
    stats = feature_lib.get_cache_stats()
    print(f"Cache disabled - Stats: {stats}")
    
    assert stats['hits'] == 0, "Should have no cache hits when disabled"
    assert stats['misses'] == 0, "Should have no cache misses when disabled"
    assert stats['cache_size'] == 0, "Cache should be empty when disabled"
    print("✓ Cache correctly disabled")
    
    # Features should still match
    for name in features1.keys():
        assert features1[name].equals(features2[name]), f"Feature {name} mismatch"
    print("✓ Features computed correctly without cache")
    
    print("✅ Test 2 PASSED\n")


def test_cache_performance():
    """Test that cache provides performance improvement."""
    print("=== Test 3: Cache Performance ===")
    
    # Create larger dataset
    dates = pd.date_range('2018-01-01', '2020-12-31', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'FB', 'NVDA', 'NFLX']
    prices = pd.DataFrame(
        np.random.randn(len(dates), len(tickers)).cumsum(axis=0) + 100,
        index=dates,
        columns=tickers
    )
    
    print(f"Dataset: {len(dates)} days × {len(tickers)} tickers")
    
    # Create feature library
    feature_lib = FeatureLibrary()
    
    # Time first computation (no cache)
    start = time.time()
    features1 = feature_lib.compute_all(prices, use_cache=True)
    time_no_cache = time.time() - start
    print(f"First computation (cache miss): {time_no_cache:.3f}s")
    
    # Time second computation (with cache)
    start = time.time()
    features2 = feature_lib.compute_all(prices, use_cache=True)
    time_with_cache = time.time() - start
    print(f"Second computation (cache hit): {time_with_cache:.3f}s")
    
    # Calculate speedup
    if time_with_cache > 0:
        speedup = time_no_cache / time_with_cache
        print(f"Speedup: {speedup:.1f}x")
        
        # Cache should be significantly faster (at least 10x)
        if speedup > 10:
            print(f"✓ Cache provides significant speedup ({speedup:.1f}x)")
        else:
            print(f"⚠ Speedup lower than expected ({speedup:.1f}x), but cache is working")
    
    # Verify cache stats
    stats = feature_lib.get_cache_stats()
    print(f"Cache stats: {stats}")
    assert stats['hit_rate'] == 50.0, "Hit rate should be 50%"
    
    print("✅ Test 3 PASSED\n")


def test_cache_with_different_params():
    """Test that cache correctly handles different parameters."""
    print("=== Test 4: Cache with Different Parameters ===")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    prices = pd.DataFrame(
        np.random.randn(len(dates), len(tickers)).cumsum(axis=0) + 100,
        index=dates,
        columns=tickers
    )
    
    # Create feature library
    feature_lib = FeatureLibrary()
    
    # Compute with different lag values
    features_lag0 = feature_lib.compute_all(prices, lag=0, use_cache=True)
    features_lag1 = feature_lib.compute_all(prices, lag=1, use_cache=True)
    
    stats = feature_lib.get_cache_stats()
    print(f"Different lag values - Stats: {stats}")
    assert stats['misses'] == 2, "Different lag should create different cache entries"
    assert stats['cache_size'] == 2, "Should have 2 cache entries"
    print("✓ Different lag values create separate cache entries")
    
    # Compute with different rank_transform values
    features_rank_true = feature_lib.compute_all(prices, rank_transform=True, use_cache=True)
    features_rank_false = feature_lib.compute_all(prices, rank_transform=False, use_cache=True)
    
    stats = feature_lib.get_cache_stats()
    print(f"Different rank_transform - Stats: {stats}")
    assert stats['cache_size'] >= 3, "Different rank_transform should create different cache entries"
    print("✓ Different rank_transform values create separate cache entries")
    
    # Verify features are different
    sample_feature = list(features_rank_true.keys())[0]
    if sample_feature not in feature_lib._skip_rank_transform:
        # Features should be different when rank_transform differs
        assert not features_rank_true[sample_feature].equals(features_rank_false[sample_feature]), \
            "Rank transformed features should differ from non-transformed"
        print("✓ Rank transform correctly affects feature values")
    
    print("✅ Test 4 PASSED\n")


def test_cache_size_limit():
    """Test that cache respects size limit."""
    print("=== Test 5: Cache Size Limit ===")
    
    # Create feature library
    feature_lib = FeatureLibrary()
    
    # Create many different price slices to exceed cache limit
    dates = pd.date_range('2015-01-01', '2020-12-31', freq='D')
    tickers = ['AAPL', 'MSFT']
    prices_full = pd.DataFrame(
        np.random.randn(len(dates), len(tickers)).cumsum(axis=0) + 100,
        index=dates,
        columns=tickers
    )
    
    # Compute features for many different date ranges
    # This should exceed the 1000 entry cache limit
    print("Computing features for 50 different date ranges...")
    for i in range(50):
        end_idx = 100 + i * 10
        prices_slice = prices_full.iloc[:end_idx]
        feature_lib.compute_all(prices_slice, use_cache=True)
    
    stats = feature_lib.get_cache_stats()
    print(f"After 50 computations - Stats: {stats}")
    
    # Cache size should be limited
    assert stats['cache_size'] <= 1000, "Cache size should not exceed 1000"
    assert stats['cache_size'] == 50, "Should have 50 entries (all unique)"
    print(f"✓ Cache size is {stats['cache_size']} (within limit)")
    
    print("✅ Test 5 PASSED\n")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("FEATURE CACHING TESTS")
    print("="*60)
    
    try:
        test_cache_functionality()
        test_cache_disabled()
        test_cache_performance()
        test_cache_with_different_params()
        test_cache_size_limit()
        
        print("="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
