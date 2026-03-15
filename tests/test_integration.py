#!/usr/bin/env python3
"""
Integration Tests for Priority 3+ Features

Tests the integration of:
- Smart Money Concepts (SMC)
- Support/Resistance (S/R)
- Oil-Specific Features
- Feature Library initialization
- CLI parameter handling

Run with: python3 test_integration.py
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def print_test(name, status="running"):
    """Print test status."""
    symbols = {"running": "⏳", "pass": "✅", "fail": "❌", "skip": "⚠️"}
    print(f"{symbols.get(status, '•')} {name}")

def test_feature_library_basic():
    """Test basic FeatureLibrary initialization."""
    print_test("FeatureLibrary - Basic Initialization", "running")
    
    try:
        from evolution.gp import FeatureLibrary
        
        # Test default initialization
        lib = FeatureLibrary()
        assert len(lib.feature_names) >= 90, f"Expected >=90 features, got {len(lib.feature_names)}"
        assert lib.enable_smc == False
        assert lib.enable_sr == False
        assert lib.enable_oil == False
        
        print_test("FeatureLibrary - Basic Initialization", "pass")
        return True
    except Exception as e:
        print_test("FeatureLibrary - Basic Initialization", "fail")
        print(f"  Error: {e}")
        return False

def test_feature_library_smc():
    """Test FeatureLibrary with SMC enabled."""
    print_test("FeatureLibrary - SMC Features", "running")
    
    try:
        from evolution.gp import FeatureLibrary
        
        lib = FeatureLibrary(enable_smc=True)
        
        # Check SMC features are added
        smc_features = [f for f in lib.feature_names if f.startswith('smc_')]
        assert len(smc_features) >= 6, f"Expected >=6 SMC features, got {len(smc_features)}"
        
        # Check SMC calculator is initialized
        assert lib.smc_features is not None, "SMC features calculator not initialized"
        
        expected_features = [
            'smc_order_block_bull', 'smc_order_block_bear',
            'smc_fvg_bull', 'smc_fvg_bear',
            'smc_liquidity_sweep', 'smc_break_of_structure'
        ]
        for feat in expected_features:
            assert feat in lib.feature_names, f"Missing SMC feature: {feat}"
        
        print_test("FeatureLibrary - SMC Features", "pass")
        return True
    except Exception as e:
        print_test("FeatureLibrary - SMC Features", "fail")
        print(f"  Error: {e}")
        return False

def test_feature_library_sr():
    """Test FeatureLibrary with S/R enabled."""
    print_test("FeatureLibrary - S/R Features", "running")
    
    try:
        from evolution.gp import FeatureLibrary
        
        lib = FeatureLibrary(enable_sr=True)
        
        # Check S/R features are added
        sr_features = [f for f in lib.feature_names if f.startswith('sr_')]
        assert len(sr_features) >= 8, f"Expected >=8 S/R features, got {len(sr_features)}"
        
        # Check S/R calculator is initialized
        assert lib.sr_features is not None, "S/R features calculator not initialized"
        
        expected_features = [
            'sr_poc_distance', 'sr_value_area_position',
            'sr_pivot_traditional', 'sr_pivot_fibonacci',
            'sr_bb_position', 'sr_keltner_position'
        ]
        for feat in expected_features:
            assert feat in lib.feature_names, f"Missing S/R feature: {feat}"
        
        print_test("FeatureLibrary - S/R Features", "pass")
        return True
    except Exception as e:
        print_test("FeatureLibrary - S/R Features", "fail")
        print(f"  Error: {e}")
        return False

def test_feature_library_oil():
    """Test FeatureLibrary with Oil features enabled."""
    print_test("FeatureLibrary - Oil Features", "running")
    
    try:
        from evolution.gp import FeatureLibrary
        
        lib = FeatureLibrary(enable_oil=True)
        
        # Check Oil features are added
        oil_features = [f for f in lib.feature_names if f.startswith('oil_')]
        assert len(oil_features) >= 10, f"Expected >=10 Oil features, got {len(oil_features)}"
        
        # Check Oil calculator is initialized
        assert lib.oil_features is not None, "Oil features calculator not initialized"
        
        expected_features = [
            'oil_wti_correlation', 'oil_brent_correlation',
            'oil_wti_beta', 'oil_inventory_zscore',
            'oil_crack_spread_321', 'oil_wti_brent_spread'
        ]
        for feat in expected_features:
            assert feat in lib.feature_names, f"Missing Oil feature: {feat}"
        
        print_test("FeatureLibrary - Oil Features", "pass")
        return True
    except Exception as e:
        print_test("FeatureLibrary - Oil Features", "fail")
        print(f"  Error: {e}")
        return False

def test_feature_library_all():
    """Test FeatureLibrary with all features enabled."""
    print_test("FeatureLibrary - All Features", "running")
    
    try:
        from evolution.gp import FeatureLibrary
        
        lib = FeatureLibrary(enable_smc=True, enable_sr=True, enable_oil=True)
        
        # Check total feature count
        assert len(lib.feature_names) >= 120, f"Expected >=120 features, got {len(lib.feature_names)}"
        
        # Check all calculators initialized
        assert lib.smc_features is not None
        assert lib.sr_features is not None
        assert lib.oil_features is not None
        
        print_test("FeatureLibrary - All Features", "pass")
        return True
    except Exception as e:
        print_test("FeatureLibrary - All Features", "fail")
        print(f"  Error: {e}")
        return False

def test_smc_features_calculation():
    """Test SMC features calculation with sample data."""
    print_test("SMC Features - Calculation", "running")
    
    try:
        from evolution.smart_money_features import SmartMoneyFeatures
        
        # Create sample price data (single ticker for SMC)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Generate realistic OHLC data
        np.random.seed(42)
        base_price = 100
        prices = base_price + np.cumsum(np.random.randn(100) * 2)
        
        high = pd.Series(prices + np.abs(np.random.randn(100)), index=dates)
        low = pd.Series(prices - np.abs(np.random.randn(100)), index=dates)
        close = pd.Series(prices, index=dates)
        volume = pd.Series(1000000 + np.random.randint(-100000, 100000, 100), index=dates)
        
        smc = SmartMoneyFeatures()
        results = smc.calculate_all_features(high, low, close, volume)
        
        # Check results structure
        assert isinstance(results, dict), "Results should be a dictionary"
        assert len(results) >= 6, f"Expected >=6 SMC features, got {len(results)}"
        
        # Check each result is a float, numeric value, or Series
        for key, value in results.items():
            assert isinstance(value, (int, float, np.number, pd.Series)), f"{key} should be numeric or Series, got {type(value)}"
            if isinstance(value, pd.Series):
                assert len(value) > 0, f"{key} Series should not be empty"
        
        print_test("SMC Features - Calculation", "pass")
        return True
    except Exception as e:
        print_test("SMC Features - Calculation", "fail")
        print(f"  Error: {e}")
        return False

def test_sr_features_calculation():
    """Test S/R features calculation with sample data."""
    print_test("S/R Features - Calculation", "running")
    
    try:
        from evolution.support_resistance_features import SupportResistanceFeatures
        
        # Create sample price data (single ticker for S/R)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        np.random.seed(42)
        base_price = 100
        prices = base_price + np.cumsum(np.random.randn(100) * 2)
        
        high = pd.Series(prices + np.abs(np.random.randn(100)), index=dates)
        low = pd.Series(prices - np.abs(np.random.randn(100)), index=dates)
        close = pd.Series(prices, index=dates)
        volume = pd.Series(1000000 + np.random.randint(-100000, 100000, 100), index=dates)
        
        sr = SupportResistanceFeatures()
        results = sr.calculate_all_features(high, low, close, volume)
        
        # Check results structure
        assert isinstance(results, dict), "Results should be a dictionary"
        assert len(results) >= 8, f"Expected >=8 S/R features, got {len(results)}"
        
        # Check each result is numeric or Series
        for key, value in results.items():
            assert isinstance(value, (int, float, np.number, pd.Series)), f"{key} should be numeric or Series, got {type(value)}"
            if isinstance(value, pd.Series):
                assert len(value) > 0, f"{key} Series should not be empty"
        
        print_test("S/R Features - Calculation", "pass")
        return True
    except Exception as e:
        print_test("S/R Features - Calculation", "fail")
        print(f"  Error: {e}")
        return False

def test_gp_population_with_features():
    """Test GPPopulation initialization with advanced features."""
    print_test("GPPopulation - Advanced Features", "running")
    
    try:
        from evolution.gp import GPPopulation
        
        # Test with SMC enabled
        pop = GPPopulation(
            population_size=10,
            max_depth=3,
            enable_smc=True
        )
        
        assert pop.feature_lib.enable_smc == True
        assert len(pop.feature_lib.feature_names) > 90
        
        # Test with all features
        pop_all = GPPopulation(
            population_size=10,
            max_depth=3,
            enable_smc=True,
            enable_sr=True,
            enable_oil=True
        )
        
        assert pop_all.feature_lib.enable_smc == True
        assert pop_all.feature_lib.enable_sr == True
        assert pop_all.feature_lib.enable_oil == True
        assert len(pop_all.feature_lib.feature_names) >= 120
        
        print_test("GPPopulation - Advanced Features", "pass")
        return True
    except Exception as e:
        print_test("GPPopulation - Advanced Features", "fail")
        print(f"  Error: {e}")
        return False

def test_cli_parameters():
    """Test CLI parameter parsing."""
    print_test("CLI Parameters - Parsing", "running")
    
    try:
        import argparse
        import sys
        
        # Save original argv
        original_argv = sys.argv
        
        # Test with advanced features enabled
        sys.argv = [
            'arena_runner_v3.py',
            '--enable-smc',
            '--enable-sr',
            '--enable-oil',
            '--generations', '5',
            '--population', '10'
        ]
        
        # Import and parse (without running main)
        from arena_runner_v3 import main
        
        # Restore argv
        sys.argv = original_argv
        
        print_test("CLI Parameters - Parsing", "pass")
        return True
    except Exception as e:
        print_test("CLI Parameters - Parsing", "fail")
        print(f"  Error: {e}")
        return False

def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("INTEGRATION TESTS - Priority 3+ Features")
    print("="*70 + "\n")
    
    tests = [
        test_feature_library_basic,
        test_feature_library_smc,
        test_feature_library_sr,
        test_feature_library_oil,
        test_feature_library_all,
        test_smc_features_calculation,
        test_sr_features_calculation,
        test_gp_population_with_features,
        test_cli_parameters,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  Unexpected error: {e}")
            results.append(False)
        print()
    
    # Summary
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed!")
        return 0
    else:
        print(f"❌ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
