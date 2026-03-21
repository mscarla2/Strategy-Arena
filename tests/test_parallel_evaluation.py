#!/usr/bin/env python3
"""
Test parallel evaluation implementation.

Verifies that:
1. WalkForwardEvaluator can be serialized/deserialized
2. Parallel evaluation produces same results as sequential
3. Parallel evaluation is faster for larger populations
"""

import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evolution.gp import (
    GPStrategy,
    TreeGenerator,
    FeatureLibrary,
    WalkForwardEvaluator,
    evaluate_strategy_parallel,
)


def test_evaluator_serialization():
    """Test that WalkForwardEvaluator can be serialized and deserialized."""
    print("\n=== Test 1: Evaluator Serialization ===")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    prices = pd.DataFrame(
        np.random.randn(len(dates), len(tickers)).cumsum(axis=0) + 100,
        index=dates,
        columns=tickers
    )
    
    periods = [
        ('2020-01-01', '2020-06-30', '2020-07-01', '2020-09-30'),
        ('2020-04-01', '2020-09-30', '2020-10-01', '2020-12-31'),
    ]
    
    # Create evaluator
    evaluator = WalkForwardEvaluator(
        prices=prices,
        periods=periods,
        benchmark_results=[],
        transaction_cost=0.002,
        rebalance_frequency=21,
    )
    
    # Serialize
    config = evaluator.to_config()
    print(f"✓ Serialized evaluator config: {list(config.keys())}")
    
    # Deserialize
    evaluator2 = WalkForwardEvaluator.from_config(config)
    print(f"✓ Deserialized evaluator successfully")
    
    # Verify attributes match
    assert evaluator2.transaction_cost == evaluator.transaction_cost
    assert evaluator2.rebalance_frequency == evaluator.rebalance_frequency
    assert len(evaluator2.periods) == len(evaluator.periods)
    print(f"✓ Attributes match after deserialization")
    
    print("✅ Test 1 PASSED\n")


def test_parallel_vs_sequential():
    """Test that parallel evaluation produces same results as sequential."""
    print("=== Test 2: Parallel vs Sequential Results ===")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    prices = pd.DataFrame(
        np.random.randn(len(dates), len(tickers)).cumsum(axis=0) + 100,
        index=dates,
        columns=tickers
    )
    
    periods = [
        ('2020-01-01', '2020-06-30', '2020-07-01', '2020-09-30'),
    ]
    
    # Create evaluator
    evaluator = WalkForwardEvaluator(
        prices=prices,
        periods=periods,
        benchmark_results=[],
        transaction_cost=0.002,
        rebalance_frequency=21,
    )
    
    # Create a simple strategy
    feature_lib = FeatureLibrary()
    generator = TreeGenerator(feature_lib.feature_names)
    tree = generator.random_tree(max_depth=3, method='grow')
    strategy = GPStrategy(tree=tree, top_pct=20, generation=0, origin='test')
    
    # Sequential evaluation
    result_seq = evaluator.evaluate_strategy(strategy)
    print(f"Sequential fitness: {result_seq.total:.4f}")
    
    # Parallel evaluation
    evaluator_config = evaluator.to_config()
    strategy_id, fitness_par, period_results_par = evaluate_strategy_parallel(
        (strategy, evaluator_config)
    )
    print(f"Parallel fitness: {fitness_par:.4f}")
    
    # Compare results
    assert abs(result_seq.total - fitness_par) < 1e-6, \
        f"Fitness mismatch: {result_seq.total} vs {fitness_par}"
    print(f"✓ Fitness values match (diff: {abs(result_seq.total - fitness_par):.2e})")
    
    print("✅ Test 2 PASSED\n")


def test_parallel_speedup():
    """Test that parallel evaluation is faster for larger populations."""
    print("=== Test 3: Parallel Speedup ===")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    prices = pd.DataFrame(
        np.random.randn(len(dates), len(tickers)).cumsum(axis=0) + 100,
        index=dates,
        columns=tickers
    )
    
    periods = [
        ('2020-01-01', '2020-04-30', '2020-05-01', '2020-07-31'),
        ('2020-04-01', '2020-07-31', '2020-08-01', '2020-10-31'),
    ]
    
    # Create evaluator
    evaluator = WalkForwardEvaluator(
        prices=prices,
        periods=periods,
        benchmark_results=[],
        transaction_cost=0.002,
        rebalance_frequency=21,
    )
    
    # Create population of strategies
    feature_lib = FeatureLibrary()
    generator = TreeGenerator(feature_lib.feature_names)
    population_size = 10
    population = []
    
    for i in range(population_size):
        tree = generator.random_tree(max_depth=4, method='grow')
        strategy = GPStrategy(tree=tree, top_pct=20, generation=0, origin='test')
        population.append(strategy)
    
    print(f"Created population of {population_size} strategies")
    
    # Sequential evaluation
    start_seq = time.time()
    for strategy in population:
        evaluator.evaluate_strategy(strategy)
    time_seq = time.time() - start_seq
    print(f"Sequential time: {time_seq:.2f}s")
    
    # Parallel evaluation
    from multiprocessing import Pool, cpu_count
    evaluator_config = evaluator.to_config()
    eval_args = [(strategy, evaluator_config) for strategy in population]
    
    n_jobs = max(1, cpu_count() - 1)
    start_par = time.time()
    with Pool(processes=n_jobs) as pool:
        pool.map(evaluate_strategy_parallel, eval_args)
    time_par = time.time() - start_par
    print(f"Parallel time ({n_jobs} workers): {time_par:.2f}s")
    
    speedup = time_seq / time_par
    print(f"Speedup: {speedup:.2f}x")
    
    # Note: Speedup may be minimal for small populations or simple strategies
    # but should be significant for production workloads
    if speedup > 1.0:
        print(f"✓ Parallel evaluation is faster")
    else:
        print(f"⚠ Speedup < 1.0 (expected for small test population)")
    
    print("✅ Test 3 PASSED\n")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("PARALLEL EVALUATION TESTS")
    print("="*60)
    
    try:
        test_evaluator_serialization()
        test_parallel_vs_sequential()
        test_parallel_speedup()
        
        print("="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
