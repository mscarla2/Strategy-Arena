#!/usr/bin/env python3
"""
Pre-flight check before running full evolution.
Run this to confirm everything is wired correctly.
"""
import warnings
warnings.filterwarnings('ignore')
import sys, pandas as pd, numpy as np
sys.path.insert(0, '.')

print("=" * 60)
print("ALPHAGENE PRE-FLIGHT CHECK")
print("=" * 60)

# ── 1. Data ───────────────────────────────────────────────────
print("\n[1] Data")
prices = pd.read_parquet('data/cache/prices_master.parquet')
volume = pd.read_parquet('data/cache/volume_master.parquet')
print(f"  Prices:  {prices.shape}  ({prices.index[0].date()} → {prices.index[-1].date()})")
print(f"  Volume:  {volume.shape}")

from data.universe import get_oil_tradeable_tickers
tradeable = get_oil_tradeable_tickers()
in_prices = [t for t in tradeable if t in prices.columns]
print(f"  Tradeable tickers: {tradeable}")
print(f"  In prices:         {in_prices}")
assert len(in_prices) >= 4, "Need at least 4 tradeable tickers"
print("  ✓ Data OK")

# ── 2. Features ───────────────────────────────────────────────
print("\n[2] Features")
from evolution.gp import FeatureLibrary, TreeGenerator, GPStrategy
lib = FeatureLibrary(enable_oil=True, enable_smc=True, enable_sr=True)
print(f"  Total features: {len(lib.feature_names)}")

gen = TreeGenerator(lib.feature_names)
n_unique = 0
n_constant = 0
for i in range(10):
    tree = gen.random_tree(max_depth=3)
    s = GPStrategy(tree=tree, top_pct=50)
    scores = s.score_stocks(prices, volume=volume)
    tradeable_scores = scores[in_prices]
    if tradeable_scores.nunique() > 1:
        n_unique += 1
    else:
        n_constant += 1

print(f"  10 random trees: {n_unique} unique scores, {n_constant} constant")
assert n_unique >= 5, f"Too many constant-score trees: {n_constant}/10"
print("  ✓ Features OK")

# ── 3. top_pct sanity ─────────────────────────────────────────
print("\n[3] Portfolio selection")
for pct in [33, 50, 67, 83]:
    tree = gen.random_tree(max_depth=3)
    s = GPStrategy(tree=tree, top_pct=pct)
    scores = s.score_stocks(prices, volume=volume)
    tradeable_scores = scores[in_prices]
    n_select = max(1, int(len(tradeable_scores) * pct / 100))
    selected = tradeable_scores.nlargest(n_select).index.tolist()
    print(f"  top_pct={pct}% → {n_select} stocks selected: {selected}")
print("  ✓ Portfolio selection OK")

# ── 4. Walk-forward quick eval ────────────────────────────────
print("\n[4] Walk-forward evaluation (1 strategy, 3 periods)")
from evolution.walkforward import WalkForwardEvaluator
from evolution.gp_fitness import calculate_fitness_v2

periods = [
    ("2022-01-01", "2022-07-01", "2022-07-01", "2022-10-01"),
    ("2022-07-01", "2023-01-01", "2023-01-01", "2023-04-01"),
    ("2023-01-01", "2023-07-01", "2023-07-01", "2023-10-01"),
]

tree = gen.random_tree(max_depth=4)
strategy = GPStrategy(tree=tree, top_pct=50)

evaluator = WalkForwardEvaluator(
    prices=prices,
    periods=periods,
    benchmark_results=[{'total_return': 0.05, 'sharpe_ratio': 0.5}] * 3,
    tradeable_tickers=in_prices,
    volume=volume,
    use_fitness_v2=True,
    universe_type='oil_microcap',
)

result = evaluator.evaluate_strategy(strategy)
print(f"  Fitness: {result.total:.4f}")
print(f"  Sharpe:  {result.avg_sharpe:.3f}")
print(f"  Return:  {result.avg_return:.1%}")
print(f"  Periods: {result.n_periods}")
assert result.n_periods > 0, "No periods evaluated"
print("  ✓ Walk-forward OK")

# ── 5. Differentiated fitness ─────────────────────────────────
print("\n[5] Fitness differentiation (5 strategies)")
fitnesses = []
for i in range(5):
    tree = gen.random_tree(max_depth=4)
    s = GPStrategy(tree=tree, top_pct=50)
    r = evaluator.evaluate_strategy(s)
    fitnesses.append(r.total)
    print(f"  Strategy {i+1}: fit={r.total:+.4f}  formula={s.get_formula()[:40]}")

assert len(set(round(f, 4) for f in fitnesses)) > 1, "All strategies have identical fitness!"
print("  ✓ Fitness differentiation OK")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED — ready to run full evolution")
print("=" * 60)
