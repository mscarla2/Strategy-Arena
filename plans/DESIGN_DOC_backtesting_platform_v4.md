# ALPHAGENE Backtesting Platform v4.0 — Design Document

**Authors:** Engineering Lead (reviewer), Senior Engineer (author)
**Date:** 2026-03-21
**Status:** DRAFT — Ready for L4 Implementation
**Priority:** P0 — Management Critical Path

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Analysis](#2-current-state-analysis)
3. [Root Cause Analysis: Fitness Ceiling at 0.59](#3-root-cause-analysis-fitness-ceiling-at-059)
4. [Workstream 1: Walk-Forward Evaluator Overhaul](#4-workstream-1-walk-forward-evaluator-overhaul)
5. [Workstream 2: Feature Library Expansion (Market-Relative & Macro)](#5-workstream-2-feature-library-expansion)
6. [Workstream 3: Fitness Function Redesign](#6-workstream-3-fitness-function-redesign)
7. [Workstream 4: Oil-Universe Specialization](#7-workstream-4-oil-universe-specialization)
8. [Workstream 5: Dashboard & UI for Management](#8-workstream-5-dashboard--ui-for-management)
9. [Implementation Plan & Sequencing](#9-implementation-plan--sequencing)
10. [Testing Strategy](#10-testing-strategy)
11. [Risk Register](#11-risk-register)
12. [Appendix: File-Level Change Map](#appendix-file-level-change-map)

---

## 1. Executive Summary

### The Problem

Despite millions invested, our GP-evolved strategies have **never exceeded a fitness of 0.59**. Management is specifically focused on **oil stock performance optimization** and has flagged that **March 2026 data is underrepresented** in our backtest window. Our quant researchers have identified a **complete absence of market-relative features** (no beta, no idiosyncratic vol, no sector correlation) as the single largest structural gap.

### The Solution

This document specifies a **five-workstream overhaul** targeting the root causes of the fitness ceiling:

| # | Workstream | Impact | Effort |
|---|-----------|--------|--------|
| 1 | Walk-Forward Evaluator Overhaul | **Critical** — fixes backtest window & recency bias | M |
| 2 | Feature Library Expansion (~90 → ~140 features) | **Critical** — adds market-relative & macro features | L |
| 3 | Fitness Function Redesign | **High** — removes artificial ceiling, adds recency weighting | M |
| 4 | Oil-Universe Specialization | **High** — management's top priority | M |
| 5 | Dashboard & UI | **Medium** — management visibility | M |

**Expected outcome:** Break through the 0.59 fitness ceiling to **0.70+** within 2 sprint cycles.

---

## 2. Current State Analysis

### 2.1 Architecture Overview

```
arena_runner_v3.py          ← CLI entry point, orchestrates everything
├── config.py               ← Global settings, walk-forward periods, costs
├── data/
│   ├── fetcher.py          ← yfinance download + parquet cache
│   ├── universe.py         ← S&P 500 + MidCap + Oil universe definitions
│   └── cache/              ← Parquet price cache
├── evolution/
│   ├── gp.py               ← MONOLITH: Nodes, Trees, FeatureLibrary, Fitness,
│   │                          WalkForwardEvaluator, GPStrategy, GPPopulation (2313 lines)
│   ├── gp_storage.py       ← SQLite serialization for strategies
│   ├── oil_specific_features.py  ← Oil market data features
│   ├── regime_detection.py ← Vol/trend/market/correlation regime detection
│   ├── fitness_calmar.py   ← Alternative Calmar-based fitness
│   ├── smart_money_features.py
│   ├── support_resistance_features.py
│   └── diversity.py, multi_objective.py
├── backtest/
│   ├── risk_management.py  ← Slippage, dilution filter, liquidity constraints
│   ├── rebalancing.py      ← Partial rebalancer
│   ├── stops.py            ← Trailing volatility stops
│   ├── position_sizing.py  ← Kelly + vol-target sizing
│   └── multi_timeframe.py  ← Timeframe configs
├── ui/
│   ├── dashboard.py        ← Streamlit app (basic)
│   ├── charts.py           ← Plotly chart builders
│   └── strategy_picker.py  ← CLI strategy viewer
└── scripts/
    └── run_arena_oil.sh    ← Oil-specific run script
```

### 2.2 Current Feature Library Inventory

The [`FeatureLibrary`](evolution/gp.py:491) currently provides **~90 features** across these categories:

| Category | Count | Examples |
|----------|-------|---------|
| Momentum | 16 | `mom_5d` through `mom_252d`, skip-month, acceleration, consistency |
| Mean Reversion | 11 | `zscore_*`, `dist_ma_*`, `reversion_speed_*`, `rsi_*` |
| Volatility | 10 | `vol_*`, `vol_ratio_*`, `vol_of_vol_*`, `vol_trend_*` |
| Drawdown/Risk | 11 | `drawdown_*`, `recovery_rate_*`, `max_dd_*`, `ulcer_index_*` |
| Higher Moments | 9 | `skew_*`, `kurt_*`, `downside_dev_*`, `left_tail_*` |
| Trend Quality | 8 | `trend_r2_*`, `trend_slope_*`, `hurst_proxy_*` |
| Price Level | 8 | `range_position_*`, `breakout_*`, `support/resistance_distance_*` |
| Cross-Sectional | 8 | `rel_strength_*`, `rel_vol_*`, `excess_mom_*`, `rel_value_*` |
| Risk-Adjusted | 8 | `sharpe_*`, `sortino_*`, `calmar_*`, `info_ratio_*` |
| Volume | 4 | `volume_trend_*`, `volume_ratio_*`, `price_volume_corr_*` |
| Pattern | 6 | `gap_reversal_*`, `streak_strength`, `return_autocorr_*` |
| **Oil-Specific** | 10 | `oil_wti_correlation`, `oil_inventory_zscore`, `oil_crack_spread_*` |
| **SMC** | 6 | `smc_order_block_*`, `smc_fvg_*`, `smc_liquidity_sweep` |
| **S/R** | 8 | `sr_poc_distance`, `sr_pivot_*`, `sr_bb_position` |

### 2.3 Current Walk-Forward Configuration

From [`arena_runner_v3.py:418`](arena_runner_v3.py:418):

```python
def _generate_walk_forward_periods(
    self,
    train_months: int = 12,   # 1 year training
    test_months: int = 3,     # 3 month test
    step_months: int = 3      # 3 month step
)
```

The oil script uses **6-month train / 3-month test / 3-month step** (see [`scripts/run_arena_oil.sh:60`](scripts/run_arena_oil.sh:60)).

### 2.4 Current Fitness Function

From [`calculate_fitness()`](evolution/gp.py:1648):

```
total = 0.30 * sharpe_component     # Excess Sharpe vs benchmark
      + 0.25 * return_component     # Excess return + IR bonus
      + 0.30 * stability_component  # Win rate - DD penalty - consistency penalty
      - 0.15 * cost_penalty          # Turnover × transaction cost
```

Clipped to `[-1, 1]`. Period count adjustment: `if n < 4: total *= n / 4`.

---

## 3. Root Cause Analysis: Fitness Ceiling at 0.59

After thorough analysis, the fitness ceiling has **five compounding root causes**:

### RC-1: Walk-Forward Window Does Not Cover March 2026

**Evidence:** Today is 2026-03-21. The walk-forward generator in [`_generate_walk_forward_periods()`](arena_runner_v3.py:418) creates periods by stepping `step_months` from `start_date`. With `start=2018-01-01` and `train=6mo, test=3mo, step=3mo`, the **last test period ends around 2025-12-31 or 2026-01-01** — it never reaches March 2026.

**Impact:** The most recent and most important data (per management) is **completely excluded** from fitness evaluation. Strategies are optimized for stale market conditions.

**Fix:** Workstream 1 — Add a "most-recent" anchored period and recency weighting.

### RC-2: No Market-Relative Features (Beta, Idiosyncratic Vol, Sector Correlation)

**Evidence:** From the feature importance analysis, `oil_inventory_zscore` (a macro feature) achieved 0.549 fitness with only 1 usage — the highest single-use fitness. Yet the library has **zero** features for:
- Market beta (time-varying)
- Idiosyncratic volatility (residual after market factor)
- Sector/universe correlation
- Beta instability (regime change signal)

**Impact:** The GP cannot distinguish "stock moved because oil moved" from "stock moved despite oil." This is the **#1 structural gap** per our quant researchers.

**Fix:** Workstream 2 — Add 50+ market-relative, microstructure, and regime features.

### RC-3: Fitness Function Has Artificial Ceiling Mechanics

**Evidence:** The fitness function clips to `[-1, 1]` and applies multiple penalties that compound:
- Sharpe variance penalty: up to 0.3
- Drawdown penalty: up to 1.0 (at 0.50 weight = 0.50 effective)
- Worst return penalty: 0.3 (at 0.25 weight = 0.075 effective)
- Consistency penalty: up to 0.4 (at 0.25 weight = 0.10 effective)
- Cost penalty: up to 1.0 (at 0.15 weight = 0.15 effective)

A strategy with **excellent** excess Sharpe (1.0 → component = 1.0) and **excellent** excess return (10% → component = 1.0) but **any** drawdown > 25% gets stability_component ≈ -0.1, yielding:

```
total = 0.30 * 0.7 + 0.25 * 1.0 + 0.30 * (-0.1) - 0.15 * 0.2
      = 0.21 + 0.25 - 0.03 - 0.03 = 0.40
```

Even a near-perfect strategy struggles to exceed 0.60 because the penalty structure is **too aggressive** for oil microcaps where 25%+ drawdowns are normal.

**Fix:** Workstream 3 — Recalibrate penalty thresholds for oil universe, add recency weighting.

### RC-4: Oil Universe Is Too Small (8 Tickers) for Cross-Sectional Features

**Evidence:** [`OIL_FOCUSED_UNIVERSE`](data/universe.py:239) contains only 8 tickers (6 microcaps + USO + BNO). Cross-sectional features like `rel_strength_*` and `excess_mom_*` need a meaningful cross-section to rank against. With 8 tickers, rank transforms produce coarse 12.5% steps.

**Impact:** Cross-sectional features are nearly useless. The GP is forced to rely on time-series features only.

**Fix:** Workstream 4 — Expand oil universe to include mid/large-cap energy (XOM, CVX, COP, etc.) as a reference panel.

### RC-5: Feature Pre-computation Is a Placeholder

**Evidence:** [`_precompute_features()`](arena_runner_v3.py:516) prints "placeholder" and does nothing. Features are recomputed from scratch for every strategy × every period × every rebalance date. This makes evolution painfully slow, limiting the number of generations and population size we can run.

**Impact:** Slow evaluation → fewer generations → less evolutionary pressure → lower fitness.

**Fix:** Workstream 1 — Implement proper feature pre-computation with period-indexed caching.

---

## 4. Workstream 1: Walk-Forward Evaluator Overhaul

### 4.1 Problem Statement

The current [`WalkForwardEvaluator`](evolution/gp.py:1912) has three deficiencies:
1. **No coverage of the most recent period** (March 2026)
2. **All periods weighted equally** — no recency bias
3. **No expanding-window option** — only fixed-window walk-forward

### 4.2 Design: Anchored Recent Period

Add a mandatory "anchored" test period that always includes the most recent data:

```python
# In _generate_walk_forward_periods():
# After generating standard periods, append an anchored recent period

# Standard periods: rolling walk-forward
periods = [...]  # existing logic

# Anchored recent period: last N months as test, preceding M months as train
recent_test_end = pd.Timestamp(self.end_date)
recent_test_start = recent_test_end - pd.DateOffset(months=test_months)
recent_train_end = recent_test_start
recent_train_start = recent_train_end - pd.DateOffset(months=train_months)

periods.append((
    recent_train_start.strftime("%Y-%m-%d"),
    recent_train_end.strftime("%Y-%m-%d"),
    recent_test_start.strftime("%Y-%m-%d"),
    recent_test_end.strftime("%Y-%m-%d"),
))
```

This guarantees March 2026 is always in the evaluation window.

### 4.3 Design: Recency-Weighted Period Scoring

Instead of `np.mean(excess_returns)`, use exponentially-weighted mean:

```python
def _recency_weighted_mean(values: List[float], half_life_periods: int = 4) -> float:
    """Exponentially-weighted mean giving more weight to recent periods."""
    n = len(values)
    if n == 0:
        return 0.0
    
    # Weights: most recent period gets weight 1.0, decays by half_life
    decay = np.log(2) / half_life_periods
    weights = np.array([np.exp(-decay * (n - 1 - i)) for i in range(n)])
    weights /= weights.sum()
    
    return np.dot(values, weights)
```

**Configurable half-life:** Default 4 periods (1 year with quarterly steps). Management can tune this to emphasize recent performance more aggressively.

### 4.4 Design: Expanding Window Option

Add an `expanding_window` mode where training data grows over time (all history up to test start):

```python
if self.expanding_window:
    # Training uses ALL data from start to test_start
    train_start = self.prices.index[0].strftime("%Y-%m-%d")
else:
    # Fixed window: train_start as generated
    train_start = period[0]
```

This is critical for oil stocks where regime changes (COVID crash, 2022 energy crisis) provide valuable training signal that fixed windows may exclude.

### 4.5 Design: Feature Pre-computation Cache

Replace the placeholder with a proper period-indexed feature cache:

```python
class FeaturePrecomputeCache:
    """Pre-compute features for all rebalance dates across all periods."""
    
    def __init__(self, feature_lib: FeatureLibrary, prices: pd.DataFrame,
                 volume: pd.DataFrame, periods: List[Tuple], 
                 rebalance_frequency: int = 21):
        self.cache: Dict[str, Dict[str, pd.Series]] = {}
        self._precompute(feature_lib, prices, volume, periods, rebalance_frequency)
    
    def _precompute(self, ...):
        """Compute features for every rebalance date in every test period."""
        all_rebalance_dates = set()
        for _, _, test_start, test_end in periods:
            test_dates = prices.loc[test_start:test_end].index
            for i, date in enumerate(test_dates):
                if i % rebalance_frequency == 0:
                    all_rebalance_dates.add(date)
        
        for date in sorted(all_rebalance_dates):
            date_key = date.strftime("%Y-%m-%d")
            prices_to_date = prices.loc[:date]
            volume_to_date = volume.loc[:date] if volume is not None else None
            self.cache[date_key] = feature_lib.compute_all(
                prices_to_date, volume_to_date, lag=1, rank_transform=True
            )
    
    def get_features(self, date: str) -> Dict[str, pd.Series]:
        return self.cache.get(date, {})
```

**Expected speedup:** 10-50× for population evaluation (features computed once, shared across all strategies).

### 4.6 Files Changed

| File | Change |
|------|--------|
| [`evolution/gp.py`](evolution/gp.py:1912) | Refactor `WalkForwardEvaluator` — add `expanding_window`, `recency_half_life`, anchored period |
| [`arena_runner_v3.py`](arena_runner_v3.py:418) | Update `_generate_walk_forward_periods()` to append anchored period |
| [`arena_runner_v3.py`](arena_runner_v3.py:516) | Replace `_precompute_features()` placeholder with `FeaturePrecomputeCache` |
| [`config.py`](config.py) | Add `RECENCY_HALF_LIFE`, `USE_EXPANDING_WINDOW` settings |
| **NEW** `evolution/feature_cache.py` | Extract `FeaturePrecomputeCache` class |

---

## 5. Workstream 2: Feature Library Expansion

### 5.1 Problem Statement

The current [`FeatureLibrary`](evolution/gp.py:491) is **100% price-derived** (except for the optional oil features). It completely lacks:
- Market-relative features (beta, idiosyncratic vol)
- Microstructure features (Amihud illiquidity, Kyle's lambda)
- Regime detection features (vol regime, trend strength, correlation spike)
- Feature engineering on existing features (momentum Sharpe, efficiency ratio)

### 5.2 New Feature Groups

#### 5.2.1 Market-Relative Features (Priority 1 — Highest Impact)

These separate stock-specific alpha from market/oil beta. **This is the single most impactful change.**

```python
# In FeatureLibrary._build_feature_specs():

# ═══════════════════════════════════════════════════════════════
# MARKET-RELATIVE FEATURES (NEW)
# ═══════════════════════════════════════════════════════════════

# Time-varying market beta
specs['market_beta_21d'] = ('market_beta', 21)
specs['market_beta_63d'] = ('market_beta', 63)
specs['market_beta_126d'] = ('market_beta', 126)

# Beta instability (rolling beta std — regime change signal)
specs['beta_instability_63d'] = ('beta_instability', 63)

# Idiosyncratic volatility (residual vol after removing market factor)
specs['idio_vol_21d'] = ('idiosyncratic_vol', 21)
specs['idio_vol_63d'] = ('idiosyncratic_vol', 63)

# Sector/universe correlation
specs['sector_correlation_21d'] = ('sector_correlation', 21)
specs['sector_correlation_63d'] = ('sector_correlation', 63)

# Universe return dispersion (cross-sectional vol)
specs['universe_dispersion_21d'] = ('universe_dispersion', 21)
```

**Implementation for `market_beta`:**

```python
elif feature_type == 'market_beta':
    period = spec[1]
    if len(prices) < period + 1:
        return pd.Series(1.0, index=tickers)
    
    returns = prices.pct_change().iloc[-period:]
    market_returns = returns.mean(axis=1)  # Equal-weight market proxy
    
    betas = {}
    for ticker in tickers:
        cov = returns[ticker].cov(market_returns)
        var = market_returns.var()
        betas[ticker] = cov / var if var > 0 else 1.0
    
    return pd.Series(betas)
```

**Implementation for `idiosyncratic_vol`:**

```python
elif feature_type == 'idiosyncratic_vol':
    period = spec[1]
    if len(prices) < period + 1:
        return pd.Series(0.0, index=tickers)
    
    returns = prices.pct_change().iloc[-period:]
    market_returns = returns.mean(axis=1)
    
    idio_vols = {}
    for ticker in tickers:
        # Regress stock returns on market returns
        cov = returns[ticker].cov(market_returns)
        var = market_returns.var()
        beta = cov / var if var > 0 else 1.0
        
        # Residual = stock return - beta * market return
        residual = returns[ticker] - beta * market_returns
        idio_vols[ticker] = residual.std() * np.sqrt(252)
    
    return pd.Series(idio_vols)
```

#### 5.2.2 Microstructure & Liquidity Features (Priority 2)

```python
# ═══════════════════════════════════════════════════════════════
# MICROSTRUCTURE FEATURES (NEW)
# ═══════════════════════════════════════════════════════════════

# Amihud illiquidity ratio (|return| / dollar volume)
specs['amihud_21d'] = ('amihud_illiquidity', 21)
specs['amihud_63d'] = ('amihud_illiquidity', 63)

# Roll spread proxy (bid-ask spread estimate from close prices)
specs['roll_spread_21d'] = ('roll_spread', 21)

# Kyle's lambda proxy (price impact per unit volume)
specs['kyle_lambda_21d'] = ('kyle_lambda', 21)

# Zero-return days fraction (illiquidity signal)
specs['zero_return_days_21d'] = ('zero_return_fraction', 21)
specs['zero_return_days_63d'] = ('zero_return_fraction', 63)

# Turnover rate
specs['turnover_rate_21d'] = ('turnover_rate', 21)
```

**Implementation for `amihud_illiquidity`:**

```python
elif feature_type == 'amihud_illiquidity':
    period = spec[1]
    if volume is None or len(prices) < period:
        return pd.Series(0.0, index=tickers)
    
    returns = prices.pct_change().iloc[-period:]
    dollar_volume = (prices * volume).iloc[-period:] if volume is not None else None
    
    if dollar_volume is None:
        return pd.Series(0.0, index=tickers)
    
    # Amihud = mean(|return| / dollar_volume)
    amihud = (returns.abs() / dollar_volume.replace(0, np.nan)).mean()
    return amihud.fillna(0)
```

#### 5.2.3 Regime Detection Features (Priority 3)

```python
# ═══════════════════════════════════════════════════════════════
# REGIME FEATURES (NEW)
# ═══════════════════════════════════════════════════════════════

# Volatility regime (percentile of current vol vs history)
specs['vol_regime'] = ('volatility_regime', 63)

# Trend strength (ADX proxy)
specs['trend_strength_21d'] = ('adx_proxy', 21)
specs['trend_strength_63d'] = ('adx_proxy', 63)

# Correlation spike (sudden increase in cross-asset correlation)
specs['correlation_spike_21d'] = ('correlation_spike', 21)

# Market breadth (% of universe above N-day MA)
specs['breadth_50d'] = ('market_breadth', 50)
specs['breadth_200d'] = ('market_breadth', 200)
```

#### 5.2.4 Feature Engineering on Existing Features (Priority 4)

```python
# ═══════════════════════════════════════════════════════════════
# ENGINEERED FEATURES (NEW)
# ═══════════════════════════════════════════════════════════════

# Momentum quality: Sharpe of momentum (smooth vs choppy gains)
specs['mom_sharpe_21d'] = ('momentum_sharpe', 21)
specs['mom_sharpe_63d'] = ('momentum_sharpe', 63)

# Drawdown-adjusted momentum
specs['dd_adj_mom_63d'] = ('drawdown_adjusted_momentum', 63)

# Vol-normalized momentum
specs['vol_adj_mom_21d'] = ('vol_adjusted_momentum', 21)
specs['vol_adj_mom_63d'] = ('vol_adjusted_momentum', 63)

# Price efficiency ratio (Kaufman AMA style: net move / total path)
specs['efficiency_ratio_21d'] = ('efficiency_ratio', 21)
specs['efficiency_ratio_63d'] = ('efficiency_ratio', 63)

# Fractal dimension proxy
specs['fractal_dim_63d'] = ('fractal_dimension', 63)
```

**Implementation for `efficiency_ratio`:**

```python
elif feature_type == 'efficiency_ratio':
    period = spec[1]
    if len(prices) < period + 1:
        return pd.Series(0.5, index=tickers)
    
    # Net price change (direction)
    net_change = (prices.iloc[-1] - prices.iloc[-period - 1]).abs()
    
    # Total path length (sum of absolute daily changes)
    daily_changes = prices.diff().iloc[-period:].abs().sum()
    
    # Efficiency = net / path (1.0 = perfectly trending, 0.0 = choppy)
    efficiency = net_change / daily_changes.replace(0, np.nan)
    return efficiency.fillna(0.5).clip(0, 1)
```

### 5.3 Feature Count Summary

| Category | Current | Added | New Total |
|----------|---------|-------|-----------|
| Momentum | 16 | 0 | 16 |
| Mean Reversion | 11 | 0 | 11 |
| Volatility | 10 | 0 | 10 |
| Drawdown/Risk | 11 | 0 | 11 |
| Higher Moments | 9 | 0 | 9 |
| Trend Quality | 8 | 0 | 8 |
| Price Level | 8 | 0 | 8 |
| Cross-Sectional | 8 | 0 | 8 |
| Risk-Adjusted | 8 | 0 | 8 |
| Volume | 4 | 0 | 4 |
| Pattern | 6 | 0 | 6 |
| **Market-Relative** | **0** | **9** | **9** |
| **Microstructure** | **0** | **7** | **7** |
| **Regime** | **0** | **5** | **5** |
| **Engineered** | **0** | **7** | **7** |
| Oil-Specific | 10 | 0 | 10 |
| SMC | 6 | 0 | 6 |
| S/R | 8 | 0 | 8 |
| **TOTAL** | **~90** | **~28** | **~141** |

### 5.4 Architectural Decision: Monolith Decomposition

The current [`evolution/gp.py`](evolution/gp.py) is a **2313-line monolith** containing nodes, trees, operators, the entire feature library, the fitness function, GPStrategy, WalkForwardEvaluator, and GPPopulation. This must be decomposed:

| New File | Extracted From | Contents |
|----------|---------------|----------|
| `evolution/nodes.py` | `gp.py:29-270` | `Node`, `FeatureNode`, `ConstantNode`, `BinaryOpNode`, `UnaryOpNode`, `ConditionalNode` |
| `evolution/tree_ops.py` | `gp.py:276-484` | `TreeGenerator`, `GPOperators` |
| `evolution/features.py` | `gp.py:491-1619` | `FeatureLibrary` (with new features) |
| `evolution/fitness.py` | `gp.py:1626-1796` | `FitnessResult`, `calculate_fitness()` |
| `evolution/strategy.py` | `gp.py:1801-1906` | `GPStrategy` |
| `evolution/walkforward.py` | `gp.py:1912-2071` | `WalkForwardEvaluator` |
| `evolution/population.py` | `gp.py:2108-2290` | `GPPopulation` |
| `evolution/feature_cache.py` | NEW | `FeaturePrecomputeCache` |
| `evolution/gp.py` | KEPT | Re-exports for backward compatibility |

### 5.5 Files Changed

| File | Change |
|------|--------|
| `evolution/features.py` | **NEW** — Extracted + expanded FeatureLibrary |
| `evolution/nodes.py` | **NEW** — Extracted node classes |
| `evolution/tree_ops.py` | **NEW** — Extracted tree operations |
| `evolution/strategy.py` | **NEW** — Extracted GPStrategy |
| `evolution/walkforward.py` | **NEW** — Extracted + enhanced WalkForwardEvaluator |
| `evolution/population.py` | **NEW** — Extracted GPPopulation |
| `evolution/feature_cache.py` | **NEW** — Feature pre-computation cache |
| [`evolution/gp.py`](evolution/gp.py) | Becomes thin re-export shim for backward compat |

---

## 6. Workstream 3: Fitness Function Redesign

### 6.1 Problem Statement

The current fitness function in [`calculate_fitness()`](evolution/gp.py:1648) has three issues:

1. **Drawdown penalty thresholds are calibrated for large-cap**, not oil microcaps where 25-35% drawdowns are routine
2. **All periods weighted equally** — no recency bias
3. **No regime-aware scoring** — a strategy that performs well in the current regime but poorly in 2020 COVID crash gets penalized equally

### 6.2 Design: Regime-Calibrated Penalties

Replace hard-coded drawdown thresholds with universe-adaptive thresholds:

```python
def _get_drawdown_thresholds(universe_type: str) -> Dict[str, float]:
    """Get drawdown penalty thresholds based on universe characteristics."""
    if universe_type == 'oil_microcap':
        return {
            'severe': 0.50,    # Was 0.35 — oil microcaps routinely hit 40%+
            'moderate': 0.35,  # Was 0.25
            'mild': 0.20,      # Was 0.15
        }
    elif universe_type == 'oil_largecap':
        return {
            'severe': 0.40,
            'moderate': 0.30,
            'mild': 0.20,
        }
    else:  # general / S&P 500
        return {
            'severe': 0.35,
            'moderate': 0.25,
            'mild': 0.15,
        }
```

### 6.3 Design: Recency-Weighted Fitness

Integrate the recency weighting from Workstream 1 into the fitness calculation:

```python
def calculate_fitness_v2(
    period_results: List[Dict],
    benchmark_results: List[Dict],
    transaction_cost: float = 0.002,
    recency_half_life: int = 4,
    universe_type: str = 'general',
) -> FitnessResult:
    """
    Enhanced fitness with recency weighting and universe-adaptive penalties.
    """
    # ... extract metrics ...
    
    # Recency-weighted means instead of simple means
    avg_excess_sharpe = _recency_weighted_mean(excess_sharpes, recency_half_life)
    avg_excess_return = _recency_weighted_mean(excess_returns, recency_half_life)
    
    # Universe-adaptive drawdown thresholds
    dd_thresholds = _get_drawdown_thresholds(universe_type)
    
    if worst_dd > dd_thresholds['severe']:
        dd_penalty = 1.0
    elif worst_dd > dd_thresholds['moderate']:
        dd_penalty = 0.6
    elif worst_dd > dd_thresholds['mild']:
        dd_penalty = 0.2
    else:
        dd_penalty = 0.0
    
    # ... rest of calculation ...
```

### 6.4 Design: Multi-Objective Composite Score

Add an optional multi-objective mode that combines Sharpe-based and Calmar-based fitness:

```python
total = (
    0.25 * sharpe_component +      # Risk-adjusted returns (Sharpe)
    0.20 * calmar_component +      # Drawdown-adjusted returns (Calmar)
    0.20 * return_component +      # Absolute excess returns
    0.20 * stability_component -   # Consistency & win rate
    0.15 * cost_penalty            # Transaction costs
)
```

This blends the existing Sharpe fitness with the Calmar fitness from [`evolution/fitness_calmar.py`](evolution/fitness_calmar.py), giving the GP a more nuanced optimization target.

### 6.5 Files Changed

| File | Change |
|------|--------|
| `evolution/fitness.py` | **NEW** — `calculate_fitness_v2()` with recency weighting, universe-adaptive penalties |
| [`evolution/fitness_calmar.py`](evolution/fitness_calmar.py) | Add recency weighting, universe-adaptive thresholds |
| [`config.py`](config.py) | Add `UNIVERSE_TYPE`, `RECENCY_HALF_LIFE` settings |
| [`arena_runner_v3.py`](arena_runner_v3.py) | Pass `universe_type` and `recency_half_life` to evaluator |

---

## 7. Workstream 4: Oil-Universe Specialization

### 7.1 Problem Statement

Management's #1 priority is oil stock performance. The current oil universe is **too small** (8 tickers) and the oil-specific features in [`OilSpecificFeatures`](evolution/oil_specific_features.py:35) have a **broken integration** — the [`calculate_all_features()`](evolution/oil_specific_features.py:335) method signature expects `(stock_prices, oil_market_data)` but the [`FeatureLibrary`](evolution/gp.py:1599) calls it with `(prices, volume, start_date, end_date)`.

### 7.2 Design: Expanded Oil Universe

```python
# data/universe.py — NEW oil universe tiers

OIL_TIER1_MAJORS = [
    "XOM", "CVX", "COP", "SLB", "EOG",  # Large-cap oil majors
    "MPC", "PSX", "VLO", "OXY", "PXD",  # Large-cap E&P / refiners
]

OIL_TIER2_MIDCAP = [
    "DVN", "FANG", "MRO", "APA", "HAL",  # Mid-cap E&P / services
    "BKR", "HES", "CTRA", "OKE", "TRGP", # Mid-cap pipeline / services
    "WMB", "KMI",                          # Midstream
]

OIL_TIER3_MICROCAP = [
    "EONR", "TPET", "USEG", "STAK", "PRSO", "BATL",  # Existing microcaps
]

OIL_BENCHMARKS = [
    "USO", "BNO",  # Oil ETF proxies
    "XLE",         # Energy sector ETF (NEW — better sector benchmark)
    "XOP",         # Oil & Gas Exploration ETF (NEW)
]

# Combined: 30+ tickers for meaningful cross-sectional analysis
OIL_EXPANDED_UNIVERSE = OIL_TIER1_MAJORS + OIL_TIER2_MIDCAP + OIL_TIER3_MICROCAP + OIL_BENCHMARKS
```

**Rationale:** With 30+ tickers, cross-sectional features become meaningful. The GP can now discover strategies like "buy microcap oil stocks with high idiosyncratic vol when large-cap oil is trending up" — impossible with only 8 tickers.

### 7.3 Design: Fix Oil Feature Integration

The current integration is broken. Fix the [`FeatureLibrary._compute_feature()`](evolution/gp.py:1599) oil branch:

```python
elif feature_type == 'oil':
    if self.oil_features is None:
        return pd.Series(0.0, index=tickers)
    
    feature_name = spec[1]
    
    # Lazy-load oil market data (cached per date range)
    if not hasattr(self, '_oil_market_data') or self._oil_market_data is None:
        from evolution.oil_specific_features import fetch_oil_market_data
        start_date = prices.index[0].strftime("%Y-%m-%d")
        end_date = prices.index[-1].strftime("%Y-%m-%d")
        self._oil_market_data = fetch_oil_market_data(start_date, end_date)
    
    # Calculate per-ticker oil features
    results = {}
    for ticker in tickers:
        try:
            ticker_features = self.oil_features.calculate_all_features(
                stock_prices=prices[ticker],
                oil_market_data=self._oil_market_data
            )
            results[ticker] = ticker_features.get(feature_name, 0.0)
        except Exception:
            results[ticker] = 0.0
    
    return pd.Series(results)
```

### 7.4 Design: Oil-Specific Benchmark Enhancement

Add XLE (Energy Select Sector SPDR) as the primary oil benchmark instead of equal-weight:

```python
# In _calculate_benchmarks():
if self.is_oil_universe:
    # Primary benchmark: XLE (energy sector ETF)
    if 'XLE' in self.prices.columns:
        xle_bench = calculate_benchmark(
            self.prices, test_start, test_end,
            method="single_ticker", benchmark_ticker="XLE"
        )
        self.benchmark_results.append(xle_bench)  # Use XLE as PRIMARY benchmark
    
    # Secondary benchmarks: USO, BNO (for reference)
    for oil_ticker in ['USO', 'BNO']:
        ...
```

### 7.5 Files Changed

| File | Change |
|------|--------|
| [`data/universe.py`](data/universe.py:220) | Add `OIL_TIER1_MAJORS`, `OIL_TIER2_MIDCAP`, `OIL_EXPANDED_UNIVERSE`, `OIL_BENCHMARKS` |
| [`evolution/oil_specific_features.py`](evolution/oil_specific_features.py:335) | Fix `calculate_all_features()` signature, add caching |
| `evolution/features.py` | Fix oil feature integration in `_compute_feature()` |
| [`arena_runner_v3.py`](arena_runner_v3.py:331) | Use expanded oil universe, XLE as primary benchmark |

---

## 8. Workstream 5: Dashboard & UI for Management

### 8.1 Problem Statement

The current [`ui/dashboard.py`](ui/dashboard.py) is a basic Streamlit app with:
- Strategy selection dropdown
- Cumulative returns chart
- Period comparison bars
- Drawdown chart
- Rolling Sharpe chart
- Fitness evolution chart

Management needs:
1. **Oil-specific views** — performance vs XLE/USO/BNO, oil price overlay
2. **Backtester controls** — run backtests from the UI, adjust parameters
3. **Feature importance visualization** — which features drive the best strategies
4. **Regime overlay** — show market regime on performance charts
5. **Real-time status** — evolution progress, current generation stats

### 8.2 Design: New Dashboard Pages

#### Page 1: Oil Performance Dashboard (Management View)

```
┌─────────────────────────────────────────────────────────────┐
│  🛢️ OIL STRATEGY PERFORMANCE                                │
├─────────────────────────────────────────────────────────────┤
│  [Strategy Selector ▼]  [Date Range ▼]  [Benchmark ▼]      │
├──────────────────────────┬──────────────────────────────────┤
│  KPI Cards:              │  Cumulative Returns Chart        │
│  • Alpha vs XLE: +12.3%  │  (Strategy vs XLE vs USO vs BNO) │
│  • Sharpe: 1.45          │                                  │
│  • Max DD: -18.2%        │                                  │
│  • Win Rate: 72%         │                                  │
│  • Calmar: 2.1           │                                  │
├──────────────────────────┼──────────────────────────────────┤
│  Oil Price Overlay       │  Period Alpha Heatmap            │
│  (WTI + strategy returns)│  (green/red by quarter)          │
├──────────────────────────┴──────────────────────────────────┤
│  Period Details Table (sortable, filterable)                 │
│  Start | End | Return | vs XLE | vs USO | Sharpe | Max DD   │
└─────────────────────────────────────────────────────────────┘
```

#### Page 2: Backtester Controls

```
┌─────────────────────────────────────────────────────────────┐
│  🧪 BACKTESTER                                               │
├─────────────────────────────────────────────────────────────┤
│  Universe: [Oil Expanded ▼]  Start: [2018-01-01]            │
│  Train Months: [6]  Test Months: [3]  Step: [3]             │
│  Population: [100]  Generations: [30]  Max Depth: [5]       │
│  ☑ Recency Weighting  ☑ Expanding Window  ☑ Oil Features    │
│  ☑ Calmar Fitness  ☑ Trailing Stops  ☑ Kelly Sizing         │
│                                                              │
│  [▶ Run Evolution]  [⏹ Stop]  [📊 View Results]             │
├─────────────────────────────────────────────────────────────┤
│  Evolution Progress:                                         │
│  Gen 12/30 ████████████░░░░░░░░ 40%                         │
│  Best Fitness: 0.623  Avg: 0.312  Stagnant: 2               │
│                                                              │
│  Fitness Evolution Chart (live updating)                     │
└─────────────────────────────────────────────────────────────┘
```

#### Page 3: Feature Importance

```
┌─────────────────────────────────────────────────────────────┐
│  📊 FEATURE IMPORTANCE                                       │
├─────────────────────────────────────────────────────────────┤
│  Top 20 Features by Weighted Score (horizontal bar chart)    │
│  ├── market_beta_63d          ████████████ 2.34              │
│  ├── idio_vol_21d             ██████████   1.98              │
│  ├── kurt_63d                 █████████    1.63              │
│  ├── oil_inventory_zscore     ████████     1.45              │
│  └── ...                                                     │
├─────────────────────────────────────────────────────────────┤
│  Feature Category Breakdown (pie chart)                      │
│  Feature Correlation Matrix (heatmap)                        │
│  Feature Usage by Fitness Tier (stacked bar)                 │
└─────────────────────────────────────────────────────────────┘
```

### 8.3 Design: WebSocket Evolution Progress

For real-time evolution monitoring, add a WebSocket-based progress reporter:

```python
# evolution/progress.py
class EvolutionProgressReporter:
    """Reports evolution progress to dashboard via file-based IPC."""
    
    def __init__(self, progress_file: str = "data/cache/evolution_progress.json"):
        self.progress_file = Path(progress_file)
    
    def report_generation(self, stats: Dict):
        """Write generation stats to progress file."""
        progress = {
            'timestamp': datetime.now().isoformat(),
            'generation': stats['generation'],
            'total_generations': stats.get('total_generations', 30),
            'best_fitness': stats['max_fitness'],
            'avg_fitness': stats['avg_fitness'],
            'stagnant': stats.get('stagnant', 0),
            'status': 'running',
        }
        self.progress_file.write_text(json.dumps(progress))
    
    def report_complete(self, final_stats: Dict):
        progress = {**final_stats, 'status': 'complete'}
        self.progress_file.write_text(json.dumps(progress))
```

The Streamlit dashboard polls this file every 2 seconds for live updates.

### 8.4 Files Changed

| File | Change |
|------|--------|
| [`ui/dashboard.py`](ui/dashboard.py) | Complete rewrite — multi-page app with oil view, backtester, feature importance |
| [`ui/charts.py`](ui/charts.py) | Add oil overlay chart, alpha heatmap, feature importance charts |
| **NEW** `ui/pages/oil_performance.py` | Oil-specific performance page |
| **NEW** `ui/pages/backtester.py` | Backtester controls page |
| **NEW** `ui/pages/feature_importance.py` | Feature importance visualization |
| **NEW** `evolution/progress.py` | Evolution progress reporter |
| [`tools/feature_importance.py`](tools/feature_importance.py) | Refactor for dashboard integration |

---

## 9. Implementation Plan & Sequencing

### Phase 1: Foundation (Week 1) — Unblock Fitness Ceiling

**Goal:** Break through 0.59 fitness ceiling.

| Day | Task | Owner | Files |
|-----|------|-------|-------|
| 1 | Decompose `gp.py` monolith into modules | L4 | `evolution/*.py` |
| 1 | Add anchored recent period to walk-forward | L4 | `arena_runner_v3.py`, `evolution/walkforward.py` |
| 2 | Implement market-relative features (beta, idio_vol, sector_corr) | L4 | `evolution/features.py` |
| 2 | Fix oil feature integration (broken `calculate_all_features` call) | L4 | `evolution/features.py`, `evolution/oil_specific_features.py` |
| 3 | Recalibrate fitness penalties for oil microcaps | L4 | `evolution/fitness.py` |
| 3 | Add recency weighting to fitness | L4 | `evolution/fitness.py` |
| 4 | Implement feature pre-computation cache | L4 | `evolution/feature_cache.py` |
| 5 | Integration testing — run oil evolution, verify fitness > 0.59 | L4 | `tests/` |

### Phase 2: Feature Depth (Week 2) — Maximize Alpha Discovery

| Day | Task | Owner | Files |
|-----|------|-------|-------|
| 1 | Add microstructure features (Amihud, Roll spread, Kyle's lambda) | L4 | `evolution/features.py` |
| 2 | Add regime features (vol_regime, trend_strength, breadth) | L4 | `evolution/features.py` |
| 2 | Add engineered features (mom_sharpe, efficiency_ratio, fractal_dim) | L4 | `evolution/features.py` |
| 3 | Expand oil universe (30+ tickers with tier system) | L4 | `data/universe.py` |
| 3 | Add XLE as primary oil benchmark | L4 | `arena_runner_v3.py` |
| 4 | Add expanding window option to walk-forward | L4 | `evolution/walkforward.py` |
| 5 | Full regression test — verify no performance degradation | L4 | `tests/` |

### Phase 3: Dashboard (Week 3) — Management Visibility

| Day | Task | Owner | Files |
|-----|------|-------|-------|
| 1-2 | Oil Performance Dashboard page | L4 | `ui/pages/oil_performance.py` |
| 2-3 | Backtester Controls page | L4 | `ui/pages/backtester.py` |
| 3-4 | Feature Importance page | L4 | `ui/pages/feature_importance.py` |
| 4 | Evolution progress reporter | L4 | `evolution/progress.py` |
| 5 | End-to-end demo for management | L4 + Lead | All |

### Dependency Graph

```
Phase 1 (Foundation)
├── Monolith decomposition ──────────────────────┐
├── Anchored recent period ──┐                    │
├── Market-relative features ├── Fitness recalib ─┤
├── Fix oil integration ─────┘                    │
└── Feature pre-compute cache ────────────────────┘
                                                   │
Phase 2 (Feature Depth)                            │
├── Microstructure features ◄──────────────────────┘
├── Regime features
├── Engineered features
├── Expanded oil universe
└── Expanding window
     │
Phase 3 (Dashboard)
├── Oil Performance page
├── Backtester Controls page
├── Feature Importance page
└── Progress reporter
```

---

## 10. Testing Strategy

### 10.1 Unit Tests

| Test | Validates | File |
|------|-----------|------|
| `test_market_beta_computation` | Beta = 1.0 for market proxy, ~0 for uncorrelated | `tests/test_features.py` |
| `test_idiosyncratic_vol` | Idio vol < total vol for all tickers | `tests/test_features.py` |
| `test_amihud_illiquidity` | Higher for low-volume stocks | `tests/test_features.py` |
| `test_efficiency_ratio_bounds` | Always in [0, 1] | `tests/test_features.py` |
| `test_recency_weighting` | Recent periods get higher weight | `tests/test_fitness.py` |
| `test_anchored_period_includes_recent` | Last period end >= today - 7 days | `tests/test_walkforward.py` |
| `test_feature_cache_consistency` | Cached features == freshly computed | `tests/test_feature_cache.py` |
| `test_oil_universe_expansion` | 30+ tickers in expanded universe | `tests/test_universe.py` |
| `test_fitness_oil_calibration` | 35% DD gets moderate (not severe) penalty for oil | `tests/test_fitness.py` |

### 10.2 Integration Tests

| Test | Validates |
|------|-----------|
| `test_full_oil_evolution_run` | End-to-end: data fetch → evolution → DB save → dashboard load |
| `test_feature_cache_speedup` | Cached evaluation ≥ 5× faster than uncached |
| `test_march_2026_coverage` | At least one test period includes March 2026 data |
| `test_backward_compatibility` | Old `from evolution.gp import ...` still works after decomposition |

### 10.3 Regression Tests

| Test | Validates |
|------|-----------|
| `test_general_universe_no_degradation` | S&P 500 fitness doesn't decrease with new features |
| `test_oil_fitness_improvement` | Oil fitness > 0.59 with new features + calibration |

---

## 11. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | Monolith decomposition breaks imports | Medium | High | Keep `gp.py` as re-export shim; add backward-compat tests |
| R2 | 141 features cause overfitting | Medium | Medium | Parsimony coefficient already penalizes complexity; add feature selection pressure |
| R3 | Feature pre-computation uses too much memory | Low | Medium | LRU cache with configurable max size; compute only for rebalance dates |
| R4 | Expanded oil universe has survivorship bias | Medium | Medium | Use point-in-time universe membership; exclude delisted tickers |
| R5 | Recency weighting causes recency bias in GP | Medium | Low | Configurable half-life; default is moderate (4 periods = 1 year) |
| R6 | Oil market data (WTI/Brent) download fails | Low | High | Cache oil data aggressively; fallback to USO/BNO as proxy |
| R7 | Dashboard Streamlit version incompatibility | Low | Low | Pin Streamlit version in `pyproject.toml` |

---

## Appendix: File-Level Change Map

### New Files

| File | Purpose | Lines (est.) |
|------|---------|-------------|
| `evolution/nodes.py` | Expression tree node classes | ~250 |
| `evolution/tree_ops.py` | Tree generation and GP operators | ~220 |
| `evolution/features.py` | Expanded FeatureLibrary (~141 features) | ~1800 |
| `evolution/fitness.py` | `calculate_fitness_v2()` with recency + universe-adaptive | ~250 |
| `evolution/strategy.py` | `GPStrategy` class | ~120 |
| `evolution/walkforward.py` | Enhanced `WalkForwardEvaluator` | ~250 |
| `evolution/population.py` | `GPPopulation` class | ~200 |
| `evolution/feature_cache.py` | `FeaturePrecomputeCache` | ~150 |
| `evolution/progress.py` | Evolution progress reporter | ~80 |
| `ui/pages/oil_performance.py` | Oil performance dashboard page | ~300 |
| `ui/pages/backtester.py` | Backtester controls page | ~250 |
| `ui/pages/feature_importance.py` | Feature importance visualization | ~200 |
| `tests/test_features.py` | Feature unit tests | ~300 |
| `tests/test_fitness.py` | Fitness function tests | ~150 |
| `tests/test_walkforward.py` | Walk-forward evaluator tests | ~150 |
| `tests/test_feature_cache.py` | Feature cache tests | ~100 |

### Modified Files

| File | Changes |
|------|---------|
| [`evolution/gp.py`](evolution/gp.py) | Becomes thin re-export shim (~50 lines) |
| [`evolution/oil_specific_features.py`](evolution/oil_specific_features.py) | Fix `calculate_all_features()` signature, add caching |
| [`evolution/fitness_calmar.py`](evolution/fitness_calmar.py) | Add recency weighting, universe-adaptive thresholds |
| [`data/universe.py`](data/universe.py) | Add expanded oil universe tiers |
| [`arena_runner_v3.py`](arena_runner_v3.py) | Anchored period, feature cache, expanded oil universe, XLE benchmark |
| [`config.py`](config.py) | Add `RECENCY_HALF_LIFE`, `USE_EXPANDING_WINDOW`, `UNIVERSE_TYPE` |
| [`ui/dashboard.py`](ui/dashboard.py) | Multi-page app structure |
| [`ui/charts.py`](ui/charts.py) | Oil overlay, alpha heatmap, feature importance charts |
| [`scripts/run_arena_oil.sh`](scripts/run_arena_oil.sh) | Updated flags for new features |

### Unchanged Files

| File | Reason |
|------|--------|
| [`data/fetcher.py`](data/fetcher.py) | Works correctly, no changes needed |
| [`backtest/risk_management.py`](backtest/risk_management.py) | Already well-implemented |
| [`backtest/rebalancing.py`](backtest/rebalancing.py) | Already well-implemented |
| [`backtest/stops.py`](backtest/stops.py) | Already well-implemented |
| [`backtest/position_sizing.py`](backtest/position_sizing.py) | Already well-implemented |
| [`evolution/gp_storage.py`](evolution/gp_storage.py) | Works correctly, no changes needed |
| [`evolution/regime_detection.py`](evolution/regime_detection.py) | Already well-implemented, will be used by new regime features |

---

*End of Design Document*
