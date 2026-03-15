#!/usr/bin/env python3
"""
Calmar Ratio-Based Fitness Function

Replaces Sharpe-based fitness with Calmar Ratio (Return / Max Drawdown).
Better suited for microcap trading where drawdowns are the primary risk.
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class CalmarFitnessResult:
    """Comprehensive fitness evaluation using Calmar Ratio."""
    total: float
    calmar_component: float
    return_component: float
    stability_component: float
    turnover_penalty: float
    
    avg_calmar: float
    calmar_std: float
    avg_return: float
    avg_drawdown: float
    avg_turnover: float
    win_rate: float
    worst_period_calmar: float
    
    n_periods: int
    period_results: List[Dict] = field(default_factory=list)


def calculate_calmar_fitness(
    period_results: List[Dict],
    benchmark_results: List[Dict],
    target_turnover: float = 0.5,  # 50% annual turnover target
) -> CalmarFitnessResult:
    """
    Calculate fitness using Calmar Ratio.
    
    Calmar Ratio = Annualized Return / Max Drawdown
    
    This is superior to Sharpe for microcaps because:
    - Drawdown is the primary risk (not volatility)
    - Easier to interpret (2.0 = 2x return vs drawdown)
    - Penalizes catastrophic losses more heavily
    
    Args:
        period_results: List of period performance dicts
        benchmark_results: List of benchmark performance dicts
        target_turnover: Target annual turnover (default 50%)
    
    Returns:
        CalmarFitnessResult with fitness score and components
    """
    
    if not period_results or not benchmark_results:
        return CalmarFitnessResult(
            total=0, calmar_component=0, return_component=0,
            stability_component=0, turnover_penalty=0, avg_calmar=0,
            calmar_std=0, avg_return=0, avg_drawdown=0, avg_turnover=0,
            win_rate=0, worst_period_calmar=-1, n_periods=0, period_results=[]
        )
    
    n = len(period_results)
    
    # Extract metrics
    returns = [r['total_return'] for r in period_results]
    drawdowns = [r.get('max_drawdown', 0.01) for r in period_results]  # Avoid division by zero
    turnovers = [r.get('turnover', 0) for r in period_results]
    
    # Calculate Calmar Ratios
    calmars = []
    for ret, dd in zip(returns, drawdowns):
        # Annualize return (assuming monthly periods)
        ann_return = (1 + ret) ** 12 - 1
        # Calmar = annualized return / max drawdown
        calmar = ann_return / max(dd, 0.01)  # Avoid division by zero
        calmars.append(calmar)
    
    # Benchmark metrics
    bench_returns = [b.get('total_return', 0) for b in benchmark_results[:n]]
    bench_drawdowns = [b.get('max_drawdown', 0.01) for b in benchmark_results[:n]]
    
    bench_calmars = []
    for ret, dd in zip(bench_returns, bench_drawdowns):
        ann_return = (1 + ret) ** 12 - 1
        calmar = ann_return / max(dd, 0.01)
        bench_calmars.append(calmar)
    
    # EXCESS metrics (strategy minus benchmark)
    excess_returns = [r - b for r, b in zip(returns, bench_returns)]
    excess_calmars = [c - b for c, b in zip(calmars, bench_calmars)]
    
    # Aggregate statistics
    avg_calmar = np.mean(calmars)
    calmar_std = np.std(calmars) if n > 1 else 0
    avg_return = np.mean(returns)
    avg_drawdown = np.mean(drawdowns)
    avg_turnover = np.mean(turnovers)
    
    avg_excess_return = np.mean(excess_returns)
    avg_excess_calmar = np.mean(excess_calmars)
    
    # Win rate = fraction of periods beating benchmark
    win_rate = sum(1 for e in excess_returns if e > 0) / n
    worst_calmar = min(calmars)
    
    # ═══════════════════════════════════════════════════════════════
    # COMPONENTS - all centered around zero for benchmark-matching
    # ═══════════════════════════════════════════════════════════════
    
    # 1. Calmar component: Excess Calmar of 1.0 = full score
    #    Example: If benchmark Calmar is 1.5 and strategy is 2.5, excess is 1.0
    calmar_component = np.clip(avg_excess_calmar / 1.0, -1, 1)
    
    # Penalize variance in Calmar (consistency matters)
    calmar_variance_penalty = min(calmar_std / 2.0, 0.3)
    calmar_component -= calmar_variance_penalty
    
    # 2. Return component: 10% excess return = full score
    return_component = np.clip(avg_excess_return / 0.10, -1, 1)
    
    # 3. Stability: win rate vs benchmark, penalize catastrophic Calmars
    stability_component = (win_rate - 0.5) * 2  # 50% win rate = 0, 100% = 1
    
    if worst_calmar < -1.0:  # Negative Calmar = losing money with drawdown
        stability_component -= 0.5
    elif worst_calmar < 0.5:  # Very low Calmar
        stability_component -= 0.2
    
    stability_component = np.clip(stability_component, -1, 1)
    
    # 4. Turnover penalty: Penalize excessive turnover
    #    Target is 50% annual (0.5), which is ~25 trades/year on 50 positions
    excess_turnover = max(0, avg_turnover - target_turnover)
    turnover_penalty = np.clip(excess_turnover / target_turnover, 0, 1)
    
    # ═══════════════════════════════════════════════════════════════
    # TOTAL - zero means "matches benchmark"
    # ═══════════════════════════════════════════════════════════════
    
    total = (
        0.40 * calmar_component +    # Primary: risk-adjusted returns
        0.25 * return_component +     # Secondary: absolute returns
        0.20 * stability_component -  # Tertiary: consistency
        0.15 * turnover_penalty       # Penalty: excessive trading
    )
    
    # Period count adjustment (need enough data)
    if n < 4:
        total *= n / 4
    
    # Clip to [-1, 1] range
    total = np.clip(total, -1, 1)
    
    return CalmarFitnessResult(
        total=total,
        calmar_component=calmar_component,
        return_component=return_component,
        stability_component=stability_component,
        turnover_penalty=turnover_penalty,
        avg_calmar=avg_calmar,
        calmar_std=calmar_std,
        avg_return=avg_return,
        avg_drawdown=avg_drawdown,
        avg_turnover=avg_turnover,
        win_rate=win_rate,
        worst_period_calmar=worst_calmar,
        n_periods=n,
        period_results=period_results
    )


def calculate_calmar_ratio(returns: pd.Series) -> float:
    """
    Calculate Calmar Ratio for a returns series.
    
    Args:
        returns: Daily returns series
    
    Returns:
        Calmar Ratio (annualized return / max drawdown)
    """
    if len(returns) == 0:
        return 0
    
    # Cumulative returns
    cumulative = (1 + returns).cumprod()
    
    # Annualized return
    total_return = cumulative.iloc[-1] - 1
    years = len(returns) / 252
    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Max drawdown
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = abs(drawdown.min())
    
    # Calmar Ratio
    if max_drawdown < 0.01:
        max_drawdown = 0.01  # Avoid division by zero
    
    calmar = ann_return / max_drawdown
    return calmar


def compare_fitness_methods(period_results: List[Dict], 
                            benchmark_results: List[Dict]) -> Dict:
    """
    Compare Sharpe-based vs Calmar-based fitness.
    
    Useful for understanding the difference between the two approaches.
    
    Returns:
        Dict with comparison metrics
    """
    # Calculate both fitness methods
    from evolution.gp import calculate_fitness as calculate_sharpe_fitness
    
    sharpe_fitness = calculate_sharpe_fitness(period_results, benchmark_results)
    calmar_fitness = calculate_calmar_fitness(period_results, benchmark_results)
    
    return {
        'sharpe_fitness': sharpe_fitness.total,
        'calmar_fitness': calmar_fitness.total,
        'sharpe_avg': sharpe_fitness.avg_sharpe,
        'calmar_avg': calmar_fitness.avg_calmar,
        'sharpe_stability': sharpe_fitness.stability_component,
        'calmar_stability': calmar_fitness.stability_component,
        'fitness_correlation': np.corrcoef([sharpe_fitness.total, calmar_fitness.total])[0, 1]
    }


# Example usage
if __name__ == "__main__":
    # Example period results
    period_results = [
        {'total_return': 0.15, 'max_drawdown': 0.08, 'turnover': 0.4},
        {'total_return': 0.12, 'max_drawdown': 0.10, 'turnover': 0.5},
        {'total_return': 0.18, 'max_drawdown': 0.12, 'turnover': 0.3},
        {'total_return': -0.05, 'max_drawdown': 0.15, 'turnover': 0.6},
        {'total_return': 0.20, 'max_drawdown': 0.09, 'turnover': 0.4},
    ]
    
    benchmark_results = [
        {'total_return': 0.10, 'max_drawdown': 0.12},
        {'total_return': 0.08, 'max_drawdown': 0.15},
        {'total_return': 0.12, 'max_drawdown': 0.10},
        {'total_return': -0.08, 'max_drawdown': 0.18},
        {'total_return': 0.15, 'max_drawdown': 0.11},
    ]
    
    fitness = calculate_calmar_fitness(period_results, benchmark_results)
    
    print("Calmar Fitness Results:")
    print(f"  Total Fitness: {fitness.total:.3f}")
    print(f"  Calmar Component: {fitness.calmar_component:.3f}")
    print(f"  Return Component: {fitness.return_component:.3f}")
    print(f"  Stability Component: {fitness.stability_component:.3f}")
    print(f"  Turnover Penalty: {fitness.turnover_penalty:.3f}")
    print(f"\nMetrics:")
    print(f"  Avg Calmar: {fitness.avg_calmar:.2f}")
    print(f"  Avg Return: {fitness.avg_return:.1%}")
    print(f"  Avg Drawdown: {fitness.avg_drawdown:.1%}")
    print(f"  Avg Turnover: {fitness.avg_turnover:.1%}")
    print(f"  Win Rate: {fitness.win_rate:.1%}")
