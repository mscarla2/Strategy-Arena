# evolution/gp_fitness.py
"""
Fitness evaluation functions for GP strategies.

Contains: FitnessResult, calculate_fitness(), calculate_fitness_v2(),
          _recency_weighted_mean(), _get_drawdown_thresholds()
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════════
# FITNESS RESULT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FitnessResult:
    """Comprehensive fitness evaluation."""
    total: float
    sharpe_component: float
    return_component: float
    stability_component: float
    cost_penalty: float
    
    avg_sharpe: float
    sharpe_std: float
    avg_return: float
    return_std: float
    avg_turnover: float
    win_rate: float
    worst_period_return: float
    information_ratio: float
    
    n_periods: int
    period_results: List[Dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# FITNESS FUNCTION v1
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_fitness(
    period_results: List[Dict],
    benchmark_results: List[Dict],
    transaction_cost: float = 0.002,
) -> FitnessResult:
    """Calculate fitness centered on benchmark-relative performance."""
    
    if not period_results:
        return FitnessResult(
            total=0, sharpe_component=0, return_component=0,
            stability_component=0, cost_penalty=0, avg_sharpe=0,
            sharpe_std=0, avg_return=0, return_std=0, avg_turnover=0,
            win_rate=0, worst_period_return=-1, information_ratio=0,
            n_periods=0, period_results=[]
        )
    
    n = len(period_results)
    # Pad benchmarks with zeros if fewer than period results
    benchmark_results = list(benchmark_results) + [{}] * max(0, n - len(benchmark_results))
    
    # Extract metrics
    sharpes = [r['sharpe_ratio'] for r in period_results]
    returns = [r['total_return'] for r in period_results]
    turnovers = [r.get('turnover', 0) for r in period_results]
    
    bench_returns = [b.get('total_return', 0) for b in benchmark_results[:n]]
    bench_sharpes = [b.get('sharpe_ratio', 0) for b in benchmark_results[:n]]
    
    # EXCESS metrics (strategy minus benchmark)
    excess_returns = [r - b for r, b in zip(returns, bench_returns)]
    excess_sharpes = [s - b for s, b in zip(sharpes, bench_sharpes)]
    
    avg_sharpe = np.mean(sharpes)
    sharpe_std = np.std(sharpes) if n > 1 else 0
    avg_return = np.mean(returns)
    return_std = np.std(returns) if n > 1 else 0
    avg_turnover = np.mean(turnovers)
    
    avg_excess_return = np.mean(excess_returns)
    avg_excess_sharpe = np.mean(excess_sharpes)
    
    # Win rate = fraction of periods beating benchmark
    win_rate = sum(1 for e in excess_returns if e > 0) / n
    worst_return = min(returns)
    worst_dd = max(r.get('max_drawdown', 0) for r in period_results)
    
    # Information ratio
    if n > 1:
        excess_std = np.std(excess_returns)
        ir = avg_excess_return / excess_std if excess_std > 0 else 0
    else:
        ir = 0
    
    # ═══════════════════════════════════════════════════════════════
    # SHARPE COMPONENT
    # Excess Sharpe component: 0.5 excess Sharpe = full score of 1.0
    # ═══════════════════════════════════════════════════════════════
    
    sharpe_component = np.clip(avg_excess_sharpe / 0.5, -1, 1)
    
    # Penalize variance in Sharpe
    sharpe_variance_penalty = min(sharpe_std / 1.0, 0.3)
    sharpe_component -= sharpe_variance_penalty
    
    # ═══════════════════════════════════════════════════════════════
    # RETURN COMPONENT
    # Excess return component: 10% annual excess = full score
    # ═══════════════════════════════════════════════════════════════
    
    return_component = np.clip(avg_excess_return / 0.10, -1, 1)
    
    # Information ratio bonus: IR of 0.5 = full bonus
    ir_bonus = np.clip(ir / 0.5, -0.5, 0.5)
    return_component = 0.7 * return_component + 0.3 * ir_bonus
    
    # ═══════════════════════════════════════════════════════════════
    # STABILITY COMPONENT
    # Penalties are explicitly weighted so their sum is bounded to 1.0,
    # preserving gradient signal across the full [-1, 1] range.
    # ═══════════════════════════════════════════════════════════════
    
    # Win rate baseline: 50% = 0, 100% = 1, 0% = -1
    stability_component = (win_rate - 0.5) * 2
    
    # Drawdown penalty (weight: 0.50)
    if worst_dd > 0.35:
        dd_penalty = 1.0   # near-disqualifying
    elif worst_dd > 0.25:
        dd_penalty = 0.6
    elif worst_dd > 0.15:
        dd_penalty = 0.2
    else:
        dd_penalty = 0.0
    
    # Worst period return penalty (weight: 0.25)
    return_penalty = 0.3 if worst_return < -0.20 else 0.0
    
    # Consistency penalty: high variance in excess returns (weight: 0.25)
    excess_return_std = np.std(excess_returns)
    consistency_penalty = np.clip(excess_return_std / 0.20, 0, 0.4)
    
    # Combine with explicit weights — max total penalty = 1.0
    stability_component -= (
        0.50 * dd_penalty +
        0.25 * return_penalty +
        0.25 * consistency_penalty
    )
    stability_component = np.clip(stability_component, -1, 1)
    
    # ═══════════════════════════════════════════════════════════════
    # COST PENALTY
    # ═══════════════════════════════════════════════════════════════
    
    annual_cost_drag = avg_turnover * 12 * transaction_cost * 2
    cost_penalty = np.clip(annual_cost_drag / 0.03, 0, 1)  # 3% drag = full penalty
    
    # ═══════════════════════════════════════════════════════════════
    # TOTAL - zero means "matches benchmark"
    # Positive = beats benchmark, Negative = loses to benchmark
    # ═══════════════════════════════════════════════════════════════
    
    total = (
        0.30 * sharpe_component +
        0.25 * return_component +
        0.30 * stability_component -
        0.15 * cost_penalty
    )
    
    # Period count adjustment (need enough data)
    if n < 4:
        total *= n / 4
    
    total = np.clip(total, -1, 1)
    
    return FitnessResult(
        total=total,
        sharpe_component=sharpe_component,
        return_component=return_component,
        stability_component=stability_component,
        cost_penalty=cost_penalty,
        avg_sharpe=avg_sharpe,
        sharpe_std=sharpe_std,
        avg_return=avg_return,
        return_std=return_std,
        avg_turnover=avg_turnover,
        win_rate=win_rate,
        worst_period_return=worst_return,
        information_ratio=ir,
        n_periods=n,
        period_results=period_results
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FITNESS FUNCTION v2 — RC-3: Recency Weighting + Universe-Adaptive Penalties
# ═══════════════════════════════════════════════════════════════════════════════

def _recency_weighted_mean(values: List[float], half_life_periods: int = 4) -> float:
    """
    Exponentially-weighted mean giving more weight to recent periods.
    
    Args:
        values: List of per-period values (oldest first, most recent last)
        half_life_periods: Number of periods for weight to decay by half
        
    Returns:
        Weighted mean
    """
    n = len(values)
    if n == 0:
        return 0.0
    if n == 1:
        return values[0]
    
    decay = np.log(2) / max(half_life_periods, 1)
    weights = np.array([np.exp(-decay * (n - 1 - i)) for i in range(n)])
    weights /= weights.sum()
    
    return float(np.dot(values, weights))


def _get_drawdown_thresholds(universe_type: str) -> Dict[str, float]:
    """
    Get drawdown penalty thresholds based on universe characteristics.
    
    Oil microcaps routinely hit 40%+ drawdowns, so the thresholds must be
    calibrated differently than for large-cap S&P 500 stocks.
    
    Args:
        universe_type: One of 'general', 'oil_microcap', 'oil_largecap'
        
    Returns:
        Dict with 'severe', 'moderate', 'mild' thresholds
    """
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


def calculate_fitness_v2(
    period_results: List[Dict],
    benchmark_results: List[Dict],
    transaction_cost: float = 0.002,
    recency_half_life: int = 4,
    universe_type: str = 'general',
) -> FitnessResult:
    """
    Enhanced fitness with recency weighting and universe-adaptive penalties.
    
    Improvements over calculate_fitness():
    - Recency-weighted means instead of simple means (recent periods matter more)
    - Universe-adaptive drawdown thresholds (oil microcaps get wider thresholds)
    - Blended Sharpe + Calmar scoring for multi-objective optimization
    
    Args:
        period_results: List of per-period performance dicts
        benchmark_results: List of benchmark performance dicts
        transaction_cost: Per-trade transaction cost
        recency_half_life: Half-life for recency weighting (in periods)
        universe_type: 'general', 'oil_microcap', or 'oil_largecap'
        
    Returns:
        FitnessResult with fitness score and components
    """
    if not period_results:
        return FitnessResult(
            total=0, sharpe_component=0, return_component=0,
            stability_component=0, cost_penalty=0, avg_sharpe=0,
            sharpe_std=0, avg_return=0, return_std=0, avg_turnover=0,
            win_rate=0, worst_period_return=-1, information_ratio=0,
            n_periods=0, period_results=[]
        )
    
    n = len(period_results)
    # Pad benchmarks with zeros if fewer than period results
    benchmark_results = list(benchmark_results) + [{}] * max(0, n - len(benchmark_results))
    
    # Extract metrics
    sharpes = [r['sharpe_ratio'] for r in period_results]
    returns = [r['total_return'] for r in period_results]
    turnovers = [r.get('turnover', 0) for r in period_results]
    
    bench_returns = [b.get('total_return', 0) for b in benchmark_results[:n]]
    bench_sharpes = [b.get('sharpe_ratio', 0) for b in benchmark_results[:n]]
    
    # EXCESS metrics (strategy minus benchmark)
    excess_returns = [r - b for r, b in zip(returns, bench_returns)]
    excess_sharpes = [s - b for s, b in zip(sharpes, bench_sharpes)]
    
    avg_sharpe = np.mean(sharpes)
    sharpe_std = np.std(sharpes) if n > 1 else 0
    avg_return = np.mean(returns)
    return_std = np.std(returns) if n > 1 else 0
    avg_turnover = np.mean(turnovers)
    
    # ═══════════════════════════════════════════════════════════════
    # RECENCY-WEIGHTED MEANS (RC-3 enhancement)
    # ═══════════════════════════════════════════════════════════════
    avg_excess_return = _recency_weighted_mean(excess_returns, recency_half_life)
    avg_excess_sharpe = _recency_weighted_mean(excess_sharpes, recency_half_life)
    
    # Win rate = fraction of periods beating benchmark
    win_rate = sum(1 for e in excess_returns if e > 0) / n
    worst_return = min(returns)
    worst_dd = max(r.get('max_drawdown', 0) for r in period_results)
    
    # Information ratio (recency-weighted)
    if n > 1:
        excess_std = np.std(excess_returns)
        ir = avg_excess_return / excess_std if excess_std > 0 else 0
    else:
        ir = 0
    
    # ═══════════════════════════════════════════════════════════════
    # SHARPE COMPONENT (same structure, recency-weighted input)
    # ═══════════════════════════════════════════════════════════════
    
    sharpe_component = np.clip(avg_excess_sharpe / 0.5, -1, 1)
    
    # Penalize variance in Sharpe
    sharpe_variance_penalty = min(sharpe_std / 1.0, 0.3)
    sharpe_component -= sharpe_variance_penalty
    
    # ═══════════════════════════════════════════════════════════════
    # CALMAR COMPONENT (NEW — blended with Sharpe for multi-objective)
    # ═══════════════════════════════════════════════════════════════
    
    # Calculate per-period Calmar ratios
    calmars = []
    bench_calmars = []
    for i in range(n):
        ret = returns[i]
        dd = max(period_results[i].get('max_drawdown', 0.01), 0.01)
        ann_ret = (1 + ret) ** 4 - 1  # Annualize from quarterly
        calmars.append(ann_ret / dd)
        
        b_ret = bench_returns[i]
        b_dd = max(benchmark_results[i].get('max_drawdown', 0.01), 0.01)
        b_ann_ret = (1 + b_ret) ** 4 - 1
        bench_calmars.append(b_ann_ret / b_dd)
    
    excess_calmars = [c - b for c, b in zip(calmars, bench_calmars)]
    avg_excess_calmar = _recency_weighted_mean(excess_calmars, recency_half_life)
    calmar_component = np.clip(avg_excess_calmar / 1.0, -1, 1)
    
    # ═══════════════════════════════════════════════════════════════
    # RETURN COMPONENT (recency-weighted)
    # ═══════════════════════════════════════════════════════════════
    
    return_component = np.clip(avg_excess_return / 0.10, -1, 1)
    
    # Information ratio bonus
    ir_bonus = np.clip(ir / 0.5, -0.5, 0.5)
    return_component = 0.7 * return_component + 0.3 * ir_bonus
    
    # ═══════════════════════════════════════════════════════════════
    # STABILITY COMPONENT — UNIVERSE-ADAPTIVE PENALTIES (RC-3)
    # ═══════════════════════════════════════════════════════════════
    
    stability_component = (win_rate - 0.5) * 2
    
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
    
    # Worst period return penalty (also universe-adaptive)
    worst_return_threshold = -0.30 if universe_type.startswith('oil') else -0.20
    return_penalty = 0.3 if worst_return < worst_return_threshold else 0.0
    
    # Consistency penalty
    excess_return_std = np.std(excess_returns)
    consistency_threshold = 0.30 if universe_type.startswith('oil') else 0.20
    consistency_penalty = np.clip(excess_return_std / consistency_threshold, 0, 0.4)
    
    stability_component -= (
        0.50 * dd_penalty +
        0.25 * return_penalty +
        0.25 * consistency_penalty
    )
    stability_component = np.clip(stability_component, -1, 1)
    
    # ═══════════════════════════════════════════════════════════════
    # COST PENALTY
    # ═══════════════════════════════════════════════════════════════
    
    annual_cost_drag = avg_turnover * 12 * transaction_cost * 2
    cost_penalty = np.clip(annual_cost_drag / 0.03, 0, 1)
    
    # ═══════════════════════════════════════════════════════════════
    # TOTAL — Blended Sharpe + Calmar (multi-objective)
    # ═══════════════════════════════════════════════════════════════
    
    total = (
        0.25 * sharpe_component +      # Risk-adjusted returns (Sharpe)
        0.20 * calmar_component +       # Drawdown-adjusted returns (Calmar)
        0.20 * return_component +       # Absolute excess returns
        0.20 * stability_component -    # Consistency & win rate
        0.15 * cost_penalty             # Transaction costs
    )
    
    # Period count adjustment
    if n < 4:
        total *= n / 4
    
    total = np.clip(total, -1, 1)
    
    return FitnessResult(
        total=total,
        sharpe_component=sharpe_component,
        return_component=return_component,
        stability_component=stability_component,
        cost_penalty=cost_penalty,
        avg_sharpe=avg_sharpe,
        sharpe_std=sharpe_std,
        avg_return=avg_return,
        return_std=return_std,
        avg_turnover=avg_turnover,
        win_rate=win_rate,
        worst_period_return=worst_return,
        information_ratio=ir,
        n_periods=n,
        period_results=period_results
    )
