"""
Impact Measurement Framework

Compares strategy performance before and after risk management enhancements.
Provides detailed metrics to quantify the impact of:
1. Slippage modeling
2. Dilution filtering
3. Liquidity constraints
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class PerformanceMetrics:
    """Performance metrics for a strategy."""
    # Returns
    total_return: float
    annualized_return: float
    avg_period_return: float
    
    # Risk
    volatility: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Consistency
    win_rate: float
    worst_period: float
    best_period: float
    return_std: float
    
    # Trading
    avg_turnover: float
    n_periods: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ImpactMetrics:
    """Metrics showing impact of risk management."""
    # Return impact
    return_reduction_pct: float  # How much returns decreased
    return_reduction_abs: float  # Absolute decrease
    
    # Risk impact
    drawdown_reduction_pct: float  # How much DD decreased
    volatility_reduction_pct: float  # How much vol decreased
    
    # Risk-adjusted impact
    sharpe_improvement: float  # Change in Sharpe
    calmar_improvement: float  # Change in Calmar
    sortino_improvement: float  # Change in Sortino
    
    # Realism gap
    backtest_vs_reality_gap: float  # Gap between naive and realistic
    
    # Slippage impact
    avg_slippage_cost_per_period: float
    total_slippage_cost: float
    
    # Dilution impact
    dilution_events_avoided: int
    dilution_cost_avoided: float
    
    # Liquidity impact
    illiquid_tickers_filtered: int
    position_size_reductions: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class ImpactAnalyzer:
    """
    Analyze impact of risk management enhancements.
    """
    
    def __init__(self):
        self.results = {}
    
    def calculate_performance_metrics(
        self,
        period_results: List[Dict],
        label: str = "strategy"
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            period_results: List of period result dicts
            label: Label for this strategy
        
        Returns:
            PerformanceMetrics object
        """
        if not period_results:
            return PerformanceMetrics(
                total_return=0, annualized_return=0, avg_period_return=0,
                volatility=0, max_drawdown=0, sharpe_ratio=0,
                sortino_ratio=0, calmar_ratio=0, win_rate=0,
                worst_period=0, best_period=0, return_std=0,
                avg_turnover=0, n_periods=0
            )
        
        # Extract metrics
        returns = [r['total_return'] for r in period_results]
        sharpes = [r.get('sharpe_ratio', 0) for r in period_results]
        turnovers = [r.get('turnover', 0) for r in period_results]
        
        # Calculate cumulative returns
        cumulative = np.cumprod([1 + r for r in returns])
        total_return = cumulative[-1] - 1
        
        # Annualized return (assuming 2-month periods)
        n_periods = len(returns)
        periods_per_year = 6  # 12 months / 2 months
        annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(periods_per_year)
        
        # Max drawdown
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Sharpe ratio
        avg_sharpe = np.mean(sharpes)
        
        # Sortino ratio (downside deviation)
        downside_returns = [r for r in returns if r < 0]
        if downside_returns:
            downside_dev = np.std(downside_returns) * np.sqrt(periods_per_year)
            sortino_ratio = annualized_return / downside_dev if downside_dev > 0 else 0
        else:
            sortino_ratio = annualized_return * 10  # Arbitrary large number
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else annualized_return * 10
        
        # Win rate
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
        
        # Best/worst periods
        worst_period = min(returns)
        best_period = max(returns)
        
        # Return std
        return_std = np.std(returns)
        
        # Avg turnover
        avg_turnover = np.mean(turnovers)
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            avg_period_return=np.mean(returns),
            volatility=volatility,
            max_drawdown=max_drawdown,
            sharpe_ratio=avg_sharpe,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            worst_period=worst_period,
            best_period=best_period,
            return_std=return_std,
            avg_turnover=avg_turnover,
            n_periods=n_periods
        )
    
    def compare_strategies(
        self,
        before_results: List[Dict],
        after_results: List[Dict],
        slippage_costs: List[float],
        dilution_events: List[int],
        illiquid_filtered: int,
        position_reductions: int
    ) -> ImpactMetrics:
        """
        Compare performance before and after risk management.
        
        Args:
            before_results: Period results without risk management
            after_results: Period results with risk management
            slippage_costs: Slippage costs per period
            dilution_events: Dilution events detected per period
            illiquid_filtered: Number of illiquid tickers filtered
            position_reductions: Number of position size reductions
        
        Returns:
            ImpactMetrics object
        """
        # Calculate metrics for both
        before_metrics = self.calculate_performance_metrics(before_results, "before")
        after_metrics = self.calculate_performance_metrics(after_results, "after")
        
        # Return impact
        return_reduction_abs = before_metrics.total_return - after_metrics.total_return
        return_reduction_pct = (return_reduction_abs / before_metrics.total_return * 100) if before_metrics.total_return != 0 else 0
        
        # Risk impact
        drawdown_reduction_abs = before_metrics.max_drawdown - after_metrics.max_drawdown
        drawdown_reduction_pct = (drawdown_reduction_abs / before_metrics.max_drawdown * 100) if before_metrics.max_drawdown != 0 else 0
        
        volatility_reduction_abs = before_metrics.volatility - after_metrics.volatility
        volatility_reduction_pct = (volatility_reduction_abs / before_metrics.volatility * 100) if before_metrics.volatility != 0 else 0
        
        # Risk-adjusted impact
        sharpe_improvement = after_metrics.sharpe_ratio - before_metrics.sharpe_ratio
        calmar_improvement = after_metrics.calmar_ratio - before_metrics.calmar_ratio
        sortino_improvement = after_metrics.sortino_ratio - before_metrics.sortino_ratio
        
        # Realism gap
        backtest_vs_reality_gap = return_reduction_abs
        
        # Slippage impact
        avg_slippage_cost = np.mean(slippage_costs) if slippage_costs else 0
        total_slippage_cost = sum(slippage_costs) if slippage_costs else 0
        
        # Dilution impact
        total_dilution_events = sum(dilution_events) if dilution_events else 0
        # Estimate cost: assume each dilution event costs 20% on average
        dilution_cost_avoided = total_dilution_events * 0.20
        
        return ImpactMetrics(
            return_reduction_pct=return_reduction_pct,
            return_reduction_abs=return_reduction_abs,
            drawdown_reduction_pct=drawdown_reduction_pct,
            volatility_reduction_pct=volatility_reduction_pct,
            sharpe_improvement=sharpe_improvement,
            calmar_improvement=calmar_improvement,
            sortino_improvement=sortino_improvement,
            backtest_vs_reality_gap=backtest_vs_reality_gap,
            avg_slippage_cost_per_period=avg_slippage_cost,
            total_slippage_cost=total_slippage_cost,
            dilution_events_avoided=total_dilution_events,
            dilution_cost_avoided=dilution_cost_avoided,
            illiquid_tickers_filtered=illiquid_filtered,
            position_size_reductions=position_reductions
        )
    
    def generate_comparison_report(
        self,
        before_results: List[Dict],
        after_results: List[Dict],
        slippage_costs: List[float],
        dilution_events: List[int],
        illiquid_filtered: int,
        position_reductions: int,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Generate comprehensive comparison report.
        
        Args:
            before_results: Period results without risk management
            after_results: Period results with risk management
            slippage_costs: Slippage costs per period
            dilution_events: Dilution events detected per period
            illiquid_filtered: Number of illiquid tickers filtered
            position_reductions: Number of position size reductions
            output_path: Optional path to save report
        
        Returns:
            Report dictionary
        """
        # Calculate metrics
        before_metrics = self.calculate_performance_metrics(before_results, "before")
        after_metrics = self.calculate_performance_metrics(after_results, "after")
        impact_metrics = self.compare_strategies(
            before_results, after_results, slippage_costs,
            dilution_events, illiquid_filtered, position_reductions
        )
        
        # Build report
        report = {
            'summary': {
                'before': before_metrics.to_dict(),
                'after': after_metrics.to_dict(),
                'impact': impact_metrics.to_dict()
            },
            'key_findings': self._generate_key_findings(
                before_metrics, after_metrics, impact_metrics
            ),
            'recommendations': self._generate_recommendations(impact_metrics)
        }
        
        # Save if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def _generate_key_findings(
        self,
        before: PerformanceMetrics,
        after: PerformanceMetrics,
        impact: ImpactMetrics
    ) -> List[str]:
        """Generate key findings from comparison."""
        findings = []
        
        # Return impact
        if impact.return_reduction_pct > 50:
            findings.append(
                f"⚠️  CRITICAL: Returns decreased by {impact.return_reduction_pct:.1f}% "
                f"({before.total_return:.1%} → {after.total_return:.1%}). "
                f"This is the 'reality gap' between backtest and live trading."
            )
        elif impact.return_reduction_pct > 20:
            findings.append(
                f"⚠️  Returns decreased by {impact.return_reduction_pct:.1f}% "
                f"({before.total_return:.1%} → {after.total_return:.1%}) due to realistic costs."
            )
        
        # Risk-adjusted improvement
        if impact.sharpe_improvement > 0.2:
            findings.append(
                f"✅ Sharpe ratio improved by {impact.sharpe_improvement:.2f} "
                f"({before.sharpe_ratio:.2f} → {after.sharpe_ratio:.2f}), "
                f"indicating better risk-adjusted returns."
            )
        
        if impact.calmar_improvement > 0.5:
            findings.append(
                f"✅ Calmar ratio improved by {impact.calmar_improvement:.2f} "
                f"({before.calmar_ratio:.2f} → {after.calmar_ratio:.2f}), "
                f"indicating smoother equity curve."
            )
        
        # Drawdown improvement
        if impact.drawdown_reduction_pct > 20:
            findings.append(
                f"✅ Max drawdown reduced by {impact.drawdown_reduction_pct:.1f}% "
                f"({before.max_drawdown:.1%} → {after.max_drawdown:.1%}), "
                f"indicating better risk management."
            )
        
        # Slippage impact
        if impact.total_slippage_cost > 0.05:
            findings.append(
                f"💰 Slippage costs: {impact.total_slippage_cost:.1%} total "
                f"({impact.avg_slippage_cost_per_period:.2%} per period). "
                f"This is the cost of trading microcaps."
            )
        
        # Dilution impact
        if impact.dilution_events_avoided > 0:
            findings.append(
                f"🚨 Avoided {impact.dilution_events_avoided} dilution events, "
                f"saving approximately {impact.dilution_cost_avoided:.1%}. "
                f"This is critical for microcap trading."
            )
        
        # Liquidity impact
        if impact.illiquid_tickers_filtered > 0:
            findings.append(
                f"🔍 Filtered {impact.illiquid_tickers_filtered} illiquid tickers. "
                f"These would have caused severe slippage in live trading."
            )
        
        return findings
    
    def _generate_recommendations(self, impact: ImpactMetrics) -> List[str]:
        """Generate recommendations based on impact."""
        recommendations = []
        
        # If returns dropped significantly
        if impact.return_reduction_pct > 50:
            recommendations.append(
                "Consider the 'after' returns as realistic expectations for live trading. "
                "The 'before' returns were inflated by ignoring transaction costs."
            )
        
        # If risk-adjusted metrics improved
        if impact.sharpe_improvement > 0 or impact.calmar_improvement > 0:
            recommendations.append(
                "Risk-adjusted metrics improved despite lower returns. "
                "This indicates a more robust strategy suitable for live trading."
            )
        
        # If dilution events detected
        if impact.dilution_events_avoided > 0:
            recommendations.append(
                "Dilution filter is critical for microcaps. "
                "Continue monitoring for ATM offerings and volume spikes."
            )
        
        # If many illiquid tickers filtered
        if impact.illiquid_tickers_filtered > 2:
            recommendations.append(
                "Consider increasing minimum ADV threshold to further reduce slippage risk. "
                f"Currently filtered {impact.illiquid_tickers_filtered} tickers."
            )
        
        # If slippage is high
        if impact.avg_slippage_cost_per_period > 0.02:
            recommendations.append(
                "Slippage costs are high (>2% per period). "
                "Consider reducing turnover or increasing position hold times."
            )
        
        return recommendations
    
    def print_comparison_table(
        self,
        before: PerformanceMetrics,
        after: PerformanceMetrics,
        impact: ImpactMetrics
    ):
        """Print formatted comparison table."""
        print("\n" + "="*80)
        print("RISK MANAGEMENT IMPACT ANALYSIS")
        print("="*80)
        
        print("\n📊 PERFORMANCE COMPARISON")
        print("-"*80)
        print(f"{'Metric':<30} {'Before':<15} {'After':<15} {'Change':<15}")
        print("-"*80)
        
        # Returns
        print(f"{'Total Return':<30} {before.total_return:>14.2%} {after.total_return:>14.2%} {impact.return_reduction_abs:>14.2%}")
        print(f"{'Annualized Return':<30} {before.annualized_return:>14.2%} {after.annualized_return:>14.2%}")
        print(f"{'Avg Period Return':<30} {before.avg_period_return:>14.2%} {after.avg_period_return:>14.2%}")
        
        # Risk
        print(f"{'Max Drawdown':<30} {before.max_drawdown:>14.2%} {after.max_drawdown:>14.2%} {-impact.drawdown_reduction_pct:>13.1f}%")
        print(f"{'Volatility':<30} {before.volatility:>14.2%} {after.volatility:>14.2%} {-impact.volatility_reduction_pct:>13.1f}%")
        
        # Risk-adjusted
        print(f"{'Sharpe Ratio':<30} {before.sharpe_ratio:>14.2f} {after.sharpe_ratio:>14.2f} {impact.sharpe_improvement:>+14.2f}")
        print(f"{'Sortino Ratio':<30} {before.sortino_ratio:>14.2f} {after.sortino_ratio:>14.2f} {impact.sortino_improvement:>+14.2f}")
        print(f"{'Calmar Ratio':<30} {before.calmar_ratio:>14.2f} {after.calmar_ratio:>14.2f} {impact.calmar_improvement:>+14.2f}")
        
        # Consistency
        print(f"{'Win Rate':<30} {before.win_rate:>14.2%} {after.win_rate:>14.2%}")
        print(f"{'Worst Period':<30} {before.worst_period:>14.2%} {after.worst_period:>14.2%}")
        print(f"{'Best Period':<30} {before.best_period:>14.2%} {after.best_period:>14.2%}")
        
        print("\n💰 COST BREAKDOWN")
        print("-"*80)
        print(f"{'Slippage Cost (Total)':<30} {impact.total_slippage_cost:>14.2%}")
        print(f"{'Slippage Cost (Per Period)':<30} {impact.avg_slippage_cost_per_period:>14.2%}")
        print(f"{'Dilution Events Avoided':<30} {impact.dilution_events_avoided:>14d}")
        print(f"{'Dilution Cost Avoided':<30} {impact.dilution_cost_avoided:>14.2%}")
        
        print("\n🔍 LIQUIDITY FILTERS")
        print("-"*80)
        print(f"{'Illiquid Tickers Filtered':<30} {impact.illiquid_tickers_filtered:>14d}")
        print(f"{'Position Size Reductions':<30} {impact.position_size_reductions:>14d}")
        
        print("\n" + "="*80)
