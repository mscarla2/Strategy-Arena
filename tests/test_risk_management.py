#!/usr/bin/env python3
"""
Test Risk Management Integration

Runs the Strategy Arena with and without risk management to measure impact.
Generates comprehensive comparison report.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.fetcher import DataFetcher
from data.universe import get_oil_universe
from backtest.risk_management import (
    MicrocapSlippageModel,
    DilutionFilter,
    LiquidityConstraint,
    RiskManager
)
from backtest.impact_measurement import ImpactAnalyzer, PerformanceMetrics


def load_oil_data():
    """Load oil stock data for testing."""
    print("📊 Loading oil stock data...")
    
    tickers = get_oil_universe()
    start_date = "2025-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    fetcher = DataFetcher()
    prices = fetcher.fetch(start_date, end_date, tickers, use_cache=True)
    
    # For this test, we'll use prices as a proxy for volume
    # In production, you'd fetch actual volume data
    volume = prices * 1000000  # Simulate volume
    
    print(f"  ✓ Loaded {len(prices)} days, {len(prices.columns)} tickers")
    print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    return prices, volume, tickers


def simulate_strategy_without_risk_mgmt(prices, volume, tickers):
    """
    Simulate strategy WITHOUT risk management.
    
    This represents the naive backtest that ignores:
    - Slippage
    - Dilution events
    - Liquidity constraints
    """
    print("\n🔴 Running strategy WITHOUT risk management...")
    
    # Simple momentum strategy: buy top 3 by 20-day return
    period_results = []
    
    # Generate test periods (2-month periods)
    start_date = prices.index[0]
    end_date = prices.index[-1]
    
    current_date = start_date + pd.DateOffset(months=2)
    
    while current_date <= end_date:
        period_start = current_date - pd.DateOffset(months=2)
        period_end = current_date
        
        # Get period data
        period_mask = (prices.index >= period_start) & (prices.index <= period_end)
        period_prices = prices[period_mask]
        
        if len(period_prices) < 20:
            current_date += pd.DateOffset(months=1)
            continue
        
        # Calculate 20-day momentum
        lookback_prices = prices[prices.index < period_start].tail(20)
        if len(lookback_prices) < 20:
            current_date += pd.DateOffset(months=1)
            continue
        
        momentum = (lookback_prices.iloc[-1] / lookback_prices.iloc[0] - 1)
        
        # Select top 3
        top_3 = momentum.nlargest(3).index.tolist()
        
        # Calculate equal-weight returns
        period_returns = period_prices[top_3].pct_change().mean(axis=1)
        
        # Calculate metrics
        total_return = (1 + period_returns).prod() - 1
        sharpe = (period_returns.mean() / period_returns.std()) * np.sqrt(252) if period_returns.std() > 0 else 0
        
        cumulative = (1 + period_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = abs(drawdown.min())
        
        # Turnover (assume 100% turnover each period for simplicity)
        turnover = 1.0
        
        period_results.append({
            'period_start': period_start.strftime('%Y-%m-%d'),
            'period_end': period_end.strftime('%Y-%m-%d'),
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'turnover': turnover,
            'n_days': len(period_returns)
        })
        
        current_date += pd.DateOffset(months=1)
    
    print(f"  ✓ Completed {len(period_results)} periods")
    
    return period_results


def simulate_strategy_with_risk_mgmt(prices, volume, tickers):
    """
    Simulate strategy WITH risk management.
    
    This represents realistic backtest that includes:
    - Slippage modeling
    - Dilution filtering
    - Liquidity constraints
    """
    print("\n🟢 Running strategy WITH risk management...")
    
    # Initialize risk manager
    risk_manager = RiskManager(
        slippage_model=MicrocapSlippageModel(base_slippage_bps=25),
        dilution_filter=DilutionFilter(volume_spike_threshold=5.0),
        liquidity_constraint=LiquidityConstraint(max_pct_adv=0.05, min_adv_dollars=50000)
    )
    
    # Filter universe by liquidity
    liquid_tickers = risk_manager.filter_universe(tickers, prices, volume)
    print(f"  Filtered to {len(liquid_tickers)} liquid tickers (from {len(tickers)})")
    
    period_results = []
    slippage_costs = []
    dilution_events = []
    position_reductions = 0
    
    # Generate test periods
    start_date = prices.index[0]
    end_date = prices.index[-1]
    
    current_date = start_date + pd.DateOffset(months=2)
    current_positions = []
    
    while current_date <= end_date:
        period_start = current_date - pd.DateOffset(months=2)
        period_end = current_date
        
        # Get period data
        period_mask = (prices.index >= period_start) & (prices.index <= period_end)
        period_prices = prices[period_mask]
        period_volume = volume[period_mask]
        
        if len(period_prices) < 20:
            current_date += pd.DateOffset(months=1)
            continue
        
        # Calculate 20-day momentum
        lookback_prices = prices[prices.index < period_start].tail(20)
        if len(lookback_prices) < 20:
            current_date += pd.DateOffset(months=1)
            continue
        
        momentum = (lookback_prices.iloc[-1] / lookback_prices.iloc[0] - 1)
        
        # Select top 3 from liquid tickers only
        liquid_momentum = momentum[liquid_tickers]
        top_3 = liquid_momentum.nlargest(3).index.tolist()
        
        # Check for dilution exits
        dilution_exits = risk_manager.check_dilution_exits(
            period_prices, period_volume, current_positions
        )
        
        if dilution_exits:
            print(f"  ⚠️  Dilution detected: {dilution_exits}")
            # Remove dilution tickers from selection
            top_3 = [t for t in top_3 if t not in dilution_exits]
            dilution_events.append(len(dilution_exits))
        else:
            dilution_events.append(0)
        
        # Calculate positions with liquidity constraints
        capital = 100000
        equal_weight = capital / len(top_3) if top_3 else 0
        positions = {ticker: equal_weight for ticker in top_3}
        
        # Enforce liquidity constraints
        adjusted_positions = risk_manager.enforce_position_limits(
            positions, period_prices, period_volume
        )
        
        # Count position reductions
        for ticker in positions:
            if adjusted_positions.get(ticker, 0) < positions[ticker]:
                position_reductions += 1
        
        # Calculate returns
        if adjusted_positions:
            period_returns = period_prices[list(adjusted_positions.keys())].pct_change().mean(axis=1)
        else:
            period_returns = pd.Series(0, index=period_prices.index)
        
        # Calculate metrics
        total_return = (1 + period_returns).prod() - 1
        sharpe = (period_returns.mean() / period_returns.std()) * np.sqrt(252) if period_returns.std() > 0 else 0
        
        cumulative = (1 + period_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = abs(drawdown.min())
        
        turnover = 1.0
        
        # Calculate slippage
        adjusted_return, slippage_cost, _ = risk_manager.calculate_adjusted_returns(
            total_return, turnover, adjusted_positions, period_prices, period_volume
        )
        
        slippage_costs.append(slippage_cost)
        
        period_results.append({
            'period_start': period_start.strftime('%Y-%m-%d'),
            'period_end': period_end.strftime('%Y-%m-%d'),
            'total_return': adjusted_return,  # Use adjusted return
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'turnover': turnover,
            'n_days': len(period_returns)
        })
        
        current_positions = list(adjusted_positions.keys())
        current_date += pd.DateOffset(months=1)
    
    print(f"  ✓ Completed {len(period_results)} periods")
    print(f"  Total dilution events: {sum(dilution_events)}")
    print(f"  Position reductions: {position_reductions}")
    
    return period_results, slippage_costs, dilution_events, len(tickers) - len(liquid_tickers), position_reductions


def main():
    """Main test function."""
    print("="*80)
    print("RISK MANAGEMENT IMPACT TEST")
    print("="*80)
    
    # Load data
    prices, volume, tickers = load_oil_data()
    
    # Run without risk management
    before_results = simulate_strategy_without_risk_mgmt(prices, volume, tickers)
    
    # Run with risk management
    after_results, slippage_costs, dilution_events, illiquid_filtered, position_reductions = \
        simulate_strategy_with_risk_mgmt(prices, volume, tickers)
    
    # Analyze impact
    print("\n📊 Analyzing impact...")
    analyzer = ImpactAnalyzer()
    
    before_metrics = analyzer.calculate_performance_metrics(before_results, "before")
    after_metrics = analyzer.calculate_performance_metrics(after_results, "after")
    impact_metrics = analyzer.compare_strategies(
        before_results, after_results, slippage_costs,
        dilution_events, illiquid_filtered, position_reductions
    )
    
    # Print comparison table
    analyzer.print_comparison_table(before_metrics, after_metrics, impact_metrics)
    
    # Generate report
    report = analyzer.generate_comparison_report(
        before_results, after_results, slippage_costs,
        dilution_events, illiquid_filtered, position_reductions,
        output_path="data/cache/risk_management_impact_report.json"
    )
    
    # Print key findings
    print("\n🔍 KEY FINDINGS")
    print("-"*80)
    for finding in report['key_findings']:
        print(f"  {finding}")
    
    print("\n💡 RECOMMENDATIONS")
    print("-"*80)
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    print("\n" + "="*80)
    print("✅ Test complete! Report saved to: data/cache/risk_management_impact_report.json")
    print("="*80)


if __name__ == "__main__":
    main()
