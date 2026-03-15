#!/usr/bin/env python3
"""
Portfolio Rebalancing Strategies

Implements various rebalancing approaches:
- Partial rebalancing (only trade if deviation > threshold)
- Full rebalancing
- Threshold-based rebalancing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass


@dataclass
class RebalanceDecision:
    """Decision about whether to rebalance a position."""
    ticker: str
    current_weight: float
    target_weight: float
    deviation: float
    deviation_pct: float
    should_trade: bool
    trade_size: float  # Positive = buy, negative = sell


class PartialRebalancer:
    """
    Implements partial rebalancing to reduce turnover.
    Only trades positions that deviate significantly from target weights.
    
    Benefits:
    - Reduces turnover by 50-70%
    - Reduces slippage costs proportionally
    - Maintains portfolio alignment with strategy
    """
    
    def __init__(self, deviation_threshold: float = 0.20):
        """
        Args:
            deviation_threshold: Only rebalance if position deviates >20% from target
                                Example: If target is 10%, only trade if actual is <8% or >12%
        """
        self.deviation_threshold = deviation_threshold
    
    def calculate_trades(self, current_weights: Dict[str, float], 
                        target_weights: Dict[str, float],
                        portfolio_value: float) -> Tuple[Dict[str, float], List[RebalanceDecision]]:
        """
        Calculate trades needed for partial rebalancing.
        
        Args:
            current_weights: Dict of ticker -> current weight (0-1)
            target_weights: Dict of ticker -> target weight (0-1)
            portfolio_value: Total portfolio value in dollars
        
        Returns:
            Tuple of (trades dict, decisions list)
            - trades: Dict of ticker -> trade size in dollars (positive = buy, negative = sell)
            - decisions: List of RebalanceDecision objects for analysis
        """
        trades = {}
        decisions = []
        
        # Get all tickers (current + target)
        all_tickers = set(current_weights.keys()) | set(target_weights.keys())
        
        for ticker in all_tickers:
            current = current_weights.get(ticker, 0)
            target = target_weights.get(ticker, 0)
            
            # Calculate deviation
            if target > 0:
                deviation = current - target
                deviation_pct = abs(deviation) / target
            else:
                # Target is 0, so we should exit if we have a position
                deviation = current
                deviation_pct = 1.0 if current > 0 else 0
            
            # Decide whether to trade
            should_trade = deviation_pct > self.deviation_threshold
            
            # Calculate trade size
            if should_trade:
                trade_size_weight = target - current
                trade_size_dollars = trade_size_weight * portfolio_value
                trades[ticker] = trade_size_dollars
            else:
                trade_size_dollars = 0
            
            # Record decision
            decision = RebalanceDecision(
                ticker=ticker,
                current_weight=current,
                target_weight=target,
                deviation=deviation,
                deviation_pct=deviation_pct,
                should_trade=should_trade,
                trade_size=trade_size_dollars
            )
            decisions.append(decision)
        
        return trades, decisions
    
    def calculate_turnover(self, trades: Dict[str, float], 
                          portfolio_value: float) -> float:
        """
        Calculate turnover percentage.
        
        Turnover = sum(abs(trades)) / portfolio_value
        
        Args:
            trades: Dict of ticker -> trade size in dollars
            portfolio_value: Total portfolio value
        
        Returns:
            Turnover as percentage (0-1)
        """
        total_traded = sum(abs(trade) for trade in trades.values())
        return total_traded / portfolio_value if portfolio_value > 0 else 0
    
    def get_statistics(self, decisions: List[RebalanceDecision]) -> Dict:
        """
        Get rebalancing statistics.
        
        Returns:
            Dict with statistics about rebalancing decisions
        """
        total_positions = len(decisions)
        positions_traded = sum(1 for d in decisions if d.should_trade)
        positions_held = total_positions - positions_traded
        
        avg_deviation = np.mean([abs(d.deviation) for d in decisions])
        max_deviation = max([abs(d.deviation) for d in decisions]) if decisions else 0
        
        return {
            'total_positions': total_positions,
            'positions_traded': positions_traded,
            'positions_held': positions_held,
            'trade_pct': positions_traded / total_positions if total_positions > 0 else 0,
            'avg_deviation': avg_deviation,
            'max_deviation': max_deviation,
            'threshold': self.deviation_threshold
        }


class ThresholdRebalancer:
    """
    Rebalances only when portfolio drift exceeds a threshold.
    
    This is a portfolio-level approach (vs position-level in PartialRebalancer).
    """
    
    def __init__(self, portfolio_drift_threshold: float = 0.05):
        """
        Args:
            portfolio_drift_threshold: Rebalance when total drift > 5%
        """
        self.portfolio_drift_threshold = portfolio_drift_threshold
    
    def calculate_portfolio_drift(self, current_weights: Dict[str, float],
                                  target_weights: Dict[str, float]) -> float:
        """
        Calculate total portfolio drift.
        
        Drift = sum(abs(current - target)) / 2
        
        Divided by 2 because each deviation is counted twice (overweight + underweight).
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
        
        Returns:
            Total drift as percentage (0-1)
        """
        all_tickers = set(current_weights.keys()) | set(target_weights.keys())
        
        total_drift = 0
        for ticker in all_tickers:
            current = current_weights.get(ticker, 0)
            target = target_weights.get(ticker, 0)
            total_drift += abs(current - target)
        
        return total_drift / 2
    
    def should_rebalance(self, current_weights: Dict[str, float],
                        target_weights: Dict[str, float]) -> Tuple[bool, float]:
        """
        Determine if portfolio should be rebalanced.
        
        Returns:
            Tuple of (should_rebalance, drift)
        """
        drift = self.calculate_portfolio_drift(current_weights, target_weights)
        should_rebalance = drift > self.portfolio_drift_threshold
        return should_rebalance, drift
    
    def calculate_trades(self, current_weights: Dict[str, float],
                        target_weights: Dict[str, float],
                        portfolio_value: float) -> Dict[str, float]:
        """
        Calculate trades for full rebalancing (if threshold exceeded).
        
        Returns:
            Dict of ticker -> trade size in dollars
        """
        should_rebalance, drift = self.should_rebalance(current_weights, target_weights)
        
        if not should_rebalance:
            return {}
        
        # Full rebalance
        trades = {}
        all_tickers = set(current_weights.keys()) | set(target_weights.keys())
        
        for ticker in all_tickers:
            current = current_weights.get(ticker, 0)
            target = target_weights.get(ticker, 0)
            trade_size = (target - current) * portfolio_value
            if abs(trade_size) > 0:
                trades[ticker] = trade_size
        
        return trades


class SmartRebalancer:
    """
    Combines partial rebalancing with portfolio drift monitoring.
    
    - Uses partial rebalancing for individual positions
    - But forces full rebalance if total drift is too high
    """
    
    def __init__(self, position_threshold: float = 0.20,
                 portfolio_threshold: float = 0.10):
        """
        Args:
            position_threshold: Position-level deviation threshold (20%)
            portfolio_threshold: Portfolio-level drift threshold (10%)
        """
        self.partial_rebalancer = PartialRebalancer(position_threshold)
        self.threshold_rebalancer = ThresholdRebalancer(portfolio_threshold)
    
    def calculate_trades(self, current_weights: Dict[str, float],
                        target_weights: Dict[str, float],
                        portfolio_value: float) -> Tuple[Dict[str, float], Dict]:
        """
        Calculate trades using smart rebalancing logic.
        
        Returns:
            Tuple of (trades, metadata)
        """
        # Check portfolio-level drift
        should_full_rebalance, drift = self.threshold_rebalancer.should_rebalance(
            current_weights, target_weights
        )
        
        if should_full_rebalance:
            # Force full rebalance
            trades = self.threshold_rebalancer.calculate_trades(
                current_weights, target_weights, portfolio_value
            )
            metadata = {
                'rebalance_type': 'full',
                'reason': 'portfolio_drift_exceeded',
                'drift': drift,
                'threshold': self.threshold_rebalancer.portfolio_drift_threshold
            }
        else:
            # Partial rebalance
            trades, decisions = self.partial_rebalancer.calculate_trades(
                current_weights, target_weights, portfolio_value
            )
            stats = self.partial_rebalancer.get_statistics(decisions)
            metadata = {
                'rebalance_type': 'partial',
                'reason': 'position_deviations',
                'drift': drift,
                **stats
            }
        
        return trades, metadata


# Example usage
if __name__ == "__main__":
    # Example: Partial Rebalancing
    rebalancer = PartialRebalancer(deviation_threshold=0.20)
    
    # Current portfolio
    current_weights = {
        'AAPL': 0.35,  # Target was 0.33, now 35% (6% deviation)
        'GOOGL': 0.28,  # Target was 0.33, now 28% (15% deviation)
        'MSFT': 0.37,  # Target was 0.33, now 37% (12% deviation)
    }
    
    # Target portfolio (equal weight)
    target_weights = {
        'AAPL': 0.33,
        'GOOGL': 0.33,
        'MSFT': 0.33,
    }
    
    portfolio_value = 100000
    
    # Calculate trades
    trades, decisions = rebalancer.calculate_trades(
        current_weights, target_weights, portfolio_value
    )
    
    print("Rebalancing Decisions:")
    for decision in decisions:
        print(f"{decision.ticker}: {decision.current_weight:.1%} -> {decision.target_weight:.1%} "
              f"(deviation: {decision.deviation_pct:.1%}, trade: {'YES' if decision.should_trade else 'NO'})")
    
    print(f"\nTrades:")
    for ticker, size in trades.items():
        print(f"{ticker}: ${size:,.2f}")
    
    print(f"\nTurnover: {rebalancer.calculate_turnover(trades, portfolio_value):.1%}")
    
    stats = rebalancer.get_statistics(decisions)
    print(f"\nStatistics:")
    print(f"  Positions traded: {stats['positions_traded']}/{stats['total_positions']} ({stats['trade_pct']:.1%})")
    print(f"  Avg deviation: {stats['avg_deviation']:.1%}")
    print(f"  Max deviation: {stats['max_deviation']:.1%}")
