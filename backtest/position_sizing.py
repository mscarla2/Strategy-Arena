#!/usr/bin/env python3
"""
Position Sizing Strategies

Implements various position sizing methods:
- Kelly Criterion
- Volatility-adjusted sizing
- Risk parity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    ticker: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    return_pct: float
    hold_days: int


class KellyCriterion:
    """
    Kelly Criterion for optimal position sizing.
    
    Kelly % = (Win Rate × Avg Win - Loss Rate × Avg Loss) / Avg Win
    
    The Kelly Criterion maximizes long-term growth rate but can be aggressive.
    We use fractional Kelly (25% max) for safety.
    """
    
    def __init__(self, lookback_trades: int = 20, max_kelly: float = 0.25):
        """
        Args:
            lookback_trades: Number of recent trades to analyze
            max_kelly: Maximum Kelly fraction (cap at 25% for safety)
        """
        self.lookback_trades = lookback_trades
        self.max_kelly = max_kelly
        self.trade_history: List[TradeRecord] = []
    
    def record_trade(self, trade: TradeRecord):
        """Record a completed trade."""
        self.trade_history.append(trade)
        if len(self.trade_history) > self.lookback_trades:
            self.trade_history.pop(0)
    
    def calculate_kelly_fraction(self, ticker: str = None) -> float:
        """
        Calculate Kelly fraction for position sizing.
        
        Args:
            ticker: Optional ticker to calculate ticker-specific Kelly
        
        Returns:
            Kelly fraction (0-1)
        """
        if len(self.trade_history) < 10:
            return 0.10  # Conservative default
        
        # Filter by ticker if specified
        if ticker:
            trades = [t for t in self.trade_history if t.ticker == ticker]
        else:
            trades = self.trade_history
        
        if len(trades) < 5:
            return 0.10
        
        # Calculate win rate and payoff ratio
        returns = [t.return_pct for t in trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        
        if not wins or not losses:
            return 0.10
        
        win_rate = len(wins) / len(returns)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        # Kelly formula
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Cap at max_kelly for safety (full Kelly is too aggressive)
        kelly = max(0, min(kelly, self.max_kelly))
        
        return kelly
    
    def calculate_position_size(self, ticker: str, capital: float, 
                               signal_strength: float = 1.0) -> float:
        """
        Calculate position size using Kelly Criterion.
        
        Args:
            ticker: Stock ticker
            capital: Available capital
            signal_strength: Signal strength (0-1)
        
        Returns:
            Position size in dollars
        """
        kelly_fraction = self.calculate_kelly_fraction(ticker)
        position_size = capital * kelly_fraction * signal_strength
        return position_size
    
    def get_statistics(self, ticker: str = None) -> Dict:
        """
        Get trading statistics for Kelly calculation.
        
        Args:
            ticker: Optional ticker to filter by
        
        Returns:
            Dict with win rate, avg win, avg loss, Kelly fraction
        """
        if ticker:
            trades = [t for t in self.trade_history if t.ticker == ticker]
        else:
            trades = self.trade_history
        
        if not trades:
            return {
                'n_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'kelly_fraction': 0
            }
        
        returns = [t.return_pct for t in trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        
        return {
            'n_trades': len(trades),
            'win_rate': len(wins) / len(returns) if returns else 0,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': abs(np.mean(losses)) if losses else 0,
            'kelly_fraction': self.calculate_kelly_fraction(ticker)
        }


class VolatilityAdjustedSizing:
    """
    Inverse volatility weighting for position sizing.
    Reduces exposure during high-volatility periods.
    
    This is a risk parity approach that allocates capital inversely
    proportional to volatility, ensuring equal risk contribution.
    """
    
    def __init__(self, lookback: int = 20, target_volatility: float = 0.15):
        """
        Args:
            lookback: Volatility calculation period (days)
            target_volatility: Target portfolio volatility (15% annual)
        """
        self.lookback = lookback
        self.target_volatility = target_volatility
    
    def calculate_volatility(self, returns: pd.Series) -> float:
        """
        Calculate annualized volatility.
        
        Args:
            returns: Daily returns series
        
        Returns:
            Annualized volatility
        """
        if len(returns) < self.lookback:
            return 0.15  # Default to 15% if insufficient data
        
        vol = returns.rolling(self.lookback).std().iloc[-1] * np.sqrt(252)
        return vol if not np.isnan(vol) else 0.15
    
    def calculate_position_size(self, ticker: str, base_size: float,
                               returns: pd.Series) -> float:
        """
        Adjust position size based on volatility.
        
        Args:
            ticker: Stock ticker
            base_size: Base position size (from Kelly or equal-weight)
            returns: Historical returns series
        
        Returns:
            Volatility-adjusted position size
        """
        current_vol = self.calculate_volatility(returns)
        
        if current_vol == 0:
            return base_size
        
        # Scale position inversely with volatility
        vol_scalar = self.target_volatility / current_vol
        
        # Cap scalar at 2x (don't over-leverage in low vol)
        vol_scalar = min(vol_scalar, 2.0)
        
        adjusted_size = base_size * vol_scalar
        return adjusted_size
    
    def calculate_portfolio_weights(self, tickers: List[str], 
                                   returns_dict: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        Calculate inverse volatility weights for portfolio.
        
        This implements risk parity: each position contributes equal risk.
        
        Args:
            tickers: List of stock tickers
            returns_dict: Dict of ticker -> returns series
        
        Returns:
            Dict of ticker -> weight (sums to 1.0)
        """
        volatilities = {}
        for ticker in tickers:
            if ticker not in returns_dict:
                continue
            vol = self.calculate_volatility(returns_dict[ticker])
            volatilities[ticker] = vol if vol > 0 else 0.01
        
        if not volatilities:
            # Equal weight if no data
            return {t: 1.0 / len(tickers) for t in tickers}
        
        # Inverse volatility weights
        inv_vols = {t: 1/v for t, v in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())
        
        weights = {t: iv / total_inv_vol for t, iv in inv_vols.items()}
        return weights
    
    def get_volatility_regime(self, returns: pd.Series) -> str:
        """
        Classify current volatility regime.
        
        Args:
            returns: Returns series
        
        Returns:
            'low', 'normal', or 'high'
        """
        current_vol = self.calculate_volatility(returns)
        
        if current_vol < 0.10:
            return 'low'
        elif current_vol < 0.25:
            return 'normal'
        else:
            return 'high'


class CombinedPositionSizer:
    """
    Combines Kelly Criterion and Volatility-Adjusted sizing.
    
    Final position size = Kelly size × Volatility scalar
    
    This provides optimal growth (Kelly) with risk management (vol adjustment).
    """
    
    def __init__(self, kelly_lookback: int = 20, kelly_max: float = 0.25,
                 vol_lookback: int = 20, target_vol: float = 0.15):
        """
        Args:
            kelly_lookback: Lookback for Kelly calculation
            kelly_max: Maximum Kelly fraction
            vol_lookback: Lookback for volatility calculation
            target_vol: Target portfolio volatility
        """
        self.kelly = KellyCriterion(kelly_lookback, kelly_max)
        self.vol_sizer = VolatilityAdjustedSizing(vol_lookback, target_vol)
    
    def record_trade(self, trade: TradeRecord):
        """Record a completed trade for Kelly calculation."""
        self.kelly.record_trade(trade)
    
    def calculate_position_size(self, ticker: str, capital: float,
                               returns: pd.Series, signal_strength: float = 1.0) -> float:
        """
        Calculate position size using combined Kelly + Vol adjustment.
        
        Args:
            ticker: Stock ticker
            capital: Available capital
            returns: Historical returns series
            signal_strength: Signal strength (0-1)
        
        Returns:
            Position size in dollars
        """
        # Kelly-based size
        kelly_size = self.kelly.calculate_position_size(ticker, capital, signal_strength)
        
        # Volatility adjustment
        final_size = self.vol_sizer.calculate_position_size(ticker, kelly_size, returns)
        
        return final_size
    
    def get_statistics(self, ticker: str = None) -> Dict:
        """Get combined statistics."""
        kelly_stats = self.kelly.get_statistics(ticker)
        
        return {
            **kelly_stats,
            'target_volatility': self.vol_sizer.target_volatility,
            'vol_lookback': self.vol_sizer.lookback
        }


# Example usage
if __name__ == "__main__":
    # Example: Kelly Criterion
    kelly = KellyCriterion(lookback_trades=20, max_kelly=0.25)
    
    # Simulate some trades
    trades = [
        TradeRecord('AAPL', '2024-01-01', '2024-01-10', 150, 155, 0.033, 10),
        TradeRecord('AAPL', '2024-01-15', '2024-01-20', 155, 152, -0.019, 5),
        TradeRecord('AAPL', '2024-01-25', '2024-02-05', 152, 160, 0.053, 11),
    ]
    
    for trade in trades:
        kelly.record_trade(trade)
    
    print("Kelly Statistics:")
    print(kelly.get_statistics('AAPL'))
    print(f"Position size for $100k: ${kelly.calculate_position_size('AAPL', 100000):.2f}")
    
    # Example: Volatility-Adjusted Sizing
    vol_sizer = VolatilityAdjustedSizing(lookback=20, target_volatility=0.15)
    
    # Simulate returns
    returns = pd.Series(np.random.randn(100) * 0.02)  # 2% daily vol
    
    print(f"\nVolatility: {vol_sizer.calculate_volatility(returns):.2%}")
    print(f"Regime: {vol_sizer.get_volatility_regime(returns)}")
    print(f"Adjusted size for $10k base: ${vol_sizer.calculate_position_size('AAPL', 10000, returns):.2f}")
