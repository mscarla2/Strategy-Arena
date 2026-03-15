#!/usr/bin/env python3
"""
Multi-Timeframe Strategy Framework

Supports multiple trading timeframes with appropriate slippage models:
- Intraday (5-min candles, 1-4 hour holds)
- Swing (daily candles, 2-10 day holds)
- Weekly (daily candles, 7-15 day holds)
- Monthly (daily candles, 20-60 day holds)

Each timeframe has:
- Appropriate slippage model (higher for intraday)
- Appropriate features (VWAP for intraday, momentum for swing)
- Appropriate fitness function (penalizes excessive turnover)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

from backtest.risk_management import MicrocapSlippageModel, DilutionFilter, LiquidityConstraint
from backtest.intraday_features import IntradayFeatures


class Timeframe(Enum):
    """Trading timeframe enum."""
    INTRADAY = "intraday"  # 5-min candles, 1-4 hour holds
    SWING = "swing"        # Daily candles, 2-10 day holds
    WEEKLY = "weekly"      # Daily candles, 7-15 day holds
    MONTHLY = "monthly"    # Daily candles, 20-60 day holds


@dataclass
class TimeframeConfig:
    """Configuration for a specific timeframe."""
    name: str
    data_interval: str  # '5min', '1d'
    target_hold_days: float  # Average hold period in days
    base_slippage_bps: float  # Base slippage in basis points
    commission_per_trade: float  # Fixed commission
    max_trades_per_day: int  # Maximum trades per day
    min_position_size: float  # Minimum position size (to amortize commission)
    
    @property
    def expected_annual_trades(self) -> float:
        """Calculate expected annual trades."""
        return 252 / self.target_hold_days
    
    @property
    def expected_annual_slippage_pct(self) -> float:
        """Calculate expected annual slippage percentage."""
        trades_per_year = self.expected_annual_trades
        slippage_per_trade = self.base_slippage_bps / 10000
        return trades_per_year * slippage_per_trade * 2  # Buy + sell


# Predefined timeframe configurations
TIMEFRAME_CONFIGS = {
    Timeframe.INTRADAY: TimeframeConfig(
        name="Intraday",
        data_interval="5min",
        target_hold_days=0.25,  # 6 hours average
        base_slippage_bps=75,  # 0.75% per trade
        commission_per_trade=6.95,
        max_trades_per_day=10,
        min_position_size=1000  # $1000 minimum to keep commission < 1.4%
    ),
    Timeframe.SWING: TimeframeConfig(
        name="Swing",
        data_interval="1d",
        target_hold_days=5,  # 5 days average
        base_slippage_bps=35,  # 0.35% per trade
        commission_per_trade=6.95,
        max_trades_per_day=3,
        min_position_size=500  # $500 minimum
    ),
    Timeframe.WEEKLY: TimeframeConfig(
        name="Weekly",
        data_interval="1d",
        target_hold_days=10,  # 10 days average
        base_slippage_bps=25,  # 0.25% per trade
        commission_per_trade=6.95,
        max_trades_per_day=2,
        min_position_size=1000  # $1000 minimum (to keep commission < 1.4%)
    ),
    Timeframe.MONTHLY: TimeframeConfig(
        name="Monthly",
        data_interval="1d",
        target_hold_days=30,  # 30 days average
        base_slippage_bps=20,  # 0.20% per trade
        commission_per_trade=6.95,
        max_trades_per_day=1,
        min_position_size=1000  # $1000 minimum (to keep commission < 1.4%)
    )
}


class MultiTimeframeStrategy:
    """
    Base class for multi-timeframe trading strategies.
    
    Handles:
    - Timeframe-specific slippage models
    - Position sizing constraints
    - Turnover penalties
    - Feature calculation
    """
    
    def __init__(self, timeframe: Timeframe, capital: float = 100000):
        """
        Initialize multi-timeframe strategy.
        
        Args:
            timeframe: Trading timeframe
            capital: Initial capital
        """
        self.timeframe = timeframe
        self.config = TIMEFRAME_CONFIGS[timeframe]
        self.capital = capital
        
        # Initialize risk management components
        self.slippage_model = MicrocapSlippageModel(
            base_slippage_bps=self.config.base_slippage_bps,
            volume_impact_factor=0.5,
            commission_per_trade=self.config.commission_per_trade
        )
        
        self.dilution_filter = DilutionFilter(
            volume_spike_threshold=5.0,
            price_drop_threshold=-0.10
        )
        
        self.liquidity_constraint = LiquidityConstraint(
            max_pct_adv=0.05,
            min_adv_dollars=50000
        )
        
        # Initialize feature calculator (for intraday)
        if timeframe == Timeframe.INTRADAY:
            self.intraday_features = IntradayFeatures()
        else:
            self.intraday_features = None
    
    def calculate_position_size(self, ticker: str, signal_strength: float,
                               prices: pd.DataFrame, volume: pd.DataFrame) -> float:
        """
        Calculate position size based on timeframe constraints.
        
        Args:
            ticker: Stock ticker
            signal_strength: Signal strength (0-1)
            prices: Price data
            volume: Volume data
        
        Returns:
            Position size in dollars (0 if trade should be skipped)
        """
        # Base position size (equal weight)
        base_size = self.capital * 0.33  # 3 positions max
        
        # Scale by signal strength
        position_size = base_size * signal_strength
        
        # Skip weak signals with small positions (commission drag too high)
        if signal_strength < 0.5 and position_size < self.config.min_position_size:
            return 0
        
        # Enforce minimum position size (to amortize commission)
        position_size = max(position_size, self.config.min_position_size)
        
        # Enforce liquidity constraints
        max_size = self.liquidity_constraint.calculate_max_position_size(
            ticker, prices, volume
        )
        position_size = min(position_size, max_size)
        
        return position_size
    
    def calculate_fitness(self, returns: float, turnover: float, 
                         max_drawdown: float, num_trades: int) -> float:
        """
        Calculate fitness score with timeframe-appropriate penalties.
        
        Fitness = (Net Returns / Max Drawdown) - Turnover Penalty
        
        Args:
            returns: Gross returns
            turnover: Portfolio turnover
            max_drawdown: Maximum drawdown
            num_trades: Number of trades
        
        Returns:
            Fitness score
        """
        # Calculate slippage cost
        avg_slippage = self.config.base_slippage_bps / 10000
        net_returns = self.slippage_model.apply_slippage_to_returns(
            returns, turnover, avg_slippage, num_trades, self.capital
        )
        
        # Calmar ratio (return / max drawdown)
        calmar = net_returns / max_drawdown if max_drawdown > 0 else 0
        
        # Turnover penalty (penalize excessive trading)
        expected_turnover = 1.0 / self.config.target_hold_days * 252  # Annual turnover
        excess_turnover = max(0, turnover - expected_turnover)
        turnover_penalty = excess_turnover * 0.1
        
        # Trade frequency penalty (penalize too many trades)
        expected_trades_per_day = 252 / self.config.target_hold_days / 252
        actual_trades_per_day = num_trades / 252
        excess_trades = max(0, actual_trades_per_day - expected_trades_per_day)
        trade_penalty = excess_trades * 0.05
        
        fitness = calmar - turnover_penalty - trade_penalty
        
        return fitness
    
    def generate_signals(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals.
        
        Override this method in subclasses to implement specific strategies.
        
        Args:
            data: Price/volume data
            features: Calculated features
        
        Returns:
            Series with signal strength (0-1) for each ticker
        """
        raise NotImplementedError("Subclasses must implement generate_signals()")
    
    def backtest(self, prices: pd.DataFrame, volume: pd.DataFrame,
                 features: Optional[pd.DataFrame] = None) -> Dict:
        """
        Backtest strategy with timeframe-appropriate slippage.
        
        Args:
            prices: Price data
            volume: Volume data
            features: Pre-calculated features (optional)
        
        Returns:
            Dict with backtest results
        """
        # Initialize tracking
        capital = self.capital
        positions = {}
        trades = []
        equity_curve = []
        
        # Calculate features if not provided
        if features is None:
            if self.timeframe == Timeframe.INTRADAY and self.intraday_features:
                features = self.intraday_features.calculate_all_features(
                    pd.DataFrame({
                        'open': prices.iloc[:, 0],  # Use first ticker as proxy
                        'high': prices.max(axis=1),
                        'low': prices.min(axis=1),
                        'close': prices.iloc[:, 0],
                        'volume': volume.iloc[:, 0]
                    })
                )
            else:
                # Use simple momentum features for daily timeframes
                features = pd.DataFrame(index=prices.index)
                for col in prices.columns:
                    features[f'{col}_momentum_5'] = prices[col].pct_change(5)
                    features[f'{col}_momentum_20'] = prices[col].pct_change(20)
        
        # Simulate trading
        for i in range(20, len(prices)):  # Start after warmup period
            current_date = prices.index[i]
            current_prices = prices.iloc[:i+1]
            current_volume = volume.iloc[:i+1]
            current_features = features.iloc[:i+1]
            
            # Generate signals
            signals = self.generate_signals(current_prices, current_features)
            
            # Check for exits (dilution, stops, etc.)
            exits = []
            for ticker in list(positions.keys()):
                if self.dilution_filter.should_exit(ticker, current_prices, current_volume, list(positions.keys())):
                    exits.append(ticker)
            
            # Execute exits
            for ticker in exits:
                if ticker in positions:
                    # Calculate exit slippage
                    position_size = positions[ticker]['size'] * current_prices[ticker].iloc[-1]
                    slippage_result = self.slippage_model.calculate_slippage(
                        ticker, position_size, current_prices, current_volume
                    )
                    
                    # Calculate P&L
                    entry_price = positions[ticker]['entry_price']
                    exit_price = current_prices[ticker].iloc[-1] * (1 - slippage_result.total_slippage)
                    pnl = (exit_price - entry_price) * positions[ticker]['size']
                    
                    capital += pnl
                    
                    trades.append({
                        'date': current_date,
                        'ticker': ticker,
                        'action': 'sell',
                        'price': exit_price,
                        'size': positions[ticker]['size'],
                        'pnl': pnl,
                        'slippage_pct': slippage_result.total_slippage * 100,
                        'reason': 'dilution'
                    })
                    
                    del positions[ticker]
            
            # Execute entries (if signals and room in portfolio)
            if len(positions) < 3:  # Max 3 positions
                for ticker in signals.nlargest(3 - len(positions)).index:
                    if ticker not in positions and signals[ticker] > 0.5:
                        # Calculate position size
                        position_size = self.calculate_position_size(
                            ticker, signals[ticker], current_prices, current_volume
                        )
                        
                        if position_size >= self.config.min_position_size:
                            # Calculate entry slippage
                            slippage_result = self.slippage_model.calculate_slippage(
                                ticker, position_size, current_prices, current_volume
                            )
                            
                            # Enter position
                            entry_price = current_prices[ticker].iloc[-1] * (1 + slippage_result.total_slippage)
                            shares = position_size / entry_price
                            
                            positions[ticker] = {
                                'entry_price': entry_price,
                                'size': shares,
                                'entry_date': current_date
                            }
                            
                            capital -= position_size
                            
                            trades.append({
                                'date': current_date,
                                'ticker': ticker,
                                'action': 'buy',
                                'price': entry_price,
                                'size': shares,
                                'slippage_pct': slippage_result.total_slippage * 100
                            })
            
            # Calculate equity
            position_value = sum(
                positions[ticker]['size'] * current_prices[ticker].iloc[-1]
                for ticker in positions
            )
            total_equity = capital + position_value
            equity_curve.append({
                'date': current_date,
                'equity': total_equity,
                'cash': capital,
                'positions': len(positions)
            })
        
        # Calculate metrics
        equity_df = pd.DataFrame(equity_curve).set_index('date')
        returns = equity_df['equity'].pct_change().dropna()
        
        total_return = (equity_df['equity'].iloc[-1] - self.capital) / self.capital
        max_dd = self._calculate_max_drawdown(equity_df['equity'])
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Calculate turnover
        total_traded = sum(abs(t.get('pnl', t['size'] * t['price'])) for t in trades)
        turnover = total_traded / self.capital
        
        # Calculate fitness
        fitness = self.calculate_fitness(total_return, turnover, max_dd, len(trades))
        
        return {
            'timeframe': self.timeframe.value,
            'total_return': total_return,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'num_trades': len(trades),
            'turnover': turnover,
            'fitness': fitness,
            'equity_curve': equity_df,
            'trades': trades,
            'final_capital': equity_df['equity'].iloc[-1]
        }
    
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown."""
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        return abs(drawdown.min())


class SimpleIntradayStrategy(MultiTimeframeStrategy):
    """
    Simple intraday strategy using VWAP and momentum.
    
    Entry: Price crosses above VWAP with positive momentum
    Exit: Price crosses below VWAP or end of day
    """
    
    def __init__(self, capital: float = 100000):
        super().__init__(Timeframe.INTRADAY, capital)
    
    def generate_signals(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """Generate intraday signals based on VWAP and momentum."""
        signals = pd.Series(0.0, index=data.columns)
        
        if 'vwap' not in features.columns:
            return signals
        
        # Get latest values
        latest_features = features.iloc[-1]
        
        for ticker in data.columns:
            if ticker in data.columns:
                current_price = data[ticker].iloc[-1]
                vwap = latest_features.get('vwap', current_price)
                
                # Signal: Price > VWAP and positive momentum
                if current_price > vwap:
                    momentum_5 = latest_features.get('momentum_5', 0)
                    if momentum_5 > 0:
                        signals[ticker] = min(1.0, momentum_5 * 10)  # Scale momentum
        
        return signals


class SimpleSwingStrategy(MultiTimeframeStrategy):
    """
    Simple swing strategy using momentum.
    
    Entry: Strong 5-day momentum
    Exit: Momentum reverses or stop loss
    """
    
    def __init__(self, capital: float = 100000):
        super().__init__(Timeframe.SWING, capital)
    
    def generate_signals(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """Generate swing signals based on momentum."""
        signals = pd.Series(0.0, index=data.columns)
        
        for ticker in data.columns:
            momentum_col = f'{ticker}_momentum_5'
            if momentum_col in features.columns:
                momentum = features[momentum_col].iloc[-1]
                if momentum > 0.05:  # 5% momentum threshold
                    signals[ticker] = min(1.0, momentum * 5)
        
        return signals


def compare_timeframes(prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """
    Compare performance across all timeframes.
    
    Args:
        prices: Price data
        volume: Volume data
    
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for timeframe in [Timeframe.SWING, Timeframe.WEEKLY, Timeframe.MONTHLY]:
        print(f"\nTesting {timeframe.value} timeframe...")
        
        if timeframe == Timeframe.INTRADAY:
            strategy = SimpleIntradayStrategy()
        else:
            strategy = SimpleSwingStrategy(timeframe)
        
        result = strategy.backtest(prices, volume)
        
        results.append({
            'Timeframe': timeframe.value,
            'Total Return': f"{result['total_return']:.1%}",
            'Max Drawdown': f"{result['max_drawdown']:.1%}",
            'Sharpe Ratio': f"{result['sharpe_ratio']:.2f}",
            'Num Trades': result['num_trades'],
            'Turnover': f"{result['turnover']:.1f}x",
            'Fitness': f"{result['fitness']:.3f}",
            'Expected Annual Slippage': f"{TIMEFRAME_CONFIGS[timeframe].expected_annual_slippage_pct:.1%}"
        })
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    print("Multi-Timeframe Strategy Framework")
    print("=" * 80)
    
    # Print timeframe configurations
    print("\nTimeframe Configurations:")
    print("-" * 80)
    for tf, config in TIMEFRAME_CONFIGS.items():
        print(f"\n{config.name}:")
        print(f"  Target Hold: {config.target_hold_days} days")
        print(f"  Base Slippage: {config.base_slippage_bps} bps")
        print(f"  Expected Annual Trades: {config.expected_annual_trades:.0f}")
        print(f"  Expected Annual Slippage: {config.expected_annual_slippage_pct:.1%}")
        print(f"  Min Position Size: ${config.min_position_size}")
    
    print("\n" + "=" * 80)
    print("✓ Framework ready for use")
