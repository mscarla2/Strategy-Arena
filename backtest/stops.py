#!/usr/bin/env python3
"""
Stop Loss Strategies

Implements various stop loss methods:
- Trailing volatility stops (ATR-based)
- Fixed percentage stops
- Time-based stops
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum


class StopType(Enum):
    """Type of stop loss."""
    TRAILING_ATR = "trailing_atr"
    FIXED_PCT = "fixed_pct"
    TIME_BASED = "time_based"


@dataclass
class StopLevel:
    """Stop loss level for a position."""
    ticker: str
    stop_price: float
    stop_type: StopType
    entry_price: float
    entry_date: str
    current_price: float
    distance_pct: float  # Distance from current price to stop


class TrailingVolatilityStop:
    """
    ATR-based trailing stop loss.
    Adjusts stop distance based on volatility.
    
    Benefits:
    - Adapts to market conditions (wider stops in volatile markets)
    - Protects profits by trailing price
    - Reduces whipsaws compared to fixed stops
    """
    
    def __init__(self, atr_multiplier: float = 2.0, lookback: int = 14):
        """
        Args:
            atr_multiplier: Stop distance = ATR × multiplier (2.0 = 2 ATRs)
            lookback: ATR calculation period (14 days standard)
        """
        self.atr_multiplier = atr_multiplier
        self.lookback = lookback
        self.stops: Dict[str, StopLevel] = {}  # ticker -> stop level
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, 
                     close: pd.Series) -> float:
        """
        Calculate Average True Range.
        
        True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
        ATR = average of True Range over lookback period
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
        
        Returns:
            Current ATR value
        """
        if len(close) < self.lookback + 1:
            return 0
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.lookback).mean().iloc[-1]
        
        return atr if not np.isnan(atr) else 0
    
    def initialize_stop(self, ticker: str, entry_price: float, entry_date: str,
                       atr: float, position_type: str = 'long') -> float:
        """
        Initialize stop for a new position.
        
        Args:
            ticker: Stock ticker
            entry_price: Entry price
            entry_date: Entry date
            atr: Current ATR value
            position_type: 'long' or 'short'
        
        Returns:
            Initial stop price
        """
        stop_distance = atr * self.atr_multiplier
        
        if position_type == 'long':
            stop_price = entry_price - stop_distance
        else:  # short
            stop_price = entry_price + stop_distance
        
        self.stops[ticker] = StopLevel(
            ticker=ticker,
            stop_price=stop_price,
            stop_type=StopType.TRAILING_ATR,
            entry_price=entry_price,
            entry_date=entry_date,
            current_price=entry_price,
            distance_pct=stop_distance / entry_price
        )
        
        return stop_price
    
    def update_stop(self, ticker: str, current_price: float, 
                   atr: float, position_type: str = 'long') -> Optional[float]:
        """
        Update trailing stop for a position.
        
        Args:
            ticker: Stock ticker
            current_price: Current market price
            atr: Current ATR value
            position_type: 'long' or 'short'
        
        Returns:
            Updated stop price (None if ticker not found)
        """
        if ticker not in self.stops:
            return None
        
        stop_distance = atr * self.atr_multiplier
        
        if position_type == 'long':
            new_stop = current_price - stop_distance
            # Only move stop up, never down
            if new_stop > self.stops[ticker].stop_price:
                self.stops[ticker].stop_price = new_stop
                self.stops[ticker].current_price = current_price
                self.stops[ticker].distance_pct = stop_distance / current_price
        else:  # short
            new_stop = current_price + stop_distance
            # Only move stop down, never up
            if new_stop < self.stops[ticker].stop_price:
                self.stops[ticker].stop_price = new_stop
                self.stops[ticker].current_price = current_price
                self.stops[ticker].distance_pct = stop_distance / current_price
        
        return self.stops[ticker].stop_price
    
    def check_stop(self, ticker: str, current_price: float, 
                  position_type: str = 'long') -> bool:
        """
        Check if stop has been hit.
        
        Args:
            ticker: Stock ticker
            current_price: Current market price
            position_type: 'long' or 'short'
        
        Returns:
            True if stop hit, False otherwise
        """
        if ticker not in self.stops:
            return False
        
        if position_type == 'long':
            return current_price <= self.stops[ticker].stop_price
        else:  # short
            return current_price >= self.stops[ticker].stop_price
    
    def get_stop_level(self, ticker: str) -> Optional[StopLevel]:
        """Get stop level for a ticker."""
        return self.stops.get(ticker)
    
    def remove_stop(self, ticker: str):
        """Remove stop for closed position."""
        if ticker in self.stops:
            del self.stops[ticker]
    
    def get_all_stops(self) -> Dict[str, StopLevel]:
        """Get all active stops."""
        return self.stops.copy()
    
    def calculate_profit_protection(self, ticker: str) -> Optional[float]:
        """
        Calculate how much profit is protected by the stop.
        
        Returns:
            Profit protection as percentage of entry price (None if no stop)
        """
        if ticker not in self.stops:
            return None
        
        stop = self.stops[ticker]
        profit_protected = (stop.stop_price - stop.entry_price) / stop.entry_price
        return profit_protected


class FixedPercentageStop:
    """
    Simple fixed percentage stop loss.
    
    Simpler than ATR stops but doesn't adapt to volatility.
    Good for backtesting comparison.
    """
    
    def __init__(self, stop_pct: float = 0.10):
        """
        Args:
            stop_pct: Stop distance as percentage (0.10 = 10%)
        """
        self.stop_pct = stop_pct
        self.stops: Dict[str, StopLevel] = {}
    
    def initialize_stop(self, ticker: str, entry_price: float, entry_date: str,
                       position_type: str = 'long') -> float:
        """Initialize stop for a new position."""
        if position_type == 'long':
            stop_price = entry_price * (1 - self.stop_pct)
        else:  # short
            stop_price = entry_price * (1 + self.stop_pct)
        
        self.stops[ticker] = StopLevel(
            ticker=ticker,
            stop_price=stop_price,
            stop_type=StopType.FIXED_PCT,
            entry_price=entry_price,
            entry_date=entry_date,
            current_price=entry_price,
            distance_pct=self.stop_pct
        )
        
        return stop_price
    
    def check_stop(self, ticker: str, current_price: float,
                  position_type: str = 'long') -> bool:
        """Check if stop has been hit."""
        if ticker not in self.stops:
            return False
        
        if position_type == 'long':
            return current_price <= self.stops[ticker].stop_price
        else:  # short
            return current_price >= self.stops[ticker].stop_price
    
    def get_stop_level(self, ticker: str) -> Optional[StopLevel]:
        """Get stop level for a ticker."""
        return self.stops.get(ticker)
    
    def remove_stop(self, ticker: str):
        """Remove stop for closed position."""
        if ticker in self.stops:
            del self.stops[ticker]


class TimeBasedStop:
    """
    Time-based stop: exit after N days regardless of price.
    
    Useful for:
    - Preventing positions from becoming "dead money"
    - Enforcing turnover targets
    - Regime-aware strategies (exit before regime change)
    """
    
    def __init__(self, max_hold_days: int = 30):
        """
        Args:
            max_hold_days: Maximum days to hold a position
        """
        self.max_hold_days = max_hold_days
        self.positions: Dict[str, Tuple[str, int]] = {}  # ticker -> (entry_date, days_held)
    
    def initialize_position(self, ticker: str, entry_date: str):
        """Initialize position tracking."""
        self.positions[ticker] = (entry_date, 0)
    
    def update_position(self, ticker: str, current_date: str) -> int:
        """
        Update days held for a position.
        
        Returns:
            Days held
        """
        if ticker not in self.positions:
            return 0
        
        entry_date, _ = self.positions[ticker]
        # Simple day counter (assumes daily data)
        days_held = (pd.Timestamp(current_date) - pd.Timestamp(entry_date)).days
        self.positions[ticker] = (entry_date, days_held)
        return days_held
    
    def check_stop(self, ticker: str) -> bool:
        """Check if time stop has been hit."""
        if ticker not in self.positions:
            return False
        
        _, days_held = self.positions[ticker]
        return days_held >= self.max_hold_days
    
    def remove_position(self, ticker: str):
        """Remove position tracking."""
        if ticker in self.positions:
            del self.positions[ticker]


# Example usage
if __name__ == "__main__":
    # Example: Trailing ATR Stop
    atr_stop = TrailingVolatilityStop(atr_multiplier=2.0, lookback=14)
    
    # Simulate price data
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    high = pd.Series(np.random.randn(30).cumsum() + 100, index=dates)
    low = high - np.random.rand(30) * 2
    close = (high + low) / 2
    
    # Calculate ATR
    atr = atr_stop.calculate_atr(high, low, close)
    print(f"ATR: ${atr:.2f}")
    
    # Initialize stop
    entry_price = close.iloc[-1]
    stop_price = atr_stop.initialize_stop('AAPL', entry_price, str(dates[-1]), atr)
    print(f"Entry: ${entry_price:.2f}, Stop: ${stop_price:.2f}")
    
    # Simulate price movement
    new_price = entry_price * 1.05  # 5% gain
    atr_stop.update_stop('AAPL', new_price, atr)
    print(f"New price: ${new_price:.2f}, Updated stop: ${atr_stop.get_stop_level('AAPL').stop_price:.2f}")
    
    # Check if stopped out
    test_price = entry_price * 0.95  # 5% loss
    is_stopped = atr_stop.check_stop('AAPL', test_price)
    print(f"Test price: ${test_price:.2f}, Stopped out: {is_stopped}")
