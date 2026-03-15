"""
Smart Money Concepts (SMC) Features for GP Evolution

Implements institutional trading patterns and market microstructure concepts:
- Order Blocks: Institutional accumulation/distribution zones
- Fair Value Gaps (FVG): Price inefficiencies that often get filled
- Liquidity Sweeps: Stop-loss hunting patterns
- Break of Structure (BOS): Trend change identification
- Change of Character (CHoCH): Momentum shift detection

These features help identify where "smart money" (institutions) are operating.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class OrderBlockType(Enum):
    """Type of order block."""
    BULLISH = "bullish"
    BEARISH = "bearish"


class StructureType(Enum):
    """Type of market structure."""
    HIGHER_HIGH = "HH"
    HIGHER_LOW = "HL"
    LOWER_HIGH = "LH"
    LOWER_LOW = "LL"


@dataclass
class OrderBlock:
    """Represents an order block zone."""
    start_idx: int
    end_idx: int
    high: float
    low: float
    volume: float
    block_type: OrderBlockType
    strength: float  # 0-1, based on volume and price action


@dataclass
class FairValueGap:
    """Represents a fair value gap."""
    idx: int
    gap_high: float
    gap_low: float
    gap_size: float
    filled: bool = False
    fill_idx: Optional[int] = None


@dataclass
class LiquiditySweep:
    """Represents a liquidity sweep event."""
    idx: int
    sweep_type: str  # 'high' or 'low'
    sweep_level: float
    reversal_confirmed: bool
    reversal_strength: float


class SmartMoneyFeatures:
    """
    Calculate Smart Money Concepts features for trading strategies.
    
    These features identify institutional trading patterns that retail traders
    often miss, providing edge in strategy discovery.
    """
    
    def __init__(
        self,
        order_block_lookback: int = 20,
        fvg_min_size: float = 0.001,  # 0.1% minimum gap
        liquidity_sweep_threshold: float = 0.002,  # 0.2% beyond level
        structure_lookback: int = 50
    ):
        """
        Args:
            order_block_lookback: Periods to look back for order blocks
            fvg_min_size: Minimum gap size as fraction of price
            liquidity_sweep_threshold: How far beyond level to confirm sweep
            structure_lookback: Periods for structure analysis
        """
        self.order_block_lookback = order_block_lookback
        self.fvg_min_size = fvg_min_size
        self.liquidity_sweep_threshold = liquidity_sweep_threshold
        self.structure_lookback = structure_lookback
    
    def calculate_order_blocks(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Identify order blocks - zones where institutions accumulated/distributed.
        
        Order blocks are the last bullish/bearish candle before a strong move.
        They act as support/resistance zones.
        
        Returns:
            Series with order block strength (positive = bullish, negative = bearish)
        """
        result = pd.Series(0.0, index=close.index)
        
        # Calculate price momentum
        returns = close.pct_change()
        
        # Identify strong moves (>2% in one period)
        strong_up = returns > 0.02
        strong_down = returns < -0.02
        
        for i in range(self.order_block_lookback, len(close)):
            # Look for bullish order blocks (before strong up move)
            if strong_up.iloc[i]:
                # Find last down candle before the move
                for j in range(i - 1, max(0, i - self.order_block_lookback), -1):
                    if close.iloc[j] < close.iloc[j - 1]:
                        # This is a potential bullish order block
                        volume_strength = volume.iloc[j] / volume.iloc[j - 20:j].mean()
                        price_range = (high.iloc[j] - low.iloc[j]) / close.iloc[j]
                        strength = min(volume_strength * price_range * 10, 1.0)
                        result.iloc[i] = strength
                        break
            
            # Look for bearish order blocks (before strong down move)
            elif strong_down.iloc[i]:
                # Find last up candle before the move
                for j in range(i - 1, max(0, i - self.order_block_lookback), -1):
                    if close.iloc[j] > close.iloc[j - 1]:
                        # This is a potential bearish order block
                        volume_strength = volume.iloc[j] / volume.iloc[j - 20:j].mean()
                        price_range = (high.iloc[j] - low.iloc[j]) / close.iloc[j]
                        strength = min(volume_strength * price_range * 10, 1.0)
                        result.iloc[i] = -strength
                        break
        
        return result
    
    def calculate_fair_value_gaps(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Identify Fair Value Gaps (FVG) - price inefficiencies.
        
        FVG occurs when there's a gap between consecutive candles:
        - Bullish FVG: Gap between candle[i-2].high and candle[i].low
        - Bearish FVG: Gap between candle[i-2].low and candle[i].high
        
        Returns:
            Tuple of (bullish_fvg_strength, bearish_fvg_strength)
        """
        bullish_fvg = pd.Series(0.0, index=close.index)
        bearish_fvg = pd.Series(0.0, index=close.index)
        
        for i in range(2, len(close)):
            # Bullish FVG: gap up
            if low.iloc[i] > high.iloc[i - 2]:
                gap_size = (low.iloc[i] - high.iloc[i - 2]) / close.iloc[i]
                if gap_size > self.fvg_min_size:
                    bullish_fvg.iloc[i] = min(gap_size * 100, 1.0)
            
            # Bearish FVG: gap down
            elif high.iloc[i] < low.iloc[i - 2]:
                gap_size = (low.iloc[i - 2] - high.iloc[i]) / close.iloc[i]
                if gap_size > self.fvg_min_size:
                    bearish_fvg.iloc[i] = min(gap_size * 100, 1.0)
        
        return bullish_fvg, bearish_fvg
    
    def calculate_liquidity_sweeps(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Detect liquidity sweeps - stop-loss hunting by institutions.
        
        A liquidity sweep occurs when:
        1. Price breaks above recent high or below recent low
        2. Volume spikes (institutions triggering stops)
        3. Price quickly reverses (institutions got their liquidity)
        
        Returns:
            Series with sweep strength (positive = bullish reversal, negative = bearish)
        """
        result = pd.Series(0.0, index=close.index)
        
        # Calculate rolling highs and lows
        rolling_high = high.rolling(window=20).max()
        rolling_low = low.rolling(window=20).min()
        
        # Calculate volume spike
        avg_volume = volume.rolling(window=20).mean()
        volume_spike = volume / avg_volume
        
        for i in range(20, len(close) - 1):
            # Check for high sweep (bearish trap -> bullish reversal)
            if high.iloc[i] > rolling_high.iloc[i - 1]:
                # Did it reverse?
                if close.iloc[i] < close.iloc[i - 1] and close.iloc[i + 1] > close.iloc[i]:
                    # Was there a volume spike?
                    if volume_spike.iloc[i] > 1.5:
                        reversal_strength = (close.iloc[i + 1] - close.iloc[i]) / close.iloc[i]
                        result.iloc[i + 1] = min(reversal_strength * 50, 1.0)
            
            # Check for low sweep (bullish trap -> bearish reversal)
            elif low.iloc[i] < rolling_low.iloc[i - 1]:
                # Did it reverse?
                if close.iloc[i] > close.iloc[i - 1] and close.iloc[i + 1] < close.iloc[i]:
                    # Was there a volume spike?
                    if volume_spike.iloc[i] > 1.5:
                        reversal_strength = (close.iloc[i] - close.iloc[i + 1]) / close.iloc[i]
                        result.iloc[i + 1] = -min(reversal_strength * 50, 1.0)
        
        return result
    
    def calculate_break_of_structure(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.Series:
        """
        Identify Break of Structure (BOS) - trend change signals.
        
        BOS occurs when:
        - In uptrend: Price breaks below previous higher low
        - In downtrend: Price breaks above previous lower high
        
        Returns:
            Series with BOS strength (positive = bullish BOS, negative = bearish)
        """
        result = pd.Series(0.0, index=close.index)
        
        # Identify swing highs and lows
        swing_highs = self._find_swing_points(high, 'high')
        swing_lows = self._find_swing_points(low, 'low')
        
        # Track current trend
        trend = 0  # 1 = uptrend, -1 = downtrend, 0 = neutral
        last_swing_high = None
        last_swing_low = None
        
        for i in range(self.structure_lookback, len(close)):
            # Update swing points
            if swing_highs.iloc[i] > 0:
                if last_swing_high is not None:
                    if high.iloc[i] > last_swing_high:
                        trend = 1  # Higher high = uptrend
                last_swing_high = high.iloc[i]
            
            if swing_lows.iloc[i] > 0:
                if last_swing_low is not None:
                    if low.iloc[i] < last_swing_low:
                        trend = -1  # Lower low = downtrend
                last_swing_low = low.iloc[i]
            
            # Check for BOS
            if trend == 1 and last_swing_low is not None:
                # In uptrend, check if we broke below last higher low
                if low.iloc[i] < last_swing_low:
                    break_size = (last_swing_low - low.iloc[i]) / close.iloc[i]
                    result.iloc[i] = -min(break_size * 50, 1.0)  # Bearish BOS
                    trend = -1
            
            elif trend == -1 and last_swing_high is not None:
                # In downtrend, check if we broke above last lower high
                if high.iloc[i] > last_swing_high:
                    break_size = (high.iloc[i] - last_swing_high) / close.iloc[i]
                    result.iloc[i] = min(break_size * 50, 1.0)  # Bullish BOS
                    trend = 1
        
        return result
    
    def calculate_change_of_character(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Identify Change of Character (CHoCH) - momentum shift detection.
        
        CHoCH is a weaker signal than BOS, indicating potential trend change.
        It occurs when price action shows signs of exhaustion.
        
        Returns:
            Series with CHoCH strength (positive = bullish, negative = bearish)
        """
        result = pd.Series(0.0, index=close.index)
        
        # Calculate momentum indicators
        returns = close.pct_change()
        momentum = returns.rolling(window=10).mean()
        momentum_change = momentum.diff()
        
        # Calculate volume trend
        volume_ma = volume.rolling(window=20).mean()
        volume_trend = (volume - volume_ma) / volume_ma
        
        # Identify CHoCH
        for i in range(20, len(close)):
            # Bullish CHoCH: momentum turning up with volume
            if momentum_change.iloc[i] > 0 and momentum.iloc[i - 1] < 0:
                if volume_trend.iloc[i] > 0:
                    strength = min(abs(momentum_change.iloc[i]) * 50, 1.0)
                    result.iloc[i] = strength
            
            # Bearish CHoCH: momentum turning down with volume
            elif momentum_change.iloc[i] < 0 and momentum.iloc[i - 1] > 0:
                if volume_trend.iloc[i] > 0:
                    strength = min(abs(momentum_change.iloc[i]) * 50, 1.0)
                    result.iloc[i] = -strength
        
        return result
    
    def _find_swing_points(
        self,
        series: pd.Series,
        point_type: str,
        window: int = 5
    ) -> pd.Series:
        """
        Find swing highs or swing lows.
        
        A swing high is a local maximum (higher than N bars on each side).
        A swing low is a local minimum (lower than N bars on each side).
        
        Args:
            series: Price series (high or low)
            point_type: 'high' or 'low'
            window: Number of bars on each side to compare
        
        Returns:
            Series with 1 at swing points, 0 elsewhere
        """
        result = pd.Series(0, index=series.index)
        
        for i in range(window, len(series) - window):
            if point_type == 'high':
                # Check if this is a local maximum
                is_swing = all(series.iloc[i] > series.iloc[i - j] for j in range(1, window + 1))
                is_swing = is_swing and all(series.iloc[i] > series.iloc[i + j] for j in range(1, window + 1))
            else:  # 'low'
                # Check if this is a local minimum
                is_swing = all(series.iloc[i] < series.iloc[i - j] for j in range(1, window + 1))
                is_swing = is_swing and all(series.iloc[i] < series.iloc[i + j] for j in range(1, window + 1))
            
            if is_swing:
                result.iloc[i] = 1
        
        return result
    
    def calculate_all_features(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> Dict[str, pd.Series]:
        """
        Calculate all Smart Money Concepts features.
        
        Returns:
            Dictionary of feature name -> feature series
        """
        features = {}
        
        # Order blocks
        features['smc_order_blocks'] = self.calculate_order_blocks(high, low, close, volume)
        
        # Fair value gaps
        bullish_fvg, bearish_fvg = self.calculate_fair_value_gaps(high, low, close)
        features['smc_fvg_bullish'] = bullish_fvg
        features['smc_fvg_bearish'] = bearish_fvg
        features['smc_fvg_net'] = bullish_fvg - bearish_fvg
        
        # Liquidity sweeps
        features['smc_liquidity_sweeps'] = self.calculate_liquidity_sweeps(high, low, close, volume)
        
        # Break of structure
        features['smc_break_of_structure'] = self.calculate_break_of_structure(high, low, close)
        
        # Change of character
        features['smc_change_of_character'] = self.calculate_change_of_character(high, low, close, volume)
        
        return features


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example: Calculate SMC features for a stock
    import yfinance as yf
    
    print("Smart Money Concepts Features - Example")
    print("=" * 60)
    
    # Fetch sample data
    ticker = "AAPL"
    data = yf.download(ticker, start="2023-01-01", end="2024-01-01", progress=False)
    
    # Initialize calculator
    smc = SmartMoneyFeatures(
        order_block_lookback=20,
        fvg_min_size=0.001,
        liquidity_sweep_threshold=0.002,
        structure_lookback=50
    )
    
    # Calculate features
    features = smc.calculate_all_features(
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        volume=data['Volume']
    )
    
    # Display results
    print(f"\nCalculated {len(features)} SMC features for {ticker}")
    print("\nFeature Summary:")
    for name, series in features.items():
        non_zero = (series != 0).sum()
        print(f"  {name:30s}: {non_zero:4d} signals ({non_zero/len(series)*100:.1f}%)")
    
    # Show recent signals
    print("\nRecent Signals (last 10 days):")
    recent = pd.DataFrame(features).tail(10)
    print(recent.to_string())
    
    print("\n" + "=" * 60)
    print("Features ready for integration into GP evolution!")
