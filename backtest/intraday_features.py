#!/usr/bin/env python3
"""
Intraday Feature Engineering for 5-Minute Candles

Specialized features for intraday trading:
- VWAP (Volume-Weighted Average Price)
- Volume Profile (support/resistance levels)
- Order Flow Imbalance (buying vs selling pressure)
- Microstructure Signals (volume spikes, divergences)
- Intraday Momentum (short-term trends)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class IntradayFeatures:
    """
    Calculate intraday-specific features for 5-minute candle data.
    
    These features are designed to capture:
    - Institutional activity (VWAP)
    - Support/resistance levels (volume profile)
    - Order flow dynamics (buying/selling pressure)
    - Microstructure patterns (tape reading)
    """
    
    def __init__(self):
        """Initialize intraday feature calculator."""
        pass
    
    def calculate_vwap(self, data: pd.DataFrame, reset_daily: bool = True) -> pd.Series:
        """
        Calculate Volume-Weighted Average Price (VWAP).
        
        VWAP is the average price weighted by volume. Institutional traders
        use VWAP as a benchmark for execution quality.
        
        Trading signals:
        - Price > VWAP: Bullish (buyers in control)
        - Price < VWAP: Bearish (sellers in control)
        - Price crossing VWAP: Potential reversal
        
        Args:
            data: DataFrame with 'close' and 'volume' columns
            reset_daily: Reset VWAP at start of each day
        
        Returns:
            Series with VWAP values
        """
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        if reset_daily:
            # Reset VWAP at start of each day
            date = data.index.date
            vwap = (typical_price * data['volume']).groupby(date).cumsum() / \
                   data['volume'].groupby(date).cumsum()
        else:
            # Cumulative VWAP
            vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        
        return vwap
    
    def calculate_vwap_bands(self, data: pd.DataFrame, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate VWAP with standard deviation bands.
        
        Similar to Bollinger Bands but using VWAP as the center line.
        
        Args:
            data: DataFrame with OHLCV data
            num_std: Number of standard deviations for bands
        
        Returns:
            Tuple of (vwap, upper_band, lower_band)
        """
        vwap = self.calculate_vwap(data)
        
        # Calculate standard deviation of price from VWAP
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        squared_diff = (typical_price - vwap) ** 2
        
        # Volume-weighted variance
        date = data.index.date
        variance = (squared_diff * data['volume']).groupby(date).cumsum() / \
                  data['volume'].groupby(date).cumsum()
        std = np.sqrt(variance)
        
        upper_band = vwap + (num_std * std)
        lower_band = vwap - (num_std * std)
        
        return vwap, upper_band, lower_band
    
    def calculate_volume_profile(self, data: pd.DataFrame, num_bins: int = 20) -> pd.DataFrame:
        """
        Calculate volume profile (volume at each price level).
        
        Volume profile shows where the most trading activity occurred,
        identifying key support/resistance levels.
        
        Args:
            data: DataFrame with OHLCV data
            num_bins: Number of price bins
        
        Returns:
            DataFrame with price levels and volume
        """
        # Bin prices into levels
        price_min = data['low'].min()
        price_max = data['high'].max()
        bins = np.linspace(price_min, price_max, num_bins + 1)
        
        # Assign each bar to a price bin
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        price_bins = pd.cut(typical_price, bins=bins)
        
        # Sum volume at each level
        volume_profile = data.groupby(price_bins)['volume'].sum()
        
        # Convert to DataFrame
        profile_df = pd.DataFrame({
            'price_level': [(interval.left + interval.right) / 2 for interval in volume_profile.index],
            'volume': volume_profile.values
        })
        
        return profile_df
    
    def calculate_poc(self, data: pd.DataFrame, num_bins: int = 20) -> float:
        """
        Calculate Point of Control (POC) - price level with highest volume.
        
        POC acts as a magnet for price and is a key support/resistance level.
        
        Args:
            data: DataFrame with OHLCV data
            num_bins: Number of price bins
        
        Returns:
            POC price level
        """
        profile = self.calculate_volume_profile(data, num_bins)
        poc = profile.loc[profile['volume'].idxmax(), 'price_level']
        return poc
    
    def calculate_order_flow_imbalance(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate order flow imbalance (buying vs selling pressure).
        
        Heuristic: If price closes in upper 50% of bar range, assume buying pressure.
        
        Values:
        - +1.0: All buying (close = high)
        - 0.0: Neutral (close = mid)
        - -1.0: All selling (close = low)
        
        Args:
            data: DataFrame with OHLCV data
        
        Returns:
            Series with imbalance values (-1 to +1)
        """
        # Calculate where close is within the bar range
        bar_range = data['high'] - data['low']
        
        # Avoid division by zero
        bar_range = bar_range.replace(0, np.nan)
        
        # Position of close within range (0 = low, 1 = high)
        range_position = (data['close'] - data['low']) / bar_range
        
        # Convert to imbalance (-1 to +1)
        imbalance = (range_position - 0.5) * 2
        
        # Fill NaN with 0 (neutral)
        imbalance = imbalance.fillna(0)
        
        return imbalance
    
    def calculate_volume_weighted_imbalance(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate volume-weighted order flow imbalance.
        
        Weights the imbalance by volume to emphasize high-volume bars.
        
        Args:
            data: DataFrame with OHLCV data
        
        Returns:
            Series with volume-weighted imbalance
        """
        imbalance = self.calculate_order_flow_imbalance(data)
        
        # Weight by volume
        weighted_imbalance = imbalance * data['volume']
        
        # Normalize by rolling volume
        rolling_volume = data['volume'].rolling(20).sum()
        normalized_imbalance = weighted_imbalance.rolling(20).sum() / rolling_volume
        
        return normalized_imbalance
    
    def detect_volume_spike(self, data: pd.DataFrame, threshold: float = 3.0, lookback: int = 20) -> pd.Series:
        """
        Detect volume spikes (volume > threshold * average).
        
        Volume spikes often precede significant price moves.
        
        Args:
            data: DataFrame with volume column
            threshold: Spike threshold (3.0 = 3x average)
            lookback: Lookback period for average
        
        Returns:
            Boolean series indicating volume spikes
        """
        avg_volume = data['volume'].rolling(lookback).mean()
        volume_ratio = data['volume'] / avg_volume
        
        spike = volume_ratio > threshold
        
        return spike
    
    def detect_price_volume_divergence(self, data: pd.DataFrame, lookback: int = 5) -> pd.Series:
        """
        Detect price-volume divergence.
        
        Divergence occurs when:
        - Price rises but volume falls (bearish)
        - Price falls but volume rises (bullish)
        
        Args:
            data: DataFrame with OHLCV data
            lookback: Lookback period for trend
        
        Returns:
            Series with divergence signals (-1 = bearish, 0 = none, +1 = bullish)
        """
        # Calculate price and volume trends
        price_change = data['close'].pct_change(lookback)
        volume_change = data['volume'].pct_change(lookback)
        
        # Detect divergence
        bullish_divergence = (price_change < -0.02) & (volume_change > 0.5)  # Price down, volume up
        bearish_divergence = (price_change > 0.02) & (volume_change < -0.3)  # Price up, volume down
        
        divergence = pd.Series(0, index=data.index)
        divergence[bullish_divergence] = 1
        divergence[bearish_divergence] = -1
        
        return divergence
    
    def calculate_intraday_momentum(self, data: pd.DataFrame, periods: list = [5, 10, 20]) -> pd.DataFrame:
        """
        Calculate intraday momentum over multiple periods.
        
        Args:
            data: DataFrame with close prices
            periods: List of lookback periods (in bars)
        
        Returns:
            DataFrame with momentum columns
        """
        momentum_df = pd.DataFrame(index=data.index)
        
        for period in periods:
            momentum_df[f'momentum_{period}'] = data['close'].pct_change(period)
        
        return momentum_df
    
    def calculate_intraday_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI) for intraday data.
        
        RSI measures momentum and identifies overbought/oversold conditions.
        
        Values:
        - > 70: Overbought (potential reversal down)
        - < 30: Oversold (potential reversal up)
        
        Args:
            data: DataFrame with close prices
            period: RSI period (14 bars = ~70 minutes for 5-min candles)
        
        Returns:
            Series with RSI values (0-100)
        """
        # Calculate price changes
        delta = data['close'].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(period).mean()
        avg_losses = losses.rolling(period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_intraday_macd(self, data: pd.DataFrame, 
                                fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence) for intraday data.
        
        MACD shows trend direction and momentum.
        
        Args:
            data: DataFrame with close prices
            fast: Fast EMA period (12 bars = ~60 minutes)
            slow: Slow EMA period (26 bars = ~130 minutes)
            signal: Signal line period (9 bars = ~45 minutes)
        
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        # Calculate EMAs
        ema_fast = data['close'].ewm(span=fast).mean()
        ema_slow = data['close'].ewm(span=slow).mean()
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line
        signal_line = macd_line.ewm(span=signal).mean()
        
        # Histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all intraday features at once.
        
        Args:
            data: DataFrame with OHLCV data
        
        Returns:
            DataFrame with all features
        """
        features = pd.DataFrame(index=data.index)
        
        # VWAP features
        vwap, vwap_upper, vwap_lower = self.calculate_vwap_bands(data)
        features['vwap'] = vwap
        features['vwap_upper'] = vwap_upper
        features['vwap_lower'] = vwap_lower
        features['price_vs_vwap'] = (data['close'] - vwap) / vwap
        
        # Order flow features
        features['order_flow_imbalance'] = self.calculate_order_flow_imbalance(data)
        features['volume_weighted_imbalance'] = self.calculate_volume_weighted_imbalance(data)
        
        # Volume features
        features['volume_spike'] = self.detect_volume_spike(data).astype(int)
        features['price_volume_divergence'] = self.detect_price_volume_divergence(data)
        
        # Momentum features
        momentum_df = self.calculate_intraday_momentum(data, periods=[5, 10, 20])
        features = pd.concat([features, momentum_df], axis=1)
        
        # Technical indicators
        features['rsi'] = self.calculate_intraday_rsi(data, period=14)
        macd, signal, hist = self.calculate_intraday_macd(data)
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = hist
        
        # POC (Point of Control) - calculated per day
        features['poc'] = np.nan
        for date in data.index.date:
            day_data = data[data.index.date == date]
            if len(day_data) > 0:
                poc = self.calculate_poc(day_data)
                features.loc[features.index.date == date, 'poc'] = poc
        
        features['price_vs_poc'] = (data['close'] - features['poc']) / features['poc']
        
        return features


def test_intraday_features():
    """Test intraday feature calculations."""
    print("=" * 80)
    print("INTRADAY FEATURES TEST")
    print("=" * 80)
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01 09:30', periods=100, freq='5min')
    
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(100) * 0.5) + np.random.rand(100) * 2,
        'low': 100 + np.cumsum(np.random.randn(100) * 0.5) - np.random.rand(100) * 2,
        'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Ensure high >= close >= low
    data['high'] = data[['high', 'close']].max(axis=1)
    data['low'] = data[['low', 'close']].min(axis=1)
    
    # Calculate features
    calculator = IntradayFeatures()
    
    print("\n1. Testing VWAP calculation...")
    vwap = calculator.calculate_vwap(data)
    print(f"  ✓ VWAP calculated: {len(vwap)} values")
    print(f"  Sample values: {vwap.head(3).values}")
    
    print("\n2. Testing order flow imbalance...")
    imbalance = calculator.calculate_order_flow_imbalance(data)
    print(f"  ✓ Imbalance calculated: {len(imbalance)} values")
    print(f"  Range: [{imbalance.min():.2f}, {imbalance.max():.2f}]")
    
    print("\n3. Testing volume spike detection...")
    spikes = calculator.detect_volume_spike(data)
    print(f"  ✓ Spikes detected: {spikes.sum()} out of {len(spikes)} bars")
    
    print("\n4. Testing all features...")
    features = calculator.calculate_all_features(data)
    print(f"  ✓ Features calculated: {len(features.columns)} features")
    print(f"  Feature names: {features.columns.tolist()}")
    
    print("\n5. Sample feature values:")
    print(features.tail(5))
    
    print("\n" + "=" * 80)
    print("✓ Test complete!")
    print("=" * 80)


if __name__ == '__main__':
    test_intraday_features()
