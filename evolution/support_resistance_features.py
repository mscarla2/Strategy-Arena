"""
Support/Resistance Features for GP Evolution

Implements technical analysis concepts for identifying key price levels:
- Volume Profile: Price levels with high trading activity
- Pivot Points: Traditional, Fibonacci, and Camarilla pivots
- Dynamic Support/Resistance: Moving averages, Bollinger Bands, Keltner Channels
- Price Action Levels: Historical highs/lows, consolidation zones

These features help identify where price is likely to find support or resistance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class PivotType(Enum):
    """Type of pivot point calculation."""
    TRADITIONAL = "traditional"
    FIBONACCI = "fibonacci"
    CAMARILLA = "camarilla"
    WOODIE = "woodie"


@dataclass
class VolumeProfileLevel:
    """Represents a volume profile level."""
    price: float
    volume: float
    is_poc: bool = False  # Point of Control (highest volume)
    in_value_area: bool = False  # Within 70% of volume


@dataclass
class PivotLevels:
    """Pivot point levels."""
    pivot: float
    r1: float
    r2: float
    r3: float
    s1: float
    s2: float
    s3: float


class SupportResistanceFeatures:
    """
    Calculate Support/Resistance features for trading strategies.
    
    These features identify key price levels where institutional and retail
    traders are likely to place orders, providing edge in strategy discovery.
    """
    
    def __init__(
        self,
        volume_profile_bins: int = 50,
        pivot_lookback: int = 1,  # Days to look back for pivot calculation
        ma_periods: List[int] = None,
        bb_period: int = 20,
        bb_std: float = 2.0,
        keltner_period: int = 20,
        keltner_atr_mult: float = 2.0
    ):
        """
        Args:
            volume_profile_bins: Number of price bins for volume profile
            pivot_lookback: Days to look back for pivot calculation
            ma_periods: Moving average periods for dynamic S/R
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviation multiplier
            keltner_period: Keltner Channel period
            keltner_atr_mult: Keltner Channel ATR multiplier
        """
        self.volume_profile_bins = volume_profile_bins
        self.pivot_lookback = pivot_lookback
        self.ma_periods = ma_periods or [20, 50, 100, 200]
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.keltner_period = keltner_period
        self.keltner_atr_mult = keltner_atr_mult
    
    def calculate_volume_profile(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        lookback: int = 20
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate volume profile and identify Point of Control (POC).
        
        Volume profile shows where most trading activity occurred.
        POC is the price level with highest volume.
        
        Returns:
            Tuple of (poc_distance, value_area_high_distance, value_area_low_distance)
            All distances are normalized by current price
        """
        poc_distance = pd.Series(0.0, index=close.index)
        va_high_distance = pd.Series(0.0, index=close.index)
        va_low_distance = pd.Series(0.0, index=close.index)
        
        for i in range(lookback, len(close)):
            # Get data for lookback period
            period_high = high.iloc[i - lookback:i]
            period_low = low.iloc[i - lookback:i]
            period_close = close.iloc[i - lookback:i]
            period_volume = volume.iloc[i - lookback:i]
            
            # Create price bins
            price_min = period_low.min()
            price_max = period_high.max()
            bins = np.linspace(price_min, price_max, self.volume_profile_bins)
            
            # Allocate volume to bins
            volume_by_price = np.zeros(len(bins) - 1)
            for j in range(len(period_close)):
                # Find which bin this price falls into
                price = period_close.iloc[j]
                bin_idx = np.searchsorted(bins, price) - 1
                bin_idx = max(0, min(bin_idx, len(volume_by_price) - 1))
                volume_by_price[bin_idx] += period_volume.iloc[j]
            
            # Find POC (highest volume bin)
            poc_idx = np.argmax(volume_by_price)
            poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2
            
            # Find Value Area (70% of volume)
            total_volume = volume_by_price.sum()
            target_volume = total_volume * 0.70
            
            # Start from POC and expand outward
            va_indices = [poc_idx]
            current_volume = volume_by_price[poc_idx]
            
            while current_volume < target_volume and len(va_indices) < len(volume_by_price):
                # Check adjacent bins
                left_idx = min(va_indices) - 1
                right_idx = max(va_indices) + 1
                
                left_vol = volume_by_price[left_idx] if left_idx >= 0 else 0
                right_vol = volume_by_price[right_idx] if right_idx < len(volume_by_price) else 0
                
                if left_vol > right_vol and left_idx >= 0:
                    va_indices.append(left_idx)
                    current_volume += left_vol
                elif right_idx < len(volume_by_price):
                    va_indices.append(right_idx)
                    current_volume += right_vol
                else:
                    break
            
            # Calculate Value Area boundaries
            va_low_idx = min(va_indices)
            va_high_idx = max(va_indices)
            va_low_price = bins[va_low_idx]
            va_high_price = bins[va_high_idx + 1]
            
            # Store distances normalized by current price
            current_price = close.iloc[i]
            poc_distance.iloc[i] = (poc_price - current_price) / current_price
            va_high_distance.iloc[i] = (va_high_price - current_price) / current_price
            va_low_distance.iloc[i] = (va_low_price - current_price) / current_price
        
        return poc_distance, va_high_distance, va_low_distance
    
    def calculate_pivot_points(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        pivot_type: PivotType = PivotType.TRADITIONAL
    ) -> Dict[str, pd.Series]:
        """
        Calculate pivot points for support/resistance levels.
        
        Pivot points are calculated from previous period's high, low, close.
        Different methods provide different levels.
        
        Returns:
            Dictionary with pivot, r1, r2, r3, s1, s2, s3 series
        """
        result = {
            'pivot': pd.Series(0.0, index=close.index),
            'r1': pd.Series(0.0, index=close.index),
            'r2': pd.Series(0.0, index=close.index),
            'r3': pd.Series(0.0, index=close.index),
            's1': pd.Series(0.0, index=close.index),
            's2': pd.Series(0.0, index=close.index),
            's3': pd.Series(0.0, index=close.index),
        }
        
        for i in range(self.pivot_lookback, len(close)):
            h = high.iloc[i - self.pivot_lookback]
            l = low.iloc[i - self.pivot_lookback]
            c = close.iloc[i - self.pivot_lookback]
            
            if pivot_type == PivotType.TRADITIONAL:
                pivot = (h + l + c) / 3
                r1 = 2 * pivot - l
                r2 = pivot + (h - l)
                r3 = h + 2 * (pivot - l)
                s1 = 2 * pivot - h
                s2 = pivot - (h - l)
                s3 = l - 2 * (h - pivot)
            
            elif pivot_type == PivotType.FIBONACCI:
                pivot = (h + l + c) / 3
                r1 = pivot + 0.382 * (h - l)
                r2 = pivot + 0.618 * (h - l)
                r3 = pivot + 1.000 * (h - l)
                s1 = pivot - 0.382 * (h - l)
                s2 = pivot - 0.618 * (h - l)
                s3 = pivot - 1.000 * (h - l)
            
            elif pivot_type == PivotType.CAMARILLA:
                pivot = (h + l + c) / 3
                range_hl = h - l
                r1 = c + range_hl * 1.1 / 12
                r2 = c + range_hl * 1.1 / 6
                r3 = c + range_hl * 1.1 / 4
                s1 = c - range_hl * 1.1 / 12
                s2 = c - range_hl * 1.1 / 6
                s3 = c - range_hl * 1.1 / 4
            
            elif pivot_type == PivotType.WOODIE:
                pivot = (h + l + 2 * c) / 4
                r1 = 2 * pivot - l
                r2 = pivot + (h - l)
                r3 = h + 2 * (pivot - l)
                s1 = 2 * pivot - h
                s2 = pivot - (h - l)
                s3 = l - 2 * (h - pivot)
            
            # Store as distance from current price (normalized)
            current_price = close.iloc[i]
            result['pivot'].iloc[i] = (pivot - current_price) / current_price
            result['r1'].iloc[i] = (r1 - current_price) / current_price
            result['r2'].iloc[i] = (r2 - current_price) / current_price
            result['r3'].iloc[i] = (r3 - current_price) / current_price
            result['s1'].iloc[i] = (s1 - current_price) / current_price
            result['s2'].iloc[i] = (s2 - current_price) / current_price
            result['s3'].iloc[i] = (s3 - current_price) / current_price
        
        return result
    
    def calculate_dynamic_support_resistance(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> Dict[str, pd.Series]:
        """
        Calculate dynamic support/resistance from moving averages and bands.
        
        Returns:
            Dictionary with MA distances, BB distances, Keltner distances
        """
        result = {}
        
        # Moving averages as dynamic S/R
        for period in self.ma_periods:
            ma = close.rolling(window=period).mean()
            distance = (ma - close) / close
            result[f'ma_{period}_distance'] = distance
            
            # Is price above or below MA?
            result[f'above_ma_{period}'] = (close > ma).astype(float)
        
        # Bollinger Bands
        bb_ma = close.rolling(window=self.bb_period).mean()
        bb_std = close.rolling(window=self.bb_period).std()
        bb_upper = bb_ma + self.bb_std * bb_std
        bb_lower = bb_ma - self.bb_std * bb_std
        
        result['bb_upper_distance'] = (bb_upper - close) / close
        result['bb_lower_distance'] = (bb_lower - close) / close
        result['bb_width'] = (bb_upper - bb_lower) / bb_ma
        result['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
        
        # Keltner Channels
        kc_ma = close.rolling(window=self.keltner_period).mean()
        atr = self._calculate_atr(high, low, close, self.keltner_period)
        kc_upper = kc_ma + self.keltner_atr_mult * atr
        kc_lower = kc_ma - self.keltner_atr_mult * atr
        
        result['kc_upper_distance'] = (kc_upper - close) / close
        result['kc_lower_distance'] = (kc_lower - close) / close
        result['kc_width'] = (kc_upper - kc_lower) / kc_ma
        result['kc_position'] = (close - kc_lower) / (kc_upper - kc_lower)
        
        # Squeeze indicator (BB inside KC)
        result['squeeze'] = ((bb_upper < kc_upper) & (bb_lower > kc_lower)).astype(float)
        
        return result
    
    def calculate_price_action_levels(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        lookback: int = 50
    ) -> Dict[str, pd.Series]:
        """
        Calculate support/resistance from price action.
        
        Identifies:
        - Recent highs/lows
        - Consolidation zones
        - Round number levels
        
        Returns:
            Dictionary with distance to key levels
        """
        result = {}
        
        # Rolling highs and lows
        rolling_high = high.rolling(window=lookback).max()
        rolling_low = low.rolling(window=lookback).min()
        
        result['distance_to_high'] = (rolling_high - close) / close
        result['distance_to_low'] = (rolling_low - close) / close
        result['range_position'] = (close - rolling_low) / (rolling_high - rolling_low)
        
        # Consolidation detection (low volatility)
        returns = close.pct_change()
        volatility = returns.rolling(window=20).std()
        avg_volatility = volatility.rolling(window=100).mean()
        result['consolidation'] = (volatility < avg_volatility * 0.5).astype(float)
        
        # Distance to round numbers (psychological levels)
        # Round to nearest 5, 10, 50, 100
        for round_level in [5, 10, 50, 100]:
            nearest_round = (close / round_level).round() * round_level
            result[f'distance_to_round_{round_level}'] = (nearest_round - close) / close
        
        return result
    
    def _calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int
    ) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def calculate_all_features(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> Dict[str, pd.Series]:
        """
        Calculate all Support/Resistance features.
        
        Returns:
            Dictionary of feature name -> feature series
        """
        features = {}
        
        # Volume profile
        poc_dist, va_high_dist, va_low_dist = self.calculate_volume_profile(
            high, low, close, volume
        )
        features['sr_poc_distance'] = poc_dist
        features['sr_va_high_distance'] = va_high_dist
        features['sr_va_low_distance'] = va_low_dist
        
        # Pivot points (traditional)
        pivots = self.calculate_pivot_points(high, low, close, PivotType.TRADITIONAL)
        for name, series in pivots.items():
            features[f'sr_pivot_{name}'] = series
        
        # Dynamic S/R
        dynamic = self.calculate_dynamic_support_resistance(high, low, close)
        features.update(dynamic)
        
        # Price action levels
        price_action = self.calculate_price_action_levels(high, low, close)
        features.update(price_action)
        
        return features


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example: Calculate S/R features for a stock
    import yfinance as yf
    
    print("Support/Resistance Features - Example")
    print("=" * 60)
    
    # Fetch sample data
    ticker = "AAPL"
    data = yf.download(ticker, start="2023-01-01", end="2024-01-01", progress=False)
    
    # Initialize calculator
    sr = SupportResistanceFeatures(
        volume_profile_bins=50,
        pivot_lookback=1,
        ma_periods=[20, 50, 100, 200],
        bb_period=20,
        bb_std=2.0,
        keltner_period=20,
        keltner_atr_mult=2.0
    )
    
    # Calculate features
    features = sr.calculate_all_features(
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        volume=data['Volume']
    )
    
    # Display results
    print(f"\nCalculated {len(features)} S/R features for {ticker}")
    print("\nFeature Categories:")
    
    categories = {
        'Volume Profile': [k for k in features.keys() if 'poc' in k or 'va_' in k],
        'Pivot Points': [k for k in features.keys() if 'pivot' in k],
        'Moving Averages': [k for k in features.keys() if 'ma_' in k],
        'Bollinger Bands': [k for k in features.keys() if 'bb_' in k],
        'Keltner Channels': [k for k in features.keys() if 'kc_' in k],
        'Price Action': [k for k in features.keys() if 'distance_to' in k or 'range_position' in k],
    }
    
    for category, feature_list in categories.items():
        print(f"\n  {category}: {len(feature_list)} features")
        for feature in feature_list[:3]:  # Show first 3
            print(f"    - {feature}")
    
    # Show recent values
    print("\nRecent Values (last 5 days):")
    recent = pd.DataFrame({
        'POC Distance': features['sr_poc_distance'],
        'Pivot': features['sr_pivot_pivot'],
        'MA 50': features['ma_50_distance'],
        'BB Position': features['bb_position'],
        'Range Position': features['range_position']
    }).tail(5)
    print(recent.to_string())
    
    print("\n" + "=" * 60)
    print("Features ready for integration into GP evolution!")
