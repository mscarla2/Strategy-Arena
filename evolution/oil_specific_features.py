"""
Oil-Specific Features for GP Evolution

Implements sector-specific features for oil & gas stocks:
- Crude Oil Price Correlation: Relationship with WTI/Brent prices
- Inventory Levels: EIA crude oil inventory data
- Refinery Utilization: Capacity utilization rates
- Crack Spreads: Refining margins (gasoline/diesel vs crude)
- Geopolitical Risk: OPEC decisions, sanctions, conflicts
- Seasonal Patterns: Driving season, heating season effects

These features help identify oil-specific alpha that generic technical
indicators might miss.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings


@dataclass
class OilMarketData:
    """Container for oil market data."""
    wti_price: pd.Series  # West Texas Intermediate crude price
    brent_price: pd.Series  # Brent crude price
    inventory: pd.Series  # EIA crude oil inventory (million barrels)
    refinery_utilization: pd.Series  # Refinery capacity utilization (%)
    gasoline_price: pd.Series  # Gasoline wholesale price
    diesel_price: pd.Series  # Diesel wholesale price


class OilSpecificFeatures:
    """
    Calculate oil-specific features for trading oil stocks.
    
    These features capture sector dynamics that affect oil company valuations
    beyond generic technical indicators.
    """
    
    def __init__(
        self,
        correlation_lookback: int = 60,
        inventory_lookback: int = 20,
        seasonal_lookback: int = 252
    ):
        """
        Args:
            correlation_lookback: Period for oil price correlation
            inventory_lookback: Period for inventory analysis
            seasonal_lookback: Period for seasonal pattern detection
        """
        self.correlation_lookback = correlation_lookback
        self.inventory_lookback = inventory_lookback
        self.seasonal_lookback = seasonal_lookback
    
    def calculate_oil_price_correlation(
        self,
        stock_returns: pd.Series,
        oil_returns: pd.Series,
        lookback: Optional[int] = None
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate rolling correlation with oil prices.
        
        High correlation = pure play on oil prices
        Low correlation = other factors dominate
        
        Returns:
            Tuple of (correlation, beta)
        """
        lookback = lookback or self.correlation_lookback
        
        # Rolling correlation
        correlation = stock_returns.rolling(window=lookback).corr(oil_returns)
        
        # Rolling beta (sensitivity to oil price changes)
        cov = stock_returns.rolling(window=lookback).cov(oil_returns)
        var = oil_returns.rolling(window=lookback).var()
        beta = cov / var
        
        return correlation.fillna(0), beta.fillna(0)
    
    def calculate_oil_price_momentum(
        self,
        oil_prices: pd.Series,
        periods: List[int] = None
    ) -> Dict[str, pd.Series]:
        """
        Calculate oil price momentum over multiple periods.
        
        Oil price trends often lead stock price movements.
        
        Returns:
            Dict of momentum features
        """
        periods = periods or [5, 10, 20, 60]
        features = {}
        
        for period in periods:
            momentum = oil_prices.pct_change(periods=period)
            features[f'oil_momentum_{period}d'] = momentum
        
        # Oil price volatility
        features['oil_volatility_20d'] = oil_prices.pct_change().rolling(window=20).std() * np.sqrt(252)
        
        # Oil price trend strength (ADX-like)
        returns = oil_prices.pct_change()
        up_days = (returns > 0).rolling(window=20).sum()
        features['oil_trend_strength'] = (up_days - 10) / 10  # Normalized to -1 to 1
        
        return features
    
    def calculate_inventory_features(
        self,
        inventory: pd.Series,
        lookback: Optional[int] = None
    ) -> Dict[str, pd.Series]:
        """
        Calculate features from EIA crude oil inventory data.
        
        High inventory = bearish for oil prices
        Low inventory = bullish for oil prices
        
        Returns:
            Dict of inventory features
        """
        lookback = lookback or self.inventory_lookback
        features = {}
        
        # Inventory level relative to historical average
        inventory_ma = inventory.rolling(window=lookback * 5).mean()
        features['inventory_deviation'] = (inventory - inventory_ma) / inventory_ma
        
        # Inventory change (week-over-week)
        features['inventory_change'] = inventory.diff()
        features['inventory_change_pct'] = inventory.pct_change()
        
        # Inventory trend (increasing or decreasing)
        inventory_slope = inventory.rolling(window=lookback).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == lookback else 0
        )
        features['inventory_trend'] = inventory_slope
        
        # Inventory surprise (vs expectations)
        # Simplified: compare to rolling average
        expected_change = inventory.diff().rolling(window=4).mean()
        actual_change = inventory.diff()
        features['inventory_surprise'] = actual_change - expected_change
        
        return features
    
    def calculate_crack_spreads(
        self,
        crude_price: pd.Series,
        gasoline_price: pd.Series,
        diesel_price: pd.Series
    ) -> Dict[str, pd.Series]:
        """
        Calculate crack spreads (refining margins).
        
        Crack spread = refined product price - crude oil price
        High spreads = profitable for refiners
        
        Returns:
            Dict of crack spread features
        """
        features = {}
        
        # 3-2-1 crack spread (3 barrels crude -> 2 barrels gasoline + 1 barrel diesel)
        # Simplified calculation
        features['crack_spread_321'] = (2 * gasoline_price + diesel_price) / 3 - crude_price
        
        # Gasoline crack spread
        features['crack_spread_gasoline'] = gasoline_price - crude_price
        
        # Diesel crack spread
        features['crack_spread_diesel'] = diesel_price - crude_price
        
        # Crack spread momentum
        features['crack_spread_momentum'] = features['crack_spread_321'].pct_change(periods=20)
        
        # Crack spread percentile (relative to history)
        for name in ['crack_spread_321', 'crack_spread_gasoline', 'crack_spread_diesel']:
            rolling_rank = features[name].rolling(window=252).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
            )
            features[f'{name}_percentile'] = rolling_rank
        
        return features
    
    def calculate_refinery_utilization(
        self,
        utilization: pd.Series
    ) -> Dict[str, pd.Series]:
        """
        Calculate features from refinery utilization rates.
        
        High utilization = strong demand for refined products
        Low utilization = weak demand or maintenance season
        
        Returns:
            Dict of utilization features
        """
        features = {}
        
        # Utilization level
        features['refinery_utilization'] = utilization
        
        # Utilization change
        features['refinery_utilization_change'] = utilization.diff()
        
        # Utilization relative to historical average
        util_ma = utilization.rolling(window=52).mean()  # 52 weeks = 1 year
        features['refinery_utilization_deviation'] = (utilization - util_ma) / util_ma
        
        # Utilization trend
        util_slope = utilization.rolling(window=12).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 12 else 0
        )
        features['refinery_utilization_trend'] = util_slope
        
        return features
    
    def calculate_seasonal_patterns(
        self,
        prices: pd.Series,
        lookback: Optional[int] = None
    ) -> Dict[str, pd.Series]:
        """
        Calculate seasonal patterns in oil stocks.
        
        - Driving season (May-Sep): High gasoline demand
        - Heating season (Nov-Mar): High heating oil demand
        - Shoulder seasons: Lower demand
        
        Returns:
            Dict of seasonal features
        """
        lookback = lookback or self.seasonal_lookback
        features = {}
        
        # Extract month from index
        if not isinstance(prices.index, pd.DatetimeIndex):
            return features
        
        months = prices.index.month
        
        # Driving season indicator (May-September)
        features['driving_season'] = ((months >= 5) & (months <= 9)).astype(float)
        
        # Heating season indicator (November-March)
        features['heating_season'] = ((months >= 11) | (months <= 3)).astype(float)
        
        # Shoulder season indicator
        features['shoulder_season'] = (~features['driving_season'].astype(bool) & 
                                       ~features['heating_season'].astype(bool)).astype(float)
        
        # Seasonal returns (average return for this month historically)
        seasonal_returns = pd.Series(0.0, index=prices.index)
        for month in range(1, 13):
            month_mask = months == month
            if month_mask.sum() > 0:
                month_returns = prices.pct_change()[month_mask]
                avg_return = month_returns.mean()
                seasonal_returns[month_mask] = avg_return
        
        features['seasonal_expected_return'] = seasonal_returns
        
        return features
    
    def calculate_wti_brent_spread(
        self,
        wti_price: pd.Series,
        brent_price: pd.Series
    ) -> Dict[str, pd.Series]:
        """
        Calculate WTI-Brent spread features.
        
        Spread reflects regional supply/demand dynamics and transportation costs.
        
        Returns:
            Dict of spread features
        """
        features = {}
        
        # Absolute spread
        features['wti_brent_spread'] = wti_price - brent_price
        
        # Relative spread (as % of Brent)
        features['wti_brent_spread_pct'] = (wti_price - brent_price) / brent_price
        
        # Spread momentum
        features['wti_brent_spread_momentum'] = features['wti_brent_spread'].pct_change(periods=20)
        
        # Spread percentile
        spread_percentile = features['wti_brent_spread'].rolling(window=252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )
        features['wti_brent_spread_percentile'] = spread_percentile
        
        return features
    
    def calculate_geopolitical_risk_proxy(
        self,
        oil_volatility: pd.Series,
        oil_returns: pd.Series
    ) -> Dict[str, pd.Series]:
        """
        Calculate proxy for geopolitical risk.
        
        High volatility + large moves = potential geopolitical events
        
        Returns:
            Dict of risk proxy features
        """
        features = {}
        
        # Volatility spike indicator
        vol_ma = oil_volatility.rolling(window=60).mean()
        vol_spike = oil_volatility / vol_ma
        features['geopolitical_risk_proxy'] = vol_spike
        
        # Large move indicator (>3% daily move)
        large_moves = (abs(oil_returns) > 0.03).astype(float)
        features['oil_large_move'] = large_moves
        
        # Consecutive large moves (potential crisis)
        features['oil_large_move_streak'] = large_moves.rolling(window=5).sum()
        
        return features
    
    def calculate_all_features(
        self,
        stock_prices: pd.Series,
        oil_market_data: OilMarketData
    ) -> Dict[str, pd.Series]:
        """
        Calculate all oil-specific features.
        
        Args:
            stock_prices: Stock price series
            oil_market_data: Oil market data container
        
        Returns:
            Dictionary of feature name -> feature series
        """
        features = {}
        
        # Calculate returns
        stock_returns = stock_prices.pct_change()
        oil_returns = oil_market_data.wti_price.pct_change()
        
        # Oil price correlation and beta
        corr, beta = self.calculate_oil_price_correlation(stock_returns, oil_returns)
        features['oil_correlation'] = corr
        features['oil_beta'] = beta
        
        # Oil price momentum
        oil_momentum = self.calculate_oil_price_momentum(oil_market_data.wti_price)
        features.update(oil_momentum)
        
        # Inventory features
        if oil_market_data.inventory is not None:
            inventory_features = self.calculate_inventory_features(oil_market_data.inventory)
            features.update(inventory_features)
        
        # Crack spreads
        if (oil_market_data.gasoline_price is not None and 
            oil_market_data.diesel_price is not None):
            crack_features = self.calculate_crack_spreads(
                oil_market_data.wti_price,
                oil_market_data.gasoline_price,
                oil_market_data.diesel_price
            )
            features.update(crack_features)
        
        # Refinery utilization
        if oil_market_data.refinery_utilization is not None:
            util_features = self.calculate_refinery_utilization(
                oil_market_data.refinery_utilization
            )
            features.update(util_features)
        
        # Seasonal patterns
        seasonal_features = self.calculate_seasonal_patterns(stock_prices)
        features.update(seasonal_features)
        
        # WTI-Brent spread
        if oil_market_data.brent_price is not None:
            spread_features = self.calculate_wti_brent_spread(
                oil_market_data.wti_price,
                oil_market_data.brent_price
            )
            features.update(spread_features)
        
        # Geopolitical risk proxy
        oil_volatility = oil_returns.rolling(window=20).std() * np.sqrt(252)
        risk_features = self.calculate_geopolitical_risk_proxy(oil_volatility, oil_returns)
        features.update(risk_features)
        
        return features


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_oil_market_data(
    start_date: str,
    end_date: str
) -> OilMarketData:
    """
    Fetch oil market data from various sources.
    
    Note: This is a simplified implementation. In production, you would:
    - Use EIA API for inventory and refinery data
    - Use commodity data providers for prices
    - Cache data locally
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        OilMarketData container
    """
    try:
        import yfinance as yf
        
        # Fetch WTI crude oil (CL=F)
        wti = yf.download("CL=F", start=start_date, end=end_date, progress=False)
        wti_price = wti['Close'] if not wti.empty else pd.Series()
        
        # Fetch Brent crude oil (BZ=F)
        brent = yf.download("BZ=F", start=start_date, end=end_date, progress=False)
        brent_price = brent['Close'] if not brent.empty else pd.Series()
        
        # Fetch gasoline (RB=F)
        gasoline = yf.download("RB=F", start=start_date, end=end_date, progress=False)
        gasoline_price = gasoline['Close'] if not gasoline.empty else pd.Series()
        
        # Fetch heating oil/diesel (HO=F)
        diesel = yf.download("HO=F", start=start_date, end=end_date, progress=False)
        diesel_price = diesel['Close'] if not diesel.empty else pd.Series()
        
        # Note: EIA data would require API key and separate fetching
        # For now, return None for inventory and refinery data
        inventory = None
        refinery_utilization = None
        
        return OilMarketData(
            wti_price=wti_price,
            brent_price=brent_price,
            inventory=inventory,
            refinery_utilization=refinery_utilization,
            gasoline_price=gasoline_price,
            diesel_price=diesel_price
        )
    
    except Exception as e:
        warnings.warn(f"Error fetching oil market data: {e}")
        # Return empty data
        return OilMarketData(
            wti_price=pd.Series(),
            brent_price=pd.Series(),
            inventory=None,
            refinery_utilization=None,
            gasoline_price=pd.Series(),
            diesel_price=pd.Series()
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Oil-Specific Features - Example")
    print("=" * 60)
    
    # Fetch oil stock data
    import yfinance as yf
    
    ticker = "XOM"  # ExxonMobil
    stock_data = yf.download(ticker, start="2023-01-01", end="2024-01-01", progress=False)
    
    # Fetch oil market data
    print("\nFetching oil market data...")
    oil_data = fetch_oil_market_data("2023-01-01", "2024-01-01")
    
    # Initialize calculator
    oil_features = OilSpecificFeatures(
        correlation_lookback=60,
        inventory_lookback=20,
        seasonal_lookback=252
    )
    
    # Calculate features
    print(f"\nCalculating oil-specific features for {ticker}...")
    features = oil_features.calculate_all_features(
        stock_prices=stock_data['Close'],
        oil_market_data=oil_data
    )
    
    # Display results
    print(f"\nCalculated {len(features)} oil-specific features")
    print("\nFeature Categories:")
    
    categories = {
        'Oil Correlation': [k for k in features.keys() if 'correlation' in k or 'beta' in k],
        'Oil Momentum': [k for k in features.keys() if 'oil_momentum' in k or 'oil_trend' in k],
        'Crack Spreads': [k for k in features.keys() if 'crack_spread' in k],
        'Seasonal': [k for k in features.keys() if 'season' in k],
        'WTI-Brent': [k for k in features.keys() if 'wti_brent' in k],
        'Geopolitical': [k for k in features.keys() if 'geopolitical' in k or 'large_move' in k],
    }
    
    for category, feature_list in categories.items():
        if feature_list:
            print(f"\n  {category}: {len(feature_list)} features")
            for feature in feature_list[:3]:  # Show first 3
                print(f"    - {feature}")
    
    # Show recent values
    print("\nRecent Values (last 5 days):")
    recent = pd.DataFrame({
        'Oil Correlation': features.get('oil_correlation', pd.Series()),
        'Oil Beta': features.get('oil_beta', pd.Series()),
        'Oil Momentum 20d': features.get('oil_momentum_20d', pd.Series()),
        'Driving Season': features.get('driving_season', pd.Series()),
    }).tail(5)
    print(recent.to_string())
    
    print("\n" + "=" * 60)
    print("Oil-specific features ready for integration!")
    print("\nNote: For production use, integrate EIA API for:")
    print("  - Weekly crude oil inventory data")
    print("  - Refinery utilization rates")
    print("  - Production and import/export data")
