"""
Regime Detection for GP Evolution

Implements market regime detection to adapt strategies based on market conditions:
- Volatility Regimes: High/low volatility periods
- Trend Regimes: Trending vs mean-reverting markets
- Market Regimes: Bull/bear/sideways markets
- Correlation Regimes: High/low correlation periods

Strategies can adapt position sizing, feature selection, and risk management
based on the detected regime.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from scipy import stats


class VolatilityRegime(Enum):
    """Volatility regime classification."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


class TrendRegime(Enum):
    """Trend regime classification."""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    SIDEWAYS = "sideways"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


class MarketRegime(Enum):
    """Market regime classification."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"


class CorrelationRegime(Enum):
    """Correlation regime classification."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass
class RegimeState:
    """Current market regime state."""
    volatility_regime: VolatilityRegime
    trend_regime: TrendRegime
    market_regime: MarketRegime
    correlation_regime: CorrelationRegime
    
    volatility_percentile: float  # 0-1
    trend_strength: float  # -1 to 1
    market_sentiment: float  # -1 to 1
    correlation_level: float  # 0-1
    
    regime_confidence: float  # 0-1


class RegimeDetector:
    """
    Detect market regimes for adaptive strategy selection.
    
    Uses multiple indicators to classify market conditions and provide
    regime-specific recommendations for position sizing and risk management.
    """
    
    def __init__(
        self,
        volatility_lookback: int = 20,
        trend_lookback: int = 50,
        market_lookback: int = 200,
        correlation_lookback: int = 20
    ):
        """
        Args:
            volatility_lookback: Period for volatility calculation
            trend_lookback: Period for trend detection
            market_lookback: Period for market regime detection
            correlation_lookback: Period for correlation calculation
        """
        self.volatility_lookback = volatility_lookback
        self.trend_lookback = trend_lookback
        self.market_lookback = market_lookback
        self.correlation_lookback = correlation_lookback
    
    def detect_volatility_regime(
        self,
        returns: pd.Series,
        lookback: Optional[int] = None
    ) -> Tuple[VolatilityRegime, float]:
        """
        Detect volatility regime.
        
        Uses rolling volatility and compares to historical distribution.
        
        Returns:
            Tuple of (regime, percentile)
        """
        lookback = lookback or self.volatility_lookback
        
        # Calculate rolling volatility
        volatility = returns.rolling(window=lookback).std() * np.sqrt(252)
        
        # Get current volatility
        current_vol = volatility.iloc[-1]
        
        # Calculate historical percentile
        historical_vol = volatility.dropna()
        percentile = stats.percentileofscore(historical_vol, current_vol) / 100
        
        # Classify regime
        if percentile < 0.25:
            regime = VolatilityRegime.LOW
        elif percentile < 0.75:
            regime = VolatilityRegime.NORMAL
        elif percentile < 0.95:
            regime = VolatilityRegime.HIGH
        else:
            regime = VolatilityRegime.EXTREME
        
        return regime, percentile
    
    def detect_trend_regime(
        self,
        prices: pd.Series,
        lookback: Optional[int] = None
    ) -> Tuple[TrendRegime, float]:
        """
        Detect trend regime using ADX and moving averages.
        
        Returns:
            Tuple of (regime, trend_strength)
        """
        lookback = lookback or self.trend_lookback
        
        # Calculate moving averages
        ma_short = prices.rolling(window=lookback // 2).mean()
        ma_long = prices.rolling(window=lookback).mean()
        
        # Trend direction
        trend_direction = (ma_short.iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1]
        
        # Calculate ADX (Average Directional Index)
        adx = self._calculate_adx(prices, lookback)
        current_adx = adx.iloc[-1]
        
        # Trend strength (-1 to 1)
        if trend_direction > 0:
            trend_strength = min(trend_direction * 10, 1.0) * (current_adx / 100)
        else:
            trend_strength = max(trend_direction * 10, -1.0) * (current_adx / 100)
        
        # Classify regime
        if current_adx < 20:
            regime = TrendRegime.SIDEWAYS
        elif trend_strength > 0.5:
            regime = TrendRegime.STRONG_UPTREND
        elif trend_strength > 0.2:
            regime = TrendRegime.WEAK_UPTREND
        elif trend_strength < -0.5:
            regime = TrendRegime.STRONG_DOWNTREND
        elif trend_strength < -0.2:
            regime = TrendRegime.WEAK_DOWNTREND
        else:
            regime = TrendRegime.SIDEWAYS
        
        return regime, trend_strength
    
    def detect_market_regime(
        self,
        prices: pd.Series,
        volume: Optional[pd.Series] = None,
        lookback: Optional[int] = None
    ) -> Tuple[MarketRegime, float]:
        """
        Detect overall market regime (bull/bear/sideways).
        
        Uses long-term moving averages and price action.
        
        Returns:
            Tuple of (regime, sentiment)
        """
        lookback = lookback or self.market_lookback
        
        # Calculate long-term moving average
        ma_long = prices.rolling(window=lookback).mean()
        
        # Current position relative to MA
        current_price = prices.iloc[-1]
        ma_value = ma_long.iloc[-1]
        distance_from_ma = (current_price - ma_value) / ma_value
        
        # Calculate returns
        returns = prices.pct_change()
        recent_returns = returns.iloc[-lookback // 4:]  # Last quarter of lookback
        
        # Market sentiment (-1 to 1)
        sentiment = distance_from_ma * 2  # Scale to roughly -1 to 1
        sentiment = np.clip(sentiment, -1, 1)
        
        # Calculate volatility for regime classification
        volatility = returns.rolling(window=20).std().iloc[-1]
        avg_volatility = returns.rolling(window=lookback).std().mean()
        
        # Classify regime
        if current_price > ma_value * 1.05 and recent_returns.mean() > 0:
            regime = MarketRegime.BULL
        elif current_price < ma_value * 0.95 and recent_returns.mean() < 0:
            regime = MarketRegime.BEAR
        elif volatility > avg_volatility * 1.5:
            regime = MarketRegime.VOLATILE
        else:
            regime = MarketRegime.SIDEWAYS
        
        return regime, sentiment
    
    def detect_correlation_regime(
        self,
        returns_matrix: pd.DataFrame,
        lookback: Optional[int] = None
    ) -> Tuple[CorrelationRegime, float]:
        """
        Detect correlation regime across assets.
        
        High correlation = risk-off, low correlation = risk-on
        
        Args:
            returns_matrix: DataFrame with returns for multiple assets
        
        Returns:
            Tuple of (regime, correlation_level)
        """
        lookback = lookback or self.correlation_lookback
        
        # Calculate rolling correlation matrix
        recent_returns = returns_matrix.iloc[-lookback:]
        corr_matrix = recent_returns.corr()
        
        # Average correlation (excluding diagonal)
        n = len(corr_matrix)
        if n < 2:
            return CorrelationRegime.LOW, 0.0
        
        # Get upper triangle (excluding diagonal)
        upper_triangle = np.triu(corr_matrix.values, k=1)
        correlations = upper_triangle[upper_triangle != 0]
        
        avg_correlation = np.mean(np.abs(correlations))
        
        # Classify regime
        if avg_correlation < 0.3:
            regime = CorrelationRegime.LOW
        elif avg_correlation < 0.6:
            regime = CorrelationRegime.MODERATE
        else:
            regime = CorrelationRegime.HIGH
        
        return regime, avg_correlation
    
    def detect_all_regimes(
        self,
        prices: pd.Series,
        returns: pd.Series,
        returns_matrix: Optional[pd.DataFrame] = None,
        volume: Optional[pd.Series] = None
    ) -> RegimeState:
        """
        Detect all market regimes.
        
        Returns:
            RegimeState with all regime classifications
        """
        # Detect each regime
        vol_regime, vol_percentile = self.detect_volatility_regime(returns)
        trend_regime, trend_strength = self.detect_trend_regime(prices)
        market_regime, market_sentiment = self.detect_market_regime(prices, volume)
        
        # Correlation regime (if returns matrix provided)
        if returns_matrix is not None and len(returns_matrix.columns) > 1:
            corr_regime, corr_level = self.detect_correlation_regime(returns_matrix)
        else:
            corr_regime = CorrelationRegime.MODERATE
            corr_level = 0.5
        
        # Calculate regime confidence
        # Higher confidence when regimes are consistent
        confidence = self._calculate_regime_confidence(
            vol_regime, trend_regime, market_regime
        )
        
        return RegimeState(
            volatility_regime=vol_regime,
            trend_regime=trend_regime,
            market_regime=market_regime,
            correlation_regime=corr_regime,
            volatility_percentile=vol_percentile,
            trend_strength=trend_strength,
            market_sentiment=market_sentiment,
            correlation_level=corr_level,
            regime_confidence=confidence
        )
    
    def get_regime_weights(
        self,
        regime_state: RegimeState
    ) -> Dict[str, float]:
        """
        Get position sizing weights based on regime.
        
        Returns:
            Dict with recommended adjustments:
            - position_size_multiplier: Scale positions by this factor
            - max_positions: Maximum number of positions
            - stop_loss_multiplier: Scale stop losses by this factor
        """
        weights = {
            'position_size_multiplier': 1.0,
            'max_positions': 10,
            'stop_loss_multiplier': 1.0
        }
        
        # Adjust for volatility regime
        if regime_state.volatility_regime == VolatilityRegime.LOW:
            weights['position_size_multiplier'] *= 1.2
            weights['stop_loss_multiplier'] *= 0.8
        elif regime_state.volatility_regime == VolatilityRegime.HIGH:
            weights['position_size_multiplier'] *= 0.7
            weights['stop_loss_multiplier'] *= 1.5
        elif regime_state.volatility_regime == VolatilityRegime.EXTREME:
            weights['position_size_multiplier'] *= 0.5
            weights['stop_loss_multiplier'] *= 2.0
        
        # Adjust for market regime
        if regime_state.market_regime == MarketRegime.BULL:
            weights['position_size_multiplier'] *= 1.1
            weights['max_positions'] = 12
        elif regime_state.market_regime == MarketRegime.BEAR:
            weights['position_size_multiplier'] *= 0.8
            weights['max_positions'] = 6
        elif regime_state.market_regime == MarketRegime.VOLATILE:
            weights['position_size_multiplier'] *= 0.6
            weights['max_positions'] = 5
        
        # Adjust for correlation regime
        if regime_state.correlation_regime == CorrelationRegime.HIGH:
            # High correlation = less diversification benefit
            weights['max_positions'] = min(weights['max_positions'], 5)
            weights['position_size_multiplier'] *= 0.9
        
        return weights
    
    def _calculate_adx(
        self,
        prices: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).
        
        ADX measures trend strength (0-100).
        """
        high = prices  # Simplified: using close as high
        low = prices   # Simplified: using close as low
        close = prices
        
        # Calculate +DM and -DM
        up_move = high.diff()
        down_move = -low.diff()
        
        plus_dm = pd.Series(0.0, index=prices.index)
        minus_dm = pd.Series(0.0, index=prices.index)
        
        plus_dm[up_move > down_move] = up_move[up_move > down_move]
        plus_dm[plus_dm < 0] = 0
        
        minus_dm[down_move > up_move] = down_move[down_move > up_move]
        minus_dm[minus_dm < 0] = 0
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smooth with EMA
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx.fillna(0)
    
    def _calculate_regime_confidence(
        self,
        vol_regime: VolatilityRegime,
        trend_regime: TrendRegime,
        market_regime: MarketRegime
    ) -> float:
        """
        Calculate confidence in regime classification.
        
        Higher confidence when regimes are consistent.
        """
        confidence = 0.5  # Base confidence
        
        # Check for consistency
        # Bull market + uptrend + normal vol = high confidence
        if market_regime == MarketRegime.BULL:
            if trend_regime in [TrendRegime.STRONG_UPTREND, TrendRegime.WEAK_UPTREND]:
                confidence += 0.2
            if vol_regime in [VolatilityRegime.LOW, VolatilityRegime.NORMAL]:
                confidence += 0.1
        
        # Bear market + downtrend = high confidence
        elif market_regime == MarketRegime.BEAR:
            if trend_regime in [TrendRegime.STRONG_DOWNTREND, TrendRegime.WEAK_DOWNTREND]:
                confidence += 0.2
            if vol_regime == VolatilityRegime.HIGH:
                confidence += 0.1
        
        # Volatile market + high vol = high confidence
        elif market_regime == MarketRegime.VOLATILE:
            if vol_regime in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME]:
                confidence += 0.3
        
        return min(confidence, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example: Detect regimes for a stock
    import yfinance as yf
    
    print("Regime Detection - Example")
    print("=" * 60)
    
    # Fetch sample data
    ticker = "SPY"
    data = yf.download(ticker, start="2020-01-01", end="2024-01-01", progress=False)
    
    # Calculate returns
    returns = data['Close'].pct_change().dropna()
    
    # Initialize detector
    detector = RegimeDetector(
        volatility_lookback=20,
        trend_lookback=50,
        market_lookback=200,
        correlation_lookback=20
    )
    
    # Detect regimes
    regime_state = detector.detect_all_regimes(
        prices=data['Close'],
        returns=returns,
        volume=data['Volume']
    )
    
    # Display results
    print(f"\nCurrent Regime State for {ticker}:")
    print(f"  Volatility: {regime_state.volatility_regime.value} "
          f"(percentile: {regime_state.volatility_percentile:.1%})")
    print(f"  Trend: {regime_state.trend_regime.value} "
          f"(strength: {regime_state.trend_strength:+.2f})")
    print(f"  Market: {regime_state.market_regime.value} "
          f"(sentiment: {regime_state.market_sentiment:+.2f})")
    print(f"  Correlation: {regime_state.correlation_regime.value} "
          f"(level: {regime_state.correlation_level:.2f})")
    print(f"  Confidence: {regime_state.regime_confidence:.1%}")
    
    # Get recommended weights
    weights = detector.get_regime_weights(regime_state)
    print(f"\nRecommended Adjustments:")
    print(f"  Position Size Multiplier: {weights['position_size_multiplier']:.2f}x")
    print(f"  Max Positions: {weights['max_positions']}")
    print(f"  Stop Loss Multiplier: {weights['stop_loss_multiplier']:.2f}x")
    
    # Analyze regime changes over time
    print(f"\nRegime History (last 12 months):")
    regime_history = []
    
    for i in range(-252, 0, 21):  # Monthly samples
        if abs(i) > len(data):
            continue
        
        period_data = data.iloc[:i]
        period_returns = period_data['Close'].pct_change().dropna()
        
        if len(period_returns) < 200:
            continue
        
        state = detector.detect_all_regimes(
            prices=period_data['Close'],
            returns=period_returns,
            volume=period_data['Volume']
        )
        
        regime_history.append({
            'date': period_data.index[-1].strftime('%Y-%m-%d'),
            'volatility': state.volatility_regime.value,
            'trend': state.trend_regime.value,
            'market': state.market_regime.value
        })
    
    for entry in regime_history[-6:]:  # Show last 6 months
        print(f"  {entry['date']}: {entry['market']:10s} | "
              f"{entry['trend']:20s} | {entry['volatility']:10s}")
    
    print("\n" + "=" * 60)
    print("Regime detection ready for integration into GP evolution!")
