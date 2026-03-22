# evolution/features.py
"""
Comprehensive feature library for GP strategies (~140 features).

Contains: FeatureLibrary
"""

import numpy as np
import pandas as pd
import hashlib
from typing import Dict, List, Optional


class FeatureLibrary:
    """
    Comprehensive feature library with ~140 orthogonalized features.
    """
    
    def __init__(self, enable_smc=False, enable_sr=False, enable_oil=False):
        self.enable_smc = enable_smc
        self.enable_sr = enable_sr
        self.enable_oil = enable_oil
        
        self.feature_specs = self._build_feature_specs()
        self.feature_names = list(self.feature_specs.keys())
        self.max_lookback = 504  # 2 years
        self.feature_categories = self._build_category_map()
        
        # Features to exclude from rank transform (already ranked or naturally bounded 0-1)
        self._skip_rank_transform = {
            'relative_strength_21d', 'relative_strength_63d', 'relative_strength_126d',
            'relative_volatility_21d', 'relative_volatility_63d',
            'range_position_21d', 'range_position_63d', 'range_position_252d',
            'high_proximity_63d', 'high_proximity_252d',
            'recovery_rate_63d', 'recovery_rate_126d',
            'mom_consistency_21d', 'mom_consistency_63d',
            'hurst_proxy_63d', 'hurst_proxy_126d',
        }
        
        # Initialize advanced feature calculators if enabled
        self.smc_features = None
        self.sr_features = None
        self.oil_features = None
        
        if self.enable_smc:
            from evolution.smart_money_features import SmartMoneyFeatures
            self.smc_features = SmartMoneyFeatures()
        
        if self.enable_sr:
            from evolution.support_resistance_features import SupportResistanceFeatures
            self.sr_features = SupportResistanceFeatures()
        
        if self.enable_oil:
            from evolution.oil_specific_features import OilSpecificFeatures
            self.oil_features = OilSpecificFeatures()
        
        # Feature cache for performance optimization
        self._feature_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _build_feature_specs(self) -> Dict:
        """Build all feature specifications."""
        specs = {}
        
        # ═══════════════════════════════════════════════════════════════════
        # MOMENTUM FEATURES
        # ═══════════════════════════════════════════════════════════════════
        
        specs['mom_5d'] = ('momentum', 5)
        specs['mom_10d'] = ('momentum', 10)
        specs['mom_21d'] = ('momentum', 21)
        specs['mom_63d'] = ('momentum', 63)
        specs['mom_126d'] = ('momentum', 126)
        specs['mom_252d'] = ('momentum', 252)
        
        specs['mom_12_1'] = ('momentum_skip', 252, 21)
        specs['mom_6_1'] = ('momentum_skip', 126, 21)
        specs['mom_3_1'] = ('momentum_skip', 63, 21)
        
        specs['mom_accel_short'] = ('momentum_accel', 5, 21)
        specs['mom_accel_medium'] = ('momentum_accel', 21, 63)
        specs['mom_accel_long'] = ('momentum_accel', 63, 252)
        
        specs['mom_consistency_21d'] = ('momentum_consistency', 21, 5)
        specs['mom_consistency_63d'] = ('momentum_consistency', 63, 21)
        
        specs['mom_smoothness_21d'] = ('momentum_smoothness', 21)
        specs['mom_smoothness_63d'] = ('momentum_smoothness', 63)
        
        # ═══════════════════════════════════════════════════════════════════
        # MEAN REVERSION FEATURES
        # ═══════════════════════════════════════════════════════════════════
        
        specs['zscore_5d'] = ('zscore', 5)
        specs['zscore_10d'] = ('zscore', 10)
        specs['zscore_21d'] = ('zscore', 21)
        specs['zscore_63d'] = ('zscore', 63)
        
        specs['dist_ma_10'] = ('distance_from_ma', 10)
        specs['dist_ma_21'] = ('distance_from_ma', 21)
        specs['dist_ma_50'] = ('distance_from_ma', 50)
        specs['dist_ma_200'] = ('distance_from_ma', 200)
        
        specs['reversion_speed_10d'] = ('reversion_speed', 10)
        specs['reversion_speed_21d'] = ('reversion_speed', 21)
        specs['reversion_speed_63d'] = ('reversion_speed', 63)
        
        specs['rsi_14'] = ('rsi', 14)
        specs['rsi_28'] = ('rsi', 28)
        
        # ═══════════════════════════════════════════════════════════════════
        # VOLATILITY FEATURES
        # ═══════════════════════════════════════════════════════════════════
        
        specs['vol_5d'] = ('volatility', 5)
        specs['vol_21d'] = ('volatility', 21)
        specs['vol_63d'] = ('volatility', 63)
        specs['vol_252d'] = ('volatility', 252)
        
        specs['vol_ratio_5_21'] = ('vol_ratio', 5, 21)
        specs['vol_ratio_21_63'] = ('vol_ratio', 21, 63)
        specs['vol_ratio_63_252'] = ('vol_ratio', 63, 252)
        
        specs['vol_of_vol_21d'] = ('vol_of_vol', 21)
        specs['vol_of_vol_63d'] = ('vol_of_vol', 63)
        
        specs['vol_trend_21d'] = ('vol_trend', 21, 63)
        
        # ═══════════════════════════════════════════════════════════════════
        # DRAWDOWN AND RISK FEATURES
        # ═══════════════════════════════════════════════════════════════════
        
        specs['drawdown_21d'] = ('drawdown', 21)
        specs['drawdown_63d'] = ('drawdown', 63)
        specs['drawdown_126d'] = ('drawdown', 126)
        specs['drawdown_252d'] = ('drawdown', 252)
        
        specs['drawdown_duration_63d'] = ('drawdown_duration', 63)
        specs['drawdown_duration_252d'] = ('drawdown_duration', 252)
        
        specs['recovery_rate_63d'] = ('recovery_rate', 63)
        specs['recovery_rate_126d'] = ('recovery_rate', 126)
        
        specs['max_dd_63d'] = ('max_drawdown', 63)
        specs['max_dd_126d'] = ('max_drawdown', 126)
        
        specs['ulcer_index_63d'] = ('ulcer_index', 63)
        
        # --- CONTINUED IN _build_feature_specs_part2 ---
        self._build_feature_specs_part2(specs)
        
        return specs
    
    def _build_feature_specs_part2(self, specs: Dict):
        """Continue building feature specs (higher moments through patterns)."""
        
        # ═══════════════════════════════════════════════════════════════════
        # HIGHER MOMENTS
        # ═══════════════════════════════════════════════════════════════════
        
        specs['skew_21d'] = ('skewness', 21)
        specs['skew_63d'] = ('skewness', 63)
        specs['skew_126d'] = ('skewness', 126)
        
        specs['kurt_21d'] = ('kurtosis', 21)
        specs['kurt_63d'] = ('kurtosis', 63)
        specs['kurt_126d'] = ('kurtosis', 126)
        
        specs['downside_dev_21d'] = ('downside_deviation', 21)
        specs['downside_dev_63d'] = ('downside_deviation', 63)
        
        specs['up_down_ratio_63d'] = ('up_down_capture', 63)
        
        specs['left_tail_21d'] = ('left_tail', 21, 0.05)
        specs['left_tail_63d'] = ('left_tail', 63, 0.05)
        
        # ═══════════════════════════════════════════════════════════════════
        # TREND QUALITY FEATURES
        # ═══════════════════════════════════════════════════════════════════
        
        specs['trend_r2_21d'] = ('trend_r2', 21)
        specs['trend_r2_63d'] = ('trend_r2', 63)
        specs['trend_r2_126d'] = ('trend_r2', 126)
        
        specs['trend_slope_21d'] = ('trend_slope', 21)
        specs['trend_slope_63d'] = ('trend_slope', 63)
        
        specs['trend_deviation_21d'] = ('trend_deviation', 21)
        specs['trend_deviation_63d'] = ('trend_deviation', 63)
        
        specs['hurst_proxy_63d'] = ('hurst_proxy', 63)
        specs['hurst_proxy_126d'] = ('hurst_proxy', 126)
        
        # ═══════════════════════════════════════════════════════════════════
        # PRICE LEVEL FEATURES
        # ═══════════════════════════════════════════════════════════════════
        
        specs['range_position_21d'] = ('range_position', 21)
        specs['range_position_63d'] = ('range_position', 63)
        specs['range_position_252d'] = ('range_position', 252)
        
        specs['high_proximity_63d'] = ('high_proximity', 63)
        specs['high_proximity_252d'] = ('high_proximity', 252)
        
        specs['breakout_21d'] = ('breakout_strength', 21)
        specs['breakout_63d'] = ('breakout_strength', 63)
        
        specs['support_distance_21d'] = ('support_distance', 21)
        specs['resistance_distance_21d'] = ('resistance_distance', 21)
        
        # ═══════════════════════════════════════════════════════════════════
        # CROSS-SECTIONAL FEATURES
        # ═══════════════════════════════════════════════════════════════════
        
        specs['rel_strength_21d'] = ('relative_strength', 21)
        specs['rel_strength_63d'] = ('relative_strength', 63)
        specs['rel_strength_126d'] = ('relative_strength', 126)
        
        specs['rel_vol_21d'] = ('relative_volatility', 21)
        specs['rel_vol_63d'] = ('relative_volatility', 63)
        
        specs['excess_mom_21d'] = ('excess_momentum', 21)
        specs['excess_mom_63d'] = ('excess_momentum', 63)
        
        specs['rel_value_21d'] = ('relative_value', 21)
        specs['rel_value_63d'] = ('relative_value', 63)
        
        # ═══════════════════════════════════════════════════════════════════
        # RISK-ADJUSTED FEATURES
        # ═══════════════════════════════════════════════════════════════════
        
        specs['sharpe_21d'] = ('sharpe_ratio', 21)
        specs['sharpe_63d'] = ('sharpe_ratio', 63)
        specs['sharpe_126d'] = ('sharpe_ratio', 126)
        
        specs['sortino_21d'] = ('sortino_ratio', 21)
        specs['sortino_63d'] = ('sortino_ratio', 63)
        
        specs['calmar_63d'] = ('calmar_ratio', 63)
        specs['calmar_126d'] = ('calmar_ratio', 126)
        
        specs['info_ratio_63d'] = ('information_ratio', 63)
        specs['info_ratio_126d'] = ('information_ratio', 126)
        
        # ═══════════════════════════════════════════════════════════════════
        # VOLUME/LIQUIDITY FEATURES (optional)
        # ═══════════════════════════════════════════════════════════════════
        
        specs['volume_trend_21d'] = ('volume_trend', 21)
        specs['volume_ratio_5_21'] = ('volume_ratio', 5, 21)
        specs['price_volume_corr_21d'] = ('price_volume_corr', 21)
        specs['volume_volatility_21d'] = ('volume_volatility', 21)
        
        # ═══════════════════════════════════════════════════════════════════
        # PATTERN FEATURES
        # ═══════════════════════════════════════════════════════════════════
        
        specs['gap_reversal_10d'] = ('gap_reversal', 10)
        specs['streak_strength'] = ('streak_strength', 21)
        specs['return_autocorr_1d'] = ('return_autocorr', 21, 1)
        specs['return_autocorr_5d'] = ('return_autocorr', 21, 5)
        specs['abs_return_21d'] = ('abs_return_momentum', 21)
        specs['abs_return_63d'] = ('abs_return_momentum', 63)
        
        # --- CONTINUED IN _build_feature_specs_part3 ---
        self._build_feature_specs_part3(specs)
    
    def _build_feature_specs_part3(self, specs: Dict):
        """Continue building feature specs (optional + NEW features)."""
        
        # ═══════════════════════════════════════════════════════════════════
        # SMART MONEY CONCEPTS (SMC) - Optional
        # ═══════════════════════════════════════════════════════════════════
        if self.enable_smc:
            specs['smc_order_block_bull'] = ('smc', 'order_block_bull')
            specs['smc_order_block_bear'] = ('smc', 'order_block_bear')
            specs['smc_fvg_bull'] = ('smc', 'fvg_bull')
            specs['smc_fvg_bear'] = ('smc', 'fvg_bear')
            specs['smc_liquidity_sweep'] = ('smc', 'liquidity_sweep')
            specs['smc_break_of_structure'] = ('smc', 'break_of_structure')
        
        # ═══════════════════════════════════════════════════════════════════
        # SUPPORT/RESISTANCE - Optional
        # ═══════════════════════════════════════════════════════════════════
        if self.enable_sr:
            specs['sr_poc_distance'] = ('sr', 'poc_distance')
            specs['sr_value_area_position'] = ('sr', 'value_area_position')
            specs['sr_pivot_traditional'] = ('sr', 'pivot_traditional')
            specs['sr_pivot_fibonacci'] = ('sr', 'pivot_fibonacci')
            specs['sr_pivot_camarilla'] = ('sr', 'pivot_camarilla')
            specs['sr_bb_position'] = ('sr', 'bb_position')
            specs['sr_keltner_position'] = ('sr', 'keltner_position')
            specs['sr_historical_level'] = ('sr', 'historical_level')
        
        # ═══════════════════════════════════════════════════════════════════
        # OIL-SPECIFIC FEATURES - Optional
        # ═══════════════════════════════════════════════════════════════════
        if self.enable_oil:
            specs['oil_wti_correlation'] = ('oil', 'wti_correlation')
            specs['oil_brent_correlation'] = ('oil', 'brent_correlation')
            specs['oil_wti_beta'] = ('oil', 'wti_beta')
            specs['oil_inventory_zscore'] = ('oil', 'inventory_zscore')
            specs['oil_inventory_change'] = ('oil', 'inventory_change')
            specs['oil_crack_spread_321'] = ('oil', 'crack_spread_321')
            specs['oil_crack_spread_532'] = ('oil', 'crack_spread_532')
            specs['oil_seasonal_driving'] = ('oil', 'seasonal_driving')
            specs['oil_seasonal_heating'] = ('oil', 'seasonal_heating')
            specs['oil_wti_brent_spread'] = ('oil', 'wti_brent_spread')
        
        # ═══════════════════════════════════════════════════════════════════
        # MARKET-RELATIVE FEATURES (NEW)
        # ═══════════════════════════════════════════════════════════════════
        
        specs['market_beta_21d'] = ('market_beta', 21)
        specs['market_beta_63d'] = ('market_beta', 63)
        specs['market_beta_126d'] = ('market_beta', 126)
        
        specs['beta_instability_63d'] = ('beta_instability', 63)
        
        specs['idio_vol_21d'] = ('idiosyncratic_vol', 21)
        specs['idio_vol_63d'] = ('idiosyncratic_vol', 63)
        
        specs['sector_correlation_21d'] = ('sector_correlation', 21)
        specs['sector_correlation_63d'] = ('sector_correlation', 63)
        
        specs['universe_dispersion_21d'] = ('universe_dispersion', 21)
        
        # ═══════════════════════════════════════════════════════════════════
        # MICROSTRUCTURE FEATURES (NEW)
        # ═══════════════════════════════════════════════════════════════════
        
        specs['amihud_21d'] = ('amihud_illiquidity', 21)
        specs['amihud_63d'] = ('amihud_illiquidity', 63)
        
        specs['roll_spread_21d'] = ('roll_spread', 21)
        
        specs['kyle_lambda_21d'] = ('kyle_lambda', 21)
        
        specs['zero_return_days_21d'] = ('zero_return_fraction', 21)
        specs['zero_return_days_63d'] = ('zero_return_fraction', 63)
        
        specs['turnover_rate_21d'] = ('turnover_rate', 21)
        
        # ═══════════════════════════════════════════════════════════════════
        # REGIME FEATURES (NEW)
        # ═══════════════════════════════════════════════════════════════════
        
        specs['vol_regime'] = ('volatility_regime', 63)
        
        specs['trend_strength_21d'] = ('adx_proxy', 21)
        specs['trend_strength_63d'] = ('adx_proxy', 63)
        
        specs['correlation_spike_21d'] = ('correlation_spike', 21)
        
        specs['breadth_50d'] = ('market_breadth', 50)
        specs['breadth_200d'] = ('market_breadth', 200)
        
        # ═══════════════════════════════════════════════════════════════════
        # ENGINEERED FEATURES (NEW)
        # ═══════════════════════════════════════════════════════════════════
        
        specs['mom_sharpe_21d'] = ('momentum_sharpe', 21)
        specs['mom_sharpe_63d'] = ('momentum_sharpe', 63)
        
        specs['dd_adj_mom_63d'] = ('drawdown_adjusted_momentum', 63)
        
        specs['vol_adj_mom_21d'] = ('vol_adjusted_momentum', 21)
        specs['vol_adj_mom_63d'] = ('vol_adjusted_momentum', 63)
        
        specs['efficiency_ratio_21d'] = ('efficiency_ratio', 21)
        specs['efficiency_ratio_63d'] = ('efficiency_ratio', 63)
        
        specs['fractal_dim_63d'] = ('fractal_dimension', 63)
    
    def _build_category_map(self) -> Dict[str, List[str]]:
        """Map features to categories for analysis."""
        categories = {
            'momentum': [],
            'mean_reversion': [],
            'volatility': [],
            'drawdown': [],
            'higher_moments': [],
            'trend': [],
            'price_level': [],
            'cross_sectional': [],
            'risk_adjusted': [],
            'volume': [],
            'pattern': [],
            'market_relative': [],
            'microstructure': [],
            'regime': [],
            'engineered': [],
        }
        
        for name in self.feature_names:
            if name.startswith('mom_') and not name.startswith('mom_sharpe'):
                categories['momentum'].append(name)
            elif name.startswith(('zscore', 'dist_ma', 'reversion', 'rsi')):
                categories['mean_reversion'].append(name)
            elif name.startswith('vol_') and not name.startswith('vol_regime') and not name.startswith('vol_adj'):
                categories['volatility'].append(name)
            elif name.startswith(('drawdown', 'recovery', 'max_dd', 'ulcer')):
                categories['drawdown'].append(name)
            elif name.startswith(('skew', 'kurt', 'downside', 'up_down', 'left_tail')):
                categories['higher_moments'].append(name)
            elif name.startswith(('trend_r2', 'trend_slope', 'trend_deviation', 'hurst')):
                categories['trend'].append(name)
            elif name.startswith(('range', 'high_prox', 'breakout', 'support', 'resistance')):
                categories['price_level'].append(name)
            elif name.startswith(('rel_', 'excess')):
                categories['cross_sectional'].append(name)
            elif name.startswith(('sharpe', 'sortino', 'calmar', 'info_ratio')):
                categories['risk_adjusted'].append(name)
            elif name.startswith(('volume', 'price_volume')):
                categories['volume'].append(name)
            elif name.startswith(('market_beta', 'beta_instability', 'idio_vol', 'sector_corr', 'universe_disp')):
                categories['market_relative'].append(name)
            elif name.startswith(('amihud', 'roll_spread', 'kyle_lambda', 'zero_return', 'turnover_rate')):
                categories['microstructure'].append(name)
            elif name.startswith(('vol_regime', 'trend_strength', 'correlation_spike', 'breadth')):
                categories['regime'].append(name)
            elif name.startswith(('mom_sharpe', 'dd_adj', 'vol_adj', 'efficiency', 'fractal')):
                categories['engineered'].append(name)
            else:
                categories['pattern'].append(name)
        
        return categories
    
    def _make_cache_key(self, prices: pd.DataFrame, lag: int, rank_transform: bool) -> str:
        """Create a hashable cache key from prices DataFrame."""
        last_date = str(prices.index[-1]) if len(prices) > 0 else "empty"
        shape_str = f"{len(prices)}x{len(prices.columns)}"
        cols_hash = hashlib.md5(','.join(sorted(prices.columns)).encode()).hexdigest()[:8]
        return f"{last_date}_{shape_str}_{cols_hash}_lag{lag}_rank{rank_transform}"
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'total': total,
            'hit_rate': hit_rate,
            'cache_size': len(self._feature_cache)
        }
    
    def clear_cache(self):
        """Clear feature cache and reset statistics."""
        self._feature_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def compute_all(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame = None,
        lag: int = 1,
        rank_transform: bool = True,
        use_cache: bool = True,
    ) -> Dict[str, pd.Series]:
        """Compute all features with optional caching."""
        if use_cache:
            cache_key = self._make_cache_key(prices, lag, rank_transform)
            if cache_key in self._feature_cache:
                self._cache_hits += 1
                return self._feature_cache[cache_key].copy()
            self._cache_misses += 1
        
        if lag > 0 and len(prices) > lag:
            lagged_prices = prices.iloc[:-lag]
            lagged_volume = volume.iloc[:-lag] if volume is not None else None
        else:
            lagged_prices = prices
            lagged_volume = volume
        
        features = {}
        for name, spec in self.feature_specs.items():
            try:
                features[name] = self._compute_feature(lagged_prices, lagged_volume, spec)
            except Exception:
                features[name] = pd.Series(0.0, index=prices.columns)
        
        if rank_transform:
            features = self._rank_transform(features)
        
        if use_cache:
            if len(self._feature_cache) >= 1000:
                oldest_key = next(iter(self._feature_cache))
                del self._feature_cache[oldest_key]
            self._feature_cache[cache_key] = features.copy()
        
        return features
    
    def _rank_transform(self, features: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """Transform features to cross-sectional percentile ranks."""
        ranked = {}
        for name, series in features.items():
            if name in self._skip_rank_transform:
                ranked[name] = series
                continue
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            if series.dtype == object:
                try:
                    series = series.astype(float)
                except (ValueError, TypeError):
                    ranked[name] = pd.Series(0.0, index=series.index)
                    continue
            if series.empty or bool(series.isna().all()):
                ranked[name] = series
                continue
            ranked[name] = series.rank(pct=True, method='average', na_option='keep')
        return ranked

    def _compute_feature(
        self,
        prices: pd.DataFrame,
        volume: Optional[pd.DataFrame],
        spec: tuple
    ) -> pd.Series:
        """Compute a single feature."""
        
        feature_type = spec[0]
        tickers = prices.columns
        
        # ═══════════════════════════════════════════════════════════════════
        # MOMENTUM
        # ═══════════════════════════════════════════════════════════════════
        
        if feature_type == 'momentum':
            period = spec[1]
            if len(prices) < period + 1:
                return pd.Series(0.0, index=tickers)
            return (prices.iloc[-1] / prices.iloc[-period - 1]) - 1
        
        elif feature_type == 'momentum_skip':
            lookback, skip = spec[1], spec[2]
            if len(prices) < lookback + skip + 1:
                return pd.Series(0.0, index=tickers)
            start_idx = -(lookback + skip + 1)
            end_idx = -(skip + 1) if skip > 0 else -1
            return (prices.iloc[end_idx] / prices.iloc[start_idx]) - 1
        
        elif feature_type == 'momentum_accel':
            short, long = spec[1], spec[2]
            if len(prices) < long + 1:
                return pd.Series(0.0, index=tickers)
            mom_short = (prices.iloc[-1] / prices.iloc[-short - 1]) - 1
            mom_long = (prices.iloc[-1] / prices.iloc[-long - 1]) - 1
            return mom_short - (mom_long * short / long)
        
        elif feature_type == 'momentum_consistency':
            period, sub_period = spec[1], spec[2]
            if len(prices) < period:
                return pd.Series(0.5, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            n_sub = period // sub_period
            def calc_consistency(col):
                positive_subs = 0
                for i in range(n_sub):
                    start = i * sub_period
                    end = (i + 1) * sub_period
                    sub_return = col.iloc[start:end].sum()
                    if sub_return > 0:
                        positive_subs += 1
                return positive_subs / max(n_sub, 1)
            return returns.apply(calc_consistency)
        
        elif feature_type == 'momentum_smoothness':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.5, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            total_return = (prices.iloc[-1] / prices.iloc[-period]) - 1
            abs_sum = returns.abs().sum()
            smoothness = total_return / abs_sum.replace(0, np.nan)
            return smoothness.fillna(0).clip(-1, 1)
        
        # ═══════════════════════════════════════════════════════════════════
        # MEAN REVERSION
        # ═══════════════════════════════════════════════════════════════════
        
        elif feature_type == 'zscore':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            recent = prices.iloc[-period:]
            mean = recent.mean()
            std = recent.std()
            current = prices.iloc[-1]
            z = (current - mean) / std.replace(0, np.nan)
            return z.fillna(0).clip(-4, 4)
        
        elif feature_type == 'distance_from_ma':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            ma = prices.iloc[-period:].mean()
            current = prices.iloc[-1]
            return (current / ma) - 1
        
        elif feature_type == 'reversion_speed':
            period = spec[1]
            if len(prices) < period * 2:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            def calc_autocorr(x):
                if len(x.dropna()) < 5:
                    return 0
                ac = x.autocorr(lag=1)
                return -ac if not np.isnan(ac) else 0
            return returns.apply(calc_autocorr)
        
        elif feature_type == 'rsi':
            period = spec[1]
            if len(prices) < period + 1:
                return pd.Series(50.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            gains = returns.clip(lower=0).mean()
            losses = (-returns.clip(upper=0)).mean()
            rs = gains / losses.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        
        # ═══════════════════════════════════════════════════════════════════
        # VOLATILITY
        # ═══════════════════════════════════════════════════════════════════
        
        elif feature_type == 'volatility':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            return returns.std() * np.sqrt(252)
        
        elif feature_type == 'vol_ratio':
            short, long = spec[1], spec[2]
            if len(prices) < long:
                return pd.Series(1.0, index=tickers)
            returns = prices.pct_change()
            vol_short = returns.iloc[-short:].std()
            vol_long = returns.iloc[-long:].std()
            return (vol_short / vol_long.replace(0, np.nan)).fillna(1.0)
        
        elif feature_type == 'vol_of_vol':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change()
            rolling_vol = returns.rolling(5).std()
            return rolling_vol.iloc[-period:].std()
        
        elif feature_type == 'vol_trend':
            short, long = spec[1], spec[2]
            if len(prices) < long:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change()
            vol_now = returns.iloc[-short:].std()
            vol_before = returns.iloc[-long:-short].std()
            return (vol_now / vol_before.replace(0, np.nan) - 1).fillna(0)

        # ═══════════════════════════════════════════════════════════════════
        # DRAWDOWN AND RISK
        # ═══════════════════════════════════════════════════════════════════
        
        elif feature_type == 'drawdown':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            recent = prices.iloc[-period:]
            peak = recent.max()
            current = prices.iloc[-1]
            return (current - peak) / peak
        
        elif feature_type == 'drawdown_duration':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            recent = prices.iloc[-period:]
            running_max = recent.cummax()
            at_high = (recent == running_max)
            def days_since_high(col):
                at_high_col = at_high[col.name]
                if at_high_col.iloc[-1]:
                    return 0
                try:
                    last_high_idx = at_high_col[::-1].idxmax()
                    return len(at_high_col) - at_high_col.index.get_loc(last_high_idx) - 1
                except:
                    return period
            return recent.apply(days_since_high) / period
        
        elif feature_type == 'recovery_rate':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.5, index=tickers)
            recent = prices.iloc[-period:]
            running_max = recent.cummax()
            drawdown = (recent - running_max) / running_max
            near_high = (drawdown > -0.05).mean()
            return near_high
        
        elif feature_type == 'max_drawdown':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            recent = prices.iloc[-period:]
            running_max = recent.cummax()
            drawdown = (recent - running_max) / running_max
            return drawdown.min()
        
        elif feature_type == 'ulcer_index':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            recent = prices.iloc[-period:]
            running_max = recent.cummax()
            drawdown_pct = (recent - running_max) / running_max * 100
            ulcer = np.sqrt((drawdown_pct ** 2).mean())
            return -ulcer / 100
        
        # ═══════════════════════════════════════════════════════════════════
        # HIGHER MOMENTS
        # ═══════════════════════════════════════════════════════════════════
        
        elif feature_type == 'skewness':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            return returns.skew().clip(-3, 3)
        
        elif feature_type == 'kurtosis':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            return returns.kurtosis().clip(-5, 10)
        
        elif feature_type == 'downside_deviation':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            negative_returns = returns.clip(upper=0)
            return negative_returns.std() * np.sqrt(252)
        
        elif feature_type == 'up_down_capture':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(1.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            market_returns = returns.mean(axis=1)
            up_days = market_returns > 0
            down_days = market_returns < 0
            if up_days.sum() > 0 and down_days.sum() > 0:
                up_capture = returns.loc[up_days].mean() / market_returns.loc[up_days].mean()
                down_capture = returns.loc[down_days].mean() / market_returns.loc[down_days].mean()
                return (up_capture - down_capture).fillna(0)
            return pd.Series(0.0, index=tickers)
        
        elif feature_type == 'left_tail':
            period, quantile = spec[1], spec[2]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            es = returns.apply(lambda x: x.quantile(quantile))
            return -es
        
        # ═══════════════════════════════════════════════════════════════════
        # TREND QUALITY
        # ═══════════════════════════════════════════════════════════════════
        
        elif feature_type == 'trend_r2':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            def calc_r2(series):
                y = series.values
                x = np.arange(len(y))
                if len(y) < 5:
                    return 0
                corr = np.corrcoef(x, y)[0, 1]
                return corr ** 2 if not np.isnan(corr) else 0
            return prices.iloc[-period:].apply(calc_r2)
        
        elif feature_type == 'trend_slope':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            def calc_slope(series):
                y = series.values
                x = np.arange(len(y))
                if len(y) < 5:
                    return 0
                slope = np.polyfit(x, y, 1)[0]
                return slope / series.mean() * period
            return prices.iloc[-period:].apply(calc_slope)
        
        elif feature_type == 'trend_deviation':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            def calc_deviation(series):
                y = series.values
                x = np.arange(len(y))
                if len(y) < 5:
                    return 0
                slope, intercept = np.polyfit(x, y, 1)
                trend_line = slope * x + intercept
                deviation = (y[-1] - trend_line[-1]) / series.mean()
                return deviation
            return prices.iloc[-period:].apply(calc_deviation)
        
        elif feature_type == 'hurst_proxy':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.5, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            def calc_hurst_proxy(col):
                if len(col.dropna()) < 20:
                    return 0.5
                r = col.cumsum()
                s = col.expanding().std()
                rs = (r.max() - r.min()) / s.iloc[-1] if s.iloc[-1] > 0 else 1
                h = np.log(rs) / np.log(len(col)) if rs > 0 else 0.5
                return min(max(h, 0), 1)
            return returns.apply(calc_hurst_proxy)

        # ═══════════════════════════════════════════════════════════════════
        # PRICE LEVEL
        # ═══════════════════════════════════════════════════════════════════
        
        elif feature_type == 'range_position':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.5, index=tickers)
            recent = prices.iloc[-period:]
            high = recent.max()
            low = recent.min()
            current = prices.iloc[-1]
            range_size = high - low
            position = (current - low) / range_size.replace(0, np.nan)
            return position.fillna(0.5)
        
        elif feature_type == 'high_proximity':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            recent = prices.iloc[-period:]
            high = recent.max()
            current = prices.iloc[-1]
            return current / high
        
        elif feature_type == 'breakout_strength':
            period = spec[1]
            if len(prices) < period + 1:
                return pd.Series(0.0, index=tickers)
            historical = prices.iloc[-(period + 1):-1]
            high = historical.max()
            current = prices.iloc[-1]
            breakout = (current / high) - 1
            return breakout.clip(0, 0.5)
        
        elif feature_type == 'support_distance':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            recent = prices.iloc[-period:]
            low = recent.min()
            current = prices.iloc[-1]
            return (current / low) - 1
        
        elif feature_type == 'resistance_distance':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            recent = prices.iloc[-period:]
            high = recent.max()
            current = prices.iloc[-1]
            return (high / current) - 1
        
        # ═══════════════════════════════════════════════════════════════════
        # CROSS-SECTIONAL
        # ═══════════════════════════════════════════════════════════════════
        
        elif feature_type == 'relative_strength':
            period = spec[1]
            if len(prices) < period + 1:
                return pd.Series(0.5, index=tickers)
            returns = (prices.iloc[-1] / prices.iloc[-period - 1]) - 1
            return returns.rank(pct=True)
        
        elif feature_type == 'relative_volatility':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.5, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            vol = returns.std()
            return vol.rank(pct=True)
        
        elif feature_type == 'excess_momentum':
            period = spec[1]
            if len(prices) < period + 1:
                return pd.Series(0.0, index=tickers)
            returns = (prices.iloc[-1] / prices.iloc[-period - 1]) - 1
            mean_return = returns.mean()
            return returns - mean_return
        
        elif feature_type == 'relative_value':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            ma = prices.iloc[-period:].mean()
            discount = prices.iloc[-1] / ma - 1
            return -(discount - discount.mean())
        
        # ═══════════════════════════════════════════════════════════════════
        # RISK-ADJUSTED
        # ═══════════════════════════════════════════════════════════════════
        
        elif feature_type == 'sharpe_ratio':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            mean_ret = returns.mean() * 252
            vol = returns.std() * np.sqrt(252)
            sharpe = mean_ret / vol.replace(0, np.nan)
            return sharpe.fillna(0).clip(-3, 3)
        
        elif feature_type == 'sortino_ratio':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            mean_ret = returns.mean() * 252
            downside = returns.clip(upper=0).std() * np.sqrt(252)
            sortino = mean_ret / downside.replace(0, np.nan)
            return sortino.fillna(0).clip(-5, 5)
        
        elif feature_type == 'calmar_ratio':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            total_return = (prices.iloc[-1] / prices.iloc[-period]) - 1
            ann_return = (1 + total_return) ** (252 / period) - 1
            recent = prices.iloc[-period:]
            running_max = recent.cummax()
            max_dd = ((recent - running_max) / running_max).min().abs()
            calmar = ann_return / max_dd.replace(0, np.nan)
            return calmar.fillna(0).clip(-5, 5)
        
        elif feature_type == 'information_ratio':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            market_returns = returns.mean(axis=1)
            excess_returns = returns.sub(market_returns, axis=0)
            mean_excess = excess_returns.mean()
            tracking_error = excess_returns.std()
            ir = mean_excess / tracking_error.replace(0, np.nan) * np.sqrt(252)
            return ir.fillna(0).clip(-3, 3)
        
        # ═══════════════════════════════════════════════════════════════════
        # VOLUME (optional)
        # ═══════════════════════════════════════════════════════════════════
        
        elif feature_type == 'volume_trend':
            if volume is None:
                return pd.Series(0.0, index=tickers)
            period = spec[1]
            if len(volume) < period:
                return pd.Series(0.0, index=tickers)
            recent = volume.iloc[-period:]
            first_half = recent.iloc[:period // 2].mean()
            second_half = recent.iloc[period // 2:].mean()
            trend = (second_half / first_half.replace(0, np.nan)) - 1
            return trend.fillna(0)
        
        elif feature_type == 'volume_ratio':
            if volume is None:
                return pd.Series(1.0, index=tickers)
            short, long = spec[1], spec[2]
            if len(volume) < long:
                return pd.Series(1.0, index=tickers)
            vol_short = volume.iloc[-short:].mean()
            vol_long = volume.iloc[-long:].mean()
            return (vol_short / vol_long.replace(0, np.nan)).fillna(1.0)
        
        elif feature_type == 'price_volume_corr':
            if volume is None:
                return pd.Series(0.0, index=tickers)
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            price_ret = prices.pct_change().iloc[-period:]
            vol_ret = volume.pct_change().iloc[-period:]
            def calc_corr(ticker):
                try:
                    return price_ret[ticker].corr(vol_ret[ticker])
                except:
                    return 0
            return pd.Series({t: calc_corr(t) for t in tickers})
        
        elif feature_type == 'volume_volatility':
            if volume is None:
                return pd.Series(0.0, index=tickers)
            period = spec[1]
            if len(volume) < period:
                return pd.Series(0.0, index=tickers)
            vol_pct = volume.pct_change().iloc[-period:]
            return vol_pct.std()

        # ═══════════════════════════════════════════════════════════════════
        # PATTERN
        # ═══════════════════════════════════════════════════════════════════
        
        elif feature_type == 'gap_reversal':
            period = spec[1]
            if len(prices) < period + 1:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            large_moves = returns.abs() > returns.std() * 2
            def gap_reversal_score(col):
                big_days = large_moves[col.name]
                if big_days.sum() == 0:
                    return 0
                reversals = 0
                total = 0
                for i, is_big in enumerate(big_days):
                    if is_big and i < len(col) - 1:
                        if np.sign(col.iloc[i]) != np.sign(col.iloc[i + 1]):
                            reversals += 1
                        total += 1
                return reversals / max(total, 1)
            return returns.apply(gap_reversal_score)
        
        elif feature_type == 'streak_strength':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            def calc_streak(col):
                sign = np.sign(col.iloc[-1])
                streak = 1
                for i in range(len(col) - 2, -1, -1):
                    if np.sign(col.iloc[i]) == sign:
                        streak += 1
                    else:
                        break
                return sign * streak / period
            return returns.apply(calc_streak)
        
        elif feature_type == 'return_autocorr':
            period, lag = spec[1], spec[2]
            if len(prices) < period + lag:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            def calc_autocorr(col):
                try:
                    ac = col.autocorr(lag=lag)
                    return ac if not np.isnan(ac) else 0
                except:
                    return 0
            return returns.apply(calc_autocorr)
        
        elif feature_type == 'abs_return_momentum':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            return returns.abs().mean() * np.sqrt(252)
        
        # ═══════════════════════════════════════════════════════════════════
        # SMART MONEY CONCEPTS (SMC)
        # ═══════════════════════════════════════════════════════════════════
        elif feature_type == 'smc':
            if self.smc_features is None:
                return pd.Series(0.0, index=tickers)
            feature_name = spec[1]
            results = {}
            for ticker in tickers:
                try:
                    ticker_prices = prices[ticker]
                    ticker_volume = volume[ticker] if volume is not None else pd.Series(0, index=ticker_prices.index)
                    smc_result = self.smc_features.calculate_all_features(
                        high=ticker_prices, low=ticker_prices,
                        close=ticker_prices, volume=ticker_volume
                    )
                    if feature_name in smc_result:
                        feat = smc_result[feature_name]
                        results[ticker] = float(feat.iloc[-1]) if hasattr(feat, 'iloc') and len(feat) > 0 else 0.0
                    else:
                        results[ticker] = 0.0
                except:
                    results[ticker] = 0.0
            return pd.Series(results)
        
        # ═══════════════════════════════════════════════════════════════════
        # SUPPORT/RESISTANCE
        # ═══════════════════════════════════════════════════════════════════
        elif feature_type == 'sr':
            if self.sr_features is None:
                return pd.Series(0.0, index=tickers)
            feature_name = spec[1]
            results = {}
            for ticker in tickers:
                try:
                    ticker_prices = prices[ticker]
                    ticker_volume = volume[ticker] if volume is not None else pd.Series(0, index=ticker_prices.index)
                    sr_result = self.sr_features.calculate_all_features(
                        high=ticker_prices, low=ticker_prices,
                        close=ticker_prices, volume=ticker_volume
                    )
                    if feature_name in sr_result:
                        feat = sr_result[feature_name]
                        results[ticker] = float(feat.iloc[-1]) if hasattr(feat, 'iloc') and len(feat) > 0 else 0.0
                    else:
                        results[ticker] = 0.0
                except:
                    results[ticker] = 0.0
            return pd.Series(results)
        
        # ═══════════════════════════════════════════════════════════════════
        # OIL-SPECIFIC FEATURES
        # ═══════════════════════════════════════════════════════════════════
        elif feature_type == 'oil':
            if self.oil_features is None:
                return pd.Series(0.0, index=tickers)
            feature_name = spec[1]
            if len(prices) == 0:
                return pd.Series(0.0, index=tickers)
            self._oil_market_data = None  # Initialize before checking
            if not hasattr(self, '_oil_market_data') or self._oil_market_data is None:
                try:
                    from evolution.oil_specific_features import OilMarketData
                    wti_proxy = prices['USO'] if 'USO' in prices.columns else prices.iloc[:, 0]
                    brent_proxy = prices['BNO'] if 'BNO' in prices.columns else wti_proxy
                    self._oil_market_data = OilMarketData(
                        wti_price=wti_proxy, brent_price=brent_proxy,
                        inventory=None, refinery_utilization=None,
                        gasoline_price=None, diesel_price=None,
                    )
                except Exception:
                    self._oil_market_data = None
            if self._oil_market_data is None:
                return pd.Series(0.0, index=tickers)
            results = {}
            for ticker in tickers:
                try:
                    if ticker in prices.columns:
                        ticker_features = self.oil_features.calculate_all_features(
                            stock_prices=prices[ticker],
                            oil_market_data=self._oil_market_data
                        )
                        if feature_name in ticker_features:
                            feat_val = ticker_features[feature_name]
                            results[ticker] = feat_val.iloc[-1] if hasattr(feat_val, 'iloc') and len(feat_val) > 0 else 0.0
                        else:
                            results[ticker] = 0.0
                    else:
                        results[ticker] = 0.0
                except Exception:
                    results[ticker] = 0.0
            return pd.Series(results)

        # ═══════════════════════════════════════════════════════════════════
        # MARKET-RELATIVE FEATURES (NEW)
        # ═══════════════════════════════════════════════════════════════════
        
        elif feature_type == 'market_beta':
            period = spec[1]
            if len(prices) < period + 1:
                return pd.Series(1.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            market_returns = returns.mean(axis=1)  # Equal-weight market proxy
            betas = {}
            for ticker in tickers:
                cov = returns[ticker].cov(market_returns)
                var = market_returns.var()
                betas[ticker] = cov / var if var > 0 else 1.0
            return pd.Series(betas)
        
        elif feature_type == 'beta_instability':
            period = spec[1]
            if len(prices) < period + 21:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-(period + 21):]
            market_returns = returns.mean(axis=1)
            # Rolling 21-day beta, then take std over the period
            rolling_betas = {}
            for ticker in tickers:
                betas_list = []
                for end in range(21, len(returns) + 1):
                    window_ret = returns[ticker].iloc[end-21:end]
                    window_mkt = market_returns.iloc[end-21:end]
                    cov = window_ret.cov(window_mkt)
                    var = window_mkt.var()
                    betas_list.append(cov / var if var > 0 else 1.0)
                rolling_betas[ticker] = np.std(betas_list) if betas_list else 0.0
            return pd.Series(rolling_betas)
        
        elif feature_type == 'idiosyncratic_vol':
            period = spec[1]
            if len(prices) < period + 1:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            market_returns = returns.mean(axis=1)
            idio_vols = {}
            for ticker in tickers:
                cov = returns[ticker].cov(market_returns)
                var = market_returns.var()
                beta = cov / var if var > 0 else 1.0
                residual = returns[ticker] - beta * market_returns
                idio_vols[ticker] = residual.std() * np.sqrt(252)
            return pd.Series(idio_vols)
        
        elif feature_type == 'sector_correlation':
            period = spec[1]
            if len(prices) < period + 1:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            market_returns = returns.mean(axis=1)
            corrs = {}
            for ticker in tickers:
                try:
                    corrs[ticker] = returns[ticker].corr(market_returns)
                except:
                    corrs[ticker] = 0.0
            return pd.Series(corrs).fillna(0)
        
        elif feature_type == 'universe_dispersion':
            period = spec[1]
            if len(prices) < period + 1:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            # Cross-sectional std of returns each day, then average
            daily_dispersion = returns.std(axis=1).mean()
            # Return same value for all tickers (market-level feature)
            return pd.Series(daily_dispersion, index=tickers)
        
        # ═══════════════════════════════════════════════════════════════════
        # MICROSTRUCTURE FEATURES (NEW)
        # ═══════════════════════════════════════════════════════════════════
        
        elif feature_type == 'amihud_illiquidity':
            period = spec[1]
            if volume is None or len(prices) < period:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            dollar_volume = (prices * volume).iloc[-period:]
            amihud = (returns.abs() / dollar_volume.replace(0, np.nan)).mean()
            return amihud.fillna(0)
        
        elif feature_type == 'roll_spread':
            period = spec[1]
            if len(prices) < period + 1:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            # Roll (1984) spread estimator: 2 * sqrt(-cov(r_t, r_{t-1}))
            spreads = {}
            for ticker in tickers:
                r = returns[ticker].dropna()
                if len(r) < 5:
                    spreads[ticker] = 0.0
                    continue
                cov = r.iloc[1:].values @ r.iloc[:-1].values / (len(r) - 1)
                spreads[ticker] = 2 * np.sqrt(-cov) if cov < 0 else 0.0
            return pd.Series(spreads)
        
        elif feature_type == 'kyle_lambda':
            period = spec[1]
            if volume is None or len(prices) < period:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            vol_data = volume.iloc[-period:]
            lambdas = {}
            for ticker in tickers:
                r = returns[ticker].dropna()
                v = vol_data[ticker].dropna()
                if len(r) < 5 or len(v) < 5:
                    lambdas[ticker] = 0.0
                    continue
                # Kyle's lambda ≈ |return| / volume
                signed_vol = np.sign(r) * v
                try:
                    cov = np.cov(r.values[:min(len(r), len(v))],
                                 signed_vol.values[:min(len(r), len(v))])
                    lambdas[ticker] = abs(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else 0.0
                except:
                    lambdas[ticker] = 0.0
            return pd.Series(lambdas)
        
        elif feature_type == 'zero_return_fraction':
            period = spec[1]
            if len(prices) < period + 1:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            zero_frac = (returns.abs() < 1e-8).mean()
            return zero_frac
        
        elif feature_type == 'turnover_rate':
            period = spec[1]
            if volume is None or len(prices) < period:
                return pd.Series(0.0, index=tickers)
            # Average daily volume / price as turnover proxy
            avg_vol = volume.iloc[-period:].mean()
            avg_price = prices.iloc[-period:].mean()
            turnover = avg_vol / avg_price.replace(0, np.nan)
            return turnover.fillna(0)
        
        # ═══════════════════════════════════════════════════════════════════
        # REGIME FEATURES (NEW)
        # ═══════════════════════════════════════════════════════════════════
        
        elif feature_type == 'volatility_regime':
            period = spec[1]
            if len(prices) < 252:
                return pd.Series(0.5, index=tickers)
            returns = prices.pct_change()
            current_vol = returns.iloc[-period:].std()
            historical_vol = returns.iloc[-252:].std()
            # Percentile of current vol vs 1-year history
            regime = current_vol / historical_vol.replace(0, np.nan)
            return regime.fillna(1.0).clip(0, 3)
        
        elif feature_type == 'adx_proxy':
            period = spec[1]
            if len(prices) < period + 1:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            # ADX proxy: ratio of |net move| to sum of |daily moves|
            net_move = returns.sum().abs()
            total_path = returns.abs().sum()
            adx = net_move / total_path.replace(0, np.nan)
            return adx.fillna(0).clip(0, 1)
        
        elif feature_type == 'correlation_spike':
            period = spec[1]
            if len(prices) < period + 63:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change()
            # Recent avg pairwise correlation vs longer-term
            recent_corr = returns.iloc[-period:].corr().values
            np.fill_diagonal(recent_corr, np.nan)
            avg_recent = np.nanmean(recent_corr)
            
            longer_corr = returns.iloc[-63:].corr().values
            np.fill_diagonal(longer_corr, np.nan)
            avg_longer = np.nanmean(longer_corr)
            
            spike = avg_recent - avg_longer
            return pd.Series(spike, index=tickers)
        
        elif feature_type == 'market_breadth':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.5, index=tickers)
            ma = prices.iloc[-period:].mean()
            current = prices.iloc[-1]
            above_ma = (current > ma).mean()
            return pd.Series(above_ma, index=tickers)
        
        # ═══════════════════════════════════════════════════════════════════
        # ENGINEERED FEATURES (NEW)
        # ═══════════════════════════════════════════════════════════════════
        
        elif feature_type == 'momentum_sharpe':
            period = spec[1]
            if len(prices) < period + 1:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            mean_ret = returns.mean()
            vol = returns.std()
            mom_sharpe = mean_ret / vol.replace(0, np.nan)
            return mom_sharpe.fillna(0).clip(-3, 3)
        
        elif feature_type == 'drawdown_adjusted_momentum':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            total_return = (prices.iloc[-1] / prices.iloc[-period]) - 1
            recent = prices.iloc[-period:]
            running_max = recent.cummax()
            max_dd = ((recent - running_max) / running_max).min().abs()
            dd_adj = total_return / max_dd.replace(0, np.nan)
            return dd_adj.fillna(0).clip(-5, 5)
        
        elif feature_type == 'vol_adjusted_momentum':
            period = spec[1]
            if len(prices) < period + 1:
                return pd.Series(0.0, index=tickers)
            total_return = (prices.iloc[-1] / prices.iloc[-period - 1]) - 1
            returns = prices.pct_change().iloc[-period:]
            vol = returns.std() * np.sqrt(252)
            vol_adj = total_return / vol.replace(0, np.nan)
            return vol_adj.fillna(0).clip(-5, 5)
        
        elif feature_type == 'efficiency_ratio':
            period = spec[1]
            if len(prices) < period + 1:
                return pd.Series(0.5, index=tickers)
            net_change = (prices.iloc[-1] - prices.iloc[-period - 1]).abs()
            daily_changes = prices.diff().iloc[-period:].abs().sum()
            efficiency = net_change / daily_changes.replace(0, np.nan)
            return efficiency.fillna(0.5).clip(0, 1)
        
        elif feature_type == 'fractal_dimension':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(1.5, index=tickers)
            def calc_fractal_dim(series):
                y = series.values
                if len(y) < 10:
                    return 1.5
                n = len(y)
                # Higuchi-inspired: compare path length at different scales
                path_1 = np.sum(np.abs(np.diff(y)))
                path_2 = np.sum(np.abs(y[2:] - y[:-2])) / 2
                if path_2 > 0 and path_1 > 0:
                    fd = 1 + np.log(path_1 / path_2) / np.log(2)
                    return min(max(fd, 1.0), 2.0)
                return 1.5
            return prices.iloc[-period:].apply(calc_fractal_dim)
        
        # Default
        return pd.Series(0.0, index=tickers)

