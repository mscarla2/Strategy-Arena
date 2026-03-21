"""
Feature Pre-computation Cache (RC-5)

Replaces the placeholder _precompute_features() with a proper period-indexed
feature cache. Features are computed once for all rebalance dates across all
walk-forward periods, then shared across all strategy evaluations.

Expected speedup: 10-50× for population evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FeaturePrecomputeCache:
    """
    Pre-compute features for all rebalance dates across all walk-forward periods.
    
    This eliminates redundant feature computation during GP evolution:
    - Without cache: features computed per-strategy × per-period × per-rebalance-date
    - With cache: features computed ONCE per-rebalance-date, shared across all strategies
    
    Usage:
        cache = FeaturePrecomputeCache(feature_lib, prices, volume, periods)
        features = cache.get_features("2025-03-01")  # instant lookup
    """
    
    def __init__(
        self,
        feature_lib,  # FeatureLibrary instance
        prices: pd.DataFrame,
        volume: Optional[pd.DataFrame],
        periods: List[Tuple[str, str, str, str]],
        rebalance_frequency: int = 21,
        max_dates: int = 500,
    ):
        """
        Args:
            feature_lib: FeatureLibrary instance for computing features
            prices: Full price DataFrame (all tickers, all dates)
            volume: Full volume DataFrame (optional)
            periods: List of (train_start, train_end, test_start, test_end) tuples
            rebalance_frequency: Days between rebalances (default: 21 = monthly)
            max_dates: Maximum number of rebalance dates to cache (memory guard)
        """
        self.feature_lib = feature_lib
        self.prices = prices
        self.volume = volume
        self.rebalance_frequency = rebalance_frequency
        self.max_dates = max_dates
        
        # Cache: date_key -> {feature_name: pd.Series}
        self.cache: Dict[str, Dict[str, pd.Series]] = {}
        self._hits = 0
        self._misses = 0
        
        # Pre-compute all features
        self._precompute(periods)
    
    def _precompute(self, periods: List[Tuple[str, str, str, str]]):
        """Compute features for every rebalance date in every test period."""
        # Collect all unique rebalance dates across all periods
        all_rebalance_dates: Set[pd.Timestamp] = set()
        
        for _, _, test_start, test_end in periods:
            test_start_dt = pd.Timestamp(test_start)
            test_end_dt = pd.Timestamp(test_end)
            
            # Get test dates that exist in our price data
            mask = (self.prices.index >= test_start_dt) & (self.prices.index <= test_end_dt)
            test_dates = self.prices.index[mask]
            
            for i, date in enumerate(test_dates):
                if i % self.rebalance_frequency == 0:
                    all_rebalance_dates.add(date)
        
        # Sort and limit
        sorted_dates = sorted(all_rebalance_dates)
        if len(sorted_dates) > self.max_dates:
            logger.warning(
                f"Limiting feature cache from {len(sorted_dates)} to {self.max_dates} dates"
            )
            sorted_dates = sorted_dates[-self.max_dates:]  # Keep most recent
        
        total = len(sorted_dates)
        logger.info(f"Pre-computing features for {total} rebalance dates...")
        
        for idx, date in enumerate(sorted_dates):
            date_key = date.strftime("%Y-%m-%d")
            
            # Get prices up to this date (for lookback computation)
            prices_to_date = self.prices.loc[:date]
            volume_to_date = self.volume.loc[:date] if self.volume is not None else None
            
            if len(prices_to_date) < 30:
                # Not enough data for meaningful features
                continue
            
            try:
                features = self.feature_lib.compute_all(
                    prices_to_date, volume_to_date, lag=1, rank_transform=True
                )
                self.cache[date_key] = features
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.warning(f"Failed to compute features for {date_key}: {e}")
                continue
            
            # Progress logging every 20%
            if total > 10 and (idx + 1) % max(1, total // 5) == 0:
                logger.info(f"  Feature cache: {idx + 1}/{total} dates computed")
        
        logger.info(
            f"Feature cache ready: {len(self.cache)} dates cached, "
            f"~{len(self.cache) * len(self.feature_lib.feature_names)} feature vectors"
        )
    
    def get_features(self, date: str) -> Optional[Dict[str, pd.Series]]:
        """
        Get pre-computed features for a specific date.
        
        Args:
            date: Date string in YYYY-MM-DD format
            
        Returns:
            Dict of {feature_name: pd.Series} or None if not cached
        """
        result = self.cache.get(date)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result
    
    def get_features_for_date(self, date: pd.Timestamp) -> Optional[Dict[str, pd.Series]]:
        """
        Get pre-computed features for a Timestamp.
        Falls back to nearest cached date if exact match not found.
        
        Args:
            date: pd.Timestamp
            
        Returns:
            Dict of {feature_name: pd.Series} or None if not cached
        """
        date_key = date.strftime("%Y-%m-%d")
        result = self.cache.get(date_key)
        
        if result is not None:
            self._hits += 1
            return result
        
        # Try to find nearest cached date (within 5 trading days)
        for offset in range(1, 6):
            for direction in [-1, 1]:
                nearby = (date + pd.Timedelta(days=direction * offset)).strftime("%Y-%m-%d")
                result = self.cache.get(nearby)
                if result is not None:
                    self._hits += 1
                    return result
        
        self._misses += 1
        return None
    
    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    @property
    def stats(self) -> Dict:
        """Cache statistics."""
        return {
            'cached_dates': len(self.cache),
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self.hit_rate,
            'features_per_date': len(self.feature_lib.feature_names),
        }
    
    def __repr__(self) -> str:
        return (
            f"FeaturePrecomputeCache("
            f"dates={len(self.cache)}, "
            f"hits={self._hits}, "
            f"misses={self._misses}, "
            f"hit_rate={self.hit_rate:.1%})"
        )
