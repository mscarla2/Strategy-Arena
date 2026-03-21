"""
Feature Pre-computation Cache (RC-5)

Replaces the placeholder _precompute_features() with a proper period-indexed
feature cache. Features are computed once for all rebalance dates across all
walk-forward periods, then shared across all strategy evaluations.

Disk persistence: cached features are stored as parquet files in
data/cache/features/ so they survive across runs. On startup, existing
cached dates are loaded from disk; only missing dates are recomputed.

Expected speedup: 10-50× for population evaluation.
"""

import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Default disk cache directory
_DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "features"


def _universe_fingerprint(tickers: List[str]) -> str:
    """Short hash of sorted ticker list — changes when the universe changes."""
    key = ",".join(sorted(tickers))
    return hashlib.md5(key.encode()).hexdigest()[:8]


class FeaturePrecomputeCache:
    """
    Pre-compute features for all rebalance dates across all walk-forward periods.

    This eliminates redundant feature computation during GP evolution:
    - Without cache: features computed per-strategy × per-period × per-rebalance-date
    - With cache: features computed ONCE per-rebalance-date, shared across all strategies

    Disk persistence:
        Each (date, universe) pair is stored as a parquet file:
            data/cache/features/{universe_hash}/{YYYY-MM-DD}.parquet
        On the next run with the same universe, cached dates are loaded
        from disk in milliseconds instead of being recomputed.

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
        cache_dir: Optional[Path] = None,
        persist_to_disk: bool = True,
    ):
        """
        Args:
            feature_lib: FeatureLibrary instance for computing features
            prices: Full price DataFrame (all tickers, all dates)
            volume: Full volume DataFrame (optional)
            periods: List of (train_start, train_end, test_start, test_end) tuples
            rebalance_frequency: Days between rebalances (default: 21 = monthly)
            max_dates: Maximum number of rebalance dates to cache (memory guard)
            cache_dir: Directory for parquet files (default: data/cache/features/)
            persist_to_disk: Whether to save/load from disk (default: True)
        """
        self.feature_lib = feature_lib
        self.prices = prices
        self.volume = volume
        self.rebalance_frequency = rebalance_frequency
        self.max_dates = max_dates
        self.persist_to_disk = persist_to_disk

        # Disk cache directory, namespaced by universe fingerprint
        self._universe_hash = _universe_fingerprint(list(prices.columns))
        if cache_dir is None:
            cache_dir = _DEFAULT_CACHE_DIR
        self._cache_dir = Path(cache_dir) / self._universe_hash
        if self.persist_to_disk:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache: date_key -> {feature_name: pd.Series}
        self.cache: Dict[str, Dict[str, pd.Series]] = {}
        self._hits = 0
        self._misses = 0
        self._disk_loads = 0
        self._disk_saves = 0

        # Pre-compute (loads from disk where possible, computes the rest)
        self._precompute(periods)

    # ──────────────────────────────────────────────────────────────────────
    # Disk I/O helpers
    # ──────────────────────────────────────────────────────────────────────

    def _parquet_path(self, date_key: str) -> Path:
        """Return the parquet file path for a given date key."""
        return self._cache_dir / f"{date_key}.parquet"

    def _load_from_disk(self, date_key: str) -> Optional[Dict[str, pd.Series]]:
        """Try to load a cached date from disk. Returns None on miss."""
        if not self.persist_to_disk:
            return None
        path = self._parquet_path(date_key)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            # Each column is a feature, index is tickers
            features = {col: df[col] for col in df.columns}
            self._disk_loads += 1
            return features
        except Exception as e:
            logger.debug(f"Failed to load {path}: {e}")
            return None

    def _save_to_disk(self, date_key: str, features: Dict[str, pd.Series]):
        """Persist a date's features to disk as parquet."""
        if not self.persist_to_disk:
            return
        try:
            df = pd.DataFrame(features)
            path = self._parquet_path(date_key)
            df.to_parquet(path, engine="pyarrow", compression="snappy")
            self._disk_saves += 1
        except Exception as e:
            logger.debug(f"Failed to save {date_key} to disk: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Core pre-computation
    # ──────────────────────────────────────────────────────────────────────

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
        logger.info(f"Feature cache: {total} rebalance dates needed")

        # Phase 1 — load what we can from disk
        dates_to_compute: List[pd.Timestamp] = []
        for date in sorted_dates:
            date_key = date.strftime("%Y-%m-%d")
            disk_features = self._load_from_disk(date_key)
            if disk_features is not None:
                self.cache[date_key] = disk_features
            else:
                dates_to_compute.append(date)

        if self._disk_loads > 0:
            logger.info(
                f"  Loaded {self._disk_loads}/{total} dates from disk cache"
            )

        # Phase 2 — compute the rest
        n_compute = len(dates_to_compute)
        if n_compute > 0:
            logger.info(f"  Computing {n_compute} new dates...")

        for idx, date in enumerate(dates_to_compute):
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
                # Persist to disk for next run
                self._save_to_disk(date_key, features)
            except Exception as e:
                logger.warning(f"Failed to compute features for {date_key}: {e}")
                continue

            # Progress logging every 20%
            if n_compute > 10 and (idx + 1) % max(1, n_compute // 5) == 0:
                logger.info(f"    Feature cache: {idx + 1}/{n_compute} new dates computed")

        logger.info(
            f"Feature cache ready: {len(self.cache)} dates "
            f"({self._disk_loads} from disk, {self._disk_saves} newly saved)"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Lookup API
    # ──────────────────────────────────────────────────────────────────────

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

    # ──────────────────────────────────────────────────────────────────────
    # Cache management
    # ──────────────────────────────────────────────────────────────────────

    def clear_disk_cache(self):
        """Remove all parquet files for this universe from disk."""
        if self._cache_dir.exists():
            for f in self._cache_dir.glob("*.parquet"):
                f.unlink()
            logger.info(f"Cleared disk cache at {self._cache_dir}")

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
            'disk_loads': self._disk_loads,
            'disk_saves': self._disk_saves,
            'cache_dir': str(self._cache_dir),
        }

    def __repr__(self) -> str:
        return (
            f"FeaturePrecomputeCache("
            f"dates={len(self.cache)}, "
            f"hits={self._hits}, "
            f"misses={self._misses}, "
            f"hit_rate={self.hit_rate:.1%}, "
            f"disk_loads={self._disk_loads}, "
            f"disk_saves={self._disk_saves})"
        )
