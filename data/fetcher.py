"""
Data fetching and caching with robust error handling.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import List, Optional
import time
import hashlib
import random

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR, MIN_TICKERS_REQUIRED


class DataFetcher:
    """Fetches and caches historical price data."""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or DATA_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Master cache file - stores all downloaded data
        self.master_cache = self.cache_dir / "prices_master.parquet"
    
    def _cache_path(self, start: str, end: str, tickers: List[str]) -> Path:
        """Generate cache file path."""
        key = f"{start}_{end}_{'_'.join(sorted(tickers)[:10])}"
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return self.cache_dir / f"prices_{h}.parquet"
    
    def _load_master_cache(self) -> Optional[pd.DataFrame]:
        """Load master cache if it exists."""
        if self.master_cache.exists():
            try:
                df = pd.read_parquet(self.master_cache)
                if not df.empty:
                    return df
            except Exception:
                pass
        return None
    
    def _save_master_cache(self, df: pd.DataFrame):
        """Save or update master cache."""
        try:
            existing = self._load_master_cache()
            
            if existing is not None:
                # Merge: add new columns and extend date range
                all_cols = list(set(existing.columns) | set(df.columns))
                
                # Combine indices
                all_dates = existing.index.union(df.index).sort_values()
                
                # Create merged dataframe
                merged = pd.DataFrame(index=all_dates, columns=all_cols)
                
                # Fill with existing data first
                for col in existing.columns:
                    merged.loc[existing.index, col] = existing[col]
                
                # Overlay with new data (overwrites if same dates)
                for col in df.columns:
                    merged.loc[df.index, col] = df[col]
                
                df = merged
            
            df.to_parquet(self.master_cache)
        except Exception as e:
            print(f"  Warning: Could not save master cache: {e}")
    
    def _download_with_retry(self, 
                              tickers: List[str], 
                              start_date: str, 
                              end_date: str,
                              max_retries: int = 3,
                              base_delay: float = 2.0) -> pd.DataFrame:
        """Download with exponential backoff retry."""
        
        all_data = {}
        failed_tickers = []
        
        # Try bulk download first
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"  Retry {attempt + 1}/{max_retries} after {delay:.1f}s delay...")
                    time.sleep(delay)
                
                data = yf.download(
                    tickers,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True,
                    threads=True,
                    timeout=30
                )
                
                if data.empty:
                    continue
                
                # Parse the response
                if len(tickers) == 1:
                    if 'Close' in data.columns:
                        all_data[tickers[0]] = data['Close']
                    elif not data.empty:
                        all_data[tickers[0]] = data.iloc[:, 0]
                
                elif isinstance(data.columns, pd.MultiIndex):
                    if 'Close' in data.columns.get_level_values(0):
                        close_data = data['Close']
                    else:
                        close_data = data.iloc[:, data.columns.get_level_values(0) == data.columns.get_level_values(0)[0]]
                        if isinstance(close_data.columns, pd.MultiIndex):
                            close_data.columns = close_data.columns.get_level_values(1)
                    
                    for col in close_data.columns:
                        if close_data[col].notna().sum() > 0:
                            all_data[col] = close_data[col]
                else:
                    for col in data.columns:
                        if data[col].notna().sum() > 0:
                            all_data[col] = data[col]
                
                if all_data:
                    break
                    
            except Exception as e:
                print(f"  Bulk download attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    failed_tickers = tickers
        
        # Fallback: individual downloads for failed tickers
        if not all_data or len(all_data) < len(tickers) * 0.5:
            remaining = [t for t in tickers if t not in all_data]
            print(f"  Trying individual downloads for {len(remaining)} tickers...")
            
            for i, ticker in enumerate(remaining):
                try:
                    # Rate limiting with jitter
                    time.sleep(0.3 + random.uniform(0, 0.2))
                    
                    stock = yf.Ticker(ticker)
                    hist = stock.history(start=start_date, end=end_date, auto_adjust=True)
                    
                    if not hist.empty and 'Close' in hist.columns:
                        all_data[ticker] = hist['Close']
                        
                    if (i + 1) % 10 == 0:
                        print(f"    Downloaded {i + 1}/{len(remaining)}...")
                        
                except Exception:
                    failed_tickers.append(ticker)
                    continue
        
        if not all_data:
            raise ValueError("No data could be downloaded")
        
        return pd.DataFrame(all_data)
    
    def fetch(self,
              start_date: str,
              end_date: str,
              tickers: List[str] = None,
              min_data_pct: float = 0.8,
              use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch adjusted close prices.
        
        Args:
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD
            tickers: List of tickers (None = auto based on date)
            min_data_pct: Minimum data coverage required
            use_cache: Whether to use cached data
        
        Returns:
            DataFrame: dates as index, tickers as columns
        """
        from data.universe import get_universe_for_period
        
        if tickers is None:
            tickers = get_universe_for_period(start_date)
        
        # Remove known problematic tickers
        tickers = [t for t in tickers if t not in ['DOW', 'BRK.B', 'BF.B']]
        
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        
        # Try master cache first
        if use_cache:
            master = self._load_master_cache()
            if master is not None:
                # Check if cache covers our date range
                cache_start = master.index.min()
                cache_end = master.index.max()
                
                # Find tickers available in cache
                available = [t for t in tickers if t in master.columns]
                
                if (len(available) >= MIN_TICKERS_REQUIRED and
                    cache_start <= start_dt and
                    cache_end >= end_dt - pd.Timedelta(days=7)):  # Allow 7 day buffer for weekends
                    
                    # Use cached data
                    df = master.loc[start_dt:end_dt, available].copy()
                    df = df.ffill(limit=5).dropna(how='all')
                    
                    if len(df) > 100 and len(df.columns) >= MIN_TICKERS_REQUIRED:
                        print(f"  Loaded from cache: {len(df)} days, {len(df.columns)} tickers")
                        return df
                    else:
                        print(f"  Cache insufficient, downloading fresh data...")
                else:
                    missing_tickers = len(tickers) - len(available)
                    print(f"  Cache missing {missing_tickers} tickers or date range, downloading...")
        
        # Download fresh data
        print(f"  Downloading {len(tickers)} tickers: {start_date} to {end_date}...")
        
        df = self._download_with_retry(tickers, start_date, end_date)
        
        # Filter by data coverage
        min_rows = len(df) * min_data_pct
        valid_cols = [c for c in df.columns if df[c].notna().sum() >= min_rows]
        
        if len(valid_cols) < MIN_TICKERS_REQUIRED:
            min_rows = len(df) * 0.5
            valid_cols = [c for c in df.columns if df[c].notna().sum() >= min_rows]
        
        if len(valid_cols) < MIN_TICKERS_REQUIRED:
            raise ValueError(
                f"Only {len(valid_cols)} tickers have sufficient data "
                f"(need {MIN_TICKERS_REQUIRED})"
            )
        
        df = df[valid_cols]
        df = df.ffill(limit=5)
        df = df.dropna(how='all')
        
        # Save to master cache
        self._save_master_cache(df)
        
        print(f"  Loaded {len(df)} days, {len(df.columns)} tickers")
        return df
    
    def clear_cache(self):
        """Clear cached data."""
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()
        print("Cache cleared")


# Convenience function
def load_prices(start_date: str, end_date: str, tickers: List[str] = None) -> pd.DataFrame:
    """Load price data."""
    return DataFetcher().fetch(start_date, end_date, tickers)