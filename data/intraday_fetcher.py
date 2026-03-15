#!/usr/bin/env python3
"""
Intraday Data Fetcher for 5-Minute Candles

Supports multiple data providers:
- Polygon.io (recommended, requires API key)
- Alpaca (free tier available)
- yfinance (fallback, limited intraday history)

Includes quality filters for microcap data:
- Volume filtering (remove low-volume bars)
- Spread filtering (remove wide-spread bars)
- Time filtering (regular trading hours only)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
from typing import List, Optional, Dict
import warnings

warnings.filterwarnings('ignore')


class IntradayDataFetcher:
    """
    Fetch and cache 5-minute candle data for microcaps.
    
    Features:
    - Multiple data provider support
    - Automatic caching
    - Quality filtering
    - Gap detection and handling
    """
    
    def __init__(self, provider='yfinance', api_key=None, cache_dir='data/cache/intraday'):
        """
        Initialize intraday data fetcher.
        
        Args:
            provider: Data provider ('polygon', 'alpaca', 'yfinance')
            api_key: API key for provider (required for polygon/alpaca)
            cache_dir: Directory for caching data
        """
        self.provider = provider
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize provider
        if provider == 'polygon':
            self._init_polygon()
        elif provider == 'alpaca':
            self._init_alpaca()
        elif provider == 'yfinance':
            self._init_yfinance()
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _init_polygon(self):
        """Initialize Polygon.io client."""
        if not self.api_key:
            raise ValueError("Polygon.io requires API key")
        
        try:
            from polygon import RESTClient
            self.client = RESTClient(self.api_key)
            print(f"✓ Initialized Polygon.io client")
        except ImportError:
            raise ImportError("Install polygon-api-client: pip install polygon-api-client")
    
    def _init_alpaca(self):
        """Initialize Alpaca client."""
        if not self.api_key:
            raise ValueError("Alpaca requires API key")
        
        try:
            from alpaca_trade_api import REST
            # API key format: "KEY_ID:SECRET_KEY"
            key_id, secret_key = self.api_key.split(':')
            self.client = REST(key_id, secret_key, base_url='https://paper-api.alpaca.markets')
            print(f"✓ Initialized Alpaca client")
        except ImportError:
            raise ImportError("Install alpaca-trade-api: pip install alpaca-trade-api")
    
    def _init_yfinance(self):
        """Initialize yfinance (no API key needed)."""
        try:
            import yfinance as yf
            self.yf = yf
            print(f"✓ Initialized yfinance (fallback provider)")
            print(f"  Note: yfinance has limited intraday history (7-60 days)")
        except ImportError:
            raise ImportError("Install yfinance: pip install yfinance")
    
    def fetch(self, ticker: str, start_date: str, end_date: str, 
              interval: str = '5min', use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch intraday data for a ticker.
        
        Args:
            ticker: Stock ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Candle interval ('1min', '5min', '15min', '30min', '1h')
            use_cache: Use cached data if available
        
        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        # Check cache
        if use_cache:
            cached_data = self._load_from_cache(ticker, start_date, end_date, interval)
            if cached_data is not None:
                print(f"  Loaded {ticker} from cache: {len(cached_data)} bars")
                return cached_data
        
        # Fetch from provider
        print(f"  Fetching {ticker} from {self.provider}...")
        
        if self.provider == 'polygon':
            data = self._fetch_polygon(ticker, start_date, end_date, interval)
        elif self.provider == 'alpaca':
            data = self._fetch_alpaca(ticker, start_date, end_date, interval)
        elif self.provider == 'yfinance':
            data = self._fetch_yfinance(ticker, start_date, end_date, interval)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        if data is None or len(data) == 0:
            print(f"  ⚠️  No data returned for {ticker}")
            return pd.DataFrame()
        
        # Apply quality filters
        data = self._apply_quality_filters(data)
        
        # Cache data
        if use_cache:
            self._save_to_cache(ticker, start_date, end_date, interval, data)
        
        print(f"  ✓ Fetched {ticker}: {len(data)} bars")
        return data
    
    def _fetch_polygon(self, ticker: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """Fetch data from Polygon.io."""
        # Map interval to Polygon format
        interval_map = {
            '1min': (1, 'minute'),
            '5min': (5, 'minute'),
            '15min': (15, 'minute'),
            '30min': (30, 'minute'),
            '1h': (1, 'hour')
        }
        
        if interval not in interval_map:
            raise ValueError(f"Unsupported interval: {interval}")
        
        multiplier, timespan = interval_map[interval]
        
        try:
            # Fetch aggregates
            aggs = self.client.get_aggs(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_=start_date,
                to=end_date,
                limit=50000
            )
            
            if not aggs:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = pd.DataFrame([{
                'timestamp': pd.Timestamp(a.timestamp, unit='ms', tz='America/New_York'),
                'open': a.open,
                'high': a.high,
                'low': a.low,
                'close': a.close,
                'volume': a.volume
            } for a in aggs])
            
            data.set_index('timestamp', inplace=True)
            return data
            
        except Exception as e:
            print(f"  ⚠️  Polygon error: {e}")
            return pd.DataFrame()
    
    def _fetch_alpaca(self, ticker: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """Fetch data from Alpaca."""
        # Map interval to Alpaca format
        interval_map = {
            '1min': '1Min',
            '5min': '5Min',
            '15min': '15Min',
            '30min': '30Min',
            '1h': '1Hour'
        }
        
        if interval not in interval_map:
            raise ValueError(f"Unsupported interval: {interval}")
        
        timeframe = interval_map[interval]
        
        try:
            # Fetch bars
            bars = self.client.get_bars(
                ticker,
                timeframe,
                start=start_date,
                end=end_date,
                limit=10000
            ).df
            
            if bars is None or len(bars) == 0:
                return pd.DataFrame()
            
            # Rename columns
            bars = bars.rename(columns={
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            })
            
            # Convert timezone
            bars.index = bars.index.tz_convert('America/New_York')
            
            return bars[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            print(f"  ⚠️  Alpaca error: {e}")
            return pd.DataFrame()
    
    def _fetch_yfinance(self, ticker: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """Fetch data from yfinance."""
        # Map interval to yfinance format
        interval_map = {
            '1min': '1m',
            '5min': '5m',
            '15min': '15m',
            '30min': '30m',
            '1h': '1h'
        }
        
        if interval not in interval_map:
            raise ValueError(f"Unsupported interval: {interval}")
        
        yf_interval = interval_map[interval]
        
        try:
            # Fetch data
            data = self.yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=yf_interval,
                progress=False
            )
            
            if data is None or len(data) == 0:
                return pd.DataFrame()
            
            # Rename columns (yfinance uses title case)
            data.columns = [col.lower() for col in data.columns]
            
            # Convert timezone if needed
            if data.index.tz is None:
                data.index = data.index.tz_localize('America/New_York')
            else:
                data.index = data.index.tz_convert('America/New_York')
            
            return data[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            print(f"  ⚠️  yfinance error: {e}")
            return pd.DataFrame()
    
    def _apply_quality_filters(self, data: pd.DataFrame, 
                               min_volume: int = 100,
                               max_spread_bps: float = 200,
                               regular_hours_only: bool = True) -> pd.DataFrame:
        """
        Apply quality filters to intraday data.
        
        Args:
            min_volume: Minimum volume per bar (shares)
            max_spread_bps: Maximum bid-ask spread (basis points)
            regular_hours_only: Keep only regular trading hours (9:30-16:00 ET)
        
        Returns:
            Filtered DataFrame
        """
        if len(data) == 0:
            return data
        
        original_len = len(data)
        
        # Filter low volume bars
        data = data[data['volume'] >= min_volume]
        
        # Filter wide spreads (estimate from high-low range)
        spread_bps = ((data['high'] - data['low']) / data['close']) * 10000
        data = data[spread_bps <= max_spread_bps]
        
        # Filter to regular trading hours
        if regular_hours_only:
            data = data.between_time('09:30', '16:00')
        
        filtered_len = len(data)
        if filtered_len < original_len:
            pct_removed = (1 - filtered_len / original_len) * 100
            print(f"    Filtered {original_len - filtered_len} bars ({pct_removed:.1f}%)")
        
        return data
    
    def _load_from_cache(self, ticker: str, start_date: str, end_date: str, interval: str) -> Optional[pd.DataFrame]:
        """Load data from cache."""
        cache_file = self.cache_dir / f"{ticker}_{interval}_{start_date}_{end_date}.parquet"
        
        if cache_file.exists():
            try:
                data = pd.read_parquet(cache_file)
                return data
            except Exception as e:
                print(f"  ⚠️  Cache read error: {e}")
                return None
        
        return None
    
    def _save_to_cache(self, ticker: str, start_date: str, end_date: str, interval: str, data: pd.DataFrame):
        """Save data to cache."""
        cache_file = self.cache_dir / f"{ticker}_{interval}_{start_date}_{end_date}.parquet"
        
        try:
            data.to_parquet(cache_file)
        except Exception as e:
            print(f"  ⚠️  Cache write error: {e}")
    
    def fetch_multiple(self, tickers: List[str], start_date: str, end_date: str,
                      interval: str = '5min', use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch intraday data for multiple tickers.
        
        Args:
            tickers: List of stock tickers
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Candle interval
            use_cache: Use cached data if available
        
        Returns:
            Dictionary of {ticker: DataFrame}
        """
        print(f"📊 Fetching intraday data for {len(tickers)} tickers...")
        print(f"  Provider: {self.provider}")
        print(f"  Interval: {interval}")
        print(f"  Date range: {start_date} to {end_date}")
        
        data_dict = {}
        
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{len(tickers)}] {ticker}")
            
            data = self.fetch(ticker, start_date, end_date, interval, use_cache)
            
            if len(data) > 0:
                data_dict[ticker] = data
            
            # Rate limiting (for API providers)
            if self.provider in ['polygon', 'alpaca'] and i < len(tickers):
                time.sleep(0.2)  # 5 requests per second
        
        print(f"✓ Fetched {len(data_dict)}/{len(tickers)} tickers successfully")
        
        return data_dict
    
    def get_aligned_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Align multiple ticker data to common timestamps.
        
        Args:
            data_dict: Dictionary of {ticker: DataFrame}
        
        Returns:
            DataFrame with MultiIndex columns (ticker, field)
        """
        if not data_dict:
            return pd.DataFrame()
        
        # Combine all data
        combined = pd.DataFrame()
        
        for ticker, data in data_dict.items():
            for col in ['open', 'high', 'low', 'close', 'volume']:
                combined[(ticker, col)] = data[col]
        
        # Forward fill missing values (up to 3 bars)
        combined = combined.fillna(method='ffill', limit=3)
        
        # Drop rows with any remaining NaN
        combined = combined.dropna()
        
        print(f"  Aligned data: {len(combined)} common timestamps")
        
        return combined


def test_intraday_fetcher():
    """Test the intraday data fetcher."""
    print("=" * 80)
    print("INTRADAY DATA FETCHER TEST")
    print("=" * 80)
    
    # Test with yfinance (no API key needed)
    fetcher = IntradayDataFetcher(provider='yfinance')
    
    # Fetch recent data (yfinance has limited history)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    # Test single ticker
    print("\n1. Testing single ticker fetch...")
    data = fetcher.fetch('AAPL', start_date, end_date, interval='5min')
    
    if len(data) > 0:
        print(f"\n✓ Successfully fetched {len(data)} bars")
        print(f"\nFirst 5 bars:")
        print(data.head())
        print(f"\nLast 5 bars:")
        print(data.tail())
    else:
        print("\n⚠️  No data returned")
    
    # Test multiple tickers
    print("\n2. Testing multiple ticker fetch...")
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    data_dict = fetcher.fetch_multiple(tickers, start_date, end_date, interval='5min')
    
    print(f"\n✓ Fetched {len(data_dict)} tickers")
    for ticker, data in data_dict.items():
        print(f"  {ticker}: {len(data)} bars")
    
    # Test aligned data
    print("\n3. Testing data alignment...")
    aligned = fetcher.get_aligned_data(data_dict)
    
    if len(aligned) > 0:
        print(f"\n✓ Aligned data: {len(aligned)} common timestamps")
        print(f"\nColumns: {aligned.columns.tolist()[:6]}...")
        print(f"\nFirst 3 rows:")
        print(aligned.head(3))
    
    print("\n" + "=" * 80)
    print("✓ Test complete!")
    print("=" * 80)


if __name__ == '__main__':
    test_intraday_fetcher()
