#!/usr/bin/env python3
"""
Test script to verify oil stock data availability and quality.
Run this before executing the full arena to catch data issues early.
"""

import sys
from datetime import datetime
from data.universe import get_oil_universe
from data.fetcher import DataFetcher

def test_oil_data():
    """Test oil stock data availability and quality."""
    print("🛢️  Testing Oil Stock Data")
    print("=" * 60)
    print()
    
    # Get oil universe
    tickers = get_oil_universe()
    print(f"📊 Testing {len(tickers)} tickers: {', '.join(tickers)}")
    print()
    
    # Initialize data fetcher
    start_date = "2025-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    print(f"📅 Date range: {start_date} to {end_date}")
    print()
    
    fetcher = DataFetcher()
    
    # Fetch data
    print("⏳ Fetching data from yfinance...")
    prices = fetcher.fetch(start_date, end_date, tickers, use_cache=False)
    print()
    
    # Analyze results
    print("📈 Data Quality Report:")
    print("-" * 60)
    
    if prices.empty:
        print("❌ ERROR: No data retrieved!")
        return False
    
    success = True
    for ticker in tickers:
        if ticker in prices.columns:
            data_points = prices[ticker].notna().sum()
            total_points = len(prices)
            coverage = (data_points / total_points) * 100
            
            if coverage >= 80:
                status = "✅"
            elif coverage >= 50:
                status = "⚠️"
                success = False
            else:
                status = "❌"
                success = False
            
            print(f"{status} {ticker:6s}: {data_points:4d}/{total_points:4d} days ({coverage:5.1f}% coverage)")
        else:
            print(f"❌ {ticker:6s}: NOT FOUND in data")
            success = False
    
    print()
    print("📊 Overall Statistics:")
    print(f"   Total trading days: {len(prices)}")
    print(f"   Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"   Missing data points: {prices.isna().sum().sum()}")
    print()
    
    if success:
        print("✅ All tickers have sufficient data (≥80% coverage)")
        print()
        print("🚀 Ready to run arena! Execute:")
        print("   bash run_oil_arena.sh")
        return True
    else:
        print("⚠️  Some tickers have insufficient data")
        print()
        print("Recommendations:")
        print("  1. Check if ticker symbols are correct")
        print("  2. Verify tickers have data for 2025")
        print("  3. Consider removing problematic tickers")
        print("  4. Try a different date range")
        return False

if __name__ == "__main__":
    try:
        success = test_oil_data()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
