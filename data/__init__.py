"""
Data module.
"""

from data.fetcher import DataFetcher, load_prices
from data.universe import (
    LONG_HISTORY_TICKERS,
    EXPANDED_UNIVERSE,
    SP500_FULL,
    SP400_MIDCAP,
    get_universe_for_period,
    get_sector_tickers,
    # Sector-specific lists
    SP500_TECHNOLOGY,
    SP500_HEALTHCARE,
    SP500_FINANCIALS,
    SP500_CONSUMER_DISCRETIONARY,
    SP500_CONSUMER_STAPLES,
    SP500_INDUSTRIALS,
    SP500_ENERGY,
    SP500_MATERIALS,
    SP500_UTILITIES,
    SP500_REAL_ESTATE,
    SP500_COMMUNICATION,
)

__all__ = [
    # Core
    "DataFetcher",
    "load_prices",
    
    # Universes
    "LONG_HISTORY_TICKERS",
    "FULL_UNIVERSE",
    "EXPANDED_UNIVERSE",
    "SP500_FULL",
    "SP400_MIDCAP",
    
    # Functions
    "get_universe_for_period",
    "get_sector_tickers",
    
    # Sectors
    "SP500_TECHNOLOGY",
    "SP500_HEALTHCARE",
    "SP500_FINANCIALS",
    "SP500_CONSUMER_DISCRETIONARY",
    "SP500_CONSUMER_STAPLES",
    "SP500_INDUSTRIALS",
    "SP500_ENERGY",
    "SP500_MATERIALS",
    "SP500_UTILITIES",
    "SP500_REAL_ESTATE",
    "SP500_COMMUNICATION",
]