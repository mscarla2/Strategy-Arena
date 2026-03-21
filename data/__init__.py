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
    # Oil universe (RC-4)
    OIL_MICROCAP_STOCKS,
    OIL_FUTURES_PROXIES,
    OIL_FOCUSED_UNIVERSE,
    OIL_REFERENCE_PANEL,
    OIL_BENCHMARKS,
    OIL_TRADEABLE_UNIVERSE,
    OIL_FULL_DOWNLOAD_UNIVERSE,
    get_oil_universe,
    get_oil_tradeable_tickers,
    get_oil_reference_panel,
    get_oil_benchmarks,
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
    
    # Oil Universe (RC-4)
    "OIL_MICROCAP_STOCKS",
    "OIL_FUTURES_PROXIES",
    "OIL_FOCUSED_UNIVERSE",
    "OIL_REFERENCE_PANEL",
    "OIL_BENCHMARKS",
    "OIL_TRADEABLE_UNIVERSE",
    "OIL_FULL_DOWNLOAD_UNIVERSE",
    "get_oil_universe",
    "get_oil_tradeable_tickers",
    "get_oil_reference_panel",
    "get_oil_benchmarks",
]