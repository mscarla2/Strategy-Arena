"""
Ticker universe management with historical availability.
Expanded to 500+ stocks for broader alpha opportunity.
"""

from typing import List

# ═══════════════════════════════════════════════════════════════════════════════
# S&P 500 COMPONENTS (as of 2024, grouped by sector)
# ═══════════════════════════════════════════════════════════════════════════════

SP500_TECHNOLOGY = [
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "CSCO", "ACN",
    "IBM", "INTC", "INTU", "QCOM", "TXN", "AMAT", "NOW", "ADI", "LRCX", "MU",
    "KLAC", "SNPS", "CDNS", "MCHP", "MSI", "HPQ", "FTNT", "KEYS", "ANSS", "GLW",
    "NXPI", "MPWR", "SWKS", "ZBRA", "TER", "AKAM", "JNPR", "FFIV", "NTAP", "WDC",
    "HPE", "ENPH", "SEDG", "CTSH", "IT", "EPAM", "PAYC", "PAYX", "FSLR", "GEN",
    "TRMB", "TYL", "VRSN", "PTC", "LDOS", "JKHY",
]

SP500_HEALTHCARE = [
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "AMGN",
    "BMY", "MDT", "GILD", "CVS", "ELV", "ISRG", "VRTX", "SYK", "CI", "REGN",
    "BSX", "ZTS", "BDX", "HUM", "MCK", "EW", "HCA", "IDXX", "DXCM", "MTD",
    "IQV", "A", "BIIB", "RMD", "ILMN", "ALGN", "CAH", "BAX", "HOLX", "CNC",
    "TECH", "VTRS", "MOH", "HSIC", "CRL", "DGX", "LH", "INCY", "PODD", "COO",
    "XRAY", "BIO", "PKI", "WAT", "STE", "RVTY",
]

SP500_FINANCIALS = [
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI", "BLK",
    "C", "AXP", "PGR", "SCHW", "MMC", "CB", "ICE", "CME", "AON", "PNC",
    "USB", "TFC", "AJG", "MCO", "MET", "AIG", "MSCI", "AFL", "PRU", "TROW",
    "AMP", "TRV", "ALL", "HIG", "NDAQ", "BK", "COF", "DFS", "STT", "CINF",
    "FRC", "FITB", "MTB", "RF", "CFG", "HBAN", "NTRS", "KEY", "WRB", "RJF",
    "L", "BRO", "SIVB", "ZION", "CMA", "FDS", "CBOE", "MKTX", "IVZ", "PBCT",
    "GL", "RE", "AIZ", "LNC", "BEN",
]

SP500_CONSUMER_DISCRETIONARY = [
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "CMG",
    "ORLY", "MAR", "AZO", "ROST", "HLT", "DHI", "YUM", "GM", "F", "EBAY",
    "APTV", "LEN", "ULTA", "DRI", "GRMN", "PHM", "TSCO", "BBY", "POOL", "NVR",
    "CCL", "RCL", "WYNN", "LVS", "MGM", "HAS", "DG", "DLTR", "GPC", "EXPE",
    "ETSY", "LULU", "CBRE", "DECK", "TPR", "VFC", "WHR", "BWA", "LEG", "PVH",
    "RL", "HBI", "NCLH", "AAP", "MHK", "NWL",
]

SP500_CONSUMER_STAPLES = [
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "EL",
    "KMB", "GIS", "STZ", "SYY", "ADM", "KHC", "HSY", "MKC", "K", "CLX",
    "KR", "WBA", "TSN", "CAG", "CHD", "MNST", "HRL", "SJM", "CPB", "TAP",
    "BG", "LW", "KDP",
]

SP500_INDUSTRIALS = [
    "CAT", "GE", "HON", "UNP", "UPS", "RTX", "BA", "DE", "LMT", "ADP",
    "MMM", "GD", "ITW", "NOC", "CSX", "NSC", "WM", "ETN", "PH", "EMR",
    "FDX", "CTAS", "JCI", "PCAR", "TT", "CARR", "ODFL", "AME", "RSG", "CPRT",
    "FAST", "ROK", "CMI", "VRSK", "GWW", "IR", "XYL", "DOV", "OTIS", "WAB",
    "LHX", "SWK", "DAL", "EXPD", "ROP", "IEX", "J", "CHRW", "HWM", "TDG",
    "FTV", "ALLE", "NLOK", "MAS", "JBHT", "PWR", "URI", "GNRC", "NDSN", "EFX",
    "PNR", "HII", "AAL", "UAL", "LUV", "LDOS", "PAYC", "SNA", "RHI",
]

SP500_ENERGY = [
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PXD", "PSX", "VLO", "OXY",
    "WMB", "KMI", "HAL", "DVN", "HES", "BKR", "FANG", "CTRA", "OKE", "TRGP",
    "MRO", "APA",
]

SP500_MATERIALS = [
    "LIN", "APD", "SHW", "FCX", "ECL", "NEM", "NUE", "DD", "DOW", "CTVA",
    "PPG", "VMC", "MLM", "ALB", "IFF", "BALL", "PKG", "AVY", "CE", "CF",
    "FMC", "EMN", "MOS", "IP", "SEE", "WRK", "AMCR",
]

SP500_UTILITIES = [
    "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "PEG",
    "WEC", "ES", "AWK", "DTE", "EIX", "FE", "ETR", "PPL", "AEE", "CMS",
    "CNP", "EVRG", "ATO", "NI", "PNW", "NRG", "LNT", "AES",
]

SP500_REAL_ESTATE = [
    "AMT", "PLD", "CCI", "EQIX", "PSA", "O", "WELL", "DLR", "SBAC", "SPG",
    "AVB", "EQR", "VICI", "VTR", "ARE", "MAA", "WY", "ESS", "INVH", "EXR",
    "UDR", "PEAK", "KIM", "BXP", "CPT", "HST", "REG", "FRT", "IRM", "CBRE",
]

SP500_COMMUNICATION = [
    "GOOGL", "GOOG", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS", "CHTR",
    "ATVI", "EA", "TTWO", "WBD", "OMC", "IPG", "PARA", "LYV", "FOXA", "FOX",
    "NWS", "NWSA", "DISH", "LUMN", "MTCH",
]

# ═══════════════════════════════════════════════════════════════════════════════
# S&P MIDCAP 400 - SELECTED HIGH LIQUIDITY
# ═══════════════════════════════════════════════════════════════════════════════

SP400_MIDCAP = [
    # Technology
    "MANH", "SAIC", "CACI", "GLOB", "PEGA", "TENB", "CYBR", "CRWD", "DDOG", "NET",
    "ZS", "OKTA", "BILL", "PCTY", "SMAR", "QLYS", "JAMF", "ALTR", "DT", "RPD",
    
    # Healthcare
    "EXAS", "NTRA", "RARE", "HZNP", "JAZZ", "UTHR", "NBIX", "MEDP", "IART", "LIVN",
    "LNTH", "NVST", "OSCR", "PGNY", "OMCL", "PRGO", "PCRX", "INGN", "NEOG", "OGN",
    
    # Financials
    "SEIC", "WTFC", "PNFP", "SNV", "GBCI", "FNB", "SBNY", "WAL", "CATY", "UBSI",
    "IBOC", "BANR", "FFIN", "FIBK", "ONB", "FHN", "BPOP", "CBU", "VLY", "TOWN",
    
    # Consumer
    "FIVE", "BOOT", "WING", "SHAK", "PLAY", "CAKE", "TXRH", "EAT", "DIN", "JACK",
    "BJRI", "RUTH", "DINE", "BJ", "PZZA", "TACO", "WEN", "FRGI", "ARCO", "LOCO",
    
    # Industrials
    "AGCO", "TTC", "OSK", "TEX", "MTW", "PRLB", "GGG", "RBC", "B", "AIT",
    "KAI", "GHC", "SXI", "ATKR", "AWI", "UFPI", "BMI", "EPAC", "WTS", "CW",
    
    # Real Estate
    "SUI", "ELS", "ACC", "AIV", "COLD", "STAG", "REXR", "FR", "TRNO", "DEI",
]

# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED UNIVERSES
# ═══════════════════════════════════════════════════════════════════════════════

# Full S&P 500
SP500_FULL = (
    SP500_TECHNOLOGY + SP500_HEALTHCARE + SP500_FINANCIALS + 
    SP500_CONSUMER_DISCRETIONARY + SP500_CONSUMER_STAPLES + SP500_INDUSTRIALS +
    SP500_ENERGY + SP500_MATERIALS + SP500_UTILITIES + SP500_REAL_ESTATE +
    SP500_COMMUNICATION
)

# S&P 500 + MidCap 400 selections
EXPANDED_UNIVERSE = list(set(SP500_FULL + SP400_MIDCAP))

# Legacy tickers with long history (for backtesting pre-2000)
LONG_HISTORY_TICKERS = [
    "MMM", "CAT", "GE", "HON", "KO", "PEP", "PG", "CL", "MCD", "WMT",
    "JNJ", "PFE", "MRK", "ABT", "LLY", "JPM", "BAC", "WFC", "C", "AXP",
    "XOM", "CVX", "COP", "IBM", "INTC", "TXN", "T", "VZ", "SO", "DUK",
    "APD", "ECL", "AAPL", "MSFT", "HD", "NKE", "DIS",
]


def get_universe_for_period(start_date: str, size: str = "large") -> List[str]:
    """
    Get appropriate ticker universe based on start date and desired size.
    
    Args:
        start_date: YYYY-MM-DD format
        size: "small" (~50), "medium" (~200), "large" (~500)
    """
    start_year = int(start_date[:4])
    
    if start_year < 1995:
        # Only the oldest, most reliable tickers
        return [
            "MMM", "KO", "PEP", "PG", "JNJ", "PFE", "MRK",
            "JPM", "XOM", "CVX", "IBM", "GE", "T", "WMT"
        ]
    
    elif start_year < 2000:
        base = LONG_HISTORY_TICKERS
        if size == "small":
            return base[:30]
        return base
    
    elif start_year < 2010:
        # Exclude post-2010 IPOs
        exclude = {"META", "TSLA", "PYPL", "NOW", "SNOW", "CRWD", "DDOG", "NET", "ZS"}
        filtered = [t for t in SP500_FULL if t not in exclude]
        
        if size == "small":
            return filtered[:50]
        elif size == "medium":
            return filtered[:200]
        return filtered
    
    elif start_year < 2015:
        exclude = {"SNOW", "CRWD", "DDOG", "NET", "ZS", "BILL"}
        filtered = [t for t in EXPANDED_UNIVERSE if t not in exclude]
        
        if size == "small":
            return filtered[:50]
        elif size == "medium":
            return filtered[:200]
        return filtered
    
    else:
        # Full universe available
        if size == "small":
            return EXPANDED_UNIVERSE[:50]
        elif size == "medium":
            return EXPANDED_UNIVERSE[:200]
        return EXPANDED_UNIVERSE


def get_sector_tickers(sector: str) -> List[str]:
    """Get tickers for a specific sector."""
    sectors = {
        "technology": SP500_TECHNOLOGY,
        "healthcare": SP500_HEALTHCARE,
        "financials": SP500_FINANCIALS,
        "consumer_discretionary": SP500_CONSUMER_DISCRETIONARY,
        "consumer_staples": SP500_CONSUMER_STAPLES,
        "industrials": SP500_INDUSTRIALS,
        "energy": SP500_ENERGY,
        "materials": SP500_MATERIALS,
        "utilities": SP500_UTILITIES,
        "real_estate": SP500_REAL_ESTATE,
        "communication": SP500_COMMUNICATION,
    }
    return sectors.get(sector.lower(), [])


OIL_MICROCAP_STOCKS = [
    "USEG",  # U.S. Energy Corp — 2018, good history
    "MXC",   # Mexco Energy — 2018, good history
    "BRN",   # Barnwell Industries — 2018, good history
    "PED",   # PEDEVCO Corp — 2018, good history
    "REI",   # Ring Energy — 2018, good history
    "PRSO",  # Peraso Inc — 2018, good history
    "BATL",  # Battalion Oil — 2019, 24% missing (ffill applied)
]

# Forward-test only — thesis tickers with insufficient backtest history
OIL_THESIS_TICKERS = [
    "TPET",  # Trio Petroleum — 2023, 64% missing
    "EONR",  # Eon Resources — 2022, 52% missing
    "STAK",  # Stack Energy — 2026, 87% missing
]

OIL_FUTURES_PROXIES = [
    "USO",   # United States Oil Fund (ETF proxy for WTI)
    "BNO",   # United States Brent Oil Fund (ETF proxy for Brent)
]

# Combined oil-focused universe (legacy — kept for backward compatibility)
OIL_FOCUSED_UNIVERSE = OIL_MICROCAP_STOCKS + OIL_FUTURES_PROXIES

# ═══════════════════════════════════════════════════════════════════════════════
# OIL EXPANDED UNIVERSE — Tiered system for meaningful cross-sectional analysis
# ═══════════════════════════════════════════════════════════════════════════════

# Reference panel — used for cross-sectional feature computation ONLY
# These are never held in the portfolio, only provide sector context
OIL_REFERENCE_PANEL = [
    # Large-cap majors
    "XOM", "CVX", "COP", "OXY", "EOG",
    # Mid-cap E&P
    "DVN", "APA", "FANG",
    # Services
    "SLB", "HAL", "BKR",
    # Refiners
    "MPC", "PSX", "VLO",
    # Sector ETFs
    "XLE", "XOP",
]

OIL_BENCHMARKS = [
    "USO",   # United States Oil Fund (ETF proxy for WTI)
    "BNO",   # United States Brent Oil Fund (ETF proxy for Brent)
    "XLE",   # Energy Select Sector SPDR (primary sector benchmark)
    "XOP",   # SPDR S&P Oil & Gas Exploration & Production ETF
]

# Tradeable universe: microcap stocks the GP can actually hold
OIL_TRADEABLE_UNIVERSE = OIL_MICROCAP_STOCKS

# Full universe for data download: tradeable + reference panel + benchmarks
OIL_FULL_DOWNLOAD_UNIVERSE = list(set(
    OIL_TRADEABLE_UNIVERSE + OIL_REFERENCE_PANEL + OIL_BENCHMARKS
))


def get_oil_universe(expanded: bool = True) -> List[str]:
    """
    Get oil-focused universe for volatile oil market analysis.

    Args:
        expanded: If True, return full download universe (tradeable + reference + benchmarks).
                  If False, return legacy focused universe for backward compatibility.

    Returns:
        List of oil tickers
    """
    if expanded:
        return OIL_FULL_DOWNLOAD_UNIVERSE
    return OIL_FOCUSED_UNIVERSE


def get_oil_tradeable_tickers() -> List[str]:
    """Get only the tradeable oil microcap tickers (portfolio holdings)."""
    return OIL_TRADEABLE_UNIVERSE


def get_oil_thesis_tickers() -> List[str]:
    """
    Get forward-test only thesis tickers (TPET, EONR, STAK).
    These have insufficient history for backtesting but are the primary
    conviction tickers for forward deployment.
    """
    return OIL_THESIS_TICKERS


def get_oil_reference_panel() -> List[str]:
    """Get reference panel tickers for cross-sectional feature computation."""
    return OIL_REFERENCE_PANEL


def get_oil_benchmarks() -> List[str]:
    """Get oil benchmark tickers."""
    return OIL_BENCHMARKS
# Quick stats
if __name__ == "__main__":
    print(f"S&P 500 tickers: {len(SP500_FULL)}")
    print(f"MidCap additions: {len(SP400_MIDCAP)}")
    print(f"Total expanded universe: {len(EXPANDED_UNIVERSE)}")
    print(f"Oil-focused universe (legacy): {len(OIL_FOCUSED_UNIVERSE)}")
    print(f"\nBy sector:")
    for name, tickers in [
        ("Technology", SP500_TECHNOLOGY),
        ("Healthcare", SP500_HEALTHCARE),
        ("Financials", SP500_FINANCIALS),
        ("Consumer Disc", SP500_CONSUMER_DISCRETIONARY),
        ("Consumer Staples", SP500_CONSUMER_STAPLES),
        ("Industrials", SP500_INDUSTRIALS),
        ("Energy", SP500_ENERGY),
        ("Materials", SP500_MATERIALS),
        ("Utilities", SP500_UTILITIES),
        ("Real Estate", SP500_REAL_ESTATE),
        ("Communication", SP500_COMMUNICATION),
    ]:
        print(f"  {name}: {len(tickers)}")
    print(f"\nOil Focus (Expanded):")
    print(f"  Tradeable (microcap): {len(OIL_TRADEABLE_UNIVERSE)}")
    print(f"  Reference Panel:      {len(OIL_REFERENCE_PANEL)}")
    print(f"  Benchmarks:           {len(OIL_BENCHMARKS)}")
    print(f"  Full Download:        {len(OIL_FULL_DOWNLOAD_UNIVERSE)}")
    print(f"  Legacy (8 tickers):   {len(OIL_FOCUSED_UNIVERSE)}")