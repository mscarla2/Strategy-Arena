"""
Unit tests for data/universe.py

Covers all universe lists, helper functions, and the expanded oil universe system.
"""
import pytest

from data.universe import (
    SP500_TECHNOLOGY, SP500_HEALTHCARE, SP500_FINANCIALS,
    SP500_CONSUMER_DISCRETIONARY, SP500_CONSUMER_STAPLES,
    SP500_INDUSTRIALS, SP500_ENERGY, SP500_MATERIALS,
    SP500_UTILITIES, SP500_REAL_ESTATE, SP500_COMMUNICATION,
    OIL_MICROCAP_STOCKS, OIL_FOCUSED_UNIVERSE, OIL_REFERENCE_PANEL,
    OIL_BENCHMARKS, OIL_TRADEABLE_UNIVERSE, OIL_FULL_DOWNLOAD_UNIVERSE,
    get_oil_universe, get_oil_tradeable_tickers,
    get_oil_reference_panel, get_oil_benchmarks,
    get_sector_tickers,
)


# ─── S&P 500 Sector Lists ─────────────────────────────────────────────────────

class TestSP500Sectors:
    @pytest.mark.parametrize("sector_list,min_count", [
        (SP500_TECHNOLOGY, 30),
        (SP500_HEALTHCARE, 30),
        (SP500_FINANCIALS, 30),
        (SP500_CONSUMER_DISCRETIONARY, 30),
        (SP500_CONSUMER_STAPLES, 20),
        (SP500_INDUSTRIALS, 30),
        (SP500_ENERGY, 10),
        (SP500_MATERIALS, 15),
        (SP500_UTILITIES, 15),
        (SP500_REAL_ESTATE, 10),
        (SP500_COMMUNICATION, 10),
    ])
    def test_sector_has_minimum_tickers(self, sector_list, min_count):
        assert len(sector_list) >= min_count

    def test_no_duplicates_within_sector(self):
        for sector in [
            SP500_TECHNOLOGY, SP500_HEALTHCARE, SP500_FINANCIALS,
            SP500_CONSUMER_DISCRETIONARY, SP500_CONSUMER_STAPLES,
            SP500_INDUSTRIALS, SP500_ENERGY, SP500_MATERIALS,
            SP500_UTILITIES, SP500_REAL_ESTATE, SP500_COMMUNICATION,
        ]:
            assert len(sector) == len(set(sector)), f"Duplicate tickers in sector: {sector}"

    def test_all_tickers_are_strings(self):
        for sector in [SP500_TECHNOLOGY, SP500_ENERGY, SP500_FINANCIALS]:
            for ticker in sector:
                assert isinstance(ticker, str)
                assert len(ticker) >= 1

    def test_key_tech_stocks_present(self):
        assert "AAPL" in SP500_TECHNOLOGY
        assert "MSFT" in SP500_TECHNOLOGY
        assert "NVDA" in SP500_TECHNOLOGY

    def test_key_energy_stocks_present(self):
        assert "XOM" in SP500_ENERGY
        assert "CVX" in SP500_ENERGY


# ─── get_sector_tickers() ─────────────────────────────────────────────────────

class TestGetSectorTickers:
    def test_known_sector_returns_list(self):
        result = get_sector_tickers("technology")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_case_insensitive(self):
        assert get_sector_tickers("Technology") == get_sector_tickers("technology")

    def test_unknown_sector_returns_empty(self):
        assert get_sector_tickers("doesnotexist") == []

    @pytest.mark.parametrize("sector", [
        "technology", "healthcare", "financials", "consumer_discretionary",
        "consumer_staples", "industrials", "energy", "materials",
        "utilities", "real_estate", "communication",
    ])
    def test_all_sectors_accessible(self, sector):
        result = get_sector_tickers(sector)
        assert isinstance(result, list)
        assert len(result) > 0


# ─── Oil Microcap Stocks ──────────────────────────────────────────────────────

class TestOilMicrocapStocks:
    def test_has_minimum_tickers(self):
        assert len(OIL_MICROCAP_STOCKS) >= 4

    def test_no_duplicates(self):
        assert len(OIL_MICROCAP_STOCKS) == len(set(OIL_MICROCAP_STOCKS))

    def test_known_microcaps_present(self):
        # These are the core microcaps established in the system
        for ticker in ["USEG", "MXC", "BRN"]:
            assert ticker in OIL_MICROCAP_STOCKS


# ─── Oil Reference Panel ──────────────────────────────────────────────────────

class TestOilReferencePanel:
    def test_has_16_tickers(self):
        assert len(OIL_REFERENCE_PANEL) == 16

    def test_includes_majors(self):
        for ticker in ["XOM", "CVX", "COP", "OXY", "EOG"]:
            assert ticker in OIL_REFERENCE_PANEL

    def test_includes_etfs(self):
        assert "XLE" in OIL_REFERENCE_PANEL
        assert "XOP" in OIL_REFERENCE_PANEL

    def test_no_duplicates(self):
        assert len(OIL_REFERENCE_PANEL) == len(set(OIL_REFERENCE_PANEL))

    def test_no_overlap_with_microcaps(self):
        overlap = set(OIL_MICROCAP_STOCKS) & set(OIL_REFERENCE_PANEL)
        assert len(overlap) == 0, f"Overlap: {overlap}"


# ─── Oil Benchmarks ───────────────────────────────────────────────────────────

class TestOilBenchmarks:
    def test_includes_required_benchmarks(self):
        for ticker in ["XLE", "XOP", "USO", "BNO"]:
            assert ticker in OIL_BENCHMARKS

    def test_no_duplicates(self):
        assert len(OIL_BENCHMARKS) == len(set(OIL_BENCHMARKS))


# ─── OIL_TRADEABLE_UNIVERSE ───────────────────────────────────────────────────

class TestOilTradeableUniverse:
    def test_equals_microcap_stocks(self):
        assert OIL_TRADEABLE_UNIVERSE == OIL_MICROCAP_STOCKS

    def test_no_overlap_with_reference_panel(self):
        overlap = set(OIL_TRADEABLE_UNIVERSE) & set(OIL_REFERENCE_PANEL)
        assert len(overlap) == 0


# ─── OIL_FULL_DOWNLOAD_UNIVERSE ───────────────────────────────────────────────

class TestOilFullDownloadUniverse:
    def test_is_superset_of_tradeable(self):
        full = set(OIL_FULL_DOWNLOAD_UNIVERSE)
        for t in OIL_TRADEABLE_UNIVERSE:
            assert t in full

    def test_is_superset_of_reference_panel(self):
        full = set(OIL_FULL_DOWNLOAD_UNIVERSE)
        for t in OIL_REFERENCE_PANEL:
            assert t in full

    def test_is_superset_of_benchmarks(self):
        full = set(OIL_FULL_DOWNLOAD_UNIVERSE)
        for t in OIL_BENCHMARKS:
            assert t in full

    def test_has_enough_for_cross_sectional(self):
        assert len(OIL_FULL_DOWNLOAD_UNIVERSE) >= 20

    def test_no_duplicates(self):
        assert len(OIL_FULL_DOWNLOAD_UNIVERSE) == len(set(OIL_FULL_DOWNLOAD_UNIVERSE))


# ─── get_oil_universe() ───────────────────────────────────────────────────────

class TestGetOilUniverse:
    def test_expanded_true_returns_full_download(self):
        result = get_oil_universe(expanded=True)
        assert set(result) == set(OIL_FULL_DOWNLOAD_UNIVERSE)

    def test_expanded_false_returns_legacy(self):
        result = get_oil_universe(expanded=False)
        assert result == OIL_FOCUSED_UNIVERSE

    def test_default_is_expanded(self):
        result = get_oil_universe()
        assert set(result) == set(OIL_FULL_DOWNLOAD_UNIVERSE)


# ─── Helper functions ─────────────────────────────────────────────────────────

class TestHelperFunctions:
    def test_get_oil_tradeable_tickers(self):
        assert get_oil_tradeable_tickers() == OIL_TRADEABLE_UNIVERSE

    def test_get_oil_reference_panel(self):
        assert get_oil_reference_panel() == OIL_REFERENCE_PANEL

    def test_get_oil_benchmarks(self):
        assert get_oil_benchmarks() == OIL_BENCHMARKS
