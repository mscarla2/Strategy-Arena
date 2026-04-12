#!/usr/bin/env python3
"""
highest_impact_collector.py
================================================================
Highest-Impact Data Collector for EONR and TPET
Pulls: Hedge book | Dilution stack | Production rates | Cash/burn
Source: SEC EDGAR API (free, no API key required)

Usage:
    python highest_impact_collector.py

Requirements:
    pip install requests pandas
"""

import requests
import json
import re
import time
from datetime import datetime, timedelta
from typing import Optional

# ── Configuration ─────────────────────────────────────────────
TICKERS = ["EONR", "TPET"]

# SEC requires a descriptive User-Agent with contact info
HEADERS = {
    "User-Agent": "ResearchScript research@youremail.com",  # ← update this
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json",
}

EDGAR_BASE   = "https://data.sec.gov"
EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
SEC_BASE     = "https://www.sec.gov"

RATE_LIMIT_DELAY = 0.15   # SEC enforces ~10 req/sec max
LOOKBACK_DAYS    = 365    # How far back to search filings


# ── EDGAR Utility Functions ────────────────────────────────────

def get_cik(ticker: str) -> Optional[str]:
    """Resolve ticker to 10-digit CIK via SEC company tickers file."""
    url  = f"{SEC_BASE}/files/company_tickers.json"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    for _, company in resp.json().items():
        if company.get("ticker", "").upper() == ticker.upper():
            return str(company["cik_str"]).zfill(10)
    return None


def get_submissions(cik: str) -> dict:
    """Fetch full filing submission history for a company."""
    url  = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_xbrl_facts(cik: str) -> Optional[dict]:
    """
    Pull all XBRL-tagged financial facts.
    Note: Many micro-caps are exempt from XBRL requirements,
    so this may return None — handled gracefully below.
    """
    url  = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def fulltext_search(ticker: str, terms: str, form_types: str = "10-Q,10-K,8-K") -> list:
    """
    Run EDGAR full-text search for specific keywords in recent filings.
    Returns list of hits with source metadata and excerpts.
    """
    params = {
        "q":         f'"{ticker}" {terms}',
        "dateRange": "custom",
        "startdt":   (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d"),
        "enddt":     datetime.now().strftime("%Y-%m-%d"),
        "forms":     form_types,
    }
    resp = requests.get(EDGAR_SEARCH, params=params, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        return []
    return resp.json().get("hits", {}).get("hits", [])


# ── XBRL Data Extraction ───────────────────────────────────────

def extract_xbrl_metrics(xbrl: dict) -> dict:
    """
    Pull the most analytically useful fields from the XBRL facts blob.
    Returns a flat dict with the most recent reported value per metric.
    """
    if not xbrl:
        return {}

    us_gaap = xbrl.get("facts", {}).get("us-gaap", {})
    dei     = xbrl.get("facts", {}).get("dei",      {})

    def latest(concept: dict, unit: str = "shares") -> Optional[dict]:
        """Return the most recently reported value for a given concept."""
        units = concept.get("units", {}).get(unit, [])
        # Prefer 10-K/10-Q over amendments and proxy filings
        valid = [u for u in units if u.get("form") in ("10-K", "10-Q", "20-F")]
        if not valid:
            valid = units
        return max(valid, key=lambda x: x.get("end", "")) if valid else None

    def fmt_shares(v):
        if v is None:
            return "N/A"
        if v >= 1_000_000_000:
            return f"{v / 1_000_000_000:.2f}B"
        if v >= 1_000_000:
            return f"{v / 1_000_000:.1f}M"
        if v >= 1_000:
            return f"{v / 1_000:.1f}K"
        return f"{v:,.0f}"

    def fmt_usd(v):
        if v is None:
            return "N/A"
        sign = "-" if v < 0 else ""
        av   = abs(v)
        if av >= 1_000_000:
            return f"{sign}${av / 1_000_000:.2f}M"
        if av >= 1_000:
            return f"{sign}${av / 1_000:.1f}K"
        return f"{sign}${av:,.0f}"

    results = {}

    # ── Shares outstanding ──
    for key in ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"]:
        rec = latest(us_gaap.get(key, {})) or latest(dei.get(key, {}))
        if rec:
            results["shares_outstanding"] = {
                "raw": rec["val"], "fmt": fmt_shares(rec["val"]), "as_of": rec.get("end")
            }
            break

    # ── Shares authorized ──
    for key in ["CommonStockSharesAuthorized"]:
        rec = latest(us_gaap.get(key, {}))
        if rec:
            results["shares_authorized"] = {
                "raw": rec["val"], "fmt": fmt_shares(rec["val"]), "as_of": rec.get("end")
            }

    # ── Warrants ──
    for key in ["ClassOfWarrantOrRightOutstanding", "WarrantsAndRightsOutstanding"]:
        rec = latest(us_gaap.get(key, {}))
        if rec:
            results["warrants_outstanding"] = {
                "raw": rec["val"], "fmt": fmt_shares(rec["val"]), "as_of": rec.get("end")
            }
            break

    # ── Cash ──
    for key in ["CashAndCashEquivalentsAtCarryingValue",
                "CashCashEquivalentsAndShortTermInvestments", "Cash"]:
        rec = latest(us_gaap.get(key, {}), unit="USD")
        if rec:
            results["cash"] = {
                "raw": rec["val"], "fmt": fmt_usd(rec["val"]), "as_of": rec.get("end")
            }
            break

    # ── Long-term debt ──
    for key in ["LongTermDebtNoncurrent", "LongTermDebt", "NotesPayable"]:
        rec = latest(us_gaap.get(key, {}), unit="USD")
        if rec:
            results["long_term_debt"] = {
                "raw": rec["val"], "fmt": fmt_usd(rec["val"]), "as_of": rec.get("end")
            }
            break

    # ── Operating cash flow (annualized → monthly burn proxy) ──
    for key in ["NetCashProvidedByUsedInOperatingActivities"]:
        rec = latest(us_gaap.get(key, {}), unit="USD")
        if rec:
            results["operating_cash_flow"] = {
                "raw": rec["val"], "fmt": fmt_usd(rec["val"]), "as_of": rec.get("end")
            }

    # ── Revenue ──
    for key in ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "OilAndGasRevenue", "OilAndCondensateRevenue"]:
        rec = latest(us_gaap.get(key, {}), unit="USD")
        if rec:
            results["revenue"] = {
                "raw": rec["val"], "fmt": fmt_usd(rec["val"]), "as_of": rec.get("end")
            }
            break

    return results


# ── Dilution & Runway Analysis ─────────────────────────────────

def compute_dilution_profile(metrics: dict) -> dict:
    """Derive dilution headroom, fully diluted count, and cash runway."""
    profile = {}

    so   = metrics.get("shares_outstanding",   {}).get("raw")
    auth = metrics.get("shares_authorized",    {}).get("raw")
    wts  = metrics.get("warrants_outstanding", {}).get("raw")
    cash = metrics.get("cash",                 {}).get("raw")
    ocf  = metrics.get("operating_cash_flow",  {}).get("raw")

    if so and auth:
        pct_used = (so / auth) * 100
        profile["authorized_headroom_shares"] = auth - so
        profile["pct_authorization_used"]     = pct_used
        profile["dilution_risk_flag"] = (
            "HIGH"     if pct_used > 85 else
            "MODERATE" if pct_used > 70 else
            "LOW"
        )

    if so and wts:
        profile["fully_diluted_shares"] = so + wts
        profile["warrant_dilution_pct"] = (wts / so) * 100

    # Estimate runway from annualized OCF and current cash
    if cash and ocf and ocf < 0:
        monthly_burn = abs(ocf) / 12
        if monthly_burn > 0:
            months = cash / monthly_burn
            profile["est_monthly_burn_usd"] = monthly_burn
            profile["est_months_runway"]    = months
            profile["capital_raise_risk"]   = (
                "HIGH"     if months < 6  else
                "MODERATE" if months < 12 else
                "LOW"
            )

    return profile


# ── Full-Text Search Categories ────────────────────────────────

SEARCH_QUERIES = {
    "hedge_book": {
        "terms":   "hedge swap collar derivative crude oil barrel strike price",
        "summary": "Hedging and derivative positions on crude oil production",
    },
    "production_rates": {
        "terms":   "barrels per day bopd boe production current net",
        "summary": "Current oil/gas production rates",
    },
    "dilution_instruments": {
        "terms":   "warrant convertible note shelf registration authorized shares S-3",
        "summary": "Potential dilution instruments and shelf registration activity",
    },
    "cash_runway": {
        "terms":   "cash equivalents liquidity burn rate going concern capital raise",
        "summary": "Cash position, burn rate, and going concern disclosures",
    },
}


# ── Per-Ticker Analysis ────────────────────────────────────────

def analyze_ticker(ticker: str) -> dict:
    result = {
        "ticker":           ticker,
        "cik":              None,
        "company_name":     None,
        "xbrl_metrics":     {},
        "dilution_profile": {},
        "recent_filings":   [],
        "search_findings":  {},
        "errors":           [],
    }

    print(f"\n{'━' * 64}")
    print(f"  TICKER: {ticker}")
    print(f"{'━' * 64}")

    # ── Step 1: Resolve CIK ──────────────────────────────────
    print("\n  [1/4]  Resolving CIK...")
    try:
        cik = get_cik(ticker)
        time.sleep(RATE_LIMIT_DELAY)
    except Exception as e:
        result["errors"].append(f"CIK lookup failed: {e}")
        print(f"         ✗ {e}")
        return result

    if not cik:
        msg = f"{ticker} not found in SEC company tickers. Verify the symbol on EDGAR."
        result["errors"].append(msg)
        print(f"         ✗ {msg}")
        return result

    result["cik"] = cik
    print(f"         ✓ CIK: {cik}")

    # ── Step 2: Submission history + filing list ─────────────
    print("\n  [2/4]  Fetching submission history...")
    try:
        submissions        = get_submissions(cik)
        time.sleep(RATE_LIMIT_DELAY)
        result["company_name"] = submissions.get("name", "N/A")
        print(f"         ✓ Company: {result['company_name']}")

        recent     = submissions.get("filings", {}).get("recent", {})
        forms      = recent.get("form",          [])
        dates      = recent.get("filingDate",     [])
        accessions = recent.get("accessionNumber",[])

        TARGET_FORMS = {"10-Q", "10-K", "8-K", "S-3", "S-1", "DEF 14A"}
        for i, form in enumerate(forms):
            if form in TARGET_FORMS:
                result["recent_filings"].append({
                    "form":      form,
                    "date":      dates[i]      if i < len(dates)      else "N/A",
                    "accession": accessions[i] if i < len(accessions) else "N/A",
                })
            if len(result["recent_filings"]) >= 15:
                break

        # Shelf registrations are the primary hard dilution signal
        shelf = [f for f in result["recent_filings"] if f["form"] in ("S-3", "S-1")]
        if shelf:
            print(f"         ⚠  DILUTION SIGNAL: {len(shelf)} shelf/S-1 filing(s) found:")
            for sf in shelf:
                print(f"            {sf['form']}  filed {sf['date']}  {sf['accession']}")
        else:
            print(f"         ✓ No shelf registrations in past {LOOKBACK_DAYS} days")

        print(f"         ✓ {len(result['recent_filings'])} relevant filings indexed")

    except Exception as e:
        result["errors"].append(f"Submissions fetch failed: {e}")
        print(f"         ✗ {e}")

    # ── Step 3: XBRL standardized financials ────────────────
    print("\n  [3/4]  Pulling XBRL financial facts...")
    try:
        xbrl_raw = get_xbrl_facts(cik)
        time.sleep(RATE_LIMIT_DELAY)

        if not xbrl_raw:
            print("         ⚠  No XBRL data on file (common for micro-caps).")
            print("            Hedge/production data relies entirely on full-text search.")
        else:
            metrics = extract_xbrl_metrics(xbrl_raw)
            result["xbrl_metrics"] = metrics

            profile = compute_dilution_profile(metrics)
            result["dilution_profile"] = profile

            print(f"         ✓ {len(metrics)} metrics extracted")

            DISPLAY_ORDER = [
                "shares_outstanding", "shares_authorized", "warrants_outstanding",
                "cash", "long_term_debt", "operating_cash_flow", "revenue",
            ]
            for key in DISPLAY_ORDER:
                if key in metrics:
                    m = metrics[key]
                    print(f"            {key:<32s}  {m['fmt']:>12s}  (as of {m.get('as_of','?')})")

            if profile:
                print(f"\n         DILUTION PROFILE:")
                if "pct_authorization_used" in profile:
                    print(f"            Authorization used:       "
                          f"{profile['pct_authorization_used']:.1f}%  "
                          f"[{profile.get('dilution_risk_flag','?')} risk]")
                if "authorized_headroom_shares" in profile:
                    print(f"            Authorized headroom:      "
                          f"{profile['authorized_headroom_shares']:>15,.0f} shares")
                if "fully_diluted_shares" in profile:
                    print(f"            Fully diluted shares:     "
                          f"{profile['fully_diluted_shares']:>15,.0f}")
                if "warrant_dilution_pct" in profile:
                    print(f"            Warrant dilution:         "
                          f"{profile['warrant_dilution_pct']:.1f}% of current float")
                if "est_months_runway" in profile:
                    print(f"            Est. cash runway:         "
                          f"~{profile['est_months_runway']:.1f} months  "
                          f"[{profile.get('capital_raise_risk','?')} risk]")
                if "est_monthly_burn_usd" in profile:
                    burn = profile["est_monthly_burn_usd"]
                    if burn >= 1_000_000:
                        burn_fmt = f"${burn/1_000_000:.2f}M/month"
                    else:
                        burn_fmt = f"${burn/1_000:.1f}K/month"
                    print(f"            Est. monthly burn:        {burn_fmt}")

    except Exception as e:
        result["errors"].append(f"XBRL fetch failed: {e}")
        print(f"         ✗ {e}")

    # ── Step 4: Full-text filing search ─────────────────────
    print("\n  [4/4]  Running full-text search across recent filings...")

    for qname, qcfg in SEARCH_QUERIES.items():
        try:
            hits = fulltext_search(ticker, qcfg["terms"])
            time.sleep(RATE_LIMIT_DELAY)
            result["search_findings"][qname] = hits

            print(f"\n         [{qname.upper()}]")
            print(f"         {qcfg['summary']}")
            print(f"         {len(hits)} matching document(s):")

            for hit in hits[:3]:
                src = hit.get("_source", {})
                hl  = hit.get("highlight", {})
                print(f"           • {src.get('form_type','?'):<8s}"
                      f" | {src.get('file_date','?')}"
                      f" | {src.get('entity_name','?')}")
                for snippets in hl.values():
                    if snippets:
                        clean = re.sub(r"<[^>]+>", "", snippets[0])[:300]
                        print(f"             └ \"{clean}...\"")
                        break

        except Exception as e:
            result["errors"].append(f"Full-text search [{qname}] failed: {e}")
            print(f"         ✗ {e}")

    return result


# ── Final Summary Printer ──────────────────────────────────────

def print_final_summary(all_results: list):
    print(f"\n\n{'═' * 64}")
    print("  FINAL SUMMARY — HIGHEST-IMPACT DATA COLLECTOR")
    print(f"  Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 64}")

    for r in all_results:
        ticker  = r["ticker"]
        name    = r.get("company_name", "N/A")
        metrics = r.get("xbrl_metrics", {})
        profile = r.get("dilution_profile", {})
        filings = r.get("recent_filings", [])
        errors  = r.get("errors", [])

        print(f"\n  {ticker}  ({name})  CIK: {r.get('cik','N/A')}")
        print(f"  {'─' * 56}")

        def fval(key, prefix=""):
            m = metrics.get(key, {})
            if not m:
                return f"  {prefix}N/A (not in XBRL — check 10-Q)"
            return f"  {prefix}{m['fmt']}  (as of {m.get('as_of','?')})"

        print(f"  Shares Outstanding:    {fval('shares_outstanding')}")
        print(f"  Shares Authorized:     {fval('shares_authorized')}")
        print(f"  Warrants Outstanding:  {fval('warrants_outstanding')}")
        print(f"  Cash:                  {fval('cash')}")
        print(f"  Long-Term Debt:        {fval('long_term_debt')}")
        print(f"  Op. Cash Flow (ann.):  {fval('operating_cash_flow')}")

        if "pct_authorization_used" in profile:
            print(f"\n  Dilution Risk:   "
                  f"{profile['pct_authorization_used']:.1f}% of authorization used  "
                  f"[{profile.get('dilution_risk_flag','?')}]")
        if "est_months_runway" in profile:
            print(f"  Capital Raise Risk: "
                  f"~{profile['est_months_runway']:.1f} months runway  "
                  f"[{profile.get('capital_raise_risk','?')}]")

        shelves = [f for f in filings if f["form"] in ("S-3", "S-1")]
        if shelves:
            print(f"\n  ⚠  SHELF FILINGS ({len(shelves)}):")
            for s in shelves:
                print(f"     {s['form']} filed {s['date']}  |  {s['accession']}")
        else:
            print(f"\n  No shelf/S-1 filings in past {LOOKBACK_DAYS} days")

        if errors:
            print(f"\n  Errors ({len(errors)}):")
            for e in errors:
                print(f"     ✗ {e}")

    print(f"\n{'═' * 64}")
    print("  IMPORTANT LIMITATIONS:")
    print("  1. XBRL data may lag the latest 10-Q by 1–3 weeks.")
    print("  2. Hedge book details are rarely XBRL-tagged — they live in")
    print("     the 'Derivatives and Hedging' footnote of the 10-Q.")
    print("  3. Production rates are almost never XBRL-tagged for micro-")
    print("     caps. Check full-text search results for bopd figures.")
    print("  4. For going concern language, search for 'substantial doubt'")
    print("     in the full-text search results under cash_runway.")
    print(f"{'═' * 64}\n")


# ── Entry Point ────────────────────────────────────────────────

def main():
    print(f"\n{'═' * 64}")
    print("  HIGHEST-IMPACT DATA COLLECTOR")
    print("  Source: SEC EDGAR (no API key required)")
    print(f"  Targets: {', '.join(TICKERS)}")
    print(f"  Lookback: {LOOKBACK_DAYS} days")
    print(f"{'═' * 64}")

    all_results = []
    for ticker in TICKERS:
        result = analyze_ticker(ticker)
        all_results.append(result)
        time.sleep(1.0)  # Polite pause between tickers

    print_final_summary(all_results)

    outfile = f"highest_impact_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Full dataset saved → {outfile}\n")


if __name__ == "__main__":
    main()