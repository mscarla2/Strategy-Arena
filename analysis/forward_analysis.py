#!/usr/bin/env python3
"""
TPET & EONR Forward Signal Monitor
────────────────────────────────────
Monitors for actionable signals based on the March 2026 catalyst analysis.
Run daily at market close.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

TICKERS = ["TPET", "EONR"]
BENCHMARKS = ["USO", "CL=F"]  # USO + WTI futures
COST_RT = 0.005
LOOKBACK_DAYS = 60


# ─────────────────────────────────────────────────────────────────────────────
# Signal definitions based on March 2026 analysis
# ─────────────────────────────────────────────────────────────────────────────

SIGNALS = {
    "TPET": {
        "war_premium_intact":    lambda r: r["uso_beta_5d"] > 2.0,
        "war_premium_fading":    lambda r: r["uso_beta_5d"] < 1.0,
        "volume_accumulation":   lambda r: r["vol_ratio"] > 3.0,
        "volume_distribution":   lambda r: r["vol_ratio"] < 0.3,
        "week1_march_entry":     lambda r: r["month"] == 3 and r["week_of_month"] <= 2,
        "week3_march_exit":      lambda r: r["month"] == 3 and r["week_of_month"] >= 3,
        "overnight_setup":       lambda r: r["drop_from_open"] >= 3.0 and r["vol_ratio"] > 3.0,
    },
    "EONR": {
        "accumulation_zone":     lambda r: r["close"] <= 0.90,
        "breakout_signal":       lambda r: r["vol_ratio"] > 3.0 and r["close_in_range"] > 0.6,
        "hedge_protected":       True,   # Always true — $70/bbl hedge through 2027
        "debt_cleared":          True,   # Always true — $3M debt
        "drilling_catalyst":     lambda r: r["month"] in [4, 5, 6],  # Q2 drilling begins
        "overnight_setup":       lambda r: r["drop_from_open"] >= 6.0
                                          and r["close_in_range"] < 0.5,
        "week2_march_entry":     lambda r: r["month"] == 3 and r["week_of_month"] == 2,
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# Download & compute
# ─────────────────────────────────────────────────────────────────────────────

def download(ticker: str, days: int = 120) -> pd.DataFrame:
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.columns = ["open", "high", "low", "close", "volume"]
    return df


def compute_signals(df: pd.DataFrame, uso_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret"]             = df["close"].pct_change()
    df["gap_pct"]         = (df["open"] - df["close"].shift(1)) / df["close"].shift(1) * 100
    df["intraday_range"]  = (df["high"] - df["low"]) / df["open"] * 100
    df["drop_from_open"]  = (df["open"] - df["low"]) / df["open"] * 100
    df["rise_from_open"]  = (df["high"] - df["open"]) / df["open"] * 100
    df["open_to_close"]   = (df["close"] - df["open"]) / df["open"] * 100
    df["close_in_range"]  = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    df["next_open"]       = df["open"].shift(-1)
    df["overnight_ret"]   = (df["next_open"] - df["close"]) / df["close"] * 100
    df["vol_20d"]         = df["volume"].rolling(20).mean()
    df["vol_ratio"]       = df["volume"] / df["vol_20d"]
    df["month"]           = df.index.month
    df["week_of_month"]   = (df.index.day - 1) // 7 + 1

    # Rolling beta to USO (5-day and 20-day)
    uso_ret = uso_df["close"].pct_change()
    aligned = df["ret"].align(uso_ret, join="inner")
    stock_ret, uso_aligned = aligned

    for window, col in [(5, "uso_beta_5d"), (20, "uso_beta_20d")]:
        betas = []
        for i in range(len(stock_ret)):
            if i < window:
                betas.append(np.nan)
                continue
            s = stock_ret.iloc[i-window:i]
            u = uso_aligned.iloc[i-window:i]
            cov = s.cov(u)
            var = u.var()
            betas.append(cov / var if var > 0 else np.nan)
        beta_series = pd.Series(betas, index=stock_ret.index)
        df[col] = beta_series.reindex(df.index)

    # Volatility regime
    df["vol_10d_ann"] = df["ret"].rolling(10).std() * np.sqrt(252) * 100
    df["vol_30d_ann"] = df["ret"].rolling(30).std() * np.sqrt(252) * 100
    df["vol_regime"]  = df["vol_10d_ann"] / df["vol_30d_ann"]

    # Momentum
    df["mom_5d"]  = df["close"].pct_change(5) * 100
    df["mom_10d"] = df["close"].pct_change(10) * 100
    df["mom_20d"] = df["close"].pct_change(20) * 100

    # Distance from March peak
    march_data = df[df["month"] == 3]
    if len(march_data) > 0:
        march_peak = march_data["close"].max()
        df["pct_from_march_peak"] = (df["close"] - march_peak) / march_peak * 100
    else:
        df["pct_from_march_peak"] = np.nan

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'═'*70}")
print(f"  TPET & EONR FORWARD SIGNAL MONITOR")
print(f"  Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{'═'*70}")

# Download benchmarks
uso_df = download("USO", 120)
wti_df = download("CL=F", 120)

SEP = "─" * 70

for ticker in TICKERS:
    df_raw = download(ticker, LOOKBACK_DAYS + 30)
    if df_raw.empty:
        print(f"\n  {ticker}: no data")
        continue

    df = compute_signals(df_raw, uso_df)
    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else latest

    print(f"\n\n{'═'*70}")
    print(f"  {ticker} — CURRENT STATE & FORWARD SIGNALS")
    print(f"  As of: {df.index[-1].date()}")
    print(f"{'═'*70}")

    # ── Current state ─────────────────────────────────────────────────────
    print(f"\n  CURRENT STATE")
    print(f"  {SEP}")
    print(f"  Price:              ${latest['close']:.3f}")
    print(f"  Today's return:     {latest['ret']*100:+.2f}%")
    print(f"  Intraday range:     {latest['intraday_range']:.1f}%  "
          f"(drop={latest['drop_from_open']:.1f}%  rise={latest['rise_from_open']:.1f}%)")
    print(f"  Volume ratio:       {latest['vol_ratio']:.2f}x  (vs 20-day avg)")
    print(f"  Close in range:     {latest['close_in_range']:.2f}  "
          f"(0=at low, 1=at high)")
    print(f"  Gap from yesterday: {latest['gap_pct']:+.2f}%")
    print(f"  Beta to USO (5d):   {latest['uso_beta_5d']:+.2f}x")
    print(f"  Beta to USO (20d):  {latest['uso_beta_20d']:+.2f}x")
    print(f"  Vol regime:         {latest['vol_regime']:.2f}x  "
          f"(>1=vol expanding, <1=vol contracting)")
    print(f"  Mom 5d/10d/20d:     {latest['mom_5d']:+.1f}% / "
          f"{latest['mom_10d']:+.1f}% / {latest['mom_20d']:+.1f}%")
    print(f"  From March peak:    {latest['pct_from_march_peak']:+.1f}%")

    # ── War premium status ────────────────────────────────────────────────
    print(f"\n  WAR PREMIUM STATUS")
    print(f"  {SEP}")
    beta_5d  = latest["uso_beta_5d"]
    beta_20d = latest["uso_beta_20d"]

    if beta_5d > 3.0:
        wp_status = "🔴 EXTREME — War premium fully intact, highly volatile"
    elif beta_5d > 2.0:
        wp_status = "🟡 ELEVATED — War premium intact but moderating"
    elif beta_5d > 1.0:
        wp_status = "🟠 FADING — War premium unwinding, reduce exposure"
    else:
        wp_status = "🟢 NORMALIZED — War premium gone, back to fundamental value"

    print(f"  Status: {wp_status}")
    print(f"  5d beta: {beta_5d:+.2f}x  |  20d beta: {beta_20d:+.2f}x")
    print(f"  Historical normal beta: ~0.57x (TPET) / ~0.60x (EONR)")
    if beta_5d > 1.0:
        implied_excess = (beta_5d - 0.6) / 0.6 * 100
        print(f"  War premium implied: ~{implied_excess:.0f}% excess sensitivity to oil")

    # ── USO / WTI current ─────────────────────────────────────────────────
    print(f"\n  OIL MACRO CONTEXT")
    print(f"  {SEP}")
    if not uso_df.empty:
        uso_latest = uso_df.iloc[-1]
        uso_mom5   = uso_df["close"].pct_change(5).iloc[-1] * 100
        uso_mom20  = uso_df["close"].pct_change(20).iloc[-1] * 100
        print(f"  USO price:    ${uso_latest['close']:.2f}  "
              f"(5d: {uso_mom5:+.1f}%  20d: {uso_mom20:+.1f}%)")
    if not wti_df.empty:
        wti_latest = wti_df.iloc[-1]
        wti_mom5   = wti_df["close"].pct_change(5).iloc[-1] * 100
        print(f"  WTI futures:  ${wti_latest['close']:.2f}  (5d: {wti_mom5:+.1f}%)")

    # ── Forward signals ───────────────────────────────────────────────────
    print(f"\n  FORWARD SIGNALS")
    print(f"  {SEP}")

    row = latest.to_dict()

    if ticker == "TPET":
        # Signal 1: War premium
        if row.get("uso_beta_5d", 0) > 2.0:
            print(f"  ✓ WAR PREMIUM INTACT — maintain position, high beta to oil")
        elif row.get("uso_beta_5d", 0) < 1.0:
            print(f"  ✗ WAR PREMIUM FADING — reduce position, beta normalizing")

        # Signal 2: Volume accumulation vs distribution
        if row.get("vol_ratio", 0) > 3.0:
            print(f"  ✓ VOLUME ACCUMULATION — {row['vol_ratio']:.1f}x normal, institutional interest")
        elif row.get("vol_ratio", 0) < 0.3:
            print(f"  ✗ VOLUME DISTRIBUTION — {row['vol_ratio']:.1f}x normal, smart money exiting")

        # Signal 3: March week pattern
        if row.get("month") == 3:
            wom = row.get("week_of_month", 0)
            if wom <= 2:
                print(f"  ✓ MARCH WEEK {wom} — historically strong (+359% week 1, +261% week 2)")
            else:
                print(f"  ✗ MARCH WEEK {wom} — historically weak (avg -15% to -35%)")

        # Signal 4: Overnight setup
        if row.get("drop_from_open", 0) >= 3.0 and row.get("vol_ratio", 0) > 3.0:
            print(f"  ✓ OVERNIGHT SETUP — drop {row['drop_from_open']:.1f}% + vol {row['vol_ratio']:.1f}x "
                  f"→ buy close, sell next open (Sharpe 3.90 historically)")

        # Signal 5: March 31 catalyst
        days_to_march31 = (pd.Timestamp("2026-03-31") - df.index[-1]).days
        if 0 <= days_to_march31 <= 14:
            print(f"  ⚡ CATALYST APPROACHING — Alberta well completion in {days_to_march31} days")
            print(f"     Watch for production announcement (30-40 bbl/day target)")

        # Signal 6: Post-March positioning
        if row.get("month") != 3:
            print(f"  📊 POST-MARCH — Monitor beta normalization")
            print(f"     Entry signal: vol_ratio > 3x AND beta > 2x simultaneously")

    elif ticker == "EONR":
        # Signal 1: Accumulation zone
        price = row.get("close", 0)
        if price <= 0.90:
            print(f"  ✓ ACCUMULATION ZONE — price ${price:.3f} ≤ $0.90 target")
            print(f"     Fundamental floor: $70/bbl hedge + $3M debt + 92-well plan")
        elif price >= 1.50:
            print(f"  ⚠ EXTENDED — price ${price:.3f}, consider trimming")

        # Signal 2: Breakout
        if row.get("vol_ratio", 0) > 3.0 and row.get("close_in_range", 0) > 0.6:
            print(f"  ✓ BREAKOUT SIGNAL — high vol + strong close = continuation likely")

        # Signal 3: Q2 drilling catalyst
        month = row.get("month", 0)
        if month in [4, 5, 6]:
            print(f"  ⚡ Q2 DRILLING CATALYST — 92-well program should begin this month")
            print(f"     Target: +500 bbl/day production increase")

        # Signal 4: Overnight setup (EONR-specific rule)
        if row.get("drop_from_open", 0) >= 6.0 and row.get("close_in_range", 0.5) < 0.5:
            print(f"  ✓ OVERNIGHT SETUP — EONR specific: drop {row['drop_from_open']:.1f}% "
                  f"+ closed bottom half of range")
            print(f"     Historical: Sharpe 1.81 when close in bottom 25-50% of range")

        # Signal 5: Hedge protection assessment
        if not wti_df.empty:
            wti_price = wti_df["close"].iloc[-1]
            if wti_price > 70:
                premium = wti_price - 70
                print(f"  ✓ HEDGE ACTIVE — WTI ${wti_price:.0f} > $70 floor "
                      f"(${premium:.0f}/bbl premium captured)")
            else:
                print(f"  ✓ HEDGE PROTECTING — WTI ${wti_price:.0f} < $70, hedge floor active")

        # Signal 6: Debt-cleared re-rating
        print(f"  ✓ DEBT CLEARED — $3M debt enables institutional eligibility")
        print(f"     Watch for: analyst initiations, fund 13F filings")

    # ── Recent price action table ──────────────────────────────────────────
    print(f"\n  RECENT 10 DAYS")
    print(f"  {SEP}")
    print(f"  {'Date':>12}  {'Close':>7}  {'Ret%':>7}  {'VolRatio':>9}  "
          f"{'Beta5d':>7}  {'VolRegime':>10}  {'OvNight%':>10}")
    for date, row_h in df.tail(10).iterrows():
        flag = ""
        if abs(row_h["ret"]) > 0.10:
            flag = " ***"
        elif row_h["vol_ratio"] > 2:
            flag = " **"
        print(f"  {str(date.date()):>12}  "
              f"${row_h['close']:>6.3f}  "
              f"{row_h['ret']*100:>+7.2f}  "
              f"{row_h['vol_ratio']:>9.2f}  "
              f"{row_h['uso_beta_5d']:>+7.2f}  "
              f"{row_h['vol_regime']:>10.2f}  "
              f"{row_h['overnight_ret']:>+10.2f}{flag}")

    # ── Scenario analysis ─────────────────────────────────────────────────
    print(f"\n  SCENARIO ANALYSIS")
    print(f"  {SEP}")

    current_price = latest["close"]

    if ticker == "TPET":
        scenarios = {
            "Hormuz reopens / ceasefire": {
                "probability": "40%",
                "beta_impact": "drops to 0.57x",
                "price_impact": f"${current_price * 0.40:.3f} - ${current_price * 0.60:.3f}",
                "timeline": "1-4 weeks",
                "action": "EXIT position before this happens",
            },
            "Conflict escalates further": {
                "probability": "20%",
                "beta_impact": "spikes to 6-8x",
                "price_impact": f"${current_price * 1.50:.3f} - ${current_price * 2.50:.3f}",
                "timeline": "days",
                "action": "HOLD / add on dips with tight stop",
            },
            "Stalemate (current regime)": {
                "probability": "40%",
                "beta_impact": "stays 2-4x",
                "price_impact": f"${current_price * 0.80:.3f} - ${current_price * 1.20:.3f}",
                "timeline": "weeks",
                "action": "TRADE the volatility (overnight edge)",
            },
        }
    else:  # EONR
        scenarios = {
            "War fades, oil drops to $70": {
                "probability": "40%",
                "fundamental_floor": "$70/bbl hedge active",
                "price_impact": f"${current_price * 0.60:.3f} - ${current_price * 0.80:.3f}",
                "timeline": "1-4 weeks",
                "action": "BUY — hedge protects, 92-well plan drives re-rating",
            },
            "92-well plan executes (+500 bbl/day)": {
                "probability": "60%",
                "revenue_impact": "+$12-15M annual revenue at $70/bbl",
                "price_impact": f"${current_price * 1.50:.3f} - ${current_price * 3.00:.3f}",
                "timeline": "3-9 months",
                "action": "HOLD / accumulate on weakness",
            },
            "Conflict deepens, oil $120+": {
                "probability": "20%",
                "note": "Hedged at $70 — MISSES upside above $70 on hedged portion",
                "price_impact": f"${current_price * 1.20:.3f} - ${current_price * 2.00:.3f}",
                "timeline": "weeks",
                "action": "HOLD — unhedged 88% captures full upside",
            },
        }

    for scenario, details in scenarios.items():
        print(f"\n  [{details.get('probability', '?')}] {scenario}")
        for k, v in details.items():
            if k not in ["probability"]:
                print(f"    {k:<20}: {v}")

print(f"\n\n{'═'*70}")
print(f"  END OF SIGNAL REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{'═'*70}\n")