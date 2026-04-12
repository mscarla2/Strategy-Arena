#!/usr/bin/env python3
"""
TPET & EONR — Master Analysis Suite
═════════════════════════════════════════════════════════════════════════════
Consolidates all analysis scripts into a single runnable pipeline:

  §0   Data Download & Enrichment
  §1   Intraday Microstructure & March Deep Dive
  §2   Statistical Significance (+ Outlier-Robustness Test)
  §3   Transaction Cost Estimation (Roll autocorr-aware + CS)
  §4   Fundamental Floor Models (EONR sensitivity table, TPET war premium)
  §5   Tail Risk & Kelly Position Sizing
  §6   Pairs Cointegration (OU half-life + cost-adjusted threshold)
  §7   Catalyst Hunting (big-move log, news)
  §8   Cross-Ticker Correlation
  §9   Monthly Regime Analysis
  §10  Overnight Edge Deep Dive (threshold sweep, year-by-year)
  §11  Forward Signal Monitor (war premium, accumulation, scenarios)

Run:  python analysis/master_analysis.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from datetime import datetime, timedelta


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TICKERS      = ["TPET", "EONR"]
BENCHMARKS   = ["USO", "CL=F", "XOP", "XLE"]
START        = "2022-01-01"
TRADING_DAYS = 252
SEP          = "─" * 72
WIDE         = "═" * 72

# Realistic round-trip cost estimates (from Phase 2 CS estimator)
COST_RT = {"TPET": 0.0617, "EONR": 0.0503}

# EONR fundamental parameters
EONR_HEDGE_PRICE   = 70.0    # $/bbl floor hedge price
EONR_LIFTING_COST  = 35.0    # $/bbl opex
EONR_EV_MULTIPLE   = 3.5     # small-cap E&P EV/EBITDA multiple
EONR_SHARES        = 25_000_000
EONR_PRODUCTION    = 200     # bpd

# Overnight edge thresholds (from sweep optimisation)
OE_THRESHOLD = {"TPET": 3.0, "EONR": 6.0}


# ─────────────────────────────────────────────────────────────────────────────
# §0  DATA DOWNLOAD & ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────

def download(ticker: str, start: str = START, days: int | None = None) -> pd.DataFrame:
    if days is not None:
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


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret"]            = df["close"].pct_change()
    df["gap_pct"]        = (df["open"] - df["close"].shift(1)) / df["close"].shift(1) * 100
    df["intraday_range"] = (df["high"] - df["low"]) / df["open"] * 100
    df["drop_from_open"] = (df["open"] - df["low"]) / df["open"] * 100
    df["rise_from_open"] = (df["high"] - df["open"]) / df["open"] * 100
    df["open_to_close"]  = (df["close"] - df["open"]) / df["open"] * 100
    df["close_in_range"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    df["next_open"]      = df["open"].shift(-1)
    df["overnight_ret"]  = (df["next_open"] - df["close"]) / df["close"] * 100
    df["vol_20d_avg"]    = df["volume"].rolling(20).mean()
    df["vol_ratio"]      = df["volume"] / df["vol_20d_avg"]
    df["month"]          = df.index.month
    df["year"]           = df.index.year
    df["dow"]            = df.index.dayofweek
    df["is_march"]       = df["month"] == 3
    df["week_of_month"]  = (df.index.day - 1) // 7 + 1
    df["vol_10d_ann"]    = df["ret"].rolling(10).std() * np.sqrt(TRADING_DAYS) * 100
    df["vol_30d_ann"]    = df["ret"].rolling(30).std() * np.sqrt(TRADING_DAYS) * 100
    df["vol_regime"]     = df["vol_10d_ann"] / df["vol_30d_ann"]
    df["mom_5d"]         = df["close"].pct_change(5) * 100
    df["mom_10d"]        = df["close"].pct_change(10) * 100
    df["mom_20d"]        = df["close"].pct_change(20) * 100
    return df


def attach_uso_beta(df: pd.DataFrame, uso_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    uso_ret = uso_df["close"].pct_change()
    stock_ret, uso_aligned = df["ret"].align(uso_ret, join="inner")
    for window, col in [(5, "uso_beta_5d"), (20, "uso_beta_20d")]:
        betas = []
        for i in range(len(stock_ret)):
            if i < window:
                betas.append(np.nan)
                continue
            s = stock_ret.iloc[i - window:i]
            u = uso_aligned.iloc[i - window:i]
            cov = s.cov(u)
            var = u.var()
            betas.append(cov / var if var > 0 else np.nan)
        beta_series = pd.Series(betas, index=stock_ret.index)
        df[col] = beta_series.reindex(df.index)
    return df


print(f"\n{WIDE}")
print("  TPET & EONR — MASTER ANALYSIS SUITE")
print(f"  Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{WIDE}\n")

print("Downloading data…")
data: dict[str, pd.DataFrame] = {}
uso_df = pd.DataFrame()

for t in TICKERS + BENCHMARKS:
    df = download(t)
    if not df.empty:
        enriched = enrich(df)
        if t in TICKERS and not uso_df.empty:
            enriched = attach_uso_beta(enriched, uso_df)
        data[t] = enriched
        if t == "USO":
            uso_df = enriched
        print(f"  {t}: {len(df)} days  ({df.index[0].date()} → {df.index[-1].date()})")

# Re-attach betas now USO is available for tickers loaded before USO
for t in TICKERS:
    if t in data and not uso_df.empty and "uso_beta_5d" not in data[t].columns:
        data[t] = attach_uso_beta(data[t], uso_df)


# ─────────────────────────────────────────────────────────────────────────────
# §1  INTRADAY MICROSTRUCTURE & MARCH DEEP DIVE
# ─────────────────────────────────────────────────────────────────────────────

MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
DAY_NAMES   = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday"}

print(f"\n\n{WIDE}")
print("  §1  INTRADAY MICROSTRUCTURE — March vs Rest of Year")
print(f"{WIDE}")

for ticker in TICKERS:
    if ticker not in data:
        continue
    df    = data[ticker]
    march = df[df["is_march"]]
    other = df[~df["is_march"]]

    print(f"\n  {ticker} — March vs Rest of Year")
    print(f"  {SEP}")
    metrics = [
        ("Intraday range (%)",      "intraday_range"),
        ("Drop from open (%)",      "drop_from_open"),
        ("Rise from open (%)",      "rise_from_open"),
        ("Open→Close (%)",          "open_to_close"),
        ("Overnight return (%)",    "overnight_ret"),
        ("Volume ratio (vs 20d)",   "vol_ratio"),
        ("Gap from prev close (%)", "gap_pct"),
    ]
    print(f"  {'Metric':<28}  {'March':>10}  {'Other':>10}  {'Diff':>10}  {'Ratio':>8}")
    print(f"  {SEP}")
    for label, col in metrics:
        m_val = march[col].mean()
        o_val = other[col].mean()
        diff  = m_val - o_val
        ratio = m_val / o_val if o_val != 0 else np.nan
        flag  = "  ◄" if abs(ratio - 1) > 0.2 else ""
        print(f"  {label:<28}  {m_val:>+10.3f}  {o_val:>+10.3f}  "
              f"{diff:>+10.3f}  {ratio:>8.2f}x{flag}")

    # March week-of-month pattern
    print(f"\n  {ticker} — March by Week")
    print(f"  {'Week':>6}  {'N':>4}  {'AvgRet%':>8}  {'TotalRet%':>10}  {'WinRate':>8}  {'VolRatio':>9}")
    for week in [1, 2, 3, 4, 5]:
        sub = march[march["week_of_month"] == week]
        if len(sub) < 2:
            continue
        tot = float((1 + sub["ret"]).prod() - 1) * 100
        wr  = (sub["ret"] > 0).mean()
        print(f"  Week {week}  {len(sub):>4}  {sub['ret'].mean()*100:>+8.3f}  "
              f"{tot:>+10.1f}%  {wr:>8.1%}  {sub['vol_ratio'].mean():>9.2f}")

    # This March (current year)
    this_march = df[(df["year"] == datetime.now().year) & (df["month"] == 3)]
    if len(this_march) > 0:
        print(f"\n  {ticker} — March {datetime.now().year} Day-by-Day Log")
        print(f"  {SEP}")
        print(f"  {'Date':>12}  {'Close':>7}  {'Ret%':>7}  {'Range%':>7}  "
              f"{'VolRatio':>9}  {'Gap%':>7}  {'OvNight%':>10}")
        for date, row in this_march.iterrows():
            flag = "  *** BIG" if abs(row["ret"]) > 0.10 else ("  **VOL" if row["vol_ratio"] > 2 else "")
            print(f"  {str(date.date()):>12}  ${row['close']:>6.3f}  "
                  f"{row['ret']*100:>+7.2f}  {row['intraday_range']:>7.2f}  "
                  f"{row['vol_ratio']:>9.2f}  {row['gap_pct']:>+7.2f}  "
                  f"{row['overnight_ret']:>+10.2f}{flag}")
        tot_m = float((1 + this_march["ret"].dropna()).prod() - 1) * 100
        print(f"\n  MTD return: {tot_m:+.1f}%  |  days: {len(this_march)}  "
              f"|  win: {(this_march['ret'] > 0).mean():.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# §2  STATISTICAL SIGNIFICANCE  (+ Outlier-Robustness Check)
# ─────────────────────────────────────────────────────────────────────────────

def significance_tests(returns: pd.Series, labels: pd.Series,
                       n_permutations: int = 5000, label_str: str = "signal") -> dict:
    """Permutation test + Mann-Whitney U on labelled returns."""
    signal_rets = returns[labels].dropna()
    null_rets   = returns[~labels].dropna()
    if len(signal_rets) < 5:
        return {"error": "Insufficient sample size"}

    _, p_t  = stats.ttest_1samp(signal_rets, 0)
    _, p_mw = stats.mannwhitneyu(signal_rets, null_rets, alternative="greater")

    observed_mean = signal_rets.mean()
    combined      = returns.dropna().values
    n_signal      = len(signal_rets)
    perm_means    = []
    rng = np.random.default_rng(42)
    for _ in range(n_permutations):
        idx = rng.permutation(len(combined))
        perm_means.append(combined[idx[:n_signal]].mean())
    p_perm = (np.array(perm_means) >= observed_mean).mean()

    boot_sharpes = []
    for _ in range(2000):
        resample = rng.choice(signal_rets.values, size=len(signal_rets), replace=True)
        if np.std(resample) > 0:
            boot_sharpes.append(np.mean(resample) / np.std(resample) * np.sqrt(TRADING_DAYS))

    ci = (np.percentile(boot_sharpes, 2.5), np.percentile(boot_sharpes, 97.5)) if boot_sharpes else (0, 0)
    return {
        "n":              n_signal,
        "mean_pct":       observed_mean * 100,
        "p_ttest":        p_t,
        "p_mann_whitney": p_mw,
        "p_permutation":  p_perm,
        "sharpe_95ci":    ci,
    }


print(f"\n\n{WIDE}")
print("  §2  STATISTICAL SIGNIFICANCE — March Seasonality")
print(f"{WIDE}")

for ticker in TICKERS:
    if ticker not in data:
        continue
    df   = data[ticker]
    rets = df["ret"].dropna()
    mask = (df.loc[rets.index, "is_march"])

    print(f"\n  {ticker}")
    print(f"  {SEP}")

    # Full-sample tests
    res = significance_tests(rets, mask)
    if "error" in res:
        print(f"  Error: {res['error']}")
        continue
    print(f"  Full sample  |  N={res['n']}  |  mean={res['mean_pct']:+.3f}%  |  "
          f"P(perm)={res['p_permutation']:.4f}  |  P(MW-U)={res['p_mann_whitney']:.4f}")
    print(f"  Bootstrap Sharpe 95% CI: [{res['sharpe_95ci'][0]:+.2f}, {res['sharpe_95ci'][1]:+.2f}]")

    # Interpretation of divergence
    perm_sig = res["p_permutation"] < 0.05
    mwu_sig  = res["p_mann_whitney"] < 0.05
    if perm_sig and not mwu_sig:
        print(f"  ⚠  DIVERGENCE: permutation significant, MWU not → likely 2-3 extreme outlier days")
        print(f"     The mean is inflated by outliers; rank distribution is ordinary.")
        print(f"     ↓ Running outlier-robustness check…")

    # ── OUTLIER-ROBUSTNESS CHECK ───────────────────────────────────────────
    march_rets = rets[mask]
    top2_idx   = march_rets.nlargest(2).index
    rets_trim  = rets.drop(top2_idx)
    mask_trim  = mask.reindex(rets_trim.index).fillna(False)

    res_trim = significance_tests(rets_trim, mask_trim, label_str="March (excl. top-2)")
    if "error" not in res_trim:
        print(f"\n  Outlier-robust (drop top-2 March returns):")
        print(f"  Trimmed      |  N={res_trim['n']}  |  mean={res_trim['mean_pct']:+.3f}%  |  "
              f"P(perm)={res_trim['p_permutation']:.4f}  |  P(MW-U)={res_trim['p_mann_whitney']:.4f}")

        perm_trim_sig = res_trim["p_permutation"] < 0.05
        mwu_trim_sig  = res_trim["p_mann_whitney"] < 0.05
        if not perm_trim_sig and not mwu_sig:
            print(f"  ✗ VERDICT: Significance COLLAPSES after removing outliers.")
            print(f"     The March 'edge' is driven by 2-3 extreme events, not a structural seasonal.")
            print(f"     This pattern has ZERO predictive value for future Marches.")
        elif perm_trim_sig and mwu_trim_sig:
            print(f"  ✓ VERDICT: Significance HOLDS after removing outliers — genuine seasonality.")
        else:
            print(f"  ~ VERDICT: Mixed — weak structural component, dominated by tail events.")

    # Top-2 outlier dates for context
    print(f"\n  Top-2 March return days:")
    for d in top2_idx:
        print(f"    {d.date()}  ret={rets[d]*100:+.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# §3  TRANSACTION COST ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def estimate_spreads(df: pd.DataFrame, ticker: str = "") -> dict:
    """
    Roll model + Corwin-Schultz proxy.
    Roll model fails silently when serial autocorrelation is positive (momentum regimes).
    This function detects and flags that condition.
    """
    rets = df["close"].pct_change().dropna()

    # Serial autocorrelation check
    cov_serial = rets.cov(rets.shift(1))
    autocorr   = rets.autocorr(lag=1)
    roll_valid = cov_serial < 0   # Roll requires negative serial covariance

    if roll_valid:
        roll_spread = 2 * np.sqrt(-cov_serial)
        roll_note   = "valid"
    else:
        roll_spread = np.nan
        roll_note   = f"INVALID (positive autocorr={autocorr:+.3f} → momentum regime, covariance={cov_serial:.6f})"

    # Corwin-Schultz proxy (high-low estimator)
    high    = df["high"]
    low     = df["low"]
    khl     = np.log(high / low) ** 2
    cs_proxy = float(np.sqrt(khl.rolling(2).mean()).mean() * 0.5)

    conservative = max(x for x in [roll_spread if roll_valid else 0, cs_proxy, 0.02] if not np.isnan(x))
    return {
        "roll_spread_pct":       roll_spread * 100 if roll_valid else np.nan,
        "roll_note":             roll_note,
        "autocorr_lag1":         autocorr,
        "cs_spread_proxy_pct":   cs_proxy * 100,
        "conservative_cost_rt_pct": conservative * 100,
    }


def evaluate_overnight_viability(df: pd.DataFrame, ticker: str, cost_rt: float,
                                  threshold: float, test_thresholds: list | None = None) -> None:
    """Reports gross/net overnight edge viability vs realistic costs."""
    df = df.copy()
    df["signal"]    = df["drop_from_open"] >= threshold
    df["net_ret_pct"] = df["overnight_ret"] / 100 - cost_rt

    trades = df[df["signal"]].dropna(subset=["next_open"])
    if len(trades) < 5:
        print(f"  {ticker}: insufficient signal days at {threshold:.0f}% threshold")
        return

    gross_mean = trades["overnight_ret"].mean() / 100
    net_mean   = trades["net_ret_pct"].mean()
    sharpe_net = net_mean / trades["net_ret_pct"].std() * np.sqrt(TRADING_DAYS) \
                 if trades["net_ret_pct"].std() > 0 else 0

    print(f"  {ticker} | threshold≥{threshold:.0f}% | N={len(trades)}  "
          f"gross={gross_mean*100:+.3f}%  net={net_mean*100:+.3f}%  "
          f"Sharpe(net)={sharpe_net:+.2f}")
    viable = net_mean > 0
    print(f"  {'✓ VIABLE' if viable else '✗ NOT VIABLE'} at {cost_rt*100:.2f}% RT cost")

    if not viable:
        # Find minimum threshold that clears cost hurdle
        thresholds_to_test = test_thresholds or [5, 8, 10, 12, 15, 20, 25]
        print(f"  Searching for viable threshold…")
        for th in thresholds_to_test:
            t = df[df["drop_from_open"] >= th].dropna(subset=["next_open"])
            if len(t) < 5:
                continue
            n_mean = t["overnight_ret"].mean() / 100 - cost_rt
            if n_mean > 0:
                print(f"  → First viable threshold: ≥{th}%  net={n_mean*100:+.3f}%  N={len(t)}")
                break
        else:
            print(f"  → No threshold up to {max(thresholds_to_test)}% clears cost hurdle")


print(f"\n\n{WIDE}")
print("  §3  TRANSACTION COST ESTIMATION")
print(f"{WIDE}")

for ticker in TICKERS:
    if ticker not in data:
        continue
    costs = estimate_spreads(data[ticker], ticker)
    print(f"\n  {ticker}")
    print(f"  {SEP}")
    print(f"  Roll spread:    "
          + (f"{costs['roll_spread_pct']:.3f}%" if not np.isnan(costs['roll_spread_pct'])
             else f"N/A — {costs['roll_note']}"))
    print(f"  Serial autocorr (lag-1): {costs['autocorr_lag1']:+.4f}")
    print(f"  CS spread proxy: {costs['cs_spread_proxy_pct']:.3f}%")
    print(f"  Conservative RT: {costs['conservative_cost_rt_pct']:.2f}%  "
          f"(floor: 2% for microcaps)")

print(f"\n  Overnight Edge Viability:")
print(f"  {SEP}")
for ticker in TICKERS:
    if ticker not in data:
        continue
    evaluate_overnight_viability(data[ticker], ticker, COST_RT[ticker],
                                  OE_THRESHOLD[ticker])


# ─────────────────────────────────────────────────────────────────────────────
# §4  FUNDAMENTAL FLOOR MODELS
# ─────────────────────────────────────────────────────────────────────────────

def eonr_floor(wti: float, hedge_pct: float = 0.12,
               production_bpd: float = EONR_PRODUCTION) -> float:
    hedged   = hedge_pct * production_bpd
    unhedged = (1 - hedge_pct) * production_bpd
    ebitda   = (hedged   * (EONR_HEDGE_PRICE - EONR_LIFTING_COST) +
                unhedged * (wti              - EONR_LIFTING_COST)) * 365
    ebitda   = max(ebitda, 0)
    return (ebitda * EONR_EV_MULTIPLE) / EONR_SHARES


def tpet_war_premium(df: pd.DataFrame, uso_df: pd.DataFrame) -> dict:
    tpet_ret, uso_ret = df["close"].pct_change().align(
        uso_df["close"].pct_change(), join="inner"
    )
    valid = tpet_ret.dropna().index.intersection(uso_ret.dropna().index)
    t = tpet_ret.loc[valid]
    u = uso_ret.loc[valid]
    beta     = t.cov(u) / u.var() if u.var() > 0 else np.nan
    excess5  = (t.tail(5) - u.tail(5)).mean()
    base5    = t.tail(5).mean()
    premium  = (excess5 / base5 * 100) if (beta is not None and beta > 1.5 and base5 != 0) else 0.0
    return {"beta": beta, "estimated_premium_pct": premium}


print(f"\n\n{WIDE}")
print("  §4  FUNDAMENTAL FLOOR MODELS")
print(f"{WIDE}")

# EONR sensitivity table
print(f"\n  EONR — Floor Sensitivity Table  (production={EONR_PRODUCTION} bpd)")
print(f"  {SEP}")
hedge_levels  = [0.12, 0.25, 0.50]
wti_scenarios = [55, 60, 65, 70, 75, 80, 85]

# Header
print(f"  {'WTI $':>7}", end="")
for h in hedge_levels:
    print(f"  {'Hedge '+str(int(h*100))+'%':>12}", end="")
print()
print(f"  {SEP}")
for wti in wti_scenarios:
    print(f"  {wti:>6.0f}", end="")
    for h in hedge_levels:
        floor = eonr_floor(wti, hedge_pct=h)
        note  = "  ← hedge floor" if wti == EONR_HEDGE_PRICE else ""
        print(f"  ${floor:>10.3f}", end="")
    print()

print(f"\n  Note: at WTI ≤ ${EONR_LIFTING_COST:.0f}/bbl lifting cost, unhedged production is cash-negative.")
print(f"  The $70/bbl hedge only protects the hedged portion; unhedged 88% has full downside exposure.")

# EONR current price vs floor
if "EONR" in data:
    current_eonr = data["EONR"]["close"].iloc[-1]
    wti_current  = data["CL=F"]["close"].iloc[-1] if "CL=F" in data else 72.50
    floor_current = eonr_floor(wti_current)
    premium_to_floor = (current_eonr / floor_current - 1) * 100 if floor_current > 0 else np.nan
    print(f"\n  Current EONR price: ${current_eonr:.3f}  |  WTI: ${wti_current:.2f}")
    print(f"  Estimated fundamental floor: ${floor_current:.3f}")
    print(f"  Premium above floor: {premium_to_floor:+.1f}%  "
          + ("(above floor — war/optionality premium)" if premium_to_floor > 0 else "(BELOW floor — possibly mispriced)"))

# TPET war premium
print(f"\n  TPET — War Premium Decomposition")
print(f"  {SEP}")
if "TPET" in data and not uso_df.empty:
    wp = tpet_war_premium(data["TPET"], uso_df)
    print(f"  Current 5d beta to USO: {wp['beta']:+.3f}x  (normal: ~0.57x)")
    print(f"  Estimated war premium:  {wp['estimated_premium_pct']:+.1f}%")
    if wp["beta"] is not None:
        if wp["beta"] < 1.0:
            print(f"  STATUS: 🟢 NORMALIZED — war premium fully unwound")
        elif wp["beta"] < 2.0:
            print(f"  STATUS: 🟠 FADING — premium still unwinding")
        else:
            print(f"  STATUS: 🔴 ELEVATED — war premium intact")


# ─────────────────────────────────────────────────────────────────────────────
# §5  TAIL RISK METRICS & KELLY POSITION SIZING
# ─────────────────────────────────────────────────────────────────────────────

def compute_tail_risk(returns: pd.Series, confidence: float = 0.95) -> dict:
    rets = returns.dropna()
    var  = np.percentile(rets, (1 - confidence) * 100)
    cvar = rets[rets <= var].mean()

    downside_std = rets[rets < 0].std() * np.sqrt(TRADING_DAYS)
    sortino      = (rets.mean() * TRADING_DAYS) / downside_std if downside_std > 0 else 0

    # Omega ratio
    threshold   = 0.0
    gains       = rets[rets > threshold] - threshold
    losses      = threshold - rets[rets <= threshold]
    omega       = gains.sum() / losses.sum() if losses.sum() > 0 else np.nan

    # Kelly criterion (full Kelly and half-Kelly)
    wins   = rets[rets > 0]
    losses_s = rets[rets <= 0]
    if len(wins) > 0 and len(losses_s) > 0:
        p        = len(wins) / len(rets)
        avg_win  = wins.mean()
        avg_loss = abs(losses_s.mean())
        b        = avg_win / avg_loss  # win/loss ratio
        kelly    = (p * b - (1 - p)) / b  # Kelly fraction
        half_kelly = kelly / 2
    else:
        kelly = half_kelly = np.nan

    return {
        "VaR_95":     var * 100,
        "CVaR_95":    cvar * 100,
        "Sortino":    sortino,
        "Omega":      omega,
        "Kelly":      kelly,
        "HalfKelly":  half_kelly,
        "ann_vol":    rets.std() * np.sqrt(TRADING_DAYS) * 100,
        "ann_ret":    rets.mean() * TRADING_DAYS * 100,
    }


print(f"\n\n{WIDE}")
print("  §5  TAIL RISK METRICS & KELLY POSITION SIZING")
print(f"{WIDE}")

for ticker in TICKERS:
    if ticker not in data:
        continue
    risk = compute_tail_risk(data[ticker]["ret"])
    print(f"\n  {ticker}")
    print(f"  {SEP}")
    print(f"  Annualised return:   {risk['ann_ret']:+.1f}%")
    print(f"  Annualised vol:       {risk['ann_vol']:.1f}%")
    print(f"  VaR (95%):           {risk['VaR_95']:+.2f}%  per day")
    print(f"  CVaR (95%):          {risk['CVaR_95']:+.2f}%  per day  "
          f"(avg loss on worst 5% of days)")
    print(f"  Sortino ratio:        {risk['Sortino']:+.2f}  "
          f"({'poor' if risk['Sortino'] < 0.5 else 'ok' if risk['Sortino'] < 1.0 else 'good'})")
    print(f"  Omega ratio:          {risk['Omega']:.3f}" if not np.isnan(risk["Omega"]) else "  Omega:  N/A")

    print(f"\n  Kelly Position Sizing:")
    if not np.isnan(risk["Kelly"]):
        kelly_pct     = risk["Kelly"] * 100
        half_kelly_pct = risk["HalfKelly"] * 100
        print(f"  Full Kelly:   {kelly_pct:+.1f}% of capital")
        print(f"  Half Kelly:   {half_kelly_pct:+.1f}% of capital  ← recommended")
        if kelly_pct < 0:
            print(f"  ✗ NEGATIVE KELLY — expected value is negative; do NOT take this position")
        elif kelly_pct < 5:
            print(f"  ⚠  Very small Kelly fraction — risk/reward barely positive")
        elif kelly_pct > 50:
            print(f"  ⚠  Large Kelly fraction — highly concentrated; use half-Kelly")
    else:
        print(f"  Insufficient return history for Kelly computation")

    # Compare to S&P 500 benchmark context
    print(f"\n  Context: S&P 500 typical Sortino ~0.8–1.2, daily VaR(95) ~-1.5%")
    print(f"  {ticker} has {abs(risk['VaR_95'])/1.5:.1f}x the daily tail risk of the S&P 500 per unit of return.")


# ─────────────────────────────────────────────────────────────────────────────
# §6  PAIRS COINTEGRATION (OU Half-Life + Cost-Adjusted Threshold)
# ─────────────────────────────────────────────────────────────────────────────

def ornstein_uhlenbeck_halflife(spread: pd.Series) -> float:
    """
    Fits an AR(1) on Δspread_t ~ β * spread_{t-1} + ε and computes
    half-life = −ln(2) / β.  Half-life < 0 means mean-diverging — not tradeable.
    """
    spread_clean = spread.dropna()
    delta        = spread_clean.diff().dropna()
    lagged       = spread_clean.shift(1).dropna()
    aligned      = delta.align(lagged, join="inner")
    d, l_        = aligned
    beta         = sm.OLS(d, sm.add_constant(l_)).fit().params.iloc[1]
    if beta >= 0:
        return np.nan   # diverging
    return -np.log(2) / beta


def analyze_pairs(df1: pd.DataFrame, df2: pd.DataFrame,
                  cost_tpet: float = COST_RT["TPET"],
                  cost_eonr: float = COST_RT["EONR"]) -> dict:
    p1, p2 = df1["close"].align(df2["close"], join="inner")
    p1, p2 = p1.dropna(), p2.dropna()
    common  = p1.index.intersection(p2.index)
    p1, p2  = p1.loc[common], p2.loc[common]

    # Engle-Granger cointegration
    score, p_value, _ = coint(p1, p2)

    # Fitted OLS hedge ratio
    ols     = sm.OLS(p1, sm.add_constant(p2)).fit()
    hedge   = ols.params.iloc[1]
    spread  = p1 - hedge * p2
    z_score = (spread.iloc[-1] - spread.mean()) / spread.std()

    # OU half-life
    halflife = ornstein_uhlenbeck_halflife(spread)

    # Cost-adjusted minimum z-score for entry
    # Each standard deviation of the spread must be large enough to cover both legs' costs
    # Minimum gross move needed = cost_tpet + cost_eonr (one-way each leg, round-trip one)
    total_cost_pct  = cost_tpet + cost_eonr   # total RT cost for both legs
    spread_std_pct  = spread.pct_change().std() * np.sqrt(TRADING_DAYS)  # annualised
    # For a 1-standard-deviation move to cover costs in `halflife` days:
    if halflife and not np.isnan(halflife) and halflife > 0:
        # Daily spread std
        daily_spread_std = spread.std()
        # We need spread_move > total_cost * current price level
        price_level = p1.mean()
        min_spread_move = total_cost_pct * price_level
        z_threshold = min_spread_move / daily_spread_std if daily_spread_std > 0 else np.nan
    else:
        z_threshold = np.nan

    return {
        "cointegration_p": p_value,
        "cointegration_score": score,
        "hedge_ratio": hedge,
        "spread_mean": spread.mean(),
        "spread_std": spread.std(),
        "current_z_score": z_score,
        "halflife_days": halflife,
        "cost_adjusted_z_threshold": z_threshold,
    }


print(f"\n\n{WIDE}")
print("  §6  PAIRS COINTEGRATION — TPET / EONR")
print(f"{WIDE}")

if "TPET" in data and "EONR" in data:
    pr = analyze_pairs(data["TPET"], data["EONR"])
    c_status = "COINTEGRATED" if pr["cointegration_p"] < 0.05 else "NOT COINTEGRATED"

    print(f"\n  Engle-Granger: {c_status}  (p={pr['cointegration_p']:.4f}, score={pr['cointegration_score']:.3f})")
    print(f"  Hedge ratio:   {pr['hedge_ratio']:.4f}x  (long 1 TPET, short {pr['hedge_ratio']:.4f} EONR)")
    print(f"  Spread mean:   ${pr['spread_mean']:.4f}  |  std: ${pr['spread_std']:.4f}")
    print(f"  Current z-score: {pr['current_z_score']:+.3f}")

    if pr["current_z_score"] < -2.0:
        print(f"  Signal: LONG TPET / SHORT EONR  (spread at discount)")
    elif pr["current_z_score"] > 2.0:
        print(f"  Signal: SHORT TPET / LONG EONR  (spread at premium)")
    else:
        print(f"  Signal: NEUTRAL (within ±2σ band)")

    print(f"\n  OU Half-Life:")
    if pr["halflife_days"] and not np.isnan(pr["halflife_days"]):
        hl = pr["halflife_days"]
        print(f"  {hl:.1f} trading days  to half-reversion")
        total_cost = COST_RT["TPET"] + COST_RT["EONR"]
        print(f"  Total RT cost both legs: {total_cost*100:.2f}%")
        print(f"  Spread must move >{total_cost*100:.2f}% in ~{hl:.0f} days to break even")
        if hl > 20:
            print(f"  ⚠  Long half-life ({hl:.0f}d) → high path-risk; spread can stay dislocated")
        elif hl < 5:
            print(f"  ✓  Short half-life ({hl:.0f}d) → fast mean-reversion, lower path-risk")
    else:
        print(f"  ✗ Spread appears DIVERGING — no reliable OU process, do NOT trade this pair")

    print(f"\n  Cost-Adjusted Entry Threshold:")
    if pr["cost_adjusted_z_threshold"] and not np.isnan(pr["cost_adjusted_z_threshold"]):
        threshold_z = pr["cost_adjusted_z_threshold"]
        current_z   = abs(pr["current_z_score"])
        print(f"  Minimum |z| for entry after costs: {threshold_z:.2f}σ")
        print(f"  Current |z|: {current_z:.2f}σ")
        if current_z >= threshold_z:
            print(f"  ✓ Current z-score CLEARS the cost-adjusted threshold")
        else:
            print(f"  ✗ Current z-score does NOT clear cost-adjusted threshold — wait for wider spread")
    else:
        print(f"  Cannot compute threshold (diverging spread or insufficient data)")

    # Rolling cointegration stability
    print(f"\n  Rolling Cointegration Stability (90-day windows):")
    p1, p2 = data["TPET"]["close"].align(data["EONR"]["close"], join="inner")
    p1, p2 = p1.dropna(), p2.dropna()
    window = 90
    results_roll = []
    for i in range(window, len(p1)):
        s1 = p1.iloc[i - window:i]
        s2 = p2.iloc[i - window:i]
        try:
            _, pv, _ = coint(s1, s2)
            results_roll.append((p1.index[i], pv))
        except Exception:
            pass
    if results_roll:
        roll_df  = pd.DataFrame(results_roll, columns=["date", "p_coint"])
        pct_sig  = (roll_df["p_coint"] < 0.05).mean()
        recent_p = roll_df.tail(30)["p_coint"].mean()
        print(f"  % of 90d windows cointegrated (p<0.05): {pct_sig:.1%}")
        print(f"  Recent 30-window avg p-value: {recent_p:.4f}")
        if pct_sig < 0.5:
            print(f"  ⚠  Relationship is UNSTABLE — cointegration holds in <50% of windows")
        else:
            print(f"  ✓  Relationship broadly stable across rolling windows")


# ─────────────────────────────────────────────────────────────────────────────
# §7  CATALYST HUNTING — Big March Move Days + News
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n\n{WIDE}")
print("  §7  CATALYST HUNTING — Big Move Days")
print(f"{WIDE}")

for ticker in TICKERS:
    if ticker not in data:
        continue
    df    = data[ticker]
    march = df[df["is_march"]].copy()

    print(f"\n  {ticker} — All March Days by Absolute Return (top 20)")
    print(f"  {SEP}")
    print(f"  {'Date':>12}  {'Ret%':>7}  {'Gap%':>7}  {'Range%':>8}  "
          f"{'VolRatio':>9}  {'OvNight%':>10}")
    for date, row in march.dropna(subset=["ret"]).sort_values("ret", key=abs, ascending=False).head(20).iterrows():
        vol_flag = " HIGH-VOL" if row["vol_ratio"] > 2 else ""
        gap_flag = f" GAP{row['gap_pct']:+.1f}%" if abs(row["gap_pct"]) > 3 else ""
        print(f"  {str(date.date()):>12}  {row['ret']*100:>+7.2f}  "
              f"{row['gap_pct']:>+7.2f}  {row['intraday_range']:>8.2f}  "
              f"{row['vol_ratio']:>9.2f}  {row['overnight_ret']:>+10.2f}"
              f"  {'▲' if row['ret'] > 0 else '▼'}{vol_flag}{gap_flag}")

    # News
    print(f"\n  {ticker} — Recent March News")
    print(f"  {SEP}")
    try:
        tk   = yf.Ticker(ticker)
        news = tk.news or []
        march_news = []
        for item in news:
            ts = item.get("providerPublishTime", 0)
            if ts:
                dt = pd.Timestamp(ts, unit="s")
                if dt.month == 3:
                    march_news.append({
                        "date": dt.date(),
                        "title": item.get("title", "")[:80],
                        "pub":   item.get("publisher", "")[:15],
                    })
        if march_news:
            for n in sorted(march_news, key=lambda x: x["date"], reverse=True)[:15]:
                print(f"  {n['date']}  [{n['pub']:<15}]  {n['title']}")
        else:
            print(f"  No March news in recent feed")
    except Exception as e:
        print(f"  News fetch failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# §8  CROSS-TICKER CORRELATION
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n\n{WIDE}")
print("  §8  CROSS-TICKER CORRELATION")
print(f"{WIDE}")

all_tickers = TICKERS + [b for b in BENCHMARKS if b in data]
ret_df = pd.DataFrame({t: data[t]["ret"] for t in all_tickers if t in data}).dropna(how="all")

for label, mask in [("MARCH only", ret_df.index.month == 3),
                    ("REST OF YEAR", ret_df.index.month != 3)]:
    sub  = ret_df.loc[mask].dropna(how="all")
    corr = sub.corr()
    print(f"\n  Correlation Matrix — {label}  (N={len(sub)})")
    print(f"  {SEP}")
    cols = corr.columns.tolist()
    print(f"  {'':>6}", end="")
    for c in cols:
        print(f"  {c:>7}", end="")
    print()
    for row_t in corr.index:
        print(f"  {row_t:>6}", end="")
        for col_t in cols:
            val = corr.loc[row_t, col_t]
            print(f"  {val:>7.3f}", end="")
        print()

# TPET vs EONR lead-lag in March
if "TPET" in ret_df.columns and "EONR" in ret_df.columns:
    march_both = ret_df.loc[ret_df.index.month == 3, ["TPET", "EONR"]].dropna()
    print(f"\n  TPET ↔ EONR Lead-Lag Analysis — March")
    print(f"  {SEP}")
    print(f"  {'Lag':>5}  {'TPET leads EONR':>20}  {'EONR leads TPET':>20}")
    for lag in range(0, 6):
        c1 = march_both["TPET"].shift(lag).corr(march_both["EONR"]) if lag > 0 else march_both["TPET"].corr(march_both["EONR"])
        c2 = march_both["EONR"].shift(lag).corr(march_both["TPET"]) if lag > 0 else c1
        print(f"  {lag:>5}  {c1:>20.4f}  {c2:>20.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# §9  MONTHLY REGIME ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n\n{WIDE}")
print("  §9  MONTHLY REGIME ANALYSIS")
print(f"{WIDE}")

for ticker in TICKERS:
    if ticker not in data:
        continue
    df = data[ticker]
    print(f"\n  {ticker} — Monthly Return Regimes (all history)")
    print(f"  {SEP}")
    print(f"  {'Month':>8}  {'N':>4}  {'AvgRet%':>8}  {'TotRet%':>9}  "
          f"{'WinRate':>8}  {'AvgRange%':>10}  {'AvgVolR':>8}  {'Sharpe':>7}")
    for month in range(1, 13):
        sub = df[df["month"] == month].dropna(subset=["ret"])
        if len(sub) < 5:
            continue
        tot = float((1 + sub["ret"]).prod() - 1) * 100
        wr  = (sub["ret"] > 0).mean()
        sh  = sub["ret"].mean() / sub["ret"].std() * np.sqrt(TRADING_DAYS) if sub["ret"].std() > 0 else 0
        flag = "  ◄ MARCH" if month == 3 else ""
        print(f"  {MONTH_NAMES[month]:>8}  {len(sub):>4}  "
              f"{sub['ret'].mean()*100:>+8.3f}  {tot:>+9.1f}%  "
              f"{wr:>8.1%}  {sub['intraday_range'].mean():>10.3f}  "
              f"{sub['vol_ratio'].mean():>8.3f}  {sh:>7.2f}{flag}")

    # March year-by-year
    print(f"\n  {ticker} — March by Year")
    print(f"  {SEP}")
    print(f"  {'Year':>6}  {'N':>4}  {'TotRet%':>9}  {'WinRate':>8}  {'Sharpe':>7}")
    for year in sorted(df["year"].unique()):
        sub = df[(df["year"] == year) & (df["month"] == 3)].dropna(subset=["ret"])
        if len(sub) < 3:
            continue
        tot = float((1 + sub["ret"]).prod() - 1) * 100
        wr  = (sub["ret"] > 0).mean()
        sh  = sub["ret"].mean() / sub["ret"].std() * np.sqrt(TRADING_DAYS) if sub["ret"].std() > 0 else 0
        flag = "  ◄ THIS YEAR" if year == datetime.now().year else ""
        print(f"  {year:>6}  {len(sub):>4}  {tot:>+9.1f}%  {wr:>8.1%}  {sh:>7.2f}{flag}")


# ─────────────────────────────────────────────────────────────────────────────
# §10  OVERNIGHT EDGE DEEP DIVE
# ─────────────────────────────────────────────────────────────────────────────

def overnight_deep_dive(ticker: str, threshold: float, df: pd.DataFrame) -> None:
    cost = (COST_RT[ticker])
    df   = df.copy()
    df["signal"]        = df["drop_from_open"] >= threshold
    df["net_ret"]       = df["overnight_ret"] / 100 - cost
    df["gross_ret"]     = df["overnight_ret"] / 100
    df["close_in_range_val"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)

    trades = df[df["signal"]].dropna(subset=["next_open"])
    if len(trades) == 0:
        print(f"  {ticker}: no trades at {threshold:.0f}% threshold")
        return

    rets  = trades["net_ret"]
    wins  = (rets > 0).sum()
    total = float((1 + rets).prod() - 1)
    wr    = wins / len(rets)
    avg_w = rets[rets > 0].mean() if wins > 0 else 0.0
    avg_l = rets[rets <= 0].mean() if (len(rets) - wins) > 0 else 0.0
    sh    = rets.mean() / rets.std() * np.sqrt(TRADING_DAYS) if rets.std() > 0 else 0
    equity = (1 + rets).cumprod()
    mdd    = float(((equity - equity.cummax()) / equity.cummax()).min())

    print(f"\n  {WIDE[:65]}")
    print(f"  {ticker} — Drop ≥ {threshold:.0f}%  |  RT cost = {cost*100:.2f}%")
    print(f"  {WIDE[:65]}")
    print(f"  Trades: {len(rets)}  |  Win: {wr:.1%}  |  EV: {(wr*avg_w+(1-wr)*avg_l)*100:+.3f}%")
    print(f"  Total: {total*100:+.1f}%  |  Sharpe: {sh:+.2f}  |  Max DD: {mdd*100:.1f}%")

    # Threshold sweep
    print(f"\n  {'Thresh':>7}  {'N':>5}  {'WinR':>6}  {'Gross%':>8}  {'Net%':>8}  {'Sharpe':>7}  {'Total%':>9}")
    for th in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30]:
        t = df[df["drop_from_open"] >= th].dropna(subset=["next_open"])
        if len(t) < 5:
            continue
        g = t["gross_ret"]
        n = t["net_ret"]
        sh_t = n.mean() / n.std() * np.sqrt(TRADING_DAYS) if n.std() > 0 else 0
        tot_t = float((1 + n).prod() - 1) * 100
        mark = "  ◄" if th == threshold else ""
        print(f"  {th:>6.0f}%  {len(t):>5}  {(n>0).mean():>6.1%}  "
              f"{g.mean()*100:>+8.3f}  {n.mean()*100:>+8.3f}  "
              f"{sh_t:>7.2f}  {tot_t:>+9.1f}%{mark}")

    # Year-by-year
    print(f"\n  Year-by-year:")
    print(f"  {'Year':>6}  {'N':>4}  {'WinR':>6}  {'AvgRet%':>8}  {'Total%':>10}  {'Sharpe':>7}")
    for year, grp in trades.groupby(trades.index.year):
        r = grp["net_ret"]
        sh_y = r.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else 0
        tot_y = float((1 + r).prod() - 1) * 100
        print(f"  {year:>6}  {len(r):>4}  {(r>0).mean():>6.1%}  "
              f"{r.mean()*100:>+8.3f}  {tot_y:>+10.1f}%  {sh_y:>7.2f}")


print(f"\n\n{WIDE}")
print("  §10  OVERNIGHT EDGE DEEP DIVE")
print(f"{WIDE}")

for ticker, threshold in OE_THRESHOLD.items():
    if ticker in data:
        overnight_deep_dive(ticker, threshold, data[ticker])


# ─────────────────────────────────────────────────────────────────────────────
# §11  FORWARD SIGNAL MONITOR
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n\n{WIDE}")
print("  §11  FORWARD SIGNAL MONITOR")
print(f"  Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{WIDE}")

wti_df = data.get("CL=F", pd.DataFrame())

for ticker in TICKERS:
    if ticker not in data:
        continue
    df     = data[ticker]
    latest = df.iloc[-1]
    row    = latest.to_dict()

    print(f"\n\n  {'═'*68}")
    print(f"  {ticker} — CURRENT STATE & FORWARD SIGNALS")
    print(f"  As of: {df.index[-1].date()}")
    print(f"  {'═'*68}")

    print(f"\n  CURRENT STATE")
    print(f"  {SEP}")
    print(f"  Price:          ${row['close']:.3f}")
    print(f"  Today return:   {row['ret']*100:+.2f}%")
    print(f"  Intraday range: {row['intraday_range']:.1f}%  "
          f"(drop={row['drop_from_open']:.1f}%  rise={row['rise_from_open']:.1f}%)")
    print(f"  Volume ratio:   {row['vol_ratio']:.2f}x  (vs 20d avg)")
    print(f"  Close in range: {row.get('close_in_range', np.nan):.2f}  (0=at low, 1=at high)")
    print(f"  Gap yesterday:  {row['gap_pct']:+.2f}%")
    if "uso_beta_5d" in row:
        print(f"  Beta USO (5d):  {row['uso_beta_5d']:+.2f}x  |  (20d): {row.get('uso_beta_20d', np.nan):+.2f}x")
    print(f"  Vol regime:     {row.get('vol_regime', np.nan):.2f}x  (>1=expanding)")
    print(f"  Momentum:       5d={row['mom_5d']:+.1f}%  10d={row['mom_10d']:+.1f}%  20d={row['mom_20d']:+.1f}%")

    # War premium section
    beta_5d = row.get("uso_beta_5d", np.nan)
    print(f"\n  WAR PREMIUM STATUS")
    print(f"  {SEP}")
    if not np.isnan(beta_5d):
        if beta_5d > 3.0:
            print(f"  🔴 EXTREME — war premium fully intact (beta={beta_5d:+.2f}x)")
        elif beta_5d > 2.0:
            print(f"  🟡 ELEVATED — war premium intact but moderating (beta={beta_5d:+.2f}x)")
        elif beta_5d > 1.0:
            print(f"  🟠 FADING — war premium unwinding (beta={beta_5d:+.2f}x)")
        else:
            print(f"  🟢 NORMALIZED — war premium gone; back to fundamental value (beta={beta_5d:+.2f}x)")
        print(f"  ⚠  Note: Phase 2 analysis shows estimated war premium ≈0%. "
              f"Signals below are cost-adjusted.")

    # Oil macro
    print(f"\n  OIL MACRO CONTEXT")
    print(f"  {SEP}")
    if "USO" in data:
        uso_l = data["USO"].iloc[-1]
        print(f"  USO:  ${uso_l['close']:.2f}  "
              f"5d={data['USO']['close'].pct_change(5).iloc[-1]*100:+.1f}%  "
              f"20d={data['USO']['close'].pct_change(20).iloc[-1]*100:+.1f}%")
    if not wti_df.empty:
        wti_l = wti_df.iloc[-1]
        print(f"  WTI:  ${wti_l['close']:.2f}  "
              f"5d={wti_df['close'].pct_change(5).iloc[-1]*100:+.1f}%")

    # Forward signals — cost-aware
    print(f"\n  FORWARD SIGNALS  (cost-adjusted: RT={COST_RT[ticker]*100:.2f}%)")
    print(f"  {SEP}")

    if ticker == "TPET":
        if row.get("uso_beta_5d", 0) > 2.0:
            print(f"  ✓ WAR PREMIUM INTACT — beta {row['uso_beta_5d']:.2f}x  (but Phase 2: premium already ~0%)")
        elif row.get("uso_beta_5d", 0) < 1.0:
            print(f"  ✗ WAR PREMIUM GONE — reduce/exit position")
        if row.get("vol_ratio", 0) > 3.0:
            print(f"  ✓ VOLUME ACCUMULATION — {row['vol_ratio']:.1f}x normal")
        elif row.get("vol_ratio", 0) < 0.3:
            print(f"  ✗ VOLUME DISTRIBUTION — {row['vol_ratio']:.1f}x normal")
        if row.get("month") == 3:
            wom = row.get("week_of_month", 0)
            sig = "✓" if wom <= 2 else "✗"
            print(f"  {sig} MARCH WEEK {wom}  (⚠ significance likely driven by outliers — see §2)")
        # Overnight setup — now explicitly notes cost hurdle
        drop = row.get("drop_from_open", 0)
        if drop >= OE_THRESHOLD["TPET"] and row.get("vol_ratio", 0) > 3.0:
            gross_hist = data["TPET"][data["TPET"]["drop_from_open"] >= OE_THRESHOLD["TPET"]]["overnight_ret"].mean() / 100
            net_hist   = gross_hist - COST_RT["TPET"]
            viability  = "✓ NET POSITIVE historically" if net_hist > 0 else "✗ NET NEGATIVE after costs"
            print(f"  {'✓' if net_hist > 0 else '✗'} OVERNIGHT SETUP — drop={drop:.1f}%  {viability} (avg net={net_hist*100:+.3f}%)")
        # Catalyst
        days_to_march31 = (pd.Timestamp("2026-03-31") - df.index[-1]).days
        if 0 <= days_to_march31 <= 14:
            print(f"  ⚡ CATALYST — Alberta well completion in {days_to_march31}d  (watch for 30-40 bbl/day)")

    elif ticker == "EONR":
        price = row.get("close", 0)
        wti_p = wti_df["close"].iloc[-1] if not wti_df.empty else 72.5
        floor = eonr_floor(wti_p)
        prem  = (price / floor - 1) * 100 if floor > 0 else np.nan
        if price <= 0.90:
            print(f"  ✓ ACCUMULATION ZONE — ${price:.3f} ≤ $0.90")
            print(f"    Fundamental floor @ WTI=${wti_p:.1f}: ${floor:.3f}  "
                  f"(current premium above floor: {prem:+.1f}%)")
        if row.get("vol_ratio", 0) > 3.0 and row.get("close_in_range", 0) > 0.6:
            print(f"  ✓ BREAKOUT SIGNAL — high volume + strong close")
        if row.get("month") in [4, 5, 6]:
            print(f"  ⚡ Q2 DRILLING — 92-well programme should begin (target +500 bbl/day)")
        drop = row.get("drop_from_open", 0)
        if drop >= OE_THRESHOLD["EONR"]:
            gross_hist = data["EONR"][data["EONR"]["drop_from_open"] >= OE_THRESHOLD["EONR"]]["overnight_ret"].mean() / 100
            net_hist   = gross_hist - COST_RT["EONR"]
            viability  = "✓ NET POSITIVE historically" if net_hist > 0 else "✗ NET NEGATIVE after costs"
            print(f"  {'✓' if net_hist > 0 else '✗'} OVERNIGHT SETUP (EONR ≥{OE_THRESHOLD['EONR']:.0f}%) — "
                  f"drop={drop:.1f}%  {viability}")
        if not wti_df.empty and wti_p > EONR_HEDGE_PRICE:
            print(f"  ✓ HEDGE ACTIVE — WTI ${wti_p:.0f} > $70 floor  "
                  f"(+${wti_p - EONR_HEDGE_PRICE:.0f}/bbl premium on hedged portion)")
        else:
            print(f"  ✓ HEDGE PROTECTING — WTI below $70 hedge floor active")

    # Recent 10 days
    print(f"\n  RECENT 10 DAYS")
    print(f"  {SEP}")
    print(f"  {'Date':>12}  {'Close':>7}  {'Ret%':>7}  {'VolR':>6}  "
          f"{'Beta5d':>7}  {'OvNight%':>10}")
    for date, row_h in df.tail(10).iterrows():
        flag = " ***" if abs(row_h["ret"]) > 0.10 else (" **" if row_h["vol_ratio"] > 2 else "")
        beta_val = f"{row_h.get('uso_beta_5d', np.nan):+.2f}" if "uso_beta_5d" in df.columns else "  N/A"
        print(f"  {str(date.date()):>12}  "
              f"${row_h['close']:>6.3f}  "
              f"{row_h['ret']*100:>+7.2f}  "
              f"{row_h['vol_ratio']:>6.2f}  "
              f"{beta_val:>7}  "
              f"{row_h['overnight_ret']:>+10.2f}{flag}")


# ─────────────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n\n{WIDE}")
print("  ANALYSIS COMPLETE")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"  Key reminders:")
print(f"  • March edge likely driven by outlier days — see §2 robustness check")
print(f"  • TPET Roll estimator unreliable during momentum; use CS spread (§3)")
print(f"  • Overnight edge net-negative at current RT costs unless threshold raised (§3/§10)")
print(f"  • EONR floor ~$0.38 at WTI $72; current price contains optionality/premium (§4)")
print(f"  • Kelly fraction is small/negative — position sizing must reflect tail risk (§5)")
print(f"  • Pairs trade: check OU half-life & cost-adjusted z-threshold before entry (§6)")
print(f"{WIDE}\n")
