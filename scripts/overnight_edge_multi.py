#!/usr/bin/env python3
"""
Overnight Mean-Reversion Edge — Oil Microcap Universe
──────────────────────────────────────────────────────
Rule:
  If intraday low falls ≥ threshold % below the day's open
  → BUY at the day's close
  → SELL at the NEXT day's open

Runs on: TPET, STAK , EONR, USEG, MXC, BRN, PED, REI, PRSO (oil microcap tradeable universe)
Reports per-ticker and combined pool statistics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

TICKERS         = ["TPET", "STAK", "EONR", "USEG", "MXC", "BRN", "PED", "REI", "PRSO"]
START           = "2018-01-01"
END             = None
THRESHOLD_PCT   = 5.0
TRANSACTION_PCT = 0.25    # Higher cost for microcaps (wider spreads)
TRADING_DAYS    = 252


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def analyse(rets: pd.Series, label: str, n_total_days: int) -> dict:
    n = len(rets)
    if n == 0:
        return {}

    wins   = (rets > 0).sum()
    losses = (rets <= 0).sum()
    wr     = wins / n
    avg_w  = rets[rets > 0].mean()  if wins   > 0 else 0.0
    avg_l  = rets[rets <= 0].mean() if losses > 0 else 0.0
    avg_r  = rets.mean()
    total  = float((1 + rets).prod() - 1)
    sharpe = (rets.mean() / rets.std() * np.sqrt(TRADING_DAYS)) if rets.std() > 0 else 0.0

    equity = (1 + rets).cumprod()
    rmax   = equity.cummax()
    max_dd = float((equity - rmax).div(rmax).min())
    ev     = wr * avg_w + (1 - wr) * avg_l
    freq   = n / n_total_days * 100

    return {
        "ticker": label, "n": n, "freq_pct": freq,
        "win_rate": wr, "avg_win": avg_w, "avg_loss": avg_l,
        "payoff": abs(avg_w / avg_l) if avg_l != 0 else float("inf"),
        "avg_ret": avg_r, "ev": ev, "total_ret": total,
        "sharpe": sharpe, "max_dd": max_dd,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Download & process
# ─────────────────────────────────────────────────────────────────────────────

cost = TRANSACTION_PCT / 100 * 2
results = []
all_trades: list[pd.DataFrame] = []

print(f"\nDownloading OHLC data for oil microcap universe…")
print(f"Tickers: {TICKERS}")
print(f"Period:  {START} → present")
print(f"Cost:    {TRANSACTION_PCT*2:.2f}% round-trip (microcap spread estimate)\n")

for ticker in TICKERS:
    try:
        raw = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
        if raw.empty:
            print(f"  {ticker}: no data, skipping")
            continue
    except Exception as e:
        print(f"  {ticker}: download error ({e}), skipping")
        continue

    # Handle MultiIndex columns from yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    if not all(c in raw.columns for c in ["Open", "High", "Low", "Close"]):
        print(f"  {ticker}: missing OHLC columns, skipping")
        continue

    df = raw[["Open", "High", "Low", "Close"]].copy()
    df.index = pd.to_datetime(df.index)
    df.columns = ["open", "high", "low", "close"]

    df["intraday_drop_pct"] = (df["open"] - df["low"]) / df["open"] * 100
    df["signal"]            = df["intraday_drop_pct"] >= THRESHOLD_PCT
    df["next_open"]         = df["open"].shift(-1)
    df["overnight_ret"]     = (df["next_open"] - df["close"]) / df["close"]
    df                      = df.dropna(subset=["next_open"])

    trades = df[df["signal"]].copy()
    if len(trades) == 0:
        print(f"  {ticker}: no trades at {THRESHOLD_PCT:.0f}% threshold")
        continue

    trades["net_ret"] = trades["overnight_ret"] - cost
    trades["ticker"]  = ticker

    stats = analyse(trades["net_ret"], ticker, len(df))
    if stats:
        stats["start"] = df.index[0].date()
        stats["end"]   = df.index[-1].date()
        results.append(stats)
        all_trades.append(trades[["ticker", "net_ret", "intraday_drop_pct",
                                   "open", "low", "close", "next_open"]])

    print(f"  {ticker}: {len(trades):>4} trades | "
          f"win={stats.get('win_rate', 0):.1%} | "
          f"ev={stats.get('ev', 0)*100:+.3f}% | "
          f"sharpe={stats.get('sharpe', 0):+.2f} | "
          f"total={stats.get('total_ret', 0)*100:+.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

SEP = "─" * 105

print(f"\n\n{'═'*105}")
print(f"  OIL MICROCAP OVERNIGHT EDGE  |  drop ≥ {THRESHOLD_PCT:.0f}% from open  |  cost={TRANSACTION_PCT*2:.2f}% round-trip")
print(f"{'═'*105}")

print(f"\n  {'Ticker':<7} {'Data':>18} {'Trades':>7} {'Freq%':>6} "
      f"{'WinRate':>8} {'AvgWin%':>8} {'AvgLoss%':>9} "
      f"{'Payoff':>8} {'EV%':>7} {'TotalRet%':>10} {'Sharpe':>8} {'MaxDD%':>8}")
print(f"  {SEP}")

for s in results:
    print(
        f"  {s['ticker']:<7} "
        f"{str(s['start'])+'→'+str(s['end']):>18} "
        f"{s['n']:>7} "
        f"{s['freq_pct']:>6.1f} "
        f"{s['win_rate']:>8.1%} "
        f"{s['avg_win']*100:>+8.3f} "
        f"{s['avg_loss']*100:>+9.3f} "
        f"{s['payoff']:>8.2f} "
        f"{s['ev']*100:>+7.3f} "
        f"{s['total_ret']*100:>+10.1f} "
        f"{s['sharpe']:>+8.2f} "
        f"{s['max_dd']*100:>+8.1f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pooled
# ─────────────────────────────────────────────────────────────────────────────

if all_trades:
    pool = pd.concat(all_trades).sort_index()
    n_pool_days = pool.index.nunique()
    pool_stats = analyse(pool["net_ret"], "POOL", n_pool_days)

    print(f"\n  {SEP}")
    print(f"  POOLED (all tickers combined, chronological)")
    print(f"  {SEP}")
    print(
        f"  {'POOL':<7} "
        f"{'all':>18} "
        f"{pool_stats['n']:>7} "
        f"{pool_stats['freq_pct']:>6.1f} "
        f"{pool_stats['win_rate']:>8.1%} "
        f"{pool_stats['avg_win']*100:>+8.3f} "
        f"{pool_stats['avg_loss']*100:>+9.3f} "
        f"{pool_stats['payoff']:>8.2f} "
        f"{pool_stats['ev']*100:>+7.3f} "
        f"{pool_stats['total_ret']*100:>+10.1f} "
        f"{pool_stats['sharpe']:>+8.2f} "
        f"{pool_stats['max_dd']*100:>+8.1f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Year-by-year pooled
# ─────────────────────────────────────────────────────────────────────────────

if all_trades:
    pool_df = pd.concat(all_trades)
    print(f"\n  {SEP}")
    print(f"  YEAR-BY-YEAR  (pooled, threshold={THRESHOLD_PCT:.0f}%)")
    print(f"  {SEP}")
    print(f"  {'Year':>6}  {'N':>5}  {'WinRate':>8}  {'AvgRet%':>8}  {'TotalRet%':>10}  {'Sharpe':>7}")
    for year, grp in pool_df.groupby(pool_df.index.year):
        r    = grp["net_ret"]
        wr   = (r > 0).mean()
        ar   = r.mean() * 100
        tot  = float((1 + r).prod() - 1) * 100
        sh   = (r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0
        mark = "  ◄ positive" if tot > 0 else ""
        print(f"  {year:>6}  {len(r):>5}  {wr:>8.1%}  {ar:>+8.3f}  {tot:>+10.1f}%  {sh:>7.2f}{mark}")


# ─────────────────────────────────────────────────────────────────────────────
# Threshold sweep per ticker
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n  {SEP}")
print(f"  THRESHOLD SWEEP per ticker  (cost={TRANSACTION_PCT*2:.2f}%)")
print(f"  {SEP}")
print(f"  {'Ticker':<7}  {'Thresh':>7}  {'N':>5}  {'WinRate':>8}  {'AvgRet%':>8}  {'Sharpe':>7}  {'TotalRet%':>10}")

for ticker in TICKERS:
    try:
        raw = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
        if raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if not all(c in raw.columns for c in ["Open", "Low", "Close"]):
            continue
    except Exception:
        continue

    df = raw[["Open", "Low", "Close"]].copy()
    df.index = pd.to_datetime(df.index)
    df.columns = ["open", "low", "close"]
    df["drop_pct"]  = (df["open"] - df["low"]) / df["open"] * 100
    df["next_open"] = df["open"].shift(-1)
    df["ov_ret"]    = (df["next_open"] - df["close"]) / df["close"]
    df = df.dropna()

    printed_header = False
    for th in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0]:
        t = df[df["drop_pct"] >= th]
        if len(t) < 5:
            continue
        r    = t["ov_ret"] - cost
        wr   = (r > 0).mean()
        sh   = r.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else 0
        tot  = float((1 + r).prod() - 1) * 100
        mark = "  ◄" if th == THRESHOLD_PCT else ""
        print(f"  {ticker:<7}  {th:>6.0f}%  {len(t):>5}  {wr:>8.1%}  "
              f"{r.mean()*100:>+8.3f}  {sh:>7.2f}  {tot:>+10.1f}%{mark}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Distribution of intraday drops (how often do big drops occur?)
# ─────────────────────────────────────────────────────────────────────────────

print(f"  {SEP}")
print(f"  INTRADAY DROP DISTRIBUTION (% of days with drop ≥ threshold)")
print(f"  {SEP}")
print(f"  {'Ticker':<7}  {'Total Days':>10}  ", end="")
thresholds = [1, 2, 3, 5, 7, 10, 15]
for th in thresholds:
    print(f"  ≥{th:>2}%", end="")
print()

for ticker in TICKERS:
    try:
        raw = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
        if raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if not all(c in raw.columns for c in ["Open", "Low"]):
            continue
    except Exception:
        continue

    df = raw[["Open", "Low"]].copy()
    df.columns = ["open", "low"]
    df["drop"] = (df["open"] - df["low"]) / df["open"] * 100
    df = df.dropna()

    print(f"  {ticker:<7}  {len(df):>10}  ", end="")
    for th in thresholds:
        pct = (df["drop"] >= th).mean() * 100
        print(f"  {pct:>4.1f}%", end="")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

if all_trades:
    out_path = Path(__file__).parent.parent / "data" / "cache" / "oil_microcap_overnight_edge.csv"
    pd.concat(all_trades).to_csv(out_path)
    print(f"\n  Full trade log saved to: {out_path}")
print()