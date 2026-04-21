#!/usr/bin/env python3
"""
Macro Component Audit
=====================
Standalone validator for three Arena infrastructure modules:
  1. Dilution Detection  — technical volume/price method vs trade outcomes
  2. Regime Detection    — VOO-based market regime labels per trade window
  3. Slippage Model      — ADV-aware slippage vs flat 0.2% for each ticker

Zero imports from side_by_side_backtest/. Reads only:
  - side_by_side_backtest/ohlcv_cache/*.parquet
  - side_by_side_backtest/watchlist_backtest.db

Run:
    python analysis/macro_audit.py
    python analysis/macro_audit.py --section dilution
    python analysis/macro_audit.py --section regime
    python analysis/macro_audit.py --section slippage
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

CACHE_DIR = ROOT / "side_by_side_backtest" / "ohlcv_cache"
DB_PATH   = ROOT / "side_by_side_backtest" / "watchlist_backtest.db"

# ── Arena module imports (no side_by_side_backtest deps) ──────────────────────
from backtest.enhanced_dilution_detection import EnhancedDilutionDetector
from backtest.risk_management import MicrocapSlippageModel
from evolution.regime_detection import RegimeDetector, VolatilityRegime, TrendRegime, MarketRegime

# ── Data loading helpers ───────────────────────────────────────────────────────

def load_parquet_for_ticker(ticker: str) -> pd.DataFrame:
    """Load and merge all parquet windows for a ticker into one sorted DataFrame."""
    files = sorted(CACHE_DIR.glob(f"{ticker}_*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    # Normalise column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    return df


def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 5-min OHLCV to daily bars."""
    if df.empty:
        return df
    return df.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}
    ).dropna(subset=["close"])


def load_trades() -> pd.DataFrame:
    """Load all trades from watchlist_backtest.db."""
    if not DB_PATH.exists():
        print(f"[WARN] DB not found: {DB_PATH}")
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT ticker, entry_ts, entry_price, exit_ts, exit_price, "
        "       outcome, pnl_pct, pattern_type, support_source "
        "FROM trades WHERE outcome != 'open'",
        conn,
    )
    conn.close()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["exit_ts"]  = pd.to_datetime(df["exit_ts"],  utc=True, errors="coerce")
    return df


def all_cached_tickers() -> List[str]:
    """Return unique tickers that have at least one parquet file."""
    tickers = set()
    for f in CACHE_DIR.glob("*.parquet"):
        tickers.add(f.stem.split("_")[0])
    return sorted(tickers)


# ── Section 1: Dilution Detection Audit ───────────────────────────────────────

def audit_dilution(trades: pd.DataFrame) -> pd.DataFrame:
    """
    For every trade in the DB, scan the 5-min bars in the window
    [entry_ts - 3 sessions, entry_ts] for a technical dilution signal
    (volume spike ≥ 3× avg + price drop ≥ 10%).

    Returns a DataFrame with one row per trade containing:
      ticker, entry_ts, outcome, pnl_pct,
      dilution_flag (bool), vol_spike, price_drop
    """
    detector = EnhancedDilutionDetector(
        volume_spike_threshold=3.0,
        price_drop_threshold=-0.10,
        lookback_days=20,
    )

    rows = []
    for _, trade in trades.iterrows():
        ticker = trade["ticker"]
        entry_ts = trade["entry_ts"]

        df5 = load_parquet_for_ticker(ticker)
        if df5.empty:
            continue

        # Use bars up to but not including the entry bar
        pre = df5[df5.index < entry_ts]
        if len(pre) < 25:
            continue

        daily = resample_to_daily(pre)
        if len(daily) < 5:
            continue

        event = detector.detect_technical_dilution(
            ticker=ticker,
            prices=daily["close"],
            volume=daily["volume"],
            date=entry_ts.to_pydatetime(),
        )

        rows.append({
            "ticker":        ticker,
            "entry_ts":      entry_ts,
            "outcome":       trade["outcome"],
            "pnl_pct":       trade["pnl_pct"],
            "dilution_flag": event is not None,
            "vol_spike":     event.volume_spike if event else 0.0,
            "price_drop":    event.price_drop   if event else 0.0,
        })

    return pd.DataFrame(rows)


def print_dilution_report(df: pd.DataFrame) -> None:
    if df.empty:
        print("[dilution] No trades found — skipping.\n")
        return

    total   = len(df)
    # Note: all SL exits are at a fixed % — no catastrophic losses exist in this DB.
    # Reframed: does the detector fire more on losses (SL hit) than wins (PT hit)?
    losses  = df[df["outcome"] == "loss"]
    wins    = df[df["outcome"] == "win"]

    flag_on_loss = int(losses["dilution_flag"].sum()) if len(losses) else 0
    flag_on_win  = int(wins["dilution_flag"].sum())   if len(wins)   else 0

    recall    = flag_on_loss / len(losses) * 100 if len(losses) else 0.0
    fp_rate   = flag_on_win  / len(wins)   * 100 if len(wins)   else 0.0

    print("=" * 60)
    print("DILUTION DETECTION AUDIT (technical method only)")
    print("=" * 60)
    print(f"  Total trades analysed : {total}")
    print()
    print("  NOTE: SL exits are bounded — no catastrophic losses in DB.")
    print("  Reframed: does dilution flag fire more on losses than wins?")
    print()
    print(f"  Losses (SL hit)       : {len(losses)}")
    print(f"  Wins  (PT hit)        : {len(wins)}")
    print(f"  Dilution flag on loss : {flag_on_loss}  → {recall:.1f}% of losses")
    print(f"  Dilution flag on win  : {flag_on_win}   → {fp_rate:.1f}% of wins (FP rate)")
    print()
    if flag_on_loss == 0 and flag_on_win == 0:
        print("  ⚠️  INCONCLUSIVE — detector never fires on this universe.")
        print("      vol_spike ≥ 3× AND price_drop ≥ 10% same daily bar is too")
        print("      strict for SL-bounded entries. Technical method alone is")
        print("      insufficient; would need SEC filing feed to add real value.")
    elif recall > fp_rate and fp_rate <= 25:
        print(f"  ✅ PASSES — flag fires {recall:.1f}% on losses vs {fp_rate:.1f}% on wins")
    else:
        print(f"  ❌ FAILS — no discrimination (loss={recall:.1f}%, win={fp_rate:.1f}%)")
    print()
    if int(df["dilution_flag"].sum()) > 0:
        print("  All flagged trades:")
        for _, r in df[df["dilution_flag"]].iterrows():
            print(f"    {r['ticker']:6s}  {str(r['entry_ts'])[:19]}  "
                  f"outcome={r['outcome']:7s}  pnl={r['pnl_pct']:+.3f}%  "
                  f"vol×{r['vol_spike']:.1f}  drop={r['price_drop']*100:.1f}%")
    print()


# ── Section 2: Regime Detection Audit ─────────────────────────────────────────

def audit_regime(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a single regime snapshot from all available VOO daily bars,
    then apply it to every trade that falls within the VOO data window.

    With only ~30 days of cached VOO data there is only enough history
    for one snapshot (needs vol_lookback bars to warm up). We use that
    snapshot as a "current regime" label and show win-rate stats for
    trades that occurred during that period.
    """
    voo_5m = load_parquet_for_ticker("VOO")
    if voo_5m.empty:
        print("[regime] No VOO data found — skipping.\n")
        return pd.DataFrame()

    voo_daily = resample_to_daily(voo_5m)
    n_daily = len(voo_daily)
    if n_daily < 5:
        print(f"[regime] Only {n_daily} VOO daily bars — insufficient. Skipping.\n")
        return pd.DataFrame()

    vol_lb   = min(20, n_daily - 1)
    trend_lb = min(50, n_daily - 1)
    mkt_lb   = min(200, n_daily - 1)

    detector = RegimeDetector(
        volatility_lookback=vol_lb,
        trend_lookback=trend_lb,
        market_lookback=mkt_lb,
    )
    voo_returns = voo_daily["close"].pct_change().dropna()

    # Compute single snapshot on full available history
    try:
        vol_reg,   _ = detector.detect_volatility_regime(voo_returns)
        trend_reg, _ = detector.detect_trend_regime(voo_daily["close"])
        mkt_reg,   _ = detector.detect_market_regime(voo_daily["close"])
    except Exception as e:
        print(f"[regime] Regime detection failed: {e}\n")
        return pd.DataFrame()

    snapshot = {
        "vol_regime":   vol_reg.value,
        "trend_regime": trend_reg.value,
        "mkt_regime":   mkt_reg.value,
    }

    # VOO window: first to last daily bar date
    voo_start = voo_daily.index[0].normalize()
    voo_end   = voo_daily.index[-1].normalize()

    print(f"[regime] VOO window: {str(voo_start)[:10]} – {str(voo_end)[:10]}  "
          f"regime={snapshot}")

    rows = []
    skipped = 0
    for _, trade in trades.iterrows():
        entry_ts  = trade["entry_ts"]
        entry_day = entry_ts.normalize()   # UTC-aware, matches voo index

        if entry_day < voo_start or entry_day > voo_end:
            skipped += 1
            continue

        rows.append({
            "ticker":   trade["ticker"],
            "entry_ts": entry_ts,
            "outcome":  trade["outcome"],
            "pnl_pct":  trade["pnl_pct"],
            **snapshot,
        })

    if skipped > 0:
        print(f"[regime] {skipped} trades outside VOO window — excluded.")

    return pd.DataFrame(rows)


def print_regime_report(df: pd.DataFrame) -> None:
    if df.empty:
        print("[regime] No data — skipping.\n")
        return

    print("=" * 60)
    print("REGIME DETECTION AUDIT (VOO daily bars)")
    print("=" * 60)

    for col, label in [("vol_regime", "Volatility"), ("mkt_regime", "Market")]:
        print(f"\n  Win rate by {label} regime:")
        grp = df.groupby(col).apply(
            lambda g: pd.Series({
                "trades":   len(g),
                "win_rate": (g["outcome"] == "win").mean() * 100,
                "avg_pnl":  g["pnl_pct"].mean(),
            }),
            include_groups=False,
        ).reset_index()
        for _, row in grp.iterrows():
            print(f"    {row[col]:20s}  n={int(row['trades']):3d}  "
                  f"WR={row['win_rate']:.1f}%  avg_pnl={row['avg_pnl']:+.2f}%")

    # Pass criterion: worst regime WR < best regime WR by ≥ 15 pp
    wr_by_mkt = df.groupby("mkt_regime")["outcome"].apply(
        lambda g: (g == "win").mean() * 100
    )
    spread = wr_by_mkt.max() - wr_by_mkt.min() if len(wr_by_mkt) > 1 else 0
    print()
    if spread >= 15:
        print(f"  ✅ PASSES — regime WR spread {spread:.1f} pp ≥ 15 pp")
    else:
        print(f"  ❌ FAILS  — regime WR spread {spread:.1f} pp < 15 pp")
    print()


# ── Section 3: Slippage Model Audit ───────────────────────────────────────────

def audit_slippage(order_sizes: List[float] = None) -> pd.DataFrame:
    """
    For each ticker with cached data, compute ADV and realistic slippage
    at each order_size vs flat 0.2%. Returns a per-ticker summary.
    """
    if order_sizes is None:
        order_sizes = [500.0, 1_000.0, 5_000.0]

    model   = MicrocapSlippageModel(base_slippage_bps=25, volume_impact_factor=0.5,
                                    commission_per_trade=6.95)
    tickers = all_cached_tickers()
    rows    = []

    # Build dummy price/volume DataFrames expected by MicrocapSlippageModel
    for ticker in tickers:
        df5 = load_parquet_for_ticker(ticker)
        if df5.empty or "volume" not in df5.columns:
            continue

        daily = resample_to_daily(df5)
        if len(daily) < 5:
            continue

        # MicrocapSlippageModel expects DataFrame with ticker as column
        prices_df = daily[["close"]].rename(columns={"close": ticker})
        volume_df = daily[["volume"]].rename(columns={"volume": ticker})

        row = {"ticker": ticker, "adv_shares": daily["volume"].tail(20).mean(),
               "last_price": daily["close"].iloc[-1]}

        for size in order_sizes:
            result = model.calculate_slippage(
                ticker=ticker,
                order_size_dollars=size,
                prices=prices_df,
                volume=volume_df,
                lookback=min(20, len(daily) - 1),
            )
            size_key = f"{int(size)}" if size < 1000 else f"{int(size/1000)}k"
            col_pct  = f"slip_{size_key}_pct"
            col_vs   = f"vs_flat_{size_key}"
            row[col_pct] = round(result.total_slippage * 100, 3)
            row[col_vs]  = round((result.total_slippage - 0.002) * 100, 3)

        rows.append(row)

    return pd.DataFrame(rows)


def print_slippage_report(df: pd.DataFrame) -> None:
    if df.empty:
        print("[slippage] No ticker data found — skipping.\n")
        return

    print("=" * 60)
    print("SLIPPAGE MODEL AUDIT (ADV-aware vs flat 0.2%)")
    print("=" * 60)

    order_cols_pct = sorted([c for c in df.columns if c.startswith("slip_") and c.endswith("_pct")])
    flat_pct       = 0.2   # sanity check G baseline

    for col in order_cols_pct:
        # col like "slip_500_pct" or "slip_1k_pct"
        size_key   = col.replace("slip_", "").replace("_pct", "")
        size_label = f"${size_key}"
        over_flat   = (df[col] > flat_pct).sum()
        pct_over    = over_flat / len(df) * 100
        median_slip = df[col].median()
        print(f"\n  Order size {size_label}:")
        print(f"    Median slippage : {median_slip:.2f}%")
        print(f"    Flat 0.2% basis : {flat_pct:.2f}%")
        print(f"    Tickers > flat  : {over_flat}/{len(df)} ({pct_over:.0f}%)")

    # Pass criterion: ≥30% of tickers exceed 0.5% at $1k order (col = slip_1k_pct)
    slip_col = next((c for c in order_cols_pct if "1k" in c), None)
    if slip_col and slip_col in df.columns:
        pct_above_half = (df[slip_col] > 0.5).mean() * 100
        print()
        if pct_above_half >= 30:
            print(f"  ✅ PASSES — {pct_above_half:.0f}% of tickers slip >0.5% at $1k "
                  f"(flat 0.2% materially understates costs)")
        else:
            print(f"  ❌ FAILS  — only {pct_above_half:.0f}% of tickers slip >0.5% at $1k")

    print("\n  Highest-slippage tickers at $1k:")
    if "slip_1k_pct" in df.columns:
        top = df.nlargest(10, "slip_1k_pct")[
            ["ticker", "adv_shares", "last_price", "slip_1k_pct"]
        ]
        for _, r in top.iterrows():
            adv_d = r["adv_shares"] * r["last_price"]
            print(f"    {r['ticker']:6s}  ADV≈${adv_d:,.0f}  "
                  f"slip@$1k={r['slip_1k_pct']:.2f}%")
    print()


# ── Main entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Macro Component Audit")
    parser.add_argument(
        "--section", choices=["dilution", "regime", "slippage", "all"],
        default="all", help="Which audit section to run (default: all)",
    )
    parser.add_argument(
        "--export", action="store_true",
        help="Save CSV reports to analysis/ directory",
    )
    args = parser.parse_args()

    run_all      = args.section == "all"
    export_dir   = Path(__file__).parent

    trades = load_trades()
    print(f"\n[info] Loaded {len(trades)} completed trades from DB\n")

    # ── Dilution ──────────────────────────────────────────────────────────────
    if run_all or args.section == "dilution":
        print("[running] Dilution detection audit …")
        dil_df = audit_dilution(trades)
        print_dilution_report(dil_df)
        if args.export and not dil_df.empty:
            out = export_dir / "audit_dilution.csv"
            dil_df.to_csv(out, index=False)
            print(f"  → Saved {out}\n")

    # ── Regime ────────────────────────────────────────────────────────────────
    if run_all or args.section == "regime":
        print("[running] Regime detection audit …")
        reg_df = audit_regime(trades)
        print_regime_report(reg_df)
        if args.export and not reg_df.empty:
            out = export_dir / "audit_regime.csv"
            reg_df.to_csv(out, index=False)
            print(f"  → Saved {out}\n")

    # ── Slippage ──────────────────────────────────────────────────────────────
    if run_all or args.section == "slippage":
        print("[running] Slippage model audit …")
        slip_df = audit_slippage()
        print_slippage_report(slip_df)
        if args.export and not slip_df.empty:
            out = export_dir / "audit_slippage.csv"
            slip_df.to_csv(out, index=False)
            print(f"  → Saved {out}\n")


if __name__ == "__main__":
    main()
