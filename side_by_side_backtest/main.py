"""
Side-by-Side Body Intraday Backtest — CLI Entry Point
=====================================================

Phases executed in order:
  1. Parse  — read scraped_watchlists.json → WatchlistEntry objects → SQLite
  2. Fetch  — download 5-min OHLCV bars for every unique ticker / date
  3. Run    — single-pass simulation with default parameters (quick sanity check)
  4. Sweep  — full profit-target / stop-loss optimisation grid
  5. Report — console dashboard + optional CSV/PNG export

Usage
-----
    # Full pipeline (default settings):
    python -m side_by_side_backtest.main

    # Custom watchlist path and provider:
    python -m side_by_side_backtest.main \
        --watchlist /path/to/scraped_watchlists.json \
        --provider alpaca

    # Quick single-pass only (no sweep):
    python -m side_by_side_backtest.main --no-sweep

    # Skip re-parse if DB already populated:
    python -m side_by_side_backtest.main --skip-parse

    # Custom sweep range and stop:
    python -m side_by_side_backtest.main \
        --pt-start 0.5 --pt-stop 5.0 --pt-step 0.25 \
        --sl-start 0.5 --sl-stop 2.0 --sl-step 0.5

    # Export charts and CSV:
    python -m side_by_side_backtest.main --export
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="side_by_side_backtest",
        description="Side-by-Side Body intraday strategy backtester.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    p.add_argument(
        "--watchlist",
        default=None,
        help="Path to scraped_watchlists.json. Ignored if --tickers is set.",
    )
    p.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help=(
            "Auto-build watchlist from ohlcv_cache/ for these tickers, "
            "e.g. --tickers UGRO ANNA TURB. Skips --watchlist parsing."
        ),
    )
    p.add_argument(
        "--same-window-support",
        action="store_true",
        help=(
            "When using --tickers, derive support from the *same* date window "
            "(leaky — use only for comparison). Default: prior-window (clean)."
        ),
    )
    p.add_argument(
        "--db",
        default=None,
        help="SQLite DB path (default: side_by_side_backtest/watchlist_backtest.db)",
    )

    # Data
    p.add_argument(
        "--provider",
        choices=["yfinance", "alpaca"],
        default="yfinance",
        help="OHLCV data provider",
    )
    p.add_argument(
        "--lookback-days",
        type=int,
        default=3,
        help="Days of history to fetch before each watchlist date",
    )
    p.add_argument(
        "--forward-days",
        type=int,
        default=2,
        help="Days to fetch after each watchlist date",
    )

    # Pipeline control
    p.add_argument("--skip-parse",  action="store_true", help="Skip Phase 1 (use existing DB)")
    p.add_argument("--skip-fetch",  action="store_true", help="Skip Phase 2 (use disk OHLCV cache)")
    p.add_argument("--no-sweep",    action="store_true", help="Skip Phase 4 (sweep) — single-pass only")
    p.add_argument("--export",      action="store_true", help="Export CSV and PNG artefacts")
    p.add_argument("--verbose",     action="store_true", help="Print per-trade details")
    p.add_argument(
        "--sanity",
        action="store_true",
        help="Run the 6-check sanity suite after the main backtest pipeline.",
    )
    p.add_argument(
        "--sanity-checks",
        nargs="+",
        default=None,
        metavar="CHECK",
        help="Subset of sanity checks to run, e.g. --sanity-checks A B F (default: all).",
    )

    # Auto-tune (Bayesian optimisation — replaces manual sweep ranges)
    p.add_argument(
        "--auto-tune",
        action="store_true",
        help="Replace Phase 4 grid sweep with Bayesian auto-tuning (optuna TPE)",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Bayesian optimisation trials (used with --auto-tune)",
    )
    p.add_argument(
        "--tune-objective",
        choices=["expectancy", "profit_factor", "win_rate"],
        default="expectancy",
        help="Metric to maximise during auto-tuning",
    )
    p.add_argument(
        "--tune-min-trades",
        type=int,
        default=5,
        help="Minimum trades required for a trial to be considered viable",
    )
    p.add_argument(
        "--tune-tolerance",
        action="store_true",
        help="Also optimise pattern_tolerance_pct during auto-tune",
    )

    # Quick single-pass parameters
    p.add_argument("--pt",  type=float, default=5.0, help="Profit target %% for the quick single-pass run")
    p.add_argument("--sl",  type=float, default=1.0, help="Stop-loss %% for the quick single-pass run")

    # Risk controls
    p.add_argument(
        "--max-loss-pct",
        type=float,
        default=5.0,
        help=(
            "Hard maximum loss cap per trade (%%). Prevents gap-through blowouts "
            "from exceeding this %% below entry price (default: 5.0)."
        ),
    )
    p.add_argument(
        "--max-attempts",
        type=int,
        default=10,
        help=(
            "Max support-touch re-entry attempts per ticker per session window. "
            "0 = unlimited. Default: 10. Prevents ATPC/AZI-style 100+ attempt churn."
        ),
    )
    p.add_argument(
        "--no-require-support-ok",
        action="store_false",
        dest="require_support_ok",
        help=(
            "Disable the support_ok filter — allow trades where price failed to "
            "respect the support level (legacy behaviour, not recommended)."
        ),
    )
    p.set_defaults(require_support_ok=True)

    # Sweep grid
    p.add_argument("--pt-start", type=float, default=0.5,  help="Sweep: PT range start (%%)")
    p.add_argument("--pt-stop",  type=float, default=5.0,  help="Sweep: PT range stop  (%%)")
    p.add_argument("--pt-step",  type=float, default=0.5,  help="Sweep: PT range step  (%%)")
    p.add_argument("--sl-start", type=float, default=1.0,  help="Sweep: SL range start (%%)")
    p.add_argument("--sl-stop",  type=float, default=1.0,  help="Sweep: SL range stop  (%%)")
    p.add_argument("--sl-step",  type=float, default=0.5,  help="Sweep: SL range step  (%%)")

    # Pattern engine
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Side-by-Side body-midpoint tolerance (fraction, e.g. 0.01 = 1%%)",
    )

    # Report
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory for exported artefacts (default: side_by_side_backtest/reports/)",
    )

    return p


# ---------------------------------------------------------------------------
# Pipeline phases
# ---------------------------------------------------------------------------

def phase_parse(args, db) -> list:
    from .parser import parse_scraped_file

    wl_path = Path(args.watchlist)
    if not wl_path.exists():
        print(f"[main] ERROR: watchlist file not found: {wl_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'═'*60}")
    print("  PHASE 1 — Parse & Store Watchlist Entries")
    print(f"{'═'*60}")

    entries = parse_scraped_file(wl_path)
    inserted = db.upsert_entries(entries)
    print(f"[main] Inserted {inserted} new entries into DB ({len(entries)} total parsed).")
    return entries


def phase_fetch(args, entries, db) -> dict:
    from .data_fetcher import fetch_bars_batch, load_30day_bars
    import pandas as pd

    print(f"\n{'═'*60}")
    print("  PHASE 2 — Fetch 5-Minute OHLCV Bars")
    print(f"{'═'*60}")

    # Strategy:
    # 1. Fetch per-entry windows (3d lookback + 2d forward around post date)
    #    → covers the EXACT date the trade would have been live
    # 2. Supplement with 30d rolling cache (fills gaps, adds recent history)
    # 3. Merge both so the simulator has the most complete picture

    if args.skip_fetch:
        # --skip-fetch: load only from 30d parquet + legacy disk cache (no network)
        bars_map: dict = {}
        for entry in entries:
            ticker = entry.ticker
            if ticker in bars_map:
                continue
            cached = load_30day_bars(ticker)
            if not cached.empty:
                bars_map[ticker] = cached
        print(f"[main] --skip-fetch: loaded {len(bars_map)} tickers from disk cache.")
        return bars_map

    # Step 1: per-entry fetch (historical accuracy)
    bars_map = fetch_bars_batch(
        entries,
        lookback_days=args.lookback_days,
        forward_days=args.forward_days,
        provider=args.provider,
    )

    # Step 2: for each ticker, merge in 30d cache if available.
    # Deduplicate each frame before concat to avoid InvalidIndexError when
    # either frame has internal duplicate timestamps (yfinance sometimes returns these).
    def _safe_dedup(df: pd.DataFrame) -> pd.DataFrame:
        if df.index.duplicated().any():
            return df[~df.index.duplicated(keep="last")]
        return df

    enriched = 0
    for ticker in list(bars_map.keys()):
        cached_30d = load_30day_bars(ticker)
        if not cached_30d.empty:
            merged = pd.concat([_safe_dedup(bars_map[ticker]), _safe_dedup(cached_30d)])
            merged = _safe_dedup(merged).sort_index()
            bars_map[ticker] = merged
            enriched += 1

    print(f"[main] {len(bars_map)} tickers fetched; {enriched} enriched with 30d cache.")
    return bars_map


def phase_single_run(args, entries, bars_map) -> list:
    from .simulator import simulate_all

    max_loss  = getattr(args, "max_loss_pct", 5.0)
    max_att   = getattr(args, "max_attempts", 0)
    req_sup   = getattr(args, "require_support_ok", False)

    print(f"\n{'═'*60}")
    print(
        f"  PHASE 3 — Single-Pass Simulation "
        f"(PT={args.pt}%  SL={args.sl}%  MAX_LOSS={max_loss}%  "
        f"MAX_ATT={max_att or '∞'}  REQ_SUP={req_sup})"
    )
    print(f"{'═'*60}")

    trades = simulate_all(
        entries,
        bars_map,
        profit_target_pct=args.pt,
        stop_loss_pct=args.sl,
        max_loss_pct=max_loss,
        pattern_tolerance_pct=args.tolerance,
        require_support_ok=req_sup,
        max_entry_attempts=max_att,
        verbose=args.verbose,
    )

    wins    = sum(1 for t in trades if t.outcome == "win")
    losses  = sum(1 for t in trades if t.outcome == "loss")
    timeouts = sum(1 for t in trades if t.outcome == "timeout")
    win_rate = wins / len(trades) if trades else 0.0

    print(
        f"\n[main] Single-pass: {len(trades)} trades | "
        f"W={wins}  L={losses}  T={timeouts} | "
        f"WR={win_rate:.1%}"
    )
    return trades


def phase_sweep(args, entries, bars_map) -> tuple:
    from .optimizer import optimize, summaries_to_dataframe
    from .simulator import simulate_all

    print(f"\n{'═'*60}")
    print("  PHASE 4 — Parameter Sweep Optimisation")
    print(f"{'═'*60}")

    result = optimize(
        entries,
        bars_map,
        profit_target_range=(args.pt_start, args.pt_stop, args.pt_step),
        stop_loss_range=(args.sl_start, args.sl_stop, args.sl_step),
        pattern_tolerance_pct=args.tolerance,
        verbose=True,
    )

    # Re-simulate with the best-by-expectancy configuration to get TradeResult list
    max_loss = getattr(args, "max_loss_pct", 5.0)
    max_att  = getattr(args, "max_attempts", 0)
    req_sup  = getattr(args, "require_support_ok", False)
    best_trades: list = []
    best = result.best_by_expectancy
    if best is not None:
        best_trades = simulate_all(
            entries,
            bars_map,
            profit_target_pct=best.profit_target_pct,
            stop_loss_pct=best.stop_loss_pct,
            max_loss_pct=max_loss,
            pattern_tolerance_pct=args.tolerance,
            require_support_ok=req_sup,
            max_entry_attempts=max_att,
            verbose=False,
        )

    return result, best_trades


def phase_auto_tune(args, entries, bars_map) -> tuple:
    """Phase 4 (alternative) — Bayesian auto-tuning via optuna TPE."""
    from .auto_tuner import auto_tune, auto_tune_to_dataframe
    from .simulator import simulate_all

    tune_result = auto_tune(
        entries,
        bars_map,
        n_trials=args.n_trials,
        min_trades=args.tune_min_trades,
        objective=args.tune_objective,
        tune_tolerance=args.tune_tolerance,
        verbose=True,
    )

    # Re-simulate with best params to produce TradeResult list for report/DB
    max_loss = getattr(args, "max_loss_pct", 5.0)
    max_att  = getattr(args, "max_attempts", 0)
    req_sup  = getattr(args, "require_support_ok", False)
    best_trades = simulate_all(
        entries,
        bars_map,
        profit_target_pct=tune_result.best_pt,
        stop_loss_pct=tune_result.best_sl,
        max_loss_pct=max_loss,
        pattern_tolerance_pct=tune_result.best_tolerance,
        require_support_ok=req_sup,
        max_entry_attempts=max_att,
        verbose=False,
    )

    # Optionally export trial history as CSV
    if args.export:
        import pandas as pd
        from pathlib import Path
        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else Path(__file__).parent / "reports"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        df = auto_tune_to_dataframe(tune_result)
        csv_path = output_dir / "auto_tune_trials.csv"
        df.to_csv(csv_path, index=False)
        print(f"[auto_tune] Trial history saved → {csv_path}")

    return tune_result, best_trades


def phase_report(args, result, best_trades) -> None:
    from .report import generate_report

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(__file__).parent / "reports"
    )

    print(f"\n{'═'*60}")
    print("  PHASE 5 — Report")
    print(f"{'═'*60}")

    generate_report(
        result,
        best_trades=best_trades if args.export else None,
        export=args.export,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ── DB connection ────────────────────────────────────────────────────
    from .db import WatchlistDB

    db_path = args.db or str(Path(__file__).parent / "watchlist_backtest.db")
    db = WatchlistDB(db_path).connect()

    try:
        # ── Phase 1: Parse ───────────────────────────────────────────────
        if args.skip_parse:
            print("[main] --skip-parse: loading entries from existing DB …")
            entries = db.load_entries()
            print(f"[main] Loaded {len(entries)} entries from DB.")
        elif args.tickers:
            # Fast path: auto-build watchlist from ohlcv_cache/ for given tickers
            from .watchlist_builder import build_watchlist_from_tickers
            tickers = [t.upper() for t in args.tickers]
            use_prior = not getattr(args, "same_window_support", False)
            print(f"\n{'═'*60}")
            print(f"  PHASE 1 — Auto-Build Watchlist from Cache ({', '.join(tickers)})")
            print(f"{'═'*60}")
            entries = build_watchlist_from_tickers(tickers, use_prior_window=use_prior)
            inserted = db.upsert_entries(entries)
            print(f"[main] Inserted {inserted} new entries into DB ({len(entries)} total).")
        else:
            if args.watchlist is None:
                print("[main] ERROR: provide --watchlist PATH or --tickers TICKER [...]", file=sys.stderr)
                sys.exit(1)
            entries = phase_parse(args, db)

        if not entries:
            print("[main] No entries found — nothing to simulate. Exiting.")
            return

        # ── Phase 2: Fetch ───────────────────────────────────────────────
        if args.skip_fetch:
            print("[main] --skip-fetch: relying on disk OHLCV cache only.")
        bars_map = phase_fetch(args, entries, db)

        if not bars_map:
            print("[main] No OHLCV data fetched — cannot simulate. Exiting.")
            return

        # ── Phase 3: Single-pass ─────────────────────────────────────────
        single_trades = phase_single_run(args, entries, bars_map)

        if args.no_sweep:
            print("\n[main] --no-sweep: skipping optimisation. Done.")
            return

        # ── Phase 4: Sweep or Auto-Tune ──────────────────────────────────
        if args.auto_tune:
            tune_result, best_trades = phase_auto_tune(args, entries, bars_map)
            # Print concise best-params banner
            print(
                f"\n[main] Auto-tune complete — "
                f"PT={tune_result.best_pt:.2f}%  "
                f"SL={tune_result.best_sl:.2f}%  "
                f"E={tune_result.best_expectancy:+.4f}  "
                f"WR={tune_result.best_win_rate:.1%}  "
                f"PF={tune_result.best_profit_factor:.3f}  "
                f"trades={tune_result.best_total_trades}"
            )
            # Wrap in a minimal OptimizationResult-compatible object for phase_report
            from .optimizer import compute_summary
            from .models import OptimizationResult, BacktestSummary
            best_summary = compute_summary(best_trades, tune_result.best_pt, tune_result.best_sl)
            opt_result = OptimizationResult(
                summaries=[best_summary],
                best_by_win_rate=best_summary,
                best_by_profit_factor=best_summary,
                best_by_expectancy=best_summary,
            )
        else:
            opt_result, best_trades = phase_sweep(args, entries, bars_map)

        # Persist best-config trades to DB
        if best_trades:
            db.clear_trades()
            db.insert_trades(best_trades)
            print(f"[main] {len(best_trades)} trades from best config saved to DB.")

        # ── Phase 5: Report ──────────────────────────────────────────────
        phase_report(args, opt_result, best_trades)

        # ── Phase 6: Sanity checks (optional) ────────────────────────────
        if getattr(args, "sanity", False):
            from .sanity_check import run_all_checks
            from .run_sanity import _print_result

            # Determine PT/SL used for best config
            if args.auto_tune:
                sanity_pt = tune_result.best_pt
                sanity_sl = tune_result.best_sl
            else:
                sanity_pt = args.pt
                sanity_sl = args.sl

            print(f"\n{'═'*60}")
            print(f"  PHASE 6 — Sanity Checks")
            print(f"{'═'*60}")

            results = run_all_checks(
                entries,
                bars_map,
                best_trades,
                pt=sanity_pt,
                sl=sanity_sl,
                checks=getattr(args, "sanity_checks", None),
            )

            for r in results:
                _print_result(r)

            passes = sum(1 for r in results if r["status"] == "PASS")
            warns  = sum(1 for r in results if r["status"] == "WARN")
            fails  = sum(1 for r in results if r["status"] == "FAIL")
            print(f"\n[main] Sanity: ✅ {passes} PASS  ⚠️  {warns} WARN  ❌ {fails} FAIL\n")

    finally:
        db.close()


if __name__ == "__main__":
    main()
