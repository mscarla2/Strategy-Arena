"""
Sanity Check CLI Entry Point
============================
Runs the 6-check diagnostic suite against any set of tickers using the
existing ohlcv_cache/ parquet files — no scraping or re-fetching needed.

Usage
-----
    # All 6 checks on UGRO, ANNA, TURB (prior-window support — clean):
    python -m side_by_side_backtest.run_sanity --tickers UGRO ANNA TURB

    # Specific checks only:
    python -m side_by_side_backtest.run_sanity --tickers UGRO ANNA TURB --checks A B F

    # Use same-window support (leaky, for comparison):
    python -m side_by_side_backtest.run_sanity --tickers UGRO --same-window

    # Custom PT/SL baseline:
    python -m side_by_side_backtest.run_sanity --tickers UGRO ANNA --pt 2.0 --sl 1.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_sanity",
        description="Side-by-Side Body backtest sanity checker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tickers", nargs="+", required=True, help="Tickers to test, e.g. UGRO ANNA TURB")
    p.add_argument("--checks", nargs="+", default=None,
                   help="Which checks to run: A B C D E F (default: all)")
    p.add_argument("--pt",  type=float, default=2.0, help="Profit-target %% for base simulation")
    p.add_argument("--sl",  type=float, default=1.0, help="Stop-loss %% for base simulation")
    p.add_argument("--same-window", action="store_true",
                   help="Use same-window support (leaky — for comparison only)")
    p.add_argument("--n-shuffles", type=int, default=20,
                   help="Number of shuffle iterations for Check A")
    p.add_argument("--export", action="store_true",
                   help="Export sanity report to reports/sanity_report.txt")
    return p


def _print_result(result: dict) -> None:
    """Pretty-print one SanityResult to the terminal."""
    status_icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️ ", "INFO": "ℹ️ "}.get(result["status"], "?")
    print(f"\n{'─'*60}")
    print(f"  {status_icon} [{result['status']}] {result['name']}")
    print(f"  {result['headline']}")

    rows = result.get("rows", [])
    if rows:
        # Print rows as a simple aligned table
        if isinstance(rows[0], dict):
            headers = list(rows[0].keys())
            col_widths = {h: max(len(h), max(len(str(r.get(h, ""))) for r in rows)) for h in headers}
            header_line = "  " + "  ".join(h.ljust(col_widths[h]) for h in headers)
            sep_line    = "  " + "  ".join("-" * col_widths[h] for h in headers)
            print(header_line)
            print(sep_line)
            for row in rows:
                print("  " + "  ".join(str(row.get(h, "")).ljust(col_widths[h]) for h in headers))


def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    tickers = [t.upper() for t in args.tickers]
    use_prior = not args.same_window

    print(f"\n{'═'*60}")
    print(f"  Side-by-Side Sanity Check Suite")
    print(f"  Tickers : {', '.join(tickers)}")
    print(f"  Support : {'prior-window (clean)' if use_prior else 'same-window (leaky)'}")
    print(f"  PT={args.pt}%  SL={args.sl}%")
    print(f"  Checks  : {', '.join(args.checks) if args.checks else 'ALL (A–F)'}")
    print(f"{'═'*60}")

    # ── Build entries from cache ─────────────────────────────────────────────
    from .watchlist_builder import build_watchlist_from_tickers
    entries = build_watchlist_from_tickers(tickers, use_prior_window=use_prior)

    if not entries:
        print("[run_sanity] ERROR: No entries generated — check ohlcv_cache/ for these tickers.")
        sys.exit(1)

    # ── Build bars_map from cache ────────────────────────────────────────────
    from .data_fetcher import fetch_bars_batch
    bars_map = fetch_bars_batch(entries, provider="schwab_data")

    if not bars_map:
        print("[run_sanity] ERROR: No OHLCV data available — run main pipeline first to populate cache.")
        sys.exit(1)

    # ── Run base simulation to get trades for checks C and F ────────────────
    from .simulator import simulate_all
    base_trades = simulate_all(entries, bars_map, profit_target_pct=args.pt, stop_loss_pct=args.sl)
    print(f"\n[run_sanity] Base simulation: {len(base_trades)} trades (PT={args.pt}%  SL={args.sl}%)")

    # ── Run sanity checks ────────────────────────────────────────────────────
    from .sanity_check import run_all_checks
    results = run_all_checks(
        entries,
        bars_map,
        base_trades,
        pt=args.pt,
        sl=args.sl,
        checks=args.checks,
    )

    # ── Print results ────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  SANITY CHECK RESULTS  ({len(results)} checks)")
    print(f"{'═'*60}")

    summary_lines = []
    for r in results:
        _print_result(r)
        icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️ "}.get(r["status"], "?")
        summary_lines.append(f"  {icon} {r['name']:25s}  {r['status']:4s}  {r['headline'][:60]}")

    print(f"\n{'═'*60}")
    print("  SUMMARY")
    print(f"{'═'*60}")
    for line in summary_lines:
        print(line)

    passes  = sum(1 for r in results if r["status"] == "PASS")
    warns   = sum(1 for r in results if r["status"] == "WARN")
    fails   = sum(1 for r in results if r["status"] == "FAIL")
    print(f"\n  ✅ {passes} PASS  ⚠️  {warns} WARN  ❌ {fails} FAIL\n")

    # ── Optional export ──────────────────────────────────────────────────────
    if args.export:
        output_dir = Path(__file__).parent / "reports"
        output_dir.mkdir(exist_ok=True)
        report_path = output_dir / "sanity_report.txt"
        with report_path.open("w") as fh:
            fh.write(f"Sanity Check Report — {', '.join(tickers)}\n")
            fh.write("=" * 60 + "\n\n")
            for line in summary_lines:
                fh.write(line.replace("✅", "PASS").replace("❌", "FAIL").replace("⚠️ ", "WARN") + "\n")
            fh.write(f"\nPASS={passes}  WARN={warns}  FAIL={fails}\n")
        print(f"[run_sanity] Report exported → {report_path}")


if __name__ == "__main__":
    main()
