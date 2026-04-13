"""
Phase 4 — Performance Dashboard / Report Generator
Renders results to the terminal and optionally exports CSV + PNG charts.

Outputs:
  1. Console summary table (Rich if available, plain-text fallback)
  2. optimization_results.csv  — full sweep grid
  3. heatmap_win_rate.png      — win-rate heat-map  (PT × SL)
  4. heatmap_profit_factor.png — profit-factor heat-map
  5. equity_curve.png          — cumulative PnL for the best configuration
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from .models import BacktestSummary, OptimizationResult, TradeResult
from .optimizer import summaries_to_dataframe

_OUTPUT_DIR = Path(__file__).parent / "reports"

# ---------------------------------------------------------------------------
# Rich / fallback table helpers
# ---------------------------------------------------------------------------

def _has_rich() -> bool:
    try:
        import rich  # noqa: F401
        return True
    except ImportError:
        return False


def _print_rich_table(df: pd.DataFrame, title: str = "") -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=title, show_lines=False, header_style="bold cyan")
    for col in df.columns:
        table.add_column(str(col), justify="right")
    for _, row in df.iterrows():
        table.add_row(*[str(v) for v in row])
    console.print(table)


def _print_plain_table(df: pd.DataFrame, title: str = "") -> None:
    if title:
        print(f"\n{'─' * 60}")
        print(f"  {title}")
        print(f"{'─' * 60}")
    print(df.to_string(index=False))
    print()


def print_table(df: pd.DataFrame, title: str = "") -> None:
    if _has_rich():
        _print_rich_table(df, title)
    else:
        _print_plain_table(df, title)


# ---------------------------------------------------------------------------
# Summary headline
# ---------------------------------------------------------------------------

def print_headline(result: OptimizationResult) -> None:
    lines = [
        "",
        "╔══════════════════════════════════════════════════════════════╗",
        "║        Side-by-Side Body Backtest — Optimization Summary      ║",
        "╚══════════════════════════════════════════════════════════════╝",
        "",
    ]
    viable = [s for s in result.summaries if s.total_trades >= 5]
    lines.append(f"  Total parameter combinations tested : {len(result.summaries)}")
    lines.append(f"  Combinations with ≥5 trades         : {len(viable)}")
    lines.append("")

    for label, best in [
        ("Best Win Rate",      result.best_by_win_rate),
        ("Best Profit Factor", result.best_by_profit_factor),
        ("Best Expectancy",    result.best_by_expectancy),
    ]:
        if best is None:
            continue
        lines.append(
            f"  {label:<22} PT={best.profit_target_pct:.1f}%  "
            f"SL={best.stop_loss_pct:.1f}%  "
            f"→  WR={best.win_rate:.1%}  "
            f"PF={best.profit_factor:.2f}  "
            f"E={best.expectancy:+.3f}  "
            f"({best.total_trades} trades)"
        )
    lines.append("")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Top-N table
# ---------------------------------------------------------------------------

def print_top_n(result: OptimizationResult, n: int = 10, sort_by: str = "expectancy") -> None:
    df = summaries_to_dataframe(result)
    if df.empty:
        print("[report] No summaries to display.")
        return

    df = df[df["total_trades"] >= 5].copy()
    if df.empty:
        print("[report] No viable configurations (< 5 trades each).")
        return

    sort_col = sort_by if sort_by in df.columns else "expectancy"
    df = df.sort_values(sort_col, ascending=False).head(n)

    display_cols = [
        "profit_target_pct", "stop_loss_pct", "total_trades",
        "win_rate", "avg_win_pct", "avg_loss_pct",
        "expectancy", "profit_factor", "avg_hold_bars",
        "support_respect_rate",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    df_display = df[display_cols].copy()

    # Format percentages
    for col in ("win_rate", "support_respect_rate"):
        if col in df_display.columns:
            df_display[col] = df_display[col].map(lambda x: f"{x:.1%}")

    for col in ("avg_win_pct", "avg_loss_pct", "expectancy"):
        if col in df_display.columns:
            df_display[col] = df_display[col].map(lambda x: f"{x:+.3f}%")

    print_table(df_display, title=f"Top {n} Configurations (sorted by {sort_col})")


# ---------------------------------------------------------------------------
# Per-session breakdown
# ---------------------------------------------------------------------------

def print_session_breakdown(best: Optional[BacktestSummary]) -> None:
    if best is None or not best.session_win_rates:
        return
    print("\n  ── Session Win Rates (best by expectancy) ──")
    for sess, wr in best.session_win_rates.items():
        bar = "█" * int(wr * 20)
        print(f"  {sess:<15} {wr:.1%}  {bar}")
    print()


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(result: OptimizationResult, output_dir: Path = _OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = summaries_to_dataframe(result)
    out_path = output_dir / "optimization_results.csv"
    df.to_csv(out_path, index=False)
    print(f"[report] CSV exported → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Heat-map charts
# ---------------------------------------------------------------------------

def _try_heatmap(df: pd.DataFrame, value_col: str, title: str, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import seaborn as sns  # type: ignore
    except ImportError:
        print(f"[report] matplotlib/seaborn not installed — skipping {value_col} heatmap.")
        return

    if "profit_target_pct" not in df.columns or "stop_loss_pct" not in df.columns:
        return

    pivot = df.pivot_table(
        index="stop_loss_pct",
        columns="profit_target_pct",
        values=value_col,
        aggfunc="mean",
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        ax=ax,
        linewidths=0.3,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Profit Target %")
    ax.set_ylabel("Stop Loss %")
    plt.tight_layout()

    fname = output_dir / f"heatmap_{value_col}.png"
    plt.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"[report] Heat-map saved → {fname}")


def export_heatmaps(result: OptimizationResult, output_dir: Path = _OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = summaries_to_dataframe(result)
    if df.empty:
        return
    _try_heatmap(df, "win_rate",       "Win Rate Heat-map (PT × SL)",       output_dir)
    _try_heatmap(df, "profit_factor",  "Profit Factor Heat-map (PT × SL)",  output_dir)
    _try_heatmap(df, "expectancy",     "Expectancy Heat-map (PT × SL)",     output_dir)


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

def export_equity_curve(
    trades: List[TradeResult],
    label: str = "Best Config",
    output_dir: Path = _OUTPUT_DIR,
) -> None:
    if not trades:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("[report] matplotlib not installed — skipping equity curve.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_trades = sorted(trades, key=lambda t: t.entry_ts)
    cumulative = [0.0]
    for t in sorted_trades:
        cumulative.append(cumulative[-1] + t.pnl_pct)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(cumulative, linewidth=1.5, color="#2196F3")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(range(len(cumulative)), cumulative, 0,
                    where=[c >= 0 for c in cumulative], alpha=0.15, color="green")
    ax.fill_between(range(len(cumulative)), cumulative, 0,
                    where=[c < 0 for c in cumulative],  alpha=0.15, color="red")
    ax.set_title(f"Cumulative PnL% — {label}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative PnL (%)")
    plt.tight_layout()

    fname = output_dir / "equity_curve.png"
    plt.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"[report] Equity curve saved → {fname}")


# ---------------------------------------------------------------------------
# Master report function
# ---------------------------------------------------------------------------

def generate_report(
    result: OptimizationResult,
    best_trades: Optional[List[TradeResult]] = None,
    export: bool = True,
    output_dir: Path = _OUTPUT_DIR,
) -> None:
    """
    Print all console output and optionally export CSV / PNG artefacts.

    Parameters
    ----------
    result      : OptimizationResult from optimizer.optimize().
    best_trades : TradeResult list from the chosen best configuration
                  (used for equity curve). If None, no equity curve is produced.
    export      : Whether to write files to disk.
    output_dir  : Where to write export artefacts.
    """
    print_headline(result)
    print_top_n(result, n=15, sort_by="expectancy")
    print_top_n(result, n=10, sort_by="win_rate")
    print_session_breakdown(result.best_by_expectancy)

    if export:
        export_csv(result, output_dir)
        export_heatmaps(result, output_dir)
        if best_trades:
            label = (
                f"PT={result.best_by_expectancy.profit_target_pct}%  "
                f"SL={result.best_by_expectancy.stop_loss_pct}%"
                if result.best_by_expectancy
                else "Best Config"
            )
            export_equity_curve(best_trades, label=label, output_dir=output_dir)
