"""
Phase 4 — Parameter Optimizer
Sweeps profit_target_pct (X%) from 0.5 to 5.0 in configurable steps,
optionally sweeps stop_loss_pct (Y%) as well, and computes performance
metrics for every (X, Y) combination.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .models import (
    BacktestSummary,
    OptimizationResult,
    SessionType,
    TradeResult,
    WatchlistEntry,
)
from .simulator import simulate_all


# ---------------------------------------------------------------------------
# Metric computation for a list of trades
# ---------------------------------------------------------------------------

def compute_summary(
    trades: List[TradeResult],
    profit_target_pct: float,
    stop_loss_pct: float,
) -> BacktestSummary:
    """Aggregate a flat list of TradeResult objects into a BacktestSummary."""
    total = len(trades)
    if total == 0:
        return BacktestSummary(
            profit_target_pct=profit_target_pct,
            stop_loss_pct=stop_loss_pct,
        )

    wins    = [t for t in trades if t.outcome == "win"]
    losses  = [t for t in trades if t.outcome == "loss"]
    timeouts = [t for t in trades if t.outcome == "timeout"]

    n_wins   = len(wins)
    n_losses = len(losses)
    n_time   = len(timeouts)

    win_rate = n_wins / total if total else 0.0

    avg_win  = float(np.mean([t.pnl_pct for t in wins]))   if wins   else 0.0
    avg_loss = float(np.mean([t.pnl_pct for t in losses])) if losses else 0.0

    # Expectancy: (WR × avg_win) + (LR × avg_loss)
    # Note: avg_loss is typically negative, so the formula stays additive.
    loss_rate = (n_losses + n_time) / total
    expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)

    # Profit Factor: gross_profit / gross_loss
    gross_profit = sum(t.pnl_pct for t in wins)
    gross_loss   = abs(sum(t.pnl_pct for t in losses + timeouts))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    avg_hold = float(np.mean([t.hold_bars for t in trades])) if trades else 0.0

    support_respected_count = sum(1 for t in trades if t.support_respected)
    support_respect_rate = support_respected_count / total if total else 0.0

    # Per-session win rates
    session_win_rates: dict[str, float] = {}
    for stype in SessionType:
        sess_trades = [t for t in trades if t.session_type == stype]
        if sess_trades:
            sess_wins = sum(1 for t in sess_trades if t.outcome == "win")
            session_win_rates[stype.value] = round(sess_wins / len(sess_trades), 4)

    return BacktestSummary(
        profit_target_pct=profit_target_pct,
        stop_loss_pct=stop_loss_pct,
        total_trades=total,
        wins=n_wins,
        losses=n_losses,
        timeouts=n_time,
        win_rate=round(win_rate, 4),
        avg_win_pct=round(avg_win, 4),
        avg_loss_pct=round(avg_loss, 4),
        expectancy=round(expectancy, 4),
        profit_factor=round(profit_factor, 4),
        avg_hold_bars=round(avg_hold, 2),
        support_respect_rate=round(support_respect_rate, 4),
        session_win_rates=session_win_rates,
    )


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def optimize(
    entries: List[WatchlistEntry],
    bars_map: Dict,
    profit_target_range: tuple = (0.5, 5.0, 0.5),   # (start, stop, step) in pct
    stop_loss_range: tuple = (1.0, 1.0, 1.0),         # fixed at 1 % by default
    pattern_tolerance_pct: float = 0.01,
    verbose: bool = True,
) -> OptimizationResult:
    """
    Sweep (profit_target_pct, stop_loss_pct) combinations and simulate each.

    Parameters
    ----------
    entries               : Parsed watchlist entries.
    bars_map              : {ticker: DataFrame} of 5-min OHLCV.
    profit_target_range   : (start%, stop%, step%) tuple — default 0.5→5.0 step 0.5.
    stop_loss_range       : Same structure; set start==stop to fix the stop.
    pattern_tolerance_pct : Passed to the pattern engine.
    verbose               : Print progress.

    Returns
    -------
    OptimizationResult with summaries and best-parameter pointers.
    """
    # Build grid
    pt_start, pt_stop, pt_step = profit_target_range
    sl_start, sl_stop, sl_step = stop_loss_range

    pt_values = list(
        np.round(np.arange(pt_start, pt_stop + pt_step * 0.01, pt_step), 4)
    )
    sl_values = list(
        np.round(np.arange(sl_start, sl_stop + sl_step * 0.01, sl_step), 4)
    )

    combinations = [(pt, sl) for pt in pt_values for sl in sl_values]
    total_combinations = len(combinations)

    if verbose:
        print(
            f"[optimizer] Starting sweep: {len(pt_values)} PT × "
            f"{len(sl_values)} SL = {total_combinations} combinations, "
            f"{len(entries)} entries."
        )

    summaries: list[BacktestSummary] = []

    for idx, (pt, sl) in enumerate(combinations, 1):
        if verbose:
            print(f"  [{idx:>3}/{total_combinations}] PT={pt:.1f}%  SL={sl:.1f}%", end=" … ")

        trades = simulate_all(
            entries,
            bars_map,
            profit_target_pct=pt,
            stop_loss_pct=sl,
            pattern_tolerance_pct=pattern_tolerance_pct,
            verbose=False,
        )

        summary = compute_summary(trades, profit_target_pct=pt, stop_loss_pct=sl)
        summaries.append(summary)

        if verbose:
            print(
                f"trades={summary.total_trades}  "
                f"WR={summary.win_rate:.1%}  "
                f"PF={summary.profit_factor:.2f}  "
                f"E={summary.expectancy:+.3f}"
            )

    # Find bests (only among summaries with at least 5 trades)
    viable = [s for s in summaries if s.total_trades >= 5]

    best_wr = max(viable, key=lambda s: s.win_rate,       default=None)
    best_pf = max(viable, key=lambda s: s.profit_factor,  default=None)
    best_ex = max(viable, key=lambda s: s.expectancy,     default=None)

    if verbose and viable:
        print("\n[optimizer] ── Best Configurations ──")
        if best_wr:
            print(f"  Best Win Rate      : PT={best_wr.profit_target_pct}%  "
                  f"SL={best_wr.stop_loss_pct}%  → {best_wr.win_rate:.1%}")
        if best_pf:
            print(f"  Best Profit Factor : PT={best_pf.profit_target_pct}%  "
                  f"SL={best_pf.stop_loss_pct}%  → {best_pf.profit_factor:.2f}")
        if best_ex:
            print(f"  Best Expectancy    : PT={best_ex.profit_target_pct}%  "
                  f"SL={best_ex.stop_loss_pct}%  → {best_ex.expectancy:+.3f}")

    return OptimizationResult(
        summaries=summaries,
        best_by_win_rate=best_wr,
        best_by_profit_factor=best_pf,
        best_by_expectancy=best_ex,
    )


# ---------------------------------------------------------------------------
# Convenience: convert results to DataFrame
# ---------------------------------------------------------------------------

def summaries_to_dataframe(result: OptimizationResult) -> pd.DataFrame:
    """Convert OptimizationResult.summaries into a tidy pandas DataFrame."""
    rows = [s.dict() for s in result.summaries]
    df = pd.DataFrame(rows)
    # Flatten session_win_rates dict into columns
    if "session_win_rates" in df.columns:
        sr = df["session_win_rates"].apply(pd.Series).add_prefix("wr_")
        df = pd.concat([df.drop(columns=["session_win_rates"]), sr], axis=1)
    return df
