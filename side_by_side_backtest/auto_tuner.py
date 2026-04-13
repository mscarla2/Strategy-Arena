"""
Auto-Tuner — Bayesian Optimisation of PT / SL Parameters
=========================================================

Uses Optuna (TPE sampler) to intelligently search the profit-target /
stop-loss space without requiring manual range specification.

Rather than an exhaustive grid sweep, the tuner runs N trials where each
trial is guided by the results of previous trials, converging on the
parameter combination that maximises **expectancy** (or an alternative
objective you choose).

Objective hierarchy
-------------------
Primary   : expectancy  (avg $ made per trade — accounts for WR and magnitude)
Tie-break : profit_factor (gross profit / gross loss)
Guard     : at least `min_trades` trades must fire for a config to be viable

Search space (defaults — all configurable)
------------------------------------------
profit_target_pct  : 0.25 – 20.0  (log-uniform to explore small and large targets)
stop_loss_pct      : 0.1  – 10.0  (log-uniform)
pattern_tolerance  : 0.001 – 0.05 (log-uniform, fraction of price)

Usage
-----
    from side_by_side_backtest.auto_tuner import auto_tune
    result = auto_tune(entries, bars_map, n_trials=100)
    print(result.best_pt, result.best_sl, result.best_expectancy)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import optuna

from .models import WatchlistEntry
from .optimizer import compute_summary
from .simulator import simulate_all

# Silence optuna's own verbose logging — we print our own progress
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AutoTuneResult:
    """Best parameters found by the Bayesian search."""
    best_pt: float                  # profit_target_pct
    best_sl: float                  # stop_loss_pct
    best_tolerance: float           # pattern_tolerance_pct
    best_expectancy: float
    best_win_rate: float
    best_profit_factor: float
    best_total_trades: int
    n_trials: int
    n_viable_trials: int            # trials with >= min_trades trades
    all_trials: List[dict] = field(default_factory=list)  # raw per-trial data

    def summary_str(self) -> str:
        return (
            f"Auto-Tune Result ({self.n_trials} trials, {self.n_viable_trials} viable)\n"
            f"  Best PT       : {self.best_pt:.2f}%\n"
            f"  Best SL       : {self.best_sl:.2f}%\n"
            f"  Best Tolerance: {self.best_tolerance*100:.2f}%\n"
            f"  Expectancy    : {self.best_expectancy:+.4f}\n"
            f"  Win Rate      : {self.best_win_rate:.1%}\n"
            f"  Profit Factor : {self.best_profit_factor:.3f}\n"
            f"  Total Trades  : {self.best_total_trades}\n"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def auto_tune(
    entries: List[WatchlistEntry],
    bars_map: Dict,
    n_trials: int = 100,
    min_trades: int = 5,
    pt_low: float = 0.25,
    pt_high: float = 20.0,
    sl_low: float = 0.1,
    sl_high: float = 10.0,
    tol_low: float = 0.001,
    tol_high: float = 0.05,
    tune_tolerance: bool = False,   # set True to also optimise pattern tolerance
    objective: str = "expectancy",  # "expectancy" | "profit_factor" | "win_rate"
    seed: int = 42,
    verbose: bool = True,
) -> AutoTuneResult:
    """
    Run Bayesian optimisation over PT / SL (and optionally pattern tolerance).

    Parameters
    ----------
    entries         : Parsed watchlist entries.
    bars_map        : {ticker: DataFrame} of 5-min OHLCV.
    n_trials        : Number of Optuna trials to run.
    min_trades      : Minimum trades for a config to be considered viable.
    pt_low/pt_high  : Search bounds for profit_target_pct.
    sl_low/sl_high  : Search bounds for stop_loss_pct.
    tol_low/tol_high: Search bounds for pattern_tolerance_pct.
    tune_tolerance  : Whether to also optimise pattern tolerance.
    objective       : Metric to maximise: "expectancy", "profit_factor", or "win_rate".
    seed            : Random seed for reproducibility.
    verbose         : Print trial-by-trial progress.

    Returns
    -------
    AutoTuneResult with the best parameters found.
    """
    all_trials: list[dict] = []

    def _objective(trial: optuna.Trial) -> float:
        pt  = trial.suggest_float("profit_target_pct", pt_low,  pt_high,  log=True)
        sl  = trial.suggest_float("stop_loss_pct",     sl_low,  sl_high,  log=True)
        tol = (
            trial.suggest_float("tolerance", tol_low, tol_high, log=True)
            if tune_tolerance
            else 0.01
        )

        trades = simulate_all(
            entries,
            bars_map,
            profit_target_pct=pt,
            stop_loss_pct=sl,
            pattern_tolerance_pct=tol,
            verbose=False,
        )

        summary = compute_summary(trades, profit_target_pct=pt, stop_loss_pct=sl)

        # Record for later inspection
        all_trials.append({
            "trial": trial.number,
            "pt": round(pt, 4),
            "sl": round(sl, 4),
            "tolerance": round(tol, 4),
            "total_trades": summary.total_trades,
            "win_rate": summary.win_rate,
            "expectancy": summary.expectancy,
            "profit_factor": summary.profit_factor,
        })

        if verbose:
            viable = summary.total_trades >= min_trades
            tag = "✓" if viable else "✗"
            print(
                f"  [{tag}] Trial {trial.number+1:>3}/{n_trials}"
                f"  PT={pt:5.2f}%  SL={sl:4.2f}%"
                f"  trades={summary.total_trades:>3}"
                f"  WR={summary.win_rate:.0%}"
                f"  E={summary.expectancy:+.3f}"
                f"  PF={summary.profit_factor:.2f}"
            )

        # Penalise configs with too few trades — return very bad score
        if summary.total_trades < min_trades:
            return -999.0

        if objective == "profit_factor":
            score = summary.profit_factor if summary.profit_factor < 1e6 else 1e6
        elif objective == "win_rate":
            score = summary.win_rate
        else:  # default: expectancy
            score = summary.expectancy

        return score

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  AUTO-TUNE — Bayesian Search ({n_trials} trials)")
        print(f"  Objective: {objective.upper()}  |  Min trades: {min_trades}")
        print(f"  PT range: {pt_low}%–{pt_high}%  |  SL range: {sl_low}%–{sl_high}%")
        if tune_tolerance:
            print(f"  Tolerance range: {tol_low*100:.1f}%–{tol_high*100:.1f}%")
        print(f"{'═'*60}")

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)

    # Pull best params
    best = study.best_params
    best_pt  = best["profit_target_pct"]
    best_sl  = best["stop_loss_pct"]
    best_tol = best.get("tolerance", 0.01)

    # Re-simulate with best params to get accurate summary
    best_trades = simulate_all(
        entries,
        bars_map,
        profit_target_pct=best_pt,
        stop_loss_pct=best_sl,
        pattern_tolerance_pct=best_tol,
        verbose=False,
    )
    best_summary = compute_summary(best_trades, best_pt, best_sl)

    # Count viable trials
    n_viable = sum(1 for t in all_trials if t["total_trades"] >= min_trades)

    result = AutoTuneResult(
        best_pt=round(best_pt, 4),
        best_sl=round(best_sl, 4),
        best_tolerance=round(best_tol, 4),
        best_expectancy=best_summary.expectancy,
        best_win_rate=best_summary.win_rate,
        best_profit_factor=best_summary.profit_factor,
        best_total_trades=best_summary.total_trades,
        n_trials=n_trials,
        n_viable_trials=n_viable,
        all_trials=all_trials,
    )

    if verbose:
        print(f"\n{result.summary_str()}")

    return result


def auto_tune_to_dataframe(result: AutoTuneResult):
    """Convert all trial records to a pandas DataFrame for analysis."""
    import pandas as pd
    return pd.DataFrame(result.all_trials).sort_values("expectancy", ascending=False)
