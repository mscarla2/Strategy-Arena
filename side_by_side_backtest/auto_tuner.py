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
from .simulator import simulate_all, build_simulation_caches, SimulationCaches

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

    # ── Walk-forward OOS fields (populated when validate_oos=True) ────────────
    oos_expectancy:     Optional[float] = None   # OOS expectancy with best params
    oos_win_rate:       Optional[float] = None
    oos_profit_factor:  Optional[float] = None
    oos_total_trades:   int             = 0
    oos_split_pct:      float           = 0.0    # fraction used for OOS (e.g. 0.3)

    def summary_str(self) -> str:
        s = (
            f"Auto-Tune Result ({self.n_trials} trials, {self.n_viable_trials} viable)\n"
            f"  Best PT       : {self.best_pt:.2f}%\n"
            f"  Best SL       : {self.best_sl:.2f}%\n"
            f"  Best Tolerance: {self.best_tolerance*100:.2f}%\n"
            f"  ── In-Sample ──────────────────────────\n"
            f"  Expectancy    : {self.best_expectancy:+.4f}\n"
            f"  Win Rate      : {self.best_win_rate:.1%}\n"
            f"  Profit Factor : {self.best_profit_factor:.3f}\n"
            f"  Total Trades  : {self.best_total_trades}\n"
        )
        if self.oos_expectancy is not None:
            s += (
                f"  ── Out-of-Sample ({self.oos_split_pct:.0%} of data) ─────\n"
                f"  OOS Expectancy  : {self.oos_expectancy:+.4f}"
                f"{'  ✅ Holds up' if self.oos_expectancy > 0 else '  ⚠️ Degraded'}\n"
                f"  OOS Win Rate    : {self.oos_win_rate:.1%}\n"
                f"  OOS Profit Factor: {self.oos_profit_factor:.3f}\n"
                f"  OOS Trades      : {self.oos_total_trades}\n"
            )
        return s


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
    n_jobs: int = -1,               # Perf 4: parallel trial workers (-1 = all cores)
    verbose: bool = True,
    validate_oos: bool = False,     # Phase E: run walk-forward OOS check after tuning
    oos_split: float = 0.30,        # fraction of entries reserved for OOS (default 30%)
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

    # ── Perf 2+3: Build S/R + pattern caches ONCE before any trials run ───────
    # This is the expensive step (≈62s for 349 tickers). By hoisting it here,
    # each of the 100 trials only costs ~3s instead of ~65s.
    if verbose:
        print("[auto_tune] Pre-computing S/R levels and pattern maps …")
    import time as _time
    _t0 = _time.perf_counter()
    _caches: SimulationCaches = build_simulation_caches(
        entries, bars_map,
        pattern_tolerance_pct=0.01,   # use default; per-trial tolerance only affects entry detection threshold
        verbose=False,
    )
    _t1 = _time.perf_counter()
    if verbose:
        print(f"[auto_tune] Cache built in {_t1-_t0:.1f}s — starting {n_trials} trials …")

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
            caches=_caches,
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
    # Perf 4: run trials in parallel across CPU cores.
    # n_jobs=-1 uses all available cores; each trial calls simulate_all independently.
    # Note: verbose printing from parallel workers may interleave — acceptable for tuning.
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False, n_jobs=n_jobs)

    # Pull best params
    best = study.best_params
    best_pt  = best["profit_target_pct"]
    best_sl  = best["stop_loss_pct"]
    best_tol = best.get("tolerance", 0.01)

    # Re-simulate with best params to get accurate summary (reuse caches)
    best_trades = simulate_all(
        entries,
        bars_map,
        profit_target_pct=best_pt,
        stop_loss_pct=best_sl,
        pattern_tolerance_pct=best_tol,
        verbose=False,
        caches=_caches,
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

    # ── Walk-forward OOS validation ───────────────────────────────────────────
    if validate_oos and entries:
        # Split entries chronologically: first (1-oos_split) for IS, rest for OOS.
        # We sort by post_timestamp so the split respects time order.
        sorted_entries = sorted(
            [e for e in entries if e.post_timestamp is not None],
            key=lambda e: e.post_timestamp,
        )
        # Any entries without timestamps go to IS
        no_ts = [e for e in entries if e.post_timestamp is None]
        all_sorted = no_ts + sorted_entries

        split_idx = max(1, int(len(all_sorted) * (1 - oos_split)))
        oos_entries = all_sorted[split_idx:]

        if oos_entries and verbose:
            print(
                f"\n[auto_tune] Walk-forward OOS: {len(oos_entries)} entries "
                f"({oos_split:.0%} of total) with best params "
                f"PT={best_pt:.2f}%  SL={best_sl:.2f}% …"
            )

        if oos_entries:
            oos_trades = simulate_all(
                oos_entries,
                bars_map,
                profit_target_pct=best_pt,
                stop_loss_pct=best_sl,
                pattern_tolerance_pct=best_tol,
                verbose=False,
                caches=_caches,
            )
            oos_summary = compute_summary(oos_trades, best_pt, best_sl)
            result.oos_expectancy    = oos_summary.expectancy
            result.oos_win_rate      = oos_summary.win_rate
            result.oos_profit_factor = oos_summary.profit_factor
            result.oos_total_trades  = oos_summary.total_trades
            result.oos_split_pct     = oos_split

            if verbose:
                degraded = (
                    oos_summary.expectancy < result.best_expectancy * 0.5
                    and oos_summary.expectancy < 0
                )
                flag = "⚠️  POSSIBLE OVERFITTING — OOS significantly worse than IS" if degraded else "✅  OOS results broadly consistent with IS"
                print(
                    f"[auto_tune] OOS result: "
                    f"E={oos_summary.expectancy:+.4f}  "
                    f"WR={oos_summary.win_rate:.1%}  "
                    f"PF={oos_summary.profit_factor:.3f}  "
                    f"trades={oos_summary.total_trades}"
                )
                print(f"[auto_tune] {flag}")

    if verbose:
        print(f"\n{result.summary_str()}")

    return result


def auto_tune_to_dataframe(result: AutoTuneResult):
    """Convert all trial records to a pandas DataFrame for analysis."""
    import pandas as pd
    return pd.DataFrame(result.all_trials).sort_values("expectancy", ascending=False)
