"""
Sanity Check Suite — 6 Independent Diagnostic Checks
=====================================================
Validates that backtest results have genuine edge and are not artifacts
of overfitting, data leakage, duplicate trades, or cherry-picked parameters.

Each check function returns a SanityResult dict:
    {
        "name":        str,        # check identifier
        "status":      str,        # "PASS" | "FAIL" | "WARN" | "INFO"
        "headline":    str,        # one-line summary
        "details":     dict,       # check-specific numeric data
        "rows":        list[dict], # optional table rows for Rich output
    }

Usage (programmatic)
--------------------
    from side_by_side_backtest.sanity_check import run_all_checks
    results = run_all_checks(entries, bars_map, trades, pt=2.0, sl=1.0)
    for r in results:
        print(r["name"], r["status"], r["headline"])

Usage (CLI)
-----------
    python -m side_by_side_backtest.run_sanity --tickers UGRO ANNA TURB
    python -m side_by_side_backtest.run_sanity --tickers UGRO --checks A B F
"""
from __future__ import annotations

import copy
import random
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

SanityResult = Dict[str, Any]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _win_rate(trades) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.outcome == "win")
    return wins / len(trades)


def _expectancy(trades) -> float:
    """Avg PnL% per trade (including losses and timeouts)."""
    if not trades:
        return 0.0
    return sum(t.pnl_pct for t in trades) / len(trades)


def _profit_factor(trades) -> float:
    gross_win  = sum(t.pnl_pct for t in trades if t.pnl_pct > 0)
    gross_loss = abs(sum(t.pnl_pct for t in trades if t.pnl_pct < 0))
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss


def _status(condition: bool, *, warn: bool = False) -> str:
    if condition:
        return "PASS"
    return "WARN" if warn else "FAIL"


# ---------------------------------------------------------------------------
# Check A — Shuffle Control
# ---------------------------------------------------------------------------

def check_a_shuffle_control(
    entries,
    bars_map: dict,
    pt: float,
    sl: float,
    n_shuffles: int = 20,
    seed: int = 42,
) -> SanityResult:
    """
    Randomise support_level → re-simulate N times.
    Real win-rate should be meaningfully above random baseline.
    PASS if real_wr > mean(shuffles) + 1.5 * stdev(shuffles).
    """
    from .simulator import simulate_all

    rng = random.Random(seed)

    # Gather all support levels to define a realistic price range
    supports = [e.support_level for e in entries if e.support_level is not None]
    if len(supports) < 2:
        return {
            "name": "A-Shuffle Control",
            "status": "WARN",
            "headline": "Not enough support levels to run shuffle test (need ≥2 entries with support).",
            "details": {},
            "rows": [],
        }

    p5  = sorted(supports)[max(0, int(len(supports) * 0.05))]
    p95 = sorted(supports)[min(len(supports) - 1, int(len(supports) * 0.95))]

    # Real win rate
    real_trades = simulate_all(entries, bars_map, profit_target_pct=pt, stop_loss_pct=sl)
    real_wr = _win_rate(real_trades)

    shuffle_wrs: list[float] = []
    for _ in range(n_shuffles):
        shuffled = []
        for e in entries:
            e2 = e.model_copy()
            e2 = e2.model_copy(update={"support_level": round(rng.uniform(p5, p95), 4)})
            shuffled.append(e2)
        sh_trades = simulate_all(shuffled, bars_map, profit_target_pct=pt, stop_loss_pct=sl)
        shuffle_wrs.append(_win_rate(sh_trades))

    if len(shuffle_wrs) < 2:
        mean_sh = shuffle_wrs[0] if shuffle_wrs else 0.0
        std_sh  = 0.0
    else:
        mean_sh = statistics.mean(shuffle_wrs)
        std_sh  = statistics.stdev(shuffle_wrs)

    threshold = mean_sh + 1.5 * std_sh
    passes = real_wr > threshold

    rows = [
        {"metric": "Real win rate",        "value": f"{real_wr:.1%}"},
        {"metric": "Shuffle mean WR",       "value": f"{mean_sh:.1%}"},
        {"metric": "Shuffle stdev",         "value": f"{std_sh:.2%}"},
        {"metric": "Pass threshold (μ+1.5σ)", "value": f"{threshold:.1%}"},
        {"metric": "Real trades",           "value": str(len(real_trades))},
    ]

    return {
        "name": "A-Shuffle Control",
        "status": _status(passes),
        "headline": (
            f"Real WR {real_wr:.1%} {'>' if passes else '<='} "
            f"shuffle baseline {threshold:.1%} — "
            f"{'edge is above noise floor' if passes else 'no edge above random baseline'}"
        ),
        "details": {
            "real_wr": real_wr,
            "shuffle_mean_wr": mean_sh,
            "shuffle_std": std_sh,
            "threshold": threshold,
            "n_shuffles": n_shuffles,
            "real_trades": len(real_trades),
        },
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Check B — Out-of-Sample Split (70/30) + Walk-Forward
# ---------------------------------------------------------------------------

def check_b_out_of_sample(
    entries,
    bars_map: dict,
    pt: float,
    sl: float,
    split: float = 0.70,
    max_wr_gap: float = 0.15,
    walk_forward: bool = True,
    wf_folds: int = 5,
    min_test_trades: int = 5,
) -> SanityResult:
    """
    Out-of-sample robustness check. Two modes:

    Walk-Forward (default, walk_forward=True):
        Splits sorted entries into `wf_folds` equal time slices.
        For each fold k: train = folds[0..k-1], test = folds[k].
        Reports per-fold WR and average test WR.
        PASS if avg test WR is within max_wr_gap of avg train WR,
        AND at least half the folds have ≥ min_test_trades.

    Simple Split (walk_forward=False):
        Chronological train/test at `split` ratio (default 70/30).
        PASS if |train_WR - test_WR| < max_wr_gap and test has ≥ min_test_trades.
    """
    from .simulator import simulate_all

    sorted_entries = sorted(
        [e for e in entries if e.post_timestamp is not None],
        key=lambda e: e.post_timestamp,
    )
    undated = [e for e in entries if e.post_timestamp is None]

    if len(sorted_entries) < 4:
        return {
            "name": "B-Out-of-Sample",
            "status": "WARN",
            "headline": f"Too few dated entries ({len(sorted_entries)}) — need ≥4 for OOS test.",
            "details": {"dated_entries": len(sorted_entries)},
            "rows": [],
        }

    # ── Walk-Forward mode ─────────────────────────────────────────────────────
    if walk_forward and len(sorted_entries) >= wf_folds * 2:
        fold_size = len(sorted_entries) // wf_folds
        rows = []
        train_wrs, test_wrs = [], []
        viable_folds = 0

        for k in range(1, wf_folds):
            train_e = sorted_entries[: k * fold_size] + undated
            test_e  = sorted_entries[k * fold_size : (k + 1) * fold_size]
            if not test_e:
                continue
            train_t = simulate_all(train_e, bars_map, profit_target_pct=pt, stop_loss_pct=sl)
            test_t  = simulate_all(test_e,  bars_map, profit_target_pct=pt, stop_loss_pct=sl)
            twr = _win_rate(train_t)
            ewr = _win_rate(test_t)
            train_wrs.append(twr)
            test_wrs.append(ewr)
            if len(test_t) >= min_test_trades:
                viable_folds += 1
            rows.append({
                "fold":        f"Fold {k}/{wf_folds-1}",
                "train_size":  str(len(train_e)),
                "test_size":   str(len(test_e)),
                "train_WR":    f"{twr:.1%}",
                "test_WR":     f"{ewr:.1%}",
                "gap":         f"{abs(twr-ewr):.1%}",
                "test_trades": str(len(test_t)),
            })

        avg_train = statistics.mean(train_wrs) if train_wrs else 0.0
        avg_test  = statistics.mean(test_wrs)  if test_wrs  else 0.0
        avg_gap   = abs(avg_train - avg_test)
        passes    = avg_gap < max_wr_gap and viable_folds >= max(1, (wf_folds - 1) // 2)

        return {
            "name": "B-Out-of-Sample",
            "status": _status(passes, warn=(viable_folds == 0)),
            "headline": (
                f"Walk-Forward ({wf_folds-1} folds) — avg train WR {avg_train:.1%} → "
                f"avg test WR {avg_test:.1%} (gap {avg_gap:.1%}) — "
                f"{viable_folds}/{wf_folds-1} folds viable — "
                f"{'stable' if passes else 'unstable / overfit'}"
            ),
            "details": {
                "mode": "walk_forward", "folds": wf_folds,
                "avg_train_wr": avg_train, "avg_test_wr": avg_test,
                "avg_gap": avg_gap, "viable_folds": viable_folds,
            },
            "rows": rows,
        }

    # ── Simple split mode ─────────────────────────────────────────────────────
    split_idx     = max(1, int(len(sorted_entries) * split))
    train_entries = sorted_entries[:split_idx] + undated
    test_entries  = sorted_entries[split_idx:]

    train_trades = simulate_all(train_entries, bars_map, profit_target_pct=pt, stop_loss_pct=sl)
    test_trades  = simulate_all(test_entries,  bars_map, profit_target_pct=pt, stop_loss_pct=sl)

    train_wr = _win_rate(train_trades)
    test_wr  = _win_rate(test_trades)
    gap      = abs(train_wr - test_wr)
    passes   = gap < max_wr_gap and len(test_trades) >= min_test_trades

    split_pct = int(split * 100)
    rows = [
        {"split": f"Train (first {split_pct}%)",    "entries": str(len(train_entries)), "trades": str(len(train_trades)), "win_rate": f"{train_wr:.1%}"},
        {"split": f"Test  (last  {100-split_pct}%)", "entries": str(len(test_entries)),  "trades": str(len(test_trades)),  "win_rate": f"{test_wr:.1%}"},
        {"split": "Gap",                              "entries": "",                      "trades": "",                     "win_rate": f"{gap:.1%} ({'OK' if passes else 'OVERFIT?'})"},
    ]
    return {
        "name": "B-Out-of-Sample",
        "status": _status(passes, warn=(gap >= max_wr_gap and len(test_trades) < min_test_trades)),
        "headline": (
            f"Split {split_pct}/{100-split_pct} — Train WR {train_wr:.1%} → "
            f"Test WR {test_wr:.1%} (gap {gap:.1%}) — "
            f"{'within tolerance' if passes else 'gap too large, possible overfit'}"
        ),
        "details": {
            "mode": "simple_split", "split": split,
            "train_trades": len(train_trades), "test_trades": len(test_trades),
            "train_wr": train_wr, "test_wr": test_wr, "gap": gap,
        },
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Check C — Duplicate Trade Deduplication
# ---------------------------------------------------------------------------

def check_c_duplicate_trades(trades) -> SanityResult:
    """
    Detect trades with identical (ticker, entry_ts).
    Reports de-duped win rate vs raw win rate.
    PASS if duplicates < 5% of total trades.
    """
    seen: dict[tuple, Any] = {}
    duplicates = 0
    deduped: list = []

    for t in trades:
        key = (t.ticker, t.entry_ts)
        if key in seen:
            duplicates += 1
        else:
            seen[key] = t
            deduped.append(t)

    raw_wr   = _win_rate(trades)
    dedup_wr = _win_rate(deduped)
    dup_pct  = duplicates / len(trades) if trades else 0.0
    passes   = dup_pct < 0.05

    rows = [
        {"metric": "Total trades (raw)",       "value": str(len(trades))},
        {"metric": "Duplicate trades",         "value": f"{duplicates} ({dup_pct:.1%})"},
        {"metric": "Unique trades",            "value": str(len(deduped))},
        {"metric": "Raw win rate",             "value": f"{raw_wr:.1%}"},
        {"metric": "De-duped win rate",        "value": f"{dedup_wr:.1%}"},
        {"metric": "WR shift from dedup",      "value": f"{dedup_wr - raw_wr:+.1%}"},
    ]

    return {
        "name": "C-Duplicate Trades",
        "status": _status(passes, warn=(dup_pct >= 0.05 and dup_pct < 0.20)),
        "headline": (
            f"{duplicates} duplicate(s) found in {len(trades)} trades ({dup_pct:.1%}) — "
            f"{'clean' if passes else 'high duplicate rate — investigate entry logic'}"
        ),
        "details": {
            "total_trades":   len(trades),
            "duplicates":     duplicates,
            "unique_trades":  len(deduped),
            "raw_wr":         raw_wr,
            "deduped_wr":     dedup_wr,
            "dup_pct":        dup_pct,
        },
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Check D — Pattern Near Support (Tight Filter)
# ---------------------------------------------------------------------------

def check_d_pattern_near_support(
    entries,
    bars_map: dict,
    pt: float,
    sl: float,
    tight_proximity_pct: float = 0.005,
    max_wr_drop: float = 0.10,
) -> SanityResult:
    """
    Re-simulate with strict_pattern_proximity=True (pattern body within 0.5% of support).
    PASS if tight_win_rate stays within 10pp of loose_win_rate.
    """
    from .simulator import simulate_all

    loose_trades = simulate_all(entries, bars_map, profit_target_pct=pt, stop_loss_pct=sl)
    tight_trades = simulate_all(
        entries,
        bars_map,
        profit_target_pct=pt,
        stop_loss_pct=sl,
        strict_pattern_proximity=True,
        pattern_proximity_pct=tight_proximity_pct,
    )

    loose_wr = _win_rate(loose_trades)
    tight_wr = _win_rate(tight_trades)
    drop     = loose_wr - tight_wr
    passes   = drop < max_wr_drop and len(tight_trades) >= 3

    rows = [
        {"filter": "Loose (default proximity)", "trades": str(len(loose_trades)), "win_rate": f"{loose_wr:.1%}"},
        {"filter": f"Tight (±{tight_proximity_pct*100:.1f}% of support)", "trades": str(len(tight_trades)), "win_rate": f"{tight_wr:.1%}"},
        {"filter": "WR drop",                   "trades": "",               "win_rate": f"{drop:+.1%} ({'OK' if passes else 'LARGE DROP'})"},
    ]

    return {
        "name": "D-Pattern Proximity",
        "status": _status(passes, warn=(drop >= max_wr_drop and len(tight_trades) < 3)),
        "headline": (
            f"Loose WR {loose_wr:.1%} → Tight WR {tight_wr:.1%} (drop {drop:.1%}) — "
            f"{'robust to proximity filter' if passes else 'large WR drop under tight filter'}"
        ),
        "details": {
            "loose_trades":       len(loose_trades),
            "tight_trades":       len(tight_trades),
            "loose_wr":           loose_wr,
            "tight_wr":           tight_wr,
            "wr_drop":            drop,
            "proximity_pct":      tight_proximity_pct,
            "max_allowed_drop":   max_wr_drop,
        },
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Check E — Realistic PT/SL Grid Sweep
# ---------------------------------------------------------------------------

def check_e_realistic_sweep(
    entries,
    bars_map: dict,
    pt_grid: Optional[List[float]] = None,
    sl_grid: Optional[List[float]] = None,
    min_trades: int = 5,
) -> SanityResult:
    """
    Sweep a realistic PT/SL grid (small, intraday-appropriate ranges) and
    report the full expectancy table. Flags whether ANY viable combo exists.
    PASS if at least one combo has positive expectancy with >= min_trades.
    """
    from .simulator import simulate_all

    if pt_grid is None:
        pt_grid = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    if sl_grid is None:
        sl_grid = [0.5, 1.0, 1.5, 2.0]

    rows: list[dict] = []
    best_expectancy = -float("inf")
    viable_combos = 0

    for pt in pt_grid:
        for sl in sl_grid:
            trades = simulate_all(entries, bars_map, profit_target_pct=pt, stop_loss_pct=sl)
            if len(trades) < min_trades:
                continue
            wr  = _win_rate(trades)
            exp = _expectancy(trades)
            pf  = _profit_factor(trades)
            viable_combos += 1
            if exp > best_expectancy:
                best_expectancy = exp
            rows.append({
                "PT%": f"{pt:.1f}%",
                "SL%": f"{sl:.1f}%",
                "trades": str(len(trades)),
                "win_rate": f"{wr:.1%}",
                "expectancy": f"{exp:+.3f}",
                "profit_factor": f"{pf:.2f}",
            })

    # Sort by expectancy descending
    rows.sort(key=lambda r: float(r["expectancy"]), reverse=True)
    passes = viable_combos > 0 and best_expectancy > 0

    headline = (
        f"{viable_combos} viable combos in realistic PT/SL grid — "
        f"best expectancy {best_expectancy:+.3f} — "
        f"{'positive edge exists' if passes else 'no positive expectancy at realistic params'}"
    ) if viable_combos > 0 else "No viable combos (< min_trades) in realistic grid."

    return {
        "name": "E-Realistic Sweep",
        "status": _status(passes, warn=(viable_combos == 0)),
        "headline": headline,
        "details": {
            "pt_grid":         pt_grid,
            "sl_grid":         sl_grid,
            "viable_combos":   viable_combos,
            "best_expectancy": best_expectancy,
            "min_trades":      min_trades,
        },
        "rows": rows[:20],  # cap at top 20 rows
    }


# ---------------------------------------------------------------------------
# Check F — Per-Ticker Breakdown
# ---------------------------------------------------------------------------

def check_f_per_ticker(trades) -> SanityResult:
    """
    Break down trades by ticker. Flags if one ticker drives >50% of wins.
    PASS if no single ticker accounts for >70% of wins.
    """
    if not trades:
        return {
            "name": "F-Per-Ticker",
            "status": "WARN",
            "headline": "No trades to analyse.",
            "details": {},
            "rows": [],
        }

    ticker_wins:   dict[str, int] = defaultdict(int)
    ticker_losses: dict[str, int] = defaultdict(int)
    ticker_pnl:    dict[str, list] = defaultdict(list)
    ticker_dates:  dict[str, list] = defaultdict(list)

    for t in trades:
        if t.outcome == "win":
            ticker_wins[t.ticker] += 1
        else:
            ticker_losses[t.ticker] += 1
        ticker_pnl[t.ticker].append(t.pnl_pct)
        if t.entry_ts:
            ticker_dates[t.ticker].append(t.entry_ts)

    total_wins = sum(ticker_wins.values()) or 1
    rows: list[dict] = []
    max_win_pct = 0.0

    for ticker in sorted(ticker_wins.keys() | ticker_losses.keys()):
        wins   = ticker_wins[ticker]
        losses = ticker_losses[ticker]
        total  = wins + losses
        wr     = wins / total if total else 0.0
        avg_pnl = statistics.mean(ticker_pnl[ticker]) if ticker_pnl[ticker] else 0.0
        win_pct = wins / total_wins
        if win_pct > max_win_pct:
            max_win_pct = win_pct
        dates = ticker_dates[ticker]
        date_range = (
            f"{min(dates).strftime('%Y-%m-%d')} → {max(dates).strftime('%Y-%m-%d')}"
            if dates else "—"
        )
        rows.append({
            "ticker":      ticker,
            "trades":      str(total),
            "wins":        str(wins),
            "win_rate":    f"{wr:.1%}",
            "avg_pnl":     f"{avg_pnl:+.2f}%",
            "% of wins":   f"{win_pct:.1%}",
            "date_range":  date_range,
        })

    # Sort by number of trades desc
    rows.sort(key=lambda r: int(r["trades"]), reverse=True)
    passes = max_win_pct <= 0.70

    dominant = max(ticker_wins, key=ticker_wins.get) if ticker_wins else "—"
    return {
        "name": "F-Per-Ticker",
        "status": _status(passes, warn=(max_win_pct > 0.50)),
        "headline": (
            f"Dominant ticker: {dominant} ({max_win_pct:.1%} of wins) — "
            f"{'diversified results' if passes else 'results heavily concentrated in one ticker'}"
        ),
        "details": {
            "ticker_wins":   dict(ticker_wins),
            "ticker_losses": dict(ticker_losses),
            "dominant":      dominant,
            "dominant_win_pct": max_win_pct,
        },
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Check G — Slippage Sensitivity
# ---------------------------------------------------------------------------

def check_g_slippage_sensitivity(
    trades,
    slippage_levels: Optional[List[float]] = None,
    fail_pf_threshold: float = 1.0,
    warn_slippage_pct: float = 0.2,
) -> SanityResult:
    """
    Re-score existing trades under increasing slippage levels.
    Entry price worsened by +slippage%; exit on wins reduced by -slippage%.
    PASS  if PF ≥ threshold at 0.2% slippage.
    WARN  if PF drops below threshold between 0.1%–0.2%.
    FAIL  if PF < threshold at just 0.1% slippage.
    """
    if not trades:
        return {"name": "G-Slippage Sensitivity", "status": "WARN",
                "headline": "No trades to analyse.", "details": {}, "rows": []}

    if slippage_levels is None:
        slippage_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

    rows: list[dict] = []
    first_fail_at: Optional[float] = None

    for slip in slippage_levels:
        slip_frac = slip / 100.0
        adj_pnls = []
        for t in trades:
            entry_adj = t.entry_price * (1 + slip_frac)
            if t.exit_price is None:
                adj_pnls.append(0.0)
                continue
            if t.outcome == "win":
                exit_adj = t.exit_price * (1 - slip_frac)
            elif t.outcome == "loss":
                exit_adj = t.exit_price * (1 + slip_frac)
            else:
                exit_adj = t.exit_price
            adj_pnls.append((exit_adj - entry_adj) / entry_adj * 100)

        wins_pnl   = sum(p for p in adj_pnls if p > 0)
        losses_pnl = abs(sum(p for p in adj_pnls if p < 0))
        pf  = wins_pnl / losses_pnl if losses_pnl > 0 else float("inf")
        wr  = sum(1 for p in adj_pnls if p > 0) / len(adj_pnls)
        exp = sum(adj_pnls) / len(adj_pnls)
        viable = pf >= fail_pf_threshold
        if not viable and first_fail_at is None:
            first_fail_at = slip
        rows.append({
            "slippage": f"{slip:.2f}%",
            "win_rate": f"{wr:.1%}",
            "expectancy": f"{exp:+.3f}",
            "profit_factor": f"{pf:.3f}",
            "viable": "✅" if viable else "❌",
        })

    if first_fail_at is None:
        status  = "PASS"
        summary = f"PF ≥{fail_pf_threshold} up to {slippage_levels[-1]:.2f}% slippage — robust"
    elif first_fail_at <= 0.10:
        status  = "FAIL"
        summary = f"PF drops below {fail_pf_threshold} at {first_fail_at:.2f}% slippage — too fragile"
    elif first_fail_at <= warn_slippage_pct:
        status  = "WARN"
        summary = f"PF drops below {fail_pf_threshold} at {first_fail_at:.2f}% slippage — marginal"
    else:
        status  = "PASS"
        summary = f"PF holds through {warn_slippage_pct:.2f}% slippage — acceptable"

    return {
        "name": "G-Slippage Sensitivity",
        "status": status, "headline": summary,
        "details": {"slippage_levels": slippage_levels,
                    "first_fail_at": first_fail_at,
                    "fail_pf_threshold": fail_pf_threshold},
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Run All
# ---------------------------------------------------------------------------

def run_all_checks(
    entries,
    bars_map: dict,
    trades,
    pt: float = 2.0,
    sl: float = 1.0,
    checks: Optional[List[str]] = None,
) -> List[SanityResult]:
    """
    Run all (or a subset of) sanity checks and return results.
    checks : Optional subset, e.g. ["A", "B", "G"]. None = run all (A–G).
    """
    _ALL = {"A", "B", "C", "D", "E", "F", "G"}
    enabled = _ALL if checks is None else {c.upper() for c in checks} & _ALL

    results: list[SanityResult] = []
    if "A" in enabled:
        results.append(check_a_shuffle_control(entries, bars_map, pt, sl))
    if "B" in enabled:
        results.append(check_b_out_of_sample(entries, bars_map, pt, sl))
    if "C" in enabled:
        results.append(check_c_duplicate_trades(trades))
    if "D" in enabled:
        results.append(check_d_pattern_near_support(entries, bars_map, pt, sl))
    if "E" in enabled:
        results.append(check_e_realistic_sweep(entries, bars_map))
    if "F" in enabled:
        results.append(check_f_per_ticker(trades))
    if "G" in enabled:
        results.append(check_g_slippage_sensitivity(trades))
    return results
