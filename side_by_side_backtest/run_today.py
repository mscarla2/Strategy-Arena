#!/usr/bin/env python3
"""
run_today.py — "How would I have done today with the card strategy?"

Loads today's watchlist posts (all sessions), fetches 30-day 5-min bars for every
ticker, then runs simulate_card_strategy() across the full bar history — scoring
every 5-min session open, entering when score >= threshold, and exiting via the
card's own TP/SL levels.  Budget mirrors autonomous config ($5k / $500 / max 10).

Usage
-----
    python side_by_side_backtest/run_today.py           # today, score >= 4.3
    python side_by_side_backtest/run_today.py --min 3.5
    python side_by_side_backtest/run_today.py --no-fetch          # cached bars only
    python side_by_side_backtest/run_today.py --date 2026-04-20   # specific date
    python side_by_side_backtest/run_today.py --session pre_market
    python side_by_side_backtest/run_today.py --budget 10000 --size 1000
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import zoneinfo

# ── path bootstrap ─────────────────────────────────────────────────────────────
_PKG  = Path(__file__).parent
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from side_by_side_backtest.card_strategy_simulator import simulate_card_strategy
from side_by_side_backtest.data_fetcher import (
    fetch_bars_for_entry,
    is_banned,
    load_30day_bars,
    refresh_today,
)
from side_by_side_backtest.models import RawWatchlist, TradeResult
from side_by_side_backtest.parser import parse_watchlist_post

_WL_PATH = _ROOT / "scraped_watchlists.json"

# ANSI colour helpers
_G   = "\033[92m"
_Y   = "\033[93m"
_R   = "\033[91m"
_DIM = "\033[2m"
_RST = "\033[0m"
_BOLD = "\033[1m"


def _c_pnl(v: float) -> str:
    """Colour a dollar PnL value green/red."""
    if v >= 0:
        return f"{_G}+${v:.2f}{_RST}"
    return f"{_R}-${abs(v):.2f}{_RST}"


def _c_pct(v: float) -> str:
    if v >= 0:
        return f"{_G}+{v:.2f}%{_RST}"
    return f"{_R}{v:.2f}%{_RST}"


def _load_posts() -> list[dict]:
    if not _WL_PATH.exists():
        print(f"{_R}ERROR:{_RST} {_WL_PATH} not found", file=sys.stderr)
        sys.exit(1)
    return json.loads(_WL_PATH.read_text())


# US Eastern timezone — posts are published in ET market hours.
# Falling back to UTC-5 offset if zoneinfo is unavailable.
try:
    _ET = zoneinfo.ZoneInfo("America/New_York")
except Exception:
    _ET = timezone(timedelta(hours=-5))


def _pick_posts(
    posts: list[dict],
    session_filter: str,
    target_date: date | None,
) -> list[dict]:
    """
    Return the post(s) whose title matches *session_filter* and whose date
    matches *target_date* (defaults to today in US Eastern time, so running
    just after UTC midnight still finds today's posts).
    """
    if target_date is None:
        target_date = datetime.now(tz=_ET).date()

    kw_map = {
        "market_open":  ["MARKET OPEN"],
        "pre_market":   ["PRE MARKET", "PREMARKET"],
        "after_hours":  ["AFTER HOUR", "AFTER-HOUR", "AH WATCH"],
        "unknown":      [],   # fall through — take any post from today
    }
    keywords = kw_map.get(session_filter, [])

    matched = []
    for p in posts:
        ts_raw = p.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            post_date = ts.astimezone(_ET).date()
        except Exception:
            continue

        if post_date != target_date:
            continue

        title = p.get("title", "").upper()
        if not keywords or any(kw in title for kw in keywords):
            matched.append(p)

    return matched


def _load_bars(entry, fetch: bool):
    """Return 30-day bars for *entry*, optionally refreshing from yfinance."""
    import pandas as pd
    ticker = entry.ticker
    bars = load_30day_bars(ticker)
    if bars.empty or fetch:
        try:
            if fetch:
                bars = refresh_today(ticker)
            if bars.empty:
                bars = fetch_bars_for_entry(entry) or pd.DataFrame()
        except Exception as exc:
            print(f"  {_DIM}[warn] {ticker}: fetch — {exc}{_RST}", file=sys.stderr)
    return bars


def _print_trades(trades: list, trade_size: float, budget: float) -> None:
    """Print per-trade table + running equity + summary."""
    W = 110
    print(f"\n{_BOLD}{'─'*W}{_RST}")
    print(
        f"{_BOLD}{'TICKER':<7} {'ENTRY TIME':<18} {'ENTRY':>8} {'TP':>8} {'SL':>8} "
        f"{'EXIT':>8} {'PNL%':>7} {'PNL$':>8} {'EQUITY':>9} {'OUTCOME':<9}{_RST}"
    )
    print("─" * W)

    equity = budget
    for t in trades:
        shares     = trade_size / t.entry_price if t.entry_price else 0
        dollar_pnl = shares * (t.exit_price - t.entry_price)
        equity    += dollar_pnl
        ts_str     = str(t.entry_ts)[:16] if t.entry_ts else "—"
        # Reconstruct TP/SL dollar prices from stored pct fields
        tp_dollar  = t.entry_price * (1 + t.profit_target_pct / 100) if t.entry_price else 0
        sl_dollar  = t.entry_price * (1 - t.stop_loss_pct   / 100) if t.entry_price else 0
        outcome_c  = (f"{_G}{t.outcome}{_RST}" if t.outcome == "win"
                      else f"{_R}{t.outcome}{_RST}" if t.outcome == "loss"
                      else f"{_DIM}{t.outcome}{_RST}")
        print(
            f"{t.ticker:<7} {ts_str:<18} "
            f"${t.entry_price:>7.4f} ${tp_dollar:>7.4f} ${sl_dollar:>7.4f} "
            f"${t.exit_price:>7.4f} "
            f"{_c_pct(t.pnl_pct):>7}  {_c_pnl(dollar_pnl):>8}  "
            f"${equity:>8.2f}  {outcome_c}"
        )
    print("─" * W)


def _run_all(args, posts: list[dict], do_fetch: bool, max_conc: int) -> None:
    """Run simulation for every unique post date in the JSON.

    Optimised: collects ALL unique tickers across ALL posts, loads bars once,
    runs simulate_card_strategy ONCE (it already scores all 30 days internally),
    then groups the returned trades by calendar date for the per-day summary.
    """
    # ── Collect all unique tickers across all posts ───────────────────────────
    all_entries: list = []
    all_dates: list[date] = []
    seen_dates: set[date] = set()

    for p in posts:
        ts_raw = p.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            d  = ts.astimezone(_ET).date()
            if d not in seen_dates:
                seen_dates.add(d)
                all_dates.append(d)
        except Exception:
            continue
        for e in parse_watchlist_post(RawWatchlist(**p)):
            all_entries.append(e)

    all_dates.sort()
    if not all_dates:
        print(f"{_Y}No posts found in {_WL_PATH}.{_RST}")
        return

    # Dedupe entries by ticker
    seen_t: set[str] = set()
    unique: list = []
    for e in all_entries:
        if e.ticker != "SPY" and e.ticker not in seen_t and not is_banned(e.ticker):
            seen_t.add(e.ticker)
            unique.append(e)

    mode_label = "CARD" if args.mode == "card" else "SBS"
    print(f"\n{_BOLD}[{mode_label}] Running all {len(all_dates)} sessions — "
          f"{len(unique)} unique tickers  ${args.size}/trade{_RST}")
    print(f"{_DIM}Loading bars …{_RST}")

    # ── Load bars once for all tickers ───────────────────────────────────────
    bars_map: dict = {}
    n = min(args.workers, len(unique))
    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = {pool.submit(_load_bars, e, do_fetch): e for e in unique}
        for fut in as_completed(futs):
            e = futs[fut]
            try:
                b = fut.result()
                if b is not None and not b.empty:
                    bars_map[e.ticker] = b
            except Exception:
                pass
    
    print(f"{_DIM}{len(bars_map)}/{len(unique)} tickers have bars — simulating …{_RST}\n")

    # ── Simulate ──────────────────────────────────────────────────────────────
    # Card: ONE call across all 30 days (score_setup is date-aware internally).
    # SBS:  Per-day calls — SBS uses support levels from each day's specific post,
    #       so mixing tickers across posts would give wrong support levels.
    if args.mode == "card":
        all_trades = simulate_card_strategy(
            unique, bars_map,
            min_score=args.min,
            budget_total=args.budget,
            trade_size=args.size,
            max_concurrent=max_conc,
            daily_loss_halt=args.halt,
            max_workers=args.workers,
            verbose=True,
            rescore_stride=1,
            disable_sr_cache=False,
        )
    else:
        # SBS per-day: entries and support levels are date-specific
        from side_by_side_backtest.simulator import simulate_all as _sim_all
        all_trades = []
        for sim_date in all_dates:
            day_matched = _pick_posts(posts, "unknown", sim_date)
            if not day_matched:
                continue
            day_entries: list = []
            for p in day_matched:
                day_entries.extend(parse_watchlist_post(RawWatchlist(**p)))
            day_seen: set[str] = set()
            day_unique: list = []
            for e in day_entries:
                if e.ticker != "SPY" and e.ticker not in day_seen and not is_banned(e.ticker):
                    day_seen.add(e.ticker); day_unique.append(e)
            day_bars = {e.ticker: bars_map[e.ticker]
                        for e in day_unique if e.ticker in bars_map}
            if not day_bars:
                continue
            day_trades = _sim_all(
                day_unique, day_bars,
                profit_target_pct=args.tp,
                stop_loss_pct=args.sl,
                max_entry_attempts=10,
                require_support_ok=True,
                verbose=False,
            )
            all_trades.extend(day_trades)

    # ── Group trades by entry date and print per-day summary ──────────────────
    def _trade_date(t) -> date:
        ts = t.entry_ts
        return ts.date() if hasattr(ts, "date") else ts.to_pydatetime().date()

    from collections import defaultdict
    by_date: dict[date, list] = defaultdict(list)
    for t in all_trades:
        by_date[_trade_date(t)].append(t)

    W = 90
    print(f"{'DATE':<12} {'TRADES':>6} {'W':>4} {'L':>4} {'WR':>6} "
          f"{'DAY P&L':>10}  {'EQUITY':>10}")
    print("─" * W)

    cumulative_equity = args.budget
    grand_trades = grand_wins = grand_losses = 0
    grand_pnl = 0.0

    for sim_date in all_dates:
        day_trades = by_date.get(sim_date, [])
        if not day_trades:
            print(f"{str(sim_date):<12} {'—':>6}")
            continue

        wins    = sum(1 for t in day_trades if t.outcome == "win")
        losses  = sum(1 for t in day_trades if t.outcome == "loss")
        wr      = wins / len(day_trades)
        day_pnl = sum((args.size / t.entry_price) * (t.exit_price - t.entry_price)
                      for t in day_trades if t.entry_price)
        cumulative_equity += day_pnl
        grand_trades += len(day_trades)
        grand_wins   += wins
        grand_losses += losses
        grand_pnl    += day_pnl

        wr_c = (f"{_G}{wr:.0%}{_RST}" if wr >= 0.6
                else f"{_Y}{wr:.0%}{_RST}" if wr >= 0.4
                else f"{_R}{wr:.0%}{_RST}")
        print(f"{str(sim_date):<12} {len(day_trades):>6} {wins:>4} {losses:>4} "
              f"{wr_c:>6}  {_c_pnl(day_pnl):>10}  ${cumulative_equity:>9.2f}")

    print("─" * W)
    overall_wr = grand_wins / grand_trades if grand_trades else 0
    print(
        f"\n  {_BOLD}Total trades:{_RST} {grand_trades}  "
        f"{_BOLD}Wins:{_RST} {grand_wins}  {_BOLD}Losses:{_RST} {grand_losses}  "
        f"{_BOLD}WR:{_RST} {overall_wr:.1%}  "
        f"{_BOLD}Net P&L:{_RST} {_c_pnl(grand_pnl)}  "
        f"{_BOLD}Final equity:{_RST} ${args.budget + grand_pnl:.2f}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Watchlist backtest — card strategy or side-by-side simulator"
    )
    ap.add_argument("--mode",     default="card", choices=["card", "sbs"],
                    help="card = score-based entry; sbs = support-touch + pattern (default: card)")
    ap.add_argument("--min",      type=float, default=4.3,
                    help="[card] Min score to enter (default 4.3)")
    ap.add_argument("--tp",       type=float, default=5.0,
                    help="[sbs] Take-profit %% (default 5.0)")
    ap.add_argument("--sl",       type=float, default=1.0,
                    help="[sbs] Stop-loss %% (default 1.0)")
    ap.add_argument("--session",  default="market_open",
                    choices=["market_open", "pre_market", "after_hours", "unknown"],
                    help="Which session post to load (default: market_open)")
    ap.add_argument("--date",     default=None,
                    help="YYYY-MM-DD of the post (default: today ET)")
    ap.add_argument("--no-fetch", action="store_true",
                    help="Use cached bars only — no live yfinance refresh")
    ap.add_argument("--workers",  type=int, default=12,
                    help="Parallel bar-fetch workers (default 12)")
    ap.add_argument("--budget",   type=float, default=5_000.0,
                    help="Total ring-fenced capital $ (default 5000)")
    ap.add_argument("--size",     type=float, default=500.0,
                    help="$ per trade (default 500)")
    ap.add_argument("--halt",     type=float, default=300.0,
                    help="Daily loss halt $ (default 300)")
    ap.add_argument("--all",      action="store_true",
                    help="Run simulation for every post date in the JSON")
    args = ap.parse_args()

    target_date: date | None = None
    if args.date:
        try:
            target_date = date.fromisoformat(args.date)
        except ValueError:
            print(f"{_R}ERROR:{_RST} --date must be YYYY-MM-DD", file=sys.stderr)
            sys.exit(1)

    do_fetch = not args.no_fetch
    max_conc = max(1, int(args.budget // args.size))
    posts    = _load_posts()

    # ── --all mode: iterate every post date ──────────────────────────────────
    if args.all:
        _run_all(args, posts, do_fetch, max_conc)
        return

    # ── 1. Find today's post ──────────────────────────────────────────────────
    matched = _pick_posts(posts, "unknown" if args.mode == "sbs" else args.session, target_date)

    if not matched:
        date_str = str(target_date or datetime.now(tz=_ET).date())
        print(f"{_Y}No '{args.session}' post found for {date_str}.{_RST}\nAvailable:")
        today = target_date or datetime.now(tz=_ET).date()
        for p in posts:
            try:
                ts = datetime.fromisoformat(p.get("timestamp","").replace("Z","+00:00"))
                if ts.astimezone(_ET).date() == today:
                    print(f"  {ts.astimezone(_ET).strftime('%H:%M ET')}  {p.get('title','')[:70]}")
            except Exception:
                pass
        sys.exit(0)

    print(f"\n{_BOLD}Post:{_RST}")
    for p in matched:
        print(f"  {p.get('timestamp','')}  {p.get('title','')[:70]}")

    # ── 2. Parse + dedupe entries ─────────────────────────────────────────────
    entries: list = []
    for p in matched:
        entries.extend(parse_watchlist_post(RawWatchlist(**p)))
    seen: set[str] = set()
    unique: list = []
    for e in entries:
        if e.ticker != "SPY" and e.ticker not in seen and not is_banned(e.ticker):
                    seen.add(e.ticker); unique.append(e)

    print(f"\n{_BOLD}{len(unique)} tickers{_RST} "
          f"({'live fetch' if do_fetch else 'cached only'}) — loading bars …\n")

    # ── 3. Fetch bars in parallel ─────────────────────────────────────────────
    bars_map: dict = {}
    n = min(args.workers, len(unique)) if unique else 1
    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = {pool.submit(_load_bars, e, do_fetch): e for e in unique}
        done = 0
        for fut in as_completed(futs):
            done += 1
            e = futs[fut]
            try:
                b = fut.result()
                if b is not None and not b.empty:
                    bars_map[e.ticker] = b
                    print(f"  [{done:>3}/{len(unique)}] {e.ticker:<6} {len(b)} bars",
                          flush=True)
                else:
                    print(f"  [{done:>3}/{len(unique)}] {e.ticker:<6} {_DIM}no bars{_RST}",
                          flush=True)
            except Exception as exc:
                print(f"  [{done:>3}/{len(unique)}] {e.ticker:<6} ERR {exc}",
                      file=sys.stderr)

    sim_date = target_date or datetime.now(tz=_ET).date()

    def _trade_date(t) -> date:
        ts = t.entry_ts
        return ts.date() if hasattr(ts, "date") else ts.to_pydatetime().date()

    def _summarise(trades: list, label: str) -> None:
        trades.sort(key=lambda t: t.entry_ts)
        _print_trades(trades, args.size, args.budget)
        wins      = sum(1 for t in trades if t.outcome == "win")
        losses    = sum(1 for t in trades if t.outcome == "loss")
        wr        = wins / len(trades) if trades else 0
        total_pnl = sum((args.size / t.entry_price) * (t.exit_price - t.entry_price)
                        for t in trades if t.entry_price)
        print(
            f"\n  {_BOLD}[{label}] Trades:{_RST} {len(trades)}  "
            f"{_BOLD}Wins:{_RST} {wins}  {_BOLD}Losses:{_RST} {losses}  "
            f"{_BOLD}WR:{_RST} {wr:.1%}  "
            f"{_BOLD}Net P&L:{_RST} {_c_pnl(total_pnl)}  "
            f"{_BOLD}Final equity:{_RST} ${args.budget + total_pnl:.2f}\n"
        )

    # ── 4a. Card strategy ─────────────────────────────────────────────────────
    if args.mode == "card":
        print(f"\n{_BOLD}[CARD] Simulation for {sim_date} …{_RST}  "
              f"(score≥{args.min}, ${args.size}/trade, "
              f"max {max_conc} concurrent, ${args.halt} halt)\n")

        all_trades = simulate_card_strategy(
            unique, bars_map,
            min_score=args.min,
            budget_total=args.budget,
            trade_size=args.size,
            max_concurrent=max_conc,
            daily_loss_halt=args.halt,
            max_workers=args.workers,
            verbose=True,
            rescore_stride=4,
            disable_sr_cache=False,
        )
        trades = [t for t in all_trades if _trade_date(t) == sim_date]
        if not trades:
            msg = (f"No trades fired at score ≥ {args.min}" if not all_trades
                   else f"No trades on {sim_date} (but {len(all_trades)} across 30d)")
            print(f"{_Y}{msg}.{_RST}")
            sys.exit(0)
        _summarise(trades, "CARD")

    # ── 4b. Side-by-Side simulator ────────────────────────────────────────────
    else:
        from side_by_side_backtest.simulator import simulate_all
        print(f"\n{_BOLD}[SBS] Simulation for {sim_date} …{_RST}  "
              f"(TP={args.tp}%, SL={args.sl}%, ${args.size}/trade)\n")

        all_trades = simulate_all(
            unique, bars_map,
            profit_target_pct=args.tp,
            stop_loss_pct=args.sl,
            max_entry_attempts=10,
            require_support_ok=True,
            verbose=False,
        )
        trades = [t for t in all_trades if _trade_date(t) == sim_date]
        if not trades:
            msg = (f"No SBS trades fired" if not all_trades
                   else f"No SBS trades on {sim_date} (but {len(all_trades)} across 30d)")
            print(f"{_Y}{msg}.{_RST}")
            sys.exit(0)
        _summarise(trades, "SBS")


if __name__ == "__main__":
    main()