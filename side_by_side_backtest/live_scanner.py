"""
Live Scanner — Real-Time Entry Alerts
======================================
Polls every POLL_INTERVAL seconds for each ticker in today's watchlist entries.
When price touches the support level AND the Side-by-Side pattern fires,
an alert is emitted via:
  • Terminal  — coloured banner (ANSI)
  • Desktop   — macOS osascript notification
  • Sound     — system bell (\\a)

Usage
-----
    # Scan watchlist from default JSON, poll every 5 minutes
    python -m side_by_side_backtest.live_scanner

    # Custom path and interval
    python -m side_by_side_backtest.live_scanner \\
        --watchlist scraped_watchlists.json --interval 300
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

_PKG = Path(__file__).parent
_DEFAULT_JSON  = _PKG.parent / "scraped_watchlists.json"
_POLL_INTERVAL = 60           # seconds between full scans (entry attempts + position checks)
_TOUCH_BAND    = 0.005        # body within 0.5% of support = "touched" (body-based, not wick)
_PATTERN_BARS  = 30           # last N bars to check for the pattern
_HEARTBEAT_FILE = _PKG / ".scanner_heartbeat.json"   # written after every scan pass

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_RED    = "\033[91m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


# Global flag — set to False by the Streamlit page when running in-process
# to suppress macOS notifications/sounds that conflict with the browser session.
_NOTIFICATIONS_ENABLED: bool = True


def _banner(sc) -> None:
    """Print a coloured ANSI alert banner to stdout."""
    now = datetime.now(tz=timezone.utc).strftime("%H:%M ET")
    next_bar = datetime.now(tz=timezone.utc).strftime("%H:%M ET")  # approximate
    sep = "═" * 42
    print(f"\n{_RED}{_BOLD}╔{sep}╗{_RESET}")
    print(f"{_RED}{_BOLD}║  🚨 ENTRY SIGNAL FIRED — {now:<16}║{_RESET}")
    print(f"{_RED}{_BOLD}║  {sc.ticker:<6}|  Support: ${sc.support:<10.2f}        ║{_RESET}")
    print(f"{_RED}{_BOLD}║  Pattern: Side-by-Side White Lines       ║{_RESET}")
    print(f"{_RED}{_BOLD}║  Score: {sc.score:.1f}/10  |  R/R: {sc.rr_ratio:.1f}:1{' '*14}║{_RESET}")
    print(f"{_RED}{_BOLD}║  → Buy at next bar open (~{next_bar})  ║{_RESET}")
    print(f"{_RED}{_BOLD}╚{sep}╝{_RESET}\n")
    sys.stdout.flush()


def _desktop_notify(sc) -> None:
    """Fire a macOS notification via osascript. No-op if notifications disabled."""
    if not _NOTIFICATIONS_ENABLED:
        return
    if sys.platform != "darwin":
        return
    msg  = f"Score {sc.score}/10 | R/R {sc.rr_ratio:.1f}:1 | Support ${sc.support:.2f}"
    title = f"🚨 ENTRY SIGNAL — {sc.ticker}"
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{msg}" with title "{title}" sound name "Glass"'],
            check=False, timeout=5,
        )
    except Exception:
        pass


def _sound() -> None:
    """Play effect.mp3 via afplay (macOS), fallback to system bell. No-op if disabled."""
    if not _NOTIFICATIONS_ENABLED:
        return
    _mp3 = Path(__file__).parent / "effect.mp3"
    if sys.platform == "darwin" and _mp3.exists():
        try:
            subprocess.run(["afplay", str(_mp3)], check=False, timeout=5)
            return
        except Exception:
            pass
    print("\a", end="", flush=True)


# ---------------------------------------------------------------------------
# Watchlist loading
# ---------------------------------------------------------------------------

def _load_entries(json_path: str, today_only: bool = True) -> list:
    """
    Parse watchlist entries from the JSON file.

    today_only=True (default): only parse posts from today's date, so the
    scanner focuses on the current watchlist rather than all historical posts.
    Set to False to scan the full history (original behaviour).
    """
    from .models import RawWatchlist
    from .parser import parse_watchlist_post
    from datetime import date

    try:
        raw = json.loads(Path(json_path).read_text())
    except Exception as exc:
        print(f"[scanner] Cannot load {json_path}: {exc}")
        return []

    if today_only:
        today_str = date.today().isoformat()  # "2026-04-20"
        raw = [p for p in raw if (p.get("timestamp") or "").startswith(today_str)]
        if not raw:
            # Fallback: use the most recent day's posts if today has none
            if raw_all := json.loads(Path(json_path).read_text()):
                latest_date = max(
                    (p.get("timestamp") or "")[:10]
                    for p in raw_all
                    if p.get("timestamp")
                )
                raw = [p for p in raw_all if (p.get("timestamp") or "").startswith(latest_date)]
                print(f"[scanner] No posts for today — using latest date: {latest_date} ({len(raw)} posts)")

    entries = []
    for post in raw:
        try:
            entries.extend(parse_watchlist_post(RawWatchlist(**post)))
        except Exception:
            pass

    if today_only:
        # Deduplicate by ticker — keep entry with highest support (most recent)
        seen: dict = {}
        for e in reversed(entries):
            seen.setdefault(e.ticker, e)
        entries = list(seen.values())
        print(f"[scanner] Today's entries: {len(entries)} tickers from {len(raw)} post(s)")

    return entries


# ---------------------------------------------------------------------------
# Per-ticker check
# ---------------------------------------------------------------------------

def _check_ticker(entry, quote: dict = None) -> Optional[object]:
    """
    Score the ticker using fresh 30-day bars.
    Returns a SetupScore always (for card strategy score-based entry),
    or None if bars unavailable.

    Bar freshness is guaranteed by the batch refresh_today() call in
    scan_once() that runs BEFORE the parallel scoring pool — bars are
    already up-to-date when _check_ticker is called.
    """
    from .data_fetcher import load_30day_bars
    from .setup_scorer import score_setup

    try:
        bars: pd.DataFrame = load_30day_bars(entry.ticker)
    except Exception:
        return None

    if bars.empty or len(bars) < 3:
        return None

    # Score the setup — score_setup handles S/R fallback internally
    try:
        sc = score_setup(entry, bars)
        # Attach the actual current market price so _execute_for_strategy
        # uses the real bid/ask level, not the watchlist support level.
        if sc is not None:
            if quote and "quote" in quote:
                q_price = float(quote["quote"].get("lastPrice", bars["close"].iloc[-1]))
                sc._current_market_price = q_price
                print(f"[scanner] {entry.ticker} using Schwab quote price: ${q_price:.3f}")
            else:
                sc._current_market_price = float(bars["close"].iloc[-1])
        return sc
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Autonomous execution hook
# ---------------------------------------------------------------------------

_COOLDOWN_MINUTES = 30   # don't re-enter same ticker+strategy within 30 min


def _is_on_cooldown(ticker: str, strategy_name: str, db_path) -> bool:
    """
    Return True if this ticker+strategy has an open position OR was entered
    within the last COOLDOWN_MINUTES. Checks the DB so it persists across
    scanner restarts.
    """
    try:
        from .db import WatchlistDB
        with WatchlistDB(db_path) as db:
            rows = db.load_actual_trades(strategy_name=strategy_name)
        # Filter to this ticker
        ticker_rows = [r for r in rows if r["ticker"] == ticker]
        if not ticker_rows:
            return False
        # Open position → always on cooldown
        if any(r["outcome"] == "open" for r in ticker_rows):
            return True
        # Recently closed → cooldown
        import pandas as pd
        last_entry = max(
            pd.Timestamp(r["entry_ts"]) for r in ticker_rows
        )
        if last_entry.tzinfo is None:
            last_entry = last_entry.tz_localize("UTC")
        elapsed = (datetime.now(tz=timezone.utc) - last_entry).total_seconds() / 60
        return elapsed < _COOLDOWN_MINUTES
    except Exception:
        return False


# File-based lock directory — one lock file per ticker+strategy combo so that
# parallel processes (background subprocess + in-process Streamlit scan) cannot
# both pass the cooldown check and double-enter the same position.
_LOCK_DIR = _PKG / ".entry_locks"


def _acquire_entry_lock(ticker: str, strategy_name: str) -> bool:
    """
    Try to create a lock file for ticker+strategy.
    Returns True if this process won the lock, False if another process already
    holds it (file was created within the last 60 seconds).
    Uses O_CREAT|O_EXCL for atomic creation — safe across processes on macOS/Linux.
    """
    import os as _os
    _LOCK_DIR.mkdir(exist_ok=True)
    lock_file = _LOCK_DIR / f"{ticker.upper()}_{strategy_name}.lock"
    # Clean up stale locks older than 60 s
    try:
        if lock_file.exists() and (time.time() - lock_file.stat().st_mtime) > 60:
            lock_file.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        fd = _os.open(str(lock_file), _os.O_CREAT | _os.O_EXCL | _os.O_WRONLY)
        _os.close(fd)
        return True
    except FileExistsError:
        return False


def _release_entry_lock(ticker: str, strategy_name: str) -> None:
    lock_file = _LOCK_DIR / f"{ticker.upper()}_{strategy_name}.lock"
    try:
        lock_file.unlink(missing_ok=True)
    except Exception:
        pass


def _execute_for_strategy(sc, monitor, engine, strategy_cfg, master_cfg) -> None:
    """
    Evaluate and execute one signal for a single strategy configuration.

    Card Strategy  : entry signal = score ≥ min_score ONLY (no pattern required).
                     The score already incorporates pattern, ADX, confluence etc.
    Backtest Strat : entry signal = pattern + support touch (score irrelevant).
                     Uses sc (already scored) but ignores score gate.
    Both strategies: cooldown — skip re-entry within 30 min of last entry.
    """
    sname    = strategy_cfg.name
    mode_tag = "[PAPER]" if master_cfg.paper_mode else "[LIVE]"

    # ── Ban check — reload from disk each call so newly banned tickers take
    #    effect immediately without restarting the scanner process.
    from .data_fetcher import _load_banned, is_banned
    try:
        _load_banned()
    except Exception:
        pass
    if is_banned(sc.ticker):
        return  # silently skip banned tickers

    # ── Cross-process entry lock — prevents duplicate orders when the
    #    Streamlit in-process scan and background subprocess overlap.
    if not _acquire_entry_lock(sc.ticker, sname):
        return  # another process is already entering this ticker+strategy

    try:
        _execute_for_strategy_inner(sc, monitor, engine, strategy_cfg, master_cfg)
    finally:
        _release_entry_lock(sc.ticker, sname)


def _execute_for_strategy_inner(sc, monitor, engine, strategy_cfg, master_cfg) -> None:
    """Inner execution — called only after the entry lock is held."""
    sname    = strategy_cfg.name
    mode_tag = "[PAPER]" if master_cfg.paper_mode else "[LIVE]"

    # ── Cooldown check (both strategies) — DB-backed, persists across restarts
    if _is_on_cooldown(sc.ticker, sname, master_cfg.db_path):
        return  # silent — don't spam console every 5 min

    # ── Re-check max_concurrent with a fresh DB read AFTER acquiring the lock ──
    # This catches the race where multiple tickers pass the engine's concurrent
    # check before any of them have written their DB row (parallel scan pass).
    try:
        from .db import WatchlistDB
        source = "paper" if master_cfg.paper_mode else "live"
        with WatchlistDB(master_cfg.db_path) as _db:
            live_open = _db.open_position_count(source=source, strategy_name=sname)
        if live_open >= strategy_cfg.max_concurrent:
            return  # silently skip — max already reached
    except Exception:
        pass

    # ── Strategy-specific entry gate ──────────────────────────────────────────
    current_mkt_price = getattr(sc, "_current_market_price", None) or sc.entry_price

    if sname == "card_strategy":
        # Card: score ≥ min_score AND price within touch_band of support
        if sc.score < strategy_cfg.min_score:
            return
        if getattr(strategy_cfg, "require_support_touch", False):
            if sc.support and sc.support > 0:
                touch_dist = abs(current_mkt_price - sc.support) / sc.support
                if touch_dist > _TOUCH_BAND:
                    print(f"{_YELLOW}{mode_tag}[{sname}] {sc.ticker} — skip: "
                          f"price ${current_mkt_price:.3f} not within {_TOUCH_BAND*100:.1f}% "
                          f"of support ${sc.support:.3f}{_RESET}")
                    return
            else:
                return  # no valid support level — skip
    else:
        # Backtest: requires support touch + pattern
        if sc.pattern_score <= 0:
            return

    decision = engine.evaluate(sc, current_price=current_mkt_price)
    if not decision.go:
        print(f"{_YELLOW}{mode_tag}[{sname}] {sc.ticker} — skip: {decision.reason}{_RESET}")
        return

    # ── PT/SL resolution ──────────────────────────────────────────────────────
    # Card strategy: TP = SetupScore.resistance (computed S/R level), SL = flat 1%
    # Backtest strategy: uses optimization_results.csv → per-ticker median → defaults
    pt_pct = strategy_cfg.default_pt_pct
    sl_pct = strategy_cfg.default_sl_pct
    # Override PT with resistance level for card strategy
    resistance_tp_price: Optional[float] = None
    if getattr(strategy_cfg, "use_resistance_as_tp", False) and sc.resistance and sc.resistance > current_mkt_price:
        resistance_tp_price = sc.resistance
        # Convert resistance price → pt_pct relative to current market price
        pt_pct = (sc.resistance / current_mkt_price - 1) * 100

    if not getattr(strategy_cfg, "use_resistance_as_tp", False):
        # Non-card path: use optimization CSV and per-ticker median
        # Layer 2: global sweep results from optimization_results.csv
        try:
            import csv as _csv
            _opt_csv = Path(__file__).parent / "reports" / "optimization_results.csv"
            if _opt_csv.exists():
                with open(_opt_csv, newline="") as _f:
                    row = next(_csv.DictReader(_f), None)
                if row:
                    pt_pct = float(row["profit_target_pct"])
                    sl_pct = float(row["stop_loss_pct"])
        except Exception:
            pass

        # Layer 1: per-ticker median from backtested trades (overrides CSV if ≥3 rows)
        try:
            from .db import WatchlistDB
            with WatchlistDB(master_cfg.db_path) as db:
                all_trades = db.load_trades()
            ticker_trades = [t for t in all_trades if t.ticker == sc.ticker]
            if len(ticker_trades) >= 3:
                import statistics
                pt_pct = statistics.median(t.profit_target_pct for t in ticker_trades)
                sl_pct = statistics.median(t.stop_loss_pct for t in ticker_trades)
        except Exception:
            pass

    print(
        f"{_GREEN}{_BOLD}{mode_tag}[{sname}] ENTERING {sc.ticker}  "
        f"qty={decision.quantity} @ limit ${decision.limit_price:.3f}  "
        f"score={sc.score:.1f}  PT={pt_pct:.2f}%  SL={sl_pct:.2f}%  "
        f"{decision.reason}{_RESET}"
    )

    # In live mode: place the order then poll for a fill within FILL_TIMEOUT_SEC.
    # Only write to DB on confirmed fill. Cancel + skip if not filled in time.
    # This prevents pending/unfilled orders from creating ghost positions.
    #
    # IMPORTANT: entry_ts is captured AFTER the fill is confirmed, not at order
    # submission time.  The position monitor uses entry_ts to filter post-entry
    # bars: if entry_ts = submission time, bars from the 5-30s fill window are
    # included — these bars may already show the stock below SL, causing
    # immediate false SL triggers.
    _FILL_POLL_INTERVAL = 2   # seconds between status checks
    _FILL_TIMEOUT_SEC   = 30  # cancel after this many seconds without a fill

    actual_entry_price = decision.limit_price or sc.entry_price  # default; overridden by fill price
    entry_ts = datetime.now(tz=timezone.utc)  # fallback for paper mode; overwritten on live fill

    if not master_cfg.paper_mode:
        from .schwab_broker import SchwabBroker
        broker = SchwabBroker(master_cfg)
        order  = broker.place_order(
            ticker=sc.ticker,
            side="buy",
            quantity=decision.quantity,
            limit_price=decision.limit_price,
        )
        if order.status == "error":
            print(f"{_YELLOW}{mode_tag}[{sname}] {sc.ticker} — order REJECTED by Schwab "
                  f"(status={order.status}): {order.message}{_RESET}")
            return  # do NOT write to DB

        if order.status == "pending" and order.order_id:
            # Poll until filled or timeout
            deadline = time.time() + _FILL_TIMEOUT_SEC
            filled_order = None
            while time.time() < deadline:
                time.sleep(_FILL_POLL_INTERVAL)
                status = broker.get_order_status(order.order_id)
                if status.status == "filled":
                    filled_order = status
                    break
                if status.status in ("cancelled", "error"):
                    print(f"{_YELLOW}{mode_tag}[{sname}] {sc.ticker} — order {status.status} "
                          f"before fill, skipping DB write{_RESET}")
                    return

            if filled_order is None:
                # Timeout — cancel the unfilled order
                broker.cancel_order(order.order_id)
                print(f"{_YELLOW}{mode_tag}[{sname}] {sc.ticker} — limit order NOT filled "
                      f"within {_FILL_TIMEOUT_SEC}s, cancelled{_RESET}")
                return  # do NOT write to DB

            # Capture fill timestamp NOW — after confirmed fill.
            # This is the correct entry_ts for the position monitor's post_entry
            # bar filter; using order-submission time would include pre-fill bars.
            entry_ts = datetime.now(tz=timezone.utc)

            # Use actual fill price if available
            if filled_order.fill_price and filled_order.fill_price > 0:
                actual_entry_price = filled_order.fill_price

    monitor.open_paper_position(
        ticker=sc.ticker,
        entry_ts=entry_ts,
        entry_price=actual_entry_price,
        quantity=decision.quantity,
        setup_score=sc.score,
        pt_pct=pt_pct,
        sl_pct=sl_pct,
        session_type=getattr(sc, "session_type", "unknown"),
        strategy_name=sname,
    )

    # Cooldown is now DB-backed via _is_on_cooldown() — no in-process dict needed


def _autonomous_execute_all(sc, monitor, engines: list) -> None:
    """
    Run the signal through ALL strategies. Each strategy evaluates independently.
    engines: list of (DecisionEngine, StrategyConfig, MasterConfig) tuples
    """
    for engine, strategy_cfg, master_cfg in engines:
        try:
            _execute_for_strategy(sc, monitor, engine, strategy_cfg, master_cfg)
        except Exception as exc:
            print(f"[autonomous] Error in {strategy_cfg.name} for {sc.ticker}: {exc}")


# ---------------------------------------------------------------------------
# Scan loop
# ---------------------------------------------------------------------------

# Maximum concurrent ticker checks per scan pass.
# Schwab supports 120 req/min; 20 workers is safe for a typical 15-ticker watchlist.
_SCAN_WORKERS = 20


def scan_once(json_path: str, max_workers: int = _SCAN_WORKERS,
              autonomous: bool = False) -> int:
    """
    Run one full scan pass in parallel. Returns number of alerts fired.

    Parameters
    ----------
    max_workers : int
        Concurrent ticker-check threads (default 8).
    autonomous  : bool
        If True, pass signals through the Decision Engine and open positions.
    """
    entries = _load_entries(json_path)
    if not entries:
        print("[scanner] No entries to scan.")
        return 0

    from .autonomous_config import CONFIG
    from .schwab_broker import SchwabBroker

    # ── Phase 1: parallel bar refresh for all watchlist tickers ──────────────
    # Refresh parquets BEFORE the scoring pool so _check_ticker always sees
    # current bars.  Without this, support touches found in simulation are
    # missed in live scanning because the cache is stale.
    _entry_tickers = list({e.ticker for e in entries})
    if _entry_tickers:
        from .data_fetcher import refresh_today
        _provider = getattr(CONFIG, "data_provider", "schwab_data")
        with ThreadPoolExecutor(max_workers=min(_SCAN_WORKERS, len(_entry_tickers))) as _pool:
            _futs = [_pool.submit(refresh_today, t, _provider) for t in _entry_tickers]
            for _f in as_completed(_futs):
                try:
                    _f.result()
                except Exception:
                    pass  # non-fatal — score from cached bars

    # ── Phase 2: Fetch Schwab quotes for all active tickers ──────────────────
    broker = SchwabBroker(CONFIG)
    
    all_tickers = list({e.ticker for e in entries})
    
    if autonomous:
        from .db import WatchlistDB
        try:
            with WatchlistDB(CONFIG.db_path) as db:
                open_trades = db.load_actual_trades(open_only=True)
                all_tickers.extend([t["ticker"] for t in open_trades])
        except Exception:
            pass
    
    all_tickers = list(set(all_tickers))
    quotes = {}
    if all_tickers:
        quotes = broker.get_quotes(all_tickers)

    # Lazy-init autonomous components (shared across alerts this pass)
    monitor  = None
    engines  = []   # list of (DecisionEngine, StrategyConfig, MasterConfig) tuples
    if autonomous:
        from .decision_engine import DecisionEngine
        from .position_monitor import PositionMonitor
        from .schwab_broker import SchwabBroker
        _broker = SchwabBroker(CONFIG) if not CONFIG.paper_mode else None
        monitor = PositionMonitor(config=CONFIG, broker=_broker)
        for strategy_cfg in CONFIG.strategies:
            engines.append((
                DecisionEngine(strategy=strategy_cfg, master_config=CONFIG),
                strategy_cfg,
                CONFIG,
            ))

    all_scores  = []   # every scored SetupScore (for autonomous execution)

    with ThreadPoolExecutor(max_workers=min(max_workers, len(entries))) as pool:
        futures = {pool.submit(_check_ticker, entry, quotes.get(entry.ticker)): entry for entry in entries}
        for fut in as_completed(futures):
            try:
                sc = fut.result()
            except Exception:
                sc = None
            if sc is not None:
                all_scores.append(sc)

    # Banners/sounds only for pattern-confirmed setups (backtest-style alert)
    # Card strategy needs no pattern — it just needs score ≥ 4.3
    pattern_alerts = [sc for sc in all_scores if sc.pattern_score > 0 and sc.support > 0
                      and min(sc.entry_price or 0, sc.support) > 0
                      and abs((sc.entry_price or 0) - sc.support) / sc.support <= _TOUCH_BAND]

    for sc in pattern_alerts:
        _banner(sc)
        _desktop_notify(sc)
        _sound()

    # Autonomous execution runs on ALL scored setups — each strategy applies its own gate
    if autonomous and monitor is not None and engines:
        for sc in all_scores:
            _autonomous_execute_all(sc, monitor, engines)

    # In autonomous mode: also check open positions for exits each cycle.
    # Use 1-min Schwab bars (last 30 min) merged on top of the 5-min parquet so
    # the bar-loop TP/SL checks fire at 1-min resolution rather than waiting for
    # the next 5-min candle.  Falls back to 5-min-only if 1-min fetch fails.
    if autonomous and monitor is not None:
        from .data_fetcher import load_30day_bars, refresh_today, _fetch_schwab_1min
        _provider = getattr(CONFIG, "data_provider", "schwab_data")
        open_tickers = set(
            t["ticker"]
            for t in monitor._db_conn().load_actual_trades(open_only=True)
        )
        bmap = {}
        for t in open_tickers:
            try:
                refresh_today(t, provider=_provider)
            except Exception:
                pass
            base = load_30day_bars(t)
            # Overlay 1-min bars for the last 30 minutes for sub-5-min exit precision
            if _provider == "schwab_data":
                try:
                    import pandas as _pd
                    one_min = _fetch_schwab_1min(t, lookback_minutes=30)
                    if not one_min.empty and not base.empty:
                        combined = _pd.concat([base, one_min])
                        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                        base = combined
                    elif not one_min.empty:
                        base = one_min
                except Exception:
                    pass  # fall back to 5-min parquet
            bmap[t] = base
        closed = monitor.check_all_positions(bmap, quotes)
        for c in closed:
            print(
                f"{_CYAN}[auto] EXIT {c['ticker']}  "
                f"reason={c['reason']}  PnL=${c['pnl_dollar']:+.2f} ({c['pnl_pct']:+.2f}%){_RESET}"
            )

    ts_now = datetime.now(tz=timezone.utc)
    ts = ts_now.strftime("%H:%M:%S UTC")
    print(f"[scanner] {ts} — scanned {len(entries)} tickers, {len(pattern_alerts)} alert(s).")

    # ── Write heartbeat so the Streamlit page can show "Last scan / Next in" ──
    try:
        _HEARTBEAT_FILE.write_text(
            json.dumps({
                "last_scan_ts": ts_now.isoformat(),
                "poll_interval": _POLL_INTERVAL,
                "tickers_scanned": len(entries),
                "alerts": len(pattern_alerts),
            })
        )
    except Exception:
        pass  # never crash the scanner over a UI hint file

    return len(pattern_alerts)


# How often the fast position monitor loop polls Schwab quotes for exits (seconds).
# With tight TP/SL (1%), 10s gives sub-15s exit latency at low API cost.
# Budget: up to 5 open positions × 1 batch quote call / 10s = 6 req/min.
_POSITION_POLL_INTERVAL = 10


def _position_monitor_loop(autonomous: bool = False) -> None:
    """
    Background thread: poll open positions every _POSITION_POLL_INTERVAL seconds
    using live Schwab quotes only (no bar refresh).  This provides sub-15-second
    exit latency for tight TP/SL without waiting for the full 60-second scan cycle.

    Runs as a daemon thread alongside run_loop().
    """
    if not autonomous:
        return

    from .autonomous_config import CONFIG
    from .schwab_broker import SchwabBroker
    from .db import WatchlistDB
    from .position_monitor import PositionMonitor

    _broker = SchwabBroker(CONFIG) if not CONFIG.paper_mode else None
    monitor  = PositionMonitor(config=CONFIG, broker=_broker)

    while True:
        time.sleep(_POSITION_POLL_INTERVAL)
        try:
            with WatchlistDB(CONFIG.db_path) as db:
                open_trades = db.load_actual_trades(open_only=True)
            if not open_trades:
                continue

            open_tickers = list({t["ticker"] for t in open_trades})
            broker_q = SchwabBroker(CONFIG) if not CONFIG.paper_mode else None
            quotes: dict = {}
            if broker_q and open_tickers:
                try:
                    quotes = broker_q.get_quotes(open_tickers)
                except Exception:
                    pass

            # Pass empty bars_map — only exit E (real-time quote) will fire.
            # The full bar-loop exits (A/B/C/D) run in the main scan_once cycle.
            closed = monitor.check_all_positions(bars_map={}, quotes_map=quotes)
            for c in closed:
                print(
                    f"{_CYAN}[pos-monitor] FAST EXIT {c['ticker']}  "
                    f"reason={c['reason']}  PnL=${c['pnl_dollar']:+.2f} ({c['pnl_pct']:+.2f}%){_RESET}"
                )
        except Exception as exc:
            pass  # never crash the background monitor


def run_loop(json_path: str, interval: int, autonomous: bool = False) -> None:
    """
    Poll indefinitely, sleeping *interval* seconds between scans.

    Spawns a background _position_monitor_loop() thread that polls open
    positions every _POSITION_POLL_INTERVAL (10s) via live Schwab quotes
    for fast TP/SL exits, independent of the main 60s scan cycle.
    """
    import threading as _threading

    if autonomous:
        _t = _threading.Thread(
            target=_position_monitor_loop,
            args=(autonomous,),
            daemon=True,
            name="position-monitor-fast",
        )
        _t.start()
        print(f"[scanner] Fast position monitor started — polling every {_POSITION_POLL_INTERVAL}s")

    mode = "AUTONOMOUS" if autonomous else "alert-only"
    print(f"[scanner] Starting live scan ({mode}) — interval {interval}s — Ctrl+C to stop")
    try:
        while True:
            scan_once(json_path, autonomous=autonomous)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[scanner] Stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="live_scanner",
        description="Real-time entry alert scanner for watchlist tickers.",
    )
    p.add_argument(
        "--watchlist", default=str(_DEFAULT_JSON),
        help="Path to scraped_watchlists.json",
    )
    p.add_argument(
        "--interval", type=int, default=_POLL_INTERVAL,
        help="Seconds between scan passes (default 300)",
    )
    p.add_argument(
        "--once", action="store_true",
        help="Run a single scan pass and exit (no loop)",
    )
    p.add_argument(
        "--autonomous", action="store_true",
        help=(
            "Enable autonomous execution mode: signals pass through the Decision "
            "Engine and positions are opened/monitored automatically. "
            "Runs in paper mode by default (autonomous_config.py paper_mode=True). "
            "Flip paper_mode=False after Schwab API credentials are configured."
        ),
    )
    p.add_argument(
        "--no-notifications", action="store_true", dest="no_notifications",
        help="Suppress macOS desktop notifications and sounds (use when running from Streamlit).",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    # Apply notification suppression flag before any scanning.
    # Set the module-level global directly (this IS the running module when
    # invoked via `python -m side_by_side_backtest.live_scanner`).
    if getattr(args, "no_notifications", False):
        _NOTIFICATIONS_ENABLED = False  # module-level global in THIS module
        # Also patch via sys.modules in case the package was already imported
        import sys as _sys
        _mod = _sys.modules.get("side_by_side_backtest.live_scanner")
        if _mod is not None:
            _mod._NOTIFICATIONS_ENABLED = False
    if args.once:
        scan_once(args.watchlist, autonomous=getattr(args, "autonomous", False))
    else:
        run_loop(args.watchlist, args.interval, autonomous=getattr(args, "autonomous", False))
