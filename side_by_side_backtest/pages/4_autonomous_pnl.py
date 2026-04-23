"""
Autonomous PnL — Paper & Live Trade Dashboard
==============================================
Tracks cumulative PnL for the autonomous trading loop.
Shows both paper trades (source='paper') and live trades (source='live')
from the actual_trades table — separate from the simulated trades table.

Sections:
  1. Summary metrics  — total PnL $, WR, expectancy, Sharpe, open positions
  2. Cumulative PnL chart — dollar-based equity curve, colour-coded by source
  3. Trade log table      — all closed trades, sortable, filterable
  4. Open positions       — currently open autonomous positions

Launch via:
    streamlit run side_by_side_backtest/app.py  →  📊 Autonomous PnL
"""
from __future__ import annotations

import subprocess
import sys
import threading as _threading
import time as _wall_clock
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

_PKG = Path(__file__).parent.parent
if str(_PKG.parent) not in sys.path:
    sys.path.insert(0, str(_PKG.parent))

from side_by_side_backtest.db import WatchlistDB
from side_by_side_backtest.autonomous_config import CONFIG

# Strategy display config: name → (display label, colour)
_STRATEGY_STYLE = {
    "backtest_strategy": ("📊 Backtest Strategy",  "#7986cb"),
}

# ---------------------------------------------------------------------------
# Immediate in-process scan — runs once on first page load, then via background loop
# ---------------------------------------------------------------------------

def _ensure_scanner_running(json_path: str, notifications: bool = False) -> None:
    """
    Run an immediate autonomous scan pass in-process on first page load,
    then spawn a background subprocess for the ongoing 5-min loop.

    notifications=False (the default) suppresses macOS desktop alerts and
    sounds entirely — both for the in-process initial scan AND for the
    background subprocess.  Only set notifications=True when the user
    explicitly enables the toggle in the sidebar.
    """
    # ── Kill stale subprocesses FIRST (before initial scan) ──────────────────
    if not st.session_state.get("scanner_proc_started"):
        subprocess.run(["pkill", "-f", "live_scanner"], capture_output=True)
        st.session_state["scanner_proc_started"] = True

    # ── Immediate first-pass (in-process, runs once per Streamlit session) ──
    if not st.session_state.get("initial_scan_done"):
        st.session_state["initial_scan_done"] = True
        try:
            with st.spinner("🤖 Running initial autonomous scan…"):
                import side_by_side_backtest.live_scanner as _ls
                # Always OFF for the in-process scan regardless of sidebar toggle —
                # the Streamlit process should never fire macOS notifications/sounds.
                _prev = _ls._NOTIFICATIONS_ENABLED
                _ls._NOTIFICATIONS_ENABLED = False
                try:
                    _ls.scan_once(json_path, autonomous=True)
                finally:
                    # Restore only if the user explicitly enabled notifications;
                    # otherwise leave disabled so subsequent in-process calls
                    # (e.g. cache-busted reruns) remain silent.
                    _ls._NOTIFICATIONS_ENABLED = notifications
            # Record the scan timestamp so the status banner can show it
            st.session_state["last_scan_ts"] = datetime.now(tz=timezone.utc)
            st.cache_data.clear()
            st.toast("✅ Initial scan complete — positions entered", icon="🤖")
        except Exception as exc:
            st.warning(f"Initial scan error: {exc}")

    # ── Background loop subprocess (ongoing 5-min polling) ──────────────────

    if st.session_state.get("scanner_proc") is not None:
        proc = st.session_state["scanner_proc"]
        if proc.poll() is None:
            return  # still running

    python = sys.executable
    cmd = [
        python, "-m", "side_by_side_backtest.live_scanner",
        "--watchlist", json_path,
        "--autonomous",
    ]
    # Always pass --no-notifications unless the user explicitly enabled the
    # sidebar toggle — macOS notifications from a background subprocess that
    # the user can't see are always surprising and unwanted by default.
    if not notifications:
        cmd.append("--no-notifications")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(_PKG.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        st.session_state["scanner_proc"] = proc
    except Exception as exc:
        st.warning(f"Could not start background scanner: {exc}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading autonomous trades…", ttl=30)
def _load_actual_trades(source_filter: Optional[str]) -> pd.DataFrame:
    """Load actual_trades from DB into a DataFrame."""
    try:
        with WatchlistDB(CONFIG.db_path) as db:
            rows = db.load_actual_trades(source=source_filter or None)
    except Exception as exc:
        st.error(f"Could not load actual_trades DB: {exc}")
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")
    df["exit_ts"]  = pd.to_datetime(df["exit_ts"],  utc=True, errors="coerce")
    df.sort_values("entry_ts", inplace=True)
    return df


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _summary(df: pd.DataFrame) -> dict:
    closed = df[df["outcome"] != "open"]
    if closed.empty:
        return {}
    wins   = closed[closed["outcome"] == "win"]
    losses = closed[closed["outcome"] == "loss"]
    wr     = len(wins) / len(closed) if len(closed) else 0
    avg_w  = wins["pnl_dollar"].mean()  if len(wins)   else 0
    avg_l  = losses["pnl_dollar"].mean() if len(losses) else 0
    return {
        "total_trades": len(closed),
        "wins":         len(wins),
        "losses":       len(losses),
        "wr":           wr,
        "total_pnl_dollar": closed["pnl_dollar"].sum(),
        "expectancy_dollar": wr * avg_w + (1 - wr) * avg_l,
        "avg_win":     avg_w,
        "avg_loss":    avg_l,
    }


def _sharpe_dollar(pnl_series: pd.Series) -> float:
    if len(pnl_series) < 2:
        return 0.0
    std = pnl_series.std()
    return round(pnl_series.mean() / std * (252 ** 0.5), 2) if std else 0.0


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _cumulative_pnl_chart(df: pd.DataFrame) -> go.Figure:
    """Dual equity curve — one line per strategy_name."""
    closed = df[df["outcome"] != "open"].copy()
    if closed.empty:
        return go.Figure()

    fig = go.Figure()

    # Separate trace per strategy so you can compare them side-by-side
    strategies_present = closed["strategy_name"].unique() if "strategy_name" in closed.columns else ["all"]
    for sname in strategies_present:
        label, color = _STRATEGY_STYLE.get(sname, (sname, "#aaaaaa"))
        sub = closed[closed["strategy_name"] == sname].sort_values("exit_ts").copy()
        sub["cum_pnl"] = sub["pnl_dollar"].cumsum()
        fig.add_trace(go.Scatter(
            x=sub["exit_ts"],
            y=sub["cum_pnl"],
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=5),
            name=label,
        ))

    # Combined total curve
    total = closed.sort_values("exit_ts").copy()
    total["cum_pnl"] = total["pnl_dollar"].cumsum()
    fig.add_trace(go.Scatter(
        x=total["exit_ts"], y=total["cum_pnl"],
        mode="lines",
        line=dict(color="#ffffff", width=1, dash="dot"),
        name="Combined",
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1)
    fig.update_layout(
        title="Cumulative PnL — Strategy Comparison",
        xaxis_title="Date", yaxis_title="Cumulative PnL ($)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40), height=420,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis=dict(gridcolor="#1f2937"), yaxis=dict(gridcolor="#1f2937"),
    )
    return fig


def _drawdown_chart(df: pd.DataFrame) -> go.Figure:
    closed = df[df["outcome"] != "open"].copy().sort_values("exit_ts")
    if len(closed) < 2:
        return go.Figure()

    cum = closed["pnl_dollar"].cumsum()
    peak = cum.cummax()
    drawdown = cum - peak

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=closed["exit_ts"],
        y=drawdown,
        fill="tozeroy",
        fillcolor="rgba(239, 83, 80, 0.2)",
        line=dict(color="#ef5350", width=1),
        name="Drawdown ($)",
    ))
    fig.update_layout(
        title="Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown from Peak ($)",
        height=200,
        margin=dict(l=40, r=20, t=40, b=30),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis=dict(gridcolor="#1f2937"),
        yaxis=dict(gridcolor="#1f2937"),
    )
    return fig


# ---------------------------------------------------------------------------
# Live scan status banner — identical style to Morning Brief
# ---------------------------------------------------------------------------

_POLL_INTERVAL  = 60    # mirrors live_scanner._POLL_INTERVAL (entry + position checks every 60 s)
_HEARTBEAT_FILE = Path(__file__).parent.parent / ".scanner_heartbeat.json"


def _read_heartbeat() -> tuple[Optional[datetime], int]:
    """
    Read the heartbeat file written by live_scanner.scan_once().
    Returns (last_scan_utc_datetime, poll_interval).
    Falls back to session_state["last_scan_ts"] before the first file write.
    """
    import json as _json
    try:
        data     = _json.loads(_HEARTBEAT_FILE.read_text())
        raw      = data.get("last_scan_ts")
        interval = int(data.get("poll_interval", _POLL_INTERVAL))
        if raw:
            return datetime.fromisoformat(raw), interval
    except Exception:
        pass
    return st.session_state.get("last_scan_ts"), _POLL_INTERVAL


def _js_live_indicator(scan_interval: int, last_scan_str: str) -> None:
    """
    Inject a client-side JS countdown + flashing dot — same as Morning Brief.
    No server round-trips; the countdown ticks purely in the browser.
    """
    html = f"""
    <div style="display:flex; align-items:center; gap:10px;
                font-family:monospace; font-size:13px; color:#cccccc;
                padding:4px 0;">
        <span id="live-dot" style="font-size:16px;">🔴</span>
        <span>
            LIVE &nbsp;|&nbsp; Last scan: <b>{last_scan_str}</b>
            &nbsp;|&nbsp; Next in: <b><span id="cd">{scan_interval}</span>s</b>
        </span>
    </div>
    <script>
        (function() {{
            let t = {scan_interval};
            const cd   = document.getElementById('cd');
            const dot  = document.getElementById('live-dot');
            const dots = ['🔴','🟠'];
            let d = 0;
            const iv = setInterval(() => {{
                t = Math.max(0, t - 1);
                cd.textContent = t;
                dot.textContent = dots[d % 2];
                d++;
                if (t <= 0) clearInterval(iv);
            }}, 1000);
        }})();
    </script>
    """
    components.html(html, height=36)


def _render_scan_status() -> None:
    """
    Show the live-scan status banner using the heartbeat file written by
    live_scanner.scan_once() — works for both in-process and subprocess scans.
    """
    last_ts, interval = _read_heartbeat()

    if last_ts is None:
        # No scan has completed yet — show a static "Scanning…" state
        _js_live_indicator(interval, "Scanning…")
        return

    now      = datetime.now(tz=timezone.utc)
    elapsed  = int((now - last_ts).total_seconds())
    next_in  = max(0, interval - elapsed)
    last_str = last_ts.astimezone().strftime("%H:%M:%S")
    _js_live_indicator(next_in, last_str)


# ---------------------------------------------------------------------------
# Watchlist panel — always shown, even before any trades fire
# ---------------------------------------------------------------------------

# Refresh-cooldown guard (mirrors Morning Brief _last_refresh_ts pattern).
# Key: ticker → monotonic wall-clock of last refresh_today() call.
_wl_last_refresh: dict = {}
_WL_REFRESH_COOLDOWN = 55.0   # seconds — slightly under the 60 s fragment interval
_wl_bg_lock = _threading.Lock()
_wl_bg_tickers: set = set()


def _wl_prefetch(tickers: list, background: bool = False) -> None:
    """Refresh today's 5-min parquets for all *tickers* — mirrors Morning Brief."""
    from side_by_side_backtest.data_fetcher import refresh_today

    now = _wall_clock.monotonic()
    due = [t for t in tickers if now - _wl_last_refresh.get(t, 0) >= _WL_REFRESH_COOLDOWN]
    if not due:
        return

    def _one(ticker: str) -> None:
        try:
            refresh_today(ticker)
            _wl_last_refresh[ticker] = _wall_clock.monotonic()
        except Exception:
            pass
        finally:
            with _wl_bg_lock:
                _wl_bg_tickers.discard(ticker)

    if background:
        with _wl_bg_lock:
            new_due = [t for t in due if t not in _wl_bg_tickers]
            _wl_bg_tickers.update(new_due)
        if not new_due:
            return
        def _run():
            with ThreadPoolExecutor(max_workers=min(8, len(new_due))) as p:
                for f in as_completed([p.submit(_one, t) for t in new_due]):
                    try: f.result()
                    except Exception: pass
        _threading.Thread(target=_run, daemon=True, name="wl-bg-refresh").start()
    else:
        with ThreadPoolExecutor(max_workers=min(8, len(due))) as p:
            for f in as_completed([p.submit(_one, t) for t in due]):
                try: f.result()
                except Exception: pass


def _score_watchlist_live(json_path: str) -> pd.DataFrame:
    """Score the latest watchlist post from freshly-refreshed on-disk parquets."""
    import json as _json
    from side_by_side_backtest.parser import parse_watchlist_post
    from side_by_side_backtest.models import RawWatchlist
    from side_by_side_backtest.data_fetcher import load_30day_bars
    from side_by_side_backtest.setup_scorer import score_setup

    try:
        raw = _json.loads(Path(json_path).read_text())
    except Exception:
        return pd.DataFrame()
    if not raw:
        return pd.DataFrame()

    latest_ts = max((p.get("timestamp") or "") for p in raw)
    latest_raw = [p for p in raw if (p.get("timestamp") or "") == latest_ts]

    entries: list = []
    for post in latest_raw:
        try:
            entries.extend(parse_watchlist_post(RawWatchlist(**post)))
        except Exception:
            pass

    seen: dict = {}
    for e in reversed(entries):
        seen.setdefault(e.ticker, e)
    entries = list(seen.values())

    rows = []
    for entry in entries:
        bars = load_30day_bars(entry.ticker)
        if bars.empty:
            continue
        try:
            sc = score_setup(entry, bars)
            rows.append({
                "Ticker":    sc.ticker,
                "Score":     sc.score,
                "Signal":    sc.signal,
                "Entry $":   f"${sc.entry_price:.3f}" if sc.entry_price else "—",
                "Support $": f"${sc.support:.3f}"     if sc.support     else "—",
                "R/R":       f"{sc.rr_ratio:.1f}:1"   if sc.rr_ratio    else "—",
                "ADX":       f"{sc.adx:.1f}"           if sc.adx         else "—",
                "Enter?":    "✅" if sc.pattern_found else "❌",
            })
        except Exception:
            pass

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Score", ascending=False)


@st.fragment(run_every=60)
def _render_live_panel(json_path: str) -> None:
    """
    Single consolidated 60-second live fragment — one status bar, one scan tick.

    Each tick (every 60 s):
      1. Refresh ticker bars (background on first render, blocking on subsequent)
      2. Score watchlist setups → display ranked table
      3. Check open-position exits (PT/SL/time-stop/momentum-fade) → toast exits
      4. Display open positions with live unrealised P&L
      5. Show ONE shared JS live-indicator at the bottom
    """
    import json as _json
    import time as _t
    from side_by_side_backtest.position_monitor import PositionMonitor
    from side_by_side_backtest.data_fetcher import load_30day_bars

    now_str = _t.strftime("%H:%M:%S", _t.localtime())
        # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Single shared live indicator
    # ═══════════════════════════════════════════════════════════════════════════
    _js_live_indicator(60, now_str)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Watchlist scores
    # ═══════════════════════════════════════════════════════════════════════════
    st.subheader("📋 Latest Watchlist Post — Scanner Universe")

    tickers: list = []
    post_title = ""
    latest_ts_str = ""
    try:
        raw = _json.loads(Path(json_path).read_text())
        if raw:
            latest_ts_str = max((p.get("timestamp") or "") for p in raw)
            latest_posts  = [p for p in raw if (p.get("timestamp") or "") == latest_ts_str]
            if latest_posts:
                post_title = latest_posts[0].get("title", "")
            from side_by_side_backtest.parser import parse_watchlist_post
            from side_by_side_backtest.models import RawWatchlist
            for post in latest_posts:
                try:
                    for e in parse_watchlist_post(RawWatchlist(**post)):
                        if e.ticker not in tickers:
                            tickers.append(e.ticker)
                except Exception:
                    pass
    except Exception:
        pass

    if post_title:
        st.caption(f"📌 Post: **{post_title}** — `{latest_ts_str[:16]}`")

    is_first = not st.session_state.get("_wl_scored", False)
    if tickers:
        if is_first:
            st.session_state["_wl_scored"] = True
            _wl_prefetch(tickers, background=True)
        else:
            _wl_prefetch(tickers, background=False)

    wdf = _score_watchlist_live(json_path)
    if wdf.empty:
        st.caption("No scored setups yet — bars may still be loading.")
    else:
        st.dataframe(wdf, use_container_width=True, hide_index=True)
        st.caption(
            f"📊 Backtest Strategy enters on any pattern + support touch."
        )

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Open position exit check + live P&L
    # ═══════════════════════════════════════════════════════════════════════════
    monitor = PositionMonitor(config=CONFIG)

    try:
        with WatchlistDB(CONFIG.db_path) as _db:
            open_rows = _db.load_actual_trades(open_only=True)
        open_tickers = list({r["ticker"] for r in open_rows})
        if open_tickers:
            bars_map = {t: load_30day_bars(t) for t in open_tickers}
            closed_exits = monitor.check_all_positions(bars_map)
            for c in closed_exits:
                st.toast(
                    f"{'✅' if c.get('pnl_dollar', 0) >= 0 else '❌'} "
                    f"**{c['ticker']}** closed — "
                    f"{c.get('reason', 'exit')}  ${c.get('pnl_dollar', 0):+.2f}",
                    icon="🤖",
                )
    except Exception as exc:
        st.warning(f"Position monitor error: {exc}")

    try:
        with WatchlistDB(CONFIG.db_path) as _db:
            open_rows = _db.load_actual_trades(open_only=True)
    except Exception:
        open_rows = []

    st.subheader(f"🔓 Open Positions ({len(open_rows)})")
    if not open_rows:
        st.caption("No open positions right now.")
    else:
        pos_rows = []
        for r in open_rows:
            ticker      = r["ticker"]
            entry_price = r.get("entry_price") or 0
            qty         = r.get("quantity") or 0
            try:
                bars = load_30day_bars(ticker)
                last = float(bars["close"].iloc[-1]) if not bars.empty else 0.0
            except Exception:
                last = 0.0
            unreal     = (last - entry_price) * qty if last and entry_price else 0.0
            unreal_pct = (last - entry_price) / entry_price * 100 if entry_price else 0.0
            pos_rows.append({
                "Ticker":     ticker,
                "Strategy":   _STRATEGY_STYLE.get(r.get("strategy_name", ""), (r.get("strategy_name", ""), ""))[0],
                "Entry $":    f"${entry_price:.3f}",
                "Last $":     f"${last:.3f}" if last else "—",
                "Unreal $":   f"${unreal:+.2f}",
                "Unreal %":   f"{unreal_pct:+.2f}%",
                "Qty":        qty,
                "PT%":        r.get("pt_pct", "—"),
                "SL%":        r.get("sl_pct", "—"),
                "Score":      r.get("setup_score", "—"),
                "Entry Time": pd.Timestamp(r["entry_ts"]).strftime("%H:%M:%S") if r.get("entry_ts") else "—",
            })
        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

        st.markdown("**🔓 Position Actions**")
        for r in open_rows:
            col1, col2 = st.columns([3, 1])
            col1.write(f"**{r['ticker']}** ({_STRATEGY_STYLE.get(r.get('strategy_name', ''), (r.get('strategy_name', ''), ''))[0]})")
            if col2.button(f"🔴 Sell {r['ticker']}", key=f"sell_{r['id']}"):
                try:
                    from side_by_side_backtest.position_monitor import PositionMonitor
                    from side_by_side_backtest.data_fetcher import load_30day_bars
                    monitor = PositionMonitor(CONFIG)
                    bars = load_30day_bars(r['ticker'])
                    last_price = float(bars["close"].iloc[-1]) if not bars.empty else r['entry_price']
                    
                    monitor._close_position(
                        row_id=r['id'],
                        trade=r,
                        exit_ts=datetime.now(tz=timezone.utc),
                        exit_price=last_price,
                        reason="manual_exit",
                        quantity=r.get('quantity', 1),
                        entry_price=r['entry_price']
                    )
                    
                    import json as _json
                    _OVERRIDE_PATH = _PKG / ".autonomous_overrides.json"
                    overrides = {"pt_pct": 1.5, "sl_pct": 0.124, "kill_switch": True}
                    if _OVERRIDE_PATH.exists():
                        try:
                            overrides = _json.loads(_OVERRIDE_PATH.read_text())
                            overrides["kill_switch"] = True
                        except Exception:
                            pass
                    _OVERRIDE_PATH.write_text(_json.dumps(overrides))
                    
                    st.success(f"Sold {r['ticker']} manually. Auto Trader HALTED.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Manual exit failed: {exc}")

    st.divider()


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Autonomous PnL",
        page_icon="🤖",
        layout="wide",
    )
    st.title("🤖 Autonomous PnL — Dual Strategy Dashboard")
    # _render_scan_status() <--- current disabled.

    mode_label = "🟡 PAPER MODE" if CONFIG.paper_mode else "🟢 LIVE MODE"
    _json_path = str(Path(__file__).parent.parent.parent / "scraped_watchlists.json")

    # ── Sidebar (must come before scanner so toggle is read first) ────────────
    with st.sidebar:
        st.markdown(f"**Trading Mode:** {mode_label}")
        st.divider()

        # Notifications toggle — disable when using from Streamlit page
        notif_enabled = st.toggle(
            "🔔 macOS notifications",
            value=False,           # default OFF when using the page
            help="Enable desktop notifications and sound alerts when signals fire. "
                 "Disable to avoid conflicts with the browser session.",
        )
        st.divider()
        st.markdown("**Strategies:**")
        for s in CONFIG.strategies:
            if s.name != "backtest_strategy":
                continue
            label, color = _STRATEGY_STYLE.get(s.name, (s.name, "#aaa"))
            st.markdown(
                f"<span style='color:{color}'>●</span> **{label}**  "
                f"  Budget ${s.budget_total:,.0f} | ${s.trade_size:.0f}/trade",
                unsafe_allow_html=True,
            )
        st.divider()
        st.markdown("**⚙️ Controls**")
        
        # Load existing overrides
        import json as _json
        _OVERRIDE_PATH = _PKG / ".autonomous_overrides.json"
        overrides = {"pt_pct": 1.5, "sl_pct": 0.124, "kill_switch": False}
        if _OVERRIDE_PATH.exists():
            try:
                overrides = _json.loads(_OVERRIDE_PATH.read_text())
            except Exception:
                pass
                
        # Tunable SP/SL
        new_pt = st.slider("Take Profit % (SP)", 0.1, 10.0, float(overrides.get("pt_pct", 1.5)), 0.1)
        new_sl = st.slider("Stop Loss % (SL)", 0.01, 5.0, float(overrides.get("sl_pct", 0.124)), 0.01)
        
        # Kill Switch
        kill = st.toggle("🛑 STOP AUTO TRADER", value=overrides.get("kill_switch", False))
        
        if new_pt != overrides.get("pt_pct") or new_sl != overrides.get("sl_pct") or kill != overrides.get("kill_switch"):
            overrides["pt_pct"] = new_pt
            overrides["sl_pct"] = new_sl
            overrides["kill_switch"] = kill
            _OVERRIDE_PATH.write_text(_json.dumps(overrides))
            st.toast("⚙️ Overrides saved!")

        st.divider()
        source_filter = st.radio(
            "Show trades from:",
            options=["All", "Paper only", "Live only"],
            index=0,
        )
        st.divider()
        col_r, col_s = st.columns(2)
        if col_r.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()
        if col_s.button("▶ Restart scanner"):
            st.session_state.pop("scanner_proc", None)
            st.session_state.pop("initial_scan_done", None)
            _ensure_scanner_running(_json_path, notifications=notif_enabled)

        st.divider()
        st.markdown("**⚠️ Danger Zone**")
        if st.button("🗑️ Clear all DB data", type="secondary",
                     help="Wipe all actual_trades, simulated trades, and watchlist entries. Cannot be undone."):
            try:
                with WatchlistDB(CONFIG.db_path) as _db:
                    _db.clear_all()
                st.session_state.pop("scanner_proc", None)
                st.session_state.pop("initial_scan_done", None)
                st.cache_data.clear()
                st.success("✅ Database cleared — all tables wiped.")
                st.rerun()
            except Exception as _e:
                st.error(f"Clear failed: {_e}")

    # ── Auto-start (after sidebar so notif_enabled is resolved) ──────────────
    _ensure_scanner_running(_json_path, notifications=notif_enabled)

    src_map = {"All": None, "Paper only": "paper", "Live only": "live"}
    df = _load_actual_trades(src_map[source_filter])

    closed = df[df["outcome"] != "open"] if not df.empty else pd.DataFrame()

    # ── Live panel: watchlist scores + open-position check (always shown) ────
    _render_live_panel(_json_path)

    if df.empty:
        st.info("🤖 Scanner is running — no trades recorded yet. First signal will appear here.")
        return

    # ── Summary metrics ───────────────────────────────────────────────────────
    sm = _summary(df)
    if sm:
        c1, c2, c3, c4, c5 = st.columns(5)
        pnl_color = "normal" if sm["total_pnl_dollar"] >= 0 else "inverse"
        c1.metric("Total PnL",    f"${sm['total_pnl_dollar']:+,.2f}")
        c2.metric("Win Rate",     f"{sm['wr']*100:.1f}%",
                  f"{sm['wins']}W / {sm['losses']}L")
        c3.metric("Expectancy",   f"${sm['expectancy_dollar']:+.2f}/trade")
        c4.metric("Trades",       sm["total_trades"])
        sharpe = _sharpe_dollar(closed["pnl_dollar"]) if not closed.empty else 0.0
        c5.metric("Sharpe",       f"{sharpe:.2f}")

    # ── Per-strategy comparison ───────────────────────────────────────────────
    if not closed.empty and "strategy_name" in closed.columns:
        strategy_rows = []
        for sname in closed["strategy_name"].unique():
            label, _ = _STRATEGY_STYLE.get(sname, (sname, "#aaa"))
            sub = closed[closed["strategy_name"] == sname]
            ssm = _summary(sub.assign(outcome=sub["outcome"]))
            strategy_rows.append({
                "Strategy":    label,
                "Trades":      ssm.get("total_trades", 0),
                "WR %":        f"{ssm.get('wr', 0)*100:.1f}%",
                "Total PnL $": f"${ssm.get('total_pnl_dollar', 0):+,.2f}",
                "Expectancy":  f"${ssm.get('expectancy_dollar', 0):+.2f}",
            })
        if strategy_rows:
            st.dataframe(pd.DataFrame(strategy_rows), width='stretch',
                         hide_index=True)

    st.divider()

    # ── Equity curve + drawdown ───────────────────────────────────────────────
    if not closed.empty:
        st.plotly_chart(_cumulative_pnl_chart(df), width='stretch')
        st.plotly_chart(_drawdown_chart(df), width='stretch')
    else:
        st.info("No closed trades yet — equity curve will appear here.")

    st.divider()

    # ── Trade log ─────────────────────────────────────────────────────────────
    st.subheader("📋 Closed Trade Log")
    if closed.empty:
        st.info("No closed trades yet.")
    else:
        disp = closed[[
            "ticker", "source", "entry_ts", "exit_ts",
            "entry_price", "exit_price", "quantity",
            "pnl_dollar", "pnl_pct", "outcome", "exit_reason", "setup_score",
        ]].copy()
        disp["entry_ts"] = disp["entry_ts"].dt.strftime("%Y-%m-%d %H:%M")
        disp["exit_ts"]  = disp["exit_ts"].dt.strftime("%Y-%m-%d %H:%M").fillna("—")
        disp["pnl_dollar"] = disp["pnl_dollar"].map(lambda x: f"${x:+.2f}")
        disp["pnl_pct"]    = disp["pnl_pct"].map(lambda x: f"{x:+.2f}%")
        disp = disp.rename(columns={
            "ticker": "Ticker", "source": "Source",
            "entry_ts": "Entry Time", "exit_ts": "Exit Time",
            "entry_price": "Entry $", "exit_price": "Exit $",
            "quantity": "Qty", "pnl_dollar": "PnL $", "pnl_pct": "PnL %",
            "outcome": "Outcome", "exit_reason": "Exit Reason",
            "setup_score": "Score",
        })
        st.dataframe(
            disp.sort_values("Entry Time", ascending=False),
            width='stretch',
            hide_index=True,
        )

        # Per-ticker breakdown
        st.subheader("Per-Ticker Summary")
        ticker_grp = closed.groupby("ticker").apply(lambda g: pd.Series({
            "Trades":     len(g),
            "WR %":       round((g["outcome"] == "win").mean() * 100, 1),
            "Total PnL $": round(g["pnl_dollar"].sum(), 2),
            "Avg PnL $":  round(g["pnl_dollar"].mean(), 2),
        }), include_groups=False).reset_index()
        ticker_grp = ticker_grp.sort_values("Total PnL $", ascending=False)
        st.dataframe(ticker_grp, width='stretch', hide_index=True)


if __name__ == "__main__":
    main()
else:
    main()
