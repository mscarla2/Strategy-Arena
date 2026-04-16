"""
Chart Viewer — Interactive 5-min Candlestick + Pattern Explorer
==============================================================
Streamlit app with TradingView-style dark theme.
Displays 5-min OHLCV candles, session shading, Side-by-Side pattern markers,
support/resistance lines, and trade entry/exit overlays.

Launch:
    streamlit run side_by_side_backtest/chart_viewer.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ── Paths ─────────────────────────────────────────────────────────────────────
_PKG_DIR   = Path(__file__).parent
_CACHE_DIR = _PKG_DIR / "ohlcv_cache"

# ── TradingView-style colour palette ─────────────────────────────────────────
_BG          = "#0d1117"          # pure black background
_BULL        = "#26a69a"          # green candle body
_BEAR        = "#ef5350"          # red candle body
_WICK        = "#555555"          # wick colour
_PREMARKET   = "rgba(101,67,33,0.35)"   # dark amber — pre-market
_AFTERHOURS  = "rgba(13,27,62,0.45)"    # dark navy — after hours
_SUPPORT     = "#ffffff"          # white dashed support line
_RESIST      = "#4caf50"          # green dashed resistance line
_PATTERN_MK  = "#f9d71c"          # yellow diamond — pattern markers
_ENTRY_MK    = "#00bcd4"          # cyan triangle-up — entry
_WIN_MK      = "#26a69a"          # green X — win exit
_LOSS_MK     = "#ef5350"          # red X — loss exit
_TIMEOUT_MK  = "#9e9e9e"          # grey — timeout exit


# ── Session boundary helpers (UTC) ───────────────────────────────────────────
def _session_label(ts: pd.Timestamp) -> str:
    """Return 'pre', 'regular', or 'after' for a UTC bar timestamp."""
    t = ts.tz_convert("UTC").time() if ts.tzinfo else ts.time()
    from datetime import time
    if time(9, 0) <= t < time(13, 30):
        return "pre"
    if time(13, 30) <= t < time(20, 0):
        return "regular"
    if time(20, 0) <= t <= time(23, 59):
        return "after"
    return "pre"  # overnight / early pre-market


# ── Cache scanner ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=60)
def scan_cache() -> Dict[str, List[str]]:
    """
    Scan ohlcv_cache/ and return {ticker: [parquet_stem, ...]} sorted by date.
    Picks up both legacy per-window files (*_YYYY-MM-DD_YYYY-MM-DD_5m.parquet)
    and the new 30-day rolling files (*_30d_5m.parquet).
    """
    result: Dict[str, List[str]] = {}

    # 30-day rolling cache (preferred)
    for pf in sorted(_CACHE_DIR.glob("*_30d_5m.parquet")):
        ticker = pf.stem.replace("_30d_5m", "").upper()
        if ticker:
            result.setdefault(ticker, []).append(pf.stem)

    # Legacy per-window files (supplement)
    for pf in sorted(_CACHE_DIR.glob("*_5m.parquet")):
        if "_30d_5m" in pf.stem:
            continue   # already handled above
        parts = pf.stem.split("_")
        if len(parts) == 4:
            ticker = parts[0]
            result.setdefault(ticker, []).append(pf.stem)

    return result


@st.cache_data(show_spinner=False)
def load_parquet(stem: str) -> pd.DataFrame:
    """Load a single cached parquet file by stem name."""
    path = _CACHE_DIR / f"{stem}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


@st.cache_data(show_spinner="Loading full history…", ttl=300)
def load_full_ticker(ticker: str) -> pd.DataFrame:
    """
    Load the best available history for a ticker.
    Priority: 30-day rolling parquet → merged legacy windows.
    Deduplicates overlapping bars and sorts by timestamp.
    """
    from side_by_side_backtest.data_fetcher import load_30day_bars

    # 1. Try 30-day rolling cache first (richest, most consistent)
    df_30d = load_30day_bars(ticker)

    # 2. Supplement with legacy per-window parquets
    legacy_stems = [s for s in scan_cache().get(ticker, []) if "_30d_5m" not in s]
    frames = [load_parquet(s) for s in legacy_stems]
    frames = [f for f in frames if not f.empty]

    if not df_30d.empty:
        frames.insert(0, df_30d)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames)
    merged = merged[~merged.index.duplicated(keep="first")].sort_index()
    return merged


# ── Chart builder ─────────────────────────────────────────────────────────────

def build_chart(
    df: pd.DataFrame,
    ticker: str = "",
    tz: str = "America/Los_Angeles",  # display timezone for x-axis
) -> go.Figure:
    """
    Build a Plotly candlestick figure matching the TradingView dark theme.
    Session-background shading is added automatically.
    S/R lines are drawn separately by the caller via sr_engine levels.
    The x-axis is displayed in *tz* (default: America/Los_Angeles = PT).
    """
    # Convert index to display timezone so chart times match the live indicator
    if df.index.tzinfo is not None:
        df = df.copy()
        df.index = df.index.tz_convert(tz)
    else:
        df = df.copy()
        df.index = df.index.tz_localize("UTC").tz_convert(tz)

    fig = go.Figure()

    # ── Candlesticks ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name=ticker,
        increasing=dict(line=dict(color=_BULL, width=1), fillcolor=_BULL),
        decreasing=dict(line=dict(color=_BEAR, width=1), fillcolor=_BEAR),
        whiskerwidth=0.3,
        showlegend=False,
        hoverinfo="skip",   # prevents per-candle hover calc; massively reduces zoom lag
        xperiod=300000,     # 5 min in ms — tells Plotly the bar interval for width calc
        xperiodalignment="middle",
    ))

    # ── Session background shading ────────────────────────────────────────────
    _add_session_shading(fig, df)

    # ── Dark theme layout + range-selector buttons ────────────────────────────
    _x_end   = df.index[-1]
    _x_start = _x_end - pd.Timedelta(days=1)

    fig.update_layout(
        plot_bgcolor=_BG, paper_bgcolor=_BG,
        font=dict(color="#cccccc", size=11),
        uirevision=ticker,
        dragmode="pan",
        hoverdistance=1,
        xaxis=dict(
            gridcolor="#1e2530", showgrid=True, zeroline=False,
            rangeslider=dict(visible=False),
            type="date",
            range=[_x_start.isoformat(), _x_end.isoformat()],
            rangeselector=dict(
                buttons=[
                    dict(count=1,  label="1D",  step="day",  stepmode="backward"),
                    dict(count=3,  label="3D",  step="day",  stepmode="backward"),
                    dict(count=7,  label="1W",  step="day",  stepmode="backward"),
                    dict(count=14, label="2W",  step="day",  stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor="#1e2530", activecolor="#26a69a",
                font=dict(color="#cccccc", size=10),
                x=0.0, y=1.02,
            ),
            # Crosshair spike line (TradingView-style)
            showspikes=True, spikemode="across",
            spikesnap="cursor", spikecolor="#555555",
            spikethickness=1, spikedash="dot",
        ),
        yaxis=dict(
            gridcolor="#1e2530", showgrid=True, zeroline=False, side="right",
            showspikes=True, spikemode="across",
            spikesnap="cursor", spikecolor="#555555",
            spikethickness=1, spikedash="dot",
        ),
        margin=dict(l=10, r=60, t=60, b=10),
        height=580,
        title=dict(text=ticker, font=dict(size=14, color="#eeeeee"), x=0.02),
        hovermode="x",          # "x" (not "x unified") reduces jitter on dense charts
        hoverlabel=dict(bgcolor="#1e2530", font_color="#cccccc"),
        modebar=dict(
            bgcolor="rgba(0,0,0,0)",
            color="#666666",
            activecolor="#26a69a",
        ),
    )
    return fig


def _add_session_shading(fig: go.Figure, df: pd.DataFrame) -> None:
    """
    Add session background shading using a single Scattergl trace per session
    type (pre-market and after-hours) instead of one vrect per run.

    Technique: for each non-regular bar we emit two x-values (bar timestamp
    repeated) with y=[y_min, y_max] so the filled area covers the full price
    range, then insert NaN between disjoint runs to break the fill.  Using
    go.Scattergl (WebGL) keeps the layer count minimal and GPU-accelerated.
    """
    if df.empty:
        return

    y_min = float(df["low"].min())
    y_max = float(df["high"].max())
    half_bar = pd.Timedelta("2.5min")  # half a 5-min bar width

    labels = [_session_label(ts) for ts in df.index]

    # Build x/y arrays for each session type with NaN gaps between runs
    session_coords: dict[str, tuple[list, list]] = {
        "pre":   ([], []),
        "after": ([], []),
    }

    i = 0
    while i < len(labels):
        sess = labels[i]
        if sess == "regular":
            i += 1
            continue
        # find end of this run
        j = i
        while j < len(labels) and labels[j] == sess:
            j += 1

        xs, ys = session_coords[sess]
        # Insert NaN gap separator between previous run and this one
        if xs:
            xs.append(None)
            ys.append(None)

        x0 = df.index[i] - half_bar
        x1 = df.index[j - 1] + half_bar
        # Rectangle via 4 corners + closing point
        xs += [x0, x1, x1, x0, x0]
        ys += [y_min, y_min, y_max, y_max, y_min]
        i = j

    _session_colours = {
        "pre":   _PREMARKET,
        "after": _AFTERHOURS,
    }
    _session_names = {
        "pre":   "Pre-market",
        "after": "After-hours",
    }

    for sess, (xs, ys) in session_coords.items():
        if not xs:
            continue
        fig.add_trace(go.Scattergl(
            x=xs, y=ys,
            fill="toself",
            fillcolor=_session_colours[sess],
            line=dict(width=0),
            mode="lines",
            name=_session_names[sess],
            hoverinfo="skip",
            showlegend=False,
        ))


# ── Overlay: pattern markers ──────────────────────────────────────────────────

def add_pattern_overlays(
    fig: go.Figure,
    patterns: list,
    df: pd.DataFrame,
    display_tz: str = "America/Los_Angeles",
) -> go.Figure:
    """Add yellow diamond markers at each detected Side-by-Side pattern bar.
    Converts timestamps to display_tz to match the PT-converted chart x-axis.
    """
    if not patterns:
        return fig

    # Build a lookup: UTC timestamp → (index position, high value)
    # df may still have UTC index here; convert to PT for x-axis matching
    df_disp = df.copy()
    if df_disp.index.tzinfo is not None:
        df_disp.index = df_disp.index.tz_convert(display_tz)
    else:
        df_disp.index = df_disp.index.tz_localize("UTC").tz_convert(display_tz)

    xs, ys, texts = [], [], []
    for p in patterns:
        if p.bar_index < len(df_disp):
            xs.append(df_disp.index[p.bar_index])
            ys.append(float(df_disp["high"].iloc[p.bar_index]) * 1.005)
            ptype = getattr(p, 'pattern_type', 'strict')
            conf  = getattr(p, 'confidence_score', 1.0)
            texts.append(
                f"{ptype.title()} S×S (conf {conf:.1f})<br>"
                f"C1 {p.candle1_open:.3f}→{p.candle1_close:.3f}<br>"
                f"C2 {p.candle2_open:.3f}→{p.candle2_close:.3f}"
            )
    fig.add_trace(go.Scattergl(
        x=xs, y=ys, mode="markers",
        marker=dict(symbol="diamond", size=10, color=_PATTERN_MK,
                    line=dict(color="#000000", width=1)),
        name="Pattern", hovertext=texts, hoverinfo="text",
        showlegend=True,
    ))
    return fig


# ── Overlay: trade entry / exit ───────────────────────────────────────────────

def add_trade_overlays(fig: go.Figure, trade, df: pd.DataFrame) -> go.Figure:
    """Add entry triangle, exit marker, and shaded PnL band for one trade."""
    if trade is None:
        return fig

    # Find closest bar index to entry_ts and exit_ts
    def _nearest_idx(ts):
        if ts is None:
            return None
        diffs = abs(df.index - ts)
        return diffs.argmin() if len(diffs) > 0 else None

    entry_idx = _nearest_idx(trade.entry_ts)
    exit_idx  = _nearest_idx(trade.exit_ts)

    # ── Entry marker (cyan triangle-up) ──────────────────────────────────────
    if entry_idx is not None:
        fig.add_trace(go.Scattergl(
            x=[df.index[entry_idx]], y=[trade.entry_price * 0.997],
            mode="markers",
            marker=dict(symbol="triangle-up", size=14, color=_ENTRY_MK,
                        line=dict(color="#ffffff", width=1)),
            name="Entry", hovertext=f"Entry @ {trade.entry_price:.3f}",
            hoverinfo="text", showlegend=True,
        ))
        tp = trade.entry_price * (1 + trade.profit_target_pct / 100)
        sl = trade.entry_price * (1 - trade.stop_loss_pct / 100)

        # TP line: draw from entry bar to the end of the visible window
        x_rest = list(df.index[entry_idx:])
        if x_rest:
            fig.add_trace(go.Scattergl(
                x=x_rest, y=[tp] * len(x_rest),
                mode="lines", line=dict(color=_BULL, width=1.5, dash="dot"),
                name=f"TP {tp:.3f}", showlegend=True,
                hovertemplate=f"TP: {tp:.3f}<extra></extra>",
            ))
            fig.add_trace(go.Scattergl(
                x=x_rest, y=[sl] * len(x_rest),
                mode="lines", line=dict(color=_BEAR, width=1.5, dash="dot"),
                name=f"SL {sl:.3f}", showlegend=True,
                hovertemplate=f"SL: {sl:.3f}<extra></extra>",
            ))
            # Entry price reference line
            fig.add_trace(go.Scattergl(
                x=x_rest, y=[trade.entry_price] * len(x_rest),
                mode="lines", line=dict(color=_ENTRY_MK, width=1, dash="dash"),
                name=f"Entry {trade.entry_price:.3f}", showlegend=True,
                hovertemplate=f"Entry: {trade.entry_price:.3f}<extra></extra>",
                opacity=0.6,
            ))

    # ── Exit marker ───────────────────────────────────────────────────────────
    if exit_idx is not None and trade.exit_price:
        colour  = _WIN_MK if trade.outcome == "win" else (_LOSS_MK if trade.outcome == "loss" else _TIMEOUT_MK)
        symbol  = "x" if trade.outcome in ("loss", "timeout") else "circle"
        fig.add_trace(go.Scattergl(
            x=[df.index[exit_idx]], y=[trade.exit_price],
            mode="markers",
            marker=dict(symbol=symbol, size=10, color=colour,
                        line=dict(color="#ffffff", width=1)),
            name=f"Exit ({trade.outcome})",
            hovertext=f"Exit @ {trade.exit_price:.3f}  PnL {trade.pnl_pct:+.2f}%",
            hoverinfo="text", showlegend=True,
        ))

    # ── Shaded band between entry and exit ────────────────────────────────────
    if entry_idx is not None and exit_idx is not None and trade.exit_price:
        fill_col = "rgba(38,166,154,0.12)" if trade.outcome == "win" else \
                   "rgba(239,83,80,0.12)"  if trade.outcome == "loss" else \
                   "rgba(120,120,120,0.10)"
        x_band = list(df.index[entry_idx : exit_idx + 1])
        if x_band:
            fig.add_trace(go.Scattergl(
                x=x_band + x_band[::-1],
                y=[trade.entry_price] * len(x_band) + [trade.exit_price] * len(x_band),
                fill="toself", fillcolor=fill_col,
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
    return fig


# ── JS live indicator (client-side countdown — zero server cost between ticks) ──

def _js_live_indicator(scan_interval: int, last_scan_str: str) -> None:
    """Inject a client-side JS countdown + flashing dot. No server round-trips."""
    html = f"""
    <div style="display:flex; align-items:center; gap:10px;
                font-family:monospace; font-size:13px; color:#cccccc;
                padding:4px 0;">
        <span id="live-dot" style="font-size:16px;">🔴</span>
        <span>
            LIVE &nbsp;|&nbsp; Last scan: <b>{last_scan_str}</b>
            &nbsp;|&nbsp; Next in: <b><span id="cd">{scan_interval}</span></b>
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
                cd.textContent = t + 's';
                dot.textContent = dots[d % 2];
                d++;
                if (t <= 0) clearInterval(iv);
            }}, 1000);
        }})();
    </script>
    """
    components.html(html, height=36)


# ── Setup Score Card ──────────────────────────────────────────────────────────

def _bar(score: float, max_score: float = 2.0) -> str:
    """ASCII progress bar for a component score (0–max_score → 8 blocks)."""
    filled = int(round(score / max_score * 8))
    return "█" * filled + "░" * (8 - filled)


def _render_setup_card(ticker: str, df: "pd.DataFrame", support: float, resistance: float) -> None:
    """
    Render a collapsible SetupScore card for the currently viewed ticker.
    Builds a minimal WatchlistEntry from the computed S/R levels so the scorer
    doesn't need a watchlist JSON — it works on whatever ticker is in the viewer.
    """
    from side_by_side_backtest.models import WatchlistEntry, SessionType
    from side_by_side_backtest.setup_scorer import score_setup

    if df.empty:
        return

    current_price = float(df["close"].iloc[-1])
    post_ts = df.index[0].to_pydatetime()

    entry_obj = WatchlistEntry(
        post_title=f"{ticker} chart viewer",
        post_timestamp=post_ts,
        ticker=ticker,
        session_type=SessionType.MARKET_OPEN,
        support_level=support,
        resistance_level=resistance,
        stop_level=round(support * 0.98, 4) if support else 0.0,
    )

    with st.expander(f"🎯 Setup Score — {ticker}", expanded=True):
        with st.spinner("Scoring setup…"):
            try:
                sc = score_setup(entry_obj, df)
            except Exception as exc:
                st.warning(f"Scoring error: {exc}")
                return

        # ── Signal banner ─────────────────────────────────────────────────────
        signal_colour = {"🟢 STRONG": "green", "🟡 WATCH": "orange", "🔴 SKIP": "red"}
        colour = signal_colour.get(sc.signal, "gray")
        st.markdown(
            f"<h3 style='color:{colour}; margin:0'>{sc.signal} &nbsp; {sc.score:.1f} / 10</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # ── Key levels ────────────────────────────────────────────────────────
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Current Price", f"${current_price:.3f}")
        col2.metric("Support",       f"${sc.support:.3f}"     if sc.support    else "—")
        col3.metric("Resistance",    f"${sc.resistance:.3f}"  if sc.resistance else "—")
        col4.metric("Stop",          f"${sc.stop:.3f}"        if sc.stop       else "—")
        col5.metric("R/R",           f"{sc.rr_ratio:.1f}:1"   if sc.rr_ratio   else "—")

        col6, col7 = st.columns(2)
        col6.metric("ADX", f"{sc.adx:.1f}")
        col7.metric("Pattern", f"✅ {sc.pattern_type}" if sc.pattern_found else "—")

        if sc.support_broken:
            st.warning("⚠️ **Support broken** — price is below watchlist level. Levels re-anchored to computed S/R.")

        st.markdown("---")

        # ── Score breakdown ───────────────────────────────────────────────────
        st.markdown("**Score breakdown** *(each component 0–2 pts)*")
        score_rows = [
            ("Pattern",      sc.pattern_score),
            ("ADX",          sc.adx_score),
            ("R/R",          sc.rr_score),
            ("Confluence",   sc.confluence_score),
            ("History",      sc.history_score),
            ("Role Rev.",    sc.role_reversal_score),
            ("Rejections",   sc.rejection_score),
            ("Rel. Volume",  sc.rel_vol_score),
            ("MACD Slope",   sc.macd_score),
            ("RSI Diverge",  sc.rsi_div_score),
            ("Regime",       sc.regime_score),
        ]
        for label, val in score_rows:
            st.text(f"  {label:<14} {_bar(val)}  {val:.1f}/2")

        # ── Badges ────────────────────────────────────────────────────────────
        badges = []
        if sc.role_reversal:
            badges.append("🔄 Role Reversal confirmed")
        if sc.rejection_count >= 2:
            badges.append(f"🛡️ {sc.rejection_count}× wick rejections")
        elif sc.rejection_count == 1:
            badges.append("🛡️ 1 wick rejection")
        if badges:
            st.caption("  " + "  |  ".join(badges))


# ── Streamlit app ─────────────────────────────────────────────────────────────

def run() -> None:
    st.set_page_config(
        page_title="Side-by-Side Chart Viewer",
        page_icon="📈",
        layout="wide",
    )

    # Dark CSS override
    st.markdown(
        f"<style>body, .stApp {{ background-color: {_BG}; color: #cccccc; }}</style>",
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("📈 Chart Viewer")
        cache = scan_cache()
        if not cache:
            st.error("No cached OHLCV data found. Run the backtest pipeline first.")
            return

        _tickers = sorted(cache.keys())
        _preselect = st.session_state.get("chart_ticker")
        _default_idx = _tickers.index(_preselect) if _preselect in _tickers else 0
        ticker = st.selectbox("Ticker", _tickers, index=_default_idx, key="cv_ticker")

        # ── Live scanner interval (replaces manual refresh + auto-refresh) ────
        st.divider()
        st.subheader("🔴 Live Scanner")
        _scan_options = {"1 min": 60, "5 min": 300, "15 min": 900, "Off": None}
        _scan_label   = st.selectbox("Auto-rescore interval", list(_scan_options.keys()), index=0)
        _refresh_secs = _scan_options[_scan_label]

        st.divider()
        st.subheader("Trade Parameters")
        pt        = st.slider("Profit Target %",      0.5, 20.0, 2.0,  0.25)
        sl        = st.slider("Stop Loss %",           0.1, 10.0, 1.0,  0.1)
        tolerance = st.slider("Pattern Tolerance %",   0.5, 5.0,  2.0,  0.5) / 100

        st.divider()
        st.subheader("Pattern Filters (Bearish S×S only)")
        adx_thresh    = st.slider("ADX threshold",        10, 40, 20, 1)
        body_mult     = st.slider("C1 body multiplier",   1.0, 3.0, 1.5, 0.1)
        require_gap   = False  # always False for intraday bearish S×S
        st.caption("🐻 Bearish Side-by-Side only — bullish variant disabled")


        st.divider()
        st.subheader("S/R Layers")
        show_pivots  = st.checkbox("Pivot points",     value=True)
        show_daily   = st.checkbox("Daily anchors",    value=True)
        show_extrema = st.checkbox("Local extrema",    value=True)
        show_vprofile= st.checkbox("Volume profile",   value=True)
        show_kmeans  = st.checkbox("K-Means levels",   value=False)

        st.divider()
        st.caption("🟡 diamond = pattern  🔵▲ = entry  🟢● = win  🔴✕ = loss\n"
                   "Color key: 🔵 pivot  🟤 daily  🟦 extrema  🟣 vol-profile  ⬜ kmeans")

    # ── Live-scan fragment ────────────────────────────────────────────────────
    # All chart rendering runs inside this fragment so only the chart re-runs
    # on the interval timer, not the full sidebar.
    @st.fragment(run_every=_refresh_secs)
    def _chart_fragment(
        _ticker, _pt, _sl, _tolerance, _adx_thresh, _body_mult, _require_gap,
        _show_pivots, _show_daily, _show_extrema, _show_vprofile, _show_kmeans,
    ):
        import time as _time
        from side_by_side_backtest.data_fetcher import refresh_today as _refresh_today

        # ── On each live tick: refresh today's bars for the selected ticker ───
        if _refresh_secs:
            _refresh_today(_ticker)
            scan_cache.clear()
            load_parquet.clear()
            load_full_ticker.clear()

        # ── Load full merged ticker history ───────────────────────────────────
        df = load_full_ticker(_ticker)
        if df.empty:
            st.warning(f"No cached data found for {_ticker}.")
            return

        # ── Ensure package root on sys.path ───────────────────────────────────
        import sys as _sys, importlib as _il
        _pkg_root = str(_PKG_DIR.parent)
        if _pkg_root not in _sys.path:
            _sys.path.insert(0, _pkg_root)

        # ── Compute multi-level S/R ───────────────────────────────────────────
        _sr  = _il.import_module("side_by_side_backtest.sr_engine")
        current_price = float(df["close"].iloc[-1])
        sr_levels = _sr.compute_sr_levels(
            df,
            current_price=current_price,
            use_pivots=_show_pivots,
            use_extrema=_show_extrema,
            use_vprofile=_show_vprofile,
            use_kmeans=_show_kmeans,
            use_daily_anchors=_show_daily,
        )

        # For simulator: nearest support below price, nearest resistance above
        support    = sr_levels.nearest_support(current_price)    or round(float(df["low"].quantile(0.15)), 4)
        resistance = sr_levels.nearest_resistance(current_price) or round(float(df["high"].quantile(0.85)), 4)

        # ── Detect patterns ───────────────────────────────────────────────────
        _pe = _il.import_module("side_by_side_backtest.pattern_engine")
        detect_side_by_side = _pe.detect_side_by_side
        df.attrs["ticker"] = _ticker
        patterns = detect_side_by_side(
            df,
            tolerance_pct=_tolerance,
            require_downtrend=True,
            adx_threshold=float(_adx_thresh),
            body_multiplier=float(_body_mult),
            require_gap=_require_gap,
        )

        # ── Simulate entry ────────────────────────────────────────────────────
        _models        = _il.import_module("side_by_side_backtest.models")
        _sim           = _il.import_module("side_by_side_backtest.simulator")
        WatchlistEntry = _models.WatchlistEntry
        SessionType    = _models.SessionType
        simulate_entry = _sim.simulate_entry

        post_ts   = df.index[0].to_pydatetime()
        entry_obj = WatchlistEntry(
            post_title=f"{_ticker} viewer",
            post_timestamp=post_ts,
            ticker=_ticker,
            session_type=SessionType.MARKET_OPEN,
            support_level=support,
            resistance_level=resistance,
            stop_level=round(support * (1 - _sl / 100), 4),
        )
        trade = simulate_entry(entry_obj, df, profit_target_pct=_pt, stop_loss_pct=_sl,
                               pattern_tolerance_pct=_tolerance)

        # ── Build chart with multi-level S/R ──────────────────────────────────
        fig = build_chart(df, ticker=_ticker)

        _colour_map = {
            "pivot_s1": "#4fc3f7", "pivot_s2": "#81d4fa",
            "pivot_r1": "#ef9a9a", "pivot_r2": "#ffcdd2",
            "pivot_p":  "#ffe082",
            "daily_low": "#ffb74d",  "daily_high": "#ff7043",
            "local_low": "#80cbc4",  "local_high": "#4db6ac",
            "poc": "#ce93d8", "vah": "#f48fb1", "val": "#a5d6a7",
            "kmeans": "#b0bec5",
        }

        # ── Consolidate S/R into one Scattergl trace per colour group ─────────
        # Replacing add_hline (SVG layout shapes redrawn on every zoom) with
        # WebGL traces that survive GPU-accelerated pan/zoom without reflow.
        x_start = df.index[0]
        x_end   = df.index[-1]

        # Group levels by colour so we emit the fewest possible traces
        from collections import defaultdict as _dd
        _sr_groups: dict = _dd(lambda: {"x": [], "y": [], "text": []})
        for lv in sr_levels.all_levels:
            colour = _colour_map.get(lv.method, "#888888")
            grp = _sr_groups[colour]
            dash   = "dash" if lv.is_support else "dot"
            width  = min(2.5, 0.8 + lv.strength * 0.25)
            fig.add_hline(
                y=lv.price,
                line=dict(color=colour, width=width, dash=dash),
                annotation_text=f"${lv.price:.3f}",
                annotation_position="right",
                annotation=dict(
                    font=dict(size=9, color=colour),
                    bgcolor="rgba(0,0,0,0)",
                    bordercolor="rgba(0,0,0,0)",
                    showarrow=False,
                    xanchor="left",
                ),
            )

        fig = add_pattern_overlays(fig, patterns, df)
        if trade:
            fig = add_trade_overlays(fig, trade, df)

        # ── Live indicator (client-side countdown, no server cost) ────────────
        if _refresh_secs:
            now_ts = _time.time()
            last_scan_str = _time.strftime('%H:%M:%S', _time.localtime(now_ts))
            _js_live_indicator(_refresh_secs, last_scan_str)

        st.plotly_chart(
            fig,
            width="stretch",
            key=f"main_chart_{_ticker}",  # stable key = Plotly preserves zoom on re-render
            config={
                "scrollZoom": True,
                "displayModeBar": True,
                "modeBarButtonsToAdd": ["toggleSpikelines"],
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "toImageButtonOptions": {"format": "png", "scale": 2},
            },
        )

        # ── Metric card ───────────────────────────────────────────────────────
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Patterns", len(patterns))
        col2.metric("S levels",  len(sr_levels.supports))
        col3.metric("R levels",  len(sr_levels.resistances))
        col4.metric("Nearest S", f"${support:.3f}")
        if trade:
            col5.metric("Outcome", trade.outcome.upper())
            delta_col = "normal" if trade.pnl_pct >= 0 else "inverse"
            col6.metric("PnL", f"{trade.pnl_pct:+.2f}%", delta_color=delta_col)
            tp_price = trade.entry_price * (1 + trade.profit_target_pct / 100)
            sl_price = trade.entry_price * (1 - trade.stop_loss_pct / 100)
            st.info(
                f"**Trade Detail** — {_ticker} | {trade.outcome.upper()}\n\n"
                f"Entry: **${trade.entry_price:.3f}** @ `{trade.entry_ts}`  \n"
                f"🟢 TP: **${tp_price:.3f}** (+{trade.profit_target_pct:.1f}%)  \n"
                f"🔴 SL: **${sl_price:.3f}** (−{trade.stop_loss_pct:.1f}%)  \n"
                f"Exit: **${trade.exit_price:.3f}** @ `{trade.exit_ts}`  \n"
                f"Hold: {trade.hold_bars} bars (~{trade.hold_bars * 5} min)  \n"
                f"PnL: **{trade.pnl_pct:+.2f}%**  |  Support respected: {'✅' if trade.support_respected else '❌'}"
            )
        else:
            col5.metric("Outcome", "No entry")
            col6.metric("PnL", "—")
            st.warning(
                "⚠️ **No trade entry found** for this ticker with the current parameters.\n\n"
                "Try: lower ADX threshold · raise pattern tolerance · lower stop loss %"
            )

        # ── Setup Score Card ──────────────────────────────────────────────────
        _render_setup_card(_ticker, df, support, resistance)

        with st.expander("📊 Chart Legend"):
            st.markdown("""
| Color | Line style | Meaning |
|-------|-----------|---------|
| 🔵 Cyan dashed | Pivot support S1/S2 |
| 🔴 Pink dotted | Pivot resistance R1/R2 |
| 🟠 Orange dashed | Daily session low |
| 🟥 Red dotted | Daily session high |
| 🟡 Yellow diamond | Side-by-Side pattern |
| 🔵 Cyan triangle | Trade entry |
| 🟢 Green circle | Winning exit |
| 🔴 Red ✕ | Losing exit |
""")

        with st.expander("Raw bar data"):
            st.dataframe(df.tail(50), width='stretch')

    # ── Invoke the fragment (auto-refreshes on interval) ──────────────────────
    _chart_fragment(
        ticker, pt, sl, tolerance, adx_thresh, body_mult, require_gap,
        show_pivots, show_daily, show_extrema, show_vprofile, show_kmeans,
    )


# Only auto-run when Streamlit executes this file directly (not when imported as a module)
if __name__ != "side_by_side_backtest.chart_viewer":
    run()
