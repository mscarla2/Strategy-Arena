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
@st.cache_data(show_spinner=False)
def scan_cache() -> Dict[str, List[str]]:
    """
    Scan ohlcv_cache/ and return {ticker: [parquet_stem, ...]} sorted by date.
    """
    result: Dict[str, List[str]] = {}
    for pf in sorted(_CACHE_DIR.glob("*_5m.parquet")):
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


@st.cache_data(show_spinner="Loading full history…")
def load_full_ticker(ticker: str) -> pd.DataFrame:
    """
    Merge ALL cached parquet files for a ticker into one continuous DataFrame.
    Deduplicates overlapping windows and sorts by timestamp.
    """
    stems = scan_cache().get(ticker, [])
    frames = [load_parquet(s) for s in stems]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames)
    merged = merged[~merged.index.duplicated(keep="first")].sort_index()
    return merged


# ── Chart builder ─────────────────────────────────────────────────────────────

def build_chart(
    df: pd.DataFrame,
    ticker: str = "",
) -> go.Figure:
    """
    Build a Plotly candlestick figure matching the TradingView dark theme.
    Session-background shading is added automatically.
    S/R lines are drawn separately by the caller via sr_engine levels.
    """
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
    ))

    # ── Session background shading ────────────────────────────────────────────
    _add_session_shading(fig, df)

    # ── Dark theme layout + range-selector buttons ────────────────────────────
    fig.update_layout(
        plot_bgcolor=_BG, paper_bgcolor=_BG,
        font=dict(color="#cccccc", size=11),
        xaxis=dict(
            gridcolor="#1e2530", showgrid=True, zeroline=False,
            rangeslider=dict(visible=False),
            type="date",   # date axis so range selector buttons work
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
        ),
        yaxis=dict(gridcolor="#1e2530", showgrid=True, zeroline=False, side="right"),
        margin=dict(l=10, r=60, t=60, b=10),
        height=580,
        title=dict(text=ticker, font=dict(size=14, color="#eeeeee"), x=0.02),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1e2530", font_color="#cccccc"),
    )
    return fig


def _add_session_shading(fig: go.Figure, df: pd.DataFrame) -> None:
    """Add coloured background rectangles for pre-market and after-hours."""
    if df.empty:
        return
    # Group consecutive bars by session type and add vrect for each run
    labels = [_session_label(ts) for ts in df.index]
    i = 0
    while i < len(labels):
        sess = labels[i]
        if sess == "regular":
            i += 1
            continue
        j = i
        while j < len(labels) and labels[j] == sess:
            j += 1
        colour = _PREMARKET if sess == "pre" else _AFTERHOURS
        fig.add_vrect(
            x0=df.index[i], x1=df.index[j - 1],
            fillcolor=colour, layer="below", line_width=0,
        )
        i = j


# ── Overlay: pattern markers ──────────────────────────────────────────────────

def add_pattern_overlays(fig: go.Figure, patterns: list, df: pd.DataFrame) -> go.Figure:
    """Add yellow diamond markers at each detected Side-by-Side pattern bar."""
    if not patterns:
        return fig
    xs, ys, texts = [], [], []
    for p in patterns:
        if p.bar_index < len(df):
            xs.append(df.index[p.bar_index])
            ys.append(float(df["high"].iloc[p.bar_index]) * 1.005)
            texts.append(
                f"Side-by-Side<br>"
                f"C1 {p.candle1_open:.3f}→{p.candle1_close:.3f}<br>"
                f"C2 {p.candle2_open:.3f}→{p.candle2_close:.3f}"
            )
    fig.add_trace(go.Scatter(
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
        fig.add_trace(go.Scatter(
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
            fig.add_trace(go.Scatter(
                x=x_rest, y=[tp] * len(x_rest),
                mode="lines", line=dict(color=_BULL, width=1.5, dash="dot"),
                name=f"TP {tp:.3f}", showlegend=True,
                hovertemplate=f"TP: {tp:.3f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=x_rest, y=[sl] * len(x_rest),
                mode="lines", line=dict(color=_BEAR, width=1.5, dash="dot"),
                name=f"SL {sl:.3f}", showlegend=True,
                hovertemplate=f"SL: {sl:.3f}<extra></extra>",
            ))
            # Entry price reference line
            fig.add_trace(go.Scatter(
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
        fig.add_trace(go.Scatter(
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
            fig.add_trace(go.Scatter(
                x=x_band + x_band[::-1],
                y=[trade.entry_price] * len(x_band) + [trade.exit_price] * len(x_band),
                fill="toself", fillcolor=fill_col,
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
    return fig


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
        _preselect = st.session_state.pop("chart_ticker", None)
        _default_idx = _tickers.index(_preselect) if _preselect in _tickers else 0
        ticker = st.selectbox("Ticker", _tickers, index=_default_idx, key="ticker")
        n_windows = len(cache.get(ticker, []))
        st.caption(f"{n_windows} date window(s) available — showing full merged history")

        st.divider()
        st.subheader("Trade Parameters")
        pt        = st.slider("Profit Target %",      0.5, 20.0, 2.0,  0.25)
        sl        = st.slider("Stop Loss %",           0.1, 10.0, 1.0,  0.1)
        tolerance = st.slider("Pattern Tolerance %",   0.5, 5.0,  2.0,  0.5) / 100

        st.divider()
        st.subheader("Pattern Filters")
        adx_thresh    = st.slider("ADX threshold",        10, 40, 20, 1)
        body_mult     = st.slider("C1 body multiplier",   1.0, 3.0, 1.5, 0.1)
        require_gap   = st.checkbox("Require gap (overnight only)", value=False)

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

    # ── Load full merged ticker history ───────────────────────────────────────
    df = load_full_ticker(ticker)
    if df.empty:
        st.warning(f"No cached data found for {ticker}. Run the backtest pipeline first.")
        return

    # ── Ensure package root on sys.path ───────────────────────────────────────
    import sys as _sys, importlib as _il
    _pkg_root = str(_PKG_DIR.parent)
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)

    # ── Compute multi-level S/R ───────────────────────────────────────────────
    _sr  = _il.import_module("side_by_side_backtest.sr_engine")
    current_price = float(df["close"].iloc[-1])
    sr_levels = _sr.compute_sr_levels(
        df,
        current_price=current_price,
        use_pivots=show_pivots,
        use_extrema=show_extrema,
        use_vprofile=show_vprofile,
        use_kmeans=show_kmeans,
        use_daily_anchors=show_daily,
    )

    # For simulator: nearest support below price, nearest resistance above
    support    = sr_levels.nearest_support(current_price)    or round(float(df["low"].quantile(0.15)), 4)
    resistance = sr_levels.nearest_resistance(current_price) or round(float(df["high"].quantile(0.85)), 4)

    # ── Detect patterns ───────────────────────────────────────────────────────
    _pe = _il.import_module("side_by_side_backtest.pattern_engine")
    detect_side_by_side = _pe.detect_side_by_side
    df.attrs["ticker"] = ticker
    patterns = detect_side_by_side(
        df,
        tolerance_pct=tolerance,
        require_downtrend=True,
        adx_threshold=float(adx_thresh),
        body_multiplier=float(body_mult),
        require_gap=require_gap,
    )

    # ── Simulate entry ────────────────────────────────────────────────────────
    _models        = _il.import_module("side_by_side_backtest.models")
    _sim           = _il.import_module("side_by_side_backtest.simulator")
    WatchlistEntry = _models.WatchlistEntry
    SessionType    = _models.SessionType
    simulate_entry = _sim.simulate_entry

    post_ts   = df.index[0].to_pydatetime()
    entry_obj = WatchlistEntry(
        post_title=f"{ticker} viewer",
        post_timestamp=post_ts,
        ticker=ticker,
        session_type=SessionType.MARKET_OPEN,
        support_level=support,
        resistance_level=resistance,
        stop_level=round(support * 0.95, 4),
    )
    trade = simulate_entry(entry_obj, df, profit_target_pct=pt, stop_loss_pct=sl,
                           pattern_tolerance_pct=tolerance)

    # ── Build chart with multi-level S/R ──────────────────────────────────────
    fig = build_chart(df, ticker=ticker)          # no single lines in base chart

    # Draw all S/R levels — no annotation text, color only
    _colour_map = {
        "pivot_s1": "#4fc3f7", "pivot_s2": "#81d4fa",
        "pivot_r1": "#ef9a9a", "pivot_r2": "#ffcdd2",
        "pivot_p":  "#ffe082",
        "daily_low": "#ffb74d",  "daily_high": "#ff7043",
        "local_low": "#80cbc4",  "local_high": "#4db6ac",
        "poc": "#ce93d8", "vah": "#f48fb1", "val": "#a5d6a7",
        "kmeans": "#b0bec5",
    }
    for lv in sr_levels.all_levels:
        colour = _colour_map.get(lv.method, "#888888")
        dash   = "dash" if lv.is_support else "dot"
        width  = min(2.5, 0.8 + lv.strength * 0.25)
        # No annotation_text — color coordination only
        fig.add_hline(
            y=lv.price,
            line=dict(color=colour, width=width, dash=dash),
        )

    fig = add_pattern_overlays(fig, patterns, df)
    if trade:
        fig = add_trade_overlays(fig, trade, df)

    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})

    # ── Metric card ───────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Patterns", len(patterns))
    col2.metric("S levels",  len(sr_levels.supports))
    col3.metric("R levels",  len(sr_levels.resistances))
    col4.metric("Nearest S", f"${support:.3f}")
    if trade:
        col5.metric("Outcome", trade.outcome.upper())
        delta_col = "normal" if trade.pnl_pct >= 0 else "inverse"
        col6.metric("PnL", f"{trade.pnl_pct:+.2f}%", delta_color=delta_col)
        # Trade detail box
        tp_price = trade.entry_price * (1 + trade.profit_target_pct / 100)
        sl_price = trade.entry_price * (1 - trade.stop_loss_pct / 100)
        st.info(
            f"**Trade Detail** — {ticker} | {trade.outcome.upper()}\n\n"
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

    # ── Color legend ──────────────────────────────────────────────────────────
    with st.expander("📊 Chart Legend"):
        st.markdown("""
| Color | Line style | Meaning |
|-------|-----------|---------|
| 🔵 Cyan `#4fc3f7` | dashed | Pivot support S1/S2 |
| 🔴 Pink `#ef9a9a` | dotted | Pivot resistance R1/R2 |
| 🟡 Yellow `#ffe082` | dotted | Pivot Point (P) |
| 🟠 Orange `#ffb74d` | dashed | Daily session low (anchor support) |
| 🟥 Red `#ff7043` | dotted | Daily session high (anchor resistance) |
| 🩵 Teal `#80cbc4` | dashed | Local fractal low (support cluster) |
| 🌊 Teal `#4db6ac` | dotted | Local fractal high (resistance cluster) |
| 🟣 Purple `#ce93d8` | dotted | Volume POC (Point of Control) |
| 🩷 Pink `#f48fb1` | dotted | Volume Area High (VAH) |
| 🟢 Green `#a5d6a7` | dashed | Volume Area Low (VAL) |
| ⬜ Grey `#b0bec5` | dotted | K-Means cluster |
| 🔵 Cyan `▲` | marker | Trade entry |
| 🟢 Green `●` | marker | Winning exit |
| 🔴 Red `✕` | marker | Losing exit |
| ⚫ Grey `✕` | marker | Timeout exit |
| 🟡 Yellow `◆` | marker | Side-by-Side White Lines pattern |
| 🟢 Cyan `─ ─` | line | Take-Profit level |
| 🔴 Red `─ ─` | line | Stop-Loss level |
""")

    with st.expander("Raw bar data"):
        st.dataframe(df.tail(50), width="stretch")


# Streamlit auto-calls the script; call run() at module level
run()
