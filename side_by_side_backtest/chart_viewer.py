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
_EQH_LINE    = "#ffd700"          # gold dashed — EQH ceiling level
_EQH_MK      = "#ff9800"          # orange square — EQH pair C3 bar
_EQH_BREAK   = "#00e676"          # bright green up-arrow — EQH breakout
_EQH_REJECT  = "#ff1744"          # bright red down-arrow — EQH rejection


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
    visible_days: int = 1,            # initial x-window; y-range computed from this window
) -> go.Figure:
    """
    Build a Plotly candlestick figure matching the TradingView dark theme.
    Session-background shading is added automatically.
    S/R lines are drawn separately by the caller via sr_engine levels.
    The x-axis is displayed in *tz* (default: America/Los_Angeles = PT).

    Y-axis is initialised to a tight range over the last *visible_days* of data
    (autorange=False, fixedrange=False so users can still drag).
    Call inject_yaxis_rescale_js(df, ticker) AFTER st.plotly_chart() to attach
    a client-side JS listener that rescales y to visible bars on every x change.
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

    # ── Tight initial y-range: computed from visible_days window only ─────────
    _x_end        = df.index[-1]
    _x_start      = _x_end - pd.Timedelta(days=visible_days)
    _visible_mask = df.index >= _x_start
    _view         = df[_visible_mask] if _visible_mask.any() else df
    _y_lo         = float(_view["low"].min())
    _y_hi         = float(_view["high"].max())
    _y_padding    = max((_y_hi - _y_lo) * 0.05, abs(_y_lo) * 0.005)
    _y_range      = [_y_lo - _y_padding, _y_hi + _y_padding]

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
            autorange=False,
            fixedrange=False,
            range=_y_range,
            showspikes=True, spikemode="across",
            spikesnap="cursor", spikecolor="#555555",
            spikethickness=1, spikedash="dot",
        ),
        margin=dict(l=10, r=80, t=60, b=10),
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


def inject_yaxis_rescale_js(df: pd.DataFrame, ticker: str, tz: str = "America/Los_Angeles") -> None:
    """
    Inject a client-side JS snippet that listens for Plotly x-range changes
    (range-selector buttons, pan, zoom) and rescales the y-axis to fit only
    the visible candlestick bars — exactly like TradingView's auto-fit.

    Must be called AFTER st.plotly_chart() so the Plotly div exists in the DOM.
    The OHLC data is embedded as a compact JSON array (ms timestamps) so the
    rescale runs entirely in the browser with zero server round-trips.
    """
    import json as _json

    # Convert to display tz (same as build_chart) so timestamps match the chart
    df_disp = df.copy()
    if df_disp.index.tzinfo is not None:
        df_disp.index = df_disp.index.tz_convert(tz)
    else:
        df_disp.index = df_disp.index.tz_localize("UTC").tz_convert(tz)

    # Compact OHLC array: [[ts_ms, low, high], ...]  — only what JS needs
    ohlc_js = _json.dumps([
        [int(ts.timestamp() * 1000), round(float(lo), 6), round(float(hi), 6)]
        for ts, lo, hi in zip(df_disp.index, df_disp["low"], df_disp["high"])
    ])

    PAD = 0.04   # 4% padding above/below the visible range

    js = f"""
<script>
(function() {{
    const OHLC  = {ohlc_js};   // [[ts_ms, low, high], ...]
    const PAD   = {PAD};
    const divId = "main_chart_{ticker}";

    function rescaleY(xStart, xEnd) {{
        const t0 = new Date(xStart).getTime();
        const t1 = new Date(xEnd).getTime();
        let lo = Infinity, hi = -Infinity;
        for (const [ts, l, h] of OHLC) {{
            if (ts >= t0 && ts <= t1) {{
                if (l < lo) lo = l;
                if (h > hi) hi = h;
            }}
        }}
        if (!isFinite(lo) || !isFinite(hi)) return;
        const pad = Math.max((hi - lo) * PAD, lo * 0.002);
        const div = document.getElementById(divId);
        if (!div) return;
        Plotly.relayout(div, {{'yaxis.range': [lo - pad, hi + pad], 'yaxis.autorange': false}});
    }}

    function attachListener() {{
        const div = document.getElementById(divId);
        if (!div || !div._fullLayout) {{
            setTimeout(attachListener, 300);
            return;
        }}
        div.on('plotly_relayout', function(evt) {{
            const x0 = evt['xaxis.range[0]'] || (evt['xaxis.range'] && evt['xaxis.range'][0]);
            const x1 = evt['xaxis.range[1]'] || (evt['xaxis.range'] && evt['xaxis.range'][1]);
            if (x0 && x1) rescaleY(x0, x1);
        }});
        // Also fire once on load with the current x-range
        const layout = div._fullLayout;
        if (layout && layout.xaxis && layout.xaxis.range) {{
            rescaleY(layout.xaxis.range[0], layout.xaxis.range[1]);
        }}
    }}
    attachListener();
}})();
</script>
"""
    components.html(js, height=0)


def _add_session_shading(fig: go.Figure, df: pd.DataFrame) -> None:
    """
    Add pre-market and after-hours background shading using layout shapes.

    Using add_shape with yref="paper" means the rectangles span the full
    chart height (0→1 in paper coordinates) regardless of price scale, and
    — critically — they are EXCLUDED from Plotly's autorange calculation so
    they can never force the y-axis to show a wider price range than the
    visible candles.
    """
    if df.empty:
        return

    half_bar = pd.Timedelta("2.5min")
    labels   = [_session_label(ts) for ts in df.index]

    _session_colours = {"pre": _PREMARKET, "after": _AFTERHOURS}

    i = 0
    while i < len(labels):
        sess = labels[i]
        if sess == "regular":
            i += 1
            continue
        # find end of this contiguous run
        j = i
        while j < len(labels) and labels[j] == sess:
            j += 1

        x0 = (df.index[i]     - half_bar).isoformat()
        x1 = (df.index[j - 1] + half_bar).isoformat()
        fig.add_shape(
            type="rect",
            xref="x", yref="paper",   # paper y → always full chart height
            x0=x0, x1=x1,
            y0=0,   y1=1,
            fillcolor=_session_colours[sess],
            line=dict(width=0),
            layer="below",
        )
        i = j


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
            ys.append(float(df_disp["high"].iloc[p.bar_index]))  # no offset — avoids autorange expansion
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


# ── Overlay: EQH (Equal Highs / Liquidity Ceiling) ────────────────────────────

def add_eqh_overlays(
    fig: go.Figure,
    eqh_pairs: list,
    eqh_signals: list,
    df: pd.DataFrame,
    display_tz: str = "America/Los_Angeles",
) -> go.Figure:
    """
    Add EQH visual overlays to the chart:
      • Gold dashed horizontal line at each EQH ceiling level
      • Orange square marker on each EQH pair C3 bar
      • Bright green up-arrow on eqh_breakout signal bars
      • Bright red down-arrow on eqh_rejection signal bars

    Parameters
    ----------
    eqh_pairs   : List of PatternMatch with pattern_type="eqh_pair"
    eqh_signals : List of PatternMatch with pattern_type="eqh_breakout"|"eqh_rejection"
    df          : OHLCV DataFrame (UTC index, used for timestamp matching)
    display_tz  : Timezone for x-axis display (must match build_chart tz arg)
    """
    if not eqh_pairs and not eqh_signals:
        return fig

    df_disp = df.copy()
    if df_disp.index.tzinfo is not None:
        df_disp.index = df_disp.index.tz_convert(display_tz)
    else:
        df_disp.index = df_disp.index.tz_localize("UTC").tz_convert(display_tz)

    # ── EQH pair markers (orange squares) and ceiling lines ──────────────────
    pair_xs, pair_ys, pair_texts = [], [], []
    seen_levels: set[float] = set()

    for p in eqh_pairs:
        if p.bar_index >= len(df_disp):
            continue
        bar_ts = df_disp.index[p.bar_index]
        bar_hi = float(df_disp["high"].iloc[p.bar_index])
        pair_xs.append(bar_ts)
        pair_ys.append(bar_hi)
        pair_texts.append(
            f"EQH Pair @ ${p.eqh_level:.3f}<br>"
            f"C2 {p.candle2_open:.3f}→{p.candle2_close:.3f}<br>"
            f"C3 {p.candle3_open:.3f}→{p.candle3_close:.3f}"
        )

        # Draw ceiling line (deduplicated by level)
        level_key = round(p.eqh_level, 3)
        if level_key > 0 and level_key not in seen_levels:
            seen_levels.add(level_key)
            fig.add_shape(
                type="line",
                xref="paper", yref="y",
                x0=0, x1=1,
                y0=p.eqh_level, y1=p.eqh_level,
                line=dict(color=_EQH_LINE, width=1.5, dash="dashdot"),
                layer="above",
            )
            fig.add_annotation(
                xref="paper", yref="y",
                x=0.01, y=p.eqh_level,
                text=f"🏛️ EQH ${p.eqh_level:.3f}",
                showarrow=False,
                font=dict(size=9, color=_EQH_LINE),
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="rgba(0,0,0,0)",
                xanchor="left",
            )

    if pair_xs:
        fig.add_trace(go.Scattergl(
            x=pair_xs, y=pair_ys, mode="markers",
            marker=dict(symbol="square", size=10, color=_EQH_MK,
                        line=dict(color="#000000", width=1)),
            name="EQH Pair", hovertext=pair_texts, hoverinfo="text",
            showlegend=True,
        ))

    # ── EQH signal markers (breakout / rejection) ────────────────────────────
    break_xs, break_ys, break_texts = [], [], []
    reject_xs, reject_ys, reject_texts = [], [], []

    for s in eqh_signals:
        if s.bar_index >= len(df_disp):
            continue
        bar_ts = df_disp.index[s.bar_index]
        if s.pattern_type == "eqh_breakout":
            bar_y  = float(df_disp["high"].iloc[s.bar_index])
            break_xs.append(bar_ts)
            break_ys.append(bar_y)
            break_texts.append(f"🟢 EQH Breakout above ${s.eqh_level:.3f}")
        elif s.pattern_type == "eqh_rejection":
            bar_y  = float(df_disp["low"].iloc[s.bar_index])
            reject_xs.append(bar_ts)
            reject_ys.append(bar_y)
            reject_texts.append(f"🔴 EQH Rejection at ${s.eqh_level:.3f}")

    if break_xs:
        fig.add_trace(go.Scattergl(
            x=break_xs, y=break_ys, mode="markers",
            marker=dict(symbol="triangle-up", size=14, color=_EQH_BREAK,
                        line=dict(color="#ffffff", width=1)),
            name="EQH Breakout", hovertext=break_texts, hoverinfo="text",
            showlegend=True,
        ))

    if reject_xs:
        fig.add_trace(go.Scattergl(
            x=reject_xs, y=reject_ys, mode="markers",
            marker=dict(symbol="triangle-down", size=14, color=_EQH_REJECT,
                        line=dict(color="#ffffff", width=1)),
            name="EQH Rejection", hovertext=reject_texts, hoverinfo="text",
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

        # TP / SL / entry lines as layout shapes (excluded from autorange)
        x0_ts = df.index[entry_idx].isoformat()
        x1_ts = df.index[-1].isoformat()
        if x0_ts:
            fig.add_shape(type="line", xref="x", yref="y",
                x0=x0_ts, x1=x1_ts, y0=tp, y1=tp,
                line=dict(color=_BULL, width=1.5, dash="dot"), layer="above")
            fig.add_shape(type="line", xref="x", yref="y",
                x0=x0_ts, x1=x1_ts, y0=sl, y1=sl,
                line=dict(color=_BEAR, width=1.5, dash="dot"), layer="above")
            fig.add_shape(type="line", xref="x", yref="y",
                x0=x0_ts, x1=x1_ts, y0=trade.entry_price, y1=trade.entry_price,
                line=dict(color=_ENTRY_MK, width=1, dash="dash"), layer="above",
                opacity=0.6)
            # Annotations use yref="y" — layout annotations do NOT affect autorange
            fig.add_annotation(xref="paper", yref="y", x=1.01, y=tp,
                text=f"TP ${tp:.3f}", showarrow=False,
                font=dict(size=9, color=_BULL), xanchor="left")
            fig.add_annotation(xref="paper", yref="y", x=1.01, y=sl,
                text=f"SL ${sl:.3f}", showarrow=False,
                font=dict(size=9, color=_BEAR), xanchor="left")

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

    # ── Shaded band between entry and exit (layout shape, excluded from autorange) ──
    if entry_idx is not None and exit_idx is not None and trade.exit_price:
        fill_col = "rgba(38,166,154,0.12)" if trade.outcome == "win" else \
                   "rgba(239,83,80,0.12)"  if trade.outcome == "loss" else \
                   "rgba(120,120,120,0.10)"
        fig.add_shape(
            type="rect",
            xref="x", yref="y",
            x0=df.index[entry_idx].isoformat(),
            x1=df.index[exit_idx].isoformat(),
            y0=min(trade.entry_price, trade.exit_price),
            y1=max(trade.entry_price, trade.exit_price),
            fillcolor=fill_col,
            line=dict(width=0),
            layer="below",
        )
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
        st.subheader("🏛️ EQH Levels")
        show_eqh = st.checkbox("Show EQH pairs", value=True,
                               help="Detect mixed-color equal-open pairs (trader's primary resistance target)")

        st.divider()
        st.caption("🟡 diamond = S×S pattern  🟠 square = EQH pair  🟢▲ = EQH breakout  🔴▼ = EQH rejection\n"
                   "🔵▲ = entry  🟢● = win  🔴✕ = loss\n"
                   "Color key: 🔵 pivot  🟤 daily  🟦 extrema  🟣 vol-profile  ⬜ kmeans  🟡 EQH ceiling")

    # ── Live-scan fragment ────────────────────────────────────────────────────
    # All chart rendering runs inside this fragment so only the chart re-runs
    # on the interval timer, not the full sidebar.
    @st.fragment(run_every=_refresh_secs)
    def _chart_fragment(
        _ticker, _pt, _sl, _tolerance, _adx_thresh, _body_mult, _require_gap,
        _show_pivots, _show_daily, _show_extrema, _show_vprofile, _show_kmeans,
        _show_eqh,
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

        # ── Direct imports — more reliable than importlib under Streamlit hot-reload ──
        # importlib.import_module() fails when sys.modules is in a partial-reload state
        # (Streamlit fragment re-executes mid-reload → KeyError on the module name).
        import sys as _sys
        _pkg_root = str(_PKG_DIR.parent)
        if _pkg_root not in _sys.path:
            _sys.path.insert(0, _pkg_root)

        from side_by_side_backtest.sr_engine import compute_sr_levels as _compute_sr_levels
        from side_by_side_backtest.pattern_engine import (
            detect_side_by_side as _detect_side_by_side,
            detect_equal_highs_pair as _detect_eqh_fn,
            detect_eqh_signal as _detect_eqh_sig,
        )

        # ── Compute multi-level S/R ───────────────────────────────────────────
        current_price = float(df["close"].iloc[-1])
        sr_levels = _compute_sr_levels(
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

        # ── Filter S/R levels to reasonable range around current price ────────
        # Prevents S/R lines far outside visible price range from distorting the chart
        _sr_band = 0.50  # only show S/R within 50% of current price
        _sr_lo = current_price * (1 - _sr_band)
        _sr_hi = current_price * (1 + _sr_band)
        filtered_sr_levels = [lv for lv in sr_levels.all_levels if _sr_lo <= lv.price <= _sr_hi]

        # ── Detect patterns ───────────────────────────────────────────────────
        df.attrs["ticker"] = _ticker
        patterns = _detect_side_by_side(
            df,
            tolerance_pct=_tolerance,
            require_downtrend=True,
            adx_threshold=float(_adx_thresh),
            body_multiplier=float(_body_mult),
            require_gap=_require_gap,
        )

        # ── Detect EQH pairs (mixed-color equal-open resistance levels) ───────
        eqh_pairs: list = []
        eqh_signals: list = []
        if _show_eqh:
            eqh_pairs   = _detect_eqh_fn(df)
            eqh_signals = _detect_eqh_sig(df, eqh_pairs)

        # ── Simulate entry ────────────────────────────────────────────────────
        # Use direct imports (not importlib) for simulator + models — these modules
        # contain dataclasses/Pydantic models and are safe to import directly.
        # importlib.import_module() for these fails under Streamlit hot-reload because
        # the fragment re-executes while sys.modules is in a partial-reload state.
        from side_by_side_backtest.models import WatchlistEntry, SessionType
        from side_by_side_backtest.simulator import simulate_entry

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
        fig = build_chart(df, ticker=_ticker, visible_days=1)

        _colour_map = {
            "pivot_s1": "#4fc3f7", "pivot_s2": "#81d4fa",
            "pivot_r1": "#ef9a9a", "pivot_r2": "#ffcdd2",
            "pivot_p":  "#ffe082",
            "daily_low": "#ffb74d",  "daily_high": "#ff7043",
            "local_low": "#80cbc4",  "local_high": "#4db6ac",
            "poc": "#ce93d8", "vah": "#f48fb1", "val": "#a5d6a7",
            "kmeans": "#b0bec5",
        }

        # ── Draw S/R lines as layout shapes (xref="paper", yref="y") ──────────
        # Only draw filtered levels within reasonable range of current price.
        # xref="paper" means the line spans the full chart width regardless of
        # the current x-range, matching add_hline behaviour visually.
        for lv in filtered_sr_levels:
            colour = _colour_map.get(lv.method, "#888888")
            dash   = "dash" if lv.is_support else "dot"
            width  = min(2.5, 0.8 + lv.strength * 0.25)
            fig.add_shape(
                type="line",
                xref="paper", yref="y",
                x0=0, x1=1,
                y0=lv.price, y1=lv.price,
                line=dict(color=colour, width=width, dash=dash),
                layer="above",
            )
            fig.add_annotation(
                xref="paper", yref="y",
                x=1.01, y=lv.price,
                text=f"${lv.price:.3f}",
                showarrow=False,
                font=dict(size=9, color=colour),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                xanchor="left",
            )

        fig = add_pattern_overlays(fig, patterns, df)
        if _show_eqh:
            fig = add_eqh_overlays(fig, eqh_pairs, eqh_signals, df)
        if trade:
            fig = add_trade_overlays(fig, trade, df)

        # ── Live indicator (client-side countdown, no server cost) ────────────
        if _refresh_secs:
            now_ts = _time.time()
            last_scan_str = _time.strftime('%H:%M:%S', _time.localtime(now_ts))
            _js_live_indicator(_refresh_secs, last_scan_str)

        st.plotly_chart(
            fig,
            width='stretch',
            key=f"main_chart_{_ticker}",  # stable key = Plotly preserves zoom on re-render
            config={
                "scrollZoom": True,
                "displayModeBar": True,
                "modeBarButtonsToAdd": ["toggleSpikelines"],
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "toImageButtonOptions": {"format": "png", "scale": 2},
            },
        )

        # ── JS y-axis rescale: fires on every x-range change client-side ─────
        inject_yaxis_rescale_js(df, _ticker)

        # ── Metric card ───────────────────────────────────────────────────────
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("S×S Patterns", len(patterns))
        col2.metric("EQH Pairs", len(eqh_pairs))
        col3.metric("Nearest S", f"${support:.3f}")
        col4.metric("S levels",  len(filtered_sr_levels))
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
        show_eqh,
    )


# Only auto-run when Streamlit executes this file directly (not when imported as a module)
if __name__ != "side_by_side_backtest.chart_viewer":
    run()
