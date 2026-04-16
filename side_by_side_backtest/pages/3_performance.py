"""
Performance Analytics — Backtest Results Dashboard
====================================================
Shows per-ticker and overall stats from the watchlist_backtest.db trade history.
Sections:
  1. Overall summary metrics
  2. Best performing plays (ranked by win-rate + expectancy)
  3. Per-ticker detail table (sortable)
  4. Equity curve
  5. PnL distribution histogram

Launch via:
    streamlit run side_by_side_backtest/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

_PKG = Path(__file__).parent.parent
if str(_PKG.parent) not in sys.path:
    sys.path.insert(0, str(_PKG.parent))

from side_by_side_backtest.db import WatchlistDB
from side_by_side_backtest.models import TradeResult


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading trade history…", ttl=60)
def _load_trades() -> pd.DataFrame:
    """Load all trades from DB into a DataFrame."""
    try:
        with WatchlistDB() as db:
            trades: List[TradeResult] = db.load_trades()
    except Exception as exc:
        st.error(f"Could not load trade DB: {exc}")
        return pd.DataFrame()

    if not trades:
        return pd.DataFrame()

    rows = [{
        "ticker":      t.ticker,
        "entry_ts":    t.entry_ts,
        "entry_price": t.entry_price,
        "exit_price":  t.exit_price or t.entry_price,
        "pnl_pct":     t.pnl_pct,
        "outcome":     t.outcome,
        "hold_bars":   t.hold_bars,
        "session":     t.session_type.value,
        "pt_pct":      t.profit_target_pct,
        "sl_pct":      t.stop_loss_pct,
    } for t in trades]

    df = pd.DataFrame(rows)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"])
    df.sort_values("entry_ts", inplace=True)
    return df


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _per_ticker_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-ticker summary statistics."""
    rows = []
    for ticker, grp in df.groupby("ticker"):
        total   = len(grp)
        wins    = (grp["outcome"] == "win").sum()
        losses  = (grp["outcome"] == "loss").sum()
        wr      = wins / total if total else 0
        avg_win  = grp.loc[grp["outcome"] == "win",  "pnl_pct"].mean() or 0
        avg_loss = grp.loc[grp["outcome"] == "loss", "pnl_pct"].mean() or 0
        expectancy = wr * avg_win + (1 - wr) * avg_loss
        total_pnl  = grp["pnl_pct"].sum()
        avg_hold   = grp["hold_bars"].mean()
        rows.append({
            "Ticker":      ticker,
            "Trades":      total,
            "Wins":        int(wins),
            "Losses":      int(losses),
            "Win Rate":    round(wr, 3),
            "Avg Win %":   round(avg_win, 2),
            "Avg Loss %":  round(avg_loss, 2),
            "Expectancy":  round(expectancy, 2),
            "Total PnL %": round(total_pnl, 2),
            "Avg Hold (bars)": round(avg_hold, 1),
        })
    stats = pd.DataFrame(rows)
    if not stats.empty:
        stats.sort_values("Expectancy", ascending=False, inplace=True)
    return stats


def _overall_metrics(df: pd.DataFrame) -> dict:
    """Compute overall portfolio-level metrics."""
    total   = len(df)
    wins    = (df["outcome"] == "win").sum()
    losses  = (df["outcome"] == "loss").sum()
    wr      = wins / total if total else 0
    avg_win  = df.loc[df["outcome"] == "win",  "pnl_pct"].mean() or 0
    avg_loss = df.loc[df["outcome"] == "loss", "pnl_pct"].mean() or 0
    pf = abs(avg_win * wins / (avg_loss * losses)) if losses and avg_loss else float("inf")
    return {
        "total": total, "wins": int(wins), "losses": int(losses),
        "wr": wr, "avg_win": avg_win, "avg_loss": avg_loss,
        "expectancy": wr * avg_win + (1 - wr) * avg_loss,
        "profit_factor": round(pf, 2),
        "total_pnl": df["pnl_pct"].sum(),
    }


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

_BENCHMARK_TICKER = "QQQ"

@st.cache_data(show_spinner=f"Fetching {_BENCHMARK_TICKER} benchmark…", ttl=3600)
def _fetch_benchmark_series(start_date: str, end_date: str) -> Optional[pd.Series]:
    """
    Download QQQ daily closes and return a cumulative-return % Series
    indexed by date, normalised so the first trading day = 0 %.
    Returns None if data is unavailable.
    """
    try:
        import yfinance as yf  # type: ignore
        raw = yf.download(
            _BENCHMARK_TICKER, start=start_date, end=end_date,
            interval="1d", progress=False, auto_adjust=True,
        )
        if raw.empty:
            return None

        # yfinance ≥0.2 returns a MultiIndex: ('Close','QQQ'), ('High','QQQ'), …
        # Handle both MultiIndex and flat-column DataFrames robustly.
        cols = raw.columns
        if isinstance(cols, pd.MultiIndex):
            close = raw[("Close", _BENCHMARK_TICKER)].dropna()
        elif "Close" in cols:
            close = raw["Close"].dropna()
        else:
            close = raw.iloc[:, 0].dropna()

        if len(close) < 2:
            return None
        cum_pct = (close / float(close.iloc[0]) - 1) * 100
        cum_pct.index = pd.to_datetime(cum_pct.index)
        return cum_pct
    except Exception as exc:
        st.warning(f"{_BENCHMARK_TICKER} benchmark unavailable: {exc}")
        return None


def _equity_curve(df: pd.DataFrame) -> go.Figure:
    df2 = df.copy().reset_index(drop=True)
    df2["cumulative_pnl"] = df2["pnl_pct"].cumsum()

    fig = go.Figure()

    # ── Strategy equity curve ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df2["entry_ts"], y=df2["cumulative_pnl"],
        mode="lines+markers", line=dict(color="#26a69a", width=2),
        marker=dict(size=5), name="Strategy (Cumulative PnL %)",
    ))

    # ── QQQ buy-and-hold benchmark ────────────────────────────────────────────
    start_dt = df2["entry_ts"].min()
    end_dt   = df2["entry_ts"].max()
    if pd.notna(start_dt) and pd.notna(end_dt) and start_dt != end_dt:
        start_str = pd.Timestamp(start_dt).strftime("%Y-%m-%d")
        # Add one calendar day so yfinance includes the final trading date's close
        end_str   = (pd.Timestamp(end_dt) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        bm_series = _fetch_benchmark_series(start_str, end_str)
        if bm_series is not None and not bm_series.empty:
            fig.add_trace(go.Scatter(
                x=bm_series.index,
                y=bm_series.values,
                mode="lines",
                line=dict(color="#ff9800", width=1.5, dash="dash"),
                name=f"{_BENCHMARK_TICKER} Buy & Hold %",
            ))

    fig.update_layout(
        title=f"Equity Curve vs {_BENCHMARK_TICKER} Buy & Hold",
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font=dict(color="#cccccc"), height=360,
        xaxis=dict(gridcolor="#1e2530"),
        yaxis=dict(gridcolor="#1e2530", title="Cumulative Return %"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)", font=dict(size=12),
        ),
    )
    return fig


def _pnl_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="pnl_pct", nbins=30,
        color="outcome",
        color_discrete_map={"win": "#26a69a", "loss": "#ef5350", "timeout": "#9e9e9e"},
        title="PnL Distribution by Outcome",
    )
    fig.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font=dict(color="#cccccc"), height=300,
        xaxis=dict(gridcolor="#1e2530", title="PnL %"),
        yaxis=dict(gridcolor="#1e2530", title="Count"),
    )
    return fig


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Performance Analytics", page_icon="📊", layout="wide")
    st.title("📊 Performance Analytics — Backtest Results")

    df = _load_trades()
    if df.empty:
        st.warning("No trade history found. Run the backtest pipeline first:\n"
                   "```\npython -m side_by_side_backtest.main --watchlist scraped_watchlists.json "
                   "--skip-fetch --auto-tune --n-trials 100 --export\n```")
        return

    m = _overall_metrics(df)

    # ── Overall metrics ───────────────────────────────────────────────────────
    st.subheader("Overall Summary")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Trades",    m["total"])
    c2.metric("Win Rate",        f"{m['wr']:.1%}")
    c3.metric("Expectancy",      f"{m['expectancy']:.2f}%")
    c4.metric("Profit Factor",   m["profit_factor"])
    c5.metric("Avg Win",         f"{m['avg_win']:.2f}%")
    c6.metric("Avg Loss",        f"{m['avg_loss']:.2f}%")

    st.plotly_chart(_equity_curve(df), width="stretch")

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(_pnl_histogram(df), width="stretch")

    # ── Per-ticker table ──────────────────────────────────────────────────────
    with col_b:
        st.subheader("🏆 Best Plays (by Expectancy)")
        stats = _per_ticker_stats(df)
        st.dataframe(stats, width='stretch', hide_index=True)

    # ── Deep-link to chart viewer ─────────────────────────────────────────────
    st.subheader("Drill into a ticker")
    top_tickers = stats["Ticker"].tolist()[:20]
    pick = st.selectbox("Select ticker to view chart", top_tickers, key="perf_ticker")
    if st.button(f"📈 Open {pick} in Chart Viewer", key="perf_cv"):
        st.session_state["chart_ticker"] = pick
        st.switch_page("pages/2_chart_viewer.py")

    with st.expander("📋 Raw trade log"):
        st.dataframe(df.sort_values("entry_ts", ascending=False), width='stretch')


if __name__ == "__main__":
    main()
