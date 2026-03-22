#!/usr/bin/env python3
"""
ALPHAGENE — Oil Performance Dashboard Page

Streamlit page showing strategy performance vs oil benchmarks (XLE, USO, BNO).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from evolution.gp_storage import GPDatabase
from ui.charts import (
    ChartTheme,
    create_oil_overlay_chart,
    create_alpha_heatmap,
    create_fitness_evolution_chart,
)

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH = "data/gp_strategies.db"


@st.cache_resource
def _get_db():
    return GPDatabase(DB_PATH)


@st.cache_data(ttl=30)
def _load_strategies(min_fitness: float = -999.0):
    db = _get_db()
    return db.get_top_strategies(limit=200, min_fitness=min_fitness)


@st.cache_data(ttl=30)
def _load_strategy_results(strategy_id: str):
    db = _get_db()
    return db.get_strategy_results(strategy_id)


@st.cache_data(ttl=30)
def _load_benchmarks_for_strategy(strategy_id: str):
    db = _get_db()
    return db.get_benchmarks_for_strategy(strategy_id)


@st.cache_data(ttl=30)
def _load_generation_stats(run_id: str):
    db = _get_db()
    return db.get_generation_stats(run_id)


# ─────────────────────────────────────────────────────────────────────────────
# KPI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_kpis(period_results: list, benchmark_results: dict):
    """Compute KPI metrics from period results."""
    if not period_results:
        return {}

    returns = [p["total_return"] for p in period_results]
    sharpes = [p["sharpe_ratio"] for p in period_results]
    drawdowns = [p["max_drawdown"] for p in period_results]
    win_rates = [p.get("win_rate", 0) for p in period_results]

    avg_return = np.mean(returns) if returns else 0
    avg_sharpe = np.mean(sharpes) if sharpes else 0
    max_dd = np.max(drawdowns) if drawdowns else 0
    avg_win_rate = np.mean(win_rates) if win_rates else 0

    # Calmar = annualised return / max drawdown
    annual_return = avg_return * 4  # quarterly periods → annualised
    calmar = annual_return / max_dd if max_dd > 0 else 0

    # Alpha vs XLE
    xle_key = None
    for key in benchmark_results:
        if "XLE" in key.upper():
            xle_key = key
            break
    if xle_key and benchmark_results[xle_key]:
        bench_returns = [b["total_return"] for b in benchmark_results[xle_key]]
        alpha_vs_xle = avg_return - np.mean(bench_returns)
    else:
        alpha_vs_xle = None

    return {
        "avg_return": avg_return,
        "alpha_vs_xle": alpha_vs_xle,
        "avg_sharpe": avg_sharpe,
        "max_dd": max_dd,
        "avg_win_rate": avg_win_rate,
        "calmar": calmar,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_cumulative_chart(period_results: list, benchmark_results: dict, name: str):
    """Cumulative returns: Strategy vs XLE vs USO vs BNO."""
    fig = go.Figure()

    if not period_results:
        fig.update_layout(title="Cumulative Returns — No Data", template="plotly_dark",
                          paper_bgcolor=ChartTheme.PAPER, plot_bgcolor=ChartTheme.BACKGROUND)
        return fig

    # Strategy
    periods = sorted(period_results, key=lambda x: x.get("period_start", ""))
    dates = [p["period_start"] for p in periods]
    rets = [p["total_return"] for p in periods]
    cum = np.cumprod([1 + r for r in rets])
    cum_pct = (cum - 1) * 100

    fig.add_trace(go.Scatter(
        x=dates, y=cum_pct, name=name,
        line=dict(color=ChartTheme.PRIMARY, width=3),
        mode="lines+markers", marker=dict(size=6),
    ))

    # Benchmarks
    colors = {
        "XLE": ChartTheme.WARNING,
        "USO": ChartTheme.ACCENT,
        "BNO": ChartTheme.SUCCESS,
        "XOP": ChartTheme.ERROR,
    }
    for bench_name, bench_periods in benchmark_results.items():
        if not bench_periods:
            continue
        sorted_b = sorted(bench_periods, key=lambda x: x.get("period_start", ""))
        b_dates = [b["period_start"] for b in sorted_b]
        b_rets = [b["total_return"] for b in sorted_b]
        b_cum = np.cumprod([1 + r for r in b_rets])
        b_cum_pct = (b_cum - 1) * 100

        # Pick colour based on ticker in name
        color = ChartTheme.GRID
        for ticker, c in colors.items():
            if ticker in bench_name.upper():
                color = c
                break

        fig.add_trace(go.Scatter(
            x=b_dates, y=b_cum_pct, name=bench_name.replace("_benchmark", ""),
            line=dict(color=color, width=2, dash="dash"),
            mode="lines+markers", marker=dict(size=4),
        ))

    fig.update_layout(
        title="📈 Cumulative Returns: Strategy vs Oil Benchmarks",
        xaxis_title="Period Start",
        yaxis_title="Cumulative Return (%)",
        template="plotly_dark",
        paper_bgcolor=ChartTheme.PAPER,
        plot_bgcolor=ChartTheme.BACKGROUND,
        font=dict(color=ChartTheme.TEXT),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _build_period_table(period_results: list, benchmark_results: dict):
    """Build a sortable period details DataFrame."""
    if not period_results:
        return pd.DataFrame()

    # Build benchmark lookup
    xle_map = {}
    uso_map = {}
    for key, periods in benchmark_results.items():
        target = None
        if "XLE" in key.upper():
            target = xle_map
        elif "USO" in key.upper():
            target = uso_map
        if target is not None:
            for b in periods:
                target[b["period_start"]] = b["total_return"]

    rows = []
    for p in sorted(period_results, key=lambda x: x.get("period_start", "")):
        ps = p.get("period_start", "")
        pe = p.get("period_end", "")
        ret = p.get("total_return", 0)
        rows.append({
            "Start": ps,
            "End": pe,
            "Return": f"{ret:.1%}",
            "vs XLE": f"{ret - xle_map.get(ps, 0):+.1%}" if ps in xle_map else "—",
            "vs USO": f"{ret - uso_map.get(ps, 0):+.1%}" if ps in uso_map else "—",
            "Sharpe": f"{p.get('sharpe_ratio', 0):.2f}",
            "Max DD": f"{p.get('max_drawdown', 0):.1%}",
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────

def render():
    """Entry point for the Oil Performance page."""
    st.header("🛢️ Oil Performance Dashboard")

    strategies = _load_strategies()
    if not strategies:
        st.info("No strategies found in the database. Run an evolution first.")
        return

    # Strategy selector
    options = {
        f"{s['strategy_id'][:8]}… | fit={s['fitness']:+.3f} | {s.get('formula', '')[:40]}": s["strategy_id"]
        for s in strategies
    }
    selected_label = st.selectbox("Select Strategy", list(options.keys()))
    strategy_id = options[selected_label]

    period_results = _load_strategy_results(strategy_id)
    benchmark_results = _load_benchmarks_for_strategy(strategy_id)

    if not period_results:
        st.warning("No period results for this strategy.")
        return

    # ── KPI cards ──────────────────────────────────────────────────────────
    kpis = _compute_kpis(period_results, benchmark_results)

    cols = st.columns(6)
    with cols[0]:
        alpha_str = f"{kpis['alpha_vs_xle']:+.1%}" if kpis.get("alpha_vs_xle") is not None else "N/A"
        st.metric("Alpha vs XLE", alpha_str)
    with cols[1]:
        st.metric("Sharpe", f"{kpis['avg_sharpe']:.2f}")
    with cols[2]:
        st.metric("Max Drawdown", f"{kpis['max_dd']:.1%}")
    with cols[3]:
        st.metric("Win Rate", f"{kpis['avg_win_rate']:.0%}")
    with cols[4]:
        st.metric("Calmar", f"{kpis['calmar']:.2f}")
    with cols[5]:
        st.metric("Avg Return", f"{kpis['avg_return']:.1%}")

    st.divider()

    # ── Cumulative returns chart ───────────────────────────────────────────
    fig_cum = _build_cumulative_chart(period_results, benchmark_results, "Strategy")
    st.plotly_chart(fig_cum, use_container_width=True)

    # ── Oil price overlay ──────────────────────────────────────────────────
    # Build strategy returns series for overlay
    sorted_periods = sorted(period_results, key=lambda x: x.get("period_start", ""))
    strat_returns = pd.Series(
        [p["total_return"] for p in sorted_periods],
        index=pd.to_datetime([p["period_start"] for p in sorted_periods]),
    )

    # USO benchmark as oil price proxy
    uso_periods = benchmark_results.get("USO_benchmark", [])
    if uso_periods:
        uso_sorted = sorted(uso_periods, key=lambda x: x.get("period_start", ""))
        uso_prices = pd.Series(
            np.cumprod([1 + b["total_return"] for b in uso_sorted]) * 100,
            index=pd.to_datetime([b["period_start"] for b in uso_sorted]),
        )
        fig_overlay = create_oil_overlay_chart(strat_returns, uso_prices)
        st.plotly_chart(fig_overlay, use_container_width=True)

    # ── Alpha heatmap ──────────────────────────────────────────────────────
    # Use XLE benchmark for alpha calculation
    xle_bench = []
    for key, periods in benchmark_results.items():
        if "XLE" in key.upper():
            xle_bench = periods
            break

    if xle_bench:
        fig_heat = create_alpha_heatmap(period_results, xle_bench)
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Period details table ───────────────────────────────────────────────
    st.subheader("📋 Period Details")
    df_table = _build_period_table(period_results, benchmark_results)
    if not df_table.empty:
        st.dataframe(df_table, use_container_width=True, hide_index=True)

    # ── Generation stats (if available) ────────────────────────────────────
    strategy_info = next((s for s in strategies if s["strategy_id"] == strategy_id), None)
    if strategy_info and strategy_info.get("run_id"):
        gen_stats = _load_generation_stats(strategy_info["run_id"])
        if gen_stats:
            st.subheader("🧬 Evolution Progress")
            fig_evo = create_fitness_evolution_chart(gen_stats)
            st.plotly_chart(fig_evo, use_container_width=True)
