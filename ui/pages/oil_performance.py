#!/usr/bin/env python3
"""
ALPHAGENE — Oil Performance Dashboard Page

Streamlit page showing strategy performance vs oil benchmarks (XLE, USO, BNO).
Layout: left panel with strategy picker + full formula; right panel with charts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

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


@st.cache_data(ttl=3600, show_spinner=False)
def _load_etf_period_returns(ticker: str, period_starts: list, period_ends: list) -> list:
    """Download ETF price data from yfinance and compute per-period returns aligned to strategy windows."""
    if not period_starts:
        return []
    try:
        start = pd.Timestamp(min(period_starts)) - pd.Timedelta(days=5)
        end = pd.Timestamp(max(period_ends)) + pd.Timedelta(days=5)
        prices = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                             auto_adjust=True, progress=False)["Close"]
        if prices is None or prices.empty:
            return []
        # Compute return for each period window
        results = []
        for ps, pe in zip(period_starts, period_ends):
            window = prices.loc[ps:pe]
            if len(window) < 2:
                continue
            ret = float(window.iloc[-1] / window.iloc[0] - 1)
            results.append({"period_start": ps, "period_end": pe, "total_return": ret})
        return results
    except Exception:
        return []


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

    # Cumulative total return
    cum_return = float(np.prod([1 + r for r in returns]) - 1)

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
        "cum_return": cum_return,
        "alpha_vs_xle": alpha_vs_xle,
        "avg_sharpe": avg_sharpe,
        "max_dd": max_dd,
        "avg_win_rate": avg_win_rate,
        "calmar": calmar,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────────────────────

def _has_meaningful_data(bench_periods: list, min_nonzero_frac: float = 0.5) -> bool:
    """Return True if benchmark has sufficient non-zero return periods."""
    if not bench_periods:
        return False
    rets = [b.get("total_return", 0) for b in bench_periods]
    nonzero = sum(1 for r in rets if abs(r) > 1e-6)
    return nonzero / len(rets) >= min_nonzero_frac


def _build_cumulative_chart(
    period_results: list,
    benchmark_results: dict,
    name: str,
    base_capital: float = 100_000.0,
):
    """Portfolio value over time: Strategy vs oil benchmarks, anchored to base_capital."""
    fig = go.Figure()

    if not period_results:
        fig.update_layout(
            title="Portfolio Value — No Data",
            template="plotly_dark",
            paper_bgcolor=ChartTheme.PAPER,
            plot_bgcolor=ChartTheme.BACKGROUND,
        )
        return fig

    # Strategy equity curve
    periods = sorted(period_results, key=lambda x: x.get("period_start", ""))
    dates = [p["period_start"] for p in periods]
    rets = [p["total_return"] for p in periods]
    cum = np.cumprod([1 + r for r in rets])
    equity = cum * base_capital

    fig.add_trace(go.Scatter(
        x=dates,
        y=equity,
        name=name,
        line=dict(color=ChartTheme.PRIMARY, width=3),
        mode="lines+markers",
        marker=dict(size=6),
        hovertemplate="%{x}<br>$%{y:,.0f}<extra>" + name + "</extra>",
    ))

    # Benchmarks — always show XLE; skip others that have no real data
    colors = {
        "XLE": ChartTheme.WARNING,
        "USO": ChartTheme.ACCENT,
        "BNO": ChartTheme.SUCCESS,
        "XOP": ChartTheme.ERROR,
    }
    for bench_name, bench_periods in benchmark_results.items():
        is_xle = "XLE" in bench_name.upper()
        if not is_xle and not _has_meaningful_data(bench_periods):
            continue
        if not bench_periods:
            continue
        sorted_b = sorted(bench_periods, key=lambda x: x.get("period_start", ""))
        b_dates = [b["period_start"] for b in sorted_b]
        b_rets = [b["total_return"] for b in sorted_b]
        b_cum = np.cumprod([1 + r for r in b_rets])
        b_equity = b_cum * base_capital

        color = ChartTheme.GRID
        for ticker, c in colors.items():
            if ticker in bench_name.upper():
                color = c
                break

        label = bench_name.replace("_benchmark", "")
        fig.add_trace(go.Scatter(
            x=b_dates,
            y=b_equity,
            name=label,
            line=dict(color=color, width=2, dash="dash"),
            mode="lines+markers",
            marker=dict(size=4),
            hovertemplate="%{x}<br>$%{y:,.0f}<extra>" + label + "</extra>",
        ))

    fig.update_layout(
        title="📈 Portfolio Value: Strategy vs Oil Benchmarks",
        xaxis_title="Period Start",
        yaxis_title="Portfolio Value ($)",
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
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

    # ── Two-column layout ──────────────────────────────────────────────────
    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.subheader("Strategy")

        # Strategy selector — compact labels
        options = {}
        for i, s in enumerate(strategies):
            label = f"#{i+1}  fit={s['fitness']:+.3f}  |  {s.get('formula', '')[:28]}…"
            options[label] = s["strategy_id"]

        selected_label = st.selectbox(
            "Select strategy",
            list(options.keys()),
            label_visibility="collapsed",
        )
        strategy_id = options[selected_label]

        # Base capital input
        base_capital = st.number_input(
            "Base Capital ($)",
            min_value=1_000,
            max_value=100_000_000,
            value=100_000,
            step=10_000,
            format="%d",
            help="Starting portfolio value used to convert returns to dollar amounts.",
        )

        # Strategy metadata
        strategy_info = next(
            (s for s in strategies if s["strategy_id"] == strategy_id), None
        )
        if strategy_info:
            st.divider()
            st.caption(f"**ID:** `{strategy_info['strategy_id']}`")
            st.caption(f"**Fitness:** `{strategy_info['fitness']:+.4f}`")
            complexity = strategy_info.get("complexity", None)
            if complexity is not None:
                st.caption(f"**Complexity:** {complexity} nodes")
            generation = strategy_info.get("generation", None)
            if generation is not None:
                st.caption(f"**Generation:** {generation}")

            # Full formula — copyable code block
            st.divider()
            st.markdown("**Formula**")
            formula = strategy_info.get("formula", "N/A")
            st.code(formula, language="text")

    # ── Right panel: KPIs + Charts ─────────────────────────────────────────
    with right_col:
        period_results = _load_strategy_results(strategy_id)
        benchmark_results = _load_benchmarks_for_strategy(strategy_id)

        if not period_results:
            st.warning("No period results for this strategy.")
            return

        kpis = _compute_kpis(period_results, benchmark_results)

        # KPI cards
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        with k1:
            st.metric(
                "Cumulative Return",
                f"{kpis['cum_return']:+.1%}",
                help=f"Based on base capital of ${base_capital:,.0f}",
            )
        with k2:
            alpha_str = (
                f"{kpis['alpha_vs_xle']:+.1%}"
                if kpis.get("alpha_vs_xle") is not None
                else "N/A"
            )
            st.metric("Alpha vs XLE", alpha_str)
        with k3:
            st.metric("Sharpe", f"{kpis['avg_sharpe']:.2f}")
        with k4:
            st.metric("Max Drawdown", f"{kpis['max_dd']:.1%}")
        with k5:
            st.metric("Win Rate", f"{kpis['avg_win_rate']:.0%}")
        with k6:
            st.metric("Calmar", f"{kpis['calmar']:.2f}")

        final_value = base_capital * (1 + kpis["cum_return"])
        st.caption(
            f"\\${base_capital:,.0f} starting capital → **\\${final_value:,.0f}** "
            f"({kpis['cum_return']:+.1%} cumulative)"
        )

        st.divider()

        # ── Resolve XLE benchmark — prefer DB, fall back to live yfinance ──
        sorted_periods = sorted(
            period_results, key=lambda x: x.get("period_start", "")
        )
        period_starts = [p["period_start"] for p in sorted_periods]
        period_ends = [p["period_end"] for p in sorted_periods]

        xle_bench = []
        for key, bperiods in benchmark_results.items():
            if "XLE" in key.upper() and _has_meaningful_data(bperiods):
                xle_bench = bperiods
                break

        if not xle_bench:
            with st.spinner("Fetching live XLE data…"):
                xle_live = _load_etf_period_returns("XLE", period_starts, period_ends)
            if xle_live:
                benchmark_results = dict(benchmark_results)
                benchmark_results["XLE_live"] = xle_live
                xle_bench = xle_live

        # Portfolio value chart
        fig_cum = _build_cumulative_chart(
            period_results, benchmark_results, "Strategy", base_capital
        )
        st.plotly_chart(fig_cum, width='stretch')

        # Oil price overlay (USO as proxy)
        strat_returns = pd.Series(
            [p["total_return"] for p in sorted_periods],
            index=pd.to_datetime(period_starts),
        )

        uso_periods = benchmark_results.get("USO_benchmark", [])
        if not uso_periods or not _has_meaningful_data(uso_periods):
            # Try live USO
            with st.spinner("Fetching live USO data…"):
                uso_live = _load_etf_period_returns("USO", period_starts, period_ends)
            if uso_live:
                uso_periods = uso_live

        if uso_periods and _has_meaningful_data(uso_periods):
            uso_sorted = sorted(uso_periods, key=lambda x: x.get("period_start", ""))
            uso_prices = pd.Series(
                np.cumprod([1 + b["total_return"] for b in uso_sorted]) * 100,
                index=pd.to_datetime([b["period_start"] for b in uso_sorted]),
            )
            fig_overlay = create_oil_overlay_chart(strat_returns, uso_prices)
            st.plotly_chart(fig_overlay, width='stretch')

        # Alpha heatmap (vs XLE)
        if xle_bench:
            fig_heat = create_alpha_heatmap(period_results, xle_bench)
            st.plotly_chart(fig_heat, width='stretch')

        # Period details table
        st.subheader("📋 Period Details")
        df_table = _build_period_table(period_results, benchmark_results)
        if not df_table.empty:
            st.dataframe(df_table, width='stretch', hide_index=True)

        # Generation stats (if available)
        if strategy_info and strategy_info.get("run_id"):
            gen_stats = _load_generation_stats(strategy_info["run_id"])
            if gen_stats:
                st.subheader("🧬 Evolution Progress")
                fig_evo = create_fitness_evolution_chart(gen_stats)
                st.plotly_chart(fig_evo, width='stretch')
