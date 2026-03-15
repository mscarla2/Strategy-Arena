#!/usr/bin/env python3
"""
ALPHAGENE Strategy Dashboard
Streamlit app for visualizing GP strategies vs benchmarks.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from evolution.gp_storage import GPDatabase

# ═══════════════════════════════════════════════════════════════════════════════
# THEME
# ═══════════════════════════════════════════════════════════════════════════════

class Theme:
    BACKGROUND = "#0d1117"
    PAPER = "#161b22"
    GRID = "#30363d"
    TEXT = "#c9d1d9"
    PRIMARY = "#58a6ff"
    SUCCESS = "#3fb950"
    WARNING = "#d29922"
    ERROR = "#f85149"
    ACCENT = "#bc8cff"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_database():
    return GPDatabase("data/gp_strategies.db")


@st.cache_data
def load_runs():
    db = get_database()
    return db.get_runs(limit=50)


@st.cache_data
def load_strategies(min_fitness: float = -1.0):
    db = get_database()
    return db.get_top_strategies(limit=100, min_fitness=min_fitness)


@st.cache_data
def load_strategy_results(strategy_id: str):
    db = get_database()
    return db.get_strategy_results(strategy_id)


@st.cache_data
def load_benchmarks(run_id: str):
    db = get_database()
    return db.get_benchmarks_for_run(run_id)


@st.cache_data
def load_generation_stats(run_id: str):
    db = get_database()
    return db.get_generation_stats(run_id)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_cumulative_returns_chart(
    strategy_results: list,
    benchmark_results: dict,
    strategy_name: str = "Strategy"
) -> go.Figure:
    """Build cumulative returns chart from period results."""
    
    fig = go.Figure()
    
    # Strategy cumulative returns
    if strategy_results:
        periods = sorted(strategy_results, key=lambda x: x.get('period_start', ''))
        dates = [p.get('period_start', '') for p in periods]
        returns = [p.get('total_return', 0) for p in periods]
        
        cumulative = []
        cum = 1.0
        for r in returns:
            cum *= (1 + r)
            cumulative.append((cum - 1) * 100)
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=cumulative,
            name=strategy_name,
            line=dict(color=Theme.PRIMARY, width=3),
            mode='lines+markers',
            marker=dict(size=8),
            hovertemplate="<b>%{x}</b><br>Cumulative: %{y:.1f}%<extra></extra>"
        ))
    
    # Benchmark cumulative returns
    for bench_name, bench_periods in benchmark_results.items():
        if bench_periods:
            sorted_bench = sorted(bench_periods, key=lambda x: x.get('period_start', ''))
            dates = [p.get('period_start', '') for p in sorted_bench]
            returns = [p.get('total_return', 0) for p in sorted_bench]
            
            cumulative = []
            cum = 1.0
            for r in returns:
                cum *= (1 + r)
                cumulative.append((cum - 1) * 100)
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=cumulative,
                name=bench_name,
                line=dict(color=Theme.WARNING, width=2, dash='dash'),
                mode='lines+markers',
                marker=dict(size=6),
                hovertemplate="<b>%{x}</b><br>Cumulative: %{y:.1f}%<extra></extra>"
            ))
    
    fig.update_layout(
        title="📈 Cumulative Returns",
        xaxis_title="Period",
        yaxis_title="Cumulative Return (%)",
        template="plotly_dark",
        paper_bgcolor=Theme.PAPER,
        plot_bgcolor=Theme.BACKGROUND,
        font=dict(color=Theme.TEXT),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def build_period_comparison_chart(
    strategy_results: list,
    benchmark_results: dict
) -> go.Figure:
    """Build period-by-period return comparison bar chart."""
    
    fig = go.Figure()
    
    if not strategy_results:
        return fig
    
    periods = sorted(strategy_results, key=lambda x: x.get('period_start', ''))
    dates = [p.get('period_start', '')[:7] for p in periods]  # YYYY-MM format
    strategy_returns = [p.get('total_return', 0) * 100 for p in periods]
    
    # Strategy bars
    fig.add_trace(go.Bar(
        x=dates,
        y=strategy_returns,
        name="Strategy",
        marker_color=Theme.PRIMARY,
        hovertemplate="<b>%{x}</b><br>Return: %{y:.1f}%<extra></extra>"
    ))
    
    # Benchmark bars
    for bench_name, bench_periods in benchmark_results.items():
        if bench_periods:
            sorted_bench = sorted(bench_periods, key=lambda x: x.get('period_start', ''))
            bench_returns = [p.get('total_return', 0) * 100 for p in sorted_bench]
            
            # Align with strategy periods
            bench_returns = bench_returns[:len(dates)]
            
            fig.add_trace(go.Bar(
                x=dates[:len(bench_returns)],
                y=bench_returns,
                name=bench_name,
                marker_color=Theme.WARNING,
                opacity=0.7,
                hovertemplate="<b>%{x}</b><br>Return: %{y:.1f}%<extra></extra>"
            ))
    
    fig.update_layout(
        title="📊 Period Returns Comparison",
        xaxis_title="Period",
        yaxis_title="Return (%)",
        template="plotly_dark",
        paper_bgcolor=Theme.PAPER,
        plot_bgcolor=Theme.BACKGROUND,
        font=dict(color=Theme.TEXT),
        barmode='group',
        hovermode='x unified'
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color=Theme.GRID)
    
    return fig


def build_drawdown_chart(strategy_results: list) -> go.Figure:
    """Build drawdown chart from period results."""
    
    fig = go.Figure()
    
    if not strategy_results:
        return fig
    
    periods = sorted(strategy_results, key=lambda x: x.get('period_start', ''))
    dates = [p.get('period_start', '') for p in periods]
    drawdowns = [-abs(p.get('max_drawdown', 0)) * 100 for p in periods]
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=drawdowns,
        fill='tozeroy',
        name="Max Drawdown",
        line=dict(color=Theme.ERROR, width=2),
        fillcolor="rgba(248, 81, 73, 0.3)",
        hovertemplate="<b>%{x}</b><br>Max DD: %{y:.1f}%<extra></extra>"
    ))
    
    fig.update_layout(
        title="📉 Period Drawdowns",
        xaxis_title="Period",
        yaxis_title="Max Drawdown (%)",
        template="plotly_dark",
        paper_bgcolor=Theme.PAPER,
        plot_bgcolor=Theme.BACKGROUND,
        font=dict(color=Theme.TEXT),
    )
    
    return fig


def build_rolling_sharpe_chart(
    strategy_results: list,
    benchmark_results: dict
) -> go.Figure:
    """Build rolling Sharpe comparison chart."""
    
    fig = go.Figure()
    
    if not strategy_results:
        return fig
    
    periods = sorted(strategy_results, key=lambda x: x.get('period_start', ''))
    dates = [p.get('period_start', '') for p in periods]
    sharpes = [p.get('sharpe_ratio', 0) for p in periods]
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=sharpes,
        name="Strategy",
        line=dict(color=Theme.PRIMARY, width=2),
        mode='lines+markers',
        hovertemplate="<b>%{x}</b><br>Sharpe: %{y:.2f}<extra></extra>"
    ))
    
    # Benchmark Sharpes
    for bench_name, bench_periods in benchmark_results.items():
        if bench_periods:
            sorted_bench = sorted(bench_periods, key=lambda x: x.get('period_start', ''))
            bench_dates = [p.get('period_start', '') for p in sorted_bench]
            bench_sharpes = [p.get('sharpe_ratio', 0) for p in sorted_bench]
            
            fig.add_trace(go.Scatter(
                x=bench_dates,
                y=bench_sharpes,
                name=bench_name,
                line=dict(color=Theme.WARNING, width=2, dash='dash'),
                mode='lines+markers',
                hovertemplate="<b>%{x}</b><br>Sharpe: %{y:.2f}<extra></extra>"
            ))
    
    fig.update_layout(
        title="📐 Period Sharpe Ratios",
        xaxis_title="Period",
        yaxis_title="Sharpe Ratio",
        template="plotly_dark",
        paper_bgcolor=Theme.PAPER,
        plot_bgcolor=Theme.BACKGROUND,
        font=dict(color=Theme.TEXT),
        hovermode='x unified'
    )
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="solid", line_color=Theme.GRID)
    fig.add_hline(y=1, line_dash="dash", line_color=Theme.SUCCESS, 
                  annotation_text="Good", annotation_position="right")
    
    return fig


def build_fitness_evolution_chart(generation_stats: list) -> go.Figure:
    """Build fitness evolution across generations."""
    
    fig = go.Figure()
    
    if not generation_stats:
        return fig
    
    gens = [s.get('generation', 0) for s in generation_stats]
    max_fitness = [s.get('max_fitness', 0) for s in generation_stats]
    avg_fitness = [s.get('avg_fitness', 0) for s in generation_stats]
    
    fig.add_trace(go.Scatter(
        x=gens,
        y=max_fitness,
        name="Max Fitness",
        line=dict(color=Theme.SUCCESS, width=3),
        mode='lines+markers',
        marker=dict(size=8, symbol='star'),
    ))
    
    fig.add_trace(go.Scatter(
        x=gens,
        y=avg_fitness,
        name="Avg Fitness",
        line=dict(color=Theme.TEXT, width=2, dash='dash'),
        mode='lines',
    ))
    
    # Fill between
    fig.add_trace(go.Scatter(
        x=gens + gens[::-1],
        y=max_fitness + avg_fitness[::-1],
        fill='toself',
        fillcolor='rgba(88, 166, 255, 0.1)',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title="🧬 Fitness Evolution",
        xaxis_title="Generation",
        yaxis_title="Fitness",
        template="plotly_dark",
        paper_bgcolor=Theme.PAPER,
        plot_bgcolor=Theme.BACKGROUND,
        font=dict(color=Theme.TEXT),
    )
    
    # Zero line (benchmark level)
    fig.add_hline(y=0, line_dash="solid", line_color=Theme.WARNING,
                  annotation_text="Benchmark", annotation_position="right")
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="AlphaGene Dashboard v3.0",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for dark theme
    st.markdown("""
        <style>
        .stApp {
            background-color: #0d1117;
        }
        .stSelectbox label, .stSlider label {
            color: #c9d1d9 !important;
        }
        .metric-card {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #58a6ff;
        }
        .metric-label {
            font-size: 12px;
            color: #8b949e;
        }
        .positive { color: #3fb950 !important; }
        .negative { color: #f85149 !important; }
        </style>
    """, unsafe_allow_html=True)
    
    # Header with version and status
    st.markdown("""
        <div style='text-align: center;'>
            <h1 style='color: #58a6ff; margin-bottom: 0;'>
                🧬 ALPHAGENE Strategy Dashboard
            </h1>
            <p style='color: #8b949e; font-size: 14px; margin-top: 5px;'>
                v3.0 | Production Ready | Genetic Programming Arena
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Strategy Selection
    st.sidebar.header("📋 Strategy Selection")
    
    strategies = load_strategies()
    
    if not strategies:
        st.warning("No strategies found. Run arena_runner_v3.py first.")
        return
    
    # Strategy dropdown - sorted by fitness (already sorted from database query)
    strategy_options = {
        f"#{i+1} | fit={s['fitness']:.3f} | {s['formula'][:40]}...": s['strategy_id']
        for i, s in enumerate(strategies)
    }
    
    selected_label = st.sidebar.selectbox(
        "Select Strategy",
        options=list(strategy_options.keys())
    )
    
    selected_id = strategy_options[selected_label]
    selected_strategy = next(s for s in strategies if s['strategy_id'] == selected_id)
    
    # Load data
    strategy_results = load_strategy_results(selected_id)
    run_id = selected_strategy.get('run_id', '')
    benchmarks = load_benchmarks(run_id) if run_id else {}
    generation_stats = load_generation_stats(run_id) if run_id else []
    
    # Strategy Info Card
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Strategy Info")
    
    # Show run metadata
    run_date = selected_strategy.get('run_date', 'N/A')
    if run_date and run_date != 'N/A':
        try:
            from datetime import datetime
            run_dt = datetime.fromisoformat(run_date.replace('Z', '+00:00'))
            run_date_str = run_dt.strftime('%Y-%m-%d %H:%M')
        except:
            run_date_str = str(run_date)[:16]
        st.sidebar.caption(f"📅 Run: {run_date_str}")
    
    st.sidebar.code(selected_strategy.get('formula', 'N/A'), language=None)
    
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Fitness", f"{selected_strategy.get('fitness', 0):.3f}")
    col2.metric("Complexity", f"{selected_strategy.get('complexity', 0)} nodes")
    
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Top %", f"{selected_strategy.get('top_pct', 0)}%")
    col2.metric("Generation", selected_strategy.get('generation', 'N/A'))
    
    # Strategy ID for reference
    st.sidebar.caption(f"🔑 ID: {selected_strategy.get('strategy_id', 'N/A')[:16]}...")
    
    # Main Content
    if not strategy_results:
        st.warning("No period results found for this strategy.")
        return
    
    # Metrics Row
    st.markdown("### 📈 Performance Summary")
    
    avg_return = np.mean([r['total_return'] for r in strategy_results])
    avg_sharpe = np.mean([r['sharpe_ratio'] for r in strategy_results])
    avg_dd = np.mean([r['max_drawdown'] for r in strategy_results])
    win_rate = sum(1 for r in strategy_results if r['total_return'] > 0) / len(strategy_results)
    
    # Benchmark comparison
    bench_return = 0
    bench_sharpe = 0
    if benchmarks:
        first_bench = list(benchmarks.values())[0]
        if first_bench:
            bench_return = np.mean([b['total_return'] for b in first_bench])
            bench_sharpe = np.mean([b['sharpe_ratio'] for b in first_bench])
    
    alpha = avg_return - bench_return
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric(
        "Avg Return",
        f"{avg_return:.1%}",
        f"{alpha:+.1%} vs bench" if bench_return else None,
        delta_color="normal"
    )
    col2.metric(
        "Avg Sharpe",
        f"{avg_sharpe:.2f}",
        f"{avg_sharpe - bench_sharpe:+.2f}" if bench_sharpe else None
    )
    col3.metric("Avg Max DD", f"{avg_dd:.1%}")
    col4.metric("Win Rate", f"{win_rate:.0%}")
    col5.metric("Periods", len(strategy_results))
    
    st.markdown("---")
    
    # Charts Row 1 - Cumulative Returns (full width)
    st.plotly_chart(
        build_cumulative_returns_chart(strategy_results, benchmarks, "Strategy"),
        use_container_width=True
    )
    
    # Charts Row 2 - Period Returns & Drawdowns
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            build_period_comparison_chart(strategy_results, benchmarks),
            use_container_width=True
        )
    
    with col2:
        st.plotly_chart(
            build_drawdown_chart(strategy_results),
            use_container_width=True
        )
    
    # Charts Row 3 - Sharpe & Fitness Evolution
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            build_rolling_sharpe_chart(strategy_results, benchmarks),
            use_container_width=True
        )
    
    with col2:
        if generation_stats:
            st.plotly_chart(
                build_fitness_evolution_chart(generation_stats),
                use_container_width=True
            )
        else:
            st.info("No generation stats available for this run.")
    
    # Period Details Table
    st.markdown("---")
    st.markdown("### 📋 Period Details")
    
    df = pd.DataFrame(strategy_results)
    if not df.empty:
        df = df[['period_start', 'period_end', 'total_return', 'sharpe_ratio', 'max_drawdown']]
        df.columns = ['Start', 'End', 'Return', 'Sharpe', 'Max DD']
        df['Return'] = df['Return'].apply(lambda x: f"{x:.1%}")
        df['Sharpe'] = df['Sharpe'].apply(lambda x: f"{x:.2f}")
        df['Max DD'] = df['Max DD'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(df, use_container_width=True, height=400)
    
    # Footer with system info
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("🧬 AlphaGene v3.0")
    with col2:
        st.caption("📊 Genetic Programming Arena")
    with col3:
        st.caption("✅ Production Ready")


if __name__ == "__main__":
    main()