#!/usr/bin/env python3
"""
ALPHAGENE — Feature Importance Page

Streamlit page showing which features contribute most to alpha generation.
"""

import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from ui.charts import (
    create_feature_importance_bar,
    create_feature_category_pie,
    ChartTheme,
)

import plotly.graph_objects as go


# ─────────────────────────────────────────────────────────────────────────────
# Feature category mapping
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_MAP = {
    # Momentum
    "momentum": "Momentum",
    "rsi": "Momentum",
    "macd": "Momentum",
    "roc": "Momentum",
    "williams": "Momentum",
    "stochastic": "Momentum",
    # Value
    "pe": "Value",
    "pb": "Value",
    "ps": "Value",
    "ev_ebitda": "Value",
    "book_value": "Value",
    "earnings": "Value",
    # Volatility
    "vol": "Volatility",
    "atr": "Volatility",
    "bollinger": "Volatility",
    "std": "Volatility",
    "variance": "Volatility",
    # Volume
    "volume": "Volume",
    "obv": "Volume",
    "vwap": "Volume",
    "adv": "Volume",
    "turnover": "Volume",
    # Oil-specific
    "oil": "Oil",
    "wti": "Oil",
    "brent": "Oil",
    "crack": "Oil",
    "inventory": "Oil",
    "rig_count": "Oil",
    "contango": "Oil",
    "backwardation": "Oil",
    "spread": "Oil",
    # Mean reversion
    "mean_rev": "Mean Reversion",
    "zscore": "Mean Reversion",
    "deviation": "Mean Reversion",
    # Quality
    "roe": "Quality",
    "roa": "Quality",
    "margin": "Quality",
    "debt": "Quality",
    # Trend
    "sma": "Trend",
    "ema": "Trend",
    "trend": "Trend",
    "ma_cross": "Trend",
    "adx": "Trend",
}


def _categorize_feature(feature_name: str) -> str:
    """Map a feature name to a category using prefix matching."""
    lower = feature_name.lower()
    
    # Match your actual feature naming conventions
    if lower.startswith('mom_') or lower.startswith('rsi') or 'momentum' in lower:
        return 'Momentum'
    elif lower.startswith('zscore') or lower.startswith('dist_ma') or lower.startswith('reversion'):
        return 'Mean Reversion'
    elif lower.startswith('vol_') or lower.startswith('vol_of'):
        return 'Volatility'
    elif lower.startswith(('drawdown', 'recovery', 'max_dd', 'ulcer')):
        return 'Drawdown/Risk'
    elif lower.startswith(('skew', 'kurt', 'downside', 'left_tail', 'up_down')):
        return 'Higher Moments'
    elif lower.startswith(('trend', 'hurst')):
        return 'Trend'
    elif lower.startswith(('range', 'breakout', 'support', 'resistance', 'high_prox')):
        return 'Price Level'
    elif lower.startswith(('rel_', 'excess_mom', 'rel_value')):
        return 'Cross-Sectional'
    elif lower.startswith(('sharpe', 'sortino', 'calmar', 'info_ratio')):
        return 'Risk-Adjusted'
    elif lower.startswith(('market_beta', 'beta_instab', 'idio_vol', 'sector_corr', 'universe_disp')):
        return 'Market-Relative'
    elif lower.startswith(('amihud', 'roll_spread', 'kyle', 'zero_return', 'turnover_rate')):
        return 'Microstructure'
    elif lower.startswith(('vol_regime', 'trend_strength', 'correlation_spike', 'breadth')):
        return 'Regime'
    elif lower.startswith(('mom_sharpe', 'dd_adj', 'vol_adj', 'efficiency', 'fractal')):
        return 'Engineered'
    elif lower.startswith(('oil_', 'wti', 'brent', 'crack', 'inventory')):
        return 'Oil'
    elif lower.startswith(('smc_',)):
        return 'SMC'
    elif lower.startswith(('sr_',)):
        return 'Support/Resistance'
    elif lower.startswith(('volume_', 'price_volume')):
        return 'Volume'
    elif lower.startswith(('gap_', 'streak', 'return_autocorr', 'abs_return')):
        return 'Pattern'
    else:
        return 'Other'


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def _load_feature_data(top_n: int = 50, min_fitness: float = 0.0) -> pd.DataFrame:
    """Load feature importance data from the analyzer."""
    try:
        from tools.feature_importance import FeatureImportanceAnalyzer
        analyzer = FeatureImportanceAnalyzer()
        df = analyzer.analyze_top_strategies(top_n=top_n, min_fitness=min_fitness)
        return df
    except Exception as exc:
        st.warning(f"Could not load feature data: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def _load_feature_by_tier(min_fitness: float = 0.0) -> pd.DataFrame:
    """Load feature usage bucketed by fitness tier."""
    try:
        from tools.feature_importance import FeatureImportanceAnalyzer
        analyzer = FeatureImportanceAnalyzer()

        # Get strategies at different fitness tiers
        tiers = {
            "Top 10": analyzer.analyze_top_strategies(top_n=10, min_fitness=min_fitness),
            "Top 25": analyzer.analyze_top_strategies(top_n=25, min_fitness=min_fitness),
            "Top 50": analyzer.analyze_top_strategies(top_n=50, min_fitness=min_fitness),
        }

        # Combine into a single DataFrame for stacked bar
        rows = []
        for tier_name, df in tiers.items():
            if df.empty:
                continue
            for _, row in df.head(15).iterrows():
                rows.append({
                    "feature": row["feature"],
                    "tier": tier_name,
                    "usage_count": row["usage_count"],
                })

        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_tier_chart(tier_df: pd.DataFrame) -> go.Figure:
    """Stacked bar chart of feature usage by fitness tier."""
    if tier_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Feature Usage by Tier — No Data", template="plotly_dark",
                          paper_bgcolor=ChartTheme.PAPER, plot_bgcolor=ChartTheme.BACKGROUND)
        return fig

    tier_colors = {
        "Top 10": ChartTheme.SUCCESS,
        "Top 25": ChartTheme.PRIMARY,
        "Top 50": ChartTheme.ACCENT,
    }

    fig = go.Figure()
    for tier_name in ["Top 50", "Top 25", "Top 10"]:
        subset = tier_df[tier_df["tier"] == tier_name]
        if subset.empty:
            continue
        fig.add_trace(go.Bar(
            x=subset["feature"],
            y=subset["usage_count"],
            name=tier_name,
            marker_color=tier_colors.get(tier_name, ChartTheme.GRID),
        ))

    fig.update_layout(
        title="📊 Feature Usage by Fitness Tier",
        xaxis_title="Feature",
        yaxis_title="Usage Count",
        barmode="group",
        template="plotly_dark",
        paper_bgcolor=ChartTheme.PAPER,
        plot_bgcolor=ChartTheme.BACKGROUND,
        font=dict(color=ChartTheme.TEXT),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────

def render():
    """Entry point for the Feature Importance page."""
    st.header("📊 Feature Importance Analysis")

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        top_n = st.slider("Analyse top N strategies", min_value=10, max_value=200, value=50, step=10)
    with col2:
        min_fitness = st.number_input(
            "Min fitness threshold", 
            value=-1.0,  # ← was 0.0
            min_value=-1.0,
            max_value=1.0,
            step=0.1, 
            format="%.2f"
        )

    features_df = _load_feature_data(top_n=top_n, min_fitness=min_fitness)

    if features_df.empty:
        st.info(
            "No feature data available. This requires strategies in the database — "
            "run an evolution first via the Backtester page."
        )
        return

    # ── Top 20 features bar chart ──────────────────────────────────────────
    fig_bar = create_feature_importance_bar(features_df)
    st.plotly_chart(fig_bar,width='stretch')

    # ── Category breakdown pie chart ───────────────────────────────────────
    features_df["category"] = features_df["feature"].apply(_categorize_feature)
    cat_df = (
        features_df.groupby("category")["usage_count"]
        .sum()
        .reset_index()
        .rename(columns={"usage_count": "count"})
        .sort_values("count", ascending=False)
    )

    col_pie, col_table = st.columns([1, 1])
    with col_pie:
        fig_pie = create_feature_category_pie(cat_df)
        st.plotly_chart(fig_pie,width='stretch')
    with col_table:
        st.subheader("Category Summary")
        st.dataframe(cat_df,width='stretch', hide_index=True)

    # ── Feature usage by fitness tier ──────────────────────────────────────
    st.subheader("Feature Usage by Fitness Tier")
    tier_df = _load_feature_by_tier(min_fitness=min_fitness)
    if not tier_df.empty:
        fig_tier = _build_tier_chart(tier_df)
        st.plotly_chart(fig_tier,width='stretch')
    else:
        st.info("Not enough data for tier analysis.")

    # ── Raw data table ─────────────────────────────────────────────────────
    with st.expander("📋 Raw Feature Data"):
        display_df = features_df[["feature", "usage_count", "usage_pct", "avg_fitness", "max_fitness", "weighted_score"]].copy()
        display_df.columns = ["Feature", "Usage", "Usage %", "Avg Fitness", "Max Fitness", "Weighted Score"]
        st.dataframe(display_df,width='stretch', hide_index=True)
