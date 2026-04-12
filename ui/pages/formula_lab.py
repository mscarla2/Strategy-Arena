#!/usr/bin/env python3
"""
ALPHAGENE — Formula Lab Page

Enter a custom GP formula string, configure a universe and date range,
then run a full walk-forward backtest and see the performance dashboard.

Benchmark selection is automatic:
  - Oil universe   → XLE
  - S&P 500 subset → VTI
"""

import sys
import datetime as _dt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from evolution.formula_parser import parse_formula
from evolution.strategy import GPStrategy
from evolution.walkforward import WalkForwardEvaluator
from data.fetcher import DataFetcher
from data.universe import get_oil_universe, get_oil_tradeable_tickers

from ui.charts import (
    ChartTheme,
    create_alpha_heatmap,
)
import plotly.graph_objects as go


# ─────────────────────────────────────────────────────────────────────────────
# Constants / defaults
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLE_FORMULA = "tanh(avg(zscore(rank(mom_21d)), avg(neg(zscore(rank(vol_21d))), zscore(rank(trend_r2_63d)))))"

OIL_UNIVERSE_LABEL = "Oil (expanded)"
SP500_LABEL = "S&P 500 (subset)"

UNIVERSE_OPTIONS = [OIL_UNIVERSE_LABEL, SP500_LABEL]

# Benchmark ticker per universe
UNIVERSE_BENCHMARK = {
    OIL_UNIVERSE_LABEL: "XLE",
    SP500_LABEL: "VTI",
}

SP500_SUBSET = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY",
    "JPM", "V", "UNH", "XOM", "JNJ", "AVGO", "MA", "PG", "HD", "CVX",
    "MRK", "ABBV", "PEP", "COST", "KO", "WMT", "CSCO", "TMO", "BAC",
]

# Oil-specific features that are only available with enable_oil=True
OIL_ONLY_FEATURES = frozenset({
    "oil_brent_correlation", "oil_crack_spread_321", "oil_crack_spread_532",
    "oil_inventory_change", "oil_inventory_zscore", "oil_seasonal_driving",
    "oil_seasonal_heating", "oil_wti_beta", "oil_wti_brent_spread",
    "oil_wti_correlation",
})


def _detect_oil_features_in_formula(formula: str) -> list[str]:
    """Return list of oil-only feature names found in the formula string."""
    return [f for f in OIL_ONLY_FEATURES if f in formula]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _generate_wf_periods(
    start: str,
    end: str,
    train_months: int,
    test_months: int,
    step_months: int,
) -> list:
    periods = []
    cur = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while cur + pd.DateOffset(months=train_months + test_months) <= end_ts:
        ts = cur.strftime("%Y-%m-%d")
        te = (cur + pd.DateOffset(months=train_months)).strftime("%Y-%m-%d")
        periods.append((ts, te, te,
                        (cur + pd.DateOffset(months=train_months + test_months)).strftime("%Y-%m-%d")))
        cur += pd.DateOffset(months=step_months)

    # anchored recent period
    recent_te = end_ts
    recent_ts = recent_te - pd.DateOffset(months=test_months)
    recent_train_e = recent_ts
    recent_train_s = recent_train_e - pd.DateOffset(months=train_months)
    anchored = (
        recent_train_s.strftime("%Y-%m-%d"),
        recent_train_e.strftime("%Y-%m-%d"),
        recent_ts.strftime("%Y-%m-%d"),
        recent_te.strftime("%Y-%m-%d"),
    )
    if not periods or anchored != periods[-1]:
        periods.append(anchored)
    return periods


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_prices(tickers_key: str, tickers: tuple, start: str, end: str):
    """Download prices via DataFetcher (cached)."""
    fetcher = DataFetcher()
    prices, volume = fetcher.fetch(start_date=start, end_date=end, tickers=list(tickers))
    prices = prices.dropna(axis=1, how="all").ffill().bfill()
    if volume is not None and not volume.empty:
        vcols = [c for c in prices.columns if c in volume.columns]
        volume = volume[vcols].reindex(prices.index).ffill().bfill().fillna(0)
    else:
        volume = None
    return prices, volume


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_benchmark_periods(ticker: str, period_starts: tuple, period_ends: tuple) -> list:
    """Fetch benchmark ETF returns aligned to walk-forward test windows."""
    if not period_starts:
        return []
    try:
        start = (pd.Timestamp(min(period_starts)) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        end = (pd.Timestamp(max(period_ends)) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        prices = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)["Close"]
        if prices is None or prices.empty:
            return []
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


def _run_backtest(
    formula: str,
    tickers: list,
    tradeable_tickers: list | None,
    benchmark_ticker: str,
    start: str,
    end: str,
    train_months: int,
    test_months: int,
    step_months: int,
    top_pct: float,
    portfolio_stop_pct: float | None,
    rebalance_days: int,
    rebalance_threshold: float,  # 0.0 = always full rebalance; >0 = partial
    enable_oil: bool = False,
) -> dict:
    """Run walk-forward backtest for a parsed formula. Returns dict with results."""
    # 1) Parse formula
    tree = parse_formula(formula)
    strategy = GPStrategy(tree=tree, top_pct=top_pct, enable_oil=enable_oil)

    # 2) Generate periods
    periods = _generate_wf_periods(start, end, train_months, test_months, step_months)
    if not periods:
        return {"error": "No valid walk-forward periods for the given date range."}

    # 3) Load prices
    data_start = (pd.Timestamp(periods[0][2]) - pd.Timedelta(days=500)).strftime("%Y-%m-%d")
    tickers_key = "_".join(sorted(tickers[:5]))  # for cache key
    with st.spinner("Loading price data…"):
        try:
            prices, volume = _fetch_prices(tickers_key, tuple(sorted(tickers)), data_start, end)
        except Exception as exc:
            return {"error": f"Data fetch failed: {exc}"}

    if len(prices) < 252:
        return {"error": f"Insufficient data: only {len(prices)} days available."}

    # 4) Evaluate
    # Optional partial rebalancing: only trade positions that drift by > threshold
    if rebalance_threshold > 0:
        from backtest.rebalancing import PartialRebalancer
        rebalancer = PartialRebalancer(deviation_threshold=rebalance_threshold)
    else:
        rebalancer = None

    evaluator = WalkForwardEvaluator(
        prices=prices,
        periods=periods,
        benchmark_results=[],
        tradeable_tickers=tradeable_tickers,
        volume=volume,
        portfolio_stop_pct=portfolio_stop_pct,
        rebalance_frequency=rebalance_days,
        rebalancer=rebalancer,
    )
    with st.spinner("Running walk-forward backtest…"):
        try:
            fitness_result = evaluator.evaluate_strategy(strategy)
        except Exception as exc:
            return {"error": f"Backtest failed: {exc}"}

    period_results = fitness_result.period_results if fitness_result.period_results else []

    # 5) Fetch benchmark
    p_starts = tuple(p.get("period_start", p.get("test_start", "")) for p in period_results)
    p_ends = tuple(p.get("period_end", p.get("test_end", "")) for p in period_results)
    with st.spinner(f"Fetching {benchmark_ticker} benchmark…"):
        bench_periods = _fetch_benchmark_periods(benchmark_ticker, p_starts, p_ends)

    return {
        "formula": formula,
        "fitness": fitness_result.total,
        "avg_sharpe": fitness_result.avg_sharpe,
        "avg_return": fitness_result.avg_return,
        "win_rate": fitness_result.win_rate,
        "period_results": period_results,
        "bench_periods": bench_periods,
        "bench_ticker": benchmark_ticker,
        "n_periods": fitness_result.n_periods,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_portfolio_chart(
    period_results: list,
    bench_periods: list,
    bench_ticker: str,
    base_capital: float,
):
    fig = go.Figure()
    if not period_results:
        fig.update_layout(title="No data", template="plotly_dark",
                          paper_bgcolor=ChartTheme.PAPER, plot_bgcolor=ChartTheme.BACKGROUND)
        return fig

    def _ps(p):
        return p.get("period_start", p.get("test_start", ""))

    sorted_periods = sorted(period_results, key=_ps)
    dates = [_ps(p) for p in sorted_periods]
    rets = [p.get("total_return", 0) for p in sorted_periods]
    # Cumulative return as %
    cum_pct = (np.cumprod([1 + r for r in rets]) - 1) * 100

    fig.add_trace(go.Scatter(
        x=dates, y=cum_pct, name="Strategy",
        line=dict(color=ChartTheme.PRIMARY, width=3),
        mode="lines+markers", marker=dict(size=6),
        hovertemplate="%{x}<br>%{y:.1f}%<extra>Strategy</extra>",
    ))

    if bench_periods:
        b_sorted = sorted(bench_periods, key=lambda x: x["period_start"])
        b_dates = [x["period_start"] for x in b_sorted]
        b_cum_pct = (np.cumprod([1 + x["total_return"] for x in b_sorted]) - 1) * 100
        fig.add_trace(go.Scatter(
            x=b_dates, y=b_cum_pct, name=bench_ticker,
            line=dict(color=ChartTheme.WARNING, width=2, dash="dash"),
            mode="lines+markers", marker=dict(size=4),
            hovertemplate=f"%{{x}}<br>%{{y:.1f}}%<extra>{bench_ticker}</extra>",
        ))

    fig.update_layout(
        title=f"📈 Cumulative Return vs {bench_ticker}",
        xaxis_title="Period",
        yaxis_title="Cumulative Return (%)",
        yaxis=dict(ticksuffix="%"),
        template="plotly_dark",
        paper_bgcolor=ChartTheme.PAPER,
        plot_bgcolor=ChartTheme.BACKGROUND,
        font=dict(color=ChartTheme.TEXT),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _build_period_table(
    period_results: list,
    bench_periods: list,
    bench_ticker: str,
) -> pd.DataFrame:
    def _ps(p):
        return p.get("period_start", p.get("test_start", ""))
    def _pe(p):
        return p.get("period_end", p.get("test_end", ""))

    bench_map = {x["period_start"]: x["total_return"] for x in (bench_periods or [])}
    rows = []
    for p in sorted(period_results, key=_ps):
        ps = _ps(p)
        ret = p.get("total_return", 0)
        rows.append({
            "Start": ps,
            "End": _pe(p),
            "Return": f"{ret:.1%}",
            f"vs {bench_ticker}": f"{ret - bench_map[ps]:+.1%}" if ps in bench_map else "—",
            "Sharpe": f"{p.get('sharpe_ratio', 0):.2f}",
            "Max DD": f"{p.get('max_drawdown', 0):.1%}",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Feature reference
# ─────────────────────────────────────────────────────────────────────────────

# Category → list of (feature_name, short_description) tuples
_FEATURE_CATEGORIES = {
    "Momentum": [
        ("mom_5d", "5-day return"), ("mom_10d", "10-day return"), ("mom_21d", "21-day return"),
        ("mom_63d", "63-day return"), ("mom_126d", "126-day return"), ("mom_252d", "252-day return"),
        ("mom_3_1", "3m–1m momentum"), ("mom_6_1", "6m–1m momentum"), ("mom_12_1", "12m–1m momentum"),
        ("mom_accel_short", "Short momentum acceleration"), ("mom_accel_medium", "Medium momentum acceleration"),
        ("mom_accel_long", "Long momentum acceleration"), ("mom_consistency_21d", "Momentum consistency 21d"),
        ("mom_consistency_63d", "Momentum consistency 63d"), ("mom_smoothness_21d", "Momentum smoothness 21d"),
        ("mom_smoothness_63d", "Momentum smoothness 63d"), ("mom_sharpe_21d", "Momentum Sharpe 21d"),
        ("mom_sharpe_63d", "Momentum Sharpe 63d"),
    ],
    "Volatility": [
        ("vol_5d", "5-day realized vol"), ("vol_21d", "21-day realized vol"),
        ("vol_63d", "63-day realized vol"), ("vol_252d", "252-day realized vol"),
        ("vol_ratio_5_21", "Vol ratio 5d/21d"), ("vol_ratio_21_63", "Vol ratio 21d/63d"),
        ("vol_ratio_63_252", "Vol ratio 63d/252d"), ("vol_of_vol_21d", "Vol of vol 21d"),
        ("vol_of_vol_63d", "Vol of vol 63d"), ("vol_regime", "Vol regime indicator"),
        ("vol_trend_21d", "Vol trend 21d"), ("vol_adj_mom_21d", "Vol-adjusted momentum 21d"),
        ("vol_adj_mom_63d", "Vol-adjusted momentum 63d"), ("rel_vol_21d", "Relative vol 21d"),
        ("rel_vol_63d", "Relative vol 63d"), ("downside_dev_21d", "Downside deviation 21d"),
        ("downside_dev_63d", "Downside deviation 63d"),
    ],
    "Risk-Adjusted Returns": [
        ("sharpe_21d", "Sharpe ratio 21d"), ("sharpe_63d", "Sharpe ratio 63d"),
        ("sharpe_126d", "Sharpe ratio 126d"), ("sortino_21d", "Sortino ratio 21d"),
        ("sortino_63d", "Sortino ratio 63d"), ("calmar_63d", "Calmar ratio 63d"),
        ("calmar_126d", "Calmar ratio 126d"), ("info_ratio_63d", "Information ratio 63d"),
        ("info_ratio_126d", "Information ratio 126d"),
    ],
    "Mean Reversion & Z-Score": [
        ("zscore_5d", "Z-score 5d"), ("zscore_10d", "Z-score 10d"),
        ("zscore_21d", "Z-score 21d"), ("zscore_63d", "Z-score 63d"),
        ("dist_ma_10", "Distance from 10d MA"), ("dist_ma_21", "Distance from 21d MA"),
        ("dist_ma_50", "Distance from 50d MA"), ("dist_ma_200", "Distance from 200d MA"),
        ("reversion_speed_10d", "Mean reversion speed 10d"), ("reversion_speed_21d", "Mean reversion speed 21d"),
        ("reversion_speed_63d", "Mean reversion speed 63d"),
        ("excess_mom_21d", "Excess momentum 21d"), ("excess_mom_63d", "Excess momentum 63d"),
    ],
    "Trend": [
        ("trend_r2_21d", "Trend R² 21d"), ("trend_r2_63d", "Trend R² 63d"),
        ("trend_r2_126d", "Trend R² 126d"), ("trend_slope_21d", "Trend slope 21d"),
        ("trend_slope_63d", "Trend slope 63d"), ("trend_deviation_21d", "Trend deviation 21d"),
        ("trend_deviation_63d", "Trend deviation 63d"), ("trend_strength_21d", "Trend strength 21d"),
        ("trend_strength_63d", "Trend strength 63d"), ("hurst_proxy_63d", "Hurst exponent proxy 63d"),
        ("hurst_proxy_126d", "Hurst exponent proxy 126d"), ("efficiency_ratio_21d", "Efficiency ratio 21d"),
        ("efficiency_ratio_63d", "Efficiency ratio 63d"), ("fractal_dim_63d", "Fractal dimension 63d"),
    ],
    "Drawdown & Recovery": [
        ("drawdown_21d", "Drawdown 21d"), ("drawdown_63d", "Drawdown 63d"),
        ("drawdown_126d", "Drawdown 126d"), ("drawdown_252d", "Drawdown 252d"),
        ("drawdown_duration_63d", "Drawdown duration 63d"), ("drawdown_duration_252d", "Drawdown duration 252d"),
        ("max_dd_63d", "Max drawdown 63d"), ("max_dd_126d", "Max drawdown 126d"),
        ("recovery_rate_63d", "Recovery rate 63d"), ("recovery_rate_126d", "Recovery rate 126d"),
        ("ulcer_index_63d", "Ulcer index 63d"), ("dd_adj_mom_63d", "DD-adjusted momentum 63d"),
    ],
    "Higher Moments": [
        ("skew_21d", "Skewness 21d"), ("skew_63d", "Skewness 63d"), ("skew_126d", "Skewness 126d"),
        ("kurt_21d", "Kurtosis 21d"), ("kurt_63d", "Kurtosis 63d"), ("kurt_126d", "Kurtosis 126d"),
        ("left_tail_21d", "Left tail risk 21d"), ("left_tail_63d", "Left tail risk 63d"),
        ("up_down_ratio_63d", "Up/down ratio 63d"), ("abs_return_21d", "Abs return 21d"),
        ("abs_return_63d", "Abs return 63d"),
    ],
    "Market-Relative": [
        ("market_beta_21d", "Beta vs market 21d"), ("market_beta_63d", "Beta vs market 63d"),
        ("market_beta_126d", "Beta vs market 126d"), ("beta_instability_63d", "Beta instability 63d"),
        ("idio_vol_21d", "Idiosyncratic vol 21d"), ("idio_vol_63d", "Idiosyncratic vol 63d"),
        ("sector_correlation_21d", "Sector correlation 21d"), ("sector_correlation_63d", "Sector correlation 63d"),
        ("universe_dispersion_21d", "Universe dispersion 21d"),
        ("rel_strength_21d", "Relative strength 21d"), ("rel_strength_63d", "Relative strength 63d"),
        ("rel_strength_126d", "Relative strength 126d"), ("rel_value_21d", "Relative value 21d"),
        ("rel_value_63d", "Relative value 63d"),
    ],
    "Microstructure": [
        ("amihud_21d", "Amihud illiquidity 21d"), ("amihud_63d", "Amihud illiquidity 63d"),
        ("roll_spread_21d", "Roll spread 21d"), ("kyle_lambda_21d", "Kyle's lambda 21d"),
        ("turnover_rate_21d", "Turnover rate 21d"), ("zero_return_days_21d", "Zero-return days 21d"),
        ("zero_return_days_63d", "Zero-return days 63d"),
    ],
    "Volume": [
        ("volume_ratio_5_21", "Volume ratio 5d/21d"), ("volume_trend_21d", "Volume trend 21d"),
        ("volume_volatility_21d", "Volume volatility 21d"), ("price_volume_corr_21d", "Price-volume correlation 21d"),
    ],
    "Price Level": [
        ("range_position_21d", "Range position 21d"), ("range_position_63d", "Range position 63d"),
        ("range_position_252d", "Range position 252d"), ("high_proximity_63d", "52w high proximity 63d"),
        ("high_proximity_252d", "52w high proximity 252d"), ("support_distance_21d", "Support distance 21d"),
        ("resistance_distance_21d", "Resistance distance 21d"), ("breakout_21d", "Breakout signal 21d"),
        ("breakout_63d", "Breakout signal 63d"),
    ],
    "Pattern": [
        ("rsi_14", "RSI 14"), ("rsi_28", "RSI 28"), ("return_autocorr_1d", "Return autocorrelation 1d"),
        ("return_autocorr_5d", "Return autocorrelation 5d"), ("gap_reversal_10d", "Gap reversal 10d"),
        ("streak_strength", "Streak strength"), ("breadth_50d", "Market breadth 50d"),
        ("breadth_200d", "Market breadth 200d"), ("correlation_spike_21d", "Correlation spike 21d"),
    ],
    "Regime": [
        ("vol_regime", "Volatility regime"), ("breadth_50d", "Breadth 50d"),
        ("breadth_200d", "Breadth 200d"), ("correlation_spike_21d", "Correlation spike 21d"),
    ],
}


def _render_feature_reference():
    """Render a categorised, searchable feature reference."""
    st.markdown("**Supported Features (137 total)**")
    search = st.text_input(
        "Search features",
        placeholder="e.g. mom, vol, sharpe …",
        label_visibility="collapsed",
    )

    search_lower = search.lower().strip() if search else ""

    for category, features in _FEATURE_CATEGORIES.items():
        if search_lower:
            matched = [(n, d) for n, d in features if search_lower in n.lower() or search_lower in d.lower()]
        else:
            matched = features

        if not matched:
            continue

        with st.expander(f"{category} ({len(matched)})", expanded=bool(search_lower)):
            rows = []
            for name, desc in matched:
                rows.append(f"`{name}` — {desc}")
            # 2-column layout
            mid = (len(rows) + 1) // 2
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("\n\n".join(rows[:mid]))
            with c2:
                st.markdown("\n\n".join(rows[mid:]))


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────

def render():
    """Entry point for the Formula Lab page."""
    st.header("🧪 Formula Lab")
    st.caption("Enter a GP formula, configure the backtest, and see the full performance dashboard.")

    left_col, right_col = st.columns([1, 2], gap="large")

    # ── LEFT PANEL ─────────────────────────────────────────────────────────
    with left_col:
        st.subheader("Formula")
        formula = st.text_area(
            "Formula",
            value=EXAMPLE_FORMULA,
            height=160,
            label_visibility="collapsed",
            help=(
                "Unary ops: neg abs sign sqrt square inv log sigmoid tanh rank zscore\n"
                "Binary ops: add sub mul div max min avg\n"
                "Unary style:  op(expr)\n"
                "Binary style: op(expr, expr)  or  (expr op expr)\n"
                "Conditional:  if(expr > 0, expr, expr)"
            ),
            placeholder="e.g. tanh(avg(zscore(rank(mom_21d)), zscore(rank(vol_21d))))",
        )

        # Validate formula on the fly
        if formula.strip():
            try:
                _tree = parse_formula(formula.strip())
                st.success(f"✓ Valid — {_tree.size()} nodes, depth {_tree.depth()}")
            except ValueError as e:
                st.error(f"Parse error: {e}")

        st.divider()
        st.subheader("Universe")
        universe_choice = st.radio(
            "Universe",
            UNIVERSE_OPTIONS,
            label_visibility="collapsed",
        )
        bench_ticker = UNIVERSE_BENCHMARK[universe_choice]
        st.caption(f"Benchmark: **{bench_ticker}**")

        # Warn if formula uses oil-only features (not supported in Formula Lab interactive mode)
        if formula.strip():
            oil_feats_used = _detect_oil_features_in_formula(formula.strip())
            if oil_feats_used:
                st.warning(
                    f"⚠️ Your formula uses oil-only features: "
                    f"`{'`, `'.join(oil_feats_used)}`.\n\n"
                    f"Oil-specific features require live commodity futures downloads "
                    f"(WTI, Brent, etc.) on every rebalance day, which is too slow "
                    f"for interactive use. **These features will return 0 for all stocks**, "
                    f"which flattens scores and makes `neg()` and similar wrappers have no effect.\n\n"
                    f"Remove these features from your formula, or use the arena runner "
                    f"(`arena_runner_v3.py --enable-oil`) for oil-feature backtesting."
                )

        top_pct = st.slider("Top % to hold", min_value=5, max_value=50, value=20, step=5)

        st.divider()
        st.subheader("Date Range")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start_date = st.date_input("Start", value=_dt.date(2020, 1, 1))
        with col_d2:
            end_date = st.date_input("End", value=_dt.date.today())

        st.divider()
        st.subheader("Walk-Forward")
        wf1, wf2, wf3 = st.columns(3)
        with wf1:
            train_months = st.number_input("Train (months)", min_value=3, max_value=60, value=12)
        with wf2:
            test_months = st.number_input("Test (months)", min_value=1, max_value=24, value=3)
        with wf3:
            step_months = st.number_input("Step (months)", min_value=1, max_value=12, value=3)

        st.divider()
        st.subheader("Rebalancing")
        rebalance_days = st.slider(
            "Rebalance every N trading days",
            min_value=1,
            max_value=63,
            value=21,
            step=1,
            help="How often to re-score stocks and adjust the portfolio. 1 = daily, 5 = weekly, 21 = monthly.",
        )
        partial_rebalance = st.checkbox(
            "Partial rebalance (threshold)",
            value=False,
            help=(
                "When enabled, only positions that have drifted more than the threshold "
                "% from their target weight are traded. This significantly reduces turnover."
            ),
        )
        if partial_rebalance:
            rebalance_threshold = st.slider(
                "Drift threshold (%)",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
                format="%d%%",
                help="Only trade a position if its weight drifts more than this % from target.",
            ) / 100.0
        else:
            rebalance_threshold = 0.0

        st.divider()
        base_capital = st.number_input(
            "Base Capital ($)",
            min_value=1_000,
            max_value=100_000_000,
            value=100_000,
            step=10_000,
            format="%d",
        )

        st.divider()
        st.subheader("Risk Management")
        use_stop = st.checkbox(
            "Portfolio stop-loss",
            value=False,
            help=(
                "When enabled, the entire portfolio moves to cash if it falls "
                "more than the threshold % below its running high-water mark. "
                "Re-entry happens on the next rebalance day after a new high is reached."
            ),
        )
        if use_stop:
            stop_pct = st.slider(
                "Stop threshold (%)",
                min_value=2,
                max_value=30,
                value=10,
                step=1,
                format="%d%%",
                help="Exit to cash when portfolio drawdown from peak exceeds this level.",
            ) / 100.0
        else:
            stop_pct = None

        run_btn = st.button("▶ Run Backtest", type="primary", use_container_width=True)

    # ── RIGHT PANEL ────────────────────────────────────────────────────────
    with right_col:
        if not run_btn:
            st.info(
                f"Configure your formula and parameters on the left, then click **▶ Run Backtest**.\n\n"
                f"Benchmark for **{universe_choice}**: **{bench_ticker}**"
            )

            st.markdown("**Supported operators**")
            ops_cols = st.columns(2)
            with ops_cols[0]:
                st.markdown(
                    "**Unary** — `op(expr)`\n"
                    "```\nneg  abs  sign  sqrt  square\n"
                    "inv  log  sigmoid  tanh\n"
                    "rank  zscore\n```"
                )
            with ops_cols[1]:
                st.markdown(
                    "**Binary** — `op(a, b)` or `(a op b)`\n"
                    "```\nadd  sub  mul  div\n"
                    "max  min  avg\n```\n\n"
                    "**Conditional**\n"
                    "```\nif(cond > 0, then, else)\n```"
                )

            st.divider()
            _render_feature_reference()
            return

        # Resolve universe
        # Note: enable_oil is intentionally NOT passed — oil-specific features
        # require live commodity futures downloads on every rebalance, which
        # makes the backtest extremely slow in an interactive context.
        # Oil-feature names in a formula simply score 0 for all stocks (a warning
        # is shown in the left panel when this is detected).
        if universe_choice == OIL_UNIVERSE_LABEL:
            tickers = get_oil_universe(expanded=True)
            tradeable = get_oil_tradeable_tickers()
        else:
            tickers = SP500_SUBSET
            tradeable = None

        result = _run_backtest(
            formula=formula.strip(),
            tickers=tickers,
            tradeable_tickers=tradeable,
            benchmark_ticker=bench_ticker,
            start=str(start_date),
            end=str(end_date),
            train_months=int(train_months),
            test_months=int(test_months),
            step_months=int(step_months),
            top_pct=float(top_pct),
            portfolio_stop_pct=stop_pct,
            rebalance_days=int(rebalance_days),
            rebalance_threshold=float(rebalance_threshold),
            enable_oil=False,
        )

        if "error" in result:
            st.error(result["error"])
            return

        period_results = result["period_results"]
        bench_periods = result["bench_periods"]
        bench_ticker = result["bench_ticker"]

        if not period_results:
            st.warning("Backtest returned no period results. Try a longer date range or simpler formula.")
            return

        # ── KPI cards
        rets = [p.get("total_return", 0) for p in period_results]
        cum_return = float(np.prod([1 + r for r in rets]) - 1)
        final_value = base_capital * (1 + cum_return)

        bench_avg = np.mean([x["total_return"] for x in bench_periods]) if bench_periods else None
        avg_ret = result["avg_return"]
        alpha_vs_bench = avg_ret - bench_avg if bench_avg is not None else None

        max_dd = max((p.get("max_drawdown", 0) for p in period_results), default=0)
        annual_ret = avg_ret * 4
        calmar = annual_ret / max_dd if max_dd > 0 else 0

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        with k1:
            st.metric("Fitness", f"{result['fitness']:+.3f}")
        with k2:
            st.metric("Cumul. Return", f"{cum_return * 100:+.1f}%")
        with k3:
            alpha_str = f"{alpha_vs_bench:+.1%}" if alpha_vs_bench is not None else "N/A"
            st.metric(f"Alpha vs {bench_ticker}", alpha_str)
        with k4:
            st.metric("Sharpe", f"{result['avg_sharpe']:.2f}")
        with k5:
            st.metric("Max DD", f"{max_dd:.1%}")
        with k6:
            st.metric("Calmar", f"{calmar:.2f}")

        st.caption(
            f"\\${base_capital:,.0f} starting capital → **\\${final_value:,.0f}** "
            f"({cum_return:+.1%} cumulative, {result['n_periods']} periods)"
        )

        st.divider()

        # Portfolio value chart
        fig_pv = _build_portfolio_chart(period_results, bench_periods, bench_ticker, base_capital)
        st.plotly_chart(fig_pv, use_container_width=True)

        # Alpha heatmap vs benchmark
        if bench_periods:
            norm_results = [
                {"period_start": p.get("period_start", p.get("test_start", "")),
                 "total_return": p.get("total_return", 0)}
                for p in period_results
            ]
            fig_heat = create_alpha_heatmap(norm_results, bench_periods)
            fig_heat.update_layout(title=f"📅 Quarterly Alpha vs {bench_ticker}")
            st.plotly_chart(fig_heat, use_container_width=True)

        # Period details table
        st.subheader("📋 Period Details")
        df_table = _build_period_table(period_results, bench_periods, bench_ticker)
        if not df_table.empty:
            st.dataframe(df_table, use_container_width=True, hide_index=True)
