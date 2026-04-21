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
        "ticker":         t.ticker,
        "entry_ts":       t.entry_ts,
        "entry_price":    t.entry_price,
        "exit_price":     t.exit_price or t.entry_price,
        "pnl_pct":        t.pnl_pct,
        "outcome":        t.outcome,
        "hold_bars":      t.hold_bars,
        "session":        t.session_type.value,
        "pt_pct":         t.profit_target_pct,
        "sl_pct":         t.stop_loss_pct,
        # Analysis tags — required for pattern-type and entry-attempt breakdowns
        "pattern_type":   t.pattern_type,
        "entry_attempt":  t.entry_attempt,
        "support_source": t.support_source,
        "bars_since_pattern": t.bars_since_pattern,
    } for t in trades]

    df = pd.DataFrame(rows)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"])
    df.sort_values("entry_ts", inplace=True)
    return df


# ---------------------------------------------------------------------------
# Risk metrics helpers
# ---------------------------------------------------------------------------

def _sharpe(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio from per-trade PnL% returns."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free
    std = float(excess.std())
    if std == 0:
        return 0.0
    # Assume ~252 trading days, ~4 trades/day max → scale by sqrt(252)
    return round(float(excess.mean()) / std * (252 ** 0.5), 3)


def _sortino(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Annualised Sortino ratio — penalises only downside volatility."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    downside_std = float(downside.std())
    if downside_std == 0:
        return 0.0
    return round(float(excess.mean()) / downside_std * (252 ** 0.5), 3)


def _max_drawdown(cumulative_pnl: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a percentage."""
    if cumulative_pnl.empty:
        return 0.0
    peak = cumulative_pnl.cummax()
    dd = cumulative_pnl - peak
    return round(float(dd.min()), 2)


def _pattern_type_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-pattern-type breakdown table."""
    if "pattern_type" not in df.columns or df["pattern_type"].isna().all():
        return pd.DataFrame()
    rows = []
    for ptype, grp in df.groupby("pattern_type"):
        total = len(grp)
        wins  = (grp["outcome"] == "win").sum()
        wr    = wins / total if total else 0
        avg_pnl = grp["pnl_pct"].mean()
        rows.append({
            "Pattern Type":  ptype,
            "Trades":        total,
            "Win Rate":      round(wr, 3),
            "Avg PnL %":     round(float(avg_pnl), 2),
            "Expectancy":    round(float(wr * grp.loc[grp["outcome"]=="win","pnl_pct"].mean() +
                                  (1-wr) * grp.loc[grp["outcome"]=="loss","pnl_pct"].mean()
                                  if (grp["outcome"]=="win").any() and (grp["outcome"]=="loss").any()
                                  else avg_pnl), 2),
        })
    return pd.DataFrame(rows).sort_values("Trades", ascending=False)


def _entry_attempt_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-entry-attempt breakdown: 1st touch vs 2nd vs 3rd+."""
    if "entry_attempt" not in df.columns or df["entry_attempt"].isna().all():
        return pd.DataFrame()
    df2 = df.copy()
    df2["attempt_group"] = df2["entry_attempt"].apply(
        lambda x: "1st touch" if x == 1 else ("2nd touch" if x == 2 else "3rd+ touch")
    )
    rows = []
    for grp_label, grp in df2.groupby("attempt_group"):
        total = len(grp)
        wins  = (grp["outcome"] == "win").sum()
        wr    = wins / total if total else 0
        rows.append({
            "Entry Attempt": grp_label,
            "Trades":        total,
            "Win Rate":      round(wr, 3),
            "Avg PnL %":     round(float(grp["pnl_pct"].mean()), 2),
        })
    order = {"1st touch": 0, "2nd touch": 1, "3rd+ touch": 2}
    result = pd.DataFrame(rows)
    if not result.empty:
        result["_ord"] = result["Entry Attempt"].map(order)
        result = result.sort_values("_ord").drop(columns=["_ord"])
    return result


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
        from side_by_side_backtest.data_fetcher import _silence_yfinance
        with _silence_yfinance():
            raw = yf.download(
                _BENCHMARK_TICKER, start=start_date, end=end_date,
                interval="1d", progress=False, auto_adjust=True,
                threads=False,
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

    df_all = _load_trades()
    if df_all.empty:
        st.warning("No trade history found. Run the backtest pipeline first:\n"
                   "```\npython -m side_by_side_backtest.main --watchlist scraped_watchlists.json "
                   "--skip-fetch --auto-tune --n-trials 100 --export\n```")
        return

    # ── Sidebar filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.header("📊 Filters")
        if "session" in df_all.columns:
            session_opts = ["All sessions"] + sorted(df_all["session"].dropna().unique().tolist())
        else:
            session_opts = ["All sessions"]
        session_sel = st.selectbox("Session type", session_opts, index=0)
        outcome_sel = st.selectbox("Outcome filter", ["All outcomes", "win", "loss", "timeout"], index=0)

    # Apply filters
    df = df_all.copy()
    if session_sel != "All sessions" and "session" in df.columns:
        df = df[df["session"] == session_sel]
    if outcome_sel != "All outcomes":
        df = df[df["outcome"] == outcome_sel]

    if df.empty:
        st.warning("No trades match the selected filters.")
        return

    m       = _overall_metrics(df)
    sharpe  = _sharpe(df["pnl_pct"])
    sortino = _sortino(df["pnl_pct"])
    cum_pnl = df["pnl_pct"].cumsum()
    mdd     = _max_drawdown(cum_pnl)

    # ── Overall metrics ───────────────────────────────────────────────────────
    st.subheader("Overall Summary")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Trades",    m["total"])
    c2.metric("Win Rate",        f"{m['wr']:.1%}")
    c3.metric("Expectancy",      f"{m['expectancy']:.2f}%")
    c4.metric("Profit Factor",   m["profit_factor"])
    c5.metric("Avg Win",         f"{m['avg_win']:.2f}%")
    c6.metric("Avg Loss",        f"{m['avg_loss']:.2f}%")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Sharpe Ratio",    sharpe)
    r2.metric("Sortino Ratio",   sortino)
    r3.metric("Max Drawdown",    f"{mdd:.2f}%")
    r4.metric("Total PnL %",     f"{m['total_pnl']:.2f}%")

    st.plotly_chart(_equity_curve(df), width='stretch')

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(_pnl_histogram(df), width='stretch')

    with col_b:
        st.subheader("🏆 Best Plays (by Expectancy)")
        stats = _per_ticker_stats(df)
        st.dataframe(stats, width='stretch', hide_index=True)

    # ── Drawdown curve ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📉 Drawdown Curve")
    if not cum_pnl.empty:
        peak      = cum_pnl.cummax()
        dd_series = cum_pnl - peak
        dd_fig    = go.Figure()
        dd_fig.add_trace(go.Scatter(
            x=df["entry_ts"].values, y=dd_series.values,
            fill="tozeroy", fillcolor="rgba(239,83,80,0.25)",
            line=dict(color="#ef5350", width=1.5),
            name="Drawdown %",
        ))
        dd_fig.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font=dict(color="#cccccc"), height=200,
            xaxis=dict(gridcolor="#1e2530"),
            yaxis=dict(gridcolor="#1e2530", title="Drawdown %"),
            margin=dict(t=10, b=20),
        )
        st.plotly_chart(dd_fig, width='stretch')

    # ── Pattern-type segmentation ─────────────────────────────────────────────
    st.subheader("🔬 Pattern-Type Breakdown")
    pt_stats = _pattern_type_stats(df)
    if pt_stats.empty:
        st.info("Pattern-type data not yet available — re-run the backtest with the updated pipeline to populate this field.")
    else:
        st.dataframe(pt_stats, width='stretch', hide_index=True)
        st.caption("**strict** = classic S×S | **exhaustion** = doji/widening C3 | **absorption** = volume-absorbed | **eqh_breakout** = EQH ceiling break | **none** = bare support touch")

    # ── Entry-attempt segmentation ────────────────────────────────────────────
    st.subheader("🎯 Entry Attempt Breakdown")
    ea_stats = _entry_attempt_stats(df)
    if ea_stats.empty:
        st.info("Entry-attempt data not yet available — re-run the backtest to populate this field.")
    else:
        st.dataframe(ea_stats, width='stretch', hide_index=True)
        st.caption("**1st touch** = first support contact of the session — typically the highest-quality setup.")

    # ── Card Strategy backtest comparison ─────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Card Strategy Backtest")
    st.caption(
        "Replays every trading session in the 30-day bar history. "
        "Score ≥ threshold → enter at session open. "
        "**TP/SL come from each card's own levels** (resistance = TP, stop = SL). "
        "Budget rules mirror autonomous config: $500/trade · max 10 concurrent · $300/day halt."
    )
    c_score, c_budget, c_size, c_halt = st.columns(4)
    card_min_score  = c_score.number_input("Min score", min_value=0.0, max_value=10.0,
                                           value=4.3, step=0.1, key="card_min_score")
    card_budget     = c_budget.number_input("Budget $", min_value=100.0, max_value=100_000.0,
                                            value=5_000.0, step=500.0, key="card_budget")
    card_trade_size = c_size.number_input("Trade size $", min_value=50.0, max_value=10_000.0,
                                          value=500.0, step=50.0, key="card_trade_size")
    card_halt       = c_halt.number_input("Daily loss halt $", min_value=0.0, max_value=5_000.0,
                                          value=300.0, step=50.0, key="card_halt")

    if st.button("▶ Run Card Strategy Backtest", key="run_card_bt"):
        with st.spinner(f"Scoring every session (score ≥ {card_min_score})… this may take 2–5 min"):
            try:
                from side_by_side_backtest.card_strategy_simulator import simulate_card_strategy
                from side_by_side_backtest.data_fetcher import (
                    load_30day_bars,
                    fetch_bars_for_entry,
                    is_banned,
                )
                from side_by_side_backtest.parser import parse_scraped_file
                from side_by_side_backtest.db import WatchlistDB

                wl_path = Path(__file__).parent.parent.parent / "scraped_watchlists.json"
                entries = parse_scraped_file(wl_path)

                # Load bars from 30d cache; fall back to live fetch if parquet
                # doesn't exist yet (mirrors Morning Brief behaviour).
                tickers = list({e.ticker for e in entries})
                entry_by_ticker = {e.ticker: e for e in entries}
                bars_map = {}
                missing_cache: list[str] = []
                for t in tickers:
                    b = load_30day_bars(t)
                    if not b.empty:
                        bars_map[t] = b
                    else:
                        missing_cache.append(t)

                # Fallback: fetch bars live for tickers without a cached parquet.
                # Exclude banned tickers upfront so they never appear in the banner.
                missing_cache = [t for t in missing_cache if not is_banned(t)]
                if missing_cache:
                    st.info(
                        f"⚡ Fetching bars for {len(missing_cache)} ticker(s) "
                        f"not yet in cache: {', '.join(missing_cache[:10])}"
                        + (" …" if len(missing_cache) > 10 else "")
                    )
                    for t in missing_cache:
                        try:
                            entry_ref = entry_by_ticker.get(t)
                            if entry_ref is not None:
                                b = fetch_bars_for_entry(entry_ref)
                                if b is not None and not b.empty:
                                    bars_map[t] = b
                        except Exception as _fe:
                            st.warning(f"Could not fetch bars for {t}: {_fe}")

                max_conc = max(1, int(card_budget // card_trade_size))

                with WatchlistDB() as _db:
                    card_trades = simulate_card_strategy(
                        entries, bars_map,
                        min_score=card_min_score,
                        db=_db,
                        verbose=True,
                        budget_total=card_budget,
                        trade_size=card_trade_size,
                        max_concurrent=max_conc,
                        daily_loss_halt=card_halt,
                    )

                if card_trades:
                    ct = pd.DataFrame([{
                        "ticker":      t.ticker,
                        "entry_ts":    t.entry_ts,
                        "entry_price": t.entry_price,
                        "exit_price":  t.exit_price or t.entry_price,
                        "pnl_pct":     t.pnl_pct,
                        "outcome":     t.outcome,
                        "hold_bars":   t.hold_bars,
                        "pt_pct":      t.profit_target_pct,
                        "sl_pct":      t.stop_loss_pct,
                    } for t in card_trades])
                    ct.sort_values("entry_ts", inplace=True)

                    wins   = (ct["outcome"] == "win").sum()
                    losses = (ct["outcome"] == "loss").sum()
                    total  = len(ct)
                    wr     = wins / total if total else 0
                    aw     = ct.loc[ct["outcome"]=="win",  "pnl_pct"].mean() or 0
                    al     = ct.loc[ct["outcome"]=="loss", "pnl_pct"].mean() or 0
                    exp    = wr * aw + (1 - wr) * al

                    # Dollar PnL column
                    ct["dollar_pnl"] = (ct["pnl_pct"] / 100) * card_trade_size
                    ct["equity"]     = card_budget + ct["dollar_pnl"].cumsum()

                    r1, r2, r3, r4, r5 = st.columns(5)
                    r1.metric("Trades",       total)
                    r2.metric("Win Rate",     f"{wr:.1%}")
                    r3.metric("Expectancy",   f"{exp:.2f}%")
                    r4.metric("Total PnL %",  f"{ct['pnl_pct'].sum():.2f}%")
                    r5.metric("Total PnL $",  f"${ct['dollar_pnl'].sum():.2f}")

                    # Equity curve
                    eq_fig = go.Figure()
                    eq_fig.add_trace(go.Scatter(
                        x=ct["entry_ts"], y=ct["equity"],
                        mode="lines", name="Equity",
                        line=dict(color="#26a69a", width=2),
                    ))
                    eq_fig.update_layout(
                        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                        font=dict(color="#cccccc"), height=220,
                        xaxis=dict(gridcolor="#1e2530"),
                        yaxis=dict(gridcolor="#1e2530", title="Equity $"),
                        margin=dict(t=10, b=20),
                    )
                    st.plotly_chart(eq_fig, width='stretch')

                    st.dataframe(ct.rename(columns={
                        "ticker":"Ticker", "entry_ts":"Entry", "entry_price":"Entry $",
                        "exit_price":"Exit $", "pnl_pct":"PnL %", "outcome":"Outcome",
                        "hold_bars":"Bars", "pt_pct":"TP%", "sl_pct":"SL%",
                        "dollar_pnl":"PnL $",
                    }).drop(columns=["equity"], errors="ignore"),
                        width='stretch', hide_index=True)
                else:
                    st.warning(
                        f"No trades fired with score \u2265 {card_min_score}. "
                        f"**{len(bars_map)}/{len(tickers)} tickers had bars** "
                        f"({len(missing_cache)} fetched live). "
                        f"Check the terminal / logs for per-date component scores "
                        f"(pattern / adx / rr / confluence / history) \u2014 "
                        f"logging level INFO is required (`--log-level info`)."
                    )
                    if not bars_map:
                        st.error(
                            "\u274c No bar data found at all. "
                            "Go to **Morning Brief** and let it refresh tickers first, "
                            "or run `refresh_cache.py` to seed the 30-day parquet cache."
                        )
                    elif len(bars_map) < len(tickers):
                        st.info(
                            f"\u2139\ufe0f {len(tickers) - len(bars_map)} ticker(s) had no bars and were skipped: "
                            + ", ".join(t for t in tickers if t not in bars_map)
                        )
            except Exception as exc:
                import traceback
                st.error(f"Card strategy backtest error: {exc}")
                st.code(traceback.format_exc())

    # ── Deep-link to chart viewer ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Drill into a ticker")
    top_tickers = stats["Ticker"].tolist()[:20] if not stats.empty else []
    if top_tickers:
        pick = st.selectbox("Select ticker to view chart", top_tickers, key="perf_ticker")
        if st.button(f"📈 Open {pick} in Chart Viewer", key="perf_cv"):
            st.session_state["chart_ticker"] = pick
            st.switch_page("pages/2_chart_viewer.py")

    with st.expander("📋 Raw trade log"):
        st.dataframe(df.sort_values("entry_ts", ascending=False), width='stretch')


if __name__ == "__main__":
    main()
