#!/usr/bin/env python3
"""
ALPHAGENE — Backtester Controls Page

Streamlit page for configuring and launching GP evolution runs,
with live progress monitoring via file-based IPC.
"""

import sys
import json
import time
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from evolution.progress import EvolutionProgressReporter
from ui.charts import create_fitness_evolution_chart, ChartTheme


PROGRESS_FILE = "data/cache/evolution_progress.json"
RUNNER_SCRIPT = "arena_runner_v3.py"


# ─────────────────────────────────────────────────────────────────────────────
# Render
# ─────────────────────────────────────────────────────────────────────────────

def render():
    """Entry point for the Backtester Controls page."""
    st.header("🧪 Backtester Controls")

    # ── Universe selector ──────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        universe = st.selectbox(
            "Universe",
            ["Oil Expanded", "Oil Legacy", "S&P 500", "Custom"],
            index=0,
        )
    with col2:
        custom_tickers = ""
        if universe == "Custom":
            custom_tickers = st.text_input(
                "Custom Tickers (space-separated)", value="AAPL MSFT GOOG"
            )

    # ── Date range ─────────────────────────────────────────────────────────
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input("Start Date", value=None)
        if start_date is None:
            import datetime as _dt
            start_date = _dt.date(2020, 1, 1)
    with col_d2:
        end_date = st.date_input("End Date", value=None)

    # ── Walk-forward params ────────────────────────────────────────────────
    st.subheader("Walk-Forward Parameters")
    wf1, wf2, wf3 = st.columns(3)
    with wf1:
        train_months = st.number_input("Train Months", min_value=3, max_value=60, value=12)
    with wf2:
        test_months = st.number_input("Test Months", min_value=1, max_value=24, value=3)
    with wf3:
        step_months = st.number_input("Step Months", min_value=1, max_value=12, value=3)

    # ── GP params ──────────────────────────────────────────────────────────
    st.subheader("GP Parameters")
    gp1, gp2, gp3 = st.columns(3)
    with gp1:
        population = st.number_input("Population", min_value=10, max_value=500, value=50)
    with gp2:
        generations = st.number_input("Generations", min_value=1, max_value=200, value=30)
    with gp3:
        max_depth = st.number_input("Max Depth", min_value=2, max_value=12, value=5)

    # ── Feature toggles ───────────────────────────────────────────────────
    st.subheader("Feature Toggles")
    tog1, tog2, tog3, tog4, tog5, tog6 = st.columns(6)
    with tog1:
        recency_weighting = st.checkbox("Recency Weighting", value=False)
        expanding_window = st.checkbox("Expanding Window", value=False)
    with tog2:
        oil_features = st.checkbox("Oil Features", value=True)
        calmar_fitness = st.checkbox("Calmar Fitness", value=False)
    with tog3:
        trailing_stops = st.checkbox("Trailing Stops", value=False)
        kelly_sizing = st.checkbox("Kelly Sizing", value=False)
    with tog4:
        fitness_v2 = st.checkbox("Fitness v2", value=False)
    with tog5:
        enable_smc = st.checkbox("Smart Money Concepts", value=False)
    with tog6:
        enable_sr = st.checkbox("Support/Resistance", value=False)

    st.divider()

    # ── Run button ─────────────────────────────────────────────────────────
    if st.button("🚀 Launch Evolution", type="primary", use_container_width=True):
        cmd = [sys.executable, RUNNER_SCRIPT]

        # Universe            
        if universe in ["Oil Expanded", "Oil Legacy"]:
            cmd += ["--universe-type", "oil_microcap"]
            cmd += ["--universe", "oil"]
        elif universe == "S&P 500":
            pass  # default universe
        elif universe == "Custom" and custom_tickers.strip():
            cmd += ["--tickers"] + custom_tickers.strip().split()

        # Dates
        cmd += ["--start", str(start_date)]
        if end_date:
            cmd += ["--end", str(end_date)]

        # Walk-forward
        cmd += [
            "--train-months", str(train_months),
            "--test-months", str(test_months),
            "--step-months", str(step_months),
        ]

        # GP
        cmd += [
            "--population", str(population),
            "--generations", str(generations),
            "--max-depth", str(max_depth),
        ]

        # Toggles
        if enable_smc:
            cmd.append("--enable-smc")
        if enable_sr:
            cmd.append("--enable-sr")
        if trailing_stops:
            cmd.append("--use-stops")
        if kelly_sizing:
            cmd.append("--use-kelly")
        if calmar_fitness:
            cmd.append("--use-calmar-fitness")
        if oil_features:
            cmd.append("--enable-oil")
        if fitness_v2:
            cmd.append("--use-fitness-v2")
        if expanding_window:
            cmd.append("--expanding-window")
        if recency_weighting and not fitness_v2:
            # Recency weighting is part of fitness v2; enable it
            cmd.append("--use-fitness-v2")

        # Clear old progress
        reporter = EvolutionProgressReporter(PROGRESS_FILE)
        reporter.clear()

        st.info(f"Launching: `{' '.join(cmd)}`")

        # Launch as detached subprocess
        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            st.success("Evolution process started! Monitor progress below.")
        except Exception as exc:
            st.error(f"Failed to launch: {exc}")

    st.divider()

    # ── Live progress display ──────────────────────────────────────────────
    st.subheader("📡 Evolution Progress")

    reporter = EvolutionProgressReporter(PROGRESS_FILE)
    progress_data = reporter.read_progress()

    if progress_data is None:
        st.info("No evolution in progress. Launch one above or wait for data.")
        return

    current = progress_data.get("current", {})
    history = progress_data.get("history", [])

    status = current.get("status", "unknown")
    gen = current.get("generation", 0)
    total = current.get("total_generations", 30)

    if status == "running":
        st.progress(min(gen / max(total, 1), 1.0), text=f"Generation {gen}/{total}")
    elif status == "complete":
        st.success(f"✅ Evolution complete — {gen} generations")
    else:
        st.warning(f"Status: {status}")

    # KPI row
    if current:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Best Fitness", f"{current.get('best_fitness', 0):+.4f}")
        with c2:
            st.metric("Avg Fitness", f"{current.get('avg_fitness', 0):+.4f}")
        with c3:
            st.metric("Stagnant", str(current.get("stagnant", 0)))
        with c4:
            st.metric("Generation", f"{gen}/{total}")

    # Fitness evolution chart
    if history:
        fig = create_fitness_evolution_chart(history)
        st.plotly_chart(fig, use_container_width=True)

    # Auto-refresh while running
    if status == "running":
        st.caption("🔄 Auto-refreshing every 2 seconds…")
        time.sleep(2)
        st.rerun()
