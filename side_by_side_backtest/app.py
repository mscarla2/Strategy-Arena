"""
Side-by-Side Backtest — Multi-Page Streamlit App
=================================================
Entry point for the unified Streamlit application.

Pages
-----
  1. Morning Brief  — watchlist triage & ranked setup table
  2. Chart Viewer   — interactive 5-min candlestick explorer

Launch
------
    streamlit run side_by_side_backtest/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure package root is importable when launched from any directory
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_PAGES_DIR = Path(__file__).parent / "pages"

# ---------------------------------------------------------------------------
# Page registry
# ---------------------------------------------------------------------------

morning_brief = st.Page(
    str(_PAGES_DIR / "1_morning_brief.py"),
    title="Morning Brief",
    icon="📋",
    default=True,
)

chart_viewer = st.Page(
    str(_PAGES_DIR / "2_chart_viewer.py"),
    title="Chart Viewer",
    icon="📈",
)

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

pg = st.navigation([morning_brief, chart_viewer])
pg.run()
