#!/usr/bin/env python3
"""
ALPHAGENE Strategy Dashboard — Multi-Page Streamlit App

Navigation hub that routes to specialised pages:
  - Oil Performance
  - Formula Lab
  - Backtester Controls
  - Feature Importance
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ALPHAGENE Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.title("ALPHAGENE")

NAV_OIL = "Oil Performance"
NAV_LAB = "Formula Lab"
NAV_BACKTEST = "Backtester"
NAV_FEATURES = "Feature Importance"

page = st.sidebar.radio(
    "Navigation",
    [NAV_OIL, NAV_LAB, NAV_BACKTEST, NAV_FEATURES],
)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

if page == NAV_OIL:
    from ui.pages.oil_performance import render
    render()
elif page == NAV_LAB:
    from ui.pages.formula_lab import render
    render()
elif page == NAV_BACKTEST:
    from ui.pages.backtester import render
    render()
elif page == NAV_FEATURES:
    from ui.pages.feature_importance import render
    render()
