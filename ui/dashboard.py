#!/usr/bin/env python3
"""
ALPHAGENE Strategy Dashboard — Multi-Page Streamlit App

Navigation hub that routes to specialised pages:
  • Oil Performance
  • Backtester Controls
  • Feature Importance
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

st.sidebar.image(
    "https://img.icons8.com/color/96/dna-helix.png",
    width=64,
)
st.sidebar.title("ALPHAGENE")
st.sidebar.caption("Genetic Programming Arena v3.0")

page = st.sidebar.selectbox(
    "Navigation",
    [
        "🛢️ Oil Performance",
        "🧪 Backtester",
        "📊 Feature Importance",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Phase 3 — Dashboard & Management UI")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🛢️ Oil Performance":
    from ui.pages.oil_performance import render
    render()
elif page == "🧪 Backtester":
    from ui.pages.backtester import render
    render()
elif page == "📊 Feature Importance":
    from ui.pages.feature_importance import render
    render()


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.divider()
col1, col2 = st.sidebar.columns(2)
with col1:
    st.sidebar.caption("🧬 AlphaGene v3.0")
with col2:
    st.sidebar.caption("✅ Production Ready")
