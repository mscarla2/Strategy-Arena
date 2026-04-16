"""
Chart Viewer — Page 2 of the multi-page app
============================================
Thin adapter: delegates entirely to the canonical chart_viewer module.
Pre-selects the ticker stored in session_state["chart_ticker"] when
navigated to from the Morning Brief "Open in Chart Viewer" button.

The actual chart logic lives in:
    side_by_side_backtest/chart_viewer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the package root is on sys.path when run as a page
_PKG_ROOT = Path(__file__).parent.parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

# Import the canonical module — its run() reads session_state["chart_ticker"]
from side_by_side_backtest.chart_viewer import run  # noqa: E402

run()
