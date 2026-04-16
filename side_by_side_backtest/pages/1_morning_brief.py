"""
Morning Brief — Watchlist Triage Page
======================================
Ranks every ticker from the scraped watchlist by SetupScore,
shows a summary table, and expands into a detailed setup card per ticker.

Launch via the multi-page app:
    streamlit run side_by_side_backtest/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

# Allow running as a standalone page too
_PKG = Path(__file__).parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG.parent))

from side_by_side_backtest.models import RawWatchlist, SessionType, WatchlistEntry
from side_by_side_backtest.parser import parse_watchlist_post
from side_by_side_backtest.setup_scorer import SetupScore, score_setup
from side_by_side_backtest.data_fetcher import fetch_bars_for_entry

_DEFAULT_JSON = _PKG.parent / "scraped_watchlists.json"


# ---------------------------------------------------------------------------
# Data helpers (cached so Streamlit doesn't re-run on every widget interaction)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading watchlist JSON…")
def _load_json(path: str) -> List[dict]:
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner="Parsing watchlist entries…")
def _parse_posts(raw: List[dict]) -> List[WatchlistEntry]:
    entries: List[WatchlistEntry] = []
    for post in raw:
        rw = RawWatchlist(**post)
        try:
            parsed = parse_watchlist_post(rw)
            entries.extend(parsed)
        except Exception:
            pass
    return entries


@st.cache_data(show_spinner="Fetching bars & scoring…", ttl=300)
def _score_entries(entries_json: str) -> List[dict]:
    """Serialise entries → score each → return list of dicts for display."""
    import json as _json
    entries = [WatchlistEntry(**e) for e in _json.loads(entries_json)]
    results: List[dict] = []
    for entry in entries:
        try:
            bars = fetch_bars_for_entry(entry) or pd.DataFrame()
        except Exception:
            bars = pd.DataFrame()
        sc = score_setup(entry, bars)
        results.append({
            "Ticker":  sc.ticker,
            "Score":   sc.score,
            "Signal":  sc.signal,
            "Entry $": sc.entry_price,
            "Support": sc.support,
            "Resist":  sc.resistance,
            "Stop":    sc.stop,
            "R/R":     sc.rr_ratio,
            "ADX":     sc.adx,
            "Pattern": "✅" if sc.pattern_found else "—",
            "_score_obj": sc,    # keep full object for card rendering
        })
    results.sort(key=lambda r: r["Score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _sidebar() -> tuple[str, Optional[str]]:
    """Render sidebar controls; return (json_path, session_filter)."""
    st.sidebar.header("📋 Morning Brief")

    json_path = st.sidebar.text_input(
        "Watchlist JSON path",
        value=str(_DEFAULT_JSON),
    )

    # Post filter — default to latest post only
    post_filter_options = ["Latest post", "All history"]
    post_filter = st.sidebar.selectbox(
        "Post filter", post_filter_options, index=0,
        help="'Latest post' scores only today's watchlist; 'All history' merges every post."
    )

    session_options = ["All", "pre_market", "market_open", "after_hours"]
    session_filter = st.sidebar.selectbox("Session filter", session_options)
    session_filter = None if session_filter == "All" else session_filter

    if st.sidebar.button("🔄 Refresh data"):
        st.cache_data.clear()

    return json_path, session_filter, post_filter


# ---------------------------------------------------------------------------
# Setup card
# ---------------------------------------------------------------------------

def _bar(score: float, max_score: float = 2.0) -> str:
    """ASCII progress bar for a component score."""
    filled = int(round(score / max_score * 8))
    return "█" * filled + "░" * (8 - filled)


def _render_card(sc: SetupScore, idx: int) -> None:
    """Render an expanded setup card for one ticker inside a Streamlit expander."""
    header = f"{sc.ticker}  ●  Score: {sc.score}/10  ●  {sc.signal}"
    with st.expander(header, expanded=False):
        st.markdown(f"**Watchlist note:** _{sc.watchlist_note}_")
        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        col1.metric("Entry target",  f"${sc.entry_price:.2f}")
        col2.metric("Take-Profit",   f"${sc.resistance:.2f}" if sc.resistance else "—")
        col3.metric("Stop-Loss",     f"${sc.stop:.2f}")

        col4, col5 = st.columns(2)
        col4.metric("R/R Ratio", f"{sc.rr_ratio:.1f}:1" if sc.rr_ratio else "—")
        col5.metric("ADX", f"{sc.adx:.1f}")

        st.markdown("---")
        st.markdown("**Score breakdown:**")
        score_rows = [
            ("Pattern",    sc.pattern_score),
            ("ADX",        sc.adx_score),
            ("R/R",        sc.rr_score),
            ("Confluence", sc.confluence_score),
            ("History",    sc.history_score),
        ]
        for label, val in score_rows:
            st.text(f"  {label:<12} {_bar(val)}  {val:.1f}/2")

        st.markdown("---")
        # idx makes the key unique even when the same ticker appears twice
        if st.button(f"📈 Open in Chart Viewer — {sc.ticker}", key=f"cv_{sc.ticker}_{idx}"):
            st.session_state["chart_ticker"] = sc.ticker
            st.switch_page("pages/2_chart_viewer.py")


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Morning Brief", page_icon="📋", layout="wide")
    st.title("📋 Morning Brief — Watchlist Triage")

    json_path, session_filter, post_filter = _sidebar()

    if not Path(json_path).exists():
        st.error(f"Watchlist JSON not found: `{json_path}`")
        return

    raw = _load_json(json_path)

    # Apply post filter BEFORE parsing (operate on raw dicts)
    if post_filter == "Latest post" and raw:
        # Sort by timestamp descending, pick all posts that share the max timestamp
        def _ts(p: dict) -> str:
            return p.get("timestamp") or ""
        latest_ts = max(_ts(p) for p in raw)
        raw = [p for p in raw if (_ts(p) or "") == latest_ts]
        st.sidebar.caption(f"Showing: {raw[0]['title'][:60]}…" if raw else "")

    all_entries = _parse_posts(raw)

    # Filter by session
    if session_filter:
        all_entries = [e for e in all_entries if e.session_type.value == session_filter]

    if not all_entries:
        st.warning("No entries found for selected session.")
        return

    # Serialise entries so @st.cache_data can hash them
    import json as _json
    entries_json = _json.dumps([e.dict() for e in all_entries], default=str)
    rows = _score_entries(entries_json)

    # Summary table (without internal _score_obj column)
    display_cols = ["Ticker", "Score", "Signal", "Entry $", "Support",
                    "Resist", "Stop", "R/R", "ADX", "Pattern"]
    df = pd.DataFrame(rows)[display_cols]
    st.subheader(f"Ranked Setups — {len(df)} tickers")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Setup Cards")
    for idx, row in enumerate(rows):
        _render_card(row["_score_obj"], idx)


if __name__ == "__main__":
    main()
