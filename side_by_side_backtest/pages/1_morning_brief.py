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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st
import base64
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Alert sound (base64-embedded so no file-serving needed)
# ---------------------------------------------------------------------------
_SOUND_PATH = Path(__file__).parent / "effect.mp3"

def _play_alert_sound() -> None:
    """Inject a hidden <audio> element that auto-plays the alert sound once."""
    if not _SOUND_PATH.exists():
        return
    try:
        b64 = base64.b64encode(_SOUND_PATH.read_bytes()).decode()
        components.html(
            f'<audio autoplay style="display:none">'
            f'<source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">'
            f'</audio>',
            height=0,
        )
    except Exception:
        pass  # sound is best-effort

# Allow running as a standalone page too
_PKG = Path(__file__).parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG.parent))

from side_by_side_backtest.models import RawWatchlist, SessionType, WatchlistEntry
from side_by_side_backtest.parser import parse_watchlist_post
from side_by_side_backtest.setup_scorer import SetupScore, score_setup
from side_by_side_backtest.data_fetcher import (
    load_30day_bars,
    fetch_bars_for_entry,   # fallback if 30d cache not seeded yet
)

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


def _score_entries_raw(entries_json: str, live_refresh: bool = False,
                       max_workers: int = 8) -> List[SetupScore]:
    """Score each entry in parallel and return a list of SetupScore objects.

    live_refresh=True: call refresh_today() for each ticker (Latest post only, small set).
    live_refresh=False: use load_30day_bars() only (All history — too many tickers to refresh).

    max_workers controls concurrency of per-ticker data fetch + scoring.
    Each worker also benefits from the parallelized component scorers inside score_setup().
    """
    import json as _json
    entries = [WatchlistEntry(**e) for e in _json.loads(entries_json)]

    def _score_one(entry) -> SetupScore:
        if live_refresh:
            from side_by_side_backtest.data_fetcher import refresh_today
            bars = refresh_today(entry.ticker)
        else:
            bars = load_30day_bars(entry.ticker)
        if bars.empty:
            try:
                bars = fetch_bars_for_entry(entry) or pd.DataFrame()
            except Exception:
                bars = pd.DataFrame()
        return score_setup(entry, bars)

    scores: List[SetupScore] = []
    n = min(max_workers, len(entries)) if entries else 1
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(_score_one, e) for e in entries]
        for fut in as_completed(futures):
            try:
                scores.append(fut.result())
            except Exception:
                pass  # individual entry failure is non-fatal

    scores.sort(key=lambda s: s.score, reverse=True)
    return scores


def _dedupe_scores(scores: List[SetupScore]) -> List[SetupScore]:
    """
    Deduplicate by ticker: keep the highest-scoring SetupScore per ticker.
    Merge all watchlist notes from every appearance into a timestamped summary
    that replaces the winning score's watchlist_note.
    """
    from collections import defaultdict
    import json as _json, re as _re

    # Group all scores per ticker
    groups: dict[str, List[SetupScore]] = defaultdict(list)
    for sc in scores:
        groups[sc.ticker].append(sc)

    deduped: List[SetupScore] = []
    for ticker, group in groups.items():
        # Best score = highest score value
        best = max(group, key=lambda s: s.score)

        if len(group) == 1:
            deduped.append(best)
            continue

        # Build merged note: deduplicate notes, add source post timestamps
        seen_notes: set[str] = set()
        merged_lines: List[str] = []
        for sc in sorted(group, key=lambda s: s.watchlist_note[:20]):
            note = sc.watchlist_note.strip()
            # Normalise whitespace for dedup
            note_key = _re.sub(r'\s+', ' ', note.lower())[:120]
            if note_key and note_key not in seen_notes:
                seen_notes.add(note_key)
                merged_lines.append(f"• {note}")

        if len(merged_lines) > 1:
            best = SetupScore(
                **{k: getattr(best, k) for k in best.__dataclass_fields__
                   if k != "watchlist_note"},
                watchlist_note="\n".join(merged_lines),
            )

        deduped.append(best)

    deduped.sort(key=lambda s: s.score, reverse=True)
    return deduped


# ---------------------------------------------------------------------------
# JS live indicator (client-side countdown — zero server cost between ticks)
# ---------------------------------------------------------------------------

def _js_live_indicator(scan_interval: int, last_scan_str: str) -> None:
    """Inject a client-side JS countdown + flashing dot. No server round-trips."""
    html = f"""
    <div style="display:flex; align-items:center; gap:10px;
                font-family:monospace; font-size:13px; color:#cccccc;
                padding:4px 0;">
        <span id="live-dot" style="font-size:16px;">🔴</span>
        <span>
            LIVE &nbsp;|&nbsp; Last scan: <b>{last_scan_str}</b>
            &nbsp;|&nbsp; Next in: <b><span id="cd">{scan_interval}</span></b>
        </span>
    </div>
    <script>
        (function() {{
            let t = {scan_interval};
            const cd   = document.getElementById('cd');
            const dot  = document.getElementById('live-dot');
            const dots = ['🔴','🟠'];
            let d = 0;
            const iv = setInterval(() => {{
                t = Math.max(0, t - 1);
                cd.textContent = t + 's';
                dot.textContent = dots[d % 2];
                d++;
                if (t <= 0) clearInterval(iv);
            }}, 1000);
        }})();
    </script>
    """
    components.html(html, height=36)


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

    st.sidebar.divider()
    st.sidebar.subheader("🔴 Live Scanner")
    _scan_options = {"1 min": 60, "5 min": 300, "15 min": 900, "Off": None}
    _scan_label   = st.sidebar.selectbox("Auto-rescore interval", list(_scan_options.keys()), index=0)
    scan_interval = _scan_options[_scan_label]

    if st.sidebar.button("🔄 Refresh data"):
        st.cache_data.clear()

    return json_path, session_filter, post_filter, scan_interval


# ---------------------------------------------------------------------------
# Setup card
# ---------------------------------------------------------------------------

def _bar(score: float, max_score: float = 2.0) -> str:
    """ASCII progress bar for a component score."""
    filled = int(round(score / max_score * 8))
    return "█" * filled + "░" * (8 - filled)


def _format_note(note: str) -> str:
    """Format watchlist note(s) for display. Multi-line (merged) notes get bullet treatment."""
    lines = [ln.strip() for ln in note.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return note.strip()
    # Already bullet-formatted by _dedupe_scores
    return "\n".join(lines)


def _render_card(sc: SetupScore, idx: int, score_history: list = None) -> None:
    """Render an expanded setup card for one ticker inside a Streamlit expander."""
    # Compute delta arrow for the header
    delta_str = ""
    if score_history and len(score_history) >= 2:
        diff = score_history[-1][1] - score_history[-2][1]
        if diff > 0.05:    delta_str = f" 🟢▲{diff:.1f}"
        elif diff < -0.05: delta_str = f" 🔴▼{abs(diff):.1f}"
        else:              delta_str = " ⬜="

    header = f"{sc.ticker}  ●  {sc.score}/10{delta_str}  ●  {sc.signal}"
    with st.expander(header, expanded=False):
        # ── Session score sparkline ──────────────────────────────────────
        if score_history and len(score_history) >= 3:
            import pandas as _pd
            spark_df = _pd.DataFrame(
                [s for _, s in score_history], columns=["Score"]
            )
            st.line_chart(spark_df, height=60, width='stretch')
        note = _format_note(sc.watchlist_note)
        lines = note.splitlines()
        if len(lines) > 1:
            st.markdown("**Trader notes** _(merged from multiple watchlist posts)_:")
            st.markdown(note)
        else:
            if sc.support_broken:
                st.warning("⚠️ **Support broken** — current price is below the watchlist level. Levels re-anchored to computed S/R.")
            elif not sc.support_ok:
                st.warning("⚠️ **Support not holding** — recent closes repeatedly below support zone. Simulator would filter this entry.")
            st.markdown(f"**Trader note:** _{note}_")
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
            ("Role Rev.",  sc.role_reversal_score),
            ("Rejections", sc.rejection_score),
            ("Rel.Volume", sc.rel_vol_score),
            ("MACD Slope", sc.macd_score),
            ("RSI Div.",   sc.rsi_div_score),
            ("Regime",     sc.regime_score),
        ]
        for label, val in score_rows:
            st.text(f"  {label:<12} {_bar(val)}  {val:.1f}/2")

        # Role reversal + rejection badges
        badges = []
        if sc.role_reversal:
            badges.append("🔄 Role Reversal confirmed")
        if sc.rejection_count >= 2:
            badges.append(f"🛡️ {sc.rejection_count}× wick rejections")
        elif sc.rejection_count == 1:
            badges.append("🛡️ 1 wick rejection")
        if sc.support_ok and not sc.support_broken:
            badges.append("✅ Support holding")
        elif not sc.support_ok:
            badges.append("🚫 Support not holding")
        if badges:
            st.caption("  " + "  |  ".join(badges))

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

    json_path, session_filter, post_filter, scan_interval = _sidebar()

    if not Path(json_path).exists():
        st.error(f"Watchlist JSON not found: `{json_path}`")
        return

    raw = _load_json(json_path)

    # Apply post filter BEFORE parsing (operate on raw dicts)
    if post_filter == "Latest post" and raw:
        def _ts(p: dict) -> str:
            return p.get("timestamp") or ""
        latest_ts = max(_ts(p) for p in raw)
        raw = [p for p in raw if (_ts(p) or "") == latest_ts]
        st.sidebar.caption(f"Showing: {raw[0]['title'][:60]}…" if raw else "")

    all_entries = _parse_posts(raw)

    if session_filter:
        all_entries = [e for e in all_entries if e.session_type.value == session_filter]

    if not all_entries:
        st.warning("No entries found for selected session.")
        return

    import json as _json
    entries_json = _json.dumps([e.dict() for e in all_entries], default=str)

    # live_refresh=True only when viewing "Latest post" (small ticker set, safe to refresh)
    _live_refresh = (post_filter == "Latest post")

    # ── Live scanner fragment — re-scores on interval, fires toast alerts ─────
    @st.fragment(run_every=scan_interval)
    def _live_section(entries_json: str, live_refresh: bool) -> None:
        import time as _time

        try:
            with st.spinner("Scoring setups…"):
                scores = _dedupe_scores(_score_entries_raw(entries_json, live_refresh=live_refresh))
        except Exception as exc:
            st.error(f"Scoring error: {exc}")
            return

        # ── Session score history: {ticker: [(timestamp, score), ...]} ───────
        import time as _ts_time
        if "score_history" not in st.session_state:
            st.session_state["score_history"] = {}
        if "prev_signals" not in st.session_state:
            st.session_state["prev_signals"] = {}
        if "alerted_tickers" not in st.session_state:
            st.session_state["alerted_tickers"] = set()

        now_ts = _ts_time.time()
        hist   = st.session_state["score_history"]
        prev_sig = st.session_state["prev_signals"]

        for sc in scores:
            t = sc.ticker
            # Append to score history (cap at 60 points per session)
            hist.setdefault(t, [])
            hist[t].append((now_ts, sc.score))
            if len(hist[t]) > 60:
                hist[t] = hist[t][-60:]

            # Detect signal transition and build reason string
            old_signal = prev_sig.get(t)
            new_signal = sc.signal
            transition = None
            reason     = ""

            if old_signal and old_signal != new_signal:
                # Determine what improved
                improved = []
                if sc.pattern_found:
                    improved.append("pattern confirmed")
                if sc.adx_score >= 1.5:
                    improved.append(f"ADX {sc.adx:.0f}")
                if sc.rr_score >= 1.5:
                    improved.append(f"R/R {sc.rr_ratio:.1f}:1")
                if sc.confluence_score >= 1.5:
                    improved.append("S/R confluence")
                if sc.role_reversal:
                    improved.append("role reversal")
                if sc.rejection_count >= 2:
                    improved.append(f"{sc.rejection_count}× rejections")
                reason = " + ".join(improved[:3]) or "multiple signals"
                transition = f"{old_signal} → {new_signal}"

            prev_sig[t] = new_signal

            # Fire toast on signal upgrade or new pattern
            if sc.pattern_found and sc.support > 0 and sc.score >= 4.0:
                key = f"{sc.ticker}_{sc.support:.3f}"
                if key not in st.session_state["alerted_tickers"]:
                    msg = f"🚨 **{sc.ticker}** — {sc.score}/10 | Support ${sc.support:.3f}"
                    if transition:
                        msg += f"\n{transition}: {reason}"
                    st.toast(msg, icon="🔔")
                    st.session_state["alerted_tickers"].add(key)
                    _play_alert_sound()
            elif transition and "STRONG" in new_signal:
                st.toast(
                    f"⬆️ **{sc.ticker}** upgraded: {transition}\n_{reason}_",
                    icon="📈",
                )
                _play_alert_sound()

        # ── Live indicator + client-side countdown ──────────────────────────
        if scan_interval:
            now_ts = _time.time()
            last_scan_str = _time.strftime('%H:%M:%S', _time.localtime(now_ts))
            _js_live_indicator(scan_interval, last_scan_str)

        # ── Ranked table ───────────────────────────────────────────────────
        def _delta_arrow(ticker: str) -> str:
            pts = hist.get(ticker, [])
            if len(pts) < 2:
                return "—"
            diff = pts[-1][1] - pts[-2][1]
            if diff > 0.05:  return f"🟢 ▲{diff:.1f}"
            if diff < -0.05: return f"🔴 ▼{abs(diff):.1f}"
            return "⬜ ="

        rows_data = [{
            "Ticker":  sc.ticker,
            "Δ":       _delta_arrow(sc.ticker),
            "Score":   sc.score,
            "Signal":  sc.signal,
            "Entry $": sc.entry_price,
            "Support": sc.support,
            "Resist":  sc.resistance,
            "Stop":    sc.stop,
            "R/R":     sc.rr_ratio,
            "ADX":     sc.adx,
            "Pattern": "✅" if sc.pattern_found else "—",
        } for sc in scores]

        display_cols = ["Ticker", "Δ", "Score", "Signal", "Entry $", "Support",
                        "Resist", "Stop", "R/R", "ADX", "Pattern"]
        df = pd.DataFrame(rows_data)[display_cols]
        st.subheader(f"Ranked Setups — {len(df)} tickers")
        st.dataframe(df, width='stretch', hide_index=True)

        st.markdown("---")
        st.subheader("Setup Cards")
        for idx, sc in enumerate(scores):
            _render_card(sc, idx, score_history=hist.get(sc.ticker, []))

    _live_section(entries_json, _live_refresh)


if __name__ == "__main__":
    main()
