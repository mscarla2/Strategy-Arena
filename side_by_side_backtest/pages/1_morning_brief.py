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
import time as _wall_clock   # wall-clock used for refresh-cooldown guard

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


# Module-level refresh-cooldown guard.
# Tracks the last wall-clock time refresh_today() was called per ticker so we
# don't hammer yfinance on every fragment tick (default every 60s).
# Key: ticker str  →  Value: float (monotonic timestamp of last refresh)
_last_refresh_ts: dict[str, float] = {}
_REFRESH_COOLDOWN = 55.0   # seconds — slightly less than the 60s scan interval

# Background-refresh state: tracks whether a fire-and-forget refresh is in
# flight so we don't spawn duplicate threads on rapid Streamlit reruns.
import threading as _threading
_bg_refresh_lock   = _threading.Lock()
_bg_refresh_tickers: set[str] = set()   # tickers currently being refreshed in bg


def _prefetch_tickers(
    tickers: list[str],
    max_workers: int = 12,
    background: bool = False,
) -> None:
    """
    Run refresh_today() for all *tickers* in parallel.

    background=False (default, used on subsequent fragment ticks):
        Blocks until all refreshes complete so the scoring phase immediately
        sees fresh data.

    background=True (used on first render):
        Fires a daemon thread and returns immediately — the first render scores
        from the existing on-disk parquets (~1ms/ticker) without waiting for
        the network.  The next fragment tick (≥60s later) will find the fresh
        parquets already written and will call with background=False.

    Refresh-cooldown guard: tickers refreshed within the last _REFRESH_COOLDOWN
    seconds are skipped entirely, preventing redundant HTTP calls on rapid reruns.
    """
    from side_by_side_backtest.data_fetcher import refresh_today

    now = _wall_clock.monotonic()
    due = [t for t in tickers if now - _last_refresh_ts.get(t, 0) >= _REFRESH_COOLDOWN]
    if not due:
        return   # all tickers refreshed recently — nothing to do

    def _refresh_one(ticker: str) -> None:
        try:
            refresh_today(ticker)
            _last_refresh_ts[ticker] = _wall_clock.monotonic()
        except Exception:
            pass  # network failure is non-fatal; scoring will use cached data
        finally:
            with _bg_refresh_lock:
                _bg_refresh_tickers.discard(ticker)

    if background:
        # Only spawn threads for tickers not already being refreshed
        with _bg_refresh_lock:
            new_due = [t for t in due if t not in _bg_refresh_tickers]
            _bg_refresh_tickers.update(new_due)
        if not new_due:
            return
        def _run_bg():
            with ThreadPoolExecutor(max_workers=min(max_workers, len(new_due))) as pool:
                futs = [pool.submit(_refresh_one, t) for t in new_due]
                for fut in as_completed(futs):
                    try: fut.result()
                    except Exception: pass
        t = _threading.Thread(target=_run_bg, daemon=True, name="bg-refresh")
        t.start()
    else:
        n = min(max_workers, len(due))
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = [pool.submit(_refresh_one, t) for t in due]
            for fut in as_completed(futures):
                fut.result()  # propagate exceptions silently (already caught inside)


def _score_entries_raw(entries_json: str, live_refresh: bool = False,
                       max_workers: int = 8) -> List[SetupScore]:
    """Score each entry in parallel and return a list of SetupScore objects.

    First render (is_first_run=True in session_state):
      Scores immediately from on-disk parquets — no network wait.
      Fires a background thread to refresh all ticker parquets concurrently.
      Result: first paint in ~300ms instead of ~600ms+.

    Subsequent fragment ticks (live_refresh=True, not first run):
      Phase 1 — blocks on _prefetch_tickers() (network, parallel, ~150ms for 8 tickers).
      Phase 2 — scores from freshly-written disk cache.

    When live_refresh=False (All history mode):
      Skips network entirely — use cached data only.

    max_workers controls concurrency of the CPU scoring phase.
    """
    import json as _json
    entries = [WatchlistEntry(**e) for e in _json.loads(entries_json)]

    unique_tickers = list({e.ticker for e in entries}) if entries else []

    # ── Phase 1: network refresh ──────────────────────────────────────────────
    if live_refresh and unique_tickers:
        is_first = not st.session_state.get("_brief_has_scored", False)
        if is_first:
            # First render: fire-and-forget so UI paints immediately from disk.
            # Mark scored now so the next tick uses the blocking path.
            st.session_state["_brief_has_scored"] = True
            _prefetch_tickers(unique_tickers, max_workers=max(max_workers, 12),
                              background=True)
        else:
            # Subsequent ticks: block until fresh data is ready before scoring.
            _prefetch_tickers(unique_tickers, max_workers=max(max_workers, 12),
                              background=False)

    # ── Phase 2: score from disk cache (CPU only, no network) ────────────────
    def _score_one(entry) -> SetupScore:
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

    st.sidebar.divider()
    st.sidebar.subheader("🎛️ Display Filters")
    min_score = st.sidebar.slider(
        "Min score threshold", min_value=0.0, max_value=10.0, value=0.0, step=0.5,
        help="Hide tickers scoring below this threshold (0 = show all)"
    )

    if st.sidebar.button("🔄 Refresh data"):
        st.cache_data.clear()

    return json_path, session_filter, post_filter, scan_interval, min_score


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
            ("EQH",        sc.eqh_score),
        ]
        for label, val in score_rows:
            st.text(f"  {label:<12} {_bar(val)}  {val:.1f}/2")

        # EQH (Equal Highs / Liquidity Ceiling) status
        if sc.eqh_level and sc.eqh_level > 0:
            eqh_icon = {
                "eqh_breakout":  "🟢 Breakout fired",
                "eqh_rejection": "🔴 Rejected",
                "approaching":   "🟡 Approaching",
            }.get(sc.eqh_signal, "🏛️ Detected")
            st.info(f"**EQH Ceiling: ${sc.eqh_level:.3f}** — {eqh_icon}")

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

    json_path, session_filter, post_filter, scan_interval, min_score = _sidebar()

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
    def _live_section(entries_json: str, live_refresh: bool, _min_score: float = 0.0) -> None:
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
            "EQH":     (f"🏛️${sc.eqh_level:.2f}" if sc.eqh_level else "—"),
        } for sc in scores]

        display_cols = ["Ticker", "Δ", "Score", "Signal", "Entry $", "Support",
                        "Resist", "Stop", "R/R", "ADX", "Pattern", "EQH"]
        df = pd.DataFrame(rows_data)[display_cols]

        # ── Min-score filter ─────────────────────────────────────────────────
        if _min_score > 0:
            df_display = df[df["Score"] >= _min_score]
            scores_display = [sc for sc in scores if sc.score >= _min_score]
        else:
            df_display = df
            scores_display = scores

        # ── Session badges ────────────────────────────────────────────────────
        _sess_counts = {}
        for sc in scores:
            _sess_label = getattr(sc, "session_type", None)
            if _sess_label:
                _sess_counts[_sess_label] = _sess_counts.get(_sess_label, 0) + 1

        _badge_parts = []
        for _sess, _cnt in sorted(_sess_counts.items()):
            _badge_parts.append(f"**{_cnt}** {_sess.replace('_', ' ')}")
        if _badge_parts:
            st.caption("Sessions: " + "  |  ".join(_badge_parts))

        n_total    = len(df)
        n_filtered = len(df_display)
        header_str = f"Ranked Setups — {n_filtered} tickers" + (
            f" (filtered from {n_total})" if n_filtered < n_total else ""
        )
        st.subheader(header_str)
        st.dataframe(df_display, width='stretch', hide_index=True)

        # ── CSV export ────────────────────────────────────────────────────────
        _csv_bytes = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Export to CSV",
            data=_csv_bytes,
            file_name="morning_brief_setups.csv",
            mime="text/csv",
            key=f"csv_export_{int(_ts_time.time())}",
        )

        st.markdown("---")
        st.subheader("Setup Cards")
        for idx, sc in enumerate(scores_display):
            _render_card(sc, idx, score_history=hist.get(sc.ticker, []))

    _live_section(entries_json, _live_refresh, min_score)


if __name__ == "__main__":
    main()
