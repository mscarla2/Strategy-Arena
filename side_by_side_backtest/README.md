# Side-by-Side Body Intraday Backtest — Trader Toolkit

A fully automated pipeline that scrapes trading watchlists, scores setups in real-time, detects the **Bearish Side-by-Side White Lines** and **Equal Highs / Liquidity Ceiling** candlestick patterns on 5-minute OHLCV data, simulates trades against human-curated support/resistance levels, and uses **Bayesian optimisation** to find the best profit-target / stop-loss parameters.

---

## What It Does

1. **Scrapes** Reddit watchlist posts → appends to `scraped_watchlists.json` (never overwrites history)
2. **Caches** 30 days of 5-min OHLCV (pre/after-market included) per ticker — intra-day gap-aware
3. **Scores** each setup 0–10 across **12 components**: Pattern, ADX, R/R, Confluence, History, Role-Reversal, Rejections, Relative Volume, MACD Slope, RSI Divergence, Regime, **EQH**
4. **Deduplicates** by ticker — keeps best score, merges watchlist notes; S/R falls back to computed levels when watchlist levels are stale/missing
5. **Triages** the morning watchlist in a ranked Streamlit table with live-scoring fragment, score delta arrows (▲▼=), session sparklines, min-score filter, and CSV export
6. **Scans** live — fires alerts (toast + macOS notification + `effect.mp3`) on signal upgrade or support touch
7. **Backtests** body-based "Wait and Stay" entry logic **and** EQH breakout mode with trailing stop, per-session analysis, and 7-check sanity suite
8. **Optimises** PT/SL parameters via Bayesian search (Optuna TPE) with optional **walk-forward OOS validation**

---

## Install

```bash
pip install -r side_by_side_backtest/requirements.txt
```

---

## 🗓️ Daily Trader Timeline

### Pre-market (~8:00 AM PT)

**Step 1 — Scrape the morning watchlist**
```bash
python watchlist_scraper.py
```
Appends only new posts. Deduplicates by `title::timestamp`.

**Step 2 — Refresh the 30-day OHLCV cache**
```bash
# Daily — intra-day gap-aware (fetches minutes missed since last bar)
./venv/bin/python -m side_by_side_backtest.refresh_cache

# First-ever run OR after a holiday
./venv/bin/python -m side_by_side_backtest.refresh_cache --full
```

**Step 3 — Open the Morning Brief**
```bash
streamlit run side_by_side_backtest/app.py
```
Open **http://localhost:8501** → **📋 Morning Brief**

- **Post filter: `Latest post`** (default) — scores only today's tickers; data refreshed in parallel background pool
- **Live Scanner: `1 min`** (default) — rescores every minute; client-side JS countdown
- **Min-score slider** — hide SKIP setups below threshold (0–10)
- **Export CSV** — download ranked table snapshot
- Ranked table sorted by score (0–10), **deduplicated** (1 row per ticker), with `Δ` column:
  - `🟢 ▲0.8` — score went up this tick
  - `🔴 ▼0.3` — score went down
  - `⬜ =` — unchanged
- **EQH column** — shows `🏛️$X.XX` when a Liquidity Ceiling is detected near current price
- Signal upgrade toast with reason: `"⬆️ ANNA upgraded: SKIP → WATCH — pattern confirmed + ADX 38"`
- Setup cards: merged watchlist notes + 12-component score breakdown bars + session sparkline
- Click **"Open in Chart Viewer"** → deep-links to chart in PT timezone

---

### Market open (9:30 AM ET)

**Step 4 — Start the live scanner** (leave running all day)
```bash
./venv/bin/python -m side_by_side_backtest.live_scanner
```
Polls every 5 minutes. Fires alert (`effect.mp3` + macOS notification) when candle **body** touches support AND pattern confirms.

**Step 5 — Act on alerts**
1. Buy at the **next 5-min bar open** (~5 min after alert)
2. **Take-Profit** at resistance level from Morning Brief card
3. **Stop-Loss** 2% below support (stop on **close below** support — wicks don't count)
4. Trailing stop activates at +1% profit

---

### End of day (4:00 PM ET)

**Step 6 — Update win-rate history + run OOS validation**
```bash
./venv/bin/python -m side_by_side_backtest.main \
  --watchlist scraped_watchlists.json \
  --skip-fetch \
  --auto-tune --n-trials 100 \
  --validate-oos --oos-split 0.30 \
  --export --sanity
```
`--skip-fetch` uses 30d parquet — zero network calls if cache is current.
`--validate-oos` splits entries chronologically (last 30%) and re-runs best params on OOS fold.

---

## Signal Engine — 12-Component SetupScore

Score is 0–24 raw (12 × 2 pts each), normalised to 0–10:

| Component | 0 pts | 1 pt | 2 pts |
|-----------|-------|------|-------|
| **Pattern** | No pattern | Pattern anywhere (not at support) | Strict/Exhaustion/Absorption ≤1% of support |
| **ADX** | < 15 | 15–25 | > 25 |
| **R/R** | Missing or < 1.0 | 1.0–1.99 | ≥ 2.0 (capped at 10:1) |
| **Confluence** | No S/R agree | 1 method | 2+ agree; VPOC bonus: 🏦VPOC |
| **History** | < 3 trades | WR 30–59% | WR ≥ 60% |
| **Role Reversal** | No flip | Soft evidence | Confirmed R→S flip |
| **Rejections** | 0 wick rejections | 1 rejection | 2+ rejections off support |
| **Rel. Volume** | C1 < 1.5× same-TOD | 1.5×–2× | ≥ 2× same-time-of-day avg |
| **MACD Slope** | Histogram falling | Rising 1 bar | Rising 3 bars (weakening sell) |
| **RSI Divergence** | None | Partial (flat RSI on lower low) | Confirmed divergence |
| **Regime** | 30-min EMA up (counter-trend) | Flat | 30-min EMA trending down |
| **EQH** | No EQH pair found | Pair found within 2% of price | Pair found ≤0.5% of price OR signal fired |

**S/R Fallback:** If watchlist support/resistance is missing or > 30% from current price, computed levels from `sr_engine` are used automatically.

**EQH signal states:** `""` (none) · `"approaching"` (near ceiling) · `"eqh_breakout"` · `"eqh_rejection"`

---

## Pattern Engine — Four Detectors

### 1. Bearish Side-by-Side (strict) — `detect_side_by_side`
- C1: Strong bearish candle (body ≥ avg body) immediately before the pair
- C2/C3: Twin **bearish** candles, same open (±3%), same body size (±50%)
- ADX ≥ 20, EMA downtrend confirmed
- `confidence_score = 1.0`

### 2. Exhaustion Side-by-Side (relaxed) — `detect_exhaustion_side_by_side`
- C1 can be up to 10 bars earlier (bottoming patterns)
- ADX ≥ 15, EMA can be flat; same-open tolerance 5%
- `confidence_score = 0.6`

### 3. Support Absorption — `detect_support_absorption`
- C1: Large bearish candle landing within 1% of support
- C2/C3: Small-body candles **staying above C1's low**; at least one must be **bullish**
- `confidence_score = 0.7`

### 4. Equal Highs / Liquidity Ceiling — `detect_equal_highs_pair` *(new)*
The trader's primary overhead-resistance target pattern. Circled in yellow on charts.

- C2 + C3: **Mixed-color** pair (one bullish + one bearish) with matching opens (±2%)
- The pair defines the EQH **ceiling price** = `max(C2.open, C3.open)`
- `confidence_score = 0.8`

**Follow-up signal — `detect_eqh_signal`:**
| Signal | Trigger | Action |
|--------|---------|--------|
| `eqh_breakout` | Candle body closes **above** ceiling | Enter long — stop-losses of shorts triggered = explosive move |
| `eqh_rejection` | Candle body closes **below** pair low | Exit/sell — bears defended level; hard decline incoming |

**EQH backtest mode:**
```bash
./venv/bin/python -m side_by_side_backtest.main \
  --tickers BIRD HR --eqh --pt 5.0 --sl 2.0 --no-sweep
```

### Entry Logic (Body-Based, bearish S×S mode)
- **Touch trigger:** `min(open, close) ≤ support × 1.005` — wick pokes don't trigger
- **Exit on close below:** only a candle **closing** below support resets the state
- **Pattern lookback:** 10 bars (~50 min)

---

## 📊 Performance Analytics Page

Open **http://localhost:8501** → **📊 Performance Analytics**

New sections added in this release:

| Section | Description |
|---------|-------------|
| **Risk metrics row** | Sharpe ratio, Sortino ratio, max drawdown %, total PnL % |
| **Drawdown curve** | Peak-to-trough running drawdown chart below equity curve |
| **Pattern-type breakdown** | Win-rate + expectancy per `pattern_type` (strict / exhaustion / absorption / eqh_breakout / none) |
| **Entry-attempt breakdown** | 1st touch vs 2nd touch vs 3rd+ touch — reveals which touch quality is highest |
| **Session-type filter** | Sidebar filter to segment all stats by pre_market / market_open / after_hours |
| **QQQ benchmark** | Equity curve overlaid with QQQ buy-and-hold normalised return |

> **Note:** Pattern-type and entry-attempt columns require trades logged with the updated pipeline (v2+). Legacy trades default to `pattern_type="none"`, `entry_attempt=1`.

---

## 🏛️ EQH Chart Overlays

In Chart Viewer, enable **"Show EQH pairs"** (sidebar checkbox):

| Visual | Meaning |
|--------|---------|
| 🟡 Gold dashed line | EQH ceiling level |
| 🟠 Orange square | EQH pair completion bar (C3) |
| 🟢 Green triangle-up | EQH breakout signal fired |
| 🔴 Red triangle-down | EQH rejection signal fired |
| Metric `EQH Pairs` | Total pairs detected in full history |

---

## 30-Day Rolling Cache

| Function | Behaviour |
|----------|-----------|
| `fetch_30day_bars(ticker)` | Full 30-day fetch → writes `{ticker}_30d_5m.parquet` |
| `refresh_today(ticker)` | Gap-aware in **minutes** — fetches exactly the missing bars since last bar |
| `load_30day_bars(ticker)` | Reads canonical parquet; empty DataFrame if not seeded |

**Pre/after-market included** (`prepost=True` to yfinance).
**Gap detection:** `gap_minutes < 6` = already current; otherwise fetches from `last_bar_date` forward.

**Morning Brief refresh is now non-blocking (first render ~instant):**
- **First render:** scores immediately from on-disk parquets; fires a background daemon thread to run `refresh_today()` for all tickers in parallel (no UI wait).
- **Subsequent ticks (≥60s):** blocks on a parallel `ThreadPoolExecutor(max_workers=12)` prefetch, then rescores from freshly-written disk — network + score in ~150ms for 8 tickers.
- **55-second cooldown guard:** tickers refreshed within the last 55s are skipped entirely, preventing redundant HTTP calls on rapid fragment reruns.

**Timing benchmark script:**
```bash
python side_by_side_backtest/_time_brief.py
```
Measures disk-only first-render path, background refresh wall-clock, and blocking subsequent-tick latency.

---

## Morning Brief Features

| Feature | Description |
|---------|-------------|
| Deduplication | 1 row per ticker; highest score wins; notes merged |
| Live Scanner | `@st.fragment(run_every=N)` — rescores without blocking tab navigation |
| Score delta (Δ) | `🟢▲` / `🔴▼` / `⬜=` column in ranked table |
| Session sparkline | Score-over-time mini chart inside setup card (after 3+ ticks) |
| Signal upgrade toast | SKIP→WATCH or WATCH→STRONG with reason string |
| Alert sound | `effect.mp3` via base64 `<audio autoplay>` in browser |
| S/R fallback | Stale/missing levels → auto-replaced by computed S/R |
| **Min-score slider** | Hide tickers below threshold (0–10, default 0) |
| **CSV export** | Download ranked table as `morning_brief_setups.csv` |
| **Session badges** | Count of pre-market / market-open / after-hours tickers above table |
| **EQH column** | Ceiling price displayed as `🏛️$X.XX` in ranked table |
| **EQH card** | Ceiling price + signal state shown inside each setup card |

---

## Database Schema

`watchlist_backtest.db` (SQLite, WAL mode):

**`trades` table** — all columns now persisted including analysis tags added in v2:

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | TEXT | Ticker symbol |
| `entry_ts` | TEXT | ISO entry timestamp |
| `entry_price` | REAL | Entry fill price |
| `exit_ts` | TEXT | ISO exit timestamp |
| `exit_price` | REAL | Exit fill price |
| `profit_target_pct` | REAL | PT% used |
| `stop_loss_pct` | REAL | SL% used |
| `outcome` | TEXT | `win` / `loss` / `timeout` |
| `pnl_pct` | REAL | PnL % |
| `support_respected` | INTEGER | 1 = support held in first 60 min |
| `support_source` | TEXT | `watchlist` or `computed` |
| `pattern_type` | TEXT | `strict` / `exhaustion` / `absorption` / `eqh_breakout` / `none` |
| `bars_since_pattern` | INTEGER | Bars between pattern bar and entry |
| `entry_attempt` | INTEGER | 1-based touch counter within session |

> Migration is automatic on connect — existing DBs gain the 4 new columns with safe defaults.

---

## 🃏 Card Mode Simulation
Backtests the strategy using the Card's own support/resistance levels instead of fixed percentages.
- **Rescore Stride:** Scans every bar (`stride=1`) for granular entry detection.
- **Regime Detection:** Identifies "choppy" vs "trending" days based on VWAP crossings in the first 60 minutes (4+ crosses = choppy).
- **Dynamic Adjustments:**
    - Exits at TP only if day is classified as "choppy".
    - Tightens stop to max 2% on choppy days.
    - Uses Linda Raschke 3/10 Oscillator (SMA based) to detect weakening momentum and tighten stops/TP during the trade.
- **Position Sizing:** Implements score-driven dynamic position sizing: `trade_size + 200 * (score - 4.3)`, capped between $100 and $1000.
- **Parallelization:** Uses `ProcessPoolExecutor` for parallel simulation across entries (faster batch runs).

---

## 📈 Chart Viewer — Y-Axis Auto-Fit & EQH Overlays

The Chart Viewer now behaves like TradingView:

| Improvement | Detail |
|-------------|--------|
| **Tight initial y-range** | On load, y-axis is set to the price range of the **last 1 day** of bars only (not the full 30-day history). `autorange=False`, `fixedrange=False` — users can still drag/zoom. |
| **Client-side JS auto-rescale** | After every x-axis pan or range-selector click, injected JS (`inject_yaxis_rescale_js`) recomputes min/max of visible candles and calls `Plotly.relayout` — zero server round-trips. |
| **Session shading via `add_shape`** | Pre-market and after-hours bands are drawn as `yref="paper"` layout shapes (excluded from autorange) — they can never force the y-axis wider than the visible candles. |
| **S/R lines via `add_shape`** | Support/resistance lines use `xref="paper"` shapes + `add_annotation` labels instead of `add_hline`, filtered to ±50% of current price. |
| **TP/SL/Entry lines via `add_shape`** | Take-profit, stop-loss, and entry reference lines are drawn as layout shapes (excluded from autorange) with right-margin annotations (`$X.XXX`). |
| **Trade band via `add_shape`** | Entry-to-exit filled rectangle uses a layout shape (excluded from autorange). |
| **EQH overlays** | `add_eqh_overlays()`: gold dashed ceiling lines, orange square markers, bright-green breakout arrows, bright-red rejection arrows. |
| **Pattern marker offset removed** | Pattern markers no longer nudge `y * 1.005` above the bar — removes spurious autorange expansion. |
| **S×S metric renamed** | `Patterns` metric card renamed to `S×S Patterns`; new `EQH Pairs` metric card added. |

---

## Walk-Forward OOS Validation

Added to `auto_tuner.py`. Enabled via `--validate-oos`:

```
[auto_tune] Walk-forward OOS (30% of data) —
  E=+0.0312  WR=58.3%  PF=1.84  trades=48  ✅ OOS holds up
```

- Splits entries **chronologically** (first 70% IS, last 30% OOS)
- Re-runs best IS params on OOS fold after Bayesian study completes
- Prints `✅ OOS holds up` or `⚠️ POSSIBLE OVERFITTING` based on whether OOS expectancy is positive
- `--oos-split 0.30` (default) — configurable fraction

---

## Sanity Check Suite (7 Checks)

```bash
./venv/bin/python -m side_by_side_backtest.run_sanity --tickers UGRO ANNA TURB
```

| Check | Tests |
|-------|-------|
| **A** — Shuffle Control | Real WR > random-support baseline |
| **B** — Out-of-Sample | Walk-forward (5 folds, 70/30) |
| **C** — Duplicate Trades | 0 duplicate (ticker, entry_ts) |
| **D** — Pattern Proximity | WR holds with strict proximity filter |
| **E** — Realistic Sweep | Positive expectancy at PT=1–5%, SL=0.5–2% |
| **F** — Per-Ticker | No single ticker > 70% of wins |
| **G** — Slippage Sensitivity | PF ≥ 1.0 at 0.2% slippage |

---

## CLI Reference

```
./venv/bin/python -m side_by_side_backtest.main [OPTIONS]
  --watchlist PATH        scraped_watchlists.json
  --tickers TICKER ...    auto-build watchlist from ohlcv_cache/ (skip JSON)
  --skip-fetch            use 30d parquet only (no network)
  --no-sweep              single-pass only (skip Phase 4 optimisation)
  --auto-tune             Bayesian PT/SL search (Optuna TPE)
  --n-trials N            default 100
  --tune-objective        expectancy | profit_factor | win_rate
  --validate-oos          walk-forward OOS check after auto-tune
  --oos-split FLOAT       OOS fraction (default 0.30)
  --eqh                   run in EQH Breakout mode (replaces bearish S×S)
  --export                CSV + PNG heatmaps
  --sanity                run 7-check suite

./venv/bin/python -m side_by_side_backtest.refresh_cache [OPTIONS]
  --full                  smart full re-fetch (merge-aware)
  --tickers TICKER...     specific tickers only
  --provider yfinance|alpaca

./venv/bin/python -m side_by_side_backtest.live_scanner [OPTIONS]
  --interval INT          poll seconds (default 300)
  --once                  single pass and exit
```

---

## Output Files

| File | Description |
|------|-------------|
| `watchlist_backtest.db` | SQLite: parsed entries + trade results with analysis tags |
| `banned_tickers.json` | Auto-banned 404/delisted tickers |
| `ohlcv_cache/{TICKER}_30d_5m.parquet` | Rolling 30-day 5-min OHLCV |
| `reports/optimization_results.csv` | Full sweep/tune grid |
| `reports/auto_tune_trials.csv` | Per-trial Bayesian search history |
| `reports/heatmap_*.png` | Win-rate / profit-factor / expectancy |
| `reports/equity_curve.png` | Cumulative PnL for best config |

---

## Unit Tests

```bash
# Run all three new unit test modules
python -m pytest tests/unit/test_pattern_engine.py \
                 tests/unit/test_setup_scorer.py \
                 tests/unit/test_simulator.py -v
```

| Module | Coverage |
|--------|---------|
| `test_pattern_engine.py` | EQH pair detection (bull+bear, bear+bull, rejects same-color, rejects far opens); breakout/rejection signal firing |
| `test_setup_scorer.py` | ADX tiers, R/R tiers, signal labels, score normalisation (max 24→10), EQH fields present |
| `test_simulator.py` | TP win, SL loss, time-stop, max-loss cap, penny-stock gate, max-attempts cap, EQH mode delegation |

---

## ⚠️ Disclaimer

This software is for **research and educational purposes only**. Past backtested performance does not guarantee future results. Do your own research. This is not financial advice. Trading involves significant risk of loss.

## File Structure

```
side_by_side_backtest/
├── app.py                  ← Multi-page Streamlit entry point (4 pages)
├── effect.mp3              ← Alert sound
├── pages/
│   ├── 1_morning_brief.py  ← Live triage + score delta + sparklines + history stats + streak
│   ├── 2_chart_viewer.py   ← Chart Viewer page (deep-link adapter)
│   ├── 3_performance.py    ← Simulated backtest analytics: equity curve, drawdown, Sharpe/Sortino,
│   │                          pattern-type breakdown, entry-attempt breakdown, session filter
│   └── 4_autonomous_pnl.py ← Autonomous PnL: paper/live trade equity curve, open positions,
│                              trade log, per-ticker breakdown
├── autonomous_config.py    ← Autonomous trading configuration (budget, trade size, circuit breaker)
├── decision_engine.py      ← Entry gate: circuit breaker + max concurrent + budget check
├── position_monitor.py     ← Exit engine: PT / SL / momentum fade / time stop
├── schwab_broker.py        ← Schwab OAuth 2.0 REST client (paper stub until credentials)
├── setup_scorer.py         ← 12-component signal scoring + history expectancy + streak
├── live_scanner.py         ← CLI alert loop + --autonomous flag for closed-loop execution
├── refresh_cache.py        ← Daily gap-aware 30d cache maintenance
├── data_fetcher.py         ← OHLCV fetcher + 30d rolling cache (intra-day aware, prepost=True)
├── simulator.py            ← Body-based entry simulator + EQH breakout mode + trailing stop
├── sr_engine.py            ← Multi-method S/R + role-reversal + rejection detection
├── pattern_engine.py       ← Strict S×S + Exhaustion S×S + Support Absorption + EQH detectors
├── models.py               ← PatternMatch, SetupScore (history stats, streak), TradeResult
├── parser.py               ← Watchlist NLP parser
├── main.py                 ← CLI backtest pipeline (--eqh, --validate-oos, --min-score)
├── optimizer.py            ← Brute-force PT×SL grid search
├── auto_tuner.py           ← Bayesian TPE search (Optuna) + walk-forward OOS validation
├── report.py               ← Console + CSV/PNG export
├── sanity_check.py         ← 7-check robustness suite
├── db.py                   ← SQLite: entries + simulated trades + actual_trades (paper/live)
└── ohlcv_cache/
    ├── {TICKER}_30d_5m.parquet   ← Rolling 30-day canonical cache
    └── {TICKER}_*_5m.parquet    ← Legacy per-window cache
```

---

## 🤖 Autonomous Trading System (v4 - Institutional)

A fully closed-loop, institutional-grade execution pipeline that wires together advanced risk management, volume constraints, and smart order routing. Operates via `live_scanner.py --autonomous`.

### ⚙️ Core Infrastructure (Phases 1 – 6)
- **Heartbeat (5-Min Aligned)**: Evaluates entries *only* on completed 5-minute candle closes, eliminating mid-bar noise and false support touches.
- **Volatility Sizing (Phase 1)**: Implements **ATR Risk Normalization** (`size_mode = "atr"`). Shares are calculated dynamically based on a fixed dollar risk budget and the ATR stop distance: `Size = Risk_Budget / (1.5 × ATR)`.
- **Liquidity Gate (Phase 2)**: Defeats the "lumpy volume" kurtosis problem in microcaps using a **Median-based Time-of-Day (TOD) Volume Profile**. Caps order size at **2% of the 20-day median volume** for the current 5-minute bucket to prevent market impact.
- **Dynamic Brackets (Phase 3)**: Bypasses static targets and enforces **Dynamic ATR Multipliers** (`SL = 1.5 × ATR`, `TP = 3.0 × ATR`) to let alpha breathe.
- **Gap Management (Phase 3)**: Traps overnight gap risk. If a stock opens below the Stop-Loss, the system immediately routes an emergency market sell at the `Open` and logs the true slippage.
- **Slicing & OCO Sync (Phase 4)**: The `SlicingEngine` splits large orders into chunks (Iceberg/TWAP) worked passively at the NBBO. The **Master Bracket Controller** dynamically scales up the Schwab-side OCO (bracket) quantity in real-time as chunk fills settle, eliminating "phantom stop" risk.

### 🎛️ Strategy Configuration — `autonomous_config.py`
The system runs both strategies in parallel, hardcoded with their **validated, unbiased champion parameters**:
- **Card Strategy (Champion 🏆)**: `min_score = 4.3`, `size_mode = "atr"`, `risk_budget = $20.00`, `use_atr = True`, `enable_slicing = True`, `liquidity_participation = 0.02`, `use_resistance_as_tp = False`.
- **Backtest Strategy (Anchor 📊)**: `min_score = 0.0` (pattern-only), `size_mode = "atr"`, `risk_budget = $10.00`, `use_atr = True`, `enable_slicing = False`, `liquidity_participation = 0.02`.

---

## 🤖 Autonomous PnL Dashboard (Upgraded)

Open **http://localhost:8501** → **🤖 Autonomous PnL**

The dashboard runs a live 60-second refresh fragment. The **🔓 Open Positions** panel has been upgraded to expose the active institutional execution state:
*   **TOD Cap**: Displays the maximum shares allowed by the Phase 2 Liquidity Gate for the current time bucket.
*   **OCO Status**: Displays the active OCO Bracket ID currently working at Schwab (rendered as `paper-oco-TICKER` in paper mode).
*   **ATR Entry**: Displays the locked-in ATR value captured at the exact moment of fill for sizing auditability.

---

---

## 🛠️ CLI Reference (Updated)

### 1. Historical Validation & PnL Re-Simulation (`run_today.py`)
Used to simulate historical performance across all 54 sessions *exactly as if you traded the production bot today* (includes bug fixes, ATR sizing, volume caps, and brackets).

```bash
# The Champion: Card Strategy (Unbiased / Loose Mode)
python3 side_by_side_backtest/run_today.py --mode card --all --size-mode atr --risk-budget 20 --use-atr --no-support-ok

# The Anchor: Backtest Strategy (Strict Heartbeat Mode)
python3 side_by_side_backtest/run_today.py --mode sbs --all --size-mode atr --risk-budget 10 --use-atr

# Core Options:
#   --mode sbs|card         Execution mode (default: sbs)
#   --all                   Run across all 54 historical sessions
#   --size-mode flat|atr    Sizing model: flat dollar vs. ATR Risk-based
#   --risk-budget X         Dollar risk per trade if size-mode=atr (default $10)
#   --use-atr               Force dynamic ATR brackets (SL=1.5x, TP=3.0x)
#   --no-support-ok         Disable the 60-min look-ahead safety gate (Loose mode)
#   --min X                 Minimum score for card mode (default 4.3)
```

### 2. Live Scanner & Paper Trading
```bash
# Run autonomous paper trading (uses CONFIG singleton)
python3 -m side_by_side_backtest.live_scanner --watchlist scraped_watchlists.json --autonomous

# Standard options:
#   --interval INT          Poll seconds (default 300 for 5-min heartbeat)
#   --once                  Single pass and exit
#   --no-notifications      Suppress macOS alerts and sounds
```

### 3. Broker Auth
```bash
# Interactive browser-based Schwab OAuth consent flow
python3 -m side_by_side_backtest.schwab_broker --auth

# Print account hashes after auth
python3 -m side_by_side_backtest.schwab_broker --get-account-hash
```

---
