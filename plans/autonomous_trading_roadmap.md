# Autonomous Trading System — Roadmap

## Vision

A fully closed-loop system:
```
Watchlist → Score (≥5) → Decide → Execute (Schwab API) → Monitor → Record real fills → Improve scoring
```

Ring-fenced budget ($2,000), $500/trade, halt if down $300/day. Paper mode first, then flip to live when Schwab API credentials arrive.

---

## Current State

The Side-by-Side Backtest system is a **manual-assist** tool:
- Morning Brief scores and ranks setups from Reddit watchlists
- Live Scanner fires alerts when a signal fires
- Simulator records **ideal** trades (perfect fills at next bar open, clean PT/SL exits)
- Performance analytics show expectancy based on **simulated** trade history
- Actual Schwab fills never enter the system

**Key gap:** `history_score` in setup cards and performance analytics are calibrated to simulated trades, not real fills. Autonomous execution will generate real fills from day one, closing this loop automatically.

---

## Configuration

All parameters live in `side_by_side_backtest/autonomous_config.py`:

```python
AUTONOMOUS_CONFIG = {
    "budget_total":        2000.0,  # $ allocated to algo
    "trade_size":           500.0,  # $ per trade (flat sizing)
    "max_concurrent":          4,   # budget / trade_size
    "daily_loss_halt":      300.0,  # $ — halt all trading if daily PnL < -this
    "min_score":               5.0, # setup score gate (5 = WATCH or STRONG)
    "trailing_activate_pct":   0.5, # % gain before trailing high-water activates
    "momentum_fade_pct":       0.3, # % retreat from high-water → early exit
    "paper_mode":             True, # flip to False when Schwab API live
    "poll_interval_sec":      300,  # 5-min bar poll interval for position monitor
}
```

---

## Architecture

```
scraped_watchlists.json / pinned tickers
        ↓
  Setup Scorer (score_setup)
        ↓ score ≥ 5?
  Decision Engine              ← checks circuit breaker + budget + max concurrent
        ↓
  Order Builder                ← quantity = $500 / entry_price, limit = entry_price + 0.5%
        ↓
  ┌── Paper mode ──┐  ┌─── Live mode ───┐
  │ Paper Logger   │  │ Schwab Broker   │  ← OAuth REST API
  └───────┬────────┘  └────────┬────────┘
          └─────────┬──────────┘
             Position Monitor              ← polls 5-min bars
                    ↓
         Exit conditions (first hit):
           A. PT hit
           B. SL hit
           C. Momentum fade (high-water retreat > momentum_fade_pct)
           D. Time stop (session close)
                    ↓
          actual_trades table              ← real/paper fills recorded
                    ↓
          Circuit breaker check            ← daily PnL < -$300 → HALT
                    ↓
          history_score update             ← real WR feeds back into scorer
```

---

## Exit Logic Detail

### Standard exits (already in simulator)
- **Take-Profit:** price ≥ entry × (1 + PT%)
- **Stop-Loss:** close ≤ entry × (1 - SL%)
- **Time stop:** session close (4 PM ET)

### Momentum Fade Exit (new — what the trader does manually)
Triggers early exit when price stalls before reaching PT:
1. Once position is up ≥ `trailing_activate_pct` (0.5%), begin tracking the highest close seen (`high_water`)
2. On each bar: if `current_close < high_water × (1 - momentum_fade_pct)` → exit at market
3. Also trigger if MACD histogram turns from rising to falling for 2 consecutive bars while position is profitable

This captures the "price isn't going up anymore but PT is still far away" scenario without waiting for a hard reversal.

---

## Build Sequence

### Step 1 — Prerequisites (Code mode)
- [ ] Add `--min-score` flag to `main.py` — close the gap between morning brief score filter and batch backtest
- [ ] Create `side_by_side_backtest/autonomous_config.py` — single source of truth for all parameters

### Step 2 — Decision Engine + Paper Logger (Code mode)
- [ ] `side_by_side_backtest/decision_engine.py`
  - Reads scorer output
  - Checks: score ≥ min_score, daily loss < halt threshold, open positions < max_concurrent, budget remaining
  - Returns: go/no-go + position size
- [ ] `side_by_side_backtest/db.py` — add `actual_trades` table with columns:
  `ticker, entry_ts, entry_price, exit_ts, exit_price, quantity, pnl_dollar, pnl_pct, outcome, source (paper/live), exit_reason`
- [ ] `side_by_side_backtest/paper_logger.py` — writes paper trades to actual_trades table

### Step 3 — Position Monitor (Code mode)
- [ ] `side_by_side_backtest/position_monitor.py`
  - Polls 5-min OHLCV bars for all open positions
  - Checks all four exit conditions per bar
  - Momentum fade: tracks high_water, checks MACD histogram
  - On exit: writes to actual_trades, updates daily PnL, checks circuit breaker

### Step 4 — Schwab Broker Client (Code mode, stub first)
- [ ] `side_by_side_backtest/schwab_broker.py`
  - OAuth 2.0 flow (browser-based first auth, refresh token thereafter)
  - `place_order(ticker, quantity, limit_price)` → order ID
  - `get_order_status(order_id)` → filled / pending / cancelled
  - `cancel_order(order_id)`
  - `get_positions()` → current open positions
  - Paper mode stub: returns mock responses, logs to actual_trades

### Step 5 — Wire into Live Scanner (Code mode)
- [ ] Extend `side_by_side_backtest/live_scanner.py`
  - On signal fire: call Decision Engine → Order Builder → Broker Client
  - Spawn Position Monitor for each accepted trade
  - Daily circuit breaker check on each poll cycle

### Step 6 — Schwab API Credentials (Manual)
- [ ] Register at developer.schwab.com
- [ ] Create app, set callback URL to https://127.0.0.1
- [ ] Store `client_id` and `client_secret` in `.env` file (never commit)
- [ ] Run first OAuth consent flow in browser
- [ ] Flip `paper_mode: False` in `autonomous_config.py`

---

## Score Gate Validation

Before running live, run the batch backtest with and without score gate to confirm ≥5 improves expectancy:

```bash
# Baseline (no gate — current)
python -m side_by_side_backtest.main \
  --watchlist scraped_watchlists.json \
  --auto-tune --n-trials 100 --sanity

# With score gate at 5
python -m side_by_side_backtest.main \
  --watchlist scraped_watchlists.json \
  --min-score 5 --auto-tune --n-trials 100 --sanity
```

Expected result: score ≥ 5 subset should show higher WR, higher expectancy, and higher profit factor than the full unfiltered universe.

---

## Risk Controls Summary

| Control | Value | Where enforced |
|---------|-------|----------------|
| Score gate | ≥ 5.0 | Decision Engine |
| Max concurrent positions | 4 | Decision Engine |
| Per-trade size | $500 flat | Order Builder |
| Daily loss halt | $300 | Circuit breaker in Position Monitor |
| Paper mode | True until API live | Broker Client |
| Session time stop | 4 PM ET | Position Monitor |
| Momentum fade | 0.3% from high-water | Position Monitor |

---

## Card Enhancements (lower priority, build alongside)

- [ ] `_load_all_history()` extended: return expectancy + avg_win + avg_loss + last_5_streak
- [ ] `SetupScore` gains new fields: `expectancy`, `avg_win`, `avg_loss`, `last_5_streak`
- [ ] `_render_card()` updated: replace single history bar with rich stats block
