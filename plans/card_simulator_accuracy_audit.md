# Card Strategy Simulator — Accuracy Audit

## What the simulator is trying to answer
> "What would have happened if you traded every qualifying morning-brief card for the last 30 days?"

The simulator in [`card_strategy_simulator.py`](../side_by_side_backtest/card_strategy_simulator.py) replays every session in the 30-day bar cache, scores each session open with `score_setup()`, and if `score >= min_score` enters at the open of the first session bar, then exits via the card's TP/SL levels or a session time-stop.

---

## Bug #1 — History score feeds forward into its own scoring window (data leakage)

### Where
[`_score_history()`](../side_by_side_backtest/setup_scorer.py:368) inside `score_setup()`, called from [`_score_all_sessions()`](../side_by_side_backtest/card_strategy_simulator.py:76).

### What happens
`_load_all_history()` loads **all trades ever stored in `watchlist_backtest.db`** — including trades that were simulated in the *future* relative to the session being scored. So when scoring session `2026-03-15`, the history component already includes trades from `2026-03-16` through `2026-04-19`.

A ticker that goes on a late-month win streak gets a boosted `history_score` of 2.0 on every earlier day it is scored — making the simulator enter it more aggressively during the early part of the window than it actually would have in real life.

### Impact
- Win rate is inflated (good tickers are entered more often because their future win record boosts the score)
- Trade count is inflated for historically winning tickers
- The `history_score` component (worth up to 2/24 = ~0.83 pts) can push a borderline ticker from SKIP to WATCH

### Fix
Pass a `cutoff_date` to `score_setup()` and filter the history lookup in `_load_all_history()` to only trades with `entry_ts < cutoff_date`.

---

## Bug #2 — S/R cache is shared across all scoring dates (stale levels)

### Where
[`_sr_cache`](../side_by_side_backtest/setup_scorer.py:33) in `setup_scorer.py`, TTL = 300 seconds.

### What happens
The simulator calls `score_setup()` for dozens of different historical dates within seconds (in parallel threads). The `_sr_cache` stores S/R levels keyed by **ticker** with a wall-clock TTL. Because all historical scoring happens in the same process, levels computed for `2026-04-19` are returned from cache when scoring `2026-03-20` — a date 30 days earlier with a completely different price.

This means:
- Support/resistance levels are anchored to the **most recent price**, not the price at the time of each historical session
- R/R ratios, confluence scores, rejection counts, and role-reversal scores are all calculated against the wrong price level

### Impact
- Every scored session uses today's S/R levels, not the period-accurate ones
- Confluence, rejection, and role-reversal scores can be wildly off for the early dates
- This also affects which support level is used as the SL and which resistance is the TP

### Fix
Either:
1. Disable the S/R cache entirely when running the card simulator (pass a unique per-date cache key), or
2. Key the cache by `(ticker, date_str)` instead of just `ticker`, and skip the wall-clock TTL check

---

## Bug #3 — Exit simulation scans ALL bars after entry, not just session bars

### Where
[`_simulate_from_entry()`](../side_by_side_backtest/card_strategy_simulator.py:179), lines 205–293.

### What happens
`post_bars = bars[bars.index > entry_ts]` returns every bar from entry forward — including bars from the next day, next week, etc. The session time-stop check at line 255 (`_bar_time_utc(ts) >= close_time`) fires as soon as *any* bar's UTC time reaches the close boundary, but overnight bars from the following day's pre-market can have times that are *less than* the close boundary, so the loop continues scanning into the next session.

Concretely: a MARKET_OPEN trade that doesn't hit TP/SL by 16:00 ET continues scanning pre-market bars from the next morning (04:00–09:30 ET), which all have timestamps `< 20:00 UTC`. The loop keeps going until either TP/SL is hit in after-hours of the SAME day or on a future day — which is not realistic.

### Impact
- Timeout trades that close at session-end are instead kept open and often find TP in the next session
- Win rate is significantly inflated (free overnight hold)
- Hold-bars metric is wrong

### Fix
Filter `post_bars` to only bars within the same calendar date AND within the session window before iterating:
```python
session_date = entry_ts.date()
post_bars = bars[
    (bars.index > entry_ts) &
    (bars.index.date == session_date) &  # same calendar day
    bars.index.map(lambda ts: _in_session(ts, session))  # same session
]
```

---

## Bug #4 — Daily loss halt uses entry date, not exit date for PnL accumulation

### Where
[`simulate_card_strategy()`](../side_by_side_backtest/card_strategy_simulator.py:389), lines 399–426.

### What happens
`trade_date = ets.date()` (entry timestamp) is used as the key for `daily_pnl`. But the dollar PnL from that trade (calculated from `exit_price`) is attributed to the entry date — even if the exit happens hours or a day later (because of Bug #3 above). This means:
- Day N's PnL accumulates trades entered on Day N, even if they didn't close until Day N+1
- The halt comparison happens BEFORE the trade is simulated, so a trade that would tip the halt is still taken

### Impact
Minor distortion in daily halt triggering — less impactful than Bug #3 but compounds with it.

### Fix
After fixing Bug #3, attribute `dollar_pnl` to `exit_ts.date()` OR check the halt gate after simulation and roll back if it would have been breached.

---

## Bug #5 — bars_map uses single latest 30-day snapshot for all historical dates

### Where
[`pages/3_performance.py`](../side_by_side_backtest/pages/3_performance.py:460-465), caller of the simulator.

### What happens
```python
bars_map[t] = load_30day_bars(t)   # loads today's 30d parquet
```
The bars file is built from today's `refresh_today()` call. So the "full bars history" passed to the simulator is exactly the same 30-day window that exists right now — but `_score_all_sessions()` iterates over every day in that window and scores it "as if" it was a live Morning Brief. 

This is actually the intended design, BUT: the bars themselves include ALL 30 days, meaning when `_score_all_sessions` scores Day 1 of the window, it passes `history = bars[bars.index < first_session_bar_ts]` — which on Day 1 is very short (only whatever bars exist before the first 5-min bar of that session). This is correct for Day 1. But for Day 25, the history slice includes bars from Days 1–25, which is the right shape.

The key concern: **the bars used for exit simulation also come from this same full 30-day set**, which means the exit loop in `_simulate_from_entry()` has access to all future bars within the 30-day window. This is what causes Bug #3 to be so severe — bars from day+1, day+2, etc. are all in scope.

### Fix
This reinforces the need for Bug #3's fix: the exit simulation must be strictly constrained to bars within the entry session only.

---

## Bug #6 — SL exit uses exact sl_price, not the bar's actual low (unrealistic fill)

### Where
[`_simulate_from_entry()`](../side_by_side_backtest/card_strategy_simulator.py:212-233), SL hit branch.

### What happens
When `low <= sl_price`, the exit price is recorded as exactly `sl_price`. In real trading, a stop-market order on a fast-moving 5-minute bar would fill at the open of the next bar (or worse). Setting `exit_price = sl_price` assumes perfect stop execution — always filling at exactly the stop level, never worse.

Similarly for TP: `exit_price = tp_price` assumes an exact limit fill at the TP level, not accounting for slippage on a limit order that's missed by a tick.

### Impact
- Loss trades are slightly understated (fill assumed at SL, not through it)
- Win trades are slightly overstated (fill assumed exactly at TP)
- Net effect inflates both win rate and average win size

### Fix
Apply a small slippage model:
- SL fill: `exit_price = low` (worst-case bar fill, or `sl_price * (1 - slippage_pct)`)
- TP fill: `exit_price = tp_price` is acceptable for limit orders, but note that if `high >= tp_price` the close might be well above TP (partial fill scenario is irrelevant for simulator purposes — limit fill at TP is fine)

---

## Bug #7 — Both TP and SL are checked but SL always wins on same-bar hit

### Where
[`_simulate_from_entry()`](../side_by_side_backtest/card_strategy_simulator.py:213-252), lines 213–234 — SL checked first unconditionally.

### What happens
The code checks `sl_hit` first. If both `low <= sl_price` AND `high >= tp_price` on the same bar, SL always takes priority. The comment says "conservative" but this is actually a pessimistic bias — there's no way to know intrabar order.

A more realistic approach checks which threshold was hit first intrabar (impossible without tick data), or splits the trade 50/50 when both fire on the same bar. The current approach creates a systematic downward bias on bars where the price whipsaws through both levels.

### Impact
Minor negative bias on PnL — makes the simulator look slightly worse than reality. Less important than Bugs 1-3 but worth noting.

---

## Bug #8 — Default fallback SL is 0.124% — far too tight for 5-min bars

### Where
[`_score_all_sessions()`](../side_by_side_backtest/card_strategy_simulator.py:168), line 168.

```python
sl_price = entry_price * (1 - 0.00124)   # 0.124% default SL
```

### What happens
When the card has no valid SL level (`sc.stop` is None or > entry), the fallback SL is entry × (1 - 0.00124). A 0.124% stop on a $5 stock is $0.006. The normal bid/ask spread on a volatile small-cap is 0.3–0.5%, meaning this stop is almost guaranteed to hit within a few bars on any normal price fluctuation — not because of a real breakdown.

### Impact
- Default-SL trades almost always become losses
- Win rate for cards without explicit stop levels is severely understated
- The `setup_scorer.py` default is 2% below entry (`entry_price * 0.98`), creating an inconsistency between what the card shows and what the backtest uses

### Fix
Align with `setup_scorer.py`: use `entry_price * (1 - 0.02)` (2%) as the fallback SL, or read `sc.stop` from the SetupScore (which already has the 2% fallback applied).

---

## Summary Table

| # | Bug | Severity | Direction of Bias |
|---|-----|----------|-------------------|
| 1 | History score data leakage (future trades in scoring) | HIGH | Inflates win rate |
| 2 | S/R cache shared across all historical dates | HIGH | Scores are wrong for most dates |
| 3 | Exit simulation crosses session boundaries | CRITICAL | Massively inflates win rate |
| 4 | Daily halt PnL keyed to entry date, not exit date | LOW | Minor halt timing error |
| 5 | bars_map contains future bars visible to exit sim | CRITICAL | Same root cause as #3 |
| 6 | SL/TP fills at exact price (no slippage) | MEDIUM | Slightly inflates P&L |
| 7 | Same-bar SL always beats TP (pessimistic) | LOW | Slightly deflates P&L |
| 8 | Default fallback SL is 0.124% (too tight) | MEDIUM | Deflates cards without explicit SL |

---

## Recommended Fix Order

1. **Bug #3 + #5** (exit session boundary) — single change, biggest accuracy impact
2. **Bug #1** (history leakage) — requires passing `cutoff_date` through the call chain
3. **Bug #2** (S/R cache) — add a `bypass_cache=True` flag in simulator context
4. **Bug #8** (fallback SL) — one-line fix, change 0.00124 → 0.02
5. **Bug #6** (SL slippage) — add optional `slippage_pct` param to `_simulate_from_entry`
6. **Bug #4** (daily halt date key) — minor cleanup
7. **Bug #7** (same-bar priority) — optional probabilistic split

---

## Implementation Plan (for Code mode)

### Step 1 — Fix exit session boundary (Bugs #3 & #5)
In `_simulate_from_entry()`, filter `post_bars` to same-day, same-session bars only:

```python
entry_date = entry_ts.tz_convert("UTC").date()
post_bars = bars[
    (bars.index > entry_ts) &
    pd.Series([ts.tz_convert("UTC").date() == entry_date for ts in bars.index],
              index=bars.index) &
    pd.Series([_in_session(ts, session) for ts in bars.index], index=bars.index)
]
```

### Step 2 — Fix history leakage (Bug #1)
Add `cutoff_date: Optional[date] = None` param to `score_setup()` and thread it through to `_score_history()` → `_load_all_history()`. In `_load_all_history()`, filter trades by `entry_ts < cutoff_date` before computing win rate.

### Step 3 — Fix S/R cache (Bug #2)  
In `card_strategy_simulator.py`, before calling `score_setup()`, clear or bypass the `_sr_cache` for the ticker. Add a `disable_sr_cache: bool = False` param to `score_setup()`.

### Step 4 — Fix fallback SL (Bug #8)
Change line 168 in `_score_all_sessions()`:
```python
# Before:
sl_price = entry_price * (1 - 0.00124)
# After:
sl_price = entry_price * 0.98   # 2% below entry, matches setup_scorer default
```

### Step 5 — Slippage model (Bug #6, optional)
Add `slippage_pct: float = 0.001` (0.1%) param to `_simulate_from_entry()`. On SL hit, use `min(sl_price, low) * (1 - slippage_pct)`.
