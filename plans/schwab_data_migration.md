# Plan: Migrate OHLCV Data from yfinance → Schwab Market Data API

## Goal
Replace yfinance as the primary 5-minute OHLCV data source with the Schwab Market Data API
(`https://api.schwabapi.com/marketdata/v1/pricehistory`). yfinance remains as a silent fallback.
**No existing parquet data is touched.**

---

## Schwab Price History Endpoint

```
GET /marketdata/v1/pricehistory
  ?symbol=TICKER
  &periodType=day
  &period=10              (up to 40 calendar days for 5-min)
  &frequencyType=minute
  &frequency=5
  &startDate=<epoch_ms>   (optional, overrides period)
  &endDate=<epoch_ms>     (optional)
  &needExtendedHoursData=true
```

Response JSON shape:
```json
{
  "symbol": "TLRY",
  "candles": [
    {"open": 7.5, "high": 7.6, "low": 7.4, "close": 7.55, "volume": 12000, "datetime": 1713877200000}
  ],
  "empty": false
}
```
`datetime` is **milliseconds since Unix epoch UTC**.

---

## Files Changed

### 1. `side_by_side_backtest/data_fetcher.py`

**Add two new private functions:**

```python
def _fetch_schwab_pricehistory(ticker, start_iso, end_iso) -> pd.DataFrame:
    """5-min OHLCV via Schwab /pricehistory. Returns empty DF on any error."""

def _fetch_schwab_daily(ticker, start_iso, end_iso) -> pd.DataFrame:
    """Daily OHLCV via Schwab /pricehistory (frequencyType=daily). Used for SPY benchmark."""
```

**Modify `fetch_5min_bars()`:**
```python
# New provider path:
if provider == "schwab_data":
    try:
        df = _fetch_schwab_pricehistory(ticker, start, end)
        if df.empty:
            raise ValueError("empty")
    except Exception as exc:
        logger.warning(f"[schwab_data] {ticker} failed ({exc}), falling back to yfinance")
        df = _fetch_yfinance(ticker, start, end)
```

**Modify `fetch_30day_bars()` and `refresh_today()`:**
- Add `provider="schwab_data"` as new default value (configurable via `AutonomousConfig.data_provider`)
- Passes `provider` through to `fetch_5min_bars()`

### 2. `side_by_side_backtest/autonomous_config.py`

```python
@dataclass
class AutonomousConfig:
    ...
    data_provider: str = "schwab_data"   # "schwab_data" | "yfinance" | "alpaca"
```

### 3. `side_by_side_backtest/live_scanner.py`

Pass `provider=CONFIG.data_provider` in:
- `refresh_today(t, provider=CONFIG.data_provider)` in the position-check bmap build
- `_check_ticker()` → `load_30day_bars()` is read-only so no change needed there

### 4. `side_by_side_backtest/pages/1_morning_brief.py`

```python
from side_by_side_backtest.autonomous_config import CONFIG
# In _prefetch_tickers():
refresh_today(ticker, provider=CONFIG.data_provider)
```

### 5. `side_by_side_backtest/simulator.py` (line ~949)

Replace direct `yf.download("SPY", ...)` with `_fetch_schwab_daily("SPY", ...)` with yfinance fallback.

### 6. `side_by_side_backtest/pages/3_performance.py` (line ~231)

Replace direct `yf.download(...)` with `_fetch_schwab_daily(...)` with yfinance fallback.

### 7. `side_by_side_backtest/refresh_cache.py`

Add `"schwab_data"` to `--provider` choices.

---

## Data Safety Guarantees

- `load_30day_bars()` is **read-only** — unchanged
- `_prune_and_save()` output format is **unchanged** — same parquet schema
- All existing `.parquet` files in `ohlcv_cache/` are untouched
- yfinance stays installed; only used as silent fallback
- The Schwab `_get_access_token()` / `_get()` methods in `schwab_broker.py` are reused as-is

---

## Fallback Chain

```
schwab_data provider
  → _fetch_schwab_pricehistory()
      → success: use Schwab data
      → error/empty: log warning → _fetch_yfinance() [existing path]
```

---

## Out of Scope

- The `evolution/` and `backtest/` directories (use yfinance via `data/fetcher.py`, separate codebase)
- Historical backtests beyond 40 calendar days of 5-min data (Schwab limit) — yfinance fallback handles this
- The `data/fetcher.py` file (separate from `side_by_side_backtest/data_fetcher.py`)
