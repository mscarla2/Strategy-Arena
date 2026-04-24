"""
Pydantic schemas for watchlist entries and trade records.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class SessionType(str, Enum):
    PRE_MARKET = "pre_market"
    MARKET_OPEN = "market_open"
    AFTER_HOURS = "after_hours"
    UNKNOWN = "unknown"


class WatchlistEntry(BaseModel):
    """Parsed, structured representation of a single ticker within a watchlist post."""

    # Source metadata
    post_title: str
    post_timestamp: Optional[datetime] = None
    raw_text: str = ""

    # Parsed fields
    ticker: str = Field(..., description="Uppercase ticker symbol, e.g. AAPL")
    session_type: SessionType = SessionType.UNKNOWN
    support_level: Optional[float] = Field(None, description="Primary support price")
    resistance_level: Optional[float] = Field(None, description="Primary resistance / target price")
    stop_level: Optional[float] = Field(None, description="Explicit stop-loss price if mentioned")
    sentiment_notes: str = ""

    @validator("ticker")
    def ticker_uppercase(cls, v: str) -> str:  # noqa: N805
        return v.upper().strip()


class RawWatchlist(BaseModel):
    """One raw scraped post from scraped_watchlists.json."""

    title: str
    content: str
    timestamp: Optional[str] = None


class OHLCV(BaseModel):
    """Single 5-minute bar."""

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class PatternMatch(BaseModel):
    """A detected Side-by-Side body pattern at a specific bar index."""

    ticker: str
    ts: datetime  # timestamp of the completion (C3) candle
    bar_index: int
    candle1_open: float
    candle1_close: float
    candle2_open: float    # C2
    candle2_close: float   # C2
    candle3_open: float = 0.0   # C3 (completion candle) — 0.0 for backward compat
    candle3_close: float = 0.0  # C3
    in_downtrend: bool
    confidence_score: float = 1.0   # 1.0 = strict S×S, 0.6 = exhaustion, 0.7 = absorption, 0.8 = eqh_pair
    pattern_type: str = "strict"    # "strict" | "exhaustion" | "absorption" | "eqh_pair" | "eqh_breakout" | "eqh_rejection"
    eqh_level: float = 0.0          # EQH ceiling price (max of C2/C3 opens); 0.0 for non-EQH patterns


class TradeResult(BaseModel):
    """Outcome of a single simulated trade."""

    ticker: str
    entry_ts: datetime
    entry_price: float
    exit_ts: Optional[datetime] = None
    exit_price: Optional[float] = None
    profit_target_pct: float  # X% used for this run
    stop_loss_pct: float
    session_type: SessionType = SessionType.UNKNOWN
    outcome: str = "open"  # "win" | "loss" | "timeout" | "open"
    pnl_pct: float = 0.0
    hold_bars: int = 0  # number of 5-min bars held
    support_respected: bool = False  # did price stay above support in first 12 bars?

    # ── Analysis tags ────────────────────────────────────────────────────────
    # support_source: was the support level taken from the watchlist ("watchlist")
    #   or silently replaced by a computed S/R level ("computed")?
    support_source: str = "watchlist"   # "watchlist" | "computed"

    # pattern_type: which detector fired the entry signal.
    # "strict"      = classic 3-candle Side-by-Side White Lines
    # "exhaustion"  = exhaustion variant (widening spread / doji C3)
    # "absorption"  = volume-absorption near support
    # "none"        = no named pattern (bare support touch)
    pattern_type: str = "none"

    # bars_since_pattern: how many 5-min bars elapsed between the pattern bar
    # and the entry bar.  0 = entered on the very next bar after the pattern.
    # High values (≥5) indicate stale pattern — useful for decay analysis.
    bars_since_pattern: int = 0

    # entry_attempt: 1-based counter of how many times support was touched
    # within this session before this trade was taken.
    # 1 = first touch of session, 2 = second touch, etc.
    entry_attempt: int = 1

    # ATR at entry time for position sizing and dynamic brackets
    atr: float = 0.0


class BacktestSummary(BaseModel):
    """Aggregated performance for one (profit_target_pct, stop_loss_pct) combination."""

    profit_target_pct: float
    stop_loss_pct: float
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    timeouts: int = 0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    expectancy: float = 0.0
    profit_factor: float = 0.0
    avg_hold_bars: float = 0.0
    support_respect_rate: float = 0.0

    # Per-session breakdown
    session_win_rates: dict = Field(default_factory=dict)


class OptimizationResult(BaseModel):
    """Full sweep output."""

    summaries: List[BacktestSummary] = Field(default_factory=list)
    best_by_win_rate: Optional[BacktestSummary] = None
    best_by_profit_factor: Optional[BacktestSummary] = None
    best_by_expectancy: Optional[BacktestSummary] = None
