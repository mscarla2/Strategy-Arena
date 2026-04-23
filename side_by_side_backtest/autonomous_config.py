"""
Autonomous Trading Configuration
=================================
Defines two parallel paper trading strategies that run simultaneously:

  Strategy A — "Card Strategy"
    Entry: SetupScore ≥ 4.3 (mirrors your manual Morning Brief decisions)
    Budget: $1,000  |  Per trade: $500
    Tag: 'card_strategy'

  Strategy B — "Backtest Strategy"
    Entry: Pattern engine + support touch only (no score gate)
    Budget: $1,000  |  Per trade: $500
    Tag: 'backtest_strategy'

Both strategies share the same paper_mode flag. Flip to False after
Schwab API credentials are configured to go live.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StrategyConfig:
    """Per-strategy configuration for the autonomous trading loop."""

    name: str                    # 'card_strategy' | 'backtest_strategy'
    display_name: str            # Human-readable label for UI

    # ── Budget & sizing ───────────────────────────────────────────────────────
    budget_total: float = 1_000.0   # $ ring-fenced for this strategy
    trade_size:   float =   500.0   # $ per trade (flat dollar sizing)

    @property
    def max_concurrent(self) -> int:
        return max(1, int(self.budget_total // self.trade_size))

    # ── Signal gate ───────────────────────────────────────────────────────────
    min_score: float = 0.0      # 0 = no gate; 4.3 = card strategy threshold

    # ── Circuit breaker ──────────────────────────────────────────────────────
    daily_loss_halt: float = 150.0  # $ — halt THIS strategy if daily PnL < -this

    # ── Entry ─────────────────────────────────────────────────────────────────
    entry_limit_slippage: float = 0.005  # limit = entry_price * (1 + this)

    # ── Exit — standard ──────────────────────────────────────────────────────
    default_pt_pct: float = 1.5
    default_sl_pct: float = 0.124
    max_hold_days:  int   = 1       # force exit after N calendar days (0 = no limit)

    # ── Exit — momentum fade ─────────────────────────────────────────────────
    trailing_activate_pct: float = 0.5
    momentum_fade_pct:     float = 0.3
    macd_fade_bars:        int   = 2


@dataclass
class AutonomousConfig:
    """Master configuration for the autonomous trading system."""

    # ── Strategies ────────────────────────────────────────────────────────────
    card_strategy: StrategyConfig = field(default_factory=lambda: StrategyConfig(
        name="card_strategy",
        display_name="📋 Card Strategy (score ≥ 4.3)",
        budget_total=5_000.0,   # $500/trade × up to 10 concurrent qualifying setups
        trade_size=500.0,
        min_score=4.3,
        daily_loss_halt=300.0,
    ))

    backtest_strategy: StrategyConfig = field(default_factory=lambda: StrategyConfig(
        name="backtest_strategy",
        display_name="📊 Backtest Strategy (pattern only)",
        budget_total=5_000.0,   # $1,000/trade × 5 concurrent pattern setups
        trade_size=1_000.0,
        min_score=0.0,          # no score gate — pure pattern + support touch
        daily_loss_halt=150.0,
    ))

    # ── Shared settings ───────────────────────────────────────────────────────
    poll_interval_sec: int = 300    # 5-min bar polling interval
    paper_mode:        bool = False  # True = paper trades; False = live Schwab orders
    data_provider:     str  = "schwab_data"  # "schwab_data" | "yfinance" | "alpaca"

    # ── Paths ─────────────────────────────────────────────────────────────────
    db_path: Path = field(
        default_factory=lambda: Path(__file__).parent / "watchlist_backtest.db"
    )
    env_path: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / ".env"
    )

    @property
    def strategies(self) -> list[StrategyConfig]:
        """Return active strategies for the autonomous trading loop.
        card_strategy is intentionally excluded — backtest_strategy only.
        """
        return [self.backtest_strategy]

    @property
    def total_budget(self) -> float:
        return sum(s.budget_total for s in self.strategies)


# Module-level singleton — import and use directly:
#   from side_by_side_backtest.autonomous_config import CONFIG
CONFIG = AutonomousConfig()

# Convenience aliases for single-strategy access
CARD_STRATEGY     = CONFIG.card_strategy
BACKTEST_STRATEGY = CONFIG.backtest_strategy
