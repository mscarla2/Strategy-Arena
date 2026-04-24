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
    min_score:           float = 0.0    # 0 = no gate; 4.0+ = card strategy threshold
    require_support_touch: bool = False  # if True, price must be within touch_band of support

    # ── Circuit breaker ──────────────────────────────────────────────────────
    daily_loss_halt: float = 150.0  # $ — halt THIS strategy if daily PnL < -this

    # ── Entry ─────────────────────────────────────────────────────────────────
    entry_limit_slippage: float = 0.005  # limit = entry_price * (1 + this)

    # ── Exit — standard ──────────────────────────────────────────────────────
    default_pt_pct:      float = 3.1
    default_sl_pct:      float = 1
    use_resistance_as_tp: bool = False  # if True, use SetupScore.resistance as TP price
    use_computed_sl:      bool = False  # if True, use SetupScore.stop as SL price
    max_hold_days:  int   = 1       # force exit after N calendar days (0 = no limit)

    # ── Exit — momentum fade ─────────────────────────────────────────────────
    macd_fade_bars:        int   = 2

    # ── Phase 6: Advanced Execution & Sizing ─────────────────────────────────
    use_atr:                    bool  = True    # Use ATR-based brackets
    tp_atr_mult:                float = 3.0     # TP = 3.0x ATR
    sl_atr_mult:                float = 1.5     # SL = 1.5x ATR
    enable_slicing:             bool  = False   # Enable Iceberg/TWAP slicing
    max_slice_shares:           int   = 1000    # Max shares per slice
    liquidity_participation:    float = 0.02    # Max 2% of TOD volume
    size_mode:                  str   = "flat"  # 'flat' | 'atr'
    risk_budget:                float = 10.0    # $ risk per trade if size_mode='atr'


@dataclass
class AutonomousConfig:
    """Master configuration for the autonomous trading system."""

    # ── Strategies ────────────────────────────────────────────────────────────
    card_strategy: StrategyConfig = field(default_factory=lambda: StrategyConfig(
        name="card_strategy",
        display_name="📋 Card Strategy (score ≥ 4.3 + ATR Sizing)",
        budget_total=5_000.0,       # $5,000 total allocated
        trade_size=1_000.0,         # Determines max_concurrent = 5
        min_score=4.3,              # Validated champion threshold
        require_support_touch=True, # Price must be near support
        daily_loss_halt=300.0,
        use_atr=True,               # Phase 3: Active ATR Brackets
        tp_atr_mult=3.0,            # 3.0x ATR Target
        sl_atr_mult=1.5,            # 1.5x ATR Stop
        enable_slicing=True,        # Phase 4: Active execution slicing
        max_slice_shares=1000,
        liquidity_participation=0.02, # Phase 2: 2% TOD Liquidity Gate
        size_mode="atr",            # Phase 1: Volatility-Adjusted Sizing
        risk_budget=20.0,           # $20 risk per trade (High confidence)
        use_resistance_as_tp=False, # Bypass rigid watchlist levels
    ))

    backtest_strategy: StrategyConfig = field(default_factory=lambda: StrategyConfig(
        name="backtest_strategy",
        display_name="📊 Backtest Strategy (pattern only + ATR Sizing)",
        budget_total=1_000.0,       # $1,000 total allocated
        trade_size=100.0,           # Determines max_concurrent = 10
        min_score=0.0,              # No score gate — pure pattern
        daily_loss_halt=150.0,
        use_atr=True,               # Phase 3: Active ATR Brackets
        tp_atr_mult=3.0,            # 3.0x ATR Target
        sl_atr_mult=1.5,            # 1.5x ATR Stop
        enable_slicing=False,       # Slicing inactive (smaller sizes)
        liquidity_participation=0.02, # Phase 2: 2% TOD Liquidity Gate
        size_mode="atr",            # Phase 1: Volatility-Adjusted Sizing
        risk_budget=10.0,           # $10 risk per trade (Conservative)
        use_resistance_as_tp=False, # Bypass rigid watchlist levels
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
        """Return active strategies for the autonomous trading loop."""
        return [self.card_strategy, self.backtest_strategy]

    @property
    def total_budget(self) -> float:
        return sum(s.budget_total for s in self.strategies)


# Module-level singleton — import and use directly:
#   from side_by_side_backtest.autonomous_config import CONFIG
CONFIG = AutonomousConfig()

# Convenience aliases for single-strategy access
CARD_STRATEGY     = CONFIG.card_strategy
BACKTEST_STRATEGY = CONFIG.backtest_strategy
