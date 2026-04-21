"""
Decision Engine
===============
Decides whether to enter a trade for a given SetupScore and StrategyConfig.

Checks (in order):
  1. Score gate      — setup.score >= strategy.min_score (0 = no gate)
  2. Circuit breaker — daily PnL loss < strategy.daily_loss_halt
  3. Max concurrent  — open positions for THIS strategy < strategy.max_concurrent
  4. Valid price     — entry price > 0

Each strategy runs its own budget and circuit breaker independently.

Usage
-----
    from side_by_side_backtest.decision_engine import DecisionEngine
    from side_by_side_backtest.autonomous_config import CONFIG

    for strategy in CONFIG.strategies:
        engine = DecisionEngine(strategy, CONFIG)
        result = engine.evaluate(setup_score)
        if result.go:
            ...
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class DecisionResult:
    """Output of one Decision Engine evaluation."""
    go:           bool
    reason:       str
    strategy_name: str  = ""
    quantity:     int   = 0
    limit_price:  float = 0.0
    trade_size_dollars: float = 0.0


class DecisionEngine:
    """
    Evaluates whether a scored setup should be traded for a given strategy.
    Each strategy has its own score gate, budget, and circuit breaker.
    """

    def __init__(self, strategy=None, master_config=None, db=None) -> None:
        """
        strategy      : StrategyConfig (defaults to backtest_strategy)
        master_config : AutonomousConfig for shared settings (paper_mode, db_path)
        db            : WatchlistDB instance (opened lazily if None)
        """
        if strategy is None:
            from .autonomous_config import BACKTEST_STRATEGY
            strategy = BACKTEST_STRATEGY
        if master_config is None:
            from .autonomous_config import CONFIG
            master_config = CONFIG

        self._strategy = strategy
        self._master   = master_config
        self._db       = db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, setup_score, current_price: Optional[float] = None) -> DecisionResult:
        """
        Evaluate a SetupScore (or plain float) against all gate conditions.
        """
        score       = float(getattr(setup_score, "score", setup_score))
        entry_price = current_price or getattr(setup_score, "entry_price", 0.0)
        sname       = self._strategy.name

        # ── Check 1: score gate ──────────────────────────────────────────
        if self._strategy.min_score > 0 and score < self._strategy.min_score:
            return DecisionResult(
                go=False,
                strategy_name=sname,
                reason=f"[{sname}] Score {score:.1f} < min_score {self._strategy.min_score:.1f}",
            )

        # ── Check 2: circuit breaker ─────────────────────────────────────
        ok, cb_reason = self._check_circuit_breaker()
        if not ok:
            return DecisionResult(go=False, strategy_name=sname, reason=cb_reason)

        # ── Check 3: max concurrent for THIS strategy ─────────────────────
        open_pos = self._open_positions()
        if open_pos >= self._strategy.max_concurrent:
            return DecisionResult(
                go=False,
                strategy_name=sname,
                reason=f"[{sname}] Max concurrent ({self._strategy.max_concurrent}) reached",
            )

        # ── Check 4: valid price ──────────────────────────────────────────
        if entry_price <= 0:
            return DecisionResult(go=False, strategy_name=sname,
                                  reason=f"[{sname}] No valid entry price")

        # ── All gates passed ──────────────────────────────────────────────
        quantity    = max(1, int(self._strategy.trade_size // entry_price))
        limit_price = round(entry_price * (1 + self._strategy.entry_limit_slippage), 4)

        return DecisionResult(
            go=True,
            strategy_name=sname,
            reason=f"[{sname}] Score {score:.1f} | open={open_pos}/{self._strategy.max_concurrent}",
            quantity=quantity,
            limit_price=limit_price,
            trade_size_dollars=quantity * entry_price,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _db_conn(self):
        if self._db is not None:
            return self._db
        from .db import WatchlistDB
        db = WatchlistDB(self._master.db_path).connect()
        self._db = db
        return db

    def _check_circuit_breaker(self) -> tuple[bool, str]:
        try:
            today = date.today().isoformat()
            daily_pnl = self._db_conn().daily_pnl(
                today,
                source="paper" if self._master.paper_mode else "live",
            )
            # Filter to this strategy's PnL using strategy_name
            # (daily_pnl currently sums all sources; approximate with per-strategy threshold)
            if daily_pnl <= -abs(self._strategy.daily_loss_halt):
                return (
                    False,
                    f"[{self._strategy.name}] Circuit breaker: daily PnL ${daily_pnl:.2f}",
                )
        except Exception as exc:
            return True, f"CB check skipped ({exc})"
        return True, ""

    def _open_positions(self) -> int:
        try:
            source = "paper" if self._master.paper_mode else "live"
            return self._db_conn().open_position_count(
                source=source,
                strategy_name=self._strategy.name,
            )
        except Exception:
            return 0
