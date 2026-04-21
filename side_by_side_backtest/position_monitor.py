"""
Position Monitor
================
Monitors open autonomous positions bar-by-bar and fires exits when
any exit condition is met.

Exit conditions (first hit wins):
  A. Take-Profit  : close >= entry * (1 + pt_pct / 100)
  B. Stop-Loss    : close <= entry * (1 - sl_pct / 100)
  C. Momentum Fade: close retreats > momentum_fade_pct% from high-water
                    (only active once position is up >= trailing_activate_pct%)
                    OR MACD histogram turns falling for macd_fade_bars consecutive bars
  D. Time Stop    : bar timestamp >= session close (21:00 UTC = 4 PM ET)

Paper mode: fills recorded to actual_trades at the simulated close price.
Live mode : sends cancel + market-sell order via SchwabBroker (stubbed until live).

Usage
-----
    from side_by_side_backtest.position_monitor import PositionMonitor
    from side_by_side_backtest.autonomous_config import CONFIG

    monitor = PositionMonitor(CONFIG)
    # Called once per 5-min poll cycle from the live scanner:
    monitor.check_all_positions(bars_map)
"""
from __future__ import annotations

import logging
from datetime import datetime, time, timezone
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Session close in UTC — 21:00 UTC = 4:00 PM ET
_SESSION_CLOSE_UTC = time(21, 0, 0)


class PositionMonitor:
    """
    Monitors all open actual_trades positions and closes them when an
    exit condition fires.
    """

    def __init__(self, config=None, db=None, broker=None) -> None:
        if config is None:
            from .autonomous_config import CONFIG
            config = CONFIG
        self._cfg    = config   # AutonomousConfig (master)
        self._db     = db       # WatchlistDB, opened lazily
        self._broker = broker   # SchwabBroker stub, None in paper mode

    # ------------------------------------------------------------------
    # Strategy-config resolver
    # ------------------------------------------------------------------

    def _strategy_cfg(self, strategy_name: str):
        """
        Return the StrategyConfig for the given strategy_name.
        Falls back to backtest_strategy if the name is unrecognised.
        Works whether self._cfg is an AutonomousConfig or a bare StrategyConfig
        (the latter being used in unit tests).
        """
        from .autonomous_config import StrategyConfig
        if isinstance(self._cfg, StrategyConfig):
            return self._cfg
        # AutonomousConfig — look up by name
        for s in self._cfg.strategies:
            if s.name == strategy_name:
                return s
        return self._cfg.backtest_strategy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_all_positions(self, bars_map: Dict[str, pd.DataFrame]) -> List[dict]:
        """
        Iterate over all open positions and evaluate exit conditions
        against the latest bars.

        bars_map : {ticker: DataFrame of 5-min OHLCV bars (UTC index)}
        Returns   : list of closed trade dicts (for logging / UI updates)
        """
        db   = self._db_conn()
        source = "paper" if self._cfg.paper_mode else "live"
        open_trades = db.load_actual_trades(source=source, open_only=True)

        closed = []
        for trade in open_trades:
            ticker = trade["ticker"]
            bars   = bars_map.get(ticker)
            if bars is None or bars.empty:
                continue

            result = self._evaluate_position(trade, bars)
            if result:
                closed.append(result)

        return closed

    # ------------------------------------------------------------------
    # Per-position evaluation
    # ------------------------------------------------------------------

    def _evaluate_position(self, trade: dict, bars: pd.DataFrame) -> Optional[dict]:
        """
        Check a single open position against its exit conditions.
        Returns a closed-trade dict if exit triggered, else None.
        """
        strategy_name = trade.get("strategy_name", "backtest_strategy")
        scfg          = self._strategy_cfg(strategy_name)

        entry_price = trade["entry_price"]
        entry_ts    = pd.Timestamp(trade["entry_ts"])
        pt_pct      = trade.get("pt_pct") or scfg.default_pt_pct
        sl_pct      = trade.get("sl_pct") or scfg.default_sl_pct
        row_id      = trade["id"]
        quantity    = trade.get("quantity", 1)

        # Only look at bars after entry
        post_entry = bars[bars.index > entry_ts].copy()
        if post_entry.empty:
            return None

        cols = [c.lower() for c in post_entry.columns]
        post_entry.columns = cols

        pt_price = entry_price * (1 + pt_pct / 100)
        sl_price = entry_price * (1 - sl_pct / 100)

        high_water    = entry_price
        macd_fall_cnt = 0
        prev_macd_hist: Optional[float] = None

        for ts, bar in post_entry.iterrows():
            close = float(bar.get("close", 0))
            if close <= 0:
                continue

            # ── Track high-water ─────────────────────────────────────
            if close > high_water:
                high_water = close

            # ── A. Take-Profit ───────────────────────────────────────
            if close >= pt_price:
                return self._close_position(
                    row_id, trade, ts, close, "pt", quantity, entry_price
                )

            # ── B. Stop-Loss ──────────────────────────────────────────
            if close <= sl_price:
                return self._close_position(
                    row_id, trade, ts, close, "sl", quantity, entry_price
                )

            # ── C. Momentum Fade ──────────────────────────────────────
            gain_pct = (close - entry_price) / entry_price * 100
            if gain_pct >= scfg.trailing_activate_pct:
                # High-water retreat check
                fade_threshold = high_water * (1 - scfg.momentum_fade_pct / 100)
                if close < fade_threshold:
                    return self._close_position(
                        row_id, trade, ts, close, "momentum_fade", quantity, entry_price
                    )

                # MACD histogram fade check
                macd_hist = self._macd_histogram(post_entry, ts)
                if macd_hist is not None and prev_macd_hist is not None:
                    if macd_hist < prev_macd_hist:
                        macd_fall_cnt += 1
                    else:
                        macd_fall_cnt = 0
                    if macd_fall_cnt >= scfg.macd_fade_bars:
                        return self._close_position(
                            row_id, trade, ts, close, "momentum_fade", quantity, entry_price
                        )
                if macd_hist is not None:
                    prev_macd_hist = macd_hist

            # ── D. Time Stop ──────────────────────────────────────────
            if hasattr(ts, 'timetz'):
                bar_time_utc = ts.tz_convert("UTC").time() if ts.tzinfo else ts.time()
                if bar_time_utc >= _SESSION_CLOSE_UTC:
                    return self._close_position(
                        row_id, trade, ts, close, "time_stop", quantity, entry_price
                    )

        return None  # position still open

    # ------------------------------------------------------------------
    # Exit execution
    # ------------------------------------------------------------------

    def _close_position(self, row_id: int, trade: dict, exit_ts,
                        exit_price: float, reason: str,
                        quantity: int, entry_price: float) -> dict:
        """Record the exit in the DB and (if live) send the sell order."""
        pnl_dollar = (exit_price - entry_price) * quantity
        pnl_pct    = (exit_price - entry_price) / entry_price * 100
        outcome    = "win" if pnl_dollar >= 0 else "loss"

        exit_ts_str = str(exit_ts)

        # Paper mode: log to DB
        db = self._db_conn()
        db.update_actual_trade_exit(
            row_id=row_id,
            exit_ts=exit_ts_str,
            exit_price=exit_price,
            exit_reason=reason,
            pnl_dollar=round(pnl_dollar, 4),
            pnl_pct=round(pnl_pct, 4),
            outcome=outcome,
        )

        # Live mode: send sell order via broker
        if not self._cfg.paper_mode and self._broker is not None:
            try:
                self._broker.place_order(
                    ticker=trade["ticker"],
                    side="sell",
                    quantity=quantity,
                    limit_price=None,  # market order on exit
                )
            except Exception as exc:
                logger.error(f"[position_monitor] Broker sell failed for {trade['ticker']}: {exc}")

        result = {
            "ticker":     trade["ticker"],
            "row_id":     row_id,
            "exit_ts":    exit_ts_str,
            "exit_price": exit_price,
            "reason":     reason,
            "pnl_dollar": round(pnl_dollar, 4),
            "pnl_pct":    round(pnl_pct, 4),
            "outcome":    outcome,
        }
        logger.info(
            f"[position_monitor] EXIT {trade['ticker']} "
            f"reason={reason}  PnL=${pnl_dollar:+.2f} ({pnl_pct:+.2f}%)"
        )
        return result

    # ------------------------------------------------------------------
    # MACD histogram helper
    # ------------------------------------------------------------------

    @staticmethod
    def _macd_histogram(bars: pd.DataFrame, as_of_ts) -> Optional[float]:
        """
        Compute MACD histogram (12/26/9) on closes up to as_of_ts.
        Returns the current histogram value, or None if insufficient data.
        """
        closes = bars.loc[bars.index <= as_of_ts, "close"] if "close" in bars.columns \
            else bars.loc[bars.index <= as_of_ts, bars.columns[3]]  # fallback 4th col

        closes = closes.dropna()
        if len(closes) < 35:
            return None

        ema12 = closes.ewm(span=12, adjust=False).mean()
        ema26 = closes.ewm(span=26, adjust=False).mean()
        macd_line   = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram   = macd_line - signal_line
        return float(histogram.iloc[-1])

    # ------------------------------------------------------------------
    # Paper trade entry recording
    # ------------------------------------------------------------------

    def open_paper_position(self, ticker: str, entry_ts, entry_price: float,
                            quantity: int, setup_score: float,
                            pt_pct: float, sl_pct: float,
                            session_type: str = "unknown",
                            strategy_name: str = "backtest_strategy") -> int:
        """
        Record a new paper trade entry in actual_trades.
        Returns the DB row_id for subsequent exit update.
        """
        is_paper = getattr(self._cfg, "paper_mode", True)
        db = self._db_conn()
        row_id = db.insert_actual_trade({
            "ticker":        ticker,
            "source":        "paper" if is_paper else "live",
            "strategy_name": strategy_name,
            "setup_score":   setup_score,
            "entry_ts":      str(entry_ts),
            "entry_price":   entry_price,
            "quantity":      quantity,
            "exit_reason":   "open",
            "outcome":       "open",
            "pt_pct":        pt_pct,
            "sl_pct":        sl_pct,
            "session_type":  session_type,
        })
        logger.info(
            f"[position_monitor] OPEN {ticker} [{strategy_name}]  "
            f"entry=${entry_price:.4f}  qty={quantity}  "
            f"PT={pt_pct:.2f}%  SL={sl_pct:.2f}%  "
            f"({'paper' if is_paper else 'live'}) row_id={row_id}"
        )
        return row_id

    # ------------------------------------------------------------------
    # Lazy DB helper
    # ------------------------------------------------------------------

    def _db_conn(self):
        if self._db is not None:
            return self._db
        from .db import WatchlistDB
        db = WatchlistDB(self._cfg.db_path).connect()
        self._db = db
        return db
