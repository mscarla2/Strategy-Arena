"""
Phase 1 — SQLite Storage Layer
Persists parsed WatchlistEntry objects so the expensive NLP parse only runs once.
Re-running with the same scraped_watchlists.json is idempotent.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from .models import SessionType, WatchlistEntry

_DEFAULT_DB = Path(__file__).parent / "watchlist_backtest.db"

_CREATE_DDL = """
CREATE TABLE IF NOT EXISTS watchlist_entries (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker            TEXT    NOT NULL,
    post_title        TEXT    NOT NULL,
    post_timestamp    TEXT,
    session_type      TEXT    NOT NULL DEFAULT 'unknown',
    support_level     REAL,
    resistance_level  REAL,
    stop_level        REAL,
    sentiment_notes   TEXT,
    raw_text          TEXT,
    UNIQUE(ticker, post_title, post_timestamp)
);

CREATE TABLE IF NOT EXISTS trades (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker              TEXT NOT NULL,
    entry_ts            TEXT NOT NULL,
    entry_price         REAL NOT NULL,
    exit_ts             TEXT,
    exit_price          REAL,
    profit_target_pct   REAL NOT NULL,
    stop_loss_pct       REAL NOT NULL,
    session_type        TEXT NOT NULL DEFAULT 'unknown',
    outcome             TEXT NOT NULL DEFAULT 'open',
    pnl_pct             REAL NOT NULL DEFAULT 0.0,
    hold_bars           INTEGER NOT NULL DEFAULT 0,
    support_respected   INTEGER NOT NULL DEFAULT 0,
    -- Analysis tags (added in Phase A migration — safe to add to existing DBs)
    support_source      TEXT NOT NULL DEFAULT 'watchlist',
    pattern_type        TEXT NOT NULL DEFAULT 'none',
    bars_since_pattern  INTEGER NOT NULL DEFAULT 0,
    entry_attempt       INTEGER NOT NULL DEFAULT 1
);
"""

# actual_trades — real or paper autonomous fills (separate from simulated trades)
_ACTUAL_TRADES_DDL = """
CREATE TABLE IF NOT EXISTS actual_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    source          TEXT    NOT NULL DEFAULT 'paper',       -- 'paper' | 'live'
    strategy_name   TEXT    NOT NULL DEFAULT 'backtest',    -- 'card_strategy' | 'backtest_strategy'
    setup_score     REAL    NOT NULL DEFAULT 0.0,           -- SetupScore at entry time

    -- Entry
    entry_ts        TEXT    NOT NULL,
    entry_price     REAL    NOT NULL,
    quantity        INTEGER NOT NULL DEFAULT 0,
    order_id        TEXT,                                   -- Schwab order ID (live only)

    -- Exit
    exit_ts         TEXT,
    exit_price      REAL,
    exit_reason     TEXT    NOT NULL DEFAULT 'open',        -- 'pt'|'sl'|'momentum_fade'|'time_stop'|'open'

    -- P&L
    pnl_dollar      REAL    NOT NULL DEFAULT 0.0,
    pnl_pct         REAL    NOT NULL DEFAULT 0.0,
    outcome         TEXT    NOT NULL DEFAULT 'open',        -- 'win'|'loss'|'open'

    -- Parameters used
    pt_pct          REAL    NOT NULL DEFAULT 0.0,
    sl_pct          REAL    NOT NULL DEFAULT 0.0,
    session_type    TEXT    NOT NULL DEFAULT 'unknown',
    atr             REAL    NOT NULL DEFAULT 0.0,

    UNIQUE(ticker, entry_ts, source, strategy_name)
);
"""

# Migration: add strategy_name column to existing actual_trades tables
_ACTUAL_TRADES_MIGRATION = [
    "ALTER TABLE actual_trades ADD COLUMN strategy_name TEXT NOT NULL DEFAULT 'backtest'",
    "ALTER TABLE actual_trades ADD COLUMN atr REAL NOT NULL DEFAULT 0.0",
]

# Migration: add analysis-tag columns to existing databases that pre-date Phase A.
# Each ALTER TABLE is guarded by a try/except so it's safe to run on a fresh DB
# (which already has the columns from _CREATE_DDL) and on any legacy DB.
_MIGRATION_DDL = [
    "ALTER TABLE trades ADD COLUMN support_source     TEXT    NOT NULL DEFAULT 'watchlist'",
    "ALTER TABLE trades ADD COLUMN pattern_type       TEXT    NOT NULL DEFAULT 'none'",
    "ALTER TABLE trades ADD COLUMN bars_since_pattern INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE trades ADD COLUMN entry_attempt      INTEGER NOT NULL DEFAULT 1",
]


class WatchlistDB:
    """Thin SQLite wrapper for parsed entries and trade results."""

    def __init__(self, db_path: str | Path = _DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> "WatchlistDB":
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.executescript(_CREATE_DDL)
        self._conn.executescript(_ACTUAL_TRADES_DDL)
        # Run schema migrations — safe on both new and legacy DBs.
        for stmt in _MIGRATION_DDL + _ACTUAL_TRADES_MIGRATION:
            try:
                self._conn.execute(stmt)
            except sqlite3.OperationalError:
                pass  # column already exists — skip
        self._conn.commit()
        return self

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "WatchlistDB":
        return self.connect()

    def __exit__(self, *_) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Call connect() or use as context manager first.")
        return self._conn

    # ------------------------------------------------------------------
    # Watchlist entries
    # ------------------------------------------------------------------

    def upsert_entries(self, entries: List[WatchlistEntry]) -> int:
        """Insert entries, silently skipping duplicates. Returns inserted count."""
        rows = [
            (
                e.ticker,
                e.post_title,
                e.post_timestamp.isoformat() if e.post_timestamp else None,
                e.session_type.value,
                e.support_level,
                e.resistance_level,
                e.stop_level,
                e.sentiment_notes,
                e.raw_text,
            )
            for e in entries
        ]
        cursor = self.conn.executemany(
            """
            INSERT OR IGNORE INTO watchlist_entries
                (ticker, post_title, post_timestamp, session_type,
                 support_level, resistance_level, stop_level,
                 sentiment_notes, raw_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()
        return cursor.rowcount

    def load_entries(
        self,
        ticker: Optional[str] = None,
        session_type: Optional[SessionType] = None,
    ) -> List[WatchlistEntry]:
        """Load entries with optional filters."""
        query = "SELECT * FROM watchlist_entries WHERE 1=1"
        params: list = []
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker.upper())
        if session_type:
            query += " AND session_type = ?"
            params.append(session_type.value)
        rows = self.conn.execute(query, params).fetchall()
        results: list[WatchlistEntry] = []
        for row in rows:
            results.append(
                WatchlistEntry(
                    ticker=row["ticker"],
                    post_title=row["post_title"],
                    post_timestamp=row["post_timestamp"],
                    session_type=SessionType(row["session_type"]),
                    support_level=row["support_level"],
                    resistance_level=row["resistance_level"],
                    stop_level=row["stop_level"],
                    sentiment_notes=row["sentiment_notes"] or "",
                    raw_text=row["raw_text"] or "",
                )
            )
        return results

    def all_tickers(self) -> List[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT ticker FROM watchlist_entries ORDER BY ticker"
        ).fetchall()
        return [r["ticker"] for r in rows]

    # ------------------------------------------------------------------
    # Trade results
    # ------------------------------------------------------------------

    def insert_trades(self, trades: list) -> None:
        """Bulk-insert TradeResult objects (including analysis-tag columns)."""
        from .models import TradeResult  # local import to avoid circular

        rows = [
            (
                t.ticker,
                t.entry_ts.isoformat(),
                t.entry_price,
                t.exit_ts.isoformat() if t.exit_ts else None,
                t.exit_price,
                t.profit_target_pct,
                t.stop_loss_pct,
                t.session_type.value,
                t.outcome,
                t.pnl_pct,
                t.hold_bars,
                int(t.support_respected),
                getattr(t, "support_source", "watchlist"),
                getattr(t, "pattern_type", "none"),
                getattr(t, "bars_since_pattern", 0),
                getattr(t, "entry_attempt", 1),
            )
            for t in trades
            if isinstance(t, TradeResult)
        ]
        self.conn.executemany(
            """
            INSERT INTO trades
                (ticker, entry_ts, entry_price, exit_ts, exit_price,
                 profit_target_pct, stop_loss_pct, session_type,
                 outcome, pnl_pct, hold_bars, support_respected,
                 support_source, pattern_type, bars_since_pattern, entry_attempt)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def upsert_trades(self, trades: list) -> None:
        """Clear existing trades and re-insert. Safe to call after every simulation run."""
        self.clear_trades()
        self.insert_trades(trades)

    def clear_trades(self) -> None:
        self.conn.execute("DELETE FROM trades")
        self.conn.commit()

    def clear_actual_trades(self) -> None:
        """Delete all rows from actual_trades (paper and live). Use for DB reset."""
        self.conn.execute("DELETE FROM actual_trades")
        self.conn.commit()

    def clear_all(self) -> None:
        """Wipe ALL tables — watchlist_entries, trades, actual_trades."""
        self.conn.execute("DELETE FROM actual_trades")
        self.conn.execute("DELETE FROM trades")
        self.conn.execute("DELETE FROM watchlist_entries")
        self.conn.commit()

    def load_trades(self, profit_target_pct: Optional[float] = None) -> list:
        """Load TradeResult rows."""
        from .models import SessionType, TradeResult  # local import

        query = "SELECT * FROM trades WHERE 1=1"
        params: list = []
        if profit_target_pct is not None:
            query += " AND profit_target_pct = ?"
            params.append(round(profit_target_pct, 4))
        rows = self.conn.execute(query, params).fetchall()

        from datetime import datetime

        results = []
        for row in rows:
            # Safely read analysis-tag columns — default to legacy values for
            # rows written before Phase A migration so old data stays usable.
            row_keys = row.keys()
            results.append(
                TradeResult(
                    ticker=row["ticker"],
                    entry_ts=datetime.fromisoformat(row["entry_ts"]),
                    entry_price=row["entry_price"],
                    exit_ts=datetime.fromisoformat(row["exit_ts"]) if row["exit_ts"] else None,
                    exit_price=row["exit_price"],
                    profit_target_pct=row["profit_target_pct"],
                    stop_loss_pct=row["stop_loss_pct"],
                    session_type=SessionType(row["session_type"]),
                    outcome=row["outcome"],
                    pnl_pct=row["pnl_pct"],
                    hold_bars=row["hold_bars"],
                    support_respected=bool(row["support_respected"]),
                    support_source=row["support_source"] if "support_source" in row_keys else "watchlist",
                    pattern_type=row["pattern_type"] if "pattern_type" in row_keys else "none",
                    bars_since_pattern=row["bars_since_pattern"] if "bars_since_pattern" in row_keys else 0,
                    entry_attempt=row["entry_attempt"] if "entry_attempt" in row_keys else 1,
                )
            )
        return results

    # ------------------------------------------------------------------
    # actual_trades — autonomous / paper trade fills
    # ------------------------------------------------------------------

    def insert_actual_trade(self, trade: dict) -> int:
        """
        Insert one actual (paper or live) trade into actual_trades.

        trade dict keys (all optional except ticker, entry_ts, entry_price):
          ticker, source, strategy_name, setup_score, entry_ts, entry_price,
          quantity, order_id, exit_ts, exit_price, exit_reason, pnl_dollar,
          pnl_pct, outcome, pt_pct, sl_pct, session_type
        Returns the row id.
        """
        cur = self.conn.execute(
            """
            INSERT OR IGNORE INTO actual_trades
                (ticker, source, strategy_name, setup_score,
                 entry_ts, entry_price, quantity,
                 order_id, exit_ts, exit_price, exit_reason,
                 pnl_dollar, pnl_pct, outcome, pt_pct, sl_pct, session_type, atr)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade.get("ticker", ""),
                trade.get("source", "paper"),
                trade.get("strategy_name", "backtest_strategy"),
                trade.get("setup_score", 0.0),
                str(trade.get("entry_ts", "")),
                trade.get("entry_price", 0.0),
                trade.get("quantity", 0),
                trade.get("order_id"),
                str(trade.get("exit_ts", "")) if trade.get("exit_ts") else None,
                trade.get("exit_price"),
                trade.get("exit_reason", "open"),
                trade.get("pnl_dollar", 0.0),
                trade.get("pnl_pct", 0.0),
                trade.get("outcome", "open"),
                trade.get("pt_pct", 0.0),
                trade.get("sl_pct", 0.0),
                trade.get("session_type", "unknown"),
                trade.get("atr", 0.0),
            ),
        )
        self.conn.commit()
        return cur.lastrowid or 0

    def update_actual_trade_exit(self, row_id: int, exit_ts: str, exit_price: float,
                                  exit_reason: str, pnl_dollar: float,
                                  pnl_pct: float, outcome: str) -> None:
        """Update exit fields on an open actual_trade row."""
        self.conn.execute(
            """
            UPDATE actual_trades
            SET exit_ts=?, exit_price=?, exit_reason=?,
                pnl_dollar=?, pnl_pct=?, outcome=?
            WHERE id=?
            """,
            (exit_ts, exit_price, exit_reason, pnl_dollar, pnl_pct, outcome, row_id),
        )
        self.conn.commit()

    def load_actual_trades(self, source: Optional[str] = None,
                           strategy_name: Optional[str] = None,
                           open_only: bool = False) -> list:
        """
        Return actual_trades rows as plain dicts.
        source='paper'|'live'|None (all).
        strategy_name='card_strategy'|'backtest_strategy'|None (all).
        open_only=True returns only rows where outcome='open'.
        """
        clauses, params = ["1=1"], []
        if source:
            clauses.append("source = ?")
            params.append(source)
        if strategy_name:
            clauses.append("strategy_name = ?")
            params.append(strategy_name)
        if open_only:
            clauses.append("outcome = 'open'")
        rows = self.conn.execute(
            f"SELECT * FROM actual_trades WHERE {' AND '.join(clauses)} "
            "ORDER BY entry_ts ASC",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def daily_pnl(self, date_iso: str, source: Optional[str] = None) -> float:
        """
        Sum of pnl_dollar for all closed actual_trades on date_iso (YYYY-MM-DD).
        Used by the circuit breaker in the Decision Engine.
        """
        clauses = ["DATE(exit_ts) = ?", "outcome != 'open'"]
        params: list = [date_iso]
        if source:
            clauses.append("source = ?")
            params.append(source)
        row = self.conn.execute(
            f"SELECT COALESCE(SUM(pnl_dollar), 0.0) FROM actual_trades "
            f"WHERE {' AND '.join(clauses)}",
            params,
        ).fetchone()
        return float(row[0]) if row else 0.0

    def open_position_count(self, source: Optional[str] = None,
                            strategy_name: Optional[str] = None) -> int:
        """Count of actual_trades rows currently open (outcome='open')."""
        clauses = ["outcome = 'open'"]
        params: list = []
        if source:
            clauses.append("source = ?")
            params.append(source)
        if strategy_name:
            clauses.append("strategy_name = ?")
            params.append(strategy_name)
        row = self.conn.execute(
            f"SELECT COUNT(*) FROM actual_trades WHERE {' AND '.join(clauses)}",
            params,
        ).fetchone()
        return int(row[0]) if row else 0
