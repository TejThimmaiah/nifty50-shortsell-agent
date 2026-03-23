"""
Database Manager
Handles SQLite DB lifecycle: schema migrations, daily backups,
archiving old trades, and exposing clean query interfaces.
Also exports trade history to CSV for Excel analysis.
"""

import csv
import logging
import os
import shutil
import sqlite3
from datetime import date, datetime, timedelta
from typing import List, Optional, Dict
from config import DB_PATH

logger = logging.getLogger(__name__)

# Schema version — increment when schema changes
SCHEMA_VERSION = 3

BACKUP_DIR = os.path.join(os.path.dirname(DB_PATH), "..", "db", "backups")


class DBManager:
    """
    Manages the SQLite trade database.
    Run migrate() at startup to ensure schema is up to date.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    # MIGRATIONS
    # ──────────────────────────────────────────────────────────────

    def migrate(self):
        """Apply all pending schema migrations. Safe to run on every startup."""
        current = self._get_schema_version()
        logger.info(f"DB schema version: {current} → target: {SCHEMA_VERSION}")

        if current < 1:
            self._apply_v1()
        if current < 2:
            self._apply_v2()
        if current < 3:
            self._apply_v3()

        self._set_schema_version(SCHEMA_VERSION)
        logger.info(f"DB migrations complete (v{SCHEMA_VERSION})")

    def _apply_v1(self):
        """V1: Core trades table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol          TEXT NOT NULL,
                    direction       TEXT NOT NULL DEFAULT 'SHORT',
                    entry_price     REAL NOT NULL,
                    exit_price      REAL,
                    quantity        INTEGER NOT NULL,
                    pnl             REAL,
                    order_id        TEXT,
                    exit_order_id   TEXT,
                    status          TEXT DEFAULT 'OPEN',
                    trade_date      TEXT NOT NULL,
                    created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
                    closed_at       TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_date ON trades(trade_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status     ON trades(status)")
        logger.info("DB v1 applied")

    def _apply_v2(self):
        """V2: Add strategy, confidence, and sector columns."""
        with sqlite3.connect(self.db_path) as conn:
            for col, definition in [
                ("strategy",   "TEXT DEFAULT 'TECHNICAL'"),
                ("confidence", "REAL DEFAULT 0.0"),
                ("sector",     "TEXT"),
                ("reason",     "TEXT"),
                ("gap_pct",    "REAL"),
                ("mtf_aligned", "INTEGER DEFAULT 0"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {definition}")
                    logger.debug(f"  Added column: {col}")
                except sqlite3.OperationalError:
                    pass   # Column already exists

            # Daily P&L summary table (pre-aggregated for fast reporting)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_summary (
                    trade_date  TEXT PRIMARY KEY,
                    total_trades INTEGER DEFAULT 0,
                    wins        INTEGER DEFAULT 0,
                    losses      INTEGER DEFAULT 0,
                    total_pnl   REAL DEFAULT 0,
                    updated_at  TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
        logger.info("DB v2 applied")

    def _apply_v3(self):
        """V3: Agent state log for audit trail."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_events (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    symbol     TEXT,
                    message    TEXT,
                    severity   TEXT DEFAULT 'INFO',
                    timestamp  TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON agent_events(event_type)")
        logger.info("DB v3 applied")

    # ──────────────────────────────────────────────────────────────
    # BACKUP
    # ──────────────────────────────────────────────────────────────

    def backup(self) -> Optional[str]:
        """Create a timestamped copy of the DB. Returns backup path."""
        if not os.path.exists(self.db_path):
            return None
        os.makedirs(BACKUP_DIR, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = os.path.join(BACKUP_DIR, f"trades_{ts}.db")
        try:
            shutil.copy2(self.db_path, dest)
            logger.info(f"DB backup created: {dest}")
            self._cleanup_old_backups(keep_days=30)
            return dest
        except Exception as e:
            logger.error(f"DB backup failed: {e}")
            return None

    def _cleanup_old_backups(self, keep_days: int = 30):
        """Remove backup files older than keep_days."""
        if not os.path.exists(BACKUP_DIR):
            return
        cutoff = datetime.now() - timedelta(days=keep_days)
        for fname in os.listdir(BACKUP_DIR):
            fpath = os.path.join(BACKUP_DIR, fname)
            mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
            if mtime < cutoff:
                os.remove(fpath)
                logger.debug(f"Removed old backup: {fname}")

    # ──────────────────────────────────────────────────────────────
    # QUERIES
    # ──────────────────────────────────────────────────────────────

    def log_event(self, event_type: str, message: str,
                  symbol: str = None, severity: str = "INFO"):
        """Write an event to the agent_events audit log."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO agent_events (event_type, symbol, message, severity) VALUES (?,?,?,?)",
                    (event_type, symbol, message, severity)
                )
        except Exception as e:
            logger.debug(f"Event log error: {e}")

    def get_trades(
        self,
        start_date: str = None,
        end_date:   str = None,
        symbol:     str = None,
        status:     str = None,
    ) -> List[Dict]:
        """Flexible trade query with optional filters."""
        where, params = [], []
        if start_date:
            where.append("trade_date >= ?"); params.append(start_date)
        if end_date:
            where.append("trade_date <= ?"); params.append(end_date)
        if symbol:
            where.append("symbol = ?");     params.append(symbol)
        if status:
            where.append("status = ?");     params.append(status)

        sql = "SELECT * FROM trades"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY trade_date DESC, id DESC"

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Trade query error: {e}")
            return []

    def update_daily_summary(self, trade_date: str):
        """Recompute and store the daily summary for a given date."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute("""
                    SELECT COUNT(*) total,
                           SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) wins,
                           SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) losses,
                           COALESCE(SUM(pnl), 0) total_pnl
                    FROM trades WHERE trade_date = ? AND status = 'CLOSED'
                """, (trade_date,)).fetchone()

                conn.execute("""
                    INSERT OR REPLACE INTO daily_summary
                    (trade_date, total_trades, wins, losses, total_pnl, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (trade_date, row[0], row[1] or 0, row[2] or 0,
                      row[3], datetime.now().isoformat()))
        except Exception as e:
            logger.error(f"Daily summary update error: {e}")

    def get_daily_summaries(self, days: int = 30) -> List[Dict]:
        """Return daily P&L summaries for the last N days."""
        start = (date.today() - timedelta(days=days)).isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM daily_summary WHERE trade_date >= ? ORDER BY trade_date",
                    (start,)
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def export_to_csv(self, output_path: str, days: int = 30) -> str:
        """Export trade history to CSV for Excel analysis."""
        start = (date.today() - timedelta(days=days)).isoformat()
        trades = self.get_trades(start_date=start)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fields = [
            "symbol", "direction", "entry_price", "exit_price",
            "quantity", "pnl", "status", "strategy", "confidence",
            "sector", "reason", "trade_date", "closed_at",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(trades)

        logger.info(f"Exported {len(trades)} trades to {output_path}")
        return output_path

    def archive_old_trades(self, days_to_keep: int = 90):
        """Move trades older than N days to an archive table."""
        cutoff = (date.today() - timedelta(days=days_to_keep)).isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trades_archive AS
                    SELECT * FROM trades WHERE 0
                """)
                conn.execute(
                    "INSERT INTO trades_archive SELECT * FROM trades WHERE trade_date < ?",
                    (cutoff,)
                )
                cur = conn.execute("DELETE FROM trades WHERE trade_date < ?", (cutoff,))
                logger.info(f"Archived {cur.rowcount} trades older than {cutoff}")
        except Exception as e:
            logger.error(f"Archive error: {e}")

    # ──────────────────────────────────────────────────────────────
    # SCHEMA VERSION
    # ──────────────────────────────────────────────────────────────

    def _get_schema_version(self) -> int:
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT value FROM schema_meta WHERE key='version'"
                ).fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0

    def _set_schema_version(self, version: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY, value TEXT
                )
            """)
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('version', ?)",
                (str(version),)
            )


# Singleton
db_manager = DBManager()
