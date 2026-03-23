"""
Trade Memory
The agent's experiential memory. Every trade is stored with its full context:
what signals fired, what the market was doing, what time, what sector,
what the outcome was. This raw experience is what the learning system trains on.

Over time the agent builds a rich understanding of:
  - Which patterns work in which market regimes
  - Which time windows are most profitable
  - Which sectors to avoid on which days
  - How to adjust confidence thresholds based on recent accuracy

Think of this as the agent's "hippocampus" — storing episodic memory.
"""

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Dict, List, Optional, Any
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

MEMORY_DB = os.path.join(os.path.dirname(__file__), "..", "db", "trade_memory.db")


@dataclass
class TradeMemory:
    """
    Complete context record for one trade.
    Everything the agent knew at entry, plus what happened.
    """
    # Identity
    trade_id:           str     = ""
    symbol:             str     = ""
    trade_date:         str     = ""
    entry_time:         str     = ""
    exit_time:          str     = ""

    # Strategy that triggered
    strategy:           str     = "TECHNICAL"      # TECHNICAL | GAP_UP | MTF | COMBINED
    signals_fired:      List[str] = field(default_factory=list)   # ["RSI_OVERBOUGHT", "BEARISH_ENGULFING", ...]
    candlestick_pattern: str    = ""
    mtf_aligned:        bool    = False
    mtf_alignment_count: int    = 0

    # Market context at entry
    nifty_change_pct:   float   = 0.0
    nifty_breadth:      str     = ""              # BULLISH | BEARISH | NEUTRAL
    market_regime:      str     = ""              # TRENDING_DOWN | RANGING | VOLATILE | TRENDING_UP
    fii_net_cr:         float   = 0.0
    india_vix:          float   = 0.0
    sector:             str     = ""
    sector_trend:       str     = ""              # DOWNTREND | SIDEWAYS | UPTREND
    time_of_day:        str     = ""              # EARLY (9:20-10:30) | MID | LATE (12-1pm)

    # Technical readings at entry
    rsi_at_entry:       float   = 0.0
    macd_histogram:     float   = 0.0
    bb_position:        float   = 0.0            # 0=lower band, 1=upper band
    volume_ratio:       float   = 0.0            # vs 20-day avg
    gap_pct:            float   = 0.0            # gap from prev close (if gap strategy)

    # Trade parameters
    entry_price:        float   = 0.0
    stop_loss:          float   = 0.0
    target:             float   = 0.0
    quantity:           int     = 0
    confidence_score:   float   = 0.0

    # Outcome
    exit_price:         float   = 0.0
    exit_reason:        str     = ""             # TARGET | STOP_LOSS | EOD_SQUAREOFF | MANUAL
    pnl:                float   = 0.0
    pnl_pct:            float   = 0.0
    holding_minutes:    int     = 0
    won:                bool    = False

    # Agent reflection (written by self-improver after close)
    lesson:             str     = ""
    what_worked:        str     = ""
    what_failed:        str     = ""


class TradeMemoryStore:
    """
    Persistent store for trade memories.
    Provides queries that the learning agents use to improve.
    """

    def __init__(self, db_path: str = None):
        self.db = db_path or MEMORY_DB
        os.makedirs(os.path.dirname(self.db), exist_ok=True)
        self._init_schema()

    # ──────────────────────────────────────────────────────────────
    # WRITE
    # ──────────────────────────────────────────────────────────────

    def record(self, memory: TradeMemory) -> str:
        """Store a trade memory. Returns its ID."""
        if not memory.trade_id:
            memory.trade_id = f"{memory.symbol}_{memory.trade_date}_{int(datetime.now().timestamp())}"

        with sqlite3.connect(self.db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO trade_memories
                (trade_id, symbol, trade_date, entry_time, exit_time,
                 strategy, signals_fired, candlestick_pattern, mtf_aligned, mtf_alignment_count,
                 nifty_change_pct, nifty_breadth, market_regime, fii_net_cr, india_vix,
                 sector, sector_trend, time_of_day,
                 rsi_at_entry, macd_histogram, bb_position, volume_ratio, gap_pct,
                 entry_price, stop_loss, target, quantity, confidence_score,
                 exit_price, exit_reason, pnl, pnl_pct, holding_minutes, won,
                 lesson, what_worked, what_failed)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                memory.trade_id, memory.symbol, memory.trade_date,
                memory.entry_time, memory.exit_time,
                memory.strategy, json.dumps(memory.signals_fired),
                memory.candlestick_pattern, int(memory.mtf_aligned), memory.mtf_alignment_count,
                memory.nifty_change_pct, memory.nifty_breadth, memory.market_regime,
                memory.fii_net_cr, memory.india_vix,
                memory.sector, memory.sector_trend, memory.time_of_day,
                memory.rsi_at_entry, memory.macd_histogram, memory.bb_position,
                memory.volume_ratio, memory.gap_pct,
                memory.entry_price, memory.stop_loss, memory.target,
                memory.quantity, memory.confidence_score,
                memory.exit_price, memory.exit_reason,
                memory.pnl, memory.pnl_pct, memory.holding_minutes, int(memory.won),
                memory.lesson, memory.what_worked, memory.what_failed,
            ))
        return memory.trade_id

    def update_outcome(self, trade_id: str, exit_price: float, exit_reason: str,
                       pnl: float, pnl_pct: float, holding_minutes: int,
                       lesson: str = "", what_worked: str = "", what_failed: str = ""):
        """Update a memory after trade closes."""
        with sqlite3.connect(self.db) as conn:
            conn.execute("""
                UPDATE trade_memories
                SET exit_price=?, exit_reason=?, pnl=?, pnl_pct=?, holding_minutes=?,
                    won=?, exit_time=?, lesson=?, what_worked=?, what_failed=?
                WHERE trade_id=?
            """, (
                exit_price, exit_reason, pnl, pnl_pct, holding_minutes,
                int(pnl > 0), datetime.now(IST).strftime("%H:%M:%S"),
                lesson, what_worked, what_failed, trade_id,
            ))

    # ──────────────────────────────────────────────────────────────
    # QUERIES FOR LEARNING
    # ──────────────────────────────────────────────────────────────

    def get_recent(self, days: int = 30) -> List[Dict]:
        """Get all trades from the last N days."""
        from datetime import timedelta
        since = (date.today() - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trade_memories WHERE trade_date >= ? AND exit_reason != '' ORDER BY trade_date DESC",
                (since,)
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_signal_stats(self, days: int = 30) -> Dict[str, Dict]:
        """
        For each signal that fired, what was the win rate?
        Returns: {"RSI_OVERBOUGHT": {"win_rate": 0.62, "avg_pnl": 450, "count": 28}, ...}
        """
        memories = self.get_recent(days)
        signal_stats: Dict[str, Dict] = {}

        for mem in memories:
            for signal in mem.get("signals_fired", []):
                if signal not in signal_stats:
                    signal_stats[signal] = {"wins": 0, "total": 0, "pnl_sum": 0}
                signal_stats[signal]["total"]   += 1
                signal_stats[signal]["pnl_sum"] += mem.get("pnl", 0)
                if mem.get("won"):
                    signal_stats[signal]["wins"] += 1

        return {
            sig: {
                "win_rate": round(d["wins"] / max(d["total"], 1), 3),
                "avg_pnl":  round(d["pnl_sum"] / max(d["total"], 1), 2),
                "count":    d["total"],
            }
            for sig, d in signal_stats.items()
            if d["total"] >= 3    # Only signals with enough samples
        }

    def get_pattern_stats(self, days: int = 30) -> Dict[str, Dict]:
        """Win rate by candlestick pattern."""
        memories = self.get_recent(days)
        stats: Dict[str, Dict] = {}
        for mem in memories:
            pat = mem.get("candlestick_pattern", "")
            if not pat:
                continue
            if pat not in stats:
                stats[pat] = {"wins": 0, "total": 0, "pnl_sum": 0}
            stats[pat]["total"] += 1
            stats[pat]["pnl_sum"] += mem.get("pnl", 0)
            if mem.get("won"):
                stats[pat]["wins"] += 1
        return {
            p: {"win_rate": round(d["wins"]/max(d["total"],1), 3),
                "avg_pnl":  round(d["pnl_sum"]/max(d["total"],1), 2),
                "count":    d["total"]}
            for p, d in stats.items() if d["total"] >= 2
        }

    def get_regime_stats(self, days: int = 60) -> Dict[str, Dict]:
        """Win rate by market regime."""
        memories = self.get_recent(days)
        stats: Dict[str, Dict] = {}
        for mem in memories:
            regime = mem.get("market_regime", "UNKNOWN")
            if not regime:
                continue
            if regime not in stats:
                stats[regime] = {"wins": 0, "total": 0, "pnl_sum": 0}
            stats[regime]["total"] += 1
            stats[regime]["pnl_sum"] += mem.get("pnl", 0)
            if mem.get("won"):
                stats[regime]["wins"] += 1
        return {
            r: {"win_rate": round(d["wins"]/max(d["total"],1), 3),
                "avg_pnl":  round(d["pnl_sum"]/max(d["total"],1), 2),
                "count":    d["total"]}
            for r, d in stats.items() if d["total"] >= 2
        }

    def get_time_stats(self, days: int = 30) -> Dict[str, Dict]:
        """Win rate by time of day (EARLY/MID/LATE)."""
        memories = self.get_recent(days)
        stats: Dict[str, Dict] = {}
        for mem in memories:
            tod = mem.get("time_of_day", "UNKNOWN")
            if tod not in stats:
                stats[tod] = {"wins": 0, "total": 0, "pnl_sum": 0}
            stats[tod]["total"]   += 1
            stats[tod]["pnl_sum"] += mem.get("pnl", 0)
            if mem.get("won"):
                stats[tod]["wins"] += 1
        return {
            t: {"win_rate": round(d["wins"]/max(d["total"],1), 3),
                "avg_pnl":  round(d["pnl_sum"]/max(d["total"],1), 2),
                "count":    d["total"]}
            for t, d in stats.items() if d["total"] >= 2
        }

    def get_sector_stats(self, days: int = 30) -> Dict[str, Dict]:
        """Win rate by sector."""
        memories = self.get_recent(days)
        stats: Dict[str, Dict] = {}
        for mem in memories:
            sec = mem.get("sector", "UNKNOWN") or "UNKNOWN"
            if sec not in stats:
                stats[sec] = {"wins": 0, "total": 0, "pnl_sum": 0}
            stats[sec]["total"]   += 1
            stats[sec]["pnl_sum"] += mem.get("pnl", 0)
            if mem.get("won"):
                stats[sec]["wins"] += 1
        return {
            s: {"win_rate": round(d["wins"]/max(d["total"],1), 3),
                "avg_pnl":  round(d["pnl_sum"]/max(d["total"],1), 2),
                "count":    d["total"]}
            for s, d in stats.items() if d["total"] >= 3
        }

    def get_rsi_optimum(self, days: int = 30) -> Dict:
        """What RSI threshold gave the best results?"""
        memories = [m for m in self.get_recent(days) if m.get("rsi_at_entry", 0) > 0]
        if not memories:
            return {"best_threshold": 70, "analysis": "insufficient data"}

        # Bucket RSI readings
        buckets = {}
        for mem in memories:
            rsi = mem["rsi_at_entry"]
            bucket = int(rsi // 5) * 5    # 65, 70, 75, 80
            if bucket not in buckets:
                buckets[bucket] = {"wins": 0, "total": 0}
            buckets[bucket]["total"] += 1
            if mem.get("won"):
                buckets[bucket]["wins"] += 1

        best = max(
            buckets.items(),
            key=lambda x: x[1]["wins"] / max(x[1]["total"], 1),
            default=(70, {})
        )
        return {
            "best_threshold": best[0],
            "win_rate_at_best": round(best[1].get("wins", 0) / max(best[1].get("total", 1), 1), 3),
            "buckets": {str(k): v for k, v in sorted(buckets.items())},
        }

    def get_lessons(self, days: int = 14) -> List[str]:
        """Get agent's own lessons from recent trades."""
        with sqlite3.connect(self.db) as conn:
            from datetime import timedelta
            since = (date.today() - timedelta(days=days)).isoformat()
            rows = conn.execute(
                "SELECT lesson FROM trade_memories WHERE trade_date >= ? AND lesson != '' ORDER BY trade_date DESC LIMIT 50",
                (since,)
            ).fetchall()
        return [r[0] for r in rows if r[0]]

    # ──────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────

    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        d = dict(row)
        try:
            d["signals_fired"] = json.loads(d.get("signals_fired", "[]") or "[]")
        except Exception:
            d["signals_fired"] = []
        d["won"] = bool(d.get("won", 0))
        return d

    def _init_schema(self):
        with sqlite3.connect(self.db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_memories (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT, trade_date TEXT, entry_time TEXT, exit_time TEXT,
                    strategy TEXT, signals_fired TEXT, candlestick_pattern TEXT,
                    mtf_aligned INTEGER, mtf_alignment_count INTEGER,
                    nifty_change_pct REAL, nifty_breadth TEXT, market_regime TEXT,
                    fii_net_cr REAL, india_vix REAL,
                    sector TEXT, sector_trend TEXT, time_of_day TEXT,
                    rsi_at_entry REAL, macd_histogram REAL, bb_position REAL,
                    volume_ratio REAL, gap_pct REAL,
                    entry_price REAL, stop_loss REAL, target REAL,
                    quantity INTEGER, confidence_score REAL,
                    exit_price REAL, exit_reason TEXT, pnl REAL,
                    pnl_pct REAL, holding_minutes INTEGER, won INTEGER,
                    lesson TEXT DEFAULT '', what_worked TEXT DEFAULT '',
                    what_failed TEXT DEFAULT '',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tm_date ON trade_memories(trade_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tm_symbol ON trade_memories(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tm_regime ON trade_memories(market_regime)")


# Singleton
memory_store = TradeMemoryStore()
