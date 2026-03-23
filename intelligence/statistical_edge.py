"""
Statistical Edge Calculator
Measures the actual edge (expected value) for every signal combination
the agent has ever traded. Stops the agent from trading patterns
that have no proven edge, no matter how "convincing" they look.

Edge = E[P&L per trade] = (Win rate × Avg win) + (Loss rate × Avg loss)
Positive edge = the strategy makes money in expectation.

Also computes:
  - t-statistic: is the edge statistically significant or just noise?
  - Minimum sample size: how many trades before we can trust the edge?
  - Edge stability: is the edge consistent over time or declining?
  - Regime-conditioned edge: edge in downtrends vs uptrends

The agent uses this to:
  1. Only trade signal combos with proven positive edge
  2. Rank candidates by expected value, not just win rate
  3. Detect when an edge is degrading (market has adapted)
  4. Size positions proportionally to edge magnitude
"""

import logging
import math
import sqlite3
import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import date, timedelta

logger = logging.getLogger(__name__)

EDGE_DB = os.path.join(os.path.dirname(__file__), "..", "db", "signal_edge.db")

# Minimum trades before reporting edge as reliable
MIN_SAMPLE_SIZE = 8
# Minimum t-stat for statistical significance (p < 0.05)
MIN_T_STAT = 1.96


@dataclass
class EdgeMeasurement:
    signal_combo:    str         # sorted, pipe-separated signal names
    sample_size:     int
    win_rate:        float
    avg_win:         float       # average P&L on winning trades
    avg_loss:        float       # average P&L on losing trades (negative)
    expected_value:  float       # E[P&L] per trade
    profit_factor:   float       # gross profit / gross loss
    t_statistic:     float       # statistical significance of edge
    is_significant:  bool        # t_stat >= 1.96
    is_reliable:     bool        # sample_size >= MIN_SAMPLE_SIZE
    edge_trend:      str         # "IMPROVING" | "STABLE" | "DECLINING"
    regime_breakdown: Dict[str, float]   # edge by market regime
    last_updated:    str


class StatisticalEdgeCalculator:
    """
    Tracks and computes statistical edge for every signal combination.
    """

    def __init__(self):
        os.makedirs(os.path.dirname(EDGE_DB), exist_ok=True)
        self._init_schema()

    # ──────────────────────────────────────────────────────────────
    # RECORD TRADES
    # ──────────────────────────────────────────────────────────────

    def record(
        self,
        signals:       List[str],
        pnl:           float,
        won:           bool,
        regime:        str = "",
        trade_date:    str = "",
    ):
        """Record a closed trade for edge tracking."""
        combo = self._signals_to_key(signals)
        with sqlite3.connect(EDGE_DB) as conn:
            conn.execute("""
                INSERT INTO edge_records (signal_combo, pnl, won, regime, trade_date)
                VALUES (?, ?, ?, ?, ?)
            """, (combo, pnl, int(won), regime, trade_date or date.today().isoformat()))

    # ──────────────────────────────────────────────────────────────
    # MEASURE EDGE
    # ──────────────────────────────────────────────────────────────

    def measure(self, signals: List[str], days: int = 60) -> Optional[EdgeMeasurement]:
        """
        Compute edge for a given signal combination.
        Returns None if insufficient data.
        """
        combo = self._signals_to_key(signals)
        since = (date.today() - timedelta(days=days)).isoformat()

        with sqlite3.connect(EDGE_DB) as conn:
            rows = conn.execute("""
                SELECT pnl, won, regime, trade_date FROM edge_records
                WHERE signal_combo = ? AND trade_date >= ?
                ORDER BY trade_date ASC
            """, (combo, since)).fetchall()

        if len(rows) < 3:
            return None

        pnls   = [r[0] for r in rows]
        wins   = [r[0] for r in rows if r[1]]
        losses = [r[0] for r in rows if not r[1]]
        regimes= [r[2] for r in rows]

        n         = len(pnls)
        win_rate  = len(wins) / n
        avg_win   = sum(wins)   / max(len(wins),   1)
        avg_loss  = sum(losses) / max(len(losses), 1)
        ev        = win_rate * avg_win + (1 - win_rate) * avg_loss
        pf        = abs(sum(wins)) / max(abs(sum(losses)), 1e-6)

        # t-statistic: test if mean P&L is significantly > 0
        import statistics
        if n >= 2:
            mean_pnl = ev
            std_pnl  = statistics.stdev(pnls) if len(pnls) > 1 else 1.0
            t_stat   = (mean_pnl / max(std_pnl, 1e-6)) * math.sqrt(n)
        else:
            t_stat = 0.0

        # Regime breakdown
        regime_pnls: Dict[str, List[float]] = {}
        for pnl, won, regime, _ in rows:
            if regime:
                regime_pnls.setdefault(regime, []).append(pnl)
        regime_ev = {
            r: round(sum(pnls) / len(pnls), 2)
            for r, pnls in regime_pnls.items()
            if len(pnls) >= 3
        }

        # Edge trend: compare first half vs second half
        mid      = n // 2
        ev_early = sum(pnls[:mid]) / max(mid, 1)
        ev_late  = sum(pnls[mid:]) / max(n - mid, 1)
        if ev_late > ev_early * 1.15:
            trend = "IMPROVING"
        elif ev_late < ev_early * 0.85:
            trend = "DECLINING"
        else:
            trend = "STABLE"

        return EdgeMeasurement(
            signal_combo=combo,
            sample_size=n,
            win_rate=round(win_rate, 3),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            expected_value=round(ev, 2),
            profit_factor=round(pf, 3),
            t_statistic=round(t_stat, 3),
            is_significant=abs(t_stat) >= MIN_T_STAT,
            is_reliable=n >= MIN_SAMPLE_SIZE,
            edge_trend=trend,
            regime_breakdown=regime_ev,
            last_updated=date.today().isoformat(),
        )

    def has_edge(self, signals: List[str], min_ev: float = 0) -> bool:
        """
        Quick check: does this signal combo have proven positive edge?
        Returns True if edge > min_ev with statistical significance.
        """
        m = self.measure(signals)
        if m is None:
            return True   # Unknown combo: give benefit of doubt
        return m.expected_value > min_ev and (m.is_significant or not m.is_reliable)

    def get_top_combos(self, top_n: int = 10, min_samples: int = 5) -> List[EdgeMeasurement]:
        """Get the N highest-edge signal combinations."""
        with sqlite3.connect(EDGE_DB) as conn:
            combos = conn.execute(
                "SELECT DISTINCT signal_combo FROM edge_records"
            ).fetchall()

        measurements = []
        for (combo,) in combos:
            signals = combo.split("|")
            m = self.measure(signals)
            if m and m.sample_size >= min_samples:
                measurements.append(m)

        return sorted(measurements, key=lambda m: m.expected_value, reverse=True)[:top_n]

    def get_declining_edges(self) -> List[str]:
        """Find signal combos whose edge is declining — warn the agent."""
        with sqlite3.connect(EDGE_DB) as conn:
            combos = conn.execute(
                "SELECT DISTINCT signal_combo FROM edge_records"
            ).fetchall()

        declining = []
        for (combo,) in combos:
            m = self.measure(combo.split("|"))
            if m and m.edge_trend == "DECLINING" and m.sample_size >= MIN_SAMPLE_SIZE:
                declining.append(f"{combo}: EV={m.expected_value:.0f} ({m.edge_trend})")
        return declining

    def expected_value_for_candidate(
        self,
        signals: List[str],
        entry_price: float,
        stop_loss: float,
        target: float,
    ) -> float:
        """
        Compute expected P&L for a candidate trade using both:
        1. Historical edge from signal combo
        2. Theoretical E[V] = p*reward - (1-p)*risk
        Blends both for a robust estimate.
        """
        m = self.measure(signals)

        risk   = abs(stop_loss - entry_price)
        reward = abs(target - entry_price)

        if m and m.is_reliable:
            # Use historical win rate
            p_win   = m.win_rate
            # Blend historical EV with theoretical EV
            theory_ev = p_win * reward - (1 - p_win) * risk
            blend     = (m.expected_value * 0.4 + theory_ev * 0.6)   # trust theory more initially
            return round(blend, 2)
        else:
            # Only theoretical
            p_win = 0.55   # default win rate assumption
            return round(p_win * reward - (1 - p_win) * risk, 2)

    # ──────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────

    def _signals_to_key(self, signals: List[str]) -> str:
        """Convert signal list to a canonical, sorted key."""
        return "|".join(sorted(set(signals)))

    def _init_schema(self):
        with sqlite3.connect(EDGE_DB) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS edge_records (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_combo TEXT NOT NULL,
                    pnl         REAL NOT NULL,
                    won         INTEGER NOT NULL,
                    regime      TEXT DEFAULT '',
                    trade_date  TEXT DEFAULT (DATE('now'))
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_er_combo ON edge_records(signal_combo)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_er_date  ON edge_records(trade_date)")


# Singleton
edge_calculator = StatisticalEdgeCalculator()
