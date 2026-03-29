"""
Weekly Performance Analytics
Aggregates daily trade data into weekly, monthly, and cumulative metrics.
Generates performance reports with charts (ASCII in terminal, HTML for web).
Identifies strategy drift and warns when performance degrades.
"""

import json
import logging
import os
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

from config import DB_PATH

logger = logging.getLogger(__name__)


@dataclass
class WeeklyStats:
    week_start: str
    week_end:   str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    total_pnl: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    max_single_loss: float
    best_trade_symbol: str
    worst_trade_symbol: str
    avg_holding_description: str  # e.g. "intraday"
    sharpe_week: float


@dataclass
class PerformanceReport:
    generated_at:    str
    period_start:    str
    period_end:      str
    total_trades:    int
    overall_pnl:     float
    overall_win_rate: float
    overall_profit_factor: float
    best_week_pnl:   float
    worst_week_pnl:  float
    max_drawdown_pct: float
    consecutive_win_streak:  int
    consecutive_loss_streak: int
    most_traded_symbol:  str
    most_profitable_symbol: str
    weekly_breakdown: List[WeeklyStats] = field(default_factory=list)
    monthly_pnl:     Dict[str, float] = field(default_factory=dict)
    symbol_stats:    Dict[str, Dict]  = field(default_factory=dict)
    alerts:          List[str]        = field(default_factory=list)


class PerformanceAnalytics:
    """
    Reads from the SQLite trade database and computes performance metrics
    across any time period.
    """

    def __init__(self, capital: float = 100_000):
        self.capital = capital

    # ──────────────────────────────────────────────────────────────
    # MAIN REPORT
    # ──────────────────────────────────────────────────────────────

    def generate_report(
        self,
        days_back: int = 30,
        save_path: Optional[str] = None,
    ) -> PerformanceReport:
        """Generate a full performance report for the last N days."""
        end_date   = date.today()
        start_date = end_date - timedelta(days=days_back)
        trades     = self._load_trades(start_date, end_date)

        if not trades:
            logger.info("No trades found in the specified period.")
            return PerformanceReport(
                generated_at=datetime.now().isoformat(),
                period_start=start_date.isoformat(),
                period_end=end_date.isoformat(),
                total_trades=0,
                overall_pnl=0, overall_win_rate=0, overall_profit_factor=0,
                best_week_pnl=0, worst_week_pnl=0, max_drawdown_pct=0,
                consecutive_win_streak=0, consecutive_loss_streak=0,
                most_traded_symbol="—", most_profitable_symbol="—",
            )

        report = self._compute_report(trades, start_date, end_date)
        report.alerts = self._generate_alerts(report)

        if save_path:
            self._save_report(report, save_path)

        return report

    def print_summary(self, report: PerformanceReport):
        """Print a clean terminal summary."""
        w = 55
        print("\n" + "═" * w)
        print(f"  PERFORMANCE REPORT  {report.period_start} → {report.period_end}")
        print("═" * w)
        pnl_sign = "+" if report.overall_pnl >= 0 else ""
        print(f"  Total P&L       : {pnl_sign}₹{report.overall_pnl:,.0f}")
        print(f"  Total trades    : {report.total_trades}")
        print(f"  Win rate        : {report.overall_win_rate:.1f}%")
        print(f"  Profit factor   : {report.overall_profit_factor:.2f}")
        print(f"  Max drawdown    : {report.max_drawdown_pct:.2f}%")
        print(f"  Win streak      : {report.consecutive_win_streak}")
        print(f"  Loss streak     : {report.consecutive_loss_streak}")
        print(f"  Best week       : +₹{report.best_week_pnl:,.0f}")
        print(f"  Worst week      : ₹{report.worst_week_pnl:,.0f}")
        print(f"  Top symbol      : {report.most_profitable_symbol}")
        print("─" * w)

        if report.weekly_breakdown:
            print("  Weekly breakdown:")
            for w_stats in report.weekly_breakdown:
                sign = "+" if w_stats.total_pnl >= 0 else ""
                wr   = f"{w_stats.win_rate_pct:.0f}%"
                print(f"    {w_stats.week_start} : {sign}₹{w_stats.total_pnl:,.0f} "
                      f"({w_stats.total_trades}T, {wr} WR)")

        if report.alerts:
            print("─" * w)
            print("  ⚠ Alerts:")
            for a in report.alerts:
                print(f"    • {a}")

        print("═" * w + "\n")

    def get_symbol_heatmap(self) -> Dict[str, Dict]:
        """Get per-symbol win rate and P&L for the last 90 days."""
        trades = self._load_trades(
            date.today() - timedelta(days=90), date.today()
        )
        return self._aggregate_by_symbol(trades)

    def get_monthly_pnl(self, months: int = 6) -> Dict[str, float]:
        """Get month-by-month P&L for the last N months."""
        start = date.today().replace(day=1) - timedelta(days=months * 31)
        trades = self._load_trades(start, date.today())
        monthly: Dict[str, float] = defaultdict(float)
        for t in trades:
            month_key = t["trade_date"][:7]   # "YYYY-MM"
            monthly[month_key] += t["pnl"] or 0
        return dict(sorted(monthly.items()))

    # ──────────────────────────────────────────────────────────────
    # COMPUTATION
    # ──────────────────────────────────────────────────────────────

    def _compute_report(
        self,
        trades: List[Dict],
        start_date: date,
        end_date: date,
    ) -> PerformanceReport:
        closed = [t for t in trades if t["status"] == "CLOSED"]

        # Overall metrics
        total_pnl    = sum(t["pnl"] or 0 for t in closed)
        wins         = [t for t in closed if (t["pnl"] or 0) > 0]
        losses       = [t for t in closed if (t["pnl"] or 0) <= 0]
        win_rate     = len(wins) / max(len(closed), 1) * 100
        gross_profit = sum(t["pnl"] for t in wins) if wins else 0
        gross_loss   = abs(sum(t["pnl"] for t in losses)) if losses else 1
        pf           = gross_profit / max(gross_loss, 1)

        # Drawdown
        equity = [self.capital]
        for t in closed:
            equity.append(equity[-1] + (t["pnl"] or 0))
        max_dd = self._max_drawdown(equity)

        # Streaks
        win_streak  = self._max_streak(closed, positive=True)
        loss_streak = self._max_streak(closed, positive=False)

        # By symbol
        by_symbol     = self._aggregate_by_symbol(closed)
        traded_counts = {s: d["count"] for s, d in by_symbol.items()}
        pnl_by_sym    = {s: d["total_pnl"] for s, d in by_symbol.items()}

        most_traded     = max(traded_counts, key=traded_counts.get) if traded_counts else "—"
        most_profitable = max(pnl_by_sym, key=pnl_by_sym.get) if pnl_by_sym else "—"

        # Weekly breakdown
        weekly = self._weekly_breakdown(closed, start_date, end_date)
        week_pnls = [w.total_pnl for w in weekly]

        # Monthly P&L
        monthly: Dict[str, float] = defaultdict(float)
        for t in closed:
            monthly[t["trade_date"][:7]] += t["pnl"] or 0

        return PerformanceReport(
            generated_at=datetime.now().isoformat(),
            period_start=start_date.isoformat(),
            period_end=end_date.isoformat(),
            total_trades=len(closed),
            overall_pnl=round(total_pnl, 2),
            overall_win_rate=round(win_rate, 1),
            overall_profit_factor=round(pf, 2),
            best_week_pnl=round(max(week_pnls, default=0), 2),
            worst_week_pnl=round(min(week_pnls, default=0), 2),
            max_drawdown_pct=round(max_dd, 2),
            consecutive_win_streak=win_streak,
            consecutive_loss_streak=loss_streak,
            most_traded_symbol=most_traded,
            most_profitable_symbol=most_profitable,
            weekly_breakdown=weekly,
            monthly_pnl=dict(monthly),
            symbol_stats=by_symbol,
        )

    def _weekly_breakdown(
        self,
        trades: List[Dict],
        start_date: date,
        end_date: date,
    ) -> List[WeeklyStats]:
        """Group trades by calendar week and compute per-week stats."""
        by_week: Dict[str, List[Dict]] = defaultdict(list)

        for t in trades:
            try:
                trade_date = date.fromisoformat(t["trade_date"])
                monday     = trade_date - timedelta(days=trade_date.weekday())
                by_week[monday.isoformat()].append(t)
            except Exception:
                continue

        result = []
        for week_start_str, week_trades in sorted(by_week.items()):
            week_start = date.fromisoformat(week_start_str)
            week_end   = week_start + timedelta(days=4)
            wins   = [t for t in week_trades if (t["pnl"] or 0) > 0]
            losses = [t for t in week_trades if (t["pnl"] or 0) <= 0]
            total_pnl  = sum(t["pnl"] or 0 for t in week_trades)
            gp         = sum(t["pnl"] for t in wins)   if wins   else 0
            gl         = abs(sum(t["pnl"] for t in losses)) if losses else 1
            wr         = len(wins) / max(len(week_trades), 1) * 100

            # Simple weekly Sharpe
            daily_pnls: Dict[str, float] = defaultdict(float)
            for t in week_trades:
                daily_pnls[t["trade_date"]] += t["pnl"] or 0
            daily_values = list(daily_pnls.values())
            import statistics
            if len(daily_values) > 1:
                mean_d = statistics.mean(daily_values)
                std_d  = statistics.stdev(daily_values)
                sharpe = (mean_d / std_d * (252 ** 0.5)) if std_d > 0 else 0
            else:
                sharpe = 0

            best_sym  = max(week_trades, key=lambda t: t["pnl"] or 0, default={}).get("symbol", "—")
            worst_sym = min(week_trades, key=lambda t: t["pnl"] or 0, default={}).get("symbol", "—")

            result.append(WeeklyStats(
                week_start=week_start.isoformat(),
                week_end=week_end.isoformat(),
                total_trades=len(week_trades),
                winning_trades=len(wins),
                losing_trades=len(losses),
                win_rate_pct=round(wr, 1),
                total_pnl=round(total_pnl, 2),
                gross_profit=round(gp, 2),
                gross_loss=round(gl, 2),
                profit_factor=round(gp / max(gl, 1), 2),
                max_single_loss=round(min(t["pnl"] or 0 for t in week_trades), 2),
                best_trade_symbol=best_sym,
                worst_trade_symbol=worst_sym,
                avg_holding_description="intraday",
                sharpe_week=round(sharpe, 2),
            ))

        return result

    def _aggregate_by_symbol(self, trades: List[Dict]) -> Dict[str, Dict]:
        """Compute per-symbol statistics."""
        by_sym: Dict[str, List[Dict]] = defaultdict(list)
        for t in trades:
            by_sym[t.get("symbol", "UNKNOWN")].append(t)

        result = {}
        for sym, sym_trades in by_sym.items():
            wins = [t for t in sym_trades if (t["pnl"] or 0) > 0]
            result[sym] = {
                "count":     len(sym_trades),
                "total_pnl": round(sum(t["pnl"] or 0 for t in sym_trades), 2),
                "win_rate":  round(len(wins) / len(sym_trades) * 100, 1),
                "avg_pnl":   round(sum(t["pnl"] or 0 for t in sym_trades) / len(sym_trades), 2),
                "best":      round(max(t["pnl"] or 0 for t in sym_trades), 2),
                "worst":     round(min(t["pnl"] or 0 for t in sym_trades), 2),
            }
        return result

    # ──────────────────────────────────────────────────────────────
    # ALERTS
    # ──────────────────────────────────────────────────────────────

    def _generate_alerts(self, report: PerformanceReport) -> List[str]:
        """Flag strategy degradation or unusual conditions."""
        alerts = []

        if report.overall_win_rate < 40 and report.total_trades >= 10:
            alerts.append(
                f"Win rate {report.overall_win_rate:.0f}% is below 40% — "
                "consider pausing and backtesting with current market conditions"
            )

        if report.overall_profit_factor < 1.0 and report.total_trades >= 5:
            alerts.append(
                f"Profit factor {report.overall_profit_factor:.2f} < 1.0 — "
                "strategy is losing money overall this period"
            )

        if report.max_drawdown_pct > 8:
            alerts.append(
                f"Max drawdown {report.max_drawdown_pct:.1f}% is high — "
                "reduce position size until strategy recovers"
            )

        if report.consecutive_loss_streak >= 5:
            alerts.append(
                f"Loss streak of {report.consecutive_loss_streak} trades — "
                "activate circuit breaker and review strategy"
            )

        if report.overall_pnl < 0 and report.total_trades >= 15:
            alerts.append(
                f"Overall P&L is negative (₹{report.overall_pnl:,.0f}) over "
                f"{report.total_trades} trades — consider paper trading until conditions improve"
            )

        return alerts

    # ──────────────────────────────────────────────────────────────
    # DB & HELPERS
    # ──────────────────────────────────────────────────────────────

    def _load_trades(self, start_date: date, end_date: date) -> List[Dict]:
        """Load trades from SQLite DB for a date range."""
        if not os.path.exists(DB_PATH):
            return []
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("""
                    SELECT symbol, direction, entry_price, exit_price,
                           quantity, pnl, status, trade_date
                    FROM trades
                    WHERE trade_date >= ? AND trade_date <= ?
                    ORDER BY trade_date ASC
                """, (start_date.isoformat(), end_date.isoformat())).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Trade load error: {e}")
            return []

    def _max_drawdown(self, equity: List[float]) -> float:
        """Calculate maximum drawdown percentage from equity curve."""
        if not equity:
            return 0.0
        peak = equity[0]
        max_dd = 0.0
        for val in equity:
            if val > peak:
                peak = val
            dd = (peak - val) / max(peak, 1) * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _max_streak(self, trades: List[Dict], positive: bool) -> int:
        """Calculate maximum consecutive win or loss streak."""
        max_s = 0
        curr  = 0
        for t in trades:
            pnl = t["pnl"] or 0
            if (positive and pnl > 0) or (not positive and pnl <= 0):
                curr += 1
                max_s = max(max_s, curr)
            else:
                curr = 0
        return max_s

    def _save_report(self, report: PerformanceReport, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)
        logger.info(f"Performance report saved: {path}")    def record_trade(self, symbol: str, direction: str,
                     entry_price: float, exit_price: float,
                     quantity: int, pnl: float, trade_date=None):
        """Record a completed trade for analytics."""
        from datetime import date
        import sqlite3 as _sq
        import config as _cfg
        td = str(trade_date or date.today())
        with _sq.connect(_cfg.DB_PATH) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO trades
                (symbol, direction, entry_price, exit_price, quantity, pnl, trade_date, status)
                VALUES (?,?,?,?,?,?,?,'CLOSED')
            """, (symbol, direction, entry_price, exit_price, quantity, pnl, td))
            conn.commit()


    def __init__(self, capital: float = 100_000):
        self.capital = capital

    # ──────────────────────────────────────────────────────────────
    # MAIN REPORT
    # ──────────────────────────────────────────────────────────────

    def generate_report(
        self,
        days_back: int = 30,
        save_path: Optional[str] = None,
    ) -> PerformanceReport:
        """Generate a full performance report for the last N days."""
        end_date   = date.today()
        start_date = end_date - timedelta(days=days_back)
        trades     = self._load_trades(start_date, end_date)

        if not trades:
            logger.info("No trades found in the specified period.")
            return PerformanceReport(
                generated_at=datetime.now().isoformat(),
                period_start=start_date.isoformat(),
                period_end=end_date.isoformat(),
                total_trades=0,
                overall_pnl=0, overall_win_rate=0, overall_profit_factor=0,
                best_week_pnl=0, worst_week_pnl=0, max_drawdown_pct=0,
                consecutive_win_streak=0, consecutive_loss_streak=0,
                most_traded_symbol="—", most_profitable_symbol="—",
            )

        report = self._compute_report(trades, start_date, end_date)
        report.alerts = self._generate_alerts(report)

        if save_path:
            self._save_report(report, save_path)

        return report

    def print_summary(self, report: PerformanceReport):
        """Print a clean terminal summary."""
        w = 55
        print("\n" + "═" * w)
        print(f"  PERFORMANCE REPORT  {report.period_start} → {report.period_end}")
        print("═" * w)
        pnl_sign = "+" if report.overall_pnl >= 0 else ""
        print(f"  Total P&L       : {pnl_sign}₹{report.overall_pnl:,.0f}")
        print(f"  Total trades    : {report.total_trades}")
        print(f"  Win rate        : {report.overall_win_rate:.1f}%")
        print(f"  Profit factor   : {report.overall_profit_factor:.2f}")
        print(f"  Max drawdown    : {report.max_drawdown_pct:.2f}%")
        print(f"  Win streak      : {report.consecutive_win_streak}")
        print(f"  Loss streak     : {report.consecutive_loss_streak}")
        print(f"  Best week       : +₹{report.best_week_pnl:,.0f}")
        print(f"  Worst week      : ₹{report.worst_week_pnl:,.0f}")
        print(f"  Top symbol      : {report.most_profitable_symbol}")
        print("─" * w)

        if report.weekly_breakdown:
            print("  Weekly breakdown:")
            for w_stats in report.weekly_breakdown:
                sign = "+" if w_stats.total_pnl >= 0 else ""
                wr   = f"{w_stats.win_rate_pct:.0f}%"
                print(f"    {w_stats.week_start} : {sign}₹{w_stats.total_pnl:,.0f} "
                      f"({w_stats.total_trades}T, {wr} WR)")

        if report.alerts:
            print("─" * w)
            print("  ⚠ Alerts:")
            for a in report.alerts:
                print(f"    • {a}")

        print("═" * w + "\n")

    def get_symbol_heatmap(self) -> Dict[str, Dict]:
        """Get per-symbol win rate and P&L for the last 90 days."""
        trades = self._load_trades(
            date.today() - timedelta(days=90), date.today()
        )
        return self._aggregate_by_symbol(trades)

    def get_monthly_pnl(self, months: int = 6) -> Dict[str, float]:
        """Get month-by-month P&L for the last N months."""
        start = date.today().replace(day=1) - timedelta(days=months * 31)
        trades = self._load_trades(start, date.today())
        monthly: Dict[str, float] = defaultdict(float)
        for t in trades:
            month_key = t["trade_date"][:7]   # "YYYY-MM"
            monthly[month_key] += t["pnl"] or 0
        return dict(sorted(monthly.items()))

    # ──────────────────────────────────────────────────────────────
    # COMPUTATION
    # ──────────────────────────────────────────────────────────────

    def _compute_report(
        self,
        trades: List[Dict],
        start_date: date,
        end_date: date,
    ) -> PerformanceReport:
        closed = [t for t in trades if t["status"] == "CLOSED"]

        # Overall metrics
        total_pnl    = sum(t["pnl"] or 0 for t in closed)
        wins         = [t for t in closed if (t["pnl"] or 0) > 0]
        losses       = [t for t in closed if (t["pnl"] or 0) <= 0]
        win_rate     = len(wins) / max(len(closed), 1) * 100
        gross_profit = sum(t["pnl"] for t in wins) if wins else 0
        gross_loss   = abs(sum(t["pnl"] for t in losses)) if losses else 1
        pf           = gross_profit / max(gross_loss, 1)

        # Drawdown
        equity = [self.capital]
        for t in closed:
            equity.append(equity[-1] + (t["pnl"] or 0))
        max_dd = self._max_drawdown(equity)

        # Streaks
        win_streak  = self._max_streak(closed, positive=True)
        loss_streak = self._max_streak(closed, positive=False)

        # By symbol
        by_symbol     = self._aggregate_by_symbol(closed)
        traded_counts = {s: d["count"] for s, d in by_symbol.items()}
        pnl_by_sym    = {s: d["total_pnl"] for s, d in by_symbol.items()}

        most_traded     = max(traded_counts, key=traded_counts.get) if traded_counts else "—"
        most_profitable = max(pnl_by_sym, key=pnl_by_sym.get) if pnl_by_sym else "—"

        # Weekly breakdown
        weekly = self._weekly_breakdown(closed, start_date, end_date)
        week_pnls = [w.total_pnl for w in weekly]

        # Monthly P&L
        monthly: Dict[str, float] = defaultdict(float)
        for t in closed:
            monthly[t["trade_date"][:7]] += t["pnl"] or 0

        return PerformanceReport(
            generated_at=datetime.now().isoformat(),
            period_start=start_date.isoformat(),
            period_end=end_date.isoformat(),
            total_trades=len(closed),
            overall_pnl=round(total_pnl, 2),
            overall_win_rate=round(win_rate, 1),
            overall_profit_factor=round(pf, 2),
            best_week_pnl=round(max(week_pnls, default=0), 2),
            worst_week_pnl=round(min(week_pnls, default=0), 2),
            max_drawdown_pct=round(max_dd, 2),
            consecutive_win_streak=win_streak,
            consecutive_loss_streak=loss_streak,
            most_traded_symbol=most_traded,
            most_profitable_symbol=most_profitable,
            weekly_breakdown=weekly,
            monthly_pnl=dict(monthly),
            symbol_stats=by_symbol,
        )

    def _weekly_breakdown(
        self,
        trades: List[Dict],
        start_date: date,
        end_date: date,
    ) -> List[WeeklyStats]:
        """Group trades by calendar week and compute per-week stats."""
        by_week: Dict[str, List[Dict]] = defaultdict(list)

        for t in trades:
            try:
                trade_date = date.fromisoformat(t["trade_date"])
                monday     = trade_date - timedelta(days=trade_date.weekday())
                by_week[monday.isoformat()].append(t)
            except Exception:
                continue

        result = []
        for week_start_str, week_trades in sorted(by_week.items()):
            week_start = date.fromisoformat(week_start_str)
            week_end   = week_start + timedelta(days=4)
            wins   = [t for t in week_trades if (t["pnl"] or 0) > 0]
            losses = [t for t in week_trades if (t["pnl"] or 0) <= 0]
            total_pnl  = sum(t["pnl"] or 0 for t in week_trades)
            gp         = sum(t["pnl"] for t in wins)   if wins   else 0
            gl         = abs(sum(t["pnl"] for t in losses)) if losses else 1
            wr         = len(wins) / max(len(week_trades), 1) * 100

            # Simple weekly Sharpe
            daily_pnls: Dict[str, float] = defaultdict(float)
            for t in week_trades:
                daily_pnls[t["trade_date"]] += t["pnl"] or 0
            daily_values = list(daily_pnls.values())
            import statistics
            if len(daily_values) > 1:
                mean_d = statistics.mean(daily_values)
                std_d  = statistics.stdev(daily_values)
                sharpe = (mean_d / std_d * (252 ** 0.5)) if std_d > 0 else 0
            else:
                sharpe = 0

            best_sym  = max(week_trades, key=lambda t: t["pnl"] or 0, default={}).get("symbol", "—")
            worst_sym = min(week_trades, key=lambda t: t["pnl"] or 0, default={}).get("symbol", "—")

            result.append(WeeklyStats(
                week_start=week_start.isoformat(),
                week_end=week_end.isoformat(),
                total_trades=len(week_trades),
                winning_trades=len(wins),
                losing_trades=len(losses),
                win_rate_pct=round(wr, 1),
                total_pnl=round(total_pnl, 2),
                gross_profit=round(gp, 2),
                gross_loss=round(gl, 2),
                profit_factor=round(gp / max(gl, 1), 2),
                max_single_loss=round(min(t["pnl"] or 0 for t in week_trades), 2),
                best_trade_symbol=best_sym,
                worst_trade_symbol=worst_sym,
                avg_holding_description="intraday",
                sharpe_week=round(sharpe, 2),
            ))

        return result

    def _aggregate_by_symbol(self, trades: List[Dict]) -> Dict[str, Dict]:
        """Compute per-symbol statistics."""
        by_sym: Dict[str, List[Dict]] = defaultdict(list)
        for t in trades:
            by_sym[t.get("symbol", "UNKNOWN")].append(t)

        result = {}
        for sym, sym_trades in by_sym.items():
            wins = [t for t in sym_trades if (t["pnl"] or 0) > 0]
            result[sym] = {
                "count":     len(sym_trades),
                "total_pnl": round(sum(t["pnl"] or 0 for t in sym_trades), 2),
                "win_rate":  round(len(wins) / len(sym_trades) * 100, 1),
                "avg_pnl":   round(sum(t["pnl"] or 0 for t in sym_trades) / len(sym_trades), 2),
                "best":      round(max(t["pnl"] or 0 for t in sym_trades), 2),
                "worst":     round(min(t["pnl"] or 0 for t in sym_trades), 2),
            }
        return result

    # ──────────────────────────────────────────────────────────────
    # ALERTS
    # ──────────────────────────────────────────────────────────────

    def _generate_alerts(self, report: PerformanceReport) -> List[str]:
        """Flag strategy degradation or unusual conditions."""
        alerts = []

        if report.overall_win_rate < 40 and report.total_trades >= 10:
            alerts.append(
                f"Win rate {report.overall_win_rate:.0f}% is below 40% — "
                "consider pausing and backtesting with current market conditions"
            )

        if report.overall_profit_factor < 1.0 and report.total_trades >= 5:
            alerts.append(
                f"Profit factor {report.overall_profit_factor:.2f} < 1.0 — "
                "strategy is losing money overall this period"
            )

        if report.max_drawdown_pct > 8:
            alerts.append(
                f"Max drawdown {report.max_drawdown_pct:.1f}% is high — "
                "reduce position size until strategy recovers"
            )

        if report.consecutive_loss_streak >= 5:
            alerts.append(
                f"Loss streak of {report.consecutive_loss_streak} trades — "
                "activate circuit breaker and review strategy"
            )

        if report.overall_pnl < 0 and report.total_trades >= 15:
            alerts.append(
                f"Overall P&L is negative (₹{report.overall_pnl:,.0f}) over "
                f"{report.total_trades} trades — consider paper trading until conditions improve"
            )

        return alerts

    # ──────────────────────────────────────────────────────────────
    # DB & HELPERS
    # ──────────────────────────────────────────────────────────────

    def _load_trades(self, start_date: date, end_date: date) -> List[Dict]:
        """Load trades from SQLite DB for a date range."""
        if not os.path.exists(DB_PATH):
            return []
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("""
                    SELECT symbol, direction, entry_price, exit_price,
                           quantity, pnl, status, trade_date
                    FROM trades
                    WHERE trade_date >= ? AND trade_date <= ?
                    ORDER BY trade_date ASC
                """, (start_date.isoformat(), end_date.isoformat())).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Trade load error: {e}")
            return []

    def _max_drawdown(self, equity: List[float]) -> float:
        """Calculate maximum drawdown percentage from equity curve."""
        if not equity:
            return 0.0
        peak = equity[0]
        max_dd = 0.0
        for val in equity:
            if val > peak:
                peak = val
            dd = (peak - val) / max(peak, 1) * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _max_streak(self, trades: List[Dict], positive: bool) -> int:
        """Calculate maximum consecutive win or loss streak."""
        max_s = 0
        curr  = 0
        for t in trades:
            pnl = t["pnl"] or 0
            if (positive and pnl > 0) or (not positive and pnl <= 0):
                curr += 1
                max_s = max(max_s, curr)
            else:
                curr = 0
        return max_s

    def _save_report(self, report: PerformanceReport, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)
        logger.info(f"Performance report saved: {path}")
