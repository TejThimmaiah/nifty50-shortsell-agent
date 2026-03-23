"""
Risk Manager Agent
The last gate before any trade is placed.
Enforces position sizing, daily loss limits, and drawdown protection.
No trade bypasses this agent.
"""

import logging
import sqlite3
from datetime import datetime, date
from typing import Optional, Tuple
from dataclasses import dataclass
from config import TRADING, DB_PATH

logger = logging.getLogger(__name__)


@dataclass
class RiskDecision:
    approved: bool
    reason: str
    adjusted_quantity: int      # may be reduced from requested
    adjusted_stop_loss: float
    max_loss_this_trade: float
    daily_loss_remaining: float


class RiskManagerAgent:
    """
    Stateful risk manager. Tracks daily P&L and enforces all risk rules.
    """

    def __init__(self, capital: float = None):
        self.capital = capital or TRADING.total_capital
        self._init_db()

    # ──────────────────────────────────────────────────────────────
    # MAIN APPROVAL GATE
    # ──────────────────────────────────────────────────────────────

    def approve_trade(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        quantity: int,
        direction: str = "SHORT",
    ) -> RiskDecision:
        """
        Main risk check. Returns RiskDecision with approved/rejected + adjustments.
        Called before EVERY trade.
        """
        # Rule 1: Check daily loss limit
        daily_pnl = self.get_today_pnl()
        max_daily_loss = self.capital * TRADING.max_daily_loss_pct / 100
        if daily_pnl <= -max_daily_loss:
            return RiskDecision(
                approved=False,
                reason=f"Daily loss limit hit: ₹{daily_pnl:.0f} (limit ₹{max_daily_loss:.0f})",
                adjusted_quantity=0,
                adjusted_stop_loss=stop_loss,
                max_loss_this_trade=0,
                daily_loss_remaining=0,
            )

        # Rule 2: Check max open positions
        open_count = self.get_open_position_count()
        if open_count >= TRADING.max_open_positions:
            return RiskDecision(
                approved=False,
                reason=f"Max open positions reached: {open_count}/{TRADING.max_open_positions}",
                adjusted_quantity=0,
                adjusted_stop_loss=stop_loss,
                max_loss_this_trade=0,
                daily_loss_remaining=max_daily_loss + daily_pnl,
            )

        # Rule 3: Position sizing — risk per trade
        risk_per_share = abs(stop_loss - entry_price)
        if risk_per_share <= 0:
            return RiskDecision(
                approved=False,
                reason="Invalid stop loss — risk per share is zero",
                adjusted_quantity=0,
                adjusted_stop_loss=stop_loss,
                max_loss_this_trade=0,
                daily_loss_remaining=max_daily_loss + daily_pnl,
            )

        max_risk_per_trade = self.capital * TRADING.max_risk_per_trade_pct / 100
        max_quantity = int(max_risk_per_trade / risk_per_share)
        adjusted_qty = min(quantity, max_quantity)

        if adjusted_qty < 1:
            return RiskDecision(
                approved=False,
                reason=f"Minimum quantity not achievable within risk limits (risk/share=₹{risk_per_share:.2f})",
                adjusted_quantity=0,
                adjusted_stop_loss=stop_loss,
                max_loss_this_trade=0,
                daily_loss_remaining=max_daily_loss + daily_pnl,
            )

        # Rule 4: Don't exceed remaining daily loss budget
        potential_loss = adjusted_qty * risk_per_share
        remaining_budget = max_daily_loss + daily_pnl   # daily_pnl is negative if losing
        if potential_loss > remaining_budget:
            # Reduce quantity to fit remaining budget
            adjusted_qty = max(1, int(remaining_budget / risk_per_share))

        # Rule 5: Stop loss sanity check for shorts
        if direction == "SHORT" and stop_loss <= entry_price:
            # SL must be ABOVE entry for shorts
            corrected_sl = round(entry_price * (1 + TRADING.stop_loss_pct / 100), 2)
            logger.warning(f"SL correction for {symbol}: {stop_loss} → {corrected_sl}")
            stop_loss = corrected_sl

        # Rule 6: Minimum price check
        if entry_price < TRADING.min_price:
            return RiskDecision(
                approved=False,
                reason=f"Price ₹{entry_price} below minimum ₹{TRADING.min_price}",
                adjusted_quantity=0,
                adjusted_stop_loss=stop_loss,
                max_loss_this_trade=0,
                daily_loss_remaining=remaining_budget,
            )

        max_loss = round(adjusted_qty * abs(stop_loss - entry_price), 2)

        logger.info(
            f"Risk approved: {symbol} | qty={adjusted_qty} | "
            f"risk=₹{max_loss:.0f} | daily_remaining=₹{remaining_budget:.0f}"
        )

        return RiskDecision(
            approved=True,
            reason="All risk checks passed",
            adjusted_quantity=adjusted_qty,
            adjusted_stop_loss=stop_loss,
            max_loss_this_trade=max_loss,
            daily_loss_remaining=remaining_budget,
        )

    # ──────────────────────────────────────────────────────────────
    # P&L TRACKING
    # ──────────────────────────────────────────────────────────────

    def record_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        quantity: int,
        order_id: str = "",
    ):
        """Record a new open trade."""
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO trades (symbol, direction, entry_price, quantity, order_id, status, trade_date)
                VALUES (?, ?, ?, ?, ?, 'OPEN', ?)
            """, (symbol, direction, entry_price, quantity, order_id, date.today().isoformat()))
        logger.info(f"Trade recorded: {direction} {quantity} {symbol} @ ₹{entry_price}")

    def close_trade(
        self,
        symbol: str,
        exit_price: float,
        order_id: str = "",
    ) -> float:
        """Mark trade as closed and compute P&L."""
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("""
                SELECT id, direction, entry_price, quantity
                FROM trades WHERE symbol=? AND status='OPEN' ORDER BY id DESC LIMIT 1
            """, (symbol,)).fetchone()

            if not row:
                logger.warning(f"No open trade found for {symbol}")
                return 0.0

            trade_id, direction, entry_price, quantity = row
            if direction == "SHORT":
                pnl = (entry_price - exit_price) * quantity
            else:
                pnl = (exit_price - entry_price) * quantity

            conn.execute("""
                UPDATE trades SET exit_price=?, pnl=?, status='CLOSED', exit_order_id=?,
                closed_at=? WHERE id=?
            """, (exit_price, pnl, order_id, datetime.now().isoformat(), trade_id))

        logger.info(f"Trade closed: {symbol} @ ₹{exit_price} | P&L = ₹{pnl:.2f}")
        return pnl

    def get_today_pnl(self) -> float:
        """Get total P&L for today (closed trades only)."""
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("""
                SELECT COALESCE(SUM(pnl), 0) FROM trades
                WHERE trade_date=? AND status='CLOSED'
            """, (date.today().isoformat(),)).fetchone()
        return float(row[0]) if row else 0.0

    def get_open_position_count(self) -> int:
        """Count currently open positions."""
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("""
                SELECT COUNT(*) FROM trades
                WHERE trade_date=? AND status='OPEN'
            """, (date.today().isoformat(),)).fetchone()
        return int(row[0]) if row else 0

    def get_daily_summary(self) -> dict:
        """Get end-of-day P&L summary."""
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("""
                SELECT symbol, direction, entry_price, exit_price, quantity, pnl, status
                FROM trades WHERE trade_date=?
            """, (date.today().isoformat(),)).fetchall()

        trades = [
            {"symbol": r[0], "direction": r[1], "entry": r[2], "exit": r[3],
             "qty": r[4], "pnl": r[5] or 0, "status": r[6]}
            for r in rows
        ]
        total_pnl = sum(t["pnl"] for t in trades)
        winners   = [t for t in trades if t["pnl"] > 0]
        losers    = [t for t in trades if t["pnl"] < 0]

        return {
            "date": date.today().isoformat(),
            "trades": trades,
            "total_pnl": round(total_pnl, 2),
            "win_count": len(winners),
            "loss_count": len(losers),
            "win_rate": round(len(winners) / max(len(trades), 1) * 100, 1),
            "best_trade": max(trades, key=lambda t: t["pnl"])["symbol"] if trades else None,
            "worst_trade": min(trades, key=lambda t: t["pnl"])["symbol"] if trades else None,
        }

    # ──────────────────────────────────────────────────────────────
    # DB INIT
    # ──────────────────────────────────────────────────────────────

    def _init_db(self):
        """Create trade database if it doesn't exist."""
        import os
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity INTEGER NOT NULL,
                    pnl REAL,
                    order_id TEXT,
                    exit_order_id TEXT,
                    status TEXT DEFAULT 'OPEN',
                    trade_date TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    closed_at TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_date ON trades(trade_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON trades(status)")
        logger.debug("Trade database initialized")
