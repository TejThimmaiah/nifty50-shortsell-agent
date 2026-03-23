"""
Position Monitor Agent
Runs on a tight 5-minute loop during market hours.
Checks live prices against stop losses and targets.
Updates trailing SL dynamically.
Sends alerts on every significant event.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Callable, Optional
from zoneinfo import ZoneInfo

from data.nse_fetcher import get_quote
from data.free_market_data import get_live_quote, free_streamer
from agents.risk_manager import RiskManagerAgent
from agents.trade_executor import TradeExecutorAgent
from config import TRADING, PAPER_TRADE

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")


class PositionMonitorAgent:
    """
    Continuously monitors open positions for:
    - Stop loss breaches
    - Target hits
    - Trailing SL updates
    - Forced square-off at 3:10 PM
    """

    def __init__(
        self,
        risk_mgr: RiskManagerAgent,
        executor: TradeExecutorAgent,
        notify_fn: Callable[[str], None] = None,
    ):
        self.risk_mgr  = risk_mgr
        self.executor  = executor
        self.notify    = notify_fn or (lambda msg: logger.info(f"NOTIFY: {msg}"))
        self.positions: Dict[str, Dict] = {}    # symbol → position metadata
        self.alerts_sent: Dict[str, str] = {}   # symbol → last alert type

    def register_position(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        target: float,
        quantity: int,
        direction: str = "SHORT",
    ):
        """Register a new position to monitor."""
        self.positions[symbol] = {
            "symbol":      symbol,
            "direction":   direction,
            "entry_price": entry_price,
            "stop_loss":   stop_loss,
            "target":      target,
            "quantity":    quantity,
            "highest_profit_pct": 0.0,
            "registered_at": datetime.now(IST).isoformat(),
            "trailing_active": False,
        }
        logger.info(
            f"Monitoring {direction} {symbol} | Entry=₹{entry_price} "
            f"| SL=₹{stop_loss} | Target=₹{target}"
        )

    def unregister_position(self, symbol: str):
        """Remove a position from monitoring (after close)."""
        self.positions.pop(symbol, None)
        self.alerts_sent.pop(symbol, None)

    def check_all(self) -> List[str]:
        """
        Check all registered positions against live prices.
        Returns list of symbols that were closed this cycle.
        """
        closed_symbols = []

        for symbol, pos in list(self.positions.items()):
            try:
                closed = self._check_position(symbol, pos)
                if closed:
                    closed_symbols.append(symbol)
            except Exception as e:
                logger.error(f"Monitor error [{symbol}]: {e}")

        return closed_symbols

    def force_square_off_all(self) -> Dict[str, float]:
        """
        3:10 PM: Force close all open positions regardless of P&L.
        Returns {symbol: pnl} for each closed position.
        """
        logger.info("=== FORCE SQUARE-OFF ALL POSITIONS ===")
        results = {}

        for symbol in list(self.positions.keys()):
            quote = get_quote(symbol)
            ltp   = float(quote.get("ltp", 0)) if quote else 0

            if PAPER_TRADE:
                pos = self.positions[symbol]
                qty = pos["quantity"]
                self.executor.cover_short(symbol, qty, ltp)
            else:
                self.executor.cancel_all_orders(symbol)
                self.executor.cover_short(symbol, self.positions[symbol]["quantity"])

            pnl = self.risk_mgr.close_trade(symbol, ltp)
            results[symbol] = pnl
            self.unregister_position(symbol)

            emoji = "✅" if pnl >= 0 else "🔴"
            self.notify(f"{emoji} EOD square-off: {symbol} @ ₹{ltp:.2f} | P&L ₹{pnl:.0f}")
            logger.info(f"Force closed {symbol} @ ₹{ltp:.2f} | P&L ₹{pnl:.2f}")

        return results

    # ──────────────────────────────────────────────────────────────
    # INTERNAL: single position check
    # ──────────────────────────────────────────────────────────────

    def _check_position(self, symbol: str, pos: Dict) -> bool:
        """
        Check one position. Returns True if the position was closed.
        """
        # Use free NSE polling — no Kite subscription needed
        quote = get_live_quote(symbol) or get_quote(symbol)
        if not quote:
            logger.warning(f"Could not fetch quote for {symbol}")
            return False

        ltp       = float(quote.get("ltp", pos["entry_price"]))
        entry     = pos["entry_price"]
        sl        = pos["stop_loss"]
        target    = pos["target"]
        qty       = pos["quantity"]
        direction = pos["direction"]

        if direction == "SHORT":
            profit_pct = (entry - ltp) / entry * 100

            # ── STOP LOSS HIT ─────────────────────────────────────
            if ltp >= sl:
                logger.warning(f"SL HIT [{symbol}]: ltp={ltp:.2f} >= sl={sl:.2f}")
                self._close_position(symbol, ltp, qty, "STOP_LOSS")
                return True

            # ── TARGET HIT ────────────────────────────────────────
            if ltp <= target:
                logger.info(f"TARGET HIT [{symbol}]: ltp={ltp:.2f} <= target={target:.2f}")
                self._close_position(symbol, ltp, qty, "TARGET")
                return True

            # ── TRAILING STOP LOSS ────────────────────────────────
            if profit_pct >= 0.8:
                new_sl = round(ltp * (1 + TRADING.trailing_sl_pct / 100), 2)
                if new_sl < sl:  # Move SL closer to entry (tighter) for shorts
                    old_sl = pos["stop_loss"]
                    pos["stop_loss"] = new_sl
                    pos["trailing_active"] = True
                    logger.info(f"Trailing SL [{symbol}]: {old_sl:.2f} → {new_sl:.2f}")

                    if self.alerts_sent.get(symbol) != "TRAILING":
                        self.notify(
                            f"📐 Trailing SL activated: {symbol}\n"
                            f"New SL: ₹{new_sl:.2f} | Profit: {profit_pct:.2f}%"
                        )
                        self.alerts_sent[symbol] = "TRAILING"

            # ── PROFIT MILESTONE ALERTS ───────────────────────────
            if profit_pct >= 1.0 and self.alerts_sent.get(symbol) != "PROFIT_1PCT":
                self.notify(f"📈 {symbol} at 1% profit | LTP ₹{ltp:.2f}")
                self.alerts_sent[symbol] = "PROFIT_1PCT"

            # ── APPROACHING SL ALERT ──────────────────────────────
            distance_to_sl = (sl - ltp) / entry * 100
            if 0 < distance_to_sl < 0.15 and self.alerts_sent.get(symbol) != "NEAR_SL":
                self.notify(f"⚠️ {symbol} approaching SL! LTP ₹{ltp:.2f} | SL ₹{sl:.2f}")
                self.alerts_sent[symbol] = "NEAR_SL"

        pos["last_ltp"] = ltp
        pos["last_profit_pct"] = profit_pct if direction == "SHORT" else 0
        return False

    def _close_position(self, symbol: str, exit_price: float, qty: int, reason: str):
        """Close position and record in DB."""
        # Cancel pending GTT orders
        self.executor.cancel_all_orders(symbol)

        # Cover the short
        self.executor.cover_short(symbol, qty, exit_price)

        # Record in DB
        pnl = self.risk_mgr.close_trade(symbol, exit_price)

        # Remove from monitoring
        self.unregister_position(symbol)

        # Notify
        emoji = "✅" if pnl >= 0 else "🔴"
        reason_text = "Target hit 🎯" if reason == "TARGET" else "Stop loss hit 🛑"
        self.notify(
            f"{emoji} {symbol} CLOSED — {reason_text}\n"
            f"Exit: ₹{exit_price:.2f} | P&L: ₹{pnl:.0f}"
        )

        logger.info(
            f"Position closed [{symbol}]: reason={reason} exit=₹{exit_price:.2f} pnl=₹{pnl:.2f}"
        )

    # ──────────────────────────────────────────────────────────────
    # STATUS
    # ──────────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        """Return current status of all monitored positions."""
        return {
            symbol: {
                "entry":    pos["entry_price"],
                "sl":       pos["stop_loss"],
                "target":   pos["target"],
                "qty":      pos["quantity"],
                "ltp":      pos.get("last_ltp"),
                "pnl_pct":  pos.get("last_profit_pct"),
                "trailing": pos.get("trailing_active", False),
            }
            for symbol, pos in self.positions.items()
        }
