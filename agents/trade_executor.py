"""
Trade Executor Agent
Executes short sell orders via Zerodha Kite Connect API.
Includes a full paper trading mode for zero-cost simulation.
Auto-places stop loss and target orders immediately after entry.
"""

import logging
import json
import time
from datetime import datetime
from typing import Optional, Dict, Tuple
from config import (
    KITE_API_KEY, KITE_ACCESS_TOKEN, PAPER_TRADE,
    TRADING, ORDER_TIMEOUT_SEC
)
from agents.self_healer import SelfHealerAgent

logger = logging.getLogger(__name__)


class TradeExecutorAgent:
    """
    Handles all order operations: entry, stop loss, target, square-off.
    Paper trading mode mirrors the same interface — switch via PAPER_TRADE env var.
    """

    def __init__(self, healer: SelfHealerAgent = None):
        self.healer = healer or SelfHealerAgent()
        self._kite = None
        self._paper_positions: Dict[str, Dict] = {}

        if not PAPER_TRADE:
            self._init_kite()
        else:
            logger.info("*** PAPER TRADING MODE ACTIVE — no real orders will be placed ***")

    # ──────────────────────────────────────────────────────────────
    # MAIN OPERATIONS
    # ──────────────────────────────────────────────────────────────

    def short_sell(
        self,
        symbol: str,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        target: float,
    ) -> Dict:
        """
        Execute a complete short trade: entry + SL + target in one call.
        Returns trade info dict with order IDs.
        """
        logger.info(
            f"SHORT SELL: {symbol} | qty={quantity} | "
            f"entry={entry_price} | SL={stop_loss} | target={target}"
        )

        if PAPER_TRADE:
            return self._paper_short(symbol, quantity, entry_price, stop_loss, target)
        else:
            return self._live_short(symbol, quantity, entry_price, stop_loss, target)

    def cover_short(self, symbol: str, quantity: int, exit_price: float = None) -> Dict:
        """Buy back (cover) the short position."""
        logger.info(f"COVER SHORT: {symbol} | qty={quantity} | price={exit_price}")

        if PAPER_TRADE:
            return self._paper_cover(symbol, quantity, exit_price)
        else:
            return self._live_cover(symbol, quantity, exit_price)

    def cancel_all_orders(self, symbol: str):
        """Cancel all pending orders for a symbol (SL, target)."""
        if PAPER_TRADE:
            pos = self._paper_positions.get(symbol, {})
            pos["sl_cancelled"] = True
            pos["target_cancelled"] = True
            logger.info(f"[PAPER] Cancelled all orders for {symbol}")
            return

        if not self._kite:
            return
        try:
            orders = self._kite.orders()
            for order in orders:
                if (order.get("tradingsymbol") == symbol and
                        order.get("status") in ("TRIGGER PENDING", "OPEN")):
                    self._kite.cancel_order(
                        order_id=order["order_id"],
                        variety=order.get("variety", "regular")
                    )
                    logger.info(f"Cancelled order {order['order_id']} for {symbol}")
        except Exception as e:
            logger.error(f"Cancel orders error [{symbol}]: {e}")
            self.healer.heal(f"Failed to cancel orders for {symbol}: {e}", {"symbol": symbol})

    def square_off_all(self) -> Dict:
        """Emergency square-off — cover all open short positions at market price."""
        logger.info("=== SQUARING OFF ALL POSITIONS ===")
        results = {}

        if PAPER_TRADE:
            for symbol, pos in list(self._paper_positions.items()):
                if pos.get("status") == "OPEN":
                    qty = pos.get("quantity", 0)
                    current_price = pos.get("entry_price", 0)  # use entry as proxy
                    result = self._paper_cover(symbol, qty, current_price)
                    results[symbol] = result
            return results

        if not self._kite:
            return {}

        try:
            positions = self._kite.positions()
            net_positions = positions.get("net", [])
            for pos in net_positions:
                sym = pos.get("tradingsymbol")
                qty = abs(int(pos.get("quantity", 0)))
                if qty > 0 and int(pos.get("quantity", 0)) < 0:  # short position
                    self.cover_short(sym, qty)
                    results[sym] = {"status": "squared_off", "quantity": qty}
        except Exception as e:
            logger.error(f"Square off error: {e}")
            self.healer.heal(f"Failed to square off positions: {e}")

        return results

    def get_positions(self) -> Dict:
        """Fetch all current open positions."""
        if PAPER_TRADE:
            return {
                sym: pos
                for sym, pos in self._paper_positions.items()
                if pos.get("status") == "OPEN"
            }

        if not self._kite:
            return {}
        try:
            positions = self._kite.positions()
            return {
                p["tradingsymbol"]: {
                    "symbol": p["tradingsymbol"],
                    "quantity": p["quantity"],
                    "avg_price": p["average_price"],
                    "unrealised_pnl": p["unrealised"],
                }
                for p in positions.get("net", [])
                if p.get("quantity") != 0
            }
        except Exception as e:
            logger.error(f"Get positions error: {e}")
            return {}

    # ──────────────────────────────────────────────────────────────
    # LIVE ZERODHA IMPLEMENTATION
    # ──────────────────────────────────────────────────────────────

    def _init_kite(self):
        """Initialize Zerodha Kite Connect session."""
        try:
            from kiteconnect import KiteConnect
            self._kite = KiteConnect(api_key=KITE_API_KEY)
            self._kite.set_access_token(KITE_ACCESS_TOKEN)
            profile = self._kite.profile()
            logger.info(f"Kite connected: {profile.get('user_name')}")
        except ImportError:
            logger.error("kiteconnect package not installed. Run: pip install kiteconnect")
            raise
        except Exception as e:
            logger.error(f"Kite init error: {e}")
            fix = self.healer.heal(
                f"Zerodha Kite Connect initialization failed: {e}",
                {"error": str(e), "api": "Zerodha Kite"}
            )
            logger.info(f"Healer: {fix.get('solution')}")
            raise

    def _live_short(self, symbol, quantity, entry_price, stop_loss, target) -> Dict:
        """Place real short sell order on Zerodha with GTT for SL and target."""
        from kiteconnect import KiteConnect

        try:
            # Entry order — MIS (intraday), MARKET
            entry_order_id = self._kite.place_order(
                tradingsymbol=symbol,
                exchange=self._kite.EXCHANGE_NSE,
                transaction_type=self._kite.TRANSACTION_TYPE_SELL,   # SHORT
                quantity=quantity,
                product=self._kite.PRODUCT_MIS,                      # Intraday margin
                order_type=self._kite.ORDER_TYPE_MARKET,
                variety=self._kite.VARIETY_REGULAR,
            )
            logger.info(f"Entry order placed: {entry_order_id}")
            time.sleep(1)   # Let the entry fill

            # GTT for stop loss (trigger above entry for shorts)
            sl_gtt_id = self._kite.place_gtt(
                trigger_type=self._kite.GTT_TYPE_SINGLE,
                tradingsymbol=symbol,
                exchange=self._kite.EXCHANGE_NSE,
                trigger_values=[stop_loss],
                last_price=entry_price,
                orders=[{
                    "transaction_type": self._kite.TRANSACTION_TYPE_BUY,
                    "quantity": quantity,
                    "order_type": self._kite.ORDER_TYPE_MARKET,
                    "product": self._kite.PRODUCT_MIS,
                }]
            )

            # GTT for target (trigger below entry for shorts)
            target_gtt_id = self._kite.place_gtt(
                trigger_type=self._kite.GTT_TYPE_SINGLE,
                tradingsymbol=symbol,
                exchange=self._kite.EXCHANGE_NSE,
                trigger_values=[target],
                last_price=entry_price,
                orders=[{
                    "transaction_type": self._kite.TRANSACTION_TYPE_BUY,
                    "quantity": quantity,
                    "order_type": self._kite.ORDER_TYPE_MARKET,
                    "product": self._kite.PRODUCT_MIS,
                }]
            )

            return {
                "status": "EXECUTED",
                "symbol": symbol,
                "quantity": quantity,
                "entry_order_id": entry_order_id,
                "sl_gtt_id": sl_gtt_id,
                "target_gtt_id": target_gtt_id,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "target": target,
                "paper": False,
            }

        except Exception as e:
            logger.error(f"Live short order failed [{symbol}]: {e}")
            fix = self.healer.search_api_error_fix(str(e), "Zerodha Kite Connect")
            logger.info(f"Healer fix: {fix}")
            return {"status": "FAILED", "error": str(e), "healer_fix": fix}

    def _live_cover(self, symbol, quantity, exit_price=None) -> Dict:
        """Cover a live short position."""
        try:
            order_id = self._kite.place_order(
                tradingsymbol=symbol,
                exchange=self._kite.EXCHANGE_NSE,
                transaction_type=self._kite.TRANSACTION_TYPE_BUY,
                quantity=quantity,
                product=self._kite.PRODUCT_MIS,
                order_type=self._kite.ORDER_TYPE_MARKET,
                variety=self._kite.VARIETY_REGULAR,
            )
            return {"status": "COVERED", "symbol": symbol, "order_id": order_id, "paper": False}
        except Exception as e:
            logger.error(f"Cover order failed [{symbol}]: {e}")
            return {"status": "FAILED", "error": str(e)}

    # ──────────────────────────────────────────────────────────────
    # PAPER TRADING IMPLEMENTATION
    # ──────────────────────────────────────────────────────────────

    def _paper_short(self, symbol, quantity, entry_price, stop_loss, target) -> Dict:
        """Simulate a short order — same interface as live."""
        self._paper_positions[symbol] = {
            "status": "OPEN",
            "direction": "SHORT",
            "symbol": symbol,
            "quantity": quantity,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target": target,
            "entry_time": datetime.now().isoformat(),
            "sl_triggered": False,
            "target_triggered": False,
        }
        fake_id = f"PAPER_{symbol}_{int(time.time())}"
        logger.info(f"[PAPER] SHORT {quantity} {symbol} @ ₹{entry_price} | SL={stop_loss} | T={target}")
        return {
            "status": "EXECUTED",
            "symbol": symbol,
            "quantity": quantity,
            "entry_order_id": fake_id,
            "sl_gtt_id": fake_id + "_SL",
            "target_gtt_id": fake_id + "_TGT",
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target": target,
            "paper": True,
        }

    def _paper_cover(self, symbol, quantity, exit_price=None) -> Dict:
        """Simulate covering a short position."""
        pos = self._paper_positions.get(symbol)
        if pos:
            ep = pos.get("entry_price", exit_price or 0)
            pnl = (ep - (exit_price or ep)) * quantity
            pos["status"] = "CLOSED"
            pos["exit_price"] = exit_price
            pos["pnl"] = pnl
            pos["exit_time"] = datetime.now().isoformat()
            logger.info(f"[PAPER] COVERED {quantity} {symbol} @ ₹{exit_price} | P&L=₹{pnl:.2f}")
            return {"status": "COVERED", "symbol": symbol, "pnl": pnl, "paper": True}
        return {"status": "NOT_FOUND", "symbol": symbol}

    def check_paper_triggers(self, symbol: str, current_price: float) -> Optional[str]:
        """Check if SL or target has been hit in paper mode. Returns 'SL', 'TARGET', or None."""
        pos = self._paper_positions.get(symbol)
        if not pos or pos.get("status") != "OPEN":
            return None
        if pos.get("direction") == "SHORT":
            if current_price >= pos["stop_loss"]:
                logger.info(f"[PAPER] SL hit for {symbol} @ ₹{current_price}")
                self._paper_cover(symbol, pos["quantity"], current_price)
                return "SL"
            if current_price <= pos["target"]:
                logger.info(f"[PAPER] Target hit for {symbol} @ ₹{current_price}")
                self._paper_cover(symbol, pos["quantity"], current_price)
                return "TARGET"
        return None
