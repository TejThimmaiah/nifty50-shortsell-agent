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
                        order.get("product") == "MIS" and
                        order.get("status") in ("TRIGGER PENDING", "OPEN")):
                    variety = order.get("variety", "regular")
                    self._kite.cancel_order(
                        order_id=order["order_id"],
                        variety=variety,
                    )
                    logger.info(f"Cancelled MIS order {order['order_id']} for {symbol} (status={order['status']})")
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
        """Fetch all current open positions from the live Zerodha account."""
        if PAPER_TRADE:
            return {
                sym: pos
                for sym, pos in self._paper_positions.items()
                if pos.get("status") == "OPEN"
            }

        if not self._ensure_kite():
            return {}
        try:
            positions = self._kite.positions()
            return {
                p["tradingsymbol"]: {
                    "symbol":          p["tradingsymbol"],
                    "quantity":        p["quantity"],
                    "avg_price":       p.get("average_price", 0),
                    "last_price":      p.get("last_price", 0),
                    "unrealised_pnl":  p.get("unrealised", 0),
                    "realised_pnl":    p.get("realised", 0),
                    "pnl":             p.get("pnl", 0),
                    "product":         p.get("product", "MIS"),
                }
                for p in positions.get("net", [])
                if p.get("quantity") != 0
            }
        except Exception as e:
            logger.error(f"Get positions error: {e}")
            return {}

    def get_account_balance(self) -> Dict:
        """
        Fetch real account balance and margin from the live Zerodha account.
        Returns actual available cash, used margin, and today's P&L.
        """
        if PAPER_TRADE:
            total_pnl = sum(
                p.get("pnl", 0)
                for p in self._paper_positions.values()
                if p.get("status") == "CLOSED"
            )
            return {
                "available_cash": PAPER_TRADE,
                "used_margin": 0,
                "total_pnl_today": total_pnl,
                "paper": True,
            }

        if not self._ensure_kite():
            return {}
        try:
            margins  = self._kite.margins()
            equity   = margins.get("equity", {})
            net_data = equity.get("net", 0)
            available = equity.get("available", {})

            # Today's P&L from positions
            positions = self._kite.positions()
            day_pnl   = sum(p.get("pnl", 0) for p in positions.get("day", []))

            result = {
                "available_cash":  float(available.get("cash", net_data) if isinstance(available, dict) else net_data),
                "used_margin":     float(equity.get("utilised", {}).get("debits", 0) if isinstance(equity.get("utilised"), dict) else 0),
                "total_pnl_today": float(day_pnl),
                "paper": False,
            }
            logger.info(
                f"Account: cash=₹{result['available_cash']:,.0f} | "
                f"margin_used=₹{result['used_margin']:,.0f} | "
                f"day_pnl=₹{result['total_pnl_today']:+,.0f}"
            )
            return result
        except Exception as e:
            logger.error(f"Get account balance error: {e}")
            return {}

    # ──────────────────────────────────────────────────────────────
    # LIVE ZERODHA IMPLEMENTATION
    # ──────────────────────────────────────────────────────────────

    def _init_kite(self):
        """Initialize Kite — non-crashing. Retries on each trade attempt."""
        import os
        api_key      = os.getenv("KITE_API_KEY", "") or KITE_API_KEY
        access_token = os.getenv("KITE_ACCESS_TOKEN", "") or KITE_ACCESS_TOKEN
        if not api_key or not access_token:
            logger.warning("Kite credentials not set — will retry when token available")
            self._kite = None
            return
        try:
            from kiteconnect import KiteConnect
            self._kite = KiteConnect(api_key=api_key)
            self._kite.set_access_token(access_token)
            profile = self._kite.profile()
            logger.info(f"✅ Kite connected: {profile.get('user_name')} ({profile.get('user_id')})")
        except Exception as e:
            logger.warning(f"Kite init: {e} — will retry")
            self._kite = None

    def _ensure_kite(self) -> bool:
        """Ensure Kite connected — always reads latest env token."""
        import os
        api_key      = os.getenv("KITE_API_KEY", "") or KITE_API_KEY
        access_token = os.getenv("KITE_ACCESS_TOKEN", "") or KITE_ACCESS_TOKEN
        if not api_key or not access_token:
            logger.warning("No Kite credentials — cannot place live orders")
            return False
        # Always re-init to pick up latest token
        try:
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(access_token)
            kite.profile()   # Validate
            self._kite = kite
            logger.info("✅ Kite ready for orders")
            return True
        except Exception as e:
            logger.error(f"Kite connection failed: {e}")
            self._kite = None
            return False

    def _live_short(self, symbol, quantity, entry_price, stop_loss, target) -> Dict:
        """
        Place a real MIS short sell order on Zerodha NSE.
        Uses:
          - SELL MIS MARKET  for entry
          - BUY  MIS SL-M    for stop loss (triggers when price RISES above SL)
          - BUY  MIS LIMIT   for target   (triggers when price FALLS to target)
        GTT orders are NOT used — they don't work reliably for intraday MIS positions.
        """
        # Always ensure fresh Kite connection before placing orders
        if not self._ensure_kite():
            return {"status": "ERROR", "message": "Kite not connected — no valid access token"}

        sl_order_id     = None
        target_order_id = None

        try:
            # ── 1. Entry: SELL MIS MARKET (short) ────────────────────────────
            entry_order_id = self._kite.place_order(
                tradingsymbol=symbol,
                exchange=self._kite.EXCHANGE_NSE,
                transaction_type=self._kite.TRANSACTION_TYPE_SELL,
                quantity=quantity,
                product=self._kite.PRODUCT_MIS,
                order_type=self._kite.ORDER_TYPE_MARKET,
                variety=self._kite.VARIETY_REGULAR,
            market_protection=0.05,
            )
            logger.info(f"✅ Entry order placed: {entry_order_id} | {symbol} SHORT {quantity} @ MARKET")
            time.sleep(1)   # Give exchange 1 second to process entry

            # ── 2. Stop Loss: BUY MIS SL-M (triggers above entry for shorts) ─
            #    trigger_price = stop_loss (price at which buy-back is triggered)
            sl_order_id = self._kite.place_order(
                tradingsymbol=symbol,
                exchange=self._kite.EXCHANGE_NSE,
                transaction_type=self._kite.TRANSACTION_TYPE_BUY,
                quantity=quantity,
                product=self._kite.PRODUCT_MIS,
                order_type=self._kite.ORDER_TYPE_SLM,   # Stop Loss Market
                trigger_price=round(stop_loss, 2),
                variety=self._kite.VARIETY_REGULAR,
                tag="SL",
            )
            logger.info(f"✅ SL order placed: {sl_order_id} | trigger={stop_loss}")

            # ── 3. Target: BUY MIS LIMIT (fills when price falls to target) ──
            target_order_id = self._kite.place_order(
                tradingsymbol=symbol,
                exchange=self._kite.EXCHANGE_NSE,
                transaction_type=self._kite.TRANSACTION_TYPE_BUY,
                quantity=quantity,
                product=self._kite.PRODUCT_MIS,
                order_type=self._kite.ORDER_TYPE_LIMIT,
                price=round(target, 2),
                variety=self._kite.VARIETY_REGULAR,
                tag="TGT",
            )
            logger.info(f"✅ Target order placed: {target_order_id} | price={target}")

            return {
                "status": "EXECUTED",
                "symbol": symbol,
                "quantity": quantity,
                "entry_order_id": entry_order_id,
                "sl_order_id": sl_order_id,
                "target_order_id": target_order_id,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "target": target,
                "paper": False,
            }

        except Exception as e:
            logger.error(f"Live short order failed [{symbol}]: {e}")
            # Cancel any orders that were already placed
            for oid in [sl_order_id, target_order_id]:
                if oid:
                    try:
                        self._kite.cancel_order(order_id=oid, variety=self._kite.VARIETY_REGULAR)
                        logger.info(f"Rolled back order {oid}")
                    except Exception:
                        pass
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
            market_protection=0.05,
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
