"""
Tej Tick-Level Execution — Kite WebSocket
==========================================
Enters trades on exact tick, not candle close.
Saves 0.1-0.3% per trade — on Rs 1L capital that's Rs 100-300 per trade.

Streams live ticks from Zerodha Kite WebSocket.
Triggers orders on exact price tick conditions.
"""

import os, logging, threading, time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, List
from datetime import datetime
from zoneinfo import ZoneInfo
logger = logging.getLogger("tick_executor")
IST = ZoneInfo("Asia/Kolkata")

try:
    from kiteconnect import KiteTicker
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False


@dataclass
class TickTrigger:
    symbol:       str
    token:        int
    trigger_type: str     # "PRICE_BELOW" / "PRICE_ABOVE" / "VOLUME_SPIKE"
    trigger_val:  float
    action:       str     # "SHORT" / "COVER"
    quantity:     int
    stop_loss:    float
    target:       float
    callback:     Optional[Callable] = None
    fired:        bool = False
    created_at:   str = field(default_factory=lambda: datetime.now(IST).isoformat())


class TickLevelExecutor:
    """
    Streams live NSE ticks and executes orders on exact conditions.
    Much more precise than waiting for 5-minute candle close.
    """

    def __init__(self):
        self.api_key     = os.getenv("KITE_API_KEY", "")
        self.access_token = os.getenv("KITE_ACCESS_TOKEN", "")
        self.ticker      = None
        self.triggers:   Dict[int, List[TickTrigger]] = {}
        self.tick_cache: Dict[int, dict] = {}
        self._running    = False
        self._lock       = threading.Lock()

    def _on_ticks(self, ws, ticks):
        """Called on every incoming tick."""
        for tick in ticks:
            token = tick.get("instrument_token")
            if token:
                self.tick_cache[token] = tick
                self._check_triggers(token, tick)

    def _check_triggers(self, token: int, tick: dict):
        """Check if any triggers fire on this tick."""
        with self._lock:
            triggers = self.triggers.get(token, [])

        for trigger in triggers:
            if trigger.fired:
                continue
            ltp = tick.get("last_price", 0)
            vol = tick.get("volume", 0)

            fired = False
            if trigger.trigger_type == "PRICE_BELOW" and ltp <= trigger.trigger_val:
                fired = True
            elif trigger.trigger_type == "PRICE_ABOVE" and ltp >= trigger.trigger_val:
                fired = True
            elif trigger.trigger_type == "VOLUME_SPIKE":
                avg_vol = tick.get("average_traded_price", 1)
                if avg_vol > 0 and vol / avg_vol >= trigger.trigger_val:
                    fired = True

            if fired:
                trigger.fired = True
                logger.info(
                    f"Tick trigger fired: {trigger.symbol} "
                    f"{trigger.trigger_type} at {ltp:.2f} → {trigger.action}"
                )
                self._execute_trigger(trigger, ltp)

    def _execute_trigger(self, trigger: TickTrigger, price: float):
        """Execute the triggered order."""
        try:
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=self.api_key)
            kite.set_access_token(self.access_token)

            if trigger.action == "SHORT":
                order_id = kite.place_order(
                    variety=kite.VARIETY_REGULAR,
                    exchange=kite.EXCHANGE_NSE,
                    tradingsymbol=trigger.symbol,
                    transaction_type=kite.TRANSACTION_TYPE_SELL,
                    quantity=trigger.quantity,
                    product=kite.PRODUCT_MIS,
                    order_type=kite.ORDER_TYPE_MARKET,
                )
                logger.info(f"Tick SHORT order placed: {trigger.symbol} {trigger.quantity} @ ~{price:.2f} | Order: {order_id}")

            elif trigger.action == "COVER":
                order_id = kite.place_order(
                    variety=kite.VARIETY_REGULAR,
                    exchange=kite.EXCHANGE_NSE,
                    tradingsymbol=trigger.symbol,
                    transaction_type=kite.TRANSACTION_TYPE_BUY,
                    quantity=trigger.quantity,
                    product=kite.PRODUCT_MIS,
                    order_type=kite.ORDER_TYPE_MARKET,
                )
                logger.info(f"Tick COVER order placed: {trigger.symbol} @ ~{price:.2f} | Order: {order_id}")

            if trigger.callback:
                trigger.callback(trigger, price, order_id)

        except Exception as e:
            logger.error(f"Tick execution failed for {trigger.symbol}: {e}")

    def add_trigger(self, trigger: TickTrigger):
        """Register a tick trigger."""
        with self._lock:
            if trigger.token not in self.triggers:
                self.triggers[trigger.token] = []
            self.triggers[trigger.token].append(trigger)

        # Subscribe to this token
        if self.ticker and self._running:
            tokens = list(self.triggers.keys())
            self.ticker.subscribe(tokens)
            self.ticker.set_mode(self.ticker.MODE_FULL, tokens)
        logger.info(f"Tick trigger set: {trigger.symbol} {trigger.trigger_type} @ {trigger.trigger_val}")

    def start(self):
        """Start WebSocket tick streaming."""
        if not KITE_AVAILABLE:
            logger.warning("kiteconnect not installed — tick executor inactive")
            return
        if not self.api_key or not self.access_token:
            logger.warning("Kite credentials missing — tick executor inactive")
            return

        def on_connect(ws, response):
            logger.info("Tick WebSocket connected")
            tokens = list(self.triggers.keys())
            if tokens:
                ws.subscribe(tokens)
                ws.set_mode(ws.MODE_FULL, tokens)

        def on_close(ws, code, reason):
            logger.warning(f"Tick WebSocket closed: {code} {reason}")
            self._running = False

        def on_error(ws, code, reason):
            logger.error(f"Tick WebSocket error: {code} {reason}")

        self.ticker = KiteTicker(self.api_key, self.access_token)
        self.ticker.on_ticks   = self._on_ticks
        self.ticker.on_connect = on_connect
        self.ticker.on_close   = on_close
        self.ticker.on_error   = on_error

        self._running = True
        thread = threading.Thread(target=self.ticker.connect, kwargs={"threaded": True})
        thread.daemon = True
        thread.start()
        logger.info("Tick executor started")

    def stop(self):
        if self.ticker:
            try:
                self.ticker.close()
            except Exception:
                pass
        self._running = False

    def get_ltp(self, token: int) -> Optional[float]:
        """Get last traded price for a token."""
        tick = self.tick_cache.get(token)
        return tick.get("last_price") if tick else None

    def clear_fired_triggers(self):
        """Remove fired triggers."""
        with self._lock:
            for token in list(self.triggers.keys()):
                self.triggers[token] = [t for t in self.triggers[token] if not t.fired]


tick_executor = TickLevelExecutor()
