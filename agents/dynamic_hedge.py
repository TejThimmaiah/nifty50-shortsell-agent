"""
Tej Dynamic Hedging — Nifty PUT Options
=========================================
Automatically buys Nifty PUT options when short exposure exceeds threshold.
Protects against sudden market reversal (stop-hunt, circuit breaker).

"I have Rs 3L short exposure. Buying 1 lot Nifty 23000 PE
 for Rs 2,500 to hedge against sudden reversal."
"""

import os, logging
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, date
from zoneinfo import ZoneInfo
logger = logging.getLogger("dynamic_hedge")
IST = ZoneInfo("Asia/Kolkata")

HEDGE_THRESHOLD   = 200_000   # Hedge when short exposure > Rs 2L
MAX_HEDGE_COST    = 5_000     # Max spend on hedge per day (Rs)
NIFTY_LOT_SIZE    = 75


@dataclass
class HedgePosition:
    symbol:     str
    strike:     int
    expiry:     str
    option_type: str  # "PE"
    lots:       int
    entry_price: float
    total_cost: float
    order_id:   str


class DynamicHedge:
    """
    Monitors total short exposure and auto-hedges with Nifty PUTs.
    """

    def __init__(self):
        self.api_key      = os.getenv("KITE_API_KEY", "")
        self.access_token = os.getenv("KITE_ACCESS_TOKEN", "")
        self.active_hedge: Optional[HedgePosition] = None
        self.today_hedge_cost = 0.0

    def _get_nifty_level(self) -> float:
        try:
            import yfinance as yf
            t    = yf.Ticker("^NSEI")
            hist = t.history(period="1d", interval="1m")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            pass
        return 23000.0

    def _get_atm_put_strike(self, nifty: float) -> int:
        """Get ATM put strike (rounded to nearest 100)."""
        return int(round(nifty / 100) * 100)

    def _get_nearest_expiry(self) -> str:
        """Get nearest weekly expiry."""
        from datetime import timedelta
        today = date.today()
        # Nifty weekly expiry is Thursday
        days_to_thu = (3 - today.weekday()) % 7
        if days_to_thu == 0:
            days_to_thu = 7
        expiry = today + timedelta(days=days_to_thu)
        return expiry.strftime("%d%b%Y").upper()

    def calculate_short_exposure(self, positions: dict) -> float:
        """Calculate total Rs value of short positions."""
        total = 0.0
        for sym, pos in positions.items():
            qty   = abs(pos.get("quantity", 0))
            price = pos.get("entry_price", 0)
            total += qty * price
        return total

    def should_hedge(self, short_exposure: float) -> bool:
        """Decide if hedge is needed."""
        if self.active_hedge:
            return False  # Already hedged
        if self.today_hedge_cost >= MAX_HEDGE_COST:
            return False  # Already spent limit
        return short_exposure >= HEDGE_THRESHOLD

    def buy_put_option(self, nifty_level: float) -> Optional[HedgePosition]:
        """Buy 1 lot Nifty ATM put option."""
        strike = self._get_atm_put_strike(nifty_level)
        expiry = self._get_nearest_expiry()
        symbol = f"NIFTY{expiry}{strike}PE"

        logger.info(f"Buying hedge PUT: {symbol}")

        try:
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=self.api_key)
            kite.set_access_token(self.access_token)

            # Get current option price
            quote = kite.ltp([f"NFO:{symbol}"])
            ltp   = quote.get(f"NFO:{symbol}", {}).get("last_price", 100)

            order_id = kite.place_order(
                variety=kite.VARIETY_REGULAR,
                exchange="NFO",
                tradingsymbol=symbol,
                transaction_type=kite.TRANSACTION_TYPE_BUY,
                quantity=NIFTY_LOT_SIZE,
                product=kite.PRODUCT_MIS,
                order_type=kite.ORDER_TYPE_MARKET,
            )

            cost = ltp * NIFTY_LOT_SIZE
            self.today_hedge_cost += cost
            self.active_hedge = HedgePosition(
                symbol=symbol, strike=strike, expiry=expiry,
                option_type="PE", lots=1, entry_price=ltp,
                total_cost=cost, order_id=str(order_id),
            )
            logger.info(f"Hedge placed: {symbol} @ Rs {ltp:.0f} | Cost: Rs {cost:.0f}")
            return self.active_hedge

        except Exception as e:
            logger.error(f"Hedge order failed: {e}")
            return None

    def auto_hedge_if_needed(self, positions: dict) -> Optional[str]:
        """
        Main entry point — call this after every new short position.
        Returns Telegram message if hedge was placed.
        """
        exposure = self.calculate_short_exposure(positions)

        if not self.should_hedge(exposure):
            return None

        nifty = self._get_nifty_level()
        hedge = self.buy_put_option(nifty)

        if hedge:
            return (
                f"🛡️ <b>Auto-Hedge Placed</b>\n\n"
                f"Short exposure: Rs {exposure:,.0f}\n"
                f"Hedge: 1 lot {hedge.symbol}\n"
                f"Cost: Rs {hedge.total_cost:,.0f}\n"
                f"Protection against sudden reversal"
            )
        return None

    def remove_hedge(self) -> bool:
        """Square off hedge position at EOD."""
        if not self.active_hedge:
            return True
        try:
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=self.api_key)
            kite.set_access_token(self.access_token)
            kite.place_order(
                variety=kite.VARIETY_REGULAR,
                exchange="NFO",
                tradingsymbol=self.active_hedge.symbol,
                transaction_type=kite.TRANSACTION_TYPE_SELL,
                quantity=NIFTY_LOT_SIZE,
                product=kite.PRODUCT_MIS,
                order_type=kite.ORDER_TYPE_MARKET,
            )
            self.active_hedge = None
            logger.info("Hedge squared off")
            return True
        except Exception as e:
            logger.error(f"Hedge square-off failed: {e}")
            return False


dynamic_hedge = DynamicHedge()
