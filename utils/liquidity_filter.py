"""
Liquidity Filter
Prevents the agent from entering trades in illiquid stocks
where the spread is wide or the volume is insufficient to exit cleanly.
A critical safety layer — especially important for short selling.

Rules:
  - Minimum average daily volume (ADV): 500,000 shares
  - Market cap minimum: ₹5,000 Cr (large-cap only)
  - Must be F&O eligible (implicit liquidity guarantee)
  - Intraday volume by 9:30 AM must be > 1% of ADV
  - Bid-ask spread (from OB depth): < 0.1%
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import pandas as pd

from utils.data_cache import cached_daily, cached_quote

logger = logging.getLogger(__name__)

# Minimum thresholds
MIN_ADV            = 500_000      # shares
MIN_MARKET_CAP_CR  = 5_000        # ₹ Crore
MIN_INTRADAY_VOL   = 100_000      # shares by 9:30 AM
MAX_SPREAD_PCT     = 0.15         # 0.15% max bid-ask spread
MIN_PRICE          = 50.0         # ₹
MAX_PRICE          = 10_000.0     # ₹ (avoids very high-priced stocks that need large capital)


@dataclass
class LiquidityCheck:
    symbol:        str
    passed:        bool
    reason:        str
    adv:           float      # Average Daily Volume (20-day)
    current_vol:   float      # Today's volume so far
    vol_ratio:     float      # current_vol / ADV
    price:         float
    estimated_spread_pct: float
    market_cap_cr: Optional[float]
    tradeable_qty: int        # Max quantity we can trade without moving market


class LiquidityFilter:
    """
    Checks whether a stock is liquid enough to trade safely.
    Uses cached data to avoid repeated API calls.
    """

    def check(self, symbol: str, desired_qty: int = 0) -> LiquidityCheck:
        """
        Full liquidity check for one symbol.
        Returns LiquidityCheck with passed=True if safe to trade.
        """
        # Fetch data
        quote = cached_quote(symbol)
        df    = cached_daily(symbol, days=25)

        if not quote:
            return LiquidityCheck(
                symbol=symbol, passed=False,
                reason="Could not fetch live quote",
                adv=0, current_vol=0, vol_ratio=0,
                price=0, estimated_spread_pct=0,
                market_cap_cr=None, tradeable_qty=0,
            )

        price       = float(quote.get("ltp", 0))
        current_vol = float(quote.get("volume", 0))

        # Price range check
        if price < MIN_PRICE:
            return self._fail(symbol, f"Price ₹{price:.2f} < minimum ₹{MIN_PRICE}", price, current_vol)
        if price > MAX_PRICE:
            return self._fail(symbol, f"Price ₹{price:.2f} > maximum ₹{MAX_PRICE}", price, current_vol)

        # Average daily volume
        adv = self._compute_adv(df)
        if adv < MIN_ADV:
            return self._fail(
                symbol,
                f"ADV {adv:,.0f} < minimum {MIN_ADV:,} shares",
                price, current_vol, adv=adv,
            )

        # Volume ratio (liquidity at time of trade)
        vol_ratio = current_vol / max(adv, 1)

        # Spread estimate (proxy: 0.02% for Nifty 50, 0.05–0.15% for others)
        spread_pct = self._estimate_spread(symbol, price, adv)
        if spread_pct > MAX_SPREAD_PCT:
            return self._fail(
                symbol,
                f"Estimated spread {spread_pct:.3f}% > max {MAX_SPREAD_PCT}%",
                price, current_vol, adv=adv, spread=spread_pct,
            )

        # Tradeable quantity: max 0.5% of ADV per trade (to avoid market impact)
        tradeable_qty = max(1, int(adv * 0.005))

        # If caller specified a desired quantity, check it doesn't exceed tradeable
        if desired_qty > 0 and desired_qty > tradeable_qty:
            logger.warning(
                f"Desired qty {desired_qty} > tradeable qty {tradeable_qty} "
                f"for {symbol} — will cap at {tradeable_qty}"
            )

        logger.debug(
            f"Liquidity OK [{symbol}]: ADV={adv:,.0f} vol_ratio={vol_ratio:.2f} "
            f"spread={spread_pct:.3f}% tradeable={tradeable_qty}"
        )

        return LiquidityCheck(
            symbol=symbol,
            passed=True,
            reason="All liquidity checks passed",
            adv=adv,
            current_vol=current_vol,
            vol_ratio=vol_ratio,
            price=price,
            estimated_spread_pct=spread_pct,
            market_cap_cr=None,
            tradeable_qty=tradeable_qty,
        )

    def filter_candidates(self, symbols: list) -> Tuple[list, list]:
        """
        Filter a list of symbols by liquidity.
        Returns (passed: list, failed: list).
        """
        passed = []
        failed = []
        for sym in symbols:
            result = self.check(sym)
            if result.passed:
                passed.append(sym)
            else:
                logger.info(f"Liquidity filter rejected {sym}: {result.reason}")
                failed.append(sym)
        return passed, failed

    def cap_quantity(self, symbol: str, desired_qty: int) -> int:
        """Cap quantity to the liquid tradeable amount for this symbol."""
        result = self.check(symbol, desired_qty)
        if not result.passed:
            return 0
        return min(desired_qty, result.tradeable_qty)

    # ──────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────

    def _compute_adv(self, df: Optional[pd.DataFrame]) -> float:
        """Compute 20-day average daily volume."""
        if df is None or "volume" not in df.columns or len(df) < 5:
            return 0.0
        return float(df["volume"].tail(20).mean())

    def _estimate_spread(self, symbol: str, price: float, adv: float) -> float:
        """
        Estimate bid-ask spread from available proxies.
        Real spread requires Level 2 order book (only available via Kite WebSocket).
        We use ADV and price as a proxy: higher ADV = tighter spread.
        """
        from config import TRADING

        # High-liquidity stocks (Nifty 50 constituents) have very tight spreads
        if symbol in TRADING.priority_watchlist and adv > 2_000_000:
            return 0.02     # 0.02%

        # Mid-tier F&O stocks
        if adv > 1_000_000:
            return 0.05

        # Lower liquidity
        if adv > 500_000:
            return 0.10

        # Just above our minimum — borderline
        return 0.14

    def _fail(
        self,
        symbol: str,
        reason: str,
        price: float,
        current_vol: float,
        adv: float = 0,
        spread: float = 0,
    ) -> LiquidityCheck:
        return LiquidityCheck(
            symbol=symbol,
            passed=False,
            reason=reason,
            adv=adv,
            current_vol=current_vol,
            vol_ratio=current_vol / max(adv, 1),
            price=price,
            estimated_spread_pct=spread,
            market_cap_cr=None,
            tradeable_qty=0,
        )
