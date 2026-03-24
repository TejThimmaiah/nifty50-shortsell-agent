"""
Tej Correlation Breakdown Detector + Smart Order Router
=========================================================
Two features in one file:

1. CORRELATION BREAKDOWN:
   Detects when historical correlations break — regime change signal.
   "Nifty usually follows SGX futures 95% of the time.
    Today it ignored a -0.8% SGX drop. Regime change — be careful."

2. SMART ORDER ROUTING:
   Splits large orders to avoid slippage.
   "Buying 100 shares at once moves price 0.15%.
    Split into 4 batches of 25 saves Rs 150 on this order."
"""

import os, time, logging, numpy as np
from dataclasses import dataclass
from typing import List, Optional
logger = logging.getLogger("correlation_order")

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False


# ══════════════════════════════════════════════════════
# PART 1 — CORRELATION BREAKDOWN DETECTOR
# ══════════════════════════════════════════════════════

@dataclass
class CorrelationAlert:
    pair:          str
    historical:    float   # Normal correlation
    current:       float   # Recent correlation
    breakdown_pct: float   # How much it deviated
    severity:      str     # "MILD" / "MODERATE" / "SEVERE"
    implication:   str


class CorrelationBreakdownDetector:
    """Monitors key correlations and alerts when they break."""

    PAIRS = {
        "SGX_NIFTY":    ("^NSEI",     "^STI"),
        "US_INDIA":     ("^NSEI",     "^GSPC"),
        "CRUDE_INFRA":  ("CL=F",      "^CNXIT"),
        "VIX_NIFTY":    ("^INDIAVIX", "^NSEI"),
        "FII_NIFTY":    ("^NSEI",     "USDINR=X"),
    }

    HISTORICAL_CORR = {
        "SGX_NIFTY":   0.88,
        "US_INDIA":    0.72,
        "CRUDE_INFRA": -0.45,
        "VIX_NIFTY":   -0.82,
        "FII_NIFTY":   -0.65,
    }

    def get_rolling_corr(self, t1: str, t2: str, window: int = 20) -> float:
        if not YF_AVAILABLE:
            return 0.7
        try:
            d1 = yf.download(t1, period="3mo", interval="1d", progress=False)["Close"].pct_change().dropna()
            d2 = yf.download(t2, period="3mo", interval="1d", progress=False)["Close"].pct_change().dropna()
            df = d1.to_frame("a").join(d2.to_frame("b")).dropna()
            if len(df) >= window:
                return float(df.tail(window).corr().iloc[0, 1])
        except Exception:
            pass
        return 0.7

    def detect_breakdowns(self) -> List[CorrelationAlert]:
        alerts = []
        for name, (t1, t2) in self.PAIRS.items():
            hist    = self.HISTORICAL_CORR.get(name, 0.7)
            current = self.get_rolling_corr(t1, t2)
            delta   = abs(current - hist)

            if delta < 0.15:
                continue

            severity = "SEVERE" if delta > 0.35 else ("MODERATE" if delta > 0.25 else "MILD")
            if current < hist:
                impl = f"{name} correlation weakened — markets may be decoupling"
            else:
                impl = f"{name} correlation strengthened — contagion risk elevated"

            alerts.append(CorrelationAlert(
                pair=name, historical=hist, current=current,
                breakdown_pct=delta * 100, severity=severity, implication=impl
            ))

        return sorted(alerts, key=lambda a: a.breakdown_pct, reverse=True)

    def format_for_telegram(self) -> str:
        alerts = self.detect_breakdowns()
        if not alerts:
            return "<b>Correlations</b>\nAll correlations normal — no regime change detected."
        msg = "<b>Correlation Breakdowns Detected!</b>\n\n"
        for a in alerts[:3]:
            sev_emoji = {"SEVERE": "🚨", "MODERATE": "⚠️", "MILD": "📊"}.get(a.severity, "📊")
            msg += (f"{sev_emoji} {a.pair}: {a.severity}\n"
                    f"   Normal: {a.historical:.2f} → Current: {a.current:.2f}\n"
                    f"   {a.implication}\n\n")
        return msg


# ══════════════════════════════════════════════════════
# PART 2 — SMART ORDER ROUTER
# ══════════════════════════════════════════════════════

@dataclass
class OrderSlice:
    quantity:    int
    delay_secs:  int
    order_type:  str   # "MARKET" / "LIMIT"
    limit_price: Optional[float]


class SmartOrderRouter:
    """
    Splits large orders to minimize slippage.
    Uses TWAP (Time-Weighted Average Price) strategy.
    """

    MAX_SINGLE_ORDER = 50     # Max qty in single order without splitting
    SLICE_DELAY      = 30     # Seconds between slices

    def plan_order(self, symbol: str, total_qty: int, side: str,
                   ltp: float, atr: float) -> List[OrderSlice]:
        """
        Plan order execution — split if needed.
        Returns list of order slices to execute sequentially.
        """
        if total_qty <= self.MAX_SINGLE_ORDER:
            return [OrderSlice(total_qty, 0, "MARKET", None)]

        # Calculate optimal slice size based on ATR
        volatility_factor = min(1.0, atr / ltp * 100)
        if volatility_factor > 0.5:
            n_slices = 4    # High volatility — more slices
        elif volatility_factor > 0.3:
            n_slices = 3
        else:
            n_slices = 2

        base_qty    = total_qty // n_slices
        remainder   = total_qty - base_qty * n_slices
        slices      = []

        for i in range(n_slices):
            qty  = base_qty + (remainder if i == n_slices - 1 else 0)
            delay = i * self.SLICE_DELAY
            slices.append(OrderSlice(
                quantity=qty,
                delay_secs=delay,
                order_type="MARKET",
                limit_price=None,
            ))

        logger.info(f"Smart routing {total_qty} {symbol}: {n_slices} slices of ~{base_qty}")
        return slices

    def execute_sliced(self, symbol: str, slices: List[OrderSlice],
                       side: str, kite=None) -> List[dict]:
        """Execute order slices sequentially."""
        results = []
        for i, sl in enumerate(slices):
            if sl.delay_secs > 0 and i > 0:
                logger.info(f"Waiting {sl.delay_secs}s before slice {i+1}...")
                time.sleep(sl.delay_secs)

            if kite is None:
                results.append({"slice": i, "qty": sl.quantity, "status": "SIMULATED"})
                continue

            try:
                from kiteconnect import KiteConnect
                txn = kite.TRANSACTION_TYPE_SELL if side == "SHORT" else kite.TRANSACTION_TYPE_BUY
                oid = kite.place_order(
                    variety=kite.VARIETY_REGULAR,
                    exchange=kite.EXCHANGE_NSE,
                    tradingsymbol=symbol,
                    transaction_type=txn,
                    quantity=sl.quantity,
                    product=kite.PRODUCT_MIS,
                    order_type=kite.ORDER_TYPE_MARKET,
                )
                results.append({"slice": i, "qty": sl.quantity, "order_id": oid, "status": "PLACED"})
                logger.info(f"Slice {i+1}/{len(slices)}: {sl.quantity} {symbol} → order {oid}")
            except Exception as e:
                logger.error(f"Slice {i+1} failed: {e}")
                results.append({"slice": i, "qty": sl.quantity, "status": "FAILED", "error": str(e)})

        return results


correlation_detector = CorrelationBreakdownDetector()
smart_router         = SmartOrderRouter()
