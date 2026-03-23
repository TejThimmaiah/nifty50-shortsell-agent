"""
Fibonacci & Pivot Point Calculator
These levels are self-fulfilling prophecies — millions of traders and
algorithms watch them, so they work because everyone acts on them.

Fibonacci Levels (from swing high/low):
  0.236, 0.382, 0.500, 0.618, 0.786
  Extension: 1.272, 1.414, 1.618, 2.000
  For shorts: 0.618 and 0.786 retracement = strong resistance

Pivot Points (calculated from previous day's OHLC):
  Classic:   PP = (H+L+C)/3,  R1=2PP-L,  S1=2PP-H
  Fibonacci: Uses Fibonacci ratios applied to PP range
  Camarilla: Tight intraday levels for short-term traders

Why this matters for short selling:
  - Price often stalls at Fibonacci levels (61.8% retracement = "golden ratio")
  - Pivot resistance levels are widely watched → self-fulfilling resistance
  - When RSI is overbought AND price is at a Fib level → very strong short signal
  - Options writers hedge at Fibonacci levels → creates real selling pressure
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FibonacciLevels:
    swing_high:    float
    swing_low:     float
    direction:     str      # UP (retracing down) | DOWN (retracing up)
    retracements:  Dict[str, float]    # {"0.236": 1234.5, "0.382": ...}
    extensions:    Dict[str, float]    # {"1.272": ..., "1.618": ...}
    nearest_resistance: Optional[float]   # nearest fib level above current price
    nearest_support:    Optional[float]   # nearest fib level below current price
    current_zone:       str              # which Fib band the price is in


@dataclass
class PivotLevels:
    pivot:    float
    r1: float; r2: float; r3: float
    s1: float; s2: float; s3: float
    camarilla_r1: float; camarilla_r2: float; camarilla_r3: float; camarilla_r4: float
    camarilla_s1: float; camarilla_s2: float; camarilla_s3: float; camarilla_s4: float
    nearest_resistance: Optional[float]
    nearest_support:    Optional[float]


@dataclass
class KeyPriceLevels:
    symbol:          str
    current_price:   float
    fibonacci:       Optional[FibonacciLevels]
    pivots:          Optional[PivotLevels]
    all_resistances: List[float]    # sorted ascending
    all_supports:    List[float]    # sorted ascending
    nearest_resistance_pct: float   # % above current price
    nearest_support_pct:    float   # % below current price
    at_fib_resistance:      bool
    at_pivot_resistance:    bool
    short_confluence:       float   # 0–1, multiple levels clustered = higher


class FiboPivotCalculator:
    """
    Computes Fibonacci levels and pivot points for short-selling resistance analysis.
    """

    FIB_RETRACE = [0.236, 0.382, 0.500, 0.618, 0.786]
    FIB_EXTEND  = [1.000, 1.272, 1.414, 1.618, 2.000, 2.618]

    def compute(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        swing_lookback: int = 20,
    ) -> KeyPriceLevels:
        """
        Compute all key price levels for a symbol.
        df: daily OHLCV (at least 20 bars).
        """
        if df is None or len(df) < 5:
            return self._empty(symbol)

        current_price = float(df["close"].iloc[-1])

        # Fibonacci levels
        fib = self._compute_fibonacci(df, current_price, swing_lookback)

        # Pivot levels (from previous day)
        pivots = self._compute_pivots(df)

        # Collect all resistance and support levels
        all_res = []
        all_sup = []

        if fib:
            for level in fib.retracements.values():
                if level > current_price * 1.001:
                    all_res.append(level)
                elif level < current_price * 0.999:
                    all_sup.append(level)

        if pivots:
            for level in [pivots.r1, pivots.r2, pivots.r3,
                          pivots.camarilla_r1, pivots.camarilla_r2]:
                if level > current_price * 1.001:
                    all_res.append(level)
            for level in [pivots.s1, pivots.s2, pivots.s3]:
                if level < current_price * 0.999:
                    all_sup.append(level)

        all_res  = sorted(set(all_res))
        all_sup  = sorted(set(all_sup), reverse=True)

        nearest_res = all_res[0]  if all_res else None
        nearest_sup = all_sup[0]  if all_sup else None

        res_pct = (nearest_res - current_price) / current_price * 100 if nearest_res else 99.0
        sup_pct = (current_price - nearest_sup) / current_price * 100 if nearest_sup else 99.0

        # Is price at a resistance zone?
        at_fib = fib and fib.nearest_resistance and \
                 abs(fib.nearest_resistance - current_price) / current_price < 0.005
        at_pvt = pivots and pivots.nearest_resistance and \
                 abs(pivots.nearest_resistance - current_price) / current_price < 0.005

        # Confluence: multiple levels within 0.5% of each other near current price
        nearby = [l for l in all_res if abs(l - current_price) / current_price < 0.015]
        confluence = min(1.0, len(nearby) * 0.25)

        return KeyPriceLevels(
            symbol=symbol,
            current_price=current_price,
            fibonacci=fib,
            pivots=pivots,
            all_resistances=all_res[:8],
            all_supports=all_sup[:8],
            nearest_resistance_pct=round(res_pct, 3),
            nearest_support_pct=round(sup_pct, 3),
            at_fib_resistance=bool(at_fib),
            at_pivot_resistance=bool(at_pvt),
            short_confluence=round(confluence, 3),
        )

    # ──────────────────────────────────────────────────────────────
    # FIBONACCI
    # ──────────────────────────────────────────────────────────────

    def _compute_fibonacci(
        self, df: pd.DataFrame, current_price: float, lookback: int
    ) -> Optional[FibonacciLevels]:
        """Find the most recent significant swing and compute Fib levels."""
        try:
            recent  = df.tail(lookback)
            sw_high = float(recent["high"].max())
            sw_low  = float(recent["low"].min())
            rng     = sw_high - sw_low

            if rng < 1e-6:
                return None

            # Determine direction: is price coming from a high or a low?
            high_idx = recent["high"].idxmax()
            low_idx  = recent["low"].idxmin()
            direction = "DOWN" if high_idx > low_idx else "UP"

            retracements = {}
            for fib in self.FIB_RETRACE:
                if direction == "DOWN":
                    # Price fell from high, retracements go UP from low
                    retracements[str(fib)] = round(sw_low + rng * fib, 2)
                else:
                    # Price rose from low, retracements go DOWN from high
                    retracements[str(fib)] = round(sw_high - rng * fib, 2)

            extensions = {}
            for ext in self.FIB_EXTEND:
                if direction == "DOWN":
                    extensions[str(ext)] = round(sw_low - rng * (ext - 1), 2)
                else:
                    extensions[str(ext)] = round(sw_high + rng * (ext - 1), 2)

            # Nearest resistance (levels above current price)
            all_levels = list(retracements.values()) + list(extensions.values())
            resistances = [l for l in all_levels if l > current_price * 1.001]
            supports    = [l for l in all_levels if l < current_price * 0.999]

            nearest_res = min(resistances) if resistances else None
            nearest_sup = max(supports)    if supports    else None

            # Determine price zone
            fib_values = sorted(retracements.values())
            zone = "BELOW_ALL"
            for i, fib_val in enumerate(fib_values):
                if current_price < fib_val:
                    zone = f"BELOW_{list(retracements.keys())[i]}"
                    break
            else:
                zone = "ABOVE_ALL"

            return FibonacciLevels(
                swing_high=round(sw_high, 2),
                swing_low=round(sw_low, 2),
                direction=direction,
                retracements=retracements,
                extensions=extensions,
                nearest_resistance=nearest_res,
                nearest_support=nearest_sup,
                current_zone=zone,
            )
        except Exception as e:
            logger.debug(f"Fibonacci error: {e}")
            return None

    # ──────────────────────────────────────────────────────────────
    # PIVOT POINTS
    # ──────────────────────────────────────────────────────────────

    def _compute_pivots(self, df: pd.DataFrame) -> Optional[PivotLevels]:
        """Compute classic and Camarilla pivot points from previous day's OHLC."""
        try:
            prev = df.iloc[-2]
            h, l, c = float(prev["high"]), float(prev["low"]), float(prev["close"])
            rng = h - l

            pp = (h + l + c) / 3

            # Classic pivots
            r1 = 2 * pp - l
            r2 = pp + rng
            r3 = h + 2 * (pp - l)
            s1 = 2 * pp - h
            s2 = pp - rng
            s3 = l - 2 * (h - pp)

            # Camarilla (tighter, preferred for intraday)
            cam_r1 = c + rng * 1.1 / 12
            cam_r2 = c + rng * 1.1 / 6
            cam_r3 = c + rng * 1.1 / 4
            cam_r4 = c + rng * 1.1 / 2
            cam_s1 = c - rng * 1.1 / 12
            cam_s2 = c - rng * 1.1 / 6
            cam_s3 = c - rng * 1.1 / 4
            cam_s4 = c - rng * 1.1 / 2

            current = float(df["close"].iloc[-1])
            all_res = sorted([r for r in [r1, r2, r3, cam_r1, cam_r2, cam_r3, cam_r4]
                              if r > current * 1.001])
            all_sup = sorted([s for s in [s1, s2, s3, cam_s1, cam_s2, cam_s3, cam_s4]
                              if s < current * 0.999], reverse=True)

            return PivotLevels(
                pivot=round(pp, 2),
                r1=round(r1,2), r2=round(r2,2), r3=round(r3,2),
                s1=round(s1,2), s2=round(s2,2), s3=round(s3,2),
                camarilla_r1=round(cam_r1,2), camarilla_r2=round(cam_r2,2),
                camarilla_r3=round(cam_r3,2), camarilla_r4=round(cam_r4,2),
                camarilla_s1=round(cam_s1,2), camarilla_s2=round(cam_s2,2),
                camarilla_s3=round(cam_s3,2), camarilla_s4=round(cam_s4,2),
                nearest_resistance=all_res[0] if all_res else None,
                nearest_support=all_sup[0]    if all_sup else None,
            )
        except Exception as e:
            logger.debug(f"Pivot error: {e}")
            return None

    def _empty(self, symbol: str) -> KeyPriceLevels:
        return KeyPriceLevels(
            symbol=symbol, current_price=0, fibonacci=None, pivots=None,
            all_resistances=[], all_supports=[], nearest_resistance_pct=99,
            nearest_support_pct=99, at_fib_resistance=False,
            at_pivot_resistance=False, short_confluence=0,
        )


# Singleton
fibo_pivot = FiboPivotCalculator()
