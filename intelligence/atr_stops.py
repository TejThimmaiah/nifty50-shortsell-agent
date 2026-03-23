"""
ATR-Adaptive Stop Loss Engine
Fixed percentage stops (0.5%) are naive — they ignore volatility.
In high-volatility markets, a 0.5% stop gets hit by noise.
In low-volatility markets, 0.5% is too loose.

ATR (Average True Range) normalizes stops to actual price volatility:
  stop_distance = ATR × multiplier
  
When volatility doubles, stop widens.
When volatility halves, stop tightens.
The dollar risk stays proportional to actual market conditions.

Advanced features:
  1. Chandelier Exit  — trailing stop below highest high since entry
  2. Parabolic SAR    — accelerating stop that follows the trend
  3. Swing Low Stop   — place SL at nearest swing low above entry (for shorts)
  4. Vol-regime stop  — different multiplier for different VIX regimes

For shorts:
  SL is ABOVE entry. ATR determines how far above.
  Tighter in low-vol (0.8x ATR), wider in high-vol (1.5x ATR).
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ATRStopResult:
    entry_price:        float
    atr:                float
    atr_pct:            float           # ATR as % of price
    vol_regime:         str             # LOW | NORMAL | HIGH | EXTREME
    recommended_sl:     float           # ATR-based stop
    chandelier_sl:      Optional[float] # Chandelier exit level
    swing_sl:           Optional[float] # Nearest swing high (for shorts)
    parabolic_sl:       Optional[float] # Parabolic SAR level
    final_sl:           float           # The stop to actually use
    sl_pct_from_entry:  float           # final SL as % above entry
    target:             float           # ATR-based target
    risk_reward:        float


class ATRStopEngine:
    """
    Computes volatility-adjusted stop losses and targets.
    Replaces the fixed config.TRADING.stop_loss_pct.
    """

    # ATR multipliers by volatility regime
    MULTIPLIERS = {
        "LOW":     {"sl": 1.2, "target": 2.5, "chandelier": 2.0},
        "NORMAL":  {"sl": 1.5, "target": 3.0, "chandelier": 2.5},
        "HIGH":    {"sl": 2.0, "target": 3.5, "chandelier": 3.0},
        "EXTREME": {"sl": 2.5, "target": 4.0, "chandelier": 3.5},
    }

    # Hard limits — never exceed these regardless of ATR
    MAX_SL_PCT   = 1.5   # SL can't be more than 1.5% above entry
    MIN_SL_PCT   = 0.25  # SL must be at least 0.25% above entry
    MIN_RR_RATIO = 1.3   # target must be at least 1.3× the stop

    def compute(
        self,
        df: pd.DataFrame,
        entry_price: float,
        direction: str = "SHORT",
        vix: float = 15.0,
        atr_period: int = 14,
    ) -> ATRStopResult:
        """
        Compute full ATR-adaptive stop loss suite.
        df: OHLCV DataFrame (at least atr_period + 5 bars)
        """
        if df is None or len(df) < atr_period + 2:
            return self._fallback(entry_price)

        atr       = self._compute_atr(df, atr_period)
        atr_pct   = (atr / entry_price) * 100
        vol_regime= self._classify_volatility(atr_pct, vix)
        mults     = self.MULTIPLIERS[vol_regime]

        # Base ATR stop
        sl_dist  = atr * mults["sl"]
        tgt_dist = atr * mults["target"]

        if direction == "SHORT":
            atr_sl  = entry_price + sl_dist
            atr_tgt = entry_price - tgt_dist
        else:
            atr_sl  = entry_price - sl_dist
            atr_tgt = entry_price + tgt_dist

        # Chandelier exit (for shorts: highest high since entry + multiplier × ATR)
        chandelier = self._chandelier_stop(df, entry_price, atr, mults["chandelier"], direction)

        # Swing structure stop
        swing = self._swing_stop(df, entry_price, direction)

        # Parabolic SAR
        psar = self._parabolic_sar(df, direction)

        # Final stop: use the TIGHTEST of the valid stops
        # (for shorts: lowest value = tightest = closest to entry from above)
        candidates = [s for s in [atr_sl, chandelier, swing] if s is not None]
        if direction == "SHORT":
            final_sl = min(candidates) if candidates else atr_sl  # tightest = closest to entry
        else:
            final_sl = max(candidates) if candidates else atr_sl

        # Apply hard limits
        sl_pct   = abs(final_sl - entry_price) / entry_price * 100
        sl_pct   = max(self.MIN_SL_PCT, min(self.MAX_SL_PCT, sl_pct))
        if direction == "SHORT":
            final_sl = entry_price * (1 + sl_pct / 100)
        else:
            final_sl = entry_price * (1 - sl_pct / 100)

        # Ensure minimum R:R
        actual_risk   = abs(final_sl - entry_price)
        min_reward    = actual_risk * self.MIN_RR_RATIO
        if direction == "SHORT":
            final_tgt = min(atr_tgt, entry_price - min_reward)
        else:
            final_tgt = max(atr_tgt, entry_price + min_reward)

        rr = abs(entry_price - final_tgt) / max(actual_risk, 1e-6)

        logger.debug(
            f"ATR stop [{direction}]: entry={entry_price:.2f} "
            f"ATR={atr:.2f} ({atr_pct:.2f}%) "
            f"regime={vol_regime} SL={final_sl:.2f} TGT={final_tgt:.2f} R:R={rr:.2f}"
        )

        return ATRStopResult(
            entry_price=entry_price,
            atr=round(atr, 2),
            atr_pct=round(atr_pct, 3),
            vol_regime=vol_regime,
            recommended_sl=round(atr_sl, 2),
            chandelier_sl=round(chandelier, 2) if chandelier else None,
            swing_sl=round(swing, 2) if swing else None,
            parabolic_sl=round(psar, 2) if psar else None,
            final_sl=round(final_sl, 2),
            sl_pct_from_entry=round(sl_pct, 3),
            target=round(final_tgt, 2),
            risk_reward=round(rr, 2),
        )

    def get_trailing_stop(
        self,
        df: pd.DataFrame,
        entry_price: float,
        current_price: float,
        original_sl: float,
        direction: str = "SHORT",
        atr_period: int = 14,
    ) -> float:
        """
        Compute updated trailing stop based on current price action.
        For shorts: stop moves DOWN as price falls (locking in profit).
        Returns new SL — never worse than original.
        """
        if df is None or len(df) < atr_period:
            return original_sl

        atr    = self._compute_atr(df, atr_period)
        result = self.compute(df, current_price, direction)

        if direction == "SHORT":
            # For a short, SL is above entry. As price falls, trail SL down.
            profit_pct = (entry_price - current_price) / entry_price * 100
            if profit_pct >= 0.8:  # Start trailing at 0.8% profit
                new_sl = current_price + atr * 1.0   # tighter trailing
                return min(original_sl, max(new_sl, result.final_sl))
        return original_sl

    # ──────────────────────────────────────────────────────────────
    # INTERNAL
    # ──────────────────────────────────────────────────────────────

    def _compute_atr(self, df: pd.DataFrame, period: int) -> float:
        """Average True Range."""
        high  = df["high"].values.astype(float)
        low   = df["low"].values.astype(float)
        close = df["close"].values.astype(float)
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        hl   = high - low
        hc   = np.abs(high - prev_close)
        lc   = np.abs(low  - prev_close)
        tr   = np.maximum(hl, np.maximum(hc, lc))
        return float(np.mean(tr[-period:]))

    def _classify_volatility(self, atr_pct: float, vix: float) -> str:
        """Classify volatility regime from ATR% and VIX."""
        if atr_pct > 2.5 or vix > 25:   return "EXTREME"
        if atr_pct > 1.5 or vix > 18:   return "HIGH"
        if atr_pct > 0.8 or vix > 13:   return "NORMAL"
        return "LOW"

    def _chandelier_stop(
        self, df: pd.DataFrame, entry: float, atr: float, mult: float, direction: str
    ) -> Optional[float]:
        """Chandelier exit: highest high (for shorts) + ATR × multiplier."""
        lookback = min(10, len(df))
        recent   = df.tail(lookback)
        if direction == "SHORT":
            highest_high = float(recent["high"].max())
            return highest_high + atr * mult
        else:
            lowest_low = float(recent["low"].min())
            return lowest_low - atr * mult

    def _swing_stop(
        self, df: pd.DataFrame, entry: float, direction: str
    ) -> Optional[float]:
        """
        Place stop at nearest swing high above entry (for shorts).
        Swing high = bar whose high is higher than both neighbors.
        """
        if len(df) < 5:
            return None
        highs = df["high"].values.astype(float)
        for i in range(len(highs) - 2, 1, -1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                if direction == "SHORT" and highs[i] > entry:
                    return float(highs[i]) * 1.001  # small buffer
        return None

    def _parabolic_sar(self, df: pd.DataFrame, direction: str) -> Optional[float]:
        """Parabolic SAR — simple implementation."""
        if len(df) < 10:
            return None
        try:
            import pandas_ta as ta
            psar_df = ta.psar(df["high"], df["low"], df["close"])
            if psar_df is None or psar_df.empty:
                return None
            # Get the bullish or bearish SAR column
            col = [c for c in psar_df.columns if "PSARl" in c or "PSARs" in c]
            if not col:
                return None
            val = psar_df[col[0]].iloc[-1]
            return float(val) if not np.isnan(val) else None
        except Exception:
            return None

    def _fallback(self, entry_price: float) -> ATRStopResult:
        """Safe fallback when data is insufficient."""
        from config import TRADING
        sl_pct = TRADING.stop_loss_pct
        tgt_pct = TRADING.target_pct
        return ATRStopResult(
            entry_price=entry_price,
            atr=entry_price * 0.01,
            atr_pct=1.0,
            vol_regime="NORMAL",
            recommended_sl=round(entry_price * (1 + sl_pct / 100), 2),
            chandelier_sl=None,
            swing_sl=None,
            parabolic_sl=None,
            final_sl=round(entry_price * (1 + sl_pct / 100), 2),
            sl_pct_from_entry=sl_pct,
            target=round(entry_price * (1 - tgt_pct / 100), 2),
            risk_reward=tgt_pct / sl_pct,
        )


# Singleton
atr_stop_engine = ATRStopEngine()
