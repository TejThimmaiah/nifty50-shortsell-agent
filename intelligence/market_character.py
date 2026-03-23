"""
Mean Reversion vs Trend Following Classifier
Markets alternate between two regimes:
  1. TRENDING    — momentum works, mean reversion fails
  2. MEAN_REVERTING — mean reversion works, momentum fails

Using the wrong strategy in the wrong regime is expensive.

Detection methods:
  1. Hurst Exponent (H):
       H > 0.5 → trending (persistent)
       H < 0.5 → mean-reverting (anti-persistent)
       H = 0.5 → random walk

  2. Autocorrelation:
       Positive autocorrelation → momentum / trend
       Negative autocorrelation → mean-reversion

  3. Variance Ratio Test (Lo & MacKinlay):
       VR > 1 → trending
       VR < 1 → mean-reverting

  4. ADX (Average Directional Index):
       ADX > 25 → trending
       ADX < 20 → ranging/mean-reverting

Implications for short selling:
  TRENDING_DOWN     → use momentum shorts (ride the trend)
  MEAN_REVERTING    → use RSI extremes + VWAP rejection (quick exits)
  RANDOM_WALK       → reduce position size, need stronger signals
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MarketCharacter:
    symbol:            str
    hurst_exponent:    float      # 0–1 (0.5 = random walk)
    autocorrelation:   float      # lag-1 autocorrelation of returns
    variance_ratio:    float      # VR(2) — 2-period variance ratio
    adx:               float      # ADX value
    regime:            str        # TRENDING | MEAN_REVERTING | RANDOM_WALK
    confidence:        float      # 0–1
    strategy_mode:     str        # MOMENTUM | REVERSION | SELECTIVE
    sl_multiplier:     float      # adjust SL width for this regime
    target_multiplier: float      # adjust target for this regime
    hold_duration:     str        # SHORT (exit same day) | MEDIUM | LONG
    description:       str


class MarketCharacterClassifier:
    """
    Classifies market character to determine the appropriate trading strategy.
    Called every morning and updated every 30 minutes.
    """

    def classify(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        lookback: int = 50,
    ) -> MarketCharacter:
        """
        Classify market character from price history.
        df: daily or intraday OHLCV, at least 30 bars.
        """
        if df is None or len(df) < 20:
            return self._default(symbol)

        df = df.tail(lookback).copy()
        closes  = df["close"].values.astype(float)
        returns = np.diff(np.log(closes))

        if len(returns) < 10:
            return self._default(symbol)

        # 1. Hurst Exponent (simplified R/S analysis)
        hurst  = self._hurst_exponent(closes)

        # 2. Lag-1 autocorrelation of returns
        acf_1  = self._autocorrelation(returns, lag=1)

        # 3. Variance Ratio
        vr     = self._variance_ratio(returns, period=2)

        # 4. ADX
        adx    = self._compute_adx(df)

        # Vote-based classification
        trending_votes    = 0
        mean_rev_votes    = 0

        if hurst > 0.55:   trending_votes  += 2
        elif hurst < 0.45: mean_rev_votes  += 2
        else:              pass  # 0.45–0.55 = ambiguous

        if acf_1 > 0.10:   trending_votes  += 1
        elif acf_1 < -0.10: mean_rev_votes += 1

        if vr > 1.05:      trending_votes  += 1
        elif vr < 0.95:    mean_rev_votes  += 1

        if adx > 25:       trending_votes  += 2
        elif adx < 18:     mean_rev_votes  += 2

        total = trending_votes + mean_rev_votes
        if total == 0:
            regime = "RANDOM_WALK"
            confidence = 0.40
        elif trending_votes > mean_rev_votes * 1.5:
            regime     = "TRENDING"
            confidence = min(0.90, 0.50 + trending_votes / total * 0.40)
        elif mean_rev_votes > trending_votes * 1.5:
            regime     = "MEAN_REVERTING"
            confidence = min(0.90, 0.50 + mean_rev_votes / total * 0.40)
        else:
            regime     = "RANDOM_WALK"
            confidence = 0.45

        # Strategy implications
        if regime == "TRENDING":
            strategy_mode  = "MOMENTUM"
            sl_mult        = 1.3     # Wider stop — don't get shaken out
            tgt_mult       = 1.5     # Wider target — let the trend run
            hold_duration  = "MEDIUM"
            desc           = f"Trending market (H={hurst:.2f}, ADX={adx:.1f}). Use momentum shorts, wider targets."
        elif regime == "MEAN_REVERTING":
            strategy_mode  = "REVERSION"
            sl_mult        = 0.8     # Tighter stop
            tgt_mult       = 0.7     # Quick profit-taking
            hold_duration  = "SHORT"
            desc           = f"Mean-reverting market (H={hurst:.2f}, ACF={acf_1:.2f}). Quick exits at RSI extremes."
        else:
            strategy_mode  = "SELECTIVE"
            sl_mult        = 1.0
            tgt_mult       = 1.0
            hold_duration  = "SHORT"
            desc           = f"Random walk (H={hurst:.2f}). Reduce size, require stronger signals."

        return MarketCharacter(
            symbol=symbol,
            hurst_exponent=round(hurst, 4),
            autocorrelation=round(acf_1, 4),
            variance_ratio=round(vr, 4),
            adx=round(adx, 2),
            regime=regime,
            confidence=round(confidence, 3),
            strategy_mode=strategy_mode,
            sl_multiplier=sl_mult,
            target_multiplier=tgt_mult,
            hold_duration=hold_duration,
            description=desc,
        )

    # ──────────────────────────────────────────────────────────────
    # STATISTICAL MEASURES
    # ──────────────────────────────────────────────────────────────

    def _hurst_exponent(self, series: np.ndarray) -> float:
        """
        Simplified Hurst Exponent via R/S analysis.
        Values: H > 0.5 = trending, H < 0.5 = mean-reverting.
        """
        try:
            n = len(series)
            if n < 20:
                return 0.5

            # R/S at different lags
            lags = [max(4, n // 8), max(8, n // 4), max(16, n // 2)]
            rs_values = []
            for lag in lags:
                if lag >= n:
                    continue
                chunk = series[-lag:]
                mean  = np.mean(chunk)
                std   = np.std(chunk)
                if std < 1e-10:
                    continue
                cumdev = np.cumsum(chunk - mean)
                r_s    = (cumdev.max() - cumdev.min()) / std
                if r_s > 0:
                    rs_values.append((np.log(lag), np.log(r_s)))

            if len(rs_values) < 2:
                return 0.5

            x = np.array([v[0] for v in rs_values])
            y = np.array([v[1] for v in rs_values])
            h = np.polyfit(x, y, 1)[0]
            return float(np.clip(h, 0.1, 0.9))

        except Exception:
            return 0.5

    def _autocorrelation(self, returns: np.ndarray, lag: int = 1) -> float:
        """Lag-N autocorrelation of returns."""
        try:
            if len(returns) < lag + 5:
                return 0.0
            r1 = returns[:-lag]
            r2 = returns[lag:]
            corr = np.corrcoef(r1, r2)[0, 1]
            return float(corr) if not np.isnan(corr) else 0.0
        except Exception:
            return 0.0

    def _variance_ratio(self, returns: np.ndarray, period: int = 2) -> float:
        """
        Variance Ratio test (Lo & MacKinlay 1988).
        VR(k) = Var(k-period returns) / (k × Var(1-period returns))
        VR > 1 = momentum, VR < 1 = mean-reversion.
        """
        try:
            if len(returns) < period * 4:
                return 1.0
            var1 = np.var(returns)
            if var1 < 1e-10:
                return 1.0
            # k-period returns
            k_returns = np.array([
                sum(returns[i:i+period])
                for i in range(0, len(returns) - period + 1, 1)
            ])
            vark = np.var(k_returns)
            return float(vark / (period * var1))
        except Exception:
            return 1.0

    def _compute_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """ADX — measures trend strength regardless of direction."""
        try:
            import pandas_ta as ta
            adx = ta.adx(df["high"], df["low"], df["close"], length=period)
            if adx is None:
                return 20.0
            col = [c for c in adx.columns if c.startswith("ADX_")]
            if col:
                return float(adx[col[0]].iloc[-1])
            return 20.0
        except Exception:
            return 20.0

    def _default(self, symbol: str) -> MarketCharacter:
        return MarketCharacter(
            symbol=symbol,
            hurst_exponent=0.5,
            autocorrelation=0.0,
            variance_ratio=1.0,
            adx=20.0,
            regime="RANDOM_WALK",
            confidence=0.4,
            strategy_mode="SELECTIVE",
            sl_multiplier=1.0,
            target_multiplier=1.0,
            hold_duration="SHORT",
            description="Insufficient data — using defaults.",
        )


# Singleton
character_classifier = MarketCharacterClassifier()
