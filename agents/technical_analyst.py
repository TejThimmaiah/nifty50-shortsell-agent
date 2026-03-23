"""
Technical Analysis Engine
Calculates all indicators needed to identify short selling opportunities.
Uses ta library (free, PyPI) — no paid data feed required.
"""

import logging
import numpy as np
import pandas as pd
import ta
import ta.momentum
import ta.trend
import ta.volatility
import ta.volume
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from config import TRADING

logger = logging.getLogger(__name__)


def _get_adaptive_rsi_threshold() -> float:
    """Get the learned RSI threshold from adaptive config, fallback to static config."""
    try:
        from intelligence.adaptive_config import adaptive_config
        return adaptive_config.get_current().rsi_overbought
    except Exception:
        return TRADING.rsi_overbought


@dataclass
class TechnicalSignal:
    """
    Technical signal for a Nifty 50 stock.
    SHORT_ONLY — this system never generates LONG or BUY signals.
    """
    symbol:            str
    signal:            str     # "STRONG_SHORT" | "SHORT" | "INSUFFICIENT"
    confidence:        float   # 0.0 – 1.0 (short conviction)
    rsi:               float
    macd_histogram:    float
    bb_position:       float   # 0=lower band, 1=upper band (high = bearish)
    is_overbought:     bool
    bearish_divergence: bool
    at_resistance:     bool
    volume_confirms:   bool
    support_level:     float
    resistance_level:  float
    entry_price:       float
    stop_loss:         float   # ABOVE entry (short SL)
    target:            float   # BELOW entry (short target)
    reason:            str


def calculate_all(df: pd.DataFrame, symbol: str = "") -> Optional[TechnicalSignal]:
    """
    Master function — runs all indicators and returns a consolidated signal.
    df must have columns: open, high, low, close, volume (lowercase).
    """
    if df is None or len(df) < 30:
        logger.warning(f"Insufficient data for {symbol}: {len(df) if df is not None else 0} candles")
        return None

    df = df.copy()

    try:
        # ── RSI ─────────────────────────────────────────────
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=TRADING.rsi_period).rsi()

        # ── MACD ─────────────────────────────────────────────
        macd_ind = ta.trend.MACD(df["close"], window_fast=12, window_slow=26, window_sign=9)
        df["macd"]        = macd_ind.macd()
        df["macd_signal"] = macd_ind.macd_signal()
        df["macd_hist"]   = macd_ind.macd_diff()

        # ── Bollinger Bands ───────────────────────────────────
        bb_ind = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
        df["bb_upper"] = bb_ind.bollinger_hband()
        df["bb_lower"] = bb_ind.bollinger_lband()
        df["bb_mid"]   = bb_ind.bollinger_mavg()
        df["bb_pct"]   = bb_ind.bollinger_pband()

        # ── EMA ──────────────────────────────────────────────
        df["ema9"]  = ta.trend.EMAIndicator(df["close"], window=9).ema_indicator()
        df["ema21"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
        df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()

        # ── Volume MA ─────────────────────────────────────────
        df["vol_ma20"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_ma20"]

        # ── Stochastic ───────────────────────────────────────
        stoch_ind = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"], window=14, smooth_window=3)
        df["stoch_k"] = stoch_ind.stoch()
        df["stoch_d"] = stoch_ind.stoch_signal()

        # ── ATR (for SL calculation) ──────────────────────────
        df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()

        # ── Support & Resistance (pivot points) ──────────────
        support, resistance = _calculate_sr_levels(df)

        latest = df.iloc[-1]
        prev   = df.iloc[-2] if len(df) > 1 else latest

        # ────────────────────────────────────────────────────
        # SIGNAL LOGIC
        # ────────────────────────────────────────────────────
        rsi          = float(latest.get("rsi", 50))
        macd_hist    = float(latest.get("macd_hist", 0))
        bb_pct       = float(latest.get("bb_pct", 0.5))
        vol_ratio    = float(latest.get("vol_ratio", 1))
        stoch_k      = float(latest.get("stoch_k", 50))
        close        = float(latest["close"])
        atr          = float(latest.get("atr", close * 0.01))

        # Individual conditions — uses LEARNED RSI threshold
        rsi_threshold      = _get_adaptive_rsi_threshold()
        is_overbought      = bool(rsi >= rsi_threshold or stoch_k >= 80)
        bearish_divergence = bool(_check_bearish_divergence(df))
        at_resistance      = bool(resistance and (abs(close - resistance) / resistance) < 0.005)
        volume_confirms    = bool(vol_ratio >= TRADING.volume_multiplier)
        macd_turning_down  = bool(macd_hist < 0 or (float(prev.get("macd_hist", 0)) > 0 and macd_hist < 0))
        bb_extended        = bool(bb_pct >= 0.9)
        ema_downtrend      = bool(float(latest.get("ema9", close)) < float(latest.get("ema21", close)))

        # Score-based confidence (short bias only)
        short_score = sum([
            is_overbought     * 0.25,
            bearish_divergence * 0.20,
            at_resistance      * 0.20,
            volume_confirms    * 0.15,
            macd_turning_down  * 0.10,
            bb_extended        * 0.05,
            ema_downtrend      * 0.05,
        ])

        # SHORT ONLY — no LONG, no NEUTRAL
        # INSUFFICIENT means not enough short signal to act
        if   short_score >= 0.65: signal = "STRONG_SHORT"
        elif short_score >= 0.45: signal = "SHORT"
        else:                     signal = "INSUFFICIENT"

        # SL above entry (short), target below entry
        entry_price  = close
        sl_price     = round(entry_price * (1 + TRADING.stop_loss_pct / 100), 2)
        target_price = round(entry_price * (1 - TRADING.target_pct / 100), 2)

        # Build reason string
        reasons = []
        if is_overbought:      reasons.append(f"RSI={rsi:.1f} (overbought)")
        if bearish_divergence: reasons.append("bearish RSI divergence")
        if at_resistance:      reasons.append(f"at resistance ₹{resistance:.2f}")
        if volume_confirms:    reasons.append(f"vol {vol_ratio:.1f}x avg")
        if macd_turning_down:  reasons.append("MACD histogram turning negative")
        if bb_extended:        reasons.append(f"BB position {bb_pct:.2f} (upper band)")

        return TechnicalSignal(
            symbol=symbol,
            signal=signal,
            confidence=round(short_score, 3),
            rsi=rsi,
            macd_histogram=macd_hist,
            bb_position=bb_pct,
            is_overbought=is_overbought,
            bearish_divergence=bearish_divergence,
            at_resistance=at_resistance,
            volume_confirms=volume_confirms,
            support_level=support or 0,
            resistance_level=resistance or 0,
            entry_price=entry_price,
            stop_loss=sl_price,
            target=target_price,
            reason="; ".join(reasons) if reasons else "No clear signal",
        )

    except Exception as e:
        logger.error(f"Technical analysis error [{symbol}]: {e}", exc_info=True)
        return None


def _calculate_sr_levels(df: pd.DataFrame, lookback: int = 20) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate support and resistance using pivot highs/lows.
    Returns (support, resistance).
    """
    try:
        recent = df.tail(lookback)
        highs = recent["high"].values
        lows  = recent["low"].values

        # Find local pivot highs (resistance)
        resistance_levels = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                resistance_levels.append(highs[i])

        # Find local pivot lows (support)
        support_levels = []
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                support_levels.append(lows[i])

        close = float(df.iloc[-1]["close"])

        # Find nearest resistance ABOVE current price
        resistance = None
        if resistance_levels:
            above = [r for r in resistance_levels if r > close]
            resistance = min(above) if above else max(resistance_levels)

        # Find nearest support BELOW current price
        support = None
        if support_levels:
            below = [s for s in support_levels if s < close]
            support = max(below) if below else min(support_levels)

        return support, resistance

    except Exception as e:
        logger.error(f"S/R calculation error: {e}")
        return None, None


def _check_bearish_divergence(df: pd.DataFrame, lookback: int = 10) -> bool:
    """
    Check for bearish RSI divergence:
    Price makes higher high but RSI makes lower high → bearish.
    """
    try:
        recent = df.tail(lookback)
        prices = recent["close"].values
        rsis   = recent["rsi"].values

        if np.any(np.isnan(rsis)):
            return False

        # Split into two halves and compare
        mid = len(recent) // 2
        first_half_price_high = prices[:mid].max()
        second_half_price_high = prices[mid:].max()
        first_half_rsi_high = rsis[:mid].max()
        second_half_rsi_high = rsis[mid:].max()

        # Bearish divergence: price higher but RSI lower
        return (second_half_price_high > first_half_price_high and
                second_half_rsi_high < first_half_rsi_high)

    except Exception:
        return False


def get_market_breadth(nifty_data: Dict) -> str:
    """Assess market breadth for short selling favorability."""
    if not nifty_data:
        return "UNKNOWN"
    try:
        advances = int(nifty_data.get("advances", 0))
        declines = int(nifty_data.get("declines", 0))
        total = advances + declines
        if total == 0:
            return "NEUTRAL"
        decline_ratio = declines / total
        if decline_ratio > 0.65:
            return "BEARISH"      # Good for shorts
        elif decline_ratio > 0.50:
            return "SLIGHTLY_BEARISH"
        elif decline_ratio < 0.35:
            return "BULLISH"      # Avoid shorts
        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"
