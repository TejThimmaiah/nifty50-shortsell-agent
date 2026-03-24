"""
Tej Global Contagion Detector
================================
When US sneezes, Tej already knows before India opens.

Monitors real-time correlation between global markets and Nifty.
Detects when normal correlations break (regime change signal).

"US futures down 1.2% at 8AM IST. Historically Nifty opens
 down 0.8-1.1% in this scenario. Increasing short bias."
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo
logger = logging.getLogger("contagion_detector")
IST = ZoneInfo("Asia/Kolkata")

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False


@dataclass
class ContagionSignal:
    global_move:    float    # % move in global markets
    predicted_nifty: float   # Predicted Nifty opening move
    confidence:     float
    signal:         str      # "BEARISH_OPEN" / "NEUTRAL" / "BULLISH_OPEN"
    correlation:    float    # Current correlation coefficient
    regime:         str      # "NORMAL_CORRELATION" / "DECOUPLED" / "AMPLIFIED"
    details:        Dict[str, float]


class GlobalContagionDetector:
    """
    Detects global market contagion and predicts Nifty impact.
    Uses rolling correlation to detect regime changes.
    """

    # Historical average correlations with Nifty
    BASE_CORRELATIONS = {
        "^GSPC":    0.72,    # S&P 500 — strongest influence
        "^IXIC":    0.68,    # NASDAQ
        "CL=F":     -0.35,   # Crude (inverse — high crude = import cost)
        "USDINR=X": -0.65,   # USD/INR (inverse)
        "GC=F":     0.15,    # Gold (mild positive)
        "^TNX":     -0.40,   # US 10Y yield (inverse)
        "DX-Y.NYB": -0.58,   # Dollar index (inverse)
    }

    IMPACT_MULTIPLIERS = {
        "^GSPC":    0.75,    # 1% S&P move = ~0.75% Nifty
        "^IXIC":    0.60,
        "CL=F":     -0.20,   # 1% crude up = -0.2% Nifty
        "USDINR=X": -1.20,   # 1% INR weaken = -1.2% Nifty
        "^TNX":     -0.30,
        "DX-Y.NYB": -0.50,
    }

    def _get_overnight_moves(self) -> Dict[str, float]:
        """Get overnight % moves in global markets."""
        moves = {}
        if not YF_AVAILABLE:
            return moves
        for ticker in self.BASE_CORRELATIONS:
            try:
                data = yf.download(ticker, period="3d", interval="1d", progress=False)
                if len(data) >= 2:
                    prev  = float(data["Close"].iloc[-2])
                    curr  = float(data["Close"].iloc[-1])
                    moves[ticker] = round((curr - prev) / prev * 100, 3)
            except Exception:
                pass
        return moves

    def _get_rolling_correlation(self, ticker: str, window: int = 30) -> float:
        """Calculate recent rolling correlation with Nifty."""
        if not YF_AVAILABLE:
            return self.BASE_CORRELATIONS.get(ticker, 0.5)
        try:
            nifty  = yf.download("^NSEI", period="3mo", interval="1d", progress=False)["Close"].pct_change().dropna()
            other  = yf.download(ticker, period="3mo", interval="1d", progress=False)["Close"].pct_change().dropna()
            df     = nifty.to_frame("nifty").join(other.to_frame("other")).dropna()
            if len(df) >= window:
                corr = float(df.tail(window).corr().iloc[0, 1])
                return round(corr, 3)
        except Exception:
            pass
        return self.BASE_CORRELATIONS.get(ticker, 0.5)

    def detect(self) -> ContagionSignal:
        """Full contagion detection."""
        overnight = self._get_overnight_moves()
        if not overnight:
            return ContagionSignal(
                global_move=0, predicted_nifty=0, confidence=0.3,
                signal="NEUTRAL", correlation=0.7,
                regime="NORMAL_CORRELATION", details={}
            )

        # Calculate weighted predicted Nifty impact
        weighted_impact = 0.0
        total_weight    = 0.0
        details         = {}

        for ticker, move in overnight.items():
            if ticker in self.IMPACT_MULTIPLIERS:
                mult     = self.IMPACT_MULTIPLIERS[ticker]
                impact   = move * mult
                weight   = abs(self.BASE_CORRELATIONS.get(ticker, 0.5))
                weighted_impact += impact * weight
                total_weight    += weight
                details[ticker]  = round(impact, 3)

        pred = weighted_impact / max(total_weight, 1e-9)

        # Detect regime
        sp500_move  = overnight.get("^GSPC", 0)
        rolling_corr = self._get_rolling_correlation("^GSPC")
        base_corr    = self.BASE_CORRELATIONS["^GSPC"]
        corr_change  = rolling_corr - base_corr

        if corr_change > 0.15:
            regime = "AMPLIFIED"     # Higher than usual correlation
        elif corr_change < -0.20:
            regime = "DECOUPLED"     # India ignoring global cues
        else:
            regime = "NORMAL_CORRELATION"

        # Signal
        if pred < -0.8:
            signal, conf = "BEARISH_OPEN", 0.80
        elif pred < -0.3:
            signal, conf = "MILDLY_BEARISH_OPEN", 0.65
        elif pred > 0.8:
            signal, conf = "BULLISH_OPEN", 0.75
        elif pred > 0.3:
            signal, conf = "MILDLY_BULLISH_OPEN", 0.60
        else:
            signal, conf = "NEUTRAL", 0.50

        global_move = overnight.get("^GSPC", 0)
        return ContagionSignal(
            global_move=round(global_move, 3),
            predicted_nifty=round(pred, 3),
            confidence=conf,
            signal=signal,
            correlation=rolling_corr,
            regime=regime,
            details=details,
        )

    def format_for_telegram(self) -> str:
        s = self.detect()
        emoji = "🔴" if s.predicted_nifty < -0.3 else ("🟢" if s.predicted_nifty > 0.3 else "🟡")
        return (
            f"<b>Global Contagion Radar</b>\n\n"
            f"US S&P500 overnight: {s.global_move:+.2f}%\n"
            f"Predicted Nifty impact: {s.predicted_nifty:+.2f}%\n"
            f"Correlation regime: {s.regime}\n"
            f"{emoji} Signal: {s.signal} ({s.confidence:.0%})\n\n"
            f"<b>Factor impacts:</b>\n" +
            "\n".join([f"• {k}: {v:+.2f}%" for k, v in s.details.items()])
        )


contagion_detector = GlobalContagionDetector()
