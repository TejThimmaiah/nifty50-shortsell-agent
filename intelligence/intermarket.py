"""
Intermarket Analysis
Markets don't trade in isolation. NSE moves are foreshadowed by:

1. SGX Nifty (Singapore futures) — leads NSE by ~45 minutes
   If SGX Nifty is down 1%+ at 8:30 AM, NSE will likely open weak
   = excellent environment for short selling

2. USD/INR — strong dollar = weak Indian markets (FII outflows)
   USD/INR rising >0.5% pre-market = bearish for Nifty

3. Crude Oil — India is a major importer
   Crude down >2% = positive for India (lower inflation, better CAD)
   Crude up >2% = negative for India (oil marketing companies, etc.)

4. US markets (Dow/S&P 500 futures) — global risk sentiment
   US futures down = Indian markets likely to follow

5. Asia (Nikkei, Hang Seng) — regional contagion
   Asian sell-off = NSE likely to suffer

All data fetched from Yahoo Finance — free, no API key.
This module gives the agent a 45-minute head start every morning.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")


@dataclass
class IntermarketBias:
    """Aggregated directional bias from all intermarket indicators."""
    bias:             str       # "BEARISH" | "SLIGHTLY_BEARISH" | "NEUTRAL" | "BULLISH"
    bias_score:       float     # -1.0 to +1.0 (negative = bearish for NSE)
    confidence:       float     # 0–1
    good_for_shorts:  bool

    # Individual readings
    sgx_nifty_chg:   Optional[float] = None
    usd_inr_chg:     Optional[float] = None
    crude_chg:        Optional[float] = None
    dow_futures_chg:  Optional[float] = None
    nikkei_chg:       Optional[float] = None
    hang_seng_chg:    Optional[float] = None
    vix_level:        Optional[float] = None

    signals:          List[str] = field(default_factory=list)
    fetched_at:       str = ""


class IntermarketAnalyser:
    """
    Fetches and interprets intermarket data every morning before NSE opens.
    Gives directional bias for the entire trading day.
    """

    def get_morning_bias(self) -> IntermarketBias:
        """
        Fetch all intermarket data and compute NSE directional bias.
        Call this at 8:50 AM IST for pre-market intelligence.
        """
        data = self._fetch_all()
        return self._compute_bias(data)

    def _fetch_all(self) -> Dict[str, Optional[float]]:
        """Fetch % changes from Yahoo Finance for all intermarket indicators."""
        import yfinance as yf

        tickers = {
            "sgx_nifty":    "^NSEI",        # Use Nifty as proxy for SGX timing
            "usd_inr":      "USDINR=X",
            "crude":        "CL=F",
            "dow_futures":  "YM=F",
            "sp500_futures":"ES=F",
            "nikkei":       "^N225",
            "hang_seng":    "^HSI",
            "vix":          "^VIX",
        }

        changes = {}
        for name, ticker in tickers.items():
            try:
                df = yf.download(
                    ticker, period="2d", interval="1d",
                    progress=False, auto_adjust=True
                )
                if df is None or len(df) < 2:
                    changes[name] = None
                    continue
                prev  = float(df["Close"].iloc[-2])
                last  = float(df["Close"].iloc[-1])
                changes[name] = round((last - prev) / prev * 100, 3)
            except Exception as e:
                logger.debug(f"Intermarket fetch error [{ticker}]: {e}")
                changes[name] = None

        return changes

    def _compute_bias(self, data: Dict) -> IntermarketBias:
        """Synthesise all intermarket signals into a single directional bias."""
        score    = 0.0
        signals  = []
        weights  = []

        # SGX Nifty (highest predictive power for NSE direction)
        sgx = data.get("sgx_nifty")
        if sgx is not None:
            contrib = -sgx * 0.35   # negative because SGX down = bearish for NSE
            score  += contrib
            weights.append(0.35)
            if sgx < -0.8:
                signals.append(f"SGX/Nifty futures {sgx:.2f}% (bearish lead)")
            elif sgx > 0.8:
                signals.append(f"SGX/Nifty futures {sgx:.2f}% (bullish lead)")

        # USD/INR — rupee weakening = capital outflows = bearish Nifty
        usd_inr = data.get("usd_inr")
        if usd_inr is not None:
            contrib = usd_inr * 0.20   # USD rising (USDINR rising) = bearish NSE
            score  += contrib
            weights.append(0.20)
            if usd_inr > 0.4:
                signals.append(f"USD/INR +{usd_inr:.2f}% (rupee weak → FII outflows)")

        # Crude oil — lower crude = good for NSE
        crude = data.get("crude")
        if crude is not None:
            contrib = crude * 0.15    # crude rising = bad for India
            score  += contrib
            weights.append(0.15)
            if crude > 2.0:
                signals.append(f"Crude +{crude:.2f}% (inflation concern)")
            elif crude < -2.0:
                signals.append(f"Crude {crude:.2f}% (import cost relief — positive)")

        # Dow futures
        dow = data.get("dow_futures")
        if dow is not None:
            contrib = -dow * 0.15
            score  += contrib
            weights.append(0.15)
            if dow < -0.8:
                signals.append(f"Dow futures {dow:.2f}% (global risk-off)")

        # Asia (Nikkei, Hang Seng)
        nikkei     = data.get("nikkei")
        hang_seng  = data.get("hang_seng")
        asia_avg   = None
        if nikkei is not None and hang_seng is not None:
            asia_avg   = (nikkei + hang_seng) / 2
            contrib    = -asia_avg * 0.15
            score     += contrib
            weights.append(0.15)
            if asia_avg < -1.0:
                signals.append(f"Asia selloff (Nikkei {nikkei:.1f}%, HSI {hang_seng:.1f}%)")

        # Normalise score based on actual data received
        if weights:
            score = score / sum(weights) * sum(weights)   # already weighted

        score = round(max(-1.0, min(1.0, score)), 3)

        # VIX
        vix = data.get("vix")

        # Classification
        if   score <= -0.40: bias = "BEARISH"
        elif score <= -0.15: bias = "SLIGHTLY_BEARISH"
        elif score >=  0.40: bias = "BULLISH"
        elif score >=  0.15: bias = "SLIGHTLY_BULLISH"
        else:                bias = "NEUTRAL"

        good_for_shorts = score <= -0.15 and bias in ("BEARISH", "SLIGHTLY_BEARISH")
        confidence      = min(1.0, len([w for w in weights if w]) / 5 * 0.8 + 0.2)

        return IntermarketBias(
            bias=bias,
            bias_score=score,
            confidence=round(confidence, 3),
            good_for_shorts=good_for_shorts,
            sgx_nifty_chg=sgx,
            usd_inr_chg=usd_inr,
            crude_chg=crude,
            dow_futures_chg=dow,
            nikkei_chg=nikkei,
            hang_seng_chg=hang_seng,
            vix_level=vix,
            signals=signals,
            fetched_at=datetime.now(IST).strftime("%H:%M IST"),
        )

    def get_sector_impact(self, crude_chg: Optional[float]) -> Dict[str, str]:
        """
        Map crude oil change to sector impact.
        High crude = bad for aviation, paints, chemicals, OMCs.
        Low crude = bad for upstream oil companies.
        """
        if crude_chg is None:
            return {}
        if crude_chg > 2:
            return {
                "avoid_shorting": ["ONGC", "COALINDIA"],   # benefit from high crude
                "prefer_shorting": ["INDIGO", "SPICEJET", "ASIANPAINT", "BPCL", "IOC"],
            }
        elif crude_chg < -2:
            return {
                "avoid_shorting": ["BPCL", "IOC", "HPCL", "INDIGO"],
                "prefer_shorting": ["ONGC", "OILIND"],
            }
        return {}


# Singleton
intermarket_analyser = IntermarketAnalyser()
