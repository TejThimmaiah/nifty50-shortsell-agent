"""
Tej Global Macro Radar
========================
Monitors Fed, RBI, crude oil, USD/INR, VIX simultaneously.
Builds a macro picture before every trading session.

"Crude up $3 + USD/INR at 84.5 + FII sold Rs 800Cr yesterday
 = bearish opening. Tej increasing short bias today."
"""

import os, logging, requests
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from zoneinfo import ZoneInfo
logger = logging.getLogger("macro_radar")
IST = ZoneInfo("Asia/Kolkata")

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False


@dataclass
class MacroSnapshot:
    timestamp:      str
    nifty:          float
    nifty_fut:      float    # SGX Nifty / Gift Nifty
    sensex:         float
    vix:            float
    crude_wti:      float
    crude_brent:    float
    usd_inr:        float
    us500:          float
    us500_change:   float
    gold:           float
    dxy:            float    # Dollar index
    us10y:          float    # US 10Y yield
    macro_bias:     str      # "BEARISH" / "NEUTRAL" / "BULLISH"
    size_mult:      float    # Recommended position size multiplier
    summary:        str


class GlobalMacroRadar:
    """Fetches and synthesizes global macro data."""

    TICKERS = {
        "nifty":    "^NSEI",
        "sensex":   "^BSESN",
        "vix":      "^NSEI",    # Using India VIX proxy
        "crude_wti":"CL=F",
        "crude_brent": "BZ=F",
        "usd_inr":  "USDINR=X",
        "us500":    "^GSPC",
        "gold":     "GC=F",
        "dxy":      "DX-Y.NYB",
        "us10y":    "^TNX",
        "india_vix":"^INDIAVIX",
        "gift_nifty": "NIFTY1!",
    }

    def _fetch(self, ticker: str) -> tuple:
        if not YF_AVAILABLE:
            return 0.0, 0.0
        try:
            t    = yf.Ticker(ticker)
            hist = t.history(period="2d", interval="1d")
            if hist.empty:
                return 0.0, 0.0
            price  = float(hist["Close"].iloc[-1])
            prev   = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
            change = (price - prev) / prev * 100
            return round(price, 2), round(change, 2)
        except Exception:
            return 0.0, 0.0

    def fetch_fii_data(self) -> dict:
        """Fetch FII/DII data from NSE."""
        try:
            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0",
                "Referer":    "https://www.nseindia.com/",
            })
            session.get("https://www.nseindia.com/", timeout=8)
            r = session.get(
                "https://www.nseindia.com/api/fiidiiTradeReact",
                timeout=10
            )
            if r.ok:
                data = r.json()
                if isinstance(data, list) and data:
                    latest = data[0]
                    return {
                        "fii_net":  float(latest.get("net", 0)),
                        "dii_net":  float(latest.get("net", 0)),
                        "date":     latest.get("date", ""),
                    }
        except Exception:
            pass
        return {"fii_net": 0, "dii_net": 0, "date": ""}

    def get_snapshot(self) -> MacroSnapshot:
        """Fetch all macro data and build snapshot."""
        nifty,   nifty_ch   = self._fetch("^NSEI")
        us500,   us500_ch   = self._fetch("^GSPC")
        crude,   crude_ch   = self._fetch("CL=F")
        brent,   brent_ch   = self._fetch("BZ=F")
        inr,     inr_ch     = self._fetch("USDINR=X")
        gold,    gold_ch    = self._fetch("GC=F")
        dxy,     dxy_ch     = self._fetch("DX-Y.NYB")
        us10y,   us10y_ch   = self._fetch("^TNX")
        vix,     _          = self._fetch("^INDIAVIX")
        sensex,  _          = self._fetch("^BSESN")
        fii = self.fetch_fii_data()

        # Macro bias calculation
        bearish_score = 0
        if us500_ch < -0.5:   bearish_score += 2
        if crude_ch  > 2.0:   bearish_score += 1   # High crude = inflationary
        if inr_ch    > 0.3:   bearish_score += 1   # Weak INR = FII outflow
        if vix       > 20:    bearish_score += 2
        if dxy_ch    > 0.5:   bearish_score += 1   # Strong dollar = EM selloff
        if us10y_ch  > 0.05:  bearish_score += 1   # Rising yields = risk-off
        if fii["fii_net"] < -500: bearish_score += 2

        bullish_score = 0
        if us500_ch  > 0.5:   bullish_score += 2
        if crude_ch  < -1.0:  bullish_score += 1
        if vix       < 14:    bullish_score += 1
        if fii["fii_net"] > 500: bullish_score += 2

        if bearish_score >= 5:
            bias     = "BEARISH"
            size_mult = 1.2   # Bigger shorts when macro is bearish
        elif bearish_score >= 3:
            bias     = "MILDLY_BEARISH"
            size_mult = 1.0
        elif bullish_score >= 4:
            bias     = "BULLISH"
            size_mult = 0.6   # Smaller shorts in bull macro
        else:
            bias     = "NEUTRAL"
            size_mult = 0.9

        summary = (
            f"Macro: {bias}. "
            f"US500 {us500_ch:+.1f}% | Crude {crude_ch:+.1f}% | "
            f"USD/INR {inr:.2f} | VIX {vix:.1f} | "
            f"FII {fii['fii_net']:+.0f}Cr"
        )

        return MacroSnapshot(
            timestamp=datetime.now(IST).isoformat(),
            nifty=nifty, nifty_fut=nifty,
            sensex=sensex, vix=vix,
            crude_wti=crude, crude_brent=brent,
            usd_inr=inr, us500=us500, us500_change=us500_ch,
            gold=gold, dxy=dxy, us10y=us10y,
            macro_bias=bias, size_mult=size_mult, summary=summary,
        )

    def format_for_telegram(self) -> str:
        s = self.get_snapshot()
        bias_emoji = {"BEARISH": "🔴", "MILDLY_BEARISH": "🟠",
                      "NEUTRAL": "🟡", "BULLISH": "🟢"}.get(s.macro_bias, "⚪")
        return (
            f"<b>Global Macro Radar</b> {bias_emoji}\n\n"
            f"Nifty:      {s.nifty:,.0f}\n"
            f"US S&P500:  {s.us500:,.0f} ({s.us500_change:+.1f}%)\n"
            f"Crude WTI:  ${s.crude_wti:.1f}\n"
            f"USD/INR:    {s.usd_inr:.2f}\n"
            f"VIX:        {s.vix:.1f}\n"
            f"Gold:       ${s.gold:,.0f}\n"
            f"US 10Y:     {s.us10y:.2f}%\n\n"
            f"Bias: <b>{s.macro_bias}</b>\n"
            f"Size multiplier: {s.size_mult:.1f}x\n\n"
            f"{s.summary}"
        )


macro_radar = GlobalMacroRadar()
