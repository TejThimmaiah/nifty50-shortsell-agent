"""
Tej Insider Filing Tracker
============================
Monitors SEBI insider trading disclosures.
When promoters sell heavily — it's a bearish signal.
When FIIs dump — Tej knows before most retail traders.

Data source: NSE bulk/block deals + SEBI filings (free public data)
"""

import logging
import requests
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger("insider_tracker")
IST = ZoneInfo("Asia/Kolkata")

NSE_BASE = "https://www.nseindia.com"
HEADERS  = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept":     "application/json",
    "Referer":    "https://www.nseindia.com/",
}


@dataclass
class InsiderActivity:
    symbol:        str
    entity:        str    # Who bought/sold
    entity_type:   str    # "PROMOTER" / "FII" / "DII" / "INSIDER"
    action:        str    # "BUY" / "SELL"
    quantity:      float
    value_cr:      float  # Value in crores
    date:          str
    signal:        str    # "BEARISH" / "BULLISH" / "NEUTRAL"


@dataclass
class InsiderReport:
    symbol:        str
    total_selling: float   # Crores sold
    total_buying:  float   # Crores bought
    net_flow:      float   # Net (negative = net selling)
    activities:    List[InsiderActivity]
    signal:        str
    confidence:    float
    summary:       str


class InsiderTracker:
    """
    Tracks institutional and promoter activity on NSE stocks.
    Uses NSE public bulk/block deal data.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._init_session()

    def _init_session(self):
        try:
            self.session.get(f"{NSE_BASE}/", timeout=10)
        except Exception:
            pass

    def get_bulk_deals(self, symbol: str, days: int = 5) -> List[dict]:
        """Fetch bulk deals from NSE for a symbol."""
        try:
            r = self.session.get(
                f"{NSE_BASE}/api/bulk-deal-archives",
                params={"symbol": symbol.upper(), "series": "EQ"},
                timeout=15
            )
            if r.ok:
                data = r.json()
                deals = data if isinstance(data, list) else data.get("data", [])
                cutoff = datetime.now(IST).date() - timedelta(days=days)
                recent = []
                for d in deals:
                    try:
                        deal_date = datetime.strptime(
                            d.get("BD_DT_DATE", "01-Jan-2020"), "%d-%b-%Y"
                        ).date()
                        if deal_date >= cutoff:
                            recent.append(d)
                    except Exception:
                        pass
                return recent
        except Exception as e:
            logger.warning(f"Bulk deals fetch failed for {symbol}: {e}")
        return []

    def get_block_deals(self, symbol: str, days: int = 5) -> List[dict]:
        """Fetch block deals from NSE."""
        try:
            r = self.session.get(
                f"{NSE_BASE}/api/block-deal-archives",
                params={"symbol": symbol.upper()},
                timeout=15
            )
            if r.ok:
                data = r.json()
                return data if isinstance(data, list) else data.get("data", [])
        except Exception as e:
            logger.warning(f"Block deals fetch failed for {symbol}: {e}")
        return []

    def _classify_entity(self, client_name: str) -> str:
        """Classify entity type from name."""
        name_upper = client_name.upper()
        if any(w in name_upper for w in ["PROMOTER", "FOUNDER", "DIRECTOR", "MD ", "CEO"]):
            return "PROMOTER"
        elif any(w in name_upper for w in ["FII", "FPI", "FOREIGN", "GLOBAL", "INTERNATIONAL",
                                            "CAPITAL LLC", "FUND LP", "PARTNERS LP"]):
            return "FII"
        elif any(w in name_upper for w in ["LIC", "SBI", "HDFC MF", "ICICI MF", "NIPPON",
                                            "KOTAK MF", "AXIS MF", "MUTUAL FUND", "MF "]):
            return "DII"
        else:
            return "INSTITUTIONAL"

    def analyze(self, symbol: str, days: int = 10) -> InsiderReport:
        """Full insider activity analysis."""
        bulk_deals  = self.get_bulk_deals(symbol, days)
        block_deals = self.get_block_deals(symbol, days)
        all_deals   = bulk_deals + block_deals

        activities    = []
        total_selling = 0.0
        total_buying  = 0.0

        for deal in all_deals:
            try:
                client   = deal.get("BD_CLIENT_NAME", deal.get("CLIENT_NAME", "Unknown"))
                qty      = float(deal.get("BD_QTY_TRD", deal.get("QUANTITY", 0)) or 0)
                price    = float(deal.get("BD_TP_WATP", deal.get("PRICE", 0)) or 0)
                value_cr = (qty * price) / 1e7
                action   = "BUY" if deal.get("BD_BUY_SELL", "B") == "B" else "SELL"
                etype    = self._classify_entity(client)
                dt       = deal.get("BD_DT_DATE", deal.get("DATE", ""))

                activity = InsiderActivity(
                    symbol=symbol, entity=client, entity_type=etype,
                    action=action, quantity=qty, value_cr=value_cr,
                    date=dt, signal="BEARISH" if action == "SELL" else "BULLISH"
                )
                activities.append(activity)

                if action == "SELL":
                    total_selling += value_cr
                else:
                    total_buying += value_cr
            except Exception:
                pass

        net_flow = total_buying - total_selling

        # Generate signal
        if not activities:
            signal, confidence, summary = "NO_DATA", 0.0, "No bulk/block deals found."
        elif net_flow < -50:  # Heavy net selling > 50 Cr
            signal     = "STRONG_BEARISH"
            confidence = 0.80
            summary    = f"Heavy insider selling: Rs {abs(net_flow):.0f} Cr net sold in {days} days"
        elif net_flow < -10:
            signal     = "BEARISH"
            confidence = 0.65
            summary    = f"Net selling: Rs {abs(net_flow):.0f} Cr in {days} days"
        elif net_flow > 50:
            signal     = "STRONG_BULLISH"
            confidence = 0.75
            summary    = f"Heavy buying: Rs {net_flow:.0f} Cr in {days} days — avoid shorting"
        elif net_flow > 10:
            signal     = "BULLISH"
            confidence = 0.60
            summary    = f"Net buying: Rs {net_flow:.0f} Cr — weak short signal"
        else:
            signal     = "NEUTRAL"
            confidence = 0.50
            summary    = f"Minimal activity: net flow Rs {net_flow:.0f} Cr"

        return InsiderReport(
            symbol=symbol, total_selling=total_selling,
            total_buying=total_buying, net_flow=net_flow,
            activities=activities[:10], signal=signal,
            confidence=confidence, summary=summary,
        )

    def format_for_telegram(self, symbol: str) -> str:
        """Format as Telegram message."""
        r    = self.analyze(symbol)
        emoji = {
            "STRONG_BEARISH": "🔴🔴", "BEARISH": "🔴",
            "NEUTRAL": "🟡", "BULLISH": "🟢", "STRONG_BULLISH": "🟢🟢",
            "NO_DATA": "⚪"
        }.get(r.signal, "⚪")

        msg = (
            f"<b>Insider Activity: {symbol}</b>\n\n"
            f"{emoji} {r.signal} ({r.confidence:.0%})\n"
            f"Buying: Rs {r.total_buying:.1f} Cr\n"
            f"Selling: Rs {r.total_selling:.1f} Cr\n"
            f"Net Flow: Rs {r.net_flow:+.1f} Cr\n\n"
            f"{r.summary}"
        )
        if r.activities:
            msg += "\n\n<b>Recent deals:</b>\n"
            for a in r.activities[:4]:
                e = "🔴" if a.action == "SELL" else "🟢"
                msg += f"{e} {a.entity[:25]} {a.action} Rs {a.value_cr:.1f}Cr\n"
        return msg


insider_tracker = InsiderTracker()
