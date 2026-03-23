"""
Screener.in Data Fetcher
Scrapes fundamental financial data from Screener.in (free, no API key).
Used to filter out stocks with strong fundamentals from short candidates.
We want to short weak companies, not strong ones going through a technical pullback.

Key filters for short candidates:
  - Promoter holding falling (insider selling signal)
  - High PE vs sector peers (overvalued)
  - Declining quarterly revenue/profit
  - High debt-to-equity
  - Negative cash flow
"""

import time
import logging
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from functools import lru_cache

logger = logging.getLogger(__name__)

BASE_URL = "https://www.screener.in"
HEADERS  = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept":     "text/html,application/xhtml+xml",
    "Referer":    "https://www.screener.in/",
}

_session = requests.Session()
_session.headers.update(HEADERS)


@dataclass
class FundamentalProfile:
    symbol: str
    company_name: str

    # Valuation
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    market_cap_cr: Optional[float] = None
    ev_ebitda: Optional[float] = None

    # Financial health
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    interest_coverage: Optional[float] = None

    # Quality
    roe: Optional[float] = None              # Return on Equity %
    roce: Optional[float] = None             # Return on Capital Employed %
    promoter_holding_pct: Optional[float] = None

    # Growth (QoQ)
    revenue_growth_pct: Optional[float] = None
    profit_growth_pct: Optional[float] = None

    # Short bias score (higher = weaker fundamentals = better short candidate)
    short_bias_score: float = 0.0
    short_bias_reason: List[str] = field(default_factory=list)

    # Raw data availability flag
    data_fetched: bool = False


def get_fundamental_profile(symbol: str) -> FundamentalProfile:
    """
    Fetch and parse fundamental data for a given NSE symbol from Screener.in.
    Returns a FundamentalProfile with a computed short_bias_score.
    """
    profile = FundamentalProfile(symbol=symbol, company_name=symbol)

    try:
        url  = f"{BASE_URL}/company/{symbol}/consolidated/"
        resp = _session.get(url, timeout=15)

        # Screener redirects to /standalone/ if no consolidated data
        if resp.status_code == 404:
            url  = f"{BASE_URL}/company/{symbol}/"
            resp = _session.get(url, timeout=15)

        if resp.status_code != 200:
            logger.warning(f"Screener.in returned {resp.status_code} for {symbol}")
            return profile

        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract company name
        name_tag = soup.find("h1", class_="h2")
        if name_tag:
            profile.company_name = name_tag.get_text(strip=True)

        # Parse key ratios section
        _parse_key_ratios(soup, profile)

        # Parse shareholding pattern
        _parse_shareholding(soup, profile)

        # Parse quarterly results for growth
        _parse_quarterly_growth(soup, profile)

        profile.data_fetched = True
        profile.short_bias_score, profile.short_bias_reason = _compute_short_bias(profile)

        logger.debug(
            f"Screener [{symbol}]: PE={profile.pe_ratio}, "
            f"D/E={profile.debt_to_equity}, "
            f"Promoter={profile.promoter_holding_pct}%, "
            f"ShortBias={profile.short_bias_score:.2f}"
        )

    except Exception as e:
        logger.warning(f"Screener.in fetch failed [{symbol}]: {e}")

    # Polite rate limit
    time.sleep(1.5)
    return profile


def get_multiple_profiles(symbols: List[str]) -> Dict[str, FundamentalProfile]:
    """Fetch profiles for multiple symbols with rate limiting."""
    results = {}
    for symbol in symbols:
        results[symbol] = get_fundamental_profile(symbol)
    return results


def screen_short_candidates(
    symbols: List[str],
    min_short_bias: float = 0.4,
) -> List[FundamentalProfile]:
    """
    Filter a list of symbols to keep only those with weak fundamentals
    that make strong short candidates.
    """
    profiles = get_multiple_profiles(symbols)
    candidates = [
        p for p in profiles.values()
        if p.short_bias_score >= min_short_bias
    ]
    candidates.sort(key=lambda x: x.short_bias_score, reverse=True)
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# PARSERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_key_ratios(soup: BeautifulSoup, profile: FundamentalProfile):
    """Parse the key ratios section from Screener.in."""
    # Screener.in renders ratios as a list of <li> elements
    ratio_section = soup.find("section", id="top-ratios")
    if not ratio_section:
        return

    for li in ratio_section.find_all("li"):
        label_tag = li.find("span", class_="name")
        value_tag = li.find("span", class_="number")
        if not label_tag or not value_tag:
            continue

        label = label_tag.get_text(strip=True).lower()
        raw   = value_tag.get_text(strip=True).replace(",", "").replace("₹", "").strip()

        try:
            val = _parse_float(raw)
            if val is None:
                continue

            if "stock p/e" in label or "p/e" in label:
                profile.pe_ratio = val
            elif "p/b" in label or "price to book" in label:
                profile.pb_ratio = val
            elif "market cap" in label:
                profile.market_cap_cr = val
            elif "debt / equity" in label or "d/e" in label:
                profile.debt_to_equity = val
            elif "current ratio" in label:
                profile.current_ratio = val
            elif "interest coverage" in label:
                profile.interest_coverage = val
            elif "roe" in label:
                profile.roe = val
            elif "roce" in label:
                profile.roce = val
        except Exception:
            continue


def _parse_shareholding(soup: BeautifulSoup, profile: FundamentalProfile):
    """Parse promoter holding from shareholding section."""
    try:
        # Screener shows shareholding in a table
        sh_section = soup.find("section", id="shareholding")
        if not sh_section:
            return

        # Find the promoter row
        for tr in sh_section.find_all("tr"):
            cells = tr.find_all("td")
            if not cells:
                continue
            label = cells[0].get_text(strip=True).lower()
            if "promoter" in label and len(cells) >= 2:
                # Latest quarter is the last column
                raw = cells[-1].get_text(strip=True).replace("%", "")
                val = _parse_float(raw)
                if val is not None:
                    profile.promoter_holding_pct = val
                    break
    except Exception as e:
        logger.debug(f"Shareholding parse error: {e}")


def _parse_quarterly_growth(soup: BeautifulSoup, profile: FundamentalProfile):
    """Parse revenue and profit growth from quarterly results."""
    try:
        qr_section = soup.find("section", id="quarters")
        if not qr_section:
            return

        rows = qr_section.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if not cells or len(cells) < 3:
                continue
            label = cells[0].get_text(strip=True).lower()

            if "sales" in label or "revenue" in label:
                # Compare last two quarters
                vals = [_parse_float(c.get_text(strip=True).replace(",", ""))
                        for c in cells[1:] if _parse_float(c.get_text(strip=True).replace(",", "")) is not None]
                if len(vals) >= 2:
                    prev, curr = vals[-2], vals[-1]
                    if prev and prev != 0:
                        profile.revenue_growth_pct = round((curr - prev) / abs(prev) * 100, 2)

            elif "net profit" in label or "profit" in label:
                vals = [_parse_float(c.get_text(strip=True).replace(",", ""))
                        for c in cells[1:] if _parse_float(c.get_text(strip=True).replace(",", "")) is not None]
                if len(vals) >= 2:
                    prev, curr = vals[-2], vals[-1]
                    if prev and prev != 0:
                        profile.profit_growth_pct = round((curr - prev) / abs(prev) * 100, 2)

    except Exception as e:
        logger.debug(f"Quarterly results parse error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SHORT BIAS SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _compute_short_bias(p: FundamentalProfile):
    """
    Score the stock for short bias based on fundamentals.
    Higher score = weaker fundamentals = better short candidate.
    Returns (score: float 0–1, reasons: list[str])
    """
    score   = 0.0
    reasons = []

    # High PE (overvalued relative to earnings)
    if p.pe_ratio is not None:
        if p.pe_ratio > 80:
            score += 0.25; reasons.append(f"Very high PE {p.pe_ratio:.0f}x")
        elif p.pe_ratio > 50:
            score += 0.15; reasons.append(f"High PE {p.pe_ratio:.0f}x")
        elif p.pe_ratio > 35:
            score += 0.08; reasons.append(f"Elevated PE {p.pe_ratio:.0f}x")

    # High debt
    if p.debt_to_equity is not None:
        if p.debt_to_equity > 2.0:
            score += 0.20; reasons.append(f"High D/E {p.debt_to_equity:.1f}x")
        elif p.debt_to_equity > 1.0:
            score += 0.10; reasons.append(f"Elevated D/E {p.debt_to_equity:.1f}x")

    # Declining revenue (weakening business)
    if p.revenue_growth_pct is not None and p.revenue_growth_pct < -5:
        score += 0.20; reasons.append(f"Revenue declining {p.revenue_growth_pct:.1f}%")
    elif p.revenue_growth_pct is not None and p.revenue_growth_pct < 0:
        score += 0.10; reasons.append(f"Revenue slightly negative {p.revenue_growth_pct:.1f}%")

    # Declining profit
    if p.profit_growth_pct is not None and p.profit_growth_pct < -10:
        score += 0.20; reasons.append(f"Profit declining {p.profit_growth_pct:.1f}%")
    elif p.profit_growth_pct is not None and p.profit_growth_pct < 0:
        score += 0.10; reasons.append(f"Profit slightly negative {p.profit_growth_pct:.1f}%")

    # Low promoter holding (insiders not confident)
    if p.promoter_holding_pct is not None:
        if p.promoter_holding_pct < 30:
            score += 0.15; reasons.append(f"Low promoter holding {p.promoter_holding_pct:.1f}%")
        elif p.promoter_holding_pct < 45:
            score += 0.07; reasons.append(f"Below-avg promoter holding {p.promoter_holding_pct:.1f}%")

    # Low ROE/ROCE (capital inefficiency)
    if p.roe is not None and p.roe < 8:
        score += 0.10; reasons.append(f"Weak ROE {p.roe:.1f}%")
    if p.roce is not None and p.roce < 8:
        score += 0.08; reasons.append(f"Weak ROCE {p.roce:.1f}%")

    return round(min(1.0, score), 3), reasons


def _parse_float(s: str) -> Optional[float]:
    """Safely parse a string to float, handling Cr/%, etc."""
    try:
        s = s.replace(",", "").replace("%", "").replace("Cr", "").strip()
        if s in ("-", "", "N/A", "—"):
            return None
        return float(s)
    except (ValueError, TypeError):
        return None
