"""
NSE Market Calendar
Tracks exchange holidays, special trading sessions, and market timing.
Self-updates by fetching the NSE holiday list at the start of each month.

TRADING DAYS:
  Monday – Friday only (NSE is closed Saturday & Sunday always)
  EXCEPT on official NSE holidays listed below

The agent ONLY trades on actual NSE trading days.
GitHub Actions cron fires Mon–Fri but the agent's _is_market_day()
check inside brain/orchestrator.py blocks trading on holidays.
"""

import json
import logging
import os
import requests
from datetime import date, datetime, timedelta
from typing import List, Optional, Set
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "db", "nse_holidays.json")

# ─────────────────────────────────────────────────────────────────────────────
# OFFICIAL NSE EQUITY MARKET HOLIDAYS (Capital Market segment)
# Source: https://www.nseindia.com/products-services/equity-market-holidays
# ─────────────────────────────────────────────────────────────────────────────

NSE_HOLIDAYS_2025 = {
    "2025-01-26",  # Republic Day
    "2025-02-26",  # Mahashivratri
    "2025-03-14",  # Holi
    "2025-04-14",  # Dr. Ambedkar Jayanti
    "2025-04-18",  # Good Friday
    "2025-05-01",  # Maharashtra Day / Labour Day
    "2025-08-15",  # Independence Day
    "2025-08-27",  # Ganesh Chaturthi
    "2025-10-02",  # Gandhi Jayanti / Dussehra
    "2025-10-24",  # Dussehra
    "2025-11-05",  # Diwali – Laxmi Pujan (Muhurat trading only)
    "2025-11-14",  # Gurunanak Jayanti
    "2025-12-25",  # Christmas
}

# 2026 NSE holidays (official list — auto-refreshed from NSE API monthly)
NSE_HOLIDAYS_2026 = {
    "2026-01-26",  # Republic Day
    "2026-02-26",  # Mahashivratri
    "2026-03-20",  # Holi (Friday)
    "2026-04-03",  # Good Friday
    "2026-04-14",  # Dr. Ambedkar Jayanti
    "2026-05-01",  # Maharashtra Day
    "2026-06-11",  # Bakri Id (Id-ul-Adha) — tentative, confirm via NSE
    "2026-08-15",  # Independence Day (Saturday — observed Friday 14th if applicable)
    "2026-09-17",  # Ganesh Chaturthi (Thursday)
    "2026-10-02",  # Gandhi Jayanti (Friday)
    "2026-10-13",  # Dussehra (Tuesday) — tentative
    "2026-10-28",  # Diwali – Laxmi Pujan — tentative
    "2026-11-03",  # Gurunanak Jayanti — tentative
    "2026-12-25",  # Christmas
}

ALL_HARDCODED_HOLIDAYS: Set[str] = NSE_HOLIDAYS_2025 | NSE_HOLIDAYS_2026


class MarketCalendar:
    """
    Determines whether NSE equity market is open on a given day and time.

    The agent ONLY executes trades on NSE trading days:
      ✅ Monday – Friday
      ❌ Saturday, Sunday — always closed
      ❌ Official NSE holidays — listed above
      ❌ Any day NSE declares a special holiday (auto-refreshed from NSE API)
    """

    def __init__(self):
        self._holidays: Set[str] = set()
        self._load_holidays()

    def is_trading_day(self, check_date: Optional[date] = None) -> bool:
        """Returns True if NSE equity market is open for trading on this date."""
        d = check_date or date.today()

        # ── Rule 1: Weekends are ALWAYS closed ────────────────────────────────
        if d.weekday() >= 5:   # 5=Saturday, 6=Sunday
            logger.debug(f"{d} is a weekend — NSE closed")
            return False

        # ── Rule 2: NSE official holidays ─────────────────────────────────────
        if d.isoformat() in self._holidays:
            logger.info(f"{d} is an NSE holiday — agent will NOT trade today")
            return False

        return True

    def is_market_open(self, check_dt: Optional[datetime] = None) -> bool:
        """Returns True if NSE equity market is currently open (9:15–15:30 IST)."""
        now = check_dt or datetime.now(IST)
        if not self.is_trading_day(now.date()):
            return False
        market_open  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return market_open <= now <= market_close

    def is_in_trading_window(self, check_dt: Optional[datetime] = None) -> bool:
        """
        Returns True if within the safe window for NEW short entries.
        9:20 AM – 1:00 PM IST only.
        No new entries after 1 PM (too close to 3:10 PM square-off).
        """
        now = check_dt or datetime.now(IST)
        if not self.is_trading_day(now.date()):
            return False
        scan_start = now.replace(hour=9,  minute=20, second=0, microsecond=0)
        entry_cut  = now.replace(hour=13, minute=0,  second=0, microsecond=0)
        return scan_start <= now <= entry_cut

    def minutes_to_open(self) -> Optional[int]:
        now = datetime.now(IST)
        if self.is_market_open(now):
            return 0
        if not self.is_trading_day():
            return None
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        if now < market_open:
            return int((market_open - now).total_seconds() / 60)
        return None

    def minutes_to_close(self) -> Optional[int]:
        now = datetime.now(IST)
        if not self.is_market_open(now):
            return None
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return int((market_close - now).total_seconds() / 60)

    def next_trading_day(self) -> date:
        d = date.today() + timedelta(days=1)
        while not self.is_trading_day(d):
            d += timedelta(days=1)
        return d

    def get_upcoming_holidays(self, days: int = 30) -> List[str]:
        start = date.today()
        return [
            (start + timedelta(days=i)).isoformat()
            for i in range(days)
            if (start + timedelta(days=i)).isoformat() in self._holidays
        ]

    def refresh_holidays(self):
        """Fetch latest holiday list from NSE API. Called monthly."""
        logger.info("Refreshing NSE holiday list from NSE API...")
        holidays = self._fetch_nse_holidays()
        if holidays:
            self._holidays = ALL_HARDCODED_HOLIDAYS | holidays
            self._save_cache(holidays)
            logger.info(f"NSE holiday list refreshed: {len(self._holidays)} holidays loaded")
        else:
            logger.warning("Could not fetch NSE holidays — using hardcoded list as fallback")

    # ──────────────────────────────────────────────────────────────
    # INTERNAL
    # ──────────────────────────────────────────────────────────────

    def _load_holidays(self):
        cached = self._load_cache()
        self._holidays = ALL_HARDCODED_HOLIDAYS.copy()
        if cached:
            self._holidays.update(cached)
        else:
            fetched = self._fetch_nse_holidays()
            if fetched:
                self._holidays.update(fetched)
                self._save_cache(fetched)

    def _fetch_nse_holidays(self) -> Optional[Set[str]]:
        try:
            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer":    "https://www.nseindia.com/",
                "Accept":     "application/json",
            })
            session.get("https://www.nseindia.com", timeout=10)
            resp = session.get(
                "https://www.nseindia.com/api/holiday-master?type=trading",
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            holidays = set()
            for entry in data.get("CM", []):
                trade_date = entry.get("tradingDate", "")
                if trade_date:
                    try:
                        d = datetime.strptime(trade_date, "%d-%b-%Y").date()
                        holidays.add(d.isoformat())
                    except ValueError:
                        pass
            return holidays if holidays else None
        except Exception as e:
            logger.warning(f"NSE holiday fetch failed: {e}")
            return None

    def _load_cache(self) -> Optional[Set[str]]:
        try:
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE) as f:
                    data = json.load(f)
                return set(data.get("holidays", []))
        except Exception:
            pass
        return None

    def _save_cache(self, holidays: Set[str]):
        try:
            os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
            with open(CACHE_FILE, "w") as f:
                json.dump({
                    "holidays":   sorted(holidays),
                    "updated_at": date.today().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Holiday cache save failed: {e}")


# Singleton
calendar = MarketCalendar()
