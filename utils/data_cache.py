"""
Data Cache Manager
In-memory + SQLite-backed cache for market data.
Prevents hitting NSE/yfinance rate limits by serving stale-but-valid data
when the same symbol is requested multiple times within the TTL window.

TTL rules:
  Live quotes    → 5 seconds  (can't be too stale intraday)
  Intraday OHLCV → 60 seconds (1-min candles update every minute)
  Daily OHLCV    → 3600 seconds (daily bars only change once per day)
  FII data       → 300 seconds (updates a few times per day)
  F&O list       → 86400 seconds (changes rarely)
  Screener       → 3600 seconds (fundamental data is slow-moving)
"""

import time
import logging
import threading
import hashlib
import json
import sqlite3
import os
from typing import Any, Optional, Dict, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Default TTLs in seconds
TTL = {
    "quote":       5,
    "intraday":    60,
    "daily":       3600,
    "fii_dii":     300,
    "fo_list":     86400,
    "screener":    3600,
    "oi":          120,
    "nifty":       10,
    "vix":         30,
    "sector":      1800,
    "options":     120,
    "holidays":    86400 * 30,
}

CACHE_DB = os.path.join(os.path.dirname(__file__), "..", "db", "data_cache.db")


class DataCache:
    """
    Two-level cache: L1 = in-memory dict (fastest), L2 = SQLite (survives restart).
    Automatically expires entries after TTL seconds.
    Thread-safe.
    """

    def __init__(self):
        self._l1: Dict[str, Tuple[Any, float]] = {}   # key → (value, expire_at)
        self._lock = threading.RLock()
        self._hits   = 0
        self._misses = 0
        self._init_db()
        self._cleanup_thread = threading.Thread(
            target=self._periodic_cleanup, daemon=True, name="cache-cleaner"
        )
        self._cleanup_thread.start()

    # ──────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached value. Returns None if missing or expired."""
        with self._lock:
            # L1 check
            if key in self._l1:
                value, expire_at = self._l1[key]
                if time.time() < expire_at:
                    self._hits += 1
                    return value
                else:
                    del self._l1[key]

            # L2 check (SQLite)
            row = self._db_get(key)
            if row:
                value, expire_at = row
                if time.time() < expire_at:
                    self._l1[key] = (value, expire_at)   # promote to L1
                    self._hits += 1
                    return value
                else:
                    self._db_delete(key)

            self._misses += 1
            return None

    def set(self, key: str, value: Any, ttl_category: str = "quote") -> None:
        """Store a value with a TTL."""
        ttl_sec    = TTL.get(ttl_category, 60)
        expire_at  = time.time() + ttl_sec

        with self._lock:
            self._l1[key] = (value, expire_at)
            # Persist to SQLite for cross-restart durability
            self._db_set(key, value, expire_at)

    def invalidate(self, key: str) -> None:
        """Remove a specific key from all cache levels."""
        with self._lock:
            self._l1.pop(key, None)
            self._db_delete(key)

    def invalidate_prefix(self, prefix: str) -> int:
        """Remove all keys starting with a prefix. Returns count removed."""
        count = 0
        with self._lock:
            to_del = [k for k in self._l1 if k.startswith(prefix)]
            for k in to_del:
                del self._l1[k]
                count += 1
            count += self._db_delete_prefix(prefix)
        return count

    def clear_all(self) -> None:
        """Wipe the entire cache (use carefully)."""
        with self._lock:
            self._l1.clear()
            self._db_clear()
        logger.info("Data cache cleared")

    def stats(self) -> Dict:
        """Return cache performance stats."""
        total = self._hits + self._misses
        return {
            "hits":      self._hits,
            "misses":    self._misses,
            "hit_rate":  round(self._hits / max(total, 1) * 100, 1),
            "l1_size":   len(self._l1),
        }

    # ──────────────────────────────────────────────────────────────
    # CONVENIENCE WRAPPERS
    # ──────────────────────────────────────────────────────────────

    def cached_call(self, key: str, fn, ttl_category: str = "quote") -> Any:
        """
        Get from cache or call fn() and cache the result.
        Usage:
            data = cache.cached_call(f"quote_{sym}", lambda: get_quote(sym), "quote")
        """
        cached = self.get(key)
        if cached is not None:
            return cached
        result = fn()
        if result is not None:
            self.set(key, result, ttl_category)
        return result

    # ──────────────────────────────────────────────────────────────
    # SQLITE BACKEND
    # ──────────────────────────────────────────────────────────────

    def _init_db(self):
        os.makedirs(os.path.dirname(CACHE_DB), exist_ok=True)
        with sqlite3.connect(CACHE_DB) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key       TEXT PRIMARY KEY,
                    value     TEXT NOT NULL,
                    expire_at REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expire ON cache(expire_at)")

    def _db_get(self, key: str) -> Optional[Tuple[Any, float]]:
        try:
            with sqlite3.connect(CACHE_DB) as conn:
                row = conn.execute(
                    "SELECT value, expire_at FROM cache WHERE key=?", (key,)
                ).fetchone()
            if row:
                return json.loads(row[0]), row[1]
        except Exception:
            pass
        return None

    def _db_set(self, key: str, value: Any, expire_at: float):
        try:
            with sqlite3.connect(CACHE_DB) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cache (key, value, expire_at) VALUES (?,?,?)",
                    (key, json.dumps(value, default=str), expire_at)
                )
        except Exception as e:
            logger.debug(f"Cache DB write error: {e}")

    def _db_delete(self, key: str):
        try:
            with sqlite3.connect(CACHE_DB) as conn:
                conn.execute("DELETE FROM cache WHERE key=?", (key,))
        except Exception:
            pass

    def _db_delete_prefix(self, prefix: str) -> int:
        try:
            with sqlite3.connect(CACHE_DB) as conn:
                cur = conn.execute("DELETE FROM cache WHERE key LIKE ?", (prefix + "%",))
                return cur.rowcount
        except Exception:
            return 0

    def _db_clear(self):
        try:
            with sqlite3.connect(CACHE_DB) as conn:
                conn.execute("DELETE FROM cache")
        except Exception:
            pass

    def _periodic_cleanup(self):
        """Background thread: purge expired entries every 5 minutes."""
        while True:
            time.sleep(300)
            try:
                now = time.time()
                with self._lock:
                    expired_l1 = [k for k, (_, exp) in self._l1.items() if exp < now]
                    for k in expired_l1:
                        del self._l1[k]
                with sqlite3.connect(CACHE_DB) as conn:
                    cur = conn.execute("DELETE FROM cache WHERE expire_at < ?", (now,))
                    if cur.rowcount:
                        logger.debug(f"Cache cleanup: removed {cur.rowcount} expired entries")
            except Exception as e:
                logger.debug(f"Cache cleanup error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Singleton — import and use this everywhere
# ─────────────────────────────────────────────────────────────────────────────
cache = DataCache()


# ─────────────────────────────────────────────────────────────────────────────
# Patching helpers — wrap NSE fetcher calls with caching transparently
# ─────────────────────────────────────────────────────────────────────────────

def cached_quote(symbol: str) -> Optional[dict]:
    from data.nse_fetcher import get_quote
    return cache.cached_call(f"quote:{symbol}", lambda: get_quote(symbol), "quote")


def cached_intraday(symbol: str, interval: str = "5m") -> Optional[object]:
    from data.nse_fetcher import get_intraday_ohlcv
    key = f"intraday:{symbol}:{interval}"
    return cache.cached_call(key, lambda: get_intraday_ohlcv(symbol, interval), "intraday")


def cached_daily(symbol: str, days: int = 60) -> Optional[object]:
    from data.nse_fetcher import get_historical_ohlcv
    key = f"daily:{symbol}:{days}"
    return cache.cached_call(key, lambda: get_historical_ohlcv(symbol, days), "daily")


def cached_fii_dii() -> dict:
    from data.nse_fetcher import get_fii_dii_data
    return cache.cached_call("fii_dii", get_fii_dii_data, "fii_dii") or {}


def cached_nifty() -> dict:
    from data.nse_fetcher import get_nifty_index
    return cache.cached_call("nifty_index", get_nifty_index, "nifty") or {}


def cached_fo_stocks() -> list:
    from data.nse_fetcher import get_fo_stocks
    return cache.cached_call("fo_stocks", get_fo_stocks, "fo_list") or []
