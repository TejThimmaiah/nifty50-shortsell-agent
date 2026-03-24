"""
Tej Working Memory — Redis
============================
Holds full current day context in memory.
News, positions, P&L, sentiment, macro — all instantly accessible.

Like a trader's short-term memory during the trading day.
Cleared at EOD. Rebuilt next morning.
"""

import os, json, logging
from datetime import datetime, date
from typing import Any, Optional
from zoneinfo import ZoneInfo
logger = logging.getLogger("working_memory")
IST = ZoneInfo("Asia/Kolkata")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


class WorkingMemory:
    """
    Fast in-memory context store for trading day.
    Falls back to dict if Redis not available.
    """

    def __init__(self):
        self._r    = None
        self._cache = {}   # Fallback dict
        self._connect()

    def _connect(self):
        if not REDIS_AVAILABLE:
            logger.info("Redis not available — using in-memory dict fallback")
            return
        try:
            import redis
            self._r = redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=3)
            self._r.ping()
            logger.info("Redis connected")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e} — using dict fallback")
            self._r = None

    def _key(self, namespace: str, key: str) -> str:
        today = date.today().isoformat()
        return f"tej:{today}:{namespace}:{key}"

    def set(self, namespace: str, key: str, value: Any, ttl: int = 86400):
        """Store value with optional TTL (default 24h)."""
        raw = json.dumps(value) if not isinstance(value, str) else value
        if self._r:
            try:
                self._r.setex(self._key(namespace, key), ttl, raw)
                return
            except Exception:
                pass
        self._cache[self._key(namespace, key)] = raw

    def get(self, namespace: str, key: str) -> Optional[Any]:
        """Get value."""
        k = self._key(namespace, key)
        raw = None
        if self._r:
            try:
                raw = self._r.get(k)
            except Exception:
                pass
        if raw is None:
            raw = self._cache.get(k)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return raw

    def append_list(self, namespace: str, key: str, value: Any, max_len: int = 100):
        """Append to a list."""
        k = self._key(namespace, key)
        raw = json.dumps(value)
        if self._r:
            try:
                self._r.lpush(k, raw)
                self._r.ltrim(k, 0, max_len - 1)
                self._r.expire(k, 86400)
                return
            except Exception:
                pass
        lst = self._cache.get(k, [])
        if isinstance(lst, str):
            try:
                lst = json.loads(lst)
            except Exception:
                lst = []
        lst.insert(0, raw)
        self._cache[k] = lst[:max_len]

    def get_list(self, namespace: str, key: str) -> list:
        """Get list."""
        k = self._key(namespace, key)
        if self._r:
            try:
                items = self._r.lrange(k, 0, -1)
                return [json.loads(i) for i in items]
            except Exception:
                pass
        lst = self._cache.get(k, [])
        if isinstance(lst, list):
            result = []
            for i in lst:
                try:
                    result.append(json.loads(i) if isinstance(i, str) else i)
                except Exception:
                    pass
            return result
        return []

    # ── High-level helpers ──────────────────────────────────────

    def store_position(self, symbol: str, position: dict):
        self.set("positions", symbol, position)

    def get_position(self, symbol: str) -> Optional[dict]:
        return self.get("positions", symbol)

    def get_all_positions(self) -> dict:
        prefix = f"tej:{date.today().isoformat()}:positions:"
        positions = {}
        if self._r:
            try:
                keys = self._r.keys(f"{prefix}*")
                for k in keys:
                    sym = k.split(":")[-1]
                    val = self._r.get(k)
                    if val:
                        positions[sym] = json.loads(val)
                return positions
            except Exception:
                pass
        for k, v in self._cache.items():
            if ":positions:" in k:
                sym = k.split(":")[-1]
                try:
                    positions[sym] = json.loads(v) if isinstance(v, str) else v
                except Exception:
                    pass
        return positions

    def store_sentiment(self, symbol: str, result: dict):
        self.set("sentiment", symbol, result, ttl=3600)

    def get_sentiment(self, symbol: str) -> Optional[dict]:
        return self.get("sentiment", symbol)

    def store_macro(self, snapshot: dict):
        self.set("macro", "latest", snapshot, ttl=1800)

    def get_macro(self) -> Optional[dict]:
        return self.get("macro", "latest")

    def log_trade(self, trade: dict):
        self.append_list("trades", "today", trade)

    def get_today_trades(self) -> list:
        return self.get_list("trades", "today")

    def set_daily_pnl(self, pnl: float):
        self.set("pnl", "today", pnl)

    def get_daily_pnl(self) -> float:
        return float(self.get("pnl", "today") or 0)

    def get_context_summary(self) -> str:
        """Full day context as text for LLM prompts."""
        macro     = self.get_macro() or {}
        positions = self.get_all_positions()
        trades    = self.get_today_trades()
        pnl       = self.get_daily_pnl()

        return (
            f"Today: {date.today()}\n"
            f"Macro: {macro.get('macro_bias','unknown')} | {macro.get('summary','')}\n"
            f"Open positions: {list(positions.keys())}\n"
            f"Trades today: {len(trades)}\n"
            f"Day P&L so far: Rs {pnl:+,.0f}"
        )

    def clear_day(self):
        """Clear working memory at EOD."""
        prefix = f"tej:{date.today().isoformat()}:"
        if self._r:
            try:
                keys = self._r.keys(f"{prefix}*")
                if keys:
                    self._r.delete(*keys)
                logger.info(f"Cleared {len(keys)} Redis keys for today")
                return
            except Exception:
                pass
        keys_to_del = [k for k in self._cache if k.startswith(prefix)]
        for k in keys_to_del:
            del self._cache[k]
        logger.info(f"Cleared {len(keys_to_del)} memory keys for today")


working_memory = WorkingMemory()
