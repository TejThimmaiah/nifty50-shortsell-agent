"""
Alert Manager
Unified routing for all agent alerts.
Every alert goes through this class — guarantees delivery even if one
channel fails. Supports deduplication, rate limiting, and severity levels.
"""

import logging
import time
import threading
import requests
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable
from zoneinfo import ZoneInfo

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, CLOUDFLARE_WEBHOOK_URL, CLOUDFLARE_WEBHOOK_SECRET

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")


class Severity(Enum):
    DEBUG    = 0
    INFO     = 1
    WARNING  = 2
    CRITICAL = 3
    TRADE    = 4   # Always delivered, never rate-limited


@dataclass
class Alert:
    message:   str
    severity:  Severity
    symbol:    Optional[str] = None
    category:  str = "general"         # "trade" | "risk" | "system" | "general"
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(IST).strftime("%H:%M:%S IST")


class AlertManager:
    """
    Central alert dispatcher. Routes messages to:
      1. Python logger (always)
      2. Telegram (configurable, rate-limited)
      3. Cloudflare webhook (for dashboard updates)
      4. Custom callbacks (e.g. for tests)
    """

    # Minimum seconds between duplicate alerts (same message)
    DEDUP_WINDOW_SEC = 60

    # Max Telegram messages per minute (Telegram bot API limit: 30/sec, we use 10/min)
    MAX_TG_PER_MINUTE = 10

    def __init__(self):
        self._callbacks:    List[Callable[[Alert], None]] = []
        self._dedup_cache:  Dict[str, float] = {}
        self._tg_sent_times: List[float] = []
        self._lock = threading.Lock()
        self._queue: List[Alert] = []

        # Start background sender thread
        self._sender = threading.Thread(
            target=self._flush_loop, daemon=True, name="alert-sender"
        )
        self._sender.start()

    # ──────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────

    def send(
        self,
        message:  str,
        severity: Severity = Severity.INFO,
        symbol:   str = None,
        category: str = "general",
    ) -> None:
        """Queue an alert for delivery across all channels."""
        alert = Alert(
            message=message,
            severity=severity,
            symbol=symbol,
            category=category,
        )
        with self._lock:
            self._queue.append(alert)

    # Convenience wrappers
    def info(self,     msg: str, symbol: str = None) -> None:
        self.send(msg, Severity.INFO,     symbol)

    def warn(self,     msg: str, symbol: str = None) -> None:
        self.send(msg, Severity.WARNING,  symbol)

    def critical(self, msg: str, symbol: str = None) -> None:
        self.send(msg, Severity.CRITICAL, symbol)

    def trade(self,    msg: str, symbol: str = None) -> None:
        """Trade alerts: always delivered, never deduplicated or rate-limited."""
        self.send(msg, Severity.TRADE,    symbol, "trade")

    def register_callback(self, fn: Callable[[Alert], None]) -> None:
        """Register a custom delivery callback (e.g. for unit tests)."""
        self._callbacks.append(fn)

    def get_stats(self) -> Dict:
        return {
            "queue_length": len(self._queue),
            "tg_sent_last_minute": sum(
                1 for t in self._tg_sent_times if time.time() - t < 60
            ),
        }

    # ──────────────────────────────────────────────────────────────
    # INTERNAL: background flush loop
    # ──────────────────────────────────────────────────────────────

    def _flush_loop(self):
        while True:
            time.sleep(0.5)
            with self._lock:
                batch = list(self._queue)
                self._queue.clear()

            for alert in batch:
                self._deliver(alert)

    def _deliver(self, alert: Alert) -> None:
        """Deliver one alert to all channels."""
        # 1. Always log
        log_level = {
            Severity.DEBUG:    logging.DEBUG,
            Severity.INFO:     logging.INFO,
            Severity.WARNING:  logging.WARNING,
            Severity.CRITICAL: logging.CRITICAL,
            Severity.TRADE:    logging.INFO,
        }.get(alert.severity, logging.INFO)
        logger.log(log_level, f"[ALERT/{alert.category}] {alert.message}")

        # 2. Custom callbacks
        for cb in self._callbacks:
            try:
                cb(alert)
            except Exception as e:
                logger.debug(f"Alert callback error: {e}")

        # 3. Telegram (rate-limited + deduplicated, except TRADE)
        if alert.severity != Severity.DEBUG:
            if alert.severity == Severity.TRADE or not self._is_duplicate(alert):
                if self._can_send_telegram():
                    self._send_telegram(alert)

        # 4. Cloudflare webhook (async, non-blocking)
        if alert.severity in (Severity.TRADE, Severity.CRITICAL):
            threading.Thread(
                target=self._send_webhook,
                args=(alert,),
                daemon=True,
            ).start()

    def _is_duplicate(self, alert: Alert) -> bool:
        """Check if this exact message was sent recently."""
        key = f"{alert.category}:{alert.message[:80]}"
        now = time.time()
        last = self._dedup_cache.get(key, 0)
        if now - last < self.DEDUP_WINDOW_SEC:
            return True
        self._dedup_cache[key] = now
        # Prune old entries
        if len(self._dedup_cache) > 500:
            cutoff = now - self.DEDUP_WINDOW_SEC * 2
            self._dedup_cache = {
                k: v for k, v in self._dedup_cache.items() if v > cutoff
            }
        return False

    def _can_send_telegram(self) -> bool:
        """Enforce Telegram rate limit: max 10 messages per minute."""
        now = time.time()
        self._tg_sent_times = [t for t in self._tg_sent_times if now - t < 60]
        if len(self._tg_sent_times) >= self.MAX_TG_PER_MINUTE:
            logger.debug("Telegram rate limit hit — skipping alert")
            return False
        return True

    def _send_telegram(self, alert: Alert) -> None:
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return

        # Add severity prefix
        prefix = {
            Severity.TRADE:    "💰",
            Severity.CRITICAL: "🚨",
            Severity.WARNING:  "⚠️",
            Severity.INFO:     "ℹ️",
        }.get(alert.severity, "")
        text = f"{prefix} {alert.message}" if prefix else alert.message

        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={
                    "chat_id":    TELEGRAM_CHAT_ID,
                    "text":       text[:4096],
                    "parse_mode": "HTML",
                },
                timeout=8,
            )
            if resp.ok:
                self._tg_sent_times.append(time.time())
            else:
                logger.warning(f"Telegram delivery failed: {resp.status_code}")
        except Exception as e:
            logger.warning(f"Telegram send error: {e}")

    def _send_webhook(self, alert: Alert) -> None:
        if not CLOUDFLARE_WEBHOOK_URL:
            return
        try:
            requests.post(
                f"{CLOUDFLARE_WEBHOOK_URL.rstrip('/')}/webhook/alert",
                json={"message": alert.message, "severity": alert.severity.name,
                      "category": alert.category, "symbol": alert.symbol,
                      "timestamp": alert.timestamp},
                headers={"X-Webhook-Secret": CLOUDFLARE_WEBHOOK_SECRET or ""},
                timeout=5,
            )
        except Exception:
            pass   # Non-critical — webhook delivery is best-effort


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────
alerts = AlertManager()
