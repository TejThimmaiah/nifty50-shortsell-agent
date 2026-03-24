"""
Tej Predictive Alerts
=======================
Tej messages you BEFORE a setup forms.

"Watch HDFCBANK in 20 mins — RSI approaching 70, 
 volume building, sentiment deteriorating. 
 Getting ready to short."

Not reactive. Predictive.
"""

import os
import logging
import requests
import asyncio
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger("predictive_alerts")
IST = ZoneInfo("Asia/Kolkata")


@dataclass
class PendingSetup:
    symbol:          str
    alert_type:      str    # "RSI_APPROACHING" / "VOLUME_BUILDING" / "BREAKDOWN_IMMINENT"
    current_value:   float
    trigger_value:   float
    eta_minutes:     int
    confidence:      float
    message:         str


class PredictiveAlerts:
    """
    Monitors forming setups and alerts before they trigger.
    Runs continuously during market hours via WebSocket ticks.
    """

    RSI_ALERT_THRESHOLD      = 65.0   # Alert when RSI hits this (approaching 70)
    VOLUME_BUILD_THRESHOLD   = 1.3    # Alert when volume is 1.3x (approaching spike)
    BREAKDOWN_BUFFER_PCT     = 0.5    # Alert when price within 0.5% of support

    def __init__(self):
        self.token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.sent_alerts = {}   # symbol → last alert time (avoid spam)

    def _send(self, message: str):
        if not self.token or not self.chat_id:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"},
                timeout=10,
            )
        except Exception as e:
            logger.error(f"Send failed: {e}")

    def _can_alert(self, symbol: str, cooldown_minutes: int = 30) -> bool:
        """Prevent duplicate alerts within cooldown period."""
        now = datetime.now(IST)
        last = self.sent_alerts.get(symbol)
        if last and (now - last).seconds < cooldown_minutes * 60:
            return False
        return True

    def check_rsi_approaching(self, symbol: str, current_rsi: float,
                               rsi_velocity: float) -> Optional[PendingSetup]:
        """
        Alert when RSI is heading toward overbought (65→70).
        rsi_velocity: RSI points per candle
        """
        if current_rsi < self.RSI_ALERT_THRESHOLD:
            return None
        if rsi_velocity <= 0:   # RSI falling, no alert
            return None

        points_to_trigger = 70 - current_rsi
        if points_to_trigger <= 0:
            return None

        candles_needed = points_to_trigger / max(rsi_velocity, 0.1)
        eta_minutes    = int(candles_needed * 5)  # 5-min candles

        if 5 <= eta_minutes <= 45:
            return PendingSetup(
                symbol=symbol,
                alert_type="RSI_APPROACHING",
                current_value=current_rsi,
                trigger_value=70.0,
                eta_minutes=eta_minutes,
                confidence=0.70,
                message=(
                    f"⚡ <b>Watch {symbol} in ~{eta_minutes} mins</b>\n"
                    f"RSI at {current_rsi:.1f} and rising fast.\n"
                    f"Approaching overbought (70) — short setup forming.\n"
                    f"Get ready."
                )
            )
        return None

    def check_volume_building(self, symbol: str, current_vol_ratio: float,
                               vol_velocity: float) -> Optional[PendingSetup]:
        """Alert when volume is building toward a spike."""
        if current_vol_ratio < self.VOLUME_BUILD_THRESHOLD:
            return None
        if current_vol_ratio > 2.0:
            return None  # Already spiked, handled elsewhere

        eta_minutes = int((2.0 - current_vol_ratio) / max(vol_velocity, 0.01) * 5)

        if 5 <= eta_minutes <= 30:
            return PendingSetup(
                symbol=symbol,
                alert_type="VOLUME_BUILDING",
                current_value=current_vol_ratio,
                trigger_value=2.0,
                eta_minutes=eta_minutes,
                confidence=0.60,
                message=(
                    f"📊 <b>Volume building: {symbol}</b>\n"
                    f"Volume at {current_vol_ratio:.1f}x average and rising.\n"
                    f"Could trigger breakout/breakdown in ~{eta_minutes} mins.\n"
                    f"Watch for direction."
                )
            )
        return None

    def check_breakdown_imminent(self, symbol: str, current_price: float,
                                  support_level: float) -> Optional[PendingSetup]:
        """Alert when price approaching key support (breakdown imminent)."""
        if support_level <= 0:
            return None
        distance_pct = (current_price - support_level) / current_price * 100

        if 0 < distance_pct <= self.BREAKDOWN_BUFFER_PCT:
            return PendingSetup(
                symbol=symbol,
                alert_type="BREAKDOWN_IMMINENT",
                current_value=current_price,
                trigger_value=support_level,
                eta_minutes=10,
                confidence=0.75,
                message=(
                    f"🎯 <b>Breakdown imminent: {symbol}</b>\n"
                    f"Price at {current_price:.2f} — only {distance_pct:.2f}% above support {support_level:.2f}\n"
                    f"If support breaks → strong short entry.\n"
                    f"Set alert at {support_level * 0.999:.2f}"
                )
            )
        return None

    def process_tick(self, symbol: str, market_data: dict) -> List[PendingSetup]:
        """
        Process a single tick/candle and check all alert conditions.
        Called continuously during market hours.
        """
        alerts = []

        rsi     = market_data.get("rsi", 50)
        rsi_vel = market_data.get("rsi_velocity", 0)
        vol_rat = market_data.get("volume_ratio", 1)
        vol_vel = market_data.get("volume_velocity", 0)
        price   = market_data.get("close", 0)
        support = market_data.get("support_level", 0)

        checks = [
            self.check_rsi_approaching(symbol, rsi, rsi_vel),
            self.check_volume_building(symbol, vol_rat, vol_vel),
            self.check_breakdown_imminent(symbol, price, support),
        ]

        for setup in checks:
            if setup and self._can_alert(symbol):
                alerts.append(setup)
                self._send(setup.message)
                self.sent_alerts[symbol] = datetime.now(IST)
                logger.info(f"Alert sent: {symbol} — {setup.alert_type}")

        return alerts

    def market_open_brief(self, watchlist_setups: list):
        """Send morning brief with stocks to watch."""
        if not watchlist_setups:
            return
        msg = "<b>Morning Watch List</b>\n\n"
        msg += "Tej is watching these stocks for short setups today:\n\n"
        for sym, reason, eta in watchlist_setups[:5]:
            msg += f"• <b>{sym}</b> — {reason} (~{eta} mins)\n"
        msg += "\nI'll alert you when setups trigger."
        self._send(msg)


predictive_alerts = PredictiveAlerts()
