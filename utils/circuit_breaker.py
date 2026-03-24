"""
Circuit Breaker — Emergency Stop System
Monitors real-time P&L and market conditions.
Automatically halts trading only when conditions make SHORT SELLING DANGEROUS.

SHORT-SELLING SPECIFIC LOGIC:
  Nifty falling is NORMAL and GOOD — that's why we short sell.
  We only halt on:
    - Extreme crash >4%: individual stocks hit lower circuit breakers
      (you can't cover a short on a stock that's locked at lower circuit)
    - VIX > 30: spreads widen dramatically, slippage kills edge
    - Our own P&L losses (we're wrong on direction, something structural changed)

  We do NOT halt when:
    - Nifty falls 1–4%: this is the ideal short-selling environment
    - Market is bearish: bearish = good for shorts
    - FII selling: selling pressure = good for shorts
"""

import logging
import threading
import time
from datetime import datetime, date
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo
from config import TRADING

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")


@dataclass
class CircuitBreakerState:
    triggered: bool = False
    trigger_reason: str = ""
    trigger_time: Optional[str] = None
    auto_reset_after_minutes: int = 0
    consecutive_losses: int = 0
    total_loss_today: float = 0.0
    checks_run: int = 0


class CircuitBreaker:
    """
    Multi-level circuit breaker for a SHORT-SELLING agent.

    Level 1 — Soft halt:     3 consecutive losing trades → pause 30 min
    Level 2 — Daily soft:    Daily loss > 3% capital → stop for the day
    Level 3 — Daily hard:    Daily loss > 5% capital → stop + notify
    Level 4 — Extreme crash: Nifty down >4% → halt (lower circuit risk on stocks)
    Level 5 — VIX spike:     VIX > 30 → halt (spreads too wide, slippage kills edge)
    Level 6 — Flash loss:    Single position loses > 2% capital instantly → close all

    NOT a circuit breaker:
      ✅ Nifty down 0.5–4%  → IDEAL environment, keep trading
      ✅ Bearish breadth     → keep trading (we're shorting)
      ✅ FII selling         → keep trading (tailwind for shorts)
    """

    LEVELS = {
        "CONSECUTIVE_LOSS": {"threshold": 3,    "pause_min": 30,  "description": "3 consecutive losses — re-evaluating"},
        "DAILY_LOSS_SOFT":  {"threshold": 0.03, "pause_min": 999, "description": "Daily loss >3% — stop for today"},
        "DAILY_LOSS_HARD":  {"threshold": 0.05, "pause_min": 999, "description": "Daily loss >5% — hard stop"},
        "EXTREME_CRASH":    {"threshold": 0.04, "pause_min": 60,  "description": "Nifty down >4% — lower circuit risk"},
        "VIX_SPIKE":        {"threshold": 30,   "pause_min": 60,  "description": "VIX > 30 — spreads too wide"},
        "FLASH_LOSS":       {"threshold": 0.02, "pause_min": 30,  "description": "Flash position loss >2%"},
        "MAX_SLIPPAGE":     {"threshold": 0.01, "pause_min": 15,  "description": "Slippage >1% — liquidity issue"},
    }

    def __init__(
        self,
        capital: float = None,
        notify_fn: Callable[[str], None] = None,
        on_halt: Callable[[], None] = None,
    ):
        self.capital       = capital or TRADING.total_capital
        self.notify        = notify_fn or (lambda m: logger.warning(f"CB: {m}"))
        self.on_halt       = on_halt or (lambda: None)
        self.state         = CircuitBreakerState()
        self._trade_history: List[float] = []    # recent trade P&Ls
        self._monitoring   = False
        self._monitor_thread: Optional[threading.Thread] = None

    # ──────────────────────────────────────────────────────────────
    # PUBLIC: called by orchestrator before each trade
    # ──────────────────────────────────────────────────────────────

    def allow_trade(self) -> tuple[bool, str]:
        """
        Gate check: returns (allowed: bool, reason: str).
        Call this before placing ANY order.
        """
        self.state.checks_run += 1

        if self.state.triggered:
            minutes_elapsed = self._minutes_since_trigger()
            if (self.state.auto_reset_after_minutes > 0 and
                    minutes_elapsed >= self.state.auto_reset_after_minutes):
                logger.info(f"Circuit breaker auto-reset after {minutes_elapsed:.0f} min")
                self.reset()
            else:
                return False, f"CIRCUIT BREAKER ACTIVE: {self.state.trigger_reason}"

        return True, "ok"

    def record_trade_result(self, pnl: float):
        """Record a closed trade's P&L. Updates consecutive loss counter."""
        self._trade_history.append(pnl)
        self.state.total_loss_today += min(0, pnl)

        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        # Level 1: consecutive losses
        if self.state.consecutive_losses >= self.LEVELS["CONSECUTIVE_LOSS"]["threshold"]:
            self._trigger(
                "CONSECUTIVE_LOSS",
                f"{self.state.consecutive_losses} consecutive losses — pausing {self.LEVELS['CONSECUTIVE_LOSS']['pause_min']} min",
                auto_reset_min=self.LEVELS["CONSECUTIVE_LOSS"]["pause_min"],
            )

        # Level 2 & 3: daily loss
        loss_pct = abs(self.state.total_loss_today) / self.capital
        if loss_pct >= self.LEVELS["DAILY_LOSS_HARD"]["threshold"]:
            self._trigger(
                "DAILY_LOSS_HARD",
                f"Daily loss {loss_pct:.1%} exceeds 5% — HALTED FOR THE DAY",
                auto_reset_min=0,
            )
        elif loss_pct >= self.LEVELS["DAILY_LOSS_SOFT"]["threshold"]:
            self._trigger(
                "DAILY_LOSS_SOFT",
                f"Daily loss {loss_pct:.1%} exceeds 3% — HALTED FOR THE DAY",
                auto_reset_min=0,
            )

    def check_position_flash_loss(
        self, symbol: str, entry_price: float, current_price: float, quantity: int
    ) -> tuple[bool, str]:
        """
        Real-time check: is this position losing faster than expected?
        Call every tick for open short positions.
        Returns (crisis: bool, message: str).
        """
        # For shorts: loss occurs when price rises above entry
        pnl = (entry_price - current_price) * quantity
        loss_pct = abs(pnl) / self.capital

        if pnl < 0 and loss_pct >= self.LEVELS["FLASH_LOSS"]["threshold"]:
            msg = (
                f"FLASH LOSS on {symbol}: ₹{abs(pnl):.0f} "
                f"({loss_pct:.1%} of capital)"
            )
            self._trigger("FLASH_LOSS", msg, auto_reset_min=30)
            return True, msg

        return False, "ok"

    def check_market_crash(self, nifty_change_pct: float):
        """
        SHORT-SELLING SPECIFIC market condition check.

        Nifty falling is our FRIEND — we only halt on extreme events:
          - Down 0% to -4%  → KEEP TRADING (ideal for shorts)
          - Down -4% to -6% → PAUSE new entries (lower circuit risk on stocks)
          - Down > -6%      → HALT (market-wide circuit breaker likely, can't trade)
        """
        if nifty_change_pct <= -4.0:
            severity = "EXTREME_CRASH"
            reason = (
                f"NIFTY_CRASH: Nifty down {nifty_change_pct:.2f}% — extreme crash, "
                f"stocks may hit lower circuit breakers. "
                f"Cannot cover shorts on circuit-locked stocks."
            )
            self._trigger(severity, reason, auto_reset_min=60)
            logger.warning(f"⚠️ EXTREME CRASH CB: {reason}")
        elif -4.0 < nifty_change_pct <= -3.0:
            severity = "NIFTY_CRASH"
            reason = (
                f"NIFTY_CRASH: Nifty down {nifty_change_pct:.2f}% — "
                f"pausing new entries to manage lower circuit risk."
            )
            self._trigger(severity, reason, auto_reset_min=30)
            logger.warning(f"⚠️ NIFTY CRASH CB: {reason}")
        elif -3.0 < nifty_change_pct <= -1.0:
            # This is the SWEET SPOT for short selling — log it as opportunity
            logger.info(
                f"📉 Nifty {nifty_change_pct:.2f}% — IDEAL short-selling environment. "
                f"No circuit breaker. Aggressive shorts mode."
            )
        elif nifty_change_pct >= 1.5:
            # Nifty rallying strongly — short selling headwind
            logger.info(
                f"📈 Nifty +{nifty_change_pct:.2f}% — market rallying. "
                f"Short positions face headwind. Reduce size."
            )

    def check_vix_spike(self, vix: float):
        """VIX > 30 means spreads are dangerously wide — slippage kills edge."""
        if vix >= self.LEVELS["VIX_SPIKE"]["threshold"]:
            self._trigger(
                "VIX_SPIKE",
                f"India VIX={vix:.1f} — spreads too wide, slippage will kill edge",
                auto_reset_min=60,
            )

    def get_short_selling_environment(self, nifty_change_pct: float, vix: float) -> dict:
        """
        Returns the quality of the current environment for short selling.
        Used by the scanner to adjust position sizes and confidence gates.
        """
        if nifty_change_pct < -3.0 and vix < 25:
            return {"quality": "EXCELLENT", "size_multiplier": 1.3,
                    "reason": f"Nifty {nifty_change_pct:.1f}% — strong short-selling tailwind"}
        elif -3.0 <= nifty_change_pct < -1.0:
            return {"quality": "GOOD",      "size_multiplier": 1.1,
                    "reason": f"Nifty {nifty_change_pct:.1f}% — favourable for shorts"}
        elif -1.0 <= nifty_change_pct < 0:
            return {"quality": "NEUTRAL",   "size_multiplier": 1.0,
                    "reason": f"Nifty {nifty_change_pct:.1f}% — mild selling, normal size"}
        elif 0 <= nifty_change_pct < 1.5:
            return {"quality": "CAUTION",   "size_multiplier": 0.7,
                    "reason": f"Nifty +{nifty_change_pct:.1f}% — slight rally, reduce size"}
        else:
            return {"quality": "POOR",      "size_multiplier": 0.4,
                    "reason": f"Nifty +{nifty_change_pct:.1f}% — strong rally, minimal shorts"}

    def check_slippage(
        self, symbol: str, expected_price: float, executed_price: float
    ):
        """Detect excessive slippage on order execution."""
        if expected_price == 0:
            return
        slippage_pct = abs(executed_price - expected_price) / expected_price
        if slippage_pct >= self.LEVELS["MAX_SLIPPAGE"]["threshold"]:
            logger.warning(
                f"High slippage [{symbol}]: expected ₹{expected_price:.2f}, "
                f"got ₹{executed_price:.2f} ({slippage_pct:.2%})"
            )
            self._trigger(
                "MAX_SLIPPAGE",
                f"Slippage {slippage_pct:.2%} on {symbol} — pausing for liquidity",
                auto_reset_min=15,
            )

    def reset(self):
        """Manually reset the circuit breaker (e.g. after reviewing logs)."""
        old_reason = self.state.trigger_reason
        self.state.triggered = False
        self.state.trigger_reason = ""
        self.state.trigger_time = None
        self.state.auto_reset_after_minutes = 0
        logger.info(f"Circuit breaker reset (was: {old_reason})")
        self.notify(f"✅ Circuit breaker reset (was: {old_reason})")

    def reset_daily(self):
        """Call at market open each morning to reset daily counters."""
        self.state = CircuitBreakerState()
        self._trade_history = []
        logger.info("Circuit breaker daily reset complete")

    def get_status(self) -> Dict:
        return {
            "triggered":          self.state.triggered,
            "reason":             self.state.trigger_reason,
            "consecutive_losses": self.state.consecutive_losses,
            "total_loss_today":   round(self.state.total_loss_today, 2),
            "loss_pct_today":     round(abs(self.state.total_loss_today) / self.capital * 100, 2),
            "checks_run":         self.state.checks_run,
        }

    # ──────────────────────────────────────────────────────────────
    # INTERNAL
    # ──────────────────────────────────────────────────────────────

    def _trigger(self, level: str, reason: str, auto_reset_min: int = 0):
        """Activate the circuit breaker."""
        if self.state.triggered:
            return   # Already halted — don't re-trigger on same condition

        self.state.triggered = True
        self.state.trigger_reason = f"[{level}] {reason}"
        self.state.trigger_time = datetime.now(IST).isoformat()
        self.state.auto_reset_after_minutes = auto_reset_min

        reset_msg = (
            f"Auto-resets in {auto_reset_min} min"
            if auto_reset_min > 0
            else "Manual reset required"
        )

        alert = (
            f"🚨 CIRCUIT BREAKER TRIGGERED [{level}]\n"
            f"Reason: {reason}\n"
            f"Trading HALTED. {reset_msg}."
        )
        logger.critical(alert)
        self.notify(alert)

        # Call halt callback (e.g. to square off all positions)
        try:
            self.on_halt()
        except Exception as e:
            logger.error(f"on_halt callback error: {e}")

    def _minutes_since_trigger(self) -> float:
        if not self.state.trigger_time:
            return 0
        triggered_at = datetime.fromisoformat(self.state.trigger_time)
        if triggered_at.tzinfo is None:
            triggered_at = triggered_at.replace(tzinfo=IST)
        return (datetime.now(IST) - triggered_at).total_seconds() / 60
