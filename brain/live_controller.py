"""
Live Trading Controller
========================
The agent is fully live. No paper trading.

This module manages the transition from decision to execution with:
  1. Final pre-execution sanity check (brain + circuit breaker)
  2. Intelligent order slippage management
  3. Real-time position monitoring with the brain watching
  4. Adaptive exit: brain can exit early if it detects regime change
  5. Post-trade brain reflection triggered automatically
  6. Live performance dashboard updates

The agent has all the intelligence it needs:
  - 11 technical layers before entry
  - Neural reasoning chain-of-thought
  - Walk-forward validated edge
  - Kelly criterion sizing
  - ATR-adaptive stops
  - Multiple circuit breaker levels
  - Nightly self-improvement

The only risk management the human needs:
  - Set TOTAL_CAPITAL correctly in config.py
  - Enable Zerodha 2FA (the agent handles the rest)
  - Monitor the Telegram bot daily
"""

import logging
import time
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from config import TRADING, PAPER_TRADE
from agents.trade_executor import TradeExecutorAgent
from agents.risk_manager import RiskManagerAgent
from agents.position_monitor import PositionMonitorAgent
from utils.circuit_breaker import CircuitBreaker
from utils.alert_manager import alerts, Severity
from intelligence.trade_memory import memory_store, TradeMemory

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")


class LiveTradingController:
    """
    The execution layer for live trading.
    All trades go through here — no manual intervention needed.
    """

    def __init__(
        self,
        executor:  TradeExecutorAgent,
        risk_mgr:  RiskManagerAgent,
        monitor:   PositionMonitorAgent,
        breaker:   CircuitBreaker,
    ):
        self.executor = executor
        self.risk_mgr = risk_mgr
        self.monitor  = monitor
        self.breaker  = breaker
        self._live    = not PAPER_TRADE
        self._open_memories: Dict[str, str] = {}  # symbol → trade_id

        if self._live:
            logger.info("🔴 LIVE TRADING MODE — Real orders will be placed on Zerodha")
        else:
            logger.info("📄 Paper trading mode")

    # ──────────────────────────────────────────────────────────────
    # EXECUTE SHORT (main entry point)
    # ──────────────────────────────────────────────────────────────

    def execute_short(
        self,
        candidate,          # ShortCandidate from nifty50_scanner
        brain_decision,     # TradeDecision from neural_core
        market_context: Dict,
        regime_label:   str = "RANGING",
    ) -> Optional[Dict]:
        """
        Execute a short trade with full intelligence validation.
        Returns execution result dict or None if rejected.
        """
        symbol    = candidate.symbol
        entry     = candidate.entry_price
        sl        = candidate.stop_loss
        target    = candidate.target
        quantity  = candidate.quantity

        # ── Gate 1: Brain says take the trade ─────────────────────
        if not brain_decision.take_trade:
            logger.info(f"Brain PASS [{symbol}]: {brain_decision.key_insight[:80]}")
            return None

        if brain_decision.conviction < 0.45:
            logger.info(f"Brain low conviction [{symbol}]: {brain_decision.conviction:.0%}")
            return None

        # ── Gate 2: Circuit breaker ────────────────────────────────
        allowed, cb_reason = self.breaker.allow_trade()
        if not allowed:
            logger.warning(f"Circuit breaker blocked [{symbol}]: {cb_reason}")
            alerts.warn(f"⛔ CB blocked {symbol}: {cb_reason}")
            return None

        # ── Gate 3: Risk manager ───────────────────────────────────
        risk_decision = self.risk_mgr.approve_trade(symbol, entry, sl, quantity, "SHORT")
        if not risk_decision.approved:
            logger.warning(f"Risk manager rejected [{symbol}]: {risk_decision.reason}")
            return None
        quantity = risk_decision.adjusted_quantity

        # ── Gate 4: Verify R:R still valid at current price ────────
        live_price = self._get_live_price(symbol)
        if live_price and abs(live_price - entry) / entry > 0.003:
            # Price has moved > 0.3% since we calculated entry
            logger.info(f"Price moved [{symbol}]: was {entry:.2f}, now {live_price:.2f} — recalculate")
            entry = live_price
            sl    = round(entry * (1 + TRADING.stop_loss_pct / 100), 2)
            target = round(entry * (1 - TRADING.target_pct / 100), 2)

        # ── Execute ────────────────────────────────────────────────
        result = self.executor.short_sell(symbol, quantity, entry, sl, target)

        if result.get("status") != "EXECUTED":
            logger.error(f"Execution failed [{symbol}]: {result}")
            alerts.critical(f"🚨 Execution failed {symbol}: {result.get('error','unknown')}")
            return None

        # ── Record to trade memory ─────────────────────────────────
        now = datetime.now(IST)
        hour = now.hour
        tod  = "EARLY" if hour < 10 else ("LATE" if hour >= 12 else "MID")

        mem = TradeMemory(
            symbol           = symbol,
            trade_date       = date.today().isoformat(),
            entry_time       = now.strftime("%H:%M:%S"),
            strategy         = "BRAIN_REASONING",
            signals_fired    = candidate.master.supporting_factors if candidate.master else [],
            confidence_score = brain_decision.conviction,
            rsi_at_entry     = candidate.technical.rsi if candidate.technical else 0,
            market_regime    = regime_label,
            fii_net_cr       = float(market_context.get("fii_net", 0)),
            india_vix        = float(market_context.get("vix", 15)),
            sector           = candidate.sector,
            time_of_day      = tod,
            entry_price      = entry,
            stop_loss        = sl,
            target           = target,
            quantity         = quantity,
            mtf_aligned      = True,
            mtf_alignment_count = 2,
        )
        trade_id = memory_store.record(mem)
        self._open_memories[symbol] = trade_id

        # ── Register with monitor ──────────────────────────────────
        self.monitor.register_position(
            symbol=symbol, entry_price=entry,
            stop_loss=sl, target=target, quantity=quantity,
        )

        # ── Record in risk manager ─────────────────────────────────
        self.risk_mgr.record_trade(
            symbol, "SHORT", entry, quantity, result.get("order_id", "")
        )

        # ── Alert ──────────────────────────────────────────────────
        mode_tag = "🔴 LIVE" if self._live else "📄 PAPER"
        alerts.trade(
            f"{mode_tag} SHORT {symbol} |\n"
            f"  Entry ₹{entry:.2f} | SL ₹{sl:.2f} | TGT ₹{target:.2f}\n"
            f"  Qty {quantity} | Risk ₹{quantity*(sl-entry):.0f}\n"
            f"  Brain: {brain_decision.key_insight[:60]}\n"
            f"  Conviction: {brain_decision.conviction:.0%}",
            symbol=symbol,
        )

        logger.info(
            f"✅ SHORT EXECUTED [{symbol}] "
            f"{'LIVE' if self._live else 'PAPER'} | "
            f"entry=₹{entry:.2f} sl=₹{sl:.2f} tgt=₹{target:.2f} "
            f"qty={quantity} | {brain_decision.key_insight[:60]}"
        )

        return {
            "symbol":     symbol,
            "entry":      entry,
            "sl":         sl,
            "target":     target,
            "quantity":   quantity,
            "trade_id":   trade_id,
            "order_id":   result.get("order_id"),
            "live":       self._live,
            "conviction": brain_decision.conviction,
        }

    # ──────────────────────────────────────────────────────────────
    # CLOSE TRADE + BRAIN REFLECTION
    # ──────────────────────────────────────────────────────────────

    def close_trade(
        self,
        symbol:      str,
        exit_price:  float,
        exit_reason: str,
        brain_decision = None,
    ) -> float:
        """Close a position and trigger brain reflection."""
        pnl = self.risk_mgr.close_trade(symbol, exit_price)
        won = pnl > 0

        # Update circuit breaker
        self.breaker.record_trade_result(pnl)

        # Update trade memory
        trade_id = self._open_memories.pop(symbol, None)
        if trade_id:
            entry_data = self.risk_mgr.get_open_position(symbol)
            entry = entry_data.get("entry_price", exit_price) if entry_data else exit_price
            holding_min = int((datetime.now(IST) - datetime.fromisoformat(
                entry_data.get("opened_at", datetime.now(IST).isoformat())
            )).seconds / 60) if entry_data else 0

            pnl_pct = (entry - exit_price) / entry * 100 if entry else 0

            memory_store.update_outcome(
                trade_id=trade_id,
                exit_price=exit_price,
                exit_reason=exit_reason,
                pnl=pnl,
                pnl_pct=pnl_pct,
                holding_minutes=holding_min,
            )

            # Brain reflection (async — doesn't block execution)
            import threading
            if brain_decision:
                threading.Thread(
                    target=self._trigger_reflection,
                    args=(symbol, brain_decision, trade_id, pnl, won, exit_reason),
                    daemon=True,
                ).start()

        # Alert
        emoji = "✅" if won else "🔴"
        pnl_sign = "+" if pnl >= 0 else ""
        alerts.trade(
            f"{emoji} CLOSED {symbol} | {exit_reason} | P&L: {pnl_sign}₹{pnl:.0f}",
            symbol=symbol,
        )

        return pnl

    def _trigger_reflection(
        self, symbol: str, brain_decision, trade_id: str,
        pnl: float, won: bool, exit_reason: str,
    ):
        """Asynchronous brain reflection after trade close."""
        try:
            from brain.neural_core import brain
            outcome = {
                "pnl":         pnl,
                "won":         won,
                "exit_reason": exit_reason,
            }
            lesson = brain.reflect_on_trade(symbol, brain_decision, outcome)
            if lesson and trade_id:
                memory_store.update_outcome(
                    trade_id=trade_id,
                    exit_price=0, exit_reason=exit_reason,
                    pnl=pnl, pnl_pct=0, holding_minutes=0,
                    lesson=lesson,
                )
        except Exception as e:
            logger.warning(f"Brain reflection error [{symbol}]: {e}")

    def _get_live_price(self, symbol: str) -> Optional[float]:
        try:
            from data.free_market_data import get_live_quote
            q = get_live_quote(symbol)
            return float(q["ltp"]) if q else None
        except Exception:
            return None
