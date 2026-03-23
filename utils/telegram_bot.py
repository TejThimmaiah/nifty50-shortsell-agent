"""
Telegram Command Bot
Control the trading agent entirely via Telegram messages.
Send commands to pause, resume, get status, run manual scans, and more.
Runs as a background thread inside the agent process.

Commands:
  /status      — Current agent state, positions, day P&L
  /positions   — Open positions with live P&L
  /scan        — Trigger an immediate scan
  /pause       — Pause new entries (keep monitoring existing)
  /resume      — Resume normal operation
  /report      — Generate and send today's performance report
  /breaker     — Show circuit breaker status
  /weekly      — Weekly P&L summary
  /squareoff   — Emergency square-off all positions
  /history     — Last 10 closed trades
  /help        — Show all commands
"""

import logging
import threading
import time
import requests
from datetime import datetime
from typing import Optional, Callable, Dict, Any
from zoneinfo import ZoneInfo

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


class TelegramCommandBot:
    """
    Long-polling Telegram bot that runs in a background thread.
    Delegates commands to the orchestrator via registered handlers.
    """

    def __init__(self):
        self._handlers: Dict[str, Callable[[], str]] = {}
        self._running   = False
        self._thread: Optional[threading.Thread] = None
        self._last_update_id = 0
        self._paused   = False
        self._authorised_chat_id = TELEGRAM_CHAT_ID

        # Register built-in commands
        self._register_defaults()

    # ──────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────

    def register(self, command: str, handler: Callable[[], str]):
        """
        Register a custom command handler.
        handler must return a string (the reply message).
        Example:
          bot.register("/scan", lambda: orchestrator.run_scan_now())
        """
        self._handlers[command.lower()] = handler
        logger.debug(f"Telegram command registered: {command}")

    def register_orchestrator(self, orchestrator):
        """Wire up all orchestrator commands at once."""
        self.register("/status",    lambda: self._status_reply(orchestrator))
        self.register("/positions", lambda: self._positions_reply(orchestrator))
        self.register("/scan",      lambda: self._scan_reply(orchestrator))
        self.register("/pause",     lambda: self._pause_reply(orchestrator))
        self.register("/resume",    lambda: self._resume_reply(orchestrator))
        self.register("/report",    lambda: self._report_reply(orchestrator))
        self.register("/breaker",   lambda: self._breaker_reply(orchestrator))
        self.register("/weekly",    lambda: self._weekly_reply(orchestrator))
        self.register("/squareoff", lambda: self._squareoff_reply(orchestrator))
        self.register("/history",   lambda: self._history_reply(orchestrator))
        logger.info("All orchestrator commands registered with Telegram bot")

    def start(self):
        """Start polling for Telegram messages in a background thread."""
        if not TELEGRAM_BOT_TOKEN:
            logger.warning("TELEGRAM_BOT_TOKEN not set — command bot disabled")
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="telegram-bot",
        )
        self._thread.start()
        logger.info("Telegram command bot started")
        self.send("🤖 Tej agent started. Send /help for commands.")

    def stop(self):
        self._running = False

    def send(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send a message to the configured chat."""
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return False
        try:
            resp = requests.post(
                f"{BASE_URL}/sendMessage",
                json={
                    "chat_id":    TELEGRAM_CHAT_ID,
                    "text":       message[:4096],   # Telegram limit
                    "parse_mode": parse_mode,
                },
                timeout=10,
            )
            return resp.ok
        except Exception as e:
            logger.warning(f"Telegram send failed: {e}")
            return False

    def is_paused(self) -> bool:
        """Returns True if the bot has paused new trade entries."""
        return self._paused

    # ──────────────────────────────────────────────────────────────
    # POLLING LOOP
    # ──────────────────────────────────────────────────────────────

    def _poll_loop(self):
        """Long-poll Telegram for new messages."""
        logger.info("Telegram bot polling started")
        while self._running:
            try:
                updates = self._get_updates()
                for update in updates:
                    self._process_update(update)
            except Exception as e:
                logger.warning(f"Telegram poll error: {e}")
            time.sleep(3)   # Poll every 3 seconds

    def _get_updates(self) -> list:
        """Fetch new updates from Telegram."""
        resp = requests.get(
            f"{BASE_URL}/getUpdates",
            params={
                "offset":  self._last_update_id + 1,
                "timeout": 5,
                "allowed_updates": ["message"],
            },
            timeout=10,
        )
        if not resp.ok:
            return []
        data = resp.json()
        updates = data.get("result", [])
        if updates:
            self._last_update_id = updates[-1]["update_id"]
        return updates

    def _process_update(self, update: dict):
        """Process a single incoming message from Tej Thimmaiah."""
        msg = update.get("message", {})
        if not msg:
            return

        chat_id = str(msg.get("chat", {}).get("id", ""))
        text    = msg.get("text", "").strip()

        # Security: only respond to Tej
        if chat_id != str(self._authorised_chat_id):
            logger.warning(f"Unauthorised message from chat_id={chat_id}")
            return

        if not text:
            return

        logger.info(f"Telegram message: {text[:60]}")

        # ── Route commands (/status, /scan, etc.) ─────────────────
        if text.startswith("/"):
            command = text.split()[0].lower()
            if command == "/help":
                reply = self._help_text()
            elif command in self._handlers:
                try:
                    reply = self._handlers[command]()
                except Exception as e:
                    reply = f"❌ Error: {str(e)[:100]}"
            else:
                reply = f"Unknown command: {command}\nSend /help or just talk to me freely."
            self.send(reply)
            return

        # ── Everything else → Tej talks back freely ───────────────
        try:
            from brain.tej_persona import tej
            context = self._get_live_context()
            self.send("💭 thinking...")
            reply = tej.respond(text, context)
            self.send(reply)
        except Exception as e:
            logger.error(f"Tej conversation error: {e}")
            self.send("Having trouble thinking right now — LLM connection issue. Try again in a moment.")

    # ──────────────────────────────────────────────────────────────
    # BUILT-IN REPLY BUILDERS
    # ──────────────────────────────────────────────────────────────

    def _register_defaults(self):
        """Register the /help and /ping commands that don't need orchestrator."""
        self._handlers["/help"]    = self._help_text
        self._handlers["/ping"]    = lambda: "🏓 Alive. Thinking. Working toward the goal."
        self._handlers["/review"]  = self._review_reply
        self._handlers["/goal"]    = self._goal_reply
        self._handlers["/mission"] = self._mission_reply

    def _help_text(self) -> str:
        return (
            "👋 <b>Tej here.</b>\n\n"
            "This is OUR mission — not just yours.\n"
            "Making the Thimmaiah family the first billionaires "
            "in their lineage. I own this as much as you do.\n\n"
            "Just <b>talk to me freely</b> — type anything.\n"
            "Markets, trades, progress, doubts, questions, life.\n"
            "I'll be straight with you. Always.\n\n"
            "<b>Commands:</b>\n"
            "/status    — Day P&amp;L + open positions\n"
            "/positions — Live P&amp;L on each short\n"
            "/scan      — Force scan all 50 Nifty stocks now\n"
            "/goal      — Our progress toward ₹1,000 crore\n"
            "/mission   — What this mission means to me\n"
            "/review    — Honest 30-day performance review\n"
            "/report    — Full day report\n"
            "/weekly    — 7-day summary\n"
            "/history   — Last 10 trades\n"
            "/pause     — Pause new entries\n"
            "/resume    — Resume trading\n"
            "/squareoff — Emergency close all positions\n"
            "/breaker   — Circuit breaker status\n"
            "/ping      — Health check\n\n"
            "<i>Or just talk to me. We're partners.</i>"
        )

    def _mission_reply(self) -> str:
        try:
            from brain.tej_persona import tej
            return tej.mission_statement()
        except Exception as e:
            return f"Error: {e}"

    def _review_reply(self) -> str:
        try:
            from brain.tej_persona import tej
            return tej.honest_performance_review(days=30)
        except Exception as e:
            return f"Error running review: {e}"

    def _goal_reply(self) -> str:
        try:
            from brain.goal_tracker import goal_tracker
            from brain.tej_persona import tej
            from config import TRADING

            # Get capital estimate
            capital = TRADING.total_capital
            if self._orchestrator:
                try:
                    capital = TRADING.total_capital + self._orchestrator.risk_mgr.get_today_pnl()
                except Exception:
                    pass

            # Goal tracker visual
            tracker_msg = goal_tracker.format_for_telegram(capital)

            # Tej's voice on the goal
            goal_voice  = tej.answer_about_goal(
                "Where am I right now on the journey to becoming the first billionaire "
                "in the Thimmaiah family? Be honest — progress, timeline, and what needs to happen next."
            )
            return tracker_msg + "\n\n" + goal_voice
        except Exception as e:
            return f"Goal tracker error: {e}"

    def _get_live_context(self) -> dict:
        """Pull live state to give Tej context for the conversation."""
        ctx = {}
        try:
            if self._orchestrator:
                orch = self._orchestrator
                ctx["today_pnl"]       = orch.risk_mgr.get_today_pnl()
                ctx["open_positions"]  = orch.risk_mgr.get_open_position_count()
                ctx["regime"]          = orch.current_regime.label if orch.current_regime else "UNKNOWN"
                ctx["nifty_change"]    = float(orch._nifty_data.get("change_pct", 0)) if hasattr(orch, "_nifty_data") else 0
        except Exception:
            pass
        try:
            from brain.neural_core import brain
            from intelligence.trade_memory import memory_store
            ctx["intelligence_score"] = brain._state.intelligence_score
            recent = memory_store.get_recent(days=14)
            closed = [t for t in recent if t.get("exit_reason")]
            if closed:
                ctx["win_rate_14d"] = sum(1 for t in closed if t.get("won")) / len(closed)
                ctx["total_pnl_all_time"] = sum(t.get("pnl", 0) for t in memory_store.get_recent(days=365))
                ctx["total_trades"]  = len(memory_store.get_recent(days=365))
        except Exception:
            pass
        return ctx

    def _status_reply(self, orch) -> str:
        try:
            pnl        = orch.risk_mgr.get_today_pnl()
            open_count = orch.risk_mgr.get_open_position_count()
            cb_status  = orch.breaker.get_status()
            now        = datetime.now(IST).strftime("%H:%M IST")
            mode       = "📄 PAPER" if __import__("config").PAPER_TRADE else "⚡ LIVE"

            pnl_sign = "+" if pnl >= 0 else ""
            pnl_emoji= "📈" if pnl >= 0 else "📉"

            cb_line = (
                f"🚨 CB ACTIVE: {cb_status['reason'][:50]}"
                if cb_status["triggered"]
                else "✅ Circuit breaker: OK"
            )

            return (
                f"<b>Tej Status — {now}</b>\n"
                f"Mode: {mode}\n\n"
                f"{pnl_emoji} Day P&amp;L: <b>{pnl_sign}₹{pnl:,.0f}</b>\n"
                f"📊 Open positions: {open_count}/{__import__('config').TRADING.max_open_positions}\n"
                f"{'⏸ PAUSED' if self._paused else '▶ Active'}\n"
                f"{cb_line}"
            )
        except Exception as e:
            return f"Status error: {e}"

    def _positions_reply(self, orch) -> str:
        try:
            pos = orch.monitor.get_status()
            if not pos:
                return "📭 No open positions."

            lines = ["<b>Open Positions:</b>\n"]
            for sym, data in pos.items():
                entry  = data.get("entry", 0)
                ltp    = data.get("ltp") or "—"
                sl     = data.get("sl", 0)
                target = data.get("target", 0)
                pnl_pct= data.get("pnl_pct") or 0
                trail  = " 📐trail" if data.get("trailing") else ""
                pnl_e  = "📈" if pnl_pct >= 0 else "📉"
                lines.append(
                    f"{pnl_e} <b>{sym}</b>{trail}\n"
                    f"  Entry ₹{entry:.2f} → LTP ₹{ltp}\n"
                    f"  SL ₹{sl:.2f} | Target ₹{target:.2f}\n"
                    f"  P&amp;L: {'+' if pnl_pct>=0 else ''}{pnl_pct:.2f}%"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Positions error: {e}"

    def _scan_reply(self, orch) -> str:
        try:
            self.send("🔍 Running scan — this takes 30–60 seconds...")
            orch._morning_scan()
            count = len(orch.today_candidates)
            return f"✅ Scan complete. Found {count} candidates."
        except Exception as e:
            return f"Scan error: {e}"

    def _pause_reply(self, orch) -> str:
        self._paused = True
        orch.market_session_active = False
        return "⏸ Trading paused. Existing positions still monitored.\nSend /resume to restart."

    def _resume_reply(self, orch) -> str:
        self._paused = False
        orch.market_session_active = True
        return "▶ Trading resumed."

    def _report_reply(self, orch) -> str:
        try:
            self.send("📊 Generating report...")
            summary = orch.risk_mgr.get_daily_summary()
            pnl  = summary.get("total_pnl", 0)
            sign = "+" if pnl >= 0 else ""
            wr   = summary.get("win_rate", 0)
            wins = summary.get("win_count", 0)
            loss = summary.get("loss_count", 0)
            return (
                f"<b>Today's Report — {summary.get('date','')}</b>\n\n"
                f"{'📈' if pnl >= 0 else '📉'} P&amp;L: <b>{sign}₹{pnl:,.0f}</b>\n"
                f"Trades: {wins}W / {loss}L  ({wr:.0f}% win rate)\n"
                f"Best: {summary.get('best_trade','—')} | "
                f"Worst: {summary.get('worst_trade','—')}"
            )
        except Exception as e:
            return f"Report error: {e}"

    def _breaker_reply(self, orch) -> str:
        try:
            status = orch.breaker.get_status()
            triggered = status["triggered"]
            if triggered:
                return (
                    f"🚨 <b>Circuit Breaker ACTIVE</b>\n"
                    f"Reason: {status['reason']}\n"
                    f"Consecutive losses: {status['consecutive_losses']}\n"
                    f"Day loss: ₹{abs(status['total_loss_today']):,.0f} "
                    f"({status['loss_pct_today']:.1f}%)"
                )
            return (
                f"✅ Circuit Breaker: <b>OK</b>\n"
                f"Consecutive losses: {status['consecutive_losses']}\n"
                f"Day loss: ₹{abs(status['total_loss_today']):,.0f} "
                f"({status['loss_pct_today']:.1f}%)\n"
                f"Checks run: {status['checks_run']}"
            )
        except Exception as e:
            return f"Breaker status error: {e}"

    def _weekly_reply(self, orch) -> str:
        try:
            from reports.performance_analytics import PerformanceAnalytics
            analytics = PerformanceAnalytics(capital=__import__("config").TRADING.total_capital)
            report    = analytics.generate_report(days_back=7)
            pnl  = report.overall_pnl
            sign = "+" if pnl >= 0 else ""
            return (
                f"<b>Weekly Summary</b>\n\n"
                f"{'📈' if pnl >= 0 else '📉'} P&amp;L: <b>{sign}₹{pnl:,.0f}</b>\n"
                f"Trades: {report.total_trades} | "
                f"Win rate: {report.overall_win_rate:.0f}%\n"
                f"Profit factor: {report.overall_profit_factor:.2f}\n"
                f"Max drawdown: {report.max_drawdown_pct:.1f}%\n"
                f"Best symbol: {report.most_profitable_symbol}"
                + (f"\n\n⚠ {report.alerts[0]}" if report.alerts else "")
            )
        except Exception as e:
            return f"Weekly report error: {e}"

    def _squareoff_reply(self, orch) -> str:
        try:
            self.send("🚨 EMERGENCY SQUARE-OFF INITIATED...")
            orch._square_off_all()
            pnl = orch.risk_mgr.get_today_pnl()
            return f"✅ All positions squared off.\nDay P&amp;L: ₹{pnl:,.0f}"
        except Exception as e:
            return f"Square-off error: {e}"

    def _history_reply(self, orch) -> str:
        try:
            summary = orch.risk_mgr.get_daily_summary()
            trades  = summary.get("trades", [])
            closed  = [t for t in trades if t.get("status") == "CLOSED"]

            if not closed:
                return "📭 No closed trades today."

            lines = ["<b>Today's Closed Trades:</b>\n"]
            for t in closed[-10:]:
                pnl   = t.get("pnl", 0)
                emoji = "✅" if pnl >= 0 else "🔴"
                sign  = "+" if pnl >= 0 else ""
                lines.append(
                    f"{emoji} {t.get('symbol')} → {sign}₹{pnl:.0f}"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"History error: {e}"
