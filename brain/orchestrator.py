"""
Brain Orchestrator
===================
The new central command, powered by the Neural Reasoning Core.

Old orchestrator: runs rules → executes if rules pass
Brain orchestrator: thinks → reasons → decides → executes → reflects → evolves

Every single trade goes through:
  1. 11 technical intelligence layers (master scorer)
  2. Neural chain-of-thought reasoning (brain)
  3. Evolution engine rule-check (evolved rules)
  4. Live execution (fully live, no paper trading)
  5. Post-trade reflection (brain learns)
  6. Nightly improvement (parameters evolve)
  7. Weekly evolution (strategy rules evolve)

The agent gets smarter with every single trade.
By Day 30, it has deep knowledge of each Nifty 50 stock.
By Day 90, it has discovered patterns no human programmed.
By Day 180, it has a trading intelligence tailored to current NSE conditions.
"""

import logging
import schedule
import time
from datetime import datetime, date
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from config import TRADING, PAPER_TRADE
from agents.nifty50_scanner import Nifty50ShortScanner, ShortCandidate
from agents.technical_analyst import calculate_all
from agents.risk_manager import RiskManagerAgent
from agents.trade_executor import TradeExecutorAgent
from agents.position_monitor import PositionMonitorAgent
from agents.self_healer import SelfHealerAgent
from agents.sentiment_agent import SentimentAgent
from reports.eod_reporter import EODReporter
from strategies.sector_rotation import SectorRotationAnalyser
from utils.circuit_breaker import CircuitBreaker
from utils.market_calendar import calendar as mkt_calendar
from utils.telegram_bot import TelegramCommandBot
from utils.alert_manager import alerts, Severity
from utils.logger import setup_logging
from intelligence.adaptive_config import adaptive_config
from intelligence.intermarket import intermarket_analyser
from intelligence.market_regime import regime_detector
from intelligence.signal_fusion import signal_fusion

# Phase 2+3 upgrades — loaded safely, fallback to None if unavailable
try:
    from utils.upgrade_loader import upgrades as _upgrades
except Exception:
    _upgrades = None
from intelligence.statistical_edge import edge_calculator
from intelligence.self_improver import self_improver
from intelligence.trade_memory import memory_store
from brain.neural_core import brain, NeuralReasoningCore
from brain.evolution_engine import evolution_engine
from brain.live_controller import LiveTradingController
from data.nse_fetcher import get_nifty_index, get_fii_dii_data, get_historical_ohlcv
from data.free_market_data import free_streamer

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")


class BrainOrchestrator:
    """
    The top-level agent brain. Fully autonomous. Always live.
    """

    def __init__(self):
        logger.info("Initialising Tej Brain Orchestrator...")

        # Core agents
        self.healer   = SelfHealerAgent()
        self.risk_mgr = RiskManagerAgent(capital=TRADING.total_capital)
        self.executor = TradeExecutorAgent()
        self.monitor  = PositionMonitorAgent(
            risk_mgr=self.risk_mgr,
            executor=self.executor,
            notify_fn=self._notify,
        )
        self.breaker  = CircuitBreaker(
            capital=TRADING.total_capital,
            notify_fn=self._notify,
            on_halt=self._emergency_halt,
        )

        # Intelligence
        self.scanner  = Nifty50ShortScanner(
            capital=TRADING.total_capital,
            healer=self.healer,
        )
        self.sentiment = SentimentAgent(healer=self.healer)
        self.sectors   = SectorRotationAnalyser()
        self.reporter  = EODReporter(risk_mgr=self.risk_mgr)

        # Brain components
        self.brain      = brain
        self.evolution  = evolution_engine
        self.controller = LiveTradingController(
            executor=self.executor,
            risk_mgr=self.risk_mgr,
            monitor=self.monitor,
            breaker=self.breaker,
        )

        # Bot
        self.tg_bot = TelegramCommandBot()
        self.tg_bot.register_orchestrator(self)

        # Session state
        self.market_session_active = False
        self.today_candidates:     List[ShortCandidate]   = []
        self.active_trades:        Dict[str, Dict]        = {}
        self.current_regime        = None
        self.adaptive_params       = adaptive_config.load()
        self.intermarket_bias      = 0.0
        self._nifty_data:          Dict = {}
        self._fii_data:            Dict = {}
        self._current_market_ctx:  Dict = {}
        # Store brain decisions for reflection
        self._pending_brain_decisions: Dict[str, object] = {}

        mode = "🔴 LIVE" if not PAPER_TRADE else "📄 PAPER"
        logger.info(
            f"BrainOrchestrator ready | Mode: {mode} | "
            f"Capital: ₹{TRADING.total_capital:,.0f} | "
            f"Universe: {len(TRADING.priority_watchlist)} Nifty50 stocks"
        )

    # ──────────────────────────────────────────────────────────────
    # RUN LOOP
    # ──────────────────────────────────────────────────────────────

    def run(self):
        """Start the brain. Runs indefinitely."""
        mode_str = "LIVE TRADING" if not PAPER_TRADE else "PAPER MODE"
        self._notify(
            f"🧠 Tej Brain started | {mode_str}\n"
            f"Universe: Nifty 50 | Direction: SHORT ONLY\n"
            f"Intelligence score: {self.brain._state.intelligence_score:.0%}"
        )

        # Start services
        self.tg_bot.start()
        free_streamer.start()
        logger.info("Free tick streamer started (NSE public API)")

        # Print brain status
        logger.info(self.brain.get_brain_status())

        # Schedule
        schedule.every().day.at("08:50").do(self._pre_market_intel)
        schedule.every().day.at("09:10").do(self.breaker.reset_daily)
        schedule.every().day.at("09:20").do(self._morning_scan)
        schedule.every(5).minutes.do(self._intraday_loop)
        schedule.every().day.at("15:10").do(self._force_square_off)
        schedule.every().day.at("15:35").do(self._end_of_day)
        schedule.every().sunday.at("20:00").do(self._weekly_evolution)
        schedule.every(30).days.do(mkt_calendar.refresh_holidays)

        # ── Startup catch-up: schedule library does NOT backfill missed tasks.
        # On GitHub Actions the runner often starts after 09:20 IST (slow boot,
        # cold pip cache).  Fire the missed tasks immediately on startup so the
        # agent actually trades today instead of silently skipping every check.
        now_ist = datetime.now(IST)
        is_trading_day = self._is_market_day()
        startup_hour   = now_ist.hour * 60 + now_ist.minute  # minutes since midnight

        if is_trading_day:
            # If we missed the 09:10 circuit-breaker reset, do it now
            if startup_hour >= 9 * 60 + 10:
                logger.info(
                    f"[STARTUP CATCH-UP] Started at {now_ist.strftime('%H:%M')} IST — "
                    f"09:10 task already passed. Resetting circuit breaker now."
                )
                try:
                    self.breaker.reset_daily()
                except Exception as _e:
                    logger.warning(f"Circuit breaker reset on startup failed: {_e}")

            # If we missed the 09:20 morning scan, fire it immediately.
            # This is the primary cause of zero-trade days on slow runners.
            if startup_hour >= 9 * 60 + 20 and not self.market_session_active:
                # Only catch up during viable trading window (before 13:00)
                if startup_hour < 13 * 60:
                    logger.info(
                        f"[STARTUP CATCH-UP] Started at {now_ist.strftime('%H:%M')} IST — "
                        f"morning scan already past. Firing _morning_scan() immediately."
                    )
                    self._morning_scan()
                else:
                    logger.info(
                        f"[STARTUP CATCH-UP] Started at {now_ist.strftime('%H:%M')} IST — "
                        f"past 13:00, too late for morning scan. Monitoring only."
                    )
                    # Still set active so the intraday monitor can close any open positions
                    self.market_session_active = True

        # Main loop
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Brain shutting down gracefully...")
                self._notify("🛑 Agent shut down by operator")
                break
            except Exception as e:
                logger.error(f"Brain loop error: {e}", exc_info=True)
                self.healer.heal(f"Brain loop error: {e}", {})
                time.sleep(10)

    # ──────────────────────────────────────────────────────────────
    # PRE-MARKET INTELLIGENCE (8:50 AM)
    # ──────────────────────────────────────────────────────────────

    def _pre_market_intel(self):
        """Fetch intermarket data and form morning beliefs before NSE opens."""
        logger.info("=== Pre-market intelligence gathering ===")
        try:
            im_bias = intermarket_analyser.get_morning_bias()
            self.intermarket_bias = im_bias.bias_score
            self._nifty_data = get_nifty_index() or {}
            self._fii_data   = get_fii_dii_data() or {}

            self.adaptive_params = adaptive_config.load()

            # Build context for morning brief
            from intelligence.trade_memory import memory_store
            from brain.neural_core import brain
            recent = memory_store.get_recent(days=14)
            closed = [t for t in recent if t.get("exit_reason")]
            win_rate = sum(1 for t in closed if t.get("won")) / max(len(closed), 1)
            all_pnl  = sum(t.get("pnl", 0) for t in memory_store.get_recent(days=365))

            morning_ctx = {
                "nifty_change":       float(self._nifty_data.get("change_pct", 0)),
                "fii_net":            float(self._fii_data.get("fii_net", 0)),
                "regime":             "UNKNOWN",
                "intelligence_score": brain._state.intelligence_score,
                "win_rate_14d":       win_rate,
                "total_pnl_all_time": all_pnl,
                "total_trades":       len(memory_store.get_recent(days=365)),
            }

            # Tej greets his owner with a personal, data-driven morning message
            from brain.tej_persona import tej
            greeting = tej.morning_greeting(morning_ctx)
            self._notify(greeting)
            logger.info(f"Tej morning greeting sent")

            # Brain observes market
            market_data = {
                "intermarket_bias":  im_bias.bias_score,
                "intermarket_label": im_bias.bias,
                "sgx_nifty_chg":     im_bias.sgx_nifty_chg,
                "crude_chg":         im_bias.crude_chg,
                "usd_inr_chg":       im_bias.usd_inr_chg,
                "vix":               im_bias.vix_level or 15,
                "fii_net":           float(self._fii_data.get("fii_net", 0)),
            }
            self.brain.observe_market(market_data)

        except Exception as e:
            logger.error(f"Pre-market intel error: {e}", exc_info=True)
            self.healer.heal(f"Pre-market error: {e}", {})

    # ──────────────────────────────────────────────────────────────
    # MORNING SCAN (9:20 AM)
    # ──────────────────────────────────────────────────────────────

    def _morning_scan(self):
        """Scan all 50 Nifty stocks and select the best short candidates."""
        if not self._is_market_day():
            logger.info("Market holiday — skipping")
            return

        logger.info("=== MORNING SCAN — Nifty 50 Short Selling ===")
        self.market_session_active = True

        try:
            # Refresh NSE data
            self._nifty_data = get_nifty_index() or {}
            self._fii_data   = get_fii_dii_data() or {}
            nifty_chg  = float(self._nifty_data.get("change_pct", 0))
            fii_net    = float(self._fii_data.get("fii_net", 0))

            # Detect market regime
            vix = float(self._nifty_data.get("india_vix", 15))
            advances = int(self._nifty_data.get("advances", 25))
            declines = int(self._nifty_data.get("declines", 25))
            adv_dec  = advances / max(advances + declines, 1)

            self.current_regime = regime_detector.detect(
                vix=vix, advance_decline=adv_dec,
                fii_net_cr=fii_net, nifty_change_1d=nifty_chg,
            )
            self.adaptive_params = adaptive_config.apply_regime_override(
                self.adaptive_params, self.current_regime.label
            )

            logger.info(
                f"Regime: {self.current_regime.label} | "
                f"Short-favourable: {self.current_regime.is_good_for_shorts} | "
                f"Nifty: {nifty_chg:+.2f}% | FII: ₹{fii_net:.0f}Cr"
            )

            if self.current_regime.label == "CRISIS":
                self._notify(
                    "🚨 EXTREME CRASH — VIX>30 + Nifty>4% down\n"
                    "Stocks at risk of lower circuit breakers.\n"
                    "Cannot safely cover shorts on circuit-locked stocks. Halting."
                )
                return

            # Show environment quality — falling Nifty = green signal for us
            env = self.breaker.get_short_selling_environment(nifty_chg, vix)
            logger.info(
                f"Short-selling environment: {env['quality']} | "
                f"Size: {env['size_multiplier']}x | {env['reason']}"
            )
            if env["quality"] in ("EXCELLENT", "GOOD"):
                self._notify(
                    f"📉 {env['quality']} short-selling environment!\n"
                    f"{env['reason']}\n"
                    f"Deploying {env['size_multiplier']}x position size"
                )

            # Build shared market context
            self._current_market_ctx = {
                "regime":           self.current_regime.label,
                "nifty_change":     nifty_chg,
                "fii_net":          fii_net,
                "vix":              vix,
                "intermarket_bias": self.intermarket_bias,
                "date":             date.today().isoformat(),
            }

            # Run scanner
            candidates = self.scanner.scan(intermarket_bias=self.intermarket_bias)
            logger.info(f"Scanner found {len(candidates)} candidates")

            # ── Brain reviews each candidate ──────────────────────
            approved_candidates = []
            for c in candidates:
                try:
                    # Get similar historical trades for memory context
                    similar = self._find_similar_trades(c)
                    hist    = self._get_historical_performance()

                    # Context for this specific stock
                    stock_ctx = {**self._current_market_ctx, "sector": c.sector}

                    # Phase 2+3 upgrade signals — enhances brain decision
                    if _upgrades:
                        try:
                            enh = _upgrades.get_enhanced_signals(c.symbol, None, stock_ctx)
                            stock_ctx.update(enh)
                            if enh.get("sentiment_signal") == "AVOID_SHORT":
                                logger.info(f"  {c.symbol}: Sentiment says AVOID — skipping")
                                continue
                            if enh.get("insider_signal") in ("STRONG_BULLISH",):
                                logger.info(f"  {c.symbol}: Insider buying — skipping")
                                continue
                        except Exception as _e:
                            logger.debug(f"Upgrade signals skipped: {_e}")

                    # Brain chain-of-thought reasoning
                    brain_decision = self.brain.decide(
                        symbol=c.symbol,
                        master_score=c.score,
                        signals=c.master.supporting_factors if c.master else [],
                        market_context=stock_ctx,
                        historical_perf=hist,
                        similar_trades=similar,
                    )

                    # Evolution engine evolved-rule check
                    ev_allowed, ev_reason = self.evolution.apply_rules(c.symbol, stock_ctx)
                    if not ev_allowed:
                        logger.info(f"  {c.symbol}: {ev_reason}")
                        continue

                    # Brain says take it
                    if brain_decision.take_trade:
                        approved_candidates.append((c, brain_decision))
                        logger.info(
                            f"  🧠 APPROVED [{c.symbol}] "
                            f"conviction={brain_decision.conviction:.0%} | "
                            f"{brain_decision.key_insight[:70]}"
                        )
                    else:
                        logger.info(
                            f"  🧠 DECLINED [{c.symbol}] | "
                            f"{brain_decision.final_verdict[:70]}"
                        )

                except Exception as e:
                    logger.error(f"Brain decision error [{c.symbol}]: {e}")

            self.today_candidates = [c for c, _ in approved_candidates]

            if not approved_candidates:
                self._notify("🔍 Morning scan complete — no setups pass brain review today")
                return

            self._notify(
                f"🔍 Brain approved {len(approved_candidates)} candidate(s):\n" +
                "\n".join(
                    f"  {c.symbol} | score={c.score:.2f} | "
                    f"R:R={c.risk_reward:.1f} | {c.reason[:50]}"
                    for c, _ in approved_candidates
                )
            )

            # Execute top brain-approved candidates
            for candidate, brain_decision in approved_candidates[:TRADING.max_open_positions]:
                if self.risk_mgr.get_open_position_count() >= TRADING.max_open_positions:
                    break
                result = self.controller.execute_short(
                    candidate=candidate,
                    brain_decision=brain_decision,
                    market_context=self._current_market_ctx,
                    regime_label=self.current_regime.label,
                )
                if result:
                    self._pending_brain_decisions[candidate.symbol] = brain_decision
                    self.active_trades[candidate.symbol] = result

        except Exception as e:
            logger.error(f"Morning scan error: {e}", exc_info=True)
            self.healer.heal(f"Morning scan error: {e}", {})

    # ──────────────────────────────────────────────────────────────
    # INTRADAY LOOP (every 5 minutes)
    # ──────────────────────────────────────────────────────────────

    def _intraday_loop(self):
        """Monitor positions and look for fresh setups every 5 minutes."""
        if not self.market_session_active:
            return

        now = datetime.now(IST)
        if not (9 <= now.hour < 15 or (now.hour == 15 and now.minute < 10)):
            return

        # Monitor existing positions
        try:
            closed = self.monitor.check_all()
            for symbol in (closed or []):
                trade_data = self.active_trades.pop(symbol, {})
                brain_dec  = self._pending_brain_decisions.pop(symbol, None)
                exit_price = float(self.risk_mgr.get_last_exit_price(symbol) or 0)
                exit_reason = "SL_OR_TARGET"
                pnl = self.controller.close_trade(symbol, exit_price, exit_reason, brain_dec)
                logger.info(f"Position closed [{symbol}]: P&L=₹{pnl:.0f}")
        except Exception as e:
            logger.error(f"Position monitor error: {e}")

        # Look for new setups if slots available (only before 1 PM)
        if (now.hour < 13 and
                self.risk_mgr.get_open_position_count() < TRADING.max_open_positions):
            try:
                fresh = self.scanner.scan_live(
                    watched_symbols=TRADING.priority_watchlist[:20],
                    market_breadth=self._current_market_ctx.get("regime", "NEUTRAL"),
                    fii_net=self._current_market_ctx.get("fii_net", 0),
                    intermarket_bias=self.intermarket_bias,
                )
                for c in fresh:
                    if c.symbol in self.active_trades:
                        continue  # already in trade
                    hist = self._get_historical_performance()
                    bd   = self.brain.decide(
                        symbol=c.symbol,
                        master_score=c.score,
                        signals=c.master.supporting_factors if c.master else [],
                        market_context={**self._current_market_ctx, "sector": c.sector},
                        historical_perf=hist,
                        similar_trades=self._find_similar_trades(c),
                    )
                    ev_ok, _ = self.evolution.apply_rules(c.symbol, self._current_market_ctx)
                    if bd.take_trade and ev_ok:
                        result = self.controller.execute_short(c, bd, self._current_market_ctx)
                        if result:
                            self._pending_brain_decisions[c.symbol] = bd
                            self.active_trades[c.symbol] = result
                        break   # one new entry per 5-min cycle
            except Exception as e:
                logger.debug(f"Intraday scan error: {e}")

    # ──────────────────────────────────────────────────────────────
    # FORCE SQUARE-OFF (3:10 PM)
    # ──────────────────────────────────────────────────────────────

    def _force_square_off(self):
        """Close ALL positions at 3:10 PM. No exceptions."""
        if not self.active_trades:
            return
        logger.info("=== 3:10 PM — Force square-off all positions ===")
        results = self.monitor.force_square_off_all()
        self.market_session_active = False
        pnl = self.risk_mgr.get_today_pnl()
        sign = "+" if pnl >= 0 else ""
        self._notify(
            f"🔔 3:10 PM square-off complete\n"
            f"Day P&L: {sign}₹{pnl:,.0f}\n"
            f"Positions closed: {len(results or [])}"
        )
        self.active_trades.clear()

    # ──────────────────────────────────────────────────────────────
    # END OF DAY (3:35 PM)
    # ──────────────────────────────────────────────────────────────

    def _end_of_day(self):
        """Post-market: EOD report + brain learning + Tej's honest daily voice summary."""
        logger.info("=== End of Day Processing ===")
        try:
            summary = self.risk_mgr.get_daily_summary()
            self.reporter.generate()
            pnl     = self.risk_mgr.get_today_pnl()

            from intelligence.postmortem import postmortem_reviewer
            closed = [t for t in summary.get("trades", []) if t.get("status") == "CLOSED"]
            if closed:
                postmortem_reviewer.review_day(closed, day_context=self._current_market_ctx)

            # Update Bayesian signal fusion
            for t in closed:
                sigs = [s.strip() for s in t.get("reason", "").split("|")
                        if any(k in s.upper() for k in ["RSI","VWAP","WYCKOFF","FII","MACD"])]
                if sigs:
                    won = t.get("pnl", 0) > 0
                    signal_fusion.record_outcome(sigs, won)
                    edge_calculator.record(sigs, t.get("pnl", 0), won)

            # Nightly self-improvement
            result  = self_improver.run_nightly_improvement()
            changes = result.get("param_changes", [])
            lessons = result.get("lessons", [])

            wins      = sum(1 for t in closed if t.get("pnl", 0) > 0)
            total_t   = len(closed)
            win_rate  = wins / max(total_t, 1)
            capital   = TRADING.total_capital + pnl

            # Update goal tracker
            from brain.goal_tracker import goal_tracker
            goal_tracker.update(capital, pnl, win_rate)

            # Tej speaks — honest EOD voice message to his owner
            from brain.tej_persona import tej
            eod_msg = tej.eod_summary({
                "today_pnl":     pnl,
                "wins":          wins,
                "losses":        total_t - wins,
                "lessons":       lessons,
                "param_changes": changes,
            })
            self._notify(eod_msg)

        except Exception as e:
            logger.error(f"End-of-day error: {e}", exc_info=True)
            # ── FIXED: Never crash silently — always notify ──
            self._notify(
                f"⚠️ EOD Report\nError during processing: {e}\n"
                "Check Zerodha app for P&L."
            )

    # ──────────────────────────────────────────────────────────────
    # EOD REPORT — exposed for TelegramCommandBot + /eod command
    # ──────────────────────────────────────────────────────────────

    def generate_eod_report(self) -> str:
        """
        Public method called by TelegramCommandBot when /eod is triggered.
        Pulls real P&L from Kite + daily summary from risk manager.
        Returns a Telegram-formatted string.
        """
        lines = ["📊 <b>EOD Report</b>\n"]

        # Real P&L from risk manager (always available)
        try:
            pnl     = self.risk_mgr.get_today_pnl()
            summary = self.risk_mgr.get_daily_summary()
            trades  = summary.get("trades", [])
            closed  = [t for t in trades if t.get("status") == "CLOSED"]
            wins    = sum(1 for t in closed if t.get("pnl", 0) > 0)
            losses  = len(closed) - wins

            sign = "+" if pnl >= 0 else ""
            lines.append(f"Day P&L   : Rs {sign}{pnl:,.2f}")
            lines.append(f"Trades    : {len(closed)} closed ({wins}W / {losses}L)")

            if closed:
                lines.append("\n<b>Trades Today:</b>")
                for t in closed:
                    t_pnl = t.get("pnl", 0)
                    emoji = "✅" if t_pnl >= 0 else "❌"
                    lines.append(
                        f"  {emoji} {t.get('symbol','?')} | "
                        f"Rs {t_pnl:+,.0f} | {t.get('exit_reason','?')}"
                    )
        except Exception as e:
            logger.error(f"generate_eod_report risk_mgr error: {e}")
            lines.append(f"⚠️ Could not fetch P&L from risk manager: {e}")

        # Open positions (if any remain)
        try:
            if self.active_trades:
                lines.append(f"\n<b>Open Positions:</b> {len(self.active_trades)}")
                for sym in self.active_trades:
                    lines.append(f"  ⚡ {sym} — still open")
        except Exception:
            pass

        # Capital update
        try:
            from config import TRADING
            pnl_val = self.risk_mgr.get_today_pnl()
            capital = TRADING.total_capital + pnl_val
            lines.append(f"\nCapital   : Rs {capital:,.0f}")
        except Exception:
            pass

        # Brain intelligence score
        try:
            lines.append(f"Intelligence: {self.brain._state.intelligence_score:.0%}")
        except Exception:
            pass

        lines.append("\n✅ Check Zerodha app to confirm all orders.")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────
    # WEEKLY EVOLUTION (Sunday 8 PM)
    # ──────────────────────────────────────────────────────────────

    def _weekly_evolution(self):
        """Sunday evening: evolve strategy, retrain ML, run WFO."""
        logger.info("=== Weekly Evolution Cycle ===")
        self._notify("🧬 Weekly evolution starting...")
        try:
            # Run evolution engine
            ev_result = self.evolution.run_evolution_cycle()
            adopted  = ev_result.get("adopted", [])
            discovered = ev_result.get("discoveries", [])

            # Retrain ML ensemble
            from intelligence.ml_ensemble import ml_predictor
            retrained = ml_predictor.train()

            # WFO on key symbols
            from intelligence.walk_forward import wfo_engine
            verdicts = {}
            for sym in TRADING.priority_watchlist[:5]:
                try:
                    r = wfo_engine.run(sym, total_days=120)
                    verdicts[sym] = r.strategy_verdict
                except Exception:
                    pass

            # Bad habit detection
            from intelligence.postmortem import postmortem_reviewer
            bad_habits = postmortem_reviewer.identify_bad_habits(days=30)

            msg = (
                f"🧬 Weekly evolution complete:\n"
                f"  Rules adopted: {len(adopted)}\n"
                f"  Patterns discovered: {len(discovered)}\n"
                f"  ML retrained: {'✅' if retrained else '⏭'}\n"
                f"  Brain intelligence: {self.brain._state.intelligence_score:.0%}\n"
                + (f"  Bad habit: {bad_habits[0]}" if bad_habits else "")
                + (f"\n  Adopted: {adopted[0][:60]}" if adopted else "")
            )
            self._notify(msg)
            logger.info(msg)

        except Exception as e:
            logger.error(f"Weekly evolution error: {e}", exc_info=True)

    # ──────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────

    def _find_similar_trades(self, candidate: ShortCandidate) -> List[Dict]:
        """Find historical trades similar to this candidate."""
        try:
            recent = memory_store.get_recent(days=60)
            return [
                t for t in recent
                if t.get("symbol") == candidate.symbol
                or t.get("sector") == candidate.sector
            ][:8]
        except Exception:
            return []

    def _get_historical_performance(self) -> Dict:
        """Get recent performance summary for brain context."""
        try:
            recent = memory_store.get_recent(days=14)
            closed = [t for t in recent if t.get("exit_reason")]
            if not closed:
                return {"win_rate": 0.5, "avg_pnl": 0, "trend": "STABLE"}
            wins   = sum(1 for t in closed if t.get("won"))
            pnls   = [t.get("pnl", 0) for t in closed]
            mid    = len(pnls) // 2
            early_avg = sum(pnls[:mid]) / max(mid, 1)
            late_avg  = sum(pnls[mid:]) / max(len(pnls) - mid, 1)
            trend  = "IMPROVING" if late_avg > early_avg else ("DECLINING" if late_avg < early_avg * 0.9 else "STABLE")
            return {
                "win_rate": wins / len(closed),
                "avg_pnl":  sum(pnls) / len(pnls),
                "trend":    trend,
                "n_trades": len(closed),
            }
        except Exception:
            return {"win_rate": 0.5, "avg_pnl": 0, "trend": "STABLE"}

    def _emergency_halt(self):
        self.market_session_active = False
        logger.critical("EMERGENCY HALT triggered")
        self.monitor.force_square_off_all()
        self.active_trades.clear()

    def _is_market_day(self) -> bool:
        return mkt_calendar.is_trading_day(date.today())

    def _notify(self, message: str):
        alerts.send(message)

    def _square_off_all(self):
        self._force_square_off()
