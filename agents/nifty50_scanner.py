"""
Nifty 50 Short Selling Scanner
Scans all 50 Nifty 50 stocks every morning.
Finds the best 1–3 short candidates for the day.
Short-only. No LONG signals accepted. No non-Nifty50 stocks.

Scan sequence:
  9:15 AM — Fetch pre-market FII/DII data + SGX Nifty bias
  9:20 AM — Scan all 50 stocks in order of priority
  By 9:35 AM — Top 3 candidates identified and ready
  9:35 AM onwards — Execute the best setups as they confirm intraday
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from config import TRADING
from data.nifty50_universe import NIFTY50, get_sector, is_nifty50
from agents.technical_analyst import calculate_all, TechnicalSignal
from agents.self_healer import SelfHealerAgent
from data.nse_fetcher import (
    get_intraday_ohlcv, get_historical_ohlcv,
    get_fii_dii_data, get_nifty_index, get_oi_data,
)
from data.free_market_data import get_live_quote
from intelligence.master_scorer import master_scorer, MasterScore

logger = logging.getLogger(__name__)


@dataclass
class ShortCandidate:
    """
    A Nifty 50 short-selling candidate.
    SHORT ONLY. MIS intraday. NSE only.
    """
    symbol:         str
    score:          float         # master score 0–1
    sector:         str           # Nifty 50 sector (FINANCIAL, IT, ENERGY, etc.)
    technical:      TechnicalSignal
    entry_price:    float
    stop_loss:      float         # ABOVE entry (short)
    target:         float         # BELOW entry (short)
    quantity:       int
    risk_amount:    float         # ₹ at risk if SL hit
    risk_reward:    float         # reward/risk ratio
    master:         Optional[MasterScore] = None
    oi_change_pct:  float = 0.0
    reason:         str   = ""    # human-readable confluence summary


class Nifty50ShortScanner:
    """
    Scans all 50 Nifty 50 stocks for intraday short opportunities.
    Returns the best 1–3 candidates each morning.
    Short-only. No exceptions.
    """

    def __init__(self, capital: float = None, healer: SelfHealerAgent = None):
        self.capital = capital or TRADING.total_capital
        self.healer  = healer

        # Pre-validate: enforce Nifty 50 mandate
        TRADING.validate()
        logger.info(
            f"Nifty50ShortScanner initialised | "
            f"Universe: {len(NIFTY50)} stocks | "
            f"Direction: SHORT_ONLY | "
            f"Capital: ₹{self.capital:,.0f}"
        )

    # ──────────────────────────────────────────────────────────────
    # MAIN SCAN (called at 9:20 AM)
    # ──────────────────────────────────────────────────────────────

    def scan(
        self,
        quick_mode: bool = False,     # True = scan Tier 1 only (faster)
        intermarket_bias: float = 0.0,
    ) -> List[ShortCandidate]:
        """
        Full scan of Nifty 50 universe.
        Returns top candidates ranked by master score, descending.
        """
        # Fetch market-wide context once
        nifty_data = get_nifty_index() or {}
        fii_data   = get_fii_dii_data() or {}
        fii_net    = float(fii_data.get("fii_net", 0))
        nifty_chg  = float(nifty_data.get("change_pct", 0))

        # Determine market breadth from Nifty
        if nifty_chg <= -0.5:
            market_breadth = "BEARISH"
        elif nifty_chg >= 0.5:
            market_breadth = "BULLISH"
        else:
            market_breadth = "NEUTRAL"

        logger.info(
            f"Scanning Nifty 50 | Nifty={nifty_chg:+.2f}% "
            f"FII=₹{fii_net:.0f}Cr | Breadth={market_breadth}"
        )

        # Choose scan list
        symbols = TRADING.priority_watchlist[:10] if quick_mode else TRADING.priority_watchlist

        candidates: List[ShortCandidate] = []

        for i, symbol in enumerate(symbols):
            # Safety check: only Nifty 50 stocks
            if not is_nifty50(symbol):
                logger.error(f"REJECTED {symbol} — not in Nifty 50. Bug in priority_watchlist.")
                continue

            try:
                candidate = self._analyse(
                    symbol, market_breadth, fii_net,
                    nifty_chg, intermarket_bias,
                )
                if candidate:
                    candidates.append(candidate)
                    logger.info(
                        f"  [{i+1:02d}/50] {symbol:12s} ✓ "
                        f"score={candidate.score:.3f} | "
                        f"entry=₹{candidate.entry_price:.2f} "
                        f"SL=₹{candidate.stop_loss:.2f} "
                        f"TGT=₹{candidate.target:.2f} "
                        f"R:R={candidate.risk_reward:.1f} | "
                        f"{candidate.reason[:60]}"
                    )
                else:
                    logger.debug(f"  [{i+1:02d}/50] {symbol:12s} — no setup")

                # Polite rate limiting
                if i % 10 == 9:
                    time.sleep(1.0)

            except Exception as e:
                logger.error(f"Scan error [{symbol}]: {e}")
                if self.healer:
                    fix = self.healer.heal(f"Scan error {symbol}: {e}", {"symbol": symbol})
                    logger.debug(f"Healer: {fix.get('solution','skip')}")

        # Sort by score. Take top 3 (max positions).
        candidates.sort(key=lambda c: c.score, reverse=True)
        top = candidates[:TRADING.max_open_positions]

        logger.info(
            f"Scan complete: {len(candidates)} candidates found | "
            f"Top {len(top)} selected | SHORT ONLY"
        )
        for rank, c in enumerate(top, 1):
            logger.info(
                f"  #{rank} {c.symbol} [{c.sector}] — "
                f"score={c.score:.3f} R:R={c.risk_reward:.1f} | {c.reason[:80]}"
            )

        return top

    # ──────────────────────────────────────────────────────────────
    # SINGLE STOCK ANALYSIS
    # ──────────────────────────────────────────────────────────────

    def _analyse(
        self,
        symbol:           str,
        market_breadth:   str,
        fii_net:          float,
        nifty_chg:        float,
        intermarket_bias: float,
    ) -> Optional[ShortCandidate]:
        """Full analysis pipeline for one Nifty 50 stock. Short bias only."""
        import datetime

        # ── Multi-timeframe data ──────────────────────────────────
        df_5m    = get_intraday_ohlcv(symbol, interval="5m",  period="1d")
        df_15m   = get_intraday_ohlcv(symbol, interval="15m", period="5d")
        df_daily = get_historical_ohlcv(symbol, days=60)

        if df_daily is None or len(df_daily) < 20:
            return None

        # ── Technical gate: must show SHORT signal ────────────────
        signal = calculate_all(df_daily, symbol)
        if signal is None:
            return None

        # STRICT: only take SHORT or STRONG_SHORT. NEUTRAL = skip. No longs.
        if signal.signal not in ("SHORT", "STRONG_SHORT"):
            return None

        # ── Live price check ──────────────────────────────────────
        quote = get_live_quote(symbol)
        if quote:
            ltp = float(quote.get("ltp", 0))
            if not (TRADING.min_price <= ltp <= TRADING.max_price):
                return None

        # ── OI data for conviction ────────────────────────────────
        oi_data       = get_oi_data(symbol)
        oi_change_pct = float(oi_data.get("oi_change_pct", 0))

        # ── Build signal list ─────────────────────────────────────
        signals_fired = []
        if signal.is_overbought:           signals_fired.append("RSI_OVERBOUGHT")
        if signal.bearish_divergence:      signals_fired.append("BEARISH_DIVERGENCE")
        if signal.at_resistance:           signals_fired.append("AT_RESISTANCE")
        if signal.volume_confirms:         signals_fired.append("VOLUME_CONFIRMS")
        if signal.macd_histogram < 0:      signals_fired.append("MACD_TURNING_DOWN")
        if signal.bb_position > 0.90:      signals_fired.append("BB_EXTENDED")
        if fii_net < -300:                 signals_fired.append("FII_SELLING")
        if market_breadth == "BEARISH":    signals_fired.append("SECTOR_DOWNTREND")
        if nifty_chg < -0.5:              signals_fired.append("NIFTY_FALLING")
        if oi_change_pct > 5:             signals_fired.append("OI_BUILDUP_BEARISH")

        # ── Build ML feature vector ───────────────────────────────
        now      = datetime.datetime.now()
        hour     = now.hour
        tod      = "EARLY" if hour < 10 else ("LATE" if hour >= 12 else "MID")

        trade_features = {
            "rsi_at_entry":        signal.rsi,
            "macd_histogram":      signal.macd_histogram,
            "bb_position":         signal.bb_position,
            "volume_ratio":        1.8 if signal.volume_confirms else 1.0,
            "gap_pct":             0.0,
            "confidence_score":    signal.confidence,
            "mtf_alignment_count": 1,
            "fii_net_cr":          fii_net,
            "india_vix":           15.0,
            "trade_date":          str(datetime.date.today()),
            "entry_time":          now.strftime("%H:%M"),
            "time_of_day":         tod,
            "sector_trend":        "DOWNTREND" if market_breadth == "BEARISH" else "SIDEWAYS",
            "signals_fired":       signals_fired,
        }

        # ── Master scorer: all 11 intelligence layers ─────────────
        master = master_scorer.score(
            symbol=symbol,
            signals=signals_fired,
            df_5m=df_5m,
            df_15m=df_15m,
            df_1d=df_daily,
            trade_features=trade_features,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            target=signal.target,
            capital=self.capital,
            regime_label=market_breadth,
            intermarket_bias=intermarket_bias,
        )

        # Veto or skip
        if master.vetoed or master.recommendation in ("SKIP", "VETO"):
            logger.debug(f"  {symbol}: {master.recommendation} (score={master.final_score:.3f})")
            return None

        # Minimum score gate
        if master.final_score < TRADING.min_confidence:
            return None

        # ── Position sizing ───────────────────────────────────────
        entry          = signal.entry_price
        sl             = signal.stop_loss
        tgt            = signal.target
        risk_per_share = max(sl - entry, 0.01)
        reward_per_share = max(entry - tgt, 0.01)
        rr             = reward_per_share / risk_per_share

        # Enforce minimum R:R
        if rr < TRADING.min_risk_reward:
            logger.debug(f"  {symbol}: R:R {rr:.2f} < minimum {TRADING.min_risk_reward}")
            return None

        max_risk_rs = self.capital * TRADING.max_risk_per_trade_pct / 100
        std_qty     = max(1, int(max_risk_rs / risk_per_share))
        kelly_qty   = master.recommended_qty or std_qty
        quantity    = min(kelly_qty, int(self.capital * TRADING.max_single_position_pct / 100 / entry))
        quantity    = max(1, quantity)

        risk_amt    = round(quantity * risk_per_share, 2)

        # Build reason string
        supporting = master.supporting_factors[:3]
        reason_parts = supporting[:]
        if master.opposing_factors:
            reason_parts.append(f"⚠ {master.opposing_factors[0]}")

        return ShortCandidate(
            symbol=symbol,
            score=master.final_score,
            sector=get_sector(symbol),
            technical=signal,
            entry_price=round(entry, 2),
            stop_loss=round(sl, 2),
            target=round(tgt, 2),
            quantity=quantity,
            risk_amount=risk_amt,
            risk_reward=round(rr, 2),
            master=master,
            oi_change_pct=oi_change_pct,
            reason=" | ".join(reason_parts),
        )

    # ──────────────────────────────────────────────────────────────
    # LIVE INTRADAY SCAN (called every 5 min during session)
    # ──────────────────────────────────────────────────────────────

    def scan_live(
        self,
        watched_symbols: List[str],
        market_breadth: str = "NEUTRAL",
        fii_net: float = 0.0,
        intermarket_bias: float = 0.0,
    ) -> List[ShortCandidate]:
        """
        Quick scan of actively watched symbols using 5-min candles.
        Used throughout the day to catch fresh setups that weren't
        present at the morning scan.
        """
        candidates = []
        for symbol in watched_symbols:
            if not is_nifty50(symbol):
                continue
            try:
                c = self._analyse(symbol, market_breadth, fii_net, 0.0, intermarket_bias)
                if c:
                    candidates.append(c)
            except Exception:
                pass
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:TRADING.max_open_positions]
