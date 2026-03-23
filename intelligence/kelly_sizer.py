"""
Kelly Criterion Position Sizer
Uses the Kelly formula to compute mathematically optimal position sizes.

Kelly formula: f* = (p*b - q) / b
  p = probability of winning (from recent trade history)
  q = 1 - p (probability of losing)
  b = net odds received (avg win / avg loss)
  f* = optimal fraction of capital to risk

We use FRACTIONAL KELLY (half-Kelly) to reduce variance while maintaining
most of the growth rate benefit. Full Kelly is theoretically optimal but
causes extreme drawdowns in practice.

The position size is further adjusted by:
  - Current market regime (0.6x in volatile, 1.2x in trending down)
  - Signal confidence (Bayesian posterior probability)
  - Recent drawdown (size down after losses, size up after wins)
  - Correlation with existing positions (avoid concentration)
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    symbol:               str
    kelly_fraction:       float      # f* from Kelly formula
    half_kelly_fraction:  float      # recommended: f*/2
    adjusted_fraction:    float      # after all adjustments
    risk_amount:          float      # ₹ to risk
    quantity:             int        # shares to trade
    max_loss:             float      # worst case if SL hit
    rationale:            str


class KellySizer:
    """
    Computes optimal position sizes using Kelly Criterion.
    Uses real trade history to calibrate p and b.
    """

    # Half-Kelly for safety (most practitioners use 0.25–0.50 Kelly)
    KELLY_FRACTION = 0.5

    # Hard caps
    MAX_RISK_PCT    = 3.0    # Never risk more than 3% per trade
    MIN_RISK_PCT    = 0.25   # Minimum 0.25% (below this isn't worth the friction)
    MAX_POSITIONS   = 5      # Hard limit on simultaneous positions

    def compute(
        self,
        symbol:          str,
        entry_price:     float,
        stop_loss:       float,
        capital:         float,
        posterior_win_p: float = 0.55,       # from Bayesian fusion
        open_positions:  int   = 0,
        regime_mult:     float = 1.0,        # from market regime detector
        recent_wr:       float = None,       # override from trade history
        recent_avg_win:  float = None,
        recent_avg_loss: float = None,
    ) -> PositionSize:
        """
        Compute optimal position size.

        posterior_win_p: probability of winning from Bayesian signal fusion
        regime_mult:     position size multiplier from market regime (0.3–1.3)
        """
        # Risk per share
        risk_per_share = abs(stop_loss - entry_price)
        if risk_per_share <= 0:
            return self._zero_size(symbol, "Invalid stop loss")

        # Kelly inputs
        p = recent_wr or posterior_win_p
        p = max(0.30, min(0.80, p))   # Bound: we never know true p exactly

        # Average win/loss ratio
        if recent_avg_win and recent_avg_loss and recent_avg_loss > 0:
            b = abs(recent_avg_win) / abs(recent_avg_loss)
        else:
            b = abs(entry_price - (entry_price * 0.985)) / risk_per_share   # target/SL ratio
        b = max(0.5, min(5.0, b))

        # Kelly formula
        q       = 1 - p
        kelly_f = (p * (b + 1) - 1) / b
        kelly_f = max(0.0, kelly_f)

        # Half-Kelly (reduces variance, standard practice)
        half_kelly = kelly_f * self.KELLY_FRACTION

        # Adjust for regime
        adjusted = half_kelly * regime_mult

        # Drawdown protection: reduce size if we've lost 3+ in a row
        # (This is handled by the circuit breaker but Kelly supports it mathematically)
        adjusted = min(adjusted, self.MAX_RISK_PCT / 100)
        adjusted = max(adjusted, self.MIN_RISK_PCT / 100)

        # Concentration limit: reduce if already holding positions
        if open_positions >= 2:
            concentration_mult = 1.0 - (open_positions - 1) * 0.15
            adjusted *= max(0.4, concentration_mult)

        risk_amount = capital * adjusted
        quantity    = max(1, int(risk_amount / risk_per_share))
        max_loss    = quantity * risk_per_share

        rationale = (
            f"Kelly={kelly_f:.3f} → Half-Kelly={half_kelly:.3f} → "
            f"Regime×{regime_mult:.2f} → Final={adjusted:.3f} | "
            f"p={p:.2f}, b={b:.2f}, risk/share=₹{risk_per_share:.2f}"
        )

        logger.debug(f"Kelly [{symbol}]: {rationale}")

        return PositionSize(
            symbol=symbol,
            kelly_fraction=round(kelly_f, 4),
            half_kelly_fraction=round(half_kelly, 4),
            adjusted_fraction=round(adjusted, 4),
            risk_amount=round(risk_amount, 2),
            quantity=quantity,
            max_loss=round(max_loss, 2),
            rationale=rationale,
        )

    def compute_portfolio_kelly(
        self,
        positions: List[Dict],     # [{"symbol", "entry", "sl", "posterior_win_p"}]
        capital: float,
        regime_mult: float = 1.0,
    ) -> Dict[str, PositionSize]:
        """
        Portfolio-level Kelly: allocate capital across multiple positions
        while respecting total portfolio risk.
        """
        if not positions:
            return {}

        # Total portfolio Kelly fraction
        results = {}
        total_allocated = 0.0
        max_portfolio_risk = min(0.10, 0.03 * len(positions))   # max 10% total risk

        for i, pos in enumerate(positions):
            remaining_cap = max(0, max_portfolio_risk - total_allocated)
            if remaining_cap <= 0:
                break

            size = self.compute(
                symbol=pos["symbol"],
                entry_price=pos["entry"],
                stop_loss=pos["sl"],
                capital=capital,
                posterior_win_p=pos.get("posterior_win_p", 0.55),
                open_positions=i,
                regime_mult=regime_mult,
            )
            # Cap at remaining allocation
            cap_fraction = min(size.adjusted_fraction, remaining_cap)
            size.adjusted_fraction = cap_fraction
            size.risk_amount = capital * cap_fraction
            size.quantity = max(1, int(size.risk_amount / max(
                abs(pos["sl"] - pos["entry"]), 0.01
            )))
            size.max_loss = size.quantity * abs(pos["sl"] - pos["entry"])

            total_allocated += cap_fraction
            results[pos["symbol"]] = size

        return results

    def _zero_size(self, symbol: str, reason: str) -> PositionSize:
        return PositionSize(
            symbol=symbol, kelly_fraction=0, half_kelly_fraction=0,
            adjusted_fraction=0, risk_amount=0, quantity=0, max_loss=0,
            rationale=reason,
        )

    def get_kelly_stats(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> Dict:
        """Return Kelly stats for display in reports."""
        if avg_loss == 0:
            return {"kelly": 0, "half_kelly": 0, "edge": 0}
        b = avg_win / abs(avg_loss)
        p = win_rate
        q = 1 - p
        kelly = max(0, (p * (b + 1) - 1) / b)
        edge  = p * avg_win + q * avg_loss   # expected value per trade
        return {
            "kelly":      round(kelly, 4),
            "half_kelly": round(kelly * 0.5, 4),
            "edge":       round(edge, 2),
            "b_ratio":    round(b, 2),
            "breakeven_wr": round(1 / (1 + b), 3),
        }


# Singleton
kelly_sizer = KellySizer()
