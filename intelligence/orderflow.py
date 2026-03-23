"""
Order Flow Analyser
Detects institutional buying or selling pressure from publicly available NSE data.
No Level-2 order book needed — we infer order flow from:

1. VWAP deviation          — price above VWAP suggests buyer control, below = sellers
2. Bid/Ask imbalance       — from NSE market depth (top 5 bids/asks)
3. Tick direction ratio    — ratio of up-ticks to down-ticks in recent bars
4. Volume-weighted delta   — up-volume vs down-volume within each candle
5. Large trade detection   — candles with volume 3x+ average = institutional prints
6. Absorption candle       — high volume but small move = absorption (reversal warning)
7. Effort vs result        — large effort (volume) with small result (price move) = exhaustion

For short selling, we want to see:
  - Price BELOW VWAP (sellers in control)
  - Bid/Ask imbalance favouring sellers (more sell orders)
  - Down-ticks dominating
  - Absorption at highs (exhausted buyers)
  - Large sell prints at resistance
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)


@dataclass
class OrderFlowSignal:
    symbol:                str
    vwap:                  float
    price_vs_vwap:         float       # % above/below VWAP (negative = below = bearish)
    bid_ask_imbalance:     float       # -1 (sellers dominate) to +1 (buyers dominate)
    tick_direction_ratio:  float       # 0=all down-ticks, 1=all up-ticks
    volume_delta:          float       # up-vol minus down-vol (negative = selling pressure)
    absorption_detected:   bool        # high vol small move = exhaustion
    large_sell_prints:     int         # number of institutional-sized sell candles
    effort_vs_result:      str         # "EXHAUSTION" | "STRONG_MOVE" | "NORMAL"
    orderflow_score:       float       # -1 (strong selling) to +1 (strong buying)
    bearish_signal:        bool        # True if order flow confirms short thesis
    summary:               str


class OrderFlowAnalyser:
    """
    Analyses order flow from OHLCV data.
    Uses publicly available data only — no paid feed required.
    """

    def analyse(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        bid_quantity: int = 0,
        ask_quantity: int = 0,
    ) -> Optional[OrderFlowSignal]:
        """
        Full order flow analysis.
        df: OHLCV DataFrame (at least 20 bars, 5m candles preferred)
        bid_quantity / ask_quantity: from NSE live quote market depth (if available)
        """
        if df is None or len(df) < 10:
            return None

        try:
            df = df.copy()
            vwap_val      = self._compute_vwap(df)
            close         = float(df["close"].iloc[-1])
            price_vs_vwap = (close - vwap_val) / vwap_val * 100

            tick_ratio    = self._tick_direction_ratio(df)
            vol_delta     = self._volume_delta(df)
            absorption    = self._detect_absorption(df)
            large_sells   = self._count_large_sell_prints(df)
            effort_result = self._effort_vs_result(df)
            bid_ask_imb   = self._bid_ask_imbalance(bid_quantity, ask_quantity)

            # Composite order flow score
            score = 0.0
            score += _norm(price_vs_vwap, -3, 3) * (-0.30)  # below VWAP = negative → bearish
            score += (0.5 - tick_ratio) * 0.25              # more down-ticks → negative
            score += _norm(vol_delta, -500000, 500000) * 0.20
            score -= bid_ask_imb * 0.15                     # seller imbalance → negative
            if absorption:
                score -= 0.15   # absorption at highs = buyers exhausted
            if large_sells >= 2:
                score -= 0.10   # institutional selling

            score = round(max(-1.0, min(1.0, score)), 3)
            bearish = score < -0.15

            parts = []
            if price_vs_vwap < -0.5:
                parts.append(f"below VWAP {price_vs_vwap:.2f}%")
            if tick_ratio < 0.40:
                parts.append(f"down-ticks dominant ({tick_ratio:.0%})")
            if absorption:
                parts.append("absorption (buyer exhaustion)")
            if large_sells >= 2:
                parts.append(f"{large_sells} large sell prints")
            if effort_result == "EXHAUSTION":
                parts.append("effort/result exhaustion")

            return OrderFlowSignal(
                symbol=symbol,
                vwap=round(vwap_val, 2),
                price_vs_vwap=round(price_vs_vwap, 3),
                bid_ask_imbalance=round(bid_ask_imb, 3),
                tick_direction_ratio=round(tick_ratio, 3),
                volume_delta=round(vol_delta),
                absorption_detected=absorption,
                large_sell_prints=large_sells,
                effort_vs_result=effort_result,
                orderflow_score=score,
                bearish_signal=bearish,
                summary="; ".join(parts) if parts else "neutral order flow",
            )

        except Exception as e:
            logger.error(f"OrderFlow error [{symbol}]: {e}")
            return None

    # ──────────────────────────────────────────────────────────────
    # VWAP
    # ──────────────────────────────────────────────────────────────

    def _compute_vwap(self, df: pd.DataFrame) -> float:
        """Volume-weighted average price for today's session."""
        typical = (df["high"] + df["low"] + df["close"]) / 3
        vol     = df["volume"].replace(0, 1)
        return float((typical * vol).sum() / vol.sum())

    def get_vwap_bands(self, df: pd.DataFrame, num_std: float = 1.5) -> Dict[str, float]:
        """VWAP with upper and lower bands (std deviation bands)."""
        typical = (df["high"] + df["low"] + df["close"]) / 3
        vol     = df["volume"].replace(0, 1)
        vwap    = float((typical * vol).sum() / vol.sum())
        std     = float(typical.std())
        return {
            "vwap":    round(vwap, 2),
            "upper_1": round(vwap + std, 2),
            "upper_2": round(vwap + std * 2, 2),
            "lower_1": round(vwap - std, 2),
            "lower_2": round(vwap - std * 2, 2),
        }

    # ──────────────────────────────────────────────────────────────
    # TICK DIRECTION
    # ──────────────────────────────────────────────────────────────

    def _tick_direction_ratio(self, df: pd.DataFrame) -> float:
        """
        Ratio of bars that closed higher than previous bar.
        High = buyers; Low = sellers.
        """
        closes  = df["close"].values
        changes = np.diff(closes)
        up_ticks   = np.sum(changes > 0)
        total_ticks = len(changes)
        return up_ticks / max(total_ticks, 1)

    # ──────────────────────────────────────────────────────────────
    # VOLUME DELTA (estimated from OHLCV)
    # ──────────────────────────────────────────────────────────────

    def _volume_delta(self, df: pd.DataFrame, lookback: int = 10) -> float:
        """
        Estimate buy vs sell volume from OHLCV.
        Method: proportion of candle range covered by close determines buy/sell split.
        close near high → most volume was buying; close near low → selling.
        """
        recent = df.tail(lookback)
        buy_vol = sell_vol = 0.0
        for _, row in recent.iterrows():
            rng = float(row["high"]) - float(row["low"])
            if rng < 1e-6:
                continue
            buy_ratio  = (float(row["close"]) - float(row["low"])) / rng
            sell_ratio = 1 - buy_ratio
            vol = float(row["volume"])
            buy_vol  += vol * buy_ratio
            sell_vol += vol * sell_ratio
        return buy_vol - sell_vol   # positive = net buying, negative = net selling

    # ──────────────────────────────────────────────────────────────
    # ABSORPTION
    # ──────────────────────────────────────────────────────────────

    def _detect_absorption(self, df: pd.DataFrame) -> bool:
        """
        Absorption: high volume but tiny price move at a recent high.
        Indicates buyers used up their ammunition without moving price.
        Classic Wyckoff sign of weakness.
        """
        if len(df) < 5:
            return False
        recent = df.tail(5)
        vol_avg = float(df["volume"].tail(20).mean())
        price_avg_move = float((df["high"] - df["low"]).tail(20).mean())

        for _, row in recent.iterrows():
            bar_vol  = float(row["volume"])
            bar_move = float(row["high"]) - float(row["low"])
            if bar_vol >= vol_avg * 2.5 and bar_move < price_avg_move * 0.5:
                return True
        return False

    # ──────────────────────────────────────────────────────────────
    # LARGE PRINTS
    # ──────────────────────────────────────────────────────────────

    def _count_large_sell_prints(self, df: pd.DataFrame, vol_threshold: float = 3.0) -> int:
        """
        Count candles with volume > threshold × average AND bearish (close < open).
        These are institutional-sized sell executions.
        """
        if len(df) < 10:
            return 0
        vol_avg = float(df["volume"].tail(20).mean())
        recent  = df.tail(8)
        count   = 0
        for _, row in recent.iterrows():
            if (float(row["volume"]) >= vol_avg * vol_threshold and
                    float(row["close"]) < float(row["open"])):
                count += 1
        return count

    # ──────────────────────────────────────────────────────────────
    # EFFORT VS RESULT
    # ──────────────────────────────────────────────────────────────

    def _effort_vs_result(self, df: pd.DataFrame) -> str:
        """
        Compare volume (effort) to price move (result) over last 3 bars.
        High effort, small result = exhaustion.
        High effort, large result = strong genuine move.
        """
        if len(df) < 4:
            return "NORMAL"
        recent    = df.tail(3)
        vol_ratio = float(recent["volume"].sum() / df["volume"].tail(20).mean())
        price_move = abs(float(recent["close"].iloc[-1]) - float(recent["close"].iloc[0]))
        avg_move   = float((df["high"] - df["low"]).tail(20).mean())

        if vol_ratio >= 2.5 and price_move < avg_move * 0.4:
            return "EXHAUSTION"
        if vol_ratio >= 2.0 and price_move > avg_move * 1.5:
            return "STRONG_MOVE"
        return "NORMAL"

    # ──────────────────────────────────────────────────────────────
    # BID/ASK IMBALANCE
    # ──────────────────────────────────────────────────────────────

    def _bid_ask_imbalance(self, bid_qty: int, ask_qty: int) -> float:
        """
        -1 = completely seller-dominated
        +1 = completely buyer-dominated
        """
        total = bid_qty + ask_qty
        if total == 0:
            return 0.0
        return (bid_qty - ask_qty) / total


def _norm(val: float, lo: float, hi: float) -> float:
    """Normalise val to [-1, 1] range."""
    if hi == lo:
        return 0.0
    return max(-1.0, min(1.0, (val - lo) / (hi - lo) * 2 - 1))


# Singleton
orderflow_analyser = OrderFlowAnalyser()
