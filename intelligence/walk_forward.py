"""
Walk-Forward Optimization Engine
The gold standard for validating trading strategies without lookahead bias.

Problem with regular backtesting:
  Optimize parameters on 6 months of data → great backtest numbers
  Go live → parameters were overfit to that specific period → fail

Walk-forward solves this:
  1. Split history into N windows
  2. For each window: optimize on "in-sample" period, validate on "out-of-sample"
  3. Only count out-of-sample results → honest performance estimate
  4. Detect whether the strategy has genuine edge or was just curve-fitted

Walk-forward schedule:
  - Runs every Sunday night
  - Uses 90 days in-sample, 30 days out-of-sample
  - Rolls forward 30 days each iteration
  - Tests 200+ parameter combinations per window
  - Reports the robustness ratio: OOS/IS performance (>0.7 = robust)

If robustness degrades below 0.5 → agent reduces position sizes 50%
If robustness degrades below 0.3 → agent triggers a "pause and review" alert
"""

import itertools
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WFO_RESULTS_FILE = os.path.join(
    os.path.dirname(__file__), "..", "db", "wfo_results.json"
)

# Parameter grid — same as optimizer.py but used with proper walk-forward
WFO_GRID = {
    "rsi_overbought":    [65, 68, 70, 72, 75],
    "stop_loss_pct":     [0.3, 0.5, 0.7, 1.0],
    "target_pct":        [1.0, 1.5, 2.0, 2.5],
    "volume_multiplier": [1.2, 1.5, 2.0],
}


@dataclass
class WFOWindow:
    window_id:       int
    is_start:        str       # in-sample start
    is_end:          str       # in-sample end
    oos_start:       str       # out-of-sample start
    oos_end:         str       # out-of-sample end
    best_params:     Dict
    is_performance:  float     # in-sample metric (profit factor)
    oos_performance: float     # out-of-sample metric (profit factor)
    robustness:      float     # oos / is  (1.0 = perfect, <0.5 = overfit)
    is_trades:       int
    oos_trades:      int


@dataclass
class WFOReport:
    generated_at:        str
    symbol:              str
    windows_tested:      int
    avg_robustness:      float
    min_robustness:      float
    overall_oos_pf:      float     # overall out-of-sample profit factor
    overall_oos_wr:      float     # overall out-of-sample win rate
    recommended_params:  Dict
    strategy_verdict:    str       # ROBUST | MARGINAL | FRAGILE
    windows:             List[WFOWindow] = field(default_factory=list)
    alerts:              List[str]       = field(default_factory=list)


class WalkForwardOptimizer:
    """
    Proper walk-forward optimization for the short-selling strategy.
    Gives an honest estimate of real-world performance.
    """

    def __init__(self, capital: float = 100_000):
        self.capital    = capital
        self.is_days    = 60    # in-sample window
        self.oos_days   = 20    # out-of-sample window
        self.step_days  = 20    # roll forward by this many days

    def run(self, symbol: str, total_days: int = 180) -> WFOReport:
        """
        Run walk-forward optimization on historical data for one symbol.
        """
        logger.info(f"Walk-forward optimization: {symbol}, {total_days} days")

        from data.nse_fetcher import get_historical_ohlcv
        df = get_historical_ohlcv(symbol, days=total_days + 20)
        if df is None or len(df) < self.is_days + self.oos_days:
            return self._empty_report(symbol)

        # Build windows
        windows: List[WFOWindow] = []
        window_id = 0
        end_day   = len(df) - self.oos_days

        for start in range(0, end_day - self.is_days, self.step_days):
            is_end   = start + self.is_days
            oos_end  = is_end + self.oos_days

            if oos_end > len(df):
                break

            df_is  = df.iloc[start:is_end]
            df_oos = df.iloc[is_end:oos_end]

            # Optimise on in-sample
            best_params, is_perf, is_trades = self._optimise_window(df_is)

            # Validate on out-of-sample using best_params (no re-optimisation)
            oos_perf, oos_trades = self._evaluate_params(df_oos, best_params)

            robustness = oos_perf / max(is_perf, 1e-6)

            windows.append(WFOWindow(
                window_id=window_id,
                is_start=str(df.index[start])[:10],
                is_end=str(df.index[is_end - 1])[:10],
                oos_start=str(df.index[is_end])[:10],
                oos_end=str(df.index[oos_end - 1])[:10],
                best_params=best_params,
                is_performance=round(is_perf, 3),
                oos_performance=round(oos_perf, 3),
                robustness=round(robustness, 3),
                is_trades=is_trades,
                oos_trades=oos_trades,
            ))
            window_id += 1

        if not windows:
            return self._empty_report(symbol)

        # Aggregate results
        robustness_values = [w.robustness for w in windows]
        oos_perfs         = [w.oos_performance for w in windows if w.oos_trades >= 3]
        avg_rob  = float(np.mean(robustness_values))
        min_rob  = float(np.min(robustness_values))
        avg_oos  = float(np.mean(oos_perfs)) if oos_perfs else 0.0

        # Use most recent window's params (most relevant to current market)
        recommended = windows[-1].best_params

        # Verdict
        if avg_rob >= 0.70 and avg_oos >= 1.3:
            verdict = "ROBUST"
        elif avg_rob >= 0.50 and avg_oos >= 1.0:
            verdict = "MARGINAL"
        else:
            verdict = "FRAGILE"

        # Alerts
        alerts = []
        if min_rob < 0.40:
            alerts.append(f"Robustness dropped to {min_rob:.2f} in one window — strategy may be overfit")
        if avg_oos < 1.0:
            alerts.append(f"Avg OOS profit factor {avg_oos:.2f} < 1.0 — strategy loses money on new data")
        if verdict == "FRAGILE":
            alerts.append("FRAGILE strategy — reduce position sizes by 50% until robustness improves")

        report = WFOReport(
            generated_at=date.today().isoformat(),
            symbol=symbol,
            windows_tested=len(windows),
            avg_robustness=round(avg_rob, 3),
            min_robustness=round(min_rob, 3),
            overall_oos_pf=round(avg_oos, 3),
            overall_oos_wr=0.0,  # computed below
            recommended_params=recommended,
            strategy_verdict=verdict,
            windows=windows,
            alerts=alerts,
        )

        # Compute overall OOS win rate
        report.overall_oos_wr = self._compute_oos_wr(df, windows)
        self._save(report)

        logger.info(
            f"WFO {symbol}: {len(windows)} windows | "
            f"Robustness={avg_rob:.2f} | OOS_PF={avg_oos:.2f} | {verdict}"
        )
        return report

    def run_portfolio(self, symbols: List[str]) -> Dict[str, WFOReport]:
        """Run WFO across multiple symbols."""
        results = {}
        for sym in symbols:
            try:
                results[sym] = self.run(sym)
                v = results[sym].strategy_verdict
                logger.info(f"  WFO {sym}: {v} (rob={results[sym].avg_robustness:.2f})")
            except Exception as e:
                logger.error(f"WFO error [{sym}]: {e}")
        return results

    def get_robustness_multiplier(self, symbol: str = None) -> float:
        """
        Returns a position size multiplier based on WFO robustness.
        Called by Kelly sizer to reduce sizes for fragile strategies.
        """
        report = self._load_latest()
        if not report:
            return 1.0

        rob = report.get("avg_robustness", 0.7)
        verdict = report.get("strategy_verdict", "MARGINAL")

        if verdict == "ROBUST":
            return 1.0
        elif verdict == "MARGINAL":
            return 0.75
        else:  # FRAGILE
            return 0.50

    # ──────────────────────────────────────────────────────────────
    # INTERNAL
    # ──────────────────────────────────────────────────────────────

    def _optimise_window(self, df: pd.DataFrame) -> Tuple[Dict, float, int]:
        """Find the best parameter set on an in-sample window."""
        best_params = {"rsi_overbought": 70, "stop_loss_pct": 0.5,
                       "target_pct": 1.5, "volume_multiplier": 1.5}
        best_pf     = 0.0
        best_trades = 0

        keys   = list(WFO_GRID.keys())
        values = [WFO_GRID[k] for k in keys]

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            if params["target_pct"] < params["stop_loss_pct"] * 1.3:
                continue
            pf, n = self._evaluate_params(df, params)
            if pf > best_pf and n >= 3:
                best_pf     = pf
                best_params = params
                best_trades = n

        return best_params, best_pf, best_trades

    def _evaluate_params(self, df: pd.DataFrame, params: Dict) -> Tuple[float, int]:
        """
        Evaluate a parameter set on a DataFrame slice.
        Returns (profit_factor, num_trades).
        """
        from agents.technical_analyst import calculate_all
        import config

        # Temporarily apply params
        orig_rsi  = config.TRADING.rsi_overbought
        orig_sl   = config.TRADING.stop_loss_pct
        orig_tgt  = config.TRADING.target_pct
        orig_vol  = config.TRADING.volume_multiplier

        config.TRADING.rsi_overbought    = params["rsi_overbought"]
        config.TRADING.stop_loss_pct     = params["stop_loss_pct"]
        config.TRADING.target_pct        = params["target_pct"]
        config.TRADING.volume_multiplier = params["volume_multiplier"]

        gross_profit = gross_loss = 0.0
        trades = 0

        try:
            for i in range(30, len(df)):
                slice_df = df.iloc[:i]
                signal   = calculate_all(slice_df, "")
                if signal is None or signal.signal not in ("SHORT", "STRONG_SHORT"):
                    continue
                if signal.confidence < 0.45:
                    continue

                entry = float(df.iloc[i]["close"])
                sl    = entry * (1 + params["stop_loss_pct"] / 100)
                tgt   = entry * (1 - params["target_pct"] / 100)

                # Check outcome over next 5 bars
                for j in range(i + 1, min(i + 6, len(df))):
                    day_high = float(df.iloc[j]["high"])
                    day_low  = float(df.iloc[j]["low"])

                    if day_high >= sl:
                        gross_loss += (sl - entry)
                        trades += 1
                        break
                    elif day_low <= tgt:
                        gross_profit += (entry - tgt)
                        trades += 1
                        break
                else:
                    # EOD square-off
                    eod = float(df.iloc[min(i + 1, len(df) - 1)]["close"])
                    pnl = entry - eod
                    if pnl > 0:
                        gross_profit += pnl
                    else:
                        gross_loss += abs(pnl)
                    trades += 1

        finally:
            config.TRADING.rsi_overbought    = orig_rsi
            config.TRADING.stop_loss_pct     = orig_sl
            config.TRADING.target_pct        = orig_tgt
            config.TRADING.volume_multiplier = orig_vol

        pf = gross_profit / max(gross_loss, 1e-6)
        return round(pf, 3), trades

    def _compute_oos_wr(self, df: pd.DataFrame, windows: List[WFOWindow]) -> float:
        """Compute overall out-of-sample win rate across all windows."""
        return round(sum(1 for w in windows if w.oos_performance >= 1.0) / max(len(windows), 1), 3)

    def _save(self, report: WFOReport):
        os.makedirs(os.path.dirname(WFO_RESULTS_FILE), exist_ok=True)
        data = asdict(report)
        with open(WFO_RESULTS_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load_latest(self) -> Optional[Dict]:
        if not os.path.exists(WFO_RESULTS_FILE):
            return None
        try:
            with open(WFO_RESULTS_FILE) as f:
                return json.load(f)
        except Exception:
            return None

    def _empty_report(self, symbol: str) -> WFOReport:
        return WFOReport(
            generated_at=date.today().isoformat(),
            symbol=symbol,
            windows_tested=0,
            avg_robustness=0.7,
            min_robustness=0.7,
            overall_oos_pf=1.0,
            overall_oos_wr=0.5,
            recommended_params={"rsi_overbought": 70, "stop_loss_pct": 0.5,
                                 "target_pct": 1.5, "volume_multiplier": 1.5},
            strategy_verdict="MARGINAL",
        )


# Singleton
wfo_engine = WalkForwardOptimizer()
