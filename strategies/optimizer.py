"""
Strategy Parameter Optimizer
Runs a grid search over key parameters to find the best settings
for the short selling strategy on historical NSE data.
Uses backtester.py — completely free, no paid data needed.

Usage:
  python -m strategies.optimizer --symbols RELIANCE,TCS,INFY --days 90
"""

import argparse
import itertools
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from backtest.backtester import Backtester
from data.nse_fetcher import get_historical_ohlcv

logger = logging.getLogger(__name__)


@dataclass
class ParamSet:
    rsi_overbought: float
    stop_loss_pct: float
    target_pct: float
    volume_multiplier: float
    rsi_period: int


@dataclass
class OptimizationResult:
    params: ParamSet
    avg_win_rate: float
    avg_total_pnl: float
    avg_profit_factor: float
    avg_sharpe: float
    avg_max_drawdown: float
    total_trades: int
    score: float           # composite score — higher is better
    symbols_tested: List[str]


class StrategyOptimizer:
    """
    Grid search optimizer for the short selling strategy parameters.
    Evaluates each parameter combination on historical data
    and ranks by composite performance score.
    """

    # Grid search space — adjust ranges to explore more combinations
    PARAM_GRID = {
        "rsi_overbought":    [65.0, 68.0, 70.0, 72.0, 75.0],
        "stop_loss_pct":     [0.3, 0.5, 0.7, 1.0],
        "target_pct":        [1.0, 1.5, 2.0, 2.5],
        "volume_multiplier": [1.2, 1.5, 2.0],
        "rsi_period":        [10, 14, 21],
    }

    # Constraints (filters out parameter sets that violate these)
    CONSTRAINTS = {
        "min_risk_reward": 1.5,      # target must be at least 1.5× the stop loss
        "max_stop_loss":   1.2,      # don't allow SL > 1.2%
    }

    def __init__(self, capital: float = 100_000):
        self.capital = capital

    def optimize(
        self,
        symbols: List[str],
        days: int = 90,
        top_n: int = 5,
    ) -> List[OptimizationResult]:
        """
        Run grid search over all parameter combinations.
        Returns top_n results sorted by composite score.
        """
        param_combinations = self._generate_combinations()
        logger.info(
            f"Optimizer: {len(param_combinations)} combinations × "
            f"{len(symbols)} symbols × {days} days"
        )

        # Pre-fetch all historical data once (avoid repeated API calls)
        logger.info("Pre-fetching historical data...")
        historical_data = {}
        for sym in symbols:
            df = get_historical_ohlcv(sym, days=days + 60)
            if df is not None and len(df) > 60:
                historical_data[sym] = df
                logger.debug(f"  {sym}: {len(df)} bars")
            else:
                logger.warning(f"  {sym}: insufficient data, skipping")

        if not historical_data:
            logger.error("No data fetched — optimizer cannot run")
            return []

        results: List[OptimizationResult] = []
        tested_symbols = list(historical_data.keys())

        for i, params in enumerate(param_combinations):
            if (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i+1}/{len(param_combinations)} combinations")

            result = self._evaluate_params(params, historical_data, days, tested_symbols)
            if result:
                results.append(result)

        results.sort(key=lambda r: r.score, reverse=True)
        top = results[:top_n]

        logger.info(f"\nTop {top_n} parameter sets:")
        for rank, r in enumerate(top, 1):
            logger.info(
                f"  #{rank}: score={r.score:.3f} | "
                f"WR={r.avg_win_rate:.0f}% | "
                f"PF={r.avg_profit_factor:.2f} | "
                f"RSI>{r.params.rsi_overbought} | "
                f"SL={r.params.stop_loss_pct}% | "
                f"TGT={r.params.target_pct}%"
            )

        return top

    def apply_best(self, result: OptimizationResult):
        """Apply the best parameter set to config.py."""
        import config
        config.TRADING.rsi_overbought    = result.params.rsi_overbought
        config.TRADING.stop_loss_pct     = result.params.stop_loss_pct
        config.TRADING.target_pct        = result.params.target_pct
        config.TRADING.volume_multiplier = result.params.volume_multiplier
        config.TRADING.rsi_period        = result.params.rsi_period
        logger.info(f"Applied optimized parameters: {result.params}")

    def save_results(self, results: List[OptimizationResult], output_dir: str = "reports"):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"optimization_{timestamp}.json")
        data = [asdict(r) for r in results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Optimization results saved: {path}")
        return path

    # ──────────────────────────────────────────────────────────────
    # INTERNAL
    # ──────────────────────────────────────────────────────────────

    def _generate_combinations(self) -> List[ParamSet]:
        """Generate all valid parameter combinations."""
        all_combos = []
        keys = list(self.PARAM_GRID.keys())
        values = [self.PARAM_GRID[k] for k in keys]

        for combo in itertools.product(*values):
            params = ParamSet(**dict(zip(keys, combo)))
            if self._is_valid(params):
                all_combos.append(params)

        return all_combos

    def _is_valid(self, params: ParamSet) -> bool:
        """Check if a parameter set satisfies constraints."""
        rr = params.target_pct / params.stop_loss_pct
        if rr < self.CONSTRAINTS["min_risk_reward"]:
            return False
        if params.stop_loss_pct > self.CONSTRAINTS["max_stop_loss"]:
            return False
        return True

    def _evaluate_params(
        self,
        params: ParamSet,
        historical_data: Dict[str, pd.DataFrame],
        days: int,
        symbols: List[str],
    ) -> Optional[OptimizationResult]:
        """Evaluate one parameter set across all symbols."""
        import config

        # Temporarily override config
        original = {
            "rsi_overbought":    config.TRADING.rsi_overbought,
            "stop_loss_pct":     config.TRADING.stop_loss_pct,
            "target_pct":        config.TRADING.target_pct,
            "volume_multiplier": config.TRADING.volume_multiplier,
            "rsi_period":        config.TRADING.rsi_period,
        }
        config.TRADING.rsi_overbought    = params.rsi_overbought
        config.TRADING.stop_loss_pct     = params.stop_loss_pct
        config.TRADING.target_pct        = params.target_pct
        config.TRADING.volume_multiplier = params.volume_multiplier
        config.TRADING.rsi_period        = params.rsi_period

        try:
            bt = Backtester(capital=self.capital)
            all_win_rates, all_pnls, all_pfs, all_sharpes, all_dds, all_trades = (
                [], [], [], [], [], []
            )

            for sym, df in historical_data.items():
                try:
                    result = bt.run(sym, days=days)
                    if result.total_trades < 2:
                        continue
                    all_win_rates.append(result.win_rate_pct)
                    all_pnls.append(result.total_pnl)
                    all_pfs.append(result.profit_factor)
                    all_sharpes.append(result.sharpe_ratio)
                    all_dds.append(result.max_drawdown)
                    all_trades.append(result.total_trades)
                except Exception:
                    continue

            if not all_win_rates:
                return None

            avg_wr  = sum(all_win_rates) / len(all_win_rates)
            avg_pnl = sum(all_pnls) / len(all_pnls)
            avg_pf  = sum(all_pfs) / len(all_pfs)
            avg_sh  = sum(all_sharpes) / len(all_sharpes)
            avg_dd  = sum(all_dds) / len(all_dds)

            # Composite score (weighted)
            score = (
                avg_wr / 100 * 0.30 +
                min(avg_pf / 3, 1.0) * 0.30 +
                min(max(avg_sh / 2, 0), 1.0) * 0.20 +
                max(0, 1 - avg_dd / 20) * 0.20
            )

            return OptimizationResult(
                params=params,
                avg_win_rate=round(avg_wr, 2),
                avg_total_pnl=round(avg_pnl, 2),
                avg_profit_factor=round(avg_pf, 3),
                avg_sharpe=round(avg_sh, 3),
                avg_max_drawdown=round(avg_dd, 2),
                total_trades=sum(all_trades),
                score=round(score, 4),
                symbols_tested=symbols,
            )
        finally:
            # Always restore original config
            for k, v in original.items():
                setattr(config.TRADING, k, v)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = argparse.ArgumentParser(description="Tej Strategy Optimizer")
    parser.add_argument("--symbols", default="RELIANCE,TCS,INFY,HDFCBANK,SBIN",
                        help="Comma-separated NSE symbols")
    parser.add_argument("--days",    type=int, default=90, help="Backtest period")
    parser.add_argument("--top",     type=int, default=5,  help="Top N results to show")
    parser.add_argument("--apply",   action="store_true",  help="Apply best params to config")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    opt = StrategyOptimizer()
    results = opt.optimize(symbols=symbols, days=args.days, top_n=args.top)

    if results:
        opt.save_results(results)
        if args.apply:
            opt.apply_best(results[0])
            logger.info(f"Best params applied: {results[0].params}")
