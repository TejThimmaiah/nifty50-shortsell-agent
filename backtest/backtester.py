"""
Backtesting Engine
Validates the short selling strategy on historical NSE data
before any real money is deployed.
Uses yfinance historical data — completely free.

Usage:
  python -m backtest.backtester --symbol RELIANCE --days 90
  python -m backtest.backtester --universe nifty50 --days 180
"""

import argparse
import logging
import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from data.nse_fetcher import get_historical_ohlcv
from agents.technical_analyst import calculate_all
from config import TRADING

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    stop_loss: float
    target: float
    quantity: int
    pnl: float
    pnl_pct: float
    exit_reason: str            # "TARGET", "STOP_LOSS", "EOD_SQUAREOFF", "SIGNAL_GONE"
    signal_confidence: float
    holding_bars: int           # number of 5-min candles held


@dataclass
class BacktestResult:
    symbol: str
    period_days: int
    start_date: str
    end_date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    total_pnl: float
    total_pnl_pct: float
    avg_win: float
    avg_loss: float
    profit_factor: float        # gross profit / gross loss
    max_drawdown: float
    sharpe_ratio: float
    best_trade_pnl: float
    worst_trade_pnl: float
    avg_holding_bars: float
    trades: List[BacktestTrade] = field(default_factory=list)


class Backtester:
    """
    Event-driven backtester for the intraday short selling strategy.
    Simulates bar-by-bar execution on historical 5-minute data.
    """

    def __init__(self, capital: float = None, commission_per_trade: float = 20.0):
        self.capital = capital or TRADING.total_capital
        self.commission = commission_per_trade   # ₹20 per trade (Zerodha flat fee)

    # ──────────────────────────────────────────────────────────────
    # MAIN BACKTEST
    # ──────────────────────────────────────────────────────────────

    def run(self, symbol: str, days: int = 90) -> BacktestResult:
        """
        Run full backtest for one symbol over `days` calendar days.
        Uses daily OHLCV for signal generation, simulates intraday execution.
        """
        logger.info(f"Backtesting {symbol} over {days} days...")

        df = get_historical_ohlcv(symbol, days=days + 60)   # +60 for indicator warmup
        if df is None or len(df) < 60:
            logger.error(f"Insufficient historical data for {symbol}")
            return self._empty_result(symbol, days)

        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        end_date   = datetime.now().strftime("%Y-%m-%d")

        # Filter to backtest period
        df_full = df.copy()
        df      = df[df.index >= start_date].copy()

        trades: List[BacktestTrade] = []
        equity_curve: List[float] = [self.capital]
        running_capital = self.capital

        for i in range(30, len(df)):  # Need 30 bars of history for indicators
            # Use all data up to current bar (no lookahead)
            df_so_far = pd.concat([df_full.iloc[:-len(df)+i], df.iloc[:i]])

            signal = calculate_all(df_so_far, symbol)
            if signal is None:
                continue

            if signal.signal not in ("STRONG_SHORT", "SHORT"):
                continue

            if signal.confidence < 0.45:
                continue

            # Simulate trade execution
            current_bar = df.iloc[i]
            entry_price = float(current_bar["close"])
            stop_loss   = round(entry_price * (1 + TRADING.stop_loss_pct / 100), 2)
            target      = round(entry_price * (1 - TRADING.target_pct / 100), 2)

            # Position sizing
            risk_per_share = stop_loss - entry_price
            max_risk = running_capital * TRADING.max_risk_per_trade_pct / 100
            quantity = max(1, int(max_risk / max(risk_per_share, 0.01)))

            # Simulate trade outcome (look forward in daily data)
            trade = self._simulate_trade(
                df=df,
                start_idx=i,
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target=target,
                quantity=quantity,
                confidence=signal.confidence,
            )

            if trade:
                # Deduct commission
                trade.pnl -= (self.commission * 2)  # entry + exit
                running_capital += trade.pnl
                equity_curve.append(running_capital)
                trades.append(trade)
                logger.debug(
                    f"  [{trade.entry_date}] {symbol}: {trade.exit_reason} "
                    f"| P&L ₹{trade.pnl:.0f} ({trade.pnl_pct:.2f}%)"
                )

        return self._compute_results(symbol, days, start_date, end_date, trades, equity_curve)

    def run_universe(self, symbols: List[str], days: int = 90) -> Dict[str, BacktestResult]:
        """Backtest multiple symbols and return combined results."""
        results = {}
        for symbol in symbols:
            try:
                result = self.run(symbol, days)
                results[symbol] = result
                logger.info(
                    f"  {symbol}: {result.win_rate_pct:.0f}% WR | "
                    f"P&L ₹{result.total_pnl:.0f} | PF {result.profit_factor:.2f}"
                )
            except Exception as e:
                logger.error(f"Backtest error [{symbol}]: {e}")
        return results

    # ──────────────────────────────────────────────────────────────
    # TRADE SIMULATION
    # ──────────────────────────────────────────────────────────────

    def _simulate_trade(
        self,
        df: pd.DataFrame,
        start_idx: int,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        target: float,
        quantity: int,
        confidence: float,
    ) -> Optional[BacktestTrade]:
        """
        Simulate a short trade by walking forward bar by bar.
        Uses daily OHLCV — assumes worst-case intraday movement:
        SL is hit if high >= SL, target hit if low <= target.
        """
        entry_bar  = df.iloc[start_idx]
        entry_date = str(df.index[start_idx])[:10]

        for j in range(start_idx + 1, min(start_idx + 6, len(df))):   # Max 5 trading days
            bar  = df.iloc[j]
            high = float(bar["high"])
            low  = float(bar["low"])
            close= float(bar["close"])
            exit_date = str(df.index[j])[:10]
            holding_bars = j - start_idx

            # Check stop loss (hit if high >= SL price in intraday)
            if high >= stop_loss:
                exit_price = stop_loss
                pnl = (entry_price - exit_price) * quantity
                return BacktestTrade(
                    symbol=symbol,
                    entry_date=entry_date,
                    exit_date=exit_date,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    stop_loss=stop_loss,
                    target=target,
                    quantity=quantity,
                    pnl=round(pnl, 2),
                    pnl_pct=round((pnl / (entry_price * quantity)) * 100, 3),
                    exit_reason="STOP_LOSS",
                    signal_confidence=confidence,
                    holding_bars=holding_bars,
                )

            # Check target (hit if low <= target)
            if low <= target:
                exit_price = target
                pnl = (entry_price - exit_price) * quantity
                return BacktestTrade(
                    symbol=symbol,
                    entry_date=entry_date,
                    exit_date=exit_date,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    stop_loss=stop_loss,
                    target=target,
                    quantity=quantity,
                    pnl=round(pnl, 2),
                    pnl_pct=round((pnl / (entry_price * quantity)) * 100, 3),
                    exit_reason="TARGET",
                    signal_confidence=confidence,
                    holding_bars=holding_bars,
                )

        # EOD square-off (last checked bar close)
        if start_idx + 1 < len(df):
            exit_bar   = df.iloc[min(start_idx + 1, len(df) - 1)]
            exit_price = float(exit_bar["close"])
            pnl = (entry_price - exit_price) * quantity
            return BacktestTrade(
                symbol=symbol,
                entry_date=entry_date,
                exit_date=str(df.index[min(start_idx + 1, len(df) - 1)])[:10],
                entry_price=entry_price,
                exit_price=exit_price,
                stop_loss=stop_loss,
                target=target,
                quantity=quantity,
                pnl=round(pnl, 2),
                pnl_pct=round((pnl / (entry_price * quantity)) * 100, 3),
                exit_reason="EOD_SQUAREOFF",
                signal_confidence=confidence,
                holding_bars=1,
            )
        return None

    # ──────────────────────────────────────────────────────────────
    # RESULT CALCULATION
    # ──────────────────────────────────────────────────────────────

    def _compute_results(
        self,
        symbol: str,
        days: int,
        start_date: str,
        end_date: str,
        trades: List[BacktestTrade],
        equity_curve: List[float],
    ) -> BacktestResult:
        if not trades:
            return self._empty_result(symbol, days, start_date, end_date)

        winners = [t for t in trades if t.pnl > 0]
        losers  = [t for t in trades if t.pnl <= 0]

        total_pnl      = sum(t.pnl for t in trades)
        gross_profit   = sum(t.pnl for t in winners) if winners else 0
        gross_loss     = abs(sum(t.pnl for t in losers)) if losers else 1
        profit_factor  = round(gross_profit / max(gross_loss, 1), 2)

        # Max drawdown
        peak = equity_curve[0]
        max_dd = 0.0
        for val in equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio (annualized, simplified)
        daily_returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe = 0.0
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = round(
                (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5), 2
            )

        return BacktestResult(
            symbol=symbol,
            period_days=days,
            start_date=start_date,
            end_date=end_date,
            total_trades=len(trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate_pct=round(len(winners) / len(trades) * 100, 1),
            total_pnl=round(total_pnl, 2),
            total_pnl_pct=round(total_pnl / self.capital * 100, 2),
            avg_win=round(sum(t.pnl for t in winners) / max(len(winners), 1), 2),
            avg_loss=round(sum(t.pnl for t in losers) / max(len(losers), 1), 2),
            profit_factor=profit_factor,
            max_drawdown=round(max_dd * 100, 2),
            sharpe_ratio=sharpe,
            best_trade_pnl=max(t.pnl for t in trades),
            worst_trade_pnl=min(t.pnl for t in trades),
            avg_holding_bars=round(sum(t.holding_bars for t in trades) / len(trades), 1),
            trades=trades,
        )

    def _empty_result(self, symbol, days, start="", end="") -> BacktestResult:
        return BacktestResult(
            symbol=symbol, period_days=days, start_date=start, end_date=end,
            total_trades=0, winning_trades=0, losing_trades=0, win_rate_pct=0,
            total_pnl=0, total_pnl_pct=0, avg_win=0, avg_loss=0,
            profit_factor=0, max_drawdown=0, sharpe_ratio=0,
            best_trade_pnl=0, worst_trade_pnl=0, avg_holding_bars=0,
        )

    # ──────────────────────────────────────────────────────────────
    # REPORTING
    # ──────────────────────────────────────────────────────────────

    def print_summary(self, result: BacktestResult):
        print("\n" + "=" * 55)
        print(f"  BACKTEST RESULTS — {result.symbol}")
        print("=" * 55)
        print(f"  Period       : {result.start_date} → {result.end_date}")
        print(f"  Total trades : {result.total_trades}")
        print(f"  Win rate     : {result.win_rate_pct:.1f}%")
        print(f"  Total P&L    : ₹{result.total_pnl:,.0f} ({result.total_pnl_pct:.2f}%)")
        print(f"  Avg win      : ₹{result.avg_win:,.0f}")
        print(f"  Avg loss     : ₹{result.avg_loss:,.0f}")
        print(f"  Profit factor: {result.profit_factor:.2f}")
        print(f"  Max drawdown : {result.max_drawdown:.2f}%")
        print(f"  Sharpe ratio : {result.sharpe_ratio:.2f}")
        print(f"  Best trade   : ₹{result.best_trade_pnl:,.0f}")
        print(f"  Worst trade  : ₹{result.worst_trade_pnl:,.0f}")
        print("=" * 55)

    def save_report(self, result: BacktestResult, output_dir: str = "reports"):
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(
            output_dir,
            f"backtest_{result.symbol}_{result.start_date}_{result.end_date}.json"
        )
        data = asdict(result)
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Backtest report saved: {filename}")
        return filename


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = argparse.ArgumentParser(description="Tej Backtester")
    parser.add_argument("--symbol",   default="RELIANCE", help="NSE symbol to backtest")
    parser.add_argument("--days",     type=int, default=90, help="Lookback days")
    parser.add_argument("--capital",  type=float, default=100000, help="Starting capital")
    parser.add_argument("--universe", help="backtest a predefined universe: nifty50|priority")
    args = parser.parse_args()

    bt = Backtester(capital=args.capital)

    if args.universe == "priority":
        from config import TRADING
        results = bt.run_universe(TRADING.priority_watchlist[:10], days=args.days)
        for sym, res in results.items():
            bt.print_summary(res)
            bt.save_report(res)
    else:
        result = bt.run(args.symbol, days=args.days)
        bt.print_summary(result)
        bt.save_report(result)
