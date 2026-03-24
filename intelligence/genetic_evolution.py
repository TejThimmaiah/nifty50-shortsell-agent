"""
Tej Genetic Strategy Evolution
=================================
100 strategy variants compete every week.
Best performer survives. Weakest die. Mutations happen.

Like natural selection — Tej's strategy evolves toward profitability
without human intervention.

Parameters that evolve:
  - RSI thresholds (entry/exit)
  - Volume multiplier
  - ATR stop multiplier
  - Min score threshold
  - Risk/reward minimum
  - Position sizing factor
  - Time filters
  - Indicator weights
"""

import os
import json
import random
import logging
import sqlite3
from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import List, Optional
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger("genetic_evolution")
IST = ZoneInfo("Asia/Kolkata")

DB_PATH = os.getenv("DB_PATH", "db/tej_brain.db")


@dataclass
class StrategyGene:
    """One strategy variant — a set of parameters."""
    gene_id:            str
    generation:         int

    # RSI parameters
    rsi_overbought:     float = 65.0
    rsi_oversold:       float = 35.0

    # Volume filter
    volume_multiplier:  float = 1.5

    # ATR stop
    atr_stop_mult:      float = 2.0
    atr_target_mult:    float = 4.0

    # Score thresholds
    min_score:          float = 0.60
    strong_score:       float = 0.75

    # Risk parameters
    min_rr:             float = 2.0
    max_risk_pct:       float = 2.0
    kelly_fraction:     float = 0.25

    # Time filter
    no_entry_after:     float = 13.0  # hour
    min_entry_after:    float = 9.5   # hour

    # Indicator weights (sum = 1.0)
    w_rsi:              float = 0.20
    w_macd:             float = 0.15
    w_volume:           float = 0.15
    w_wyckoff:          float = 0.15
    w_orderflow:        float = 0.15
    w_sentiment:        float = 0.10
    w_ml:               float = 0.10

    # Fitness (set after backtesting)
    fitness:            float = 0.0
    win_rate:           float = 0.0
    profit_factor:      float = 0.0
    total_trades:       int   = 0


class GeneticEvolution:
    """
    Genetic algorithm for autonomous strategy evolution.
    Runs every Monday. Produces next generation by week end.
    """

    POPULATION_SIZE = 50
    ELITE_COUNT     = 10   # Top 10 survive unchanged
    MUTATION_RATE   = 0.15
    CROSSOVER_RATE  = 0.7

    # Parameter bounds [min, max]
    BOUNDS = {
        "rsi_overbought":    [60, 80],
        "rsi_oversold":      [20, 45],
        "volume_multiplier": [1.0, 3.0],
        "atr_stop_mult":     [1.5, 3.5],
        "atr_target_mult":   [2.5, 6.0],
        "min_score":         [0.50, 0.80],
        "strong_score":      [0.65, 0.90],
        "min_rr":            [1.5, 4.0],
        "max_risk_pct":      [1.0, 3.0],
        "kelly_fraction":    [0.1, 0.4],
        "no_entry_after":    [12.0, 14.0],
        "min_entry_after":   [9.25, 10.5],
    }

    def __init__(self):
        self.population: List[StrategyGene] = []
        self.generation  = 0
        self.best_gene: Optional[StrategyGene] = None
        self._load_population()

    def _load_population(self):
        """Load saved population from DB."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cur  = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS gene_pool (
                    gene_id TEXT PRIMARY KEY,
                    generation INTEGER,
                    params TEXT,
                    fitness REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    total_trades INTEGER,
                    created_at TEXT
                )
            """)
            conn.commit()
            cur.execute("SELECT params, fitness FROM gene_pool ORDER BY fitness DESC LIMIT 1")
            row = cur.fetchone()
            if row:
                params = json.loads(row[0])
                self.best_gene = StrategyGene(**params)
                logger.info(f"Best gene loaded — fitness: {row[1]:.4f}")
            conn.close()
        except Exception as e:
            logger.error(f"Load population failed: {e}")

    def _random_gene(self, generation: int) -> StrategyGene:
        """Create a random strategy gene within bounds."""
        gene = StrategyGene(
            gene_id=f"G{generation}_{random.randint(1000,9999)}",
            generation=generation,
        )
        for param, (mn, mx) in self.BOUNDS.items():
            setattr(gene, param, round(random.uniform(mn, mx), 3))

        # Normalize weights
        total_w = gene.w_rsi + gene.w_macd + gene.w_volume + gene.w_wyckoff + gene.w_orderflow + gene.w_sentiment + gene.w_ml
        for w in ["w_rsi","w_macd","w_volume","w_wyckoff","w_orderflow","w_sentiment","w_ml"]:
            setattr(gene, w, round(getattr(gene, w) / total_w, 3))
        return gene

    def _crossover(self, parent1: StrategyGene, parent2: StrategyGene, gen: int) -> StrategyGene:
        """Uniform crossover — randomly pick each param from either parent."""
        child = StrategyGene(gene_id=f"G{gen}_{random.randint(1000,9999)}", generation=gen)
        for param in self.BOUNDS:
            if random.random() < 0.5:
                setattr(child, param, getattr(parent1, param))
            else:
                setattr(child, param, getattr(parent2, param))
        return child

    def _mutate(self, gene: StrategyGene) -> StrategyGene:
        """Randomly mutate some parameters."""
        mutant = deepcopy(gene)
        for param, (mn, mx) in self.BOUNDS.items():
            if random.random() < self.MUTATION_RATE:
                current = getattr(mutant, param)
                delta   = (mx - mn) * 0.1 * random.choice([-1, 1])
                new_val = max(mn, min(mx, current + delta))
                setattr(mutant, param, round(new_val, 3))
        mutant.gene_id = f"M{self.generation}_{random.randint(1000,9999)}"
        return mutant

    def _fitness(self, gene: StrategyGene, backtest_results: dict) -> float:
        """
        Calculate fitness score from backtest results.
        Balances: profit factor, win rate, Sharpe ratio, drawdown.
        """
        pf  = backtest_results.get("profit_factor", 1.0)
        wr  = backtest_results.get("win_rate", 0.5)
        sr  = backtest_results.get("sharpe", 0.0)
        mdd = backtest_results.get("max_drawdown", 0.1)
        n   = backtest_results.get("total_trades", 0)

        if n < 5:
            return 0.0

        # Penalize too few or too many trades
        trade_factor = min(1.0, n / 20)

        fitness = (
            0.35 * min(pf / 3, 1.0) +       # Profit factor (cap at 3x)
            0.25 * wr +                       # Win rate
            0.20 * min(sr / 2, 1.0) +        # Sharpe ratio (cap at 2)
            0.10 * (1 - min(mdd / 0.2, 1)) + # Drawdown penalty
            0.10 * trade_factor               # Enough trades to be valid
        )
        return round(fitness, 4)

    def evolve(self, backtest_fn) -> StrategyGene:
        """
        Run one generation of evolution.
        backtest_fn: callable(gene) -> dict with results
        Returns best gene of new generation.
        """
        self.generation += 1
        logger.info(f"Generation {self.generation} — population: {len(self.population)}")

        # Initialize if empty
        if not self.population:
            self.population = [self._random_gene(self.generation)
                               for _ in range(self.POPULATION_SIZE)]

        # Evaluate fitness
        for gene in self.population:
            try:
                results = backtest_fn(gene)
                gene.fitness       = self._fitness(gene, results)
                gene.win_rate      = results.get("win_rate", 0)
                gene.profit_factor = results.get("profit_factor", 1)
                gene.total_trades  = results.get("total_trades", 0)
            except Exception as e:
                logger.error(f"Backtest failed for {gene.gene_id}: {e}")
                gene.fitness = 0.0

        # Sort by fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        self.best_gene = self.population[0]
        logger.info(
            f"Best: {self.best_gene.gene_id} — "
            f"fitness: {self.best_gene.fitness:.4f} | "
            f"WR: {self.best_gene.win_rate:.0%} | "
            f"PF: {self.best_gene.profit_factor:.2f}"
        )

        # Build next generation
        next_gen = []

        # Elites survive unchanged
        next_gen.extend(deepcopy(self.population[:self.ELITE_COUNT]))

        # Fill rest with crossover + mutation
        while len(next_gen) < self.POPULATION_SIZE:
            # Tournament selection
            t1 = random.sample(self.population[:20], 3)
            t2 = random.sample(self.population[:20], 3)
            p1 = max(t1, key=lambda g: g.fitness)
            p2 = max(t2, key=lambda g: g.fitness)

            if random.random() < self.CROSSOVER_RATE:
                child = self._crossover(p1, p2, self.generation)
            else:
                child = deepcopy(p1)

            child = self._mutate(child)
            next_gen.append(child)

        self.population = next_gen
        self._save_best()
        return self.best_gene

    def _save_best(self):
        """Save best gene to DB."""
        if not self.best_gene:
            return
        try:
            conn = sqlite3.connect(DB_PATH)
            cur  = conn.cursor()
            params = {k: v for k, v in asdict(self.best_gene).items()}
            cur.execute("""
                INSERT OR REPLACE INTO gene_pool
                (gene_id, generation, params, fitness, win_rate, profit_factor, total_trades, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.best_gene.gene_id,
                self.generation,
                json.dumps(params),
                self.best_gene.fitness,
                self.best_gene.win_rate,
                self.best_gene.profit_factor,
                self.best_gene.total_trades,
                datetime.now(IST).isoformat(),
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Save best gene failed: {e}")

    def get_best_params(self) -> dict:
        """Get best strategy parameters as config dict."""
        if not self.best_gene:
            return {}
        return {
            "rsi_overbought":   self.best_gene.rsi_overbought,
            "rsi_oversold":     self.best_gene.rsi_oversold,
            "volume_mult":      self.best_gene.volume_multiplier,
            "atr_stop":         self.best_gene.atr_stop_mult,
            "min_score":        self.best_gene.min_score,
            "min_rr":           self.best_gene.min_rr,
            "no_entry_after":   f"{int(self.best_gene.no_entry_after):02d}:{int((self.best_gene.no_entry_after % 1)*60):02d}",
        }

    def format_for_telegram(self) -> str:
        """Format evolution status as Telegram message."""
        if not self.best_gene:
            return "No evolution data yet — first generation runs Monday."
        g = self.best_gene
        return (
            f"<b>Strategy Evolution — Generation {self.generation}</b>\n\n"
            f"Best Gene: {g.gene_id}\n"
            f"Fitness: {g.fitness:.4f} | WR: {g.win_rate:.0%} | PF: {g.profit_factor:.2f}\n"
            f"Trades validated: {g.total_trades}\n\n"
            f"<b>Evolved Parameters:</b>\n"
            f"RSI entry: &gt;{g.rsi_overbought:.0f}\n"
            f"Min score: {g.min_score:.2f}\n"
            f"ATR stop: {g.atr_stop_mult:.1f}x\n"
            f"R:R minimum: {g.min_rr:.1f}:1\n"
            f"Stop trading after: {int(g.no_entry_after):02d}:00"
        )


genetic_evolution = GeneticEvolution()
