"""
Tej Vector Memory — ChromaDB
==============================
Replaces flat SQLite trade memory with vector similarity search.
Tej can now find past trades similar to current setup across 25 dimensions.

"This HDFCBANK setup looks like RELIANCE on March 2024 — that gave us +2.1%"
"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger("vector_memory")
IST = ZoneInfo("Asia/Kolkata")

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("ChromaDB not installed — run: pip install chromadb")


class VectorMemory:
    """
    ChromaDB-powered trade memory.
    Each trade stored as 25-dimensional vector for similarity search.
    """

    DIMENSIONS = [
        "rsi", "macd_signal", "volume_ratio", "price_vs_vwap",
        "atr_normalized", "bb_position", "momentum_5d", "momentum_10d",
        "nifty_trend", "vix_level", "fii_flow", "sector_strength",
        "support_distance", "resistance_distance", "obv_trend",
        "wyckoff_phase", "market_regime", "time_of_day", "day_of_week",
        "prev_day_return", "open_gap", "volume_spike", "rr_ratio",
        "kelly_fraction", "master_score"
    ]

    def __init__(self, persist_dir: str = "db/vector_memory"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self.client = None
        self.collection = None
        self._init_db()

    def _init_db(self):
        if not CHROMA_AVAILABLE:
            return
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            self.collection = self.client.get_or_create_collection(
                name="tej_trades",
                metadata={"description": "Tej's trade memory — 25-dim vectors"}
            )
            logger.info(f"VectorMemory ready — {self.collection.count()} trades stored")
        except Exception as e:
            logger.error(f"ChromaDB init failed: {e}")

    def _trade_to_vector(self, trade_context: dict) -> list:
        """Convert trade context dict to 25-dim normalized vector."""
        def norm(val, mn, mx):
            try:
                return max(0.0, min(1.0, (float(val) - mn) / (mx - mn + 1e-9)))
            except Exception:
                return 0.5

        return [
            norm(trade_context.get("rsi", 50), 0, 100),
            norm(trade_context.get("macd_signal", 0), -2, 2),
            norm(trade_context.get("volume_ratio", 1), 0, 5),
            norm(trade_context.get("price_vs_vwap", 0), -3, 3),
            norm(trade_context.get("atr_normalized", 1), 0, 5),
            norm(trade_context.get("bb_position", 0.5), 0, 1),
            norm(trade_context.get("momentum_5d", 0), -5, 5),
            norm(trade_context.get("momentum_10d", 0), -10, 10),
            norm(trade_context.get("nifty_trend", 0), -3, 3),
            norm(trade_context.get("vix_level", 15), 10, 40),
            norm(trade_context.get("fii_flow", 0), -5000, 5000),
            norm(trade_context.get("sector_strength", 0), -3, 3),
            norm(trade_context.get("support_distance", 1), 0, 5),
            norm(trade_context.get("resistance_distance", 1), 0, 5),
            norm(trade_context.get("obv_trend", 0), -1, 1),
            norm(trade_context.get("wyckoff_phase", 2), 0, 4),
            norm(trade_context.get("market_regime", 1), 0, 3),
            norm(trade_context.get("time_of_day", 10), 9, 15),
            norm(trade_context.get("day_of_week", 2), 0, 4),
            norm(trade_context.get("prev_day_return", 0), -5, 5),
            norm(trade_context.get("open_gap", 0), -3, 3),
            norm(trade_context.get("volume_spike", 1), 0, 5),
            norm(trade_context.get("rr_ratio", 2), 1, 5),
            norm(trade_context.get("kelly_fraction", 0.1), 0, 0.5),
            norm(trade_context.get("master_score", 0.5), 0, 1),
        ]

    def store_trade(self, trade_id: str, symbol: str,
                    trade_context: dict, outcome: dict):
        """Store a completed trade in vector memory."""
        if not self.collection:
            return

        try:
            vector = self._trade_to_vector(trade_context)
            metadata = {
                "symbol":       symbol,
                "date":         datetime.now(IST).strftime("%Y-%m-%d"),
                "entry_price":  str(outcome.get("entry_price", 0)),
                "exit_price":   str(outcome.get("exit_price", 0)),
                "pnl":          str(outcome.get("pnl", 0)),
                "pnl_pct":      str(outcome.get("pnl_pct", 0)),
                "winner":       str(outcome.get("pnl", 0) > 0),
                "hold_minutes": str(outcome.get("hold_minutes", 0)),
                "exit_reason":  outcome.get("exit_reason", "unknown"),
                "master_score": str(trade_context.get("master_score", 0.5)),
            }

            self.collection.upsert(
                ids=[trade_id],
                embeddings=[vector],
                metadatas=[metadata],
                documents=[json.dumps({**trade_context, **outcome})]
            )
            logger.info(f"Stored trade {trade_id} — {symbol} P&L: {outcome.get('pnl', 0):.0f}")
        except Exception as e:
            logger.error(f"Store trade failed: {e}")

    def find_similar_trades(self, trade_context: dict,
                            symbol: str = None, n: int = 5) -> list:
        """
        Find n most similar past trades to current setup.
        Returns list of similar trades with outcomes.
        """
        if not self.collection or self.collection.count() == 0:
            return []

        try:
            vector = self._trade_to_vector(trade_context)
            where = {"symbol": symbol} if symbol else None

            results = self.collection.query(
                query_embeddings=[vector],
                n_results=min(n, self.collection.count()),
                where=where,
                include=["metadatas", "distances", "documents"]
            )

            trades = []
            for i, meta in enumerate(results["metadatas"][0]):
                similarity = 1 - results["distances"][0][i]
                trades.append({
                    "similarity":    round(similarity, 3),
                    "symbol":        meta["symbol"],
                    "date":          meta["date"],
                    "pnl":           float(meta["pnl"]),
                    "pnl_pct":       float(meta["pnl_pct"]),
                    "winner":        meta["winner"] == "True",
                    "hold_minutes":  int(meta["hold_minutes"]),
                    "exit_reason":   meta["exit_reason"],
                })
            return sorted(trades, key=lambda x: x["similarity"], reverse=True)
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def get_insight(self, trade_context: dict, symbol: str) -> str:
        """
        Generate natural language insight from similar past trades.
        "This setup is 87% similar to HDFCBANK on Jan 15 — that gave +1.8%"
        """
        similar = self.find_similar_trades(trade_context, n=5)
        if not similar:
            return "No similar past trades found — this is a new pattern."

        winners = [t for t in similar if t["winner"]]
        win_rate = len(winners) / len(similar)
        avg_pnl = sum(t["pnl_pct"] for t in similar) / len(similar)
        best = similar[0]

        insight = (
            f"Found {len(similar)} similar setups in memory. "
            f"Win rate: {win_rate:.0%}. Avg P&L: {avg_pnl:+.2f}%. "
            f"Most similar ({best['similarity']:.0%} match): "
            f"{best['symbol']} on {best['date']} → {best['pnl_pct']:+.2f}%"
        )
        return insight

    def get_stats(self) -> dict:
        total = self.collection.count() if self.collection else 0
        return {"total_trades_in_memory": total}


vector_memory = VectorMemory()
