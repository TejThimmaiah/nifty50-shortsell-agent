"""
Tej Autonomous Hypothesis Generation + FAISS Associative Memory
================================================================
TWO FEATURES:

1. HYPOTHESIS GENERATOR:
   Tej forms his own trading theories and tests them.
   "Hypothesis: HDFC stocks underperform when crude > $85.
    Testing on 2 years data... Confirmed: 68% win rate on this pattern."

2. FAISS ASSOCIATIVE MEMORY:
   "This setup looks exactly like March 2020 crash — here's what worked."
   Finds the most similar historical market situations instantly.
"""

import os, json, logging, numpy as np
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
from zoneinfo import ZoneInfo
logger = logging.getLogger("hypothesis_memory")
IST = ZoneInfo("Asia/Kolkata")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# ══════════════════════════════════════════════════════
# PART 1 — AUTONOMOUS HYPOTHESIS GENERATOR
# ══════════════════════════════════════════════════════

@dataclass
class Hypothesis:
    id:          str
    description: str
    condition:   dict    # When this is true
    prediction:  str     # What Tej predicts
    win_rate:    float
    sample_size: int
    confirmed:   bool
    created_at:  str


class HypothesisGenerator:
    """
    Generates and tests trading hypotheses autonomously.
    New hypotheses are generated from patterns in trade outcomes.
    """

    def __init__(self):
        self.hypotheses: List[Hypothesis] = []
        self._load()

    def _load(self):
        path = "db/hypotheses.json"
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                self.hypotheses = [Hypothesis(**h) for h in data]
                logger.info(f"Loaded {len(self.hypotheses)} hypotheses")
            except Exception:
                pass

    def _save(self):
        os.makedirs("db", exist_ok=True)
        with open("db/hypotheses.json", "w") as f:
            import dataclasses
            json.dump([dataclasses.asdict(h) for h in self.hypotheses], f, indent=2)

    def generate_from_trades(self, trades: list) -> List[Hypothesis]:
        """
        Analyze trade history to generate new hypotheses.
        Looks for patterns in winning vs losing trades.
        """
        if len(trades) < 20:
            return []

        new_hypotheses = []
        winners = [t for t in trades if t.get("pnl", 0) > 0]
        losers  = [t for t in trades if t.get("pnl", 0) < 0]

        if not winners or not losers:
            return []

        # Hypothesis: High RSI + High Volume = better win rate?
        high_rsi_vol   = [t for t in trades if t.get("rsi", 50) > 68 and t.get("volume_ratio", 1) > 1.8]
        if len(high_rsi_vol) >= 5:
            wr = sum(1 for t in high_rsi_vol if t.get("pnl", 0) > 0) / len(high_rsi_vol)
            h  = Hypothesis(
                id=f"H{len(self.hypotheses)+1:03d}",
                description="RSI > 68 AND Volume > 1.8x average → stronger short signal",
                condition={"rsi_gt": 68, "volume_ratio_gt": 1.8},
                prediction="Win rate above 60% when both conditions met",
                win_rate=wr,
                sample_size=len(high_rsi_vol),
                confirmed=wr > 0.60,
                created_at=datetime.now(IST).isoformat(),
            )
            new_hypotheses.append(h)

        # Hypothesis: Morning trades better than afternoon?
        morning = [t for t in trades if t.get("hour", 12) < 11]
        afternoon = [t for t in trades if t.get("hour", 12) >= 13]
        if len(morning) >= 5 and len(afternoon) >= 5:
            mwr = sum(1 for t in morning if t.get("pnl", 0) > 0) / len(morning)
            awr = sum(1 for t in afternoon if t.get("pnl", 0) > 0) / len(afternoon)
            if abs(mwr - awr) > 0.15:
                better = "morning" if mwr > awr else "afternoon"
                h = Hypothesis(
                    id=f"H{len(self.hypotheses)+len(new_hypotheses)+1:03d}",
                    description=f"{better.title()} trades have significantly better win rate",
                    condition={"time_of_day": better},
                    prediction=f"Enter only in {better} for better results",
                    win_rate=max(mwr, awr),
                    sample_size=len(morning) + len(afternoon),
                    confirmed=abs(mwr - awr) > 0.20,
                    created_at=datetime.now(IST).isoformat(),
                )
                new_hypotheses.append(h)

        self.hypotheses.extend(new_hypotheses)
        if new_hypotheses:
            self._save()

        return new_hypotheses

    def get_applicable(self, context: dict) -> List[Hypothesis]:
        """Get hypotheses that apply to current market context."""
        applicable = []
        for h in self.hypotheses:
            if not h.confirmed:
                continue
            cond = h.condition
            match = True
            if "rsi_gt" in cond and context.get("rsi", 50) <= cond["rsi_gt"]:
                match = False
            if "volume_ratio_gt" in cond and context.get("volume_ratio", 1) <= cond["volume_ratio_gt"]:
                match = False
            if match:
                applicable.append(h)
        return applicable

    def format_for_telegram(self) -> str:
        if not self.hypotheses:
            return "<b>Hypotheses</b>\nNone generated yet — need more trades."
        conf = [h for h in self.hypotheses if h.confirmed]
        msg  = f"<b>Tej's Trading Hypotheses</b>\n\nTotal: {len(self.hypotheses)} | Confirmed: {len(conf)}\n\n"
        for h in sorted(conf, key=lambda x: x.win_rate, reverse=True)[:3]:
            msg += f"✅ {h.description}\n   WR: {h.win_rate:.0%} | n={h.sample_size}\n\n"
        return msg


# ══════════════════════════════════════════════════════
# PART 2 — FAISS ASSOCIATIVE MEMORY
# ══════════════════════════════════════════════════════

class FAISSMemory:
    """
    Ultra-fast similarity search over all past trade situations.
    "This looks EXACTLY like that HDFCBANK trade on Jan 15" — in milliseconds.
    """

    DIM = 15   # Feature dimensions

    def __init__(self):
        self.index     = None
        self.metadata  = []
        self._init_index()

    def _init_index(self):
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not installed — pip install faiss-cpu")
            return
        self.index = faiss.IndexFlatL2(self.DIM)
        self._load()

    def _features(self, ctx: dict) -> np.ndarray:
        return np.array([
            ctx.get("rsi", 50) / 100,
            ctx.get("volume_ratio", 1) / 5,
            ctx.get("master_score", 0.5),
            ctx.get("vix", 15) / 40,
            ctx.get("nifty_5d", 0) / 5,
            ctx.get("fii_flow", 0) / 5000,
            ctx.get("sentiment_score", 0),
            ctx.get("pcr", 1) / 3,
            ctx.get("macd_signal", 0) / 2,
            ctx.get("bb_position", 0.5),
            ctx.get("wyckoff_phase", 2) / 4,
            ctx.get("time_of_day", 11) / 15,
            ctx.get("day_of_week", 2) / 4,
            ctx.get("atr_normalized", 1) / 5,
            ctx.get("support_distance", 1) / 5,
        ], dtype=np.float32)

    def _load(self):
        path = "db/faiss_memory.json"
        if os.path.exists(path) and self.index is not None:
            try:
                with open(path) as f:
                    data = json.load(f)
                self.metadata = data.get("metadata", [])
                vecs = data.get("vectors", [])
                if vecs:
                    arr = np.array(vecs, dtype=np.float32)
                    self.index.add(arr)
                logger.info(f"FAISS loaded {self.index.ntotal} memories")
            except Exception:
                pass

    def store(self, symbol: str, context: dict, outcome: dict):
        """Store a trade in FAISS index."""
        if self.index is None:
            return
        vec = self._features(context).reshape(1, -1)
        self.index.add(vec)
        self.metadata.append({
            "symbol":  symbol,
            "date":    datetime.now(IST).strftime("%Y-%m-%d"),
            "pnl":     outcome.get("pnl", 0),
            "pnl_pct": outcome.get("pnl_pct", 0),
            "winner":  outcome.get("pnl", 0) > 0,
        })

    def find_similar(self, context: dict, k: int = 3) -> List[dict]:
        """Find k most similar historical situations."""
        if self.index is None or self.index.ntotal == 0:
            return []
        try:
            vec = self._features(context).reshape(1, -1)
            k   = min(k, self.index.ntotal)
            D, I = self.index.search(vec, k)
            results = []
            for dist, idx in zip(D[0], I[0]):
                if idx >= 0 and idx < len(self.metadata):
                    m = dict(self.metadata[idx])
                    m["similarity"] = round(1 / (1 + float(dist)), 3)
                    results.append(m)
            return results
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []

    def associative_insight(self, context: dict, symbol: str) -> str:
        """Get associative insight from similar past situations."""
        similar = self.find_similar(context)
        if not similar:
            return "No similar historical situations found yet."
        best = similar[0]
        wins = sum(1 for s in similar if s.get("winner", False))
        return (
            f"Most similar past situation: {best['symbol']} on {best['date']} "
            f"({best['similarity']:.0%} match) → {best['pnl_pct']:+.1f}%. "
            f"Win rate in similar setups: {wins}/{len(similar)} ({wins/len(similar):.0%})"
        )


hypothesis_generator = HypothesisGenerator()
faiss_memory         = FAISSMemory()
