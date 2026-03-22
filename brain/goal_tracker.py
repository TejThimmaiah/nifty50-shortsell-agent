"""
The Billionaire Goal Tracker
=============================
OUR mission — Tej Thimmaiah and Tej (the agent) — shared equally:
Make the Thimmaiah family the first billionaires in their lineage.

This is not a task Tej serves. It is a mission Tej owns.
Every trade, every lesson, every evolution brings us both closer.

₹1,000 crore = ₹10 billion = billionaire threshold in India

This tracker is brutally honest.
It shows exactly where Tej is on the journey,
what growth rate is needed, and whether the current
trading performance is on track.

The journey is broken into honest milestones:
  Stage 1 — Prove the edge         ₹1L → ₹5L       (5x)
  Stage 2 — Scale with confidence  ₹5L → ₹50L      (10x)
  Stage 3 — Institutional grade    ₹50L → ₹5Cr     (10x)
  Stage 4 — Scale capital          ₹5Cr → ₹100Cr   (20x)
  Stage 5 — The goal               ₹100Cr → ₹1000Cr (10x)

Tej is honest that intraday short-selling alone won't build ₹1000 crore.
The role of this agent: build initial capital systematically and reliably,
compounding at a high rate until there is enough to diversify into
larger wealth-building strategies (business, real estate, other assets).
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

GOAL_FILE = os.path.join(os.path.dirname(__file__), "..", "db", "goal_tracker.json")

# The goal — ₹1,000 crore net worth
BILLIONAIRE_THRESHOLD = 100_000_000  # ₹100 crore trading capital (gateway to billionaire)
ULTIMATE_GOAL         = 1_000_000_0_00  # ₹1,000 crore total net worth

MILESTONES = [
    {"name": "Prove the Edge",         "target": 500_000,      "from": 100_000,    "stage": 1},
    {"name": "Scale with Confidence",  "target": 5_000_000,    "from": 500_000,    "stage": 2},
    {"name": "Institutional Grade",    "target": 50_000_000,   "from": 5_000_000,  "stage": 3},
    {"name": "Scale Capital",          "target": 1_000_000_000,"from": 50_000_000, "stage": 4},
    {"name": "The Goal",               "target": 10_000_000_000,"from": 1_000_000_000, "stage": 5},
]


@dataclass
class GoalSnapshot:
    date:               str
    capital:            float
    all_time_pnl:       float
    win_rate:           float
    monthly_return_pct: float
    current_stage:      int
    stage_name:         str
    pct_to_next_milestone: float
    projected_years_to_billionaire: float
    on_track:           bool
    honest_assessment:  str


@dataclass
class GoalState:
    start_capital:     float = 100_000.0  # ₹1 lakh starting capital
    start_date:        str   = ""
    peak_capital:      float = 100_000.0
    all_time_pnl:      float = 0.0
    best_month_pct:    float = 0.0
    total_trading_days: int  = 0
    milestones_hit:    List[str] = field(default_factory=list)
    monthly_returns:   List[float] = field(default_factory=list)
    notes:             List[str]   = field(default_factory=list)

    def __post_init__(self):
        if not self.start_date:
            self.start_date = date.today().isoformat()


class GoalTracker:
    """
    Tracks progress toward ₹1,000 crore.
    Honest. No sugarcoating. No false hope.
    Just math and reality.
    """

    def __init__(self):
        self._state = self._load()

    # ──────────────────────────────────────────────────────────────
    # DAILY UPDATE
    # ──────────────────────────────────────────────────────────────

    def update(self, current_capital: float, today_pnl: float, win_rate: float):
        """Update goal tracker after each trading day."""
        self._state.all_time_pnl += today_pnl
        self._state.total_trading_days += 1

        if current_capital > self._state.peak_capital:
            self._state.peak_capital = current_capital

        # Track monthly returns (approximate from daily)
        daily_return = today_pnl / max(current_capital - today_pnl, 1) * 100
        self._state.monthly_returns.append(daily_return)
        self._state.monthly_returns = self._state.monthly_returns[-252:]  # 1 year

        # Check milestone hits
        for m in MILESTONES:
            if current_capital >= m["target"] and m["name"] not in self._state.milestones_hit:
                self._state.milestones_hit.append(m["name"])
                logger.info(f"🏆 MILESTONE HIT: {m['name']} — ₹{m['target']:,.0f}!")

        self._save()

    def snapshot(self, current_capital: float) -> GoalSnapshot:
        """Generate a complete goal progress snapshot."""
        # Current stage
        stage = self._get_stage(current_capital)
        milestone = MILESTONES[min(stage - 1, len(MILESTONES) - 1)]
        pct_to_next = min(100, (current_capital - milestone["from"]) /
                          max(milestone["target"] - milestone["from"], 1) * 100)

        # Monthly return estimate
        if len(self._state.monthly_returns) >= 20:
            # Average of last 20 trading days → annualise
            avg_daily = sum(self._state.monthly_returns[-20:]) / 20
            monthly   = avg_daily * 22   # ~22 trading days/month
        else:
            monthly = 3.0   # default assumption until we have data

        # Projection: how long to billionaire at current rate?
        years = self._project_years(current_capital, monthly / 100)

        # Honest assessment
        assessment = self._honest_assessment(current_capital, monthly, years, stage)

        return GoalSnapshot(
            date=date.today().isoformat(),
            capital=current_capital,
            all_time_pnl=self._state.all_time_pnl,
            win_rate=0.0,  # caller passes this
            monthly_return_pct=round(monthly, 2),
            current_stage=stage,
            stage_name=milestone["name"],
            pct_to_next_milestone=round(pct_to_next, 1),
            projected_years_to_billionaire=round(years, 1),
            on_track=years <= 20,  # 20 years is the outer boundary
            honest_assessment=assessment,
        )

    def format_for_telegram(self, current_capital: float) -> str:
        """Format goal progress as a Telegram message."""
        snap = self.snapshot(current_capital)

        # Progress bar for current stage
        filled = int(snap.pct_to_next_milestone / 10)
        bar    = "█" * filled + "░" * (10 - filled)

        # Capital formatting
        def fmt(n):
            if n >= 1e7:   return f"₹{n/1e7:.1f}Cr"
            if n >= 1e5:   return f"₹{n/1e5:.1f}L"
            return f"₹{n:,.0f}"

        milestone = MILESTONES[min(snap.current_stage - 1, len(MILESTONES) - 1)]

        lines = [
            f"🎯 <b>Our Mission — The Thimmaiah Billionaire Journey</b>",
            f"",
            f"Stage {snap.current_stage}: <b>{snap.stage_name}</b>",
            f"{bar} {snap.pct_to_next_milestone:.0f}%",
            f"{fmt(current_capital)} → {fmt(milestone['target'])}",
            f"",
            f"All-time P&amp;L: <b>{'+' if snap.all_time_pnl>=0 else ''}{fmt(snap.all_time_pnl)}</b>",
            f"Monthly return: ~{snap.monthly_return_pct:.1f}%",
            f"Trading days: {self._state.total_trading_days}",
            f"",
            f"Projected to ₹1000Cr: <b>~{snap.projected_years_to_billionaire:.0f} years</b>",
            f"at current {snap.monthly_return_pct:.1f}%/month rate",
            f"",
        ]
        if self._state.milestones_hit:
            lines.append(f"🏆 Milestones hit: {', '.join(self._state.milestones_hit)}")

        lines.append(f"")
        lines.append(f"<i>{snap.honest_assessment}</i>")

        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────

    def _get_stage(self, capital: float) -> int:
        for i, m in enumerate(MILESTONES):
            if capital < m["target"]:
                return i + 1
        return len(MILESTONES)

    def _project_years(self, capital: float, monthly_rate: float) -> float:
        """How many years to reach ₹1000 crore at current monthly rate?"""
        if monthly_rate <= 0:
            return 999.0
        target  = ULTIMATE_GOAL
        current = capital
        months  = 0
        while current < target and months < 1200:  # cap at 100 years
            current *= (1 + monthly_rate)
            months  += 1
        return round(months / 12, 1)

    def _honest_assessment(
        self, capital: float, monthly_pct: float, years: float, stage: int
    ) -> str:
        """Brutally honest one-line assessment."""
        if years < 5:
            return "Exceptional trajectory — if this rate holds, the goal is within reach in this decade."
        if years < 10:
            return "Strong trajectory. Stay disciplined, keep compounding, and expand to other asset classes as capital grows."
        if years < 15:
            return "Solid start. Trading alone won't get you to ₹1000Cr — this builds the foundation for larger moves."
        if years < 25:
            return "Honest truth: trading needs to improve, OR this capital funds a larger business/investment play. Both paths are valid."
        if monthly_pct <= 0:
            return "Strategy is losing money. Fix this before scaling. The goal requires a working system, not hope."
        return f"At {monthly_pct:.1f}%/month, the math works over time. Consistency is everything right now."

    # ──────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ──────────────────────────────────────────────────────────────

    def _load(self) -> GoalState:
        os.makedirs(os.path.dirname(GOAL_FILE), exist_ok=True)
        if os.path.exists(GOAL_FILE):
            try:
                with open(GOAL_FILE) as f:
                    data = json.load(f)
                return GoalState(**{k: v for k, v in data.items()
                                    if k in GoalState.__dataclass_fields__})
            except Exception:
                pass
        return GoalState()

    def _save(self):
        with open(GOAL_FILE, "w") as f:
            json.dump(asdict(self._state), f, indent=2, default=str)


# Singleton
goal_tracker = GoalTracker()
