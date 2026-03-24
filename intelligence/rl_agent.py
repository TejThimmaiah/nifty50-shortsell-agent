"""
Tej Reinforcement Learning — Ray RLlib
========================================
Tej learns optimal entry/exit through trial and reward.
Like AlphaGo but for Nifty 50 short selling.

State:  25 market indicators
Action: SHORT / SKIP / HOLD
Reward: P&L + risk-adjusted return
"""

import os, json, logging, numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
logger = logging.getLogger("rl_agent")
IST = ZoneInfo("Asia/Kolkata")

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

try:
    from ray import tune
    from ray.rllib.algorithms.ppo import PPOConfig
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


class TradingEnv(gym.Env if GYM_AVAILABLE else object):
    """Nifty 50 short-selling gym environment for RL training."""

    metadata = {"render_modes": []}

    OBS_DIM  = 25
    ACT_DIM  = 3   # 0=SKIP, 1=SHORT, 2=HOLD

    def __init__(self, config=None):
        if not GYM_AVAILABLE:
            return
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.ACT_DIM)
        self.data         = []
        self.step_idx     = 0
        self.position     = None
        self.capital      = 100_000.0
        self.episode_pnl  = 0.0

    def _get_obs(self) -> np.ndarray:
        if not self.data or self.step_idx >= len(self.data):
            return np.zeros(self.OBS_DIM, dtype=np.float32)
        row = self.data[self.step_idx]
        obs = [
            self._norm(row.get("rsi", 50), 0, 100),
            self._norm(row.get("macd", 0), -2, 2),
            self._norm(row.get("volume_ratio", 1), 0, 5),
            self._norm(row.get("vwap_dist", 0), -3, 3),
            self._norm(row.get("atr", 1), 0, 5),
            self._norm(row.get("bb_pos", 0.5), 0, 1),
            self._norm(row.get("mom_5", 0), -5, 5),
            self._norm(row.get("mom_10", 0), -10, 10),
            self._norm(row.get("nifty_ret", 0), -3, 3),
            self._norm(row.get("vix", 15), 10, 40),
            self._norm(row.get("fii", 0), -5000, 5000),
            self._norm(row.get("sector", 0), -3, 3),
            self._norm(row.get("support_dist", 1), 0, 5),
            self._norm(row.get("resist_dist", 1), 0, 5),
            self._norm(row.get("obv", 0), -1, 1),
            self._norm(row.get("wyckoff", 2), 0, 4),
            self._norm(row.get("regime", 1), 0, 3),
            self._norm(row.get("hour", 10), 9, 15),
            self._norm(row.get("dow", 2), 0, 4),
            self._norm(row.get("prev_ret", 0), -5, 5),
            self._norm(row.get("gap", 0), -3, 3),
            self._norm(row.get("vol_spike", 1), 0, 5),
            self._norm(row.get("rr", 2), 1, 5),
            self._norm(row.get("kelly", 0.1), 0, 0.5),
            self._norm(row.get("score", 0.5), 0, 1),
        ]
        return np.array(obs, dtype=np.float32)

    def _norm(self, v, mn, mx):
        return float(max(-1.0, min(1.0, 2 * (v - mn) / (mx - mn + 1e-9) - 1)))

    def reset(self, seed=None, options=None):
        self.step_idx    = 0
        self.position    = None
        self.episode_pnl = 0.0
        return self._get_obs(), {}

    def step(self, action):
        row     = self.data[self.step_idx] if self.data else {}
        reward  = 0.0
        price   = row.get("close", 100)
        sl      = row.get("stop_loss", price * 1.01)
        target  = row.get("target", price * 0.98)

        if action == 1 and self.position is None:   # SHORT
            self.position = {"entry": price, "sl": sl, "target": target}
            reward = 0.0  # No reward on entry

        elif self.position is not None:              # HOLD / SKIP with position
            entry  = self.position["entry"]
            if price >= self.position["sl"]:         # Stop hit — loss
                pnl    = entry - price
                reward = pnl / entry * 100 - 0.5    # Penalty + loss
                self.episode_pnl += pnl
                self.position = None
            elif price <= self.position["target"]:   # Target hit — profit
                pnl    = entry - price
                reward = pnl / entry * 100 + 1.0    # Bonus for hitting target
                self.episode_pnl += pnl
                self.position = None
            else:
                reward = 0.01                        # Small reward for holding winner

        elif action == 0:                            # SKIP
            reward = 0.0

        self.step_idx += 1
        done   = self.step_idx >= len(self.data) - 1
        trunc  = False
        return self._get_obs(), reward, done, trunc, {"pnl": self.episode_pnl}

    def load_data(self, data: list):
        """Load historical market data for training."""
        self.data     = data
        self.step_idx = 0

    def render(self):
        pass


class RLAgent:
    """
    PPO-based RL agent for Tej.
    Trains weekly on historical data.
    Uses learned policy for live trading signals.
    """

    MODEL_PATH = "db/rl_model"

    def __init__(self):
        self.policy = None
        self._load()

    def _load(self):
        if not RAY_AVAILABLE:
            logger.warning("Ray RLlib not installed — pip install ray[rllib]")
            return
        if os.path.exists(self.MODEL_PATH):
            logger.info("RL model found — loading...")

    def train(self, market_data: list, iterations: int = 50):
        """Train RL agent on historical data."""
        if not RAY_AVAILABLE or not GYM_AVAILABLE:
            logger.warning("Ray or Gymnasium not available — skipping RL training")
            return

        import ray
        ray.init(ignore_reinit_error=True, num_cpus=1)

        try:
            config = (
                PPOConfig()
                .environment(TradingEnv)
                .rollouts(num_rollout_workers=1)
                .training(
                    train_batch_size=512,
                    lr=3e-4,
                    gamma=0.99,
                    lambda_=0.95,
                )
                .framework("torch")
            )
            algo = config.build()
            best_reward = -float("inf")

            for i in range(iterations):
                result = algo.train()
                mean_r = result.get("episode_reward_mean", 0)
                if mean_r > best_reward:
                    best_reward = mean_r
                    algo.save(self.MODEL_PATH)
                if i % 10 == 0:
                    logger.info(f"RL iter {i}/{iterations} — reward: {mean_r:.3f}")

            logger.info(f"RL training complete — best reward: {best_reward:.3f}")
            ray.shutdown()
        except Exception as e:
            logger.error(f"RL training failed: {e}")

    def get_action(self, obs: np.ndarray) -> dict:
        """Get RL policy action for current market state."""
        if self.policy is None:
            return {"action": "SKIP", "confidence": 0.5, "source": "fallback"}
        try:
            action_id = self.policy.compute_single_action(obs)
            actions   = {0: "SKIP", 1: "SHORT", 2: "HOLD"}
            return {
                "action":     actions.get(action_id, "SKIP"),
                "confidence": 0.75,
                "source":     "rl_policy",
            }
        except Exception:
            return {"action": "SKIP", "confidence": 0.5, "source": "fallback"}


rl_agent = RLAgent()
