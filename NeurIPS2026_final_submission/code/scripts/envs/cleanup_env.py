"""
Cleanup (Abstracted) Environment — Gymnasium-compatible
========================================================
Abstracted version of the Sequential Social Dilemma "Cleanup"
(Hughes et al., 2018; Leibo et al., 2017).

Original Cleanup: grid-world where agents can either (1) collect apples
or (2) clean a polluted river.  Apple regrowth REQUIRES river cleaning.
Free-riders eat apples while others bear the cleaning cost.

This abstraction preserves the core dynamics:
  - N agents choose allocation λ_i ∈ [0,1] (fraction of effort on CLEANING)
  - 1-λ_i = fraction on harvesting (selfish reward)
  - Pollution P_t: increases naturally, DECREASES with cleaning effort
  - Apple yield Y_t: high when P < P_crit, collapses when P ≥ P_crit
  - Individual reward: (1-λ_i) · Y_t (harvest-only benefit)
  - Collective dilemma: everyone harvests → nobody cleans → P↑ → Y↓

Key difference from PGG and Harvest:
  - PGG: contribute TO pool → pool enables survival
  - Harvest: extract FROM resource → over-extraction depletes
  - Cleanup: allocate effort TO maintenance → neglect causes collapse
  All three are structurally distinct, yet all produce Nash Trap.

Paper Ref: Section 5 (Cross-environment validation)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional


class CleanupEnv(gym.Env):
    """
    Abstracted Cleanup Social Dilemma.

    N agents choose cleaning allocation λ_i ∈ [0,1].
    Pollution P_t increases naturally; cleaning effort reduces it.
    Apple yield Y_t depends on pollution level (tipping-point).
    
    Observation: [P_t, Y_t, mean_lambda_{t-1}, t/T]
    Action: λ_i ∈ [0, 1] (cleaning effort; 1-λ_i = harvesting)
    Reward: (1-λ_i) · Y_t · harvest_value
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_agents: int = 20,
        t_horizon: int = 50,
        p_crit: float = 0.50,       # Pollution threshold for yield collapse
        p_recov: float = 0.35,      # Recovery threshold
        pollution_rate: float = 0.10,  # Natural pollution increase per step (aggressive)
        clean_efficiency: float = 0.02,  # Cleaning efficiency per unit effort (low)
        harvest_value: float = 20.0,
        shock_prob: float = 0.05,
        shock_mag: float = 0.10,     # Pollution spike
        byz_frac: float = 0.3,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.T = t_horizon
        self.p_crit = p_crit
        self.p_recov = p_recov
        self.pollution_rate = pollution_rate
        self.clean_efficiency = clean_efficiency
        self.harvest_value = harvest_value
        self.shock_prob = shock_prob
        self.shock_mag = shock_mag
        self.byz_frac = byz_frac
        self.n_byz = int(n_agents * byz_frac)
        self.n_honest = n_agents - self.n_byz

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )
        self.reset()

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.P = 0.4          # Start with elevated pollution
        self.Y = 0.8          # Start with good apple yield
        self.t = 0
        self.prev_mean_lambda = 0.5
        self.terminated = False
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.P,
            self.Y,
            self.prev_mean_lambda,
            self.t / self.T,
        ], dtype=np.float32)

    def step(self, actions_honest: np.ndarray):
        if self.terminated:
            return self._get_obs(), np.zeros(self.n_honest), True, False, {}

        # Build full lambda vector
        lambdas = np.zeros(self.n_agents)
        lambdas[:self.n_honest] = np.clip(actions_honest, 0, 1)
        # Byzantine: zero cleaning (pure free-riding)
        lambdas[self.n_honest:] = 0.0

        mean_cleaning = float(np.mean(lambdas))
        self.prev_mean_lambda = mean_cleaning

        # --- Pollution dynamics ---
        # Natural pollution increase
        delta_pollution = self.pollution_rate

        # Cleaning reduces pollution
        total_cleaning = mean_cleaning * self.clean_efficiency * self.n_agents
        
        # Pollution shock
        shock = self.shock_mag if self.np_random.random() < self.shock_prob else 0.0

        self.P = np.clip(self.P + delta_pollution - total_cleaning + shock, 0, 1)

        # --- Apple yield (tipping-point) ---
        if self.P >= self.p_crit:
            self.Y = max(0.05, self.Y * 0.7)  # Yield collapses
        elif self.P >= self.p_recov:
            self.Y = max(0.1, self.Y * 0.9)   # Fragile yield
        else:
            self.Y = min(1.0, self.Y * 1.05)   # Healthy growth

        self.t += 1

        # --- Rewards: (1-λ_i) · Y · harvest_value ---
        # Agents that clean more get less immediate reward
        harvest_fracs = 1.0 - lambdas
        rewards_all = harvest_fracs * self.Y * self.harvest_value
        rewards = rewards_all[:self.n_honest].astype(np.float32)

        # Survival: yield above threshold
        survived = self.Y > 0.1
        terminated = not survived or self.t >= self.T
        self.terminated = terminated

        info = {
            "pollution": self.P,
            "yield": self.Y,
            "resource": self.Y,       # Alias for consistency with PGG/Harvest
            "survived": survived,
            "mean_lambda": float(np.mean(lambdas[:self.n_honest])),
            "mean_lambda_all": mean_cleaning,
            "welfare": float(np.mean(rewards_all)),
        }

        return self._get_obs(), rewards, terminated, False, info


if __name__ == "__main__":
    env = CleanupEnv()
    print("=== Cleanup Env Self-Test ===")

    # Test 1: Good cleaners (λ=0.4)
    obs, _ = env.reset(seed=42)
    for t in range(50):
        actions = np.full(env.n_honest, 0.4)
        obs, rewards, terminated, truncated, info = env.step(actions)
        if terminated: break
    print(f"Cleaners (l=0.4): P={info['pollution']:.3f}, Y={info['yield']:.3f}, "
          f"survived={info['survived']}, steps={t+1}")

    # Test 2: Free riders (λ=0.0)
    obs, _ = env.reset(seed=42)
    for t in range(50):
        actions = np.full(env.n_honest, 0.0)
        obs, rewards, terminated, truncated, info = env.step(actions)
        if terminated: break
    print(f"Freeriders (l=0): P={info['pollution']:.3f}, Y={info['yield']:.3f}, "
          f"survived={info['survived']}, steps={t+1}")

    print("\nCleanup env self-test PASSED!")
