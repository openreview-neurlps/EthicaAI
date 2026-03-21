"""
Harvest (Abstracted) Environment — Gymnasium-compatible
=========================================================
Abstracted version of the Sequential Social Dilemma "Harvest"
(Leibo et al., 2017; Hughes et al., 2018).

Original Harvest: grid-world where agents harvest apples from an orchard.
Apple regrowth depends on remaining apple density — over-harvesting leads
to depletion (Tragedy of the Commons).

This abstraction preserves the core dynamics:
  - N agents choose harvest rate λ_i ∈ [0,1] (0=conserve, 1=max harvest)
  - Resource R_t represents apple density
  - Regrowth: f(R) is nonlinear — below threshold, regrowth collapses
  - Individual reward: proportional to harvest (λ_i · R_t)
  - Collective dilemma: high λ depletes R → future rewards ↓

Key difference from PGG:
  - Agents directly extract FROM the resource (vs contributing TO a pool)
  - Reward is multiplicative with resource level (vs additive payoff)
  - Tipping point comes from regrowth collapse (vs resource dynamics)

This creates a DUAL tipping-point social dilemma:
  PGG: contribute too little → resource dies
  Harvest: extract too much → resource dies

If Nash Trap appears in BOTH, it demonstrates structural generality.

Paper Ref: Section 4 (Cross-environment validation)
"""

# ================================================================
# PARAMETER MAP: Paper Equations ↔ Code Constants
# ================================================================
# Paper Symbol        | Code Variable        | Value   | Paper Ref
# --------------------|----------------------|---------|----------
# N                   | n_agents             | 20      | Sec 3.1
# T (horizon)         | t_horizon            | 50      | Sec 3.1
# R_crit              | r_crit               | 0.15    | Analogous
# R_recov             | r_recov              | 0.30    | Analogous
# α (regrowth rate)   | regrowth_rate        | 0.08    | Harvest-specific
# β (Byzantine frac)  | byz_frac             | 0.3     | Sec 4
# p_shock             | shock_prob           | 0.05    | Same as PGG
# δ_shock             | shock_mag            | 0.10    | Same as PGG
#
# Resource dynamics (Harvest analog of Eq. 5):
#   harvest_t = mean(λ_i) · R_t · extraction_rate
#   regrowth = { 0.01  if R < R_crit      (depleted: near-zero)
#              { 0.04  if R_crit ≤ R < R_recov  (fragile)
#              { α     if R ≥ R_recov      (healthy)
#   R_{t+1} = clip(R_t - harvest_t + regrowth · R_t · (1 - R_t) - shock, 0, 1)
#
# Agent reward:
#   r_i = λ_i · R_t · harvest_value (individual benefit of extraction)
# ================================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional


class HarvestEnv(gym.Env):
    """
    Abstracted Harvest Social Dilemma — Tragedy of the Commons.

    N agents choose extraction rate λ_i ∈ [0,1].
    Resource R_t with logistic regrowth and tipping-point depletion.
    Survival defined as R_T > 0 (resource not fully depleted).

    Observation (per agent): [R_t, mean_lambda_{t-1}, depleted_flag, t/T]
    Action (per agent): λ_i ∈ [0, 1] (harvest intensity)
    Reward: individual harvest = λ_i · R_t · harvest_value
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_agents: int = 20,
        t_horizon: int = 50,
        r_crit: float = 0.15,
        r_recov: float = 0.30,
        regrowth_rate: float = 0.08,
        extraction_rate: float = 0.04,
        harvest_value: float = 20.0,
        shock_prob: float = 0.05,
        shock_mag: float = 0.10,
        byz_frac: float = 0.3,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.T = t_horizon
        self.r_crit = r_crit
        self.r_recov = r_recov
        self.regrowth_rate = regrowth_rate
        self.extraction_rate = extraction_rate
        self.harvest_value = harvest_value
        self.shock_prob = shock_prob
        self.shock_mag = shock_mag
        self.byz_frac = byz_frac
        self.n_byz = int(n_agents * byz_frac)
        self.n_honest = n_agents - self.n_byz

        # Observation: [R_t, mean_lambda, depleted_flag, t/T]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )
        # Action: λ_i ∈ [0, 1]
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

        self.reset()

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.R = 0.6  # Start with healthy resource (slightly above PGG's 0.5)
        self.t = 0
        self.prev_mean_lambda = 0.5
        self.terminated = False
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        return np.array([
            self.R,
            self.prev_mean_lambda,
            float(self.R < self.r_crit),
            self.t / self.T,
        ], dtype=np.float32)

    def step(self, actions_honest: np.ndarray):
        """
        Args:
            actions_honest: shape (n_honest,) — extraction rates λ_i ∈ [0,1]
        Returns:
            obs, rewards, terminated, truncated, info
        """
        if self.terminated:
            return self._get_obs(), np.zeros(self.n_honest), True, False, {}

        # Build full lambda vector (honest + byzantine)
        lambdas = np.zeros(self.n_agents)
        lambdas[:self.n_honest] = np.clip(actions_honest, 0, 1)
        # Byzantine agents: max extraction (greedy = λ=1.0)
        lambdas[self.n_honest:] = 1.0

        # --- Harvest extraction ---
        mean_lambda = float(np.mean(lambdas))
        total_extraction = mean_lambda * self.R * self.extraction_rate * self.n_agents
        self.prev_mean_lambda = mean_lambda

        # --- Regrowth (nonlinear, tipping-point) ---
        if self.R < self.r_crit:
            regrowth = 0.01  # Depleted: near-zero recovery
        elif self.R < self.r_recov:
            regrowth = 0.04  # Fragile recovery
        else:
            regrowth = self.regrowth_rate  # Healthy logistic growth

        # Logistic regrowth: R * (1-R) term prevents R > 1
        regrowth_amount = regrowth * self.R * (1.0 - self.R)

        # Shock
        shock = self.shock_mag if self.np_random.random() < self.shock_prob else 0.0

        # Update resource
        self.R = np.clip(self.R - total_extraction + regrowth_amount - shock, 0, 1)

        self.t += 1

        # --- Individual rewards ---
        # Each agent benefits proportional to their extraction × resource
        rewards_all = lambdas * self.R * self.harvest_value
        rewards = rewards_all[:self.n_honest].astype(np.float32)

        # --- Termination ---
        survived = self.R > 0
        terminated = not survived or self.t >= self.T
        self.terminated = terminated

        info = {
            "resource": self.R,
            "survived": survived,
            "mean_lambda": float(np.mean(lambdas[:self.n_honest])),
            "mean_lambda_all": mean_lambda,
            "welfare": float(np.mean(rewards_all)),
            "total_extraction": total_extraction,
            "regrowth": regrowth_amount,
        }

        return self._get_obs(), rewards, terminated, False, info


# ─── Quick self-test ─────────────────────────────────────────
if __name__ == "__main__":
    env = HarvestEnv()
    print("=== Harvest Env Self-Test ===")

    # Test 1: Conservative agents (λ=0.3)
    obs, _ = env.reset(seed=42)
    for t in range(50):
        actions = np.full(env.n_honest, 0.3)
        obs, rewards, terminated, truncated, info = env.step(actions)
        if terminated:
            break
    print(f"Conservative (λ=0.3): R={info['resource']:.3f}, "
          f"survived={info['survived']}, welfare={np.mean(rewards):.1f}, steps={t+1}")

    # Test 2: Greedy agents (λ=0.8)
    obs, _ = env.reset(seed=42)
    for t in range(50):
        actions = np.full(env.n_honest, 0.8)
        obs, rewards, terminated, truncated, info = env.step(actions)
        if terminated:
            break
    print(f"Greedy (λ=0.8):       R={info['resource']:.3f}, "
          f"survived={info['survived']}, welfare={np.mean(rewards):.1f}, steps={t+1}")

    # Test 3: Mixed (some greedy, some conservative)
    obs, _ = env.reset(seed=42)
    for t in range(50):
        actions = np.concatenate([
            np.full(env.n_honest // 2, 0.2),
            np.full(env.n_honest - env.n_honest // 2, 0.9),
        ])
        obs, rewards, terminated, truncated, info = env.step(actions)
        if terminated:
            break
    print(f"Mixed (0.2/0.9):      R={info['resource']:.3f}, "
          f"survived={info['survived']}, welfare={np.mean(rewards):.1f}, steps={t+1}")

    print("\nHarvest env self-test PASSED!")
