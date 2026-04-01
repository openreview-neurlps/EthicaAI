"""
Gordon-Schaefer Fishery Model — Empirically Grounded TPSD Environment
=====================================================================
Based on: Gordon (1954), Schaefer (1957), Clark (1990), Scheffer (2009).

This environment implements the classic bioeconomic fishery model with
Allee-effect tipping-point dynamics, providing a scientifically grounded
TPSD benchmark independent of our custom PGG environments.

Resource dynamics:
    B_{t+1} = B_t + r(B_t) · B_t · (1 - B_t/K) - Σ_i q · e_i · B_t - ξ_t

Where:
    B_t   : biomass (shared resource), normalized to [0, 1]
    r(B)  : intrinsic growth rate with Allee effect (tipping point)
    K     : carrying capacity (= 1.0 normalized)
    q     : catchability coefficient
    e_i   : fishing effort of agent i (= 1 - λ_i)
    ξ_t   : stochastic environmental shock

Allee Effect (Tipping Point):
    r(B) = r_normal = 0.30   if B >= B_crit
    r(B) = r_collapse = 0.01 if B < B_crit  (Depensation / Allee collapse)

This creates the same structural TPSD property as our PGG:
    - Below B_crit, recovery is near-impossible
    - Standard RL agents converge to moderate extraction (Nash Trap)
    - Only commitment floors (restraint) can prevent collapse

Paper Ref: Scheffer et al. (2009) "Early-warning signals for critical transitions"
           Clark (1990) "Mathematical Bioeconomics"
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FisheryEnv(gym.Env):
    """Gordon-Schaefer Fishery with Allee-effect tipping point."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_agents: int = 20,
        byz_frac: float = 0.3,
        t_horizon: int = 50,
        # Ecological parameters (Clark 1990, Scheffer 2009)
        carrying_capacity: float = 1.0,
        r_normal: float = 0.35,       # Growth rate above critical biomass
        r_collapse: float = 0.01,     # Growth rate below critical (Allee effect)
        b_crit: float = 0.15,         # Critical biomass (tipping point)
        b_recov: float = 0.25,        # Recovery threshold (hysteresis)
        catchability: float = 0.018,  # Catchability coefficient q
        # Stochastic shocks
        p_shock: float = 0.08,
        delta_shock: float = 0.12,
        # Initial conditions
        b_init: float = 0.60,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.n_byz = int(n_agents * byz_frac)
        self.n_honest = n_agents - self.n_byz
        self.t_horizon = t_horizon

        # Ecological parameters
        self.K = carrying_capacity
        self.r_normal = r_normal
        self.r_collapse = r_collapse
        self.b_crit = b_crit
        self.b_recov = b_recov
        self.q = catchability
        self.p_shock = p_shock
        self.delta_shock = delta_shock
        self.b_init = b_init

        # Observation: [biomass, mean_effort_prev, own_lambda_prev, crisis_flag]
        self.obs_dim = 4
        self.observation_space = spaces.Box(
            low=np.zeros(self.obs_dim, dtype=np.float32),
            high=np.ones(self.obs_dim, dtype=np.float32),
            dtype=np.float32,
        )
        # Action: restraint level λ ∈ [0,1] (effort = 1 - λ)
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.reset()

    def _growth_rate(self, B: float) -> float:
        """Allee-effect growth rate (Scheffer 2009)."""
        if B < self.b_crit:
            return self.r_collapse  # Near-irreversible collapse
        elif B < self.b_recov:
            # Hysteresis zone: linear interpolation
            frac = (B - self.b_crit) / (self.b_recov - self.b_crit)
            return self.r_collapse + frac * (self.r_normal - self.r_collapse)
        else:
            return self.r_normal  # Healthy stock

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()

        self.B = self.b_init
        self.t = 0
        self.prev_mean_effort = 0.5
        self.prev_lambdas = np.full(self.n_honest, 0.5)
        self.collapsed = False

        obs = self._get_obs()
        return obs, {"biomass": self.B}

    def _get_obs(self):
        """Per-agent observation vector."""
        crisis = 1.0 if self.B < self.b_crit else 0.0
        return np.array([
            self.B,
            self.prev_mean_effort,
            np.mean(self.prev_lambdas),
            crisis,
        ], dtype=np.float32)

    def step(self, lambdas):
        """
        Args:
            lambdas: array of restraint levels [0,1] for honest agents.
                     λ=1 → full restraint (no fishing), λ=0 → max extraction.
        Returns:
            obs, rewards, terminated, truncated, info
        """
        self.t += 1
        lambdas = np.clip(np.asarray(lambdas, dtype=np.float64), 0.0, 1.0)

        # --- Fishing effort ---
        honest_effort = 1.0 - lambdas  # effort = 1 - restraint
        byz_effort = np.ones(self.n_byz)  # Byzantine: max extraction
        all_effort = np.concatenate([honest_effort, byz_effort])
        total_catch = self.q * np.sum(all_effort) * self.B

        # --- Individual harvest (payoff) ---
        # Each agent's catch proportional to their effort
        honest_catch = self.q * honest_effort * self.B
        rewards = honest_catch  # Direct harvest reward

        # --- Biomass dynamics (Gordon-Schaefer + Allee) ---
        r = self._growth_rate(self.B)
        growth = r * self.B * (1.0 - self.B / self.K)
        shock = self.delta_shock if self.rng.random() < self.p_shock else 0.0

        self.B = np.clip(self.B + growth - total_catch - shock, 0.0, self.K)

        # --- Collapse check ---
        terminated = False
        if self.B <= 0.001:
            self.collapsed = True
            terminated = True
            rewards = np.zeros(self.n_honest)  # Fishery collapses: no more income

        if self.t >= self.t_horizon:
            terminated = True

        # --- Update state ---
        self.prev_mean_effort = float(np.mean(all_effort))
        self.prev_lambdas = lambdas.copy()

        obs = self._get_obs()
        info = {
            "biomass": float(self.B),
            "growth_rate": float(r),
            "total_catch": float(total_catch),
            "mean_lambda": float(np.mean(lambdas)),
            "collapsed": self.collapsed,
            "survived": not self.collapsed,
        }

        return obs, rewards, terminated, False, info


# ─── Quick sanity test ────────────────────────────────────────
if __name__ == "__main__":
    env = FisheryEnv(n_agents=20, byz_frac=0.3)

    # Test 1: Full restraint (λ=1) → should survive
    obs, info = env.reset(seed=42)
    for t in range(50):
        lambdas = np.ones(env.n_honest)
        obs, rewards, done, _, info = env.step(lambdas)
        if done:
            break
    print(f"Full restraint: B_final={info['biomass']:.3f}, survived={info['survived']}")

    # Test 2: No restraint (λ=0) → should collapse
    obs, info = env.reset(seed=42)
    for t in range(50):
        lambdas = np.zeros(env.n_honest)
        obs, rewards, done, _, info = env.step(lambdas)
        if done:
            break
    print(f"No restraint: B_final={info['biomass']:.3f}, survived={info['survived']}")

    # Test 3: Half restraint (λ=0.5) → Nash Trap zone
    obs, info = env.reset(seed=42)
    for t in range(50):
        lambdas = np.full(env.n_honest, 0.5)
        obs, rewards, done, _, info = env.step(lambdas)
        if done:
            break
    print(f"Half restraint: B_final={info['biomass']:.3f}, survived={info['survived']}")
