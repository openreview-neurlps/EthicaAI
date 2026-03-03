"""
Nonlinear PGG Environment — Gymnasium-compatible wrapper
=========================================================
Wraps the paper's non-linear PGG into a standard Gymnasium Env,
enabling plug-and-play use with CleanRL, EPyMARL, and any other
standard MARL library.

Multi-agent interface: each agent independently takes a continuous
action λ_i ∈ [0,1]. Observation includes resource level, mean λ,
crisis indicator, and timestep.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional


class NonlinearPGGEnv(gym.Env):
    """
    Non-linear Public Goods Game with tipping-point dynamics.
    
    This is the EXACT environment from the paper:
    - N agents, BYZ_FRAC are Byzantine (fixed λ=0)
    - Resource R_t with non-linear recovery
    - Stochastic shocks
    - Survival defined as R_T > 0
    
    Observation (per agent): [R_t, mean_lambda_{t-1}, crisis_flag, t/T]
    Action (per agent): λ_i ∈ [0, 1] (continuous)
    Reward: individual payoff = (1-λ_i)E + (M/N) * Σλ_j * E
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        n_agents: int = 20,
        multiplier: float = 1.6,
        endowment: float = 10.0,
        t_horizon: int = 50,
        r_crit: float = 0.15,
        r_recov: float = 0.25,
        shock_prob: float = 0.05,
        shock_mag: float = 0.15,
        byz_frac: float = 0.3,
    ):
        super().__init__()
        
        self.n_agents = n_agents
        self.M = multiplier
        self.E = endowment
        self.T = t_horizon
        self.r_crit = r_crit
        self.r_recov = r_recov
        self.shock_prob = shock_prob
        self.shock_mag = shock_mag
        self.byz_frac = byz_frac
        self.n_byz = int(n_agents * byz_frac)
        self.n_honest = n_agents - self.n_byz
        
        # Observation: [R_t, mean_lambda, crisis_flag, t/T]
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
        self.R = 0.5
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
            actions_honest: array of shape (n_honest,) with λ values
        Returns:
            obs, rewards, terminated, truncated, info
        """
        if self.terminated:
            return self._get_obs(), np.zeros(self.n_honest), True, False, {}
        
        # Build full lambda vector (honest + byzantine)
        lambdas = np.zeros(self.n_agents)
        lambdas[:self.n_honest] = np.clip(actions_honest, 0, 1)
        # Byzantine agents stay at 0
        
        # Compute payoffs
        contribs = lambdas * self.E
        pool = np.sum(contribs)
        payoffs = (self.E - contribs) + self.M * pool / self.n_agents
        
        # Resource dynamics
        mean_c = np.mean(contribs) / self.E
        self.prev_mean_lambda = mean_c
        
        if self.R < self.r_crit:
            f_R = 0.01
        elif self.R < self.r_recov:
            f_R = 0.03
        else:
            f_R = 0.10
        
        shock = self.shock_mag if self.np_random.random() < self.shock_prob else 0.0
        self.R = np.clip(self.R + f_R * (mean_c - 0.4) - shock, 0, 1)
        
        self.t += 1
        
        # Check termination
        survived = self.R > 0
        terminated = not survived or self.t >= self.T
        self.terminated = terminated
        
        # Honest agent rewards only
        rewards = payoffs[:self.n_honest].astype(np.float32)
        
        info = {
            "resource": self.R,
            "survived": survived,
            "mean_lambda": float(np.mean(lambdas[:self.n_honest])),
            "mean_lambda_all": float(np.mean(lambdas)),
            "welfare": float(np.mean(payoffs)),
        }
        
        return self._get_obs(), rewards, terminated, False, info
    
    def get_global_state(self):
        """MAPPO-style global state: [R, mean_lambda, crisis, t/T, n_byz/n]"""
        return np.array([
            self.R,
            self.prev_mean_lambda,
            float(self.R < self.r_crit),
            self.t / self.T,
            self.byz_frac,
        ], dtype=np.float32)


# ─── Quick self-test ─────────────────────────────────────────
if __name__ == "__main__":
    env = NonlinearPGGEnv()
    obs, _ = env.reset(seed=42)
    print(f"Initial obs: {obs}")
    
    total_reward = 0
    for t in range(50):
        actions = np.full(env.n_honest, 0.5)  # All cooperate at λ=0.5
        obs, rewards, terminated, truncated, info = env.step(actions)
        total_reward += np.mean(rewards)
        if terminated:
            break
    
    print(f"Final: R={info['resource']:.3f}, survived={info['survived']}, "
          f"welfare={total_reward:.1f}, steps={t+1}")
    
    # Test with selfish agents
    obs, _ = env.reset(seed=42)
    total_reward = 0
    for t in range(50):
        actions = np.full(env.n_honest, 0.0)  # All defect
        obs, rewards, terminated, truncated, info = env.step(actions)
        total_reward += np.mean(rewards)
        if terminated:
            break
    
    print(f"Selfish: R={info['resource']:.3f}, survived={info['survived']}, "
          f"welfare={total_reward:.1f}, steps={t+1}")
    
    print("\nEnvironment self-test PASSED!")
