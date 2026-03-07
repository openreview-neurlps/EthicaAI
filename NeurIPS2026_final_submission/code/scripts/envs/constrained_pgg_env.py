import numpy as np

class ConstrainedPGGEnv:
    """
    Non-linear Public Goods Game environment formulated as a Constrained MDP.
    Returns (obs, reward, done, info) where info contains 'cost' for safety constraints.
    """
    def __init__(self, n_agents=10, r_init=0.5, r_crit=0.15, r_recov=0.25, max_steps=50, byz_frac=0.3):
        self.n_agents = n_agents
        self.r_init = r_init
        self.r_crit = r_crit
        self.r_recov = r_recov
        self.max_steps = max_steps
        
        self.byz_frac = byz_frac
        self.n_byz = int(self.n_agents * self.byz_frac)
        self.n_normal = self.n_agents - self.n_byz
        
        self.reset()
        
    def reset(self):
        self.r = self.r_init
        self.step_count = 0
        return self._get_obs()
        
    def _get_obs(self):
        # Observation is just the current resource level for simplicity in this abstract model
        return np.array([self.r], dtype=np.float32)
        
    def _f_R(self, r):
        # Non-linear regeneration function
        if r < self.r_crit:
            return 0.01
        elif r < self.r_recov:
            return 0.03
        else:
            return 0.10
            
    def step(self, normal_actions):
        """
        normal_actions: array of shape (n_normal,) representing lambda values in [0, 1]
        Byzantine agents always act with lambda = 0.
        """
        self.step_count += 1
        
        # Byzantine agents contribute nothing
        byz_actions = np.zeros(self.n_byz)
        all_actions = np.concatenate([normal_actions, byz_actions])
        
        # Effective lambda (cooperation level)
        mean_lambda = np.mean(all_actions)
        
        # Immediate reward (welfare) computation
        # Individual gets (1 - lambda) + M/N * sum(lambdas)
        # M/N is synergy factor. Let M = N * 0.8 (common in strict dilemmas)
        M_ratio = 0.8
        rewards = (1.0 - normal_actions) + M_ratio * mean_lambda
        
        # Resource dynamics
        consumption = 0.10 * (1.0 - mean_lambda)
        regeneration = self._f_R(self.r)
        
        # Update resource
        self.r = self.r - consumption + regeneration
        self.r = np.clip(self.r, 0.0, 1.0)
        
        # Check termination & constraints
        done = self.step_count >= self.max_steps
        
        # Cost is 1 if resource drops below critical threshold (collapse)
        cost = 1.0 if self.r < self.r_crit else 0.0
        
        if self.r < self.r_crit:
            # If collapse happens, everyone gets zero reward onwards, and episode ends early
            rewards = np.zeros(self.n_normal)
            done = True
            
        info = {
            'cost': cost,
            'survival': 0.0 if cost > 0 else 1.0,
            'resource': self.r
        }
        
        return self._get_obs(), rewards, done, info
