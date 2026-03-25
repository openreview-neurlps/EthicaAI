"""
PettingZoo ParallelEnv Wrapper for Nonlinear PGG
================================================
Converts the custom Gymnasium environment into a standard AEC/Parallel
PettingZoo environment. This enables seamless integration with PureJaxRL,
RLlib, and other state-of-the-art Deep MARL frameworks.
"""
import functools
import numpy as np
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv
from nonlinear_pgg_env import NonlinearPGGEnv

class PettingZooPGGEnv(ParallelEnv):
    metadata = {
        "name": "nonlinear_pgg_v1",
        "is_parallelizable": True,
        "render_modes": [],
    }

    def __init__(self, **kwargs):
        """
        Initializes the wrapped Nonlinear PGG environment.
        Accepts the same kwargs as NonlinearPGGEnv.
        """
        super().__init__()
        self.env = NonlinearPGGEnv(**kwargs)
        
        # Only honest agents act and receive rewards in this setup
        self.possible_agents = [f"agent_{i}" for i in range(self.env.n_honest)]
        self.agents = self.possible_agents[:]
        
        self._action_spaces = {agent: self.env.action_space for agent in self.possible_agents}
        self._observation_spaces = {agent: self.env.observation_space for agent in self.possible_agents}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        global_obs, _ = self.env.reset(seed=seed, options=options)
        
        # All agents receive the same global observation (homogeneous state)
        observations = {agent: global_obs.copy() for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos

    def step(self, actions: dict):
        """
        Takes a dict of actions {agent_id: action_value}.
        """
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # Convert action dict to array format expected by the base env
        action_array = np.zeros(self.env.n_honest, dtype=np.float32)
        for i, agent in enumerate(self.possible_agents):
            if agent in actions:
                # Continuous action is typically a 1D array
                action_array[i] = actions[agent][0] if isinstance(actions[agent], np.ndarray) else actions[agent]

        global_obs, rewards_array, terminated, truncated, global_info = self.env.step(action_array)

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for i, agent in enumerate(self.possible_agents):
            observations[agent] = global_obs.copy()
            rewards[agent] = float(rewards_array[i])
            terminations[agent] = terminated
            truncations[agent] = truncated
            infos[agent] = global_info.copy()

        if terminated or truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def state(self):
        """Returns the global state for Centralized Critic architectures (e.g. MAPPO)."""
        return self.env.get_global_state()


# ─── Quick self-test ─────────────────────────────────────────
if __name__ == "__main__":
    from pettingzoo.test import parallel_api_test
    
    print("Running PettingZoo Parallel API test...")
    env = PettingZooPGGEnv()
    parallel_api_test(env, num_cycles=100)
    print("API test PASSED!")
    
    print("\nRunning simulated interaction...")
    obs, info = env.reset(seed=42)
    print(f"Number of acting agents: {len(env.agents)}")
    print(f"Initial Obs for agent_0: {obs['agent_0']}")
    
    actions = {agent: np.array([0.5], dtype=np.float32) for agent in env.agents}
    next_obs, rewards, terminations, truncations, infos = env.step(actions)
    
    print(f"Reward for agent_0 after 1 step (mean λ=0.5): {rewards['agent_0']:.2f}")
    print(f"Termination status: {terminations['agent_0']}")
    print("Done!")
