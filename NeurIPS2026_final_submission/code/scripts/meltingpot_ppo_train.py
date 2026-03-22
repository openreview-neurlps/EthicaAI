import json
import os
import argparse
import numpy as np

import ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces

import meltingpot
from meltingpot import substrate

class SimpleMeltingPotEnv(MultiAgentEnv):
    """A minimal Ray RLlib wrapper for Melting Pot 2.4.0 substrates."""
    def __init__(self, config):
        sub_name = config["substrate"]
        env_config = substrate.get_config(sub_name)
        self.env = substrate.build(sub_name, roles=env_config.default_player_roles)
        self._obs_spec = self.env.observation_spec()
        self._act_spec = self.env.action_spec()
        self._num_players = len(self._obs_spec)
        self._agent_ids = {f"agent_{i}" for i in range(self._num_players)}
        
        # Action space: Discrete
        self.action_space = spaces.Dict({
            f"agent_{i}": spaces.Discrete(self._act_spec[i].num_values)
            for i in range(self._num_players)
        })
        
        # Obs space: Only use RGB for simplicity and CNN training
        self.observation_space = spaces.Dict({
            f"agent_{i}": spaces.Dict({
                "RGB": spaces.Box(
                    low=0, high=255, 
                    shape=self._obs_spec[i]["RGB"].shape, 
                    dtype=np.uint8
                )
            }) for i in range(self._num_players)
        })
        super().__init__()

    def reset(self, *, seed=None, options=None):
        timestep = self.env.reset()
        obs = {f"agent_{i}": {"RGB": timestep.observation[i]["RGB"]} for i in range(self._num_players)}
        return obs, {}

    def step(self, action_dict):
        # Default action is 0 for agents that didn't provide an action (e.g., if dead)
        actions = [action_dict.get(f"agent_{i}", 0) for i in range(self._num_players)]
        timestep = self.env.step(actions)
        
        obs = {f"agent_{i}": {"RGB": timestep.observation[i]["RGB"]} for i in range(self._num_players)}
        rews = {f"agent_{i}": timestep.reward[i] for i in range(self._num_players)}
        
        is_last = bool(timestep.last())
        terminated = {f"agent_{i}": is_last for i in range(self._num_players)}
        terminated["__all__"] = is_last
        truncated = {f"agent_{i}": False for i in range(self._num_players)}
        truncated["__all__"] = False
        
        info = {f"agent_{i}": {} for i in range(self._num_players)}
        return obs, rews, terminated, truncated, info


def env_creator(env_config):
    return SimpleMeltingPotEnv(env_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=10, help="Number of training iterations")
    parser.add_argument("--sub", type=str, default="commons_harvest__open")
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)
    from ray.tune.registry import register_env
    register_env("meltingpot_env", env_creator)

    print(f"Starting PPO training on {args.sub} for {args.iters} iterations...")
    
    config = (
        PPOConfig()
        .environment("meltingpot_env", env_config={"substrate": args.sub})
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .env_runners(num_env_runners=0, num_envs_per_env_runner=1)
        .training(
            gamma=0.99,
            lr=1e-4,
            model={
                "conv_filters": [[16, [8, 8], 4], [32, [4, 4], 2], [256, [11, 11], 1]],
            }
        )
        .multi_agent(
            policies={"default_policy"},
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "default_policy"
        )
    )

    stop = {"training_iteration": args.iters}
    
    analysis = tune.run(
        "PPO",
        config=config.to_dict(),
        stop=stop,
        verbose=1,
    )
    
    # Extract learning curve
    df = analysis.results_df
    if not df.empty and "episode_reward_mean" in df.columns:
        # Tune saves results in a dataframe, let's extract the time series
        # Note: we might have multiple trials, but since it's 1 trial, we take the first
        trial_id = df.index[0]
        hist_df = analysis.trial_dataframes[trial_id]
        
        learning_curve = hist_df["episode_reward_mean"].tolist()
        
        results = {
            "substrate": args.sub,
            "iterations": args.iters,
            "final_reward_mean": learning_curve[-1] if learning_curve else 0.0,
            "learning_curve": learning_curve
        }
        
        out_dir = os.path.join(os.path.dirname(__file__) or ".", "..", "outputs", "meltingpot")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"ppo_learning_curve_{args.sub}.json")
        
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved learning curve to {out_path}")
    else:
        print("Warning: Could not extract learning curve from Ray results.")
        
    print("Training Complete!")
    ray.shutdown()
