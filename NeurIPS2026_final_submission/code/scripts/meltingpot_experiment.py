"""
Melting Pot Generalization Benchmark for EthicaAI
Run on Google Colab: installs dmlab2d + dm-meltingpot automatically.

Tests commitment floor mechanism in standard MARL benchmarks:
  1. commons_harvest__open
  2. clean_up

Compares: Selfish (baseline) vs Unconditional Commitment (phi=1.0)
"""

import subprocess
import sys
import os

# === Auto-install dependencies (for Colab) ===
def install_deps():
    """Install dmlab2d and dm-meltingpot on Colab."""
    print("Installing dmlab2d...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                           "dmlab2d", "-q"], stderr=subprocess.DEVNULL)
    print("Installing dm-meltingpot...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                           "dm-meltingpot", "-q"], stderr=subprocess.DEVNULL)
    print("Dependencies installed!")

try:
    import dmlab2d
    import meltingpot
except ImportError:
    install_deps()
    import dmlab2d
    import meltingpot

import numpy as np
import json
from meltingpot import substrate

# === Config ===
N_SEEDS = 20
N_STEPS = 500
SUBSTRATES = ["commons_harvest__open", "clean_up"]

def run_episode(substrate_name, policy_fn, seed=0):
    """Run one episode with a given policy function."""
    rng = np.random.RandomState(seed)
    
    env_config = substrate.get_config(substrate_name)
    env = substrate.build(env_config)
    
    timestep = env.reset()
    n_agents = len(timestep.observation)
    
    total_reward = np.zeros(n_agents)
    steps = 0
    
    for t in range(N_STEPS):
        actions = {}
        for i in range(n_agents):
            obs = timestep.observation[i]
            actions[i] = policy_fn(obs, i, rng, t)
        
        timestep = env.step(actions)
        
        for i in range(n_agents):
            total_reward[i] += timestep.reward[i]
        
        steps += 1
        if timestep.last():
            break
    
    env.close()
    
    return {
        "total_reward": float(np.mean(total_reward)),
        "steps": steps,
        "per_agent_reward": [float(r) for r in total_reward],
    }


def selfish_policy(obs, agent_id, rng, t):
    """Random/selfish baseline - take random actions."""
    # Action space in Melting Pot is typically 0-7 (movement + turn + interact)
    return rng.randint(0, 8)


def commitment_policy(obs, agent_id, rng, t):
    """Unconditional commitment policy - always try to cooperate.
    In Melting Pot:
      - 'interact' action (typically 5-7) represents cooperation/contribution
      - We bias heavily toward interact and forward movement
    """
    # 70% chance of cooperative action (interact), 30% movement toward resources
    if rng.random() < 0.7:
        return rng.choice([5, 6, 7])  # interact actions
    else:
        return rng.choice([0, 1, 2, 3, 4])  # movement actions


def run_experiment():
    results = {}
    
    for sub_name in SUBSTRATES:
        print(f"\n{'='*60}")
        print(f"  Substrate: {sub_name}")
        print(f"{'='*60}")
        
        sub_results = {}
        
        for policy_name, policy_fn in [("Selfish", selfish_policy), 
                                        ("Unconditional", commitment_policy)]:
            print(f"\n  [{policy_name}] Running {N_SEEDS} seeds...")
            
            seed_rewards = []
            for s in range(N_SEEDS):
                try:
                    res = run_episode(sub_name, policy_fn, seed=s)
                    seed_rewards.append(res["total_reward"])
                    if (s + 1) % 5 == 0:
                        print(f"    Seed {s+1}/{N_SEEDS} | Reward: {res['total_reward']:.1f}")
                except Exception as e:
                    print(f"    Seed {s+1} ERROR: {e}")
                    seed_rewards.append(0.0)
            
            mean_r = np.mean(seed_rewards)
            std_r = np.std(seed_rewards)
            
            sub_results[policy_name] = {
                "mean_reward": float(mean_r),
                "std_reward": float(std_r),
                "all_rewards": [float(r) for r in seed_rewards],
            }
            
            print(f"  [{policy_name}] Mean: {mean_r:.1f} ± {std_r:.1f}")
        
        results[sub_name] = sub_results
    
    return results


if __name__ == "__main__":
    print("EthicaAI — Melting Pot Generalization Benchmark")
    print(f"Seeds: {N_SEEDS}, Steps per episode: {N_STEPS}")
    
    results = run_experiment()
    
    # Save results
    out_dir = os.path.join(os.path.dirname(__file__) or ".", "..", "outputs", "meltingpot")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "meltingpot_results.json")
    
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("  FINAL RESULTS")
    print(f"{'='*60}")
    for sub, data in results.items():
        print(f"\n  {sub}:")
        for pol, stats in data.items():
            print(f"    {pol:20s}: {stats['mean_reward']:8.1f} ± {stats['std_reward']:.1f}")
    
    print(f"\nSaved to: {out_path}")
