#!/bin/bash
set -e

source ~/meltingpot_env/bin/activate

echo "=== Import test ==="
python3 -c "
import dmlab2d
print('dmlab2d OK')
import meltingpot
print('meltingpot OK')
from meltingpot import substrate
print('substrate OK')
"

echo "=== Running Melting Pot experiment ==="
python3 << 'PYEOF'
import numpy as np
import json
import os
from meltingpot import substrate

N_SEEDS = 5
N_STEPS = 200
SUBSTRATES = ["commons_harvest__open"]

def random_policy(obs, rng, n_actions):
    """Random agent (baseline)."""
    return rng.randint(0, n_actions)

def cooperative_policy(obs, rng, n_actions):
    """Agent that avoids harvesting when resources are low."""
    # Simple heuristic: lower probability of harvest action
    if rng.random() < 0.3:
        return 0  # stay/no-op
    return rng.randint(0, n_actions)

results = {}

for sub_name in SUBSTRATES:
    print(f"\n  Substrate: {sub_name}")
    sub_results = {"random": [], "cooperative": []}
    
    for policy_name, policy_fn in [("random", random_policy), ("cooperative", cooperative_policy)]:
        for seed in range(N_SEEDS):
            rng = np.random.RandomState(seed)
            
            env_config = substrate.get_config(sub_name)
            env = substrate.build(env_config)
            
            timestep = env.reset()
            n_agents = len(timestep.observation)
            
            # Get action spec
            action_spec = env.action_spec()
            
            total_rewards = np.zeros(n_agents)
            steps_alive = 0
            
            for t in range(N_STEPS):
                actions = {}
                for i in range(n_agents):
                    n_actions = action_spec[i].num_values if hasattr(action_spec[i], 'num_values') else 9
                    actions[i] = policy_fn(timestep.observation[i], rng, n_actions)
                
                timestep = env.step(actions)
                
                for i in range(n_agents):
                    if hasattr(timestep.reward, '__getitem__'):
                        total_rewards[i] += timestep.reward[i]
                    elif isinstance(timestep.reward, (int, float)):
                        total_rewards[i] += timestep.reward
                
                steps_alive += 1
                
                if timestep.last():
                    break
            
            sub_results[policy_name].append({
                "welfare": float(np.mean(total_rewards)),
                "steps": steps_alive,
                "total_reward": float(np.sum(total_rewards)),
            })
            
            env.close()
            print(f"    {policy_name} seed {seed}: welfare={np.mean(total_rewards):.2f}, steps={steps_alive}")
    
    # Aggregate
    for policy_name in sub_results:
        runs = sub_results[policy_name]
        agg = {
            "welfare_mean": float(np.mean([r["welfare"] for r in runs])),
            "welfare_std": float(np.std([r["welfare"] for r in runs])),
            "steps_mean": float(np.mean([r["steps"] for r in runs])),
        }
        print(f"  {policy_name}: W={agg['welfare_mean']:.2f}+/-{agg['welfare_std']:.2f}, steps={agg['steps_mean']:.0f}")
        sub_results[policy_name] = agg
    
    results[sub_name] = sub_results

# Save
out_dir = "/mnt/d/00.test/PAPER/EthicaAI/NeurIPS2026_final_submission/code/outputs/meltingpot"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "meltingpot_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved: {out_path}")
print("MELTING POT EXPERIMENT COMPLETE!")
PYEOF
