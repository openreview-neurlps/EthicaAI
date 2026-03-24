import numpy as np
import json
import time
from meltingpot import substrate

# Config
N_SEEDS = 10
N_STEPS = 500
SUBSTRATE = 'commons_harvest__open'

print(f'Loading substrate: {SUBSTRATE}')
env_config = substrate.get_config(SUBSTRATE)
print(f'Config loaded successfully!')

def run_episode(substrate_name, policy_fn, seed=0, n_steps=500):
    rng = np.random.RandomState(seed)
    
    env_config = substrate.get_config(substrate_name)
    roles = env_config.default_player_roles
    env = substrate.build(substrate_name, roles=roles)
    
    timestep = env.reset()
    n_agents = len(timestep.observation)
    
    total_rewards = np.zeros(n_agents)
    steps_alive = 0
    
    for t in range(n_steps):
        actions = []
        for i in range(n_agents):
            obs = timestep.observation[i]
            actions.append(policy_fn(obs, rng, i))
        
        timestep = env.step(actions)
        
        for i in range(n_agents):
            r = timestep.reward[i] if hasattr(timestep.reward, '__getitem__') else 0
            total_rewards[i] += float(r)
        
        steps_alive += 1
        if timestep.last():
            break
            
    env.close()
    return {
        'welfare': float(np.mean(total_rewards)),
        'total_reward': float(np.sum(total_rewards)),
        'steps': steps_alive,
        'n_agents': n_agents,
    }

# Policy functions
def random_policy(obs, rng, agent_id):
    return rng.randint(0, 8)

def greedy_harvest_policy(obs, rng, agent_id):
    if rng.random() < 0.7:
        return rng.randint(1, 5)  # move
    else:
        return rng.randint(0, 8)  # random

def restrained_policy(obs, rng, agent_id):
    if rng.random() < 0.3:
        return 0  # noop
    return rng.randint(1, 5)  # move

policies = {
    'Random': random_policy,
    'Greedy Harvest': greedy_harvest_policy,
    'Restrained (Floor-like)': restrained_policy,
}

results = {}
t0 = time.time()

for name, policy_fn in policies.items():
    print(f'\n=== {name} ===')
    runs = []
    for seed in range(N_SEEDS):
        r = run_episode(SUBSTRATE, policy_fn, seed=seed, n_steps=N_STEPS)
        runs.append(r)
        print(f'  Seed {seed}: W={r["welfare"]:.2f}, steps={r["steps"]}')
    
    agg = {
        'welfare_mean': float(np.mean([r['welfare'] for r in runs])),
        'welfare_std': float(np.std([r['welfare'] for r in runs])),
        'steps_mean': float(np.mean([r['steps'] for r in runs])),
        'n_agents': runs[0]['n_agents'],
    }
    results[name] = agg

total_time = time.time() - t0

# Save results
output = {
    'experiment': 'Melting Pot ' + SUBSTRATE,
    'seeds': N_SEEDS,
    'steps': N_STEPS,
    'results': results,
    'time_seconds': total_time,
}

out_path = '/mnt/d/00.test/PAPER/EthicaAI/NeurIPS2026_final_submission/code/outputs/meltingpot_wsl_results.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)

print("\nSaved to:", out_path)
print("ALL DONE. Time: {:.1f}s".format(total_time))
