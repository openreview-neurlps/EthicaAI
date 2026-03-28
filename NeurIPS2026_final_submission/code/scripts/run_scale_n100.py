"""
N=100 Scale Validation: PyTorch Deep MARL
==========================================
Validates that:
1. Nash Trap persists at N=100 (Selfish REINFORCE)
2. MACCL commitment floor still works at N=100
3. Scaling does not break the mechanism

Runs on GPU, 10 seeds (N=100 is 5x more agents → longer per-seed).
"""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Beta

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'envs'))
from pettingzoo_pgg_env import PettingZooPGGEnv

HIDDEN_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'scale_n100_deep')

# ============================================================
# Networks (same architecture as 20-agent, tests generalization)
# ============================================================
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_DIM), nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.Tanh()
        )
        self.alpha_head = nn.Linear(HIDDEN_DIM, act_dim)
        self.beta_head = nn.Linear(HIDDEN_DIM, act_dim)
        self.value_head = nn.Linear(HIDDEN_DIM, 1)

    def get_action_value(self, x, action=None):
        features = self.net(x)
        alpha = torch.nn.functional.softplus(self.alpha_head(features)) + 1.0
        beta = torch.nn.functional.softplus(self.beta_head(features)) + 1.0
        v = self.value_head(features)
        dist = Beta(alpha, beta)
        if action is None:
            raw_action = dist.sample()
        else:
            raw_action = action
        return raw_action, dist.log_prob(raw_action).sum(dim=-1), v

    def get_action_value_floor(self, x, floor_val, action=None):
        raw, lp, v = self.get_action_value(x, action)
        projected = raw + torch.nn.functional.relu(floor_val - raw)
        return projected, raw, lp, v


class FloorNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor(0.0))
        self.w2 = nn.Parameter(torch.tensor(0.0))
        self.w3 = nn.Parameter(torch.tensor(5.0))

    def forward(self, R):
        return torch.sigmoid(self.w1 * R + self.w2 * (R ** 2) + self.w3)


# ============================================================
# Selfish REINFORCE at N=100
# ============================================================
def run_selfish_seed(seed, n_agents=100, byz_frac=0.30, max_steps=8000):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = PettingZooPGGEnv(n_agents=n_agents, byz_frac=byz_frac)
    num_agents = len(env.possible_agents)
    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]

    # Shared policy (parameter sharing for scalability)
    policy = PolicyNet(obs_dim).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    global_step = 0
    obs, _ = env.reset(seed=seed)
    all_survivals = []
    all_lambdas = []
    all_welfares = []

    while global_step < max_steps:
        batch_obs, batch_acts, batch_logprobs, batch_rewards, batch_values = [], [], [], [], []
        ep_lambda_sum, ep_steps = 0, 0

        for _ in range(200):
            global_step += 1
            actions = {}
            step_lam = 0

            with torch.no_grad():
                for agent_id in env.agents:
                    o = torch.tensor(obs[agent_id], dtype=torch.float32).to(DEVICE).unsqueeze(0)
                    a, lp, v = policy.get_action_value(o)
                    actions[agent_id] = a.cpu().numpy().flatten()
                    step_lam += float(actions[agent_id][0])
                    batch_obs.append(obs[agent_id])
                    batch_acts.append(a.cpu().numpy().flatten())
                    batch_logprobs.append(lp.item())
                    batch_values.append(v.item())

            next_obs, rewards, terms, truncs, infos = env.step(actions)
            n_act = len(actions)
            ep_lambda_sum += step_lam / max(n_act, 1)
            ep_steps += 1

            for aid in actions:
                batch_rewards.append(rewards[aid])

            obs = next_obs
            if not env.agents:
                survived = float(infos[list(actions.keys())[0]].get('survived', False))
                welfare = sum(rewards[a] for a in actions) / max(n_act, 1)
                all_survivals.append(survived)
                all_lambdas.append(ep_lambda_sum / max(ep_steps, 1))
                all_welfares.append(welfare)
                ep_lambda_sum, ep_steps = 0, 0
                obs, _ = env.reset(seed=seed + global_step)

        # PPO update
        if len(batch_obs) > 0:
            b_obs = torch.tensor(np.array(batch_obs), dtype=torch.float32).to(DEVICE)
            b_acts = torch.tensor(np.array(batch_acts), dtype=torch.float32).to(DEVICE)
            b_lp = torch.tensor(np.array(batch_logprobs), dtype=torch.float32).to(DEVICE)
            b_r = torch.tensor(np.array(batch_rewards), dtype=torch.float32).to(DEVICE)
            b_v = torch.tensor(np.array(batch_values), dtype=torch.float32).to(DEVICE)

            returns = b_r + 0.99 * torch.cat([b_v[1:], torch.tensor([0.0]).to(DEVICE)])
            adv = returns - b_v
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            idx = np.arange(len(b_obs))
            for _ in range(4):
                np.random.shuffle(idx)
                for s in range(0, len(b_obs), 128):
                    e = s + 128
                    mb = idx[s:e]
                    _, nlp, nv = policy.get_action_value(b_obs[mb], b_acts[mb])
                    ratio = (nlp - b_lp[mb]).exp()
                    pg = -torch.min(adv[mb] * ratio, adv[mb] * torch.clamp(ratio, 0.8, 1.2)).mean()
                    vl = 0.5 * ((nv.view(-1) - returns[mb]) ** 2).mean()
                    loss = pg + 0.5 * vl
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    optimizer.step()

    n = min(30, len(all_survivals))
    return {
        'survival': float(np.mean(all_survivals[-n:])) * 100,
        'lambda': float(np.mean(all_lambdas[-n:])),
        'welfare': float(np.mean(all_welfares[-n:])),
    }


# ============================================================
# MACCL at N=100
# ============================================================
def run_maccl_seed(seed, n_agents=100, byz_frac=0.30, max_steps=8000):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = PettingZooPGGEnv(n_agents=n_agents, byz_frac=byz_frac)
    num_agents = len(env.possible_agents)
    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]

    floor_net = FloorNetwork().to(DEVICE)
    mu = torch.tensor(0.0, requires_grad=True, device=DEVICE)
    primal_opt = optim.Adam(floor_net.parameters(), lr=5e-3)
    policy = PolicyNet(obs_dim).to(DEVICE)
    agent_opt = optim.Adam(policy.parameters(), lr=3e-4)

    SAFETY_DELTA = 0.05
    global_step = 0
    obs, _ = env.reset(seed=seed)
    all_survivals = []
    all_lambdas = []
    all_welfares = []

    while global_step < max_steps:
        batch_obs, batch_acts, batch_logprobs, batch_rewards, batch_values = [], [], [], [], []
        ep_survivals, ep_welfares, ep_lambdas = [], [], []
        ep_lam_sum, ep_steps = 0, 0

        for _ in range(200):
            global_step += 1
            actions = {}
            step_lam = 0

            with torch.no_grad():
                for agent_id in env.agents:
                    o = torch.tensor(obs[agent_id], dtype=torch.float32).to(DEVICE).unsqueeze(0)
                    R_val = o[0, 0]
                    f_val = floor_net(R_val)
                    a_proj, a_raw, lp, v = policy.get_action_value_floor(o, f_val)
                    actions[agent_id] = a_proj.cpu().numpy().flatten()
                    step_lam += float(a_proj.cpu().numpy().flatten()[0])
                    batch_obs.append(obs[agent_id])
                    batch_acts.append(a_raw.cpu().numpy().flatten())
                    batch_logprobs.append(lp.item())
                    batch_values.append(v.item())

            next_obs, rewards, terms, truncs, infos = env.step(actions)
            n_act = len(actions)
            ep_lam_sum += step_lam / max(n_act, 1)
            ep_steps += 1

            for aid in actions:
                batch_rewards.append(rewards[aid])

            obs = next_obs
            if not env.agents:
                survived = float(infos[list(actions.keys())[0]].get('survived', False))
                welfare = sum(rewards[a] for a in actions) / max(n_act, 1)
                ep_survivals.append(survived)
                ep_welfares.append(welfare)
                ep_lambdas.append(ep_lam_sum / max(ep_steps, 1))
                ep_lam_sum, ep_steps = 0, 0
                obs, _ = env.reset(seed=seed + global_step)

        # PPO update
        if len(batch_obs) > 0:
            b_obs = torch.tensor(np.array(batch_obs), dtype=torch.float32).to(DEVICE)
            b_acts = torch.tensor(np.array(batch_acts), dtype=torch.float32).to(DEVICE)
            b_lp = torch.tensor(np.array(batch_logprobs), dtype=torch.float32).to(DEVICE)
            b_r = torch.tensor(np.array(batch_rewards), dtype=torch.float32).to(DEVICE)
            b_v = torch.tensor(np.array(batch_values), dtype=torch.float32).to(DEVICE)

            returns = b_r + 0.99 * torch.cat([b_v[1:], torch.tensor([0.0]).to(DEVICE)])
            adv = returns - b_v
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            idx = np.arange(len(b_obs))
            for _ in range(4):
                np.random.shuffle(idx)
                for s in range(0, len(b_obs), 128):
                    e = s + 128
                    mb = idx[s:e]
                    R_vals = b_obs[mb, 0]
                    f_vals = floor_net(R_vals).unsqueeze(-1)
                    _, _, nlp, nv = policy.get_action_value_floor(b_obs[mb], f_vals, b_acts[mb])
                    ratio = (nlp - b_lp[mb]).exp()
                    pg = -torch.min(adv[mb] * ratio, adv[mb] * torch.clamp(ratio, 0.8, 1.2)).mean()
                    vl = 0.5 * ((nv.view(-1) - returns[mb]) ** 2).mean()
                    loss = pg + 0.5 * vl
                    agent_opt.zero_grad()
                    loss.backward()
                    agent_opt.step()

        # Primal-Dual
        if len(ep_survivals) > 0:
            surv_rate = np.mean(ep_survivals)
            cv = SAFETY_DELTA - surv_rate
            with torch.no_grad():
                mu.add_(1e-2 * cv)
                mu.clamp_(min=0.0)
            primal_opt.zero_grad()
            R_all = b_obs[:, 0]
            f_all = floor_net(R_all).unsqueeze(-1)
            a_proj, _, _, _ = policy.get_action_value_floor(b_obs, f_all, b_acts)
            surrogate = -(adv.detach().unsqueeze(-1) * a_proj).mean()
            penalty = mu.item() * torch.relu(torch.tensor(cv)).to(DEVICE)
            (surrogate + penalty).backward()
            primal_opt.step()

            all_survivals.extend(ep_survivals)
            all_lambdas.extend(ep_lambdas)
            all_welfares.extend(ep_welfares)

    n = min(30, len(all_survivals))
    return {
        'survival': float(np.mean(all_survivals[-n:])) * 100,
        'lambda': float(np.mean(all_lambdas[-n:])),
        'welfare': float(np.mean(all_welfares[-n:])),
        'floor_R01': float(floor_net(torch.tensor(0.1).to(DEVICE)).item()),
    }


# ============================================================
# Bootstrap CI
# ============================================================
def bootstrap_ci(data, n_boot=10000):
    data = np.array(data)
    if len(data) == 0:
        return (0.0, 0.0)
    boots = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    N_SEEDS = args.seeds
    np.random.seed(42)
    t_total = time.time()

    print(f"{'='*60}")
    print(f"  N=100 Scale Validation (PyTorch, {DEVICE})")
    print(f"  {N_SEEDS} seeds per method")
    print(f"{'='*60}")

    # 1. Selfish REINFORCE at N=100
    print(f"\n--- Selfish REINFORCE (N=100, Byz=30%) ---")
    selfish_results = []
    for s in range(N_SEEDS):
        t0 = time.time()
        r = run_selfish_seed(42 + s)
        print(f"  Seed {s+1:2d}/{N_SEEDS}: Surv={r['survival']:5.1f}%  λ={r['lambda']:.3f}  Welf={r['welfare']:.2f}  [{time.time()-t0:.0f}s]")
        selfish_results.append(r)

    # 2. MACCL at N=100
    print(f"\n--- MACCL (N=100, Byz=30%) ---")
    maccl_results = []
    for s in range(N_SEEDS):
        t0 = time.time()
        r = run_maccl_seed(42 + s)
        print(f"  Seed {s+1:2d}/{N_SEEDS}: Surv={r['survival']:5.1f}%  λ={r['lambda']:.3f}  Welf={r['welfare']:.2f}  Floor={r['floor_R01']:.3f}  [{time.time()-t0:.0f}s]")
        maccl_results.append(r)

    # Aggregate
    def aggregate(results, name):
        survs = [r['survival'] for r in results]
        lams = [r['lambda'] for r in results]
        welfs = [r['welfare'] for r in results]
        return {
            'method': name,
            'n_agents': 100,
            'byz_frac': 0.30,
            'n_seeds': len(results),
            'survival_mean': float(np.mean(survs)),
            'survival_std': float(np.std(survs)),
            'survival_ci95': bootstrap_ci(survs),
            'lambda_mean': float(np.mean(lams)),
            'lambda_std': float(np.std(lams)),
            'lambda_ci95': bootstrap_ci(lams),
            'welfare_mean': float(np.mean(welfs)),
            'welfare_std': float(np.std(welfs)),
            'welfare_ci95': bootstrap_ci(welfs),
            'per_seed_survival': survs,
            'per_seed_lambda': lams,
            'per_seed_welfare': welfs,
        }

    output = {
        'experiment': 'N=100 Scale Validation (PyTorch Deep MARL)',
        'config': {'n_agents': 100, 'byz_frac': 0.30, 'device': str(DEVICE), 'torch_version': torch.__version__},
        'selfish': aggregate(selfish_results, 'Selfish REINFORCE'),
        'maccl': aggregate(maccl_results, 'MACCL'),
        'total_time_seconds': time.time() - t_total,
    }

    json_path = os.path.join(OUTPUT_DIR, 'scale_n100_deep_results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  Selfish: Surv={output['selfish']['survival_mean']:.1f}% CI{output['selfish']['survival_ci95']}  λ={output['selfish']['lambda_mean']:.3f}")
    print(f"  MACCL:   Surv={output['maccl']['survival_mean']:.1f}% CI{output['maccl']['survival_ci95']}  λ={output['maccl']['lambda_mean']:.3f}")
    print(f"  Total: {time.time()-t_total:.0f}s")
    print(f"  Saved: {json_path}")
    print(f"{'='*60}")
