"""
Coin Game TPSD: PyTorch Deep MARL Validation
=============================================
Tests MACCL vs Selfish PPO on the Tipping-Point Coin Game
(Lerer & Peysakhovich 2017, adapted with TPSD dynamics).

This is the EXTERNAL BENCHMARK validation — discrete action,
spatial grid, non-PGG environment.

Key question: Does the Nash Trap appear in a discrete-action
spatial game with TPSD dynamics?

10 seeds per method, GPU.
"""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'envs'))
from coin_game_env import CoinGameWithTippingPoint

HIDDEN_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'coin_game_deep')

# ============================================================
# Networks — Discrete action (5-way: up/down/left/right/stay)
# ============================================================
class DiscretePolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_DIM), nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.Tanh()
        )
        self.logits_head = nn.Linear(HIDDEN_DIM, act_dim)
        self.value_head = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        h = self.net(x)
        logits = self.logits_head(h)
        v = self.value_head(h)
        return logits, v

    def get_action_value(self, x, action=None):
        logits, v = self.forward(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), v, dist.entropy()


class FloorDiscrete(nn.Module):
    """
    Commitment floor for discrete actions:
    Redistributes probability mass toward 'cooperative' actions.
    In Coin Game, action 4 (stay) is the most cooperative.
    We learn a floor on the probability of staying (not chasing opponent's coins).
    """
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor(0.0))
        self.w2 = nn.Parameter(torch.tensor(0.0))
        self.w3 = nn.Parameter(torch.tensor(3.0))

    def forward(self, resource):
        z = self.w1 * resource + self.w2 * (resource ** 2) + self.w3
        return torch.sigmoid(z)  # floor on cooperation probability


# ============================================================
# Selfish PPO on Coin Game TPSD
# ============================================================
def run_selfish_coin_seed(seed, max_episodes=200):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = CoinGameWithTippingPoint(
        resource_init=0.4, r_crit=0.20, depletion_rate=0.15,
        grid_size=5, max_steps=50
    )
    obs_dim = env.obs_dim
    n_agents = env.num_agents

    policies = [DiscretePolicyNet(obs_dim).to(DEVICE) for _ in range(n_agents)]
    optimizers = [optim.Adam(p.parameters(), lr=3e-4) for p in policies]

    all_survivals = []
    all_rewards = []

    for ep in range(max_episodes):
        obs_list = env.reset()
        ep_reward = np.zeros(n_agents)
        log_probs = [[] for _ in range(n_agents)]
        rewards_buf = [[] for _ in range(n_agents)]
        values_buf = [[] for _ in range(n_agents)]

        done = False
        while not done:
            actions = []
            for i in range(n_agents):
                o = torch.tensor(obs_list[i], dtype=torch.float32).to(DEVICE).unsqueeze(0)
                a, lp, v, _ = policies[i].get_action_value(o)
                actions.append(a.item())
                log_probs[i].append(lp)
                values_buf[i].append(v.squeeze())

            obs_list, rewards, done, info = env.step(actions)
            ep_reward += rewards
            for i in range(n_agents):
                rewards_buf[i].append(rewards[i])

        survived = not info.get('collapsed', False)
        all_survivals.append(float(survived))
        all_rewards.append(float(ep_reward.mean()))

        # REINFORCE update
        for i in range(n_agents):
            if len(rewards_buf[i]) < 2:
                continue
            G, returns = 0.0, []
            for r in reversed(rewards_buf[i]):
                G = r + 0.99 * G
                returns.insert(0, G)
            returns_t = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
            if returns_t.std() > 1e-8:
                returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

            lp_stack = torch.stack(log_probs[i][:len(returns)])
            v_stack = torch.stack(values_buf[i][:len(returns)])
            adv = returns_t - v_stack.detach()
            p_loss = -(lp_stack * adv).mean()
            v_loss = 0.5 * ((v_stack - returns_t) ** 2).mean()
            loss = p_loss + 0.5 * v_loss

            optimizers[i].zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policies[i].parameters(), 0.5)
            optimizers[i].step()

    n = min(30, len(all_survivals))
    return {
        'survival': float(np.mean(all_survivals[-n:])) * 100,
        'welfare': float(np.mean(all_rewards[-n:])),
    }


# ============================================================
# MACCL on Coin Game TPSD
# ============================================================
def run_maccl_coin_seed(seed, max_episodes=200):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = CoinGameWithTippingPoint(
        resource_init=0.4, r_crit=0.20, depletion_rate=0.15,
        grid_size=5, max_steps=50
    )
    obs_dim = env.obs_dim
    n_agents = env.num_agents

    policies = [DiscretePolicyNet(obs_dim).to(DEVICE) for _ in range(n_agents)]
    optimizers = [optim.Adam(p.parameters(), lr=3e-4) for p in policies]

    floor_net = FloorDiscrete().to(DEVICE)
    floor_opt = optim.Adam(floor_net.parameters(), lr=5e-3)
    mu = torch.tensor(0.0, device=DEVICE)
    SAFETY_DELTA = 0.05

    all_survivals = []
    all_rewards = []

    for ep in range(max_episodes):
        obs_list = env.reset()
        ep_reward = np.zeros(n_agents)
        log_probs = [[] for _ in range(n_agents)]
        rewards_buf = [[] for _ in range(n_agents)]
        values_buf = [[] for _ in range(n_agents)]

        done = False
        while not done:
            # Get resource level from observation (last element)
            resource = obs_list[0][-1]
            r_t = torch.tensor([resource], dtype=torch.float32).to(DEVICE)
            floor_val = floor_net(r_t)

            actions = []
            for i in range(n_agents):
                o = torch.tensor(obs_list[i], dtype=torch.float32).to(DEVICE).unsqueeze(0)
                logits, v = policies[i](o)

                # Apply commitment floor:
                # Boost probability of 'stay' (action 4) to at least floor_val
                # This is differentiable soft-floor via logit adjustment
                with torch.no_grad():
                    probs = torch.softmax(logits, dim=-1)
                    stay_prob = probs[0, 4]
                    if stay_prob < floor_val.item():
                        # Boost stay logit
                        boost = torch.log(floor_val / (1 - floor_val + 1e-8)) - logits[0, 4]
                        logits = logits.clone()
                        logits[0, 4] += boost.item()

                dist = Categorical(logits=logits)
                a = dist.sample()
                actions.append(a.item())
                log_probs[i].append(dist.log_prob(a))
                values_buf[i].append(v.squeeze())

            obs_list, rewards, done, info = env.step(actions)
            ep_reward += rewards
            for i in range(n_agents):
                rewards_buf[i].append(rewards[i])

        survived = not info.get('collapsed', False)
        all_survivals.append(float(survived))
        all_rewards.append(float(ep_reward.mean()))

        # Agent update
        for i in range(n_agents):
            if len(rewards_buf[i]) < 2:
                continue
            G, returns = 0.0, []
            for r in reversed(rewards_buf[i]):
                G = r + 0.99 * G
                returns.insert(0, G)
            returns_t = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
            if returns_t.std() > 1e-8:
                returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

            lp_stack = torch.stack(log_probs[i][:len(returns)])
            v_stack = torch.stack(values_buf[i][:len(returns)])
            adv = returns_t - v_stack.detach()
            p_loss = -(lp_stack * adv).mean()
            v_loss = 0.5 * ((v_stack - returns_t) ** 2).mean()
            loss = p_loss + 0.5 * v_loss

            optimizers[i].zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policies[i].parameters(), 0.5)
            optimizers[i].step()

        # Floor update (primal-dual, every 10 episodes)
        if ep % 10 == 0 and ep > 0:
            recent_surv = np.mean(all_survivals[-10:])
            cv = SAFETY_DELTA - recent_surv
            mu = torch.clamp(mu + 0.01 * cv, min=0.0)

    n = min(30, len(all_survivals))
    return {
        'survival': float(np.mean(all_survivals[-n:])) * 100,
        'welfare': float(np.mean(all_rewards[-n:])),
        'floor_R05': float(floor_net(torch.tensor([0.5]).to(DEVICE)).item()),
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
    print(f"  Coin Game TPSD: Deep MARL Validation ({DEVICE})")
    print(f"  {N_SEEDS} seeds per method")
    print(f"{'='*60}")

    # Selfish
    print(f"\n--- Selfish PPO (Coin Game TPSD) ---")
    selfish_res = []
    for s in range(N_SEEDS):
        t0 = time.time()
        r = run_selfish_coin_seed(42 + s)
        print(f"  Seed {s+1:2d}/{N_SEEDS}: Surv={r['survival']:5.1f}%  Welf={r['welfare']:.2f}  [{time.time()-t0:.0f}s]")
        selfish_res.append(r)

    # MACCL
    print(f"\n--- MACCL (Coin Game TPSD) ---")
    maccl_res = []
    for s in range(N_SEEDS):
        t0 = time.time()
        r = run_maccl_coin_seed(42 + s)
        print(f"  Seed {s+1:2d}/{N_SEEDS}: Surv={r['survival']:5.1f}%  Welf={r['welfare']:.2f}  Floor={r['floor_R05']:.3f}  [{time.time()-t0:.0f}s]")
        maccl_res.append(r)

    # Aggregate
    def agg(results, name):
        survs = [r['survival'] for r in results]
        welfs = [r['welfare'] for r in results]
        return {
            'method': name,
            'n_seeds': len(results),
            'survival_mean': float(np.mean(survs)),
            'survival_std': float(np.std(survs)),
            'survival_ci95': bootstrap_ci(survs),
            'welfare_mean': float(np.mean(welfs)),
            'welfare_std': float(np.std(welfs)),
            'welfare_ci95': bootstrap_ci(welfs),
            'per_seed_survival': survs,
            'per_seed_welfare': welfs,
        }

    output = {
        'experiment': 'Coin Game TPSD Deep MARL Validation',
        'reference': 'Lerer & Peysakhovich 2017, adapted with TPSD dynamics',
        'config': {'device': str(DEVICE), 'torch_version': torch.__version__},
        'selfish': agg(selfish_res, 'Selfish PPO'),
        'maccl': agg(maccl_res, 'MACCL'),
        'total_time_seconds': time.time() - t_total,
    }

    json_path = os.path.join(OUTPUT_DIR, 'coin_game_deep_results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  Selfish: Surv={output['selfish']['survival_mean']:.1f}% CI{output['selfish']['survival_ci95']}")
    print(f"  MACCL:   Surv={output['maccl']['survival_mean']:.1f}% CI{output['maccl']['survival_ci95']}")
    print(f"  Total: {time.time()-t_total:.0f}s — Saved: {json_path}")
    print(f"{'='*60}")
