"""
PyTorch REINFORCE Nash Trap Verification
=========================================
Validates that the Nash Trap persists under a proper PyTorch-based
REINFORCE implementation with torch.distributions, autograd, and
Adam optimizer — addressing potential concerns about NumPy-only results.

Uses the SAME NonlinearPGGEnv environment as all other experiments.
Compares three architectures against the NumPy baseline.

Dependencies: PyTorch, NumPy, envs/nonlinear_pgg_env.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from envs.nonlinear_pgg_env import NonlinearPGGEnv

# ============================================================
# Configuration
# ============================================================
N_AGENTS = 20
BYZ_FRAC = 0.30
M_OVER_N = 0.08
N_SEEDS = 20
N_EPISODES = 300
T_HORIZON = 50
GAMMA = 0.99
LR = 3e-4

if os.environ.get("ETHICAAI_FAST") == "1":
    print("  [FAST MODE]")
    N_SEEDS = 3
    N_EPISODES = 50

# Force CPU — small per-agent models have CUDA transfer overhead
DEVICE = torch.device("cpu")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'pytorch_reinforce')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Policy Networks (PyTorch)
# ============================================================
class LinearPolicy(nn.Module):
    """Linear policy: obs -> Beta distribution parameters."""
    def __init__(self, obs_dim=4):
        super().__init__()
        self.alpha_head = nn.Linear(obs_dim, 1)
        self.beta_head = nn.Linear(obs_dim, 1)

    def forward(self, obs):
        alpha = torch.clamp(F.softplus(self.alpha_head(obs)), min=0.5, max=10.0)
        beta = torch.clamp(F.softplus(self.beta_head(obs)), min=0.5, max=10.0)
        return alpha.squeeze(-1), beta.squeeze(-1)

    @property
    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class MLPPolicy(nn.Module):
    """2-layer MLP policy: obs -> hidden -> Beta distribution."""
    def __init__(self, obs_dim=4, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.alpha_head = nn.Linear(hidden, 1)
        self.beta_head = nn.Linear(hidden, 1)

    def forward(self, obs):
        h = self.net(obs)
        alpha = torch.clamp(F.softplus(self.alpha_head(h)), min=0.5, max=10.0)
        beta = torch.clamp(F.softplus(self.beta_head(h)), min=0.5, max=10.0)
        return alpha.squeeze(-1), beta.squeeze(-1)

    @property
    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class MLPCriticPolicy(nn.Module):
    """MLP policy + value baseline (actor-critic style REINFORCE)."""
    def __init__(self, obs_dim=4, hidden=32):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.alpha_head = nn.Linear(hidden, 1)
        self.beta_head = nn.Linear(hidden, 1)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        h = self.actor(obs)
        alpha = torch.clamp(F.softplus(self.alpha_head(h)), min=0.5, max=10.0)
        beta = torch.clamp(F.softplus(self.beta_head(h)), min=0.5, max=10.0)
        return alpha.squeeze(-1), beta.squeeze(-1)

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    @property
    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# REINFORCE Training Loop
# ============================================================
def train_reinforce(policy_class, seed, n_episodes=N_EPISODES, lr=LR):
    """Train independent REINFORCE agents with PyTorch autograd."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    M = M_OVER_N * N_AGENTS
    env = NonlinearPGGEnv(n_agents=N_AGENTS, multiplier=M, byz_frac=BYZ_FRAC)
    n_honest = env.n_honest

    # Create independent agents
    agents = [policy_class().to(DEVICE) for _ in range(n_honest)]
    optimizers = [optim.Adam(a.parameters(), lr=lr) for a in agents]
    has_critic = hasattr(agents[0], 'value')

    ep_lambdas = []
    ep_survivals = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        obs_t = torch.FloatTensor(obs).to(DEVICE)

        # Collect trajectory
        log_probs = [[] for _ in range(n_honest)]
        rewards_all = [[] for _ in range(n_honest)]
        values_all = [[] for _ in range(n_honest)]
        lam_sum, steps = 0.0, 0
        survived = True

        for t in range(T_HORIZON):
            lambdas = np.zeros(n_honest)
            for i in range(n_honest):
                alpha, beta_param = agents[i](obs_t)
                dist = Beta(alpha, beta_param)
                action = dist.sample()
                lp = dist.log_prob(action)
                log_probs[i].append(lp)

                if has_critic:
                    v = agents[i].value(obs_t)
                    values_all[i].append(v)

                # Clamp to valid range
                lam_i = float(torch.clamp(action, 0.01, 0.99).item())
                lambdas[i] = lam_i

            obs_next, rewards, terminated, _, info = env.step(lambdas)
            for i in range(n_honest):
                rewards_all[i].append(float(rewards[i]))

            lam_sum += float(lambdas.mean())
            steps += 1
            if terminated:
                survived = info.get("survived", False)
                break
            obs = obs_next
            obs_t = torch.FloatTensor(obs).to(DEVICE)

        # REINFORCE update for each agent
        for i in range(n_honest):
            if len(rewards_all[i]) < 2:
                continue

            # Compute returns
            G, returns = 0.0, []
            for r in reversed(rewards_all[i]):
                G = r + GAMMA * G
                returns.insert(0, G)
            returns_t = torch.FloatTensor(returns).to(DEVICE)

            # Normalize returns
            if returns_t.std() > 1e-8:
                returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

            # Compute policy loss
            log_prob_stack = torch.stack(log_probs[i][:len(returns)])

            if has_critic and len(values_all[i]) >= len(returns):
                values_t = torch.stack(values_all[i][:len(returns)])
                advantage = returns_t - values_t.detach()
                policy_loss = -(log_prob_stack * advantage).mean()
                value_loss = nn.functional.mse_loss(values_t, returns_t)
                loss = policy_loss + 0.5 * value_loss
            else:
                loss = -(log_prob_stack * returns_t).mean()

            optimizers[i].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agents[i].parameters(), 0.5)
            optimizers[i].step()

        mean_lam = lam_sum / max(steps, 1)
        ep_lambdas.append(mean_lam)
        ep_survivals.append(float(survived))

    # Final metrics (last 30 episodes)
    final_lam = float(np.mean(ep_lambdas[-30:]))
    final_surv = float(np.mean(ep_survivals[-30:]) * 100)
    return final_lam, final_surv, agents[0].param_count


# ============================================================
# Main
# ============================================================
def run_all():
    print("=" * 70)
    print("  PyTorch REINFORCE: Nash Trap Verification")
    print(f"  Device: {DEVICE}")
    print(f"  Seeds={N_SEEDS}, Episodes={N_EPISODES}")
    print("=" * 70)

    t0 = time.time()
    results = {}

    configs = [
        ("PyTorch Linear", LinearPolicy),
        ("PyTorch MLP (32h)", MLPPolicy),
        ("PyTorch MLP+Critic (32h)", MLPCriticPolicy),
    ]

    for name, policy_cls in configs:
        print(f"\n[{name}]")
        print("-" * 50)

        lams, survs = [], []
        params = None
        for s in range(N_SEEDS):
            lam, surv, pc = train_reinforce(policy_cls, seed=42 + s)
            lams.append(lam)
            survs.append(surv)
            params = pc
            if (s + 1) % 5 == 0:
                print(f"  Seed {s+1}/{N_SEEDS}: lam={lam:.3f}, surv={surv:.0f}%")

        mean_lam = float(np.mean(lams))
        std_lam = float(np.std(lams))
        mean_surv = float(np.mean(survs))
        std_surv = float(np.std(survs))
        trapped = mean_lam < 0.85

        results[name] = {
            "params_per_agent": params,
            "lambda_mean": mean_lam,
            "lambda_std": std_lam,
            "survival_mean": mean_surv,
            "survival_std": std_surv,
            "trapped": trapped,
            "per_seed_lambda": lams,
            "per_seed_survival": survs,
        }

        status = "IN TRAP" if trapped else "ESCAPED"
        print(f"  => lam={mean_lam:.3f}+/-{std_lam:.3f}, "
              f"surv={mean_surv:.1f}+/-{std_surv:.1f}% [{status}]")

    elapsed = time.time() - t0

    # Save
    output = {
        "experiment": "PyTorch REINFORCE Nash Trap Verification",
        "config": {
            "N_AGENTS": N_AGENTS,
            "BYZ_FRAC": BYZ_FRAC,
            "M_OVER_N": M_OVER_N,
            "N_SEEDS": N_SEEDS,
            "N_EPISODES": N_EPISODES,
            "T_HORIZON": T_HORIZON,
            "GAMMA": GAMMA,
            "LR": LR,
            "DEVICE": str(DEVICE),
            "framework": "PyTorch",
            "torch_version": torch.__version__,
            "distribution": "Beta(alpha, beta)",
            "optimizer": "Adam",
            "grad_clip": 0.5,
        },
        "results": results,
        "time_seconds": elapsed,
    }

    json_path = os.path.join(OUTPUT_DIR, "pytorch_reinforce_results.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  COMPLETE in {elapsed:.0f}s")
    for name, r in results.items():
        status = "IN TRAP" if r["trapped"] else "ESCAPED"
        print(f"  {name:30s}: lam={r['lambda_mean']:.3f}+/-{r['lambda_std']:.3f} [{status}]")
    print(f"\n  Saved: {json_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_all()
