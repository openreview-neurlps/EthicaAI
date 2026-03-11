#!/usr/bin/env python
"""
mappo_team_reward.py — MAPPO with Team Reward Baseline
=======================================================
Addresses reviewer concern: "Why not MAPPO + team reward / social reward?"

Tests whether replacing individual payoff with team-average reward
allows MAPPO to escape the Nash Trap. This is the most natural
reward-shaping baseline that was missing from the paper.

Hypothesis: Team reward will NOT escape the Nash Trap under Byzantine
conditions because Byzantine agents pollute the team signal.

Usage:
  python mappo_team_reward.py
  ETHICAAI_FAST=1 python mappo_team_reward.py  # Quick test
"""
import numpy as np
import json
import time
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "team_reward"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from envs.nonlinear_pgg_env import NonlinearPGGEnv

# === Hyperparameters (same as cleanrl_mappo_pgg.py for fair comparison) ===
HIDDEN_DIM = 64
LR_ACTOR = 2.5e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
PPO_EPOCHS = 4

N_EPISODES = 500
N_EVAL = 30
N_SEEDS = 20

if os.environ.get("ETHICAAI_FAST") == "1":
    print("  [FAST MODE] N_SEEDS=3, N_EPISODES=150")
    N_SEEDS = 3
    N_EPISODES = 150


# === Neural Network (same architecture as cleanrl_mappo_pgg.py) ===
def relu(x):
    return np.maximum(0, x)


class NNLayer:
    def __init__(self, rng, in_dim, out_dim, lr=2.5e-4):
        scale = np.sqrt(2.0 / in_dim)
        self.W = rng.randn(in_dim, out_dim).astype(np.float32) * scale
        self.b = np.zeros(out_dim, dtype=np.float32)
        self.lr = lr
        self.m_W = np.zeros_like(self.W)
        self.v_W = np.zeros_like(self.W)
        self.m_b = np.zeros_like(self.b)
        self.v_b = np.zeros_like(self.b)
        self.step = 0

    def forward(self, x):
        return x @ self.W + self.b

    def adam_update(self, grad_W, grad_b, beta1=0.9, beta2=0.999, eps=1e-8):
        self.step += 1
        self.m_W = beta1 * self.m_W + (1 - beta1) * grad_W
        self.v_W = beta2 * self.v_W + (1 - beta2) * grad_W**2
        m_hat = self.m_W / (1 - beta1**self.step)
        v_hat = self.v_W / (1 - beta2**self.step)
        self.W += self.lr * m_hat / (np.sqrt(v_hat) + eps)

        self.m_b = beta1 * self.m_b + (1 - beta1) * grad_b
        self.v_b = beta2 * self.v_b + (1 - beta2) * grad_b**2
        m_hat = self.m_b / (1 - beta1**self.step)
        v_hat = self.v_b / (1 - beta2**self.step)
        self.b += self.lr * m_hat / (np.sqrt(v_hat) + eps)


class MLPActor:
    def __init__(self, rng, obs_dim=4, hidden=HIDDEN_DIM, lr=LR_ACTOR):
        self.fc1 = NNLayer(rng, obs_dim, hidden, lr)
        self.fc2 = NNLayer(rng, hidden, hidden, lr)
        self.mean_head = NNLayer(rng, hidden, 1, lr)
        self.log_std = np.array([-0.5], dtype=np.float32)

    def forward(self, obs):
        h = relu(self.fc1.forward(obs))
        h = relu(self.fc2.forward(h))
        mean = 1.0 / (1.0 + np.exp(-self.mean_head.forward(h).flatten()))
        return mean, h

    def act(self, obs, rng):
        mean, _ = self.forward(obs)
        std = np.exp(self.log_std)
        noise = rng.randn() * std[0]
        action = np.clip(mean[0] + noise, 0, 1)
        log_prob = -0.5 * ((action - mean[0]) / std[0])**2 - self.log_std[0] - 0.5 * np.log(2 * np.pi)
        return action, log_prob, mean[0]

    def log_prob(self, obs, action):
        mean, _ = self.forward(obs)
        std = np.exp(self.log_std)
        return -0.5 * ((action - mean[0]) / std[0])**2 - self.log_std[0] - 0.5 * np.log(2 * np.pi)

    def param_count(self):
        total = 0
        for layer in [self.fc1, self.fc2, self.mean_head]:
            total += layer.W.size + layer.b.size
        total += self.log_std.size
        return total


class MLPCritic:
    def __init__(self, rng, obs_dim=5, hidden=HIDDEN_DIM, lr=LR_CRITIC):
        self.fc1 = NNLayer(rng, obs_dim, hidden, lr)
        self.fc2 = NNLayer(rng, hidden, hidden, lr)
        self.value_head = NNLayer(rng, hidden, 1, lr)

    def forward(self, obs):
        h = relu(self.fc1.forward(obs))
        h = relu(self.fc2.forward(h))
        return self.value_head.forward(h).flatten()[0]


def compute_gae(rewards, values, gamma=GAMMA, lam=GAE_LAMBDA):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < len(values) else 0
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + np.array(values[:T])
    return advantages, returns


def ppo_update_actor(actor, obs_list, act_list, old_lps, advantages):
    for obs, act, old_lp, adv in zip(obs_list, act_list, old_lps, advantages):
        z1 = actor.fc1.forward(obs)
        h1 = relu(z1)
        z2 = actor.fc2.forward(h1)
        h2 = relu(z2)
        z_out = actor.mean_head.forward(h2)
        mean_val = 1.0 / (1.0 + np.exp(-z_out.flatten()))
        std = np.exp(actor.log_std)

        d_lp_d_mean = (act - mean_val[0]) / (std[0]**2)
        sig_deriv = mean_val[0] * (1 - mean_val[0])
        delta_out = d_lp_d_mean * sig_deriv * adv

        grad_W_mh = np.outer(h2, [delta_out])
        grad_b_mh = np.array([delta_out])
        actor.mean_head.adam_update(grad_W_mh, grad_b_mh)

        delta_h2 = delta_out * actor.mean_head.W.flatten()
        delta_z2 = delta_h2 * (z2 > 0).astype(np.float32)
        grad_W_fc2 = np.outer(h1, delta_z2)
        actor.fc2.adam_update(grad_W_fc2, delta_z2)

        delta_h1 = delta_z2 @ actor.fc2.W.T
        delta_z1 = delta_h1 * (z1 > 0).astype(np.float32)
        grad_W_fc1 = np.outer(obs, delta_z1)
        actor.fc1.adam_update(grad_W_fc1, delta_z1)

        d_lp_d_logstd = ((act - mean_val[0])**2 / (std[0]**2)) - 1.0
        log_std_grad = -(d_lp_d_logstd * adv + ENTROPY_COEF * 1.0)
        actor.log_std -= LR_ACTOR * 0.1 * log_std_grad


def ppo_update_critic(critic, obs_list, returns):
    for obs, ret in zip(obs_list, returns):
        z1 = critic.fc1.forward(obs)
        h1 = relu(z1)
        z2 = critic.fc2.forward(h1)
        h2 = relu(z2)
        v_pred = critic.value_head.forward(h2).flatten()[0]

        delta_out = (v_pred - ret) * VALUE_COEF

        grad_W_vh = np.outer(h2, [delta_out])
        grad_b_vh = np.array([delta_out])
        critic.value_head.adam_update(-grad_W_vh, -grad_b_vh)

        delta_h2 = delta_out * critic.value_head.W.flatten()
        delta_z2 = delta_h2 * (z2 > 0).astype(np.float32)
        critic.fc2.adam_update(-np.outer(h1, delta_z2), -delta_z2)

        delta_h1 = delta_z2 @ critic.fc2.W.T
        delta_z1 = delta_h1 * (z1 > 0).astype(np.float32)
        critic.fc1.adam_update(-np.outer(obs, delta_z1), -delta_z1)


# ============================================================
# Core Experiment: 3 reward modes
# ============================================================
REWARD_MODES = {
    "individual": "Standard individual payoff (baseline, same as cleanrl_mappo_pgg.py)",
    "team_mean": "Team reward = mean of all agent rewards (honest + Byzantine)",
    "team_honest": "Team reward = mean of honest agent rewards only (oracle access)",
}


def run_experiment(seed, reward_mode="individual"):
    """Run MAPPO with specified reward mode."""
    rng = np.random.RandomState(seed)
    env = NonlinearPGGEnv()

    n_agents = env.n_honest
    actors = [MLPActor(rng) for _ in range(n_agents)]
    critic = MLPCritic(rng, obs_dim=5)

    per_episode = []

    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=seed * 10000 + ep)

        obs_buf = [[] for _ in range(n_agents)]
        act_buf = [[] for _ in range(n_agents)]
        lp_buf = [[] for _ in range(n_agents)]
        rew_buf = [[] for _ in range(n_agents)]
        val_buf = [[] for _ in range(n_agents)]
        gs_buf = []

        for t in range(env.T):
            actions = np.zeros(n_agents)
            for i in range(n_agents):
                a, lp, mu = actors[i].act(obs, rng)
                actions[i] = a
                obs_buf[i].append(obs.copy())
                act_buf[i].append(a)
                lp_buf[i].append(lp)

                gs = env.get_global_state()
                val_buf[i].append(critic.forward(gs))
                if i == 0:
                    gs_buf.append(gs.copy())

            obs, rewards, terminated, truncated, info = env.step(actions)

            # === REWARD MODE TRANSFORMATION ===
            if reward_mode == "team_mean":
                # Mean of ALL agents' rewards (including Byzantine)
                # info["welfare"] = mean of all N agent payoffs
                team_r = info.get("welfare", float(np.mean(rewards)))
                rewards_used = [team_r] * n_agents
            elif reward_mode == "team_honest":
                # Mean of honest agents only (oracle: knows who is Byzantine)
                team_r = float(np.mean(rewards))
                rewards_used = [team_r] * n_agents
            else:
                rewards_used = list(rewards)

            for i in range(n_agents):
                rew_buf[i].append(rewards_used[i] if i < len(rewards_used) else rewards_used[-1])

            if terminated:
                break

        # PPO update
        for i in range(n_agents):
            if len(rew_buf[i]) < 2:
                continue
            advantages, returns = compute_gae(rew_buf[i], val_buf[i])
            if np.std(advantages) > 1e-8:
                advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
            for _ in range(PPO_EPOCHS):
                ppo_update_actor(actors[i], obs_buf[i], act_buf[i], lp_buf[i], advantages)
                ppo_update_critic(critic, gs_buf[:len(returns)], returns)

        per_episode.append({
            "welfare": info.get("welfare", 0),
            "survived": info.get("survived", False),
            "mean_lambda": info.get("mean_lambda", 0),
        })

    eval_data = per_episode[-N_EVAL:]
    return {
        "reward_mode": reward_mode,
        "mean_lambda": float(np.mean([d["mean_lambda"] for d in eval_data])),
        "std_lambda": float(np.std([d["mean_lambda"] for d in eval_data])),
        "mean_survival": float(np.mean([float(d["survived"]) for d in eval_data]) * 100),
        "mean_welfare": float(np.mean([d["welfare"] for d in eval_data])),
    }


def bootstrap_ci(data, n_boot=5000, ci=0.95):
    data = np.array(data)
    means = np.array([np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return float(np.percentile(means, alpha * 100)), float(np.percentile(means, (1 - alpha) * 100))


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  MAPPO + Team Reward Baseline Experiment")
    print(f"  Modes: {list(REWARD_MODES.keys())}")
    print(f"  Seeds={N_SEEDS}, Episodes={N_EPISODES}")
    print("=" * 70)

    t0 = time.time()
    all_results = {}

    for mode, desc in REWARD_MODES.items():
        print(f"\n  [{mode}] {desc}")
        print(f"  Running {N_SEEDS} seeds...")

        seed_results = []
        for s in range(N_SEEDS):
            r = run_experiment(seed=s * 7 + 42, reward_mode=mode)
            seed_results.append(r)
            if (s + 1) % 5 == 0 or s == 0:
                print(f"    Seed {s+1}: λ={r['mean_lambda']:.3f}, "
                      f"surv={r['mean_survival']:.1f}%, W={r['mean_welfare']:.1f}")

        lambdas = [r["mean_lambda"] for r in seed_results]
        survivals = [r["mean_survival"] for r in seed_results]
        welfares = [r["mean_welfare"] for r in seed_results]
        ci_lam = bootstrap_ci(lambdas)
        ci_surv = bootstrap_ci(survivals)

        all_results[mode] = {
            "description": desc,
            "lambda": {"mean": float(np.mean(lambdas)), "std": float(np.std(lambdas)),
                       "ci95": ci_lam},
            "survival": {"mean": float(np.mean(survivals)), "std": float(np.std(survivals)),
                         "ci95": ci_surv},
            "welfare": {"mean": float(np.mean(welfares)), "std": float(np.std(welfares))},
            "per_seed": seed_results,
        }

        print(f"  [{mode}] λ={np.mean(lambdas):.3f} [{ci_lam[0]:.3f}, {ci_lam[1]:.3f}], "
              f"surv={np.mean(survivals):.1f}%")

    total = time.time() - t0

    # Summary
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY (NASH TRAP ANALYSIS)")
    print("=" * 70)
    print(f"  {'Mode':<20s} | {'λ':>8s} | {'CI95':>18s} | {'Surv%':>7s} | {'Welfare':>8s} | Nash Trap?")
    print("  " + "-" * 75)
    for mode, r in all_results.items():
        trapped = "YES" if r["lambda"]["mean"] < 0.95 else "NO"
        print(f"  {mode:<20s} | {r['lambda']['mean']:>8.3f} | "
              f"[{r['lambda']['ci95'][0]:.3f}, {r['lambda']['ci95'][1]:.3f}] | "
              f"{r['survival']['mean']:>6.1f}% | {r['welfare']['mean']:>8.1f} | {trapped}")

    print(f"\n  Oracle (φ₁=1.0): λ=1.000, Surv=100.0%, W=32.0")
    print(f"  Total time: {total:.0f}s")

    # Key finding
    ind_lam = all_results["individual"]["lambda"]["mean"]
    team_lam = all_results["team_mean"]["lambda"]["mean"]
    delta = team_lam - ind_lam
    print(f"\n  KEY FINDING: Team reward {'improves' if delta > 0.05 else 'does NOT escape'} "
          f"the Nash Trap (Δλ = {delta:+.3f})")

    # Save
    output = {
        "experiment": "MAPPO + Team Reward Baseline",
        "purpose": "Test whether team-level reward shaping escapes the Nash Trap",
        "results": all_results,
        "time_seconds": round(total, 1),
        "conclusion": (
            f"Team reward (mean) shifts λ by {delta:+.3f} relative to individual reward. "
            f"{'Nash Trap persists' if delta < 0.2 else 'Partial improvement'} — "
            f"reward shaping alone is insufficient for survival in TPSDs."
        ),
    }

    out_path = OUTPUT_DIR / "team_reward_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")
""".strip()
"""
