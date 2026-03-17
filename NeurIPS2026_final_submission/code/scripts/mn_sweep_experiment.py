"""
M/N Phase Transition Experiment (v2 - NonlinearPGGEnv based)
=============================================================
Sweeps M/N ratio from 0.05 to 1.0 to find the Nash Trap boundary.
Uses the canonical NonlinearPGGEnv for environment consistency.

Output: JSON + phase diagram figure.
Dependencies: NumPy, matplotlib, envs/nonlinear_pgg_env.py
"""

import numpy as np
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from envs.nonlinear_pgg_env import NonlinearPGGEnv

# ============================================================
# Configuration
# ============================================================
N_AGENTS = 20
N_EPISODES = 300
N_SEEDS = 20
N_EVAL = 30  # last N episodes for evaluation
STATE_DIM = 4
GAMMA = 0.99
BYZ_FRAC = 0.30

# M/N sweep range
MN_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90, 1.00]

if os.environ.get("ETHICAAI_FAST") == "1":
    print("  [FAST MODE] seeds=3, episodes=50, 5 M/N points")
    N_SEEDS = 3
    N_EPISODES = 50
    MN_RATIOS = [0.05, 0.25, 0.50, 0.75, 1.00]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'mn_sweep')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Linear Agent (REINFORCE) — same as reinforce_nash_trap.py
# ============================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


class LinearAgent:
    def __init__(self, rng):
        self.w = rng.randn(STATE_DIM) * 0.01
        self.b = 0.0
        self.lr = 0.01

    def act(self, obs, rng, noise_scale):
        logit = obs @ self.w + self.b
        base = sigmoid(logit)
        return float(np.clip(base + rng.randn() * noise_scale, 0.01, 0.99))

    def update(self, obs_list, act_list, returns):
        for t in range(len(returns)):
            obs = obs_list[t]
            act = act_list[t]
            logit = obs @ self.w + self.b
            pred = sigmoid(logit)
            grad = act - pred
            self.w += self.lr * returns[t] * grad * obs
            self.b += self.lr * returns[t] * grad


# ============================================================
# Training loop with NonlinearPGGEnv
# ============================================================
def run_condition(mn_ratio, n_seeds, n_episodes):
    """Run full experiment for a given M/N ratio using NonlinearPGGEnv."""
    M = mn_ratio * N_AGENTS
    n_honest = int(N_AGENTS * (1 - BYZ_FRAC))
    all_seed_results = []

    for seed in range(n_seeds):
        rng = np.random.RandomState(42 + seed)
        env = NonlinearPGGEnv(
            n_agents=N_AGENTS, multiplier=M,
            byz_frac=BYZ_FRAC, t_horizon=50,
        )
        agents = [LinearAgent(np.random.RandomState(seed * 100 + i))
                  for i in range(n_honest)]

        ep_lams, ep_survs, ep_welfares = [], [], []

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed * 10000 + ep)
            noise = 0.15 - 0.13 * min(ep / n_episodes, 1.0)

            agent_obs = [[] for _ in range(n_honest)]
            agent_acts = [[] for _ in range(n_honest)]
            agent_rewards = [[] for _ in range(n_honest)]

            lam_sum, steps = 0.0, 0
            survived = True

            for t in range(50):
                lambdas = np.zeros(n_honest)
                for i in range(n_honest):
                    lam_i = agents[i].act(obs, rng, noise)
                    lambdas[i] = lam_i
                    agent_obs[i].append(obs.copy())
                    agent_acts[i].append(lam_i)

                obs_next, rewards, terminated, truncated, info = env.step(lambdas)

                for i in range(n_honest):
                    agent_rewards[i].append(float(rewards[i]))

                lam_sum += float(lambdas.mean())
                steps += 1

                if terminated:
                    survived = info.get("survived", False)
                    break
                obs = obs_next

            # REINFORCE update
            for i in range(n_honest):
                if len(agent_rewards[i]) < 2:
                    continue
                rew = agent_rewards[i]
                returns = np.zeros(len(rew))
                G = 0
                for t_idx in reversed(range(len(rew))):
                    G = rew[t_idx] + GAMMA * G
                    returns[t_idx] = G
                if returns.std() > 1e-8:
                    returns = (returns - returns.mean()) / returns.std()
                agents[i].update(agent_obs[i], agent_acts[i], returns)

            ep_lams.append(lam_sum / max(steps, 1))
            ep_survs.append(float(survived))
            ep_welfares.append(info.get("welfare", 0.0))

        # Summarize last N_EVAL episodes
        all_seed_results.append({
            "mean_lambda": float(np.mean(ep_lams[-N_EVAL:])),
            "survival_pct": float(np.mean(ep_survs[-N_EVAL:]) * 100),
            "welfare": float(np.mean(ep_welfares[-N_EVAL:])),
        })

    lams = [r["mean_lambda"] for r in all_seed_results]
    survs = [r["survival_pct"] for r in all_seed_results]
    welfs = [r["welfare"] for r in all_seed_results]

    return {
        "mn_ratio": mn_ratio,
        "M": M,
        "N": N_AGENTS,
        "lambda_mean": float(np.mean(lams)),
        "lambda_std": float(np.std(lams)),
        "lambda_ci95": [
            float(np.mean(lams) - 1.96 * np.std(lams) / np.sqrt(len(lams))),
            float(np.mean(lams) + 1.96 * np.std(lams) / np.sqrt(len(lams))),
        ],
        "survival_mean": float(np.mean(survs)),
        "survival_std": float(np.std(survs)),
        "welfare_mean": float(np.mean(welfs)),
        "welfare_std": float(np.std(welfs)),
        "trapped": float(np.mean(lams)) < 0.85,
        "n_seeds": n_seeds,
        "seed_results": all_seed_results,
    }


def run_oracle(mn_ratio, n_seeds):
    """Unconditional commitment (phi1=1.0) baseline."""
    M = mn_ratio * N_AGENTS
    n_honest = int(N_AGENTS * (1 - BYZ_FRAC))
    survs, welfs = [], []

    for seed in range(n_seeds):
        env = NonlinearPGGEnv(
            n_agents=N_AGENTS, multiplier=M,
            byz_frac=BYZ_FRAC, t_horizon=50,
        )
        ep_s, ep_w = [], []
        for ep in range(50):
            obs, _ = env.reset(seed=seed * 10000 + ep)
            survived = True
            for t in range(50):
                lambdas = np.ones(n_honest)  # unconditional
                obs, rewards, terminated, _, info = env.step(lambdas)
                if terminated:
                    survived = info.get("survived", False)
                    break
            ep_s.append(float(survived))
            ep_w.append(info.get("welfare", 0.0))
        survs.append(np.mean(ep_s) * 100)
        welfs.append(np.mean(ep_w))

    return {
        "survival_mean": float(np.mean(survs)),
        "welfare_mean": float(np.mean(welfs)),
    }


# ============================================================
# Plotting
# ============================================================
def plot_results(results, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    mn = [r["mn_ratio"] for r in results]
    lam = [r["lambda_mean"] for r in results]
    lam_s = [r["lambda_std"] for r in results]
    surv = [r["survival_mean"] for r in results]
    surv_s = [r["survival_std"] for r in results]
    osurv = [r.get("oracle_survival", 100.0) for r in results]

    ax = axes[0]
    ax.errorbar(mn, lam, yerr=lam_s, fmt='o-', color='#e74c3c',
                linewidth=2, markersize=8, capsize=4, label='Learned (REINFORCE)')
    ax.axhline(y=1.0, color='#2ecc71', linestyle='--', alpha=0.8, label='Oracle phi1=1.0')
    ax.axhspan(0.3, 0.7, alpha=0.05, color='red')
    ax.set_xlabel('M/N ratio', fontsize=12)
    ax.set_ylabel('Converged lambda', fontsize=12)
    ax.set_title('(a) Commitment Level vs M/N', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.05, 1.1)

    ax = axes[1]
    ax.errorbar(mn, surv, yerr=surv_s, fmt='s-', color='#e74c3c',
                linewidth=2, markersize=8, capsize=4, label='Selfish RL')
    ax.plot(mn, osurv, 'D--', color='#2ecc71',
            linewidth=2, markersize=7, label='Unconditional phi1=1.0')
    ax.set_xlabel('M/N ratio', fontsize=12)
    ax.set_ylabel('Survival Rate (%)', fontsize=12)
    ax.set_title('(b) Survival vs M/N', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-5, 105)

    ax = axes[2]
    gaps = [1.0 - l for l in lam]
    colors = ['#e74c3c' if g > 0.15 else '#f39c12' if g > 0.05 else '#2ecc71' for g in gaps]
    ax.bar(range(len(mn)), gaps, color=colors, alpha=0.85,
           edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(mn)))
    ax.set_xticklabels([f'{v:.2f}' for v in mn], fontsize=8, rotation=45)
    ax.set_xlabel('M/N ratio', fontsize=12)
    ax.set_ylabel('Commitment Gap (1 - lambda)', fontsize=12)
    ax.set_title('(c) Gap to Optimal', fontsize=13, fontweight='bold')

    plt.suptitle('M/N Phase Transition: Nash Trap Boundary\n'
                 f'N={N_AGENTS}, Byz={BYZ_FRAC*100:.0f}%, '
                 f'{N_SEEDS} seeds x {N_EPISODES} episodes',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {out_path}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  M/N PHASE TRANSITION (v2 - NonlinearPGGEnv)")
    print(f"  N={N_AGENTS}, Byz={BYZ_FRAC*100:.0f}%")
    print(f"  M/N sweep: {MN_RATIOS}")
    print(f"  Seeds={N_SEEDS}, Episodes={N_EPISODES}")
    print("=" * 70)

    t0 = time.time()
    all_results = []

    for idx, mn in enumerate(MN_RATIOS):
        M = mn * N_AGENTS
        print(f"\n[{idx+1}/{len(MN_RATIOS)}] M/N = {mn:.2f} (M = {M:.1f})")
        print("-" * 50)

        result = run_condition(mn, N_SEEDS, N_EPISODES)
        oracle = run_oracle(mn, N_SEEDS)
        result["oracle_survival"] = oracle["survival_mean"]
        result["oracle_welfare"] = oracle["welfare_mean"]

        status = "[TRAPPED]" if result["trapped"] else "[ESCAPED]"
        print(f"  RL:     lam={result['lambda_mean']:.3f}+/-{result['lambda_std']:.3f}, "
              f"Surv={result['survival_mean']:.1f}% {status}")
        print(f"  Oracle: Surv={oracle['survival_mean']:.1f}%")

        all_results.append(result)

    elapsed = time.time() - t0

    output = {
        "experiment": "M/N Phase Transition (v2 - NonlinearPGGEnv)",
        "config": {
            "N_AGENTS": N_AGENTS, "BYZ_FRAC": BYZ_FRAC,
            "N_EPISODES": N_EPISODES, "N_SEEDS": N_SEEDS,
            "MN_RATIOS": MN_RATIOS,
        },
        "results": all_results,
        "summary": {
            "trapped_count": sum(1 for r in all_results if r["trapped"]),
            "total_conditions": len(all_results),
            "boundary_mn": None,
        },
        "time_seconds": elapsed,
    }

    for i in range(len(all_results) - 1):
        if all_results[i]["trapped"] and not all_results[i+1]["trapped"]:
            output["summary"]["boundary_mn"] = {
                "lower": all_results[i]["mn_ratio"],
                "upper": all_results[i+1]["mn_ratio"],
            }
            break

    json_path = os.path.join(OUTPUT_DIR, "mn_sweep_results.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[Save] {json_path}")

    plot_results(all_results, os.path.join(OUTPUT_DIR, "mn_phase_transition.png"))

    print(f"\n{'=' * 70}")
    print(f"  RESULTS ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
    print(f"  {'M/N':>6s}  {'M':>6s}  {'lam':>10s}  {'Surv%':>8s}  {'Oracle':>8s}  Status")
    print(f"  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in all_results:
        s = "TRAP" if r["trapped"] else "FREE"
        print(f"  {r['mn_ratio']:6.2f}  {r['M']:6.1f}  "
              f"{r['lambda_mean']:.3f}+/-{r['lambda_std']:.2f}  "
              f"{r['survival_mean']:6.1f}%  {r['oracle_survival']:6.1f}%  {s}")

    if output["summary"]["boundary_mn"]:
        b = output["summary"]["boundary_mn"]
        print(f"\n  >> Boundary: M/N in ({b['lower']:.2f}, {b['upper']:.2f})")
    else:
        if all(r["trapped"] for r in all_results):
            print(f"\n  >> Nash Trap persists across ALL M/N ratios")
    print(f"{'=' * 70}")
