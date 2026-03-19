"""
2D Phase Diagram: M/N x N Heatmap
===================================
Generates a 2D heatmap showing the Nash Trap boundary as a function
of both M/N ratio and total agent count N.

This combines the M/N sweep (varying M/N at fixed N=20) with
N-scaling (varying N at fixed M/N) to produce a comprehensive
phase diagram for the paper.

Output: phase_diagram_2d.png + phase_diagram_2d.json
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
from matplotlib.colors import LinearSegmentedColormap

from envs.nonlinear_pgg_env import NonlinearPGGEnv

# ============================================================
# Configuration
# ============================================================
MN_RATIOS = [0.05, 0.15, 0.30, 0.50, 0.75, 1.00]
N_VALUES = [5, 10, 20, 50]
N_SEEDS = 20
N_EPISODES = 100
GAMMA = 0.99
BYZ_FRAC = 0.30

if os.environ.get("ETHICAAI_FAST") == "1":
    print("  [FAST MODE]")
    MN_RATIOS = [0.10, 0.50, 1.00]
    N_VALUES = [5, 10, 20]
    N_SEEDS = 3
    N_EPISODES = 50

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'phase_diagram')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def run_condition(N, mn_ratio, n_seeds, n_episodes):
    """Run REINFORCE for a given (N, M/N) pair."""
    M = mn_ratio * N
    n_honest = N - int(N * BYZ_FRAC)
    STATE_DIM = 4
    results = []

    for seed in range(n_seeds):
        rng = np.random.RandomState(42 + seed)
        env = NonlinearPGGEnv(n_agents=N, multiplier=M, byz_frac=BYZ_FRAC)

        agents_w = [rng.randn(STATE_DIM) * 0.01 for _ in range(n_honest)]
        agents_b = [0.0] * n_honest
        lr = 0.01

        ep_lams, ep_survs = [], []

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed * 10000 + ep)
            noise = 0.15 - 0.13 * min(ep / n_episodes, 1.0)

            a_obs = [[] for _ in range(n_honest)]
            a_acts = [[] for _ in range(n_honest)]
            a_rew = [[] for _ in range(n_honest)]
            lam_sum, steps = 0.0, 0
            survived = True

            for t in range(50):
                lambdas = np.zeros(n_honest)
                for i in range(n_honest):
                    logit = obs @ agents_w[i] + agents_b[i]
                    base = sigmoid(logit)
                    lam_i = float(np.clip(base + rng.randn() * noise, 0.01, 0.99))
                    lambdas[i] = lam_i
                    a_obs[i].append(obs.copy())
                    a_acts[i].append(lam_i)

                obs_next, rewards, terminated, _, info = env.step(lambdas)
                for i in range(n_honest):
                    a_rew[i].append(float(rewards[i]))
                lam_sum += float(lambdas.mean())
                steps += 1
                if terminated:
                    survived = info.get("survived", False)
                    break
                obs = obs_next

            # REINFORCE
            for i in range(n_honest):
                if len(a_rew[i]) < 2:
                    continue
                G, returns = 0, np.zeros(len(a_rew[i]))
                for t_idx in reversed(range(len(a_rew[i]))):
                    G = a_rew[i][t_idx] + GAMMA * G
                    returns[t_idx] = G
                if returns.std() > 1e-8:
                    returns = (returns - returns.mean()) / returns.std()
                for t_idx in range(len(returns)):
                    o = a_obs[i][t_idx]
                    a = a_acts[i][t_idx]
                    logit = o @ agents_w[i] + agents_b[i]
                    pred = sigmoid(logit)
                    grad = a - pred
                    agents_w[i] += lr * returns[t_idx] * grad * o
                    agents_b[i] += lr * returns[t_idx] * grad

            ep_lams.append(lam_sum / max(steps, 1))
            ep_survs.append(float(survived))

        results.append({
            "lambda": float(np.mean(ep_lams[-30:])),
            "survival": float(np.mean(ep_survs[-30:]) * 100),
        })

    return {
        "lambda_mean": float(np.mean([r["lambda"] for r in results])),
        "survival_mean": float(np.mean([r["survival"] for r in results])),
        "trapped": float(np.mean([r["lambda"] for r in results])) < 0.85,
    }


def plot_phase_diagram(data, mn_ratios, n_values, out_dir):
    """Generate 2D heatmap phase diagram."""

    # Create matrices
    lam_matrix = np.zeros((len(n_values), len(mn_ratios)))
    surv_matrix = np.zeros((len(n_values), len(mn_ratios)))

    for i, N in enumerate(n_values):
        for j, mn in enumerate(mn_ratios):
            key = f"N{N}_MN{mn:.2f}"
            if key in data:
                lam_matrix[i, j] = data[key]["lambda_mean"]
                surv_matrix[i, j] = data[key]["survival_mean"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # (a) Lambda heatmap
    ax = axes[0]
    trap_cmap = LinearSegmentedColormap.from_list(
        'trap', ['#c0392b', '#e74c3c', '#f39c12', '#f1c40f', '#2ecc71'], N=256)
    im1 = ax.imshow(lam_matrix, aspect='auto', cmap=trap_cmap,
                    vmin=0.3, vmax=1.0, origin='lower')
    ax.set_xticks(range(len(mn_ratios)))
    ax.set_xticklabels([f'{v:.2f}' for v in mn_ratios], fontsize=10)
    ax.set_yticks(range(len(n_values)))
    ax.set_yticklabels([str(v) for v in n_values], fontsize=10)
    ax.set_xlabel('M/N ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('N (agents)', fontsize=12, fontweight='bold')
    ax.set_title('(a) Converged Commitment Level', fontsize=13, fontweight='bold')
    plt.colorbar(im1, ax=ax, label='Mean lambda', shrink=0.8)

    # Annotate cells
    for i in range(len(n_values)):
        for j in range(len(mn_ratios)):
            val = lam_matrix[i, j]
            color = 'white' if val < 0.55 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

    # (b) Survival heatmap
    ax = axes[1]
    surv_cmap = LinearSegmentedColormap.from_list(
        'surv', ['#c0392b', '#e74c3c', '#f39c12', '#2ecc71', '#27ae60'], N=256)
    im2 = ax.imshow(surv_matrix, aspect='auto', cmap=surv_cmap,
                    vmin=0, vmax=100, origin='lower')
    ax.set_xticks(range(len(mn_ratios)))
    ax.set_xticklabels([f'{v:.2f}' for v in mn_ratios], fontsize=10)
    ax.set_yticks(range(len(n_values)))
    ax.set_yticklabels([str(v) for v in n_values], fontsize=10)
    ax.set_xlabel('M/N ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('N (agents)', fontsize=12, fontweight='bold')
    ax.set_title('(b) Survival Rate (%)', fontsize=13, fontweight='bold')
    plt.colorbar(im2, ax=ax, label='Survival %', shrink=0.8)

    for i in range(len(n_values)):
        for j in range(len(mn_ratios)):
            val = surv_matrix[i, j]
            color = 'white' if val < 60 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

    plt.suptitle('Nash Trap Phase Diagram\n'
                 f'Byz={BYZ_FRAC*100:.0f}%, {N_SEEDS} seeds x {N_EPISODES} ep',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, 'phase_diagram_2d.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {path}")


if __name__ == "__main__":
    print("=" * 70)
    print("  2D PHASE DIAGRAM: M/N x N")
    print(f"  M/N: {MN_RATIOS}")
    print(f"  N:   {N_VALUES}")
    print(f"  Seeds={N_SEEDS}, Episodes={N_EPISODES}")
    print("=" * 70)

    t0 = time.time()
    all_data = {}

    total_conditions = len(MN_RATIOS) * len(N_VALUES)
    idx = 0

    for N in N_VALUES:
        for mn in MN_RATIOS:
            idx += 1
            key = f"N{N}_MN{mn:.2f}"
            M = mn * N
            print(f"\n[{idx}/{total_conditions}] N={N}, M/N={mn:.2f} (M={M:.1f})")

            result = run_condition(N, mn, N_SEEDS, N_EPISODES)
            all_data[key] = result

            status = "TRAP" if result["trapped"] else "FREE"
            print(f"  lam={result['lambda_mean']:.3f}, surv={result['survival_mean']:.0f}% {status}")

    elapsed = time.time() - t0

    output = {
        "experiment": "2D Phase Diagram (M/N x N)",
        "config": {
            "MN_RATIOS": MN_RATIOS,
            "N_VALUES": N_VALUES,
            "N_SEEDS": N_SEEDS, "N_EPISODES": N_EPISODES,
            "BYZ_FRAC": BYZ_FRAC,
        },
        "data": all_data,
        "time_seconds": elapsed,
    }

    json_path = os.path.join(OUTPUT_DIR, "phase_diagram_2d.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[Save] {json_path}")

    plot_phase_diagram(all_data, MN_RATIOS, N_VALUES, OUTPUT_DIR)

    print(f"\n{'=' * 70}")
    print(f"  COMPLETE ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
