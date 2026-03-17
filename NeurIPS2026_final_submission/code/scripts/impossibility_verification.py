"""
Impossibility Theorem Empirical Verification
==============================================
Verifies the three predictions of Theorem 2:
  (i)   Signal dilution: gradient magnitude ~ O(1/N)
  (ii)  Basin dominance: trap convergence rate ~ 1 - exp(-cN)
  (iii) Escape time:     escape requires exp(Omega(N)) episodes

Sweeps N in {5, 10, 20, 50, 100} to measure scaling laws.

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
N_VALUES = [5, 10, 20, 50, 100]
N_SEEDS = 20
N_EPISODES = 200
BYZ_FRAC = 0.30
GAMMA = 0.99

if os.environ.get("ETHICAAI_FAST") == "1":
    print("  [FAST MODE]")
    N_VALUES = [5, 10, 20]
    N_SEEDS = 5
    N_EPISODES = 50

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'impossibility')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# ============================================================
# Part (i): Signal Dilution — Gradient magnitude vs N
# ============================================================
def measure_gradient_signal(N, n_seeds=10):
    """
    Estimate |d P_surv / d lambda_i| empirically via finite difference.
    For each seed, perturb one agent's lambda by epsilon and measure
    the change in survival probability across K rollouts.
    """
    n_honest = N - int(N * BYZ_FRAC)  # Must match NonlinearPGGEnv calculation
    M = 1.6  # Fixed multiplier (as in paper)
    FD_EPS = 0.05
    K_ROLLOUTS = 50

    gradient_magnitudes = []

    for seed in range(n_seeds):
        rng = np.random.RandomState(42 + seed)
        base_lambda = 0.5  # Evaluate gradient near the Nash Trap

        # Baseline survival at base_lambda
        surv_base = 0
        for k in range(K_ROLLOUTS):
            env = NonlinearPGGEnv(n_agents=N, multiplier=M, byz_frac=BYZ_FRAC)
            obs, _ = env.reset(seed=seed * 1000 + k)
            survived = True
            for t in range(50):
                lambdas = np.full(n_honest, base_lambda)
                obs, _, terminated, _, info = env.step(lambdas)
                if terminated:
                    survived = info.get("survived", False)
                    break
            surv_base += int(survived)
        P_base = surv_base / K_ROLLOUTS

        # Perturbed survival: increase agent 0's lambda by epsilon
        surv_plus = 0
        for k in range(K_ROLLOUTS):
            env = NonlinearPGGEnv(n_agents=N, multiplier=M, byz_frac=BYZ_FRAC)
            obs, _ = env.reset(seed=seed * 1000 + k)
            survived = True
            for t in range(50):
                lambdas = np.full(n_honest, base_lambda)
                lambdas[0] = base_lambda + FD_EPS  # perturb agent 0
                obs, _, terminated, _, info = env.step(lambdas)
                if terminated:
                    survived = info.get("survived", False)
                    break
            surv_plus += int(survived)
        P_plus = surv_plus / K_ROLLOUTS

        grad_mag = abs(P_plus - P_base) / FD_EPS
        gradient_magnitudes.append(grad_mag)

    return {
        "N": N,
        "gradient_mean": float(np.mean(gradient_magnitudes)),
        "gradient_std": float(np.std(gradient_magnitudes)),
        "expected_1_over_N": 1.0 / N,
    }


# ============================================================
# Part (ii): Basin Dominance — Trap convergence rate vs N
# ============================================================
def measure_basin(N, n_seeds=20):
    """
    For each seed, initialize agents with random lambda_0 in [0,1],
    run REINFORCE for N_EPISODES, check if converges to trap (lambda < 0.85).
    """
    n_honest = N - int(N * BYZ_FRAC)
    M = 1.6
    STATE_DIM = 4
    trapped_count = 0
    final_lambdas = []

    for seed in range(n_seeds):
        rng = np.random.RandomState(42 + seed)
        env = NonlinearPGGEnv(n_agents=N, multiplier=M, byz_frac=BYZ_FRAC)

        # Random initialization for agent parameters
        agents_w = [rng.randn(STATE_DIM) * 0.5 for _ in range(n_honest)]
        agents_b = [rng.randn() * 0.5 for _ in range(n_honest)]
        lr = 0.01

        ep_lams = []
        for ep in range(N_EPISODES):
            obs, _ = env.reset(seed=seed * 10000 + ep)
            noise = 0.15 - 0.13 * min(ep / N_EPISODES, 1.0)

            a_obs = [[] for _ in range(n_honest)]
            a_acts = [[] for _ in range(n_honest)]
            a_rew = [[] for _ in range(n_honest)]
            lam_sum, steps = 0.0, 0

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

        final_lam = np.mean(ep_lams[-30:])
        final_lambdas.append(final_lam)
        if final_lam < 0.85:
            trapped_count += 1

    return {
        "N": N,
        "trap_rate": float(trapped_count / n_seeds),
        "expected_lower_bound": float(1.0 - np.exp(-0.1 * N)),
        "final_lambda_mean": float(np.mean(final_lambdas)),
        "final_lambda_std": float(np.std(final_lambdas)),
    }


# ============================================================
# Part (iii): Escape Time — Episodes needed to escape vs N
# ============================================================
def measure_escape_time(N, n_seeds=10):
    """
    Start at Nash Trap (lambda ~ 0.5), add cooperative bias noise,
    measure how many episodes until lambda > 0.85 (if ever).
    """
    n_honest = N - int(N * BYZ_FRAC)
    M = 1.6
    STATE_DIM = 4
    MAX_EP = 500
    escape_times = []

    for seed in range(n_seeds):
        rng = np.random.RandomState(42 + seed)
        env = NonlinearPGGEnv(n_agents=N, multiplier=M, byz_frac=BYZ_FRAC)

        # Initialize in trap zone
        agents_w = [np.zeros(STATE_DIM) for _ in range(n_honest)]
        agents_b = [0.0 for _ in range(n_honest)]  # sigmoid(0) = 0.5
        lr = 0.01

        escaped = False
        escape_ep = MAX_EP

        for ep in range(MAX_EP):
            obs, _ = env.reset(seed=seed * 10000 + ep)
            noise = 0.10  # Fixed noise to allow exploration

            a_obs = [[] for _ in range(n_honest)]
            a_acts = [[] for _ in range(n_honest)]
            a_rew = [[] for _ in range(n_honest)]
            lam_sum, steps = 0.0, 0

            for t in range(50):
                lambdas = np.zeros(n_honest)
                for i in range(n_honest):
                    logit = obs @ agents_w[i] + agents_b[i]
                    base = sigmoid(logit)
                    # Add upward bias to help escape
                    lam_i = float(np.clip(base + rng.randn() * noise + 0.02, 0.01, 0.99))
                    lambdas[i] = lam_i
                    a_obs[i].append(obs.copy())
                    a_acts[i].append(lam_i)

                obs_next, rewards, terminated, _, info = env.step(lambdas)
                for i in range(n_honest):
                    a_rew[i].append(float(rewards[i]))

                lam_sum += float(lambdas.mean())
                steps += 1
                if terminated:
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

            mean_lam = lam_sum / max(steps, 1)
            if mean_lam > 0.85:
                escaped = True
                escape_ep = ep
                break

        escape_times.append(escape_ep if not escaped else escape_ep)

    return {
        "N": N,
        "escape_rate": float(sum(1 for t in escape_times if t < MAX_EP) / len(escape_times)),
        "mean_escape_time": float(np.mean(escape_times)),
        "max_episodes_tested": MAX_EP,
    }


# ============================================================
# Plotting
# ============================================================
def plot_all(grad_results, basin_results, escape_results, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # (a) Signal dilution
    ax = axes[0]
    Ns = [r["N"] for r in grad_results]
    grads = [r["gradient_mean"] for r in grad_results]
    grad_stds = [r["gradient_std"] for r in grad_results]
    theory = [r["expected_1_over_N"] for r in grad_results]

    ax.errorbar(Ns, grads, yerr=grad_stds, fmt='o-', color='#e74c3c',
                linewidth=2, markersize=8, capsize=4, label='Measured')
    ax.plot(Ns, theory, 's--', color='#3498db', linewidth=2, markersize=7,
            label='Theory: O(1/N)')
    ax.set_xlabel('N (agents)', fontsize=12)
    ax.set_ylabel('|grad P_surv / grad lambda_i|', fontsize=12)
    ax.set_title('(a) Signal Dilution', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=9)

    # (b) Basin dominance
    ax = axes[1]
    Ns = [r["N"] for r in basin_results]
    rates = [r["trap_rate"] for r in basin_results]
    bounds = [r["expected_lower_bound"] for r in basin_results]

    ax.plot(Ns, rates, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='Measured')
    ax.plot(Ns, bounds, 's--', color='#3498db', linewidth=2, markersize=7,
            label='Theory: 1 - exp(-cN)')
    ax.set_xlabel('N (agents)', fontsize=12)
    ax.set_ylabel('Trap convergence rate', fontsize=12)
    ax.set_title('(b) Basin Dominance', fontsize=13, fontweight='bold')
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=9)

    # (c) Escape time
    ax = axes[2]
    Ns = [r["N"] for r in escape_results]
    times = [r["mean_escape_time"] for r in escape_results]
    rates = [r["escape_rate"] for r in escape_results]

    ax.bar(range(len(Ns)), times,
           color=['#2ecc71' if r > 0 else '#e74c3c' for r in rates],
           alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(Ns)))
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel('N (agents)', fontsize=12)
    ax.set_ylabel('Episodes to escape (or max)', fontsize=12)
    ax.set_title('(c) Escape Time', fontsize=13, fontweight='bold')

    # Add escape rate text
    for i, (n, t, r) in enumerate(zip(Ns, times, rates)):
        ax.text(i, t + 10, f'{r*100:.0f}%', ha='center', fontsize=9)

    plt.suptitle('Impossibility Theorem Verification\n'
                 'Gradient Learner Scaling Laws in TPSDs',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'impossibility_verification.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {out_dir}/impossibility_verification.png")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  IMPOSSIBILITY THEOREM VERIFICATION")
    print(f"  N sweep: {N_VALUES}")
    print(f"  Seeds={N_SEEDS}, Episodes={N_EPISODES}")
    print("=" * 70)

    t0 = time.time()

    # Part (i): Signal dilution
    print("\n--- Part (i): Signal Dilution ---")
    grad_results = []
    for N in N_VALUES:
        print(f"  N={N}...")
        r = measure_gradient_signal(N, min(N_SEEDS, 10))
        grad_results.append(r)
        print(f"    grad = {r['gradient_mean']:.4f} +/- {r['gradient_std']:.4f} "
              f"(theory: {r['expected_1_over_N']:.4f})")

    # Part (ii): Basin dominance
    print("\n--- Part (ii): Basin Dominance ---")
    basin_results = []
    for N in N_VALUES:
        print(f"  N={N}...")
        r = measure_basin(N, N_SEEDS)
        basin_results.append(r)
        print(f"    trap_rate = {r['trap_rate']:.2f} "
              f"(bound: {r['expected_lower_bound']:.2f})")

    # Part (iii): Escape time
    print("\n--- Part (iii): Escape Time ---")
    escape_results = []
    for N in N_VALUES:
        print(f"  N={N}...")
        r = measure_escape_time(N, min(N_SEEDS, 10))
        escape_results.append(r)
        print(f"    escape_rate = {r['escape_rate']:.2f}, "
              f"mean_time = {r['mean_escape_time']:.0f}")

    elapsed = time.time() - t0

    # Save
    output = {
        "experiment": "Impossibility Theorem Verification",
        "signal_dilution": grad_results,
        "basin_dominance": basin_results,
        "escape_time": escape_results,
        "time_seconds": elapsed,
    }

    json_path = os.path.join(OUTPUT_DIR, "impossibility_results.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[Save] {json_path}")

    # Plot
    plot_all(grad_results, basin_results, escape_results, OUTPUT_DIR)

    print(f"\n{'=' * 70}")
    print(f"  COMPLETE ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
