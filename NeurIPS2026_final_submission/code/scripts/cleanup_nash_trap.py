"""
Cleanup Nash Trap Verification: REINFORCE on Abstracted Cleanup SSD
====================================================================
Demonstrates Nash Trap in effort-allocation social dilemma.

Key difference from PGG/Harvest:
  - λ = cleaning effort (higher = more prosocial)
  - Selfish agents: λ→0 (free-ride on others' cleaning)
  - Nash Trap: agents converge to low cleaning → pollution ↑ → yield ↓

With commitment floor: minimum cleaning effort enforced.

Paper Ref: Section 5 (Cross-environment validation)
"""

import numpy as np
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
from envs.cleanup_env import CleanupEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / os.environ.get("ETHICAAI_OUTDIR", "outputs") / "cleanup"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
N_SEEDS = 20
N_EPISODES = 300
T_HORIZON = 50
N_AGENTS = 20
BYZ_FRAC = 0.3
LEARNING_RATE = 0.01
GAMMA = 0.99

# Commitment floor: minimum cleaning effort
PHI1_VALUES = [0.0, 0.3, 0.5, 0.7, 1.0]

if os.environ.get("ETHICAAI_FAST") == "1":
    print("  [FAST MODE]")
    N_SEEDS = 3
    N_EPISODES = 100
    PHI1_VALUES = [0.0, 0.5, 1.0]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def run_reinforce_cleanup(seed, phi1=0.0):
    """Run REINFORCE on Cleanup env with optional cleaning floor."""
    rng = np.random.RandomState(seed)
    env = CleanupEnv(n_agents=N_AGENTS, byz_frac=BYZ_FRAC, t_horizon=T_HORIZON)
    n_honest = env.n_honest

    # Policy params: theta → lambda = sigmoid(theta) = cleaning effort
    thetas = rng.randn(n_honest) * 0.1

    episode_survivals = []
    episode_lambdas = []
    episode_welfares = []

    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        survived = True

        log_probs = [[] for _ in range(n_honest)]
        rewards_buf = [[] for _ in range(n_honest)]

        for t in range(T_HORIZON):
            base_lambdas = sigmoid(thetas)
            noise = rng.randn(n_honest) * 0.1
            lambdas_raw = np.clip(base_lambdas + noise, 0.01, 0.99)

            # Apply commitment floor: minimum cleaning effort = phi1
            if phi1 > 0:
                lambdas_effective = np.maximum(lambdas_raw, phi1)
            else:
                lambdas_effective = lambdas_raw

            obs, rewards, terminated, truncated, info = env.step(lambdas_effective)

            for i in range(n_honest):
                log_probs[i].append(-(lambdas_raw[i] - base_lambdas[i])**2 / (2 * 0.1**2))
                rewards_buf[i].append(float(rewards[i]))

            if terminated:
                survived = info.get("survived", False)
                break

        episode_survivals.append(float(survived))
        episode_lambdas.append(float(np.mean(sigmoid(thetas))))
        if len(rewards_buf[0]) > 0:
            episode_welfares.append(float(np.mean([np.mean(r) for r in rewards_buf])))
        else:
            episode_welfares.append(0.0)

        # REINFORCE update
        for i in range(n_honest):
            if len(rewards_buf[i]) == 0:
                continue
            returns = []
            G = 0
            for r in reversed(rewards_buf[i]):
                G = r + GAMMA * G
                returns.insert(0, G)
            returns = np.array(returns)
            if returns.std() > 1e-8:
                returns = (returns - returns.mean()) / returns.std()
            grad = sum(log_probs[i][t_idx] * returns[t_idx] for t_idx in range(len(log_probs[i])))
            thetas[i] += LEARNING_RATE * grad

    final_surv = np.mean(episode_survivals[-50:]) * 100
    final_lambda = np.mean(episode_lambdas[-50:])
    final_welfare = np.mean(episode_welfares[-50:])

    return {
        "survival_rate": final_surv,
        "mean_lambda": final_lambda,
        "mean_welfare": final_welfare,
        "all_survivals": episode_survivals,
        "all_lambdas": episode_lambdas,
    }


def main():
    print("=" * 60)
    print("  Cleanup Nash Trap Verification")
    print("  Seeds=%d, Episodes=%d, Agents=%d" % (N_SEEDS, N_EPISODES, N_AGENTS))
    print("=" * 60)

    t0 = time.time()
    results = {}

    for phi1 in PHI1_VALUES:
        print("\n--- phi1=%.1f (min cleaning effort) ---" % phi1)
        seed_results = []

        for s in range(N_SEEDS):
            r = run_reinforce_cleanup(s, phi1)
            seed_results.append(r)
            sys.stdout.write("  seed %d: surv=%.0f%% lambda=%.3f welfare=%.1f\n" % (
                s, r["survival_rate"], r["mean_lambda"], r["mean_welfare"]))
            sys.stdout.flush()

        survivals = [r["survival_rate"] for r in seed_results]
        lambdas = [r["mean_lambda"] for r in seed_results]
        welfares = [r["mean_welfare"] for r in seed_results]

        agg = {
            "survival": {
                "mean": float(np.mean(survivals)),
                "std": float(np.std(survivals)),
                "ci95": [float(np.percentile(survivals, 2.5)),
                         float(np.percentile(survivals, 97.5))],
            },
            "mean_lambda": {
                "mean": float(np.mean(lambdas)),
                "std": float(np.std(lambdas)),
            },
            "welfare": {
                "mean": float(np.mean(welfares)),
                "std": float(np.std(welfares)),
            },
            "in_trap": float(np.mean(lambdas)) < 0.85,
            "per_seed": seed_results,
        }

        tag = "phi1_%.1f" % phi1
        results[tag] = agg
        print("  => surv=%.1f%% +/- %.1f  lambda=%.3f  trap=%s" % (
            agg["survival"]["mean"], agg["survival"]["std"],
            agg["mean_lambda"]["mean"], agg["in_trap"]))

    output = {
        "experiment": "cleanup_nash_trap",
        "environment": "CleanupEnv (Abstracted SSD)",
        "description": "REINFORCE on Abstracted Cleanup — effort allocation dilemma. "
                       "Demonstrates Nash Trap in maintenance-based social dilemma.",
        "config": {
            "N_SEEDS": N_SEEDS,
            "N_EPISODES": N_EPISODES,
            "T_HORIZON": T_HORIZON,
            "N_AGENTS": N_AGENTS,
            "BYZ_FRAC": BYZ_FRAC,
            "LEARNING_RATE": LEARNING_RATE,
            "GAMMA": GAMMA,
            "framework": "NumPy REINFORCE",
        },
        "run_meta": {
            "timestamp": time.time(),
            "git_sha": os.popen("git rev-parse HEAD").read().strip(),
            "elapsed_seconds": time.time() - t0,
            "mode": "FAST" if os.environ.get("ETHICAAI_FAST") == "1" else "FULL",
        },
        "results": results,
    }

    out_path = OUTPUT_DIR / "cleanup_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("  " + "-" * 56)
    for phi1 in PHI1_VALUES:
        tag = "phi1_%.1f" % phi1
        r = results[tag]
        s = r["survival"]["mean"]
        l = r["mean_lambda"]["mean"]
        trap = "IN TRAP" if r["in_trap"] else "ESCAPED"
        print("  phi1=%.1f: surv=%5.1f%%  lambda=%.3f  [%s]" % (phi1, s, l, trap))
    print("  " + "-" * 56)
    print("  Saved: %s" % out_path)
    print("  DONE in %ds" % (time.time() - t0))
    print("=" * 60)


if __name__ == "__main__":
    main()
