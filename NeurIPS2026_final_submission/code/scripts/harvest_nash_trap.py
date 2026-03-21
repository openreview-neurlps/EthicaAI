"""
Harvest Nash Trap Verification: REINFORCE on Abstracted Harvest SSD
====================================================================
Demonstrates that the Nash Trap phenomenon generalizes beyond PGG
to the Harvest (Tragedy of the Commons) environment.

Experimental protocol (mirrors reinforce_baseline.py):
  1. Selfish REINFORCE → agents learn extraction rates → Nash Trap
  2. With commitment floor (cap extraction) → survival improves
  3. Multiple seeds for statistical robustness

Expected result: agents converge to ~moderate extraction (λ≈0.4-0.6)
that depletes resources faster than regrowth (= Nash Trap).

Paper Ref: Section 5 (Cross-environment validation)
"""

import numpy as np
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
from envs.harvest_env import HarvestEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / os.environ.get("ETHICAAI_OUTDIR", "outputs") / "harvest"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# Configuration
# ================================================================
N_SEEDS = 20
N_EPISODES = 300
T_HORIZON = 50
N_AGENTS = 20
BYZ_FRAC = 0.3

LEARNING_RATE = 0.01
GAMMA = 0.99    # Discount factor

# Commitment floor: cap extraction at (1 - phi1)
# phi1=0: no cap (selfish); phi1=0.5: max extraction = 0.5
PHI1_VALUES = [0.0, 0.3, 0.5, 0.7, 1.0]

if os.environ.get("ETHICAAI_FAST") == "1":
    print("  [FAST MODE]")
    N_SEEDS = 3
    N_EPISODES = 100
    PHI1_VALUES = [0.0, 0.5, 1.0]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def run_reinforce_harvest(seed, phi1=0.0):
    """Run REINFORCE on Harvest env with optional extraction cap."""
    rng = np.random.RandomState(seed)
    env = HarvestEnv(n_agents=N_AGENTS, byz_frac=BYZ_FRAC, t_horizon=T_HORIZON)
    n_honest = env.n_honest

    # Per-agent policy parameters (sigmoid parameterization)
    # theta → lambda = sigmoid(theta), same as PGG
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
            # Sample actions using sigmoid policy + noise
            base_lambdas = sigmoid(thetas)
            noise = rng.randn(n_honest) * 0.1
            # In Harvest: lambda = extraction rate
            # Higher lambda = more extraction = more selfish
            lambdas_raw = np.clip(base_lambdas + noise, 0.01, 0.99)

            # Apply extraction cap: max extraction = (1 - phi1)
            # phi1=0 → no cap; phi1=0.5 → max extraction = 0.5
            if phi1 > 0:
                lambdas_effective = np.minimum(lambdas_raw, 1.0 - phi1)
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

        # REINFORCE update (only for non-capped actions)
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

            grad = 0
            for t_idx in range(len(log_probs[i])):
                grad += log_probs[i][t_idx] * returns[t_idx]
            thetas[i] += LEARNING_RATE * grad

    # Final statistics (last 50 episodes)
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
    print("  Harvest Nash Trap Verification")
    print("  Seeds=%d, Episodes=%d, Agents=%d" % (N_SEEDS, N_EPISODES, N_AGENTS))
    print("  phi1 values: %s" % PHI1_VALUES)
    print("=" * 60)

    t0 = time.time()
    results = {}

    for phi1 in PHI1_VALUES:
        print("\n--- phi1=%.1f (extraction cap = %.1f) ---" % (phi1, 1.0 - phi1))
        seed_results = []

        for s in range(N_SEEDS):
            r = run_reinforce_harvest(s, phi1)
            seed_results.append(r)
            sys.stdout.write("  seed %d: surv=%.0f%% lambda=%.3f welfare=%.1f\n" % (
                s, r["survival_rate"], r["mean_lambda"], r["mean_welfare"]))
            sys.stdout.flush()

        # Aggregate
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

    # Save
    output = {
        "experiment": "harvest_nash_trap",
        "environment": "HarvestEnv (Abstracted SSD)",
        "description": "REINFORCE on Abstracted Harvest — Tragedy of the Commons. "
                       "Demonstrates Nash Trap persistence in extraction-based social dilemma.",
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

    out_path = OUTPUT_DIR / "harvest_results.json"
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
