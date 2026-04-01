"""
Fishery Nash Trap Experiment — Gordon-Schaefer Bioeconomic Model
================================================================
Demonstrates that the Nash Trap phenomenon persists in a scientifically
grounded ecological model (Gordon 1954, Clark 1990, Scheffer 2009),
independent of our custom PGG design.

Protocol:
  1. REINFORCE agents learn fishing restraint levels λ ∈ [0,1]
  2. λ=1 → full restraint (cooperative), λ=0 → max extraction (selfish)
  3. Sweep commitment floors φ₁ ∈ {0.0, 0.3, 0.5, 0.7, 1.0}
  4. 20 seeds × 300 episodes × 50 timesteps

Expected: agents converge to λ ≈ 0.3-0.5 (Nash Trap) with low survival.
Only commitment floor φ₁ ≥ 0.7 achieves reliable survival.

Paper Ref: Section 5 (Cross-environment validation), Appendix (Fishery)
"""

import numpy as np
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from envs.fishery_env import FisheryEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / os.environ.get("ETHICAAI_OUTDIR", "outputs") / "fishery"
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
GAMMA = 0.99
NOISE_STD = 0.10

PHI1_VALUES = [0.0, 0.3, 0.5, 0.7, 1.0]
EVAL_LAST = 30  # Evaluate last N episodes

if os.environ.get("ETHICAAI_FAST") == "1":
    print("  [FAST MODE]")
    N_SEEDS = 5
    N_EPISODES = 150
    PHI1_VALUES = [0.0, 0.5, 1.0]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def run_reinforce_fishery(seed, phi1=0.0):
    """Run REINFORCE on Fishery env with optional restraint floor."""
    rng = np.random.RandomState(seed)
    env = FisheryEnv(
        n_agents=N_AGENTS, byz_frac=BYZ_FRAC, t_horizon=T_HORIZON,
    )
    n_honest = env.n_honest

    # Per-agent policy: θ → λ = sigmoid(θ) (restraint level)
    thetas = rng.randn(n_honest) * 0.1

    episode_data = {
        "survivals": [],
        "lambdas": [],
        "welfares": [],
        "biomass_final": [],
    }

    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=seed * 10000 + ep)

        log_probs = [[] for _ in range(n_honest)]
        rewards_buf = [[] for _ in range(n_honest)]
        ep_lambdas = []

        for t in range(T_HORIZON):
            # Sample actions
            base_lambdas = sigmoid(thetas)
            noise = rng.randn(n_honest) * NOISE_STD
            lambdas_raw = np.clip(base_lambdas + noise, 0.01, 0.99)

            # Apply commitment floor: minimum restraint level
            if phi1 > 0:
                lambdas_effective = np.maximum(lambdas_raw, phi1)
            else:
                lambdas_effective = lambdas_raw

            obs, rewards, terminated, _, info = env.step(lambdas_effective)
            ep_lambdas.append(float(np.mean(lambdas_effective)))

            for i in range(n_honest):
                log_probs[i].append(
                    -(lambdas_raw[i] - base_lambdas[i])**2 / (2 * NOISE_STD**2)
                )
                rewards_buf[i].append(float(rewards[i]))

            if terminated:
                break

        # Episode stats
        survived = info.get("survived", not info.get("collapsed", True))
        episode_data["survivals"].append(float(survived))
        episode_data["lambdas"].append(float(np.mean(ep_lambdas)) if ep_lambdas else 0.5)
        episode_data["biomass_final"].append(float(info.get("biomass", 0.0)))

        if rewards_buf[0]:
            episode_data["welfares"].append(
                float(np.mean([sum(r) for r in rewards_buf]))
            )
        else:
            episode_data["welfares"].append(0.0)

        # REINFORCE update (skip if floor overrides everything)
        if phi1 < 0.99:
            for i in range(n_honest):
                if not rewards_buf[i]:
                    continue
                returns = []
                G = 0
                for r in reversed(rewards_buf[i]):
                    G = r + GAMMA * G
                    returns.insert(0, G)
                returns = np.array(returns)
                if returns.std() > 1e-8:
                    returns = (returns - returns.mean()) / returns.std()

                grad = sum(
                    log_probs[i][t_idx] * returns[t_idx]
                    for t_idx in range(len(log_probs[i]))
                )
                thetas[i] += LEARNING_RATE * grad

    # Final evaluation (last EVAL_LAST episodes)
    eval_s = episode_data["survivals"][-EVAL_LAST:]
    eval_l = episode_data["lambdas"][-EVAL_LAST:]
    eval_w = episode_data["welfares"][-EVAL_LAST:]
    eval_b = episode_data["biomass_final"][-EVAL_LAST:]

    return {
        "survival_rate": float(np.mean(eval_s) * 100),
        "mean_lambda": float(np.mean(eval_l)),
        "mean_welfare": float(np.mean(eval_w)),
        "mean_biomass_final": float(np.mean(eval_b)),
        "in_trap": float(np.mean(eval_l)) < 0.85,
        "all_survivals": episode_data["survivals"],
        "all_lambdas": episode_data["lambdas"],
    }


def bootstrap_ci(values, n_boot=5000, alpha=0.05):
    """Bootstrap confidence interval."""
    values = np.array(values)
    boot_means = [
        np.mean(np.random.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ]
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return [float(lo), float(hi)]


def main():
    print("=" * 65)
    print("  Gordon-Schaefer Fishery Nash Trap Experiment")
    print("  Model: Logistic growth + Allee effect tipping point")
    print("  Ref: Gordon(1954), Clark(1990), Scheffer(2009)")
    print("  Seeds=%d, Episodes=%d, Agents=%d, Byz=%.0f%%" % (
        N_SEEDS, N_EPISODES, N_AGENTS, BYZ_FRAC * 100))
    print("  phi1 values: %s" % PHI1_VALUES)
    print("=" * 65)

    t0 = time.time()
    results = {}

    for phi1 in PHI1_VALUES:
        print("\n--- phi1=%.1f (min restraint = %.1f) ---" % (phi1, phi1))
        seed_results = []

        for s in range(N_SEEDS):
            r = run_reinforce_fishery(s, phi1)
            seed_results.append(r)
            if s % 5 == 0:
                print("  seed %2d: surv=%5.1f%% λ=%.3f welfare=%.3f B=%.3f %s" % (
                    s, r["survival_rate"], r["mean_lambda"],
                    r["mean_welfare"], r["mean_biomass_final"],
                    "TRAP" if r["in_trap"] else "OK"))

        # Aggregate across seeds
        survivals = [r["survival_rate"] for r in seed_results]
        lambdas = [r["mean_lambda"] for r in seed_results]
        welfares = [r["mean_welfare"] for r in seed_results]
        biomasses = [r["mean_biomass_final"] for r in seed_results]
        trap_rates = [r["in_trap"] for r in seed_results]

        agg = {
            "survival": {
                "mean": float(np.mean(survivals)),
                "std": float(np.std(survivals)),
                "ci95": bootstrap_ci(survivals),
            },
            "mean_lambda": {
                "mean": float(np.mean(lambdas)),
                "std": float(np.std(lambdas)),
                "ci95": bootstrap_ci(lambdas),
            },
            "welfare": {
                "mean": float(np.mean(welfares)),
                "std": float(np.std(welfares)),
            },
            "biomass_final": {
                "mean": float(np.mean(biomasses)),
                "std": float(np.std(biomasses)),
            },
            "trap_rate": float(np.mean(trap_rates) * 100),
            "n_seeds": N_SEEDS,
            "per_seed": seed_results,
        }

        tag = "phi1_%.1f" % phi1
        results[tag] = agg
        print("  => surv=%.1f%% ± %.1f  λ=%.3f ± %.3f  trap=%.0f%%" % (
            agg["survival"]["mean"], agg["survival"]["std"],
            agg["mean_lambda"]["mean"], agg["mean_lambda"]["std"],
            agg["trap_rate"]))

    # ── Save ──
    output = {
        "experiment": "fishery_nash_trap",
        "environment": "Gordon-Schaefer Fishery (Allee-effect TPSD)",
        "description": (
            "REINFORCE on Gordon-Schaefer bioeconomic fishery model with "
            "Allee-effect tipping point. Demonstrates Nash Trap persistence "
            "in a scientifically grounded ecological model independent of "
            "our custom PGG. Based on Gordon(1954), Clark(1990), Scheffer(2009)."
        ),
        "ecological_model": {
            "type": "Gordon-Schaefer with Allee effect",
            "dynamics": "B_{t+1} = B_t + r(B)*B*(1-B/K) - q*Σe_i*B - shock",
            "tipping_point": "r(B<B_crit) = 0.01 (Allee collapse)",
            "references": [
                "Gordon, H.S. (1954). The Economic Theory of a Common-Property Resource: The Fishery.",
                "Schaefer, M.B. (1957). Some Considerations of Population Dynamics...",
                "Clark, C.W. (1990). Mathematical Bioeconomics.",
                "Scheffer, M. et al. (2009). Early-warning signals for critical transitions.",
            ],
        },
        "config": {
            "N_SEEDS": N_SEEDS,
            "N_EPISODES": N_EPISODES,
            "T_HORIZON": T_HORIZON,
            "N_AGENTS": N_AGENTS,
            "BYZ_FRAC": BYZ_FRAC,
            "LEARNING_RATE": LEARNING_RATE,
            "GAMMA": GAMMA,
            "NOISE_STD": NOISE_STD,
            "r_normal": 0.30,
            "r_collapse": 0.01,
            "b_crit": 0.15,
            "b_recov": 0.25,
            "catchability": 0.08,
            "framework": "NumPy REINFORCE (CleanRL convention)",
        },
        "run_meta": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_seconds": time.time() - t0,
            "mode": "FAST" if os.environ.get("ETHICAAI_FAST") == "1" else "FULL",
        },
        "results": results,
    }

    out_path = OUTPUT_DIR / "fishery_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    # ── Summary ──
    print("\n" + "=" * 65)
    print("  GORDON-SCHAEFER FISHERY — RESULTS SUMMARY")
    print("  " + "-" * 61)
    for phi1 in PHI1_VALUES:
        tag = "phi1_%.1f" % phi1
        r = results[tag]
        s = r["survival"]["mean"]
        l = r["mean_lambda"]["mean"]
        trap = r["trap_rate"]
        status = "IN TRAP" if trap > 50 else "ESCAPED"
        print("  φ₁=%.1f: surv=%5.1f%%  λ=%.3f  trap_rate=%3.0f%%  [%s]" % (
            phi1, s, l, trap, status))
    print("  " + "-" * 61)
    print("  Saved: %s" % out_path)
    print("  DONE in %.0fs" % (time.time() - t0))
    print("=" * 65)


if __name__ == "__main__":
    main()
