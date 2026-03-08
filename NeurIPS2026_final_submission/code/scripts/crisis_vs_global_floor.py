"""
Phase A: Crisis-only vs Global Floor Ablation
==============================================
Separates two commitment floor mechanisms:
  1. GLOBAL floor: effective_lambda = max(learned_lambda, phi1) [ALL states]
  2. CRISIS-ONLY floor: effective_lambda = max(learned_lambda, phi1) ONLY when R < R_override

This directly tests whether unconditional (global) commitment is necessary,
or whether crisis-triggered intervention suffices.

Grid: phi1 in {0.0, 0.3, 0.5, 0.7, 1.0} x floor_type in {global, crisis} x Byz in {0%, 30%} x 20 seeds
"""
import numpy as np
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
from cleanrl_mappo_pgg import (
    NonlinearPGGEnv, MLPActor, MLPCritic, compute_gae,
    ppo_update_actor, bootstrap_ci, relu, NNLayer,
    GAMMA, GAE_LAMBDA, CLIP_EPS, HIDDEN_DIM
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "crisis_vs_global"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Hyperparameters ===
N_EPISODES = 300
N_EVAL = 30
N_SEEDS = 20
T_HORIZON = 50
N_AGENTS = 20
R_CRIT = 0.15
R_OVERRIDE = 0.25  # Crisis threshold for crisis-only mode

PHI1_VALUES = [0.0, 0.3, 0.5, 0.7, 1.0]
BYZ_FRACS = [0.0, 0.30]
FLOOR_TYPES = ["global", "crisis_only"]

if os.environ.get("ETHICAAI_FAST") == "1":
    print("  [FAST MODE] N_SEEDS=2, N_EPISODES=30")
    N_SEEDS = 2
    N_EPISODES = 30
    N_EVAL = 10


def run_ippo_with_floor(seed, phi1, byz_frac, floor_type):
    """Train IPPO agents with specified floor type."""
    rng = np.random.RandomState(seed)
    env = NonlinearPGGEnv(byz_frac=byz_frac)
    n_honest = int(N_AGENTS * (1 - byz_frac))

    actors = [MLPActor(np.random.RandomState(seed * 100 + i)) for i in range(n_honest)]
    critic = MLPCritic(rng)

    episode_data = []

    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        all_obs = [[] for _ in range(n_honest)]
        all_acts = [[] for _ in range(n_honest)]
        all_log_probs = [[] for _ in range(n_honest)]
        all_rewards = [[] for _ in range(n_honest)]
        all_values = [[] for _ in range(n_honest)]

        total_welfare = 0.0
        lam_sum = 0.0
        steps = 0
        survived = True
        agent_rewards_sum = np.zeros(n_honest)

        for t in range(T_HORIZON):
            lambdas = np.zeros(n_honest)
            current_R = obs[0] if len(obs) > 0 else 1.0  # Resource state

            for i in range(n_honest):
                mean, _ = actors[i].forward(obs)
                std = np.exp(actors[i].log_std[0])
                learned_lambda = float(np.clip(mean[0] + rng.randn() * std, 0.01, 0.99))

                # Apply floor based on type
                if floor_type == "global":
                    effective_lambda = max(learned_lambda, phi1)
                elif floor_type == "crisis_only":
                    if current_R < R_OVERRIDE:
                        effective_lambda = max(learned_lambda, phi1)
                    else:
                        effective_lambda = learned_lambda
                else:
                    effective_lambda = learned_lambda

                lambdas[i] = effective_lambda
                all_obs[i].append(obs.copy())
                all_acts[i].append(effective_lambda)
                all_log_probs[i].append(actors[i].log_prob(obs, effective_lambda))
                val = critic.forward(obs)
                all_values[i].append(float(val))

            obs, rewards, terminated, truncated, info = env.step(lambdas)

            for i in range(n_honest):
                r = float(rewards[i]) if hasattr(rewards, '__len__') else float(rewards)
                all_rewards[i].append(r)
                agent_rewards_sum[i] += r

            team_reward = float(np.mean(rewards)) if hasattr(rewards, '__len__') else float(rewards)
            total_welfare += team_reward
            lam_sum += float(lambdas.mean())
            steps += 1

            if terminated:
                survived = info.get("survived", False)
                break

        # PPO update
        for i in range(n_honest):
            if len(all_rewards[i]) < 2:
                continue
            advantages, returns = compute_gae(all_rewards[i], all_values[i])
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / advantages.std()
            ppo_update_actor(actors[i], all_obs[i], all_acts[i],
                           all_log_probs[i], advantages)

        episode_data.append({
            "welfare": total_welfare / max(steps, 1),
            "mean_lambda": lam_sum / max(steps, 1),
            "survived": survived,
            "agent_rewards": agent_rewards_sum.tolist(),
        })

    # Evaluate on last N_EVAL episodes
    eval_eps = episode_data[-N_EVAL:]
    return {
        "survival": float(np.mean([e["survived"] for e in eval_eps]) * 100),
        "welfare": float(np.mean([e["welfare"] for e in eval_eps])),
        "lambda": float(np.mean([e["mean_lambda"] for e in eval_eps])),
    }


def main():
    print("=" * 70)
    print("  Phase A: Crisis-only vs Global Floor Ablation")
    print(f"  N={N_AGENTS}, Episodes={N_EPISODES}, Seeds={N_SEEDS}")
    print("=" * 70)

    t0 = time.time()
    results = {}

    for floor_type in FLOOR_TYPES:
        results[floor_type] = {}
        for phi1 in PHI1_VALUES:
            results[floor_type][str(phi1)] = {}
            for byz in BYZ_FRACS:
                byz_key = f"byz_{int(byz*100)}"
                print(f"\n  [{floor_type}] phi1={phi1:.1f}, Byz={byz*100:.0f}%: "
                      f"Running {N_SEEDS} seeds...")

                seed_results = []
                for s in range(N_SEEDS):
                    r = run_ippo_with_floor(s, phi1, byz, floor_type)
                    seed_results.append(r)
                    if (s + 1) % 5 == 0:
                        print(f"    Seed {s}: surv={r['survival']:.0f}%, "
                              f"W={r['welfare']:.1f}, lam={r['lambda']:.3f}")

                surv = [r["survival"] for r in seed_results]
                welf = [r["welfare"] for r in seed_results]

                results[floor_type][str(phi1)][byz_key] = {
                    "survival_mean": float(np.mean(surv)),
                    "survival_std": float(np.std(surv)),
                    "survival_ci95": bootstrap_ci(surv),
                    "welfare_mean": float(np.mean(welf)),
                    "welfare_std": float(np.std(welf)),
                }

    out_path = OUTPUT_DIR / "crisis_vs_global_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  COMPLETE in {elapsed:.0f}s")
    print(f"\n  Summary (Byz=30%):")
    print(f"  {'phi1':>5} | {'Global Surv':>12} | {'Crisis Surv':>12} | {'Global W':>10} | {'Crisis W':>10}")
    print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}")
    for phi1 in PHI1_VALUES:
        g = results["global"][str(phi1)]["byz_30"]
        c = results["crisis_only"][str(phi1)]["byz_30"]
        print(f"  {phi1:5.1f} | {g['survival_mean']:10.0f}% | {c['survival_mean']:10.0f}% | "
              f"{g['welfare_mean']:10.1f} | {c['welfare_mean']:10.1f}")
    print(f"{'=' * 70}")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
