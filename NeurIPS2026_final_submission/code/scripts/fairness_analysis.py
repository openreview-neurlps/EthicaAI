"""
Phase C: Fairness & Equity Analysis
====================================
Computes fairness metrics across all commitment floor levels:
  - Gini coefficient of agent rewards
  - Min-agent (worst-case) return
  - Reward quantiles (10th, 25th, 50th percentile)
  - Comparison: selfish RL vs commitment floors

This addresses the reviewer attack: "ethics paper without fairness metrics"
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
    ppo_update_actor, bootstrap_ci, GAMMA, GAE_LAMBDA, CLIP_EPS, HIDDEN_DIM
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fairness"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_EPISODES = 200
N_EVAL = 30
N_SEEDS = 20
T_HORIZON = 50
N_AGENTS = 20
BYZ_FRAC = 0.30

PHI1_VALUES = [0.0, 0.3, 0.5, 0.7, 1.0]

if os.environ.get("ETHICAAI_FAST") == "1":
    print("  [FAST MODE]")
    N_SEEDS = 2
    N_EPISODES = 30
    N_EVAL = 10


def gini_coefficient(rewards):
    """Compute Gini coefficient for reward distribution."""
    rewards = np.array(rewards, dtype=float)
    if len(rewards) == 0 or rewards.sum() == 0:
        return 0.0
    rewards = np.sort(rewards)
    n = len(rewards)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * rewards) - (n + 1) * np.sum(rewards)) / (n * np.sum(rewards)))


def run_with_fairness(seed, phi1):
    """Run IPPO with commitment floor, tracking per-agent fairness."""
    rng = np.random.RandomState(seed)
    env = NonlinearPGGEnv(byz_frac=BYZ_FRAC)
    n_honest = int(N_AGENTS * (1 - BYZ_FRAC))

    actors = [MLPActor(np.random.RandomState(seed * 100 + i)) for i in range(n_honest)]
    episode_data = []

    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        all_obs = [[] for _ in range(n_honest)]
        all_acts = [[] for _ in range(n_honest)]
        all_log_probs = [[] for _ in range(n_honest)]
        all_rewards = [[] for _ in range(n_honest)]
        all_values = [[] for _ in range(n_honest)]

        agent_total_rewards = np.zeros(n_honest)
        steps = 0
        survived = True
        critic = MLPCritic(rng)

        for t in range(T_HORIZON):
            lambdas = np.zeros(n_honest)
            for i in range(n_honest):
                mean, _ = actors[i].forward(obs)
                std = np.exp(actors[i].log_std[0])
                learned_lam = float(np.clip(mean[0] + rng.randn() * std, 0.01, 0.99))
                effective_lam = max(learned_lam, phi1)
                lambdas[i] = effective_lam
                all_obs[i].append(obs.copy())
                all_acts[i].append(effective_lam)
                all_log_probs[i].append(actors[i].log_prob(obs, effective_lam))
                all_values[i].append(float(critic.forward(obs)))

            obs, rewards, terminated, truncated, info = env.step(lambdas)
            for i in range(n_honest):
                r = float(rewards[i]) if hasattr(rewards, '__len__') else float(rewards)
                all_rewards[i].append(r)
                agent_total_rewards[i] += r

            steps += 1
            if terminated:
                survived = info.get("survived", False)
                break

        for i in range(n_honest):
            if len(all_rewards[i]) < 2:
                continue
            advantages, returns = compute_gae(all_rewards[i], all_values[i])
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / advantages.std()
            ppo_update_actor(actors[i], all_obs[i], all_acts[i], all_log_probs[i], advantages)

        episode_data.append({
            "survived": survived,
            "agent_rewards": agent_total_rewards.tolist(),
            "gini": gini_coefficient(agent_total_rewards),
            "min_agent": float(np.min(agent_total_rewards)),
            "max_agent": float(np.max(agent_total_rewards)),
            "median_agent": float(np.median(agent_total_rewards)),
            "p10_agent": float(np.percentile(agent_total_rewards, 10)),
            "p25_agent": float(np.percentile(agent_total_rewards, 25)),
        })

    eval_eps = episode_data[-N_EVAL:]
    return {
        "survival": float(np.mean([e["survived"] for e in eval_eps]) * 100),
        "gini_mean": float(np.mean([e["gini"] for e in eval_eps])),
        "gini_std": float(np.std([e["gini"] for e in eval_eps])),
        "min_agent_mean": float(np.mean([e["min_agent"] for e in eval_eps])),
        "max_agent_mean": float(np.mean([e["max_agent"] for e in eval_eps])),
        "p10_mean": float(np.mean([e["p10_agent"] for e in eval_eps])),
        "p25_mean": float(np.mean([e["p25_agent"] for e in eval_eps])),
        "median_mean": float(np.mean([e["median_agent"] for e in eval_eps])),
    }


def main():
    print("=" * 70)
    print("  Phase C: Fairness & Equity Analysis")
    print(f"  Seeds={N_SEEDS}, Episodes={N_EPISODES}")
    print("=" * 70)

    t0 = time.time()
    results = {}

    for phi1 in PHI1_VALUES:
        print(f"\n  phi1={phi1:.1f}: Running {N_SEEDS} seeds...")
        seed_results = []
        for s in range(N_SEEDS):
            r = run_with_fairness(s, phi1)
            seed_results.append(r)

        ginis = [r["gini_mean"] for r in seed_results]
        mins = [r["min_agent_mean"] for r in seed_results]
        survs = [r["survival"] for r in seed_results]

        results[str(phi1)] = {
            "survival_mean": float(np.mean(survs)),
            "gini_mean": float(np.mean(ginis)),
            "gini_std": float(np.std(ginis)),
            "min_agent_mean": float(np.mean(mins)),
            "min_agent_std": float(np.std(mins)),
            "p10_mean": float(np.mean([r["p10_mean"] for r in seed_results])),
            "p25_mean": float(np.mean([r["p25_mean"] for r in seed_results])),
            "median_mean": float(np.mean([r["median_mean"] for r in seed_results])),
        }
        print(f"    -> surv={np.mean(survs):.0f}%, Gini={np.mean(ginis):.3f}, "
              f"min_agent={np.mean(mins):.2f}")

    out_path = OUTPUT_DIR / "fairness_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  COMPLETE in {elapsed:.0f}s")
    print(f"\n  Summary:")
    print(f"  {'phi1':>5} | {'Surv':>6} | {'Gini':>6} | {'MinAgent':>9} | {'P10':>6}")
    print(f"  {'-'*5}-+-{'-'*6}-+-{'-'*6}-+-{'-'*9}-+-{'-'*6}")
    for phi1 in PHI1_VALUES:
        d = results[str(phi1)]
        print(f"  {phi1:5.1f} | {d['survival_mean']:5.0f}% | {d['gini_mean']:6.3f} | "
              f"{d['min_agent_mean']:9.2f} | {d['p10_mean']:6.2f}")
    print(f"{'=' * 70}")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
