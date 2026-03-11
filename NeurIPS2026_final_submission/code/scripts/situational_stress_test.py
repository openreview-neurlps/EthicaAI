"""
Situational Stress-Test — Demonstrates worst-case brittleness of situational commitment.
Tests 3 stress conditions where situational commitment degrades sharply
while unconditional floors maintain survival.

Conditions:
  A: Near-tipping start (R0 = R_crit + 0.02)
  B: Long horizon (T = 200)
  C: Shock burst (p_shock = 0.15, delta_shock = 0.20)
"""
import numpy as np
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from envs.nonlinear_pgg_env import NonlinearPGGEnv

N_SEEDS = int(os.environ.get("STRESS_SEEDS", 20))
N_EPISODES = int(os.environ.get("STRESS_EPISODES", 200))
N_AGENTS = 5
BYZ_FRAC = 0.30
N_BYZ = max(1, int(N_AGENTS * BYZ_FRAC))

STRESS_CONDITIONS = {
    "baseline_severe": {
        "desc": "Severe TPSD (default shock params)",
        "R_init": 1.0, "T": 50,
        "p_shock": 0.05, "delta_shock": 0.15,
    },
    "near_tipping_start": {
        "desc": "R0 near tipping point",
        "R_init": 0.17, "T": 50,
        "p_shock": 0.05, "delta_shock": 0.15,
    },
    "long_horizon": {
        "desc": "Extended horizon T=200",
        "R_init": 1.0, "T": 200,
        "p_shock": 0.05, "delta_shock": 0.15,
    },
    "shock_burst": {
        "desc": "Extreme shocks (p=0.15, delta=0.20)",
        "R_init": 1.0, "T": 50,
        "p_shock": 0.15, "delta_shock": 0.20,
    },
    "combined_worst": {
        "desc": "Near-tipping + long horizon + burst shocks",
        "R_init": 0.17, "T": 200,
        "p_shock": 0.15, "delta_shock": 0.20,
    },
}

POLICIES = {
    "situational": "Situational commitment (g(theta, R))",
    "unconditional": "Unconditional floor (phi1=1.0)",
    "selfish_rl": "Nash Trap baseline (lambda~0.5)",
}


def make_env(condition):
    env = NonlinearPGGEnv(n_agents=N_AGENTS)
    env.R = condition["R_init"]
    return env


def situational_action(R, R_crit=0.15, theta=0.7):
    """Situational commitment: reduces lambda when R < R_crit."""
    if R < R_crit:
        return max(0.0, np.sin(theta) * 0.3)
    else:
        return 0.7 + 0.3 * np.sin(theta)


def run_episode(condition, policy_name, seed):
    rng = np.random.RandomState(seed)
    env = make_env(condition)
    T = condition["T"]
    p_shock = condition["p_shock"]
    delta_shock = condition["delta_shock"]
    R_crit = 0.15

    total_reward = 0.0
    survived = True
    min_R = env.R
    crisis_steps = 0

    for t in range(T):
        # Generate commitments
        actions = []
        for i in range(N_AGENTS):
            if i < N_BYZ:
                # Byzantine: always defect
                actions.append(0.0)
            else:
                if policy_name == "situational":
                    actions.append(situational_action(env.R, R_crit))
                elif policy_name == "unconditional":
                    actions.append(1.0)
                elif policy_name == "selfish_rl":
                    actions.append(0.5 + rng.normal(0, 0.02))
                else:
                    actions.append(0.5)

        actions = np.clip(actions, 0, 1)
        lam = np.mean(actions)

        # PGG dynamics
        contributions = actions * env.E
        pool = np.sum(contributions) * env.M
        share = pool / N_AGENTS
        rewards = share - contributions + (1 - actions) * env.E
        total_reward += np.mean(rewards)

        # Resource dynamics
        env.R += 0.05 * (lam - 0.5)

        # Recovery
        if env.R < R_crit:
            env.R += 0.01 * env.R
            crisis_steps += 1
        else:
            env.R += 0.05 * (1.0 - env.R)

        # Stochastic shock
        if rng.random() < p_shock:
            env.R -= delta_shock

        min_R = min(min_R, env.R)

        if env.R <= 0:
            survived = False
            break

    return {
        "reward": total_reward / T,
        "survived": survived,
        "min_R": min_R,
        "crisis_steps": crisis_steps,
        "steps": t + 1,
    }


def main():
    print("=" * 60)
    print("  SITUATIONAL STRESS-TEST")
    print(f"  Seeds={N_SEEDS}, Episodes={N_EPISODES}")
    print("=" * 60)

    results = {}

    for cond_name, cond in STRESS_CONDITIONS.items():
        print(f"\n  [{cond_name}] {cond['desc']}")
        cond_results = {}

        for policy_name in POLICIES:
            survivals = []
            rewards = []
            min_Rs = []
            crisis_counts = []

            for seed in range(N_SEEDS):
                ep_results = []
                for ep in range(N_EPISODES):
                    r = run_episode(cond, policy_name, seed * 1000 + ep)
                    ep_results.append(r)

                seed_surv = np.mean([r["survived"] for r in ep_results]) * 100
                seed_reward = np.mean([r["reward"] for r in ep_results])
                seed_min_R = np.mean([r["min_R"] for r in ep_results])
                seed_crisis = np.mean([r["crisis_steps"] for r in ep_results])

                survivals.append(seed_surv)
                rewards.append(seed_reward)
                min_Rs.append(seed_min_R)
                crisis_counts.append(seed_crisis)

            surv_mean = np.mean(survivals)
            surv_std = np.std(survivals)
            tail_5 = np.percentile(survivals, 5)

            cond_results[policy_name] = {
                "survival_mean": round(surv_mean, 1),
                "survival_std": round(surv_std, 1),
                "survival_tail5": round(tail_5, 1),
                "reward_mean": round(np.mean(rewards), 2),
                "min_R_mean": round(np.mean(min_Rs), 3),
                "crisis_steps_mean": round(np.mean(crisis_counts), 1),
            }

            print(f"    {policy_name:18s}: Surv={surv_mean:.1f}% (tail5={tail_5:.1f}%), "
                  f"Crisis={np.mean(crisis_counts):.1f} steps")

        results[cond_name] = {
            "description": cond["desc"],
            "params": {k: v for k, v in cond.items() if k != "desc"},
            "policies": cond_results,
        }

    # Save
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "stress_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stress_test_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved: {out_path}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  STRESS-TEST SUMMARY TABLE (for paper)")
    print(f"{'='*70}")
    print(f"  {'Condition':25s} | {'Selfish':>10s} | {'Situational':>12s} | {'Unconditional':>14s}")
    print(f"  {'-'*25}-+-{'-'*10}-+-{'-'*12}-+-{'-'*14}")
    for cond_name, cond_data in results.items():
        sel = cond_data["policies"]["selfish_rl"]["survival_mean"]
        sit = cond_data["policies"]["situational"]["survival_mean"]
        unc = cond_data["policies"]["unconditional"]["survival_mean"]
        print(f"  {cond_name:25s} | {sel:>9.1f}% | {sit:>11.1f}% | {unc:>13.1f}%")

    print(f"\n  Tail-risk (5th percentile survival):")
    for cond_name, cond_data in results.items():
        sit_tail = cond_data["policies"]["situational"]["survival_tail5"]
        unc_tail = cond_data["policies"]["unconditional"]["survival_tail5"]
        print(f"  {cond_name:25s} | Sit tail5={sit_tail:.1f}% | Unc tail5={unc_tail:.1f}%")


if __name__ == "__main__":
    main()
