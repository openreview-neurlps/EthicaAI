"""
Situational Stress-Test v2 — Uses EXACT NonlinearPGGEnv.step()
==============================================================
Demonstrates worst-case brittleness of situational commitment using
the identical environment from the paper (NonlinearPGGEnv).

Conditions:
  A: Baseline severe (default params)
  B: Near-tipping start (R0 = r_crit + 0.02)
  C: Long horizon (T = 200)
  D: Shock burst (p_shock = 0.15, shock_mag = 0.20)
  E: Combined worst (near-tipping + long horizon + burst shocks)
"""
import numpy as np
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from envs.nonlinear_pgg_env import NonlinearPGGEnv

N_SEEDS = int(os.environ.get("STRESS_SEEDS", "20"))
N_EPISODES = int(os.environ.get("STRESS_EPISODES", "200"))
N_AGENTS = int(os.environ.get("STRESS_N", "20"))
BYZ_FRAC = 0.30
R_CRIT = 0.15


def situational_action(R, r_crit=R_CRIT, theta=0.7):
    """Situational commitment: reduces lambda when R < r_crit."""
    if R < r_crit:
        return max(0.0, np.sin(theta) * 0.3)
    else:
        return 0.7 + 0.3 * np.sin(theta)


STRESS_CONDITIONS = {
    "baseline_severe": {
        "desc": "Severe TPSD (default params, paper Sec 4.2)",
        "R_override": None,
        "t_horizon": 50,
        "shock_prob": 0.05,
        "shock_mag": 0.15,
    },
    "near_tipping_start": {
        "desc": "R0 near tipping point (R0 = r_crit + 0.02 = 0.17)",
        "R_override": R_CRIT + 0.02,
        "t_horizon": 50,
        "shock_prob": 0.05,
        "shock_mag": 0.15,
    },
    "long_horizon": {
        "desc": "Extended horizon T=200",
        "R_override": None,
        "t_horizon": 200,
        "shock_prob": 0.05,
        "shock_mag": 0.15,
    },
    "shock_burst": {
        "desc": "Extreme shocks (p=0.15, delta=0.20)",
        "R_override": None,
        "t_horizon": 50,
        "shock_prob": 0.15,
        "shock_mag": 0.20,
    },
    "combined_worst": {
        "desc": "Near-tipping + long horizon + burst shocks",
        "R_override": R_CRIT + 0.02,
        "t_horizon": 200,
        "shock_prob": 0.15,
        "shock_mag": 0.20,
    },
}

POLICIES = {
    "selfish_rl": "Nash Trap baseline (lambda~0.5)",
    "situational": "Situational commitment g(theta, R)",
    "unconditional": "Unconditional floor (phi1=1.0)",
}


def make_actions(policy_name, R, n_honest, rng):
    """Generate honest-agent actions for a given policy."""
    if policy_name == "selfish_rl":
        return np.clip(0.5 + rng.normal(0, 0.02, size=n_honest), 0, 1)
    elif policy_name == "situational":
        lam = situational_action(R)
        return np.full(n_honest, lam)
    elif policy_name == "unconditional":
        return np.ones(n_honest)
    else:
        return np.full(n_honest, 0.5)


def run_episode(condition, policy_name, seed):
    """Run a single episode using NonlinearPGGEnv.step()."""
    env = NonlinearPGGEnv(
        n_agents=N_AGENTS,
        t_horizon=condition["t_horizon"],
        shock_prob=condition["shock_prob"],
        shock_mag=condition["shock_mag"],
        byz_frac=BYZ_FRAC,
    )
    obs, _ = env.reset(seed=seed)

    # Override R0 if needed
    if condition["R_override"] is not None:
        env.R = condition["R_override"]

    rng = np.random.RandomState(seed)
    total_reward = 0.0
    min_R = env.R
    crisis_steps = 0
    survived = True

    for t in range(condition["t_horizon"]):
        actions = make_actions(policy_name, env.R, env.n_honest, rng)
        obs, rewards, terminated, truncated, info = env.step(actions)

        total_reward += float(np.mean(rewards))
        min_R = min(min_R, info["resource"])
        if info["resource"] < R_CRIT:
            crisis_steps += 1

        if terminated:
            survived = info["survived"]
            break

    return {
        "reward": total_reward / max(t + 1, 1),
        "survived": survived,
        "min_R": min_R,
        "crisis_steps": crisis_steps,
        "steps": t + 1,
        "final_R": info["resource"],
        "mean_lambda": info.get("mean_lambda", 0),
    }


def wilson_ci(successes, total, z=1.96):
    """Wilson score interval for binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return max(0, center - margin), min(1, center + margin)


def main():
    start = time.time()
    print("=" * 65)
    print(f"  SITUATIONAL STRESS-TEST v2 (env.step-based)")
    print(f"  N={N_AGENTS}, Byz={BYZ_FRAC:.0%}, Seeds={N_SEEDS}, Episodes={N_EPISODES}")
    print("=" * 65)

    results = {}

    for cond_name, cond in STRESS_CONDITIONS.items():
        print(f"\n  [{cond_name}] {cond['desc']}")
        cond_results = {}

        for policy_name, policy_desc in POLICIES.items():
            survivals = []
            rewards = []
            min_Rs = []
            crisis_counts = []

            for seed in range(N_SEEDS):
                ep_results = []
                for ep in range(N_EPISODES):
                    r = run_episode(cond, policy_name, seed * 10000 + ep)
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
            surv_lo, surv_hi = wilson_ci(
                sum(1 for s in survivals if s >= 50), len(survivals)
            )

            cond_results[policy_name] = {
                "description": policy_desc,
                "survival_mean": round(surv_mean, 1),
                "survival_std": round(surv_std, 1),
                "survival_tail5": round(tail_5, 1),
                "survival_ci95": [round(surv_lo * 100, 1), round(surv_hi * 100, 1)],
                "reward_mean": round(np.mean(rewards), 2),
                "min_R_mean": round(np.mean(min_Rs), 4),
                "crisis_steps_mean": round(np.mean(crisis_counts), 1),
            }

            print(
                f"    {policy_name:18s}: Surv={surv_mean:5.1f}% "
                f"(tail5={tail_5:5.1f}%), Crisis={np.mean(crisis_counts):4.1f} steps"
            )

        results[cond_name] = {
            "description": cond["desc"],
            "params": {
                "n_agents": N_AGENTS,
                "byz_frac": BYZ_FRAC,
                "t_horizon": cond["t_horizon"],
                "shock_prob": cond["shock_prob"],
                "shock_mag": cond["shock_mag"],
                "R_override": cond["R_override"],
                "r_crit": R_CRIT,
                "env_class": "NonlinearPGGEnv",
                "env_step_based": True,
            },
            "policies": cond_results,
        }

    elapsed = time.time() - start

    # Save
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "stress_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stress_test_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"  Elapsed: {elapsed:.0f}s")

    # Summary table
    print(f"\n{'='*72}")
    print(f"  STRESS-TEST SUMMARY (N={N_AGENTS}, {N_SEEDS} seeds x {N_EPISODES} episodes)")
    print(f"{'='*72}")
    print(f"  {'Condition':25s} | {'Selfish':>8s} | {'Situational':>12s} | {'Unconditional':>14s}")
    print(f"  {'-'*25}-+-{'-'*8}-+-{'-'*12}-+-{'-'*14}")
    for cond_name, cond_data in results.items():
        sel = cond_data["policies"]["selfish_rl"]["survival_mean"]
        sit = cond_data["policies"]["situational"]["survival_mean"]
        unc = cond_data["policies"]["unconditional"]["survival_mean"]
        print(f"  {cond_name:25s} | {sel:>7.1f}% | {sit:>11.1f}% | {unc:>13.1f}%")

    print(f"\n  Tail-risk (5th percentile survival):")
    for cond_name, cond_data in results.items():
        sit_tail = cond_data["policies"]["situational"]["survival_tail5"]
        unc_tail = cond_data["policies"]["unconditional"]["survival_tail5"]
        print(f"  {cond_name:25s} | Sit tail5={sit_tail:5.1f}% | Unc tail5={unc_tail:5.1f}%")


if __name__ == "__main__":
    main()
