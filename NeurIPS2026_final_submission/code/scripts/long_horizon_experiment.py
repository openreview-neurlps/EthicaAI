#!/usr/bin/env python
"""
long_horizon_experiment.py — Nash Trap Persistence Under Extended Horizons
==========================================================================
Tests whether the Nash Trap persists under T=200 (4x the default T=50).
Addresses reviewer concern: "T=50 is short. Does the Trap persist?"

Usage:
  python long_horizon_experiment.py
  ETHICAAI_FAST=1 python long_horizon_experiment.py
"""
import numpy as np
import json
import time
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
OUTPUT_DIR = PROJECT_ROOT / os.environ.get("ETHICAAI_OUTDIR", "outputs") / "long_horizon"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from envs.nonlinear_pgg_env import NonlinearPGGEnv
from cleanrl_mappo_pgg import MLPActor, MLPCritic, compute_gae, ppo_update_actor, ppo_update_critic, bootstrap_ci

FAST = os.environ.get("ETHICAAI_FAST") == "1"
N_EPISODES = 500 if not FAST else 50
N_EVAL = 30 if not FAST else 10
N_SEEDS = 10 if not FAST else 3
HORIZONS = [50, 100, 200]


def run_ippo_horizon(seed, T):
    """Run IPPO with variable time horizon."""
    rng = np.random.RandomState(seed)
    env = NonlinearPGGEnv(t_horizon=T)
    n = env.n_honest

    actors = [MLPActor(np.random.RandomState(seed * 100 + i)) for i in range(n)]
    critics = [MLPCritic(np.random.RandomState(seed * 100 + i), obs_dim=4) for i in range(n)]

    episodes = []
    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        obs_buf = [[] for _ in range(n)]
        act_buf = [[] for _ in range(n)]
        lp_buf = [[] for _ in range(n)]
        rew_buf = [[] for _ in range(n)]
        val_buf = [[] for _ in range(n)]

        for t in range(T):
            actions = np.zeros(n)
            for i in range(n):
                a, lp, mu = actors[i].act(obs, rng)
                actions[i] = a
                obs_buf[i].append(obs.copy())
                act_buf[i].append(a)
                lp_buf[i].append(lp)
                val_buf[i].append(critics[i].forward(obs))

            obs, rewards, done, _, info = env.step(actions)
            for i in range(n):
                rew_buf[i].append(rewards[i])
            if done:
                break

        for i in range(n):
            if len(rew_buf[i]) < 2:
                continue
            adv, ret = compute_gae(rew_buf[i], val_buf[i])
            if np.std(adv) > 1e-8:
                adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
            for _ in range(4):
                ppo_update_actor(actors[i], obs_buf[i], act_buf[i], lp_buf[i], adv)
                ppo_update_critic(critics[i], obs_buf[i], ret)

        episodes.append({
            "mean_lambda": info.get("mean_lambda", 0),
            "survived": info.get("survived", False),
            "welfare": info.get("welfare", 0),
        })

    ev = episodes[-N_EVAL:]
    return {
        "lambda": float(np.mean([d["mean_lambda"] for d in ev])),
        "survival": float(np.mean([float(d["survived"]) for d in ev]) * 100),
        "welfare": float(np.mean([d["welfare"] for d in ev])),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  Long Horizon Experiment: Nash Trap Persistence")
    print(f"  Horizons: {HORIZONS}, Seeds={N_SEEDS}")
    print("=" * 60)

    t0 = time.time()
    results = {}

    for T in HORIZONS:
        print(f"\n  T={T}, {N_SEEDS} seeds...")
        seed_data = []
        for s in range(N_SEEDS):
            r = run_ippo_horizon(s * 7 + 42, T)
            seed_data.append(r)
            if (s + 1) % 3 == 0 or s == 0:
                print(f"    Seed {s+1}: λ={r['lambda']:.3f}, surv={r['survival']:.1f}%")

        lams = [r["lambda"] for r in seed_data]
        survs = [r["survival"] for r in seed_data]
        ci_lam = bootstrap_ci(lams)

        results[f"T={T}"] = {
            "horizon": T,
            "lambda_mean": float(np.mean(lams)),
            "lambda_std": float(np.std(lams)),
            "lambda_ci95": ci_lam,
            "survival_mean": float(np.mean(survs)),
            "survival_std": float(np.std(survs)),
            "nash_trap": float(np.mean(lams)) < 0.95,
        }
        trapped = "TRAPPED" if np.mean(lams) < 0.95 else "ESCAPED"
        print(f"  T={T}: λ={np.mean(lams):.3f} [{ci_lam[0]:.3f}, {ci_lam[1]:.3f}], "
              f"surv={np.mean(survs):.1f}%, {trapped}")

    total = time.time() - t0

    print(f"\n{'='*60}")
    print("  HORIZON INVARIANCE TABLE")
    print(f"{'='*60}")
    for k, v in results.items():
        trapped = "YES" if v["nash_trap"] else "NO"
        print(f"  {k:>6} | λ={v['lambda_mean']:.3f} ± {v['lambda_std']:.3f} | "
              f"surv={v['survival_mean']:.1f}% | Trap: {trapped}")

    output = {
        "experiment": "Long Horizon Nash Trap Persistence",
        "results": results,
        "time_seconds": round(total, 1),
    }
    out_path = OUTPUT_DIR / "long_horizon_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"  Total: {total:.0f}s")
