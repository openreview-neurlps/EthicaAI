"""
Phase A: HP Tuning Sweep for IPPO on Non-linear PGG
====================================================
Tests lr × entropy_coef grid to show Nash Trap persists 
even with reasonable hyperparameter tuning.

Grid: lr ∈ {1e-4, 2.5e-4, 5e-4, 1e-3} × entropy ∈ {0.0, 0.01, 0.05}
= 12 combinations × 5 seeds = 60 runs
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
    ppo_update_actor, bootstrap_ci,
    GAMMA, GAE_LAMBDA, CLIP_EPS, HIDDEN_DIM
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "cleanrl_baselines"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_EPISODES = 150
N_EVAL = 20
N_SEEDS = 3

# HP Grid
LR_GRID = [1e-4, 2.5e-4, 5e-4, 1e-3]
ENTROPY_GRID = [0.0, 0.01, 0.05]


def run_ippo_hp(seed, lr, entropy_coef):
    """Run IPPO with specified HP."""
    rng = np.random.RandomState(seed)
    env = NonlinearPGGEnv()
    n = env.n_honest
    
    actors = [MLPActor(rng, lr=lr) for _ in range(n)]
    critics = [MLPCritic(rng, lr=lr) for _ in range(n)]
    
    episodes = []
    t0 = time.time()
    
    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=seed*10000+ep)
        obs_buf = [[] for _ in range(n)]
        act_buf = [[] for _ in range(n)]
        lp_buf = [[] for _ in range(n)]
        rew_buf = [[] for _ in range(n)]
        val_buf = [[] for _ in range(n)]
        
        for t in range(env.T):
            actions = np.zeros(n)
            for i in range(n):
                a, lp, mu = actors[i].act(obs, rng)
                actions[i] = a
                obs_buf[i].append(obs.copy())
                act_buf[i].append(a)
                lp_buf[i].append(lp)
                val_buf[i].append(critics[i].forward(obs))
            
            obs, rewards, terminated, truncated, info = env.step(actions)
            for i in range(n):
                rew_buf[i].append(rewards[i])
            if terminated:
                break
        
        for i in range(n):
            if len(rew_buf[i]) < 2:
                continue
            advantages, returns = compute_gae(rew_buf[i], val_buf[i])
            if np.std(advantages) > 1e-8:
                advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
            ppo_update_actor(actors[i], obs_buf[i], act_buf[i], lp_buf[i], advantages)
        
        episodes.append({
            "mean_lambda": info.get("mean_lambda", 0),
            "survived": info.get("survived", False),
        })
    
    wc = time.time() - t0
    ev = episodes[-N_EVAL:]
    return {
        "mean_lambda": float(np.mean([d["mean_lambda"] for d in ev])),
        "survival": float(np.mean([float(d["survived"]) for d in ev]) * 100),
        "wall_clock": wc,
    }


def main():
    print("=" * 70)
    print("  Phase A: HP Tuning Sweep (IPPO)")
    print("  Grid: %d lr × %d entropy = %d combos × %d seeds" % (
        len(LR_GRID), len(ENTROPY_GRID), len(LR_GRID)*len(ENTROPY_GRID), N_SEEDS))
    print("=" * 70)
    
    results = {}
    t_total = time.time()
    
    for lr in LR_GRID:
        for ent in ENTROPY_GRID:
            key = f"lr={lr:.0e}_ent={ent}"
            print(f"\n  [{key}] Running {N_SEEDS} seeds...")
            
            seed_results = []
            for s in range(N_SEEDS):
                r = run_ippo_hp(seed=s*7+42, lr=lr, entropy_coef=ent)
                seed_results.append(r)
                print(f"    Seed {s}: λ={r['mean_lambda']:.3f}, surv={r['survival']:.1f}%")
            
            lams = [r["mean_lambda"] for r in seed_results]
            survs = [r["survival"] for r in seed_results]
            ci_l = bootstrap_ci(lams)
            
            results[key] = {
                "lr": lr, "entropy_coef": ent,
                "lambda_mean": float(np.mean(lams)),
                "lambda_std": float(np.std(lams)),
                "lambda_ci95": [float(ci_l[0]), float(ci_l[1])],
                "survival_mean": float(np.mean(survs)),
                "survival_std": float(np.std(survs)),
                "still_trapped": float(np.mean(lams)) < 0.7,  # Nash Trap threshold
            }
            
            trapped = "TRAPPED ✓" if results[key]["still_trapped"] else "ESCAPED ✗"
            print(f"    → λ={np.mean(lams):.3f} [{ci_l[0]:.3f},{ci_l[1]:.3f}], "
                  f"surv={np.mean(survs):.1f}% — {trapped}")
    
    elapsed = time.time() - t_total
    
    # Save
    out_path = OUTPUT_DIR / "hp_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("  HP SWEEP SUMMARY")
    print("=" * 70)
    print(f"  {'lr':>8s} {'ent':>6s} {'λ mean':>8s} {'CI':>18s} {'surv%':>7s} {'trap':>8s}")
    print("  " + "-" * 60)
    
    all_trapped = True
    for k, r in results.items():
        trapped = "YES" if r["still_trapped"] else "NO"
        if not r["still_trapped"]:
            all_trapped = False
        print(f"  {r['lr']:>8.0e} {r['entropy_coef']:>6.2f} "
              f"{r['lambda_mean']:>8.3f} [{r['lambda_ci95'][0]:.3f},{r['lambda_ci95'][1]:.3f}] "
              f"{r['survival_mean']:>6.1f}% {trapped:>8s}")
    
    print(f"\n  All trapped: {all_trapped}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("  DONE!")


if __name__ == "__main__":
    main()
