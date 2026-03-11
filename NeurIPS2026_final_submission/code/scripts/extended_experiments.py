"""
Phase 3: Extended Experiments for Strong Accept
================================================
Additional experiments to strengthen claims:
  (A) N=500 scale test (non-linear PGG)
  (B) No-shock Nash Trap verification
  (C) M/N multiplier sweep (Theorem 1 generality)
"""
import numpy as np
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
from envs.nonlinear_pgg_env import NonlinearPGGEnv
from cleanrl_mappo_pgg import (
    MLPActor, MLPCritic, compute_gae, ppo_update_actor, ppo_update_critic,
    bootstrap_ci, GAMMA, GAE_LAMBDA, relu
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "extended"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_EPISODES = 150
N_EVAL = 30
T_HORIZON = 50

FAST = os.environ.get("ETHICAAI_FAST") == "1"
if FAST:
    print("  [FAST MODE]")
    N_EPISODES = 15
    N_EVAL = 5


def run_ippo(seed, n_agents=20, byz_frac=0.3, multiplier=None, shock_prob=None, shock_mag=None):
    """Run IPPO on a configurable PGG environment."""
    rng = np.random.RandomState(seed)
    
    env_kwargs = {"n_agents": n_agents, "byz_frac": byz_frac}
    if multiplier is not None:
        env_kwargs["multiplier"] = multiplier
    if shock_prob is not None:
        env_kwargs["shock_prob"] = shock_prob
    if shock_mag is not None:
        env_kwargs["shock_mag"] = shock_mag
    
    env = NonlinearPGGEnv(**env_kwargs)
    n_honest = env.n_honest
    
    actors = [MLPActor(np.random.RandomState(seed * 100 + i)) for i in range(n_honest)]
    critics = [MLPCritic(np.random.RandomState(seed * 100 + i)) for i in range(n_honest)]
    
    episodes = []
    
    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        obs_buf = [[] for _ in range(n_honest)]
        act_buf = [[] for _ in range(n_honest)]
        lp_buf = [[] for _ in range(n_honest)]
        rew_buf = [[] for _ in range(n_honest)]
        val_buf = [[] for _ in range(n_honest)]
        
        for t in range(T_HORIZON):
            actions = np.zeros(n_honest)
            for i in range(n_honest):
                a, lp, mu = actors[i].act(obs, rng)
                actions[i] = a
                obs_buf[i].append(obs.copy())
                act_buf[i].append(a)
                lp_buf[i].append(lp)
                val_buf[i].append(critics[i].forward(obs))
            
            obs, rewards, done, _, info = env.step(actions)
            for i in range(n_honest):
                rew_buf[i].append(rewards[i])
            if done:
                break
        
        for i in range(n_honest):
            if len(rew_buf[i]) < 2:
                continue
            adv, ret = compute_gae(rew_buf[i], val_buf[i])
            if np.std(adv) > 1e-8:
                adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
            for _ in range(4):  # PPO_EPOCHS
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


def experiment_scale(seeds_per=5):
    """Experiment A: Scale test N=50, 100, 200, 500."""
    print("\n" + "=" * 60)
    print("  Experiment A: Scale Test")
    print("=" * 60)
    
    scales = [50, 100, 200, 500] if not FAST else [50, 100]
    results = {}
    
    for N in scales:
        print(f"\n  N={N}, {seeds_per} seeds...")
        seed_data = []
        for s in range(seeds_per):
            r = run_ippo(s * 7 + 42, n_agents=N)
            seed_data.append(r)
            sys.stdout.write(".")
            sys.stdout.flush()
        
        lams = [r["lambda"] for r in seed_data]
        survs = [r["survival"] for r in seed_data]
        
        results[f"N={N}"] = {
            "n_agents": N,
            "lambda_mean": float(np.mean(lams)),
            "lambda_std": float(np.std(lams)),
            "survival_mean": float(np.mean(survs)),
            "survival_std": float(np.std(survs)),
            "welfare_mean": float(np.mean([r["welfare"] for r in seed_data])),
        }
        print(f"\n    N={N}: λ={np.mean(lams):.3f}±{np.std(lams):.3f}, Surv={np.mean(survs):.0f}%")
    
    return results


def experiment_no_shock(seeds_per=5):
    """Experiment B: Nash Trap without shocks."""
    print("\n" + "=" * 60)
    print("  Experiment B: No-Shock Nash Trap")
    print("=" * 60)
    
    conditions = [
        ("shock", 0.15, 0.20),
        ("no_shock", 0.0, 0.0),
    ]
    results = {}
    
    for label, sp, sm in conditions:
        print(f"\n  [{label}] {seeds_per} seeds...")
        seed_data = []
        for s in range(seeds_per):
            r = run_ippo(s * 7 + 42, shock_prob=sp, shock_mag=sm)
            seed_data.append(r)
            sys.stdout.write(".")
            sys.stdout.flush()
        
        lams = [r["lambda"] for r in seed_data]
        survs = [r["survival"] for r in seed_data]
        
        results[label] = {
            "shock_prob": sp,
            "shock_mag": sm,
            "lambda_mean": float(np.mean(lams)),
            "lambda_std": float(np.std(lams)),
            "survival_mean": float(np.mean(survs)),
            "survival_std": float(np.std(survs)),
        }
        print(f"\n    {label}: λ={np.mean(lams):.3f}, Surv={np.mean(survs):.0f}%")
    
    return results


def experiment_multiplier_sweep(seeds_per=5):
    """Experiment C: M/N multiplier sweep."""
    print("\n" + "=" * 60)
    print("  Experiment C: Multiplier Sweep")
    print("=" * 60)
    
    multipliers = [1.2, 1.5, 1.8, 2.0, 2.5, 3.0] if not FAST else [1.5, 2.0, 3.0]
    results = {}
    
    for M in multipliers:
        print(f"\n  M={M}, {seeds_per} seeds...")
        seed_data = []
        for s in range(seeds_per):
            r = run_ippo(s * 7 + 42, multiplier=M)
            seed_data.append(r)
            sys.stdout.write(".")
            sys.stdout.flush()
        
        lams = [r["lambda"] for r in seed_data]
        survs = [r["survival"] for r in seed_data]
        
        results[f"M={M}"] = {
            "multiplier": M,
            "lambda_mean": float(np.mean(lams)),
            "lambda_std": float(np.std(lams)),
            "survival_mean": float(np.mean(survs)),
            "survival_std": float(np.std(survs)),
            "welfare_mean": float(np.mean([r["welfare"] for r in seed_data])),
        }
        print(f"\n    M={M}: λ={np.mean(lams):.3f}, Surv={np.mean(survs):.0f}%")
    
    return results


def main():
    print("=" * 60)
    print("  Phase 3: Extended Experiments for Strong Accept")
    print("=" * 60)
    
    t0 = time.time()
    n_seeds = 3 if FAST else 10
    
    all_results = {}
    
    # A: Scale
    all_results["scale"] = experiment_scale(seeds_per=n_seeds)
    
    # B: No-shock
    all_results["no_shock"] = experiment_no_shock(seeds_per=n_seeds)
    
    # C: Multiplier sweep
    all_results["multiplier"] = experiment_multiplier_sweep(seeds_per=n_seeds)
    
    # Save
    out_path = OUTPUT_DIR / "extended_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  ALL EXTENDED EXPERIMENTS COMPLETE in {elapsed:.0f}s")
    print(f"  Saved: {out_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
