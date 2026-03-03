"""
Phase B: Commitment Function Ablation
======================================
Tests 5 variants of g(θ,R) to show the Moral Commitment Spectrum
is not dependent on any single component.

Ablations:
1. no_crisis:    Remove crisis zone (always use normal g)
2. no_abundance: Remove abundance zone bonus
3. alpha_zero:   No smoothing (instant response)
4. alpha_one:    Maximum smoothing (no adaptation)
5. no_restraint: Remove restraint cost (β=0)
"""
import numpy as np
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
from cleanrl_mappo_pgg import NonlinearPGGEnv, bootstrap_ci

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_EPISODES = 200
N_EVAL = 30
N_SEEDS = 5
N_AGENTS = 14  # honest agents (20 total - 30% byz)
T = 50

# Default commitment parameters
DEFAULT_PARAMS = {
    "alpha": 0.9,         # Smoothing
    "crisis_threshold": 0.15,
    "abundance_threshold": 0.25,
    "crisis_lambda": 1.0,  # Unconditional in crisis
    "normal_slope": 0.5,
    "abundance_bonus": 0.3,
    "restraint_beta": 0.1,
}


def compute_lambda(R_t, params, prev_lambda):
    """Compute commitment λ based on resource state."""
    alpha = params["alpha"]
    
    if R_t < params["crisis_threshold"]:
        target = params["crisis_lambda"]
    elif R_t >= params["abundance_threshold"]:
        target = min(1.0, params["normal_slope"] + params["abundance_bonus"])
    else:
        # Normal zone
        target = params["normal_slope"]
    
    # Apply restraint cost
    target = max(0.0, target - params["restraint_beta"] * (1 - R_t))
    target = np.clip(target, 0.0, 1.0)
    
    # Smoothing
    new_lambda = alpha * prev_lambda + (1 - alpha) * target
    return np.clip(new_lambda, 0.0, 1.0)


def make_ablation_params(variant):
    """Return modified params for each ablation."""
    p = DEFAULT_PARAMS.copy()
    
    if variant == "no_crisis":
        p["crisis_threshold"] = -1.0  # Never triggers
    elif variant == "no_abundance":
        p["abundance_bonus"] = 0.0
    elif variant == "alpha_zero":
        p["alpha"] = 0.0
    elif variant == "alpha_one":
        p["alpha"] = 1.0  # Lambda never changes
    elif variant == "no_restraint":
        p["restraint_beta"] = 0.0
    elif variant == "full":
        pass  # default
    
    return p


def run_ablation(seed, variant):
    """Run one seed with given ablation variant."""
    rng = np.random.RandomState(seed)
    env = NonlinearPGGEnv()
    params = make_ablation_params(variant)
    
    episodes = []
    
    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=seed*10000+ep)
        agent_lambdas = np.full(N_AGENTS, 0.5)  # Initial
        
        for t in range(T):
            R_t = obs[2] if len(obs) > 2 else 0.5  # Resource level from obs
            
            # Update each agent's lambda
            for i in range(N_AGENTS):
                agent_lambdas[i] = compute_lambda(R_t, params, agent_lambdas[i])
            
            obs, rewards, terminated, truncated, info = env.step(agent_lambdas)
            if terminated:
                break
        
        episodes.append({
            "mean_lambda": float(np.mean(agent_lambdas)),
            "survived": info.get("survived", False),
            "welfare": info.get("welfare", 0),
        })
    
    ev = episodes[-N_EVAL:]
    return {
        "mean_lambda": float(np.mean([d["mean_lambda"] for d in ev])),
        "survival": float(np.mean([float(d["survived"]) for d in ev]) * 100),
        "welfare": float(np.mean([d["welfare"] for d in ev])),
    }


def main():
    variants = ["full", "no_crisis", "no_abundance", "alpha_zero", "alpha_one", "no_restraint"]
    labels = {
        "full": "Full g(θ,R)",
        "no_crisis": "No crisis zone",
        "no_abundance": "No abundance bonus",
        "alpha_zero": "No smoothing (α=0)",
        "alpha_one": "No adaptation (α=1)",
        "no_restraint": "No restraint cost (β=0)",
    }
    
    print("=" * 70)
    print("  Phase B: Commitment Function Ablation")
    print("  %d variants × %d seeds" % (len(variants), N_SEEDS))
    print("=" * 70)
    
    results = {}
    t0 = time.time()
    
    for var in variants:
        print(f"\n  [{labels[var]}] Running {N_SEEDS} seeds...")
        seeds_data = []
        
        for s in range(N_SEEDS):
            r = run_ablation(seed=s*7+42, variant=var)
            seeds_data.append(r)
            print(f"    Seed {s}: λ={r['mean_lambda']:.3f}, surv={r['survival']:.1f}%, W={r['welfare']:.1f}")
        
        lams = [r["mean_lambda"] for r in seeds_data]
        survs = [r["survival"] for r in seeds_data]
        wels = [r["welfare"] for r in seeds_data]
        ci_s = bootstrap_ci(survs)
        
        results[var] = {
            "label": labels[var],
            "lambda_mean": float(np.mean(lams)),
            "lambda_std": float(np.std(lams)),
            "survival_mean": float(np.mean(survs)),
            "survival_ci95": [float(ci_s[0]), float(ci_s[1])],
            "welfare_mean": float(np.mean(wels)),
        }
        
        print(f"    → λ={np.mean(lams):.3f}, surv={np.mean(survs):.1f}% [{ci_s[0]:.1f},{ci_s[1]:.1f}]")
    
    elapsed = time.time() - t0
    
    out_path = OUTPUT_DIR / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")
    
    print("\n" + "=" * 70)
    print("  ABLATION SUMMARY")
    print("=" * 70)
    print(f"  {'Variant':>25s} {'λ':>7s} {'Surv%':>7s} {'Welfare':>8s}")
    print("  " + "-" * 50)
    for var in variants:
        r = results[var]
        print(f"  {r['label']:>25s} {r['lambda_mean']:>7.3f} {r['survival_mean']:>6.1f}% {r['welfare_mean']:>8.1f}")
    
    print(f"\n  Total time: {elapsed:.0f}s")
    print("  DONE!")


if __name__ == "__main__":
    main()
