"""
Phase B (Table 5): Crisis Commitment (phi_1) Sweep
===================================================
Tests phi_1 in {0.00, 0.21, 0.50, 1.00} under Byz=0% and Byz=30%.
Outputs Welfare and Survival (with ± std/CI).
"""
import numpy as np
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
from cleanrl_mappo_pgg import NonlinearPGGEnv

OUTPUT_DIR = Path(__file__).resolve().parent.parent / os.environ.get("ETHICAAI_OUTDIR", "outputs") / "phi1_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_EPISODES = 150
N_EVAL = 30
N_SEEDS = 20
T = 50
TOTAL_AGENTS = 20

def compute_lambda(R_t, phi_1, prev_lambda):
    """Compute commitment λ based on resource state."""
    alpha = 0.9
    crisis_threshold = 0.15
    abundance_threshold = 0.25
    normal_slope = 0.5
    abundance_bonus = 0.3
    restraint_beta = 0.1
    
    if R_t < crisis_threshold:
        target = phi_1
    elif R_t >= abundance_threshold:
        target = min(1.0, normal_slope + abundance_bonus)
    else:
        target = normal_slope
        
    target = max(0.0, target - restraint_beta * (1 - R_t))
    target = np.clip(target, 0.0, 1.0)
    
    new_lambda = alpha * prev_lambda + (1 - alpha) * target
    return np.clip(new_lambda, 0.0, 1.0)

def run_phi1(seed, phi_1, byz_frac):
    rng = np.random.RandomState(seed)
    env = NonlinearPGGEnv(byz_frac=byz_frac)
    n_honest = int(TOTAL_AGENTS * (1 - byz_frac))
    
    episodes = []
    
    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=seed*10000+ep)
        agent_lambdas = np.full(n_honest, 0.5)
        
        for t in range(T):
            R_t = obs[0]  # obs = [R, prev_mean_lambda, crisis_flag, t/T]
            
            for i in range(n_honest):
                agent_lambdas[i] = compute_lambda(R_t, phi_1, agent_lambdas[i])
            
            obs, rewards, terminated, truncated, info = env.step(agent_lambdas)
            if terminated:
                break
        
        episodes.append({
            "survived": info.get("survived", False),
            "welfare": info.get("welfare", 0.0),
        })
    
    # 20 evals
    ev = episodes[-20:]
    return {
        "survival": float(np.mean([float(d["survived"]) for d in ev]) * 100),
        "welfare": float(np.mean([d["welfare"] for d in ev])),
    }

def main():
    phi1_values = [0.00, 0.21, 0.50, 1.00]
    byz_fracs = [0.0, 0.3]
    
    print("=" * 70)
    print("  Phase B: Crisis Commitment (phi_1) Sweep (Table 5)")
    print("=" * 70)
    
    results = {}
    t0 = time.time()
    
    for phi_1 in phi1_values:
        results[str(phi_1)] = {}
        row_str = f"phi_1={phi_1:.2f} | "
        
        for bf in byz_fracs:
            seeds_data = []
            
            for s in range(N_SEEDS):
                r = run_phi1(seed=s*7+42, phi_1=phi_1, byz_frac=bf)
                seeds_data.append(r)
            
            survs = [r["survival"] for r in seeds_data]
            wels = [r["welfare"] for r in seeds_data]
            
            res = {
                "survival_mean": float(np.mean(survs)),
                "survival_std": float(np.std(survs)),
                "welfare_mean": float(np.mean(wels)),
                "welfare_std": float(np.std(wels)),
            }
            results[str(phi_1)][f"byz_{int(bf*100)}"] = res
            
            row_str += f"W({bf*100:.0f}%)={res['welfare_mean']:.1f}±{res['welfare_std']:.1f}  Alive={res['survival_mean']:.0f}±{res['survival_std']:.0f}% | "
        
        print(f"    -> {row_str}")

    out_path = OUTPUT_DIR / "phi1_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"  Took: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
