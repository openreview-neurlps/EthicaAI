"""
Full Obs baseline for Table 8 — generates the missing JSON source.
Runs the SAME code as partial_obs_experiment.py but with obs_type="full".
"""
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
from partial_obs_experiment import run_rule, run_ippo

OUTPUT_DIR = Path(__file__).resolve().parent.parent / os.environ.get("ETHICAAI_OUTDIR", "outputs") / "partial_obs"


def main():
    print("Full Obs Baseline for Table 8")
    print("=" * 50)
    
    n_seeds_rule = 10
    n_seeds_ippo = 3
    
    fast = os.environ.get("ETHICAAI_FAST") == "1"
    if fast:
        print("[FAST MODE]")
        n_seeds_rule = 3
        n_seeds_ippo = 2
    
    result = {}
    
    for alg in ["situational", "unconditional", "ippo"]:
        n_seeds = n_seeds_ippo if alg == "ippo" else n_seeds_rule
        seeds_data = []
        
        for s in range(n_seeds):
            if alg == "ippo":
                r = run_ippo(s * 7 + 42, "full", 0.0, 0)
            else:
                r = run_rule(s * 7 + 42, alg, "full", 0.0, 0)
            seeds_data.append(r)
            sys.stdout.write(".")
        sys.stdout.flush()
        
        survs = [r["survival"] for r in seeds_data]
        lams = [r["lambda"] for r in seeds_data]
        wels = [r["welfare"] for r in seeds_data]
        
        result[alg] = {
            "lambda_mean": float(np.mean(lams)),
            "lambda_std": float(np.std(lams)),
            "survival_mean": float(np.mean(survs)),
            "survival_std": float(np.std(survs)),
            "welfare_mean": float(np.mean(wels)),
            "welfare_std": float(np.std(wels)),
        }
        
        print(f"\n  {alg:>13s}: Surv={np.mean(survs):5.1f}% | λ={np.mean(lams):.3f} | W={np.mean(wels):.1f}")
    
    # Merge into existing partial_obs_results.json
    existing_path = OUTPUT_DIR / "partial_obs_results.json"
    if existing_path.exists():
        with open(existing_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}
    
    all_results["full_obs"] = result
    
    with open(existing_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n  Merged into: {existing_path}")
    print(f"\n  TABLE 8 values:")
    print(f"    Selfish RL (IPPO): {result['ippo']['survival_mean']:.0f}%")
    print(f"    Situational:      {result['situational']['survival_mean']:.0f}%")
    print(f"    Unconditional:    {result['unconditional']['survival_mean']:.0f}%")


if __name__ == "__main__":
    main()
