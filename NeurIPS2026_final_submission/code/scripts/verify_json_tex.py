"""Cross-verify TeX Table 3 values against JSON output files."""
import json
import os

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")

def load(path):
    fp = os.path.join(BASE, path)
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        return json.load(f)

def main():
    # 1. REINFORCE
    d = load("ppo_nash_trap/ippo_results.json")
    if d:
        print("=== REINFORCE ===")
        if "algorithms" in d:
            algos = d["algorithms"]
            if isinstance(algos, list):
                for a in algos:
                    if isinstance(a, dict):
                        print(f"  {a.get('name','?')}: lam={a.get('lambda_mean','?')}, surv={a.get('survival_mean','?')}, welf={a.get('welfare_mean','?')}")
            elif isinstance(algos, dict):
                for k, v in algos.items():
                    if isinstance(v, dict):
                        print(f"  {k}: lam={v.get('lambda_mean','?')}, surv={v.get('survival_mean','?')}, welf={v.get('welfare_mean','?')}")
        else:
            print(f"  Top keys: {list(d.keys())[:8]}")
            for k, v in d.items():
                if isinstance(v, dict) and "lambda_mean" in v:
                    print(f"  {k}: lam={v['lambda_mean']}, surv={v.get('survival_mean','?')}, welf={v.get('welfare_mean','?')}")

    # 2. CleanRL
    d = load("cleanrl_baselines/cleanrl_baseline_results.json")
    if d:
        print("\n=== CleanRL ===")
        for k, v in d.items():
            if isinstance(v, dict) and "lambda_mean" in v:
                print(f"  {k}: lam={v['lambda_mean']}, std={v.get('lambda_std','?')}, surv={v.get('survival_mean','?')}, welf={v.get('welfare_mean','?')}")

    # 3. QMIX
    d = load("cleanrl_baselines/qmix_real_results.json")
    if d:
        print("\n=== QMIX ===")
        print(f"  lam={d.get('lambda_mean','?')}, std={d.get('lambda_std','?')}, surv={d.get('survival_mean','?')}, welf={d.get('welfare_mean','?')}")

    # 4. LOLA
    d = load("cleanrl_baselines/lola_results.json")
    if d:
        print("\n=== LOLA ===")
        print(f"  lam={d.get('lambda_mean','?')}, std={d.get('lambda_std','?')}, surv={d.get('survival_mean','?')}, welf={d.get('welfare_mean','?')}")

    # 5. phi1
    d = load("phi1_ablation/phi1_results.json")
    if d:
        print("\n=== phi1 ===")
        for k in ["0.0", "0.21", "0.5", "1.0"]:
            if k in d:
                v = d[k]
                for byz in ["byz_0", "byz_30"]:
                    if byz in v:
                        b = v[byz]
                        print(f"  phi1={k}, {byz}: welf={b.get('welfare_mean','?')}, surv={b.get('survival_mean','?')}")

if __name__ == "__main__":
    main()
