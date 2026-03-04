"""
Generate LaTeX table rows from JSON experiment outputs.
Ensures paper tables always match the actual results.

Usage:
  python generate_tables.py
"""
import json
import os

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs")


def load_json(rel_path):
    """Load a JSON file relative to the outputs directory."""
    path = os.path.join(OUTPUTS_DIR, rel_path)
    if not os.path.exists(path):
        print(f"  [SKIP] {rel_path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def fmt(val, decimals=1):
    """Format a float value."""
    if isinstance(val, (int, float)):
        return f"{val:.{decimals}f}"
    return str(val)


def generate_table3():
    """Generate Table 3 (Nash Trap across algorithms) from JSON results."""
    print("\n=== Table 3: Nash Trap Across Algorithms ===\n")

    # CleanRL baselines
    data = load_json("cleanrl_baselines/cleanrl_baseline_results.json")
    if data:
        for key in ["CleanRL IPPO", "CleanRL MAPPO"]:
            if key in data:
                d = data[key]
                lam = d["lambda"]
                surv = d["survival"]
                w = d["welfare"]
                print(f"{key}: λ={fmt(lam['mean'], 3)} ± {fmt(lam['std'], 3)}, "
                      f"surv={fmt(surv['mean'])}%, W={fmt(w['mean'])}")

    # QMIX
    qmix = load_json("cleanrl_baselines/qmix_baseline_results.json")
    if qmix and "CleanRL QMIX" in qmix:
        d = qmix["CleanRL QMIX"]
        lam = d["lambda"]
        surv = d["survival"]
        w = d["welfare"]
        print(f"CleanRL QMIX: λ={fmt(lam['mean'], 3)} ± {fmt(lam['std'], 3)}, "
              f"surv={fmt(surv['mean'])}%, W={fmt(w['mean'])}")


def generate_hp_sweep_table():
    """Generate HP Sweep table (Appendix H) from JSON results."""
    print("\n=== Appendix H: HP Sweep ===\n")

    data = load_json("hp_sweep/hp_sweep_results.json")
    if not data:
        print("  [SKIP] hp_sweep/hp_sweep_results.json not found")
        return

    print(f"{'LR':>12} {'Entropy':>10} {'λ_mean':>10} {'Surv%':>8} {'Trapped':>8}")
    print("-" * 55)
    
    # data is a dict of { "combo_key": { "lr": ..., "entropy_coef": ..., "lambda_mean": ..., "survival_mean": ..., "still_trapped": ... } }
    for key, run in data.items():
        lr = run.get("lr", "?")
        ent = run.get("entropy_coef", "?")
        lam = run.get("lambda_mean", 0)
        surv = run.get("survival_mean", 0)
        trapped = "Yes" if run.get("still_trapped", True) else "No"
        print(f"{lr:>12.0e} {ent:>10.2f} {lam:>10.3f} {surv:>8.1f} {trapped:>8}")


if __name__ == "__main__":
    print("=" * 60)
    print("  LaTeX Table Generator — EthicaAI")
    print("=" * 60)
    generate_table3()
    generate_hp_sweep_table()
    print("\n[DONE]")
