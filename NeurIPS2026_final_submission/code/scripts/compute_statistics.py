"""
Compute actual statistical tests for paper claims.
Reads JSON outputs and computes Mann-Whitney U, Cliff's delta, effect sizes.
Outputs values that should replace hardcoded claims in the paper.
"""
import numpy as np
import json
from pathlib import Path
from scipy import stats

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"


def cliffs_delta(x, y):
    """Compute Cliff's delta effect size."""
    n_x, n_y = len(x), len(y)
    count = 0
    for xi in x:
        for yi in y:
            if xi > yi:
                count += 1
            elif xi < yi:
                count -= 1
    return count / (n_x * n_y)


def load_per_seed_lambda(path, key=None):
    """Load per-seed lambda values from JSON."""
    with open(path) as f:
        data = json.load(f)
    if key:
        data = data[key]
    if "per_seed_lambda" in data:
        return data["per_seed_lambda"]
    elif "lambda" in data and isinstance(data["lambda"], dict):
        return None  # aggregated only
    return None


def main():
    print("=" * 60)
    print("  Statistical Verification (No Hardcoding)")
    print("=" * 60)

    # Oracle: lambda=1.0 for all seeds (by definition)
    n_oracle = 20
    oracle_lambda = [1.0] * n_oracle

    # Gather all available per-seed data
    algorithms = {}

    # 1. Main baselines from reproduce outputs
    reproduce_path = OUTPUTS / "reproduce"
    if reproduce_path.exists():
        for f in reproduce_path.glob("*.json"):
            with open(f) as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, dict) and "per_seed_lambda" in v:
                        algorithms[k] = v["per_seed_lambda"]

    # 2. CleanRL baselines
    cleanrl_path = OUTPUTS / "cleanrl_baselines" / "cleanrl_baseline_results.json"
    if cleanrl_path.exists():
        with open(cleanrl_path) as f:
            data = json.load(f)
        for k, v in data.items():
            if "per_seed_lambda" in v:
                algorithms[f"cleanrl_{k}"] = v["per_seed_lambda"]

    # 3. QMIX
    qmix_path = OUTPUTS / "cleanrl_baselines" / "qmix_real_results.json"
    if qmix_path.exists():
        with open(qmix_path) as f:
            data = json.load(f)
        if "per_seed_lambda" in data:
            algorithms["QMIX"] = data["per_seed_lambda"]

    # 4. LOLA
    lola_path = OUTPUTS / "cleanrl_baselines" / "lola_results.json"
    if lola_path.exists():
        with open(lola_path) as f:
            data = json.load(f)
        if "per_seed_lambda" in data:
            algorithms["LOLA"] = data["per_seed_lambda"]

    # 5. IQL
    iql_path = OUTPUTS / "cleanrl_baselines" / "iql_baseline_results.json"
    if iql_path.exists():
        with open(iql_path) as f:
            data = json.load(f)
        if "per_seed_lambda" in data:
            algorithms["IQL"] = data["per_seed_lambda"]

    if not algorithms:
        print("  WARNING: No per-seed data found. Run experiments first.")
        return

    print(f"\n  Found {len(algorithms)} algorithms with per-seed data")

    # Compute statistics
    results = {}
    all_p_values = []

    for name, seed_lams in algorithms.items():
        oracle_slice = oracle_lambda[:len(seed_lams)]
        
        # Mann-Whitney U
        u_stat, p_value = stats.mannwhitneyu(
            seed_lams, oracle_slice, alternative='two-sided'
        )
        all_p_values.append((name, p_value))

        # Cliff's delta
        delta = cliffs_delta(seed_lams, oracle_slice)

        results[name] = {
            "n_seeds": len(seed_lams),
            "lambda_mean": float(np.mean(seed_lams)),
            "lambda_std": float(np.std(seed_lams)),
            "mann_whitney_u": float(u_stat),
            "p_value_raw": float(p_value),
            "cliffs_delta": float(delta),
        }

        print(f"  {name:25s}: λ={np.mean(seed_lams):.3f}±{np.std(seed_lams):.3f} | "
              f"U={u_stat:.0f} | p={p_value:.2e} | Cliff's δ={delta:.3f}")

    # Holm-Bonferroni correction
    sorted_pvals = sorted(all_p_values, key=lambda x: x[1])
    k = len(sorted_pvals)
    print(f"\n  Holm-Bonferroni correction (K={k}):")
    min_delta = 1.0
    for rank, (name, raw_p) in enumerate(sorted_pvals):
        adjusted_p = min(raw_p * (k - rank), 1.0)
        results[name]["p_value_holm"] = float(adjusted_p)
        delta_abs = abs(results[name]["cliffs_delta"])
        min_delta = min(min_delta, delta_abs)
        print(f"    {name:25s}: raw p={raw_p:.2e} → adjusted p={adjusted_p:.2e}")

    # Summary for paper
    print(f"\n  ===== VALUES FOR PAPER (replace hardcoded) =====")
    print(f"  Min |Cliff's δ| across all algorithms: {min_delta:.2f}")
    all_holm = [results[n]["p_value_holm"] for n in results]
    max_holm_p = max(all_holm)
    print(f"  Max Holm-corrected p-value: {max_holm_p:.2e}")
    
    if max_holm_p < 0.001:
        print(f"  → Paper can claim: 'All corrected p < 0.001'")
    elif max_holm_p < 0.01:
        print(f"  → Paper should claim: 'All corrected p < 0.01'")
    else:
        print(f"  → Paper should claim: 'All corrected p < {max_holm_p:.1e}'")

    if min_delta >= 0.9:
        print(f"  → Paper can claim: 'Cliff's δ exceed 0.90'")
    else:
        print(f"  → Paper should claim: 'Cliff's |δ| ≥ {min_delta:.2f}'")

    # Save
    out_path = OUTPUTS / "extended" / "statistical_verification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
