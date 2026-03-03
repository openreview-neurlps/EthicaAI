"""
Phase D: LOLA Divergence Analysis in N-agent PGG
=================================================
Shows that LOLA's second-order opponent gradient correction
diverges for N>2 agents in the non-linear PGG setting.

LOLA update: θ_i += η * [∇_θi J_i + Σ_{j≠i} (∇_θj θ_i) * ∇_θi ∇_θj J_i]
The second term scales as O(N) and the cross-derivatives become
unstable in non-linear dynamics.
"""
import numpy as np
import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "lola_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_payoff_gradient(lambdas, M, N_total, R_crit=0.15):
    """
    Compute ∂r_i/∂λ_i for agent i in the non-linear PGG.
    r_i = (1-λ_i) + M/N * Σ_j λ_j * p_surv(R)
    """
    N = len(lambdas)
    total_contrib = np.sum(lambdas)
    R = total_contrib / N  # Simplified resource proxy
    
    # Survival probability (sigmoid approximation)
    p_surv = 1.0 / (1.0 + np.exp(-20 * (R - R_crit)))
    dp_dR = p_surv * (1 - p_surv) * 20
    
    grads = np.zeros(N)
    for i in range(N):
        # Direct cost
        direct = -1.0
        # Survival benefit (through contribution to R)  
        benefit = M / N * p_surv + M / N * total_contrib * dp_dR / N
        grads[i] = direct + benefit
    
    return grads, p_surv


def compute_lola_correction(lambdas, M, N_total, lr_opp=0.01):
    """
    Compute LOLA's second-order correction term.
    For each agent i, the correction involves:
    Σ_{j≠i} ∇_θj(θ_i's learning step) * ∇_θi(∇_θj J_i)
    
    Returns: correction magnitude per agent, gradient norm
    """
    N = len(lambdas)
    eps = 1e-4
    
    corrections = np.zeros(N)
    cross_deriv_norms = np.zeros(N)
    
    for i in range(N):
        total_correction = 0.0
        for j in range(N):
            if i == j:
                continue
            
            # ∂²J_i / ∂λ_i ∂λ_j via numerical differentiation
            lam_pp = lambdas.copy()
            lam_pp[i] += eps; lam_pp[j] += eps
            g_pp, _ = compute_payoff_gradient(lam_pp, M, N_total)
            
            lam_pm = lambdas.copy()
            lam_pm[i] += eps; lam_pm[j] -= eps
            g_pm, _ = compute_payoff_gradient(lam_pm, M, N_total)
            
            lam_mp = lambdas.copy()
            lam_mp[i] -= eps; lam_mp[j] += eps
            g_mp, _ = compute_payoff_gradient(lam_mp, M, N_total)
            
            lam_mm = lambdas.copy()
            lam_mm[i] -= eps; lam_mm[j] -= eps
            g_mm, _ = compute_payoff_gradient(lam_mm, M, N_total)
            
            cross = (g_pp[i] - g_pm[i] - g_mp[i] + g_mm[i]) / (4 * eps * eps)
            
            # j's gradient (learning direction)
            g_j, _ = compute_payoff_gradient(lambdas, M, N_total)
            
            # LOLA correction: lr_opp * g_j[j] * cross
            correction_j = lr_opp * g_j[j] * cross
            total_correction += correction_j
            cross_deriv_norms[i] += abs(cross)
        
        corrections[i] = total_correction
    
    return corrections, cross_deriv_norms


def main():
    print("=" * 70)
    print("  Phase D: LOLA Divergence Analysis")
    print("=" * 70)
    
    M = 1.6  # PGG multiplier
    N_values = [2, 5, 10, 20, 50, 100]
    
    results = {}
    
    for N in N_values:
        # Start at Nash Trap region
        lambdas = np.full(N, 0.5)
        N_total = int(N / 0.7)  # Including Byzantine
        
        grads, p_surv = compute_payoff_gradient(lambdas, M, N_total)
        corrections, cross_norms = compute_lola_correction(lambdas, M, N_total)
        
        grad_norm = np.mean(np.abs(grads))
        correction_norm = np.mean(np.abs(corrections))
        cross_norm = np.mean(cross_norms)
        
        # Ratio: if correction >> gradient, LOLA is unstable
        ratio = correction_norm / (grad_norm + 1e-10)
        
        results[str(N)] = {
            "N": N,
            "grad_norm": float(grad_norm),
            "lola_correction_norm": float(correction_norm),
            "cross_deriv_norm": float(cross_norm),
            "correction_to_grad_ratio": float(ratio),
            "p_surv": float(p_surv),
            "diverges": ratio > 1.0,
        }
        
        status = "DIVERGES" if ratio > 1.0 else "stable"
        print(f"  N={N:>4d}: |∇J|={grad_norm:.4f}, |LOLA|={correction_norm:.4f}, "
              f"ratio={ratio:.2f} — {status}")
    
    # Simulate LOLA trajectory for N=20
    print("\n  LOLA trajectory simulation (N=20, 100 steps):")
    N = 20
    N_total = int(N / 0.7)
    lambdas = np.full(N, 0.5)
    lr = 0.01
    trajectory = [float(np.mean(lambdas))]
    
    for step in range(100):
        grads, _ = compute_payoff_gradient(lambdas, M, N_total)
        corrections, _ = compute_lola_correction(lambdas, M, N_total)
        
        # LOLA update
        lambdas = lambdas + lr * (grads + corrections)
        lambdas = np.clip(lambdas, 0.0, 1.0)
        trajectory.append(float(np.mean(lambdas)))
        
        if step % 20 == 0:
            print(f"    Step {step}: λ_mean={np.mean(lambdas):.4f}")
    
    results["trajectory_N20"] = trajectory
    results["trajectory_converges_to"] = float(trajectory[-1])
    
    # Save
    out_path = OUTPUT_DIR / "lola_analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")
    
    print("\n" + "=" * 70)
    print("  CONCLUSION")
    print("=" * 70)
    diverge_at = min([r["N"] for r in results.values() 
                      if isinstance(r, dict) and r.get("diverges", False)], default=None)
    if diverge_at:
        print(f"  LOLA correction exceeds gradient at N≥{diverge_at}")
        print(f"  → Second-order opponent modeling is UNSTABLE in N-agent PGG")
    print(f"  N=20 trajectory converges to λ={trajectory[-1]:.4f} (still trapped)")
    print("  DONE!")


if __name__ == "__main__":
    main()
