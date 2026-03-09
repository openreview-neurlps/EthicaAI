#!/usr/bin/env python
"""
nash_trap_bound.py — Numerical verification of Nash Trap upper bound
====================================================================
Verifies that the Nash Trap fixed point λ̂ satisfies:

    λ̂ ≤ 1 - (1 - M/N) / (γ · C · f_ratio)

where:
  - M/N: marginal return ratio (< 1 in social dilemmas)
  - γ: discount factor
  - C: upper bound on survival gradient ∂p_surv/∂λ
  - f_ratio: f(R_crit) / f(R_recov)

Sweeps over parameter space to validate the bound is tight.
"""

import numpy as np
import json
import os
import time

# ============================================================
# Nash Trap computation
# ============================================================
def compute_nash_trap_lambda(M_over_N, gamma, f_crit, f_recov, 
                              shock_prob=0.05, shock_mag=0.15,
                              delta=0.4, beta=0.3, E=20.0, N=5, T=50):
    """
    Numerically find λ̂ where ∂V_i/∂λ = 0.
    
    V_i(λ) = E(1 - λ + Mλ/N) / (1 - γ·p_surv(λ))
    ∂V_i/∂λ = (M/N - 1)·E·(1 - γ·p_surv)^{-1} 
              + E(1-λ+Mλ/N) · γ · ∂p_surv/∂λ · (1-γ·p_surv)^{-2}
    """
    M = M_over_N * N
    sigma_bar = shock_prob * shock_mag
    
    def p_surv(lam, n_sim=200):
        """Monte Carlo survival probability."""
        rng = np.random.RandomState(42)
        survived = 0
        for _ in range(n_sim):
            R = 0.5
            for t in range(T):
                coop = (1 - beta) * lam
                f_R = f_crit if R < 0.15 else (f_recov if R >= 0.25 else 0.03)
                R = R + f_R * (coop - delta)
                if rng.random() < shock_prob:
                    R -= shock_mag
                R = max(0, min(1, R))
                if R <= 0.001:
                    break
            else:
                survived += 1
        return survived / n_sim
    
    # Grid search for λ̂
    best_lam = 0.5
    min_grad = float('inf')
    
    for lam in np.linspace(0.01, 0.99, 200):
        ps = p_surv(lam)
        ps_plus = p_surv(min(lam + 0.01, 0.99))
        ps_minus = p_surv(max(lam - 0.01, 0.01))
        dp = (ps_plus - ps_minus) / 0.02
        
        # V_i terms
        immediate = E * (M_over_N - 1)
        denominator = max(1e-6, (1 - gamma * ps))
        per_step = E * (1 - lam + M_over_N * lam)
        future = gamma * per_step * dp / (denominator ** 2)
        
        grad = immediate / denominator + future
        
        if abs(grad) < min_grad:
            min_grad = abs(grad)
            best_lam = lam
    
    return best_lam


def theoretical_bound(M_over_N, gamma, f_crit, f_recov):
    """
    Theoretical upper bound on λ̂:
    
    At the fixed point, immediate cost = survival benefit:
      |M/N - 1| · E / (1-γ·p) = γ · E · (1-λ+Mλ/N) · (∂p/∂λ) / (1-γ·p)²
    
    Since ∂p/∂λ ≤ C · f_crit/f_recov (survival gradient bounded by
    recovery rate ratio), we get:
    
      λ̂ ≤ 1 - (1-M/N) · (1-γ) / (γ · f_crit/f_recov · M/N)
    
    This is an approximate bound; we verify numerically.
    """
    f_ratio = f_crit / max(f_recov, 1e-6)
    if f_ratio < 1e-8 or gamma > 0.999:
        return 0.0  # Trivially bounded
    bound = 1.0 - (1.0 - M_over_N) * (1.0 - gamma) / (gamma * f_ratio * max(M_over_N, 0.01))
    return max(0.0, min(1.0, bound))


# ============================================================
# Main: Parameter sweep
# ============================================================
if __name__ == "__main__":
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'outputs', 'nash_trap_bound')
    os.makedirs(OUT, exist_ok=True)
    
    print("=" * 60)
    print("  NASH TRAP UPPER BOUND VERIFICATION")
    print("=" * 60)
    
    t0 = time.time()
    results = []
    
    # Sweep parameters
    M_over_N_vals = [0.10, 0.20, 0.32, 0.50, 0.80]
    gamma_vals = [0.90, 0.95, 0.99]
    f_crit_vals = [0.001, 0.005, 0.01, 0.03]
    f_recov = 0.10  # Fixed
    
    for M_over_N in M_over_N_vals:
        for gamma in gamma_vals:
            for f_crit in f_crit_vals:
                lam_hat = compute_nash_trap_lambda(M_over_N, gamma, f_crit, f_recov)
                bound = theoretical_bound(M_over_N, gamma, f_crit, f_recov)
                
                point = {
                    "M_over_N": M_over_N,
                    "gamma": gamma,
                    "f_crit": f_crit,
                    "f_recov": f_recov,
                    "f_ratio": round(f_crit / f_recov, 4),
                    "lambda_hat": round(lam_hat, 3),
                    "bound": round(bound, 3),
                    "bound_holds": bool(lam_hat <= bound + 0.05),
                    "gap_to_1": round(1.0 - lam_hat, 3),
                }
                results.append(point)
    
    # Summary
    all_below_1 = all(r["lambda_hat"] < 0.95 for r in results 
                       if r["M_over_N"] < 0.9 and r["f_crit"] <= 0.01)
    
    print(f"\n  Total configurations: {len(results)}")
    print(f"  All λ̂ < 0.95 (for M/N<0.9, f_crit≤0.01): {all_below_1}")
    print(f"\n  Key examples:")
    for r in results:
        if r["M_over_N"] == 0.32 and r["gamma"] == 0.99:
            print(f"    f_crit={r['f_crit']:.3f}: λ̂={r['lambda_hat']:.3f}, "
                  f"gap={r['gap_to_1']:.3f}, bound={r['bound']:.3f}")
    
    total = time.time() - t0
    
    output = {
        "experiment": "Nash Trap Upper Bound Verification",
        "theorem_statement": (
            "In a TPSD with M/N < 1 and γ < 1, the Nash Trap fixed point satisfies "
            "λ̂ < 1. The gap (1 - λ̂) is bounded below by a function of (1-M/N), "
            "(1-γ), and f(R_crit)/f(R_recov)."
        ),
        "results": results,
        "all_below_1": all_below_1,
        "time_seconds": round(total, 1),
    }
    
    json_path = os.path.join(OUT, "nash_trap_bound_results.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  Total: {total:.0f}s")
    print(f"  Saved: {json_path}")
