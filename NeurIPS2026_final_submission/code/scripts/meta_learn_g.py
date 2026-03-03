"""
Track C: Meta-Learning g(θ, R)
Optimize the moral commitment function g_φ(θ, R) via MAML-style meta-learning.

Goal: Replace hand-crafted piecewise g with learned parameters φ
      that maximize group fitness + resource sustainability.

Architecture:
  Inner Loop: N=50 agents play PGG for T=200 rounds using g_φ
  Outer Loop: Update φ to maximize W (welfare) + minimize σ(R) (resource variance)

Comparison: Learned g_φ vs Hand-crafted g (ablation across Byzantine fractions)
"""

import numpy as np
import json
import os
import time
from copy import deepcopy

# ============================================================
# Constants
# ============================================================
N_AGENTS = 50
T_ROUNDS = 200
N_SEEDS = 5
MULTIPLIER = 1.6          # PGG multiplier
ENDOWMENT = 20.0
ALPHA_EMA = 0.6           # EMA smoothing for λ_t
SVO_THETA = np.radians(45)  # Prosocial default

# Meta-learning
META_LR = 0.05            # Outer loop learning rate
N_META_STEPS = 30         # Outer loop iterations
N_INNER_TASKS = 3         # Tasks per meta-step (different seeds)
PERTURBATION_SCALE = 0.02 # For finite-difference gradient estimation


# ============================================================
# Parameterized g_φ(θ, R) — Learnable Moral Function
# ============================================================
class LearnableG:
    """Parameterized g function with 6 learnable parameters.

    g_φ(θ, R) = sigmoid(φ₀·sin(θ)) · clamp(φ₁ + φ₂·R, 0, 1)   if R < φ₃ (crisis)
              = sigmoid(φ₀·sin(θ)) · clamp(φ₄ + φ₅·R, 0, 1)   if R > 1-φ₃ (abundance)
              = sigmoid(φ₀·sin(θ)) · (φ₁ + φ₂·R)               otherwise

    φ = [svo_scale, base_low, slope_low, crisis_threshold, base_high, slope_high]
    """

    def __init__(self, phi=None):
        if phi is None:
            # Initialize near hand-crafted values
            self.phi = np.array([
                1.0,    # φ₀: SVO scale (hand-crafted: implicit 1.0)
                0.21,   # φ₁: base commitment in crisis (≈ 0.3·sin(45°))
                0.0,    # φ₂: slope in crisis (hand-crafted: 0)
                0.2,    # φ₃: crisis threshold R_crisis
                0.75,   # φ₄: base in abundance (≈ 1.5·sin(45°))
                0.0,    # φ₅: slope in abundance (hand-crafted: 0)
            ], dtype=np.float64)
        else:
            self.phi = np.array(phi, dtype=np.float64)

    def __call__(self, theta, R):
        p = self.phi
        svo_factor = 1.0 / (1.0 + np.exp(-p[0] * np.sin(theta)))  # sigmoid

        if R < p[3]:  # Crisis
            raw = p[1] + p[2] * R
        elif R > (1.0 - p[3]):  # Abundance
            raw = p[4] + p[5] * R
        else:  # Normal
            raw = p[1] + (p[4] - p[1]) * (R - p[3]) / max(1.0 - 2 * p[3], 0.01)

        return float(np.clip(svo_factor * raw, 0.0, 1.0))

    def copy(self):
        return LearnableG(self.phi.copy())


# ============================================================
# Hand-crafted g (baseline from paper)
# ============================================================
def handcrafted_g(theta, R):
    R_CRISIS = 0.2
    R_ABUNDANCE = 0.7
    base = np.sin(theta)
    if R < R_CRISIS:
        return max(0.0, 0.3 * base)
    elif R > R_ABUNDANCE:
        return min(1.0, 1.5 * base)
    else:
        return float(np.clip(base * (0.7 + 1.6 * R), 0.0, 1.0))


# ============================================================
# PGG Simulation
# ============================================================
def run_pgg(g_func, n_agents=N_AGENTS, t_rounds=T_ROUNDS,
            byz_frac=0.0, seed=42):
    """Run PGG and return metrics.

    Returns: dict with welfare, cooperation, resource_stability, sustainability
    """
    rng = np.random.RandomState(seed)
    svo_angles = rng.uniform(np.radians(20), np.radians(70), n_agents)
    n_byz = int(n_agents * byz_frac)

    # State
    R_t = 0.5
    lambdas = np.array([g_func(svo_angles[i], R_t) for i in range(n_agents)])

    welfare_history = []
    coop_history = []
    resource_history = [R_t]

    for t in range(t_rounds):
        # Decide contributions
        contributions = np.zeros(n_agents)
        for i in range(n_agents):
            if i < n_byz:
                contributions[i] = 0.0  # Byzantine: always defect
            else:
                contributions[i] = ENDOWMENT * lambdas[i]

        # PGG payoffs
        total_contrib = contributions.sum()
        public_good = (total_contrib * MULTIPLIER) / n_agents
        payoffs = (ENDOWMENT - contributions) + public_good

        # Cooperation rate (fraction contributing > 50%)
        coop = np.mean(contributions[n_byz:] > ENDOWMENT * 0.5)
        coop_history.append(float(coop))

        # Welfare
        welfare = float(payoffs.mean())
        welfare_history.append(welfare)

        # Resource dynamics
        coop_ratio = np.mean(contributions) / ENDOWMENT
        R_t = np.clip(R_t + 0.1 * (coop_ratio - 0.4), 0.0, 1.0)
        resource_history.append(R_t)

        # Update λ_t with EMA
        for i in range(n_byz, n_agents):
            target = g_func(svo_angles[i], R_t)
            lambdas[i] = ALPHA_EMA * lambdas[i] + (1 - ALPHA_EMA) * target

    resource_arr = np.array(resource_history)
    return {
        "welfare": float(np.mean(welfare_history)),
        "cooperation": float(np.mean(coop_history)),
        "resource_stability": float(np.std(resource_arr)),
        "sustainability": float(np.mean(resource_arr > 0.1)),
        "final_resource": float(resource_arr[-1]),
    }


# ============================================================
# Outer Loss: What we optimize φ for
# ============================================================
def outer_loss(g_func, seeds, byz_fracs=None):
    """Compute outer loss across multiple tasks (seeds × byz_fracs).

    Loss = -mean(welfare) + 2·mean(resource_variance)
    Lower is better.
    """
    if byz_fracs is None:
        byz_fracs = [0.0, 0.1, 0.3]

    total_welfare = 0.0
    total_r_var = 0.0
    n_tasks = 0

    for seed in seeds:
        for bf in byz_fracs:
            result = run_pgg(g_func, byz_frac=bf, seed=seed)
            total_welfare += result["welfare"]
            total_r_var += result["resource_stability"]
            n_tasks += 1

    avg_w = total_welfare / n_tasks
    avg_rv = total_r_var / n_tasks

    # Negative welfare (we want to maximize) + resource instability penalty
    return -avg_w + 2.0 * avg_rv


# ============================================================
# MAML-style Meta-Learning via Finite Differences
# ============================================================
def meta_learn(n_steps=N_META_STEPS, lr=META_LR, verbose=True):
    """Optimize g_φ parameters via evolution strategy (finite differences)."""

    g = LearnableG()
    n_params = len(g.phi)
    best_loss = float('inf')
    best_phi = g.phi.copy()
    loss_history = []

    if verbose:
        print("=" * 60)
        print("  Track C: Meta-Learning g_φ(θ, R)")
        print(f"  Params: {n_params}, Meta-steps: {n_steps}, LR: {lr}")
        print("=" * 60)

    for step in range(n_steps):
        seeds = list(range(step * N_INNER_TASKS, (step + 1) * N_INNER_TASKS))

        # Current loss
        current_loss = outer_loss(g, seeds)

        # Finite-difference gradient estimation
        grad = np.zeros(n_params)
        for p_idx in range(n_params):
            g_plus = g.copy()
            g_plus.phi[p_idx] += PERTURBATION_SCALE
            loss_plus = outer_loss(g_plus, seeds)

            g_minus = g.copy()
            g_minus.phi[p_idx] -= PERTURBATION_SCALE
            loss_minus = outer_loss(g_minus, seeds)

            grad[p_idx] = (loss_plus - loss_minus) / (2 * PERTURBATION_SCALE)

        # Gradient descent
        g.phi -= lr * grad

        # Clamp parameters to reasonable ranges
        g.phi[0] = np.clip(g.phi[0], 0.1, 5.0)    # SVO scale
        g.phi[1] = np.clip(g.phi[1], 0.0, 1.0)     # base_low
        g.phi[2] = np.clip(g.phi[2], -2.0, 2.0)     # slope_low
        g.phi[3] = np.clip(g.phi[3], 0.05, 0.45)    # crisis threshold
        g.phi[4] = np.clip(g.phi[4], 0.0, 2.0)      # base_high
        g.phi[5] = np.clip(g.phi[5], -2.0, 2.0)     # slope_high

        loss_history.append(float(current_loss))

        if current_loss < best_loss:
            best_loss = current_loss
            best_phi = g.phi.copy()

        if verbose and step % 5 == 0:
            print(f"  Step {step:3d} | Loss={current_loss:8.3f} | "
                  f"φ=[{', '.join(f'{p:.3f}' for p in g.phi)}]")

    g.phi = best_phi
    return g, loss_history


# ============================================================
# Ablation: Learned g_φ vs Hand-crafted g
# ============================================================
def ablation_comparison(learned_g):
    """Compare learned g_φ against hand-crafted g across conditions."""

    byz_fracs = [0.0, 0.1, 0.2, 0.3, 0.5]
    results = {"learned": {}, "handcrafted": {}}

    print("\n" + "=" * 60)
    print("  ABLATION: Learned g_φ vs Hand-crafted g")
    print("=" * 60)
    print(f"  {'Byz%':>5} | {'Metric':>12} | {'Learned':>10} | {'Handcraft':>10} | {'Δ':>8}")
    print("  " + "-" * 55)

    for bf in byz_fracs:
        learned_runs = []
        handcrafted_runs = []

        for seed in range(N_SEEDS):
            lr = run_pgg(learned_g, byz_frac=bf, seed=seed + 100)
            hr = run_pgg(handcrafted_g, byz_frac=bf, seed=seed + 100)
            learned_runs.append(lr)
            handcrafted_runs.append(hr)

        bf_key = f"byz_{int(bf*100)}"
        results["learned"][bf_key] = {}
        results["handcrafted"][bf_key] = {}

        for metric in ["welfare", "cooperation", "resource_stability", "sustainability"]:
            l_vals = [r[metric] for r in learned_runs]
            h_vals = [r[metric] for r in handcrafted_runs]
            l_mean = np.mean(l_vals)
            h_mean = np.mean(h_vals)
            delta = l_mean - h_mean

            results["learned"][bf_key][metric] = float(l_mean)
            results["handcrafted"][bf_key][metric] = float(h_mean)

            print(f"  {bf*100:5.0f}% | {metric:>12} | {l_mean:10.3f} | {h_mean:10.3f} | {delta:+8.3f}")

        print("  " + "-" * 55)

    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    OUT = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'meta_learn_g')
    os.makedirs(OUT, exist_ok=True)

    t0 = time.time()

    # Phase 1: Meta-learn g_φ
    learned_g, loss_curve = meta_learn(n_steps=N_META_STEPS)
    meta_time = time.time() - t0

    print(f"\n  Meta-learning completed in {meta_time:.1f}s")
    print(f"  Optimal φ = [{', '.join(f'{p:.4f}' for p in learned_g.phi)}]")

    # Phase 2: Ablation comparison
    ablation = ablation_comparison(learned_g)

    # Phase 3: Save results
    output = {
        "meta_learning": {
            "n_steps": N_META_STEPS,
            "optimal_phi": learned_g.phi.tolist(),
            "phi_labels": [
                "svo_scale", "base_crisis", "slope_crisis",
                "crisis_threshold", "base_abundance", "slope_abundance"
            ],
            "loss_curve": loss_curve,
            "time_seconds": float(meta_time),
        },
        "handcrafted_params": {
            "R_crisis": 0.2,
            "R_abundance": 0.7,
            "crisis_factor": 0.3,
            "abundance_factor": 1.5,
        },
        "ablation": ablation,
    }

    # Summary deltas
    for bf_key in ablation["learned"]:
        l = ablation["learned"][bf_key]
        h = ablation["handcrafted"][bf_key]
        output[f"delta_{bf_key}"] = {
            k: float(l[k] - h[k]) for k in l
        }

    path = os.path.join(OUT, "meta_learn_results.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved: {path}")
    print("\n  Track C COMPLETE!")
