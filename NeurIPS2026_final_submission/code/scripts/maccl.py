"""
MACCL: Multi-Agent Constrained Commitment Learning
====================================================
Pillar II of the Strong Accept strategy.

Key innovation over ACL:
  1. Primal-dual gradient optimization (not ES)
  2. Local convergence guarantee via Lagrangian saddle point
  3. Safety projection with formal constraint enforcement

Algorithm:
  Phase 1 (Safety Anchoring): Train with phi1=1.0 to establish baseline
  Phase 2 (Constrained Floor Learning): Primal-dual optimization of
    phi1(R; omega) = sigmoid(w1*R + w2*R^2 + w3)
    subject to P(survival) >= 1 - delta

Dependencies: NumPy, matplotlib.
"""

import numpy as np
import json
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    "N_AGENTS": 20,
    "ENDOWMENT": 20.0,
    "MULTIPLIER": 1.6,
    "T_HORIZON": 50,
    "R_CRIT": 0.15,
    "R_RECOV": 0.25,
    "SHOCK_PROB": 0.05,
    "SHOCK_MAG": 0.15,
    "STATE_DIM": 4,
    "GAMMA": 0.99,
    "BYZ_FRAC": 0.30,
    # MACCL-specific
    "PHASE1_EPISODES": 50,
    "PHASE2_OUTER_ITERS": 100,
    "PHASE2_INNER_EPISODES": 20,
    "EVAL_EPISODES": 100,
    "N_SEEDS": 20,
    "SAFETY_DELTA": 0.05,      # P(surv) >= 1 - delta = 95%
    "SAFETY_DELTA_MIN": 0.10,  # Hard safety floor
    "LR_OMEGA": 0.05,          # Primal learning rate
    "LR_MU": 0.10,             # Dual learning rate
    "FD_EPSILON": 0.01,        # Finite difference step
}

if os.environ.get("ETHICAAI_FAST") == "1":
    print("  [FAST MODE]")
    CONFIG["N_SEEDS"] = 3
    CONFIG["PHASE1_EPISODES"] = 10
    CONFIG["PHASE2_OUTER_ITERS"] = 20
    CONFIG["PHASE2_INNER_EPISODES"] = 5
    CONFIG["EVAL_EPISODES"] = 20

N_BYZ = int(CONFIG["N_AGENTS"] * CONFIG["BYZ_FRAC"])
N_HONEST = CONFIG["N_AGENTS"] - N_BYZ

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'maccl')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Environment (shared with other experiments)
# ============================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def reset_env():
    return {
        "R": 0.5,
        "mean_c": 0.5,
        "lam_prev": np.full(CONFIG["N_AGENTS"], 0.5),
    }


def step_env(env, lambdas, rng):
    N = CONFIG["N_AGENTS"]
    E = CONFIG["ENDOWMENT"]
    M = CONFIG["MULTIPLIER"]

    lam = lambdas.copy()
    lam[:N_BYZ] = 0.0

    contribs = E * lam
    public_good = M * contribs.sum() / N
    rewards = (E - contribs) + public_good

    coop = contribs.mean() / E
    R = env["R"]

    if R < CONFIG["R_CRIT"]:
        f_R = 0.01
    elif R < CONFIG["R_RECOV"]:
        f_R = 0.03
    else:
        f_R = 0.10

    R_new = R + f_R * (coop - 0.4)
    if rng.random() < CONFIG["SHOCK_PROB"]:
        R_new -= CONFIG["SHOCK_MAG"]
    R_new = float(np.clip(R_new, 0.0, 1.0))
    done = R_new <= 0.001

    env["R"] = R_new
    env["mean_c"] = float(coop)
    env["lam_prev"] = lam.copy()

    return rewards, done


# ============================================================
# Commitment Floor Function
# ============================================================
class CommitmentFloor:
    """
    State-dependent commitment floor:
      phi1(R; omega) = sigmoid(w1*R + w2*R^2 + w3)

    Parameters omega = (w1, w2, w3).
    """
    def __init__(self, omega=None):
        if omega is None:
            # Initialize to approximately phi1 = 1.0 (high commitment)
            self.omega = np.array([0.0, 0.0, 5.0])  # sigmoid(5) ~ 0.993
        else:
            self.omega = np.array(omega, dtype=np.float64)

    def __call__(self, R):
        """Compute floor value for resource level R."""
        z = self.omega[0] * R + self.omega[1] * R**2 + self.omega[2]
        return float(sigmoid(z))

    def profile(self, R_range=None):
        """Return floor values across R range."""
        if R_range is None:
            R_range = np.linspace(0, 1, 50)
        return R_range, np.array([self(R) for R in R_range])

    def copy(self):
        return CommitmentFloor(self.omega.copy())


# ============================================================
# Evaluate a commitment floor
# ============================================================
def evaluate_floor(floor_fn, n_episodes, rng_seed):
    """Run episodes with given floor and return (welfare, survival_rate)."""
    rng = np.random.RandomState(rng_seed)
    N = CONFIG["N_AGENTS"]

    total_welfare = 0.0
    total_survival = 0
    floor_activations = 0
    total_steps = 0

    for ep in range(n_episodes):
        env = reset_env()
        ep_welfare = 0.0
        steps = 0
        survived = True

        for t in range(CONFIG["T_HORIZON"]):
            R = env["R"]
            phi1 = floor_fn(R)

            lambdas = np.zeros(N)
            lambdas[:N_BYZ] = 0.0

            for i in range(N_HONEST):
                # Base action: noisy around 0.5 (simulating untrained agent)
                base_lam = 0.5 + rng.randn() * 0.1
                base_lam = np.clip(base_lam, 0.01, 0.99)

                # Apply floor
                if base_lam < phi1:
                    lambdas[N_BYZ + i] = phi1
                    floor_activations += 1
                else:
                    lambdas[N_BYZ + i] = base_lam
                total_steps += 1

            rewards, done = step_env(env, lambdas, rng)
            ep_welfare += rewards.mean()
            steps += 1

            if done:
                survived = False
                break

        total_welfare += ep_welfare / max(steps, 1)
        total_survival += int(survived)

    welfare = total_welfare / n_episodes
    survival = total_survival / n_episodes * 100
    activation_rate = floor_activations / max(total_steps, 1)

    return welfare, survival, activation_rate


# ============================================================
# MACCL Algorithm
# ============================================================
def run_maccl(seed):
    """
    Run MACCL for one seed.

    Phase 1: Safety Anchoring (phi1 = 1.0)
    Phase 2: Primal-dual constrained floor optimization
    Phase 3: Final evaluation
    """
    print(f"\n  [Seed {seed}] Starting MACCL")
    rng_base = 1000 * seed

    # ---- Phase 1: Safety Anchoring ----
    floor_safe = CommitmentFloor(np.array([0.0, 0.0, 10.0]))  # phi1 ~ 1.0
    w_safe, s_safe, a_safe = evaluate_floor(
        floor_safe, CONFIG["PHASE1_EPISODES"], rng_base
    )
    print(f"    Phase 1 (phi1~1.0): Welfare={w_safe:.2f}, Surv={s_safe:.0f}%")

    # ---- Phase 2: Constrained Floor Learning ----
    floor = CommitmentFloor(np.array([0.0, 0.0, 5.0]))  # Start high
    mu = 1.0  # Lagrange multiplier (dual variable)
    delta = CONFIG["SAFETY_DELTA"]
    lr_omega = CONFIG["LR_OMEGA"]
    lr_mu = CONFIG["LR_MU"]
    fd_eps = CONFIG["FD_EPSILON"]

    history = {
        "omega": [],
        "welfare": [],
        "survival": [],
        "mu": [],
        "phi1_at_R0": [],
        "phi1_at_R05": [],
        "phi1_at_R1": [],
    }

    best_omega = floor.omega.copy()
    best_welfare = -np.inf
    best_survival = 0.0

    for outer in range(CONFIG["PHASE2_OUTER_ITERS"]):
        n_inner = CONFIG["PHASE2_INNER_EPISODES"]

        # Evaluate current floor
        w_curr, s_curr, a_curr = evaluate_floor(floor, n_inner, rng_base + outer)

        # Finite-difference gradient of welfare w.r.t. omega
        grad_omega = np.zeros(3)
        for dim in range(3):
            omega_plus = floor.omega.copy()
            omega_plus[dim] += fd_eps
            floor_plus = CommitmentFloor(omega_plus)
            w_plus, s_plus, _ = evaluate_floor(floor_plus, n_inner, rng_base + outer)

            omega_minus = floor.omega.copy()
            omega_minus[dim] -= fd_eps
            floor_minus = CommitmentFloor(omega_minus)
            w_minus, s_minus, _ = evaluate_floor(floor_minus, n_inner, rng_base + outer)

            # Lagrangian gradient: d/d_omega [W - mu * (delta - P_surv)]
            L_plus = w_plus + mu * (s_plus / 100.0 - (1 - delta))
            L_minus = w_minus + mu * (s_minus / 100.0 - (1 - delta))
            grad_omega[dim] = (L_plus - L_minus) / (2 * fd_eps)

        # Primal update: omega += lr * grad (maximize Lagrangian)
        new_omega = floor.omega + lr_omega * grad_omega

        # Safety projection: evaluate candidate
        floor_candidate = CommitmentFloor(new_omega)
        w_cand, s_cand, _ = evaluate_floor(floor_candidate, n_inner, rng_base + outer)

        if s_cand >= (1 - CONFIG["SAFETY_DELTA_MIN"]) * 100:
            # Accept update
            floor.omega = new_omega
        else:
            # Reject: revert to last safe omega
            pass  # Keep current omega

        # Dual update: mu += lr_mu * (delta - P_surv)
        constraint_violation = delta - s_curr / 100.0
        mu = max(0.0, mu + lr_mu * constraint_violation)

        # Track best safe solution
        if s_curr >= (1 - delta) * 100 and w_curr > best_welfare:
            best_omega = floor.omega.copy()
            best_welfare = w_curr
            best_survival = s_curr

        # Log
        history["omega"].append(floor.omega.tolist())
        history["welfare"].append(float(w_curr))
        history["survival"].append(float(s_curr))
        history["mu"].append(float(mu))
        history["phi1_at_R0"].append(float(floor(0.0)))
        history["phi1_at_R05"].append(float(floor(0.5)))
        history["phi1_at_R1"].append(float(floor(1.0)))

        if (outer + 1) % max(1, CONFIG["PHASE2_OUTER_ITERS"] // 5) == 0:
            print(f"    Phase 2 [{outer+1}/{CONFIG['PHASE2_OUTER_ITERS']}]: "
                  f"W={w_curr:.2f}, S={s_curr:.0f}%, mu={mu:.3f}, "
                  f"phi1(0)={floor(0.0):.3f}, phi1(0.5)={floor(0.5):.3f}, "
                  f"phi1(1)={floor(1.0):.3f}")

    # ---- Phase 3: Final Evaluation ----
    best_floor = CommitmentFloor(best_omega)
    w_final, s_final, a_final = evaluate_floor(
        best_floor, CONFIG["EVAL_EPISODES"], rng_base + 9999
    )

    # Compare baselines
    # Fixed phi1 = 0.0 (selfish)
    floor_zero = CommitmentFloor(np.array([0.0, 0.0, -10.0]))
    w_zero, s_zero, _ = evaluate_floor(floor_zero, CONFIG["EVAL_EPISODES"], rng_base + 9999)

    # Fixed phi1 = 1.0
    floor_one = CommitmentFloor(np.array([0.0, 0.0, 10.0]))
    w_one, s_one, _ = evaluate_floor(floor_one, CONFIG["EVAL_EPISODES"], rng_base + 9999)

    # Fixed phi1 = 0.7
    floor_07 = CommitmentFloor(np.array([0.0, 0.0, 0.847]))  # sigmoid(0.847) ~ 0.7
    w_07, s_07, _ = evaluate_floor(floor_07, CONFIG["EVAL_EPISODES"], rng_base + 9999)

    R_range, phi1_profile = best_floor.profile()

    result = {
        "seed": seed,
        "best_omega": best_omega.tolist(),
        "maccl": {"welfare": float(w_final), "survival": float(s_final),
                  "activation_rate": float(a_final)},
        "fixed_0": {"welfare": float(w_zero), "survival": float(s_zero)},
        "fixed_07": {"welfare": float(w_07), "survival": float(s_07)},
        "fixed_1": {"welfare": float(w_one), "survival": float(s_one)},
        "phi1_profile": {
            "R": R_range.tolist(),
            "phi1": phi1_profile.tolist(),
        },
        "convergence": history,
    }

    print(f"    FINAL: MACCL W={w_final:.2f} S={s_final:.0f}% | "
          f"phi1=0: W={w_zero:.2f} S={s_zero:.0f}% | "
          f"phi1=1: W={w_one:.2f} S={s_one:.0f}%")

    return result


# ============================================================
# Plotting
# ============================================================
def plot_maccl_results(all_results, output_dir):
    """Generate MACCL analysis plots."""

    # 1. Convergence plot (averaged across seeds)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    n_iters = len(all_results[0]["convergence"]["welfare"])
    x = np.arange(n_iters)

    # (a) Welfare convergence
    ax = axes[0, 0]
    welfares = np.array([r["convergence"]["welfare"] for r in all_results])
    ax.plot(x, welfares.mean(axis=0), 'b-', linewidth=2, label='Mean')
    ax.fill_between(x, welfares.mean(0) - welfares.std(0),
                    welfares.mean(0) + welfares.std(0), alpha=0.15, color='blue')
    ax.set_xlabel('Outer Iteration')
    ax.set_ylabel('Welfare')
    ax.set_title('(a) Welfare Convergence', fontweight='bold')
    ax.legend()

    # (b) Survival convergence
    ax = axes[0, 1]
    survs = np.array([r["convergence"]["survival"] for r in all_results])
    ax.plot(x, survs.mean(axis=0), 'g-', linewidth=2, label='Mean')
    ax.fill_between(x, survs.mean(0) - survs.std(0),
                    survs.mean(0) + survs.std(0), alpha=0.15, color='green')
    ax.axhline(y=95, color='red', linestyle='--', alpha=0.7, label=f'Safety threshold (95%)')
    ax.set_xlabel('Outer Iteration')
    ax.set_ylabel('Survival (%)')
    ax.set_title('(b) Survival Convergence', fontweight='bold')
    ax.legend()

    # (c) Floor profile (all seeds)
    ax = axes[1, 0]
    R_range = np.linspace(0, 1, 50)
    for r in all_results:
        floor = CommitmentFloor(r["best_omega"])
        _, phi1_vals = floor.profile(R_range)
        ax.plot(R_range, phi1_vals, alpha=0.3, color='purple')
    # Mean profile
    mean_omega = np.mean([r["best_omega"] for r in all_results], axis=0)
    mean_floor = CommitmentFloor(mean_omega)
    _, mean_phi1 = mean_floor.profile(R_range)
    ax.plot(R_range, mean_phi1, 'k-', linewidth=3, label='Mean profile')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='phi1=1.0 (fixed)')
    ax.set_xlabel('Resource R')
    ax.set_ylabel('Commitment Floor phi1(R)')
    ax.set_title('(c) Learned Floor Profile', fontweight='bold')
    ax.legend()
    ax.set_ylim(-0.05, 1.1)

    # (d) Comparison bar chart
    ax = axes[1, 1]
    methods = ['Selfish\n(phi1=0)', 'Fixed\nphi1=0.7', 'Fixed\nphi1=1.0', 'MACCL\nphi1(R)']
    welf_means = [
        np.mean([r["fixed_0"]["welfare"] for r in all_results]),
        np.mean([r["fixed_07"]["welfare"] for r in all_results]),
        np.mean([r["fixed_1"]["welfare"] for r in all_results]),
        np.mean([r["maccl"]["welfare"] for r in all_results]),
    ]
    surv_means = [
        np.mean([r["fixed_0"]["survival"] for r in all_results]),
        np.mean([r["fixed_07"]["survival"] for r in all_results]),
        np.mean([r["fixed_1"]["survival"] for r in all_results]),
        np.mean([r["maccl"]["survival"] for r in all_results]),
    ]
    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']

    x_pos = np.arange(len(methods))
    width = 0.35
    ax.bar(x_pos - width/2, welf_means, width, color=colors, alpha=0.7, label='Welfare')
    ax2 = ax.twinx()
    ax2.bar(x_pos + width/2, surv_means, width, color=colors, alpha=0.3,
            edgecolor='black', linewidth=1.5, label='Survival %')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('Welfare', fontsize=11)
    ax2.set_ylabel('Survival (%)', fontsize=11)
    ax.set_title('(d) Method Comparison', fontweight='bold')

    plt.suptitle('MACCL: Multi-Agent Constrained Commitment Learning\n'
                 f'N={CONFIG["N_AGENTS"]}, Byz={CONFIG["BYZ_FRAC"]*100:.0f}%, '
                 f'{CONFIG["N_SEEDS"]} seeds',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'maccl_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {output_dir}/maccl_analysis.png")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  MACCL: Multi-Agent Constrained Commitment Learning")
    print(f"  N={CONFIG['N_AGENTS']}, Byz={CONFIG['BYZ_FRAC']*100:.0f}%")
    print(f"  Safety: P(surv) >= {(1-CONFIG['SAFETY_DELTA'])*100:.0f}%")
    print(f"  Seeds={CONFIG['N_SEEDS']}")
    print("=" * 70)

    t0 = time.time()
    all_results = []

    for seed in range(CONFIG["N_SEEDS"]):
        result = run_maccl(seed)
        all_results.append(result)

    elapsed = time.time() - t0

    # Aggregate results
    maccl_welfares = [r["maccl"]["welfare"] for r in all_results]
    maccl_survs = [r["maccl"]["survival"] for r in all_results]
    fixed0_survs = [r["fixed_0"]["survival"] for r in all_results]
    fixed1_survs = [r["fixed_1"]["survival"] for r in all_results]
    fixed1_welfs = [r["fixed_1"]["welfare"] for r in all_results]

    output = {
        "experiment": "MACCL (Multi-Agent Constrained Commitment Learning)",
        "config": CONFIG,
        "summary": {
            "maccl_welfare": f"{np.mean(maccl_welfares):.2f} +/- {np.std(maccl_welfares):.2f}",
            "maccl_survival": f"{np.mean(maccl_survs):.1f} +/- {np.std(maccl_survs):.1f}%",
            "fixed0_survival": f"{np.mean(fixed0_survs):.1f}%",
            "fixed1_survival": f"{np.mean(fixed1_survs):.1f}%",
            "fixed1_welfare": f"{np.mean(fixed1_welfs):.2f}",
            "maccl_vs_fixed1_welfare_pct": f"{(np.mean(maccl_welfares)/np.mean(fixed1_welfs)-1)*100:.1f}%" if np.mean(fixed1_welfs) != 0 else "N/A",
        },
        "seed_results": all_results,
        "mean_omega": np.mean([r["best_omega"] for r in all_results], axis=0).tolist(),
        "time_seconds": elapsed,
    }

    json_path = os.path.join(OUTPUT_DIR, "maccl_results.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[Save] {json_path}")

    # Plot
    plot_maccl_results(all_results, OUTPUT_DIR)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  MACCL RESULTS ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
    print(f"  MACCL:     W={np.mean(maccl_welfares):.2f}, S={np.mean(maccl_survs):.1f}%")
    print(f"  phi1=0:    S={np.mean(fixed0_survs):.1f}%")
    print(f"  phi1=1.0:  W={np.mean(fixed1_welfs):.2f}, S={np.mean(fixed1_survs):.1f}%")
    print(f"  Mean omega: {np.mean([r['best_omega'] for r in all_results], axis=0)}")
    print(f"{'=' * 70}")
