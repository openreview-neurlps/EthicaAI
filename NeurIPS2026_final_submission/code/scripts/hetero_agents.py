"""
Heterogeneous Agent Experiment for MACCL
==========================================
Tests whether MACCL learns differentiated commitment floors
for agents with different endowments.

Key hypothesis: Low-endowment agents need higher floors because
they contribute less per unit of lambda; if MACCL learns
group-specific floors, this refutes "floors are one-size-fits-all".

Setup:
  - N=20 agents, 3 groups:
      Low  (E=10, 7 agents)
      Med  (E=20, 7 agents)
      High (E=30, 6 agents)
  - Nonlinear PGG with tipping point
  - 30% Byzantine fraction
  - MACCL with per-group floor parameters

Dependencies: NumPy, matplotlib.
"""

import numpy as np
import json
import os
import time
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    "N_AGENTS": 20,
    "GROUPS": OrderedDict([
        ("low",  {"endowment": 10.0, "count": 7}),
        ("med",  {"endowment": 20.0, "count": 7}),
        ("high", {"endowment": 30.0, "count": 6}),
    ]),
    "MULTIPLIER": 1.6,
    "T_HORIZON": 50,
    "R_CRIT": 0.15,
    "R_RECOV": 0.25,
    "SHOCK_PROB": 0.05,
    "SHOCK_MAG": 0.15,
    "GAMMA": 0.99,
    "BYZ_FRAC": 0.30,
    # MACCL-specific
    "TRAIN_EPISODES": 200,
    "EVAL_EPISODES": 50,
    "PHASE2_OUTER_ITERS": 100,
    "PHASE2_INNER_EPISODES": 10,
    "N_SEEDS": 20,
    "SAFETY_DELTA": 0.05,
    "SAFETY_DELTA_MIN": 0.10,
    "LR_OMEGA": 0.08,
    "LR_MU": 0.10,
    "FD_EPSILON": 0.02,
}

FAST_MODE = os.environ.get("ETHICAAI_FAST") == "1"
if FAST_MODE:
    print("  [FAST MODE]")
    CONFIG["N_SEEDS"] = 3
    CONFIG["TRAIN_EPISODES"] = 30
    CONFIG["EVAL_EPISODES"] = 15
    CONFIG["PHASE2_OUTER_ITERS"] = 30
    CONFIG["PHASE2_INNER_EPISODES"] = 5

N = CONFIG["N_AGENTS"]
N_BYZ = int(N * CONFIG["BYZ_FRAC"])
N_HONEST = N - N_BYZ
M = CONFIG["MULTIPLIER"]

# Build agent-to-group mapping
# Byzantine agents are first N_BYZ indices (assigned proportionally across groups)
AGENT_ENDOWMENTS = np.zeros(N)
AGENT_GROUP = [""] * N
GROUP_NAMES = list(CONFIG["GROUPS"].keys())

_idx = 0
for gname, ginfo in CONFIG["GROUPS"].items():
    for _ in range(ginfo["count"]):
        AGENT_ENDOWMENTS[_idx] = ginfo["endowment"]
        AGENT_GROUP[_idx] = gname
        _idx += 1

# Byzantines are the first N_BYZ agents (spread across groups as they fall)
BYZ_MASK = np.zeros(N, dtype=bool)
BYZ_MASK[:N_BYZ] = True

# Honest agent indices per group
GROUP_HONEST_INDICES = {}
for gname in GROUP_NAMES:
    GROUP_HONEST_INDICES[gname] = [
        i for i in range(N) if AGENT_GROUP[i] == gname and not BYZ_MASK[i]
    ]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'outputs', 'hetero_agents')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Environment
# ============================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def reset_env():
    return {"R": 0.5, "mean_c": 0.5}


def env_step(R, coop_rate, rng):
    """Nonlinear PGG resource dynamics with tipping point."""
    RC = CONFIG["R_CRIT"]
    RR = CONFIG["R_RECOV"]
    if R < RC:
        f = 0.01
    elif R < RR:
        f = 0.03
    else:
        f = 0.10
    shock = CONFIG["SHOCK_MAG"] if rng.random() < CONFIG["SHOCK_PROB"] else 0.0
    return float(np.clip(R + f * (coop_rate - 0.4) - shock, 0.0, 1.0))


def step_env(env, lambdas, rng):
    """Full environment step with heterogeneous endowments."""
    lam = lambdas.copy()
    lam[BYZ_MASK] = 0.0

    contribs = AGENT_ENDOWMENTS * lam
    public_good = M * contribs.sum() / N
    rewards = (AGENT_ENDOWMENTS - contribs) + public_good

    # Cooperation rate: weighted by endowment
    total_possible = AGENT_ENDOWMENTS.sum()
    coop_rate = contribs.sum() / total_possible

    R = env["R"]
    R_new = env_step(R, coop_rate, rng)
    done = R_new <= 0.001

    env["R"] = R_new
    env["mean_c"] = float(coop_rate)

    return rewards, done


# ============================================================
# Per-Group Commitment Floor
# ============================================================
class GroupCommitmentFloor:
    """
    State-dependent commitment floor per group:
      phi1_g(R; omega_g) = sigmoid(w1*R + w2*R^2 + w3)

    Each group has its own omega_g = (w1, w2, w3).
    """
    def __init__(self, group_omegas=None):
        if group_omegas is None:
            # Initialize all groups to high commitment
            self.omegas = {g: np.array([0.0, 0.0, 5.0]) for g in GROUP_NAMES}
        else:
            self.omegas = {g: np.array(v, dtype=np.float64)
                           for g, v in group_omegas.items()}

    def __call__(self, R, group):
        """Compute floor for resource R and group name."""
        omega = self.omegas[group]
        z = omega[0] * R + omega[1] * R**2 + omega[2]
        return float(sigmoid(z))

    def all_omegas_flat(self):
        """Return all omegas as a single flat array (for gradient computation)."""
        return np.concatenate([self.omegas[g] for g in GROUP_NAMES])

    def set_from_flat(self, flat):
        """Set omegas from a flat array."""
        idx = 0
        for g in GROUP_NAMES:
            self.omegas[g] = flat[idx:idx+3].copy()
            idx += 3

    def copy(self):
        return GroupCommitmentFloor({g: v.copy() for g, v in self.omegas.items()})

    def profile(self, R_range=None):
        """Return floor values per group across R range."""
        if R_range is None:
            R_range = np.linspace(0, 1, 50)
        profiles = {}
        for g in GROUP_NAMES:
            profiles[g] = np.array([self(R, g) for R in R_range])
        return R_range, profiles


# ============================================================
# Evaluate a group commitment floor
# ============================================================
def evaluate_floor(floor_fn, n_episodes, rng_seed):
    """Run episodes with group-specific floors.
    Returns per-group welfare, overall survival, activation rates."""
    rng = np.random.RandomState(rng_seed)

    total_welfare_by_group = {g: 0.0 for g in GROUP_NAMES}
    total_welfare = 0.0
    total_survival = 0
    floor_activations_by_group = {g: 0 for g in GROUP_NAMES}
    total_steps_by_group = {g: 0 for g in GROUP_NAMES}

    for ep in range(n_episodes):
        env = reset_env()
        ep_welfare = 0.0
        ep_welfare_by_group = {g: 0.0 for g in GROUP_NAMES}
        steps = 0
        survived = True

        for t in range(CONFIG["T_HORIZON"]):
            R = env["R"]

            lambdas = np.zeros(N)
            for i in range(N):
                if BYZ_MASK[i]:
                    lambdas[i] = 0.0
                    continue
                g = AGENT_GROUP[i]
                phi1 = floor_fn(R, g)
                base_lam = 0.5 + rng.randn() * 0.1
                base_lam = np.clip(base_lam, 0.01, 0.99)
                if base_lam < phi1:
                    lambdas[i] = phi1
                    floor_activations_by_group[g] += 1
                else:
                    lambdas[i] = base_lam
                total_steps_by_group[g] += 1

            rewards, done = step_env(env, lambdas, rng)
            ep_welfare += rewards.mean()
            for g in GROUP_NAMES:
                idxs = GROUP_HONEST_INDICES[g]
                if idxs:
                    ep_welfare_by_group[g] += np.mean(rewards[idxs])
            steps += 1

            if done:
                survived = False
                break

        denom = max(steps, 1)
        total_welfare += ep_welfare / denom
        for g in GROUP_NAMES:
            total_welfare_by_group[g] += ep_welfare_by_group[g] / denom
        total_survival += int(survived)

    welfare = total_welfare / n_episodes
    welfare_by_group = {g: total_welfare_by_group[g] / n_episodes for g in GROUP_NAMES}
    survival = total_survival / n_episodes * 100
    activation_by_group = {
        g: floor_activations_by_group[g] / max(total_steps_by_group[g], 1)
        for g in GROUP_NAMES
    }

    return welfare, welfare_by_group, survival, activation_by_group


# ============================================================
# Uniform floor (single floor for all groups, for baseline)
# ============================================================
class UniformFloor:
    """A floor that applies the same value to all groups."""
    def __init__(self, phi=1.0):
        self.phi = phi
    def __call__(self, R, group):
        return self.phi


# ============================================================
# MACCL with Heterogeneous Floors
# ============================================================
def run_hetero_maccl(seed):
    """
    Run MACCL with per-group floor optimization.

    Phase 1: Safety Anchoring (all floors ~ 1.0)
    Phase 2: Primal-dual constrained optimization with per-group omega
    Phase 3: Final evaluation + comparisons
    """
    print(f"\n  [Seed {seed}] Starting Hetero-MACCL")
    rng_base = 1000 * seed

    # ---- Phase 1: Safety Anchoring ----
    floor_safe = GroupCommitmentFloor()  # Default: all sigmoid(5) ~ 0.993
    for g in GROUP_NAMES:
        floor_safe.omegas[g] = np.array([0.0, 0.0, 10.0])

    w_safe, wg_safe, s_safe, _ = evaluate_floor(
        floor_safe, max(CONFIG["PHASE2_INNER_EPISODES"], 10), rng_base
    )
    print(f"    Phase 1 (all phi1~1.0): W={w_safe:.2f}, S={s_safe:.0f}%")

    # ---- Phase 2: Constrained Floor Learning ----
    # Start from moderate floor (~0.7) so optimizer can push groups apart
    floor = GroupCommitmentFloor(
        {g: np.array([0.0, 0.0, 0.847]) for g in GROUP_NAMES}  # sigmoid(0.847)~0.7
    )
    mu = 1.0  # Lagrange multiplier
    delta = CONFIG["SAFETY_DELTA"]
    lr_omega = CONFIG["LR_OMEGA"]
    lr_mu = CONFIG["LR_MU"]
    fd_eps = CONFIG["FD_EPSILON"]

    n_params = 3 * len(GROUP_NAMES)  # 3 params per group

    history = {
        "welfare": [],
        "survival": [],
        "mu": [],
    }
    for g in GROUP_NAMES:
        history[f"phi1_{g}_at_R0"] = []
        history[f"phi1_{g}_at_R05"] = []
        history[f"phi1_{g}_at_R1"] = []
        history[f"omega_{g}"] = []

    best_flat = floor.all_omegas_flat()
    best_welfare = -np.inf
    best_survival = 0.0

    for outer in range(CONFIG["PHASE2_OUTER_ITERS"]):
        n_inner = CONFIG["PHASE2_INNER_EPISODES"]
        eval_seed = rng_base + outer

        # Evaluate current floor
        w_curr, wg_curr, s_curr, act_curr = evaluate_floor(floor, n_inner, eval_seed)

        # Finite-difference gradient over all group parameters
        flat = floor.all_omegas_flat()
        grad = np.zeros(n_params)

        for dim in range(n_params):
            flat_plus = flat.copy()
            flat_plus[dim] += fd_eps
            floor_plus = floor.copy()
            floor_plus.set_from_flat(flat_plus)
            w_plus, _, s_plus, _ = evaluate_floor(floor_plus, n_inner, eval_seed)

            flat_minus = flat.copy()
            flat_minus[dim] -= fd_eps
            floor_minus = floor.copy()
            floor_minus.set_from_flat(flat_minus)
            w_minus, _, s_minus, _ = evaluate_floor(floor_minus, n_inner, eval_seed)

            # Lagrangian: L = W + mu * (P_surv - (1 - delta))
            L_plus = w_plus + mu * (s_plus / 100.0 - (1 - delta))
            L_minus = w_minus + mu * (s_minus / 100.0 - (1 - delta))
            grad[dim] = (L_plus - L_minus) / (2 * fd_eps)

        # Primal update
        new_flat = flat + lr_omega * grad

        # Safety projection
        floor_candidate = floor.copy()
        floor_candidate.set_from_flat(new_flat)
        _, _, s_cand, _ = evaluate_floor(floor_candidate, n_inner, eval_seed)

        if s_cand >= (1 - CONFIG["SAFETY_DELTA_MIN"]) * 100:
            floor.set_from_flat(new_flat)
        # else: keep current

        # Dual update
        constraint_violation = delta - s_curr / 100.0
        mu = max(0.0, mu + lr_mu * constraint_violation)

        # Track best
        if s_curr >= (1 - delta) * 100 and w_curr > best_welfare:
            best_flat = floor.all_omegas_flat()
            best_welfare = w_curr
            best_survival = s_curr

        # Log
        history["welfare"].append(float(w_curr))
        history["survival"].append(float(s_curr))
        history["mu"].append(float(mu))
        for g in GROUP_NAMES:
            history[f"phi1_{g}_at_R0"].append(float(floor(0.0, g)))
            history[f"phi1_{g}_at_R05"].append(float(floor(0.5, g)))
            history[f"phi1_{g}_at_R1"].append(float(floor(1.0, g)))
            history[f"omega_{g}"].append(floor.omegas[g].tolist())

        if (outer + 1) % max(1, CONFIG["PHASE2_OUTER_ITERS"] // 5) == 0:
            floor_strs = ", ".join(
                f"{g}={floor(0.5, g):.3f}" for g in GROUP_NAMES
            )
            print(f"    Phase 2 [{outer+1}/{CONFIG['PHASE2_OUTER_ITERS']}]: "
                  f"W={w_curr:.2f}, S={s_curr:.0f}%, mu={mu:.3f}, "
                  f"phi1(R=0.5): {floor_strs}")

    # ---- Phase 3: Final Evaluation ----
    best_floor = GroupCommitmentFloor()
    best_floor.set_from_flat(best_flat)

    n_eval = CONFIG["EVAL_EPISODES"]
    eval_seed = rng_base + 9999

    w_final, wg_final, s_final, act_final = evaluate_floor(
        best_floor, n_eval, eval_seed
    )

    # Baselines: uniform floors
    w_zero, wg_zero, s_zero, _ = evaluate_floor(
        UniformFloor(0.0), n_eval, eval_seed
    )
    w_one, wg_one, s_one, _ = evaluate_floor(
        UniformFloor(1.0), n_eval, eval_seed
    )
    w_07, wg_07, s_07, _ = evaluate_floor(
        UniformFloor(0.7), n_eval, eval_seed
    )

    # Floor profiles
    R_range = np.linspace(0, 1, 50)
    _, profiles = best_floor.profile(R_range)

    result = {
        "seed": seed,
        "best_omegas": {g: best_floor.omegas[g].tolist() for g in GROUP_NAMES},
        "maccl": {
            "welfare": float(w_final),
            "welfare_by_group": {g: float(v) for g, v in wg_final.items()},
            "survival": float(s_final),
            "activation_by_group": {g: float(v) for g, v in act_final.items()},
        },
        "uniform_0": {
            "welfare": float(w_zero), "survival": float(s_zero),
            "welfare_by_group": {g: float(v) for g, v in wg_zero.items()},
        },
        "uniform_07": {
            "welfare": float(w_07), "survival": float(s_07),
            "welfare_by_group": {g: float(v) for g, v in wg_07.items()},
        },
        "uniform_1": {
            "welfare": float(w_one), "survival": float(s_one),
            "welfare_by_group": {g: float(v) for g, v in wg_one.items()},
        },
        "floor_at_R05": {g: float(best_floor(0.5, g)) for g in GROUP_NAMES},
        "floor_profiles": {
            "R": R_range.tolist(),
            **{g: profiles[g].tolist() for g in GROUP_NAMES},
        },
        "convergence": history,
    }

    floor_strs = ", ".join(f"{g}={best_floor(0.5, g):.3f}" for g in GROUP_NAMES)
    print(f"    FINAL: MACCL W={w_final:.2f} S={s_final:.0f}% | "
          f"phi1(0.5): {floor_strs} | "
          f"Uniform-1: W={w_one:.2f} S={s_one:.0f}%")

    return result


# ============================================================
# Plotting
# ============================================================
def plot_hetero_results(all_results, output_dir):
    """Generate analysis plots for heterogeneous agent experiment."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = {"low": "#e74c3c", "med": "#f39c12", "high": "#2ecc71"}

    n_iters = len(all_results[0]["convergence"]["welfare"])
    x = np.arange(n_iters)

    # (a) Welfare convergence
    ax = axes[0, 0]
    welfares = np.array([r["convergence"]["welfare"] for r in all_results])
    ax.plot(x, welfares.mean(axis=0), 'b-', linewidth=2)
    ax.fill_between(x, welfares.mean(0) - welfares.std(0),
                    welfares.mean(0) + welfares.std(0), alpha=0.15, color='blue')
    ax.set_xlabel('Outer Iteration')
    ax.set_ylabel('Welfare')
    ax.set_title('(a) Welfare Convergence', fontweight='bold')

    # (b) Survival convergence
    ax = axes[0, 1]
    survs = np.array([r["convergence"]["survival"] for r in all_results])
    ax.plot(x, survs.mean(axis=0), 'g-', linewidth=2)
    ax.fill_between(x, survs.mean(0) - survs.std(0),
                    survs.mean(0) + survs.std(0), alpha=0.15, color='green')
    ax.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='Safety (95%)')
    ax.set_xlabel('Outer Iteration')
    ax.set_ylabel('Survival (%)')
    ax.set_title('(b) Survival Convergence', fontweight='bold')
    ax.legend()

    # (c) Floor evolution at R=0.5 per group
    ax = axes[0, 2]
    for g in GROUP_NAMES:
        key = f"phi1_{g}_at_R05"
        vals = np.array([r["convergence"][key] for r in all_results])
        E_g = CONFIG["GROUPS"][g]["endowment"]
        ax.plot(x, vals.mean(axis=0), color=colors[g], linewidth=2,
                label=f'{g.capitalize()} (E={E_g:.0f})')
        ax.fill_between(x, vals.mean(0) - vals.std(0),
                        vals.mean(0) + vals.std(0), alpha=0.1, color=colors[g])
    ax.set_xlabel('Outer Iteration')
    ax.set_ylabel('Floor phi1(R=0.5)')
    ax.set_title('(c) Floor Evolution per Group', fontweight='bold')
    ax.legend()
    ax.set_ylim(-0.05, 1.1)

    # (d) Final floor profiles per group (all seeds + mean)
    ax = axes[1, 0]
    R_range = np.linspace(0, 1, 50)
    for g in GROUP_NAMES:
        E_g = CONFIG["GROUPS"][g]["endowment"]
        all_profiles = []
        for r in all_results:
            floor = GroupCommitmentFloor(
                {gn: np.array(r["best_omegas"][gn]) for gn in GROUP_NAMES}
            )
            _, profs = floor.profile(R_range)
            all_profiles.append(profs[g])
            ax.plot(R_range, profs[g], alpha=0.15, color=colors[g])
        mean_prof = np.mean(all_profiles, axis=0)
        ax.plot(R_range, mean_prof, color=colors[g], linewidth=3,
                label=f'{g.capitalize()} (E={E_g:.0f})')
    ax.set_xlabel('Resource R')
    ax.set_ylabel('Commitment Floor phi1(R)')
    ax.set_title('(d) Learned Floor Profiles', fontweight='bold')
    ax.legend()
    ax.set_ylim(-0.05, 1.1)

    # (e) Floor comparison at R=0.5 (box plot)
    ax = axes[1, 1]
    data = []
    labels = []
    box_colors = []
    for g in GROUP_NAMES:
        vals = [r["floor_at_R05"][g] for r in all_results]
        data.append(vals)
        E_g = CONFIG["GROUPS"][g]["endowment"]
        labels.append(f'{g.capitalize()}\n(E={E_g:.0f})')
        box_colors.append(colors[g])
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel('Floor phi1(R=0.5)')
    ax.set_title('(e) Floor by Group at R=0.5', fontweight='bold')

    # (f) Welfare comparison: MACCL vs uniform baselines
    ax = axes[1, 2]
    methods = ['Uniform\nphi1=0', 'Uniform\nphi1=0.7', 'Uniform\nphi1=1.0',
               'MACCL\n(per-group)']
    welf_means = [
        np.mean([r["uniform_0"]["welfare"] for r in all_results]),
        np.mean([r["uniform_07"]["welfare"] for r in all_results]),
        np.mean([r["uniform_1"]["welfare"] for r in all_results]),
        np.mean([r["maccl"]["welfare"] for r in all_results]),
    ]
    surv_means = [
        np.mean([r["uniform_0"]["survival"] for r in all_results]),
        np.mean([r["uniform_07"]["survival"] for r in all_results]),
        np.mean([r["uniform_1"]["survival"] for r in all_results]),
        np.mean([r["maccl"]["survival"] for r in all_results]),
    ]
    bar_colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
    x_pos = np.arange(len(methods))
    width = 0.35
    ax.bar(x_pos - width/2, welf_means, width, color=bar_colors, alpha=0.7, label='Welfare')
    ax2 = ax.twinx()
    ax2.bar(x_pos + width/2, surv_means, width, color=bar_colors, alpha=0.3,
            edgecolor='black', linewidth=1.5, label='Survival %')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('Welfare')
    ax2.set_ylabel('Survival (%)')
    ax.set_title('(f) Method Comparison', fontweight='bold')

    plt.suptitle('Heterogeneous MACCL: Per-Group Commitment Floors\n'
                 f'N={N}, Groups: Low(E=10,n=7) Med(E=20,n=7) High(E=30,n=6), '
                 f'Byz={CONFIG["BYZ_FRAC"]*100:.0f}%, '
                 f'{CONFIG["N_SEEDS"]} seeds',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hetero_agents_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {output_dir}/hetero_agents_analysis.png")


# ============================================================
# Statistical analysis
# ============================================================
def analyze_differentiation(all_results):
    """Analyze whether floors differ significantly across groups."""
    from itertools import combinations

    floors_at_05 = {g: [] for g in GROUP_NAMES}
    for r in all_results:
        for g in GROUP_NAMES:
            floors_at_05[g].append(r["floor_at_R05"][g])

    analysis = {}
    for g in GROUP_NAMES:
        vals = floors_at_05[g]
        E_g = CONFIG["GROUPS"][g]["endowment"]
        analysis[g] = {
            "endowment": E_g,
            "floor_mean": float(np.mean(vals)),
            "floor_std": float(np.std(vals)),
            "floor_min": float(np.min(vals)),
            "floor_max": float(np.max(vals)),
        }

    # Pairwise comparisons (bootstrap-style)
    pairwise = {}
    for g1, g2 in combinations(GROUP_NAMES, 2):
        v1 = np.array(floors_at_05[g1])
        v2 = np.array(floors_at_05[g2])
        diff = v1 - v2
        mean_diff = float(np.mean(diff))
        se_diff = float(np.std(diff) / max(np.sqrt(len(diff)), 1))
        # Simple t-test approximation
        t_stat = mean_diff / max(se_diff, 1e-8)
        significant = abs(t_stat) > 2.0  # ~95% CI
        pairwise[f"{g1}_vs_{g2}"] = {
            "mean_diff": mean_diff,
            "se": se_diff,
            "t_stat": float(t_stat),
            "significant_at_95": significant,
        }

    # Key test: does low > high? (theory predicts yes)
    low_vals = np.array(floors_at_05["low"])
    high_vals = np.array(floors_at_05["high"])
    low_gt_high_frac = float(np.mean(low_vals > high_vals))

    return {
        "per_group": analysis,
        "pairwise": pairwise,
        "low_floor_gt_high_floor_fraction": low_gt_high_frac,
        "theory_supported": low_gt_high_frac > 0.5,
    }


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    group_desc = ", ".join(
        f"{g.capitalize()}(E={info['endowment']:.0f},n={info['count']})"
        for g, info in CONFIG["GROUPS"].items()
    )
    print("=" * 70)
    print("  Heterogeneous Agent MACCL Experiment")
    print(f"  N={N}, Groups: {group_desc}")
    print(f"  Byz={CONFIG['BYZ_FRAC']*100:.0f}%, Safety >= {(1-CONFIG['SAFETY_DELTA'])*100:.0f}%")
    print(f"  Seeds={CONFIG['N_SEEDS']}, Outer iters={CONFIG['PHASE2_OUTER_ITERS']}")
    print("=" * 70)

    t0 = time.time()
    all_results = []

    for seed in range(CONFIG["N_SEEDS"]):
        result = run_hetero_maccl(seed)
        all_results.append(result)

    elapsed = time.time() - t0

    # Statistical analysis
    differentiation = analyze_differentiation(all_results)

    # Aggregate
    maccl_welfares = [r["maccl"]["welfare"] for r in all_results]
    maccl_survs = [r["maccl"]["survival"] for r in all_results]
    uni1_welfares = [r["uniform_1"]["welfare"] for r in all_results]
    uni1_survs = [r["uniform_1"]["survival"] for r in all_results]

    output = {
        "experiment": "Heterogeneous Agent MACCL",
        "config": {k: v for k, v in CONFIG.items() if k != "GROUPS"},
        "groups": {g: dict(info) for g, info in CONFIG["GROUPS"].items()},
        "summary": {
            "maccl_welfare": f"{np.mean(maccl_welfares):.2f} +/- {np.std(maccl_welfares):.2f}",
            "maccl_survival": f"{np.mean(maccl_survs):.1f} +/- {np.std(maccl_survs):.1f}%",
            "uniform1_welfare": f"{np.mean(uni1_welfares):.2f}",
            "uniform1_survival": f"{np.mean(uni1_survs):.1f}%",
            "maccl_vs_uniform1_welfare_pct": (
                f"{(np.mean(maccl_welfares)/np.mean(uni1_welfares)-1)*100:.1f}%"
                if np.mean(uni1_welfares) != 0 else "N/A"
            ),
        },
        "floor_differentiation": differentiation,
        "mean_floors_at_R05": {
            g: float(np.mean([r["floor_at_R05"][g] for r in all_results]))
            for g in GROUP_NAMES
        },
        "seed_results": all_results,
        "time_seconds": elapsed,
    }

    json_path = os.path.join(OUTPUT_DIR, "hetero_results.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[Save] {json_path}")

    # Plot
    plot_hetero_results(all_results, OUTPUT_DIR)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  HETEROGENEOUS MACCL RESULTS ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
    print(f"  MACCL:      W={np.mean(maccl_welfares):.2f}, S={np.mean(maccl_survs):.1f}%")
    print(f"  Uniform-1:  W={np.mean(uni1_welfares):.2f}, S={np.mean(uni1_survs):.1f}%")
    print(f"\n  Mean floors at R=0.5:")
    for g in GROUP_NAMES:
        E_g = CONFIG["GROUPS"][g]["endowment"]
        mean_f = np.mean([r["floor_at_R05"][g] for r in all_results])
        std_f = np.std([r["floor_at_R05"][g] for r in all_results])
        print(f"    {g.capitalize():>5} (E={E_g:>2.0f}): phi1={mean_f:.3f} +/- {std_f:.3f}")

    print(f"\n  Floor differentiation analysis:")
    for key, val in differentiation["pairwise"].items():
        sig = "YES" if val["significant_at_95"] else "no"
        print(f"    {key}: diff={val['mean_diff']:.4f}, t={val['t_stat']:.2f}, sig={sig}")

    theory = differentiation["theory_supported"]
    frac = differentiation["low_floor_gt_high_floor_fraction"]
    print(f"\n  Theory test (low floor > high floor): {frac*100:.0f}% of seeds")
    print(f"  Theory supported: {theory}")
    print(f"{'=' * 70}")
