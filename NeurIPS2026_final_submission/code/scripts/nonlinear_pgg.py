"""
Track C-2: Non-linear Resource Dynamics with Tipping Points

Tests whether phi1=1.0 (unconditional commitment) prevents irreversible
system collapse in environments with:
  1. Tipping Point: R < R_crit => recovery rate drops to near-zero
  2. Hysteresis: Recovery requires R > R_crit + delta (harder to restore)
  3. Stochastic Shocks: Random resource crises every ~30 rounds

Comparison: phi1=1.0 (learned) vs phi1=0.21 (handcrafted) vs phi1=0.0 (selfish)
"""

import numpy as np
import json
import os
import time

# ============================================================
# Constants
# ============================================================
N_AGENTS = 50
T_ROUNDS = 300   # Longer episodes to observe collapse/recovery
N_SEEDS = 20     # More seeds for statistical power
MULTIPLIER = 1.6
ENDOWMENT = 20.0
ALPHA_EMA = 0.6

# Non-linear parameters
R_CRIT = 0.15          # Tipping point threshold
R_RECOVERY = 0.25      # Hysteresis: must exceed this to resume normal recovery
SHOCK_PROB = 0.033      # ~1 shock per 30 rounds
SHOCK_MAGNITUDE = 0.3   # R drops by this amount during shock


# ============================================================
# g functions
# ============================================================
def make_g(phi1):
    """Create g function with specified crisis commitment level."""
    def g(theta, R):
        phi = [1.297, phi1, 0.004, 0.307, 1.652, 0.405]
        svo = 1.0 / (1.0 + np.exp(-phi[0] * np.sin(theta)))
        if R < phi[3]:
            raw = phi[1] + phi[2] * R
        elif R > (1.0 - phi[3]):
            raw = phi[4] + phi[5] * R
        else:
            raw = phi[1] + (phi[4] - phi[1]) * (R - phi[3]) / max(1.0 - 2*phi[3], 0.01)
        return float(np.clip(svo * raw, 0.0, 1.0))
    return g


# ============================================================
# Non-linear Resource Dynamics
# ============================================================
def resource_update_nonlinear(R_t, coop_ratio, rng, model="tipping"):
    """Non-linear resource update with tipping point and hysteresis.

    Models:
      - "linear": Standard linear recovery (baseline)
      - "tipping": Recovery rate collapses below R_crit
      - "hysteresis": Tipping + harder recovery (must exceed R_recovery)
      - "shock": Hysteresis + random exogenous shocks
    """
    base_recovery = 0.1 * (coop_ratio - 0.4)

    if model == "linear":
        R_new = R_t + base_recovery

    elif model == "tipping":
        if R_t < R_CRIT:
            # Below tipping point: recovery rate is 10% of normal
            R_new = R_t + base_recovery * 0.1
        else:
            R_new = R_t + base_recovery

    elif model == "hysteresis":
        if R_t < R_CRIT:
            # Below tipping: near-zero recovery
            R_new = R_t + base_recovery * 0.1
        elif R_t < R_RECOVERY:
            # In hysteresis band: slow recovery (30% of normal)
            R_new = R_t + base_recovery * 0.3
        else:
            R_new = R_t + base_recovery

    elif model == "shock":
        # Hysteresis + random shocks
        if R_t < R_CRIT:
            R_new = R_t + base_recovery * 0.1
        elif R_t < R_RECOVERY:
            R_new = R_t + base_recovery * 0.3
        else:
            R_new = R_t + base_recovery

        # Exogenous shock
        if rng.random() < SHOCK_PROB:
            R_new -= SHOCK_MAGNITUDE

    else:
        raise ValueError(f"Unknown model: {model}")

    return float(np.clip(R_new, 0.0, 1.0))


# ============================================================
# PGG with non-linear dynamics
# ============================================================
def run_pgg_nonlinear(g_func, resource_model="tipping",
                      n_agents=N_AGENTS, t_rounds=T_ROUNDS,
                      byz_frac=0.0, seed=42):
    rng = np.random.RandomState(seed)
    svo_angles = rng.uniform(np.radians(20), np.radians(70), n_agents)
    n_byz = int(n_agents * byz_frac)

    R_t = 0.5
    lambdas = np.array([g_func(svo_angles[i], R_t) for i in range(n_agents)])

    welfare_hist = []
    resource_hist = [R_t]
    collapse_rounds = 0       # Rounds where R < R_crit
    min_resource = R_t
    recovery_events = 0       # Times R crossed back above R_recovery
    was_below_crit = False

    for t in range(t_rounds):
        contributions = np.zeros(n_agents)
        for i in range(n_agents):
            if i < n_byz:
                contributions[i] = 0.0
            else:
                contributions[i] = ENDOWMENT * lambdas[i]

        total_contrib = contributions.sum()
        public_good = (total_contrib * MULTIPLIER) / n_agents
        payoffs = (ENDOWMENT - contributions) + public_good
        welfare_hist.append(float(payoffs.mean()))

        coop_ratio = np.mean(contributions) / ENDOWMENT
        R_t = resource_update_nonlinear(R_t, coop_ratio, rng, resource_model)
        resource_hist.append(R_t)

        if R_t < R_CRIT:
            collapse_rounds += 1
            was_below_crit = True
        elif was_below_crit and R_t > R_RECOVERY:
            recovery_events += 1
            was_below_crit = False

        min_resource = min(min_resource, R_t)

        for i in range(n_byz, n_agents):
            target = g_func(svo_angles[i], R_t)
            lambdas[i] = ALPHA_EMA * lambdas[i] + (1 - ALPHA_EMA) * target

    resource_arr = np.array(resource_hist)
    return {
        "welfare": float(np.mean(welfare_hist)),
        "final_resource": float(resource_arr[-1]),
        "min_resource": float(min_resource),
        "collapse_rounds": int(collapse_rounds),
        "collapse_fraction": float(collapse_rounds / t_rounds),
        "recovery_events": int(recovery_events),
        "system_alive": float(resource_arr[-1] > R_CRIT),
        "avg_resource": float(np.mean(resource_arr)),
    }


# ============================================================
# Main experiments
# ============================================================
def run_experiment(phi1_values, resource_models, byz_fracs, seeds):
    results = {}

    for model in resource_models:
        results[model] = {}
        for phi1 in phi1_values:
            g = make_g(phi1)
            phi_key = f"phi1_{phi1:.2f}"
            results[model][phi_key] = {}

            for bf in byz_fracs:
                bf_key = f"byz_{int(bf*100)}"
                runs = []
                for seed in seeds:
                    r = run_pgg_nonlinear(g, resource_model=model,
                                          byz_frac=bf, seed=seed)
                    runs.append(r)

                # Aggregate
                agg = {}
                for key in runs[0]:
                    vals = [r[key] for r in runs]
                    agg[key] = float(np.mean(vals))
                    agg[f"{key}_se"] = float(np.std(vals) / np.sqrt(len(vals)))

                results[model][phi_key][bf_key] = agg

    return results


if __name__ == "__main__":
    OUT = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'nonlinear_pgg')
    os.makedirs(OUT, exist_ok=True)

    phi1_values = [0.0, 0.21, 0.5, 1.0]
    resource_models = ["linear", "tipping", "hysteresis", "shock"]
    byz_fracs = [0.0, 0.3, 0.5]
    seeds = list(range(N_SEEDS))

    t0 = time.time()

    print("=" * 70)
    print("  Track C-2: Non-linear Resource Dynamics with Tipping Points")
    print(f"  {len(phi1_values)} phi1 x {len(resource_models)} models x "
          f"{len(byz_fracs)} byz x {N_SEEDS} seeds = "
          f"{len(phi1_values)*len(resource_models)*len(byz_fracs)*N_SEEDS} runs")
    print("=" * 70)

    results = run_experiment(phi1_values, resource_models, byz_fracs, seeds)
    total_time = time.time() - t0

    # Print key results
    print(f"\n  Completed in {total_time:.1f}s\n")

    for model in resource_models:
        print(f"\n  === Resource Model: {model.upper()} ===")
        print(f"  {'phi1':>6} | {'Byz%':>5} | {'Welfare':>8} | {'Collapse%':>9} | "
              f"{'MinR':>6} | {'Recovery':>8} | {'Alive%':>7}")
        print(f"  {'-'*65}")

        for phi1 in phi1_values:
            phi_key = f"phi1_{phi1:.2f}"
            for bf in byz_fracs:
                bf_key = f"byz_{int(bf*100)}"
                r = results[model][phi_key][bf_key]
                print(f"  {phi1:6.2f} | {bf*100:5.0f} | {r['welfare']:8.2f} | "
                      f"{r['collapse_fraction']*100:8.1f}% | {r['min_resource']:6.3f} | "
                      f"{r['recovery_events']:8.1f} | {r['system_alive']*100:6.1f}%")

    # Save
    output = {
        "config": {
            "n_agents": N_AGENTS,
            "t_rounds": T_ROUNDS,
            "n_seeds": N_SEEDS,
            "R_crit": R_CRIT,
            "R_recovery": R_RECOVERY,
            "shock_prob": SHOCK_PROB,
            "shock_magnitude": SHOCK_MAGNITUDE,
            "phi1_values": phi1_values,
            "resource_models": resource_models,
            "byz_fracs": byz_fracs,
        },
        "results": results,
        "time_seconds": float(total_time),
    }

    path = os.path.join(OUT, "nonlinear_results.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Results: {path}")
    print("\n  Track C-2 COMPLETE!")
