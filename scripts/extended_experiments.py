"""
Extended Experiments for Reviewer 2 Response
=============================================
1. Scale test: N=100 non-linear PGG (Nash Trap + Unconditional)
2. Inequity Aversion baseline (same-class decentralized comparison)
3. f(R_t) parameter sensitivity (R_crit, recovery rates)
"""

import numpy as np
import json
import os
import time

# ============================================================
# Core PGG Engine (parameterized)
# ============================================================
def run_experiment(
    n_agents=20, t_rounds=100, multiplier=1.6, endowment=20.0,
    r_crit=0.15, r_recov=0.25, shock_prob=0.05, shock_mag=0.15,
    recovery_rates=(0.01, 0.03, 0.10),
    commitment_fn=None, n_byz=0, n_episodes=50, n_seeds=5,
    label=""
):
    """Run PGG simulation with given commitment function.
    commitment_fn(agent_idx, obs, env_state, rng) -> lambda value
    """
    all_metrics = []

    for seed in range(n_seeds):
        seed_metrics = []
        for ep in range(n_episodes):
            rng = np.random.RandomState(seed * 10000 + ep)
            R = 0.5
            lam_prev = np.full(n_agents, 0.5)
            mean_c = 0.5
            ep_welfare, ep_lam, survived = [], [], True

            for t in range(t_rounds):
                # Get lambdas
                obs = {"R": R, "mean_c": mean_c, "lam_prev": lam_prev}
                lambdas = np.array([
                    commitment_fn(i, obs, rng) for i in range(n_agents)
                ])
                # Byzantine: first n_byz agents always contribute 0
                lambdas[:n_byz] = 0.0

                # Payoffs
                contribs = endowment * lambdas
                public = (contribs.sum() * multiplier) / n_agents
                rewards = (endowment - contribs) + public

                # Resource dynamics
                coop = contribs.mean() / endowment
                f_R = recovery_rates[2]  # normal
                if R < r_crit:
                    f_R = recovery_rates[0]
                elif R < r_recov:
                    f_R = recovery_rates[1]

                R_new = R + f_R * (coop - 0.4)
                if rng.random() < shock_prob:
                    R_new -= shock_mag
                R_new = float(np.clip(R_new, 0.0, 1.0))

                ep_welfare.append(float(rewards.mean()))
                honest_lam = float(lambdas[n_byz:].mean())
                ep_lam.append(honest_lam)
                lam_prev = lambdas.copy()
                mean_c = float(coop)
                R = R_new

                if R <= 0.001:
                    survived = False
                    break

            seed_metrics.append({
                "welfare": float(np.mean(ep_welfare)),
                "mean_lam": float(np.mean(ep_lam)),
                "survived": survived,
            })
        all_metrics.append(seed_metrics)

    # Aggregate
    flat = [m for s in all_metrics for m in s]
    result = {
        "label": label,
        "n_agents": n_agents,
        "n_byz": n_byz,
        "welfare": float(np.mean([m["welfare"] for m in flat])),
        "mean_lam": float(np.mean([m["mean_lam"] for m in flat])),
        "survival": float(np.mean([m["survived"] for m in flat])),
    }
    return result


# ============================================================
# Commitment Functions
# ============================================================
def make_fixed_commitment(lam_val):
    """Fixed lambda for all agents."""
    def fn(i, obs, rng):
        return lam_val
    return fn


def make_selfish_rl_commitment():
    """Simulates the Nash Trap: agents converge to ~0.5."""
    def fn(i, obs, rng):
        return 0.5 + rng.normal(0, 0.02)
    return fn


def make_unconditional_commitment(phi1=1.0):
    """Ethical prior with unconditional crisis commitment."""
    def fn(i, obs, rng):
        R = obs["R"]
        if R < 0.15:
            return phi1
        elif R > 0.5:
            return min(1.0, 0.8)
        else:
            return 0.5 + 0.5 * R
    return fn


def make_inequity_aversion_commitment(alpha=0.5, beta=0.3):
    """
    Fehr-Schmidt Inequity Aversion (1999).
    Agent adjusts lambda based on comparison with mean:
      lambda_i = 0.5 + beta * (mean_lam - lam_prev_i) - alpha * max(0, lam_prev_i - mean_lam)
    Decentralized: only uses observable mean contribution.
    """
    def fn(i, obs, rng):
        my_lam = obs["lam_prev"][i]
        mean_lam = obs["lam_prev"].mean()
        # Inequity aversion adjustment
        adv_disutil = alpha * max(0, my_lam - mean_lam)  # disadvantageous
        dis_adv_util = beta * max(0, mean_lam - my_lam)  # advantageous
        new_lam = my_lam + dis_adv_util - adv_disutil
        return float(np.clip(new_lam + rng.normal(0, 0.02), 0.0, 1.0))
    return fn


def make_social_influence_commitment(influence_weight=0.3):
    """
    Social Influence (Jaques et al., 2019 style).
    Agent moves toward group mean with some weight.
    Decentralized: uses observable mean contribution.
    """
    def fn(i, obs, rng):
        my_lam = obs["lam_prev"][i]
        mean_lam = obs["lam_prev"].mean()
        new_lam = (1 - influence_weight) * my_lam + influence_weight * mean_lam
        return float(np.clip(new_lam + rng.normal(0, 0.02), 0.0, 1.0))
    return fn


# ============================================================
# Main Experiments
# ============================================================
if __name__ == "__main__":
    OUT = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'extended_experiments')
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    print("=" * 70)
    print("  EXTENDED EXPERIMENTS - Reviewer 2 Response")
    print("=" * 70)

    all_results = {}

    # ----------------------------------------------------------
    # Experiment 1: Scale Test (N=100)
    # ----------------------------------------------------------
    print("\n[1/3] SCALE TEST: N=100, Non-linear PGG")
    print("-" * 50)

    for byz_frac in [0.0, 0.3]:
        n_byz = int(100 * byz_frac)
        byz_label = f"byz{int(byz_frac*100)}"

        # Nash Trap (selfish RL ~ fixed 0.5)
        r = run_experiment(
            n_agents=100, n_byz=n_byz, n_episodes=50, n_seeds=5,
            commitment_fn=make_selfish_rl_commitment(),
            label=f"N100_selfish_{byz_label}"
        )
        all_results[r["label"]] = r
        print(f"  Selfish (Byz={byz_frac:.0%}): W={r['welfare']:.1f}, "
              f"λ={r['mean_lam']:.3f}, Surv={r['survival']*100:.1f}%")

        # Unconditional commitment
        r = run_experiment(
            n_agents=100, n_byz=n_byz, n_episodes=50, n_seeds=5,
            commitment_fn=make_unconditional_commitment(phi1=1.0),
            label=f"N100_unconditional_{byz_label}"
        )
        all_results[r["label"]] = r
        print(f"  Unconditional (Byz={byz_frac:.0%}): W={r['welfare']:.1f}, "
              f"λ={r['mean_lam']:.3f}, Surv={r['survival']*100:.1f}%")

        # Situational commitment (crisis -> 0.3)
        def make_situational():
            def fn(i, obs, rng):
                R = obs["R"]
                if R < 0.15:
                    return max(0, 0.3 + rng.normal(0, 0.02))
                elif R > 0.5:
                    return min(1.0, 0.9 + rng.normal(0, 0.02))
                else:
                    return float(np.clip(0.5 + 0.8 * R + rng.normal(0, 0.02), 0, 1))
            return fn

        r = run_experiment(
            n_agents=100, n_byz=n_byz, n_episodes=50, n_seeds=5,
            commitment_fn=make_situational(),
            label=f"N100_situational_{byz_label}"
        )
        all_results[r["label"]] = r
        print(f"  Situational (Byz={byz_frac:.0%}): W={r['welfare']:.1f}, "
              f"λ={r['mean_lam']:.3f}, Surv={r['survival']*100:.1f}%")

    # ----------------------------------------------------------
    # Experiment 2: Same-class Baselines
    # ----------------------------------------------------------
    print("\n[2/3] SAME-CLASS BASELINES: N=20 + N=100")
    print("-" * 50)

    for n_ag in [20, 100]:
        n_byz = int(n_ag * 0.3)
        prefix = f"N{n_ag}"

        # Inequity Aversion
        r = run_experiment(
            n_agents=n_ag, n_byz=n_byz, n_episodes=50, n_seeds=5,
            commitment_fn=make_inequity_aversion_commitment(alpha=0.5, beta=0.3),
            label=f"{prefix}_inequity_aversion_byz30"
        )
        all_results[r["label"]] = r
        print(f"  [{prefix}] Inequity Aversion (Byz=30%): W={r['welfare']:.1f}, "
              f"λ={r['mean_lam']:.3f}, Surv={r['survival']*100:.1f}%")

        # Social Influence
        r = run_experiment(
            n_agents=n_ag, n_byz=n_byz, n_episodes=50, n_seeds=5,
            commitment_fn=make_social_influence_commitment(influence_weight=0.3),
            label=f"{prefix}_social_influence_byz30"
        )
        all_results[r["label"]] = r
        print(f"  [{prefix}] Social Influence (Byz=30%): W={r['welfare']:.1f}, "
              f"λ={r['mean_lam']:.3f}, Surv={r['survival']*100:.1f}%")

        # Unconditional for comparison
        r = run_experiment(
            n_agents=n_ag, n_byz=n_byz, n_episodes=50, n_seeds=5,
            commitment_fn=make_unconditional_commitment(phi1=1.0),
            label=f"{prefix}_unconditional_byz30"
        )
        all_results[r["label"]] = r
        print(f"  [{prefix}] Unconditional (Byz=30%): W={r['welfare']:.1f}, "
              f"λ={r['mean_lam']:.3f}, Surv={r['survival']*100:.1f}%")

    # ----------------------------------------------------------
    # Experiment 3: f(R_t) Sensitivity
    # ----------------------------------------------------------
    print("\n[3/3] f(R_t) SENSITIVITY: R_crit, recovery rates")
    print("-" * 50)

    sensitivity_results = []

    for r_crit_val in [0.10, 0.15, 0.20, 0.25]:
        for recovery_scale in [0.5, 1.0, 2.0]:
            base_rates = (0.01, 0.03, 0.10)
            scaled_rates = tuple(r * recovery_scale for r in base_rates)

            # Selfish
            r_self = run_experiment(
                n_agents=20, n_byz=6, n_episodes=50, n_seeds=5,
                r_crit=r_crit_val, recovery_rates=scaled_rates,
                commitment_fn=make_selfish_rl_commitment(),
                label=f"sens_selfish_rcrit{r_crit_val}_scale{recovery_scale}"
            )
            # Unconditional
            r_uncond = run_experiment(
                n_agents=20, n_byz=6, n_episodes=50, n_seeds=5,
                r_crit=r_crit_val, recovery_rates=scaled_rates,
                commitment_fn=make_unconditional_commitment(phi1=1.0),
                label=f"sens_uncond_rcrit{r_crit_val}_scale{recovery_scale}"
            )

            row = {
                "r_crit": r_crit_val,
                "recovery_scale": recovery_scale,
                "selfish_survival": r_self["survival"],
                "selfish_welfare": r_self["welfare"],
                "unconditional_survival": r_uncond["survival"],
                "unconditional_welfare": r_uncond["welfare"],
            }
            sensitivity_results.append(row)
            print(f"  R_crit={r_crit_val:.2f}, Scale={recovery_scale:.1f}x: "
                  f"Selfish Surv={r_self['survival']*100:.0f}% vs "
                  f"Uncond Surv={r_uncond['survival']*100:.0f}%")

    all_results["sensitivity"] = sensitivity_results

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------
    total = time.time() - t0
    output = {"results": all_results, "time_seconds": float(total)}
    path = os.path.join(OUT, "extended_results.json")
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  COMPLETE in {total:.1f}s | Results: {path}")
    print(f"{'=' * 70}")
