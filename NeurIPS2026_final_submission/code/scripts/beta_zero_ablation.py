"""
Beta-Zero Ablation: Separating Byzantine adversary effects from tipping-point effects.
=======================================================================================

Motivation (Reviewer concern):
  All prior experiments fix Byz=30%, making it impossible to tell whether
  the Nash Trap arises from adversaries or from tipping-point dynamics.

Design:
  {Linear PGG (f=0.10 constant), Nonlinear TPSD (f=piecewise, f_crit=0.01)}
  x {beta=0.0, beta=0.3}
  x {REINFORCE, IPPO}
  x 20 seeds

Key hypothesis: The Nash Trap exists EVEN at beta=0.0 in TPSD, proving
it is the tipping-point structure, not adversaries, that causes the trap.

Usage:
  ETHICAAI_FAST=1 python beta_zero_ablation.py   # Quick smoke test (2 seeds)
  python beta_zero_ablation.py                     # Full 20-seed run
"""
import numpy as np
import json
import os
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N = 20          # Total agents (honest + byzantine)
T = 50          # Steps per episode
SEEDS = 20      # Seeds per condition
TRAIN_EP = 200  # Training episodes
EVAL_EP = 50    # Evaluation episodes (last 30 used for metrics)
M_MULT = 1.6   # PGG multiplier
E = 20.0        # Endowment
RC = 0.15       # Critical resource threshold
RR = 0.25       # Recovery resource threshold
LR = 0.01       # Learning rate

FAST = os.environ.get("ETHICAAI_FAST", "0") == "1"
if FAST:
    SEEDS = 2
    TRAIN_EP = 50
    EVAL_EP = 10
    print("[FAST MODE] seeds=2, train=50, eval=10")

# Experimental grid
BETAS = [0.0, 0.3]
ENV_TYPES = ["Linear", "TPSD"]
ALGO_NAMES = ["REINFORCE", "IPPO"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Environment step functions
# ---------------------------------------------------------------------------

def env_step_linear(R, cr, rng):
    """Linear PGG: constant recovery f=0.10, no tipping point."""
    f = 0.10
    shock = 0.15 if rng.random() < 0.05 else 0.0
    return float(np.clip(R + f * (cr - 0.4) - shock, 0, 1))


def env_step_tpsd(R, cr, rng):
    """Nonlinear TPSD: piecewise f with f_crit=0.01 below R_crit."""
    if R < RC:
        f = 0.01       # Near-irreversible below critical threshold
    elif R < RR:
        f = 0.03       # Hysteresis zone
    else:
        f = 0.10       # Normal recovery
    shock = 0.15 if rng.random() < 0.05 else 0.0
    return float(np.clip(R + f * (cr - 0.4) - shock, 0, 1))


# ---------------------------------------------------------------------------
# Agent classes (following mechanism_comparison_fair.py pattern)
# ---------------------------------------------------------------------------

class Selfish:
    """REINFORCE agent: linear policy, no value baseline."""
    def __init__(self):
        self.w = np.zeros(2)
        self.b = 0.0

    def act(self, obs, rng):
        p = sigmoid(self.w @ obs + self.b)
        return float(np.clip(p + rng.normal(0, 0.05), 0, 1))

    def update(self, r, a, obs):
        p = sigmoid(self.w @ obs + self.b)
        g = a - p
        self.w += LR * r * g * obs
        self.b += LR * r * g


class IPPO(Selfish):
    """IPPO: REINFORCE with learned value baseline (advantage = r - V(s))."""
    def __init__(self):
        super().__init__()
        self.vw = np.zeros(2)
        self.vb = 0.0

    def value(self, obs):
        return float(self.vw @ obs + self.vb)

    def update(self, r, a, obs):
        v = self.value(obs)
        adv = r - v
        p = sigmoid(self.w @ obs + self.b)
        g = a - p
        self.w += LR * adv * g * obs
        self.b += LR * adv * g
        # Value function update
        self.vw += LR * 0.5 * (r - v) * obs
        self.vb += LR * 0.5 * (r - v)


ALGO_CLASSES = {
    "REINFORCE": Selfish,
    "IPPO": IPPO,
}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def episode(agents, seed, byz_frac, env_step_fn, train=False):
    """Run one episode of the PGG.

    Args:
        agents: list of N agent objects (first nb are ignored / byzantine)
        seed: random seed for this episode
        byz_frac: fraction of byzantine agents (0.0 or 0.3)
        env_step_fn: env_step_linear or env_step_tpsd
        train: whether to call agent.update()

    Returns:
        dict with welfare, survival, mean_lambda
    """
    rng = np.random.RandomState(seed)
    nb = int(N * byz_frac)
    R = 0.5
    welf = []
    lhist = []
    alive = True

    for t in range(T):
        obs = np.array([R, t / T])
        lam = np.zeros(N)
        for i in range(N):
            lam[i] = 0.0 if i < nb else agents[i].act(obs, rng)

        c = E * lam
        pg = (c.sum() * M_MULT) / N
        pay = (E - c) + pg

        welf.append(float(pay.mean()))
        honest_mean = float(lam[nb:].mean()) if nb < N else float(lam.mean())
        lhist.append(honest_mean)

        R = env_step_fn(R, np.mean(c) / E, rng)
        if R <= 0:
            alive = False
            break

        if train:
            for i in range(nb, N):
                agents[i].update(pay[i], lam[i], obs)

    return {
        "welfare": float(np.mean(welf)),
        "survival": float(alive),
        "mean_lambda": float(np.mean(lhist)) if lhist else 0.5,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_condition(env_type, beta, algo_name):
    """Run all seeds for one (env_type, beta, algo) condition."""
    env_step_fn = env_step_linear if env_type == "Linear" else env_step_tpsd
    agent_cls = ALGO_CLASSES[algo_name]

    per_seed = []
    for s in range(SEEDS):
        agents = [agent_cls() for _ in range(N)]

        # Training
        for ep in range(TRAIN_EP):
            episode(agents, 42 + s * 1000 + ep, beta, env_step_fn, train=True)

        # Evaluation
        evals = [
            episode(agents, 42 + s * 1000 + TRAIN_EP + ep, beta, env_step_fn, train=False)
            for ep in range(EVAL_EP)
        ]
        last = evals[-30:] if len(evals) >= 30 else evals
        seed_agg = {k: float(np.mean([r[k] for r in last])) for k in last[0]}
        seed_agg["trapped"] = seed_agg["mean_lambda"] < 0.85
        per_seed.append(seed_agg)

    # Aggregate across seeds
    all_lam = [r["mean_lambda"] for r in per_seed]
    all_surv = [r["survival"] for r in per_seed]
    all_welf = [r["welfare"] for r in per_seed]
    all_trap = [r["trapped"] for r in per_seed]

    agg = {
        "mean_lambda": round(float(np.mean(all_lam)), 4),
        "mean_lambda_std": round(float(np.std(all_lam)), 4),
        "survival_pct": round(float(np.mean(all_surv)) * 100, 1),
        "survival_pct_std": round(float(np.std(all_surv)) * 100, 1),
        "welfare": round(float(np.mean(all_welf)), 2),
        "welfare_std": round(float(np.std(all_welf)), 2),
        "trap_rate": round(float(np.mean(all_trap)), 3),
    }
    return agg, per_seed


def main():
    t0 = time.time()
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "beta_zero_ablation"
    )
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("  Beta-Zero Ablation: Adversaries vs Tipping-Point Dynamics")
    print("  Grid: {Linear, TPSD} x {beta=0.0, beta=0.3} x {REINFORCE, IPPO}")
    print(f"  Seeds={SEEDS}  Train={TRAIN_EP}  Eval={EVAL_EP}")
    print("=" * 70)

    results = {}
    conditions = [
        (env_type, beta, algo)
        for env_type in ENV_TYPES
        for beta in BETAS
        for algo in ALGO_NAMES
    ]

    for idx, (env_type, beta, algo) in enumerate(conditions, 1):
        label = f"{env_type} | beta={beta} | {algo}"
        print(f"\n[{idx}/{len(conditions)}] {label}", flush=True)

        agg, per_seed = run_condition(env_type, beta, algo)
        results[label] = {
            "env_type": env_type,
            "beta": beta,
            "algo": algo,
            "aggregate": agg,
            "per_seed": per_seed,
        }

        print(
            f"  => lambda={agg['mean_lambda']:.3f}+/-{agg['mean_lambda_std']:.3f}  "
            f"S={agg['survival_pct']:.1f}%  "
            f"W={agg['welfare']:.1f}  "
            f"trap_rate={agg['trap_rate']:.0%}"
        )

    elapsed = time.time() - t0

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  RESULTS  ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
    header = f"  {'Condition':<35} {'lambda':>10} {'Surv%':>8} {'Welfare':>10} {'Trap%':>8}"
    print(header)
    print(f"  {'-' * 73}")
    for label, r in results.items():
        a = r["aggregate"]
        print(
            f"  {label:<35} "
            f"{a['mean_lambda']:.3f}+/-{a['mean_lambda_std']:.3f} "
            f"{a['survival_pct']:>6.1f}% "
            f"{a['welfare']:>9.1f} "
            f"{a['trap_rate']*100:>6.1f}%"
        )

    # -----------------------------------------------------------------------
    # Key comparisons for the reviewer response
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  KEY COMPARISONS")
    print(f"{'=' * 70}")

    for algo in ALGO_NAMES:
        tpsd_b0 = results.get(f"TPSD | beta=0.0 | {algo}", {}).get("aggregate", {})
        tpsd_b3 = results.get(f"TPSD | beta=0.3 | {algo}", {}).get("aggregate", {})
        lin_b0 = results.get(f"Linear | beta=0.0 | {algo}", {}).get("aggregate", {})
        lin_b3 = results.get(f"Linear | beta=0.3 | {algo}", {}).get("aggregate", {})

        if tpsd_b0 and lin_b0:
            print(f"\n  [{algo}] Tipping-point effect at beta=0.0:")
            print(f"    Linear  lambda={lin_b0.get('mean_lambda',0):.3f}  trap={lin_b0.get('trap_rate',0):.0%}")
            print(f"    TPSD    lambda={tpsd_b0.get('mean_lambda',0):.3f}  trap={tpsd_b0.get('trap_rate',0):.0%}")
            trap_diff = tpsd_b0.get("trap_rate", 0) - lin_b0.get("trap_rate", 0)
            print(f"    => TPSD trap rate excess: {trap_diff:+.0%}")

        if tpsd_b0 and tpsd_b3:
            print(f"\n  [{algo}] Byzantine effect in TPSD:")
            print(f"    beta=0.0  lambda={tpsd_b0.get('mean_lambda',0):.3f}  trap={tpsd_b0.get('trap_rate',0):.0%}")
            print(f"    beta=0.3  lambda={tpsd_b3.get('mean_lambda',0):.3f}  trap={tpsd_b3.get('trap_rate',0):.0%}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    output = {
        "experiment": "beta_zero_ablation",
        "purpose": "Separate Byzantine adversary effects from tipping-point effects",
        "hypothesis": "Nash Trap exists even at beta=0 in TPSD, proving tipping-point causation",
        "config": {
            "N": N, "T": T, "seeds": SEEDS, "train_ep": TRAIN_EP,
            "eval_ep": EVAL_EP, "M_mult": M_MULT, "E": E,
            "R_crit": RC, "R_recov": RR, "betas": BETAS,
            "env_types": ENV_TYPES, "algos": ALGO_NAMES,
            "mode": "FAST" if FAST else "FULL",
        },
        "results": {
            label: {
                "env_type": r["env_type"],
                "beta": r["beta"],
                "algo": r["algo"],
                "mean_lambda": r["aggregate"]["mean_lambda"],
                "mean_lambda_std": r["aggregate"]["mean_lambda_std"],
                "survival_pct": r["aggregate"]["survival_pct"],
                "welfare": r["aggregate"]["welfare"],
                "trap_rate": r["aggregate"]["trap_rate"],
            }
            for label, r in results.items()
        },
        "per_seed_data": {
            label: r["per_seed"] for label, r in results.items()
        },
        "time_seconds": round(elapsed, 1),
    }

    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
