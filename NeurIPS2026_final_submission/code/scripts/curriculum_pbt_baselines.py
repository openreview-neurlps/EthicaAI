"""Curriculum Learning + PBT baselines for EthicaAI.

Reviewers asked: Can optimistic initialization or population-based methods
escape the Nash Trap without commitment floors?

Baselines:
1. Optimistic init: L0 ∈ {0.6, 0.7, 0.8, 0.9} instead of 0.5
2. Curriculum: Start with mild TPSD (f_crit=0.50), gradually increase to severe (0.01)
3. PBT: Population of 10 agents, select top-50% by survival, mutate
"""
import numpy as np
import json
import os
import time

N = 20; T = 50; E_ENDOW = 20.0; M_MULT = 1.6; BYZ = 0.3
RC = 0.15; RR = 0.25; LR = 0.01
SEEDS = 20; TRAIN_EP = 300; EVAL_EP = 50

FAST = os.environ.get("ETHICAAI_FAST", "0") == "1"
if FAST:
    SEEDS = 3; TRAIN_EP = 50; EVAL_EP = 10

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

def env_step(R, coop_rate, rng, f_crit=0.01):
    if R < RC: f = f_crit
    elif R < RR: f = max(f_crit, 0.03)
    else: f = 0.10
    shock = 0.15 if rng.random() < 0.05 else 0
    return float(np.clip(R + f * (coop_rate - 0.4) - shock, 0, 1))

def episode(w, b, seed, f_crit=0.01, train=False):
    nb = int(N * BYZ)
    rng = np.random.RandomState(seed)
    R = 0.5; welf = []; alive = True
    for t in range(T):
        obs = np.array([R, t / T])
        lam = np.zeros(N)
        for i in range(N):
            lam[i] = 0.0 if i < nb else float(np.clip(sigmoid(w[i] @ obs + b[i]) + rng.normal(0, 0.05), 0, 1))
        c = E_ENDOW * lam
        pg = (c.sum() * M_MULT) / N
        pay = (E_ENDOW - c) + pg
        welf.append(float(pay.mean()))
        R = env_step(R, np.mean(c) / E_ENDOW, rng, f_crit)
        if R <= 0.001: alive = False; break
        if train:
            for i in range(nb, N):
                p = sigmoid(w[i] @ obs + b[i])
                g = lam[i] - p
                w[i] += LR * pay[i] * g * obs
                b[i] += LR * pay[i] * g
    return {'welfare': float(np.mean(welf)), 'survival': float(alive),
            'mean_lambda': float(np.mean([sigmoid(w[i] @ np.array([0.5, 0.5]) + b[i]) for i in range(int(N*BYZ), N)]))}


def run_optimistic_init():
    """Baseline 1: Optimistic initialization at various L0."""
    results = {}
    for init_lambda in [0.5, 0.6, 0.7, 0.8, 0.9]:
        init_b = np.log(init_lambda / (1 - init_lambda + 1e-8))  # inverse sigmoid
        seed_results = []
        for s in range(SEEDS):
            w = np.zeros((N, 2))
            b = np.full(N, init_b)
            for ep in range(TRAIN_EP):
                episode(w, b, 42 + s * 1000 + ep, f_crit=0.01, train=True)
            evals = [episode(w, b, 42 + s * 1000 + TRAIN_EP + ep, f_crit=0.01)
                     for ep in range(EVAL_EP)]
            last = evals[-30:]
            seed_results.append({k: float(np.mean([r[k] for r in last])) for k in last[0]})
        agg = {k: round(float(np.mean([r[k] for r in seed_results])), 3) for k in seed_results[0]}
        agg['trap_rate'] = round(sum(1 for r in seed_results if r['mean_lambda'] < 0.85) / len(seed_results), 2)
        results[f"init_{init_lambda}"] = agg
        print(f"  OptInit L0={init_lambda}: lam={agg['mean_lambda']:.3f} surv={agg['survival']*100:.0f}% trap={agg['trap_rate']*100:.0f}%")
    return results


def run_curriculum():
    """Baseline 2: Curriculum — start mild, increase severity."""
    results = {}
    phases = [(0.50, 100), (0.10, 100), (0.01, 100)]  # (f_crit, episodes)
    seed_results = []
    for s in range(SEEDS):
        w = np.zeros((N, 2))
        b = np.zeros(N)
        for f_crit, n_ep in phases:
            for ep in range(n_ep):
                episode(w, b, 42 + s * 10000 + ep, f_crit=f_crit, train=True)
        evals = [episode(w, b, 42 + s * 10000 + 300 + ep, f_crit=0.01)
                 for ep in range(EVAL_EP)]
        last = evals[-30:]
        seed_results.append({k: float(np.mean([r[k] for r in last])) for k in last[0]})
    agg = {k: round(float(np.mean([r[k] for r in seed_results])), 3) for k in seed_results[0]}
    agg['trap_rate'] = round(sum(1 for r in seed_results if r['mean_lambda'] < 0.85) / len(seed_results), 2)
    results['curriculum'] = agg
    print(f"  Curriculum: lam={agg['mean_lambda']:.3f} surv={agg['survival']*100:.0f}% trap={agg['trap_rate']*100:.0f}%")
    return results


def run_pbt():
    """Baseline 3: Population-Based Training."""
    POP_SIZE = 10
    GENERATIONS = 30
    EP_PER_GEN = 10
    seed_results = []

    for s in range(SEEDS):
        rng = np.random.RandomState(42 + s)
        # Initialize population
        pop = [{'w': np.zeros((N, 2)), 'b': np.zeros(N)} for _ in range(POP_SIZE)]

        for gen in range(GENERATIONS):
            # Evaluate each member
            fitness = []
            for member in pop:
                evals = [episode(member['w'], member['b'], 42 + s * 10000 + gen * 100 + ep, f_crit=0.01)
                         for ep in range(EP_PER_GEN)]
                fitness.append(np.mean([e['survival'] for e in evals]))

            # Select top 50%
            ranked = np.argsort(fitness)[::-1]
            survivors = [pop[i] for i in ranked[:POP_SIZE // 2]]

            # Reproduce with mutation
            new_pop = []
            for surv in survivors:
                new_pop.append(surv)  # Keep parent
                child = {'w': surv['w'].copy() + rng.randn(*surv['w'].shape) * 0.1,
                          'b': surv['b'].copy() + rng.randn(*surv['b'].shape) * 0.1}
                new_pop.append(child)
            pop = new_pop

            # Train each member
            for member in pop:
                for ep in range(EP_PER_GEN):
                    episode(member['w'], member['b'], 42 + s * 10000 + gen * 100 + 50 + ep, f_crit=0.01, train=True)

        # Evaluate best
        best = pop[0]
        evals = [episode(best['w'], best['b'], 42 + s * 10000 + 9000 + ep, f_crit=0.01)
                 for ep in range(EVAL_EP)]
        last = evals[-30:]
        seed_results.append({k: float(np.mean([r[k] for r in last])) for k in last[0]})

    agg = {k: round(float(np.mean([r[k] for r in seed_results])), 3) for k in seed_results[0]}
    agg['trap_rate'] = round(sum(1 for r in seed_results if r['mean_lambda'] < 0.85) / len(seed_results), 2)
    print(f"  PBT: lam={agg['mean_lambda']:.3f} surv={agg['survival']*100:.0f}% trap={agg['trap_rate']*100:.0f}%")
    return {'pbt': agg}


if __name__ == '__main__':
    t0 = time.time()
    all_results = {}

    print("\n=== Optimistic Initialization ===")
    all_results['optimistic_init'] = run_optimistic_init()

    print("\n=== Curriculum Learning ===")
    all_results['curriculum'] = run_curriculum()

    print("\n=== Population-Based Training ===")
    all_results['pbt'] = run_pbt()

    # Reference: standard init (already known)
    print("\n=== Standard Init (reference) ===")
    print("  Standard: λ~0.5, trap=100% (from existing results)")

    elapsed = time.time() - t0
    print(f"\nTime: {elapsed:.0f}s")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'curriculum_pbt')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'curriculum_pbt_results.json')
    with open(path, 'w') as f:
        json.dump({'results': all_results, 'time_s': round(elapsed)}, f, indent=2, ensure_ascii=False)
    print(f"Saved: {path}")
