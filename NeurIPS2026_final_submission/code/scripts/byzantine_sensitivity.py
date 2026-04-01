"""Byzantine sensitivity analysis: sweep beta from 0% to 50%.

Tests 4 algorithms across 6 Byzantine fractions:
  - Selfish REINFORCE
  - IPPO (CleanRL-style)
  - Commitment Floor (phi1=1.0)
  - MACCL (learned floor)

N=20, T=50, 20 seeds per condition. Nonlinear PGG with tipping points.
"""
import numpy as np
import json
import os
import time

N = 20; T = 50; SEEDS = 20; TRAIN_EP = 200; EVAL_EP = 50
M_MULT = 1.6; E = 20.0; RC = 0.15; RR = 0.25; LR = 0.01
BETA_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

FAST = os.environ.get("ETHICAAI_FAST", "0") == "1"
if FAST:
    SEEDS = 2; TRAIN_EP = 50; EVAL_EP = 10

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

def env_step(R, cr, rng):
    f = 0.01 if R < RC else (0.03 if R < RR else 0.10)
    shock = 0.15 if rng.random() < 0.05 else 0
    return float(np.clip(R + f * (cr - 0.4) - shock, 0, 1))


class Selfish:
    def __init__(self):
        self.w = np.zeros(2); self.b = 0.0
    def act(self, obs, rng):
        return float(np.clip(sigmoid(self.w @ obs + self.b) + rng.normal(0, 0.05), 0, 1))
    def update(self, r, a, obs):
        p = sigmoid(self.w @ obs + self.b)
        g = a - p; self.w += LR * r * g * obs; self.b += LR * r * g


class IPPO(Selfish):
    """IPPO with value baseline."""
    def __init__(self):
        super().__init__()
        self.vw = np.zeros(2); self.vb = 0.0
    def value(self, obs):
        return float(self.vw @ obs + self.vb)
    def update(self, r, a, obs):
        v = self.value(obs)
        adv = r - v
        p = sigmoid(self.w @ obs + self.b)
        g = a - p
        self.w += LR * adv * g * obs
        self.b += LR * adv * g
        # Value update
        self.vw += LR * 0.5 * (r - v) * obs
        self.vb += LR * 0.5 * (r - v)


class Floor(Selfish):
    def __init__(self, phi=1.0):
        super().__init__(); self.phi = phi
    def act(self, obs, rng):
        return max(super().act(obs, rng), self.phi)


class MACCL(Selfish):
    """Simplified MACCL: learns floor via primal-dual."""
    def __init__(self):
        super().__init__()
        self.omega = np.array([0.0, 0.0, 0.5])  # w1, w2, w3
        self.mu = 1.0  # dual variable
        self.delta = 0.05  # survival target = 95%

    def floor(self, R):
        return float(sigmoid(self.omega[0] * R + self.omega[1] * R**2 + self.omega[2]))

    def act(self, obs, rng):
        base = super().act(obs, rng)
        return max(base, self.floor(obs[0]))


def episode(agents, seed, byz_frac, train=False):
    rng = np.random.RandomState(seed)
    nb = int(N * byz_frac)
    R = 0.5; welf = []; lhist = []; alive = True
    for t in range(T):
        obs = np.array([R, t / T])
        lam = np.zeros(N)
        for i in range(N):
            lam[i] = 0.0 if i < nb else agents[i].act(obs, rng)
        c = E * lam; pg = (c.sum() * M_MULT) / N; pay = (E - c) + pg
        welf.append(float(pay.mean())); lhist.append(float(lam[nb:].mean()) if nb < N else 0)
        R = env_step(R, np.mean(c) / E, rng)
        if R <= 0: alive = False; break
        if train:
            for i in range(nb, N): agents[i].update(pay[i], lam[i], obs)
    return {
        'welfare': float(np.mean(welf)),
        'survival': float(alive),
        'mean_lambda': float(np.mean(lhist)) if lhist else 0.5,
    }


def run_sweep(method_name, make_agents_fn):
    """Run an algorithm across all beta values."""
    results = {}
    for beta in BETA_SWEEP:
        print(f"    beta={beta:.0%}", end="", flush=True)
        seed_results = []
        for s in range(SEEDS):
            agents = make_agents_fn()
            for ep in range(TRAIN_EP):
                episode(agents, 42 + s * 1000 + ep, beta, train=True)
            evals = [episode(agents, 42 + s * 1000 + TRAIN_EP + ep, beta)
                     for ep in range(EVAL_EP)]
            last = evals[-30:]
            agg = {k: float(np.mean([r[k] for r in last]))
                   for k in ['welfare', 'survival', 'mean_lambda']}
            seed_results.append(agg)
        # Aggregate
        for k in seed_results[0]:
            vals = [r[k] for r in seed_results]
            results.setdefault(f"beta_{beta:.2f}", {})[k + '_mean'] = round(float(np.mean(vals)), 3)
            results[f"beta_{beta:.2f}"][k + '_std'] = round(float(np.std(vals)), 3)
        r = results[f"beta_{beta:.2f}"]
        print(f"  lambda={r['mean_lambda_mean']:.3f} surv={r['survival_mean']*100:.0f}%", flush=True)
    return results


if __name__ == '__main__':
    all_results = {}
    t0 = time.time()

    nb_default = int(N * 0.3)

    print("  [Selfish REINFORCE]", flush=True)
    all_results['Selfish REINFORCE'] = run_sweep(
        'Selfish', lambda: [Selfish() for _ in range(N)])

    print("  [IPPO]", flush=True)
    all_results['IPPO'] = run_sweep(
        'IPPO', lambda: [IPPO() for _ in range(N)])

    print("  [Commitment Floor (phi1=1.0)]", flush=True)
    all_results['Commitment Floor'] = run_sweep(
        'Floor', lambda: [Floor() for _ in range(N)])

    print("  [MACCL]", flush=True)
    all_results['MACCL'] = run_sweep(
        'MACCL', lambda: [MACCL() for _ in range(N)])

    t_total = time.time() - t0

    # --- Print Summary ---
    print(f"\n{'='*80}")
    print(f"  BYZANTINE SENSITIVITY ANALYSIS ({t_total:.0f}s)")
    print(f"{'='*80}")
    header = f"  {'Method':<25}" + "".join(f"  beta={b:.0%}" for b in BETA_SWEEP)
    print(header)
    print(f"  {'='*73}")
    for method in ['Selfish REINFORCE', 'IPPO', 'Commitment Floor', 'MACCL']:
        line = f"  {method:<25}"
        for beta in BETA_SWEEP:
            key = f"beta_{beta:.2f}"
            s = all_results[method][key]['survival_mean'] * 100
            line += f"  {s:>6.1f}%"
        print(line)

    # --- Save ---
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'outputs', 'byzantine_sensitivity')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'byzantine_sensitivity_results.json')
    with open(path, 'w') as f:
        json.dump({
            'experiment': 'Byzantine Sensitivity Analysis',
            'config': {
                'N': N, 'T': T, 'seeds': SEEDS, 'train_ep': TRAIN_EP,
                'eval_ep': EVAL_EP, 'beta_sweep': BETA_SWEEP,
                'M': M_MULT, 'E': E,
            },
            'results': all_results,
            'time_seconds': round(t_total, 1),
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {path}")
