"""S3: Fair mechanism comparison — Selfish vs LIO(linear) vs LIO(MLP) vs IA vs Floor.
Upgrades LIO to 2-layer MLP for fair comparison."""
import numpy as np, json, os, time

N = 20; T = 50; SEEDS = 20; TRAIN_EP = 200; EVAL_EP = 50
M_MULT = 1.6; E = 20.0; BYZ = 0.3; RC = 0.15; RR = 0.25; LR = 0.01

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
        p = sigmoid(self.w @ obs + self.b)
        return float(np.clip(p + rng.normal(0, 0.05), 0, 1))
    def update(self, r, a, obs):
        p = sigmoid(self.w @ obs + self.b)
        g = a - p; self.w += LR * r * g * obs; self.b += LR * r * g

class LIO_Linear(Selfish):
    """Original LIO: linear incentive model."""
    def __init__(self):
        super().__init__(); self.wi = np.zeros(2); self.bi = 0.0
    def incentive(self, lj, obs):
        p = sigmoid(self.wi @ obs + self.bi)
        return float(p * lj * 0.5)
    def update(self, r, a, obs):
        super().update(r, a, obs)
        p = sigmoid(self.wi @ obs + self.bi)
        g = 0.5 - p; self.wi += LR * r * g * obs; self.bi += LR * r * g

class LIO_MLP(Selfish):
    """Upgraded LIO: 2-layer MLP incentive network (h=32)."""
    def __init__(self, h=32):
        super().__init__()
        self.W1 = np.random.randn(2, h) * 0.1
        self.b1 = np.zeros(h)
        self.W2 = np.random.randn(h, 1) * 0.1
        self.b2 = np.zeros(1)
        self.h = h
    def incentive(self, lj, obs):
        hidden = np.maximum(0, obs @ self.W1 + self.b1)  # ReLU
        inc = sigmoid(hidden @ self.W2 + self.b2)[0]
        return float(inc * lj * 0.5)
    def update(self, r, a, obs):
        super().update(r, a, obs)
        # Update incentive network with REINFORCE gradient
        hidden = np.maximum(0, obs @ self.W1 + self.b1)
        out = sigmoid(hidden @ self.W2 + self.b2)[0]
        # Gradient through sigmoid
        d_out = out * (1 - out)
        # Gradient through ReLU
        d_hidden = (hidden > 0).astype(float)
        # Update W2, b2
        g2 = hidden.reshape(-1, 1) * d_out
        self.W2 += LR * r * g2 * 0.01
        self.b2 += LR * r * d_out * 0.01
        # Update W1, b1
        backprop = d_out * self.W2.flatten() * d_hidden
        self.W1 += LR * r * np.outer(obs, backprop) * 0.01
        self.b1 += LR * r * backprop * 0.01

class IA(Selfish):
    """Fehr-Schmidt Inequity Aversion."""
    def __init__(self, alpha=0.5, beta=0.25):
        super().__init__(); self.al = alpha; self.be = beta

class Floor:
    def __init__(self, phi=1.0):
        self.phi = phi; self.w = np.zeros(2); self.b = 0.0
    def act(self, obs, rng):
        p = sigmoid(self.w @ obs + self.b)
        return max(float(np.clip(p + rng.normal(0, 0.05), 0, 1)), self.phi)
    def update(self, r, a, obs):
        p = sigmoid(self.w @ obs + self.b)
        g = a - p; self.w += LR * r * g * obs; self.b += LR * r * g

def episode(agents, seed, train=False):
    rng = np.random.RandomState(seed); nb = int(N * BYZ); R = 0.5
    welf = []; lhist = []; alive = True
    for t in range(T):
        obs = np.array([R, t / T]); lam = np.zeros(N)
        for i in range(N):
            lam[i] = 0.0 if i < nb else agents[i].act(obs, rng)
        c = E * lam; pg = (c.sum() * M_MULT) / N; pay = (E - c) + pg
        # LIO incentives
        inc = np.zeros(N)
        for i in range(nb, N):
            if hasattr(agents[i], 'incentive'):
                for j in range(nb, N):
                    if i != j:
                        v = agents[i].incentive(lam[j], obs)
                        inc[j] += v; pay[i] -= abs(v) * 0.1
        # IA modification
        for i in range(nb, N):
            if hasattr(agents[i], 'al'):
                mo = (pay.sum() - pay[i]) / (N - 1)
                pay[i] -= agents[i].al * max(mo - pay[i], 0) + agents[i].be * max(pay[i] - mo, 0)
        pay += inc; welf.append(float(pay.mean()))
        R = env_step(R, np.mean(c) / E, rng); lhist.append(lam[nb:].mean())
        if R <= 0: alive = False; break
        if train:
            for i in range(nb, N): agents[i].update(pay[i], lam[i], obs)
    return {
        'welfare': float(np.mean(welf)),
        'survival': float(alive),
        'mean_lambda': float(np.mean(lhist)) if lhist else 0.5
    }

if __name__ == '__main__':
    methods = {
        'Selfish REINFORCE': Selfish,
        'LIO-Linear': LIO_Linear,
        'LIO-MLP (h=32)': LIO_MLP,
        'Inequity Aversion': IA,
        'Commitment Floor': Floor,
    }
    results = {}; t0 = time.time()
    for name, cls in methods.items():
        print(f"  [{name}]", flush=True)
        rs = []
        for s in range(SEEDS):
            agents = [cls() for _ in range(N)]
            for ep in range(TRAIN_EP):
                episode(agents, 42 + s * 1000 + ep, True)
            evals = [episode(agents, 42 + s * 1000 + TRAIN_EP + ep) for ep in range(EVAL_EP)]
            last = evals[-30:]
            rs.append({k: float(np.mean([r[k] for r in last])) for k in last[0]})
            if (s + 1) % 5 == 0:
                print(f"    Seed {s + 1}/{SEEDS}", flush=True)
        agg = {}
        for k in rs[0]:
            v = [r[k] for r in rs]
            agg[k] = round(float(np.mean(v)), 3)
            agg[k + '_std'] = round(float(np.std(v)), 3)
        results[name] = agg
        print(f"    => lambda={agg['mean_lambda']:.3f}  S={agg['survival']*100:.1f}%  W={agg['welfare']:.1f}")

    t_total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  FAIR COMPARISON RESULTS ({t_total:.0f}s)")
    print(f"{'='*70}")
    print(f"  {'Method':<25} {'lambda':>8} {'Surv%':>8} {'Welfare':>10}")
    print(f"  {'-'*53}")
    for name, r in results.items():
        print(f"  {name:<25} {r['mean_lambda']:.3f}+/-{r['mean_lambda_std']:.3f} {r['survival']*100:>7.1f}% {r['welfare']:>9.1f}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'mechanism_comparison')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'mechanism_comparison_fair.json')
    with open(path, 'w') as f:
        json.dump({
            'results': results,
            'time_seconds': t_total,
            'config': {'N': N, 'T': T, 'seeds': SEEDS, 'train': TRAIN_EP, 'eval': EVAL_EP, 'byz': BYZ},
            'note': 'LIO-MLP uses 2-layer ReLU network (h=32) for incentive function'
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {path}")
