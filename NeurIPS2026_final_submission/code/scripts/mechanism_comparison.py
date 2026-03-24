"""Fast mechanism comparison — Selfish vs LIO vs IA vs Floor."""
import numpy as np, json, os, time

N = 20; T = 50; SEEDS = 20; TRAIN_EP = 100; EVAL_EP = 50
M_MULT = 1.6; E = 20.0; BYZ = 0.3; RC = 0.15; RR = 0.25; LR = 0.01

def env_step(R, cr, rng):
    base = 0.1 * (cr - 0.4)
    f = 0.01 if R < RC else (0.03 if R < RR else 0.10)
    shock = 0.15 if rng.random() < 0.05 else 0
    return float(np.clip(R + f * base - shock, 0, 1))

class Selfish:
    def __init__(self):
        self.w = np.zeros(2); self.b = 0.0
    def act(self, obs, rng):
        p = 1/(1+np.exp(-np.clip(self.w@obs + self.b, -10, 10)))
        return float(np.clip(p + rng.normal(0, 0.05), 0, 1))
    def update(self, r, a, obs):
        p = 1/(1+np.exp(-np.clip(self.w@obs + self.b, -10, 10)))
        g = a - p; self.w += LR*r*g*obs; self.b += LR*r*g

class LIO(Selfish):
    def __init__(self):
        super().__init__(); self.wi = np.zeros(2); self.bi = 0.0
    def incentive(self, lj, obs):
        p = 1/(1+np.exp(-np.clip(self.wi@obs + self.bi, -10, 10)))
        return float(p * lj * 0.5)
    def update(self, r, a, obs):
        super().update(r, a, obs)
        p = 1/(1+np.exp(-np.clip(self.wi@obs + self.bi, -10, 10)))
        g = 0.5-p; self.wi += LR*r*g*obs; self.bi += LR*r*g

class IA(Selfish):
    def __init__(self, alpha=0.5, beta=0.25):
        super().__init__(); self.al = alpha; self.be = beta

class Floor:
    def __init__(self, phi=1.0):
        self.phi = phi; self.w = np.zeros(2); self.b = 0.0
    def act(self, obs, rng):
        p = 1/(1+np.exp(-np.clip(self.w@obs + self.b, -10, 10)))
        return max(float(np.clip(p + rng.normal(0, 0.05), 0, 1)), self.phi)
    def update(self, r, a, obs):
        p = 1/(1+np.exp(-np.clip(self.w@obs + self.b, -10, 10)))
        g = a-p; self.w += LR*r*g*obs; self.b += LR*r*g

def episode(agents, seed, train=False):
    rng = np.random.RandomState(seed); nb = int(N*BYZ); R = 0.5
    welf = []; lhist = []; alive = True
    for t in range(T):
        obs = np.array([R, t/T]); lam = np.zeros(N)
        for i in range(N):
            lam[i] = 0.0 if i < nb else agents[i].act(obs, rng)
        c = E * lam; pg = (c.sum()*M_MULT)/N; pay = (E-c)+pg
        # LIO incentives
        inc = np.zeros(N)
        for i in range(nb, N):
            if hasattr(agents[i], 'incentive'):
                for j in range(nb, N):
                    if i != j:
                        v = agents[i].incentive(lam[j], obs)
                        inc[j] += v; pay[i] -= abs(v)*0.1
        # IA modification
        for i in range(nb, N):
            if hasattr(agents[i], 'al'):
                mo = (pay.sum()-pay[i])/(N-1)
                pay[i] -= agents[i].al*max(mo-pay[i], 0) + agents[i].be*max(pay[i]-mo, 0)
        pay += inc; welf.append(float(pay.mean()))
        R = env_step(R, np.mean(c)/E, rng); lhist.append(lam[nb:].mean())
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
        'LIO (incentives)': LIO,
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
                episode(agents, 42+s*1000+ep, True)
            evals = [episode(agents, 42+s*1000+TRAIN_EP+ep) for ep in range(EVAL_EP)]
            last = evals[-30:]
            rs.append({k: float(np.mean([r[k] for r in last])) for k in last[0]})
            if (s+1) % 5 == 0:
                print(f"    Seed {s+1}/{SEEDS}", flush=True)
        agg = {}
        for k in rs[0]:
            v = [r[k] for r in rs]
            agg[k] = round(float(np.mean(v)), 3)
            agg[k + '_std'] = round(float(np.std(v)), 3)
        results[name] = agg
        lam = agg['mean_lambda']
        lam_std = agg['mean_lambda_std']
        surv = agg['survival'] * 100
        welf = agg['welfare']
        print(f"    Result: lambda={lam:.3f}+/-{lam_std:.3f}  S={surv:.1f}%  W={welf:.1f}")

    # Summary table
    t_total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  RESULTS ({t_total:.0f}s)")
    print(f"{'='*70}")
    print(f"  {'Method':<25} {'lambda':>8} {'Surv%':>8} {'Welfare':>10}")
    print(f"  {'-'*53}")
    for name, r in results.items():
        lam = r['mean_lambda']
        lam_std = r['mean_lambda_std']
        surv = r['survival'] * 100
        welf = r['welfare']
        print(f"  {name:<25} {lam:.3f}+/-{lam_std:.3f} {surv:>7.1f}% {welf:>9.1f}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'mechanism_comparison')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'mechanism_comparison.json')
    with open(path, 'w') as f:
        json.dump({
            'results': results,
            'time_seconds': t_total,
            'config': {'N': N, 'T': T, 'seeds': SEEDS, 'train': TRAIN_EP, 'eval': EVAL_EP, 'byz': BYZ}
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {path}")
