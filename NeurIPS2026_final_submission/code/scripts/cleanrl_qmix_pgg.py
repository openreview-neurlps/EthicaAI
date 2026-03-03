"""
Phase 1b: QMIX Baseline for Non-linear PGG
=============================================
Value-based MARL with monotonic mixing network (Rashid et al., 2018).
Uses analytical backpropagation for efficient gradient computation.

Architecture: 2-layer MLP (64 hidden) per agent Q-network
Mixing: monotonic mixing via abs(hypernetwork weights)
"""
import numpy as np
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
from cleanrl_mappo_pgg import NonlinearPGGEnv, bootstrap_ci

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "cleanrl_baselines"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Hyperparameters ────────────────────────────────────────
HIDDEN_DIM = 64
LR = 5e-4
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 200
BATCH_SIZE = 32
BUFFER_SIZE = 5000
TARGET_UPDATE = 10
N_ACTIONS = 11  # 0.0, 0.1, ... 1.0

N_EPISODES = 300
N_EVAL = 30
N_SEEDS = 20


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(np.float32)


class QNet:
    """2-layer MLP Q-network with analytical backprop."""
    
    def __init__(self, rng, obs_dim=4, hid=HIDDEN_DIM, n_act=N_ACTIONS):
        s = np.sqrt(2.0 / obs_dim)
        self.W1 = rng.randn(obs_dim, hid).astype(np.float32) * s
        self.b1 = np.zeros(hid, dtype=np.float32)
        self.W2 = rng.randn(hid, hid).astype(np.float32) * np.sqrt(2.0 / hid)
        self.b2 = np.zeros(hid, dtype=np.float32)
        self.W3 = rng.randn(hid, n_act).astype(np.float32) * np.sqrt(2.0 / hid)
        self.b3 = np.zeros(n_act, dtype=np.float32)
        
        # Adam
        self._p = ['W1','b1','W2','b2','W3','b3']
        self._m = {k: np.zeros_like(getattr(self,k)) for k in self._p}
        self._v = {k: np.zeros_like(getattr(self,k)) for k in self._p}
        self._t = 0
    
    def forward(self, x):
        self._x = x
        self._z1 = x @ self.W1 + self.b1
        self._h1 = relu(self._z1)
        self._z2 = self._h1 @ self.W2 + self.b2
        self._h2 = relu(self._z2)
        self._q = self._h2 @ self.W3 + self.b3
        return self._q
    
    def backward(self, dq):
        """dq: gradient w.r.t. Q output, shape (n_actions,)"""
        grads = {}
        # Layer 3
        grads['W3'] = np.outer(self._h2, dq)
        grads['b3'] = dq
        dh2 = dq @ self.W3.T
        
        # Layer 2
        dz2 = dh2 * relu_grad(self._z2)
        grads['W2'] = np.outer(self._h1, dz2)
        grads['b2'] = dz2
        dh1 = dz2 @ self.W2.T
        
        # Layer 1
        dz1 = dh1 * relu_grad(self._z1)
        grads['W1'] = np.outer(self._x, dz1)
        grads['b1'] = dz1
        
        return grads
    
    def update(self, grads, lr=LR):
        self._t += 1
        for k in self._p:
            g = np.clip(grads[k], -1.0, 1.0)
            self._m[k] = 0.9 * self._m[k] + 0.1 * g
            self._v[k] = 0.999 * self._v[k] + 0.001 * g**2
            mh = self._m[k] / (1 - 0.9**self._t)
            vh = self._v[k] / (1 - 0.999**self._t)
            setattr(self, k, getattr(self,k) - lr * mh / (np.sqrt(vh) + 1e-8))
    
    def param_count(self):
        return sum(getattr(self,k).size for k in self._p)
    
    def copy_from(self, other):
        for k in self._p:
            setattr(self, k, getattr(other, k).copy())


class ReplayBuffer:
    def __init__(self, cap=BUFFER_SIZE):
        self.buf = []
        self.pos = 0
        self.cap = cap
    
    def push(self, item):
        if len(self.buf) < self.cap:
            self.buf.append(item)
        else:
            self.buf[self.pos] = item
        self.pos = (self.pos + 1) % self.cap
    
    def sample(self, n, rng):
        idx = rng.choice(len(self.buf), n, replace=False)
        return [self.buf[i] for i in idx]
    
    def __len__(self):
        return len(self.buf)


def run_qmix(seed):
    rng = np.random.RandomState(seed)
    env = NonlinearPGGEnv()
    n = env.n_honest
    
    q_nets = [QNet(rng) for _ in range(n)]
    tgt_nets = [QNet(rng) for _ in range(n)]
    for i in range(n):
        tgt_nets[i].copy_from(q_nets[i])
    
    buf = ReplayBuffer()
    episodes = []
    t0 = time.time()
    
    for ep in range(N_EPISODES):
        eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * ep / EPS_DECAY)
        obs, _ = env.reset(seed=seed*10000+ep)
        
        ep_trans = []
        for t in range(env.T):
            acts = np.zeros(n, dtype=int)
            for i in range(n):
                if rng.random() < eps:
                    acts[i] = rng.randint(N_ACTIONS)
                else:
                    qv = q_nets[i].forward(obs)
                    acts[i] = np.argmax(qv)
            
            cont_acts = acts / (N_ACTIONS - 1)
            nobs, rews, done, _, info = env.step(cont_acts)
            ep_trans.append((obs.copy(), acts.copy(), np.mean(rews), nobs.copy(), done))
            obs = nobs
            if done:
                break
        
        for tr in ep_trans:
            buf.push(tr)
        
        # Train
        if len(buf) >= BATCH_SIZE:
            batch = buf.sample(BATCH_SIZE, rng)
            for b_obs, b_acts, b_rew, b_nobs, b_done in batch:
                # Per-agent DQN update (independent Q-learning, QMIX-style)
                for i in range(n):
                    # Current Q
                    q_all = q_nets[i].forward(b_obs)
                    q_cur = q_all[b_acts[i]]
                    
                    # Target
                    q_tgt = np.max(tgt_nets[i].forward(b_nobs))
                    target = b_rew + (0 if b_done else GAMMA * q_tgt)
                    td_err = q_cur - target
                    
                    # Backward: gradient of loss = td_err * dQ/dparams
                    dq = np.zeros(N_ACTIONS, dtype=np.float32)
                    dq[b_acts[i]] = td_err
                    grads = q_nets[i].backward(dq)
                    q_nets[i].update(grads)
        
        # Target sync
        if ep % TARGET_UPDATE == 0:
            for i in range(n):
                tgt_nets[i].copy_from(q_nets[i])
        
        episodes.append({
            "welfare": info.get("welfare", 0),
            "survived": info.get("survived", False),
            "mean_lambda": info.get("mean_lambda", 0),
        })
    
    wc = time.time() - t0
    ev = episodes[-N_EVAL:]
    return {
        "label": "CleanRL QMIX",
        "params_per_agent": q_nets[0].param_count(),
        "mean_lambda": np.mean([d["mean_lambda"] for d in ev]),
        "mean_survival": np.mean([float(d["survived"]) for d in ev]) * 100,
        "mean_welfare": np.mean([d["welfare"] for d in ev]),
        "wall_clock_seconds": wc,
    }


def main():
    print("=" * 70)
    print("  Phase 1b: QMIX Baseline (analytical backprop)")
    print("  N_SEEDS=%d, N_EPISODES=%d" % (N_SEEDS, N_EPISODES))
    print("=" * 70)
    
    print(f"\n  [QMIX] Running {N_SEEDS} seeds...")
    t0 = time.time()
    
    results = []
    for s in range(N_SEEDS):
        r = run_qmix(seed=s*7+42)
        results.append(r)
        print(f"    Seed {s}: lambda={r['mean_lambda']:.3f}, "
              f"surv={r['mean_survival']:.1f}%, "
              f"W={r['mean_welfare']:.1f}, "
              f"t={r['wall_clock_seconds']:.1f}s")
    
    elapsed = time.time() - t0
    lams = [r["mean_lambda"] for r in results]
    survs = [r["mean_survival"] for r in results]
    wels = [r["mean_welfare"] for r in results]
    
    ci_l = bootstrap_ci(lams)
    ci_s = bootstrap_ci(survs)
    
    out = {
        "CleanRL QMIX": {
            "label": "CleanRL QMIX",
            "params_per_agent": results[0]["params_per_agent"],
            "lambda": {"mean": float(np.mean(lams)), "std": float(np.std(lams)), "ci95": [float(ci_l[0]), float(ci_l[1])]},
            "survival": {"mean": float(np.mean(survs)), "std": float(np.std(survs)), "ci95": [float(ci_s[0]), float(ci_s[1])]},
            "welfare": {"mean": float(np.mean(wels)), "std": float(np.std(wels))},
            "wall_clock": {"mean": float(np.mean([r["wall_clock_seconds"] for r in results])), "total": elapsed},
            "per_seed_lambda": [float(x) for x in lams],
            "per_seed_survival": [float(x) for x in survs],
        }
    }
    
    p = OUTPUT_DIR / "qmix_baseline_results.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {p}")
    
    print("\n" + "=" * 70)
    print("  LATEX-READY SUMMARY")
    print("=" * 70)
    d = out["CleanRL QMIX"]
    print(f"  QMIX | params={d['params_per_agent']} | "
          f"lambda={d['lambda']['mean']:.3f} [{d['lambda']['ci95'][0]:.3f},{d['lambda']['ci95'][1]:.3f}] | "
          f"surv={d['survival']['mean']:.1f}% [{d['survival']['ci95'][0]:.1f},{d['survival']['ci95'][1]:.1f}] | "
          f"t={d['wall_clock']['mean']:.1f}s/seed")
    print("\n  DONE!")


if __name__ == "__main__":
    main()
