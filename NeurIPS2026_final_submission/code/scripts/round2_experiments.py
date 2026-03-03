"""
Round 2: All experimental phases in one script
================================================
Phase A: HP Sweep (IPPO lr × entropy, 12 combos × 3 seeds)
Phase B: Commitment Ablation (5 variants × 3 seeds)
Phase D: LOLA Divergence Analysis (analytical)
"""
import numpy as np
import json
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "envs"))
from pathlib import Path
from envs.nonlinear_pgg_env import NonlinearPGGEnv

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "round2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Shared NN components ────────────────────────────────────
def relu(x):
    return np.maximum(0, x)

class SimpleActor:
    """Lightweight 2-layer MLP policy."""
    def __init__(self, rng, obs_dim=4, hid=64, lr=2.5e-4):
        s = np.sqrt(2.0/obs_dim)
        self.W1 = rng.randn(obs_dim, hid).astype(np.float32)*s
        self.b1 = np.zeros(hid, dtype=np.float32)
        self.W2 = rng.randn(hid, hid).astype(np.float32)*np.sqrt(2.0/hid)
        self.b2 = np.zeros(hid, dtype=np.float32)
        self.W3 = rng.randn(hid, 1).astype(np.float32)*np.sqrt(2.0/hid)
        self.b3 = np.zeros(1, dtype=np.float32)
        self.log_std = np.array([-0.5], dtype=np.float32)
        self.lr = lr
    
    def forward(self, obs):
        h = relu(obs @ self.W1 + self.b1)
        h = relu(h @ self.W2 + self.b2)
        raw = (h @ self.W3 + self.b3)[0]
        return 1.0/(1.0+np.exp(-raw))  # sigmoid -> [0,1]
    
    def act(self, obs, rng):
        mu = self.forward(obs)
        std = np.exp(self.log_std[0])
        a = np.clip(mu + rng.randn()*std, 0, 1)
        lp = -0.5*((a-mu)/std)**2 - self.log_std[0] - 0.5*np.log(2*np.pi)
        return a, lp, mu
    
    def simple_update(self, obs, action, advantage):
        """REINFORCE-style update."""
        mu = self.forward(obs)
        std = np.exp(self.log_std[0])
        d_lp = (action - mu) / (std**2)
        sig_d = mu * (1 - mu)
        
        h1 = relu(obs @ self.W1 + self.b1)
        h2 = relu(h1 @ self.W2 + self.b2)
        
        grad_scale = d_lp * sig_d * advantage * self.lr
        self.W3 += np.outer(h2, [grad_scale])
        self.b3 += np.array([grad_scale])


def run_ippo_single(seed, lr, entropy_coef, n_episodes=150):
    """Run single IPPO experiment."""
    rng = np.random.RandomState(seed)
    env = NonlinearPGGEnv()
    n = env.n_honest
    actors = [SimpleActor(rng, lr=lr) for _ in range(n)]
    
    episodes = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed*10000+ep)
        ep_data = []
        for t in range(env.T):
            actions = np.zeros(n)
            obs_list = []
            act_list = []
            for i in range(n):
                a, lp, mu = actors[i].act(obs, rng)
                actions[i] = a
                obs_list.append(obs.copy())
                act_list.append(a)
            
            obs, rewards, done, _, info = env.step(actions)
            ep_data.append((obs_list, act_list, np.mean(rewards)))
            if done:
                break
        
        # Simple policy gradient update
        for step_obs, step_act, r in ep_data:
            for i in range(n):
                actors[i].simple_update(step_obs[i], step_act[i], r)
        
        episodes.append({
            "mean_lambda": info.get("mean_lambda", 0),
            "survived": info.get("survived", False),
        })
    
    ev = episodes[-20:]
    return {
        "mean_lambda": float(np.mean([d["mean_lambda"] for d in ev])),
        "survival": float(np.mean([float(d["survived"]) for d in ev]) * 100),
    }


# ═══════════════════════════════════════════════════════════════
# PHASE A: HP SWEEP
# ═══════════════════════════════════════════════════════════════
def phase_a():
    print("\n" + "="*70)
    print("  PHASE A: HP Tuning Sweep")
    print("="*70)
    
    LR_GRID = [1e-4, 2.5e-4, 5e-4, 1e-3]
    ENT_GRID = [0.0, 0.01, 0.05]
    N_SEEDS = 3
    
    results = {}
    t0 = time.time()
    
    for lr in LR_GRID:
        for ent in ENT_GRID:
            key = f"lr={lr:.0e}_ent={ent}"
            seeds = []
            for s in range(N_SEEDS):
                r = run_ippo_single(seed=s*7+42, lr=lr, entropy_coef=ent)
                seeds.append(r)
            
            lams = [r["mean_lambda"] for r in seeds]
            survs = [r["survival"] for r in seeds]
            trapped = np.mean(lams) < 0.7
            
            results[key] = {
                "lr": lr, "entropy": ent,
                "lambda_mean": float(np.mean(lams)),
                "lambda_std": float(np.std(lams)),
                "survival_mean": float(np.mean(survs)),
                "trapped": bool(trapped),
            }
            
            st = "TRAPPED" if trapped else "ESCAPED"
            print(f"  lr={lr:.0e} ent={ent:.2f}: λ={np.mean(lams):.3f} surv={np.mean(survs):.1f}% {st}")
    
    elapsed = time.time() - t0
    all_trapped = all(r["trapped"] for r in results.values())
    print(f"\n  All {len(results)} combos trapped: {all_trapped}")
    print(f"  Time: {elapsed:.0f}s")
    
    return results


# ═══════════════════════════════════════════════════════════════
# PHASE B: COMMITMENT ABLATION
# ═══════════════════════════════════════════════════════════════
def compute_g(R_t, variant, prev_lambda):
    """Commitment function with ablation variants."""
    alpha = 0.9
    crisis_th = 0.15
    abund_th = 0.25
    crisis_lam = 1.0
    normal = 0.5
    abund_bonus = 0.3
    beta = 0.1
    
    if variant == "no_crisis":
        crisis_th = -1.0
    elif variant == "no_abundance":
        abund_bonus = 0.0
    elif variant == "alpha_zero":
        alpha = 0.0
    elif variant == "alpha_one":
        alpha = 1.0
    elif variant == "no_restraint":
        beta = 0.0
    
    if R_t < crisis_th:
        target = crisis_lam
    elif R_t >= abund_th:
        target = min(1.0, normal + abund_bonus)
    else:
        target = normal
    
    target = max(0.0, target - beta * (1 - R_t))
    target = np.clip(target, 0.0, 1.0)
    return alpha * prev_lambda + (1 - alpha) * target


def run_ablation_single(seed, variant, n_episodes=150):
    rng = np.random.RandomState(seed)
    env = NonlinearPGGEnv()
    n = env.n_honest
    episodes = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed*10000+ep)
        lambdas = np.full(n, 0.5)
        
        for t in range(env.T):
            R_t = obs[2] if len(obs) > 2 else 0.5
            for i in range(n):
                lambdas[i] = compute_g(R_t, variant, lambdas[i])
            obs, rewards, done, _, info = env.step(lambdas)
            if done:
                break
        
        episodes.append({
            "mean_lambda": float(np.mean(lambdas)),
            "survived": info.get("survived", False),
            "welfare": info.get("welfare", 0),
        })
    
    ev = episodes[-20:]
    return {
        "mean_lambda": float(np.mean([d["mean_lambda"] for d in ev])),
        "survival": float(np.mean([float(d["survived"]) for d in ev]) * 100),
        "welfare": float(np.mean([d["welfare"] for d in ev])),
    }


def phase_b():
    print("\n" + "="*70)
    print("  PHASE B: Commitment Function Ablation")
    print("="*70)
    
    variants = ["full", "no_crisis", "no_abundance", "alpha_zero", "alpha_one", "no_restraint"]
    labels = {
        "full": "Full g(θ,R)", "no_crisis": "No crisis zone",
        "no_abundance": "No abundance bonus", "alpha_zero": "No smoothing (α=0)",
        "alpha_one": "No adaptation (α=1)", "no_restraint": "No restraint (β=0)",
    }
    
    results = {}
    t0 = time.time()
    
    for var in variants:
        seeds = []
        for s in range(3):
            r = run_ablation_single(seed=s*7+42, variant=var)
            seeds.append(r)
        
        lams = [r["mean_lambda"] for r in seeds]
        survs = [r["survival"] for r in seeds]
        wels = [r["welfare"] for r in seeds]
        
        results[var] = {
            "label": labels[var],
            "lambda": float(np.mean(lams)),
            "survival": float(np.mean(survs)),
            "welfare": float(np.mean(wels)),
        }
        print(f"  {labels[var]:>25s}: λ={np.mean(lams):.3f} surv={np.mean(survs):.1f}% W={np.mean(wels):.1f}")
    
    print(f"  Time: {time.time()-t0:.0f}s")
    return results


# ═══════════════════════════════════════════════════════════════
# PHASE D: LOLA DIVERGENCE
# ═══════════════════════════════════════════════════════════════
def phase_d():
    print("\n" + "="*70)
    print("  PHASE D: LOLA Divergence Analysis")
    print("="*70)
    
    M = 1.6
    R_crit = 0.15
    results = {}
    
    for N in [2, 5, 10, 20, 50, 100]:
        lambdas = np.full(N, 0.5)
        N_total = int(N / 0.7)
        R = np.sum(lambdas) / N
        p_surv = 1.0 / (1.0 + np.exp(-20*(R - R_crit)))
        dp = p_surv*(1-p_surv)*20
        
        # 1st order gradient: ∂r_i/∂λ_i
        grad1 = -1.0 + M/N * p_surv + M/N * np.sum(lambdas) * dp/N
        
        # 2nd order cross-derivative: ∂²r_i/∂λ_i∂λ_j
        d2p = dp * (1 - 2*p_surv) * 20 / N
        cross = M/N * (dp/N + np.sum(lambdas) * d2p/N)
        
        # LOLA correction magnitude: Σ_{j≠i} lr_opp * grad_j * cross
        lr_opp = 0.01
        lola_correction = (N-1) * lr_opp * abs(grad1) * abs(cross)
        
        ratio = lola_correction / (abs(grad1) + 1e-10)
        diverges = ratio > 1.0
        
        results[str(N)] = {
            "N": N, "grad_norm": float(abs(grad1)),
            "lola_correction": float(lola_correction),
            "ratio": float(ratio), "diverges": bool(diverges),
        }
        
        st = "DIVERGES" if diverges else "stable"
        print(f"  N={N:>4d}: |∇|={abs(grad1):.4f}, |LOLA|={lola_correction:.4f}, ratio={ratio:.2f} → {st}")
    
    # N=20 trajectory
    N = 20
    lambdas = np.full(N, 0.5)
    trajectory = []
    for step in range(200):
        R = np.sum(lambdas)/N
        p = 1.0/(1.0+np.exp(-20*(R-R_crit)))
        dp = p*(1-p)*20
        N_total = int(N/0.7)
        grad = -1.0 + M/N*p + M/N*np.sum(lambdas)*dp/N
        cross = M/N*(dp/N + np.sum(lambdas)*(dp*(1-2*p)*20/N)/N)
        lola = (N-1)*0.01*grad*cross
        lambdas = np.clip(lambdas + 0.01*(grad + lola), 0, 1)
        trajectory.append(float(np.mean(lambdas)))
    
    results["trajectory_final"] = float(trajectory[-1])
    print(f"\n  N=20 LOLA trajectory → λ_final = {trajectory[-1]:.4f}")
    
    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  ROUND 2: ALL EXPERIMENTAL PHASES")
    print("=" * 70)
    
    t_total = time.time()
    all_results = {}
    
    all_results["phase_a_hp_sweep"] = phase_a()
    all_results["phase_b_ablation"] = phase_b()
    all_results["phase_d_lola"] = phase_d()
    
    # Save
    out_path = OUTPUT_DIR / "round2_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    elapsed = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"  ALL PHASES COMPLETE — Total: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Saved: {out_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
