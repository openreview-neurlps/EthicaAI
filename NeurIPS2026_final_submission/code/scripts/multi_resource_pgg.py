"""
Multi-Resource PGG + MACCL-Neural Experiment
==============================================
Addresses two structural weaknesses:

F1: "Environment is bandit-level (scalar action)"
  → K=3 Multi-Resource PGG: each agent outputs λ^(k) for K resources
  → 3D action space per agent (not 1D)
  → Tests if Nash Trap persists in multi-dimensional settings

W3: "MACCL = 3D hyperparameter tuning"
  → MACCL-Neural: MLP-parameterized floor (~80 parameters)
  → Genuine function approximation, not HP tuning

Output: multi_resource_results.json
"""

import numpy as np
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

# ============================================================
# Configuration
# ============================================================
N_SEEDS = 20
N_EPISODES = 300
N_AGENTS = 20
BYZ_FRAC = 0.3
K_RESOURCES = 3
GAMMA = 0.99

if os.environ.get("ETHICAAI_FAST") == "1":
    N_SEEDS = 3
    N_EPISODES = 50

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'multi_resource')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# ============================================================
# Multi-Resource PGG Environment
# ============================================================
class MultiResourcePGG:
    """
    K-Resource PGG with independent tipping-point dynamics per resource.
    
    Each agent outputs K commitment levels: λ_i^(k) ∈ [0,1] for k=1..K.
    Each resource has independent recovery dynamics.
    Survival requires ALL K resources to survive.
    
    This extends scalar PGG to multi-dimensional action space,
    addressing the "bandit-level environment" criticism.
    """
    def __init__(self, n_agents=20, k_resources=3, multiplier=1.6,
                 endowment=20.0, byz_frac=0.3):
        self.N = n_agents
        self.K = k_resources
        self.M = multiplier
        self.E = endowment
        self.n_byz = int(n_agents * byz_frac)
        self.n_honest = n_agents - self.n_byz
        self.T = 50
        
        # Per-resource parameters (vary slightly for diversity)
        self.r_crit = np.array([0.15, 0.18, 0.12])[:k_resources]
        self.r_recov = np.array([0.25, 0.28, 0.22])[:k_resources]
        self.shock_probs = np.array([0.05, 0.04, 0.06])[:k_resources]
        self.shock_mags = np.array([0.15, 0.12, 0.18])[:k_resources]
        
    def reset(self, seed=None):
        self.rng = np.random.RandomState(seed)
        self.R = np.full(self.K, 0.5)  # K resources
        self.t = 0
        self.prev_mean_lambdas = np.full(self.K, 0.5)
        return self._obs()
    
    def _obs(self):
        """Observation: [R_1, ..., R_K, mean_λ_1, ..., mean_λ_K, crisis_flags, t/T]"""
        crisis = (self.R < self.r_crit).astype(np.float32)
        return np.concatenate([self.R, self.prev_mean_lambdas, crisis,
                              [self.t / self.T]]).astype(np.float32)
    
    def obs_dim(self):
        return 3 * self.K + 1  # R + mean_λ + crisis per resource + t/T
    
    def step(self, lambdas_honest):
        """
        lambdas_honest: (n_honest, K) array — commitment per resource per agent
        Returns: obs, mean_reward, all_survived, terminated
        """
        # Full lambda: honest + byzantine (byz = 0 on all resources)
        full_lambdas = np.zeros((self.N, self.K))
        full_lambdas[:self.n_honest] = np.clip(lambdas_honest, 0, 1)
        
        # Per-resource payoffs and dynamics
        total_payoff = 0
        all_survived = True
        
        for k in range(self.K):
            contribs_k = full_lambdas[:, k] * self.E / self.K
            pool_k = np.sum(contribs_k)
            payoff_k = (self.E / self.K - contribs_k) + self.M * pool_k / self.N
            total_payoff += np.mean(payoff_k[:self.n_honest])
            
            mean_c_k = np.mean(full_lambdas[:, k])
            self.prev_mean_lambdas[k] = mean_c_k
            
            # Non-linear recovery
            if self.R[k] < self.r_crit[k]:
                f_R = 0.01
            elif self.R[k] < self.r_recov[k]:
                f_R = 0.03
            else:
                f_R = 0.10
            
            shock = self.shock_mags[k] if self.rng.random() < self.shock_probs[k] else 0.0
            self.R[k] = np.clip(self.R[k] + f_R * (mean_c_k - 0.4) - shock, 0, 1)
            
            if self.R[k] <= 0:
                all_survived = False
        
        self.t += 1
        terminated = (not all_survived) or (self.t >= self.T)
        
        return self._obs(), total_payoff, all_survived, terminated


# ============================================================
# Method 1: Selfish RL (REINFORCE, multi-dim)
# ============================================================
def run_selfish_multi(seed, n_episodes):
    env = MultiResourcePGG(N_AGENTS, K_RESOURCES, byz_frac=BYZ_FRAC)
    n_honest = env.n_honest
    K = env.K
    obs_dim = env.obs_dim()
    rng = np.random.RandomState(seed)
    
    # Per-agent, per-resource weights
    W = rng.randn(n_honest, K, obs_dim) * 0.01
    B = np.zeros((n_honest, K))
    lr = 0.01
    
    ep_welfares, ep_survivals, ep_lams = [], [], []
    
    for ep in range(n_episodes):
        obs = env.reset(seed=seed * 10000 + ep)
        noise = max(0.02, 0.15 - 0.13 * min(ep / n_episodes, 1.0))
        
        traj_obs, traj_acts, traj_rews = [], [], []
        total_w, lam_sum, steps = 0.0, 0.0, 0
        survived = True
        
        for t in range(50):
            lambdas = np.zeros((n_honest, K))
            for i in range(n_honest):
                for k in range(K):
                    logit = obs @ W[i, k] + B[i, k]
                    base = sigmoid(logit)
                    lambdas[i, k] = float(np.clip(base + rng.randn() * noise, 0.01, 0.99))
            
            traj_obs.append(obs.copy())
            traj_acts.append(lambdas.copy())
            
            obs, reward, surv, terminated = env.step(lambdas)
            traj_rews.append(reward)
            total_w += reward
            lam_sum += float(np.mean(lambdas))
            steps += 1
            
            if terminated:
                survived = surv
                break
        
        ep_welfares.append(total_w / max(steps, 1))
        ep_survivals.append(float(survived))
        ep_lams.append(lam_sum / max(steps, 1))
        
        # REINFORCE update
        if len(traj_rews) >= 2:
            G, returns = 0, np.zeros(len(traj_rews))
            for t_idx in reversed(range(len(traj_rews))):
                G = traj_rews[t_idx] + GAMMA * G
                returns[t_idx] = G
            if returns.std() > 1e-8:
                returns = (returns - returns.mean()) / returns.std()
            
            for t_idx in range(len(returns)):
                o = traj_obs[t_idx]
                for i in range(n_honest):
                    for k in range(K):
                        a = traj_acts[t_idx][i, k]
                        logit = o @ W[i, k] + B[i, k]
                        pred = sigmoid(logit)
                        grad = a - pred
                        W[i, k] += lr * returns[t_idx] * grad * o
                        B[i, k] += lr * returns[t_idx] * grad
    
    return {
        "welfare_mean": float(np.mean(ep_welfares[-30:])),
        "survival_pct": float(np.mean(ep_survivals[-30:]) * 100),
        "lambda_mean": float(np.mean(ep_lams[-30:])),
    }


# ============================================================
# Method 2: Fixed Commitment (all resources)
# ============================================================
def run_fixed_multi(seed, phi1, n_episodes):
    env = MultiResourcePGG(N_AGENTS, K_RESOURCES, byz_frac=BYZ_FRAC)
    ep_welfares, ep_survivals = [], []
    
    for ep in range(n_episodes):
        obs = env.reset(seed=seed * 10000 + ep)
        total_w, steps = 0.0, 0
        survived = True
        
        for t in range(50):
            lambdas = np.full((env.n_honest, env.K), phi1)
            obs, reward, surv, terminated = env.step(lambdas)
            total_w += reward
            steps += 1
            if terminated:
                survived = surv
                break
        
        ep_welfares.append(total_w / max(steps, 1))
        ep_survivals.append(float(survived))
    
    return {
        "welfare_mean": float(np.mean(ep_welfares[-30:])),
        "survival_pct": float(np.mean(ep_survivals[-30:]) * 100),
    }


# ============================================================
# Method 3: MACCL-Neural (MLP floor, ~80 params)
# ============================================================
class SimpleMLP:
    """2-layer MLP for MACCL-Neural floor function."""
    def __init__(self, input_dim, hidden=16, output_dim=1, seed=0):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(input_dim, hidden) * 0.1
        self.b1 = np.zeros(hidden)
        self.W2 = rng.randn(hidden, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)
        self.params = [self.W1, self.b1, self.W2, self.b2]
    
    def forward(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        out = sigmoid(h @ self.W2 + self.b2)
        return out.flatten()
    
    def n_params(self):
        return sum(p.size for p in self.params)
    
    def get_flat(self):
        return np.concatenate([p.ravel() for p in self.params])
    
    def set_flat(self, flat):
        idx = 0
        for p in self.params:
            size = p.size
            p[:] = flat[idx:idx+size].reshape(p.shape)
            idx += size


def run_maccl_neural(seed, n_episodes):
    """MACCL with MLP-parameterized floor: phi1^(k)(s; theta)."""
    env = MultiResourcePGG(N_AGENTS, K_RESOURCES, byz_frac=BYZ_FRAC)
    obs_dim = env.obs_dim()
    K = env.K
    
    # One MLP per resource (or shared — here per-resource)
    mlps = [SimpleMLP(obs_dim, hidden=16, output_dim=1, seed=seed*10+k) for k in range(K)]
    total_params = sum(m.n_params() for m in mlps)
    
    mu_dual = 1.0
    lr_theta = 0.005
    lr_mu = 0.05
    delta = 0.05
    eps_fd = 0.05  # finite difference epsilon
    
    ep_welfares, ep_survivals, ep_floors = [], [], []
    
    for ep in range(n_episodes):
        obs = env.reset(seed=seed * 10000 + ep)
        total_w, steps = 0.0, 0
        survived = True
        floors = []
        
        for t in range(50):
            # Compute floor for each resource
            phi_k = np.array([mlps[k].forward(obs)[0] for k in range(K)])
            floors.append(phi_k.copy())
            lambdas = np.tile(np.maximum(phi_k, 0.01), (env.n_honest, 1))
            
            obs, reward, surv, terminated = env.step(lambdas)
            total_w += reward
            steps += 1
            if terminated:
                survived = surv
                break
        
        ep_welfares.append(total_w / max(steps, 1))
        ep_survivals.append(float(survived))
        ep_floors.append(float(np.mean(floors)))
        
        # Finite-difference primal-dual update (every 5 episodes)
        if (ep + 1) % 5 == 0 and ep >= 10:
            recent_surv = np.mean(ep_survivals[-10:])
            recent_welfare = np.mean(ep_welfares[-10:])
            
            # Dual update
            mu_dual = max(0, mu_dual + lr_mu * (delta - (1.0 - recent_surv)))
            
            # Primal: finite-difference gradient for each MLP
            for k in range(K):
                flat = mlps[k].get_flat().copy()
                # Sample a random direction (SPSA-style)
                direction = np.random.RandomState(ep * K + k).choice([-1, 1], size=len(flat))
                
                # Evaluate perturbed +
                mlps[k].set_flat(flat + eps_fd * direction)
                obs_t = env.reset(seed=seed * 10000 + ep)
                w_plus = 0
                for t2 in range(min(25, 50)):
                    phi_t = np.array([mlps[kk].forward(obs_t)[0] for kk in range(K)])
                    lam_t = np.tile(np.maximum(phi_t, 0.01), (env.n_honest, 1))
                    obs_t, r_t, _, term_t = env.step(lam_t)
                    w_plus += r_t
                    if term_t: break
                
                # Evaluate perturbed -
                mlps[k].set_flat(flat - eps_fd * direction)
                obs_t = env.reset(seed=seed * 10000 + ep)
                w_minus = 0
                for t2 in range(min(25, 50)):
                    phi_t = np.array([mlps[kk].forward(obs_t)[0] for kk in range(K)])
                    lam_t = np.tile(np.maximum(phi_t, 0.01), (env.n_honest, 1))
                    obs_t, r_t, _, term_t = env.step(lam_t)
                    w_minus += r_t
                    if term_t: break
                
                # SPSA gradient estimate
                grad = (w_plus - w_minus) / (2 * eps_fd) * direction
                
                # Lagrangian update
                new_flat = flat + lr_theta * (grad + mu_dual * 0.01 * direction)
                mlps[k].set_flat(new_flat)
    
    return {
        "welfare_mean": float(np.mean(ep_welfares[-30:])),
        "survival_pct": float(np.mean(ep_survivals[-30:]) * 100),
        "floor_mean": float(np.mean(ep_floors[-30:])),
        "total_params": total_params,
    }


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print(f"  MULTI-RESOURCE PGG + MACCL-NEURAL EXPERIMENT")
    print(f"  N={N_AGENTS}, K={K_RESOURCES}, Byz={BYZ_FRAC}")
    print(f"  Seeds={N_SEEDS}, Episodes={N_EPISODES}")
    print("=" * 70)
    
    t0 = time.time()
    results = {}
    
    # 1. Selfish RL (multi-dim)
    print("\n  [Selfish RL (K=3)]", end=" ", flush=True)
    selfish_data = []
    for s in range(N_SEEDS):
        r = run_selfish_multi(s, N_EPISODES)
        selfish_data.append(r)
        if (s+1) % 5 == 0: print(f"s{s+1}", end=" ", flush=True)
    results["Selfish_K3"] = {
        "welfare": float(np.mean([d["welfare_mean"] for d in selfish_data])),
        "welfare_std": float(np.std([d["welfare_mean"] for d in selfish_data])),
        "survival": float(np.mean([d["survival_pct"] for d in selfish_data])),
        "lambda_mean": float(np.mean([d["lambda_mean"] for d in selfish_data])),
        "lambda_per_resource": "per-agent per-resource mean",
    }
    r = results["Selfish_K3"]
    print(f"=> Surv={r['survival']:.1f}%  λ={r['lambda_mean']:.3f}  W={r['welfare']:.2f}")
    
    # 2. Fixed phi1=1.0 (all resources)
    print("  [Fixed phi1=1.0 (K=3)]", end=" ", flush=True)
    fixed_data = [run_fixed_multi(s, 1.0, N_EPISODES) for s in range(N_SEEDS)]
    results["Fixed_K3"] = {
        "welfare": float(np.mean([d["welfare_mean"] for d in fixed_data])),
        "survival": float(np.mean([d["survival_pct"] for d in fixed_data])),
    }
    print(f"=> Surv={results['Fixed_K3']['survival']:.1f}%")
    
    # 3. MACCL-Neural
    print("  [MACCL-Neural (K=3)]", end=" ", flush=True)
    maccl_data = []
    for s in range(N_SEEDS):
        r = run_maccl_neural(s, N_EPISODES)
        maccl_data.append(r)
        if (s+1) % 5 == 0: print(f"s{s+1}", end=" ", flush=True)
    results["MACCL_Neural_K3"] = {
        "welfare": float(np.mean([d["welfare_mean"] for d in maccl_data])),
        "welfare_std": float(np.std([d["welfare_mean"] for d in maccl_data])),
        "survival": float(np.mean([d["survival_pct"] for d in maccl_data])),
        "survival_std": float(np.std([d["survival_pct"] for d in maccl_data])),
        "floor_mean": float(np.mean([d["floor_mean"] for d in maccl_data])),
        "total_params": maccl_data[0]["total_params"],
    }
    r = results["MACCL_Neural_K3"]
    print(f"=> Surv={r['survival']:.1f}%  floor={r['floor_mean']:.3f}  params={r['total_params']}")
    
    elapsed = time.time() - t0
    
    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY: Multi-Resource PGG (K=3)")
    print(f"  {'Method':25s}  {'Survival':>10s}  {'Welfare':>10s}  {'λ/Floor':>10s}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*10}  {'-'*10}")
    for m in ["Selfish_K3", "Fixed_K3", "MACCL_Neural_K3"]:
        r = results[m]
        lf = r.get("lambda_mean", r.get("floor_mean", "-"))
        lf_str = f"{lf:.3f}" if isinstance(lf, float) else "-"
        print(f"  {m:25s}  {r['survival']:8.1f}%  {r['welfare']:10.2f}  {lf_str:>10s}")
    print(f"{'='*70}")
    print(f"  Time: {elapsed:.0f}s")
    
    # Key finding
    selfish_lam = results["Selfish_K3"]["lambda_mean"]
    print(f"\n  KEY FINDING: Selfish agents converge to λ≈{selfish_lam:.3f} across ALL {K_RESOURCES} resources")
    if selfish_lam < 0.6:
        print(f"  → Nash Trap CONFIRMED in multi-dimensional action space (K={K_RESOURCES})")
    else:
        print(f"  → Nash Trap NOT observed — check parameters")
    
    # Save
    output = {
        "experiment": f"Multi-Resource PGG (K={K_RESOURCES}) + MACCL-Neural",
        "config": {
            "N_AGENTS": N_AGENTS, "K_RESOURCES": K_RESOURCES,
            "BYZ_FRAC": BYZ_FRAC, "N_SEEDS": N_SEEDS, "N_EPISODES": N_EPISODES,
        },
        "results": results,
        "time_seconds": elapsed,
        "key_findings": {
            "nash_trap_multidim": f"Selfish agents converge to λ≈{selfish_lam:.3f} across K={K_RESOURCES} resources",
            "maccl_neural_params": results["MACCL_Neural_K3"]["total_params"],
            "environment_complexity": f"3K+1={3*K_RESOURCES+1}D observation, K={K_RESOURCES}D action per agent",
        }
    }
    
    json_path = os.path.join(OUTPUT_DIR, "multi_resource_results.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {json_path}")
