"""
MACCL Cross-Environment Validation
====================================
Applies MACCL (Multi-Agent Constrained Commitment Learning) across
three environments to demonstrate generalization:
  1. PGG (NonlinearPGGEnv) — provision dilemma
  2. CPR (CommonPoolResource) — appropriation dilemma  
  3. Cleanup (CleanupEnv) — pollution/harvest dilemma

For each environment, compares:
  - Selfish RL (no commitment)
  - Fixed commitment (phi1=1.0)
  - MACCL (learned state-dependent phi1)

Output: multi_env_maccl_results.json + comparison table
Dependencies: NumPy, environments from scripts/
"""

import numpy as np
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from envs.nonlinear_pgg_env import NonlinearPGGEnv
from cpr_experiment import CommonPoolResource
from cleanup_commons import CleanupEnv

# ============================================================
# Configuration
# ============================================================
N_SEEDS = 20
N_EPISODES = 200
GAMMA = 0.99

if os.environ.get("ETHICAAI_FAST") == "1":
    print("  [FAST MODE]")
    N_SEEDS = 3
    N_EPISODES = 50

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'maccl_multi_env')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# ============================================================
# Environment Adapters — Uniform interface
# ============================================================
class PGGAdapter:
    """Adapts NonlinearPGGEnv to a common interface."""
    def __init__(self, seed):
        self.env = NonlinearPGGEnv(n_agents=20, multiplier=1.6, byz_frac=0.3)
        self.n_honest = self.env.n_honest
        self.name = "PGG"
    
    def reset(self, seed):
        obs, _ = self.env.reset(seed=seed)
        return obs
    
    def step(self, lambdas):
        """lambdas: commitment levels for honest agents."""
        obs, rewards, terminated, _, info = self.env.step(lambdas)
        return obs, float(np.mean(rewards)), info.get("survived", True), terminated
    
    def obs_dim(self):
        return 4


class CPRAdapter:
    """Adapts CommonPoolResource to a common interface."""
    def __init__(self, seed):
        self.env = CommonPoolResource(n_agents=20, byz_frac=0.3)
        self.n_honest = self.env.n_honest
        self.rng = np.random.RandomState(seed)
        self.name = "CPR"
    
    def reset(self, seed):
        self.rng = np.random.RandomState(seed)
        R = self.env.reset(self.rng)
        return np.array([R / self.env.K, 0.5, float(R < self.env.r_crit), 0.0], dtype=np.float32)
    
    def step(self, lambdas):
        """lambdas: restraint levels for honest agents (maps to CPR lambda)."""
        full_lambdas = np.zeros(self.env.N)
        full_lambdas[:self.n_honest] = lambdas
        payoffs, survived, R, terminated = self.env.step(full_lambdas, self.rng)
        obs = np.array([R / self.env.K, float(np.mean(lambdas)),
                       float(R < self.env.r_crit), self.env.t / self.env.T], dtype=np.float32)
        return obs, float(np.mean(payoffs[:self.n_honest])), survived, terminated
    
    def obs_dim(self):
        return 4


class CleanupAdapter:
    """Adapts CleanupEnv to a common interface."""
    def __init__(self, seed):
        self.env = CleanupEnv()
        self.n_honest = 5  # CleanupEnv uses N_AGENTS=5
        self.name = "Cleanup"
    
    def reset(self, seed):
        obs = self.env.reset()
        return np.array([obs[0], obs[1], obs[2], 0.0], dtype=np.float32)  # pad to 4D
    
    def step(self, lambdas):
        """lambdas: cleaning commitment for agents."""
        rewards, obs, collapsed = self.env.step(lambdas)
        survived = not collapsed
        obs_4d = np.array([obs[0], obs[1], obs[2], 0.0], dtype=np.float32)
        return obs_4d, float(np.mean(rewards)), survived, collapsed or self.env.t >= 150
    
    def obs_dim(self):
        return 4


# ============================================================
# MACCL Core — Environment-agnostic
# ============================================================
def maccl_commitment_floor(obs, omega):
    """State-dependent commitment floor: phi1 = sigmoid(w1*R + w2*R^2 + w3)."""
    R = obs[0]  # Resource/pollution level
    logit = omega[0] * R + omega[1] * R**2 + omega[2]
    return sigmoid(logit)


def run_maccl(env_class, seed, n_episodes):
    """Run MACCL on a given environment."""
    env = env_class(seed)
    n_honest = env.n_honest
    
    # MACCL parameters
    omega = np.array([0.0, 0.0, 2.0])  # Start with high commitment
    mu_dual = 1.0  # Lagrange multiplier
    lr_omega = 0.01
    lr_mu = 0.05
    delta = 0.05  # Survival constraint: P_surv >= 95%
    
    ep_welfares, ep_survivals, ep_floors = [], [], []
    
    for ep in range(n_episodes):
        obs = env.reset(seed=seed * 10000 + ep)
        total_w, steps = 0.0, 0
        survived = True
        floors = []
        
        for t in range(50 if env.name != "Cleanup" else 150):
            phi1 = maccl_commitment_floor(obs, omega)
            floors.append(phi1)
            lambdas = np.full(n_honest, max(phi1, 0.01))
            
            obs, reward, surv, terminated = env.step(lambdas)
            total_w += reward
            steps += 1
            
            if terminated:
                survived = surv
                break
        
        ep_welfares.append(total_w / max(steps, 1))
        ep_survivals.append(float(survived))
        ep_floors.append(float(np.mean(floors)))
        
        # Primal-dual update (every 5 episodes for stability)
        if (ep + 1) % 5 == 0 and ep >= 10:
            recent_surv = np.mean(ep_survivals[-10:])
            recent_welfare = np.mean(ep_welfares[-10:])
            
            # Dual update: mu tracks survival constraint violation
            constraint_violation = delta - (1.0 - recent_surv)
            mu_dual = max(0, mu_dual + lr_mu * constraint_violation)
            
            # Primal update: gradient of Lagrangian w.r.t omega
            # Finite difference
            for dim in range(3):
                omega_plus = omega.copy()
                omega_plus[dim] += 0.1
                
                # Quick eval with perturbed omega
                test_w = 0
                obs_t = env.reset(seed=seed * 10000 + ep)
                for t in range(min(30, 50)):
                    phi_t = maccl_commitment_floor(obs_t, omega_plus)
                    lambdas_t = np.full(n_honest, max(phi_t, 0.01))
                    obs_t, r_t, _, term_t = env.step(lambdas_t)
                    test_w += r_t
                    if term_t:
                        break
                
                grad = (test_w - recent_welfare * 30) / 0.1
                omega[dim] += lr_omega * (grad + mu_dual * 0.1)  # Lagrangian gradient
    
    return {
        "welfare_mean": float(np.mean(ep_welfares[-30:])),
        "survival_pct": float(np.mean(ep_survivals[-30:]) * 100),
        "floor_mean": float(np.mean(ep_floors[-30:])),
    }


def run_selfish(env_class, seed, n_episodes):
    """Selfish RL baseline (REINFORCE with linear policy)."""
    env = env_class(seed)
    n_honest = env.n_honest
    STATE_DIM = env.obs_dim()
    
    agents_w = [np.random.RandomState(seed * 100 + i).randn(STATE_DIM) * 0.01
                for i in range(n_honest)]
    agents_b = [0.0] * n_honest
    lr = 0.01
    rng = np.random.RandomState(42 + seed)
    
    ep_welfares, ep_survivals, ep_lams = [], [], []
    
    for ep in range(n_episodes):
        obs = env.reset(seed=seed * 10000 + ep)
        noise = 0.15 - 0.13 * min(ep / n_episodes, 1.0)
        
        a_obs = [[] for _ in range(n_honest)]
        a_acts = [[] for _ in range(n_honest)]
        a_rew = [[] for _ in range(n_honest)]
        total_w, lam_sum, steps = 0.0, 0.0, 0
        survived = True
        
        max_t = 50 if env.name != "Cleanup" else 150
        for t in range(max_t):
            lambdas = np.zeros(n_honest)
            for i in range(n_honest):
                logit = obs[:STATE_DIM] @ agents_w[i] + agents_b[i]
                base = sigmoid(logit)
                lam_i = float(np.clip(base + rng.randn() * noise, 0.01, 0.99))
                lambdas[i] = lam_i
                a_obs[i].append(obs[:STATE_DIM].copy())
                a_acts[i].append(lam_i)
            
            obs, reward, surv, terminated = env.step(lambdas)
            for i in range(n_honest):
                a_rew[i].append(reward)
            
            total_w += reward
            lam_sum += float(lambdas.mean())
            steps += 1
            
            if terminated:
                survived = surv
                break
        
        # REINFORCE update
        for i in range(n_honest):
            if len(a_rew[i]) < 2:
                continue
            G, returns = 0, np.zeros(len(a_rew[i]))
            for t_idx in reversed(range(len(a_rew[i]))):
                G = a_rew[i][t_idx] + GAMMA * G
                returns[t_idx] = G
            if returns.std() > 1e-8:
                returns = (returns - returns.mean()) / returns.std()
            for t_idx in range(len(returns)):
                o = a_obs[i][t_idx]
                a = a_acts[i][t_idx]
                logit = o @ agents_w[i] + agents_b[i]
                pred = sigmoid(logit)
                grad = a - pred
                agents_w[i] += lr * returns[t_idx] * grad * o
                agents_b[i] += lr * returns[t_idx] * grad
        
        ep_welfares.append(total_w / max(steps, 1))
        ep_survivals.append(float(survived))
        ep_lams.append(lam_sum / max(steps, 1))
    
    return {
        "welfare_mean": float(np.mean(ep_welfares[-30:])),
        "survival_pct": float(np.mean(ep_survivals[-30:]) * 100),
        "lambda_mean": float(np.mean(ep_lams[-30:])),
    }


def run_fixed_commitment(env_class, seed, phi1, n_episodes):
    """Fixed commitment floor baseline."""
    env = env_class(seed)
    n_honest = env.n_honest
    
    ep_welfares, ep_survivals = [], []
    
    for ep in range(n_episodes):
        obs = env.reset(seed=seed * 10000 + ep)
        total_w, steps = 0.0, 0
        survived = True
        
        max_t = 50 if env.name != "Cleanup" else 150
        for t in range(max_t):
            lambdas = np.full(n_honest, phi1)
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
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  MACCL CROSS-ENVIRONMENT VALIDATION")
    print(f"  Seeds={N_SEEDS}, Episodes={N_EPISODES}")
    print("=" * 70)
    
    t0 = time.time()
    all_results = {}
    
    env_classes = [
        ("PGG", PGGAdapter),
        ("CPR", CPRAdapter),
        ("Cleanup", CleanupAdapter),
    ]
    
    for env_name, env_class in env_classes:
        print(f"\n{'='*50}")
        print(f"  Environment: {env_name}")
        print(f"{'='*50}")
        
        env_results = {}
        
        # 1. Selfish RL
        print(f"  [Selfish RL]")
        selfish_data = []
        for s in range(N_SEEDS):
            r = run_selfish(env_class, s, N_EPISODES)
            selfish_data.append(r)
        env_results["selfish"] = {
            "welfare": float(np.mean([d["welfare_mean"] for d in selfish_data])),
            "welfare_std": float(np.std([d["welfare_mean"] for d in selfish_data])),
            "survival": float(np.mean([d["survival_pct"] for d in selfish_data])),
            "survival_std": float(np.std([d["survival_pct"] for d in selfish_data])),
        }
        print(f"    Surv={env_results['selfish']['survival']:.1f}%  W={env_results['selfish']['welfare']:.2f}")
        
        # 2. Fixed phi1=1.0
        print(f"  [Fixed phi1=1.0]")
        fixed_data = []
        for s in range(N_SEEDS):
            r = run_fixed_commitment(env_class, s, 1.0, N_EPISODES)
            fixed_data.append(r)
        env_results["fixed_1.0"] = {
            "welfare": float(np.mean([d["welfare_mean"] for d in fixed_data])),
            "welfare_std": float(np.std([d["welfare_mean"] for d in fixed_data])),
            "survival": float(np.mean([d["survival_pct"] for d in fixed_data])),
            "survival_std": float(np.std([d["survival_pct"] for d in fixed_data])),
        }
        print(f"    Surv={env_results['fixed_1.0']['survival']:.1f}%  W={env_results['fixed_1.0']['welfare']:.2f}")
        
        # 3. MACCL
        print(f"  [MACCL]")
        maccl_data = []
        for s in range(N_SEEDS):
            r = run_maccl(env_class, s, N_EPISODES)
            maccl_data.append(r)
        env_results["maccl"] = {
            "welfare": float(np.mean([d["welfare_mean"] for d in maccl_data])),
            "welfare_std": float(np.std([d["welfare_mean"] for d in maccl_data])),
            "survival": float(np.mean([d["survival_pct"] for d in maccl_data])),
            "survival_std": float(np.std([d["survival_pct"] for d in maccl_data])),
            "floor_mean": float(np.mean([d["floor_mean"] for d in maccl_data])),
        }
        print(f"    Surv={env_results['maccl']['survival']:.1f}%  W={env_results['maccl']['welfare']:.2f}  floor={env_results['maccl']['floor_mean']:.3f}")
        
        all_results[env_name] = env_results
    
    elapsed = time.time() - t0
    
    # Summary table
    print(f"\n{'='*70}")
    print("  SUMMARY TABLE")
    print(f"  {'Env':10s}  {'Method':15s}  {'Survival':>10s}  {'Welfare':>10s}")
    print(f"  {'-'*10}  {'-'*15}  {'-'*10}  {'-'*10}")
    for env_name in ["PGG", "CPR", "Cleanup"]:
        for method in ["selfish", "fixed_1.0", "maccl"]:
            r = all_results[env_name][method]
            print(f"  {env_name:10s}  {method:15s}  {r['survival']:8.1f}%  {r['welfare']:10.2f}")
    print(f"{'='*70}")
    
    # Save
    output = {
        "experiment": "MACCL Cross-Environment Validation",
        "config": {"N_SEEDS": N_SEEDS, "N_EPISODES": N_EPISODES},
        "results": all_results,
        "time_seconds": elapsed,
    }
    
    json_path = os.path.join(OUTPUT_DIR, "multi_env_maccl_results.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {json_path}")
    print(f"  Total time: {elapsed:.0f}s")
