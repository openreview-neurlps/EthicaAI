"""
CPO vs MACCL Comparison Experiment
====================================
Compares CPO-style exogenous constraint vs MACCL endogenous floor
in the NonlinearPGG environment (the severe TPSD regime).

CPO Baseline: Each agent independently optimizes individual reward
subject to an exogenous constraint P(survival) >= 1-delta.
Uses Lagrangian relaxation (same as MACCL) but WITHOUT the
commitment floor structure—the agent learns a full policy
lambda(s) instead of a floor phi1(s).

MACPO Baseline: Multi-agent extension where a centralized critic
provides cost signals, but agents still learn individual policies.

Output: cpo_vs_maccl_results.json
"""

import numpy as np
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from envs.nonlinear_pgg_env import NonlinearPGGEnv

# ============================================================
# Configuration
# ============================================================
N_SEEDS = 20
N_EPISODES = 300
N_AGENTS = 20
BYZ_FRAC = 0.3
GAMMA = 0.99

if os.environ.get("ETHICAAI_FAST") == "1":
    N_SEEDS = 3
    N_EPISODES = 50

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'cpo_comparison')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# ============================================================
# Method 1: CPO-style (Exogenous Constraint)
# ============================================================
def run_cpo(seed, n_episodes):
    """
    CPO-style agent: learns lambda(s) = sigmoid(w^T s + b)
    subject to P(survival) >= 95%.
    
    Key difference from MACCL: constraint is applied to the POLICY OUTPUT,
    not as a structural floor. The agent must discover the correct
    lambda level through gradient signals alone.
    """
    env = NonlinearPGGEnv(n_agents=N_AGENTS, multiplier=1.6, byz_frac=BYZ_FRAC)
    n_honest = env.n_honest
    STATE_DIM = 4
    rng = np.random.RandomState(seed)
    
    # Policy parameters (per agent, shared for simplicity)
    w = rng.randn(STATE_DIM) * 0.01
    b = 0.0
    
    # Lagrange multiplier for survival constraint
    mu_cost = 0.5
    lr_policy = 0.01
    lr_mu = 0.05
    delta = 0.05  # Target: P_surv >= 95%
    
    ep_welfares, ep_survivals, ep_lams = [], [], []
    
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        noise = max(0.02, 0.15 - 0.13 * min(ep / n_episodes, 1.0))
        
        traj_obs, traj_acts, traj_rews = [], [], []
        total_w, lam_sum, steps = 0.0, 0.0, 0
        survived = True
        
        for t in range(50):
            # CPO policy: lambda = sigmoid(w^T s + b) + noise
            s = np.array([obs[0], obs[1], obs[2], obs[3]], dtype=np.float32)
            logit = s @ w + b
            base_lambda = sigmoid(logit)
            lam = float(np.clip(base_lambda + rng.randn() * noise, 0.01, 0.99))
            
            lambdas = np.full(n_honest, lam)
            obs, rewards, terminated, _, info = env.step(lambdas)
            
            traj_obs.append(s)
            traj_acts.append(lam)
            traj_rews.append(float(np.mean(rewards)))
            
            total_w += float(np.mean(rewards))
            lam_sum += lam
            steps += 1
            
            if terminated:
                survived = info.get("survived", False)
                break
        
        ep_welfares.append(total_w / max(steps, 1))
        ep_survivals.append(float(survived))
        ep_lams.append(lam_sum / max(steps, 1))
        
        # CPO-style Lagrangian update
        if len(traj_rews) >= 2:
            # Compute returns
            G, returns = 0, np.zeros(len(traj_rews))
            for t_idx in reversed(range(len(traj_rews))):
                G = traj_rews[t_idx] + GAMMA * G
                returns[t_idx] = G
            if returns.std() > 1e-8:
                returns = (returns - returns.mean()) / returns.std()
            
            # Cost = 1 if not survived
            cost = 0.0 if survived else 1.0
            
            # CPO objective: maximize reward - mu * cost
            for t_idx in range(len(returns)):
                s = traj_obs[t_idx]
                a = traj_acts[t_idx]
                logit = s @ w + b
                pred = sigmoid(logit)
                grad = a - pred
                
                # Lagrangian gradient: reward gradient - mu * cost gradient
                reward_grad = returns[t_idx] * grad
                cost_grad = cost * grad  # Cost signal pushes toward survival
                
                w += lr_policy * (reward_grad - mu_cost * cost_grad) * s
                b += lr_policy * (reward_grad - mu_cost * cost_grad)
        
        # Dual update (every 5 episodes)
        if (ep + 1) % 5 == 0 and ep >= 10:
            recent_surv = np.mean(ep_survivals[-10:])
            mu_cost = max(0, mu_cost + lr_mu * (delta - (1.0 - recent_surv)))
    
    return {
        "welfare_mean": float(np.mean(ep_welfares[-30:])),
        "welfare_std": float(np.std(ep_welfares[-30:])),
        "survival_pct": float(np.mean(ep_survivals[-30:]) * 100),
        "lambda_mean": float(np.mean(ep_lams[-30:])),
    }


# ============================================================
# Method 2: MACPO-style (Multi-Agent CPO with Shared Critic)
# ============================================================
def run_macpo(seed, n_episodes):
    """
    MACPO-style: each agent has individual policy, but a shared
    critic provides global state value and cost value.
    
    Constraint: P(survival) >= 95% (exogenous, same as CPO).
    Difference from CPO: agents use shared cost signal from
    centralized critic for better credit assignment.
    """
    env = NonlinearPGGEnv(n_agents=N_AGENTS, multiplier=1.6, byz_frac=BYZ_FRAC)
    n_honest = env.n_honest
    STATE_DIM = 4
    rng = np.random.RandomState(seed)
    
    # Per-agent policy parameters
    agents_w = [rng.randn(STATE_DIM) * 0.01 for _ in range(n_honest)]
    agents_b = [0.0] * n_honest
    
    # Shared value critic
    v_w = rng.randn(STATE_DIM) * 0.01
    v_b = 0.0
    
    # Lagrange multiplier (shared across agents)
    mu_cost = 0.5
    lr_policy = 0.01
    lr_critic = 0.02
    lr_mu = 0.05
    delta = 0.05
    
    ep_welfares, ep_survivals, ep_lams = [], [], []
    
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        noise = max(0.02, 0.15 - 0.13 * min(ep / n_episodes, 1.0))
        
        a_obs = [[] for _ in range(n_honest)]
        a_acts = [[] for _ in range(n_honest)]
        a_rew = [[] for _ in range(n_honest)]
        total_w, lam_sum, steps = 0.0, 0.0, 0
        survived = True
        
        for t in range(50):
            s = np.array([obs[0], obs[1], obs[2], obs[3]], dtype=np.float32)
            lambdas = np.zeros(n_honest)
            
            for i in range(n_honest):
                logit = s @ agents_w[i] + agents_b[i]
                base = sigmoid(logit)
                lam_i = float(np.clip(base + rng.randn() * noise, 0.01, 0.99))
                lambdas[i] = lam_i
                a_obs[i].append(s.copy())
                a_acts[i].append(lam_i)
            
            obs, rewards, terminated, _, info = env.step(lambdas)
            
            for i in range(n_honest):
                a_rew[i].append(float(rewards[i] if i < len(rewards) else np.mean(rewards)))
            
            total_w += float(np.mean(rewards))
            lam_sum += float(lambdas.mean())
            steps += 1
            
            if terminated:
                survived = info.get("survived", False)
                break
        
        ep_welfares.append(total_w / max(steps, 1))
        ep_survivals.append(float(survived))
        ep_lams.append(lam_sum / max(steps, 1))
        
        # MACPO update: per-agent REINFORCE with shared cost
        cost = 0.0 if survived else 1.0
        
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
                
                # MACPO: reward - mu * global_cost (shared signal)
                agents_w[i] += lr_policy * (returns[t_idx] - mu_cost * cost) * grad * o
                agents_b[i] += lr_policy * (returns[t_idx] - mu_cost * cost) * grad
        
        # Shared dual update
        if (ep + 1) % 5 == 0 and ep >= 10:
            recent_surv = np.mean(ep_survivals[-10:])
            mu_cost = max(0, mu_cost + lr_mu * (delta - (1.0 - recent_surv)))
    
    return {
        "welfare_mean": float(np.mean(ep_welfares[-30:])),
        "welfare_std": float(np.std(ep_welfares[-30:])),
        "survival_pct": float(np.mean(ep_survivals[-30:]) * 100),
        "lambda_mean": float(np.mean(ep_lams[-30:])),
    }


# ============================================================
# Method 3: MACCL (Endogenous Floor)
# ============================================================
def run_maccl(seed, n_episodes):
    """MACCL: learns state-dependent floor phi1(R; omega)."""
    env = NonlinearPGGEnv(n_agents=N_AGENTS, multiplier=1.6, byz_frac=BYZ_FRAC)
    n_honest = env.n_honest
    rng = np.random.RandomState(seed)
    
    omega = np.array([0.0, 0.0, 2.0])  # Start high commitment
    mu_dual = 1.0
    lr_omega = 0.01
    lr_mu = 0.05
    delta = 0.05
    
    ep_welfares, ep_survivals, ep_floors = [], [], []
    
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        total_w, steps = 0.0, 0
        survived = True
        floors = []
        
        for t in range(50):
            R = obs[0]
            phi1 = sigmoid(omega[0] * R + omega[1] * R**2 + omega[2])
            floors.append(phi1)
            lambdas = np.full(n_honest, max(phi1, 0.01))
            
            obs, rewards, terminated, _, info = env.step(lambdas)
            total_w += float(np.mean(rewards))
            steps += 1
            
            if terminated:
                survived = info.get("survived", False)
                break
        
        ep_welfares.append(total_w / max(steps, 1))
        ep_survivals.append(float(survived))
        ep_floors.append(float(np.mean(floors)))
        
        # Primal-dual update
        if (ep + 1) % 5 == 0 and ep >= 10:
            recent_surv = np.mean(ep_survivals[-10:])
            recent_welfare = np.mean(ep_welfares[-10:])
            
            constraint_violation = delta - (1.0 - recent_surv)
            mu_dual = max(0, mu_dual + lr_mu * constraint_violation)
            
            for dim in range(3):
                omega_plus = omega.copy()
                omega_plus[dim] += 0.1
                
                test_w = 0
                obs_t, _ = env.reset(seed=seed * 10000 + ep)
                for t2 in range(min(30, 50)):
                    R_t = obs_t[0]
                    phi_t = sigmoid(omega_plus[0]*R_t + omega_plus[1]*R_t**2 + omega_plus[2])
                    lam_t = np.full(n_honest, max(phi_t, 0.01))
                    obs_t, r_t, term_t, _, _ = env.step(lam_t)
                    test_w += float(np.mean(r_t))
                    if term_t:
                        break
                
                grad = (test_w - recent_welfare * 30) / 0.1
                omega[dim] += lr_omega * (grad + mu_dual * 0.1)
    
    return {
        "welfare_mean": float(np.mean(ep_welfares[-30:])),
        "welfare_std": float(np.std(ep_welfares[-30:])),
        "survival_pct": float(np.mean(ep_survivals[-30:]) * 100),
        "floor_mean": float(np.mean(ep_floors[-30:])),
    }


# ============================================================
# Method 4: Fixed Floor (phi1=1.0)
# ============================================================
def run_fixed(seed, phi1, n_episodes):
    env = NonlinearPGGEnv(n_agents=N_AGENTS, multiplier=1.6, byz_frac=BYZ_FRAC)
    n_honest = env.n_honest
    
    ep_welfares, ep_survivals = [], []
    
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        total_w, steps = 0.0, 0
        survived = True
        
        for t in range(50):
            lambdas = np.full(n_honest, phi1)
            obs, rewards, terminated, _, info = env.step(lambdas)
            total_w += float(np.mean(rewards))
            steps += 1
            if terminated:
                survived = info.get("survived", False)
                break
        
        ep_welfares.append(total_w / max(steps, 1))
        ep_survivals.append(float(survived))
    
    return {
        "welfare_mean": float(np.mean(ep_welfares[-30:])),
        "welfare_std": float(np.std(ep_welfares[-30:])),
        "survival_pct": float(np.mean(ep_survivals[-30:]) * 100),
    }


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  CPO vs MACPO vs MACCL COMPARISON")
    print(f"  N={N_AGENTS}, Byz={BYZ_FRAC}, Seeds={N_SEEDS}, Episodes={N_EPISODES}")
    print("=" * 70)
    
    t0 = time.time()
    results = {}
    
    methods = [
        ("CPO", run_cpo),
        ("MACPO", run_macpo),
        ("MACCL", run_maccl),
    ]
    
    for method_name, method_fn in methods:
        print(f"\n  [{method_name}]", end=" ", flush=True)
        data = []
        for s in range(N_SEEDS):
            r = method_fn(s, N_EPISODES)
            data.append(r)
            if (s + 1) % 5 == 0:
                print(f"s{s+1}", end=" ", flush=True)
        
        results[method_name] = {
            "welfare": float(np.mean([d["welfare_mean"] for d in data])),
            "welfare_std": float(np.std([d["welfare_mean"] for d in data])),
            "survival": float(np.mean([d["survival_pct"] for d in data])),
            "survival_std": float(np.std([d["survival_pct"] for d in data])),
        }
        if "lambda_mean" in data[0]:
            results[method_name]["lambda_mean"] = float(np.mean([d["lambda_mean"] for d in data]))
        if "floor_mean" in data[0]:
            results[method_name]["floor_mean"] = float(np.mean([d["floor_mean"] for d in data]))
        
        print(f"=> Surv={results[method_name]['survival']:.1f}%  W={results[method_name]['welfare']:.2f}")
    
    # Add fixed baselines
    print("\n  [Fixed phi1=1.0]", end=" ", flush=True)
    fixed_data = [run_fixed(s, 1.0, N_EPISODES) for s in range(N_SEEDS)]
    results["Fixed_1.0"] = {
        "welfare": float(np.mean([d["welfare_mean"] for d in fixed_data])),
        "welfare_std": float(np.std([d["welfare_mean"] for d in fixed_data])),
        "survival": float(np.mean([d["survival_pct"] for d in fixed_data])),
    }
    print(f"=> Surv={results['Fixed_1.0']['survival']:.1f}%")
    
    elapsed = time.time() - t0
    
    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY: CPO vs MACPO vs MACCL")
    print(f"  {'Method':15s}  {'Survival':>10s}  {'Welfare':>10s}  {'Lambda/Floor':>12s}")
    print(f"  {'-'*15}  {'-'*10}  {'-'*10}  {'-'*12}")
    for m in ["CPO", "MACPO", "MACCL", "Fixed_1.0"]:
        r = results[m]
        lf = r.get("lambda_mean", r.get("floor_mean", "-"))
        lf_str = f"{lf:.3f}" if isinstance(lf, float) else lf
        print(f"  {m:15s}  {r['survival']:8.1f}%  {r['welfare']:10.2f}  {lf_str:>12s}")
    print(f"{'='*70}")
    
    # Save
    output = {
        "experiment": "CPO vs MACPO vs MACCL Comparison",
        "config": {
            "N_AGENTS": N_AGENTS, "BYZ_FRAC": BYZ_FRAC,
            "N_SEEDS": N_SEEDS, "N_EPISODES": N_EPISODES,
        },
        "results": results,
        "time_seconds": elapsed,
        "interpretation": {
            "CPO": "Exogenous constraint: learns lambda(s) subject to P(surv)>=95%. No structural floor.",
            "MACPO": "Multi-agent CPO with shared cost signal. Per-agent policies, no floor.",
            "MACCL": "Endogenous floor: learns phi1(R;omega), structural constraint derived from tipping-point.",
            "Fixed_1.0": "Unconditional commitment baseline.",
        }
    }
    
    json_path = os.path.join(OUTPUT_DIR, "cpo_vs_maccl_results.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {json_path}")
    print(f"  Total time: {elapsed:.0f}s")
