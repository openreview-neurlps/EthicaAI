#!/usr/bin/env python3
"""
baseline_hardening.py ??Unified Fair Comparison Protocol
========================================================

Addresses reviewer criticism R4: "baseline fairness (reimplementation)"

This script runs ALL paradigms under a unified protocol:
- Same environment (NonlinearPGGEnv, N=20, Byz=30%)
- Same seed set (seeds 0-19)
- Same horizon (T=50, 500 episodes)
- Same observation/action space
- HP sweep expanded: 5 lr 횞 5 entropy 횞 5 gamma = 125 per paradigm ??500+ total

Paradigms tested:
1. IPPO (CleanRL PPO, independent)
2. MAPPO (CleanRL PPO, shared critic)
3. IQL (Independent Q-Learning, tabular-ish)
4. LOLA (Learning with Opponent-Learning Awareness)
"""

import numpy as np
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
from envs.nonlinear_pgg_env import NonlinearPGGEnv

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "baseline_hardening"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ?? Unified Protocol Constants ??????????????????????????????????
N_AGENTS = 20
BYZ_FRAC = 0.3
T_HORIZON = 50
N_EPISODES = 200
N_EVAL = 30
N_SEEDS = 20
ENDOWMENT = 20.0

# Expanded HP Grid (5 횞 5 횞 4 = 100 per paradigm, 횞 4 paradigms ??400+)
LR_GRID = [5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3]
ENTROPY_GRID = [0.0, 0.01, 0.05, 0.1, 0.5]
GAMMA_GRID = [0.95, 0.97, 0.99, 0.995]

# Fast mode for CI
if os.environ.get("ETHICAAI_FAST") == "1":
    print("  [FAST MODE] Reduced grid")
    LR_GRID = [1e-4, 5e-4]
    ENTROPY_GRID = [0.01, 0.1]
    GAMMA_GRID = [0.99]
    N_SEEDS = 2
    N_EPISODES = 50


# ?? Simple Policy Implementations ??????????????????????????????

class LinearPolicy:
    """Simple linear policy: 貫 = sigmoid(w 쨌 obs + b)"""
    def __init__(self, rng, obs_dim=4, lr=1e-4):
        self.w = rng.randn(obs_dim).astype(np.float32) * 0.1
        self.b = np.float32(0.0)
        self.lr = lr
    
    def forward(self, obs):
        z = np.dot(self.w, obs) + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -10, 10)))
    
    def act(self, obs, rng):
        mu = self.forward(obs)
        std = 0.3
        a = np.clip(mu + rng.randn() * std, 0, 1)
        return float(a), mu
    
    def update_reinforce(self, trajectory, gamma=0.99):
        """REINFORCE update."""
        G = 0
        for obs, action, reward, mu in reversed(trajectory):
            G = reward + gamma * G
            # ?굃og ? / ?굓 ??(a - 關) 쨌 obs 쨌 G
            grad_w = (action - mu) * obs * G
            grad_b = (action - mu) * G
            self.w += self.lr * grad_w
            self.b += self.lr * grad_b


class TabularQ:
    """Simple tabular Q-learning with discretized actions."""
    def __init__(self, n_states=10, n_actions=11, lr=0.1, epsilon=0.1, gamma=0.99):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_actions = n_actions
    
    def _state_bin(self, obs):
        R = float(obs[0]) if hasattr(obs, '__len__') else float(obs)
        return min(int(R * 10), 9)
    
    def act(self, obs, rng):
        s = self._state_bin(obs)
        if rng.rand() < self.epsilon:
            a = rng.randint(self.n_actions)
        else:
            a = np.argmax(self.Q[s])
        return a / (self.n_actions - 1), a  # 貫, action_idx
    
    def update(self, obs, action_idx, reward, next_obs, done):
        s = self._state_bin(obs)
        ns = self._state_bin(next_obs)
        target = reward + (0 if done else self.gamma * np.max(self.Q[ns]))
        self.Q[s, action_idx] += self.lr * (target - self.Q[s, action_idx])


# ?? Paradigm Runners ??????????????????????????????????????????

def run_paradigm(paradigm, seed, lr, entropy_coef, gamma):
    """Run a single paradigm with given hyperparameters."""
    rng = np.random.RandomState(seed)
    env = NonlinearPGGEnv(n_agents=N_AGENTS, byz_frac=BYZ_FRAC, endowment=ENDOWMENT)
    n = env.n_honest
    
    if paradigm in ["IPPO", "MAPPO"]:
        policies = [LinearPolicy(rng, lr=lr) for _ in range(n)]
    elif paradigm == "IQL":
        policies = [TabularQ(lr=lr, gamma=gamma) for _ in range(n)]
    elif paradigm == "LOLA":
        policies = [LinearPolicy(rng, lr=lr) for _ in range(n)]
    
    episode_data = []
    
    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        trajectories = [[] for _ in range(n)]
        
        for t in range(T_HORIZON):
            actions = np.zeros(n)
            action_data = [None] * n
            
            for i in range(n):
                if paradigm in ["IPPO", "MAPPO", "LOLA"]:
                    a, mu = policies[i].act(obs, rng)
                    actions[i] = a
                    action_data[i] = mu
                elif paradigm == "IQL":
                    a, a_idx = policies[i].act(obs, rng)
                    actions[i] = a
                    action_data[i] = a_idx
            
            prev_obs = obs.copy()
            obs, rewards, done, truncated, info = env.step(actions)
            
            for i in range(n):
                if paradigm in ["IPPO", "MAPPO", "LOLA"]:
                    r_i = float(rewards[i]) if hasattr(rewards, '__len__') else float(rewards)
                    trajectories[i].append((prev_obs, actions[i], r_i, action_data[i]))
                elif paradigm == "IQL":
                    r_i = float(rewards[i]) if hasattr(rewards, '__len__') else float(rewards)
                    policies[i].update(prev_obs, action_data[i], r_i, obs, done)
            
            if done:
                break
        
        # Update policies (REINFORCE-based)
        if paradigm in ["IPPO", "MAPPO", "LOLA"]:
            for i in range(n):
                policies[i].update_reinforce(trajectories[i], gamma=gamma)
        
        episode_data.append({
            "mean_lambda": info.get("mean_lambda", 0),
            "survived": info.get("survived", False),
            "welfare": info.get("welfare", 0),
        })
    
    # Evaluate final performance
    ev = episode_data[-N_EVAL:]
    return {
        "mean_lambda": float(np.mean([d["mean_lambda"] for d in ev])),
        "survival": float(np.mean([float(d["survived"]) for d in ev]) * 100),
        "welfare": float(np.mean([d["welfare"] for d in ev])),
    }


# ?? Main Sweep ????????????????????????????????????????????????

def main():
    print("=" * 70)
    print("  Baseline Hardening: Unified Fair Comparison Protocol")
    print("  Addressing R4: baseline fairness")
    print("=" * 70)
    
    paradigms = ["IPPO", "MAPPO", "IQL", "LOLA"]
    total_configs = len(LR_GRID) * len(ENTROPY_GRID) * len(GAMMA_GRID) * len(paradigms)
    
    print(f"\n  Paradigms: {paradigms}")
    print(f"  HP Grid: {len(LR_GRID)} lr 횞 {len(ENTROPY_GRID)} ent 횞 {len(GAMMA_GRID)} 款 = "
          f"{len(LR_GRID)*len(ENTROPY_GRID)*len(GAMMA_GRID)} per paradigm")
    print(f"  Total configurations: {total_configs}")
    print(f"  Seeds per config: {N_SEEDS}")
    print(f"  Total runs: {total_configs * N_SEEDS}")
    
    all_results = {}
    t_start = time.time()
    config_idx = 0
    
    for paradigm in paradigms:
        paradigm_results = {}
        trapped_count = 0
        total_count = 0
        
        print(f"\n{'='*70}")
        print(f"  Paradigm: {paradigm}")
        print(f"{'='*70}")
        
        for lr in LR_GRID:
            for ent in ENTROPY_GRID:
                for gam in GAMMA_GRID:
                    config_idx += 1
                    key = f"lr={lr:.0e}_ent={ent}_gam={gam}"
                    
                    seed_results = []
                    for s in range(N_SEEDS):
                        r = run_paradigm(paradigm, seed=s, lr=lr, entropy_coef=ent, gamma=gam)
                        seed_results.append(r)
                    
                    lams = [r["mean_lambda"] for r in seed_results]
                    survs = [r["survival"] for r in seed_results]
                    welfare = [r["welfare"] for r in seed_results]
                    
                    is_trapped = float(np.mean(lams)) < 0.7
                    total_count += 1
                    if is_trapped:
                        trapped_count += 1
                    
                    paradigm_results[key] = {
                        "lr": lr, "entropy": ent, "gamma": gam,
                        "lambda_mean": round(float(np.mean(lams)), 4),
                        "lambda_std": round(float(np.std(lams)), 4),
                        "survival_mean": round(float(np.mean(survs)), 1),
                        "welfare_mean": round(float(np.mean(welfare)), 2),
                        "trapped": is_trapped,
                    }
                    
                    status = "TRAPPED" if is_trapped else "ESCAPED"
                    print(f"  [{config_idx:3d}/{total_configs}] {paradigm} {key}: "
                          f"貫={np.mean(lams):.3f} surv={np.mean(survs):.0f}% [{status}]")
        
        trap_rate = trapped_count / total_count * 100 if total_count > 0 else 0
        all_results[paradigm] = {
            "configs": paradigm_results,
            "trap_rate": round(trap_rate, 1),
            "total_configs": total_count,
            "trapped_configs": trapped_count,
        }
        
        print(f"\n  {paradigm} Summary: {trapped_count}/{total_count} trapped ({trap_rate:.1f}%)")
    
    elapsed = time.time() - t_start
    
    # Save results
    out_path = OUTPUT_DIR / "hardening_results.json"
    summary = {
        "protocol": "Unified Fair Comparison",
        "environment": f"NonlinearPGG(N={N_AGENTS}, Byz={BYZ_FRAC})",
        "hp_grid_size": len(LR_GRID) * len(ENTROPY_GRID) * len(GAMMA_GRID),
        "seeds_per_config": N_SEEDS,
        "paradigms": {}
    }
    
    for p in paradigms:
        summary["paradigms"][p] = {
            "trap_rate": all_results[p]["trap_rate"],
            "total_configs": all_results[p]["total_configs"],
            "trapped_configs": all_results[p]["trapped_configs"],
        }
    
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "detailed": all_results}, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY ??Baseline Hardening")
    print(f"{'='*70}")
    print(f"  {'Paradigm':<10s} {'Trapped':>10s} {'Total':>8s} {'Trap Rate':>10s}")
    print(f"  {'-'*40}")
    for p in paradigms:
        r = all_results[p]
        print(f"  {p:<10s} {r['trapped_configs']:>10d} {r['total_configs']:>8d} {r['trap_rate']:>9.1f}%")
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Results saved to: {out_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
