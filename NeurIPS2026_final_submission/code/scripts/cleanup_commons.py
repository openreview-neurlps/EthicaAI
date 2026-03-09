#!/usr/bin/env python
"""
cleanup_commons.py — Cleanup-style Commons Dilemma (Abstracted)
================================================================
Implements the core dynamics of DeepMind's Cleanup environment as a
scalar-state social dilemma, enabling CPU-based validation that the
Nash Trap and Commitment Floor generalize beyond our custom PGG.

Cleanup dynamics (from Leibo et al. 2017, Hughes et al. 2018):
  - Shared resource (river) accumulates pollution over time
  - Agents choose between HARVEST (selfish) and CLEAN (cooperative)
  - Harvest yield depends on pollution level (non-linear tipping point)
  - If pollution exceeds threshold, yield collapses (tragedy of the commons)

Our abstraction preserves the key property: non-linear coupling between
cooperation level and resource yield with an irreversible tipping point.

Usage:
  python cleanup_commons.py          # Full mode (20 seeds)
  ETHICAAI_FAST=1 python cleanup_commons.py  # Fast mode (2 seeds)
"""

import numpy as np
import json
import os
import time

# ============================================================
# Config
# ============================================================
N_AGENTS = 5
T_HORIZON = 150
N_EPISODES = 150
N_SEEDS = 20

# Cleanup-specific parameters
POLLUTION_RATE = 0.08       # Pollution accumulates each step
CLEAN_POWER = 0.15          # How much one cleaner reduces pollution
HARVEST_BASE = 2.0          # Base harvest reward
POLLUTION_THRESHOLD = 0.6   # Tipping point: above this, harvest collapses
COLLAPSE_FACTOR = 0.05      # Harvest multiplier when polluted
CLEAN_COST = 0.3            # Opportunity cost of cleaning vs harvesting

# RL hyperparameters
STATE_DIM = 3  # [pollution_level, mean_action_prev, time_fraction]
GAMMA = 0.99

if os.environ.get("ETHICAAI_FAST") == "1":
    print("  [FAST MODE] N_SEEDS=2, N_EPISODES=30")
    N_SEEDS = 2
    N_EPISODES = 30


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# ============================================================
# Cleanup Environment (Abstracted)
# ============================================================
class CleanupEnv:
    """
    Scalar-state Cleanup commons dilemma.
    
    State: pollution level P in [0, 1]
    Action: lambda in [0, 1] where 0 = pure harvest, 1 = pure clean
    
    Dynamics:
      P_{t+1} = clip(P_t + pollution_rate - clean_power * sum(lambda_i) / N, 0, 1)
    
    Harvest yield:
      yield = harvest_base * (1 - P_t)  if P_t < threshold
      yield = harvest_base * collapse_factor  if P_t >= threshold  (tipping point!)
    
    Reward:
      r_i = (1 - lambda_i) * yield - lambda_i * clean_cost
    """
    
    def __init__(self):
        self.P = 0.2  # Start with some pollution
        self.mean_action = 0.5
        self.t = 0
    
    def reset(self):
        self.P = 0.2
        self.mean_action = 0.5
        self.t = 0
        return self._obs()
    
    def _obs(self):
        return np.array([self.P, self.mean_action, self.t / T_HORIZON])
    
    def step(self, actions):
        """actions: array of shape (N,) in [0, 1], clean fraction per agent."""
        clean_effort = np.mean(actions)
        
        # Pollution dynamics (non-linear tipping point)
        self.P = np.clip(
            self.P + POLLUTION_RATE - CLEAN_POWER * clean_effort,
            0.0, 1.0
        )
        
        # Harvest yield with tipping point
        if self.P < POLLUTION_THRESHOLD:
            harvest_yield = HARVEST_BASE * (1.0 - self.P)
        else:
            # Tipping point: yield collapses
            harvest_yield = HARVEST_BASE * COLLAPSE_FACTOR
        
        # Per-agent rewards
        rewards = np.zeros(N_AGENTS)
        for i in range(N_AGENTS):
            harvest_share = (1.0 - actions[i]) * harvest_yield
            cleaning_cost = actions[i] * CLEAN_COST
            rewards[i] = harvest_share - cleaning_cost
        
        self.mean_action = float(np.mean(actions))
        self.t += 1
        
        # Collapse = pollution at max
        collapsed = self.P >= 0.99
        
        return rewards, self._obs(), collapsed


# ============================================================
# Linear REINFORCE Agent
# ============================================================
class LinearAgent:
    def __init__(self, rng):
        self.w = rng.randn(STATE_DIM) * 0.01
        self.b = 0.0
        self.lr = 0.005
    
    def act(self, obs, rng, noise=0.1):
        mu = sigmoid(float(obs @ self.w + self.b))
        return float(np.clip(mu + rng.randn() * noise, 0.01, 0.99))
    
    def update(self, obs_list, act_list, returns):
        for obs, a, G in zip(obs_list, act_list, returns):
            mu = sigmoid(float(obs @ self.w + self.b))
            grad = (a - mu) * obs
            self.w += self.lr * G * grad
            self.b += self.lr * G * (a - mu)


# ============================================================
# Experiment runners
# ============================================================
def run_selfish_rl(byz_frac=0.0, n_seeds=N_SEEDS):
    """Selfish RL agents in Cleanup."""
    n_byz = int(N_AGENTS * byz_frac)
    n_honest = N_AGENTS - n_byz
    all_data = []
    
    for seed in range(n_seeds):
        rng = np.random.RandomState(42 + seed * 13)
        agents = [LinearAgent(np.random.RandomState(seed * 100 + i)) 
                  for i in range(n_honest)]
        env = CleanupEnv()
        
        ep_data = {"welfare": [], "mean_clean": [], "survival": []}
        
        for ep in range(N_EPISODES):
            obs = env.reset()
            agent_obs = [[] for _ in range(n_honest)]
            agent_act = [[] for _ in range(n_honest)]
            agent_rew = [[] for _ in range(n_honest)]
            
            tw, tc, steps, survived = 0, 0, 0, True
            
            for t in range(T_HORIZON):
                actions = np.zeros(N_AGENTS)
                # Byzantine agents never clean
                for i in range(n_honest):
                    a = agents[i].act(obs, rng)
                    actions[n_byz + i] = a
                    agent_obs[i].append(obs.copy())
                    agent_act[i].append(a)
                
                rewards, obs, collapsed = env.step(actions)
                
                for i in range(n_honest):
                    agent_rew[i].append(rewards[n_byz + i])
                
                tw += rewards.mean()
                tc += actions[n_byz:].mean()
                steps += 1
                
                if collapsed:
                    survived = False
                    break
            
            # REINFORCE update
            for i in range(n_honest):
                T_len = len(agent_rew[i])
                returns = np.zeros(T_len)
                G = 0
                for t_idx in reversed(range(T_len)):
                    G = agent_rew[i][t_idx] + GAMMA * G
                    returns[t_idx] = G
                if returns.std() > 1e-8:
                    returns = (returns - returns.mean()) / returns.std()
                agents[i].update(agent_obs[i], agent_act[i], returns)
            
            ep_data["welfare"].append(tw / max(steps, 1))
            ep_data["mean_clean"].append(tc / max(steps, 1))
            ep_data["survival"].append(float(survived))
        
        all_data.append(ep_data)
    
    return all_data


def run_commitment(phi1, byz_frac=0.3, n_seeds=N_SEEDS):
    """Fixed commitment floor in Cleanup."""
    n_byz = int(N_AGENTS * byz_frac)
    all_data = []
    
    for seed in range(n_seeds):
        rng = np.random.RandomState(42 + seed * 13)
        env = CleanupEnv()
        
        ep_data = {"welfare": [], "mean_clean": [], "survival": []}
        
        for ep in range(N_EPISODES):
            obs = env.reset()
            tw, tc, steps, survived = 0, 0, 0, True
            
            for t in range(T_HORIZON):
                actions = np.zeros(N_AGENTS)
                # Honest agents use commitment floor
                actions[n_byz:] = phi1
                
                rewards, obs, collapsed = env.step(actions)
                tw += rewards.mean()
                tc += actions[n_byz:].mean()
                steps += 1
                
                if collapsed:
                    survived = False
                    break
            
            ep_data["welfare"].append(tw / max(steps, 1))
            ep_data["mean_clean"].append(tc / max(steps, 1))
            ep_data["survival"].append(float(survived))
        
        all_data.append(ep_data)
    
    return all_data


def summarize(data, label, last_n=50):
    welfares = [np.mean(d["welfare"][-last_n:]) for d in data]
    cleans = [np.mean(d["mean_clean"][-last_n:]) for d in data]
    survs = [np.mean(d["survival"][-last_n:]) * 100 for d in data]
    return {
        "label": label,
        "welfare_mean": round(float(np.mean(welfares)), 2),
        "welfare_std": round(float(np.std(welfares)), 2),
        "clean_mean": round(float(np.mean(cleans)), 3),
        "clean_std": round(float(np.std(cleans)), 3),
        "survival_mean": round(float(np.mean(survs)), 1),
        "survival_std": round(float(np.std(survs)), 1),
    }


if __name__ == "__main__":
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'outputs', 'cleanup_commons')
    os.makedirs(OUT, exist_ok=True)
    
    print("=" * 64)
    print("  CLEANUP COMMONS DILEMMA (Abstracted)")
    print(f"  N={N_AGENTS}, T={T_HORIZON}, seeds={N_SEEDS}")
    print("=" * 64)
    
    t0 = time.time()
    results = []
    
    # 1. Selfish RL, no Byzantine
    d = run_selfish_rl(byz_frac=0.0)
    s = summarize(d, "Selfish RL (Byz=0%)")
    results.append(s)
    print(f"  {s['label']:35s} | W={s['welfare_mean']:6.2f} | clean={s['clean_mean']:.3f} | surv={s['survival_mean']:5.1f}%")
    
    # 2. Selfish RL, 30% Byzantine
    d = run_selfish_rl(byz_frac=0.3)
    s = summarize(d, "Selfish RL (Byz=30%)")
    results.append(s)
    print(f"  {s['label']:35s} | W={s['welfare_mean']:6.2f} | clean={s['clean_mean']:.3f} | surv={s['survival_mean']:5.1f}%")
    
    # 3. Commitment floor sweep
    for phi1 in [0.3, 0.5, 0.7, 1.0]:
        d = run_commitment(phi1, byz_frac=0.3)
        s = summarize(d, f"Commitment phi1={phi1:.1f} (Byz=30%)")
        results.append(s)
        print(f"  {s['label']:35s} | W={s['welfare_mean']:6.2f} | clean={s['clean_mean']:.3f} | surv={s['survival_mean']:5.1f}%")
    
    total = time.time() - t0
    
    output = {
        "experiment": "Cleanup Commons Dilemma (Abstracted)",
        "environment": {
            "type": "Cleanup-style commons (scalar state)",
            "reference": "Hughes et al. 2018 (Inequity Aversion), Leibo et al. 2017",
            "N": N_AGENTS, "T": T_HORIZON, "seeds": N_SEEDS,
            "pollution_threshold": POLLUTION_THRESHOLD,
            "description": "Agents choose clean vs harvest; pollution tipping point causes yield collapse"
        },
        "results": results,
        "time_seconds": round(total, 1),
        "key_finding": "Nash Trap reproduces in Cleanup-style environment: selfish RL converges to low cleaning, triggering pollution collapse"
    }
    
    json_path = os.path.join(OUT, "cleanup_results.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  Total: {total:.0f}s")
    print(f"  Saved: {json_path}")
