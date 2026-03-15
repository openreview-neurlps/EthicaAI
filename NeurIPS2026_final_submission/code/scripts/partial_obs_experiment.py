"""
Phase D: Partial Observability Experiment
===========================================
Tests algorithms under Noisy, Delayed, and Local observability of R_t.
"""
import numpy as np
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
from cleanrl_mappo_pgg import (
    NonlinearPGGEnv, MLPActor, MLPCritic, compute_gae, ppo_update_actor, bootstrap_ci
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / os.environ.get("ETHICAAI_OUTDIR", "outputs") / "partial_obs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_EPISODES_IPPO = 150
N_EPISODES_RULE = 50
T = 50
N_SEEDS_RULE = 10
N_SEEDS_IPPO = 3

class PartialObsEnv(NonlinearPGGEnv):
    def __init__(self, obs_type="full", noise_std=0.0, delay_k=0, **kwargs):
        self.obs_type = obs_type
        self.noise_std = noise_std
        self.delay_k = delay_k
        self.R_history = []
        super().__init__(**kwargs)
        
    def reset(self, seed=None):
        obs, info = super().reset(seed=seed)
        self.R_history = [self.R] * max(1, self.delay_k + 1)
        return self._modify_obs(obs), info
        
    def step(self, actions_honest):
        obs, rewards, terminated, truncated, info = super().step(actions_honest)
        self.R_history.append(self.R)
        return self._modify_obs(obs), rewards, terminated, truncated, info
        
    def _modify_obs(self, obs):
        if self.obs_type == "full":
            return obs
            
        mod_obs = obs.copy()
        
        if self.obs_type == "noisy":
            mod_obs[0] = np.clip(self.R + self.np_random.normal(0, self.noise_std), 0, 1)
            mod_obs[2] = float(mod_obs[0] < self.r_crit)
        elif self.obs_type == "delayed":
            delayed_R = self.R_history[max(0, len(self.R_history) - 1 - self.delay_k)]
            mod_obs[0] = delayed_R
            mod_obs[2] = float(delayed_R < self.r_crit)
        elif self.obs_type == "local":
            mod_obs[0] = 0.5  # Masked
            mod_obs[2] = 0.0
            
        return mod_obs


def compute_lambda(R_t, variant, prev_lambda):
    alpha = 0.9
    if variant == "situational":
        crisis_lam = 0.21
    else:  # unconditional
        crisis_lam = 1.0
        
    if R_t < 0.15:
        target = crisis_lam
    elif R_t >= 0.25:
        target = min(1.0, 0.5 + 0.3)
    else:
        target = 0.5
        
    target = max(0.0, target - 0.1 * (1 - R_t))
    target = np.clip(target, 0.0, 1.0)
    return np.clip(alpha * prev_lambda + (1 - alpha) * target, 0.0, 1.0)


def run_rule(seed, variant, obs_type, noise_std=0.0, delay_k=0):
    rng = np.random.RandomState(seed)
    env = PartialObsEnv(obs_type=obs_type, noise_std=noise_std, delay_k=delay_k, byz_frac=0.3)
    
    episodes = []
    n = env.n_honest
    
    for ep in range(N_EPISODES_RULE):
        obs, _ = env.reset(seed=seed*10000+ep)
        lambdas = np.full(n, 0.5)
        
        for t in range(T):
            R_obs = obs[0]
            for i in range(n):
                lambdas[i] = compute_lambda(R_obs, variant, lambdas[i])
            obs, rewards, done, _, info = env.step(lambdas)
            if done: break
            
        episodes.append({
            "mean_lambda": float(np.mean(lambdas)),
            "survived": info.get("survived", False),
            "welfare": info.get("welfare", 0.0),
        })
        
    ev = episodes[-10:]
    return {
        "lambda": float(np.mean([d["mean_lambda"] for d in ev])),
        "survival": float(np.mean([float(d["survived"]) for d in ev]) * 100),
        "welfare": float(np.mean([d["welfare"] for d in ev])),
    }


def run_ippo(seed, obs_type, noise_std=0.0, delay_k=0):
    rng = np.random.RandomState(seed)
    env = PartialObsEnv(obs_type=obs_type, noise_std=noise_std, delay_k=delay_k, byz_frac=0.3)
    n = env.n_honest
    
    actors = [MLPActor(rng, lr=2.5e-4) for _ in range(n)]
    critics = [MLPCritic(rng, lr=2.5e-4) for _ in range(n)]
    
    episodes = []
    
    for ep in range(N_EPISODES_IPPO):
        obs, _ = env.reset(seed=seed*10000+ep)
        obs_buf = [[] for _ in range(n)]
        act_buf = [[] for _ in range(n)]
        lp_buf = [[] for _ in range(n)]
        rew_buf = [[] for _ in range(n)]
        val_buf = [[] for _ in range(n)]
        
        for t in range(T):
            actions = np.zeros(n)
            for i in range(n):
                a, lp, mu = actors[i].act(obs, rng)
                actions[i] = a
                obs_buf[i].append(obs.copy())
                act_buf[i].append(a)
                lp_buf[i].append(lp)
                val_buf[i].append(critics[i].forward(obs))
            
            obs, rewards, done, _, info = env.step(actions)
            for i in range(n): rew_buf[i].append(rewards[i])
            if done: break
        
        for i in range(n):
            if len(rew_buf[i]) < 2: continue
            adv, ret = compute_gae(rew_buf[i], val_buf[i])
            if np.std(adv) > 1e-8: adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
            ppo_update_actor(actors[i], obs_buf[i], act_buf[i], lp_buf[i], adv, entropy_coef=0.01)
            
        episodes.append({
            "mean_lambda": float(info.get("mean_lambda", 0)),
            "survived": info.get("survived", False),
            "welfare": float(info.get("welfare", 0.0)),
        })
        
    ev = episodes[-20:]
    return {
        "lambda": float(np.mean([d["mean_lambda"] for d in ev])),
        "survival": float(np.mean([float(d["survived"]) for d in ev]) * 100),
        "welfare": float(np.mean([d["welfare"] for d in ev])),
    }


def main():
    print("=" * 70)
    print("  Phase D: Partial Observability Experiment")
    print("=" * 70)
    
    conditions = [
        ("noisy", 0.05, 0), ("noisy", 0.10, 0), ("noisy", 0.20, 0),
        ("delayed", 0.0, 1), ("delayed", 0.0, 2), ("delayed", 0.0, 5),
        ("local", 0.0, 0)
    ]
    
    results = {}
    
    for obs_type, nstd, dly in conditions:
        cond_k = f"{obs_type}_std{nstd}_dly{dly}"
        print(f"\n[{cond_k}]")
        results[cond_k] = {}
        
        for alg in ["situational", "unconditional", "ippo"]:
            seeds_data = []
            n_seeds = N_SEEDS_IPPO if alg == "ippo" else N_SEEDS_RULE
            
            for s in range(n_seeds):
                if alg == "ippo":
                    r = run_ippo(s*7+42, obs_type, nstd, dly)
                else:
                    r = run_rule(s*7+42, alg, obs_type, nstd, dly)
                seeds_data.append(r)
                sys.stdout.write(".")
            sys.stdout.flush()
            
            lams = [r["lambda"] for r in seeds_data]
            survs = [r["survival"] for r in seeds_data]
            wels = [r["welfare"] for r in seeds_data]
            
            results[cond_k][alg] = {
                "lambda_mean": float(np.mean(lams)), "lambda_std": float(np.std(lams)),
                "survival_mean": float(np.mean(survs)), "survival_std": float(np.std(survs)),
                "welfare_mean": float(np.mean(wels)), "welfare_std": float(np.std(wels)),
            }
            
            print(f" {alg:>13s}: Surv={np.mean(survs):5.1f}±{np.std(survs):3.1f}% | W={np.mean(wels):4.1f}±{np.std(wels):3.1f}")
            
    out_path = OUTPUT_DIR / "partial_obs_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

if __name__ == "__main__":
    main()
