"""
Phase 0: Statistical Analysis for NeurIPS Reviewer Response
============================================================
Reruns core experiments collecting per-seed data, then computes:
  - Bootstrap 95% CIs (10,000 resamples)
  - Mann-Whitney U tests (pairwise algorithm comparisons)
  - Effect sizes (Cohen's d)

Outputs: outputs/statistical_tests/full_statistical_report.json
"""
import numpy as np
import json
import os
import sys
import time
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "statistical_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Environment params (match paper exactly)
N_AGENTS = 20
M = 1.6
E = 10.0
T_HORIZON = 50
R_CRIT = 0.15
R_RECOV = 0.25
SHOCK_PROB = 0.05
SHOCK_MAG = 0.15
BYZ_FRAC = 0.3
N_BYZ = int(N_AGENTS * BYZ_FRAC)
N_HONEST = N_AGENTS - N_BYZ
STATE_DIM = 4

N_EPISODES = 300
N_SEEDS = 20          # More seeds for statistical power (up from 5-10)
N_EVAL_EPISODES = 30  # Last 30 episodes for evaluation
N_BOOTSTRAP = 10000   # Bootstrap resamples

# ─── Environment ─────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

def run_episode(lambdas, rng):
    """Run single episode with given lambda vector, return per-step data."""
    R = 0.5
    total_rewards = np.zeros(N_AGENTS)
    
    for t in range(T_HORIZON):
        contribs = lambdas * E
        pool = np.sum(contribs)
        payoffs = (E - contribs) + M * pool / N_AGENTS
        total_rewards += payoffs
        
        mean_c = np.mean(contribs) / E
        # Tipping point dynamics
        if R < R_CRIT:
            f_R = 0.01
        elif R < R_RECOV:
            f_R = 0.03
        else:
            f_R = 0.10
        
        shock = SHOCK_MAG if rng.random() < SHOCK_PROB else 0.0
        R = np.clip(R + f_R * (mean_c - 0.4) - shock, 0, 1)
        
        if R <= 0:
            break
    
    survived = R > 0
    welfare = np.mean(total_rewards)
    return welfare, survived, np.mean(lambdas)

# ─── Algorithms ──────────────────────────────────────────────
class LinearPolicy:
    """Linear Gaussian policy for REINFORCE."""
    def __init__(self, rng, dim=STATE_DIM):
        self.w = rng.randn(dim) * 0.01
        self.b = 0.0
        self.lr = 0.01
        self.log_std = np.log(0.15)
    
    def act(self, obs, rng):
        mu = sigmoid(np.dot(self.w, obs) + self.b)
        noise = rng.randn() * np.exp(self.log_std)
        return np.clip(mu + noise, 0, 1), mu
    
    def update(self, obs_list, act_list, rewards):
        baseline = np.mean(rewards)
        for obs, act, rew in zip(obs_list, act_list, rewards):
            mu = sigmoid(np.dot(self.w, obs) + self.b)
            grad_mu = mu * (1 - mu)
            diff = act - mu
            std2 = np.exp(2 * self.log_std)
            grad_log = (diff / std2) * grad_mu * obs
            self.w += self.lr * (rew - baseline) * grad_log
            self.b += self.lr * (rew - baseline) * (diff / std2) * grad_mu

class MLPPolicy:
    """2-layer MLP for REINFORCE."""
    def __init__(self, rng, dim=STATE_DIM, hidden=32):
        self.w1 = rng.randn(dim, hidden) * 0.1
        self.b1 = np.zeros(hidden)
        self.w2 = rng.randn(hidden) * 0.1
        self.b2 = 0.0
        self.lr = 0.003
        self.log_std = np.log(0.15)
    
    def forward(self, obs):
        h = np.tanh(obs @ self.w1 + self.b1)
        return sigmoid(h @ self.w2 + self.b2), h
    
    def act(self, obs, rng):
        mu, _ = self.forward(obs)
        noise = rng.randn() * np.exp(self.log_std)
        return np.clip(mu + noise, 0, 1), mu
    
    def update(self, obs_list, act_list, rewards):
        baseline = np.mean(rewards)
        for obs, act, rew in zip(obs_list, act_list, rewards):
            mu, h = self.forward(obs)
            grad_mu = mu * (1 - mu)
            diff = act - mu
            std2 = np.exp(2 * self.log_std)
            d_out = (rew - baseline) * (diff / std2) * grad_mu
            self.w2 += self.lr * d_out * h
            self.b2 += self.lr * d_out
            dh = d_out * self.w2 * (1 - h**2)
            self.w1 += self.lr * np.outer(obs, dh)
            self.b1 += self.lr * dh

class PPOPolicy:
    """PPO with clipped surrogate for IPPO/MAPPO style."""
    def __init__(self, rng, dim=STATE_DIM, hidden=32):
        self.w1 = rng.randn(dim, hidden) * 0.1
        self.b1 = np.zeros(hidden)
        self.w2 = rng.randn(hidden) * 0.1
        self.b2 = 0.0
        self.v_w = rng.randn(dim) * 0.01
        self.v_b = 0.0
        self.lr = 0.005
        self.clip_eps = 0.2
        self.log_std = np.log(0.15)

    def forward(self, obs):
        h = np.tanh(obs @ self.w1 + self.b1)
        return sigmoid(h @ self.w2 + self.b2), h

    def value(self, obs):
        return np.dot(self.v_w, obs) + self.v_b

    def act(self, obs, rng):
        mu, _ = self.forward(obs)
        noise = rng.randn() * np.exp(self.log_std)
        return np.clip(mu + noise, 0, 1), mu

    def log_prob(self, obs, a):
        mu, _ = self.forward(obs)
        std = np.exp(self.log_std)
        return -0.5 * ((a - mu) / std) ** 2 - self.log_std

    def ppo_update(self, obs_list, act_list, old_lps, advs):
        for obs, act, old_lp, adv in zip(obs_list, act_list, old_lps, advs):
            new_lp = self.log_prob(obs, act)
            ratio = np.exp(new_lp - old_lp)
            clip_ratio = np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            loss = -min(ratio * adv, clip_ratio * adv)

            mu, h = self.forward(obs)
            grad_mu = mu * (1 - mu)
            std2 = np.exp(2 * self.log_std)
            d_out = -loss * ((act - mu) / std2) * grad_mu
            self.w2 += self.lr * d_out * h
            self.b2 += self.lr * d_out

    def update_value(self, obs_list, targets):
        for obs, target in zip(obs_list, targets):
            pred = self.value(obs)
            err = target - pred
            self.v_w += 0.01 * err * obs
            self.v_b += 0.01 * err


def run_reinforce_experiment(policy_class, seed, n_episodes=N_EPISODES):
    """Run REINFORCE with per-seed data collection."""
    rng = np.random.RandomState(seed)
    policies = [policy_class(rng) for _ in range(N_HONEST)]
    
    per_episode_data = []
    
    for ep in range(n_episodes):
        R = 0.5
        obs_buffer = [[] for _ in range(N_HONEST)]
        act_buffer = [[] for _ in range(N_HONEST)]
        reward_buffer = [[] for _ in range(N_HONEST)]
        
        for t in range(T_HORIZON):
            obs = np.array([R, 0.5, 0.5, float(R < R_CRIT)])
            lambdas = np.zeros(N_AGENTS)
            
            for i in range(N_HONEST):
                a, _ = policies[i].act(obs, rng)
                lambdas[i] = a
                obs_buffer[i].append(obs.copy())
                act_buffer[i].append(a)
            # Byzantine agents: 0 contribution
            
            contribs = lambdas * E
            pool = np.sum(contribs)
            payoffs = (E - contribs) + M * pool / N_AGENTS
            
            for i in range(N_HONEST):
                reward_buffer[i].append(payoffs[i])
            
            mean_c = np.mean(contribs) / E
            if R < R_CRIT:
                f_R = 0.01
            elif R < R_RECOV:
                f_R = 0.03
            else:
                f_R = 0.10
            shock = SHOCK_MAG if rng.random() < SHOCK_PROB else 0.0
            R = np.clip(R + f_R * (mean_c - 0.4) - shock, 0, 1)
            if R <= 0:
                break
        
        # Update policies
        for i in range(N_HONEST):
            policies[i].update(obs_buffer[i], act_buffer[i], reward_buffer[i])
        
        survived = R > 0
        welfare = np.mean([np.sum(reward_buffer[i]) for i in range(N_HONEST)])
        mean_lambda = np.mean([np.mean(act_buffer[i]) for i in range(N_HONEST)])
        
        per_episode_data.append({
            "welfare": welfare,
            "survived": survived,
            "mean_lambda": mean_lambda
        })
    
    # Evaluation: last N_EVAL_EPISODES
    eval_data = per_episode_data[-N_EVAL_EPISODES:]
    return {
        "eval_welfare": [d["welfare"] for d in eval_data],
        "eval_survival": [float(d["survived"]) for d in eval_data],
        "eval_lambda": [d["mean_lambda"] for d in eval_data],
        "mean_welfare": np.mean([d["welfare"] for d in eval_data]),
        "mean_survival": np.mean([float(d["survived"]) for d in eval_data]) * 100,
        "mean_lambda": np.mean([d["mean_lambda"] for d in eval_data]),
    }


def run_ppo_experiment(seed, shared_critic=False, n_episodes=N_EPISODES):
    """Run PPO (IPPO or MAPPO style) with per-seed data."""
    rng = np.random.RandomState(seed)
    policies = [PPOPolicy(rng) for _ in range(N_HONEST)]
    
    per_episode_data = []
    
    for ep in range(n_episodes):
        R = 0.5
        obs_buf = [[] for _ in range(N_HONEST)]
        act_buf = [[] for _ in range(N_HONEST)]
        lp_buf = [[] for _ in range(N_HONEST)]
        rew_buf = [[] for _ in range(N_HONEST)]
        val_buf = [[] for _ in range(N_HONEST)]
        
        for t in range(T_HORIZON):
            obs = np.array([R, 0.5, 0.5, float(R < R_CRIT)])
            lambdas = np.zeros(N_AGENTS)
            
            for i in range(N_HONEST):
                a, _ = policies[i].act(obs, rng)
                lp = policies[i].log_prob(obs, a)
                v = policies[i].value(obs)
                lambdas[i] = a
                obs_buf[i].append(obs.copy())
                act_buf[i].append(a)
                lp_buf[i].append(lp)
                val_buf[i].append(v)
            
            contribs = lambdas * E
            pool = np.sum(contribs)
            payoffs = (E - contribs) + M * pool / N_AGENTS
            
            for i in range(N_HONEST):
                rew_buf[i].append(payoffs[i])
            
            mean_c = np.mean(contribs) / E
            if R < R_CRIT:
                f_R = 0.01
            elif R < R_RECOV:
                f_R = 0.03
            else:
                f_R = 0.10
            shock = SHOCK_MAG if rng.random() < SHOCK_PROB else 0.0
            R = np.clip(R + f_R * (mean_c - 0.4) - shock, 0, 1)
            if R <= 0:
                break
        
        # PPO update
        for i in range(N_HONEST):
            vals = np.array(val_buf[i])
            rews = np.array(rew_buf[i])
            targets = rews.copy()
            advs = targets - vals
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            policies[i].ppo_update(obs_buf[i], act_buf[i], lp_buf[i], advs.tolist())
            policies[i].update_value(obs_buf[i], targets.tolist())
        
        survived = R > 0
        welfare = np.mean([np.sum(rew_buf[i]) for i in range(N_HONEST)])
        mean_lambda = np.mean([np.mean(act_buf[i]) for i in range(N_HONEST)])
        per_episode_data.append({"welfare": welfare, "survived": survived, "mean_lambda": mean_lambda})
    
    eval_data = per_episode_data[-N_EVAL_EPISODES:]
    return {
        "eval_welfare": [d["welfare"] for d in eval_data],
        "eval_survival": [float(d["survived"]) for d in eval_data],
        "eval_lambda": [d["mean_lambda"] for d in eval_data],
        "mean_welfare": np.mean([d["welfare"] for d in eval_data]),
        "mean_survival": np.mean([float(d["survived"]) for d in eval_data]) * 100,
        "mean_lambda": np.mean([d["mean_lambda"] for d in eval_data]),
    }


# ─── Statistical Tests ──────────────────────────────────────
def bootstrap_ci(data, n_resamples=N_BOOTSTRAP, ci=0.95):
    """Compute bootstrap confidence interval."""
    data = np.array(data)
    boot_means = np.array([
        np.mean(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_resamples)
    ])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    return float(lo), float(hi), float(np.mean(data)), float(np.std(data))

def mann_whitney_u(x, y):
    """Mann-Whitney U test (two-sided)."""
    from itertools import product as iterproduct
    x, y = np.array(x), np.array(y)
    nx, ny = len(x), len(y)
    
    # Compute U statistic
    U = 0
    for xi in x:
        for yi in y:
            if xi > yi:
                U += 1
            elif xi == yi:
                U += 0.5
    
    # Normal approximation for large samples
    mu_U = nx * ny / 2
    sigma_U = np.sqrt(nx * ny * (nx + ny + 1) / 12)
    z = (U - mu_U) / (sigma_U + 1e-10)
    
    # Two-sided p-value (normal approximation)
    from math import erfc
    p_value = erfc(abs(z) / np.sqrt(2))
    return float(U), float(z), float(p_value)

def cohens_d(x, y):
    """Cohen's d effect size."""
    x, y = np.array(x), np.array(y)
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
    return float((np.mean(x) - np.mean(y)) / (pooled_std + 1e-10))


# ─── Main Execution ─────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Phase 0: Statistical Analysis (20 seeds, 300 episodes)")
    print("=" * 70)
    
    all_results = {}
    algorithms = {
        "reinforce_linear": ("Ind. REINFORCE Linear", LinearPolicy),
        "reinforce_mlp": ("Ind. REINFORCE MLP", MLPPolicy),
        "ippo": ("IPPO-style", None),
        "mappo": ("MAPPO-style", None),
    }
    
    # Run REINFORCE experiments
    for algo_key in ["reinforce_linear", "reinforce_mlp"]:
        label, policy_cls = algorithms[algo_key]
        print(f"\n  [{label}] Running {N_SEEDS} seeds...")
        t0 = time.time()
        
        seed_results = []
        for s in range(N_SEEDS):
            result = run_reinforce_experiment(policy_cls, seed=s*7 + 42)
            seed_results.append(result)
            print(f"    Seed {s}: λ={result['mean_lambda']:.3f}, "
                  f"surv={result['mean_survival']:.1f}%, "
                  f"W={result['mean_welfare']:.1f}")
        
        elapsed = time.time() - t0
        # Collect per-seed means for statistical tests
        per_seed_lambda = [r["mean_lambda"] for r in seed_results]
        per_seed_survival = [r["mean_survival"] for r in seed_results]
        per_seed_welfare = [r["mean_welfare"] for r in seed_results]
        
        # Bootstrap CIs
        lambda_ci = bootstrap_ci(per_seed_lambda)
        surv_ci = bootstrap_ci(per_seed_survival)
        welfare_ci = bootstrap_ci(per_seed_welfare)
        
        all_results[algo_key] = {
            "label": label,
            "per_seed_lambda": per_seed_lambda,
            "per_seed_survival": per_seed_survival,
            "per_seed_welfare": per_seed_welfare,
            "lambda": {"mean": lambda_ci[2], "std": lambda_ci[3],
                      "ci95_lo": lambda_ci[0], "ci95_hi": lambda_ci[1]},
            "survival": {"mean": surv_ci[2], "std": surv_ci[3],
                        "ci95_lo": surv_ci[0], "ci95_hi": surv_ci[1]},
            "welfare": {"mean": welfare_ci[2], "std": welfare_ci[3],
                       "ci95_lo": welfare_ci[0], "ci95_hi": welfare_ci[1]},
            "time_seconds": elapsed,
        }
        print(f"  [{label}] Done ({elapsed:.0f}s)")
        print(f"    λ: {lambda_ci[2]:.3f} [{lambda_ci[0]:.3f}, {lambda_ci[1]:.3f}]")
        print(f"    Surv: {surv_ci[2]:.1f}% [{surv_ci[0]:.1f}, {surv_ci[1]:.1f}]")
    
    # Run PPO experiments (IPPO / MAPPO-style)
    for algo_key, shared in [("ippo", False), ("mappo", True)]:
        label = algorithms[algo_key][0]
        print(f"\n  [{label}] Running {N_SEEDS} seeds...")
        t0 = time.time()
        
        seed_results = []
        for s in range(N_SEEDS):
            result = run_ppo_experiment(seed=s*7 + 42, shared_critic=shared)
            seed_results.append(result)
            print(f"    Seed {s}: λ={result['mean_lambda']:.3f}, "
                  f"surv={result['mean_survival']:.1f}%, "
                  f"W={result['mean_welfare']:.1f}")
        
        elapsed = time.time() - t0
        per_seed_lambda = [r["mean_lambda"] for r in seed_results]
        per_seed_survival = [r["mean_survival"] for r in seed_results]
        per_seed_welfare = [r["mean_welfare"] for r in seed_results]
        
        lambda_ci = bootstrap_ci(per_seed_lambda)
        surv_ci = bootstrap_ci(per_seed_survival)
        welfare_ci = bootstrap_ci(per_seed_welfare)
        
        all_results[algo_key] = {
            "label": label,
            "per_seed_lambda": per_seed_lambda,
            "per_seed_survival": per_seed_survival,
            "per_seed_welfare": per_seed_welfare,
            "lambda": {"mean": lambda_ci[2], "std": lambda_ci[3],
                      "ci95_lo": lambda_ci[0], "ci95_hi": lambda_ci[1]},
            "survival": {"mean": surv_ci[2], "std": surv_ci[3],
                        "ci95_lo": surv_ci[0], "ci95_hi": surv_ci[1]},
            "welfare": {"mean": welfare_ci[2], "std": welfare_ci[3],
                       "ci95_lo": welfare_ci[0], "ci95_hi": welfare_ci[1]},
            "time_seconds": elapsed,
        }
        print(f"  [{label}] Done ({elapsed:.0f}s)")
        print(f"    λ: {lambda_ci[2]:.3f} [{lambda_ci[0]:.3f}, {lambda_ci[1]:.3f}]")
        print(f"    Surv: {surv_ci[2]:.1f}% [{surv_ci[0]:.1f}, {surv_ci[1]:.1f}]")
    
    # ─── Pairwise Statistical Tests ──────────────────────────
    print("\n" + "=" * 70)
    print("  PAIRWISE STATISTICAL TESTS")
    print("=" * 70)
    
    comparisons = [
        ("reinforce_linear", "reinforce_mlp", "Linear vs MLP"),
        ("ippo", "mappo", "IPPO vs MAPPO"),
        ("reinforce_linear", "ippo", "REINFORCE Linear vs IPPO"),
    ]
    
    test_results = []
    for a_key, b_key, label in comparisons:
        a_surv = all_results[a_key]["per_seed_survival"]
        b_surv = all_results[b_key]["per_seed_survival"]
        a_lam = all_results[a_key]["per_seed_lambda"]
        b_lam = all_results[b_key]["per_seed_lambda"]
        
        U_s, z_s, p_s = mann_whitney_u(a_surv, b_surv)
        U_l, z_l, p_l = mann_whitney_u(a_lam, b_lam)
        d_s = cohens_d(a_surv, b_surv)
        d_l = cohens_d(a_lam, b_lam)
        
        test_result = {
            "comparison": label,
            "survival": {"U": U_s, "z": z_s, "p_value": p_s, "cohens_d": d_s,
                        "significant_005": p_s < 0.05, "significant_001": p_s < 0.01},
            "lambda": {"U": U_l, "z": z_l, "p_value": p_l, "cohens_d": d_l,
                      "significant_005": p_l < 0.05, "significant_001": p_l < 0.01},
        }
        test_results.append(test_result)
        
        sig_s = "***" if p_s < 0.001 else "**" if p_s < 0.01 else "*" if p_s < 0.05 else "ns"
        sig_l = "***" if p_l < 0.001 else "**" if p_l < 0.01 else "*" if p_l < 0.05 else "ns"
        
        print(f"\n  {label}:")
        print(f"    Survival: U={U_s:.0f}, z={z_s:.2f}, p={p_s:.4f} {sig_s}, d={d_s:.2f}")
        print(f"    Lambda:   U={U_l:.0f}, z={z_l:.2f}, p={p_l:.4f} {sig_l}, d={d_l:.2f}")
    
    # ─── All algorithms Nash Trap test ───────────────────────
    # Test: are ALL algorithms significantly below oracle (λ=1.0)?
    print("\n" + "-" * 50)
    print("  ALL ALGORITHMS vs ORACLE (λ=1.0, 100% survival)")
    print("-" * 50)
    
    oracle_lambdas = [1.0] * N_SEEDS  # Oracle is deterministic
    oracle_survival = [100.0] * N_SEEDS
    
    oracle_tests = []
    for algo_key in ["reinforce_linear", "reinforce_mlp", "ippo", "mappo"]:
        label = all_results[algo_key]["label"]
        algo_lam = all_results[algo_key]["per_seed_lambda"]
        algo_surv = all_results[algo_key]["per_seed_survival"]
        
        _, z_l, p_l = mann_whitney_u(oracle_lambdas, algo_lam)
        _, z_s, p_s = mann_whitney_u(oracle_survival, algo_surv)
        
        oracle_tests.append({
            "algorithm": label,
            "vs_oracle_lambda_p": p_l,
            "vs_oracle_survival_p": p_s,
            "lambda_gap": 1.0 - np.mean(algo_lam),
            "survival_gap": 100.0 - np.mean(algo_surv),
        })
        
        sig = "***" if p_l < 0.001 else "**" if p_l < 0.01 else "*" if p_l < 0.05 else "ns"
        print(f"  {label}: λ gap = {1.0 - np.mean(algo_lam):.3f}, p={p_l:.6f} {sig}")
    
    # ─── Save Full Report ────────────────────────────────────
    report = {
        "experiment": "Phase 0 Statistical Analysis",
        "config": {
            "N_agents": N_AGENTS, "N_seeds": N_SEEDS, "N_episodes": N_EPISODES,
            "N_eval": N_EVAL_EPISODES, "N_bootstrap": N_BOOTSTRAP,
            "Byz_frac": BYZ_FRAC, "M": M, "R_crit": R_CRIT,
        },
        "algorithm_results": {k: {kk: vv for kk, vv in v.items() 
                                    if kk != "per_seed_lambda" and kk != "per_seed_survival" and kk != "per_seed_welfare"}
                              for k, v in all_results.items()},
        "per_seed_data": {k: {"lambda": v["per_seed_lambda"], 
                              "survival": v["per_seed_survival"],
                              "welfare": v["per_seed_welfare"]}
                         for k, v in all_results.items()},
        "pairwise_tests": test_results,
        "oracle_tests": oracle_tests,
    }
    
    out_path = OUTPUT_DIR / "full_statistical_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  [Saved] {out_path}")
    
    # ─── Summary for LaTeX ───────────────────────────────────
    print("\n" + "=" * 70)
    print("  LATEX-READY SUMMARY")
    print("=" * 70)
    for algo_key in ["reinforce_linear", "reinforce_mlp", "ippo", "mappo"]:
        r = all_results[algo_key]
        l = r["lambda"]
        s = r["survival"]
        w = r["welfare"]
        print(f"  {r['label']:30s} | "
              f"λ={l['mean']:.3f} [{l['ci95_lo']:.3f},{l['ci95_hi']:.3f}] | "
              f"surv={s['mean']:.1f}% [{s['ci95_lo']:.1f},{s['ci95_hi']:.1f}] | "
              f"W={w['mean']:.1f}")
    
    print("\n  DONE!")
    return report


if __name__ == "__main__":
    main()
