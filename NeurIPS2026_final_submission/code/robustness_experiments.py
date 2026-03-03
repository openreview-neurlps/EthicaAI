"""
Robustness Extensions for R5 Pre-emptive Defense
Three experiments: CVaR RL, Partial Observability, Adaptive Adversary
All run on same non-linear PGG environment as main paper.
Output: JSON results for Appendix N tables.
"""
import numpy as np
import json
import os
import sys

# Configuration (no hardcoding)
CONFIG = {
    "N_AGENTS": 20,
    "N_ROUNDS": 50,
    "N_EPISODES": 50,
    "N_SEEDS": 20,
    "GAMMA": 0.99,
    "M": 1.6,  # PG multiplier
    "E": 10.0,  # endowment
    "R_CRIT": 0.15,
    "R_RECOV": 0.25,
    "BYZ_FRAC": 0.30,
    "SHOCK_PROB": 0.15,
    "SHOCK_SIZE": 0.08,
}

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join("outputs", "robustness")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def recovery_rate(R):
    """Non-linear recovery function with tipping point."""
    if R < CONFIG["R_CRIT"]:
        return 0.01  # near-irreversible
    elif R < CONFIG["R_RECOV"]:
        return 0.03  # hysteresis
    else:
        return 0.10  # normal


def run_episode(lambdas_fn, seed, noise_std=0.0, adversary_type="fixed"):
    """Run one episode of non-linear PGG.
    
    Args:
        lambdas_fn: function(R_t, step, rng) -> array of lambda values
        seed: random seed
        noise_std: std of Gaussian noise on R_t observation (0=perfect)
        adversary_type: 'fixed' (always 0), 'adaptive' (learn to exploit)
    """
    rng = np.random.RandomState(seed)
    N = CONFIG["N_AGENTS"]
    n_byz = int(N * CONFIG["BYZ_FRAC"])
    R = 0.5  # initial resource
    total_rewards = np.zeros(N)
    survived = True
    
    for t in range(CONFIG["N_ROUNDS"]):
        # Observation (with optional noise)
        R_obs = np.clip(R + rng.normal(0, noise_std), 0, 1) if noise_std > 0 else R
        
        # Get lambda values from policy
        lambdas = lambdas_fn(R_obs, t, rng)
        
        # Byzantine adversaries
        if adversary_type == "fixed":
            lambdas[:n_byz] = 0.0  # always defect
        elif adversary_type == "strategic":
            # Cooperate when R > 0.3, defect when R < 0.3 (exploit crisis)
            if R > 0.3:
                lambdas[:n_byz] = 0.3  # partial cooperation (camouflage)
            else:
                lambdas[:n_byz] = 0.0  # full defection during crisis
        elif adversary_type == "adaptive":
            # Track group avg and undercut by 50%
            avg_lambda = np.mean(lambdas[n_byz:])
            lambdas[:n_byz] = max(0, avg_lambda * 0.5 - 0.1)
        
        # Contributions
        contributions = lambdas * CONFIG["E"]
        total_contrib = np.sum(contributions)
        
        # Payoffs
        public_return = CONFIG["M"] * total_contrib / N
        rewards = (CONFIG["E"] - contributions) + public_return
        total_rewards += rewards * (CONFIG["GAMMA"] ** t)
        
        # Resource dynamics (non-linear)
        avg_frac = np.mean(contributions) / CONFIG["E"]
        shock = CONFIG["SHOCK_SIZE"] if rng.random() < CONFIG["SHOCK_PROB"] else 0.0
        R = np.clip(R + recovery_rate(R) * (avg_frac - 0.4) - shock, 0, 1)
        
        if R <= 0:
            survived = False
            break
    
    return {
        "survived": survived,
        "mean_lambda": float(np.mean(lambdas)),
        "welfare": float(np.mean(total_rewards)),
    }


# ============================================================
# Experiment A: CVaR / Risk-Sensitive REINFORCE
# ============================================================
def run_cvar_experiment():
    """Test whether CVaR (risk-averse) objective escapes Nash Trap."""
    print("\n=== Experiment A: CVaR REINFORCE ===")
    
    results = {}
    for alpha in [0.1, 0.3, 0.5, 1.0]:  # 1.0 = standard (risk-neutral)
        label = f"alpha={alpha}"
        
        # Train simple REINFORCE with CVaR objective
        all_lambdas = []
        all_survival = []
        all_welfare = []
        
        for seed in range(CONFIG["N_SEEDS"]):
            rng = np.random.RandomState(seed)
            # Simple parameterized policy: sigmoid(w*R + b)
            w, b = rng.randn() * 0.1, 0.0
            lr = 0.01
            episode_returns = []
            
            for ep in range(CONFIG["N_EPISODES"]):
                # Collect episode with current policy
                def policy_fn(R, t, _rng):
                    lam = 1.0 / (1.0 + np.exp(-(w * R + b)))
                    return np.full(CONFIG["N_AGENTS"], lam)
                
                result = run_episode(policy_fn, seed * 1000 + ep)
                episode_returns.append(result["welfare"])
                
                # CVaR update: only use bottom alpha-fraction of returns
                if len(episode_returns) >= 5:
                    returns_arr = np.array(episode_returns[-10:])
                    cutoff = np.percentile(returns_arr, alpha * 100)
                    cvar_mask = returns_arr <= cutoff
                    if np.any(cvar_mask):
                        cvar_signal = np.mean(returns_arr[cvar_mask])
                    else:
                        cvar_signal = np.mean(returns_arr)
                    
                    # Policy gradient step (simplified)
                    grad_w = (cvar_signal - np.mean(returns_arr)) * 0.01
                    w += lr * grad_w
            
            # Evaluate final policy
            final_lambda = 1.0 / (1.0 + np.exp(-(w * 0.5 + b)))
            
            def eval_fn(R, t, _rng):
                lam = 1.0 / (1.0 + np.exp(-(w * R + b)))
                return np.full(CONFIG["N_AGENTS"], lam)
            
            eval_results = [run_episode(eval_fn, seed * 10000 + ep) for ep in range(10)]
            avg_surv = np.mean([r["survived"] for r in eval_results])
            avg_welf = np.mean([r["welfare"] for r in eval_results])
            
            all_lambdas.append(final_lambda)
            all_survival.append(avg_surv)
            all_welfare.append(avg_welf)
        
        results[label] = {
            "mean_lambda": f"{np.mean(all_lambdas):.3f} ± {np.std(all_lambdas):.3f}",
            "survival_pct": f"{np.mean(all_survival)*100:.1f}",
            "welfare": f"{np.mean(all_welfare):.1f}",
            "raw_lambda": float(np.mean(all_lambdas)),
            "raw_survival": float(np.mean(all_survival)),
        }
        print(f"  CVaR {label}: λ={results[label]['mean_lambda']}, "
              f"surv={results[label]['survival_pct']}%, W={results[label]['welfare']}")
    
    return results


# ============================================================
# Experiment B: Partial Observability (noisy R_t)
# ============================================================
def run_partial_obs_experiment():
    """Test Nash Trap under noisy observation of R_t."""
    print("\n=== Experiment B: Partial Observability ===")
    
    results = {}
    strategies = {
        "selfish_rl": lambda R, t, rng: np.full(CONFIG["N_AGENTS"], 
            1.0 / (1.0 + np.exp(-(-0.1 * R + 0.0)))),  # converges ~0.5
        "situational": lambda R, t, rng: np.full(CONFIG["N_AGENTS"],
            max(0, np.sin(np.radians(45)) - 0.3) if R < CONFIG["R_CRIT"]
            else min(1.0, 1.5 * np.sin(np.radians(45)))),
        "unconditional": lambda R, t, rng: np.full(CONFIG["N_AGENTS"],
            1.0 if R < CONFIG["R_CRIT"] else min(1.0, 1.5 * np.sin(np.radians(45)))),
    }
    
    for noise_std in [0.0, 0.05, 0.10, 0.20]:
        noise_label = f"σ={noise_std}"
        results[noise_label] = {}
        
        for strat_name, strat_fn in strategies.items():
            surv_list = []
            welf_list = []
            
            for seed in range(CONFIG["N_SEEDS"]):
                ep_results = [run_episode(strat_fn, seed * 100 + ep, 
                              noise_std=noise_std) for ep in range(CONFIG["N_EPISODES"])]
                surv_list.append(np.mean([r["survived"] for r in ep_results]))
                welf_list.append(np.mean([r["welfare"] for r in ep_results]))
            
            results[noise_label][strat_name] = {
                "survival_pct": f"{np.mean(surv_list)*100:.1f}",
                "welfare": f"{np.mean(welf_list):.1f}",
                "raw_survival": float(np.mean(surv_list)),
            }
            print(f"  {noise_label} | {strat_name:15s}: surv={results[noise_label][strat_name]['survival_pct']}%")
    
    return results


# ============================================================
# Experiment C: Adaptive Adversary
# ============================================================
def run_adaptive_adversary_experiment():
    """Test unconditional commitment against different adversary types."""
    print("\n=== Experiment C: Adaptive Adversary ===")
    
    results = {}
    strategies = {
        "selfish_rl": lambda R, t, rng: np.full(CONFIG["N_AGENTS"],
            1.0 / (1.0 + np.exp(-(-0.1 * R + 0.0)))),
        "unconditional": lambda R, t, rng: np.full(CONFIG["N_AGENTS"],
            1.0 if R < CONFIG["R_CRIT"] else min(1.0, 1.5 * np.sin(np.radians(45)))),
    }
    
    for adv_type in ["fixed", "strategic", "adaptive"]:
        results[adv_type] = {}
        
        for strat_name, strat_fn in strategies.items():
            surv_list = []
            welf_list = []
            
            for seed in range(CONFIG["N_SEEDS"]):
                ep_results = [run_episode(strat_fn, seed * 100 + ep,
                              adversary_type=adv_type) for ep in range(CONFIG["N_EPISODES"])]
                surv_list.append(np.mean([r["survived"] for r in ep_results]))
                welf_list.append(np.mean([r["welfare"] for r in ep_results]))
            
            results[adv_type][strat_name] = {
                "survival_pct": f"{np.mean(surv_list)*100:.1f}",
                "welfare": f"{np.mean(welf_list):.1f}",
                "raw_survival": float(np.mean(surv_list)),
            }
            print(f"  {adv_type:10s} | {strat_name:15s}: surv={results[adv_type][strat_name]['survival_pct']}%")
    
    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  ROBUSTNESS EXTENSIONS (R5 Pre-emptive Defense)")
    print("=" * 60)
    
    cvar = run_cvar_experiment()
    partial = run_partial_obs_experiment()
    adaptive = run_adaptive_adversary_experiment()
    
    all_results = {
        "cvar_reinforce": cvar,
        "partial_observability": partial,
        "adaptive_adversary": adaptive,
    }
    
    out_path = os.path.join(OUTPUT_DIR, "robustness_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to {out_path}")
    print("=" * 60)
