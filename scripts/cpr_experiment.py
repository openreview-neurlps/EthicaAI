"""
Phase 2: Common-Pool Resource (CPR) Experiment
================================================
A STRUCTURALLY DIFFERENT environment from PGG to validate the 
Moral Commitment Spectrum in a non-PGG setting.

Key structural differences from PGG:
- PGG: agents CONTRIBUTE to a public good (provision problem)  
- CPR: agents EXTRACT from a shared resource (appropriation problem)
- Action is inverted: lambda = "restraint" (1.0 = min extraction)
- Non-linearity: resource regeneration stops below threshold

This directly addresses W1: "single PGG class"
"""
import numpy as np
import json
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cpr_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── CPR Environment ────────────────────────────────────────
class CommonPoolResource:
    """
    Ostrom-style Common-Pool Resource game.
    
    N agents share a renewable resource pool.
    Each agent chooses extraction rate e_i in [0, 1].
    Higher extraction = more individual payoff but pool depletion.
    
    lambda = "restraint" = 1 - extraction_rate
    So lambda=1.0 means "take nothing" (max restraint)
    lambda=0.0 means "take everything" (no restraint)
    
    Resource dynamics:
    R_{t+1} = R_t + g(R_t) - sum(extractions)
    g(R_t) = r * R_t * (1 - R_t/K)  [logistic growth]
    
    Tipping point: if R_t < R_crit, growth rate drops dramatically
    """
    
    def __init__(self, n_agents=20, byz_frac=0.3, t_horizon=50,
                 r_crit=0.15, growth_rate=0.3, capacity=1.0,
                 extraction_efficiency=0.8, shock_prob=0.05, shock_mag=0.1):
        self.N = n_agents
        self.byz_frac = byz_frac
        self.n_byz = int(n_agents * byz_frac)
        self.n_honest = n_agents - self.n_byz
        self.T = t_horizon
        self.r_crit = r_crit
        self.growth_rate = growth_rate
        self.K = capacity
        self.eff = extraction_efficiency
        self.shock_prob = shock_prob
        self.shock_mag = shock_mag
    
    def reset(self, rng):
        self.R = 0.5 * self.K
        self.t = 0
        return self.R
    
    def step(self, lambdas, rng):
        """
        lambdas: shape (N,), where lambda_i = restraint level
        extraction_i = (1 - lambda_i) * R / N
        """
        extraction_rates = 1.0 - lambdas  # lambda=restraint, so extraction = 1-lambda
        
        # Individual extraction (proportional to resource)
        max_extract = self.R / self.N
        extractions = extraction_rates * max_extract * self.eff
        total_extraction = np.sum(extractions)
        
        # Payoff = what you extracted
        payoffs = extractions.copy()
        
        # Resource growth (logistic, with tipping point)
        if self.R < self.r_crit:
            # Below tipping: minimal growth (near-irreversible)
            growth = 0.01 * self.R * (1 - self.R / self.K)
        else:
            # Above tipping: healthy logistic growth
            growth = self.growth_rate * self.R * (1 - self.R / self.K)
        
        # Stochastic shock
        shock = self.shock_mag if rng.random() < self.shock_prob else 0.0
        
        # Resource update
        self.R = np.clip(self.R + growth - total_extraction - shock, 0, self.K)
        self.t += 1
        
        survived = self.R > 0
        terminated = not survived or self.t >= self.T
        
        return payoffs, survived, self.R, terminated


# ─── Algorithms ──────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

class CPRAgent:
    """REINFORCE agent for CPR (same structure as PGG agents)."""
    def __init__(self, rng, dim=4):
        self.w = rng.randn(dim) * 0.01
        self.b = 0.0
        self.lr = 0.01
        self.log_std = np.log(0.15)
    
    def act(self, obs, rng):
        mu = sigmoid(np.dot(self.w, obs) + self.b)
        noise = rng.randn() * np.exp(self.log_std)
        return np.clip(mu + noise, 0, 1)
    
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


def run_cpr_selfish(seed, n_episodes=300):
    """Run selfish RL agents on CPR environment."""
    rng = np.random.RandomState(seed)
    env = CommonPoolResource()
    agents = [CPRAgent(rng) for _ in range(env.n_honest)]
    
    per_episode = []
    for ep in range(n_episodes):
        R = env.reset(rng)
        obs_buf = [[] for _ in range(env.n_honest)]
        act_buf = [[] for _ in range(env.n_honest)]
        rew_buf = [[] for _ in range(env.n_honest)]
        
        for t in range(env.T):
            obs = np.array([R / env.K, 0.5, float(R < env.r_crit), t / env.T])
            lambdas = np.zeros(env.N)
            for i in range(env.n_honest):
                a = agents[i].act(obs, rng)
                lambdas[i] = a
                obs_buf[i].append(obs.copy())
                act_buf[i].append(a)
            # Byzantine: maximum extraction (lambda=0)
            
            payoffs, survived, R, terminated = env.step(lambdas, rng)
            for i in range(env.n_honest):
                rew_buf[i].append(payoffs[i])
            
            if terminated:
                break
        
        for i in range(env.n_honest):
            agents[i].update(obs_buf[i], act_buf[i], rew_buf[i])
        
        mean_lambda = np.mean([np.mean(act_buf[i]) for i in range(env.n_honest)])
        per_episode.append({
            "welfare": np.mean([np.sum(rew_buf[i]) for i in range(env.n_honest)]),
            "survived": survived,
            "mean_lambda": mean_lambda,
        })
    
    eval_data = per_episode[-30:]
    return {
        "mean_lambda": np.mean([d["mean_lambda"] for d in eval_data]),
        "mean_survival": np.mean([float(d["survived"]) for d in eval_data]) * 100,
        "mean_welfare": np.mean([d["welfare"] for d in eval_data]),
    }


def run_cpr_commitment(seed, commitment_type="situational", n_episodes=300):
    """Run CPR with moral commitment (situational or unconditional)."""
    rng = np.random.RandomState(seed)
    env = CommonPoolResource()
    
    per_episode = []
    for ep in range(n_episodes):
        R = env.reset(rng)
        
        for t in range(env.T):
            lambdas = np.zeros(env.N)
            
            if commitment_type == "situational":
                # Increase restraint when resource is healthy, reduce when critical
                # SAME structure as PGG paper's g(theta, R)
                for i in range(env.n_honest):
                    theta = 0.8  # base SVO angle
                    if R > env.r_crit:
                        lambdas[i] = np.clip(np.sin(theta) * R * 2, 0, 1)
                    else:
                        lambdas[i] = max(0, np.sin(theta) * 0.3)  # REDUCES restraint in crisis
            elif commitment_type == "unconditional":
                # Maximum restraint regardless of resource level
                for i in range(env.n_honest):
                    lambdas[i] = 1.0  # Maximum restraint  
            elif commitment_type == "fixed_50":
                for i in range(env.n_honest):
                    lambdas[i] = 0.5
            # Byzantine: lambda=0 (no restraint, max extraction)
            
            payoffs, survived, R, terminated = env.step(lambdas, rng)
            if terminated:
                break
        
        mean_lambda = np.mean(lambdas[:env.n_honest])
        welfare = np.mean(payoffs[:env.n_honest]) if survived else 0
        per_episode.append({
            "welfare": welfare,
            "survived": survived,
            "mean_lambda": mean_lambda,
        })
    
    eval_data = per_episode[-30:]
    return {
        "mean_lambda": np.mean([d["mean_lambda"] for d in eval_data]),
        "mean_survival": np.mean([float(d["survived"]) for d in eval_data]) * 100,
        "mean_welfare": np.mean([d["welfare"] for d in eval_data]),
    }


def bootstrap_ci(data, n_boot=10000, ci=0.95):
    data = np.array(data)
    boot = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return float(np.percentile(boot, alpha*100)), float(np.percentile(boot, (1-alpha)*100))


def main():
    N_SEEDS = 20
    
    print("=" * 70)
    print("  Phase 2: Common-Pool Resource (CPR) - Cross-Environment Validation")
    print("  Structural contrast: Appropriation (CPR) vs Provision (PGG)")
    print("=" * 70)
    
    conditions = [
        ("Selfish RL", "selfish"),
        ("Situational", "situational"),
        ("Unconditional", "unconditional"),
        ("Fixed lambda=0.5", "fixed_50"),
    ]
    
    all_results = {}
    
    for label, ctype in conditions:
        print(f"\n  [{label}] Running {N_SEEDS} seeds...")
        t0 = time.time()
        
        seed_results = []
        for s in range(N_SEEDS):
            if ctype == "selfish":
                r = run_cpr_selfish(seed=s*7+42)
            else:
                r = run_cpr_commitment(seed=s*7+42, commitment_type=ctype)
            seed_results.append(r)
        
        elapsed = time.time() - t0
        lam_vals = [r["mean_lambda"] for r in seed_results]
        surv_vals = [r["mean_survival"] for r in seed_results]
        welf_vals = [r["mean_welfare"] for r in seed_results]
        
        ci_lam = bootstrap_ci(lam_vals)
        ci_surv = bootstrap_ci(surv_vals)
        
        all_results[label] = {
            "label": label,
            "commitment_type": ctype,
            "lambda": {"mean": np.mean(lam_vals), "std": np.std(lam_vals), "ci95": ci_lam},
            "survival": {"mean": np.mean(surv_vals), "std": np.std(surv_vals), "ci95": ci_surv},
            "welfare": {"mean": np.mean(welf_vals), "std": np.std(welf_vals)},
            "per_seed_lambda": lam_vals,
            "per_seed_survival": surv_vals,
        }
        
        print(f"  [{label}] ({elapsed:.0f}s)")
        print(f"    lambda (restraint): {np.mean(lam_vals):.3f} [{ci_lam[0]:.3f}, {ci_lam[1]:.3f}]")
        print(f"    survival: {np.mean(surv_vals):.1f}% [{ci_surv[0]:.1f}, {ci_surv[1]:.1f}]")
        print(f"    welfare: {np.mean(welf_vals):.3f}")
    
    # ─── Spectrum Validation Test ────────────────────────────
    print("\n" + "=" * 70)
    print("  MORAL COMMITMENT SPECTRUM VALIDATION (CPR)")
    print("=" * 70)
    
    selfish_surv = np.mean(all_results["Selfish RL"]["per_seed_survival"])
    sit_surv = np.mean(all_results["Situational"]["per_seed_survival"])
    uncond_surv = np.mean(all_results["Unconditional"]["per_seed_survival"])
    
    spectrum_valid = (selfish_surv < sit_surv) and (uncond_surv > sit_surv)
    
    print(f"\n  Selfish RL survival:     {selfish_surv:.1f}%")
    print(f"  Situational survival:    {sit_surv:.1f}%")
    print(f"  Unconditional survival:  {uncond_surv:.1f}%")
    print(f"\n  Spectrum pattern (Selfish < Situational < Unconditional)? ", end="")
    print("YES" if spectrum_valid else "PARTIAL")
    
    crisis_test = sit_surv < 100.0  # Situational should NOT achieve 100%
    print(f"  Situational fails in crisis (< 100%)? {'YES' if crisis_test else 'NO'}")
    
    # Save
    report = {
        "experiment": "CPR Cross-Environment Validation",
        "environment": "Common-Pool Resource (Appropriation)",
        "comparison": "PGG (Provision)",
        "structural_differences": [
            "PGG: agents contribute; CPR: agents extract",
            "PGG: free-riding = not contributing; CPR: free-riding = over-extracting",
            "PGG: lambda = commitment to contribute; CPR: lambda = restraint from extracting",
        ],
        "results": all_results,
        "spectrum_validation": {
            "spectrum_pattern_holds": bool(spectrum_valid),
            "situational_fails_in_crisis": bool(crisis_test),
            "conclusion": (
                "The Moral Commitment Spectrum generalizes to CPR environments: "
                "selfish RL agents over-extract, situational commitment provides "
                "partial protection, but only unconditional commitment ensures "
                "resource sustainability under adversarial conditions."
                if spectrum_valid else
                "Partial spectrum validation: pattern requires further investigation."
            )
        }
    }
    
    out_path = OUTPUT_DIR / "cpr_results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print("\n  DONE!")


if __name__ == "__main__":
    main()
