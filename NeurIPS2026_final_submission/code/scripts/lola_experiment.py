"""
LOLA (Learning with Opponent-Learning Awareness) for N-agent PGG
=================================================================
Implements LOLA (Foerster et al., 2018) adapted for N-agent setting.

LOLA key idea: Agent i anticipates how its gradient step affects
opponents' parameters, and accounts for how opponents' learning
will change the game.

Standard PG:    罐_i ??罐_i + 慣 쨌 ??{罐_i} V_i(罐)
LOLA:           罐_i ??罐_i + 慣 쨌 ??{罐_i} V_i(罐 + 慣쨌?놴_{-i})

In N-agent PGG, the LOLA update for agent i considers the
anticipated policy updates of ALL other honest agents.

This tests whether opponent-modeling can escape the Nash Trap
(reviewer request for opponent-shaping comparison).

Dependencies: NumPy only.
"""
import numpy as np
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
from envs.nonlinear_pgg_env import NonlinearPGGEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cleanrl_baselines"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Hyperparameters ===
N_AGENTS = 20
BYZ_FRAC = 0.3
N_HONEST = int(N_AGENTS * (1 - BYZ_FRAC))
STATE_DIM = 4
N_EPISODES = 300
N_EVAL = 30
N_SEEDS = 20
T_HORIZON = 50
GAMMA = 0.99
LR_AGENT = 0.01
LOLA_LR = 0.003  # learning rate for the LOLA correction term

if os.environ.get("ETHICAAI_FAST") == "1":
    print("  [FAST MODE] N_SEEDS=2, N_EPISODES=30")
    N_SEEDS = 2
    N_EPISODES = 30


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


class LOLAAgent:
    """Linear policy with LOLA update capability.
    
    Policy: 貫 = sigmoid(w 쨌 obs + b), with Gaussian exploration.
    LOLA corrects the gradient by anticipating opponents' updates.
    """
    def __init__(self, rng):
        self.w = rng.randn(STATE_DIM).astype(np.float32) * 0.01
        self.b = np.float32(0.0)
    
    def forward(self, obs):
        return sigmoid(float(obs @ self.w + self.b))
    
    def act(self, obs, rng, noise_scale=0.1):
        mu = self.forward(obs)
        return float(np.clip(mu + rng.randn() * noise_scale, 0.01, 0.99))
    
    def get_params(self):
        return np.concatenate([self.w, [self.b]])
    
    def set_params(self, vec):
        self.w = vec[:STATE_DIM].copy().astype(np.float32)
        self.b = np.float32(vec[STATE_DIM])
    
    def policy_gradient(self, obs_list, act_list, returns):
        """Compute ?눸?J = E[?눸?log ?(a|s) 쨌 G_t]"""
        grad_w = np.zeros(STATE_DIM, dtype=np.float32)
        grad_b = np.float32(0.0)
        
        for obs, act, G in zip(obs_list, act_list, returns):
            mu = self.forward(obs)
            # ?굃og N(a|關,?)/?궽?쨌 ?궽??궽?
            # = (a-關)/?짼 쨌 關(1-關) 쨌 obs
            sigma = 0.1
            d_logp = (act - mu) / (sigma**2) * mu * (1 - mu)
            grad_w += G * d_logp * obs
            grad_b += G * d_logp
        
        n = max(len(returns), 1)
        return np.concatenate([grad_w / n, [grad_b / n]])


def run_lola(seed):
    """Train LOLA agents on non-linear PGG."""
    rng = np.random.RandomState(42 + seed)
    env = NonlinearPGGEnv(byz_frac=BYZ_FRAC)
    
    agents = [LOLAAgent(np.random.RandomState(seed * 100 + i)) for i in range(N_HONEST)]
    
    ep_data = {"welfare": [], "mean_lam": [], "survival": []}
    
    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        noise = 0.15 - 0.13 * min(ep / N_EPISODES, 1.0)
        
        # Per-agent trajectories
        all_obs = [[] for _ in range(N_HONEST)]
        all_acts = [[] for _ in range(N_HONEST)]
        all_rewards = [[] for _ in range(N_HONEST)]
        
        total_w, lam_sum, steps = 0.0, 0.0, 0
        survived = True
        
        for t in range(T_HORIZON):
            lambdas = np.zeros(N_HONEST)
            for i in range(N_HONEST):
                lam_i = agents[i].act(obs, rng, noise)
                lambdas[i] = lam_i
                all_obs[i].append(obs.copy())
                all_acts[i].append(lam_i)
            
            obs_next, rewards, terminated, truncated, info = env.step(lambdas)
            
            for i in range(N_HONEST):
                all_rewards[i].append(float(rewards[i]))
            
            total_w += float(np.mean(rewards))
            lam_sum += float(lambdas.mean())
            steps += 1
            
            if terminated:
                survived = info.get("survived", False)
                break
            obs = obs_next
        
        # Compute returns for each agent
        all_returns = []
        for i in range(N_HONEST):
            rew = all_rewards[i]
            returns = np.zeros(len(rew))
            G = 0
            for t in reversed(range(len(rew))):
                G = rew[t] + GAMMA * G
                returns[t] = G
            if returns.std() > 1e-8:
                returns = (returns - returns.mean()) / returns.std()
            all_returns.append(returns)
        
        # LOLA update for each agent
        # Step 1: Compute standard policy gradients for all agents
        standard_grads = []
        for i in range(N_HONEST):
            grad = agents[i].policy_gradient(all_obs[i], all_acts[i], all_returns[i])
            standard_grads.append(grad)
        
        # Step 2: LOLA correction ??anticipate opponents' updates
        # For agent i, imagine all j?쟧 take a gradient step, then compute
        # how agent i's value changes in the updated landscape
        for i in range(N_HONEST):
            original_params = agents[i].get_params().copy()
            
            # Imagine opponents making their standard gradient steps
            # (simplified N-agent LOLA: average opponent gradient effect)
            opponent_effect = np.zeros(STATE_DIM + 1, dtype=np.float32)
            
            for j in range(N_HONEST):
                if j == i:
                    continue
                # Opponent j would update: 罐_j ??罐_j + LR 쨌 ??j V_j
                # Effect on agent i's gradient (cross-derivative approximation):
                # ?G_i ??d짼V_i/(d罐_i쨌d罐_j) 쨌 (LR 쨌 ??j V_j)
                # Simplified: finite difference of agent i's gradient w.r.t agent j's params
                old_params_j = agents[j].get_params().copy()
                agents[j].set_params(old_params_j + LOLA_LR * standard_grads[j])
                
                # Recompute agent i's gradient in the world where j has updated
                # (using same trajectory ??this is the 1-step LOLA approximation)
                grad_after = agents[i].policy_gradient(all_obs[i], all_acts[i], all_returns[i])
                opponent_effect += (grad_after - standard_grads[i])
                
                agents[j].set_params(old_params_j)
            
            # LOLA gradient = standard_grad + LOLA_correction
            lola_correction = opponent_effect / max(N_HONEST - 1, 1)
            total_grad = standard_grads[i] + LOLA_LR * lola_correction
            
            agents[i].set_params(original_params + LR_AGENT * total_grad)
        
        ep_data["welfare"].append(total_w / max(steps, 1))
        ep_data["mean_lam"].append(lam_sum / max(steps, 1))
        ep_data["survival"].append(float(survived))
        
        if (ep + 1) % 50 == 0:
            r = slice(-30, None)
            w = np.mean(ep_data["welfare"][r])
            l = np.mean(ep_data["mean_lam"][r])
            s = np.mean(ep_data["survival"][r]) * 100
            print(f"    ep {ep+1}: W={w:.1f}, 貫={l:.3f}, Surv={s:.0f}%")
    
    # Eval
    eval_w = ep_data["welfare"][-N_EVAL:]
    eval_l = ep_data["mean_lam"][-N_EVAL:]
    eval_s = ep_data["survival"][-N_EVAL:]
    
    return {
        "welfare": float(np.mean(eval_w)),
        "lambda": float(np.mean(eval_l)),
        "survival": float(np.mean(eval_s) * 100),
    }


def bootstrap_ci(data, n_boot=1000, ci=0.95):
    arr = np.array(data)
    boots = np.array([np.mean(np.random.choice(arr, len(arr), replace=True)) for _ in range(n_boot)])
    lo = np.percentile(boots, (1-ci)/2*100)
    hi = np.percentile(boots, (1+ci)/2*100)
    return [float(lo), float(hi)]


def main():
    print("=" * 65)
    print("  LOLA (Learning with Opponent-Learning Awareness)")
    print(f"  N={N_AGENTS}, Byz={BYZ_FRAC*100:.0f}%")
    print(f"  Episodes={N_EPISODES}, Seeds={N_SEEDS}")
    print(f"  LOLA_LR={LOLA_LR}, Agent_LR={LR_AGENT}")
    print("=" * 65)
    
    t0 = time.time()
    all_results = []
    
    for s in range(N_SEEDS):
        print(f"\n  Seed {s+1}/{N_SEEDS}")
        r = run_lola(s)
        all_results.append(r)
        print(f"    ??貫={r['lambda']:.3f}, Surv={r['survival']:.0f}%, W={r['welfare']:.1f}")
    
    lams = [r["lambda"] for r in all_results]
    survs = [r["survival"] for r in all_results]
    welfs = [r["welfare"] for r in all_results]
    
    output = {
        "algorithm": "LOLA (N-agent, linear policy)",
        "description": "Learning with Opponent-Learning Awareness "
                       "(Foerster et al., 2018) adapted for N-agent PGG. "
                       "1-step LOLA correction with cross-derivative approximation.",
        "lambda_mean": float(np.mean(lams)),
        "lambda_std": float(np.std(lams)),
        "lambda_ci95": bootstrap_ci(lams),
        "survival_mean": float(np.mean(survs)),
        "survival_std": float(np.std(survs)),
        "survival_ci95": bootstrap_ci(survs),
        "welfare_mean": float(np.mean(welfs)),
        "welfare_std": float(np.std(welfs)),
        "per_seed_lambda": lams,
        "per_seed_survival": survs,
        "per_seed_welfare": welfs,
        "n_seeds": N_SEEDS,
        "n_episodes": N_EPISODES,
    }
    
    out_path = OUTPUT_DIR / "lola_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")
    
    elapsed = time.time() - t0
    print(f"\n{'=' * 65}")
    print(f"  LOLA COMPLETE in {elapsed:.0f}s")
    print(f"  貫={np.mean(lams):.3f}짹{np.std(lams):.3f}")
    print(f"  Survival={np.mean(survs):.0f}짹{np.std(survs):.0f}%")
    print(f"  Welfare={np.mean(welfs):.1f}짹{np.std(welfs):.1f}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
