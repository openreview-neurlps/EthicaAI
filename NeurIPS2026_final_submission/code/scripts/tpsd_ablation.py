"""
TPSD ON/OFF Ablation — The definitive negative control.

Uses the SAME PGG environment with identical REINFORCE agents,
but toggles the tipping-point dynamics:
  - Linear PGG (no tipping point → f(R) = constant): SHOULD NOT trap
  - Nonlinear PGG (with tipping point → f(R) = piecewise): SHOULD trap

This directly isolates TPSD structure as the cause of the Nash Trap.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


class PGGEnv:
    """Public Goods Game with configurable tipping-point dynamics."""
    
    def __init__(self, N=20, E=20.0, M=1.6, T=50, byz_frac=0.3,
                 tipping_point=True, R_crit=0.15, R_recov=0.25,
                 p_shock=0.05, delta_shock=0.15):
        self.N = N
        self.E = E
        self.M = M
        self.T = T
        self.N_byz = int(N * byz_frac)
        self.N_honest = N - self.N_byz
        self.tipping_point = tipping_point
        self.R_crit = R_crit
        self.R_recov = R_recov
        self.p_shock = p_shock
        self.delta_shock = delta_shock
        self.obs_dim = 4  # R_t, mean_c, lambda_prev, crisis_flag
    
    def f(self, R):
        """Recovery function: linear vs nonlinear."""
        if not self.tipping_point:
            return 0.10  # Constant: no tipping point
        # Nonlinear with tipping point
        if R < self.R_crit:
            return 0.01  # Near-irreversible
        elif R < self.R_recov:
            return 0.03  # Hysteresis
        else:
            return 0.10  # Normal
    
    def reset(self):
        self.R = 0.5
        self.t = 0
        self.prev_mean_c = 0.5
        self.prev_lambdas = np.full(self.N_honest, 0.5)
        return self._get_obs()
    
    def _get_obs(self):
        """Per-agent observation."""
        crisis = 1.0 if self.R < self.R_crit else 0.0
        obs = []
        for i in range(self.N_honest):
            obs.append(np.array([
                self.R,
                self.prev_mean_c / self.E,
                self.prev_lambdas[i],
                crisis
            ]))
        return obs
    
    def step(self, lambdas):
        """
        lambdas: cooperation levels for honest agents [0, 1]
        Returns: obs, rewards, done, info
        """
        self.t += 1
        
        # Contributions
        honest_c = lambdas * self.E
        byz_c = np.zeros(self.N_byz)  # Byzantine always defect
        all_c = np.concatenate([honest_c, byz_c])
        mean_c = np.mean(all_c)
        
        # Individual payoffs for honest agents
        rewards = np.array([
            (self.E - honest_c[i]) + self.M * np.sum(all_c) / self.N
            for i in range(self.N_honest)
        ])
        
        # Resource dynamics
        shock = self.delta_shock if np.random.random() < self.p_shock else 0.0
        dR = self.f(self.R) * (mean_c / self.E - 0.4) - shock
        self.R = np.clip(self.R + dR, 0.0, 1.0)
        
        # Check for collapse
        done = self.t >= self.T
        collapsed = False
        if self.R <= 0:
            done = True
            collapsed = True
            rewards = np.zeros(self.N_honest)
        
        # Update state
        self.prev_mean_c = mean_c
        self.prev_lambdas = lambdas.copy()
        
        info = {
            'R': self.R,
            'mean_c': mean_c,
            'collapsed': collapsed,
            'mean_lambda': float(np.mean(lambdas))
        }
        
        return self._get_obs(), rewards, done, info


class LinearAgent:
    """REINFORCE agent with linear policy."""
    
    def __init__(self, obs_dim=4, lr=0.01, noise_std=0.1):
        self.W = np.zeros(obs_dim)
        self.b = 0.0
        self.lr = lr
        self.noise_std = noise_std
        self.trajectory = []  # (obs, lambda, reward)
    
    def act(self, obs):
        z = np.dot(self.W, obs) + self.b
        lam = sigmoid(z)
        noise = np.random.normal(0, self.noise_std)
        lam_noisy = np.clip(lam + noise, 0, 1)
        self.trajectory.append((obs, lam, lam_noisy))
        return lam_noisy
    
    def store_reward(self, r):
        obs, lam, lam_noisy = self.trajectory[-1]
        self.trajectory[-1] = (obs, lam, lam_noisy, r)
    
    def update(self, gamma=0.99):
        if len(self.trajectory) == 0:
            return
        
        # Compute returns
        rewards = [t[3] for t in self.trajectory if len(t) == 4]
        if len(rewards) == 0:
            self.trajectory = []
            return
        
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = np.array(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Update
        valid = [t for t in self.trajectory if len(t) == 4]
        for (obs, lam, lam_noisy, _), G in zip(valid, returns):
            grad_log_pi = (lam_noisy - lam) / (self.noise_std ** 2)
            sig_grad = lam * (1 - lam)
            self.W += self.lr * G * grad_log_pi * sig_grad * obs
            self.b += self.lr * G * grad_log_pi * sig_grad
        
        self.trajectory = []


def run_experiment(tipping_point, n_seeds=20, n_episodes=300, n_eval=30, label=""):
    """Run REINFORCE agents in PGG with or without tipping point."""
    results = {'label': label, 'tipping_point': tipping_point, 'per_seed': []}
    
    for seed in range(n_seeds):
        np.random.seed(seed * 42 + 7)
        
        env = PGGEnv(N=20, E=20.0, M=1.6, T=50, byz_frac=0.3,
                     tipping_point=tipping_point)
        agents = [LinearAgent(obs_dim=4) for _ in range(env.N_honest)]
        
        ep_data = {'lambdas': [], 'survivals': [], 'welfare': []}
        
        for ep in range(n_episodes):
            obs = env.reset()
            ep_lambdas = []
            ep_rewards = np.zeros(env.N_honest)
            
            for t in range(env.T):
                lambdas = np.array([agents[i].act(obs[i]) for i in range(env.N_honest)])
                ep_lambdas.append(float(np.mean(lambdas)))
                
                obs, rewards, done, info = env.step(lambdas)
                ep_rewards += rewards
                
                for i in range(env.N_honest):
                    agents[i].store_reward(rewards[i])
                
                if done:
                    break
            
            for agent in agents:
                agent.update()
            
            ep_data['lambdas'].append(float(np.mean(ep_lambdas)))
            ep_data['survivals'].append(0 if info.get('collapsed') else 1)
            ep_data['welfare'].append(float(np.mean(ep_rewards)))
        
        # Eval: last n_eval episodes
        eval_lam = np.mean(ep_data['lambdas'][-n_eval:])
        eval_surv = np.mean(ep_data['survivals'][-n_eval:])
        eval_w = np.mean(ep_data['welfare'][-n_eval:])
        
        seed_result = {
            'seed': seed,
            'mean_lambda': float(eval_lam),
            'survival_rate': float(eval_surv),
            'welfare': float(eval_w),
            'trapped': eval_lam < 0.85
        }
        results['per_seed'].append(seed_result)
        
        if seed % 5 == 0:
            print(f"  [{label}] Seed {seed}: λ={eval_lam:.3f}, "
                  f"surv={eval_surv:.1%}, W={eval_w:.1f}, "
                  f"trapped={seed_result['trapped']}")
    
    # Aggregate
    all_lam = [s['mean_lambda'] for s in results['per_seed']]
    all_surv = [s['survival_rate'] for s in results['per_seed']]
    all_w = [s['welfare'] for s in results['per_seed']]
    
    results['aggregate'] = {
        'mean_lambda': float(np.mean(all_lam)),
        'std_lambda': float(np.std(all_lam)),
        'mean_survival': float(np.mean(all_surv)),
        'mean_welfare': float(np.mean(all_w)),
        'trap_rate': float(np.mean([s['trapped'] for s in results['per_seed']])),
        'n_seeds': n_seeds
    }
    
    return results


def main():
    output_dir = Path(__file__).parent.parent / 'outputs' / 'tpsd_ablation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("TPSD ON/OFF Ablation — Definitive Negative Control")
    print("=" * 60)
    
    # Experiment 1: Linear PGG (NO tipping point)
    print("\n[1/2] Linear PGG (constant f=0.10) — expecting higher survival")
    results_linear = run_experiment(
        tipping_point=False,
        n_seeds=20,
        n_episodes=300,
        label="PGG_Linear"
    )
    
    # Experiment 2: Nonlinear PGG (WITH tipping point)
    print("\n[2/2] Nonlinear PGG (tipping point) — expecting Nash Trap")
    results_nonlinear = run_experiment(
        tipping_point=True,
        n_seeds=20,
        n_episodes=300,
        label="PGG_TPSD"
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    for name, res in [("Linear PGG (no TPSD)", results_linear),
                       ("Nonlinear PGG (TPSD)", results_nonlinear)]:
        agg = res['aggregate']
        print(f"\n{name}:")
        print(f"  λ̄ = {agg['mean_lambda']:.3f} ± {agg['std_lambda']:.3f}")
        print(f"  Survival = {agg['mean_survival']:.1%}")
        print(f"  Welfare = {agg['mean_welfare']:.1f}")
        print(f"  Trap Rate = {agg['trap_rate']:.1%}")
    
    # Key comparison
    linear_surv = results_linear['aggregate']['mean_survival']
    nonlinear_surv = results_nonlinear['aggregate']['mean_survival']
    diff = linear_surv - nonlinear_surv
    
    print(f"\n{'=' * 30}")
    print(f"Survival gap: {diff:.1%} (Linear - Nonlinear)")
    print(f"TPSD structure {'DOES' if diff > 0.2 else 'may not'} cause differential failure")
    
    # Save
    combined = {
        'experiment': 'tpsd_onoff_ablation',
        'purpose': 'Isolate TPSD structure as cause of Nash Trap',
        'linear_pgg': results_linear,
        'nonlinear_pgg': results_nonlinear,
        'conclusion': {
            'survival_gap': float(diff),
            'linear_survival': float(linear_surv),
            'nonlinear_survival': float(nonlinear_surv),
            'tpsd_causes_differential_failure': bool(diff > 0.1)
        }
    }
    
    output_file = output_dir / 'tpsd_ablation_results.json'
    with open(output_file, 'w') as f:
        json.dump(combined, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
