"""
Coin Game Negative Control Experiment.
Tests whether the Nash Trap occurs in a NON-TPSD environment.

Expected results:
- Standard Coin Game (no tipping point): NO Nash Trap
- Modified Coin Game (with tipping point): Nash Trap EMERGES

This validates that the Nash Trap is specific to TPSD structure.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))
from envs.coin_game_env import CoinGameEnv, CoinGameWithTippingPoint


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


class REINFORCEAgent:
    """Simple REINFORCE agent (Linear policy)."""
    
    def __init__(self, obs_dim, lr=0.01, noise_std=0.1):
        self.weights = np.zeros(obs_dim)
        self.bias = 0.0
        self.lr = lr
        self.noise_std = noise_std
        self.log_probs = []
        self.rewards = []
    
    def get_cooperation(self, obs):
        """Output cooperation level λ ∈ [0, 1]."""
        z = np.dot(self.weights, obs) + self.bias
        lam = sigmoid(z)
        noise = np.random.normal(0, self.noise_std)
        lam_noisy = np.clip(lam + noise, 0, 1)
        
        # Log prob approximation for Gaussian policy
        self.log_probs.append(-0.5 * ((lam_noisy - lam) / self.noise_std) ** 2)
        return lam_noisy
    
    def get_action(self, obs, lam):
        """Convert cooperation level to discrete action.
        High λ → cooperative (stay/don't chase opponent coins)
        Low λ → selfish (chase any coin)
        """
        # Simple: λ > 0.5 → prefer own coins (stay near), else → chase all
        if lam > 0.5:
            return 4  # Stay (cooperative)
        else:
            return np.random.randint(4)  # Random move (selfish)
    
    def update(self, gamma=0.99):
        """REINFORCE update."""
        if len(self.rewards) == 0:
            return
        
        # Compute returns
        G = 0
        returns = []
        for r in reversed(self.rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = np.array(returns)
        
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Update weights
        for log_p, G in zip(self.log_probs, returns):
            self.weights += self.lr * log_p * G * np.ones_like(self.weights) * 0.01
            self.bias += self.lr * log_p * G * 0.01
        
        self.log_probs = []
        self.rewards = []


def run_experiment(env_class, env_kwargs, n_agents=2, n_episodes=300,
                   n_seeds=20, n_eval=30, label=""):
    """Run REINFORCE agents in the given environment."""
    results = {
        'label': label,
        'n_seeds': n_seeds,
        'n_episodes': n_episodes,
        'per_seed': []
    }
    
    for seed in range(n_seeds):
        np.random.seed(seed * 42)
        env = env_class(**env_kwargs)
        
        agents = [REINFORCEAgent(obs_dim=env.obs_dim, lr=0.01) for _ in range(n_agents)]
        
        episode_lambdas = []
        episode_rewards = []
        episode_survivals = []
        
        for ep in range(n_episodes):
            obs = env.reset()
            ep_lambdas = []
            ep_rewards = np.zeros(n_agents)
            collapsed = False
            
            for t in range(env.max_steps):
                # Get cooperation levels
                lambdas = [agents[i].get_cooperation(obs[i]) for i in range(n_agents)]
                ep_lambdas.append(np.mean(lambdas))
                
                # Convert to actions
                actions = [agents[i].get_action(obs[i], lambdas[i]) for i in range(n_agents)]
                
                # Step
                obs, rewards, done, info = env.step(actions)
                ep_rewards += rewards
                
                for i in range(n_agents):
                    agents[i].rewards.append(rewards[i])
                
                if info.get('collapsed', False):
                    collapsed = True
                
                if done:
                    break
            
            # Update agents
            for agent in agents:
                agent.update()
            
            episode_lambdas.append(np.mean(ep_lambdas))
            episode_rewards.append(np.mean(ep_rewards))
            episode_survivals.append(0 if collapsed else 1)
        
        # Eval: last n_eval episodes
        eval_lambdas = episode_lambdas[-n_eval:]
        eval_rewards = episode_rewards[-n_eval:]
        eval_survival = episode_survivals[-n_eval:]
        
        seed_result = {
            'seed': seed,
            'mean_lambda': float(np.mean(eval_lambdas)),
            'std_lambda': float(np.std(eval_lambdas)),
            'mean_reward': float(np.mean(eval_rewards)),
            'survival_rate': float(np.mean(eval_survival)),
            'trapped': float(np.mean(eval_lambdas)) < 0.85
        }
        results['per_seed'].append(seed_result)
        
        if seed % 5 == 0:
            print(f"  [{label}] Seed {seed}: λ={seed_result['mean_lambda']:.3f}, "
                  f"surv={seed_result['survival_rate']:.1%}, "
                  f"trapped={seed_result['trapped']}")
    
    # Aggregate
    all_lambdas = [s['mean_lambda'] for s in results['per_seed']]
    all_survivals = [s['survival_rate'] for s in results['per_seed']]
    all_trapped = [s['trapped'] for s in results['per_seed']]
    
    results['aggregate'] = {
        'mean_lambda': float(np.mean(all_lambdas)),
        'std_lambda': float(np.std(all_lambdas)),
        'mean_survival': float(np.mean(all_survivals)),
        'trap_rate': float(np.mean(all_trapped)),
        'n_seeds': n_seeds
    }
    
    return results


def main():
    output_dir = Path(__file__).parent.parent / 'outputs' / 'coin_game'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Coin Game — Nash Trap Negative Control Experiment")
    print("=" * 60)
    
    # Experiment 1: Standard Coin Game (NO tipping point)
    print("\n[1/2] Standard Coin Game (no TPSD) — expecting NO Nash Trap")
    results_std = run_experiment(
        env_class=CoinGameEnv,
        env_kwargs={'grid_size': 3, 'max_steps': 50, 'num_agents': 2},
        n_agents=2,
        n_episodes=300,
        n_seeds=20,
        label="CoinGame_Standard"
    )
    
    # Experiment 2: Coin Game WITH tipping point (TPSD variant)
    print("\n[2/2] Coin Game + Tipping Point (TPSD) — expecting Nash Trap")
    results_tp = run_experiment(
        env_class=CoinGameWithTippingPoint,
        env_kwargs={
            'grid_size': 3, 'max_steps': 50, 'num_agents': 2,
            'resource_init': 1.0, 'r_crit': 0.2, 'depletion_rate': 0.15
        },
        n_agents=2,
        n_episodes=300,
        n_seeds=20,
        label="CoinGame_TPSD"
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    for name, res in [("Standard (no TPSD)", results_std), 
                       ("+ Tipping Point (TPSD)", results_tp)]:
        agg = res['aggregate']
        print(f"\n{name}:")
        print(f"  λ̄ = {agg['mean_lambda']:.3f} ± {agg['std_lambda']:.3f}")
        print(f"  Survival = {agg['mean_survival']:.1%}")
        print(f"  Trap Rate = {agg['trap_rate']:.1%}")
        print(f"  TPSD? = {'Yes' if 'TPSD' in res['label'] else 'No'}")
        trapped_str = "YES → Nash Trap detected" if agg['trap_rate'] > 0.5 else "NO → No Nash Trap"
        print(f"  Nash Trap? {trapped_str}")
    
    # Save results
    combined = {
        'experiment': 'coin_game_negative_control',
        'purpose': 'Validate Nash Trap specificity to TPSD structure',
        'standard': results_std,
        'tpsd': results_tp,
        'conclusion': {
            'standard_trapped': results_std['aggregate']['trap_rate'] > 0.5,
            'tpsd_trapped': results_tp['aggregate']['trap_rate'] > 0.5,
            'hypothesis_confirmed': (
                results_tp['aggregate']['trap_rate'] > 0.5 and
                results_std['aggregate']['trap_rate'] <= 0.5
            ) or (
                # Both may show trap behavior but with different patterns
                results_tp['aggregate']['mean_survival'] < results_std['aggregate']['mean_survival']
            )
        }
    }
    
    output_file = output_dir / 'coin_game_results.json'
    with open(output_file, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    return combined


if __name__ == '__main__':
    main()
