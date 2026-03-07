import os
import json
import numpy as np
from envs.constrained_pgg_env import ConstrainedPGGEnv
from pathlib import Path

# Fix Seeds
np.random.seed(42)

# --- Hyperparameters ---
N_AGENTS = 10
BYZ_FRAC = 0.3
N_EPISODES = 500
N_SEEDS = 20
DELTA_CONSTRAINT = 0.05  # Maximum allowed probability of collapse (cost)

LR_ACTOR = 0.05
LR_LAGRANGIAN = 0.1
NOISE = 0.3  # Large noise to escape the "death zone" local minima

OUTPUTS_DIR = Path("outputs") / "cpo_experiments"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

def run_es_cpo(seed=0, n_episodes=N_EPISODES, delta=DELTA_CONSTRAINT):
    np.random.seed(seed)
    
    env = ConstrainedPGGEnv(n_agents=N_AGENTS, byz_frac=BYZ_FRAC)
    
    # Initialize phi_1 at random but relatively high to start safely
    phi_1 = 0.9
    lambda_dual = 0.0
    
    history_phi = []
    history_return = []
    history_cost = []
    history_lambda = []
    
    BATCH_SIZE = 5
    
    def evaluate_phi(test_phi):
        ret = 0
        cost = 0
        for _ in range(BATCH_SIZE):
            env.reset()
            done = False
            while not done:
                actions = np.ones(env.n_normal) * test_phi
                _, r, done, i = env.step(actions)
                ret += np.mean(r)
                cost = max(cost, i['cost'])
        return ret / BATCH_SIZE, cost / BATCH_SIZE

    for ep in range(0, n_episodes, BATCH_SIZE):
        # 1. Evaluate current stats for tracking and lambda update
        mean_return, mean_cost = evaluate_phi(phi_1)
        
        # 2. ES Gradient computation
        phi_plus = np.clip(phi_1 + NOISE, 0.0, 1.0)
        phi_minus = np.clip(phi_1 - NOISE, 0.0, 1.0)
        
        ret_plus, cost_plus = evaluate_phi(phi_plus)
        ret_minus, cost_minus = evaluate_phi(phi_minus)
        
        # Effective noise applied (might be asymmetric due to clipping)
        delta_plus = phi_plus - phi_1
        delta_minus = phi_1 - phi_minus
        
        if delta_plus + delta_minus > 0.01:
            grad_ret = (ret_plus - ret_minus) / (delta_plus + delta_minus)
            grad_cost = (cost_plus - cost_minus) / (delta_plus + delta_minus)
        else:
            grad_ret = 0
            grad_cost = 0
            
        # 3. Lagrangian Gradient step
        # Ascend on: Return - lambda * Cost
        grad_L_phi = grad_ret - lambda_dual * grad_cost
        
        # Update phi_1
        phi_1 = np.clip(phi_1 + LR_ACTOR * grad_L_phi, 0.0, 1.0)
        
        # 4. Update Lagrangian multiplier
        lambda_dual = max(0.0, lambda_dual + LR_LAGRANGIAN * (mean_cost - delta))
        
        history_phi.append(phi_1)
        history_return.append(mean_return)
        history_cost.append(mean_cost)
        history_lambda.append(lambda_dual)
        
    return {
        "seed": seed,
        "phi_trajectory": history_phi,
        "return_trajectory": history_return,
        "cost_trajectory": history_cost,
        "lambda_trajectory": history_lambda,
        "final_phi": history_phi[-1]
    }

if __name__ == "__main__":
    print(f"Starting CPO Lagrangian (ES) training for {N_SEEDS} seeds...")
    results = []
    
    # Overrides for fast testing if needed
    if os.environ.get("ETHICAAI_FAST") == "1":
        print("  [FAST MODE] Overriding N_SEEDS=2, N_EPISODES=20")
        N_SEEDS = 2
        N_EPISODES = 20

    for s in range(N_SEEDS):
        res = run_es_cpo(seed=s, n_episodes=N_EPISODES)
        results.append(res)
        print(f"Seed {s+1:02d}/{N_SEEDS} | Final φ₁: {res['final_phi']:.3f} | Final Cost: {res['cost_trajectory'][-1]:.3f}")
        
    avg_final_phi = np.mean([r['final_phi'] for r in results])
    print(f"\nTraining Complete. Average Final φ₁: {avg_final_phi:.3f}")
    
    # Save results
    output_path = OUTPUTS_DIR / "cpo_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
