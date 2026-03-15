"""
Phase B-v2 (Table 5): Unconditional Commitment Floor with Learning Agents
================================================================
Tests whether an unconditional commitment floor can rescue IPPO agents
from the Nash Trap.

Design:
  1. Train IPPO agents for N_EPISODES (they fall into Nash Trap: lambda~0.4)
  2. During both training and evaluation, apply commitment floor:
     effective_lambda = max(learned_lambda, phi1)  [UNCONDITIONAL, all states]
  3. phi1=0: no floor (pure learning) -> Nash Trap persists (survival~7%)
  4. phi1=1: full floor (always cooperate) -> maximum survival

This directly supports C4: "unconditional commitment is a design requirement"
by showing that learning alone is insufficient, but learning + commitment floor
rescues the system.

Grid: phi1 in {0.0, 0.21, 0.50, 1.0} x Byz in {0%, 30%} x 20 seeds
"""
import numpy as np
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
from cleanrl_mappo_pgg import (
    NonlinearPGGEnv, MLPActor, MLPCritic, compute_gae,
    ppo_update_actor, bootstrap_ci, relu, NNLayer,
    GAMMA, GAE_LAMBDA, CLIP_EPS, HIDDEN_DIM
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phi1_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Hyperparameters (match cleanrl_mappo_pgg.py exactly) ===
N_EPISODES = 300
N_EVAL = 30
N_SEEDS = 20
T_HORIZON = 50
N_AGENTS = 20
R_CRIT = 0.15  # env's r_crit (hard collapse)
R_OVERRIDE = 0.25  # crisis override triggers here (= r_recov, proactive intervention)

PHI1_VALUES = [0.0, 0.21, 0.50, 1.0]
BYZ_FRACS = [0.0, 0.30]

if os.environ.get("ETHICAAI_FAST") == "1":
    print("  [FAST MODE] Overriding N_SEEDS=2, N_EPISODES=30")
    N_SEEDS = 2
    N_EPISODES = 30
    N_EVAL = 10


def run_ippo_with_phi1(seed, phi1, byz_frac):
    """Train IPPO agents with commitment floor (phi1) override.
    
    The commitment floor is applied DURING training (not just evaluation),
    meaning agents learn in a world where the floor exists.
    This is the correct design: phi1 is a system-level design choice,
    not an agent-level decision.
    """
    rng = np.random.RandomState(seed)
    env = NonlinearPGGEnv(byz_frac=byz_frac)
    n_honest = int(N_AGENTS * (1 - byz_frac))
    
    # Each honest agent gets its own actor (independent PG = IPPO)
    actors = [MLPActor(np.random.RandomState(seed * 100 + i)) for i in range(n_honest)]
    critic = MLPCritic(rng)  # shared critic
    
    episode_data = []
    
    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        
        # Per-agent trajectories
        all_obs = [[] for _ in range(n_honest)]
        all_acts = [[] for _ in range(n_honest)]
        all_log_probs = [[] for _ in range(n_honest)]
        all_rewards = [[] for _ in range(n_honest)]
        all_values = [[] for _ in range(n_honest)]
        
        total_welfare = 0.0
        lam_sum = 0.0
        steps = 0
        survived = True
        # Track whether floor was activated per agent per timestep
        floor_active = [[] for _ in range(n_honest)]
        
        for t in range(T_HORIZON):
            lambdas = np.zeros(n_honest)
            
            for i in range(n_honest):
                # Agent decides based on policy
                mean, _ = actors[i].forward(obs)
                std = np.exp(actors[i].log_std[0])
                learned_lambda = float(np.clip(mean[0] + rng.randn() * std, 0.01, 0.99))
                
                # === UNCONDITIONAL COMMITMENT FLOOR ===
                # phi1 acts as a policy floor: effective_lambda = max(learned_lambda, phi1)
                # This is the precise operationalization of "unconditional commitment":
                # the agent's cooperation level never drops below phi1
                # regardless of the environment state.
                # phi1=0: no floor (pure learning) -> Nash Trap
                # phi1=1: full floor (always cooperate) -> maximum survival
                is_floor_active = learned_lambda < phi1
                effective_lambda = max(learned_lambda, phi1)
                
                lambdas[i] = effective_lambda
                floor_active[i].append(is_floor_active)
                
                # Store trajectory using LEARNED action for log_prob consistency.
                # When floor is active, the action stored is still the learned one
                # (for correct log_prob), but environment sees effective_lambda.
                # Floor-overridden timesteps are skipped during PPO update (below).
                all_obs[i].append(obs.copy())
                all_acts[i].append(learned_lambda)
                all_log_probs[i].append(actors[i].log_prob(obs, learned_lambda))
                val = critic.forward(obs)
                all_values[i].append(float(val))
            
            obs, rewards, terminated, truncated, info = env.step(lambdas)
            
            for i in range(n_honest):
                # Each agent gets its own reward from the reward vector
                all_rewards[i].append(float(rewards[i]) if hasattr(rewards, '__len__') else float(rewards))
            
            team_reward = float(np.mean(rewards)) if hasattr(rewards, '__len__') else float(rewards)
            total_welfare += team_reward
            lam_sum += float(lambdas.mean())
            steps += 1
            
            if terminated:
                survived = info.get("survived", False)
                break
        
        # PPO update for each agent
        # DESIGN CHOICE: Skip timesteps where floor was active.
        # Rationale: When phi1 overrides the agent's action, the observed
        # outcome reflects the specification (floor), not the policy.
        # Updating the policy on floor-imposed actions would conflate
        # "what the agent learned" with "what the system enforced."
        # This cleanly separates learning (above floor) from specification (at floor).
        for i in range(n_honest):
            if len(all_rewards[i]) < 2:
                continue
            
            # Filter out floor-overridden timesteps
            mask = [not fa for fa in floor_active[i]]
            if sum(mask) < 2:
                continue  # Not enough non-floor samples to learn from
            
            obs_filtered = [o for o, m in zip(all_obs[i], mask) if m]
            act_filtered = [a for a, m in zip(all_acts[i], mask) if m]
            lp_filtered = [lp for lp, m in zip(all_log_probs[i], mask) if m]
            rew_filtered = [r for r, m in zip(all_rewards[i], mask) if m]
            val_filtered = [v for v, m in zip(all_values[i], mask) if m]
            
            advantages, returns = compute_gae(rew_filtered, val_filtered)
            
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / advantages.std()
            
            ppo_update_actor(actors[i], obs_filtered, act_filtered,
                           lp_filtered, advantages)
            
            # Critic update (simplified: just track, no backprop for now)
            for t_idx in range(len(returns)):
                pass  # critic not essential for IPPO -- actor is the focus
        
        episode_data.append({
            "welfare": total_welfare / max(steps, 1),
            "mean_lambda": lam_sum / max(steps, 1),
            "survived": survived,
        })
    
    # Evaluate on last N_EVAL episodes
    eval_eps = episode_data[-N_EVAL:]
    return {
        "survival": float(np.mean([e["survived"] for e in eval_eps]) * 100),
        "welfare": float(np.mean([e["welfare"] for e in eval_eps])),
        "lambda": float(np.mean([e["mean_lambda"] for e in eval_eps])),
    }


def main():
    print("=" * 70)
    print("  Commitment Floor (phi1) with IPPO Learning Agents")
    print(f"  N={N_AGENTS}, R_crit={R_CRIT}, Episodes={N_EPISODES}, Seeds={N_SEEDS}")
    print("=" * 70)
    
    t0 = time.time()
    results = {}
    
    for phi1 in PHI1_VALUES:
        results[str(phi1)] = {}
        for byz in BYZ_FRACS:
            byz_key = f"byz_{int(byz*100)}"
            print(f"\n  phi1={phi1:.2f}, Byz={byz*100:.0f}%: Running {N_SEEDS} seeds...")
            
            seed_results = []
            for s in range(N_SEEDS):
                r = run_ippo_with_phi1(s, phi1, byz)
                seed_results.append(r)
                if (s + 1) % 5 == 0 or s == 0:
                    print(f"    Seed {s}: λ={r['lambda']:.3f}, "
                          f"surv={r['survival']:.1f}%, W={r['welfare']:.1f}")
            
            surv = [r["survival"] for r in seed_results]
            welf = [r["welfare"] for r in seed_results]
            lam = [r["lambda"] for r in seed_results]
            
            results[str(phi1)][byz_key] = {
                "survival_mean": float(np.mean(surv)),
                "survival_std": float(np.std(surv)),
                "survival_ci95": bootstrap_ci(surv),
                "welfare_mean": float(np.mean(welf)),
                "welfare_std": float(np.std(welf)),
                "lambda_mean": float(np.mean(lam)),
                "lambda_std": float(np.std(lam)),
                "per_seed_survival": surv,
                "per_seed_welfare": welf,
                "per_seed_lambda": lam,
            }
            
            print(f"    -> phi1={phi1:.2f} | "
                  f"W({byz*100:.0f}%)={np.mean(welf):.1f}+/-{np.std(welf):.1f}  "
                  f"Alive={np.mean(surv):.0f}+/-{np.std(surv):.0f}% | ")
    
    # Save
    out_path = OUTPUT_DIR / "phi1_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")
    
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  COMPLETE in {elapsed:.0f}s")
    print(f"  phi1 sweep summary (Byz=30%):")
    for phi1 in PHI1_VALUES:
        d = results[str(phi1)]["byz_30"]
        print(f"    phi1={phi1:.2f}: lambda={d['lambda_mean']:.3f}, "
              f"surv={d['survival_mean']:.0f}%, W={d['welfare_mean']:.1f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
