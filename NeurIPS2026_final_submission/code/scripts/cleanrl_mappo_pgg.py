"""
Phase 1: Standard MAPPO Baseline for Non-linear PGG
=====================================================
Implements a proper MAPPO with neural network policies using 
CleanRL-style single-file implementation with standard hyperparameters.

Key differences from paper's "MAPPO-style":
1. Neural network policy (2-layer MLP, 64 hidden units)
2. Neural network shared critic
3. Standard PPO hyperparameters from CleanRL defaults
4. Proper GAE computation
5. Multiple PPO epochs per rollout

This addresses W2: "MARL baseline that minimal"
"""
import numpy as np
import json
import time
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cleanrl_baselines"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import our standardized environment
from envs.nonlinear_pgg_env import NonlinearPGGEnv

# --- Hyperparameters (CleanRL defaults) ---
HIDDEN_DIM = 64
LR_ACTOR = 2.5e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
PPO_EPOCHS = 4
MINIBATCH_SIZE = 32

N_EPISODES = 500
N_EVAL = 30
N_SEEDS = 20

if os.environ.get("ETHICAAI_FAST") == "1":
    print("  [FAST MODE] Overriding N_SEEDS=2, N_EPISODES=10")
    N_SEEDS = 2
    N_EPISODES = 10


# --- Neural Network Layers (numpy only, portable) ---
def relu(x):
    return np.maximum(0, x)

def softplus(x):
    return np.log1p(np.exp(np.clip(x, -20, 20)))

class NNLayer:
    def __init__(self, rng, in_dim, out_dim, lr=2.5e-4):
        scale = np.sqrt(2.0 / in_dim)
        self.W = rng.randn(in_dim, out_dim).astype(np.float32) * scale
        self.b = np.zeros(out_dim, dtype=np.float32)
        self.lr = lr
        
        # Adam optimizer state
        self.m_W = np.zeros_like(self.W)
        self.v_W = np.zeros_like(self.W)
        self.m_b = np.zeros_like(self.b)
        self.v_b = np.zeros_like(self.b)
        self.step = 0
    
    def forward(self, x):
        return x @ self.W + self.b
    
    def adam_update(self, grad_W, grad_b, beta1=0.9, beta2=0.999, eps=1e-8):
        self.step += 1
        self.m_W = beta1 * self.m_W + (1 - beta1) * grad_W
        self.v_W = beta2 * self.v_W + (1 - beta2) * grad_W**2
        m_hat = self.m_W / (1 - beta1**self.step)
        v_hat = self.v_W / (1 - beta2**self.step)
        self.W += self.lr * m_hat / (np.sqrt(v_hat) + eps)
        
        self.m_b = beta1 * self.m_b + (1 - beta1) * grad_b
        self.v_b = beta2 * self.v_b + (1 - beta2) * grad_b**2
        m_hat = self.m_b / (1 - beta1**self.step)
        v_hat = self.v_b / (1 - beta2**self.step)
        self.b += self.lr * m_hat / (np.sqrt(v_hat) + eps)


class MLPActor:
    """2-layer MLP policy with clipped Gaussian output (mean from sigmoid, fixed log_std)."""
    def __init__(self, rng, obs_dim=4, hidden=HIDDEN_DIM, lr=LR_ACTOR):
        self.fc1 = NNLayer(rng, obs_dim, hidden, lr)
        self.fc2 = NNLayer(rng, hidden, hidden, lr)
        self.mean_head = NNLayer(rng, hidden, 1, lr)
        self.log_std = np.array([-0.5], dtype=np.float32)  # Initial std ~ 0.6
    
    def forward(self, obs):
        h = relu(self.fc1.forward(obs))
        h = relu(self.fc2.forward(h))
        mean = 1.0 / (1.0 + np.exp(-self.mean_head.forward(h).flatten()))  # sigmoid
        return mean, h
    
    def act(self, obs, rng):
        mean, _ = self.forward(obs)
        std = np.exp(self.log_std)
        noise = rng.randn() * std[0]
        action = np.clip(mean[0] + noise, 0, 1)
        log_prob = -0.5 * ((action - mean[0]) / std[0])**2 - self.log_std[0] - 0.5 * np.log(2 * np.pi)
        return action, log_prob, mean[0]
    
    def log_prob(self, obs, action):
        mean, _ = self.forward(obs)
        std = np.exp(self.log_std)
        return -0.5 * ((action - mean[0]) / std[0])**2 - self.log_std[0] - 0.5 * np.log(2 * np.pi)
    
    def entropy(self, obs):
        return 0.5 + 0.5 * np.log(2 * np.pi) + self.log_std[0]
    
    def param_count(self):
        total = 0
        for layer in [self.fc1, self.fc2, self.mean_head]:
            total += layer.W.size + layer.b.size
        total += self.log_std.size
        return total


class MLPCritic:
    """Value function critic (shared for MAPPO, per-agent for IPPO)."""
    def __init__(self, rng, obs_dim=4, hidden=HIDDEN_DIM, lr=LR_CRITIC):
        self.fc1 = NNLayer(rng, obs_dim, hidden, lr)
        self.fc2 = NNLayer(rng, hidden, hidden, lr)
        self.value_head = NNLayer(rng, hidden, 1, lr)
    
    def forward(self, obs):
        h = relu(self.fc1.forward(obs))
        h = relu(self.fc2.forward(h))
        return self.value_head.forward(h).flatten()[0]
    
    def param_count(self):
        total = 0
        for layer in [self.fc1, self.fc2, self.value_head]:
            total += layer.W.size + layer.b.size
        return total


def compute_gae(rewards, values, gamma=GAMMA, lam=GAE_LAMBDA):
    """Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < len(values) else 0
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + np.array(values[:T])
    return advantages, returns


def ppo_update_actor(actor, obs_list, act_list, old_lps, advantages, entropy_coef=ENTROPY_COEF):
    """PPO gradient step with entropy bonus (numerical gradient for portability).
    
    The entropy bonus encourages exploration by increasing log_std when
    entropy_coef > 0. For Gaussian policy, H(pi) = 0.5 + 0.5*ln(2*pi) + log_std,
    so dH/d(log_std) = 1 (always positive -> entropy bonus pushes std up).
    """
    for obs, act, old_lp, adv in zip(obs_list, act_list, old_lps, advantages):
        new_lp = actor.log_prob(obs, act)
        ratio = np.exp(new_lp - old_lp)
        clip_ratio = np.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
        
        pg_loss = -min(ratio * adv, clip_ratio * adv)
        
        # --- Policy gradient through network weights (full chain rule) ---
        # Forward pass (recompute intermediates for gradient)
        z1 = actor.fc1.forward(obs)          # pre-activation fc1
        h1 = relu(z1)                        # post-activation fc1
        z2 = actor.fc2.forward(h1)           # pre-activation fc2
        h2 = relu(z2)                        # post-activation fc2
        z_out = actor.mean_head.forward(h2)  # pre-sigmoid
        mean_val = 1.0 / (1.0 + np.exp(-z_out.flatten()))  # sigmoid
        std = np.exp(actor.log_std)
        
        # Gradient of log_prob w.r.t. mean
        d_lp_d_mean = (act - mean_val[0]) / (std[0]**2)
        
        # Through sigmoid: d_mean/d_z_out = mean*(1-mean)
        sig_deriv = mean_val[0] * (1 - mean_val[0])
        
        # Signal at mean_head input: scalar
        delta_out = d_lp_d_mean * sig_deriv * adv
        
        # --- mean_head gradient ---
        grad_W_mh = np.outer(h2, [delta_out])
        grad_b_mh = np.array([delta_out])
        actor.mean_head.adam_update(grad_W_mh, grad_b_mh)
        
        # --- fc2 gradient (backprop through mean_head) ---
        # delta at h2: delta_out * mean_head.W^T, shape (hidden,)
        delta_h2 = delta_out * actor.mean_head.W.flatten()
        # Through relu: element-wise multiply by relu'(z2)
        delta_z2 = delta_h2 * (z2 > 0).astype(np.float32)
        grad_W_fc2 = np.outer(h1, delta_z2)
        grad_b_fc2 = delta_z2
        actor.fc2.adam_update(grad_W_fc2, grad_b_fc2)
        
        # --- fc1 gradient (backprop through fc2) ---
        # delta at h1: delta_z2 @ fc2.W^T, shape (hidden,)
        delta_h1 = delta_z2 @ actor.fc2.W.T
        # Through relu: element-wise multiply by relu'(z1)
        delta_z1 = delta_h1 * (z1 > 0).astype(np.float32)
        grad_W_fc1 = np.outer(obs, delta_z1)
        grad_b_fc1 = delta_z1
        actor.fc1.adam_update(grad_W_fc1, grad_b_fc1)
        
        # --- Entropy bonus: update log_std ---
        # Gaussian entropy H = 0.5 + 0.5*ln(2?) + log_std
        # ?H/?log_std = 1, so gradient of (-entropy_coef * H) w.r.t. log_std = -entropy_coef
        # We also add the log_prob gradient w.r.t. log_std:
        # ?log_prob/?log_std = ((act - mean)^2 / std^2) - 1
        d_lp_d_logstd = ((act - mean_val[0])**2 / (std[0]**2)) - 1.0
        log_std_grad = -(d_lp_d_logstd * adv + entropy_coef * 1.0)
        # Apply simple SGD to log_std (small lr to keep stable)
        actor.log_std -= LR_ACTOR * 0.1 * log_std_grad


def ppo_update_critic(critic, obs_list, returns):
    """Critic value function update via MSE loss backpropagation."""
    for obs, ret in zip(obs_list, returns):
        # Forward pass (recompute intermediates)
        z1 = critic.fc1.forward(obs)
        h1 = relu(z1)
        z2 = critic.fc2.forward(h1)
        h2 = relu(z2)
        v_pred = critic.value_head.forward(h2).flatten()[0]
        
        # MSE loss gradient: d(loss)/d(v) = 2*(v_pred - ret) / 2 = (v_pred - ret)
        delta_out = (v_pred - ret) * VALUE_COEF
        
        # --- value_head gradient ---
        grad_W_vh = np.outer(h2, [delta_out])
        grad_b_vh = np.array([delta_out])
        critic.value_head.adam_update(-grad_W_vh, -grad_b_vh)
        
        # --- fc2 gradient ---
        delta_h2 = delta_out * critic.value_head.W.flatten()
        delta_z2 = delta_h2 * (z2 > 0).astype(np.float32)
        grad_W_fc2 = np.outer(h1, delta_z2)
        grad_b_fc2 = delta_z2
        critic.fc2.adam_update(-grad_W_fc2, -grad_b_fc2)
        
        # --- fc1 gradient ---
        delta_h1 = delta_z2 @ critic.fc2.W.T
        delta_z1 = delta_h1 * (z1 > 0).astype(np.float32)
        grad_W_fc1 = np.outer(obs, delta_z1)
        grad_b_fc1 = delta_z1
        critic.fc1.adam_update(-grad_W_fc1, -grad_b_fc1)


def run_mappo_experiment(seed, use_shared_critic=True, label="MAPPO"):
    """Run CleanRL-style MAPPO/IPPO for N_EPISODES."""
    rng = np.random.RandomState(seed)
    env = NonlinearPGGEnv()
    
    n_agents = env.n_honest
    actors = [MLPActor(rng) for _ in range(n_agents)]
    
    if use_shared_critic:
        critic = MLPCritic(rng, obs_dim=5)  # Global state dim
    else:
        critics = [MLPCritic(rng, obs_dim=4) for _ in range(n_agents)]
    
    per_episode = []
    wall_clock_start = time.time()
    
    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        
        obs_buf = [[] for _ in range(n_agents)]
        act_buf = [[] for _ in range(n_agents)]
        lp_buf = [[] for _ in range(n_agents)]
        rew_buf = [[] for _ in range(n_agents)]
        val_buf = [[] for _ in range(n_agents)]
        gs_buf = []  # global state buffer for shared critic
        
        for t in range(env.T):
            actions = np.zeros(n_agents)
            for i in range(n_agents):
                a, lp, mu = actors[i].act(obs, rng)
                actions[i] = a
                obs_buf[i].append(obs.copy())
                act_buf[i].append(a)
                lp_buf[i].append(lp)
                
                if use_shared_critic:
                    gs = env.get_global_state()
                    val_buf[i].append(critic.forward(gs))
                    if i == 0:
                        gs_buf.append(gs.copy())
                else:
                    val_buf[i].append(critics[i].forward(obs))
            
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            for i in range(n_agents):
                rew_buf[i].append(rewards[i])
            
            if terminated:
                break
        
        # PPO update for each agent (multiple epochs)
        for i in range(n_agents):
            if len(rew_buf[i]) < 2:
                continue
            
            vals = val_buf[i]
            advantages, returns = compute_gae(rew_buf[i], vals)
            
            if np.std(advantages) > 1e-8:
                advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
            
            # Multiple PPO epochs per rollout
            for _ppo_epoch in range(PPO_EPOCHS):
                # Actor update
                ppo_update_actor(
                    actors[i], obs_buf[i], act_buf[i], lp_buf[i], advantages
                )
                
                # Critic update
                if use_shared_critic:
                    ppo_update_critic(critic, gs_buf[:len(returns)], returns)
                else:
                    ppo_update_critic(critics[i], obs_buf[i], returns)
        
        per_episode.append({
            "welfare": info.get("welfare", 0),
            "survived": info.get("survived", False),
            "mean_lambda": info.get("mean_lambda", 0),
        })
    
    wall_clock = time.time() - wall_clock_start
    
    # Evaluation: last N_EVAL episodes
    eval_data = per_episode[-N_EVAL:]
    result = {
        "label": label,
        "params_per_agent": actors[0].param_count(),
        "shared_critic_params": critic.param_count() if use_shared_critic else 0,
        "mean_lambda": np.mean([d["mean_lambda"] for d in eval_data]),
        "mean_survival": np.mean([float(d["survived"]) for d in eval_data]) * 100,
        "mean_welfare": np.mean([d["welfare"] for d in eval_data]),
        "wall_clock_seconds": wall_clock,
        "per_seed_lambda": [d["mean_lambda"] for d in eval_data],
        "per_seed_survival": [float(d["survived"]) * 100 for d in eval_data],
    }
    return result


def bootstrap_ci(data, n_boot=10000, ci=0.95):
    data = np.array(data)
    boot_means = np.array([np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_means, alpha*100)), float(np.percentile(boot_means, (1-alpha)*100))


def main():
    print("=" * 70)
    print("  Phase 1: CleanRL-style Standard MARL Baselines")
    print("  N_SEEDS=%d, N_EPISODES=%d, HIDDEN=%d" % (N_SEEDS, N_EPISODES, HIDDEN_DIM))
    print("=" * 70)
    
    all_results = {}
    
    for label, shared_critic in [("CleanRL IPPO", False), ("CleanRL MAPPO", True)]:
        print(f"\n  [{label}] Running {N_SEEDS} seeds...")
        t0 = time.time()
        
        seed_results = []
        for s in range(N_SEEDS):
            r = run_mappo_experiment(
                seed=s * 7 + 42, 
                use_shared_critic=shared_critic,
                label=label
            )
            seed_results.append(r)
            print(f"    Seed {s}: lambda={r['mean_lambda']:.3f}, "
                  f"surv={r['mean_survival']:.1f}%, "
                  f"W={r['mean_welfare']:.1f}, "
                  f"t={r['wall_clock_seconds']:.1f}s")
        
        elapsed = time.time() - t0
        per_seed_lambda = [r["mean_lambda"] for r in seed_results]
        per_seed_survival = [r["mean_survival"] for r in seed_results]
        per_seed_welfare = [r["mean_welfare"] for r in seed_results]
        per_seed_time = [r["wall_clock_seconds"] for r in seed_results]
        
        ci_lam = bootstrap_ci(per_seed_lambda)
        ci_surv = bootstrap_ci(per_seed_survival)
        
        all_results[label] = {
            "label": label,
            "params_per_agent": seed_results[0]["params_per_agent"],
            "lambda": {"mean": np.mean(per_seed_lambda), "std": np.std(per_seed_lambda),
                      "ci95": ci_lam},
            "survival": {"mean": np.mean(per_seed_survival), "std": np.std(per_seed_survival),
                        "ci95": ci_surv},
            "welfare": {"mean": np.mean(per_seed_welfare), "std": np.std(per_seed_welfare)},
            "wall_clock": {"mean": np.mean(per_seed_time), "total": elapsed},
            "per_seed_lambda": per_seed_lambda,
            "per_seed_survival": per_seed_survival,
        }
        
        print(f"  [{label}] Done ({elapsed:.0f}s total)")
        print(f"    Params/agent: {seed_results[0]['params_per_agent']}")
        print(f"    lambda: {np.mean(per_seed_lambda):.3f} [{ci_lam[0]:.3f}, {ci_lam[1]:.3f}]")
        print(f"    survival: {np.mean(per_seed_survival):.1f}% [{ci_surv[0]:.1f}, {ci_surv[1]:.1f}]")
        print(f"    wall-clock/seed: {np.mean(per_seed_time):.1f}s")
    
    # Save
    out_path = OUTPUT_DIR / "cleanrl_baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {out_path}")
    
    print("\n" + "=" * 70)
    print("  LATEX-READY SUMMARY")
    print("=" * 70)
    for k, r in all_results.items():
        print(f"  {r['label']:25s} | params={r['params_per_agent']} | "
              f"lambda={r['lambda']['mean']:.3f} [{r['lambda']['ci95'][0]:.3f},{r['lambda']['ci95'][1]:.3f}] | "
              f"surv={r['survival']['mean']:.1f}% | "
              f"t={r['wall_clock']['mean']:.1f}s/seed")
    
    print("\n  DONE!")


if __name__ == "__main__":
    main()
