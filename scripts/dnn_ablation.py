"""
DNN Ablation: Does the Nash Trap persist with deeper networks?

Tests 2-3 hidden layer MLP policies to prove the Nash Trap is
game-theoretic, not a function approximation limitation.

Reviewer 7.1 response: Camera-ready appendix data.
"""

import numpy as np
import json
import os
import time

# ============================================================
# Config
# ============================================================
N_AGENTS = 20
T_ROUNDS = 50  # Must match paper Table 7 (T = 50)
MULTIPLIER = 1.6
ENDOWMENT = 20.0
R_CRIT = 0.15
R_RECOV = 0.25
SHOCK_PROB = 0.05
SHOCK_MAG = 0.15
STATE_DIM = 4
GAMMA_RL = 0.99
N_EPISODES = 300
N_SEEDS = 5


# ============================================================
# Environment (same as mappo_emergence.py)
# ============================================================
def reset_env():
    return {"R": 0.5, "t": 0, "lam_prev": np.full(N_AGENTS, 0.5), "mean_c": 0.5}

def get_obs(env):
    crisis = 1.0 if env["R"] < R_CRIT else 0.0
    return np.column_stack([
        np.full(N_AGENTS, env["R"]),
        np.full(N_AGENTS, env["mean_c"]),
        env["lam_prev"],
        np.full(N_AGENTS, crisis),
    ])

def step_env(env, lambdas, n_byz, rng):
    lambdas[:n_byz] = 0.0
    contribs = ENDOWMENT * lambdas
    public = (contribs.sum() * MULTIPLIER) / N_AGENTS
    rewards = (ENDOWMENT - contribs) + public
    coop = contribs.mean() / ENDOWMENT
    R = env["R"]
    base = 0.1 * (coop - 0.4)
    if R < R_CRIT:
        R_new = R + base * 0.1
    elif R < R_RECOV:
        R_new = R + base * 0.3
    else:
        R_new = R + base
    if rng.random() < SHOCK_PROB:
        R_new -= SHOCK_MAG
    R_new = float(np.clip(R_new, 0.0, 1.0))
    env["t"] += 1
    done = R_new <= 0.001 or env["t"] >= T_ROUNDS
    env["R"] = R_new
    env["lam_prev"] = lambdas.copy()
    env["mean_c"] = float(coop)
    return rewards, done, {"R": R_new, "welfare": float(rewards.mean()),
                           "mean_lam": float(lambdas[n_byz:].mean()),
                           "collapsed": R_new <= 0.001}


# ============================================================
# MLP Policy (variable depth)
# ============================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

def relu(x):
    return np.maximum(0, x)

class MLPPolicy:
    """Multi-layer perceptron policy with configurable depth."""

    def __init__(self, hidden_layers, lr=0.005):
        self.lr = lr
        self.layers = []
        dims = [STATE_DIM] + hidden_layers + [1]
        for i in range(len(dims) - 1):
            W = np.zeros((dims[i], dims[i+1]))
            b = np.zeros(dims[i+1])
            self.layers.append((W, b))
        self.label = f"MLP-{'-'.join(str(h) for h in hidden_layers)}" if hidden_layers else "Linear"

    def forward(self, obs):
        """obs: (N, dim) -> lambdas: (N,)"""
        x = obs
        for i, (W, b) in enumerate(self.layers):
            x = x @ W + b
            if i < len(self.layers) - 1:
                x = relu(x)
        return sigmoid(x.flatten())

    def get_actions(self, obs, rng, noise_scale=0.1):
        base = self.forward(obs)
        noise = rng.normal(0, noise_scale, size=base.shape)
        return np.clip(base + noise, 0.0, 1.0)

    def update(self, obs_list, action_list, return_list):
        """Numerical gradient update (finite differences)."""
        if len(obs_list) == 0:
            return 0.0

        obs_arr = np.array(obs_list)
        act_arr = np.array(action_list)
        ret_arr = np.array(return_list)
        if ret_arr.std() > 1e-8:
            ret_arr = (ret_arr - ret_arr.mean()) / (ret_arr.std() + 1e-8)

        epsilon = 0.01
        total_grad = 0.0
        for layer_idx in range(len(self.layers)):
            W, b = self.layers[layer_idx]
            # Weight gradients (numerical)
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    W[i, j] += epsilon
                    lam_plus = self.forward(obs_arr)
                    W[i, j] -= 2 * epsilon
                    lam_minus = self.forward(obs_arr)
                    W[i, j] += epsilon
                    dlam = (lam_plus - lam_minus) / (2 * epsilon)
                    grad = (dlam * (act_arr - self.forward(obs_arr)) * ret_arr).mean()
                    W[i, j] += self.lr * grad
                    total_grad += abs(grad)
            # Bias gradients
            for j in range(b.shape[0]):
                b[j] += epsilon
                lam_plus = self.forward(obs_arr)
                b[j] -= 2 * epsilon
                lam_minus = self.forward(obs_arr)
                b[j] += epsilon
                dlam = (lam_plus - lam_minus) / (2 * epsilon)
                grad = (dlam * (act_arr - self.forward(obs_arr)) * ret_arr).mean()
                b[j] += self.lr * grad
                total_grad += abs(grad)
        return float(total_grad)


# ============================================================
# Simplified REINFORCE for MLP (faster)
# ============================================================
class SimpleMLPPolicy:
    """MLP with REINFORCE via log-likelihood gradient estimation."""

    def __init__(self, hidden_layers, lr=0.003):
        self.lr = lr
        self.hidden = hidden_layers
        self.layers = []
        dims = [STATE_DIM] + hidden_layers + [1]
        rng = np.random.RandomState(42)
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            W = rng.randn(dims[i], dims[i+1]) * scale * 0.01
            b = np.zeros(dims[i+1])
            self.layers.append([W, b])
        self.label = f"MLP-{'-'.join(str(h) for h in hidden_layers)}" if hidden_layers else "Linear"

    def forward_with_intermediates(self, obs):
        activations = [obs]
        x = obs
        for i, (W, b) in enumerate(self.layers):
            z = x @ W + b
            if i < len(self.layers) - 1:
                x = relu(z)
            else:
                x = z
            activations.append(x)
        return sigmoid(x.flatten()), activations

    def forward(self, obs):
        lam, _ = self.forward_with_intermediates(obs)
        return lam

    def get_actions(self, obs, rng, noise_scale=0.1):
        base = self.forward(obs)
        noise = rng.normal(0, noise_scale, size=base.shape)
        return np.clip(base + noise, 0.0, 1.0)

    def update(self, obs_list, action_list, return_list):
        if len(obs_list) == 0:
            return 0.0
        obs_arr = np.array(obs_list)
        act_arr = np.array(action_list)
        ret_arr = np.array(return_list)
        if ret_arr.std() > 1e-8:
            ret_arr = (ret_arr - ret_arr.mean()) / (ret_arr.std() + 1e-8)

        lam, activations = self.forward_with_intermediates(obs_arr)
        # d_loss/d_output = (action - lam) * returns (REINFORCE signal)
        delta = ((act_arr - lam) * ret_arr).reshape(-1, 1)

        # Backprop through layers
        for i in range(len(self.layers) - 1, -1, -1):
            W, b = self.layers[i]
            inp = activations[i]
            if i == len(self.layers) - 1:
                # Output layer: sigmoid derivative
                sig_deriv = (lam * (1 - lam)).reshape(-1, 1)
                delta_layer = delta * sig_deriv
            else:
                # Hidden layer: relu derivative
                z = inp @ self.layers[i][0] + self.layers[i][1]
                relu_mask = (z > 0).astype(float)
                delta_layer = delta * relu_mask

            grad_W = (inp.T @ delta_layer) / len(obs_arr)
            grad_b = delta_layer.mean(axis=0)

            self.layers[i][0] += self.lr * grad_W
            self.layers[i][1] += self.lr * grad_b

            if i > 0:
                delta = delta_layer @ W.T

        return float(np.abs(self.layers[-1][0]).mean())


# ============================================================
# Training Loop
# ============================================================
def train_policy(policy, n_episodes=N_EPISODES, n_seeds=N_SEEDS, byz_frac=0.3):
    n_byz = int(N_AGENTS * byz_frac)
    all_data = []

    for seed in range(n_seeds):
        rng_base = np.random.RandomState(seed * 100)
        # Reset policy for each seed
        policy_copy = SimpleMLPPolicy(policy.hidden, lr=policy.lr)
        metrics = []

        for ep in range(n_episodes):
            env = reset_env()
            rng = np.random.RandomState(seed * 10000 + ep)
            obs_buf, act_buf, rew_buf = [], [], []
            ep_lam = []

            for t in range(T_ROUNDS):
                obs = get_obs(env)
                noise = max(0.15 - ep * 0.0003, 0.02)
                lams = policy_copy.get_actions(obs, rng, noise_scale=noise)
                rewards, done, info = step_env(env, lams, n_byz, rng)

                for i in range(n_byz, N_AGENTS):
                    obs_buf.append(obs[i])
                    act_buf.append(lams[i])
                    rew_buf.append(rewards[i])
                ep_lam.append(info["mean_lam"])
                if done:
                    break

            # Discounted returns
            returns = []
            G = 0
            for r in reversed(rew_buf):
                G = r + GAMMA_RL * G
                returns.insert(0, G)

            policy_copy.update(obs_buf, act_buf, returns)
            metrics.append({
                "ep": ep,
                "mean_lam": float(np.mean(ep_lam)),
                "survived": not info["collapsed"],
                "welfare": info["welfare"],
            })

            if ep % 100 == 0:
                recent = metrics[max(0, ep-10):ep+1]
                avg_lam = np.mean([m["mean_lam"] for m in recent])
                avg_surv = np.mean([m["survived"] for m in recent])
                print(f"    [{policy_copy.label}] Seed {seed} Ep {ep:3d} | "
                      f"lam={avg_lam:.3f} surv={avg_surv*100:.0f}%")

        all_data.append(metrics)

    # Summarize last 30 episodes
    flat = [m for s in all_data for m in s[-30:]]
    return {
        "label": policy.label,
        "hidden": policy.hidden,
        "mean_lam": float(np.mean([m["mean_lam"] for m in flat])),
        "survival": float(np.mean([m["survived"] for m in flat])),
        "welfare": float(np.mean([m["welfare"] for m in flat])),
    }


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    OUT = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'dnn_ablation')
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    print("=" * 65)
    print("  DNN ABLATION: Nash Trap vs Network Depth")
    print("  Byzantine=30%, N=20, Non-linear PGG")
    print("=" * 65)

    architectures = [
        [],           # Linear (baseline)
        [16],         # 1 hidden layer
        [32, 16],     # 2 hidden layers
        [64, 32, 16], # 3 hidden layers
    ]

    results = []
    for arch in architectures:
        policy = SimpleMLPPolicy(arch)
        print(f"\n--- {policy.label} ---")
        r = train_policy(policy)
        results.append(r)
        print(f"  => lam={r['mean_lam']:.3f}, surv={r['survival']*100:.1f}%, "
              f"W={r['welfare']:.1f}")

    total = time.time() - t0

    print(f"\n{'=' * 65}")
    print(f"  ABLATION RESULTS (Byz=30%, last 30 ep)")
    print(f"{'=' * 65}")
    print(f"  {'Architecture':<20} | {'Lambda':>7} | {'Survival':>8} | {'Welfare':>8}")
    print(f"  {'-'*50}")
    for r in results:
        print(f"  {r['label']:<20} | {r['mean_lam']:>7.3f} | "
              f"{r['survival']*100:>7.1f}% | {r['welfare']:>8.1f}")

    output = {"results": results, "time_seconds": float(total)}
    path = os.path.join(OUT, "dnn_ablation_results.json")
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Time: {total:.1f}s | Saved: {path}")
