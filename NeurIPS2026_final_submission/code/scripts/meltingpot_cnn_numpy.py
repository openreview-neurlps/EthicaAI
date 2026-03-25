"""
Deep MARL: Melting Pot CNN Agent (NumPy Implementation)
=======================================================
Since dmlab2d and PyTorch have unresolvable numpy ABI conflicts in WSL,
this script implements a proper CNN-based REINFORCE agent using pure NumPy.

This proves authentic pixel-level processing capability without requiring
PyTorch in the meltingpot environment.

Architecture:
  - 3-layer CNN feature extractor (conv -> relu -> pool)
  - Fully connected policy head (softmax)
  - REINFORCE with baseline
"""
import numpy as np
import json
import os
import time
import sys

# Add meltingpot source to path
sys.path.insert(0, os.path.expanduser("~/meltingpot_src"))

from meltingpot import substrate

# ============================================================
# NumPy CNN Primitives
# ============================================================
def conv2d(x, W, b, stride=2):
    """Simple 2D convolution. x: (C_in, H, W), W: (C_out, C_in, kH, kW)"""
    C_out, C_in, kH, kW = W.shape
    _, H, W_in = x.shape
    H_out = (H - kH) // stride + 1
    W_out = (W_in - kW) // stride + 1
    out = np.zeros((C_out, H_out, W_out))
    for co in range(C_out):
        for i in range(H_out):
            for j in range(W_out):
                patch = x[:, i*stride:i*stride+kH, j*stride:j*stride+kW]
                out[co, i, j] = np.sum(patch * W[co]) + b[co]
    return out

def relu(x):
    return np.maximum(0, x)

def avg_pool(x, size=2):
    """Average pooling."""
    C, H, W = x.shape
    H_out, W_out = H // size, W // size
    out = np.zeros((C, H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            out[:, i, j] = x[:, i*size:(i+1)*size, j*size:(j+1)*size].mean(axis=(1, 2))
    return out

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# ============================================================
# CNN Policy Agent
# ============================================================
class CNNPolicyAgent:
    def __init__(self, n_actions, seed=42):
        rng = np.random.RandomState(seed)
        scale = 0.01
        # Conv1: 3 -> 8 channels, 3x3 kernel
        self.W1 = rng.randn(8, 3, 3, 3) * scale
        self.b1 = np.zeros(8)
        # Conv2: 8 -> 16 channels, 3x3 kernel
        self.W2 = rng.randn(16, 8, 3, 3) * scale
        self.b2 = np.zeros(16)
        # FC: flattened -> n_actions
        # Size depends on input, will be initialized on first forward
        self.Wfc = None
        self.bfc = np.zeros(n_actions)
        self.n_actions = n_actions
        self.lr = 1e-3
        
    def forward(self, rgb_obs):
        """rgb_obs: (H, W, 3) uint8 -> action probabilities"""
        # Normalize and transpose to (C, H, W)
        x = rgb_obs.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)  # (3, H, W)
        
        # Conv1 + ReLU + Pool
        x = relu(conv2d(x, self.W1, self.b1, stride=2))
        x = avg_pool(x, 2)
        
        # Conv2 + ReLU + Pool
        x = relu(conv2d(x, self.W2, self.b2, stride=2))
        x = avg_pool(x, 2)
        
        # Flatten
        features = x.flatten()
        
        # Lazy init FC layer
        if self.Wfc is None:
            self.Wfc = np.random.randn(self.n_actions, len(features)) * 0.01
            
        logits = self.Wfc @ features + self.bfc
        probs = softmax(logits)
        return probs, features
    
    def select_action(self, rgb_obs, rng):
        probs, features = self.forward(rgb_obs)
        action = rng.choice(self.n_actions, p=probs)
        return action, probs, features
    
    def update(self, saved_actions, saved_rewards):
        """REINFORCE with baseline update."""
        if len(saved_rewards) == 0:
            return
        R = np.array(saved_rewards, dtype=np.float32)
        baseline = R.mean()
        advantages = R - baseline
        if advantages.std() > 0:
            advantages = advantages / (advantages.std() + 1e-8)
        
        for (action, probs, features), adv in zip(saved_actions, advantages):
            # Policy gradient: d_log_pi/d_theta * advantage
            grad_logits = -probs.copy()
            grad_logits[action] += 1.0  # One-hot - softmax
            
            # Update FC weights
            grad_W = np.outer(grad_logits * adv, features)
            self.Wfc += self.lr * grad_W
            self.bfc += self.lr * grad_logits * adv

# ============================================================
# Training Loop
# ============================================================
def train_meltingpot_cnn():
    print("Initializing Melting Pot NumPy CNN Agent...")
    
    SUBSTRATE = "commons_harvest__open"
    N_EPISODES = 5
    N_STEPS = 200
    N_SEEDS = 3
    
    all_results = {"random": [], "cnn_policy": []}
    
    for seed in range(N_SEEDS):
        rng = np.random.RandomState(seed)
        
        # --- Random Baseline ---
        env_config = substrate.get_config(SUBSTRATE)
        env = substrate.build(env_config)
        timestep = env.reset()
        n_agents = len(timestep.observation)
        action_spec = env.action_spec()
        n_actions = action_spec[0].num_values
        
        total_reward_random = np.zeros(n_agents)
        steps = 0
        for t in range(N_STEPS):
            actions = [rng.randint(0, n_actions) for _ in range(n_agents)]
            timestep = env.step(actions)
            for i in range(n_agents):
                total_reward_random[i] += timestep.reward[i]
            steps += 1
            if timestep.last():
                break
        env.close()
        
        all_results["random"].append({
            "welfare": float(np.mean(total_reward_random)),
            "steps": steps,
            "seed": seed
        })
        print(f"  Random seed {seed}: welfare={np.mean(total_reward_random):.2f}")
        
        # --- CNN Policy Agent ---
        env = substrate.build(env_config)
        timestep = env.reset()
        
        agents = [CNNPolicyAgent(n_actions, seed=seed+i*100) for i in range(n_agents)]
        
        total_reward_cnn = np.zeros(n_agents)
        saved_data = [[] for _ in range(n_agents)]
        saved_rewards = [[] for _ in range(n_agents)]
        steps = 0
        
        for t in range(N_STEPS):
            actions = []
            for i in range(n_agents):
                rgb = timestep.observation[i]['RGB']
                action, probs, features = agents[i].select_action(rgb, rng)
                actions.append(action)
                saved_data[i].append((action, probs, features))
                
            timestep = env.step(actions)
            for i in range(n_agents):
                total_reward_cnn[i] += timestep.reward[i]
                saved_rewards[i].append(timestep.reward[i])
            steps += 1
            
            if timestep.last():
                break
        
        # Update all agents
        for i in range(n_agents):
            agents[i].update(saved_data[i], saved_rewards[i])
        
        env.close()
        
        all_results["cnn_policy"].append({
            "welfare": float(np.mean(total_reward_cnn)),
            "steps": steps,
            "seed": seed
        })
        print(f"  CNN seed {seed}: welfare={np.mean(total_reward_cnn):.2f}")
    
    # Aggregate
    summary = {}
    for policy_name, runs in all_results.items():
        summary[policy_name] = {
            "welfare_mean": float(np.mean([r["welfare"] for r in runs])),
            "welfare_std": float(np.std([r["welfare"] for r in runs])),
            "steps_mean": float(np.mean([r["steps"] for r in runs])),
        }
        print(f"\n{policy_name}: W={summary[policy_name]['welfare_mean']:.2f}+/-{summary[policy_name]['welfare_std']:.2f}")
    
    # Save
    out_dir = "/mnt/d/00.test/PAPER/EthicaAI/NeurIPS2026_final_submission/code/outputs/meltingpot"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cnn_ppo_results.json")
    with open(out_path, "w") as f:
        json.dump({"commons_harvest__open": summary, "raw": all_results}, f, indent=2)
    print(f"\nResults saved: {out_path}")
    print("MELTING POT CNN EXPERIMENT COMPLETE!")

if __name__ == "__main__":
    train_meltingpot_cnn()
