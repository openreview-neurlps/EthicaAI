"""
Deep MARL Implementation: Melting Pot CNN+PPO Agent
===================================================
This script implements a native Deep Convolutional Neural Network (CNN) 
for the dm-meltingpot `commons_harvest__open` environment.

It extracts raw RGB frame observations and processes them through a CNN 
feature extractor, solving the "fake deep learning" criticism of the prior 
linear REINFORCE model. 
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import dmlab2d
from meltingpot.python import substrate

# Hyperparameters
LR = 5e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 32
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.01
VF_COEF = 0.5
MAX_STEPS = 500 * 50 # 50 episodes of 500 steps

class MeltingPotCNN(nn.Module):
    """ Standard CNN for processing Melting Pot RGB pixels (e.g., 88x88x3). """
    def __init__(self, action_dim):
        super().__init__()
        
        # Melting pot usually returns small partial observable views
        # We assume RGB inputs (C, H, W)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Define dummy forward to find output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 88, 88)
            # Actually meltingpot default sprite size varies, assume ~ 88x88 for commons harvest or dynamically calculate?
            # It's safer to use AdaptiveAvgPool2d to enforce a fixed size before linear layer.
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), # Force output to 32 x 4 x 4 = 512
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)

    def get_action_and_value(self, x, action=None):
        # x is assumed to be (B, C, H, W) and normalized [0, 1]
        features = self.fc(self.cnn(x))
        logits = self.actor(features)
        v = self.critic(features)
        
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
            
        return action, dist.log_prob(action), dist.entropy(), v

    def get_value(self, x):
        return self.critic(self.fc(self.cnn(x)))

def train_meltingpot_cnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Melting Pot CNN PPO on {device}...")
    
    # 1. Environment Setup
    # Meltingpot uses dm_env interface, we will build a simplified rollout loop
    try:
        env = substrate.build("commons_harvest__open")
    except Exception as e:
        print(f"Meltingpot build failed (are we in WSL?). Mocking interface for dev: {e}")
        # In case we run this script outside WSL to check syntax
        return
        
    action_spec = env.action_spec()
    action_dim = action_spec[0].num_values # Discrete actions (usually 7 or 8)
    num_agents = len(action_spec)
    
    agents = {f"agent_{i}": MeltingPotCNN(action_dim).to(device) for i in range(num_agents)}
    optimizers = {f"agent_{i}": optim.Adam(agents[f"agent_{i}"].parameters(), lr=LR) for i in range(num_agents)}
    
    global_step = 0
    start_time = time.time()
    
    timestep = env.reset()
    
    # Training Loop Outline (Single Node IPPO)
    while global_step < MAX_STEPS:
        batch_obs = {f"agent_{i}": [] for i in range(num_agents)}
        batch_acts = {f"agent_{i}": [] for i in range(num_agents)}
        batch_logprobs = {f"agent_{i}": [] for i in range(num_agents)}
        batch_rewards = {f"agent_{i}": [] for i in range(num_agents)}
        batch_values = {f"agent_{i}": [] for i in range(num_agents)}
        batch_dones = {f"agent_{i}": [] for i in range(num_agents)}
        
        for _ in range(100): # Collect 100 steps
            global_step += 1
            actions = []
            
            with torch.no_grad():
                for i in range(num_agents):
                    # Melting Pot returns list of obs dicts
                    # RGB is shape (H, W, 3) -> convert to (1, 3, H, W)
                    agent_rgb = timestep.observation[i]['RGB']
                    agent_obs = torch.tensor(agent_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                    
                    a, logprob, _, v = agents[f"agent_{i}"].get_action_and_value(agent_obs)
                    actions.append(a.item())
                    
                    batch_obs[f"agent_{i}"].append(agent_obs.cpu().numpy().squeeze(0))
                    batch_acts[f"agent_{i}"].append(a.item())
                    batch_logprobs[f"agent_{i}"].append(logprob.item())
                    batch_values[f"agent_{i}"].append(v.item())
            
            timestep = env.step(actions)
            
            for i in range(num_agents):
                batch_rewards[f"agent_{i}"].append(timestep.reward[i])
                batch_dones[f"agent_{i}"].append(timestep.last())
                
            if timestep.last():
                timestep = env.reset()
                
        # --- PPO Update Logic (same as IPPO script) ---
        # Update each agent using their respective collected trajectory
        for agent_idx in range(num_agents):
            agent_id = f"agent_{agent_idx}"
            if len(batch_obs[agent_id]) == 0: continue
            
            b_obs = torch.tensor(np.array(batch_obs[agent_id]), dtype=torch.float32).to(device)
            b_acts = torch.tensor(np.array(batch_acts[agent_id]), dtype=torch.long).to(device)
            b_logprobs = torch.tensor(np.array(batch_logprobs[agent_id]), dtype=torch.float32).to(device)
            b_rewards = torch.tensor(np.array(batch_rewards[agent_id]), dtype=torch.float32).to(device)
            b_values = torch.tensor(np.array(batch_values[agent_id]), dtype=torch.float32).to(device)
            b_dones = torch.tensor(np.array(batch_dones[agent_id]), dtype=torch.float32).to(device)
            
            with torch.no_grad():
                agent_rgb = timestep.observation[agent_idx]['RGB']
                next_o = torch.tensor(agent_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                next_v = agents[agent_id].get_value(next_o).item()
                
            advantages = torch.zeros_like(b_rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(len(b_rewards))):
                if t == len(b_rewards) - 1:
                    nextnonterminal = 1.0 - timestep.last()
                    nextvalues = next_v
                else:
                    nextnonterminal = 1.0 - b_dones[t]
                    nextvalues = b_values[t + 1]
                delta = b_rewards[t] + GAMMA * nextvalues * nextnonterminal - b_values[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + b_values
            
            b_inds = np.arange(len(b_obs))
            for epoch in range(UPDATE_EPOCHS):
                np.random.shuffle(b_inds)
                for start in range(0, len(b_obs), MINIBATCH_SIZE):
                    end = start + MINIBATCH_SIZE
                    mb_inds = b_inds[start:end]
                    
                    _, newlogprob, entropy, newvalue = agents[agent_id].get_action_and_value(b_obs[mb_inds], b_acts[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    
                    mb_advantages = advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    v_loss = 0.5 * ((newvalue.view(-1) - returns[mb_inds]) ** 2).mean()
                    entropy_loss = entropy.mean()
                    
                    loss = pg_loss - ENTROPY_COEF * entropy_loss + VF_COEF * v_loss
                    
                    optimizers[agent_id].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agents[agent_id].parameters(), 0.5)
                    optimizers[agent_id].step()

        if global_step % 500 == 0:
            print(f"[{global_step}/{MAX_STEPS}] Melting Pot CNN PPO | SPS: {int(global_step / (time.time() - start_time))} | Reward: {b_rewards.mean().item():.3f}")

    import json
    
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'meltingpot')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'cnn_ppo_results.json')
    
    results = {
        "commons_harvest__open": {
            "cnn_ppo": {
                "welfare_mean": b_rewards.mean().item(),
                "steps_mean": sum([len(batch_obs[f"agent_{i}"]) for i in range(num_agents)]) / num_agents
            }
        }
    }
    
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Melting Pot Authentic CNN PPO Training Complete! Results saved to {out_file}")

if __name__ == "__main__":
    train_meltingpot_cnn()
