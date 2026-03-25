"""
Deep MARL Implementation: Independent PPO (IPPO) in PyTorch
===========================================================
This script implements a fully functional deep IPPO agent compatible with
the PettingZooPGGEnv. It replaces the naive NumPy implementation with a 
proper Actor-Critic architecture, Generalized Advantage Estimation (GAE), 
and PPO clipped objective.

Actor: 2-layer MLP (hidden_dim=64) -> Beta Distribution (for continuous [0,1] action)
Critic: 2-layer MLP (hidden_dim=64) -> Value Function
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Beta
import numpy as np

# Adjust path to import the env
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'envs'))
from pettingzoo_pgg_env import PettingZooPGGEnv

# Hyperparameters matching 2026 Deep MARL Standards
HIDDEN_DIM = 64
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 64
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.01
VF_COEF = 0.5
MAX_STEPS = 50 * 300  # 300 episodes of 50 steps

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim=1):
        super().__init__()
        # Shared feature extractor (optional, but using separate for simplicity)
        
        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, 1)
        )
        
        # Actor Network (Outputs alpha and beta for Beta distribution)
        self.actor_base = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Tanh(),
        )
        self.actor_alpha = nn.Linear(HIDDEN_DIM, act_dim)
        self.actor_beta = nn.Linear(HIDDEN_DIM, act_dim)
        
    def get_value(self, x):
        return self.critic(x)
        
    def get_action_and_value(self, x, action=None):
        features = self.actor_base(x)
        # Softplus to ensure alpha, beta > 0. Add 1.0 to base to prevent collapse
        alpha = torch.nn.functional.softplus(self.actor_alpha(features)) + 1.0
        beta = torch.nn.functional.softplus(self.actor_beta(features)) + 1.0
        
        dist = Beta(alpha, beta)
        
        if action is None:
            action = dist.sample()
            
        return action, dist.log_prob(action).sum(dim=-1), dist.entropy().sum(dim=-1), self.critic(x)

def train_ippo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training IPPO on {device}")
    
    env = PettingZooPGGEnv()
    num_agents = len(env.possible_agents)
    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
    
    # Independent networks
    agents = {agent_id: ActorCritic(obs_dim).to(device) for agent_id in env.possible_agents}
    optimizers = {agent_id: optim.Adam(agents[agent_id].parameters(), lr=LR, eps=1e-5) for agent_id in env.possible_agents}
    
    # Rollout buffers
    num_steps = 200  # steps per update
    
    # Training Loop
    global_step = 0
    start_time = time.time()
    
    obs, _ = env.reset()
    
    while global_step < MAX_STEPS:
        # Buffers for current rollout
        batch_obs = {a: [] for a in env.possible_agents}
        batch_acts = {a: [] for a in env.possible_agents}
        batch_logprobs = {a: [] for a in env.possible_agents}
        batch_rewards = {a: [] for a in env.possible_agents}
        batch_values = {a: [] for a in env.possible_agents}
        batch_dones = {a: [] for a in env.possible_agents}
        
        # Collect trajectories
        for _ in range(num_steps):
            global_step += 1
            actions = {}
            with torch.no_grad():
                for agent_id in env.agents:
                    o = torch.tensor(obs[agent_id], dtype=torch.float32).to(device).unsqueeze(0)
                    a, logprob, _, v = agents[agent_id].get_action_and_value(o)
                    actions[agent_id] = a.cpu().numpy().flatten()
                    
                    batch_obs[agent_id].append(obs[agent_id])
                    batch_acts[agent_id].append(actions[agent_id])
                    batch_logprobs[agent_id].append(logprob.item())
                    batch_values[agent_id].append(v.item())
            
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            
            for agent_id in actions.keys():
                batch_rewards[agent_id].append(rewards[agent_id])
                batch_dones[agent_id].append(terminations[agent_id] or truncations[agent_id])
                
            obs = next_obs
            
            if not env.agents:
                obs, _ = env.reset()
                
        # PPO Update for each agent
        for agent_id in env.possible_agents:
            if len(batch_obs[agent_id]) == 0: continue
            
            b_obs = torch.tensor(np.array(batch_obs[agent_id]), dtype=torch.float32).to(device)
            b_acts = torch.tensor(np.array(batch_acts[agent_id]), dtype=torch.float32).to(device)
            b_logprobs = torch.tensor(np.array(batch_logprobs[agent_id]), dtype=torch.float32).to(device)
            b_rewards = torch.tensor(np.array(batch_rewards[agent_id]), dtype=torch.float32).to(device)
            b_values = torch.tensor(np.array(batch_values[agent_id]), dtype=torch.float32).to(device)
            b_dones = torch.tensor(np.array(batch_dones[agent_id]), dtype=torch.float32).to(device)
            
            # Bootstrap value
            with torch.no_grad():
                if agent_id in obs:
                    next_o = torch.tensor(obs[agent_id], dtype=torch.float32).to(device).unsqueeze(0)
                    next_v = agents[agent_id].get_value(next_o).item()
                else:
                    next_v = 0.0
                    
            # Compute GAE
            advantages = torch.zeros_like(b_rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(len(b_rewards))):
                if t == len(b_rewards) - 1:
                    nextnonterminal = 1.0 - (1.0 if not env.agents else 0.0) # simplified
                    nextvalues = next_v
                else:
                    nextnonterminal = 1.0 - b_dones[t]
                    nextvalues = b_values[t + 1]
                delta = b_rewards[t] + GAMMA * nextvalues * nextnonterminal - b_values[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + b_values
            
            # Optimize policy and value network
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
                    
                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    # Value loss
                    v_loss = 0.5 * ((newvalue.view(-1) - returns[mb_inds]) ** 2).mean()
                    
                    # Entropy loss
                    entropy_loss = entropy.mean()
                    
                    loss = pg_loss - ENTROPY_COEF * entropy_loss + VF_COEF * v_loss
                    
                    optimizers[agent_id].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agents[agent_id].parameters(), 0.5)
                    optimizers[agent_id].step()

        if global_step % 2000 == 0:
            avg_rew = torch.mean(b_rewards).item()
            avg_act = torch.mean(b_acts).item()
            print(f"[{global_step}/{MAX_STEPS}] SPS: {int(global_step / (time.time() - start_time))} | Mean Reward: {avg_rew:.2f} | Mean Lambda: {avg_act:.3f}")

    print("Training Complete!")
    
    # Save dummy state dict to verify
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'deep_models')
    os.makedirs(out_dir, exist_ok=True)
    torch.save(agents[env.possible_agents[0]].state_dict(), os.path.join(out_dir, 'ippo_agent_0.pt'))
    print(f"Model saved to {out_dir}/ippo_agent_0.pt")

if __name__ == "__main__":
    train_ippo()
