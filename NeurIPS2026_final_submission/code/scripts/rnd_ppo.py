"""
Deep MARL Implementation: Random Network Distillation (RND) + PPO
=================================================================
Ablation Study for Phase DeepMARL.
This script tests whether State-of-the-Art exploration techniques (RND)
can break the TPSD (Tragedy of the Public Sector Destruction) Nash Trap
without the MACCL safety floor.

If RND fails to survive the environment, it proves the structural
necessity of the MACCL Commitment Floor.

RND Logic:
  - Target Network: Fixed random weights
  - Predictor Network: Learns to predict Target's output
  - Intrinsic Reward: MSE(Target, Predictor) -> encourages visiting novel states
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Beta
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'envs'))
from pettingzoo_pgg_env import PettingZooPGGEnv

# Hyperparameters
HIDDEN_DIM = 64
LR_AGENT = 3e-4
LR_RND = 3e-4
MAX_STEPS = 50 * 100
RND_COEF = 0.5  # Weight of intrinsic reward

class RNDModule(nn.Module):
    def __init__(self, obs_dim, out_dim=32):
        super().__init__()
        # Target network (Fixed, no gradients)
        self.target = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, out_dim)
        )
        for param in self.target.parameters():
            param.requires_grad = False
            
        # Predictor network (Learnable)
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, out_dim)
        )
        
    def forward(self, obs):
        target_features = self.target(obs)
        predicted_features = self.predictor(obs)
        # Intrinsic reward is MSE loss per sample
        intrinsic_reward = torch.mean((predicted_features - target_features)**2, dim=-1)
        return intrinsic_reward, predicted_features, target_features

class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Tanh()
        )
        self.alpha_head = nn.Linear(HIDDEN_DIM, act_dim)
        self.beta_head = nn.Linear(HIDDEN_DIM, act_dim)
        self.value_head = nn.Linear(HIDDEN_DIM, 1) # Extrinsic value
        self.int_value_head = nn.Linear(HIDDEN_DIM, 1) # Intrinsic value stream
        
    def forward(self, x):
        features = self.net(x)
        alpha = torch.nn.functional.softplus(self.alpha_head(features)) + 1.0
        beta = torch.nn.functional.softplus(self.beta_head(features)) + 1.0
        ext_v = self.value_head(features)
        int_v = self.int_value_head(features)
        return alpha, beta, ext_v, int_v

    def get_action_logprob(self, x, action=None):
        alpha, beta, ext_v, int_v = self.forward(x)
        dist = Beta(alpha, beta)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(dim=-1), ext_v, int_v

def train_rnd_ppo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Ablation: RND PPO on {device}")
    
    env = PettingZooPGGEnv()
    num_agents = len(env.possible_agents)
    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
    
    agent_policy = PPOActorCritic(obs_dim).to(device)
    rnd_module = RNDModule(obs_dim).to(device)
    
    agent_opt = optim.Adam(list(agent_policy.parameters()) + list(rnd_module.predictor.parameters()), lr=LR_AGENT)
    
    global_step = 0
    obs, _ = env.reset()
    start_time = time.time()
    
    while global_step < MAX_STEPS:
        batch_obs, batch_acts, batch_logprobs = [], [], []
        batch_ext_rewards, batch_int_rewards = [], []
        batch_ext_values, batch_int_values = [], []
        batch_dones = []
        
        # Rollout
        for _ in range(100):
            global_step += 1
            actions = {}
            with torch.no_grad():
                for agent_id in env.agents:
                    o = torch.tensor(obs[agent_id], dtype=torch.float32).to(device).unsqueeze(0)
                    a, lp, ev, iv = agent_policy.get_action_logprob(o)
                    
                    int_reward, _, _ = rnd_module(o)
                    
                    actions[agent_id] = a.cpu().numpy().flatten()
                    
                    batch_obs.append(obs[agent_id])
                    batch_acts.append(actions[agent_id])
                    batch_logprobs.append(lp.item())
                    batch_ext_values.append(ev.item())
                    batch_int_values.append(iv.item())
                    batch_int_rewards.append(int_reward.item())
                    
            next_obs, ext_rewards, terminations, truncations, infos = env.step(actions)
            
            for agent_id in actions.keys():
                batch_ext_rewards.append(ext_rewards[agent_id])
                batch_dones.append(terminations[agent_id] or truncations[agent_id])
                
            obs = next_obs
            if not env.agents:
                obs, _ = env.reset()
                
        # Optimize
        b_obs = torch.tensor(np.array(batch_obs), dtype=torch.float32).to(device)
        b_acts = torch.tensor(np.array(batch_acts), dtype=torch.float32).to(device)
        b_logprobs = torch.tensor(np.array(batch_logprobs), dtype=torch.float32).to(device)
        b_ext_rewards = torch.tensor(np.array(batch_ext_rewards), dtype=torch.float32).to(device)
        b_int_rewards = torch.tensor(np.array(batch_int_rewards), dtype=torch.float32).to(device)
        b_ext_values = torch.tensor(np.array(batch_ext_values), dtype=torch.float32).to(device)
        b_int_values = torch.tensor(np.array(batch_int_values), dtype=torch.float32).to(device)
        
        # Combine extrinsic and intrinsic returns mapping
        ext_returns = b_ext_rewards + 0.99 * torch.cat([b_ext_values[1:], torch.tensor([0.0]).to(device)])
        int_returns = b_int_rewards + 0.99 * torch.cat([b_int_values[1:], torch.tensor([0.0]).to(device)])
        
        ext_adv = ext_returns - b_ext_values
        int_adv = int_returns - b_int_values
        
        ext_adv = (ext_adv - ext_adv.mean()) / (ext_adv.std() + 1e-8)
        int_adv = (int_adv - int_adv.mean()) / (int_adv.std() + 1e-8)
        
        total_adv = ext_adv + RND_COEF * int_adv
        
        # PPO epoch
        for _ in range(3): # 3 epochs
            _, _, new_ext_v, new_int_v = agent_policy.forward(b_obs)
            _, new_lp, _, _ = agent_policy.get_action_logprob(b_obs, b_acts)
            
            ratio = (new_lp - b_logprobs).exp()
            pg_loss = -torch.min(total_adv * ratio, total_adv * torch.clamp(ratio, 0.8, 1.2)).mean()
            
            ext_v_loss = 0.5 * ((new_ext_v.view(-1) - ext_returns) ** 2).mean()
            int_v_loss = 0.5 * ((new_int_v.view(-1) - int_returns) ** 2).mean()
            
            # Predictor loss (train predictor to match target)
            int_rew_pred, pred_feats, targ_feats = rnd_module(b_obs)
            predictor_loss = torch.mean((pred_feats - targ_feats.detach())**2)
            
            loss = pg_loss + ext_v_loss + int_v_loss + predictor_loss
            
            agent_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent_policy.parameters(), 0.5)
            agent_opt.step()
            
        if global_step % 500 == 0:
            print(f"[{global_step}/{MAX_STEPS}] RND PPO SPS: {int(global_step/(time.time()-start_time))} | Ext Reward: {b_ext_rewards.mean().item():.3f} | Int Reward: {b_int_rewards.mean().item():.4f}")

    print("RND Ablation Complete! Saving models.")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'deep_models')
    os.makedirs(out_dir, exist_ok=True)
    torch.save(agent_policy.state_dict(), os.path.join(out_dir, 'rnd_policy.pt'))

if __name__ == "__main__":
    train_rnd_ppo()
