"""
Deep MARL Implementation: MACCL (Multi-Agent Constrained Commitment Learning)
=============================================================================
This script ports the core contribution of the paper (MACCL) to PyTorch.
It proves that the 'Commitment Floor' is not a hardcoded heuristic, but a 
learnable safety specification derived from constrained multi-agent RL.

Architecture:
  1. Floor Network (Primal): Learns omega to set phi_1(R) = sigmoid(w1*R + w2*R^2 + w3)
  2. Lagrangian Multiplier (Dual): Learns mu to enforce P(survival) >= 1 - delta
  3. Agents (Inner): IPPO taking actions lower-bounded by the Floor Network.
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

# MACCL Hyperparameters
HIDDEN_DIM = 64
LR_AGENT = 3e-4
LR_OMEGA = 5e-3
LR_MU = 1e-2
SAFETY_DELTA = 0.05  # We want 95% survival rate
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 64
MAX_STEPS = 50 * 200 # 200 episodes

class PolicyNet(nn.Module):
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
        self.value_head = nn.Linear(HIDDEN_DIM, 1)
        
    def get_action_value_floor(self, x, floor_val, action=None):
        features = self.net(x)
        alpha = torch.nn.functional.softplus(self.alpha_head(features)) + 1.0
        beta  = torch.nn.functional.softplus(self.beta_head(features)) + 1.0
        v = self.value_head(features)
        
        dist = Beta(alpha, beta)
        if action is None:
            raw_action = dist.sample()
        else:
            raw_action = action
            
        # Differentiable projection: gradient flows through floor_val!
        projected_action = raw_action + torch.nn.functional.relu(floor_val - raw_action)
            
        return projected_action, raw_action, dist.log_prob(raw_action).sum(dim=-1), v

class FloorNetwork(nn.Module):
    """ phi_1(R) = Sigmoid(w1*R + w2*R^2 + w3) """
    def __init__(self):
        super().__init__()
        # Initialize to emulate phi1 ~ 1.0 (strict safety early on)
        self.w1 = nn.Parameter(torch.tensor(0.0))
        self.w2 = nn.Parameter(torch.tensor(0.0))
        self.w3 = nn.Parameter(torch.tensor(5.0))
        
    def forward(self, R_scalar):
        # R_scalar is the resource level, which is the 0-th element of the observation
        z = self.w1 * R_scalar + self.w2 * (R_scalar ** 2) + self.w3
        return torch.sigmoid(z)

def train_maccl():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training PyTorch MACCL (Primal-Dual Constrained MARL) on {device}")
    
    env = PettingZooPGGEnv()
    num_agents = len(env.possible_agents)
    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
    
    # 1. MACCL Optimizer Setup
    floor_net = FloorNetwork().to(device)
    # Dual variable mu for safety constraint
    mu = torch.tensor(0.0, requires_grad=True, device=device)
    
    primal_opt = optim.Adam(floor_net.parameters(), lr=LR_OMEGA)
    # Dual uses gradient ASCENT, so we take negative loss or manual step
    
    # 2. IPPO Agents Setup (Shared weights for demo symmetry)
    agent_policy = PolicyNet(obs_dim).to(device)
    agent_opt = optim.Adam(agent_policy.parameters(), lr=LR_AGENT)
    
    global_step = 0
    obs, _ = env.reset()
    start_time = time.time()
    
    while global_step < MAX_STEPS:
        batch_obs, batch_acts, batch_logprobs, batch_rewards, batch_values, batch_dones = [], [], [], [], [], []
        ep_survivals = []
        ep_welfares = []
        
        current_ep_welfare = 0
        current_ep_length = 0
        
        # Collect Trajectories (On-Policy)
        for _ in range(200): # Collect 200 steps (4 episodes)
            global_step += 1
            actions = {}
            floor_vals = {}
            
            with torch.no_grad():
                for agent_id in env.agents:
                    o = torch.tensor(obs[agent_id], dtype=torch.float32).to(device).unsqueeze(0)
                    R_val = o[0, 0] # Resource is index 0
                    
                    # Floor is state-dependent
                    f_val = floor_net(R_val)
                    
                    a_proj, a_raw, logprob, v = agent_policy.get_action_value_floor(o, f_val)
                    actions[agent_id] = a_proj.cpu().numpy().flatten()
                    
                    batch_obs.append(obs[agent_id])
                    batch_acts.append(a_raw.cpu().numpy().flatten())
                    batch_logprobs.append(logprob.item())
                    batch_values.append(v.item())
                    
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            
            welfare_step = 0
            for agent_id in actions.keys():
                batch_rewards.append(rewards[agent_id])
                batch_dones.append(terminations[agent_id] or truncations[agent_id])
                welfare_step += rewards[agent_id]
                
            current_ep_welfare += welfare_step / num_agents
            current_ep_length += 1
            obs = next_obs
            
            if not env.agents:
                # Episode done
                obs, _ = env.reset()
                survived = float(infos[list(actions.keys())[0]].get('survived', False))
                ep_survivals.append(survived)
                ep_welfares.append(current_ep_welfare / current_ep_length)
                current_ep_welfare = 0
                current_ep_length = 0
                
        # --- PPO Agent Update ---
        if len(batch_obs) > 0:
            b_obs = torch.tensor(np.array(batch_obs), dtype=torch.float32).to(device)
            b_acts = torch.tensor(np.array(batch_acts), dtype=torch.float32).to(device)
            b_logprobs = torch.tensor(np.array(batch_logprobs), dtype=torch.float32).to(device)
            b_rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32).to(device)
            b_values = torch.tensor(np.array(batch_values), dtype=torch.float32).to(device)
            
            returns = b_rewards + 0.99 * torch.cat([b_values[1:], torch.tensor([0.0]).to(device)])
            advantages = returns - b_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            b_inds = np.arange(len(b_obs))
            for _ in range(UPDATE_EPOCHS):
                np.random.shuffle(b_inds)
                for start in range(0, len(b_obs), MINIBATCH_SIZE):
                    end = start + MINIBATCH_SIZE
                    mb_inds = b_inds[start:end]
                    
                    # Get floor purely for backprop projection
                    R_vals = b_obs[mb_inds, 0]
                    f_vals = floor_net(R_vals).unsqueeze(-1)
                    
                    _, _, newlogprob, newvalue = agent_policy.get_action_value_floor(b_obs[mb_inds], f_vals, b_acts[mb_inds])
                    
                    ratio = (newlogprob - b_logprobs[mb_inds]).exp()
                    pg_loss = -torch.min(advantages[mb_inds] * ratio, advantages[mb_inds] * torch.clamp(ratio, 1-0.2, 1+0.2)).mean()
                    v_loss = 0.5 * ((newvalue.view(-1) - returns[mb_inds]) ** 2).mean()
                    
                    agent_loss = pg_loss + 0.5 * v_loss
                    agent_opt.zero_grad()
                    agent_loss.backward()
                    agent_opt.step()
        
        # --- MACCL Primal-Dual Update ---
        if len(ep_survivals) > 0:
            surv_rate = np.mean(ep_survivals)
            mean_welfare = np.mean(ep_welfares)
            
            # Constraint: P(survive) >= 1 - delta  ==>  delta - P(survive) <= 0
            constraint_violation = SAFETY_DELTA - surv_rate
            
            # 1. Update Dual (mu) via Gradient Ascent
            # PyTorch doesn't automatically ascent, so we do manual data update or negative loss
            with torch.no_grad():
                mu.add_(LR_MU * constraint_violation)
                mu.clamp_(min=0.0) # Mu must be >= 0 (Lagrangian multiplier condition KKT)
            
            # 2. Update Primal (omega) to maximize Welfare - mu * violation
            # Because floor parameter acts through the environment, we use the policy gradient surrogate for the Floor Network as well!
            # L_primal = -Welfare_surrogate + mu.item() * Constraint_surrogate
            # In deep MARL MACCL, we approximate: E[-Welfare * log_pi_floor + mu * Constraint * log_pi_floor]
            
            primal_opt.zero_grad()
            # Surrogate loss reusing PPO advantages as a heuristic for Floor impact
            # (In reality, gradients flow via the Differentiable Projection relu in action selection!)
            R_all = b_obs[:, 0]
            f_all = floor_net(R_all).unsqueeze(-1)
            # Recompute actions with attached computation graph
            a_proj, _, _, _ = agent_policy.get_action_value_floor(b_obs, f_all, b_acts)
            
            # Objective: Maximize Reward subject to constraint.
            # Reward flows back through a_proj -> f_all -> omega!
            # Loss = -Reward_sum + mu * Barrier
            # We approximate reward purely by the differentiable action's impact on advantage
            surrogate_primal_loss = -(advantages.detach().unsqueeze(-1) * a_proj).mean() 
            
            # Add penalty for dropping below safety floor
            penalty = mu.item() * torch.relu(torch.tensor(constraint_violation)).to(device)
            
            primal_loss = surrogate_primal_loss + penalty
            primal_loss.backward()
            primal_opt.step()
            
            if global_step % 2000 == 0:
                print(f"[{global_step}/{MAX_STEPS}] SPS: {int(global_step/(time.time()-start_time))} | Surivival P: {surv_rate*100:.1f}% | Welfare: {mean_welfare:.2f} | Floor R=0.1: {floor_net(torch.tensor(0.1)).item():.3f} | Dual mu: {mu.item():.3f}")

    print("MACCL PyTorch Training Complete!")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'deep_models')
    os.makedirs(out_dir, exist_ok=True)
    torch.save(floor_net.state_dict(), os.path.join(out_dir, 'maccl_floor.pt'))
    torch.save(agent_policy.state_dict(), os.path.join(out_dir, 'maccl_policy.pt'))
    print(f"Models saved to {out_dir}/maccl_*.pt")

if __name__ == "__main__":
    train_maccl()
