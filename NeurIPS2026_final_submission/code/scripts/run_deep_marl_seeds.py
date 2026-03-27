"""
Deep MARL 20-Seed Experiment Runner
====================================
MACCL, LIO, RND PPO를 20 seeds로 실행하여 per-seed 정량 지표를 수집하고
Bootstrap 95% CI를 포함한 결과 JSON을 저장합니다.

Usage:
  python run_deep_marl_seeds.py                    # 전체 실행 (20 seeds × 3 methods)
  python run_deep_marl_seeds.py --method maccl     # MACCL만 실행
  python run_deep_marl_seeds.py --seeds 3 --fast   # 빠른 테스트 (3 seeds, 단축)
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Beta, Normal

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'envs'))
from pettingzoo_pgg_env import PettingZooPGGEnv

# ============================================================
# Shared Config
# ============================================================
HIDDEN_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'deep_marl_20seeds')

# ============================================================
# Network Definitions (shared across methods)
# ============================================================
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_DIM), nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.Tanh()
        )
        self.alpha_head = nn.Linear(HIDDEN_DIM, act_dim)
        self.beta_head = nn.Linear(HIDDEN_DIM, act_dim)
        self.value_head = nn.Linear(HIDDEN_DIM, 1)

    def get_action_value_floor(self, x, floor_val, action=None):
        features = self.net(x)
        alpha = torch.nn.functional.softplus(self.alpha_head(features)) + 1.0
        beta = torch.nn.functional.softplus(self.beta_head(features)) + 1.0
        v = self.value_head(features)
        dist = Beta(alpha, beta)
        if action is None:
            raw_action = dist.sample()
        else:
            raw_action = action
        projected_action = raw_action + torch.nn.functional.relu(floor_val - raw_action)
        return projected_action, raw_action, dist.log_prob(raw_action).sum(dim=-1), v

    def get_action_logprob(self, x, action=None):
        features = self.net(x)
        alpha = torch.nn.functional.softplus(self.alpha_head(features)) + 1.0
        beta = torch.nn.functional.softplus(self.beta_head(features)) + 1.0
        dist = Beta(alpha, beta)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(dim=-1)


class FloorNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor(0.0))
        self.w2 = nn.Parameter(torch.tensor(0.0))
        self.w3 = nn.Parameter(torch.tensor(5.0))

    def forward(self, R_scalar):
        z = self.w1 * R_scalar + self.w2 * (R_scalar ** 2) + self.w3
        return torch.sigmoid(z)


class IncentiveNet(nn.Module):
    def __init__(self, obs_dim, act_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1), nn.Sigmoid()
        )
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x) * 2.0


class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_DIM), nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.Tanh()
        )
        self.alpha_head = nn.Linear(HIDDEN_DIM, act_dim)
        self.beta_head = nn.Linear(HIDDEN_DIM, act_dim)
        self.value_head = nn.Linear(HIDDEN_DIM, 1)
        self.int_value_head = nn.Linear(HIDDEN_DIM, 1)

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


class RNDModule(nn.Module):
    def __init__(self, obs_dim, out_dim=32):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, out_dim)
        )
        for param in self.target.parameters():
            param.requires_grad = False
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, out_dim)
        )

    def forward(self, obs):
        target_features = self.target(obs)
        predicted_features = self.predictor(obs)
        intrinsic_reward = torch.mean((predicted_features - target_features)**2, dim=-1)
        return intrinsic_reward, predicted_features, target_features


# ============================================================
# MACCL Single-Seed Runner
# ============================================================
def run_maccl_seed(seed, max_steps=10000, fast=False):
    if fast:
        max_steps = 3000
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = PettingZooPGGEnv()
    num_agents = len(env.possible_agents)
    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]

    floor_net = FloorNetwork().to(DEVICE)
    mu = torch.tensor(0.0, requires_grad=True, device=DEVICE)
    primal_opt = optim.Adam(floor_net.parameters(), lr=5e-3)
    agent_policy = PolicyNet(obs_dim).to(DEVICE)
    agent_opt = optim.Adam(agent_policy.parameters(), lr=3e-4)

    SAFETY_DELTA = 0.05
    global_step = 0
    obs, _ = env.reset(seed=seed)

    all_survivals = []
    all_welfares = []
    all_lambdas = []

    while global_step < max_steps:
        batch_obs, batch_acts, batch_logprobs = [], [], []
        batch_rewards, batch_values = [], []
        ep_survivals = []
        ep_welfares = []
        ep_lambdas = []

        current_ep_welfare = 0
        current_ep_length = 0
        current_ep_lambda_sum = 0

        for _ in range(200):
            global_step += 1
            actions = {}
            step_lambda_sum = 0

            with torch.no_grad():
                for agent_id in env.agents:
                    o = torch.tensor(obs[agent_id], dtype=torch.float32).to(DEVICE).unsqueeze(0)
                    R_val = o[0, 0]
                    f_val = floor_net(R_val)
                    a_proj, a_raw, logprob, v = agent_policy.get_action_value_floor(o, f_val)
                    actions[agent_id] = a_proj.cpu().numpy().flatten()
                    step_lambda_sum += float(a_proj.cpu().numpy().flatten()[0])
                    batch_obs.append(obs[agent_id])
                    batch_acts.append(a_raw.cpu().numpy().flatten())
                    batch_logprobs.append(logprob.item())
                    batch_values.append(v.item())

            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            n_acting = len(actions)

            welfare_step = 0
            for agent_id in actions.keys():
                batch_rewards.append(rewards[agent_id])
                welfare_step += rewards[agent_id]

            current_ep_welfare += welfare_step / max(n_acting, 1)
            current_ep_lambda_sum += step_lambda_sum / max(n_acting, 1)
            current_ep_length += 1
            obs = next_obs

            if not env.agents:
                obs, _ = env.reset(seed=seed + global_step)
                survived = float(infos[list(actions.keys())[0]].get('survived', False))
                ep_survivals.append(survived)
                ep_welfares.append(current_ep_welfare / max(current_ep_length, 1))
                ep_lambdas.append(current_ep_lambda_sum / max(current_ep_length, 1))
                current_ep_welfare = 0
                current_ep_length = 0
                current_ep_lambda_sum = 0

        # PPO Agent Update
        if len(batch_obs) > 0:
            b_obs = torch.tensor(np.array(batch_obs), dtype=torch.float32).to(DEVICE)
            b_acts = torch.tensor(np.array(batch_acts), dtype=torch.float32).to(DEVICE)
            b_logprobs = torch.tensor(np.array(batch_logprobs), dtype=torch.float32).to(DEVICE)
            b_rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32).to(DEVICE)
            b_values = torch.tensor(np.array(batch_values), dtype=torch.float32).to(DEVICE)

            returns = b_rewards + 0.99 * torch.cat([b_values[1:], torch.tensor([0.0]).to(DEVICE)])
            advantages = returns - b_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            b_inds = np.arange(len(b_obs))
            for _ in range(4):
                np.random.shuffle(b_inds)
                for start in range(0, len(b_obs), 64):
                    end = start + 64
                    mb_inds = b_inds[start:end]
                    R_vals = b_obs[mb_inds, 0]
                    f_vals = floor_net(R_vals).unsqueeze(-1)
                    _, _, newlogprob, newvalue = agent_policy.get_action_value_floor(b_obs[mb_inds], f_vals, b_acts[mb_inds])
                    ratio = (newlogprob - b_logprobs[mb_inds]).exp()
                    pg_loss = -torch.min(advantages[mb_inds] * ratio, advantages[mb_inds] * torch.clamp(ratio, 0.8, 1.2)).mean()
                    v_loss = 0.5 * ((newvalue.view(-1) - returns[mb_inds]) ** 2).mean()
                    agent_loss = pg_loss + 0.5 * v_loss
                    agent_opt.zero_grad()
                    agent_loss.backward()
                    agent_opt.step()

        # Primal-Dual Update
        if len(ep_survivals) > 0:
            surv_rate = np.mean(ep_survivals)
            constraint_violation = SAFETY_DELTA - surv_rate
            with torch.no_grad():
                mu.add_(1e-2 * constraint_violation)
                mu.clamp_(min=0.0)

            primal_opt.zero_grad()
            R_all = b_obs[:, 0]
            f_all = floor_net(R_all).unsqueeze(-1)
            a_proj, _, _, _ = agent_policy.get_action_value_floor(b_obs, f_all, b_acts)
            surrogate = -(advantages.detach().unsqueeze(-1) * a_proj).mean()
            penalty = mu.item() * torch.relu(torch.tensor(constraint_violation)).to(DEVICE)
            primal_loss = surrogate + penalty
            primal_loss.backward()
            primal_opt.step()

            all_survivals.extend(ep_survivals)
            all_welfares.extend(ep_welfares)
            all_lambdas.extend(ep_lambdas)

    # Final metrics (last 30 episodes)
    final_surv = float(np.mean(all_survivals[-30:])) * 100 if len(all_survivals) >= 30 else float(np.mean(all_survivals)) * 100
    final_welfare = float(np.mean(all_welfares[-30:])) if len(all_welfares) >= 30 else float(np.mean(all_welfares))
    final_lambda = float(np.mean(all_lambdas[-30:])) if len(all_lambdas) >= 30 else float(np.mean(all_lambdas))
    floor_at_01 = float(floor_net(torch.tensor(0.1).to(DEVICE)).item())
    dual_mu = float(mu.item())

    return {
        'survival': final_surv,
        'welfare': final_welfare,
        'lambda': final_lambda,
        'floor_R01': floor_at_01,
        'dual_mu': dual_mu,
        'trapped': final_surv < 95.0
    }


# ============================================================
# LIO Single-Seed Runner
# ============================================================
def run_lio_seed(seed, max_steps=5000, fast=False):
    if fast:
        max_steps = 2000

    torch.manual_seed(seed)
    np.random.seed(seed)

    import higher

    env = PettingZooPGGEnv()
    num_agents = len(env.possible_agents)
    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]

    agent_policy = PolicyNet(obs_dim).to(DEVICE)
    agent_opt = optim.Adam(agent_policy.parameters(), lr=3e-4)
    incentive_net = IncentiveNet(obs_dim).to(DEVICE)
    incentive_opt = optim.Adam(incentive_net.parameters(), lr=1e-4)

    global_step = 0
    obs, _ = env.reset(seed=seed)

    all_survivals = []
    all_welfares = []
    all_lambdas = []
    UNROLL_LENGTH = 5

    while global_step < max_steps:
        with higher.innerloop_ctx(agent_policy, agent_opt, copy_initial_weights=False) as (f_policy, diffopt):
            # Inner loop
            for inner_step in range(UNROLL_LENGTH):
                actions = {}
                log_probs = {}
                obs_t = {}
                step_lambda_sum = 0

                for agent_id in env.agents:
                    o = torch.tensor(obs[agent_id], dtype=torch.float32).to(DEVICE).unsqueeze(0)
                    a, lp = f_policy.get_action_logprob(o)
                    actions[agent_id] = a.detach().cpu().numpy().flatten()
                    step_lambda_sum += float(actions[agent_id][0])
                    log_probs[agent_id] = lp
                    obs_t[agent_id] = o

                next_obs, rewards, terminations, truncations, infos = env.step(actions)
                global_step += 1

                if not env.agents:
                    survived = float(infos[list(actions.keys())[0]].get('survived', False))
                    n_acting = len(actions)
                    welfare = sum(rewards[a] for a in actions.keys()) / max(n_acting, 1)
                    mean_lam = step_lambda_sum / max(n_acting, 1)
                    all_survivals.append(survived)
                    all_welfares.append(welfare)
                    all_lambdas.append(mean_lam)
                    obs, _ = env.reset(seed=seed + global_step)
                    break

                total_agent_loss = 0
                for agent_id in actions.keys():
                    env_reward = torch.tensor([rewards[agent_id]], dtype=torch.float32).to(DEVICE)
                    act_t = torch.tensor(actions[agent_id], dtype=torch.float32).to(DEVICE).unsqueeze(0)
                    inc = incentive_net(obs_t[agent_id], act_t)
                    total_agent_loss += -log_probs[agent_id] * (env_reward + inc).detach()

                total_agent_loss = total_agent_loss / num_agents
                diffopt.step(total_agent_loss)
                obs = next_obs
            else:
                # Outer loop update
                actions = {}
                log_probs = {}
                outer_lambda_sum = 0

                for agent_id in env.agents:
                    o = torch.tensor(obs[agent_id], dtype=torch.float32).to(DEVICE).unsqueeze(0)
                    a, lp = f_policy.get_action_logprob(o)
                    actions[agent_id] = a.detach().cpu().numpy().flatten()
                    outer_lambda_sum += float(actions[agent_id][0])
                    log_probs[agent_id] = lp

                next_obs, rewards, terminations, truncations, infos = env.step(actions)
                global_step += 1

                welfare_loss = 0
                n_acting = len(actions)
                step_welfare = 0
                for agent_id in actions.keys():
                    env_reward = torch.tensor([rewards[agent_id]], dtype=torch.float32).to(DEVICE)
                    step_welfare += env_reward.item()
                    welfare_loss += -log_probs[agent_id] * env_reward.detach()

                welfare_loss = welfare_loss / max(n_acting, 1)

                incentive_opt.zero_grad()
                welfare_loss.backward()
                nn.utils.clip_grad_norm_(incentive_net.parameters(), 1.0)
                incentive_opt.step()

                if not env.agents:
                    survived = float(infos[list(actions.keys())[0]].get('survived', False))
                    all_survivals.append(survived)
                    all_welfares.append(step_welfare / max(n_acting, 1))
                    all_lambdas.append(outer_lambda_sum / max(n_acting, 1))
                    obs, _ = env.reset(seed=seed + global_step)
                else:
                    obs = next_obs

    final_surv = float(np.mean(all_survivals[-30:])) * 100 if len(all_survivals) >= 30 else float(np.mean(all_survivals)) * 100 if all_survivals else 0.0
    final_welfare = float(np.mean(all_welfares[-30:])) if len(all_welfares) >= 30 else float(np.mean(all_welfares)) if all_welfares else 0.0
    final_lambda = float(np.mean(all_lambdas[-30:])) if len(all_lambdas) >= 30 else float(np.mean(all_lambdas)) if all_lambdas else 0.0

    return {
        'survival': final_surv,
        'welfare': final_welfare,
        'lambda': final_lambda,
        'trapped': final_surv < 95.0
    }


# ============================================================
# RND PPO Single-Seed Runner
# ============================================================
def run_rnd_seed(seed, max_steps=5000, fast=False):
    if fast:
        max_steps = 2000

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = PettingZooPGGEnv()
    num_agents = len(env.possible_agents)
    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]

    agent_policy = PPOActorCritic(obs_dim).to(DEVICE)
    rnd_module = RNDModule(obs_dim).to(DEVICE)
    agent_opt = optim.Adam(list(agent_policy.parameters()) + list(rnd_module.predictor.parameters()), lr=3e-4)

    RND_COEF = 0.5
    global_step = 0
    obs, _ = env.reset(seed=seed)

    all_survivals = []
    all_welfares = []
    all_lambdas = []
    all_int_rewards = []

    while global_step < max_steps:
        batch_obs, batch_acts, batch_logprobs = [], [], []
        batch_ext_rewards, batch_int_rewards = [], []
        batch_ext_values, batch_int_values = [], []

        ep_lambda_sum = 0
        ep_steps = 0

        for _ in range(100):
            global_step += 1
            actions = {}
            step_lambda_sum = 0

            with torch.no_grad():
                for agent_id in env.agents:
                    o = torch.tensor(obs[agent_id], dtype=torch.float32).to(DEVICE).unsqueeze(0)
                    a, lp, ev, iv = agent_policy.get_action_logprob(o)
                    int_reward, _, _ = rnd_module(o)
                    actions[agent_id] = a.cpu().numpy().flatten()
                    step_lambda_sum += float(actions[agent_id][0])
                    batch_obs.append(obs[agent_id])
                    batch_acts.append(actions[agent_id])
                    batch_logprobs.append(lp.item())
                    batch_ext_values.append(ev.item())
                    batch_int_values.append(iv.item())
                    batch_int_rewards.append(int_reward.item())

            next_obs, ext_rewards, terminations, truncations, infos = env.step(actions)
            n_acting = len(actions)
            ep_lambda_sum += step_lambda_sum / max(n_acting, 1)
            ep_steps += 1

            for agent_id in actions.keys():
                batch_ext_rewards.append(ext_rewards[agent_id])

            obs = next_obs
            if not env.agents:
                survived = float(infos[list(actions.keys())[0]].get('survived', False))
                welfare = sum(ext_rewards[a] for a in actions.keys()) / max(n_acting, 1)
                mean_lam = ep_lambda_sum / max(ep_steps, 1)
                all_survivals.append(survived)
                all_welfares.append(welfare)
                all_lambdas.append(mean_lam)
                ep_lambda_sum = 0
                ep_steps = 0
                obs, _ = env.reset(seed=seed + global_step)

        # PPO Update
        if len(batch_obs) > 0:
            b_obs = torch.tensor(np.array(batch_obs), dtype=torch.float32).to(DEVICE)
            b_acts = torch.tensor(np.array(batch_acts), dtype=torch.float32).to(DEVICE)
            b_logprobs = torch.tensor(np.array(batch_logprobs), dtype=torch.float32).to(DEVICE)
            b_ext_rewards = torch.tensor(np.array(batch_ext_rewards), dtype=torch.float32).to(DEVICE)
            b_int_rewards = torch.tensor(np.array(batch_int_rewards), dtype=torch.float32).to(DEVICE)
            b_ext_values = torch.tensor(np.array(batch_ext_values), dtype=torch.float32).to(DEVICE)
            b_int_values = torch.tensor(np.array(batch_int_values), dtype=torch.float32).to(DEVICE)

            ext_returns = b_ext_rewards + 0.99 * torch.cat([b_ext_values[1:], torch.tensor([0.0]).to(DEVICE)])
            int_returns = b_int_rewards + 0.99 * torch.cat([b_int_values[1:], torch.tensor([0.0]).to(DEVICE)])
            ext_adv = ext_returns - b_ext_values
            int_adv = int_returns - b_int_values
            ext_adv = (ext_adv - ext_adv.mean()) / (ext_adv.std() + 1e-8)
            int_adv = (int_adv - int_adv.mean()) / (int_adv.std() + 1e-8)
            total_adv = ext_adv + RND_COEF * int_adv

            for _ in range(3):
                _, new_lp, new_ext_v, new_int_v = agent_policy.get_action_logprob(b_obs, b_acts)
                ratio = (new_lp - b_logprobs).exp()
                pg_loss = -torch.min(total_adv * ratio, total_adv * torch.clamp(ratio, 0.8, 1.2)).mean()
                ext_v_loss = 0.5 * ((new_ext_v.view(-1) - ext_returns) ** 2).mean()
                int_v_loss = 0.5 * ((new_int_v.view(-1) - int_returns) ** 2).mean()
                _, pred_feats, targ_feats = rnd_module(b_obs)
                predictor_loss = torch.mean((pred_feats - targ_feats.detach())**2)
                loss = pg_loss + ext_v_loss + int_v_loss + predictor_loss
                agent_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent_policy.parameters(), 0.5)
                agent_opt.step()

            all_int_rewards.append(float(b_int_rewards.mean().item()))

    final_surv = float(np.mean(all_survivals[-30:])) * 100 if len(all_survivals) >= 30 else float(np.mean(all_survivals)) * 100 if all_survivals else 0.0
    final_welfare = float(np.mean(all_welfares[-30:])) if len(all_welfares) >= 30 else float(np.mean(all_welfares)) if all_welfares else 0.0
    final_lambda = float(np.mean(all_lambdas[-30:])) if len(all_lambdas) >= 30 else float(np.mean(all_lambdas)) if all_lambdas else 0.0
    final_int_reward = float(all_int_rewards[-1]) if all_int_rewards else 0.0

    return {
        'survival': final_surv,
        'welfare': final_welfare,
        'lambda': final_lambda,
        'int_reward_final': final_int_reward,
        'trapped': final_surv < 95.0
    }


# ============================================================
# Bootstrap CI
# ============================================================
def bootstrap_ci(data, n_boot=10000, ci=0.95):
    data = np.array(data)
    if len(data) == 0:
        return (0.0, 0.0)
    boot_means = np.array([np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_means, alpha * 100)), float(np.percentile(boot_means, (1 - alpha) * 100))


# ============================================================
# Main Runner
# ============================================================
def run_method(method_name, run_fn, n_seeds=20, fast=False):
    print(f"\n{'='*60}")
    print(f"  {method_name} — {n_seeds} seeds on {DEVICE}")
    print(f"{'='*60}")

    per_seed = []
    for s in range(n_seeds):
        seed = 42 + s
        t0 = time.time()
        result = run_fn(seed, fast=fast)
        elapsed = time.time() - t0
        per_seed.append(result)
        print(f"  Seed {s+1:2d}/{n_seeds}: Surv={result['survival']:5.1f}%  Welf={result['welfare']:6.2f}  λ={result['lambda']:.3f}  [{elapsed:.1f}s]")

    # Aggregate
    survivals = [r['survival'] for r in per_seed]
    welfares = [r['welfare'] for r in per_seed]
    lambdas = [r['lambda'] for r in per_seed]

    np.random.seed(42)
    surv_ci = bootstrap_ci(survivals)
    welf_ci = bootstrap_ci(welfares)
    lam_ci = bootstrap_ci(lambdas)

    summary = {
        'method': method_name,
        'n_seeds': n_seeds,
        'device': str(DEVICE),
        'survival_mean': float(np.mean(survivals)),
        'survival_std': float(np.std(survivals)),
        'survival_ci95': surv_ci,
        'welfare_mean': float(np.mean(welfares)),
        'welfare_std': float(np.std(welfares)),
        'welfare_ci95': welf_ci,
        'lambda_mean': float(np.mean(lambdas)),
        'lambda_std': float(np.std(lambdas)),
        'lambda_ci95': lam_ci,
        'per_seed_survival': survivals,
        'per_seed_welfare': welfares,
        'per_seed_lambda': lambdas,
        'trapped': float(np.mean(survivals)) < 95.0,
    }

    # Method-specific extra fields
    if method_name == 'MACCL':
        floors = [r.get('floor_R01', 0) for r in per_seed]
        summary['per_seed_floor_R01'] = floors
        summary['floor_R01_mean'] = float(np.mean(floors))
    elif method_name == 'RND_PPO':
        int_rews = [r.get('int_reward_final', 0) for r in per_seed]
        summary['per_seed_int_reward_final'] = int_rews
        summary['int_reward_final_mean'] = float(np.mean(int_rews))

    print(f"\n  SUMMARY: Surv={summary['survival_mean']:.1f}% CI{surv_ci}  Welf={summary['welfare_mean']:.2f}  λ={summary['lambda_mean']:.3f}")
    return summary


def main():
    parser = argparse.ArgumentParser(description='Deep MARL 20-Seed Runner')
    parser.add_argument('--method', type=str, default='all', choices=['all', 'maccl', 'lio', 'rnd'])
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--fast', action='store_true', help='Reduced steps for quick test')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}
    t_total = time.time()

    methods = {
        'maccl': ('MACCL', run_maccl_seed),
        'lio': ('LIO', run_lio_seed),
        'rnd': ('RND_PPO', run_rnd_seed),
    }

    if args.method == 'all':
        to_run = list(methods.items())
    else:
        to_run = [(args.method, methods[args.method])]

    for key, (name, fn) in to_run:
        summary = run_method(name, fn, n_seeds=args.seeds, fast=args.fast)
        results[key] = summary

    # Save combined JSON
    output = {
        'experiment': 'Deep MARL 20-Seed Validation',
        'config': {
            'device': str(DEVICE),
            'torch_version': torch.__version__,
            'n_seeds': args.seeds,
            'fast_mode': args.fast,
        },
        'results': results,
        'total_time_seconds': time.time() - t_total,
    }

    json_path = os.path.join(OUTPUT_DIR, 'deep_marl_20seeds_results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  ALL DONE in {time.time()-t_total:.0f}s")
    print(f"  Results saved to: {json_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
