"""
EthicaAI — Melting Pot Benchmark v2
Commitment Floors for Tipping-Point Commons

수정 사항 (v1 → v2):
1. Action space 정확한 매핑 (action 7 = interact)
2. clean_up 메커니즘 반영 (강 청소 → 사과 수확 2단계)
3. 에피소드 길이 500 → 1000 step
4. Observation 기반 행동: RGB에서 강/사과 타일 감지
5. Reward shaping: 강 청소 시 intrinsic reward 추가
6. Floor 메커니즘: action 7 (interact) 직접 부스팅
"""

import numpy as np
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from meltingpot import substrate

# ─── 설정 ───────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

N_SEEDS = 20
N_STEPS = 1000  # v1에서 500 → 1000으로 증가
TRAIN_EPISODES = 500  # v1에서 100 → 500
TRAIN_HORIZON = 500  # 에피소드 당 스텝 (v1에서 200 → 500)
SUBSTRATES = ['commons_harvest__open', 'clean_up']
FLOOR_PROBS = [0.0, 0.3, 0.5, 0.7]

# ─── Melting Pot Action Space (정확한 매핑) ───
# 0=noop, 1=forward, 2=backward, 3=strafe_left, 4=strafe_right,
# 5=turn_left, 6=turn_right, 7=interact (fire/clean/harvest)
ACTION_NOOP = 0
ACTION_FORWARD = 1
ACTION_INTERACT = 7  # 핵심 협력 행동
N_ACTIONS = 8

# 이동 행동 (강/사과를 찾아가기 위한)
MOVE_ACTIONS = [1, 2, 3, 4, 5, 6]  # forward, backward, strafe, turn
# 협력 행동 (강 청소 또는 사과 수확)
COOP_ACTIONS = [7]  # interact만이 실제 협력


# ─── RGB 기반 타일 감지 (clean_up 전용) ───────
def detect_tile_type(obs_rgb):
    """
    clean_up 기판의 RGB 관찰에서 에이전트 주변 타일 유형 감지.
    중앙 근처 픽셀의 색상으로 판단.

    clean_up 색상 (대략):
    - 강 (오염): 갈색/어두운 색 (R>100, G<80, B<80)
    - 강 (깨끗): 파란색 (R<80, G<80, B>150)
    - 사과: 녹색 (R<80, G>150, B<80)
    - 빈 땅: 회색/검정
    """
    h, w = obs_rgb.shape[:2]
    # 에이전트 시야 중앙 상단 (앞에 있는 타일)
    center_region = obs_rgb[h//4:h//2, w//3:2*w//3, :]

    avg_r = np.mean(center_region[:, :, 0])
    avg_g = np.mean(center_region[:, :, 1])
    avg_b = np.mean(center_region[:, :, 2])

    # 강 (오염됨) 감지
    if avg_r > 80 and avg_g < 60 and avg_b < 60:
        return 'dirty_river'
    # 사과 감지
    if avg_g > 100 and avg_r < 80:
        return 'apple'
    # 강 (깨끗) 감지
    if avg_b > 100 and avg_r < 80 and avg_g < 80:
        return 'clean_river'
    return 'empty'


# ─── 정책 정의 (v2: 정확한 action 매핑) ─────
def selfish_policy(obs, agent_id, rng, t):
    """완전 이기적: 랜덤 이동, interact는 낮은 확률"""
    return rng.randint(0, N_ACTIONS)


def make_commitment_floor_policy(phi):
    """
    Commitment Floor 정책 (v2):
    - P(interact) >= phi (최소 협력 확률)
    - 나머지 확률로 이동
    - clean_up에서는 RGB 관찰 기반으로 강 근처에서 interact 확률 추가 증가
    """
    def policy(obs, agent_id, rng, t):
        # 기본 floor: phi 확률로 interact
        if rng.random() < phi:
            return ACTION_INTERACT

        # 나머지: 주로 전진 (탐색) + 가끔 회전
        r = rng.random()
        if r < 0.5:
            return ACTION_FORWARD
        elif r < 0.7:
            return rng.choice([5, 6])  # 회전
        else:
            return rng.randint(0, N_ACTIONS)  # 랜덤

    return policy


def make_observation_aware_policy(phi):
    """
    관찰 기반 Commitment Floor (clean_up 전용):
    - 강(오염) 감지 → interact 확률 대폭 증가
    - 사과 감지 → interact
    - 그 외 → 탐색 (forward + 회전)
    """
    def policy(obs, agent_id, rng, t):
        obs_rgb = obs.get('RGB', np.zeros((88, 88, 3), dtype=np.uint8))
        tile = detect_tile_type(obs_rgb)

        if tile == 'dirty_river':
            # 오염된 강 앞 → 높은 확률로 청소
            if rng.random() < max(phi, 0.8):
                return ACTION_INTERACT
        elif tile == 'apple':
            # 사과 앞 → 수확
            return ACTION_INTERACT

        # 기본 floor + 탐색
        if rng.random() < phi:
            return ACTION_INTERACT

        # 탐색: forward 위주
        if rng.random() < 0.6:
            return ACTION_FORWARD
        return rng.choice([5, 6])  # 회전

    return policy


# ─── 에피소드 실행 ──────────────────────────
def run_episode(substrate_name, policy_fn, seed=0, n_steps=N_STEPS):
    rng = np.random.RandomState(seed)
    env_config = substrate.get_config(substrate_name)
    env = substrate.build(substrate_name, roles=env_config.default_player_roles)
    timestep = env.reset()
    n_agents = len(timestep.observation)
    total_reward = np.zeros(n_agents)

    for t in range(n_steps):
        actions = {}
        for i in range(n_agents):
            actions[i] = policy_fn(timestep.observation[i], i, rng, t)
        timestep = env.step(actions)
        for i in range(n_agents):
            total_reward[i] += float(timestep.reward[i])
        if timestep.last():
            break

    env.close()
    return {
        'mean_reward': float(np.mean(total_reward)),
        'total_reward': float(np.sum(total_reward)),
        'per_agent': total_reward.tolist(),
    }


# ─── CNN 정책 (IPPO) ────────────────────────
class CNNPolicy(nn.Module):
    def __init__(self, n_actions=N_ACTIONS):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 88, 88)
            conv_out = self.conv(dummy).shape[1]
        self.fc = nn.Sequential(nn.Linear(conv_out, 256), nn.ReLU())
        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        feat = self.fc(self.conv(x))
        return self.actor(feat), self.critic(feat)

    def get_action(self, obs, floor_prob=0.0):
        logits, value = self.forward(obs)
        probs = torch.softmax(logits, dim=-1)

        # Floor: interact(action 7) 확률을 최소 floor_prob으로 보장
        if floor_prob > 0:
            interact_idx = ACTION_INTERACT
            current_interact_prob = probs[:, interact_idx]
            # interact 확률이 floor_prob 미만이면 부스팅
            deficit = torch.clamp(floor_prob - current_interact_prob, min=0)
            if deficit.sum() > 0:
                boost = torch.zeros_like(probs)
                boost[:, interact_idx] = deficit
                # 다른 action에서 균등하게 차감
                reduction = deficit / (N_ACTIONS - 1)
                for a in range(N_ACTIONS):
                    if a != interact_idx:
                        boost[:, a] = -reduction
                probs = torch.clamp(probs + boost, min=1e-8)
                probs = probs / probs.sum(dim=-1, keepdim=True)

        dist = Categorical(probs=probs)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)


# ─── IPPO 학습 (v2: reward shaping + 정확한 floor) ───
def train_ippo(substrate_name, floor_prob=0.0, seed=0, verbose=True,
               n_episodes=TRAIN_EPISODES, horizon=TRAIN_HORIZON):
    rng = np.random.RandomState(seed)
    env_config = substrate.get_config(substrate_name)

    # 에이전트 수 확인
    env = substrate.build(substrate_name, roles=env_config.default_player_roles)
    ts = env.reset()
    n_agents = len(ts.observation)
    env.close()

    policies = [CNNPolicy().to(device) for _ in range(n_agents)]
    optimizers = [optim.Adam(p.parameters(), lr=3e-4) for p in policies]

    train_rewards = []

    for ep in range(n_episodes):
        env = substrate.build(substrate_name, roles=env_config.default_player_roles)
        ts = env.reset()

        ep_data = {i: {'obs': [], 'acts': [], 'logp': [], 'vals': [], 'rews': []}
                   for i in range(n_agents)}
        ep_reward = 0

        for t in range(horizon):
            actions = {}
            for i in range(n_agents):
                obs_rgb = ts.observation[i].get('RGB', np.zeros((88, 88, 3), dtype=np.uint8))
                obs_t = torch.tensor(obs_rgb, device=device, dtype=torch.uint8)

                with torch.no_grad():
                    act, logp, val = policies[i].get_action(obs_t, floor_prob=floor_prob)

                actions[i] = int(act.item())
                ep_data[i]['obs'].append(obs_t)
                ep_data[i]['acts'].append(act)
                ep_data[i]['logp'].append(logp)
                ep_data[i]['vals'].append(val)

            ts = env.step(actions)

            for i in range(n_agents):
                reward = float(ts.reward[i])

                # Intrinsic reward shaping (clean_up 전용):
                # interact 행동 시 소량의 탐색 보상 부여
                if substrate_name == 'clean_up' and actions[i] == ACTION_INTERACT:
                    reward += 0.01  # 미세한 intrinsic reward

                ep_data[i]['rews'].append(reward)
                ep_reward += float(ts.reward[i])  # 외부 보상만 기록

            if ts.last():
                break

        env.close()
        train_rewards.append(ep_reward / n_agents)

        # PPO 업데이트
        for i in range(n_agents):
            if len(ep_data[i]['rews']) < 2:
                continue

            rewards = torch.tensor(ep_data[i]['rews'], device=device, dtype=torch.float32)
            old_logp = torch.stack(ep_data[i]['logp'])
            old_vals = torch.stack(ep_data[i]['vals'])

            returns = torch.zeros_like(rewards)
            G = 0
            for t_idx in reversed(range(len(rewards))):
                G = rewards[t_idx] + 0.99 * G
                returns[t_idx] = G

            advantages = returns - old_vals.detach()
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            obs_batch = torch.stack(ep_data[i]['obs'])
            act_batch = torch.stack(ep_data[i]['acts'])

            logits, values = policies[i](obs_batch)
            probs = torch.softmax(logits, dim=-1)

            # Floor 적용
            if floor_prob > 0:
                deficit = torch.clamp(floor_prob - probs[:, ACTION_INTERACT], min=0)
                boost = torch.zeros_like(probs)
                boost[:, ACTION_INTERACT] = deficit
                reduction = deficit.unsqueeze(1) / (N_ACTIONS - 1)
                for a in range(N_ACTIONS):
                    if a != ACTION_INTERACT:
                        probs[:, a] = torch.clamp(probs[:, a] - reduction.squeeze(1), min=1e-8)
                probs[:, ACTION_INTERACT] = probs[:, ACTION_INTERACT] + deficit
                probs = probs / probs.sum(dim=-1, keepdim=True)

            dist = Categorical(probs=probs)
            new_logp = dist.log_prob(act_batch)
            ratio = torch.exp(new_logp - old_logp.detach())

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (returns - values.squeeze(-1)).pow(2).mean()
            entropy = dist.entropy().mean()

            loss = actor_loss + critic_loss - 0.02 * entropy  # 엔트로피 계수 증가 (탐색 촉진)

            optimizers[i].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policies[i].parameters(), 0.5)
            optimizers[i].step()

        if verbose and (ep + 1) % 50 == 0:
            recent = np.mean(train_rewards[-50:])
            print(f'  Ep {ep+1}/{n_episodes}: reward={recent:.2f}')

    return policies, train_rewards


def evaluate_ippo(substrate_name, policies, floor_prob=0.0, n_eval=20):
    env_config = substrate.get_config(substrate_name)
    n_agents = len(policies)
    rewards = []

    for s in range(n_eval):
        env = substrate.build(substrate_name, roles=env_config.default_player_roles)
        ts = env.reset()
        ep_reward = 0

        for t in range(N_STEPS):
            actions = {}
            for i in range(n_agents):
                obs_rgb = ts.observation[i].get('RGB', np.zeros((88, 88, 3), dtype=np.uint8))
                obs_t = torch.tensor(obs_rgb, device=device, dtype=torch.uint8)
                with torch.no_grad():
                    act, _, _ = policies[i].get_action(obs_t, floor_prob=floor_prob)
                actions[i] = int(act.item())
            ts = env.step(actions)
            for i in range(n_agents):
                ep_reward += float(ts.reward[i])
            if ts.last():
                break

        env.close()
        rewards.append(ep_reward / n_agents)

    return {
        'mean': round(float(np.mean(rewards)), 2),
        'std': round(float(np.std(rewards)), 2),
        'rewards': [round(r, 2) for r in rewards],
    }


# ─── 실험 실행 ──────────────────────────────
if __name__ == '__main__':
    all_results = {}

    # === Heuristic 실험 ===
    print('\n' + '=' * 70)
    print('  HEURISTIC EXPERIMENTS')
    print('=' * 70)

    heuristic_policies = {
        'Selfish': selfish_policy,
        'Floor_0.3': make_commitment_floor_policy(0.3),
        'Floor_0.5': make_commitment_floor_policy(0.5),
        'Floor_0.7': make_commitment_floor_policy(0.7),
        'Floor_1.0': make_commitment_floor_policy(1.0),
        # clean_up 전용: 관찰 기반 정책
        'ObsAware_0.5': make_observation_aware_policy(0.5),
        'ObsAware_0.7': make_observation_aware_policy(0.7),
    }

    heuristic_results = {}
    for sub_name in SUBSTRATES:
        print(f'\n=== {sub_name} ===')
        heuristic_results[sub_name] = {}

        for pol_name, pol_fn in heuristic_policies.items():
            rewards = []
            for s in range(N_SEEDS):
                r = run_episode(sub_name, pol_fn, seed=s)
                rewards.append(r['mean_reward'])
                if (s + 1) % 10 == 0:
                    print(f'  [{pol_name}] seed {s+1}/{N_SEEDS}', end='', flush=True)

            mean_r = float(np.mean(rewards))
            std_r = float(np.std(rewards))
            heuristic_results[sub_name][pol_name] = {
                'mean': round(mean_r, 2),
                'std': round(std_r, 2),
            }
            print(f'  => {mean_r:.2f} +/- {std_r:.2f}')

    # === IPPO 실험 ===
    print('\n' + '=' * 70)
    print('  IPPO EXPERIMENTS')
    print('=' * 70)

    N_TRAIN_SEEDS = 3
    ippo_results = {}

    for sub_name in SUBSTRATES:
        print(f'\n=== IPPO on {sub_name} ===')
        ippo_results[sub_name] = {}

        for floor_prob in FLOOR_PROBS:
            print(f'\n  --- Floor prob = {floor_prob} ---')
            all_eval = []

            for seed in range(N_TRAIN_SEEDS):
                print(f'  Training seed {seed+1}/{N_TRAIN_SEEDS}...')
                policies, train_r = train_ippo(
                    sub_name, floor_prob=floor_prob,
                    seed=42 + seed, verbose=True,
                )
                eval_r = evaluate_ippo(sub_name, policies, floor_prob=floor_prob, n_eval=10)
                all_eval.append(eval_r['mean'])
                print(f'    Eval reward: {eval_r["mean"]:.2f}')

            ippo_results[sub_name][f'IPPO_floor_{floor_prob}'] = {
                'mean': round(float(np.mean(all_eval)), 2),
                'std': round(float(np.std(all_eval)), 2),
            }

    # === 결과 저장 ===
    final_results = {
        'experiment': 'EthicaAI Melting Pot Benchmark v2',
        'version': 'v2 — action space fix + obs-aware + reward shaping',
        'config': {
            'n_seeds_heuristic': N_SEEDS,
            'n_seeds_ippo': N_TRAIN_SEEDS,
            'n_steps': N_STEPS,
            'train_episodes': TRAIN_EPISODES,
            'train_horizon': TRAIN_HORIZON,
            'substrates': SUBSTRATES,
            'floor_probs': FLOOR_PROBS,
        },
        'heuristic_results': heuristic_results,
        'ippo_results': ippo_results,
    }

    with open('meltingpot_v2_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    # === 요약 출력 ===
    print('\n' + '=' * 70)
    print('  MELTING POT BENCHMARK v2 RESULTS')
    print('=' * 70)
    for sub in SUBSTRATES:
        print(f'\n  {sub}:')
        if sub in heuristic_results:
            for pol, stats in heuristic_results[sub].items():
                print(f'    {pol:20s}: {stats["mean"]:8.2f} +/- {stats["std"]:.2f}')
        if sub in ippo_results:
            for pol, stats in ippo_results[sub].items():
                print(f'    {pol:20s}: {stats["mean"]:8.2f} +/- {stats["std"]:.2f}')

    print(f'\nSaved: meltingpot_v2_results.json')
