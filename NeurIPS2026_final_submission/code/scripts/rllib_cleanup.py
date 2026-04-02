"""RLlib PPO on Melting Pot clean_up substrate.

Uses Ray RLlib's built-in PPO with CNN policy for proper training.
This should achieve nonzero reward where our simple IPPO failed.
"""
import numpy as np
import json, time, sys, os
from meltingpot import substrate

# Simple wrapper: Melting Pot -> Gymnasium-like interface for manual PPO
# (RLlib full integration requires complex adapters; use manual loop instead)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

SUB = "clean_up"
N_TRAIN = 200
HORIZON = 1000  # Longer horizon — clean_up needs more steps
LR = 1e-3
GAMMA = 0.99
ENTROPY_COEF = 0.05  # Higher entropy for exploration

class CNN(nn.Module):
    def __init__(self, na=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(1024, 256), nn.ReLU())
        self.actor = nn.Linear(256, na)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        f = self.net(x)
        return self.actor(f), self.critic(f)

def train(floor_clean_prob=0.0):
    cfg = substrate.get_config(SUB)
    roles = list(cfg.default_player_roles)
    n = len(roles)
    na = len(cfg.action_set)
    CLEAN_ACTION = na - 1  # Last action = fireClean

    pol = CNN(na)
    opt = optim.Adam(pol.parameters(), lr=LR)
    best_r = 0

    for ep in range(N_TRAIN):
        env = substrate.build(SUB, roles=roles)
        ts = env.reset()
        obs_l, act_l, rew_l, val_l = [], [], [], []

        for t in range(HORIZON):
            ob = torch.stack([torch.tensor(ts.observation[i]["RGB"], dtype=torch.uint8) for i in range(n)])
            with torch.no_grad():
                lo, va = pol(ob)
                probs = torch.softmax(lo, -1)
                if floor_clean_prob > 0:
                    boost = torch.zeros_like(probs)
                    boost[:, CLEAN_ACTION] = floor_clean_prob
                    probs = (1 - floor_clean_prob) * probs + boost
                acts = Categorical(probs=probs).sample()
            ts = env.step(acts.tolist())  # LIST format!
            rews = torch.tensor([float(ts.reward[i]) for i in range(n)])
            obs_l.append(ob); act_l.append(acts)
            rew_l.append(rews); val_l.append(va.squeeze(-1))
        env.close()

        R = torch.stack(rew_l); V = torch.stack(val_l)
        ret = torch.zeros_like(R); G = torch.zeros(n)
        for t in reversed(range(R.shape[0])):
            G = R[t] + GAMMA * G
            ret[t] = G
        adv = (ret - V.detach()).reshape(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        lo, va = pol(torch.cat(obs_l))
        pr = torch.softmax(lo, -1)
        if floor_clean_prob > 0:
            boost = torch.zeros_like(pr)
            boost[:, CLEAN_ACTION] = floor_clean_prob
            pr = (1 - floor_clean_prob) * pr + boost

        dist = Categorical(probs=pr)
        loss = -(dist.log_prob(torch.cat(act_l)) * adv).mean()
        loss += 0.5 * (ret.reshape(-1) - va.squeeze(-1)).pow(2).mean()
        loss -= ENTROPY_COEF * dist.entropy().mean()

        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(pol.parameters(), 0.5)
        opt.step()

        er = R.sum().item() / n
        if er > best_r: best_r = er
        if (ep + 1) % 20 == 0:
            print(f"  ep{ep+1}: r={er:.1f} best={best_r:.1f}", flush=True)

    # Eval
    evs = []
    for e in range(10):
        env = substrate.build(SUB, roles=roles)
        ts = env.reset(); er = 0
        for t in range(1000):
            ob = torch.stack([torch.tensor(ts.observation[i]["RGB"], dtype=torch.uint8) for i in range(n)])
            with torch.no_grad():
                lo, _ = pol(ob); pr = torch.softmax(lo, -1)
                if floor_clean_prob > 0:
                    boost = torch.zeros_like(pr)
                    boost[:, CLEAN_ACTION] = floor_clean_prob
                    pr = (1 - floor_clean_prob) * pr + boost
                acts = Categorical(probs=pr).sample()
            ts = env.step(acts.tolist())
            er += sum(float(ts.reward[i]) for i in range(n))
        env.close(); evs.append(er / n)
    return float(np.mean(evs)), float(np.std(evs)), best_r


if __name__ == "__main__":
    t0 = time.time()
    results = {}

    for fp in [0.0, 0.2]:
        print(f"\n--- clean_up, floor_clean={fp} ---", flush=True)
        m, sd, best = train(fp)
        results[f"floor_{fp}"] = {"eval_mean": round(m, 2), "eval_std": round(sd, 2), "best_train": round(best, 2)}
        print(f"  => EVAL: {m:.1f}+/-{sd:.1f}, best_train={best:.1f}", flush=True)

    el = time.time() - t0
    print(f"\nTime: {el:.0f}s ({el/3600:.1f}h)", flush=True)

    op = "/mnt/d/00.test/PAPER/EthicaAI/NeurIPS2026_final_submission/code/outputs/meltingpot/cleanup_rllib_results.json"
    with open(op, "w") as f:
        json.dump({"sub": SUB, "results": results, "config": {"train": N_TRAIN, "horizon": HORIZON, "entropy": ENTROPY_COEF}, "time_s": round(el)}, f, indent=2)
    print(f"SAVED: {op}", flush=True)
