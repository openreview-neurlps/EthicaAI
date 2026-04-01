"""Overnight clean_up CNN-PPO training (1000 episodes, WSL2)."""
import torch, torch.nn as nn, torch.optim as optim
from torch.distributions import Categorical
import numpy as np, json, time, sys
from meltingpot import substrate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

SUB = "clean_up"
N_TRAIN = 1000; N_EVAL = 10; N_SEEDS = 2; HORIZON = 300
LR = 1e-3; GAMMA = 0.99; CLEAN_ACTION = 8

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,16,3,2,1), nn.ReLU(),
            nn.Conv2d(16,32,3,2,1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4)), nn.Flatten(),
            nn.Linear(512,128), nn.ReLU())
        self.actor = nn.Linear(128,9)
        self.critic = nn.Linear(128,1)
    def forward(self, x):
        if x.dim()==3: x=x.unsqueeze(0)
        x = x.permute(0,3,1,2).float()/255.0
        f = self.net(x)
        return self.actor(f), self.critic(f)

def run(floor_prob, seed):
    cfg = substrate.get_config(SUB)
    roles = list(cfg.default_player_roles)
    n = len(roles)
    pol = CNN().to(device)
    opt = optim.Adam(pol.parameters(), lr=LR)
    best_r = 0
    for ep in range(N_TRAIN):
        env = substrate.build(SUB, roles=roles)
        ts = env.reset()
        obs_l, act_l, rew_l, val_l = [],[],[],[]
        for t in range(HORIZON):
            ob = torch.stack([torch.tensor(ts.observation[i]["RGB"], device=device, dtype=torch.uint8) for i in range(n)])
            with torch.no_grad():
                lo, va = pol(ob)
                pr = torch.softmax(lo,-1)
                if floor_prob > 0:
                    b = torch.zeros_like(pr); b[:,CLEAN_ACTION]=floor_prob
                    pr = (1-floor_prob)*pr + b
                acts = Categorical(probs=pr).sample()
            ts = env.step(acts.cpu().tolist())
            rews = torch.tensor([float(ts.reward[i]) for i in range(n)], device=device)
            obs_l.append(ob); act_l.append(acts); rew_l.append(rews); val_l.append(va.squeeze(-1))
        env.close()
        R = torch.stack(rew_l); V = torch.stack(val_l)
        ret = torch.zeros_like(R); G = torch.zeros(n, device=device)
        for t in reversed(range(R.shape[0])): G=R[t]+GAMMA*G; ret[t]=G
        adv = (ret-V.detach()).reshape(-1)
        adv = (adv-adv.mean())/(adv.std()+1e-8)
        lo,va = pol(torch.cat(obs_l))
        pr = torch.softmax(lo,-1)
        if floor_prob>0:
            b=torch.zeros_like(pr); b[:,CLEAN_ACTION]=floor_prob
            pr=(1-floor_prob)*pr+b
        loss = -(Categorical(probs=pr).log_prob(torch.cat(act_l))*adv).mean() + 0.5*(ret.reshape(-1)-va.squeeze(-1)).pow(2).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(pol.parameters(), 0.5)
        opt.step()
        er = R.sum().item()/n
        if er > best_r: best_r = er
        if (ep+1)%100==0:
            print(f"  ep{ep+1}: r={er:.1f} best={best_r:.1f}", flush=True)
    evs = []
    for e in range(N_EVAL):
        env = substrate.build(SUB, roles=roles)
        ts = env.reset(); er=0
        for t in range(500):
            ob = torch.stack([torch.tensor(ts.observation[i]["RGB"], device=device, dtype=torch.uint8) for i in range(n)])
            with torch.no_grad():
                lo,_ = pol(ob); pr=torch.softmax(lo,-1)
                if floor_prob>0:
                    b=torch.zeros_like(pr); b[:,CLEAN_ACTION]=floor_prob
                    pr=(1-floor_prob)*pr+b
                acts = Categorical(probs=pr).sample()
            ts = env.step(acts.cpu().tolist())
            er += sum(float(ts.reward[i]) for i in range(n))
        env.close(); evs.append(er/n)
    return float(np.mean(evs)), float(np.std(evs))

results = {}; t0 = time.time()
for fp in [0.0, 0.2, 0.4]:
    print(f"\n--- floor={fp} ---", flush=True)
    ss = []
    for s in range(N_SEEDS):
        print(f"  Seed {s+1}/{N_SEEDS}", flush=True)
        m,sd = run(fp, 42+s)
        ss.append(m)
        print(f"    => {m:.1f}+/-{sd:.1f}", flush=True)
    results[f"floor_{fp}"] = {"mean":round(float(np.mean(ss)),2), "std":round(float(np.std(ss)),2), "seeds":[round(r,2) for r in ss]}

el = time.time()-t0
print(f"\nTime: {el:.0f}s ({el/3600:.1f}h)", flush=True)
op = "/mnt/d/00.test/PAPER/EthicaAI/NeurIPS2026_final_submission/code/outputs/meltingpot/cleanup_long_results.json"
with open(op,"w") as f:
    json.dump({"sub":SUB,"results":results,"config":{"train":N_TRAIN,"eval":N_EVAL,"seeds":N_SEEDS,"horizon":HORIZON},"time_s":round(el)},f,indent=2)
print(f"SAVED: {op}", flush=True)
