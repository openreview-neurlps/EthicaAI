#!/bin/bash
# Run clean_up CNN-PPO with 3 seeds in parallel on Linux server
# Usage: bash server_cleanup_3seeds.sh
# Requires: Python 3.10, pip install dmlab2d==1.0.0 dm-meltingpot torch "chex<0.1.87"

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/../outputs/meltingpot"
mkdir -p "$OUTPUT_DIR"

echo "=== Melting Pot clean_up: 3 seeds × 500ep × 4 floors ==="
echo "Starting 3 parallel seeds..."

for SEED in 42 43 44; do
    python3 -c "
import torch, torch.nn as nn, torch.optim as optim
from torch.distributions import Categorical
import numpy as np, json, time, sys
from meltingpot import substrate

SUB = 'clean_up'
N_TRAIN = 500; HORIZON = 1000; LR = 1e-3; GAMMA = 0.99; ENT = 0.05; SEED = $SEED

class CNN(nn.Module):
    def __init__(self, na=9):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3,32,3,2,1), nn.ReLU(), nn.Conv2d(32,64,3,2,1), nn.ReLU(), nn.Conv2d(64,64,3,2,1), nn.ReLU(), nn.AdaptiveAvgPool2d((4,4)), nn.Flatten(), nn.Linear(1024,256), nn.ReLU())
        self.actor = nn.Linear(256,na); self.critic = nn.Linear(256,1)
    def forward(self, x):
        if x.dim()==3: x=x.unsqueeze(0)
        x = x.permute(0,3,1,2).float()/255.0
        f = self.net(x); return self.actor(f), self.critic(f)

cfg = substrate.get_config(SUB)
roles = list(cfg.default_player_roles)
n = len(roles); na = len(cfg.action_set); CLEAN = na - 1

results = {}
t0 = time.time()
for fp in [0.0, 0.1, 0.2, 0.3]:
    pol = CNN(na); opt = optim.Adam(pol.parameters(), lr=LR); best_r = 0
    for ep in range(N_TRAIN):
        env = substrate.build(SUB, roles=roles); ts = env.reset()
        obs_l, act_l, rew_l, val_l = [],[],[],[]
        for t in range(HORIZON):
            ob = torch.stack([torch.tensor(ts.observation[i]['RGB'], dtype=torch.uint8) for i in range(n)])
            with torch.no_grad():
                lo, va = pol(ob); pr = torch.softmax(lo,-1)
                if fp > 0:
                    boost = torch.zeros_like(pr); boost[:,CLEAN] = fp
                    pr = (1-fp)*pr + boost
                acts = Categorical(probs=pr).sample()
            ts = env.step(acts.tolist())
            rews = torch.tensor([float(ts.reward[i]) for i in range(n)])
            obs_l.append(ob); act_l.append(acts); rew_l.append(rews); val_l.append(va.squeeze(-1))
        env.close()
        R = torch.stack(rew_l); V = torch.stack(val_l)
        ret = torch.zeros_like(R); G = torch.zeros(n)
        for t in reversed(range(R.shape[0])): G=R[t]+GAMMA*G; ret[t]=G
        adv = (ret-V.detach()).reshape(-1); adv = (adv-adv.mean())/(adv.std()+1e-8)
        lo,va = pol(torch.cat(obs_l)); pr = torch.softmax(lo,-1)
        if fp>0: boost=torch.zeros_like(pr); boost[:,CLEAN]=fp; pr=(1-fp)*pr+boost
        dist = Categorical(probs=pr)
        loss = -(dist.log_prob(torch.cat(act_l))*adv).mean() + 0.5*(ret.reshape(-1)-va.squeeze(-1)).pow(2).mean() - ENT*dist.entropy().mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(pol.parameters(), 0.5); opt.step()
        er = R.sum().item()/n
        if er > best_r: best_r = er
        if (ep+1)%100==0: print(f'  seed=$SEED floor={fp} ep{ep+1}: r={er:.1f} best={best_r:.1f}', flush=True)
    # Eval
    evs = []
    for e in range(10):
        env = substrate.build(SUB, roles=roles); ts = env.reset(); er=0
        for t in range(1000):
            ob = torch.stack([torch.tensor(ts.observation[i]['RGB'], dtype=torch.uint8) for i in range(n)])
            with torch.no_grad():
                lo,_ = pol(ob); pr=torch.softmax(lo,-1)
                if fp>0: boost=torch.zeros_like(pr); boost[:,CLEAN]=fp; pr=(1-fp)*pr+boost
                acts = Categorical(probs=pr).sample()
            ts = env.step(acts.tolist()); er += sum(float(ts.reward[i]) for i in range(n))
        env.close(); evs.append(er/n)
    results[f'floor_{fp}'] = {'eval_mean': round(float(np.mean(evs)),2), 'eval_std': round(float(np.std(evs)),2), 'best_train': round(best_r,2)}
    print(f'  seed=$SEED floor={fp}: eval={np.mean(evs):.1f}+/-{np.std(evs):.1f}', flush=True)

with open('$OUTPUT_DIR/cleanup_seed${SEED}.json', 'w') as f:
    json.dump({'seed': $SEED, 'results': results, 'time_s': round(time.time()-t0)}, f, indent=2)
print(f'SAVED: cleanup_seed${SEED}.json ({time.time()-t0:.0f}s)')
" > "$OUTPUT_DIR/cleanup_seed${SEED}.log" 2>&1 &
    echo "  Seed $SEED started (PID: $!)"
done

echo ""
echo "All 3 seeds running in parallel."
echo "Monitor: tail -f $OUTPUT_DIR/cleanup_seed*.log"
echo "Results: $OUTPUT_DIR/cleanup_seed*.json"
wait
echo "ALL DONE"
