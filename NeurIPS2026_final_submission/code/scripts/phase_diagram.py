"""
Phase Diagram: phi1 x beta -> Survival Heatmap
==========================================
Sweeps commitment floor phi1 in [0, 1] x Byzantine fraction beta in [0, 0.5]
to produce a 2D heatmap showing the "commitment phase transition".

This is the visual centerpiece of the paper -- directly maps Theorem 1.
"""
import numpy as np
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
from cleanrl_mappo_pgg import (
    NonlinearPGGEnv, MLPActor, MLPCritic, compute_gae,
    ppo_update_actor, GAMMA, GAE_LAMBDA, CLIP_EPS, HIDDEN_DIM
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase_diagram"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Grid resolution
PHI1_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
BETA_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

N_EPISODES = 200
N_EVAL = 30
N_SEEDS = 20
T_HORIZON = 50
N_AGENTS = 20

if os.environ.get("ETHICAAI_FAST") == "1":
    print("  [FAST MODE]")
    PHI1_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]
    BETA_GRID = [0.0, 0.15, 0.30, 0.45]
    N_SEEDS = 2
    N_EPISODES = 50
    N_EVAL = 10


def run_single(seed, phi1, beta):
    """Run IPPO with commitment floor phi1 and Byzantine fraction beta."""
    rng = np.random.RandomState(seed)
    env = NonlinearPGGEnv(byz_frac=beta)
    n_honest = env.n_honest
    
    actors = [MLPActor(np.random.RandomState(seed * 100 + i)) for i in range(n_honest)]
    
    ep_survived = []
    
    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        survived = True
        
        for t in range(T_HORIZON):
            lambdas = np.zeros(n_honest)
            for i in range(n_honest):
                mean, _ = actors[i].forward(obs)
                std = np.exp(actors[i].log_std[0])
                learned = float(np.clip(mean[0] + rng.randn() * std, 0.01, 0.99))
                lambdas[i] = max(learned, phi1)
            
            obs, rewards, terminated, truncated, info = env.step(lambdas)
            if terminated:
                survived = info.get("survived", False)
                break
        
        ep_survived.append(float(survived))
    
    return float(np.mean(ep_survived[-N_EVAL:]) * 100)


def main():
    print("=" * 60)
    print("  Phase Diagram: phi1 x beta -> Survival")
    print("  phi1 grid: %d points, beta grid: %d points" % (len(PHI1_GRID), len(BETA_GRID)))
    print("  Seeds=%d, Episodes=%d" % (N_SEEDS, N_EPISODES))
    total = len(PHI1_GRID) * len(BETA_GRID) * N_SEEDS
    print("  Total runs: %d" % total)
    print("=" * 60)
    
    t0 = time.time()
    heatmap = {}
    
    for pi, phi1 in enumerate(PHI1_GRID):
        heatmap[str(phi1)] = {}
        for bi, beta in enumerate(BETA_GRID):
            survivals = []
            for s in range(N_SEEDS):
                surv = run_single(s, phi1, beta)
                survivals.append(surv)
            
            mean_surv = float(np.mean(survivals))
            heatmap[str(phi1)][str(beta)] = mean_surv
            
            done = (pi * len(BETA_GRID) + bi + 1)
            pct = done / (len(PHI1_GRID) * len(BETA_GRID)) * 100
            print("  phi1=%.2f beta=%.2f: surv=%5.1f%%  [%d%%]" % (phi1, beta, mean_surv, pct))
    
    # Build matrix
    matrix = []
    for phi1 in PHI1_GRID:
        row = []
        for beta in BETA_GRID:
            row.append(heatmap[str(phi1)][str(beta)])
        matrix.append(row)
    
    output = {
        "phi1_grid": PHI1_GRID,
        "beta_grid": BETA_GRID,
        "survival_matrix": matrix,
        "n_seeds": N_SEEDS,
        "n_episodes": N_EPISODES,
    }
    
    out_path = OUTPUT_DIR / "phase_diagram.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # ASCII heatmap
    print("\n" + "=" * 60)
    print("  PHASE DIAGRAM (survival %%)")
    header = "  beta->  "
    for beta in BETA_GRID:
        header += " %.2f" % beta
    print(header)
    for pi, phi1 in enumerate(PHI1_GRID):
        line = "  phi1=%.1f" % phi1
        for bi in range(len(BETA_GRID)):
            v = matrix[pi][bi]
            line += " %5.0f" % v
        print(line)
    
    elapsed = time.time() - t0
    print("\n  Saved: %s" % out_path)
    print("  DONE in %ds" % elapsed)
    print("=" * 60)


if __name__ == "__main__":
    main()
