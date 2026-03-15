"""
Phase Diagram WITH Learning: phi1 x beta -> Survival Heatmap
=============================================================
Companion to phase_diagram.py (floor-only). This version TRAINS the IPPO
agents for N_EPISODES before evaluating survival, to confirm that:

  1. Learned policies also exhibit the same phase boundary as the
     floor-only structural diagram.
  2. Learning DOES NOT escape the Nash Trap without the floor (phi1=0).
  3. Learning + floor yields the same survival landscape as floor alone.

Together, the two diagrams support Theorem 1: phi1* is a necessary
condition independent of the learning algorithm.

Design choices:
  - Coarse grid (6 phi1 x 6 beta) for fast completion
  - PPO update is applied each episode (same as cleanrl_mappo_pgg.py)
  - Floor-overridden timesteps are excluded from PPO updates
    (same design as phi1_with_learning.py) for policy gradient correctness
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
OUTPUT_DIR = PROJECT_ROOT / os.environ.get("ETHICAAI_OUTDIR", "outputs") / "phase_diagram_learned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Grid (coarse for tractability, fine enough to show boundary) ---
PHI1_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BETA_GRID = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]

N_EPISODES = 200
N_EVAL = 30
N_SEEDS = 5       # Coarse sweep: 5 seeds; representative points get 20
T_HORIZON = 50
N_AGENTS = 20

if os.environ.get("ETHICAAI_FAST") == "1":
    print("  [FAST MODE]")
    PHI1_GRID = [0.0, 0.5, 1.0]
    BETA_GRID = [0.0, 0.30]
    N_SEEDS = 2
    N_EPISODES = 50
    N_EVAL = 10


def run_single_with_learning(seed, phi1, beta):
    """Run IPPO WITH learning and commitment floor phi1."""
    rng = np.random.RandomState(seed)
    env = NonlinearPGGEnv(byz_frac=beta)
    n_honest = env.n_honest

    actors = [MLPActor(np.random.RandomState(seed * 100 + i)) for i in range(n_honest)]

    ep_survived = []
    total_floor_count = 0
    total_step_count = 0
    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        survived = True

        # Per-agent trajectory buffers
        all_obs = [[] for _ in range(n_honest)]
        all_acts = [[] for _ in range(n_honest)]
        all_log_probs = [[] for _ in range(n_honest)]
        all_rewards = [[] for _ in range(n_honest)]
        all_values = [[] for _ in range(n_honest)]
        floor_active = [[] for _ in range(n_honest)]

        for t in range(T_HORIZON):
            lambdas = np.zeros(n_honest)
            for i in range(n_honest):
                mean, _ = actors[i].forward(obs)
                std = np.exp(actors[i].log_std[0])
                learned = float(np.clip(mean[0] + rng.randn() * std, 0.01, 0.99))

                is_floor = learned < phi1
                effective = max(learned, phi1)
                lambdas[i] = effective
                floor_active[i].append(is_floor)

                # Store trajectory with LEARNED action for correct log_prob
                all_obs[i].append(obs.copy())
                all_acts[i].append(learned)
                all_log_probs[i].append(actors[i].log_prob(obs, learned))
                all_values[i].append(0.0)  # No critic for simplicity

            obs, rewards, terminated, truncated, info = env.step(lambdas)

            for i in range(n_honest):
                all_rewards[i].append(float(rewards[i]) if hasattr(rewards, '__len__') else float(rewards))

            if terminated:
                survived = info.get("survived", False)
                break

        ep_survived.append(float(survived))

        # Track floor activation across all agents/timesteps this episode
        for i in range(n_honest):
            total_floor_count += sum(floor_active[i])
            total_step_count += len(floor_active[i])

        # PPO update: skip floor-overridden timesteps
        for i in range(n_honest):
            mask = [not fa for fa in floor_active[i]]
            if sum(mask) < 2:
                continue

            obs_f = [o for o, m in zip(all_obs[i], mask) if m]
            act_f = [a for a, m in zip(all_acts[i], mask) if m]
            lp_f = [lp for lp, m in zip(all_log_probs[i], mask) if m]
            rew_f = [r for r, m in zip(all_rewards[i], mask) if m]
            val_f = [v for v, m in zip(all_values[i], mask) if m]

            advantages, returns = compute_gae(rew_f, val_f)

            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / advantages.std()

            ppo_update_actor(actors[i], obs_f, act_f, lp_f, advantages)

    floor_rate = total_floor_count / max(total_step_count, 1)
    return float(np.mean(ep_survived[-N_EVAL:]) * 100), floor_rate


def main():
    print("=" * 60)
    print("  Phase Diagram WITH Learning: phi1 x beta -> Survival")
    print("  phi1 grid: %d points, beta grid: %d points" % (len(PHI1_GRID), len(BETA_GRID)))
    print("  Seeds=%d, Episodes=%d (with PPO learning)" % (N_SEEDS, N_EPISODES))
    total = len(PHI1_GRID) * len(BETA_GRID) * N_SEEDS
    print("  Total runs: %d" % total)
    print("=" * 60)

    t0 = time.time()
    heatmap = {}

    for pi, phi1 in enumerate(PHI1_GRID):
        heatmap[str(phi1)] = {}
        floor_rates = {}
        for bi, beta in enumerate(BETA_GRID):
            survivals = []
            seed_floor_rates = []
            for s in range(N_SEEDS):
                surv, f_rate = run_single_with_learning(s, phi1, beta)
                survivals.append(surv)
                seed_floor_rates.append(f_rate)

            mean_surv = float(np.mean(survivals))
            mean_floor = float(np.mean(seed_floor_rates))
            heatmap[str(phi1)][str(beta)] = mean_surv
            heatmap[str(phi1)].setdefault("surv_per_seed", {})[str(beta)] = survivals
            floor_rates[str(beta)] = {
                "mean": mean_floor,
                "per_seed": seed_floor_rates,
            }

            done = (pi * len(BETA_GRID) + bi + 1)
            pct = done / (len(PHI1_GRID) * len(BETA_GRID)) * 100
            print("  phi1=%.2f beta=%.2f: surv=%5.1f%% floor=%.1f%%  [%d%%]" % (
                phi1, beta, mean_surv, mean_floor * 100, pct))

        heatmap[str(phi1)]["floor_rates"] = floor_rates

    # Build matrices
    matrix = []
    floor_matrix = []
    survival_per_seed_matrix = []
    floor_per_seed_matrix = []
    for phi1 in PHI1_GRID:
        row = []
        f_row = []
        surv_ps_row = []
        floor_ps_row = []
        for beta in BETA_GRID:
            row.append(heatmap[str(phi1)][str(beta)])
            f_row.append(heatmap[str(phi1)]["floor_rates"][str(beta)]["mean"])
            surv_ps_row.append(heatmap[str(phi1)]["surv_per_seed"][str(beta)])
            floor_ps_row.append(heatmap[str(phi1)]["floor_rates"][str(beta)]["per_seed"])
        matrix.append(row)
        floor_matrix.append(f_row)
        survival_per_seed_matrix.append(surv_ps_row)
        floor_per_seed_matrix.append(floor_ps_row)

    output = {
        "phi1_grid": PHI1_GRID,
        "beta_grid": BETA_GRID,
        "survival_matrix": matrix,
        "floor_activation_matrix": floor_matrix,
        "survival_per_seed": survival_per_seed_matrix,
        "floor_activation_per_seed": floor_per_seed_matrix,
        "run_meta": {
            "timestamp": time.time(),
            "git_sha": os.popen("git rev-parse HEAD").read().strip(),
            "mode": "FAST" if os.environ.get("ETHICAAI_FAST") == "1" else "FULL",
        },
        "n_seeds": N_SEEDS,
        "n_episodes": N_EPISODES,
        "n_eval": N_EVAL,
        "t_horizon": T_HORIZON,
        "learning": True,
        "floor_skip_update": True,
        "update_rule": "PPO (clipped Gaussian, numerical gradient)",
        "description": "Phase diagram WITH IPPO learning. Floor-overridden timesteps excluded from updates.",
    }

    out_path = OUTPUT_DIR / "phase_diagram_learned.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    # ASCII heatmap
    print("\n" + "=" * 60)
    print("  PHASE DIAGRAM WITH LEARNING (survival %%)")
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
