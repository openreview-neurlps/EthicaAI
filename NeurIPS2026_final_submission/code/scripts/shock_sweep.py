"""
Phase B: Shock Dynamics Sweep
==============================
Tests Nash Trap robustness across different shock regimes:
  - p_shock: probability of shock each timestep
  - delta_shock: magnitude of resource loss
  - burst_len: consecutive shock bursts

Grid: p_shock x delta_shock x burst_len x seeds 20
Reports: survival, welfare, CVaR(return), collapse-time distribution
"""
import numpy as np
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
from cleanrl_mappo_pgg import (
    NonlinearPGGEnv, MLPActor, MLPCritic, compute_gae,
    ppo_update_actor, bootstrap_ci, GAMMA, GAE_LAMBDA, CLIP_EPS, HIDDEN_DIM
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "shock_sweep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_EPISODES = 200
N_EVAL = 30
N_SEEDS = 20
T_HORIZON = 50
N_AGENTS = 20
BYZ_FRAC = 0.30

P_SHOCKS = [0.05, 0.10, 0.20, 0.30]
DELTA_SHOCKS = [0.10, 0.20, 0.30]
BURST_LENS = [1, 3, 5]

if os.environ.get("ETHICAAI_FAST") == "1":
    print("  [FAST MODE]")
    N_SEEDS = 2
    N_EPISODES = 30
    N_EVAL = 10
    P_SHOCKS = [0.10, 0.30]
    DELTA_SHOCKS = [0.10, 0.30]
    BURST_LENS = [1, 5]


class ShockPGGEnv(NonlinearPGGEnv):
    """PGG with configurable shock dynamics."""
    def __init__(self, p_shock=0.1, delta_shock=0.2, burst_len=1, **kwargs):
        super().__init__(**kwargs)
        self.p_shock = p_shock
        self.delta_shock = delta_shock
        self.burst_len = burst_len
        self._burst_remaining = 0

    def step(self, actions):
        obs, rewards, terminated, truncated, info = super().step(actions)

        # Apply shock dynamics
        if self._burst_remaining > 0:
            self.R = max(0, self.R - self.delta_shock)
            self._burst_remaining -= 1
        elif np.random.random() < self.p_shock:
            self._burst_remaining = self.burst_len - 1
            self.R = max(0, self.R - self.delta_shock)

        # Update obs with new R
        obs = self._get_obs()

        # Check collapse
        if self.R <= 0:
            terminated = True
            info["survived"] = False

        return obs, rewards, terminated, truncated, info


def run_selfish_rl(seed, env_kwargs):
    """Run selfish IPPO agents in shock environment."""
    rng = np.random.RandomState(seed)
    env = ShockPGGEnv(byz_frac=BYZ_FRAC, **env_kwargs)
    n_honest = int(N_AGENTS * (1 - BYZ_FRAC))

    actors = [MLPActor(np.random.RandomState(seed * 100 + i)) for i in range(n_honest)]
    episode_data = []

    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        all_obs = [[] for _ in range(n_honest)]
        all_acts = [[] for _ in range(n_honest)]
        all_log_probs = [[] for _ in range(n_honest)]
        all_rewards = [[] for _ in range(n_honest)]
        all_values = [[] for _ in range(n_honest)]

        total_welfare = 0.0
        steps = 0
        survived = True
        collapse_time = T_HORIZON

        critic_dummy = MLPCritic(rng)  # simplified

        for t in range(T_HORIZON):
            lambdas = np.zeros(n_honest)
            for i in range(n_honest):
                mean, _ = actors[i].forward(obs)
                std = np.exp(actors[i].log_std[0])
                lam = float(np.clip(mean[0] + rng.randn() * std, 0.01, 0.99))
                lambdas[i] = lam
                all_obs[i].append(obs.copy())
                all_acts[i].append(lam)
                all_log_probs[i].append(actors[i].log_prob(obs, lam))
                all_values[i].append(float(critic_dummy.forward(obs)))

            obs, rewards, terminated, truncated, info = env.step(lambdas)
            for i in range(n_honest):
                r = float(rewards[i]) if hasattr(rewards, '__len__') else float(rewards)
                all_rewards[i].append(r)

            total_welfare += float(np.mean(rewards)) if hasattr(rewards, '__len__') else float(rewards)
            steps += 1
            if terminated:
                survived = info.get("survived", False)
                collapse_time = t + 1
                break

        for i in range(n_honest):
            if len(all_rewards[i]) < 2:
                continue
            advantages, returns = compute_gae(all_rewards[i], all_values[i])
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / advantages.std()
            ppo_update_actor(actors[i], all_obs[i], all_acts[i], all_log_probs[i], advantages)

        episode_data.append({
            "welfare": total_welfare / max(steps, 1),
            "survived": survived,
            "collapse_time": collapse_time,
        })

    eval_eps = episode_data[-N_EVAL:]
    returns = [e["welfare"] for e in eval_eps]
    sorted_returns = sorted(returns)
    n_tail = max(1, int(len(sorted_returns) * 0.1))
    cvar_10 = float(np.mean(sorted_returns[:n_tail]))

    return {
        "survival": float(np.mean([e["survived"] for e in eval_eps]) * 100),
        "welfare": float(np.mean(returns)),
        "cvar_10": cvar_10,
        "mean_collapse_time": float(np.mean([e["collapse_time"] for e in eval_eps])),
    }


def main():
    print("=" * 70)
    print("  Phase B: Shock Dynamics Sweep")
    print(f"  Seeds={N_SEEDS}, Episodes={N_EPISODES}")
    print("=" * 70)

    t0 = time.time()
    results = {}
    total = len(P_SHOCKS) * len(DELTA_SHOCKS) * len(BURST_LENS)
    done = 0

    for p_s in P_SHOCKS:
        for d_s in DELTA_SHOCKS:
            for b_l in BURST_LENS:
                key = f"p{p_s}_d{d_s}_b{b_l}"
                done += 1
                print(f"\n  [{done}/{total}] p_shock={p_s}, delta={d_s}, burst={b_l}: "
                      f"{N_SEEDS} seeds...")

                env_kwargs = {"p_shock": p_s, "delta_shock": d_s, "burst_len": b_l}
                seed_results = []
                for s in range(N_SEEDS):
                    r = run_selfish_rl(s, env_kwargs)
                    seed_results.append(r)

                surv = [r["survival"] for r in seed_results]
                welf = [r["welfare"] for r in seed_results]
                cvar = [r["cvar_10"] for r in seed_results]

                results[key] = {
                    "p_shock": p_s, "delta_shock": d_s, "burst_len": b_l,
                    "survival_mean": float(np.mean(surv)),
                    "survival_std": float(np.std(surv)),
                    "welfare_mean": float(np.mean(welf)),
                    "cvar_10_mean": float(np.mean(cvar)),
                    "cvar_10_std": float(np.std(cvar)),
                }
                print(f"    -> surv={np.mean(surv):.0f}%, W={np.mean(welf):.1f}, "
                      f"CVaR10={np.mean(cvar):.2f}")

    out_path = OUTPUT_DIR / "shock_sweep_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  COMPLETE in {elapsed:.0f}s. Saved: {out_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
