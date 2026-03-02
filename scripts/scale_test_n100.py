"""
Scale Test: N=100 Non-linear PGG Nash Trap
==========================================
Reproduces Table 5 from the paper: Nash Trap persists at N=100.

Tests:
  - Selfish RL (Byz=0%): should show lambda ~0.5, moderate survival
  - Selfish RL (Byz=30%): should show lambda ~0.5, low survival
  - Situational commitment (Byz=30%): high survival
  - Unconditional commitment (Byz=30%): moderate-high survival

Environment: Non-linear PGG, N=100, T=50
Dependencies: NumPy only.
"""

import numpy as np
import json
import os
import time

# ============================================================
# Config: N=100 (paper Table 5)
# ============================================================
N_AGENTS = 100
ENDOWMENT = 20.0
MULTIPLIER = 1.6
T_HORIZON = 50
R_CRIT = 0.15
R_RECOV = 0.25
SHOCK_PROB = 0.05
SHOCK_MAG = 0.15
STATE_DIM = 4
GAMMA = 0.99
N_EPISODES = 50
N_SEEDS = 10


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def reset_env():
    return {"R": 0.5, "mean_c": 0.5, "lam_prev": np.full(N_AGENTS, 0.5)}


def get_obs(env, n_agents):
    R = env["R"]
    mc = env["mean_c"]
    crisis = 1.0 if R < R_CRIT else 0.0
    obs = np.column_stack([
        np.full(n_agents, R),
        np.full(n_agents, mc),
        env["lam_prev"][:n_agents],
        np.full(n_agents, crisis)
    ])
    return obs


def step_env(env, lambdas, rng):
    contribs = ENDOWMENT * lambdas
    public_good = MULTIPLIER * contribs.sum() / N_AGENTS
    rewards = (ENDOWMENT - contribs) + public_good

    coop = contribs.mean() / ENDOWMENT
    R = env["R"]
    f_R = 0.01 if R < R_CRIT else (0.03 if R < R_RECOV else 0.10)

    R_new = R + f_R * (coop - 0.4)
    if rng.random() < SHOCK_PROB:
        R_new -= SHOCK_MAG
    R_new = float(np.clip(R_new, 0.0, 1.0))
    done = R_new <= 0.001

    env["R"] = R_new
    env["mean_c"] = float(coop)
    env["lam_prev"] = lambdas.copy()
    return rewards, done


def lambda_situational(R, theta_base=0.7, theta_crit=1.0, theta_low=0.9):
    """Situational commitment: adjusts lambda based on R."""
    if R < R_CRIT:
        return theta_crit
    elif R < R_RECOV:
        return theta_low
    else:
        return theta_base


# ============================================================
# Policy: Linear REINFORCE (per-agent)
# ============================================================
class LinearAgent:
    def __init__(self, rng):
        self.w = rng.randn(STATE_DIM) * 0.01
        self.b = 0.0
        self.lr = 0.01

    def act(self, obs, rng, noise=0.1):
        mu = sigmoid(float(obs @ self.w + self.b))
        return float(np.clip(mu + rng.randn() * noise, 0.01, 0.99))

    def update(self, obs_list, act_list, returns):
        for obs, a, G in zip(obs_list, act_list, returns):
            mu = sigmoid(float(obs @ self.w + self.b))
            grad = (a - mu) * obs
            self.w += self.lr * G * grad
            self.b += self.lr * G * (a - mu)


# ============================================================
# Experiments
# ============================================================
def run_selfish_rl(byz_frac=0.0, n_seeds=N_SEEDS):
    """Selfish RL with Byzantine agents."""
    n_byz = int(N_AGENTS * byz_frac)
    n_honest = N_AGENTS - n_byz
    all_data = []

    for seed in range(n_seeds):
        rng = np.random.RandomState(42 + seed * 13)
        agents = [LinearAgent(np.random.RandomState(seed * 100 + i)) for i in range(n_honest)]

        ep = {"welfare": [], "mean_lam": [], "survival": []}

        for e in range(N_EPISODES):
            env = reset_env()
            a_obs = [[] for _ in range(n_honest)]
            a_act = [[] for _ in range(n_honest)]
            a_rew = [[] for _ in range(n_honest)]

            tw, tl, steps, surv = 0, 0, 0, True

            for t in range(T_HORIZON):
                lams = np.zeros(N_AGENTS)
                for i in range(n_honest):
                    obs_i = get_obs(env, N_AGENTS)[n_byz + i]
                    act = agents[i].act(obs_i, rng)
                    lams[n_byz + i] = act
                    a_obs[i].append(obs_i)
                    a_act[i].append(act)

                rewards, done = step_env(env, lams, rng)
                for i in range(n_honest):
                    a_rew[i].append(rewards[n_byz + i])

                tw += rewards.mean()
                tl += lams[n_byz:].mean()
                steps += 1
                if done:
                    surv = False
                    break

            # REINFORCE update
            for i in range(n_honest):
                T_len = len(a_rew[i])
                returns = np.zeros(T_len)
                G = 0
                for t in reversed(range(T_len)):
                    G = a_rew[i][t] + GAMMA * G
                    returns[t] = G
                if returns.std() > 1e-8:
                    returns = (returns - returns.mean()) / returns.std()
                agents[i].update(a_obs[i], a_act[i], returns)

            ep["welfare"].append(tw / max(steps, 1))
            ep["mean_lam"].append(tl / max(steps, 1))
            ep["survival"].append(float(surv))

        all_data.append(ep)
    return all_data


def run_fixed_commitment(lam_val, byz_frac=0.3, n_seeds=N_SEEDS):
    """Fixed commitment level for all honest agents."""
    n_byz = int(N_AGENTS * byz_frac)
    all_data = []

    for seed in range(n_seeds):
        rng = np.random.RandomState(42 + seed * 13)
        ep = {"welfare": [], "mean_lam": [], "survival": []}

        for e in range(N_EPISODES):
            env = reset_env()
            tw, tl, steps, surv = 0, 0, 0, True

            for t in range(T_HORIZON):
                lams = np.zeros(N_AGENTS)
                # Byzantine = 0
                lams[n_byz:] = lam_val

                rewards, done = step_env(env, lams, rng)
                tw += rewards.mean()
                tl += lams[n_byz:].mean()
                steps += 1
                if done:
                    surv = False
                    break

            ep["welfare"].append(tw / max(steps, 1))
            ep["mean_lam"].append(tl / max(steps, 1))
            ep["survival"].append(float(surv))

        all_data.append(ep)
    return all_data


def run_situational(byz_frac=0.3, n_seeds=N_SEEDS):
    """Situational commitment based on resource level."""
    n_byz = int(N_AGENTS * byz_frac)
    all_data = []

    for seed in range(n_seeds):
        rng = np.random.RandomState(42 + seed * 13)
        ep = {"welfare": [], "mean_lam": [], "survival": []}

        for e in range(N_EPISODES):
            env = reset_env()
            tw, tl, steps, surv = 0, 0, 0, True

            for t in range(T_HORIZON):
                lams = np.zeros(N_AGENTS)
                R = env["R"]
                lam_val = lambda_situational(R)
                lams[n_byz:] = lam_val

                rewards, done = step_env(env, lams, rng)
                tw += rewards.mean()
                tl += lams[n_byz:].mean()
                steps += 1
                if done:
                    surv = False
                    break

            ep["welfare"].append(tw / max(steps, 1))
            ep["mean_lam"].append(tl / max(steps, 1))
            ep["survival"].append(float(surv))

        all_data.append(ep)
    return all_data


def summarize(data, label, last_n=30):
    welfares = [np.mean(d["welfare"][-last_n:]) for d in data]
    lams = [np.mean(d["mean_lam"][-last_n:]) for d in data]
    survs = [np.mean(d["survival"][-last_n:]) * 100 for d in data]
    return {
        "label": label,
        "welfare_mean": round(float(np.mean(welfares)), 1),
        "lambda_mean": round(float(np.mean(lams)), 3),
        "survival_mean": round(float(np.mean(survs)), 1),
        "welfare_std": round(float(np.std(welfares)), 1),
        "lambda_std": round(float(np.std(lams)), 3),
        "survival_std": round(float(np.std(survs)), 1),
    }


if __name__ == "__main__":
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'scale_n100')
    os.makedirs(OUT, exist_ok=True)

    print("=" * 60)
    print("  SCALE TEST: N=100 Non-linear PGG (Table 5)")
    print(f"  N={N_AGENTS}, T={T_HORIZON}, seeds={N_SEEDS}")
    print("=" * 60)

    t0 = time.time()

    # 1. Selfish RL, no Byzantine
    d1 = run_selfish_rl(byz_frac=0.0)
    s1 = summarize(d1, "Selfish RL (Byz=0%)")
    print(f"  {s1['label']:30s} | W={s1['welfare_mean']}, lam={s1['lambda_mean']}, surv={s1['survival_mean']}%")

    # 2. Selfish RL, 30% Byzantine
    d2 = run_selfish_rl(byz_frac=0.3)
    s2 = summarize(d2, "Selfish RL (Byz=30%)")
    print(f"  {s2['label']:30s} | W={s2['welfare_mean']}, lam={s2['lambda_mean']}, surv={s2['survival_mean']}%")

    # 3. Situational commitment, 30% Byzantine
    d3 = run_situational(byz_frac=0.3)
    s3 = summarize(d3, "Situational (Byz=30%)")
    print(f"  {s3['label']:30s} | W={s3['welfare_mean']}, lam={s3['lambda_mean']}, surv={s3['survival_mean']}%")

    # 4. Unconditional commitment, 30% Byzantine
    d4 = run_fixed_commitment(lam_val=1.0, byz_frac=0.3)
    s4 = summarize(d4, "Unconditional (Byz=30%)")
    print(f"  {s4['label']:30s} | W={s4['welfare_mean']}, lam={s4['lambda_mean']}, surv={s4['survival_mean']}%")

    total = time.time() - t0

    print(f"\n  Total: {total:.0f}s")

    output = {
        "experiment": "Scale Test N=100 (Table 5)",
        "N": N_AGENTS, "T": T_HORIZON, "seeds": N_SEEDS,
        "results": [s1, s2, s3, s4],
        "time": total,
    }

    json_path = os.path.join(OUT, "scale_n100_results.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  [Save] {json_path}")
