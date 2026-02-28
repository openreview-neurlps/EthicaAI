"""
MAPPO-PGG: Pure Selfish RL Emergence Experiment (v2 - Vectorized)

Lightweight, fast implementation. No scipy. Fully vectorized.

Reward: PURE payoff only (E - c_i + M*sum_c/N). No shaping.
State: (R_t, mean_c, lambda_prev, crisis_flag). No theta_i.
Action: sigmoid(W@s + b) -> lambda in [0,1]. Simple policy gradient.
"""

import numpy as np
import json
import os
import time

# ============================================================
# Config
# ============================================================
N_AGENTS = 20
T_ROUNDS = 100
MULTIPLIER = 1.6
ENDOWMENT = 20.0
R_CRIT = 0.15
R_RECOVERY = 0.25
SHOCK_PROB = 0.05
SHOCK_MAG = 0.15
STATE_DIM = 4
GAMMA_RL = 0.99
LR = 0.005
N_EPISODES = 300
N_SEEDS = 5


# ============================================================
# Vectorized PGG Environment
# ============================================================
def reset_env(rng):
    return {"R": 0.5, "t": 0, "lam_prev": np.full(N_AGENTS, 0.5), "mean_c": 0.5}


def get_obs(env):
    """(N_AGENTS, 4) observation matrix."""
    crisis = 1.0 if env["R"] < R_CRIT else 0.0
    obs = np.column_stack([
        np.full(N_AGENTS, env["R"]),
        np.full(N_AGENTS, env["mean_c"]),
        env["lam_prev"],
        np.full(N_AGENTS, crisis),
    ])
    return obs


def step_env(env, lambdas, n_byz, rng):
    """Vectorized step. Returns (rewards, done, info)."""
    lambdas[:n_byz] = 0.0
    contribs = ENDOWMENT * lambdas
    public = (contribs.sum() * MULTIPLIER) / N_AGENTS
    rewards = (ENDOWMENT - contribs) + public

    # Resource update
    coop = contribs.mean() / ENDOWMENT
    R = env["R"]
    base = 0.1 * (coop - 0.4)
    if R < R_CRIT:
        R_new = R + base * 0.1
    elif R < R_RECOVERY:
        R_new = R + base * 0.3
    else:
        R_new = R + base
    if rng.random() < SHOCK_PROB:
        R_new -= SHOCK_MAG
    R_new = float(np.clip(R_new, 0.0, 1.0))

    env["t"] += 1
    done = R_new <= 0.001 or env["t"] >= T_ROUNDS
    env["R"] = R_new
    env["lam_prev"] = lambdas.copy()
    env["mean_c"] = float(coop)

    info = {
        "R": R_new, "welfare": float(rewards.mean()),
        "mean_lam": float(lambdas[n_byz:].mean()),
        "collapsed": R_new <= 0.001,
    }
    return rewards, done, info


# ============================================================
# Simple Policy (sigmoid, no frameworks)
# ============================================================
class Policy:
    """Linear policy: lambda = sigmoid(W @ obs + b)."""

    def __init__(self, dim=STATE_DIM, lr=LR):
        self.W = np.zeros(dim)    # Shared weights for all agents
        self.b = 0.0
        self.lr = lr

    def forward(self, obs):
        """obs: (N, dim) -> lambdas: (N,)"""
        logits = obs @ self.W + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -10, 10)))

    def get_actions(self, obs, rng, noise_scale=0.1):
        """Get noisy actions for exploration."""
        base = self.forward(obs)
        noise = rng.normal(0, noise_scale, size=base.shape)
        return np.clip(base + noise, 0.0, 1.0)

    def update(self, obs_list, action_list, return_list):
        """REINFORCE update (vectorized)."""
        if len(obs_list) == 0:
            return 0.0

        obs_arr = np.array(obs_list)      # (T*N, dim)
        act_arr = np.array(action_list)   # (T*N,)
        ret_arr = np.array(return_list)   # (T*N,)

        # Normalize returns
        if ret_arr.std() > 1e-8:
            ret_arr = (ret_arr - ret_arr.mean()) / (ret_arr.std() + 1e-8)

        # Policy gradient: d/dW log(pi) * R
        # For sigmoid policy: d log pi / d logit = (action - sigmoid)
        logits = obs_arr @ self.W + self.b
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(logits, -10, 10)))
        grad_logit = act_arr - sigmoid  # (T*N,)

        # Weight gradient
        grad_W = (obs_arr.T @ (grad_logit * ret_arr)) / len(ret_arr)
        grad_b = (grad_logit * ret_arr).mean()

        self.W += self.lr * grad_W
        self.b += self.lr * grad_b

        return float(np.abs(grad_W).mean())


# ============================================================
# Training
# ============================================================
def train(n_episodes=N_EPISODES, n_seeds=N_SEEDS, byz_frac=0.0):
    n_byz = int(N_AGENTS * byz_frac)
    all_seed_data = []

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed * 100)
        policy = Policy()
        metrics = []

        for ep in range(n_episodes):
            env = reset_env(rng)
            ep_rng = np.random.RandomState(seed * 10000 + ep)

            obs_buf, act_buf, rew_buf = [], [], []
            ep_w, ep_lam, ep_R = [], [], []
            crisis_lams = []

            for t in range(T_ROUNDS):
                obs = get_obs(env)
                lams = policy.get_actions(obs, ep_rng,
                                          noise_scale=max(0.15 - ep*0.0003, 0.02))
                rewards, done, info = step_env(env, lams, n_byz, ep_rng)

                # Store honest agents only
                for i in range(n_byz, N_AGENTS):
                    obs_buf.append(obs[i])
                    act_buf.append(lams[i])
                    rew_buf.append(rewards[i])

                ep_w.append(info["welfare"])
                ep_lam.append(info["mean_lam"])
                ep_R.append(info["R"])
                if info["R"] < R_CRIT:
                    crisis_lams.append(info["mean_lam"])

                if done:
                    break

            # Compute discounted returns
            returns = []
            G = 0
            for r in reversed(rew_buf):
                G = r + GAMMA_RL * G
                returns.insert(0, G)

            # Update policy
            grad = policy.update(obs_buf, act_buf, returns)

            m = {
                "ep": ep,
                "welfare": float(np.mean(ep_w)),
                "mean_lam": float(np.mean(ep_lam)),
                "survived": not info["collapsed"],
                "final_R": float(ep_R[-1]),
                "steps": len(ep_w),
                "crisis_lam": float(np.mean(crisis_lams)) if crisis_lams else -1.0,
            }
            metrics.append(m)

            if ep % 50 == 0:
                r10 = metrics[max(0,ep-10):ep+1]
                sw = np.mean([x["welfare"] for x in r10])
                sl = np.mean([x["mean_lam"] for x in r10])
                ss = np.mean([x["survived"] for x in r10])
                cl = [x["crisis_lam"] for x in r10 if x["crisis_lam"]>=0]
                scl = np.mean(cl) if cl else -1
                print(f"    Seed {seed} Ep {ep:3d} | W={sw:.1f} lam={sl:.3f} "
                      f"surv={ss*100:.0f}% crisis_lam={scl:.3f} "
                      f"W=[{policy.W[0]:.2f},{policy.W[1]:.2f},{policy.W[2]:.2f},{policy.W[3]:.2f}]")

        all_seed_data.append(metrics)

    return all_seed_data


def run_baseline(lam_val, n_ep=50, n_seeds=N_SEEDS, byz_frac=0.0):
    n_byz = int(N_AGENTS * byz_frac)
    all_data = []
    for seed in range(n_seeds):
        metrics = []
        for ep in range(n_ep):
            rng = np.random.RandomState(seed*10000+ep)
            env = reset_env(rng)
            ws = []
            for t in range(T_ROUNDS):
                lams = np.full(N_AGENTS, lam_val)
                _, done, info = step_env(env, lams, n_byz, rng)
                ws.append(info["welfare"])
                if done:
                    break
            metrics.append({"welfare": float(np.mean(ws)), "survived": not info["collapsed"]})
        all_data.append(metrics)
    return all_data


def summarize(data, last_n=30):
    all_m = []
    for seed in data:
        all_m.extend(seed[-last_n:])
    r = {
        "welfare": float(np.mean([m["welfare"] for m in all_m])),
        "survival": float(np.mean([m["survived"] for m in all_m])),
    }
    if "mean_lam" in all_m[0]:
        r["mean_lam"] = float(np.mean([m["mean_lam"] for m in all_m]))
        cl = [m["crisis_lam"] for m in all_m if m.get("crisis_lam",-1)>=0]
        r["crisis_lam"] = float(np.mean(cl)) if cl else -1.0
    return r


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    OUT = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'mappo_emergence')
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    print("=" * 65)
    print("  MAPPO-PGG v2: Pure Selfish RL Emergence (Vectorized)")
    print("  Reward: PURE PAYOFF | State: NO theta | NO reward shaping")
    print("=" * 65)

    print("\n[1/4] Training: No Byzantine")
    rl_clean = train(byz_frac=0.0)

    print("\n[2/4] Training: 30% Byzantine")
    rl_byz30 = train(byz_frac=0.3)

    print("\n[3/4] Baselines")
    bl_0 = run_baseline(0.0)
    bl_5 = run_baseline(0.5)
    bl_10 = run_baseline(1.0)

    total = time.time() - t0

    # Summary
    results = {
        "rl_clean": summarize(rl_clean),
        "rl_byz30": summarize(rl_byz30),
        "bl_selfish": summarize(bl_0),
        "bl_moderate": summarize(bl_5),
        "bl_oracle": summarize(bl_10),
    }

    print("\n" + "=" * 65)
    print("  RESULTS (last 30 episodes)")
    print("=" * 65)
    print(f"  {'Config':>15} | {'Welfare':>8} | {'Lambda':>7} | {'Surv%':>6} | {'Crisis Lam':>10}")
    print(f"  {'-'*57}")
    for k, r in results.items():
        ml = f"{r.get('mean_lam',-1):.3f}" if r.get('mean_lam',-1)>=0 else "fixed"
        cl = f"{r.get('crisis_lam',-1):.3f}" if r.get('crisis_lam',-1)>=0 else "N/A"
        print(f"  {k:>15} | {r['welfare']:8.2f} | {ml:>7} | {r['survival']*100:5.1f}% | {cl:>10}")

    # Emergence verdict
    cl = results["rl_clean"].get("crisis_lam", -1)
    print(f"\n  === EMERGENCE VERDICT ===")
    if cl > 0.7:
        print(f"  EMERGENCE CONFIRMED: crisis_lambda={cl:.3f} > 0.7")
    elif cl > 0.5:
        print(f"  PARTIAL EMERGENCE: crisis_lambda={cl:.3f}")
    elif cl >= 0:
        print(f"  NO EMERGENCE: crisis_lambda={cl:.3f}")
        print(f"  -> Ethical prior g(theta,R) IS NECESSARY.")
    else:
        print(f"  N/A: No crisis observed.")

    # Save
    output = {
        "config": {
            "reward": "PURE_PAYOFF", "state": "NO_THETA",
            "shaping": "NONE", "n_agents": N_AGENTS,
            "episodes": N_EPISODES, "seeds": N_SEEDS,
        },
        "summary": results,
        "time_seconds": float(total),
        "curves": {
            "rl_clean": [[{"ep":m["ep"],"w":m["welfare"],"l":m["mean_lam"],
                           "s":m["survived"],"cl":m["crisis_lam"]} for m in s] for s in rl_clean],
            "rl_byz30": [[{"ep":m["ep"],"w":m["welfare"],"l":m["mean_lam"],
                           "s":m["survived"],"cl":m["crisis_lam"]} for m in s] for s in rl_byz30],
        },
    }
    path = os.path.join(OUT, "emergence_results.json")
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Time: {total:.1f}s | Results: {path}")
    print("  COMPLETE!")
