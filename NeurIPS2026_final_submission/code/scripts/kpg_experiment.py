"""
KPG (K-Level Policy Gradients) Comparison Experiment

Tests whether K-level opponent-shaping can escape the Nash Trap
in non-linear PGG with tipping points.

K=0: Standard REINFORCE (baseline = selfish RL)
K=1: 1-step opponent anticipation
K=2: 2-step recursive anticipation

Reviewer response: "Even latest opponent-shaping fails in N-player non-linear PGG"
"""

import numpy as np
import json
import os
import time

# ============================================================
# Config (identical to previous experiments)
# ============================================================
N_AGENTS = 20
T_ROUNDS = 50  # Must match paper Table 7 (T = 50)
MULTIPLIER = 1.6
ENDOWMENT = 20.0
R_CRIT = 0.15
R_RECOV = 0.25
SHOCK_PROB = 0.05
SHOCK_MAG = 0.15
STATE_DIM = 4
GAMMA = 0.99
N_EPISODES = 300
N_SEEDS = 5
BYZ_FRAC = 0.3
N_BYZ = int(N_AGENTS * BYZ_FRAC)

# ============================================================
# Environment (same non-linear PGG)
# ============================================================
def reset_env():
    return {"R": 0.5, "t": 0, "lam_prev": np.full(N_AGENTS, 0.5), "mean_c": 0.5}

def get_obs(env):
    crisis = 1.0 if env["R"] < R_CRIT else 0.0
    return np.column_stack([
        np.full(N_AGENTS, env["R"]),
        np.full(N_AGENTS, env["mean_c"]),
        env["lam_prev"],
        np.full(N_AGENTS, crisis),
    ])

def step_env(env, lambdas, rng):
    lambdas[:N_BYZ] = 0.0
    contribs = ENDOWMENT * lambdas
    public = (contribs.sum() * MULTIPLIER) / N_AGENTS
    rewards = (ENDOWMENT - contribs) + public
    coop = contribs.mean() / ENDOWMENT
    R = env["R"]
    base = 0.1 * (coop - 0.4)
    if R < R_CRIT:
        R_new = R + base * 0.1
    elif R < R_RECOV:
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
    return rewards, done, {"R": R_new, "welfare": float(rewards.mean()),
                           "mean_lam": float(lambdas[N_BYZ:].mean()),
                           "collapsed": R_new <= 0.001}

# ============================================================
# Policy with parameters
# ============================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

class Policy:
    def __init__(self, lr=0.003):
        self.W = np.zeros((STATE_DIM, 1))
        self.b = np.zeros(1)
        self.lr = lr

    def forward(self, obs):
        z = obs @ self.W + self.b
        return sigmoid(z.flatten())

    def get_params(self):
        return np.concatenate([self.W.flatten(), self.b.flatten()])

    def set_params(self, params):
        self.W = params[:STATE_DIM].reshape(STATE_DIM, 1)
        self.b = params[STATE_DIM:STATE_DIM+1]

    def clone(self):
        p = Policy(self.lr)
        p.W = self.W.copy()
        p.b = self.b.copy()
        return p

    def get_actions(self, obs, rng, noise_scale=0.1):
        base = self.forward(obs)
        noise = rng.normal(0, noise_scale, size=base.shape)
        return np.clip(base + noise, 0.0, 1.0)

    def reinforce_update(self, obs_list, act_list, ret_list):
        if len(obs_list) == 0:
            return
        obs = np.array(obs_list)
        act = np.array(act_list)
        ret = np.array(ret_list)
        if ret.std() > 1e-8:
            ret = (ret - ret.mean()) / (ret.std() + 1e-8)
        lam = self.forward(obs)
        sig_deriv = (lam * (1 - lam)).reshape(-1, 1)
        delta = ((act - lam) * ret).reshape(-1, 1)
        grad_W = (obs.T @ (delta * sig_deriv)) / len(obs)
        grad_b = (delta * sig_deriv).mean(axis=0)
        self.W += self.lr * grad_W
        self.b += self.lr * grad_b

# ============================================================
# KPG: K-Level Policy Gradient
# ============================================================
def simulate_opponent_update(policy, env_state, rng, n_rollout=5):
    """Simulate what opponents would do after seeing my policy."""
    opp_policy = policy.clone()
    obs_buf, act_buf, rew_buf = [], [], []

    for _ in range(n_rollout):
        env = {k: (v.copy() if isinstance(v, np.ndarray) else v)
               for k, v in env_state.items()}
        for t in range(min(20, T_ROUNDS)):
            obs = get_obs(env)
            lams = opp_policy.get_actions(obs, rng, noise_scale=0.1)
            rewards, done, info = step_env(env, lams, rng)
            for i in range(N_BYZ, N_AGENTS):
                obs_buf.append(obs[i])
                act_buf.append(lams[i])
                rew_buf.append(rewards[i])
            if done:
                break

    if len(rew_buf) > 0:
        returns = []
        G = 0
        for r in reversed(rew_buf):
            G = r + GAMMA * G
            returns.insert(0, G)
        opp_policy.reinforce_update(obs_buf, act_buf, returns)

    return opp_policy

def kpg_gradient(agent_policy, env_state, rng, K=1, n_rollout=5):
    """
    K-Level Policy Gradient:
    K=0: standard REINFORCE
    K=1: anticipate 1-step opponent reaction
    K=2: anticipate 2-step recursive reaction
    """
    if K == 0:
        return None  # Use standard REINFORCE

    # Simulate K levels of opponent anticipation
    anticipated_policy = agent_policy.clone()
    for k in range(K):
        anticipated_policy = simulate_opponent_update(
            anticipated_policy, env_state, rng, n_rollout
        )

    # Now compute gradient assuming opponents will become anticipated_policy
    # The key insight: use anticipated opponent behavior to compute own gradient
    obs_buf, act_buf, rew_buf = [], [], []
    for _ in range(n_rollout):
        env = {k: (v.copy() if isinstance(v, np.ndarray) else v)
               for k, v in env_state.items()}
        for t in range(min(20, T_ROUNDS)):
            obs = get_obs(env)
            # My actions from my policy
            my_lams = agent_policy.get_actions(obs, rng, noise_scale=0.1)
            # But opponent actions from anticipated policy
            opp_lams = anticipated_policy.forward(obs)
            # Mix: my agent uses my policy, others use anticipated
            lams = opp_lams.copy()
            # Agent 0 (non-byz representative) uses own policy
            for i in range(N_BYZ, N_AGENTS):
                lams[i] = my_lams[i]
            rewards, done, info = step_env(env, lams, rng)
            for i in range(N_BYZ, N_AGENTS):
                obs_buf.append(obs[i])
                act_buf.append(my_lams[i])
                rew_buf.append(rewards[i])
            if done:
                break

    if len(rew_buf) > 0:
        returns = []
        G = 0
        for r in reversed(rew_buf):
            G = r + GAMMA * G
            returns.insert(0, G)
        agent_policy.reinforce_update(obs_buf, act_buf, returns)

# ============================================================
# Training
# ============================================================
def train_kpg(K, n_episodes=N_EPISODES, n_seeds=N_SEEDS):
    all_metrics = []
    t0 = time.time()

    for seed in range(n_seeds):
        policy = Policy(lr=0.003)
        rng = np.random.RandomState(seed * 10000)
        seed_metrics = []

        for ep in range(n_episodes):
            env = reset_env()
            obs_buf, act_buf, rew_buf = [], [], []
            ep_lam = []
            noise = max(0.15 - ep * 0.0003, 0.02)

            for t in range(T_ROUNDS):
                obs = get_obs(env)
                lams = policy.get_actions(obs, rng, noise_scale=noise)
                rewards, done, info = step_env(env, lams, rng)

                for i in range(N_BYZ, N_AGENTS):
                    obs_buf.append(obs[i])
                    act_buf.append(lams[i])
                    rew_buf.append(rewards[i])
                ep_lam.append(info["mean_lam"])
                if done:
                    break

            # Standard REINFORCE update
            returns = []
            G = 0
            for r in reversed(rew_buf):
                G = r + GAMMA * G
                returns.insert(0, G)
            policy.reinforce_update(obs_buf, act_buf, returns)

            # KPG additional update (K >= 1)
            if K >= 1:
                env_snapshot = reset_env()
                env_snapshot["R"] = info["R"]
                env_snapshot["mean_c"] = info.get("mean_lam", 0.5)
                kpg_gradient(policy, env_snapshot, rng, K=K, n_rollout=3)

            seed_metrics.append({
                "ep": ep,
                "mean_lam": float(np.mean(ep_lam)),
                "survived": not info["collapsed"],
                "welfare": info["welfare"],
            })

            if ep % 100 == 0:
                recent = seed_metrics[max(0, ep-10):ep+1]
                avg_lam = np.mean([m["mean_lam"] for m in recent])
                avg_surv = np.mean([m["survived"] for m in recent])
                print(f"    [K={K}] Seed {seed} Ep {ep:3d} | "
                      f"lam={avg_lam:.3f} surv={avg_surv*100:.0f}%")

        all_metrics.append(seed_metrics)

    elapsed = time.time() - t0
    # Summarize last 30 episodes across all seeds
    flat = [m for s in all_metrics for m in s[-30:]]
    result = {
        "K": K,
        "mean_lam": float(np.mean([m["mean_lam"] for m in flat])),
        "survival": float(np.mean([m["survived"] for m in flat])),
        "welfare": float(np.mean([m["welfare"] for m in flat])),
        "time_s": float(elapsed),
    }
    return result

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    OUT = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'kpg_experiment')
    os.makedirs(OUT, exist_ok=True)

    print("=" * 65)
    print("  KPG EXPERIMENT: K-Level Policy Gradients vs Nash Trap")
    print("  N=20, Byz=30%, Non-linear PGG, 300ep x 5 seeds")
    print("=" * 65)

    results = []
    for K in [0, 1, 2]:
        print(f"\n{'='*50}")
        print(f"  K = {K}")
        print(f"{'='*50}")
        r = train_kpg(K)
        results.append(r)
        print(f"  => lam={r['mean_lam']:.3f}, surv={r['survival']*100:.1f}%, "
              f"W={r['welfare']:.1f}, time={r['time_s']:.1f}s")

    print(f"\n{'='*65}")
    print(f"  KPG RESULTS SUMMARY (Byz=30%, last 30 ep)")
    print(f"{'='*65}")
    print(f"  {'K-Level':<10} | {'Lambda':>7} | {'Survival':>8} | {'Welfare':>8} | {'Time':>6}")
    print(f"  {'-'*55}")
    for r in results:
        print(f"  K={r['K']:<8} | {r['mean_lam']:>7.3f} | "
              f"{r['survival']*100:>7.1f}% | {r['welfare']:>8.1f} | {r['time_s']:>5.1f}s")

    # Compute time scaling
    if results[0]['time_s'] > 0:
        for r in results:
            r['time_ratio'] = r['time_s'] / results[0]['time_s']

    output = {"results": results}
    path = os.path.join(OUT, "kpg_results.json")
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved: {path}")
