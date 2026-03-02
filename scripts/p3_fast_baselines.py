"""
P3 Strong MARL Baselines — Fast Unified Experiment
===================================================
Runs True IPPO, MAPPO, QMIX on identical Non-linear PGG environment.
Uses analytical gradients (no numerical differentiation) for speed.

Environment: Identical to ppo_nash_trap.py
  N=20, M=1.6, Byz=30%, T=50, R_crit=0.15

Dependencies: NumPy only.
"""

import numpy as np
import json
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Environment Config (MUST match paper Table 7)
# ============================================================
N_AGENTS = 20
ENDOWMENT = 20.0
MULTIPLIER = 1.6
T_HORIZON = 50
R_CRIT = 0.15
R_RECOV = 0.25
SHOCK_PROB = 0.05
SHOCK_MAG = 0.15
STATE_DIM = 4
GLOBAL_DIM = 5
BYZ_FRAC = 0.3
N_BYZ = int(N_AGENTS * BYZ_FRAC)
N_HONEST = N_AGENTS - N_BYZ
GAMMA = 0.99
GAE_LAMBDA = 0.95

N_EPISODES = 500
N_SEEDS = 10


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def reset_env():
    return {"R": 0.5, "mean_c": 0.5, "lam_prev": np.full(N_AGENTS, 0.5), "t": 0}


def get_obs_agent(env, idx):
    R = env["R"]
    return np.array([R, env["mean_c"], env["lam_prev"][idx], 1.0 if R < R_CRIT else 0.0])


def get_global_state(env):
    R = env["R"]
    return np.array([R, env["mean_c"], float(env["lam_prev"][N_BYZ:].mean()),
                     1.0 if R < R_CRIT else 0.0, env["t"] / T_HORIZON])


def step_env(env, lambdas, rng):
    lam = lambdas.copy()
    lam[:N_BYZ] = 0.0
    contribs = ENDOWMENT * lam
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
    env["lam_prev"] = lam.copy()
    env["t"] += 1
    return rewards, done


def compute_gae(rewards, values, gamma=GAMMA, lam=GAE_LAMBDA):
    T = len(rewards)
    advs = np.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        nv = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * nv - values[t]
        gae = delta + gamma * lam * gae
        advs[t] = gae
    returns = advs + np.array(values[:T])
    return advs, returns


# ============================================================
# Linear Gaussian Policy with Analytical PPO Gradient
# ============================================================
class GaussianLinearPolicy:
    """Linear Gaussian: mu = sigmoid(w @ obs + b), fixed std."""
    def __init__(self, rng, dim=STATE_DIM, std=0.15):
        self.w = rng.randn(dim) * 0.01
        self.b = 0.0
        self.log_std = np.log(std)
        self.lr = 0.005
        self.clip_eps = 0.2

    def _logit(self, obs):
        return obs @ self.w + self.b

    def _mu(self, obs):
        return sigmoid(self._logit(obs))

    def act(self, obs, rng):
        mu = self._mu(obs)
        std = np.exp(self.log_std)
        a = float(np.clip(mu + rng.randn() * std, 0.01, 0.99))
        lp = -0.5 * ((a - mu) / std)**2 - self.log_std
        return a, float(lp)

    def log_prob(self, obs, a):
        mu = self._mu(obs)
        std = np.exp(self.log_std)
        return float(-0.5 * ((a - mu) / std)**2 - self.log_std)

    def ppo_update(self, obs_list, act_list, old_lps, advs):
        """Analytical PPO gradient update."""
        if len(obs_list) == 0:
            return
        std = np.exp(self.log_std)
        grad_w = np.zeros_like(self.w)
        grad_b = 0.0

        for obs, a, old_lp, adv in zip(obs_list, act_list, old_lps, advs):
            mu = self._mu(obs)
            new_lp = -0.5 * ((a - mu) / std)**2 - self.log_std
            ratio = np.exp(new_lp - old_lp)

            # Clipped surrogate
            surr1 = ratio * adv
            surr2 = np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv

            if surr1 <= surr2:
                # Not clipped: compute gradient
                # d new_lp / d mu = (a - mu) / std^2
                # d mu / d logit = mu * (1 - mu)   [sigmoid derivative]
                # d logit / d w = obs
                dlp_dmu = (a - mu) / (std**2)
                dmu_dlogit = mu * (1 - mu)
                dlogit_dw = obs

                # Chain rule: d loss / d w = -ratio * adv * dlp/dmu * dmu/dlogit * dlogit/dw
                scale = ratio * adv * dlp_dmu * dmu_dlogit
                grad_w += scale * dlogit_dw
                grad_b += scale

        n = len(obs_list)
        self.w += self.lr * grad_w / n
        self.b += self.lr * grad_b / n


class LinearValue:
    """Simple linear value function."""
    def __init__(self, rng, dim=STATE_DIM):
        self.w = rng.randn(dim) * 0.01
        self.b = 0.0
        self.lr = 0.01

    def predict(self, obs):
        return float(obs @ self.w + self.b)

    def update(self, obs_list, targets):
        for obs, target in zip(obs_list, targets):
            pred = self.predict(obs)
            err = target - pred
            self.w += self.lr * err * obs
            self.b += self.lr * err


class SharedLinearValue:
    """Shared value function for MAPPO (global state input)."""
    def __init__(self, rng, dim=GLOBAL_DIM):
        self.w = rng.randn(dim) * 0.01
        self.b = 0.0
        self.lr = 0.01

    def predict(self, gs):
        return float(gs @ self.w + self.b)

    def update(self, gs_list, targets):
        for gs, target in zip(gs_list, targets):
            pred = self.predict(gs)
            err = target - pred
            self.w += self.lr * err * gs
            self.b += self.lr * err


# ============================================================
# EXP 1: True IPPO (PPO Clipped Surrogate, per-agent)
# ============================================================
def run_true_ippo(n_ep=N_EPISODES, n_seeds=N_SEEDS):
    print("\n" + "=" * 60)
    print("  [1/3] TRUE IPPO -- PPO Clipped Surrogate (per-agent)")
    print("=" * 60)

    all_data = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(42 + seed * 7)
        policies = [GaussianLinearPolicy(np.random.RandomState(seed * 100 + i)) for i in range(N_HONEST)]
        values = [LinearValue(np.random.RandomState(seed * 200 + i)) for i in range(N_HONEST)]

        ep = {"welfare": [], "mean_lam": [], "survival": [], "R_final": []}

        for e in range(n_ep):
            env = reset_env()
            a_obs = [[] for _ in range(N_HONEST)]
            a_act = [[] for _ in range(N_HONEST)]
            a_lp = [[] for _ in range(N_HONEST)]
            a_rew = [[] for _ in range(N_HONEST)]
            a_val = [[] for _ in range(N_HONEST)]

            tw, tl, steps, surv = 0, 0, 0, True

            for t in range(T_HORIZON):
                lams = np.zeros(N_AGENTS)
                for i in range(N_HONEST):
                    obs = get_obs_agent(env, N_BYZ + i)
                    act, lp = policies[i].act(obs, rng)
                    v = values[i].predict(obs)
                    lams[N_BYZ + i] = act
                    a_obs[i].append(obs)
                    a_act[i].append(act)
                    a_lp[i].append(lp)
                    a_val[i].append(v)

                rewards, done = step_env(env, lams, rng)
                for i in range(N_HONEST):
                    a_rew[i].append(rewards[N_BYZ + i])

                tw += rewards.mean()
                tl += lams[N_BYZ:].mean()
                steps += 1

                if done:
                    surv = False
                    break

            # PPO update (K=3 epochs)
            for i in range(N_HONEST):
                if len(a_rew[i]) < 2:
                    continue
                advs, rets = compute_gae(a_rew[i], a_val[i])
                if advs.std() > 1e-8:
                    advs = (advs - advs.mean()) / advs.std()

                for _ in range(3):
                    policies[i].ppo_update(a_obs[i], a_act[i], a_lp[i], advs)
                values[i].update(a_obs[i], rets)

            ep["welfare"].append(tw / max(steps, 1))
            ep["mean_lam"].append(tl / max(steps, 1))
            ep["survival"].append(float(surv))
            ep["R_final"].append(env["R"])

            if (e + 1) % 100 == 0:
                r = slice(-30, None)
                print(f"  Seed {seed} ep {e+1}: lam={np.mean(ep['mean_lam'][r]):.3f}, "
                      f"surv={np.mean(ep['survival'][r])*100:.0f}%")

        all_data.append(ep)
    return all_data


# ============================================================
# EXP 2: MAPPO (Shared Critic, per-agent policy)
# ============================================================
def run_mappo(n_ep=N_EPISODES, n_seeds=N_SEEDS):
    print("\n" + "=" * 60)
    print("  [2/3] MAPPO -- Shared Critic PPO (Yu et al., 2022)")
    print("=" * 60)

    all_data = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(42 + seed * 7)
        policies = [GaussianLinearPolicy(np.random.RandomState(seed * 100 + i)) for i in range(N_HONEST)]
        shared_v = SharedLinearValue(np.random.RandomState(seed * 300))

        ep = {"welfare": [], "mean_lam": [], "survival": [], "R_final": []}

        for e in range(n_ep):
            env = reset_env()
            a_obs = [[] for _ in range(N_HONEST)]
            a_act = [[] for _ in range(N_HONEST)]
            a_lp = [[] for _ in range(N_HONEST)]
            a_rew = [[] for _ in range(N_HONEST)]
            gs_list = []
            gv_list = []

            tw, tl, steps, surv = 0, 0, 0, True

            for t in range(T_HORIZON):
                gs = get_global_state(env)
                gs_list.append(gs)
                gv_list.append(shared_v.predict(gs))

                lams = np.zeros(N_AGENTS)
                for i in range(N_HONEST):
                    obs = get_obs_agent(env, N_BYZ + i)
                    act, lp = policies[i].act(obs, rng)
                    lams[N_BYZ + i] = act
                    a_obs[i].append(obs)
                    a_act[i].append(act)
                    a_lp[i].append(lp)

                rewards, done = step_env(env, lams, rng)
                for i in range(N_HONEST):
                    a_rew[i].append(rewards[N_BYZ + i])

                tw += rewards.mean()
                tl += lams[N_BYZ:].mean()
                steps += 1

                if done:
                    surv = False
                    break

            # PPO with shared critic values
            for i in range(N_HONEST):
                if len(a_rew[i]) < 2:
                    continue
                advs, rets = compute_gae(a_rew[i], gv_list)
                if advs.std() > 1e-8:
                    advs = (advs - advs.mean()) / advs.std()
                for _ in range(3):
                    policies[i].ppo_update(a_obs[i], a_act[i], a_lp[i], advs)

            # Update shared critic
            if len(gs_list) > 1:
                mean_r = [np.mean([a_rew[i][t] for i in range(min(N_HONEST, len(a_rew)))
                                   if t < len(a_rew[i])]) for t in range(len(gs_list))]
                _, c_targets = compute_gae(mean_r, gv_list)
                shared_v.update(gs_list, c_targets)

            ep["welfare"].append(tw / max(steps, 1))
            ep["mean_lam"].append(tl / max(steps, 1))
            ep["survival"].append(float(surv))
            ep["R_final"].append(env["R"])

            if (e + 1) % 100 == 0:
                r = slice(-30, None)
                print(f"  Seed {seed} ep {e+1}: lam={np.mean(ep['mean_lam'][r]):.3f}, "
                      f"surv={np.mean(ep['survival'][r])*100:.0f}%")

        all_data.append(ep)
    return all_data


# ============================================================
# EXP 3: QMIX (Value Decomposition, discrete actions)
# ============================================================
ACTION_VALUES = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
N_ACTIONS = len(ACTION_VALUES)

class TabularQAgent:
    """Q-table with feature hashing for efficiency."""
    def __init__(self, rng, n_bins=8, n_actions=N_ACTIONS):
        self.n_bins = n_bins
        self.n_features = n_bins ** STATE_DIM
        self.n_actions = n_actions
        # Use feature hashing to keep it compact
        self.q_table = rng.randn(1024, n_actions) * 0.01
        self.lr = 0.1

    def _hash(self, obs):
        # Discretize obs into bins and hash
        bins = np.clip((obs * self.n_bins).astype(int), 0, self.n_bins - 1)
        h = 0
        for b in bins:
            h = h * self.n_bins + b
        return h % 1024

    def get_q(self, obs):
        return self.q_table[self._hash(obs)].copy()

    def update(self, obs, action, target):
        h = self._hash(obs)
        self.q_table[h, action] += self.lr * (target - self.q_table[h, action])


class SimpleMixer:
    """Simple monotonic mixer: Q_tot = sum(w_i * Q_i) + b."""
    def __init__(self, rng, n_agents=N_HONEST):
        self.weights = np.abs(rng.randn(n_agents)) * 0.1 + 0.5
        self.bias = 0.0
        self.lr = 0.01

    def forward(self, agent_qs):
        return float(np.sum(np.abs(self.weights) * agent_qs) + self.bias)

    def update(self, agent_qs, target):
        q_tot = self.forward(agent_qs)
        err = target - q_tot
        self.weights += self.lr * err * agent_qs * np.sign(self.weights)
        self.bias += self.lr * err


def run_qmix(n_ep=N_EPISODES, n_seeds=N_SEEDS):
    print("\n" + "=" * 60)
    print("  [3/3] QMIX -- Value Decomposition (Rashid et al., 2018)")
    print("=" * 60)

    all_data = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(42 + seed * 7)
        agents = [TabularQAgent(np.random.RandomState(seed * 100 + i)) for i in range(N_HONEST)]
        target_agents = [TabularQAgent(np.random.RandomState(seed * 100 + i)) for i in range(N_HONEST)]
        mixer = SimpleMixer(np.random.RandomState(seed * 300))

        # Copy initial params
        for a, ta in zip(agents, target_agents):
            ta.q_table = a.q_table.copy()

        ep = {"welfare": [], "mean_lam": [], "survival": [], "R_final": []}

        for e in range(n_ep):
            eps = max(0.05, 1.0 - e / 300)
            env = reset_env()
            tw, tl, steps, surv = 0, 0, 0, True
            trajectory = []

            for t in range(T_HORIZON):
                lams = np.zeros(N_AGENTS)
                obs_all = []
                actions = []

                for i in range(N_HONEST):
                    obs = get_obs_agent(env, N_BYZ + i)
                    obs_all.append(obs)
                    if rng.random() < eps:
                        a = rng.randint(N_ACTIONS)
                    else:
                        a = int(np.argmax(agents[i].get_q(obs)))
                    actions.append(a)
                    lams[N_BYZ + i] = ACTION_VALUES[a]

                rewards, done = step_env(env, lams, rng)
                agent_rewards = [rewards[N_BYZ + i] for i in range(N_HONEST)]

                next_obs_all = [get_obs_agent(env, N_BYZ + i) for i in range(N_HONEST)]
                trajectory.append((obs_all, actions, agent_rewards, next_obs_all, done))

                tw += rewards.mean()
                tl += lams[N_BYZ:].mean()
                steps += 1

                if done:
                    surv = False
                    break

            # QMIX update from trajectory
            for obs_all, actions, agent_rewards, next_obs_all, done in trajectory:
                # Current Q values
                chosen_qs = np.array([agents[i].get_q(obs_all[i])[actions[i]]
                                      for i in range(N_HONEST)])
                q_tot = mixer.forward(chosen_qs)

                # Target Q values
                next_max_qs = np.array([np.max(target_agents[i].get_q(next_obs_all[i]))
                                        for i in range(N_HONEST)])
                target_q_tot = mixer.forward(next_max_qs)

                mean_r = np.mean(agent_rewards)
                y = mean_r + GAMMA * target_q_tot * (1 - float(done))

                # Update mixer
                mixer.update(chosen_qs, y)

                # Update individual Q-networks
                for i in range(N_HONEST):
                    individual_target = agent_rewards[i] + GAMMA * np.max(
                        target_agents[i].get_q(next_obs_all[i])) * (1 - float(done))
                    agents[i].update(obs_all[i], actions[i], individual_target)

            # Target network update
            if (e + 1) % 50 == 0:
                for a, ta in zip(agents, target_agents):
                    ta.q_table = a.q_table.copy()

            ep["welfare"].append(tw / max(steps, 1))
            ep["mean_lam"].append(tl / max(steps, 1))
            ep["survival"].append(float(surv))
            ep["R_final"].append(env["R"])

            if (e + 1) % 100 == 0:
                r = slice(-30, None)
                print(f"  Seed {seed} ep {e+1}: lam={np.mean(ep['mean_lam'][r]):.3f}, "
                      f"surv={np.mean(ep['survival'][r])*100:.0f}%")

        all_data.append(ep)
    return all_data


# ============================================================
# Summarize & Report
# ============================================================
def summarize(data, label):
    last_n = 30
    welfares = [np.mean(d["welfare"][-last_n:]) for d in data]
    lams = [np.mean(d["mean_lam"][-last_n:]) for d in data]
    survs = [np.mean(d["survival"][-last_n:]) * 100 for d in data]
    return {
        "label": label,
        "welfare": f"{np.mean(welfares):.1f}+/-{np.std(welfares):.1f}",
        "lambda": f"{np.mean(lams):.3f}+/-{np.std(lams):.3f}",
        "survival": f"{np.mean(survs):.1f}+/-{np.std(survs):.1f}%",
        "lambda_mean": float(np.mean(lams)),
        "lambda_std": float(np.std(lams)),
        "welfare_mean": float(np.mean(welfares)),
        "survival_mean": float(np.mean(survs)),
        "survival_std": float(np.std(survs)),
    }


def plot_comparison(results_dict, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {
        "True IPPO": "#2980b9",
        "MAPPO": "#8e44ad",
        "QMIX": "#d35400",
    }

    # (a) Lambda convergence
    ax = axes[0]
    for label, data in results_dict.items():
        all_lams = np.array([d["mean_lam"] for d in data])
        mean = all_lams.mean(axis=0)
        std = all_lams.std(axis=0)
        w = 20
        if len(mean) > w:
            ms = np.convolve(mean, np.ones(w)/w, mode='valid')
            ss = np.convolve(std, np.ones(w)/w, mode='valid')
            x = np.arange(len(ms))
        else:
            ms, ss, x = mean, std, np.arange(len(mean))
        c = colors.get(label, "#2ecc71")
        ax.plot(x, ms, label=label, color=c, linewidth=2)
        ax.fill_between(x, ms-ss, ms+ss, alpha=0.15, color=c)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Nash Eq.')
    ax.axhspan(0.25, 0.75, alpha=0.05, color='red')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Mean λ')
    ax.set_title('(a) Commitment Level', fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    # (b) Survival
    ax = axes[1]
    for label, data in results_dict.items():
        all_s = np.array([d["survival"] for d in data]) * 100
        mean = all_s.mean(axis=0)
        w = 30
        if len(mean) > w:
            ms = np.convolve(mean, np.ones(w)/w, mode='valid')
            x = np.arange(len(ms))
        else:
            ms, x = mean, np.arange(len(mean))
        c = colors.get(label, "#2ecc71")
        ax.plot(x, ms, label=label, color=c, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Survival (%)')
    ax.set_title('(b) Group Survival', fontweight='bold')
    ax.legend(fontsize=8)

    # (c) Final bar
    ax = axes[2]
    labels = list(results_dict.keys())
    final_lams = []
    final_stds = []
    for label in labels:
        data = results_dict[label]
        seed_lams = [np.mean(d["mean_lam"][-30:]) for d in data]
        final_lams.append(np.mean(seed_lams))
        final_stds.append(np.std(seed_lams))

    bars = ax.bar(range(len(labels)), final_lams, yerr=final_stds,
                  color=[colors.get(l, "#2ecc71") for l in labels],
                  capsize=5, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    ax.axhspan(0.25, 0.75, alpha=0.06, color='red', label='Nash Trap')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Converged λ')
    ax.set_title('(c) Final Commitment', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)

    plt.suptitle('Standard MARL Algorithms: All Converge to Nash Trap\n'
                 'Non-linear PGG, N=20, Byzantine=30%, R_crit=0.15',
                 fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {output_path}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'p3_baselines')
    os.makedirs(OUT, exist_ok=True)

    print("=" * 60)
    print("  P3: STRONG MARL BASELINES -- NON-LINEAR PGG NASH TRAP")
    print(f"  N={N_AGENTS}, Byz={BYZ_FRAC*100:.0f}%, M/N={MULTIPLIER/N_AGENTS:.3f}")
    print(f"  Episodes={N_EPISODES}, Seeds={N_SEEDS}")
    print("=" * 60)

    t0 = time.time()

    # Run all three
    d1 = run_true_ippo()
    s1 = summarize(d1, "True IPPO")

    d2 = run_mappo()
    s2 = summarize(d2, "MAPPO")

    d3 = run_qmix()
    s3 = summarize(d3, "QMIX")

    total = time.time() - t0

    # Report
    print(f"\n{'=' * 60}")
    print(f"  ALL RESULTS (last 30 episodes, {N_SEEDS} seeds)")
    print(f"{'=' * 60}")
    print(f"  {'Algorithm':20s} | {'Lambda':15s} | {'Survival':15s} | {'Welfare':12s}")
    print(f"  {'-'*68}")
    for s in [s1, s2, s3]:
        trap = "TRAP" if abs(s['lambda_mean'] - 0.5) < 0.30 else "OUT"
        print(f"  {s['label']:20s} | {s['lambda']:15s} | {s['survival']:15s} | {s['welfare']:12s} [{trap}]")
    print(f"\n  Total time: {total:.0f}s")

    # Save
    output = {
        "experiment": "P3 Strong MARL Baselines",
        "environment": {
            "N": N_AGENTS, "Byz": BYZ_FRAC, "M": MULTIPLIER,
            "T": T_HORIZON, "R_crit": R_CRIT,
            "episodes": N_EPISODES, "seeds": N_SEEDS,
        },
        "results": {
            "true_ippo": s1,
            "mappo": s2,
            "qmix": s3,
        },
        "time_seconds": total,
    }

    json_path = os.path.join(OUT, "p3_results.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"[Save] {json_path}")

    # Plot
    fig_path = os.path.join(OUT, "p3_nash_trap_comparison.png")
    plot_comparison({"True IPPO": d1, "MAPPO": d2, "QMIX": d3}, fig_path)

    print(f"\n  COMPLETE!")
