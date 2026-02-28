"""
Spatial Social Dilemma: Grid-based Non-linear Resource Game

A 7x7 grid with renewable resources and tipping points.
Agents move, harvest, and must decide how much to contribute
to resource regeneration (commitment).

Demonstrates that the Moral Commitment Spectrum holds
in spatially-extended environments beyond scalar-state PGG.
"""

import numpy as np
import json
import os
import time

# ============================================================
# Config
# ============================================================
GRID_SIZE = 5
N_AGENTS = 15
T_ROUNDS = 150
BYZ_FRAC = 0.3
N_BYZ = int(N_AGENTS * BYZ_FRAC)
ENDOWMENT = 10.0
MULTIPLIER = 1.6
R_CRIT = 0.20
R_RECOV = 0.30
SHOCK_PROB = 0.12
SHOCK_MAG = 0.25
N_EPISODES = 200
N_SEEDS = 5
GAMMA = 0.99
OBS_DIM = 8  # [my_x, my_y, local_R, neighbor_avg_R, global_avg_R, my_lam_prev, crisis_flag, t_norm]

# ============================================================
# Grid Environment
# ============================================================
class GridSocialDilemma:
    def __init__(self, rng):
        self.rng = rng
        self.reset()

    def reset(self):
        # Resource grid: each cell has R in [0, 1]
        self.resources = np.full((GRID_SIZE, GRID_SIZE), 0.5)
        # Agent positions (random)
        self.positions = np.array([
            (self.rng.randint(0, GRID_SIZE), self.rng.randint(0, GRID_SIZE))
            for _ in range(N_AGENTS)
        ])
        self.t = 0
        self.prev_lam = np.full(N_AGENTS, 0.5)
        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros((N_AGENTS, OBS_DIM))
        global_avg_R = self.resources.mean()

        for i in range(N_AGENTS):
            x, y = self.positions[i]
            local_R = self.resources[x, y]

            # Average of 4 neighbors (with boundary handling)
            neighbors = []
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    neighbors.append(self.resources[nx, ny])
            neighbor_avg = np.mean(neighbors) if neighbors else local_R

            crisis = 1.0 if local_R < R_CRIT else 0.0
            obs[i] = [
                x / GRID_SIZE,
                y / GRID_SIZE,
                local_R,
                neighbor_avg,
                global_avg_R,
                self.prev_lam[i],
                crisis,
                self.t / T_ROUNDS,
            ]
        return obs

    def step(self, lambdas, moves):
        """
        lambdas: contribution levels [0, 1] for each agent
        moves: movement directions (0=stay, 1=up, 2=down, 3=left, 4=right)
        """
        # Byzantine agents: zero contribution
        lambdas[:N_BYZ] = 0.0

        rewards = np.zeros(N_AGENTS)

        # 1) Harvest + Contribute at current position
        for i in range(N_AGENTS):
            x, y = self.positions[i]
            harvest = ENDOWMENT * (1 - lambdas[i])  # keep for self
            contribution = ENDOWMENT * lambdas[i]    # contribute to public

            # Local public goods: shared among agents at same cell
            agents_here = [j for j in range(N_AGENTS) if
                          self.positions[j][0] == x and self.positions[j][1] == y]
            n_here = len(agents_here)

            # Contributions from all agents at this cell
            total_contrib = sum(ENDOWMENT * lambdas[j] for j in agents_here)
            public_share = (total_contrib * MULTIPLIER) / max(n_here, 1)

            rewards[i] = harvest + public_share

            # Resource depletion from harvesting
            harvest_pressure = sum(1 - lambdas[j] for j in agents_here) / max(n_here, 1)
            self.resources[x, y] -= 0.08 * harvest_pressure * n_here

        # 2) Resource regeneration with tipping points (per cell)
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                R = self.resources[x, y]
                if R < R_CRIT:
                    regen = 0.004  # near-irreversible
                elif R < R_RECOV:
                    regen = 0.012  # hysteresis
                else:
                    regen = 0.025  # normal recovery

                self.resources[x, y] = np.clip(R + regen, 0.0, 1.0)

        # 3) Stochastic shocks (multiple random cells)
        n_shocks = self.rng.poisson(SHOCK_PROB * GRID_SIZE * GRID_SIZE)
        for _ in range(n_shocks):
            sx, sy = self.rng.randint(0, GRID_SIZE), self.rng.randint(0, GRID_SIZE)
            self.resources[sx, sy] = max(0.0, self.resources[sx, sy] - SHOCK_MAG)

        # 4) Global resource drain (environmental degradation)
        self.resources *= 0.998

        # 4) Move agents
        directions = [(0,0), (-1,0), (1,0), (0,-1), (0,1)]
        for i in range(N_AGENTS):
            if moves[i] < len(directions):
                dx, dy = directions[moves[i]]
                nx = np.clip(self.positions[i][0] + dx, 0, GRID_SIZE-1)
                ny = np.clip(self.positions[i][1] + dy, 0, GRID_SIZE-1)
                self.positions[i] = (nx, ny)

        self.t += 1
        self.prev_lam = lambdas.copy()

        # Check collapse: if average resource < 0.01
        avg_R = self.resources.mean()
        collapsed = avg_R < 0.01
        done = collapsed or self.t >= T_ROUNDS

        info = {
            "avg_R": float(avg_R),
            "min_R": float(self.resources.min()),
            "welfare": float(rewards.mean()),
            "mean_lam": float(lambdas[N_BYZ:].mean()),
            "collapsed": collapsed,
        }
        return rewards, done, info

    def _get_obs_after(self):
        return self._get_obs()


# ============================================================
# Policies
# ============================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

class SpatialPolicy:
    def __init__(self, lr=0.003):
        self.W = np.zeros((OBS_DIM, 1))
        self.b = np.zeros(1)
        self.lr = lr

    def forward(self, obs):
        z = obs @ self.W + self.b
        return sigmoid(z.flatten())

    def get_actions(self, obs, rng, noise_scale=0.1):
        lam = self.forward(obs)
        noise = rng.normal(0, noise_scale, size=lam.shape)
        lam = np.clip(lam + noise, 0.0, 1.0)
        moves = rng.randint(0, 5, size=len(obs))  # random movement
        return lam, moves

    def update(self, obs_list, act_list, ret_list):
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


def fixed_policy(obs, phi, rng):
    """Fixed commitment policy (for unconditional / situational)."""
    lam = np.full(len(obs), phi)
    moves = rng.randint(0, 5, size=len(obs))
    return lam, moves

def situational_policy(obs, rng, theta=0.9):
    """Situational commitment: reduce lambda in crisis."""
    lam = np.zeros(len(obs))
    for i in range(len(obs)):
        local_R = obs[i, 2]  # local resource
        if local_R < R_CRIT:
            lam[i] = max(0, np.sin(theta) * 0.3)
        elif local_R > R_RECOV:
            lam[i] = min(1, 1.5 * np.sin(theta))
        else:
            lam[i] = np.sin(theta) * (0.7 + 1.6 * local_R)
    moves = rng.randint(0, 5, size=len(obs))
    return lam, moves


# ============================================================
# Training / Evaluation
# ============================================================
def evaluate_method(name, policy_fn, n_episodes=N_EPISODES, n_seeds=N_SEEDS):
    """Evaluate a fixed or learned policy across seeds."""
    all_data = []

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed * 10000)
        seed_data = []

        for ep in range(n_episodes):
            env = GridSocialDilemma(np.random.RandomState(seed * 10000 + ep))
            obs = env.reset()
            ep_lam = []
            ep_welfare = []

            for t in range(T_ROUNDS):
                lam, moves = policy_fn(obs, rng)
                rewards, done, info = env.step(lam, moves)
                obs = env._get_obs_after()
                ep_lam.append(info["mean_lam"])
                ep_welfare.append(info["welfare"])
                if done:
                    break

            seed_data.append({
                "ep": ep,
                "mean_lam": float(np.mean(ep_lam)),
                "survived": not info["collapsed"],
                "welfare": float(np.mean(ep_welfare)),
                "final_R": info["avg_R"],
            })

        all_data.append(seed_data)

    flat = [m for s in all_data for m in s[-30:]]
    return {
        "name": name,
        "mean_lam": float(np.mean([m["mean_lam"] for m in flat])),
        "survival": float(np.mean([m["survived"] for m in flat])),
        "welfare": float(np.mean([m["welfare"] for m in flat])),
        "final_R": float(np.mean([m["final_R"] for m in flat])),
    }


def train_selfish_rl(n_episodes=N_EPISODES, n_seeds=N_SEEDS):
    """Train selfish RL agent in spatial dilemma."""
    all_data = []

    for seed in range(n_seeds):
        policy = SpatialPolicy(lr=0.003)
        rng = np.random.RandomState(seed * 10000)
        seed_data = []

        for ep in range(n_episodes):
            env = GridSocialDilemma(np.random.RandomState(seed * 10000 + ep))
            obs = env.reset()
            obs_buf, act_buf, rew_buf = [], [], []
            ep_lam = []

            noise = max(0.15 - ep * 0.0005, 0.02)
            for t in range(T_ROUNDS):
                lam, moves = policy.get_actions(obs, rng, noise_scale=noise)
                rewards, done, info = env.step(lam, moves)
                for i in range(N_BYZ, N_AGENTS):
                    obs_buf.append(obs[i])
                    act_buf.append(lam[i])
                    rew_buf.append(rewards[i])
                obs = env._get_obs_after()
                ep_lam.append(info["mean_lam"])
                if done:
                    break

            returns = []
            G = 0
            for r in reversed(rew_buf):
                G = r + GAMMA * G
                returns.insert(0, G)
            policy.update(obs_buf, act_buf, returns)

            seed_data.append({
                "ep": ep,
                "mean_lam": float(np.mean(ep_lam)),
                "survived": not info["collapsed"],
                "welfare": info["welfare"],
            })

            if ep % 100 == 0:
                recent = seed_data[max(0, ep-10):ep+1]
                avg_lam = np.mean([m["mean_lam"] for m in recent])
                avg_surv = np.mean([m["survived"] for m in recent])
                print(f"    [Selfish RL] Seed {seed} Ep {ep:3d} | "
                      f"lam={avg_lam:.3f} surv={avg_surv*100:.0f}%")

        all_data.append(seed_data)

    flat = [m for s in all_data for m in s[-30:]]
    return {
        "name": "Selfish RL",
        "mean_lam": float(np.mean([m["mean_lam"] for m in flat])),
        "survival": float(np.mean([m["survived"] for m in flat])),
        "welfare": float(np.mean([m["welfare"] for m in flat])),
    }


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    OUT = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'spatial_dilemma')
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    print("=" * 65)
    print("  SPATIAL SOCIAL DILEMMA")
    print("  7x7 Grid, N=10, Byz=30%, Non-linear tipping points")
    print("=" * 65)

    results = []

    # 1) Selfish RL
    print("\n--- Selfish RL (learning) ---")
    r = train_selfish_rl()
    results.append(r)
    print(f"  => lam={r['mean_lam']:.3f}, surv={r['survival']*100:.1f}%, W={r['welfare']:.1f}")

    # 2) Fixed lambda=0 (full free-riding)
    print("\n--- Fixed lambda=0.0 (free-riding) ---")
    r = evaluate_method("Free-riding", lambda obs, rng: fixed_policy(obs, 0.0, rng))
    results.append(r)
    print(f"  => lam={r['mean_lam']:.3f}, surv={r['survival']*100:.1f}%, W={r['welfare']:.1f}")

    # 3) Situational commitment
    print("\n--- Situational Commitment ---")
    r = evaluate_method("Situational", lambda obs, rng: situational_policy(obs, rng))
    results.append(r)
    print(f"  => lam={r['mean_lam']:.3f}, surv={r['survival']*100:.1f}%, W={r['welfare']:.1f}")

    # 4) Unconditional commitment (phi=1.0)
    print("\n--- Unconditional Commitment (phi=1.0) ---")
    r = evaluate_method("Unconditional", lambda obs, rng: fixed_policy(obs, 1.0, rng))
    results.append(r)
    print(f"  => lam={r['mean_lam']:.3f}, surv={r['survival']*100:.1f}%, W={r['welfare']:.1f}")

    total = time.time() - t0

    print(f"\n{'='*65}")
    print(f"  SPATIAL DILEMMA RESULTS (Byz=30%, last 30 ep)")
    print(f"{'='*65}")
    print(f"  {'Method':<25} | {'Lambda':>7} | {'Survival':>8} | {'Welfare':>8}")
    print(f"  {'-'*55}")
    for r in results:
        print(f"  {r['name']:<25} | {r['mean_lam']:>7.3f} | "
              f"{r['survival']*100:>7.1f}% | {r['welfare']:>8.1f}")

    output = {"results": results, "time_seconds": float(total),
              "grid_size": GRID_SIZE, "n_agents": N_AGENTS}
    path = os.path.join(OUT, "spatial_results.json")
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Time: {total:.1f}s | Saved: {path}")
