"""
C1: Recovery rate boundary experiment.
Sweeps f(R_crit) ∈ {0.01, 0.02, 0.03, 0.05, 0.07, 0.10} to find where Nash Trap disappears.
This directly addresses Reviewer 1's W1: "What if f(R_crit) is not 0.01?"
"""
import numpy as np
import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
from envs.nonlinear_pgg_env import NonlinearPGGEnv
from cleanrl_mappo_pgg import MLPActor, MLPCritic, compute_gae, ppo_update_actor, ppo_update_critic, bootstrap_ci

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "recovery_boundary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FAST = os.environ.get("ETHICAAI_FAST") == "1"
N_EPISODES = 150 if not FAST else 15
N_EVAL = 30 if not FAST else 5
N_SEEDS = 10 if not FAST else 3


class TunableRecoveryEnv(NonlinearPGGEnv):
    """PGG environment with tunable f(R_crit) value."""

    def __init__(self, f_crit=0.01, **kwargs):
        super().__init__(**kwargs)
        self.f_crit = f_crit

    def _recovery_rate(self, R):
        if R < self.r_crit:
            return self.f_crit  # tunable!
        elif R < self.r_recov:
            return 0.03
        else:
            return 0.10

    def step(self, actions_honest):
        # Override step to use tunable recovery rate
        if self.terminated:
            return self._get_obs(), np.zeros(self.n_honest), True, False, {}

        lambdas = np.zeros(self.n_agents)
        lambdas[:self.n_honest] = np.clip(actions_honest, 0, 1)

        contribs = lambdas * self.E
        pool = np.sum(contribs)
        payoffs = (self.E - contribs) + self.M * pool / self.n_agents

        mean_c = np.mean(contribs) / self.E
        self.prev_mean_lambda = mean_c

        f_R = self._recovery_rate(self.R)
        shock = self.shock_mag if self.np_random.random() < self.shock_prob else 0.0
        self.R = np.clip(self.R + f_R * (mean_c - 0.4) - shock, 0, 1)

        self.t += 1
        survived = self.R > 0
        terminated = not survived or self.t >= self.T
        self.terminated = terminated

        rewards = payoffs[:self.n_honest].astype(np.float32)
        info = {
            "resource": self.R,
            "survived": survived,
            "mean_lambda": float(np.mean(lambdas[:self.n_honest])),
            "welfare": float(np.mean(payoffs)),
        }

        return self._get_obs(), rewards, terminated, False, info


def run_ippo_tunable(seed, f_crit):
    """Run IPPO on tunable-recovery PGG."""
    rng = np.random.RandomState(seed)
    env = TunableRecoveryEnv(f_crit=f_crit, byz_frac=0.3)
    n = env.n_honest

    actors = [MLPActor(np.random.RandomState(seed * 100 + i)) for i in range(n)]
    critics = [MLPCritic(np.random.RandomState(seed * 100 + i)) for i in range(n)]

    episodes = []
    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        obs_buf = [[] for _ in range(n)]
        act_buf = [[] for _ in range(n)]
        lp_buf = [[] for _ in range(n)]
        rew_buf = [[] for _ in range(n)]
        val_buf = [[] for _ in range(n)]

        for t in range(50):
            actions = np.zeros(n)
            for i in range(n):
                a, lp, mu = actors[i].act(obs, rng)
                actions[i] = a
                obs_buf[i].append(obs.copy())
                act_buf[i].append(a)
                lp_buf[i].append(lp)
                val_buf[i].append(critics[i].forward(obs))

            obs, rewards, done, _, info = env.step(actions)
            for i in range(n):
                rew_buf[i].append(rewards[i])
            if done:
                break

        for i in range(n):
            if len(rew_buf[i]) < 2:
                continue
            adv, ret = compute_gae(rew_buf[i], val_buf[i])
            if np.std(adv) > 1e-8:
                adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
            for _ in range(4):
                ppo_update_actor(actors[i], obs_buf[i], act_buf[i], lp_buf[i], adv)
                ppo_update_critic(critics[i], obs_buf[i], ret)

        episodes.append({
            "mean_lambda": info.get("mean_lambda", 0),
            "survived": info.get("survived", False),
            "welfare": info.get("welfare", 0),
        })

    ev = episodes[-N_EVAL:]
    return {
        "lambda": float(np.mean([d["mean_lambda"] for d in ev])),
        "survival": float(np.mean([float(d["survived"]) for d in ev]) * 100),
        "welfare": float(np.mean([d["welfare"] for d in ev])),
    }


def main():
    print("=" * 60)
    print("  C1: Recovery Rate Boundary Experiment")
    print(f"  Seeds={N_SEEDS}, Episodes={N_EPISODES}")
    print("=" * 60)

    f_crits = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50]
    results = {}

    for fc in f_crits:
        print(f"\n  f(R_crit)={fc:.2f}, {N_SEEDS} seeds...")
        seed_data = []
        for s in range(N_SEEDS):
            r = run_ippo_tunable(s * 7 + 42, fc)
            seed_data.append(r)
            sys.stdout.write(".")
            sys.stdout.flush()

        lams = [r["lambda"] for r in seed_data]
        survs = [r["survival"] for r in seed_data]

        results[f"f_crit_{fc:.2f}"] = {
            "f_crit": fc,
            "lambda_mean": float(np.mean(lams)),
            "lambda_std": float(np.std(lams)),
            "survival_mean": float(np.mean(survs)),
            "survival_std": float(np.std(survs)),
            "welfare_mean": float(np.mean([r["welfare"] for r in seed_data])),
            "nash_trap": float(np.mean(lams)) < 0.95,
            "phi1_star": float((0.0075 / fc + 0.4) / 0.7),  # Theorem 1:
        }
        trapped = "TRAPPED" if np.mean(lams) < 0.95 else "ESCAPED"
        phi1 = (0.0075 / fc + 0.4) / 0.7
        print(f"\n    f={fc:.2f}: lam={np.mean(lams):.3f} Surv={np.mean(survs):.0f}% "
              f"phi1*={phi1:.2f} {trapped}")

    out_path = OUTPUT_DIR / "recovery_boundary_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  BOUNDARY TABLE (for paper)")
    print(f"{'=' * 60}")
    print(f"  {'f(R_crit)':>10} | {'φ₁* (Thm1)':>10} | {'λ (IPPO)':>10} | {'Surv%':>6} | Nash Trap?")
    print(f"  {'-'*10} | {'-'*10} | {'-'*10} | {'-'*6} | {'-'*10}")
    for k, v in results.items():
        phi1 = v["phi1_star"]
        trapped = "Yes" if v["nash_trap"] else "No"
        print(f"  {v['f_crit']:>10.2f} | {phi1:>10.2f} | {v['lambda_mean']:>10.3f} | {v['survival_mean']:>5.0f}% | {trapped}")

    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
