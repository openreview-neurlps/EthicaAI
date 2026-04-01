"""M-FOS and POLA baseline comparison in N-agent Nonlinear PGG.

Implements:
  - M-FOS (Model-Free Opponent Shaping, Lu et al. 2022):
    Meta-learning over episode batches. Inner loop: standard REINFORCE.
    Outer loop: meta-gradient through inner learning steps (mean-field approx).
  - POLA (Proximal LOLA, Zhao et al. 2022):
    LOLA with KL-penalty proximal constraint (mu=0.1).
  - Baselines: Selfish REINFORCE, LOLA, Commitment Floor (for reference).

All methods use the same NonlinearPGGEnv with tipping-point dynamics.
N=20, Byz=30%, T=50, 20 seeds, 200 train + 50 eval episodes.
"""
import numpy as np
import json
import os
import time
import copy

N = 20; T = 50; SEEDS = 20; TRAIN_EP = 200; EVAL_EP = 50
M_MULT = 1.6; E = 20.0; BYZ = 0.3; RC = 0.15; RR = 0.25
LR = 0.01; LOLA_LR = 0.005; POLA_MU = 0.1
MFOS_INNER_STEPS = 10; MFOS_OUTER_BATCHES = 20

FAST = os.environ.get("ETHICAAI_FAST", "0") == "1"
if FAST:
    SEEDS = 2; TRAIN_EP = 50; EVAL_EP = 10

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

def env_step(R, cr, rng):
    f = 0.01 if R < RC else (0.03 if R < RR else 0.10)
    shock = 0.15 if rng.random() < 0.05 else 0
    return float(np.clip(R + f * (cr - 0.4) - shock, 0, 1))


# === Agent Classes ===

class LinearAgent:
    """Base linear policy: lambda = sigmoid(w @ obs + b)."""
    def __init__(self):
        self.w = np.zeros(2)
        self.b = 0.0

    def get_params(self):
        return np.concatenate([self.w, [self.b]])

    def set_params(self, p):
        self.w = p[:2].copy()
        self.b = float(p[2])

    def act(self, obs, rng):
        p = sigmoid(self.w @ obs + self.b)
        return float(np.clip(p + rng.normal(0, 0.05), 0, 1))

    def policy_prob(self, obs):
        return float(sigmoid(self.w @ obs + self.b))

    def update(self, r, a, obs):
        p = sigmoid(self.w @ obs + self.b)
        g = a - p
        self.w += LR * r * g * obs
        self.b += LR * r * g


class LOLAAgent(LinearAgent):
    """LOLA: 1-step opponent learning anticipation (mean-field for N agents)."""
    pass


class POLAAgent(LinearAgent):
    """POLA: Proximal LOLA with KL penalty mu=0.1."""
    pass


class MFOSAgent(LinearAgent):
    """M-FOS: Model-free opponent shaping via meta-learning."""
    def __init__(self):
        super().__init__()
        self.meta_w = np.zeros(2)
        self.meta_b = 0.0

    def get_meta_params(self):
        return np.concatenate([self.meta_w, [self.meta_b]])

    def set_meta_params(self, p):
        self.meta_w = p[:2].copy()
        self.meta_b = float(p[2])


class FloorAgent:
    """Commitment floor phi1=1.0."""
    def __init__(self, phi=1.0):
        self.phi = phi
        self.w = np.zeros(2)
        self.b = 0.0

    def act(self, obs, rng):
        p = sigmoid(self.w @ obs + self.b)
        return max(float(np.clip(p + rng.normal(0, 0.05), 0, 1)), self.phi)

    def update(self, r, a, obs):
        p = sigmoid(self.w @ obs + self.b)
        g = a - p
        self.w += LR * r * g * obs
        self.b += LR * r * g


# === Episode Runner ===

def episode(agents, seed, train=False, method='selfish'):
    rng = np.random.RandomState(seed)
    nb = int(N * BYZ)
    R = 0.5
    welf = []; lhist = []; alive = True
    traj = []  # for LOLA/POLA/M-FOS

    for t in range(T):
        obs = np.array([R, t / T])
        lam = np.zeros(N)
        for i in range(N):
            lam[i] = 0.0 if i < nb else agents[i].act(obs, rng)

        c = E * lam
        pg = (c.sum() * M_MULT) / N
        pay = (E - c) + pg

        welf.append(float(pay.mean()))
        lhist.append(float(lam[nb:].mean()))
        traj.append({'obs': obs, 'lam': lam.copy(), 'pay': pay.copy()})

        R = env_step(R, np.mean(c) / E, rng)
        if R <= 0:
            alive = False
            break

        if train and method == 'selfish':
            for i in range(nb, N):
                agents[i].update(pay[i], lam[i], obs)

    return {
        'welfare': float(np.mean(welf)),
        'survival': float(alive),
        'mean_lambda': float(np.mean(lhist)) if lhist else 0.5,
        'traj': traj if train else None,
    }


# === LOLA Training ===

def lola_train_step(agents, traj, nb):
    """1-step LOLA update using mean-field approximation."""
    for t_data in traj:
        obs = t_data['obs']
        lam = t_data['lam']
        pay = t_data['pay']

        # Mean-field: average opponent lambda
        for i in range(nb, N):
            p_i = agents[i].policy_prob(obs)
            # Own gradient
            g_own = (lam[i] - p_i)
            # Opponent anticipation: d(pay_i)/d(lambda_-i) * d(lambda_-i)/d(theta_-i)
            # In PGG: d(pay_i)/d(lambda_j) = E * M / N for all j != i
            dpay_dlamj = E * M_MULT / N
            # Mean-field: average opponent policy gradient
            opp_grads = []
            for j in range(nb, N):
                if j != i:
                    p_j = agents[j].policy_prob(obs)
                    opp_grads.append(p_j * (1 - p_j))
            mean_opp_grad = np.mean(opp_grads) if opp_grads else 0

            # LOLA correction: anticipate opponent learning
            lola_correction = LOLA_LR * dpay_dlamj * (N - nb - 1) * mean_opp_grad

            total_signal = pay[i] * g_own + lola_correction * g_own
            agents[i].w += LR * total_signal * obs
            agents[i].b += LR * total_signal


# === POLA Training ===

def pola_train_step(agents, traj, nb, prev_params):
    """POLA: LOLA + KL-penalty proximal constraint."""
    for t_data in traj:
        obs = t_data['obs']
        lam = t_data['lam']
        pay = t_data['pay']

        for i in range(nb, N):
            p_i = agents[i].policy_prob(obs)
            g_own = (lam[i] - p_i)

            dpay_dlamj = E * M_MULT / N
            opp_grads = []
            for j in range(nb, N):
                if j != i:
                    p_j = agents[j].policy_prob(obs)
                    opp_grads.append(p_j * (1 - p_j))
            mean_opp_grad = np.mean(opp_grads) if opp_grads else 0

            lola_correction = LOLA_LR * dpay_dlamj * (N - nb - 1) * mean_opp_grad

            # KL penalty: pull toward previous params
            p_prev = sigmoid(prev_params[i][:2] @ obs + prev_params[i][2])
            kl_grad = (p_i - p_prev)

            total_signal = pay[i] * g_own + lola_correction * g_own - POLA_MU * kl_grad
            agents[i].w += LR * total_signal * obs
            agents[i].b += LR * total_signal


# === M-FOS Training ===

def mfos_train(agents, seed, nb):
    """M-FOS: meta-learning over episode batches.
    Inner loop: standard REINFORCE for MFOS_INNER_STEPS episodes.
    Outer loop: compute meta-gradient via finite differences."""

    for outer in range(MFOS_OUTER_BATCHES):
        # Save initial params
        init_params = [a.get_params() for a in agents[nb:]]

        # Inner loop: run MFOS_INNER_STEPS episodes with standard learning
        inner_agents = [copy.deepcopy(a) for a in agents[nb:]]
        for step in range(MFOS_INNER_STEPS):
            ep_seed = seed * 10000 + outer * 100 + step
            rng = np.random.RandomState(ep_seed)
            R = 0.5
            for t in range(T):
                obs = np.array([R, t / T])
                lam = np.zeros(N)
                for i in range(nb):
                    lam[i] = 0.0
                for i, ia in enumerate(inner_agents):
                    lam[nb + i] = ia.act(obs, rng)
                c = E * lam
                pg = (c.sum() * M_MULT) / N
                pay = (E - c) + pg
                R = env_step(R, np.mean(c) / E, rng)
                if R <= 0:
                    break
                for i, ia in enumerate(inner_agents):
                    ia.update(pay[nb + i], lam[nb + i], obs)

        # Evaluate after inner loop
        eval_r = episode(
            [LinearAgent() for _ in range(nb)] + inner_agents,
            seed * 10000 + outer * 100 + MFOS_INNER_STEPS,
            train=False
        )
        meta_reward = eval_r['welfare']

        # Finite-difference meta-gradient
        eps_fd = 0.01
        for i in range(len(init_params)):
            for d in range(3):  # w[0], w[1], b
                # Perturb +
                perturbed = [copy.deepcopy(a) for a in agents[nb:]]
                p = perturbed[i].get_params()
                p[d] += eps_fd
                perturbed[i].set_params(p)
                # Run inner loop with perturbation
                for step in range(min(MFOS_INNER_STEPS, 3)):  # shortened for efficiency
                    ep_seed = seed * 10000 + outer * 100 + step
                    rng_p = np.random.RandomState(ep_seed)
                    R_p = 0.5
                    for t in range(T):
                        obs = np.array([R_p, t / T])
                        lam = np.zeros(N)
                        for ii in range(nb):
                            lam[ii] = 0.0
                        for ii, ia in enumerate(perturbed):
                            lam[nb + ii] = ia.act(obs, rng_p)
                        c = E * lam
                        pg = (c.sum() * M_MULT) / N
                        pay = (E - c) + pg
                        R_p = env_step(R_p, np.mean(c) / E, rng_p)
                        if R_p <= 0:
                            break
                        for ii, ia in enumerate(perturbed):
                            ia.update(pay[nb + ii], lam[nb + ii], obs)

                eval_p = episode(
                    [LinearAgent() for _ in range(nb)] + perturbed,
                    seed * 10000 + outer * 100 + MFOS_INNER_STEPS,
                    train=False
                )
                grad_d = (eval_p['welfare'] - meta_reward) / eps_fd

                # Apply meta-gradient to original agent
                p = agents[nb + i].get_params()
                p[d] += LR * 0.1 * grad_d  # smaller meta-lr
                agents[nb + i].set_params(p)


# === Main Runner ===

def run_method(method_name, make_agents, train_fn, seeds):
    """Run a method across seeds."""
    results = []
    for s in range(seeds):
        agents = make_agents()
        train_fn(agents, s)
        # Evaluate
        evals = [episode(agents, 42 + s * 1000 + TRAIN_EP + ep, train=False)
                 for ep in range(EVAL_EP)]
        last = evals[-30:]
        agg = {k: float(np.mean([r[k] for r in last]))
               for k in ['welfare', 'survival', 'mean_lambda']}
        results.append(agg)
        if (s + 1) % 5 == 0:
            print(f"    Seed {s+1}/{seeds}", flush=True)
    return results


if __name__ == '__main__':
    nb = int(N * BYZ)
    all_results = {}
    t0 = time.time()

    # --- 1. Selfish REINFORCE ---
    print("  [Selfish REINFORCE]", flush=True)
    def train_selfish(agents, seed):
        for ep in range(TRAIN_EP):
            episode(agents, 42 + seed * 1000 + ep, train=True, method='selfish')
    rs = run_method('Selfish REINFORCE',
                    lambda: [LinearAgent() for _ in range(N)],
                    train_selfish, SEEDS)
    all_results['Selfish REINFORCE'] = rs

    # --- 2. LOLA ---
    print("  [LOLA]", flush=True)
    def train_lola(agents, seed):
        for ep in range(TRAIN_EP):
            r = episode(agents, 42 + seed * 1000 + ep, train=True, method='lola_collect')
            lola_train_step(agents, r['traj'] or [], nb)
    # Fix: collect trajectory without standard update
    def episode_collect(agents, seed):
        """Run episode and collect trajectory without updating."""
        rng = np.random.RandomState(seed)
        R = 0.5; welf = []; lhist = []; traj = []; alive = True
        for t in range(T):
            obs = np.array([R, t / T])
            lam = np.zeros(N)
            for i in range(N):
                lam[i] = 0.0 if i < nb else agents[i].act(obs, rng)
            c = E * lam; pg = (c.sum() * M_MULT) / N; pay = (E - c) + pg
            welf.append(float(pay.mean())); lhist.append(float(lam[nb:].mean()))
            traj.append({'obs': obs, 'lam': lam.copy(), 'pay': pay.copy()})
            R = env_step(R, np.mean(c) / E, rng)
            if R <= 0: alive = False; break
        return {'welfare': float(np.mean(welf)), 'survival': float(alive),
                'mean_lambda': float(np.mean(lhist)) if lhist else 0.5, 'traj': traj}

    def train_lola_v2(agents, seed):
        for ep in range(TRAIN_EP):
            r = episode_collect(agents, 42 + seed * 1000 + ep)
            lola_train_step(agents, r['traj'], nb)
    rs = run_method('LOLA',
                    lambda: [LOLAAgent() for _ in range(N)],
                    train_lola_v2, SEEDS)
    all_results['LOLA'] = rs

    # --- 3. POLA ---
    print("  [POLA (mu=0.1)]", flush=True)
    def train_pola(agents, seed):
        for ep in range(TRAIN_EP):
            prev_params = [agents[i].get_params() for i in range(N)]
            r = episode_collect(agents, 42 + seed * 1000 + ep)
            pola_train_step(agents, r['traj'], nb, prev_params)
    rs = run_method('POLA',
                    lambda: [POLAAgent() for _ in range(N)],
                    train_pola, SEEDS)
    all_results['POLA'] = rs

    # --- 4. M-FOS ---
    print("  [M-FOS (meta-learning)]", flush=True)
    def train_mfos(agents, seed):
        mfos_train(agents, seed, nb)
        # Additional standard training after meta-initialization
        for ep in range(TRAIN_EP):
            episode(agents, 42 + seed * 1000 + ep, train=True, method='selfish')
    rs = run_method('M-FOS',
                    lambda: [MFOSAgent() for _ in range(N)],
                    train_mfos, SEEDS)
    all_results['M-FOS'] = rs

    # --- 5. Commitment Floor (reference) ---
    print("  [Commitment Floor (phi1=1.0)]", flush=True)
    def train_floor(agents, seed):
        for ep in range(TRAIN_EP):
            episode(agents, 42 + seed * 1000 + ep, train=True, method='selfish')
    rs = run_method('Commitment Floor',
                    lambda: [FloorAgent() if i >= nb else LinearAgent() for i in range(N)],
                    train_floor, SEEDS)
    all_results['Commitment Floor'] = rs

    t_total = time.time() - t0

    # --- Aggregate ---
    summary = {}
    for name, rs in all_results.items():
        agg = {}
        for k in rs[0]:
            vals = [r[k] for r in rs]
            agg[k + '_mean'] = round(float(np.mean(vals)), 3)
            agg[k + '_std'] = round(float(np.std(vals)), 3)
        agg['n_seeds'] = len(rs)
        summary[name] = agg

    # --- Print ---
    print(f"\n{'='*75}")
    print(f"  M-FOS / POLA COMPARISON ({t_total:.0f}s)")
    print(f"{'='*75}")
    print(f"  {'Method':<25} {'lambda':>10} {'Surv%':>10} {'Welfare':>10}")
    print(f"  {'-'*57}")
    for name in ['Selfish REINFORCE', 'LOLA', 'POLA', 'M-FOS', 'Commitment Floor']:
        r = summary[name]
        print(f"  {name:<25} {r['mean_lambda_mean']:.3f}+/-{r['mean_lambda_std']:.3f} "
              f"{r['survival_mean']*100:>7.1f}% {r['welfare_mean']:>9.1f}")

    # --- Save ---
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'outputs', 'mechanism_comparison')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'mfos_pola_results.json')
    with open(path, 'w') as f:
        json.dump({
            'experiment': 'M-FOS and POLA Baseline Comparison',
            'config': {
                'N': N, 'T': T, 'seeds': SEEDS, 'train_ep': TRAIN_EP,
                'eval_ep': EVAL_EP, 'byz': BYZ, 'M': M_MULT, 'E': E,
                'pola_mu': POLA_MU, 'mfos_inner': MFOS_INNER_STEPS,
                'mfos_outer': MFOS_OUTER_BATCHES,
            },
            'results': summary,
            'per_seed': {name: rs for name, rs in all_results.items()},
            'time_seconds': round(t_total, 1),
            'note': ('M-FOS: meta-learning with inner/outer loops (Lu et al. 2022). '
                     'POLA: proximal LOLA with KL penalty mu=0.1 (Zhao et al. 2022). '
                     'Both use mean-field approximation for N=20 scaling. '
                     '1-step opponent anticipation (faithful N-agent requires O(N^2) Hessians).')
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {path}")
