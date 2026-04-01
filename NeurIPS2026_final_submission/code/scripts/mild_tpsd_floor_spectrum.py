"""Floor Spectrum across TPSD severity: demonstrates partial floors suffice in mild TPSDs.

Theorem 2 predicts phi1* = (sigma_bar/f(R_crit) + delta) / (1-beta).
This experiment sweeps f_crit and phi1 to verify:
  - Severe (f_crit=0.01): only phi1=1.0 achieves 100% survival
  - Mild (f_crit=0.10): phi1=0.6 already achieves >90% survival
  - Very mild (f_crit=0.50): phi1=0.5 suffices

This directly refutes the "floors are trivially 1.0" criticism.
"""
import numpy as np
import json
import os
import time

N = 20; T = 50; E_ENDOW = 20.0; M_MULT = 1.6; BYZ = 0.3
RC = 0.15; RR = 0.25
SEEDS = 20; TRAIN_EP = 200; EVAL_EP = 50; LR = 0.01

FAST = os.environ.get("ETHICAAI_FAST", "0") == "1"
if FAST:
    SEEDS = 3; TRAIN_EP = 50; EVAL_EP = 10

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

def env_step(R, coop_rate, rng, f_crit):
    if R < RC:
        f = f_crit
    elif R < RR:
        f = max(f_crit, 0.03)
    else:
        f = 0.10
    shock = 0.15 if rng.random() < 0.05 else 0
    return float(np.clip(R + f * (coop_rate - 0.4) - shock, 0, 1))

def episode(agents_w, agents_b, phi1, seed, f_crit, train=False):
    nb = int(N * BYZ)
    rng = np.random.RandomState(seed)
    R = 0.5; welf = []; alive = True
    for t in range(T):
        obs = np.array([R, t / T])
        lam = np.zeros(N)
        for i in range(N):
            if i < nb:
                lam[i] = 0.0
            else:
                base = float(np.clip(sigmoid(agents_w[i] @ obs + agents_b[i]) + rng.normal(0, 0.05), 0, 1))
                lam[i] = max(base, phi1)
        c = E_ENDOW * lam
        pg = (c.sum() * M_MULT) / N
        pay = (E_ENDOW - c) + pg
        welf.append(float(pay.mean()))
        R = env_step(R, np.mean(c) / E_ENDOW, rng, f_crit)
        if R <= 0.001:
            alive = False; break
        if train:
            for i in range(nb, N):
                p = sigmoid(agents_w[i] @ obs + agents_b[i])
                g = lam[i] - p
                agents_w[i] += LR * pay[i] * g * obs
                agents_b[i] += LR * pay[i] * g
    return {'welfare': float(np.mean(welf)), 'survival': float(alive)}


if __name__ == '__main__':
    t0 = time.time()

    f_crit_values = [0.01, 0.05, 0.10, 0.50]
    phi1_values = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Theorem 2: phi1* = (sigma_bar/f_crit + 0.4) / 0.7
    # sigma_bar = p_shock * delta_shock = 0.05 * 0.15 = 0.0075
    sigma_bar = 0.05 * 0.15

    results = {}
    for f_crit in f_crit_values:
        thm2 = min(1.0, (sigma_bar / f_crit + 0.4) / (1 - BYZ))
        print(f"\n  [f_crit={f_crit}] Thm2 prediction: phi1*={thm2:.3f}")
        results[f"f_{f_crit}"] = {'f_crit': f_crit, 'theorem2_phi1': round(thm2, 3), 'sweeps': {}}

        for phi1 in phi1_values:
            survs = []; welfs = []
            for s in range(SEEDS):
                w = np.zeros((N, 2))
                b = np.zeros(N)
                for ep in range(TRAIN_EP):
                    episode(w, b, phi1, 42 + s * 1000 + ep, f_crit, train=True)
                evals = [episode(w, b, phi1, 42 + s * 1000 + TRAIN_EP + ep, f_crit)
                         for ep in range(EVAL_EP)]
                last = evals[-30:]
                survs.append(float(np.mean([e['survival'] for e in last])))
                welfs.append(float(np.mean([e['welfare'] for e in last])))

            sm = float(np.mean(survs)); wm = float(np.mean(welfs))
            results[f"f_{f_crit}"]['sweeps'][f"phi_{phi1}"] = {
                'phi1': phi1, 'survival_mean': round(sm, 3),
                'welfare_mean': round(wm, 2),
                'survival_std': round(float(np.std(survs)), 3),
            }
            marker = " <<<" if abs(phi1 - thm2) < 0.15 else ""
            print(f"    phi1={phi1:.1f}: surv={sm*100:.1f}% welf={wm:.1f}{marker}")

    t_total = time.time() - t0

    # Summary: find minimum phi1 for >90% survival at each f_crit
    print(f"\n{'='*60}")
    print(f"  FLOOR SPECTRUM SUMMARY ({t_total:.0f}s)")
    print(f"{'='*60}")
    print(f"  {'f_crit':>8} {'Thm2':>8} {'Min phi1 for 90%':>18} {'Match?':>8}")
    print(f"  {'-'*44}")
    for key, r in results.items():
        thm2 = r['theorem2_phi1']
        min_phi = None
        for phi_key in sorted(r['sweeps'].keys()):
            sv = r['sweeps'][phi_key]
            if sv['survival_mean'] >= 0.90:
                min_phi = sv['phi1']
                break
        if min_phi is not None:
            match = "YES" if abs(min_phi - thm2) < 0.2 else "~"
            print(f"  {r['f_crit']:>8.2f} {thm2:>8.3f} {min_phi:>18.1f} {match:>8}")
        else:
            print(f"  {r['f_crit']:>8.2f} {thm2:>8.3f} {'N/A':>18} {'':>8}")

    # Save
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'outputs', 'mild_tpsd_spectrum')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'mild_tpsd_spectrum_results.json')
    with open(path, 'w') as f:
        json.dump({
            'experiment': 'Floor Spectrum across TPSD Severity',
            'config': {'N': N, 'T': T, 'seeds': SEEDS, 'byz': BYZ,
                       'f_crit_values': f_crit_values, 'phi1_values': phi1_values},
            'results': results,
            'time_seconds': round(t_total, 1),
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {path}")
