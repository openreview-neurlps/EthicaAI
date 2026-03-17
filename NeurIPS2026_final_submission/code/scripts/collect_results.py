"""Collect and display all experiment results."""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(BASE, '..')

print("=" * 70)
print("  EXPERIMENT RESULTS SUMMARY")
print("=" * 70)

# 1. M/N Sweep
mn_path = os.path.join(CODE_DIR, 'outputs', 'mn_sweep', 'mn_sweep_results.json')
if os.path.exists(mn_path):
    d = json.load(open(mn_path))
    print(f"\n[1] M/N Sweep: {d['config']['N_SEEDS']} seeds x {len(d['results'])} points")
    print(f"  {'M/N':>6s}  {'lam':>10s}  {'Surv%':>8s}  {'Oracle':>8s}  Status")
    for r in d['results']:
        s = "TRAP" if r['trapped'] else "FREE"
        print(f"  {r['mn_ratio']:6.2f}  {r['lambda_mean']:.3f}+/-{r['lambda_std']:.2f}  "
              f"{r['survival_mean']:6.1f}%  {r.get('oracle_survival', 0):6.1f}%  {s}")
    b = d['summary'].get('boundary_mn')
    if b:
        print(f"  >> Boundary: ({b['lower']:.2f}, {b['upper']:.2f})")
    else:
        print("  >> No boundary - trap persists at all M/N!")
else:
    print("\n[1] M/N Sweep: NOT FOUND")

# 2. LOLA
lola_path = os.path.join(CODE_DIR, 'outputs', 'cleanrl_baselines', 'lola_results.json')
if os.path.exists(lola_path):
    d = json.load(open(lola_path))
    print(f"\n[2] LOLA: {d.get('n_seeds', '?')} seeds")
    print(f"  lam={d['lambda_mean']:.3f}+/-{d['lambda_std']:.3f}")
    print(f"  surv={d['survival_mean']:.1f}+/-{d['survival_std']:.1f}%")
else:
    print("\n[2] LOLA: NOT FOUND")

# 3. QMIX
qmix_path = os.path.join(CODE_DIR, 'outputs', 'cleanrl_baselines', 'qmix_real_results.json')
if os.path.exists(qmix_path):
    d = json.load(open(qmix_path))
    print(f"\n[3] QMIX: {d.get('n_seeds', '?')} seeds")
    print(f"  lam={d['lambda_mean']:.3f}+/-{d['lambda_std']:.3f}")
    print(f"  surv={d['survival_mean']:.1f}+/-{d['survival_std']:.1f}%")
else:
    print("\n[3] QMIX: RUNNING or NOT FOUND")

# 4. Impossibility
imp_path = os.path.join(CODE_DIR, 'outputs', 'impossibility', 'impossibility_results.json')
if os.path.exists(imp_path):
    d = json.load(open(imp_path))
    print(f"\n[4] Impossibility Verification:")
    print("  Signal Dilution:")
    for r in d['signal_dilution']:
        print(f"    N={r['N']:3d}: grad={r['gradient_mean']:.4f}  theory={r['expected_1_over_N']:.4f}")
    print("  Basin Dominance:")
    for r in d['basin_dominance']:
        print(f"    N={r['N']:3d}: trap_rate={r['trap_rate']:.2f}  bound={r['expected_lower_bound']:.2f}")
    print("  Escape Time:")
    for r in d['escape_time']:
        print(f"    N={r['N']:3d}: escape_rate={r['escape_rate']:.2f}  mean_time={r['mean_escape_time']:.0f}")
else:
    print("\n[4] Impossibility: NOT FOUND")

# 5. MACCL
maccl_path = os.path.join(CODE_DIR, 'outputs', 'maccl', 'maccl_results.json')
if os.path.exists(maccl_path):
    d = json.load(open(maccl_path))
    print(f"\n[5] MACCL:")
    print(f"  {d['summary']}")
else:
    print("\n[5] MACCL: NOT FOUND")

print(f"\n{'=' * 70}")
