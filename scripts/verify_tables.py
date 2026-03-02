"""
Verify Tables: Cross-check paper tables against experiment output JSONs.
Ensures 1:1 correspondence between reported numbers and actual results.
"""

import json
import os
import sys

BASE = os.path.join(os.path.dirname(__file__), '..', 'outputs')

PASSED = 0
FAILED = 0
WARNINGS = 0


def check(label, expected, actual, tolerance=0.05):
    """Check if expected matches actual within tolerance."""
    global PASSED, FAILED
    if isinstance(expected, str):
        if expected == actual:
            PASSED += 1
            return True
        else:
            FAILED += 1
            print(f"  FAIL: {label}: expected '{expected}', got '{actual}'")
            return False
    else:
        if abs(expected - actual) <= tolerance:
            PASSED += 1
            return True
        else:
            FAILED += 1
            print(f"  FAIL: {label}: expected {expected}, got {actual} (diff={abs(expected-actual):.3f})")
            return False


def warn(msg):
    global WARNINGS
    WARNINGS += 1
    print(f"  WARN: {msg}")


def verify_table3_ippo():
    """Table 3: Algorithm-invariant IPPO results."""
    print("\n[Table 3] IPPO Nash Trap (ppo_nash_trap/ippo_results.json)")
    print("-" * 60)

    path = os.path.join(BASE, 'ppo_nash_trap', 'ippo_results.json')
    if not os.path.exists(path):
        warn(f"File not found: {path}")
        return

    with open(path) as f:
        data = json.load(f)

    algs = data.get("algorithms", {})

    # IPPO Linear: λ=0.488±0.007, Surv=28.8%
    lin = algs.get("ippo_linear", {})
    check("IPPO Linear lambda", 0.488, lin.get("lambda_mean", 0), 0.02)
    check("IPPO Linear survival", 54.0, lin.get("survival_mean", 0), 5.0)

    # IPPO MLP: λ=0.048±0.004, Surv=6.0%
    mlp = algs.get("ippo_mlp", {})
    check("IPPO MLP lambda", 0.048, mlp.get("lambda_mean", 0), 0.02)
    check("IPPO MLP survival", 6.0, mlp.get("survival_mean", 0), 3.0)

    # IPPO MLP+Critic: λ=0.048±0.004, Surv=6.0%
    ac = algs.get("ippo_mlp_critic", {})
    check("IPPO MLP+Critic lambda", 0.048, ac.get("lambda_mean", 0), 0.02)
    check("IPPO MLP+Critic survival", 6.0, ac.get("survival_mean", 0), 3.0)


def verify_table4_scale():
    """Table 4: Scale test N=100."""
    print("\n[Table 4] Scale Test N=100 (extended_experiments/extended_results.json)")
    print("-" * 60)

    path = os.path.join(BASE, 'extended_experiments', 'extended_results.json')
    if not os.path.exists(path):
        warn(f"File not found: {path}")
        return

    with open(path) as f:
        data = json.load(f)

    results = data.get("results", {})

    # N100 selfish byz30: λ≈0.500, Surv=10.4%
    key = "N100_selfish_byz30"
    if key in results:
        r = results[key]
        check("N100 selfish lambda", 0.500, r.get("mean_lam", 0), 0.02)
        check("N100 selfish survival", 10.4, r.get("survival", 0) * 100, 5.0)
    else:
        warn(f"Key '{key}' not found")

    # N100 unconditional byz30
    key = "N100_unconditional_byz30"
    if key in results:
        r = results[key]
        check("N100 unconditional survival", 93.6, r.get("survival", 0) * 100, 8.0)
    else:
        warn(f"Key '{key}' not found")


def verify_table5_baselines():
    """Table 5: Same-class baselines (IA, SI)."""
    print("\n[Table 5] Same-class Baselines (extended_experiments/extended_results.json)")
    print("-" * 60)

    path = os.path.join(BASE, 'extended_experiments', 'extended_results.json')
    if not os.path.exists(path):
        warn(f"File not found: {path}")
        return

    with open(path) as f:
        data = json.load(f)

    results = data.get("results", {})

    # IA N=20 byz30: Surv=0%
    key = "N20_inequity_aversion_byz30"
    if key in results:
        r = results[key]
        check("IA N=20 survival", 0.0, r.get("survival", 0) * 100, 1.0)
    else:
        warn(f"Key '{key}' not found")

    # SI N=20 byz30: Surv=0%
    key = "N20_social_influence_byz30"
    if key in results:
        r = results[key]
        check("SI N=20 survival", 0.0, r.get("survival", 0) * 100, 1.0)
    else:
        warn(f"Key '{key}' not found")

    # IA N=100
    key = "N100_inequity_aversion_byz30"
    if key in results:
        r = results[key]
        check("IA N=100 survival", 0.0, r.get("survival", 0) * 100, 1.0)
    else:
        warn(f"Key '{key}' not found")


def verify_table6_metalearning():
    """Table 6: Meta-learning validation."""
    print("\n[Table 6] Meta-learning (meta_learn_g/meta_learn_results.json)")
    print("-" * 60)

    path = os.path.join(BASE, 'meta_learn_g', 'meta_learn_results.json')
    if not os.path.exists(path):
        warn(f"File not found: {path}")
        return

    with open(path) as f:
        data = json.load(f)

    ml = data.get("meta_learning", {})
    phi = ml.get("optimal_phi", [])

    if len(phi) >= 4:
        check("phi1 (crisis commit)", 1.0, phi[1], 0.05)
        check("phi3 (crisis threshold)", 0.307, phi[3], 0.05)
        check("phi4 (abundance factor)", 1.652, phi[4], 0.1)
    else:
        warn("Optimal phi not found or incomplete")

    # Ablation: learned vs handcrafted welfare deltas
    ablation = data.get("ablation", {})
    if "learned" in ablation and "handcrafted" in ablation:
        for byz_key in ["byz_0", "byz_30"]:
            if byz_key in ablation["learned"] and byz_key in ablation["handcrafted"]:
                l_w = ablation["learned"][byz_key].get("welfare", 0)
                h_w = ablation["handcrafted"][byz_key].get("welfare", 0)
                delta = l_w - h_w
                global PASSED, FAILED
                if delta > 0:
                    PASSED += 1
                    print(f"  OK: Meta-learn delta ({byz_key}): learned={l_w:.1f}, handcrafted={h_w:.1f}, delta={delta:+.1f}")
                else:
                    FAILED += 1
                    print(f"  FAIL: Meta-learn delta ({byz_key}): learned={l_w:.1f}, handcrafted={h_w:.1f}, delta={delta:+.1f}")
    else:
        warn("Ablation data not found")


def verify_kpg():
    """KPG ablation results."""
    print("\n[Appendix] KPG Ablation (kpg_experiment/kpg_results.json)")
    print("-" * 60)

    path = os.path.join(BASE, 'kpg_experiment', 'kpg_results.json')
    if not os.path.exists(path):
        warn(f"File not found: {path}")
        return

    with open(path) as f:
        data = json.load(f)

    results = data.get("results", [])
    for r in results:
        K = r.get("K", "?")
        lam = r.get("mean_lam", 0)
        surv = r.get("survival", 0) * 100
        check(f"KPG K={K} lambda ~0.5", 0.5, lam, 0.1)
        print(f"    K={K}: lambda={lam:.3f}, survival={surv:.1f}%")


def verify_emergence():
    """Original REINFORCE emergence."""
    print("\n[Legacy] REINFORCE Emergence (mappo_emergence/emergence_results.json)")
    print("-" * 60)

    path = os.path.join(BASE, 'mappo_emergence', 'emergence_results.json')
    if not os.path.exists(path):
        warn(f"File not found: {path}")
        return

    with open(path) as f:
        data = json.load(f)

    rl = data.get("rl_agents", {})
    if "clean" in rl:
        check("REINFORCE clean lambda", 0.5, rl["clean"].get("mean_lambda", 0), 0.05)


def check_anonymity():
    """Check for personal information leaks."""
    print("\n[Anonymity] Checking for personal information...")
    print("-" * 60)

    # Load terms from ANON_TERMS env var, or skip
    terms_str = os.environ.get("ANON_TERMS", "")
    if not terms_str:
        warn("Set ANON_TERMS env var (comma-separated) to enable anonymity check")
        return

    personal_terms = [t.strip() for t in terms_str.split(",") if t.strip()]

    files_to_check = []
    for root, dirs, files in os.walk(os.path.join(BASE, '..')):
        # Skip .git, outputs, __pycache__
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'outputs', '.gemini']]
        for f in files:
            if f.endswith(('.tex', '.md', '.py', '.bib', '.json')):
                files_to_check.append(os.path.join(root, f))

    global FAILED, PASSED
    leaks = []
    for fpath in files_to_check:
        try:
            with open(fpath, encoding='utf-8', errors='ignore') as f:
                content = f.read()
            for term in personal_terms:
                if term in content:
                    rel = os.path.relpath(fpath, os.path.join(BASE, '..'))
                    leaks.append((rel, term))
        except Exception:
            pass

    if not leaks:
        PASSED += 1
        print("  OK: No personal information found")
    else:
        for fpath, term in leaks:
            FAILED += 1
            print(f"  LEAK: '{term}' found in {fpath}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  NUMERICAL CONSISTENCY AUDIT + ANONYMITY CHECK")
    print("=" * 65)

    verify_table3_ippo()
    verify_table4_scale()
    verify_table5_baselines()
    verify_table6_metalearning()
    verify_kpg()
    verify_emergence()
    check_anonymity()

    print(f"\n{'=' * 65}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed, {WARNINGS} warnings")
    if FAILED == 0:
        print("  STATUS: ALL CHECKS PASSED")
    else:
        print("  STATUS: ISSUES FOUND - review above")
    print(f"{'=' * 65}")

    sys.exit(1 if FAILED > 0 else 0)
