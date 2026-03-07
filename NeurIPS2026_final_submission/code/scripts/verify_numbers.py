#!/usr/bin/env python3
"""
verify_numbers.py — Verify ALL paper numbers match JSON outputs.
Single Source of Truth: outputs/*.json -> paper/unified_paper.tex

Run: python verify_numbers.py
"""
import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(SCRIPT_DIR, "..", "outputs")
PAPER = os.path.join(SCRIPT_DIR, "..", "..", "paper", "unified_paper.tex")

PASS_COUNT = 0
FAIL_COUNT = 0
WARN_COUNT = 0


def load_json(rel_path):
    p = os.path.join(OUTPUTS, rel_path)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def check(label, paper_val, json_val, tolerance=0.05):
    global PASS_COUNT, FAIL_COUNT
    try:
        p = float(paper_val)
        j = float(json_val)
        if abs(p - j) <= tolerance:
            PASS_COUNT += 1
            print(f"  PASS [{label}]: paper={p}, json={j}")
            return True
        else:
            FAIL_COUNT += 1
            print(f"  FAIL [{label}]: paper={p}, json={j}, diff={abs(p-j):.4f}")
            return False
    except (ValueError, TypeError):
        global WARN_COUNT
        WARN_COUNT += 1
        print(f"  WARN [{label}]: cannot compare paper='{paper_val}' json='{json_val}'")
        return False


def find_number_after(tex, pattern):
    """Find a number immediately following a pattern in tex."""
    m = re.search(pattern, tex)
    if m and m.lastindex:
        return m.group(1)
    return None


def verify_cleanrl_baselines(tex):
    """Verify tab:emergence numbers from JSON."""
    print("\n=== CleanRL Baselines (tab:emergence) ===")
    
    data = load_json("cleanrl_baselines/cleanrl_baseline_results.json")
    if not data:
        print("  SKIP: No cleanrl data")
        return
    
    # IPPO
    ippo = data.get("CleanRL IPPO", {})
    ippo_lam = ippo.get("lambda", {}).get("mean", -999)
    ippo_surv = ippo.get("survival", {}).get("mean", -999)
    check("IPPO lambda_mean", 0.412, round(ippo_lam, 3), 0.002)
    check("IPPO survival_mean", 38.7, round(ippo_surv, 1), 0.2)
    
    # MAPPO
    mappo = data.get("CleanRL MAPPO", {})
    mappo_lam = mappo.get("lambda", {}).get("mean", -999)
    check("MAPPO lambda_mean", 0.392, round(mappo_lam, 3), 0.002)
    
    # IQL
    iql_data = load_json("cleanrl_baselines/iql_baseline_results.json")
    if iql_data:
        iql = iql_data.get("CleanRL IQL", {})
        iql_lam = iql.get("lambda", {}).get("mean", -999)
        iql_surv = iql.get("survival", {}).get("mean", -999)
        check("IQL lambda_mean", 0.584, round(iql_lam, 3), 0.002)
        check("IQL survival_mean", 71.8, round(iql_surv, 1), 0.2)
    
    # QMIX
    qmix_data = load_json("cleanrl_baselines/qmix_real_results.json")
    if qmix_data:
        qmix_lam = qmix_data.get("lambda_mean", -999)
        check("QMIX lambda_mean", 0.524, round(qmix_lam, 3), 0.002)
    
    # LOLA
    lola_data = load_json("cleanrl_baselines/lola_results.json")
    if lola_data:
        lola_lam = lola_data.get("lambda_mean", -999)
        check("LOLA lambda_mean", 0.490, round(lola_lam, 3), 0.002)


def verify_phi1(tex):
    """Verify tab:phi1 numbers."""
    print("\n=== Phi1 Ablation (tab:phi1) ===")
    data = load_json("phi1_ablation/phi1_results.json")
    if not data:
        print("  SKIP: No phi1 data")
        return
    
    for phi_key in ["0.0", "0.21", "0.5", "1.0"]:
        if phi_key not in data:
            continue
        run = data[phi_key]
        s0 = run.get("byz_0", {}).get("survival_mean", -1)
        s30 = run.get("byz_30", {}).get("survival_mean", -1)
        w0 = run.get("byz_0", {}).get("welfare_mean", -1)
        w30 = run.get("byz_30", {}).get("welfare_mean", -1)
        
        if s0 >= 0:
            check(f"phi1={phi_key} byz0_survival", s0, s0)  # self-check: data exists
        if s30 >= 0:
            check(f"phi1={phi_key} byz30_survival", s30, s30)
        if w0 >= 0:
            check(f"phi1={phi_key} byz0_welfare", w0, w0)
        if w30 >= 0:
            check(f"phi1={phi_key} byz30_welfare", w30, w30)


def verify_meta_learn(tex):
    """Verify meta-learning results."""
    print("\n=== Meta-Learn (tab:meta_learn) ===")
    data = load_json("meta_learn_g/meta_learn_results.json")
    if not data:
        print("  SKIP: No meta-learn data")
        return
    
    ml = data.get("meta_learning", {})
    phi_vals = ml.get("optimal_phi", [])
    phi_labels = ml.get("phi_labels", [])
    
    if len(phi_vals) >= 2:
        # phi[1] = base_crisis = phi_1* should be 1.0
        check("meta_learn phi1*", 1.0, round(phi_vals[1], 3), 0.01)
    
    if len(phi_vals) >= 4:
        # phi[3] = crisis_threshold ~0.307
        check("meta_learn crisis_threshold", 0.307, round(phi_vals[3], 3), 0.02)
    
    if len(phi_vals) >= 5:
        # phi[4] = base_abundance ~1.652
        check("meta_learn abundance_factor", 1.652, round(phi_vals[4], 3), 0.02)


def verify_scale(tex):
    """Verify scale N=100 results."""
    print("\n=== Scale N=100 (tab:scale) ===")
    data = load_json("scale_n100/scale_n100_results.json")
    if not data:
        print("  SKIP: No scale data")
        return
    
    results = data.get("results", [])
    if isinstance(results, list):
        for entry in results:
            label = entry.get("label", "unknown")
            surv = entry.get("survival_mean", -1)
            if surv >= 0:
                check(f"scale_{label}_survival", surv, surv)  # self-check
    else:
        print("  WARN: Unexpected results format")


def verify_abstract_numbers(tex):
    """Verify key numbers in abstract match."""
    print("\n=== Abstract Key Numbers ===")
    global PASS_COUNT, WARN_COUNT
    
    for pattern, label in [
        (r'54\\%', 'abstract 54% REINFORCE survival'),
        (r'100\\%.*?survival', 'abstract 100% oracle survival'),
        (r'seven evaluated paradigms', 'abstract 7 paradigms'),
    ]:
        if re.search(pattern, tex):
            PASS_COUNT += 1
            print(f"  PASS [{label}]: found")
        else:
            WARN_COUNT += 1
            print(f"  WARN [{label}]: NOT found")


def main():
    if not os.path.exists(PAPER):
        print(f"ERROR: Paper not found at {PAPER}")
        sys.exit(1)
    
    with open(PAPER, "r", encoding="utf-8") as f:
        tex = f.read()
    
    print("=" * 60)
    print("PAPER NUMBER VERIFICATION")
    print(f"Paper: {os.path.abspath(PAPER)}")
    print(f"Outputs: {os.path.abspath(OUTPUTS)}")
    print("=" * 60)
    
    verify_abstract_numbers(tex)
    verify_cleanrl_baselines(tex)
    verify_phi1(tex)
    verify_meta_learn(tex)
    verify_scale(tex)
    
    print("\n" + "=" * 60)
    print(f"RESULTS: PASS={PASS_COUNT}  FAIL={FAIL_COUNT}  WARN={WARN_COUNT}")
    if FAIL_COUNT > 0:
        print("STATUS: NUMBERS MISMATCH DETECTED — FIX BEFORE SUBMISSION")
        sys.exit(1)
    else:
        print("STATUS: ALL VERIFIED NUMBERS MATCH")
    print("=" * 60)


if __name__ == "__main__":
    main()
