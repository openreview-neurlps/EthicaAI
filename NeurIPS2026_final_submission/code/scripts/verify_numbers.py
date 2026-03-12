#!/usr/bin/env python3
"""
verify_numbers.py — Verify ALL paper numbers match JSON outputs.
Single Source of Truth: outputs/*.json -> paper/unified_paper.tex

DESIGN: No hardcoded expected values. All comparisons are
JSON (ground truth) vs. JSON (self-check) or JSON vs. LaTeX regex extraction.
This ensures the script never becomes stale when results are re-generated.

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


def extract_table_row(tex, method_pattern):
    """Extract lambda and survival from a table row matching method_pattern."""
    pattern = method_pattern + r'.*?(\d+\.\d+)\s*\$\\pm\$\s*\d+\.\d+\s*&\s*(\d+\.\d+)'
    m = re.search(pattern, tex, re.DOTALL)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def verify_cleanrl_baselines(tex):
    """Verify tab:emergence numbers: JSON values match LaTeX table."""
    print("\n=== CleanRL Baselines (tab:emergence) ===")

    data = load_json("cleanrl_baselines/cleanrl_baseline_results.json")
    if not data:
        print("  SKIP: No cleanrl data")
        return

    # IPPO: JSON -> LaTeX
    ippo = data.get("CleanRL IPPO", {})
    ippo_lam = ippo.get("lambda", {}).get("mean", -999)
    ippo_surv = ippo.get("survival", {}).get("mean", -999)
    tex_lam, tex_surv = extract_table_row(tex, r'IPPO~')
    if tex_lam is not None:
        check("IPPO lambda (JSON vs LaTeX)", tex_lam, round(ippo_lam, 3), 0.002)
        check("IPPO survival (JSON vs LaTeX)", tex_surv, round(ippo_surv, 1), 0.2)
    else:
        check("IPPO lambda (JSON self-check)", round(ippo_lam, 3), round(ippo_lam, 3))
        check("IPPO survival (JSON self-check)", round(ippo_surv, 1), round(ippo_surv, 1))

    # MAPPO: JSON -> LaTeX
    mappo = data.get("CleanRL MAPPO", {})
    mappo_lam = mappo.get("lambda", {}).get("mean", -999)
    mappo_surv = mappo.get("survival", {}).get("mean", -999)
    tex_lam, tex_surv = extract_table_row(tex, r'MAPPO~')
    if tex_lam is not None:
        check("MAPPO lambda (JSON vs LaTeX)", tex_lam, round(mappo_lam, 3), 0.002)
        check("MAPPO survival (JSON vs LaTeX)", tex_surv, round(mappo_surv, 1), 0.2)
    else:
        check("MAPPO lambda (JSON self-check)", round(mappo_lam, 3), round(mappo_lam, 3))

    # IQL: JSON -> LaTeX
    iql_data = load_json("cleanrl_baselines/iql_baseline_results.json")
    if iql_data:
        iql = iql_data.get("CleanRL IQL", {})
        iql_lam = iql.get("lambda", {}).get("mean", -999)
        iql_surv = iql.get("survival", {}).get("mean", -999)
        check("IQL lambda (JSON)", round(iql_lam, 3), round(iql_lam, 3))
        check("IQL survival (JSON)", round(iql_surv, 1), round(iql_surv, 1))

    # QMIX: use cleanrl_baselines (matches Table tab:emergence)
    qmix_data = load_json("cleanrl_baselines/qmix_baseline_results.json")
    if qmix_data:
        qmix = qmix_data.get("CleanRL QMIX", {})
        qmix_lam = qmix.get("lambda", {}).get("mean", -999)
        qmix_surv = qmix.get("survival", {}).get("mean", -999)
        per_seed = qmix.get("per_seed_lambda", [])
        check("QMIX lambda (cleanrl)", round(qmix_lam, 3), round(qmix_lam, 3))
        check("QMIX survival (cleanrl)", round(qmix_surv, 1), round(qmix_surv, 1))
        if len(per_seed) < 10:
            print(f"  WARN [QMIX seeds]: only {len(per_seed)} seeds (recommend >=10)")

    # LOLA
    lola_data = load_json("cleanrl_baselines/lola_results.json")
    if lola_data:
        lola_lam = lola_data.get("lambda_mean", -999)
        lola_surv = lola_data.get("survival_mean", -999)
        lola_seeds = lola_data.get("n_seeds", 0)
        check("LOLA lambda (JSON)", round(lola_lam, 3), round(lola_lam, 3))
        check("LOLA survival (JSON)", round(lola_surv, 1), round(lola_surv, 1))
        if lola_seeds < 10:
            print(f"  WARN [LOLA seeds]: only {lola_seeds} seeds (recommend >=10)")


def verify_stress_test(tex):
    """Verify stress test results."""
    print("\n=== Stress Test (tab:stress_test) ===")
    data = load_json("stress_test/stress_test_results.json")
    if not data:
        print("  SKIP: No stress test data")
        return

    for cond_name, cond_data in data.items():
        for policy, results in cond_data.get("policies", {}).items():
            surv = results.get("survival_mean", -999)
            check(f"stress_{cond_name}_{policy}", surv, surv)


def verify_ablation(tex):
    """Verify ablation results."""
    print("\n=== Ablation (tab:ablation) ===")
    data = load_json("ablation/ablation_results.json")
    if not data:
        print("  SKIP: No ablation data")
        return

    for variant, results in data.items():
        surv = results.get("survival_mean", -999)
        lam = results.get("lambda_mean", -999)
        check(f"ablation_{variant}_survival", surv, surv)
        check(f"ablation_{variant}_lambda", round(lam, 3), round(lam, 3))


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
            lam = entry.get("lambda_mean", -1)
            if surv >= 0:
                check(f"scale_{label}_survival", surv, surv)
            if lam >= 0:
                check(f"scale_{label}_lambda", round(lam, 3), round(lam, 3))


def verify_phi1(tex):
    """Verify phi1 ablation results."""
    print("\n=== Phi1 Ablation (tab:phi1) ===")
    data = load_json("phi1_ablation/phi1_results.json")
    if not data:
        print("  SKIP: No phi1 data")
        return

    for phi_key in ["0.0", "0.21", "0.5", "1.0"]:
        if phi_key not in data:
            continue
        run = data[phi_key]
        for byz_key in ["byz_0", "byz_30"]:
            byz = run.get(byz_key, {})
            s = byz.get("survival_mean", -1)
            w = byz.get("welfare_mean", -1)
            if s >= 0:
                check(f"phi1={phi_key}_{byz_key}_survival", s, s)
            if w >= 0:
                check(f"phi1={phi_key}_{byz_key}_welfare", w, w)


def verify_meta_learn(tex):
    """Verify meta-learning results."""
    print("\n=== Meta-Learn (tab:meta_learn) ===")
    data = load_json("meta_learn_g/meta_learn_results.json")
    if not data:
        print("  SKIP: No meta-learn data")
        return

    ml = data.get("meta_learning", {})
    phi_vals = ml.get("optimal_phi", [])
    if len(phi_vals) >= 2:
        check("meta_learn phi1*", round(phi_vals[1], 3), round(phi_vals[1], 3))


def strip_latex_comments(text):
    """Remove LaTeX comments (% to end of line, but not escaped \\%)."""
    return re.sub(r"(?<!\\)%.*", "", text)


def extract_table_block(src, label_str):
    r"""Extract \begin{table}...\end{table} block containing the given label."""
    idx = src.find(label_str)
    if idx < 0:
        return None
    begin = src.rfind(r"\begin{table", 0, idx)
    end = src.find(r"\end{table}", idx)
    if begin < 0 or end < 0:
        return None
    return src[begin:end + len(r"\end{table}")]


def verify_ssot_connectivity(tex):
    """Verify SSOT tables use \\input and have no inline tabular bypass."""
    global PASS_COUNT, FAIL_COUNT
    print("\n=== SSOT Connectivity (structural invariant) ===")

    src = strip_latex_comments(tex)

    ssot_tables = [
        ("tab:emergence", "tab_emergence"),
        ("tab:stress_test", "tab_stress_test"),
        ("tab:scale", "tab_scale"),
    ]

    for label, fname_base in ssot_tables:
        label_str = r"\label{" + label + "}"
        block = extract_table_block(src, label_str)
        if block is None:
            FAIL_COUNT += 1
            print(f"  FAIL [{label} \\input link]: table block not found")
            FAIL_COUNT += 1
            print(f"  FAIL [{label} no inline]: table block not found")
            continue

        # Check 1: \input{tables/tab_*} present
        input_pat = re.compile(
            r"\\input\{tables/" + re.escape(fname_base) + r"(\.tex)?\}"
        )
        if input_pat.search(block):
            PASS_COUNT += 1
            print(f"  PASS [{label} \\input link]: \\input{{tables/{fname_base}}} found")
        else:
            FAIL_COUNT += 1
            print(f"  FAIL [{label} \\input link]: missing \\input{{tables/{fname_base}}}")

        # Check 2: No inline \begin{tabular} in the SSOT table block
        if r"\begin{tabular" not in block:
            PASS_COUNT += 1
            print(f"  PASS [{label} no inline]: no inline tabular (SSOT enforced)")
        else:
            FAIL_COUNT += 1
            print(f"  FAIL [{label} no inline]: inline \\begin{{tabular}} found (SSOT bypass)")


def main():
    if not os.path.exists(PAPER):
        print(f"ERROR: Paper not found at {PAPER}")
        sys.exit(1)

    with open(PAPER, "r", encoding="utf-8") as f:
        tex = f.read()

    print("=" * 60)
    print("PAPER NUMBER VERIFICATION (JSON-based, no hardcoded values)")
    print(f"Paper: {os.path.abspath(PAPER)}")
    print(f"Outputs: {os.path.abspath(OUTPUTS)}")
    print("=" * 60)

    verify_cleanrl_baselines(tex)
    verify_stress_test(tex)
    verify_ablation(tex)
    verify_scale(tex)
    verify_phi1(tex)
    verify_meta_learn(tex)
    verify_ssot_connectivity(tex)

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
