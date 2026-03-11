"""
update_paper.py — Reads JSON experiment results and updates unified_paper.tex
Run this AFTER full_run.py completes to inject 20-seed results into the paper.

Usage:
  python update_paper.py              # Preview changes
  python update_paper.py --apply      # Apply changes to TeX file
"""
import json
import re
import sys
import os
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent / "paper" / "unified_paper.tex"
OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"

APPLY = "--apply" in sys.argv


def load_json(relpath):
    p = OUTPUTS / relpath
    if not p.exists():
        print(f"  [SKIP] {relpath} not found")
        return None
    with open(p) as f:
        return json.load(f)


def update_table_emergence(tex, data):
    """Update Table 1 (tab:emergence) IPPO/MAPPO/IQL/QMIX rows."""
    if not data:
        return tex
    algs = data.get("algorithms", {})
    updates = {}
    
    # Map algorithm keys to table row patterns
    mappings = {
        "ippo": (r"IPPO~.*?\{schulman2017proximal\}", "ippo"),
        "mappo": (r"MAPPO~.*?\{yu2022surprising\}", "mappo"),
    }
    
    for key, alg_data in algs.items():
        lam = alg_data.get("lambda_mean", 0)
        lam_std = alg_data.get("lambda_std", 0)
        surv = alg_data.get("survival_mean", 0)
        welf = alg_data.get("welfare_mean", 0)
        print(f"  {key}: lam={lam:.3f}+/-{lam_std:.3f}, surv={surv:.1f}%, W={welf:.1f}")
    
    return tex


def update_coin_game(tex, data):
    """Update Coin Game table values."""
    if not data:
        return tex
    
    results = data.get("results", {})
    replacements = []
    
    for policy_key, row_pattern in [
        ("selfish", r"Selfish \(greedy\)\s*&\s*\$[\d.]+\$\s*&\s*\$[\d.]+\$"),
        ("fixed_0.7", r"Committed \$\\phi_1\{=\}0\.7\$\s*&\s*\$[\d.]+\$\s*&\s*\$[\d.]+\$"),
        ("fixed_1.0", r"Committed \$\\phi_1\{=\}1\.0\$\s*&\s*\$[\d.]+\$\s*&\s*\$[\d.]+\$"),
        ("acl", r"ACL \$\\phi_1\(R\)\$\s*&\s*\$[\d.]+\$\s*&\s*\$[\d.]+\$"),
    ]:
        if policy_key in results:
            w = results[policy_key]["welfare"]
            s = results[policy_key]["survival"]
            
            label_map = {
                "selfish": "Selfish (greedy)",
                "fixed_0.7": "Committed $\\phi_1{=}0.7$",
                "fixed_1.0": "Committed $\\phi_1{=}1.0$",
                "acl": "ACL $\\phi_1(R)$",
            }
            label = label_map[policy_key]
            new_row = f"{label} & ${w}$ & ${s}$"
            
            match = re.search(row_pattern, tex)
            if match:
                old = match.group(0)
                tex = tex.replace(old, new_row)
                print(f"  Coin Game {policy_key}: W={w}, S={s}%")
    
    return tex


def update_partial_obs(tex, data):
    """Update Table 8 (partial_obs) with full_obs baseline."""
    if not data:
        return tex
    
    full_obs = data.get("full_obs")
    if full_obs:
        ippo_surv = round(full_obs["ippo"]["survival_mean"])
        sit_surv = round(full_obs["situational"]["survival_mean"])
        unc_surv = round(full_obs["unconditional"]["survival_mean"])
        
        old = re.search(
            r"Full Obs \(Baseline\)\s*&\s*\d+\\%\s*&\s*\d+\\%\s*&\s*\d+\\%",
            tex
        )
        if old:
            new = f"Full Obs (Baseline) & {ippo_surv}\\% & {sit_surv}\\% & {unc_surv}\\%"
            tex = tex.replace(old.group(0), new)
            print(f"  Full Obs: IPPO={ippo_surv}%, Sit={sit_surv}%, Unc={unc_surv}%")
    
    return tex


def update_recovery_boundary(tex, data):
    """Update recovery boundary table."""
    if not data:
        return tex
    
    for key, vals in data.items():
        fc = vals["f_crit"]
        lam = vals["lambda_mean"]
        surv = vals["survival_mean"]
        phi1 = vals["phi1_star"]
        trapped = "Yes" if vals["nash_trap"] else "No"
        print(f"  f={fc:.2f}: lam={lam:.3f}, surv={surv:.0f}%, phi1*={phi1:.2f}, {trapped}")
    
    return tex


def main():
    print("=" * 60)
    print("  Paper Update Script")
    print(f"  Mode: {'APPLY' if APPLY else 'PREVIEW'}")
    print("=" * 60)
    
    tex = PAPER.read_text(encoding="utf-8")
    original = tex
    
    # 1. CleanRL baselines
    print("\n[1] CleanRL Baselines:")
    bl = load_json("cleanrl_baselines/cleanrl_baseline_results.json")
    if bl:
        for algo_name, algo_data in bl.items():
            if isinstance(algo_data, dict) and "lambda_mean" in algo_data:
                print(f"  {algo_name}: lam={algo_data['lambda_mean']:.3f}, "
                      f"surv={algo_data.get('survival_mean', 'N/A')}")
    
    # 2. Coin Game
    print("\n[2] Coin Game:")
    cg = load_json("coin_game/coin_game_results.json")
    tex = update_coin_game(tex, cg)
    
    # 3. Partial Obs
    print("\n[3] Partial Observability:")
    po = load_json("partial_obs/partial_obs_results.json")
    tex = update_partial_obs(tex, po)
    
    # 4. Recovery Boundary
    print("\n[4] Recovery Boundary:")
    rb = load_json("recovery_boundary/recovery_boundary_results.json")
    tex = update_recovery_boundary(tex, rb)
    
    # Summary
    changed = tex != original
    print(f"\n{'='*60}")
    print(f"  Changes detected: {'YES' if changed else 'NO'}")
    
    if APPLY and changed:
        PAPER.write_text(tex, encoding="utf-8")
        print(f"  Written to: {PAPER}")
        print("  Run pdflatex to rebuild PDF.")
    elif APPLY and not changed:
        print("  No changes to apply.")
    else:
        print("  Run with --apply to write changes.")


if __name__ == "__main__":
    main()
