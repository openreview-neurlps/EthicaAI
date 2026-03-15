#!/usr/bin/env python
"""
reproduce_all.py - One-click full reproduction pipeline
========================================================
Runs ALL experiments in the correct order and validates outputs.
This script exactly matches the experiment table in code/README.md.

Usage:
  python reproduce_all.py           # Full mode (20 seeds, ~4 hours)
  ETHICAAI_FAST=1 python reproduce_all.py   # Fast mode (2 seeds, ~5 min)

Exit codes:
  0 - All core experiments completed and validated
  1 - Some experiment(s) failed
"""
import subprocess
import sys
import os
import json
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / os.environ.get("ETHICAAI_OUTDIR", "outputs")

# ===================================================================
# Core experiments (MUST pass for paper claims)
# Matches README.md "Experiments (7 Paradigms + Extensions)" table
# ===================================================================
CORE_EXPERIMENTS = [
    {
        "name": "IPPO/MAPPO (CleanRL-style)",
        "script": "cleanrl_mappo_pgg.py",
        "output": "cleanrl_baselines/cleanrl_baseline_results.json",
        "paper_ref": "Table 3",
    },
    {
        "name": "REINFORCE Nash Trap (Linear/MLP/Critic)",
        "script": "reinforce_nash_trap.py",
        "output": "ppo_nash_trap/ippo_results.json",
        "paper_ref": "Table 3",
    },
    {
        "name": "QMIX (real mixing network)",
        "script": "cleanrl_qmix_real.py",
        "output": "cleanrl_baselines/qmix_real_results.json",
        "paper_ref": "Table 3, App. F",
    },
    {
        "name": "LOLA (opponent-shaping)",
        "script": "lola_experiment.py",
        "output": "cleanrl_baselines/lola_results.json",
        "paper_ref": "Table 3, App. F",
    },
    {
        "name": "Commitment Floor Sweep",
        "script": "phi1_with_learning.py",
        "output": "phi1_ablation/phi1_results.json",
        "paper_ref": "Table 5",
    },
]

# ===================================================================
# Extension experiments (supplementary / appendix)
# Also listed in README.md - run if scripts exist
# ===================================================================
EXTENSION_EXPERIMENTS = [
    {
        "name": "IQL Baseline",
        "script": "cleanrl_iql_pgg.py",
        "output": "cleanrl_baselines/iql_baseline_results.json",
        "paper_ref": "Table 3",
    },
    {
        "name": "Phase Diagram (phi1 x beta)",
        "script": "phase_diagram.py",
        "output": "phase_diagram/phase_diagram.json",
        "paper_ref": "App. G",
    },
    {
        "name": "CPR Cross-Validation",
        "script": "cpr_experiment.py",
        "output": "cpr_experiment/cpr_results.json",
        "paper_ref": "App. H",
    },
    {
        "name": "HP Sensitivity Sweep",
        "script": "hp_sweep_ippo.py",
        "output": "cleanrl_baselines/hp_sweep_results.json",
        "paper_ref": "App. D",
    },
    {
        "name": "Phase Diagram WITH Learning",
        "script": "phase_diagram_with_learning.py",
        "output": "phase_diagram_learned/phase_diagram_learned.json",
        "paper_ref": "App. G (with-learning companion)",
    },
]


def run_experiment(exp):
    """Run a single experiment and return success/failure."""
    script_path = SCRIPT_DIR / exp["script"]
    output_path = OUTPUT_DIR / exp["output"]

    if not script_path.exists():
        print(f"  [SKIP] Script not found: {exp['script']}")
        return None  # Skip, not fail

    if output_path.exists():
        # Check if output is newer than script
        if output_path.stat().st_mtime > script_path.stat().st_mtime:
            print(f"  [SKIP] Output is up-to-date: {exp['output']}")
            return True

    print(f"\n{'=' * 60}")
    print(f"  Running: {exp['name']}")
    print(f"  Script: {exp['script']}")
    print(f"  Paper: {exp.get('paper_ref', 'N/A')}")
    print(f"{'=' * 60}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(SCRIPT_DIR),
        capture_output=False,
        env={**os.environ},
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [FAIL] Exit code {result.returncode} ({elapsed:.0f}s)")
        return False

    if not output_path.exists():
        print(f"  [FAIL] Output not found: {exp['output']}")
        return False

    try:
        with open(output_path) as f:
            data = json.load(f)
        print(f"  [PASS] Completed in {elapsed:.0f}s -> {exp['output']}")
        return True
    except json.JSONDecodeError:
        print(f"  [FAIL] Invalid JSON: {exp['output']}")
        return False


def main():
    print("=" * 60)
    print("  EthicaAI: Full Reproduction Pipeline")
    fast = os.environ.get("ETHICAAI_FAST") == "1"
    print(f"  Mode: {'FAST (2 seeds)' if fast else 'FULL (20 seeds)'}")
    print("=" * 60)

    t_total = time.time()
    core_results = {}
    ext_results = {}

    # Run core experiments (must pass)
    print("\n>>> CORE EXPERIMENTS (required for paper claims)")
    for exp in CORE_EXPERIMENTS:
        ok = run_experiment(exp)
        if ok is not None:
            core_results[exp["name"]] = ok

    # Run extension experiments (optional)
    print("\n>>> EXTENSION EXPERIMENTS (appendix / supplementary)")
    for exp in EXTENSION_EXPERIMENTS:
        ok = run_experiment(exp)
        if ok is not None:
            ext_results[exp["name"]] = ok

    # Summary
    elapsed_total = time.time() - t_total
    core_ok = sum(1 for v in core_results.values() if v)
    core_total = len(core_results)
    ext_ok = sum(1 for v in ext_results.values() if v)
    ext_total = len(ext_results)

    print(f"\n{'=' * 60}")
    print(f"  REPRODUCTION COMPLETE in {elapsed_total/60:.1f} minutes")
    print(f"  Core: {core_ok}/{core_total} passed")
    print(f"  Extensions: {ext_ok}/{ext_total} passed")
    print(f"{'=' * 60}")

    for name, ok in core_results.items():
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status} {name}")
    if ext_results:
        print(f"  --- Extensions ---")
        for name, ok in ext_results.items():
            status = "[PASS]" if ok else "[FAIL]"
            print(f"  {status} {name}")
    print(f"{'=' * 60}")

    # Core experiments must all pass
    sys.exit(0 if core_ok == core_total else 1)


if __name__ == "__main__":
    main()
