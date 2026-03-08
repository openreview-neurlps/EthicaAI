#!/usr/bin/env python

"""

reproduce_all.py ??One-click reproduction pipeline

====================================================

Runs ALL experiments in the correct order and validates outputs.



Usage:

  python reproduce_all.py           # Full mode (20 seeds, ~4 hours)

  ETHICAAI_FAST=1 python reproduce_all.py   # Fast mode (2 seeds, ~5 minutes)



Exit codes:

  0 ??All experiments completed and validated

  1 ??Some experiment(s) failed

"""

import subprocess

import sys

import os

import json

import time

from pathlib import Path



SCRIPT_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = SCRIPT_DIR.parent / "outputs"



# Ordered list of experiments to run

EXPERIMENTS = [

    {

        "name": "IPPO/MAPPO/IQL (CleanRL-style)",

        "script": "cleanrl_mappo_pgg.py",

        "output": "cleanrl_baselines/cleanrl_baseline_results.json",

    },

    {

        "name": "REINFORCE Nash Trap (Linear/MLP/MLP+Critic)",

        "script": "ppo_nash_trap.py",

        "output": "ppo_nash_trap/ippo_results.json",

    },

    {

        "name": "QMIX (Real Mixing Network)",

        "script": "cleanrl_qmix_real.py",

        "output": "cleanrl_baselines/qmix_real_results.json",

    },

    {

        "name": "LOLA (Opponent-Shaping)",

        "script": "lola_experiment.py",

        "output": "cleanrl_baselines/lola_results.json",

    },

    {

        "name": "???Commitment Floor Sweep",

        "script": "phi1_with_learning.py",

        "output": "phi1_ablation/phi1_results.json",

    },

]



# Optional experiments (skip if script not found)

OPTIONAL = [

    {

        "name": "IQL Baseline",

        "script": "cleanrl_iql_pgg.py",

        "output": "cleanrl_baselines/iql_baseline_results.json",

    },

]





def run_experiment(exp):

    """Run a single experiment and return success/failure."""

    script_path = SCRIPT_DIR / exp["script"]

    output_path = OUTPUT_DIR / exp["output"]

    

    if not script_path.exists():

        print(f"  ??Script not found: {exp['script']}")

        return False

    

    print(f"\n{'??' * 60}")

    print(f"  Running: {exp['name']}")

    print(f"  Script: {exp['script']}")

    print(f"{'??' * 60}")

    

    t0 = time.time()

    result = subprocess.run(

        [sys.executable, str(script_path)],

        cwd=str(SCRIPT_DIR),

        capture_output=False,

        env={**os.environ},

    )

    elapsed = time.time() - t0

    

    if result.returncode != 0:

        print(f"  ??FAILED (exit code {result.returncode}, {elapsed:.0f}s)")

        return False

    

    # Validate output exists and is valid JSON

    if not output_path.exists():

        print(f"  ??Output not found: {exp['output']}")

        return False

    

    try:

        with open(output_path) as f:

            data = json.load(f)

        print(f"  ??Completed in {elapsed:.0f}s ??{exp['output']}")

        return True

    except json.JSONDecodeError:

        print(f"  ??Invalid JSON: {exp['output']}")

        return False





def main():

    print("=" * 60)

    print("  EthicaAI: Full Reproduction Pipeline")

    fast = os.environ.get("ETHICAAI_FAST") == "1"

    print(f"  Mode: {'FAST (2 seeds)' if fast else 'FULL (20 seeds)'}")

    print("=" * 60)

    

    t_total = time.time()

    results = {}

    

    for exp in EXPERIMENTS:

        ok = run_experiment(exp)

        results[exp["name"]] = ok

    

    for exp in OPTIONAL:

        if (SCRIPT_DIR / exp["script"]).exists():

            ok = run_experiment(exp)

            results[exp["name"]] = ok

    

    # Summary

    elapsed_total = time.time() - t_total

    n_ok = sum(1 for v in results.values() if v)

    n_total = len(results)

    

    print(f"\n{'=' * 60}")

    print(f"  REPRODUCTION COMPLETE in {elapsed_total/60:.1f} minutes")

    print(f"  {n_ok}/{n_total} experiments passed")

    print(f"{'??' * 60}")

    for name, ok in results.items():

        status = "[PASS]" if ok else "[FAIL]"

        print(f"  {status} {name}")

    print(f"{'=' * 60}")

    

    sys.exit(0 if n_ok == n_total else 1)





if __name__ == "__main__":

    main()

