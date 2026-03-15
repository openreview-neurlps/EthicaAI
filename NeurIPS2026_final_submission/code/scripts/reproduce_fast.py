#!/usr/bin/env python
"""
reproduce_fast.py - Quick smoke-test reproduction (NeurIPS submission)
======================================================================
Runs core experiments with 2 seeds for rapid verification (~5 minutes).
For full reproduction (20 seeds), use: python reproduce_all.py

Usage:
  python reproduce_fast.py

Exit codes:
  0 - All smoke tests passed
  1 - Some test(s) failed
"""
import subprocess
import sys
import os
import json
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / os.environ.get("ETHICAAI_OUTDIR", "outputs")

# Core experiments for smoke testing
SMOKE_TESTS = [
    {
        "name": "REINFORCE Nash Trap (3 architectures)",
        "script": "reinforce_nash_trap.py",
        "output": "ppo_nash_trap/ippo_results.json",
    },
    {
        "name": "Commitment Floor Sweep",
        "script": "phi1_with_learning.py",
        "output": "phi1_ablation/phi1_results.json",
    },
]


def run_smoke_test(exp):
    """Run a single experiment in FAST mode and validate output."""
    script_path = SCRIPT_DIR / exp["script"]
    output_path = OUTPUT_DIR / exp["output"]

    if not script_path.exists():
        print(f"  [ERROR] Script not found: {exp['script']}")
        return False

    print(f"\n{'=' * 60}")
    print(f"  Smoke test: {exp['name']}")
    print(f"  Script: {exp['script']}")
    print(f"{'=' * 60}")

    t0 = time.time()
    env = {**os.environ, "ETHICAAI_FAST": "1"}
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(SCRIPT_DIR),
        capture_output=False,
        env=env,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [FAIL] FAILED (exit code {result.returncode}, {elapsed:.0f}s)")
        return False

    if not output_path.exists():
        print(f"  [FAIL] Output not found: {exp['output']}")
        return False

    try:
        with open(output_path) as f:
            data = json.load(f)
        print(f"  [PASS] Passed in {elapsed:.0f}s -> {exp['output']}")
        return True
    except json.JSONDecodeError:
        print(f"  [FAIL] Invalid JSON: {exp['output']}")
        return False


def main():
    print("=" * 60)
    print("  EthicaAI: FAST Smoke Test (2 seeds, ~5 min)")
    print("  For full reproduction: python reproduce_all.py")
    print("=" * 60)

    t_total = time.time()
    results = {}

    for exp in SMOKE_TESTS:
        ok = run_smoke_test(exp)
        results[exp["name"]] = ok

    elapsed_total = time.time() - t_total
    n_ok = sum(1 for v in results.values() if v)
    n_total = len(results)

    print(f"\n{'=' * 60}")
    print(f"  SMOKE TEST COMPLETE in {elapsed_total/60:.1f} minutes")
    print(f"  {n_ok}/{n_total} tests passed")
    print(f"{'=' * 60}")
    for name, ok in results.items():
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status} {name}")
    print(f"{'=' * 60}")

    sys.exit(0 if n_ok == n_total else 1)


if __name__ == "__main__":
    main()
