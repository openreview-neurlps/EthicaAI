"""
reproduce_quick.py -- One-Command Core Experiment Reproduction
==============================================================
Reproduces ALL key results from the paper in a single command.

Usage:
  python scripts/reproduce_quick.py           # Full reproduction
  python scripts/reproduce_quick.py --fast    # Quick smoke test (2 seeds)

Expected runtime:
  --fast:  ~2 minutes
  full:    ~15 minutes (CPU)

Experiments reproduced:
  1. Table 3: Nash Trap (Ind. REINFORCE x3 + True IPPO + MAPPO + QMIX)
  2. Table 5: Scale test N=100
  3. DNN Ablation (Appendix)
"""

import subprocess
import sys
import os
import time
import json
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "reproduce")


def run_script(name, script, extra_env=None):
    """Run a script and return success/failure."""
    script_path = os.path.join(SCRIPT_DIR, script)
    if not os.path.exists(script_path):
        print(f"  [SKIP] {name}: {script} not found")
        return False, 0

    print(f"\n  [{name}] Running {script}...")
    t0 = time.time()
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=SCRIPT_DIR,
        env=env,
        capture_output=True,
        text=True,
        errors='replace'
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        # Print last 5 lines of output
        lines = result.stdout.strip().split('\n')
        for line in lines[-5:]:
            print(f"    {line}")
        print(f"  [{name}] OK ({elapsed:.0f}s)")
    else:
        print(f"  [{name}] FAILED ({elapsed:.0f}s)")
        # Print error
        err_lines = (result.stderr or result.stdout or "").strip().split('\n')
        for line in err_lines[-5:]:
            print(f"    ERR: {line}")

    return result.returncode == 0, elapsed


def main():
    parser = argparse.ArgumentParser(description="EthicaAI Quick Reproduction")
    parser.add_argument("--fast", action="store_true", help="Quick smoke test (2 seeds)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  EthicaAI -- Core Experiment Reproduction")
    print("  NeurIPS 2026")
    print(f"  Mode: {'FAST (smoke test)' if args.fast else 'FULL'}")
    print("=" * 65)

    t0 = time.time()
    results = []

    # 1. Table 3: Nash Trap (Ind. REINFORCE)
    ok, t = run_script("Table 3: Ind. REINFORCE", "ppo_nash_trap.py")
    results.append(("Table 3: Ind. REINFORCE (Linear/MLP/Critic)", ok, t))

    # 2. Table 3 extended: Strong MARL baselines (True IPPO, MAPPO, QMIX)
    ok, t = run_script("Table 3: Strong MARL", "p3_fast_baselines.py")
    results.append(("Table 3: True IPPO + MAPPO + QMIX", ok, t))

    # 3. DNN Ablation
    ok, t = run_script("DNN Ablation", "dnn_ablation.py")
    results.append(("Appendix: DNN Ablation", ok, t))

    # 4. KPG Experiment
    ok, t = run_script("KPG Experiment", "kpg_experiment.py")
    results.append(("Appendix: KPG Experiment", ok, t))

    # 5. Scale test N=100
    ok, t = run_script("Scale Test N=100", "scale_test_n100.py")
    results.append(("Table 5: Scale Test N=100", ok, t))

    total = time.time() - t0

    # Summary
    print(f"\n\n{'='*65}")
    print(f"  REPRODUCTION SUMMARY ({total:.0f}s total)")
    print(f"{'='*65}")

    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)

    for name, ok, t in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name:45s} ({t:.0f}s)")

    print(f"\n  {passed} passed, {failed} failed")
    print(f"  Total runtime: {total:.0f}s")

    # Save
    summary = {
        "total_time": total,
        "results": [{"name": n, "passed": ok, "time": t} for n, ok, t in results],
        "passed": passed,
        "failed": failed,
    }
    with open(os.path.join(OUTPUT_DIR, "reproduction_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
