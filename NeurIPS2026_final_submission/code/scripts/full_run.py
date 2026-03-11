"""
Full Run Master — runs ALL experiments with ETHICAAI_FAST=0 (20 seeds).
Outputs: JSON results → ready for update_paper.py to inject into TeX.
"""
import subprocess
import sys
import os
import time

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = [
    ("IPPO Nash Trap", "cleanrl_mappo_pgg.py"),
    ("Coin Game", "coin_game_tpsd.py"),
    ("Partial Obs (+ Full Obs)", "partial_obs_experiment.py"),
    ("Full Obs Baseline", "full_obs_baseline.py"),
    ("Recovery Boundary", "recovery_boundary.py"),
    ("Extended Experiments", "extended_experiments.py"),
]


def run_experiment(name, script):
    path = os.path.join(SCRIPTS_DIR, script)
    if not os.path.exists(path):
        print(f"  [SKIP] {script} not found")
        return False

    print(f"\n{'='*60}")
    print(f"  Running: {name} ({script})")
    print(f"{'='*60}")
    t0 = time.time()

    env = os.environ.copy()
    # CRITICAL: remove FAST flag for full run
    env.pop("ETHICAAI_FAST", None)

    result = subprocess.run(
        [sys.executable, path],
        env=env,
        cwd=SCRIPTS_DIR,
        capture_output=False,
    )

    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAIL (rc={result.returncode})"
    print(f"  {status} in {elapsed:.0f}s")
    return result.returncode == 0


def main():
    print("=" * 60)
    print("  FULL RUN -- ALL EXPERIMENTS (FAST=0, 20 seeds)")
    print("=" * 60)

    t_total = time.time()
    results = {}

    for name, script in EXPERIMENTS:
        ok = run_experiment(name, script)
        results[name] = ok

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  FULL RUN COMPLETE ({elapsed:.0f}s = {elapsed/60:.1f}min)")
    print(f"{'='*60}")
    for name, ok in results.items():
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {name}")

    # Signal completion
    done_path = os.path.join(SCRIPTS_DIR, "..", "outputs", "FULL_RUN_DONE.txt")
    with open(done_path, "w") as f:
        f.write(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {elapsed:.0f}s\n")
        for name, ok in results.items():
            f.write(f"  {'OK' if ok else 'FAIL'}: {name}\n")

    print(f"\n  Signal file: {done_path}")
    print("  Run 'python update_paper.py' to inject results into TeX.")


if __name__ == "__main__":
    main()
