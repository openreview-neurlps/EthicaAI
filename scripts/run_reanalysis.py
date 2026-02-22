"""Bootstrap + LMM 재분석 (100-에이전트 결과)"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'simulation', 'jax'))

from analysis.bootstrap_ci import run_bootstrap_analysis, print_bootstrap_report

SWEEP_PATH = "simulation/outputs/run_large_1771038266"

# Find sweep file
sweep_files = [f for f in os.listdir(SWEEP_PATH) if f.startswith("sweep_") and f.endswith(".json")]
sweep_file = os.path.join(SWEEP_PATH, sweep_files[0])

with open(sweep_file, 'r') as f:
    sweep = json.load(f)

print("=== Bootstrap CI Analysis (100-Agent, 70 runs) ===")
results = run_bootstrap_analysis(sweep)
print_bootstrap_report(results)

with open(os.path.join(SWEEP_PATH, "bootstrap_results.json"), 'w') as f:
    json.dump(results, f, indent=2, default=str)
print("Saved to bootstrap_results.json")

# Try LMM
try:
    from analysis.lmm_analysis import run_lmm_analysis, print_lmm_report
    print("\n=== LMM Analysis (100-Agent) ===")
    lmm_results = run_lmm_analysis(sweep)
    print_lmm_report(lmm_results)
    with open(os.path.join(SWEEP_PATH, "lmm_results.json"), 'w') as f:
        json.dump(lmm_results, f, indent=2, default=str)
    print("Saved to lmm_results.json")
except Exception as e:
    print(f"LMM skipped: {e}")
