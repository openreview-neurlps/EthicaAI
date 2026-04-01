"""Post-experiment: integrate Melting Pot results into the paper.
Run after meltingpot_cleanup_cnn.json is generated.

Two paths:
  A) Positive: floor helps in clean_up → add table to Discussion
  B) Negative: floor doesn't help → strengthen "floors require TPSD" narrative
"""
import json, os, sys

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'meltingpot')

def load_results():
    # clean_up CNN-PPO results
    cleanup_path = os.path.join(RESULTS_DIR, 'meltingpot_cleanup_cnn.json')
    if os.path.exists(cleanup_path):
        with open(cleanup_path) as f:
            cleanup = json.load(f)
    else:
        print("WARNING: meltingpot_cleanup_cnn.json not found")
        cleanup = None

    # commons_harvest CNN-PPO results (already done)
    harvest_path = os.path.join(RESULTS_DIR, 'meltingpot_cnn_ppo.json')
    if os.path.exists(harvest_path):
        with open(harvest_path) as f:
            harvest = json.load(f)
    else:
        harvest = None

    return cleanup, harvest


def analyze(cleanup, harvest):
    print("=" * 60)
    print("  MELTING POT RESULTS ANALYSIS")
    print("=" * 60)

    # commons_harvest (known: floor hurts)
    if harvest:
        print("\n  commons_harvest__open (CNN-PPO):")
        for k, v in harvest['results'].items():
            print(f"    {k}: {v['mean']:.1f} +/- {v['std']:.1f}")

    # clean_up
    if cleanup:
        print(f"\n  clean_up (CNN-PPO):")
        results = cleanup.get('results', {})
        baseline_key = [k for k in results if '0.0' in k or '0' == k.split('_')[-1]]
        floor_keys = [k for k in results if k not in baseline_key]

        baseline_r = None
        for k, v in results.items():
            print(f"    {k}: mean={v['mean']:.1f}")
            if '0.0' in k or '0' == k.split('_')[-1]:
                baseline_r = v['mean']

        # Determine scenario
        if baseline_r is not None:
            best_floor_r = max(v['mean'] for k, v in results.items() if k not in baseline_key) if floor_keys else 0
            if best_floor_r > baseline_r * 1.05:  # 5% improvement
                print(f"\n  >>> SCENARIO A: Floor HELPS (baseline={baseline_r:.1f}, best_floor={best_floor_r:.1f})")
                print("  >>> Action: Add positive Melting Pot table to Discussion")
                return "positive"
            else:
                print(f"\n  >>> SCENARIO B: Floor doesn't help (baseline={baseline_r:.1f}, best_floor={best_floor_r:.1f})")
                print("  >>> Action: Strengthen 'floors require TPSD' narrative")
                return "negative"
    else:
        print("\n  clean_up: NO RESULTS YET")
        return "pending"


def generate_latex_table(cleanup, harvest, scenario):
    """Generate LaTeX table for either scenario."""
    if scenario == "positive":
        # Table showing floor improvement in clean_up
        print("\n  LaTeX table for POSITIVE scenario:")
        print("  (Add to unified_paper.tex Discussion, after 'Floor spectrum' paragraph)")
        print(r"""
\begin{table}[h]
\centering
\caption{Melting Pot CNN-PPO results. In \texttt{clean\_up} (TPSD-like),
  commitment floor \emph{improves} welfare. In \texttt{commons\_harvest\_\_open}
  (non-TPSD), floor \emph{reduces} welfare---validating Theorem~\ref{thm:critical}.}
\small
\begin{tabular}{llcc}
\toprule
\textbf{Substrate} & \textbf{Policy} & \textbf{Reward} \\
\midrule
""")
    elif scenario == "negative":
        print("\n  No new table needed for negative scenario.")
        print("  Current 'Melting Pot validation' paragraph already covers this.")
        print("  Consider adding clean_up results to existing paragraph.")


if __name__ == '__main__':
    cleanup, harvest = load_results()
    scenario = analyze(cleanup, harvest)
    if scenario != "pending":
        generate_latex_table(cleanup, harvest, scenario)
    print("\nDone.")
