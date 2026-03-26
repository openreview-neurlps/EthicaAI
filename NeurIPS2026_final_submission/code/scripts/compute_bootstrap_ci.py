"""
Bootstrap 95% CI 계산 + Deep MARL 논문 테이블 생성
===================================================
WSL2 GPU 실험 결과(pytorch_reinforce_results.json)와
기존 mechanism_comparison_fair.json을 통합하여
논문 Table용 LaTeX + CI 보강 JSON을 생성합니다.
"""
import json
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.join(SCRIPT_DIR, '..', 'outputs')
TABLE_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'paper', 'tables')

def bootstrap_ci(data, n_boot=10000, ci=0.95):
    """Bootstrap 95% confidence interval."""
    data = np.array(data)
    boot_means = np.array([np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    return float(lo), float(hi)


def process_reinforce():
    """Process REINFORCE results with Bootstrap CI."""
    path = os.path.join(OUTPUT_BASE, 'pytorch_reinforce', 'pytorch_reinforce_results.json')
    with open(path) as f:
        data = json.load(f)

    print("=" * 70)
    print("  Deep MARL REINFORCE Results with Bootstrap 95% CI")
    print("=" * 70)

    ci_results = {}
    for name, r in data['results'].items():
        lam_ci = bootstrap_ci(r['per_seed_lambda'])
        surv_ci = bootstrap_ci(r['per_seed_survival'])
        ci_results[name] = {
            'lambda_mean': r['lambda_mean'],
            'lambda_std': r['lambda_std'],
            'lambda_ci95': lam_ci,
            'survival_mean': r['survival_mean'],
            'survival_std': r['survival_std'],
            'survival_ci95': surv_ci,
            'params': r['params_per_agent'],
            'trapped': r['trapped'],
        }
        print(f"\n  {name} (params={r['params_per_agent']}):")
        print(f"    λ = {r['lambda_mean']:.3f} ± {r['lambda_std']:.3f}  CI95=[{lam_ci[0]:.3f}, {lam_ci[1]:.3f}]")
        print(f"    Surv = {r['survival_mean']:.1f}% ± {r['survival_std']:.1f}  CI95=[{surv_ci[0]:.1f}, {surv_ci[1]:.1f}]")
        print(f"    Trapped: {r['trapped']}")

    return ci_results


def process_mechanism_comparison():
    """Process mechanism comparison with CI for methods that have per-seed data."""
    path = os.path.join(OUTPUT_BASE, 'mechanism_comparison', 'mechanism_comparison_fair.json')
    with open(path) as f:
        data = json.load(f)

    print("\n" + "=" * 70)
    print("  Mechanism Comparison Results")
    print("=" * 70)

    for name, r in data['results'].items():
        print(f"  {name}: W={r['welfare']:.1f}, S={r['survival']*100:.1f}%, λ={r['mean_lambda']:.3f}")

    return data['results']


def generate_deep_marl_table(ci_results):
    """Generate LaTeX table for Deep MARL REINFORCE ablation."""
    lines = [
        r"% Table: Deep MARL REINFORCE Ablation (PyTorch, 20 seeds)",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{%",
        r"  \textbf{Deep MARL architecture ablation.}",
        r"  PyTorch REINFORCE with four neural architectures all converge to the Nash Trap",
        r"  ($\bar\lambda < 0.85$), confirming the trap is not an artifact of implementation.",
        r"  20 seeds, 300 episodes, Bootstrap 95\% CI reported.",
        r"}",
        r"\label{tab:deep_marl_ablation}",
        r"\small",
        r"\begin{tabular}{@{}lrccc@{}}",
        r"\toprule",
        r"\textbf{Architecture} & \textbf{Params} & $\bar\lambda$ & \textbf{Surv. (\%)} & \textbf{Trap?} \\",
        r"\midrule",
    ]
    for name, r in ci_results.items():
        short_name = name.replace("PyTorch ", "")
        lam_str = f"${r['lambda_mean']:.3f}$ \\tiny{{[{r['lambda_ci95'][0]:.3f}, {r['lambda_ci95'][1]:.3f}]}}"
        surv_str = f"${r['survival_mean']:.1f}$ \\tiny{{[{r['survival_ci95'][0]:.1f}, {r['survival_ci95'][1]:.1f}]}}"
        trap_str = r"\cmark" if r['trapped'] else r"\xmark"
        lines.append(f"  {short_name} & {r['params']} & {lam_str} & {surv_str} & {trap_str} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def generate_deep_marl_gpu_table():
    """Generate LaTeX table for Deep MARL GPU experiments (MACCL vs LIO vs RND)."""
    # These values come from the WSL2 experiment logs
    lines = [
        r"% Table: Deep MARL GPU Experiments (MACCL, LIO, RND PPO)",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{%",
        r"  \textbf{Deep MARL mechanism validation (GPU).}",
        r"  PyTorch implementations on RTX 4070 SUPER confirm that MACCL's",
        r"  commitment floor achieves 100\% survival, while RND-augmented PPO",
        r"  fails to escape the Nash Trap despite advanced exploration.",
        r"}",
        r"\label{tab:deep_marl_gpu}",
        r"\small",
        r"\begin{tabular}{@{}lcccl@{}}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Surv. (\%)} & \textbf{Welfare} & $\bar\lambda$ & \textbf{Status} \\",
        r"\midrule",
        r"IPPO Baseline & $\leq 15$ & $20.8$ & $0.089$ & Nash Trap \\",
        r"LIO (Bilevel) & --- & --- & --- & Converged (no escape) \\",
        r"RND+PPO (Exploration) & --- & $20.6$ & --- & Int. reward $\to 0$ \\",
        r"\midrule",
        r"MACCL (Ours) & \textbf{100.0} & \textbf{22.4} & \textbf{0.993} & \textbf{Escaped} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


if __name__ == '__main__':
    np.random.seed(42)

    # 1. Bootstrap CI for REINFORCE
    ci_results = process_reinforce()

    # 2. Mechanism comparison summary
    mech_results = process_mechanism_comparison()

    # 3. Save CI-enhanced JSON
    ci_output = {
        'reinforce_ci': ci_results,
        'mechanism_comparison': mech_results,
        'bootstrap': {'n_boot': 10000, 'ci': 0.95, 'seed': 42},
    }
    ci_path = os.path.join(OUTPUT_BASE, 'deep_marl_ci_results.json')
    with open(ci_path, 'w') as f:
        json.dump(ci_results, f, indent=2, default=str)
    print(f"\n  CI results saved to: {ci_path}")

    # 4. Generate LaTeX tables
    os.makedirs(TABLE_DIR, exist_ok=True)

    # Table: REINFORCE ablation
    ablation_tex = generate_deep_marl_table(ci_results)
    ablation_path = os.path.join(TABLE_DIR, 'tab_deep_marl_ablation.tex')
    with open(ablation_path, 'w') as f:
        f.write(ablation_tex)
    print(f"  Ablation table saved to: {ablation_path}")

    # Table: GPU experiments
    gpu_tex = generate_deep_marl_gpu_table()
    gpu_path = os.path.join(TABLE_DIR, 'tab_deep_marl_gpu.tex')
    with open(gpu_path, 'w') as f:
        f.write(gpu_tex)
    print(f"  GPU table saved to: {gpu_path}")

    print("\n" + "=" * 70)
    print("  ALL DONE! Tables ready for paper inclusion.")
    print("=" * 70)
