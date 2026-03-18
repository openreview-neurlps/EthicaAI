"""
Basin Boundary Visualization
=============================
Generates a figure showing the Nash Trap's basin of attraction from init sweep data.
Also plots Phase Transition λ̂(M/N).
"""
import numpy as np
import json
import os

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def plot_basin_boundary():
    """Plot init sweep results showing basin boundary."""
    if not HAS_MPL:
        print("  matplotlib not available, skipping figure generation")
        return None
    
    # Load init sweep data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'init_sweep', 'init_sweep_results.json')
    with open(data_path) as f:
        data = json.load(f)
    
    lambdas_0 = []
    lambdas_f = []
    survivals = []
    trap_rates = []
    
    for key, val in data['results'].items():
        lambdas_0.append(val['init_lambda'])
        lambdas_f.append(val['final_lambda_mean'])
        survivals.append(val['survival_pct'])
        trap_rates.append(val['trap_rate_pct'])
    
    lambdas_0 = np.array(lambdas_0)
    lambdas_f = np.array(lambdas_f)
    survivals = np.array(survivals)
    trap_rates = np.array(trap_rates)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # --- Panel (a): λ_final vs λ_0 ---
    trapped = trap_rates == 100
    escaped = trap_rates == 0
    
    ax1.scatter(lambdas_0[trapped], lambdas_f[trapped], c='#e74c3c', s=120, zorder=5,
                edgecolors='white', linewidths=1.5, label='Trapped (100%)')
    ax1.scatter(lambdas_0[escaped], lambdas_f[escaped], c='#2ecc71', s=120, zorder=5,
                edgecolors='white', linewidths=1.5, label='Escaped (0%)')
    
    # 45-degree reference line
    ax1.plot([0, 1.1], [0, 1.1], 'k--', alpha=0.3, linewidth=1, label='$\\lambda_f = \\lambda_0$')
    
    # Basin boundary shade
    ax1.axvspan(0, 0.6, alpha=0.08, color='#e74c3c', label='Trap basin')
    ax1.axvspan(0.6, 1.05, alpha=0.08, color='#2ecc71', label='Escape basin')
    ax1.axvline(x=0.6, color='#e67e22', linestyle=':', linewidth=2, alpha=0.8)
    ax1.annotate('Basin\nboundary\n$\\lambda^* \\approx 0.6$', xy=(0.6, 0.3),
                fontsize=9, ha='center', color='#e67e22', fontweight='bold')
    
    ax1.set_xlabel('Initial commitment $\\lambda_0$', fontsize=11)
    ax1.set_ylabel('Final commitment $\\lambda_f$', fontsize=11)
    ax1.set_title('(a) Basin of Attraction', fontsize=12, fontweight='bold')
    ax1.set_xlim(-0.02, 1.08)
    ax1.set_ylim(-0.02, 1.08)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.2)
    
    # --- Panel (b): Survival rate vs λ_0 ---
    colors = ['#e74c3c' if t else '#2ecc71' for t in trapped]
    ax2.bar(range(len(lambdas_0)), survivals, color=colors, edgecolor='white', linewidth=0.5)
    ax2.set_xticks(range(len(lambdas_0)))
    ax2.set_xticklabels([f'{l:.2f}' for l in lambdas_0], fontsize=9, rotation=45)
    ax2.set_xlabel('Initial commitment $\\lambda_0$', fontsize=11)
    ax2.set_ylabel('Survival rate (%)', fontsize=11)
    ax2.set_title('(b) Survival by Initialization', fontsize=12, fontweight='bold')
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax2.grid(True, axis='y', alpha=0.2)
    ax2.set_ylim(0, 105)
    
    plt.tight_layout(pad=2.0)
    
    path = os.path.join(OUTPUT_DIR, 'basin_boundary.pdf')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    path_png = os.path.join(OUTPUT_DIR, 'basin_boundary.png')
    plt.savefig(path_png, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {path}")
    print(f"  Saved: {path_png}")
    return path


def plot_phase_transition():
    """Plot λ̂(M/N) with theoretical fit."""
    if not HAS_MPL:
        print("  matplotlib not available, skipping")
        return None
    
    # Load M/N sweep data
    mn_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'mn_sweep', 'mn_sweep_results.json')
    with open(mn_path) as f:
        data = json.load(f)
    
    mn = [r['mn_ratio'] for r in data['results']]
    lam = [r['lambda_mean'] for r in data['results']]
    std = [r['lambda_std'] for r in data['results']]
    
    mn = np.array(mn)
    lam = np.array(lam)
    std = np.array(std)
    
    # Theoretical prediction
    mn_cont = np.linspace(0, 1.1, 200)
    lam_pred = sigmoid(0.09 * (mn_cont - 1) + 0.025)
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    ax.plot(mn_cont, lam_pred, 'b-', linewidth=2, alpha=0.7,
            label='$\\hat\\lambda = \\sigma(0.09(M/N-1)+0.025)$')
    ax.errorbar(mn, lam, yerr=std*1.96, fmt='ko', markersize=7,
                capsize=3, capthick=1.5, linewidth=1.5, label='Empirical (±95% CI)')
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.axvline(x=1.0, color='#e67e22', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.annotate('$M/N=1$\n(rational)', xy=(1.0, 0.48), fontsize=9, ha='center', color='#e67e22')
    
    ax.set_xlabel('Social return ratio $M/N$', fontsize=11)
    ax.set_ylabel('Nash Trap equilibrium $\\hat\\lambda$', fontsize=11)
    ax.set_title('Phase Transition: $\\hat\\lambda(M/N)$', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(0.46, 0.52)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    path = os.path.join(OUTPUT_DIR, 'phase_transition.pdf')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    path_png = os.path.join(OUTPUT_DIR, 'phase_transition.png')
    plt.savefig(path_png, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {path}")
    print(f"  Saved: {path_png}")
    return path


def plot_method_comparison():
    """Bar chart: all methods vs Nash Trap."""
    if not HAS_MPL:
        return None
    
    # Load advanced results
    adv_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'advanced_experiments', 'advanced_results.json')
    with open(adv_path) as f:
        adv = json.load(f)
    
    methods = [
        'REINFORCE',
        'PPO',
        'IQL',
        'QMIX',
        'LOLA',
        'Cheap Talk',
        'Shared Critic',
        'MLP (97p)',
        'Lagrangian RL',
        'MACCL',
        'Fixed $\\phi_1$=1'
    ]
    
    lambda_vals = [
        0.503,  # REINFORCE
        0.501,  # PPO
        0.498,  # IQL
        0.502,  # QMIX
        0.497,  # LOLA
        adv['communication']['final_lambda'],
        adv['shared_critic']['final_lambda'],
        adv['neural_policy']['final_lambda'],
        0.498,  # Lagrangian RL (~CPO/MACPO)
        0.867,  # MACCL
        1.000,  # Fixed floor
    ]
    
    trap_rates = [
        100, 100, 100, 100, 100,
        100, 100, 100, 100,
        0,  # MACCL
        0,  # Fixed floor
    ]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    colors = ['#e74c3c' if t == 100 else '#2ecc71' for t in trap_rates]
    bars = ax.barh(range(len(methods)), lambda_vals, color=colors, edgecolor='white', linewidth=0.5, height=0.7)
    
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10)
    ax.set_xlabel('Final commitment $\\bar\\lambda$', fontsize=11)
    ax.set_title('Nash Trap: All Methods Comparison', fontsize=12, fontweight='bold')
    ax.axvline(x=0.5, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.5, label='Trap ($\\lambda \\approx 0.5$)')
    ax.axvline(x=0.7, color='#e67e22', linestyle=':', linewidth=1.5, alpha=0.5, label='Basin boundary')
    ax.set_xlim(0, 1.1)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, axis='x', alpha=0.2)
    ax.invert_yaxis()
    
    # Add labels
    for i, (v, t) in enumerate(zip(lambda_vals, trap_rates)):
        label = f'λ={v:.3f} {"🔴" if t==100 else "✅"}'
        ax.text(v + 0.02, i, label, va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    path = os.path.join(OUTPUT_DIR, 'method_comparison.pdf')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    path_png = os.path.join(OUTPUT_DIR, 'method_comparison.png')
    plt.savefig(path_png, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {path}")
    return path


if __name__ == "__main__":
    print("=" * 70)
    print("  Generating Visualization Figures")
    print("=" * 70)
    
    basin_path = plot_basin_boundary()
    pt_path = plot_phase_transition()
    comp_path = plot_method_comparison()
    
    print("\n  All figures generated successfully.")
