"""
Paper 2 Figures: Nash Trap Conceptual Diagram + RL Training Curves

Generates:
  - Fig 1: Nash Trap landscape (lambda vs payoff/survival)
  - Fig 2: RL training curves (lambda over episodes, crisis behavior)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 200,
})

OUT = os.path.join(os.path.dirname(__file__), '..', 'paper')

# ============================================================
# Figure 1: Nash Trap Conceptual Diagram
# ============================================================
def fig1_nash_trap():
    fig, ax1 = plt.subplots(figsize=(7, 4.2))

    lam = np.linspace(0, 1, 200)

    # Individual expected payoff (PGG with free-riding incentive)
    # At lambda=0, keep E but system dies -> 0 long-term
    # At lambda=0.5, moderate payoff, moderate survival
    # At lambda=1.0, lowest individual payoff per round but best survival
    individual_payoff = 20 * (1 - lam) + 1.6 * 20 * (0.3 + 0.7 * lam * 0.7) / 1.0
    # Adjust for collapse risk: low lambda -> system dies -> payoff crashes
    collapse_penalty = np.where(lam < 0.3, (lam / 0.3) ** 2, 1.0)
    adjusted_payoff = individual_payoff * collapse_penalty * 0.6

    # Survival probability (sigmoid around lambda threshold)
    survival = 1.0 / (1.0 + np.exp(-15 * (lam - 0.45)))
    # Under shock + byzantine, survival needs higher lambda
    survival_byz = 1.0 / (1.0 + np.exp(-12 * (lam - 0.75)))

    # Plot individual payoff
    color_payoff = '#2196F3'
    ax1.plot(lam, adjusted_payoff, color=color_payoff, linewidth=2.5,
             label='Individual Payoff (per round)')
    ax1.set_xlabel(r'Commitment Level $\lambda$')
    ax1.set_ylabel('Individual Payoff', color=color_payoff)
    ax1.tick_params(axis='y', labelcolor=color_payoff)

    # Mark Nash Trap
    nash_idx = np.argmax(adjusted_payoff)
    nash_lam = lam[nash_idx]
    ax1.annotate(f'Nash Trap\n($\\lambda \\approx {nash_lam:.2f}$)',
                 xy=(nash_lam, adjusted_payoff[nash_idx]),
                 xytext=(nash_lam - 0.15, adjusted_payoff[nash_idx] + 1.5),
                 fontsize=11, fontweight='bold', color='#D32F2F',
                 arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1.5),
                 ha='center')

    # Plot survival on secondary axis
    ax2 = ax1.twinx()
    color_surv = '#4CAF50'
    ax2.plot(lam, survival * 100, color=color_surv, linewidth=2, linestyle='--',
             label='Survival % (clean)')
    ax2.plot(lam, survival_byz * 100, color='#FF9800', linewidth=2, linestyle=':',
             label='Survival % (Byz 30% + shock)')
    ax2.set_ylabel('System Survival (%)', color=color_surv)
    ax2.tick_params(axis='y', labelcolor=color_surv)
    ax2.set_ylim(-5, 110)

    # Mark Social Optimum
    ax2.annotate('Social Optimum\n($\\lambda = 1.0$)',
                 xy=(1.0, 100), xytext=(0.72, 55),
                 fontsize=11, fontweight='bold', color='#2E7D32',
                 arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5),
                 ha='center')

    # Valley annotation
    ax1.annotate('', xy=(0.85, adjusted_payoff[int(0.85*199)]),
                 xytext=(nash_lam + 0.05, adjusted_payoff[nash_idx] - 0.5),
                 arrowprops=dict(arrowstyle='<->', color='#9E9E9E', lw=1.2))
    ax1.text(0.68, adjusted_payoff[nash_idx] - 2.8,
             '"Cost Valley"\n(individual loss)',
             fontsize=9, color='#757575', ha='center', style='italic')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left',
               framealpha=0.9, edgecolor='#BDBDBD')

    ax1.set_xlim(-0.02, 1.02)
    ax1.set_title('Figure 1: The Nash Trap in Non-linear PGG', fontweight='bold')
    ax1.axvline(x=nash_lam, color='#D32F2F', alpha=0.2, linewidth=8)
    ax1.axvline(x=1.0, color='#4CAF50', alpha=0.2, linewidth=8)

    plt.tight_layout()
    path = os.path.join(OUT, 'fig_p2_nash_trap.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  [Figure 1] Nash Trap -> {path}")
    return path


# ============================================================
# Figure 2: RL Training Curves
# ============================================================
def fig2_training_curves():
    # Load training data
    data_path = os.path.join(os.path.dirname(__file__), '..',
                             'outputs', 'mappo_emergence', 'emergence_results.json')
    with open(data_path, 'r') as f:
        data = json.load(f)

    curves_clean = data["training_curves"]["rl_clean"]  # List of seeds, each a list of dicts
    curves_byz = data["training_curves"]["rl_byz30"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, curves, title, color in [
        (axes[0], curves_clean, 'No Byzantine (Clean)', '#1976D2'),
        (axes[1], curves_byz, '30% Byzantine', '#E64A19'),
    ]:
        # Aggregate across seeds
        all_eps = set()
        for seed in curves:
            for pt in seed:
                all_eps.add(pt["ep"])
        eps = sorted(all_eps)

        # Mean lambda per episode
        lam_per_ep = {e: [] for e in eps}
        cl_per_ep = {e: [] for e in eps}
        surv_per_ep = {e: [] for e in eps}

        for seed in curves:
            for pt in seed:
                lam_per_ep[pt["ep"]].append(pt["lam"])
                if pt["cl"] >= 0:
                    cl_per_ep[pt["ep"]].append(pt["cl"])
                surv_per_ep[pt["ep"]].append(1.0 if pt["surv"] else 0.0)

        ep_arr = np.array(eps)
        lam_mean = np.array([np.mean(lam_per_ep[e]) for e in eps])
        lam_std = np.array([np.std(lam_per_ep[e]) for e in eps])
        cl_mean = np.array([np.mean(cl_per_ep[e]) if cl_per_ep[e] else np.nan for e in eps])
        surv_mean = np.array([np.mean(surv_per_ep[e]) for e in eps])

        # Plot lambda
        ax.fill_between(ep_arr, lam_mean - lam_std, lam_mean + lam_std,
                         alpha=0.15, color=color)
        ax.plot(ep_arr, lam_mean, color=color, linewidth=2,
                label=r'Mean $\lambda$ (all rounds)')

        # Plot crisis lambda
        valid = ~np.isnan(cl_mean)
        if valid.any():
            ax.plot(ep_arr[valid], cl_mean[valid], color='#D32F2F',
                    linewidth=2, linestyle='--',
                    label=r'Crisis $\lambda$ ($R_t < R_{\mathrm{crit}}$)')

        # Oracle line
        ax.axhline(y=1.0, color='#4CAF50', linestyle=':', alpha=0.7,
                   linewidth=1.5, label=r'Oracle ($\lambda=1.0$)')

        # Nash Trap line
        ax.axhline(y=0.5, color='#FF9800', linestyle='-.', alpha=0.7,
                   linewidth=1.5, label=r'Nash Trap ($\lambda=0.5$)')

        # Survival as background
        ax_surv = ax.twinx()
        ax_surv.fill_between(ep_arr, 0, surv_mean * 100,
                              alpha=0.08, color='#4CAF50')
        ax_surv.set_ylim(0, 110)
        if ax == axes[1]:
            ax_surv.set_ylabel('Survival %', fontsize=10, color='#4CAF50')
        ax_surv.tick_params(axis='y', labelcolor='#4CAF50', labelsize=9)

        ax.set_xlabel('Episode')
        ax.set_ylim(-0.05, 1.15)
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    axes[0].set_ylabel(r'Commitment Level $\lambda$')

    fig.suptitle('Figure 2: RL Agents Remain Trapped at $\\lambda \\approx 0.5$',
                 fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT, 'fig_p2_training_curves.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  [Figure 2] Training Curves -> {path}")
    return path


# ============================================================
if __name__ == "__main__":
    print("  Generating Paper 2 Figures...")
    fig1_nash_trap()
    fig2_training_curves()
    print("  Done!")
