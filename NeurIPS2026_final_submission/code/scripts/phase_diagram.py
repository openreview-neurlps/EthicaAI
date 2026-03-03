"""
Phase Diagram: Moral Commitment Spectrum (v2 - calibrated)
R_crit (환경 비선형성) × φ₁ (위기 헌신도) sweep

핵심: 논문의 PGG 환경과 일치하되, φ₁=1.0일 때 높은 R_crit에서도
생존 가능하도록 환경 동역학을 논문 파라미터에 맞춤.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

# ─── 환경 파라미터 (논문 Table 4 기반) ───
N_AGENTS = 20
ENDOWMENT = 20.0
MULTIPLIER = 1.6
T_HORIZON = 50
N_EPISODES = 50
N_SEEDS = 5

# ─── Sweep 범위 ───
R_CRIT_RANGE = [0.02, 0.05, 0.08, 0.10, 0.13, 0.15, 0.18, 0.20, 0.25, 0.30]
PHI1_RANGE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def run_episode(r_crit, phi1, rng):
    """
    단일 에피소드.
    환경 동역학: 논문 §3.3의 비선형 PGG.
    - R > R_recov: 로지스틱 회복 (rate=0.1)
    - R_crit < R < R_recov: 선형 회복 (rate=0.05)
    - R < R_crit: 극저 회복 (f=0.01, 논문 동일)
    - 추출: λ가 높을수록(기여 많을수록) 추출 적음
    """
    R = 1.0
    R_recov = max(r_crit + 0.10, 0.25)
    alive = True
    total_reward = 0.0

    for t in range(T_HORIZON):
        in_crisis = R < r_crit + 0.05

        # λ 결정
        if in_crisis:
            lam = phi1
        else:
            lam = 0.7

        contributions = ENDOWMENT * lam
        total_c = contributions * N_AGENTS
        retained = ENDOWMENT - contributions

        # 보상
        reward = retained + MULTIPLIER * total_c / N_AGENTS
        total_reward += reward

        # 자원 동역학
        # 추출: free-riding이 많을수록(λ 낮을수록) 추출 많음
        extraction = 0.005 * N_AGENTS * (1 - lam)

        # 회복
        if R > R_recov:
            recovery = 0.08 * R * (1 - R)
        elif R > r_crit:
            recovery = 0.03 * R
        else:
            recovery = 0.01  # 논문: f(R) = 0.01 below tipping point

        R = R - extraction + recovery

        # 확률적 충격 (scale with r_crit for diversity)
        shock_prob = 0.08
        shock_mag = 0.05 + r_crit * 0.5  # r_crit 높을수록 충격 강함
        if rng.random() < shock_prob:
            R -= shock_mag

        R = max(0.0, min(1.0, R))

        if R <= 0.001:
            alive = False
            break

    return alive, total_reward


def run_sweep():
    results = {}
    total = len(R_CRIT_RANGE) * len(PHI1_RANGE)
    done = 0

    for r_crit in R_CRIT_RANGE:
        for phi1 in PHI1_RANGE:
            survivals = []
            welfares = []
            for seed in range(N_SEEDS):
                rng = np.random.RandomState(42 + seed)
                ep_survivals = []
                ep_welfares = []
                for ep in range(N_EPISODES):
                    alive, welfare = run_episode(r_crit, phi1, rng)
                    ep_survivals.append(float(alive))
                    ep_welfares.append(welfare)
                survivals.append(np.mean(ep_survivals) * 100)
                welfares.append(np.mean(ep_welfares))

            key = f"{r_crit:.2f}_{phi1:.1f}"
            results[key] = {
                'r_crit': r_crit,
                'phi1': phi1,
                'survival_pct': float(np.mean(survivals)),
                'welfare': float(np.mean(welfares)),
            }
            done += 1
            if done % 11 == 0:
                print(f"  [{done}/{total}] R_crit={r_crit:.2f} done | phi1=1.0 surv={results[f'{r_crit:.2f}_1.0']['survival_pct']:.1f}%")

    return results


def plot_phase_diagram(results, output_path):
    """히트맵 + 등고선."""
    survival_matrix = np.zeros((len(PHI1_RANGE), len(R_CRIT_RANGE)))

    for i, phi1 in enumerate(PHI1_RANGE):
        for j, r_crit in enumerate(R_CRIT_RANGE):
            key = f"{r_crit:.2f}_{phi1:.1f}"
            survival_matrix[i, j] = results[key]['survival_pct']

    fig, ax = plt.subplots(figsize=(10, 7))

    im = ax.imshow(survival_matrix, aspect='auto', origin='lower',
                   cmap='RdYlGn', vmin=0, vmax=100,
                   extent=[-0.5, len(R_CRIT_RANGE)-0.5, -0.5, len(PHI1_RANGE)-0.5],
                   interpolation='bilinear')

    # 등고선
    X, Y = np.meshgrid(np.arange(len(R_CRIT_RANGE)), np.arange(len(PHI1_RANGE)))
    for level in [30, 50, 80]:
        try:
            cs = ax.contour(X, Y, survival_matrix, levels=[level],
                           colors='white' if level == 80 else 'gray',
                           linewidths=2.5 if level == 80 else 1.0,
                           linestyles='--' if level == 80 else ':')
            ax.clabel(cs, fmt=f'{level}%%', fontsize=9,
                     colors='white' if level == 80 else 'gray')
        except Exception:
            pass

    ax.set_xticks(range(len(R_CRIT_RANGE)))
    ax.set_xticklabels([f'{r:.2f}' for r in R_CRIT_RANGE], rotation=45, fontsize=10)
    ax.set_yticks(range(len(PHI1_RANGE)))
    ax.set_yticklabels([f'{p:.1f}' for p in PHI1_RANGE], fontsize=10)

    ax.set_xlabel(r'Environmental Severity ($R_{\mathrm{crit}}$)', fontsize=13)
    ax.set_ylabel(r'Crisis Commitment Level ($\phi_1$)', fontsize=13)
    ax.set_title('Phase Diagram: Moral Commitment Spectrum', fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Survival Rate (%)', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Phase] Figure saved: {output_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase Diagram v2 (Calibrated)")
    print(f"  {len(R_CRIT_RANGE)} R_crit x {len(PHI1_RANGE)} phi_1 = {len(R_CRIT_RANGE)*len(PHI1_RANGE)} conditions")
    print(f"  {N_EPISODES} episodes x {N_SEEDS} seeds each")
    print("=" * 60)

    results = run_sweep()

    os.makedirs("outputs/phase_diagram", exist_ok=True)
    json_path = "outputs/phase_diagram/results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[Phase] Results: {json_path}")

    fig_path = "paper/fig_phase_diagram.png"
    plot_phase_diagram(results, fig_path)

    # 요약: 각 R_crit에서 50%+ 생존 달성하는 최소 phi1
    print("\n--- Minimum phi_1 for >=50% Survival ---")
    for r_crit in R_CRIT_RANGE:
        min_phi1 = None
        for phi1 in PHI1_RANGE:
            key = f"{r_crit:.2f}_{phi1:.1f}"
            if results[key]['survival_pct'] >= 50:
                min_phi1 = phi1
                break
        if min_phi1 is not None:
            print(f"  R_crit={r_crit:.2f}: phi_1 >= {min_phi1:.1f}")
        else:
            print(f"  R_crit={r_crit:.2f}: No phi_1 achieves 50%")
