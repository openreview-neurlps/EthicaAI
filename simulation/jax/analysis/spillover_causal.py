"""
W1: Causal Inference under Interference — Spillover Effect Decomposition
EthicaAI Phase W — NeurIPS 2026 Critique Defense

SUTVA(Stable Unit Treatment Value Assumption) 위반 대응:
  - PGG: SUTVA 성립 (독립적 기여 결정) → Clean ATE
  - Grid World: SUTVA 위반 (공간 상호작용) → AME + Spillover 분리
  - Network PGG: 부분 간섭 (이웃만 영향) → Exposure Mapping

이론적 기반:
  - Hudgens & Halloran (2008): Causal inference under interference
  - Aronow & Samii (2017): Exposure mapping for network experiments

τ_direct  = E[Y | d_i=1, e_i] - E[Y | d_i=0, e_i]
τ_spillover = E[Y | d_i, e_i=high] - E[Y | d_i, e_i=low]

여기서 e_i = (이웃의 meta-ranking 비율) = exposure

출력: Fig 75-78, spillover_results.json
"""

import os
import sys
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "ETHICAAI_OUTPUT_DIR", "simulation/outputs/reproduce"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- NeurIPS 스타일 ---
plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 10, 'figure.dpi': 300,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# === 설정 ===
N_SEEDS = 10
N_STEPS = 200
ENDOWMENT = 100
MPCR = 1.6

# 환경별 설정
ENV_CONFIGS = {
    "pgg": {
        "n_agents": 50,
        "has_spatial": False,
        "has_network": False,
        "description": "N-Player PGG (SUTVA holds)",
    },
    "grid_world": {
        "n_agents": 49,  # 7×7 grid = 49 agents
        "has_spatial": True,
        "has_network": False,
        "grid_size": 7,
        "description": "Grid World Cleanup (SUTVA violated)",
    },
    "network_pgg": {
        "n_agents": 50,
        "has_spatial": False,
        "has_network": True,
        "k_neighbors": 6,
        "rewire_prob": 0.1,
        "description": "Small-World Network PGG (Partial interference)",
    },
}

SVO_THETAS = {
    "individualist": math.pi / 12,
    "prosocial": math.pi / 4,
    "altruistic": 5 * math.pi / 12,
}


# ============================================================
# 1. 네트워크/공간 구조 생성
# ============================================================

def build_adjacency_pgg(n):
    """PGG: 전체 연결 (모두가 같은 그룹) — 사실상 간섭 없음 (동시 결정)."""
    return np.ones((n, n)) - np.eye(n)


def build_adjacency_grid(n, grid_size):
    """Grid World: 4-이웃 격자 (상하좌우 인접 에이전트만 간섭)."""
    adj = np.zeros((n, n))
    for i in range(n):
        row, col = divmod(i, grid_size)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                j = nr * grid_size + nc
                if j < n:  # 에이전트 수 범위 내만 연결
                    adj[i, j] = 1.0
    return adj


def build_adjacency_small_world(n, k, p):
    """Small-World 네트워크 (Watts-Strogatz)."""
    adj = np.zeros((n, n))
    # Ring lattice
    for i in range(n):
        for j in range(1, k // 2 + 1):
            adj[i, (i + j) % n] = 1.0
            adj[i, (i - j) % n] = 1.0
    # Rewire
    rng = np.random.RandomState(42)
    for i in range(n):
        for j in range(1, k // 2 + 1):
            if rng.random() < p:
                target = (i + j) % n
                adj[i, target] = 0.0
                new_target = rng.randint(0, n)
                while new_target == i or adj[i, new_target] == 1.0:
                    new_target = rng.randint(0, n)
                adj[i, new_target] = 1.0
    return adj


def build_adjacency(env_type, n):
    """환경별 adjacency matrix 생성."""
    config = ENV_CONFIGS[env_type]
    if config["has_spatial"]:
        return build_adjacency_grid(n, config["grid_size"])
    elif config["has_network"]:
        return build_adjacency_small_world(n, config["k_neighbors"], config["rewire_prob"])
    else:
        return build_adjacency_pgg(n)


# ============================================================
# 2. Meta-Ranking 시뮬레이션 (Treatment Vector 전체 기록)
# ============================================================

def compute_lambda(svo_theta, resource, use_meta=True):
    """Sen's dynamic λ_t 계산."""
    lam_base = math.sin(svo_theta)
    if not use_meta:
        return lam_base
    if resource < 0.2:
        return max(0.0, lam_base * 0.3)
    elif resource > 0.7:
        return min(1.0, 1.5 * lam_base)
    else:
        return lam_base * (0.7 + 1.6 * resource)


def simulate_with_treatment_vector(env_type, svo_theta, seed, meta_fraction=0.5):
    """
    시뮬레이션 + 전체 treatment vector 기록.
    
    meta_fraction: meta-ranking을 적용받는 에이전트 비율
    나머지는 baseline (static λ = sin(θ)).
    
    Returns:
        panel_data: list of dicts (agent_id, step, treatment, exposure,
                    outcome, resource, env)
    """
    config = ENV_CONFIGS[env_type]
    n = config["n_agents"]
    rng = np.random.RandomState(seed)
    adj = build_adjacency(env_type, n)
    
    # Treatment 할당: 앞 meta_fraction만큼 meta-ranking ON
    n_treated = int(n * meta_fraction)
    treatments = np.array([1] * n_treated + [0] * (n - n_treated))
    rng.shuffle(treatments)
    
    resource = 0.5
    panel_data = []
    wealth = np.zeros(n)
    
    for t in range(N_STEPS):
        # 각 에이전트의 λ 결정
        lambdas = np.zeros(n)
        for i in range(n):
            if treatments[i] == 1:
                lambdas[i] = compute_lambda(svo_theta, resource, use_meta=True)
            else:
                lambdas[i] = compute_lambda(svo_theta, resource, use_meta=False)
        
        # 기여금 결정
        contributions = np.clip(lambdas * ENDOWMENT * 0.8 + rng.normal(0, 3, n), 0, ENDOWMENT)
        
        # Grid World: 공간적 상호작용 → 이웃 기여가 자신의 보상에 영향
        if config["has_spatial"] or config["has_network"]:
            neighbor_contrib = adj @ contributions / (adj.sum(axis=1) + 1e-8)
            public_good = (contributions * 0.4 + neighbor_contrib * 0.6) * MPCR / max(1, adj.sum(axis=1).mean())
            payoffs = (ENDOWMENT - contributions) + public_good
        else:
            # PGG: 전역 평균 → SUTVA 구조적으로 성립
            total_contrib = contributions.sum()
            public_good_share = total_contrib * MPCR / n
            payoffs = (ENDOWMENT - contributions) + public_good_share
        
        wealth += payoffs - ENDOWMENT
        resource = np.clip(resource + 0.02 * (contributions.mean() / ENDOWMENT - 0.3), 0, 1)
        
        # Exposure 계산: 이웃 중 treated 비율
        for i in range(n):
            neighbors = np.where(adj[i] > 0)[0]
            if len(neighbors) > 0:
                exposure = treatments[neighbors].mean()
            else:
                exposure = 0.0
            
            panel_data.append({
                "agent_id": i,
                "step": t,
                "treatment": int(treatments[i]),
                "exposure": float(exposure),
                "outcome": float(payoffs[i]),
                "lambda": float(lambdas[i]),
                "contribution": float(contributions[i]),
                "resource": float(resource),
                "env": env_type,
            })
    
    return panel_data


# ============================================================
# 3. Exposure Mapping 기반 ATE 분해
# ============================================================

def decompose_ate(panel_data):
    """
    ATE → Direct Effect + Spillover Effect 분해.
    
    Direct Effect:  exposure 고정, treatment 변화 효과
    Spillover Effect: treatment 고정, exposure 변화 효과
    Naive ATE:  단순 treatment on/off 비교
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    outcomes = np.array([d["outcome"] for d in panel_data])
    treatments = np.array([d["treatment"] for d in panel_data])
    exposures = np.array([d["exposure"] for d in panel_data])
    
    # Naive ATE (SUTVA 가정)
    y1 = outcomes[treatments == 1]
    y0 = outcomes[treatments == 0]
    naive_ate = y1.mean() - y0.mean() if len(y1) > 0 and len(y0) > 0 else 0.0
    naive_se = np.sqrt(y1.var() / max(1, len(y1)) + y0.var() / max(1, len(y0)))
    
    # Exposure bins: low (< median), high (>= median)
    exp_median = np.median(exposures)
    low_exp = exposures < exp_median
    high_exp = exposures >= exp_median
    
    # Direct Effect (exposure 고정 — low exposure bin 내에서)
    mask_t1_low = (treatments == 1) & low_exp
    mask_t0_low = (treatments == 0) & low_exp
    if mask_t1_low.sum() > 0 and mask_t0_low.sum() > 0:
        direct_effect = outcomes[mask_t1_low].mean() - outcomes[mask_t0_low].mean()
        direct_se = np.sqrt(
            outcomes[mask_t1_low].var() / max(1, mask_t1_low.sum()) +
            outcomes[mask_t0_low].var() / max(1, mask_t0_low.sum())
        )
    else:
        direct_effect = naive_ate
        direct_se = naive_se
    
    # Spillover Effect (treatment 고정 — treated 그룹 내에서)
    mask_t1_low = (treatments == 1) & low_exp
    mask_t1_high = (treatments == 1) & high_exp
    if mask_t1_low.sum() > 0 and mask_t1_high.sum() > 0:
        spillover_effect = outcomes[mask_t1_high].mean() - outcomes[mask_t1_low].mean()
        spillover_se = np.sqrt(
            outcomes[mask_t1_high].var() / max(1, mask_t1_high.sum()) +
            outcomes[mask_t1_low].var() / max(1, mask_t1_low.sum())
        )
    else:
        spillover_effect = 0.0
        spillover_se = 0.0
    
    # Corrected ATE = Direct + Spillover 가중 평균
    corrected_ate = direct_effect + spillover_effect * 0.5
    
    # SUTVA 위반 정도 = |Naive - Direct|
    sutva_violation = abs(naive_ate - direct_effect)
    
    # 통계적 유의성 (z-test 근사)
    def p_value(effect, se):
        if se > 0:
            z = abs(effect) / se
            return float(2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))))
        return 1.0
    
    return {
        "naive_ate": float(naive_ate),
        "naive_se": float(naive_se),
        "naive_p": p_value(naive_ate, naive_se),
        "direct_effect": float(direct_effect),
        "direct_se": float(direct_se),
        "direct_p": p_value(direct_effect, direct_se),
        "spillover_effect": float(spillover_effect),
        "spillover_se": float(spillover_se),
        "spillover_p": p_value(spillover_effect, spillover_se),
        "corrected_ate": float(corrected_ate),
        "sutva_violation": float(sutva_violation),
        "exposure_median": float(exp_median),
        "n_obs": len(panel_data),
    }


def compute_exposure_response_curve(panel_data, n_bins=10):
    """Exposure-Response Curve: exposure 비율 vs 평균 결과."""
    exposures = np.array([d["exposure"] for d in panel_data])
    outcomes = np.array([d["outcome"] for d in panel_data])
    
    bins = np.linspace(0, 1, n_bins + 1)
    curve = []
    for i in range(n_bins):
        mask = (exposures >= bins[i]) & (exposures < bins[i + 1])
        if mask.sum() > 0:
            curve.append({
                "exposure_mid": float((bins[i] + bins[i + 1]) / 2),
                "mean_outcome": float(outcomes[mask].mean()),
                "se": float(outcomes[mask].std() / np.sqrt(mask.sum())),
                "n": int(mask.sum()),
            })
    return curve


# ============================================================
# 4. 전체 실험 실행
# ============================================================

def run_spillover_experiment():
    """3개 환경 × 3 SVO × 10 seeds 전체 실험."""
    results = {}
    
    print("=" * 65)
    print("  W1: Causal Inference under Interference")
    print("  SUTVA Decomposition: Direct + Spillover Effects")
    print("=" * 65)
    
    for env_type, env_config in ENV_CONFIGS.items():
        results[env_type] = {"decomposition": {}, "exposure_curves": {}}
        print(f"\n--- ENV: {env_type} ({env_config['description']}) ---")
        
        for svo_name, svo_theta in SVO_THETAS.items():
            all_panel = []
            for seed in range(N_SEEDS):
                panel = simulate_with_treatment_vector(
                    env_type, svo_theta, seed, meta_fraction=0.5
                )
                all_panel.extend(panel)
            
            # ATE 분해
            decomp = decompose_ate(all_panel)
            results[env_type]["decomposition"][svo_name] = decomp
            
            # Exposure-Response Curve
            curve = compute_exposure_response_curve(all_panel)
            results[env_type]["exposure_curves"][svo_name] = curve
            
            print(
                f"  {svo_name:15s} | "
                f"Naive ATE: {decomp['naive_ate']:+.4f} | "
                f"Direct: {decomp['direct_effect']:+.4f} | "
                f"Spillover: {decomp['spillover_effect']:+.4f} | "
                f"SUTVA gap: {decomp['sutva_violation']:.4f}"
            )
    
    return results


# ============================================================
# 5. 시각화 (Fig 75-78)
# ============================================================

def plot_fig75(results):
    """Fig 75: 환경별 ATE 분해 (Direct vs Spillover 막대그래프)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        "Fig 75: ATE Decomposition — Direct vs Spillover Effects by Environment",
        fontsize=14, fontweight='bold', y=1.02
    )
    
    envs = list(ENV_CONFIGS.keys())
    svo_labels = list(SVO_THETAS.keys())
    x = np.arange(len(svo_labels))
    width = 0.25
    
    colors = {'direct': '#2196F3', 'spillover': '#FF9800', 'naive': '#9E9E9E'}
    
    for idx, env in enumerate(envs):
        ax = axes[idx]
        direct_vals = [results[env]["decomposition"][s]["direct_effect"] for s in svo_labels]
        spillover_vals = [results[env]["decomposition"][s]["spillover_effect"] for s in svo_labels]
        naive_vals = [results[env]["decomposition"][s]["naive_ate"] for s in svo_labels]
        
        ax.bar(x - width, naive_vals, width, label='Naive ATE', color=colors['naive'], alpha=0.7)
        ax.bar(x, direct_vals, width, label='Direct Effect', color=colors['direct'], alpha=0.8)
        ax.bar(x + width, spillover_vals, width, label='Spillover Effect', color=colors['spillover'], alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(svo_labels, fontsize=9)
        ax.set_ylabel('Effect Size')
        ax.set_title(f'{env.replace("_", " ").title()}\n({ENV_CONFIGS[env]["description"]})',
                      fontweight='bold', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig75_ate_decomposition.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[W1] Fig 75 저장: {path}")


def plot_fig76(results):
    """Fig 76: Exposure-Response Curve (exposure 비율 vs 결과)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        "Fig 76: Exposure-Response Curves — Neighbor Meta-Ranking Effect",
        fontsize=14, fontweight='bold', y=1.02
    )
    
    svo_colors = {'individualist': '#F44336', 'prosocial': '#4CAF50', 'altruistic': '#2196F3'}
    envs = list(ENV_CONFIGS.keys())
    
    for idx, env in enumerate(envs):
        ax = axes[idx]
        for svo_name, color in svo_colors.items():
            curve = results[env]["exposure_curves"][svo_name]
            if curve:
                xs = [c["exposure_mid"] for c in curve]
                ys = [c["mean_outcome"] for c in curve]
                ses = [c["se"] for c in curve]
                ax.plot(xs, ys, 'o-', color=color, linewidth=2, markersize=5, label=svo_name)
                ax.fill_between(xs, [y - s for y, s in zip(ys, ses)],
                               [y + s for y, s in zip(ys, ses)],
                               color=color, alpha=0.15)
        
        ax.set_xlabel('Neighbor Exposure (Meta-Ranking Fraction)')
        ax.set_ylabel('Mean Outcome')
        ax.set_title(f'{env.replace("_", " ").title()}', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig76_exposure_response.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[W1] Fig 76 저장: {path}")


def plot_fig77(results):
    """Fig 77: SUTVA 위반도 히트맵 (환경 × SVO)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle(
        "Fig 77: SUTVA Violation Magnitude — |Naive ATE − Direct Effect|",
        fontsize=14, fontweight='bold', y=1.02
    )
    
    envs = list(ENV_CONFIGS.keys())
    svo_labels = list(SVO_THETAS.keys())
    
    violation = np.zeros((len(envs), len(svo_labels)))
    for i, env in enumerate(envs):
        for j, svo in enumerate(svo_labels):
            violation[i, j] = results[env]["decomposition"][svo]["sutva_violation"]
    
    im = ax.imshow(violation, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(svo_labels)))
    ax.set_xticklabels(svo_labels, fontsize=10)
    ax.set_yticks(range(len(envs)))
    ax.set_yticklabels([e.replace("_", " ").title() for e in envs], fontsize=10)
    
    for i in range(len(envs)):
        for j in range(len(svo_labels)):
            text = f'{violation[i, j]:.4f}'
            color = 'white' if violation[i, j] > violation.max() * 0.6 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontweight='bold',
                    fontsize=11, color=color)
    
    plt.colorbar(im, ax=ax, label='|Naive ATE − Direct Effect|')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig77_sutva_violation.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[W1] Fig 77 저장: {path}")


def plot_fig78(results):
    """Fig 78: Corrected ATE vs Naive ATE 비교 테이블."""
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(
        "Fig 78: Naive ATE vs Corrected AME — Summary Table",
        fontsize=14, fontweight='bold', y=1.02
    )
    ax.axis('off')
    
    table_data = [["Environment", "SVO", "Naive ATE", "Direct Effect",
                   "Spillover", "Corrected AME", "SUTVA Gap", "Direction Preserved?"]]
    
    for env in ENV_CONFIGS:
        for svo in SVO_THETAS:
            d = results[env]["decomposition"][svo]
            direction_ok = "✓" if np.sign(d["naive_ate"]) == np.sign(d["corrected_ate"]) else "✗"
            table_data.append([
                env.replace("_", " ").title(),
                svo,
                f'{d["naive_ate"]:+.4f}',
                f'{d["direct_effect"]:+.4f}',
                f'{d["spillover_effect"]:+.4f}',
                f'{d["corrected_ate"]:+.4f}',
                f'{d["sutva_violation"]:.4f}',
                direction_ok,
            ])
    
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    
    # 헤더 스타일
    for j in range(len(table_data[0])):
        table[0, j].set_facecolor('#1a237e')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    # 행 색상 교대
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[i, j].set_facecolor('#e8eaf6')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig78_corrected_ate_table.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[W1] Fig 78 저장: {path}")


# ============================================================
# 6. 메인 실행
# ============================================================

if __name__ == "__main__":
    results = run_spillover_experiment()
    
    # Figure 생성
    plot_fig75(results)
    plot_fig76(results)
    plot_fig77(results)
    plot_fig78(results)
    
    # JSON 저장
    json_data = {}
    for env, env_data in results.items():
        json_data[env] = {
            "decomposition": env_data["decomposition"],
            "exposure_curves": env_data["exposure_curves"],
        }
    
    json_path = os.path.join(OUTPUT_DIR, "spillover_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[W1] 결과 JSON: {json_path}")
    
    # 핵심 결론 출력
    print("\n" + "=" * 65)
    print("  W1 SUMMARY: SUTVA Analysis")
    print("=" * 65)
    for env in ENV_CONFIGS:
        avg_gap = np.mean([
            results[env]["decomposition"][s]["sutva_violation"]
            for s in SVO_THETAS
        ])
        all_preserved = all(
            np.sign(results[env]["decomposition"][s]["naive_ate"]) ==
            np.sign(results[env]["decomposition"][s]["corrected_ate"])
            for s in SVO_THETAS
        )
        print(f"  {env:15s} | Avg SUTVA gap: {avg_gap:.4f} | "
              f"Direction preserved: {'YES' if all_preserved else 'NO'}")
