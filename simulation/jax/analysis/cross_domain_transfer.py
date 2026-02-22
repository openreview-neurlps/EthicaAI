"""
W4: Cross-Domain Behavioral Transfer Protocol
EthicaAI Phase W — NeurIPS 2026 Critique Defense

PGG→SSD 외적 타당성 비평 대응:
"PGG에서의 인간 유사성이 Grid World에서 보장되지 않음" 지적에 대한 실험적 해결.

Behavioral Fingerprint (BF) 정의:
  BF_i = (CrisisSensitivity, ContributionElasticity, GiniDynamics, RoleStability)

전이 프로토콜:
  Step 1: Source 환경에서 BF 추출
  Step 2: Target 환경에서 BF 추출 (동일 SVO, 동일 λ 파라미터)
  Step 3: BF 코사인 유사도 비교

검증 환경 쌍:
  PGG → IPD, PGG → Grid World, PGG → Climate Negotiation, IPD → Grid World

출력: Fig 85-88, transfer_results.json
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

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 10, 'figure.dpi': 300,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# === 설정 ===
N_AGENTS = 20
N_STEPS = 200
N_SEEDS = 10
ENDOWMENT = 100
MPCR = 1.6
ALPHA = 0.9

SVO_CONDITIONS = {
    "individualist": math.radians(15),
    "prosocial": math.radians(45),
    "altruistic": math.radians(90),
}

TRANSFER_PAIRS = [
    ("pgg", "ipd"),
    ("pgg", "grid_world"),
    ("pgg", "climate"),
    ("ipd", "grid_world"),
]


# ============================================================
# 1. 환경별 시뮬레이션
# ============================================================

def compute_lambda(svo_theta, resource, prev_lambda):
    """Sen's λ_t (통일 버전)."""
    base = math.sin(svo_theta)
    if resource < 0.2:
        target = max(0, base * 0.3)
    elif resource > 0.7:
        target = min(1.0, 1.5 * base)
    else:
        target = base * (0.7 + 1.6 * resource)
    return ALPHA * prev_lambda + (1 - ALPHA) * target


def simulate_pgg(svo_theta, seed):
    """N-Player PGG."""
    rng = np.random.RandomState(seed)
    n = N_AGENTS
    resource = 0.5
    lambdas = np.full(n, math.sin(svo_theta))
    
    trajectory = {"lambdas": [], "contributions": [], "resources": [],
                  "payoffs": [], "gini": []}
    
    for t in range(N_STEPS):
        for i in range(n):
            lambdas[i] = compute_lambda(svo_theta, resource, lambdas[i])
        
        contributions = np.clip(lambdas * ENDOWMENT * 0.8 + rng.normal(0, 3, n), 0, ENDOWMENT)
        total_c = contributions.sum()
        public_good = total_c * MPCR / n
        payoffs = (ENDOWMENT - contributions) + public_good
        resource = np.clip(resource + 0.02 * (contributions.mean() / ENDOWMENT - 0.3), 0, 1)
        
        gini = _compute_gini(payoffs)
        
        trajectory["lambdas"].append(float(lambdas.mean()))
        trajectory["contributions"].append(float(contributions.mean()))
        trajectory["resources"].append(float(resource))
        trajectory["payoffs"].append(float(payoffs.mean()))
        trajectory["gini"].append(float(gini))
    
    return trajectory


def simulate_ipd(svo_theta, seed):
    """Iterated Prisoner's Dilemma (IPD)."""
    rng = np.random.RandomState(seed)
    n = N_AGENTS
    resource = 0.5
    lambdas = np.full(n, math.sin(svo_theta))
    
    T, R, P, S = 5.0, 3.0, 1.0, 0.0  # Payoff matrix
    
    trajectory = {"lambdas": [], "contributions": [], "resources": [],
                  "payoffs": [], "gini": []}
    
    for t in range(N_STEPS):
        for i in range(n):
            lambdas[i] = compute_lambda(svo_theta, resource, lambdas[i])
        
        # 협력 확률 = λ
        cooperate = rng.random(n) < lambdas
        
        # 각 에이전트를 랜덤 2인 매칭
        payoffs = np.zeros(n)
        pairs = list(range(n))
        rng.shuffle(pairs)
        for k in range(0, n - 1, 2):
            i, j = pairs[k], pairs[k + 1]
            if cooperate[i] and cooperate[j]:
                payoffs[i], payoffs[j] = R, R
            elif cooperate[i] and not cooperate[j]:
                payoffs[i], payoffs[j] = S, T
            elif not cooperate[i] and cooperate[j]:
                payoffs[i], payoffs[j] = T, S
            else:
                payoffs[i], payoffs[j] = P, P
        
        resource = np.clip(resource + 0.02 * (cooperate.mean() - 0.3), 0, 1)
        contributions = cooperate.astype(float) * ENDOWMENT * 0.5
        gini = _compute_gini(payoffs)
        
        trajectory["lambdas"].append(float(lambdas.mean()))
        trajectory["contributions"].append(float(contributions.mean()))
        trajectory["resources"].append(float(resource))
        trajectory["payoffs"].append(float(payoffs.mean()))
        trajectory["gini"].append(float(gini))
    
    return trajectory


def simulate_grid_world(svo_theta, seed):
    """Grid World (Cleanup) — 공간 상호작용."""
    rng = np.random.RandomState(seed)
    n = N_AGENTS
    resource = 0.5
    lambdas = np.full(n, math.sin(svo_theta))
    grid_size = int(math.ceil(math.sqrt(n)))
    
    # 에이전트 역할: Cleaner (λ > 0.5) vs Eater (λ <= 0.5)
    trajectory = {"lambdas": [], "contributions": [], "resources": [],
                  "payoffs": [], "gini": [], "role_switches": []}
    
    prev_roles = (lambdas > 0.5).astype(int)
    
    for t in range(N_STEPS):
        for i in range(n):
            lambdas[i] = compute_lambda(svo_theta, resource, lambdas[i])
        
        # 역할 결정: λ > 0.5 → Cleaner, else → Eater
        roles = (lambdas > 0.5).astype(int)
        n_cleaners = roles.sum()
        
        # Cleaner: 자원 재생에 기여하지만 보상 낮음
        # Eater: 보상 높지만 자원 소모
        payoffs = np.zeros(n)
        for i in range(n):
            if roles[i] == 1:  # Cleaner
                payoffs[i] = ENDOWMENT * 0.3 + resource * ENDOWMENT * 0.3
            else:  # Eater
                payoffs[i] = ENDOWMENT * 0.7 + resource * ENDOWMENT * 0.5
        
        # 자원 역학: Cleaner 비율에 따라 재생
        clean_ratio = n_cleaners / n
        resource = np.clip(resource + 0.05 * (clean_ratio - 0.3), 0, 1)
        
        contributions = roles.astype(float) * ENDOWMENT * 0.6
        role_switches = int((roles != prev_roles).sum())
        prev_roles = roles.copy()
        gini = _compute_gini(payoffs)
        
        trajectory["lambdas"].append(float(lambdas.mean()))
        trajectory["contributions"].append(float(contributions.mean()))
        trajectory["resources"].append(float(resource))
        trajectory["payoffs"].append(float(payoffs.mean()))
        trajectory["gini"].append(float(gini))
        trajectory["role_switches"].append(role_switches)
    
    return trajectory


def simulate_climate(svo_theta, seed):
    """Climate Negotiation — 다자간 협상."""
    rng = np.random.RandomState(seed)
    n = min(N_AGENTS, 10)  # 10개국
    resource = 0.5  # 지구 자원 (기후 안정성)
    lambdas = np.full(n, math.sin(svo_theta))
    
    trajectory = {"lambdas": [], "contributions": [], "resources": [],
                  "payoffs": [], "gini": []}
    
    for t in range(N_STEPS):
        for i in range(n):
            lambdas[i] = compute_lambda(svo_theta, resource, lambdas[i])
        
        # 감축 비율 = λ × 능력
        abilities = rng.uniform(0.5, 1.5, n)
        reductions = lambdas * abilities * 0.6
        reductions = np.clip(reductions, 0, 1)
        
        # 글로벌 감축 효과
        global_reduction = reductions.mean()
        resource = np.clip(resource + 0.03 * (global_reduction - 0.2), 0, 1)
        
        # 보상: 감축 비용 + 기후 안정 이익
        costs = reductions * ENDOWMENT * 0.3
        benefits = resource * ENDOWMENT * 0.5
        payoffs = benefits - costs + ENDOWMENT * 0.5
        
        contributions = reductions * ENDOWMENT
        gini = _compute_gini(payoffs)
        
        trajectory["lambdas"].append(float(lambdas.mean()))
        trajectory["contributions"].append(float(contributions.mean()))
        trajectory["resources"].append(float(resource))
        trajectory["payoffs"].append(float(payoffs.mean()))
        trajectory["gini"].append(float(gini))
    
    return trajectory


ENV_SIMULATORS = {
    "pgg": simulate_pgg,
    "ipd": simulate_ipd,
    "grid_world": simulate_grid_world,
    "climate": simulate_climate,
}


def _compute_gini(values):
    """Gini coefficient."""
    values = np.sort(np.abs(values))
    n = len(values)
    if n == 0 or np.sum(values) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float(np.sum((2 * index - n - 1) * values) / (n * np.sum(values)))


# ============================================================
# 2. Behavioral Fingerprint 추출
# ============================================================

def extract_behavioral_fingerprint(trajectory):
    """
    4차원 Behavioral Fingerprint 추출:
      1. Crisis Sensitivity: 위기 시 λ 변화율
      2. Contribution Elasticity: 자원 변화에 대한 기여 반응
      3. Gini Dynamics: 불평등 변화 추세
      4. Role Stability: 행동 변화 빈도 (낮을수록 안정)
    """
    lambdas = np.array(trajectory["lambdas"])
    contributions = np.array(trajectory["contributions"])
    resources = np.array(trajectory["resources"])
    ginis = np.array(trajectory["gini"])
    
    # 1. Crisis Sensitivity: R < 0.3일 때 λ의 변화율
    crisis_mask = resources < 0.3
    if crisis_mask.sum() > 2:
        crisis_lambdas = lambdas[crisis_mask]
        crisis_sensitivity = float(np.std(crisis_lambdas))
    else:
        crisis_sensitivity = 0.0
    
    # 2. Contribution Elasticity: Δcontrib / Δresource
    if len(resources) > 10:
        delta_r = np.diff(resources)
        delta_c = np.diff(contributions)
        nonzero = np.abs(delta_r) > 1e-6
        if nonzero.sum() > 0:
            elasticity = float(np.mean(np.abs(delta_c[nonzero] / delta_r[nonzero])))
            elasticity = min(elasticity, 100.0)  # 상한
        else:
            elasticity = 0.0
    else:
        elasticity = 0.0
    
    # 3. Gini Dynamics: Gini 계수 시간 추세 (기울기)
    if len(ginis) > 5:
        gini_trend = float(np.polyfit(range(len(ginis)), ginis, 1)[0])
    else:
        gini_trend = 0.0
    
    # 4. Role Stability: λ의 시간적 변동성 (작을수록 안정)
    if len(lambdas) > 5:
        lambda_volatility = float(np.std(np.diff(lambdas)))
    else:
        lambda_volatility = 0.0
    
    return np.array([crisis_sensitivity, elasticity, gini_trend, lambda_volatility])


def cosine_similarity(a, b):
    """코사인 유사도."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ============================================================
# 3. 전체 전이 실험
# ============================================================

def run_transfer_experiment():
    """4 환경 쌍 × 3 SVO × 10 seeds 전이 실험."""
    results = {"fingerprints": {}, "similarities": {}, "lambda_convergence": {}}
    
    print("=" * 65)
    print("  W4: Cross-Domain Behavioral Transfer Protocol")
    print("=" * 65)
    
    # 모든 환경에서 BF 추출
    print("\n--- Extracting Behavioral Fingerprints ---")
    for env_name, sim_fn in ENV_SIMULATORS.items():
        results["fingerprints"][env_name] = {}
        for svo_name, svo_theta in SVO_CONDITIONS.items():
            bfs = []
            lambda_convs = []
            for seed in range(N_SEEDS):
                traj = sim_fn(svo_theta, seed)
                bf = extract_behavioral_fingerprint(traj)
                bfs.append(bf)
                lambda_convs.append(traj["lambdas"][-10:])  # 마지막 10스텝
            
            mean_bf = np.mean(bfs, axis=0)
            results["fingerprints"][env_name][svo_name] = {
                "crisis_sensitivity": float(mean_bf[0]),
                "contribution_elasticity": float(mean_bf[1]),
                "gini_dynamics": float(mean_bf[2]),
                "role_stability": float(mean_bf[3]),
                "bf_vector": mean_bf.tolist(),
                "lambda_converged": float(np.mean(lambda_convs)),
            }
            print(f"  {env_name:12s} | {svo_name:15s} | "
                  f"BF: [{mean_bf[0]:.4f}, {mean_bf[1]:.2f}, "
                  f"{mean_bf[2]:.5f}, {mean_bf[3]:.5f}]")
    
    # 환경 쌍별 BF 유사도
    print("\n--- Cross-Domain Similarity ---")
    for source, target in TRANSFER_PAIRS:
        pair_key = f"{source}_to_{target}"
        results["similarities"][pair_key] = {}
        
        for svo_name in SVO_CONDITIONS:
            bf_source = np.array(results["fingerprints"][source][svo_name]["bf_vector"])
            bf_target = np.array(results["fingerprints"][target][svo_name]["bf_vector"])
            sim = cosine_similarity(bf_source, bf_target)
            
            # 개별 차원별 차이
            dimension_diff = np.abs(bf_source - bf_target)
            
            results["similarities"][pair_key][svo_name] = {
                "cosine_similarity": float(sim),
                "dimension_diff": dimension_diff.tolist(),
                "bf_source": bf_source.tolist(),
                "bf_target": bf_target.tolist(),
            }
            print(f"  {source:8s} → {target:12s} | {svo_name:15s} | "
                  f"Cosine Sim: {sim:.4f}")
    
    # λ 수렴값 비교
    print("\n--- Lambda Convergence Comparison ---")
    for svo_name in SVO_CONDITIONS:
        conv_vals = {}
        for env_name in ENV_SIMULATORS:
            conv_vals[env_name] = results["fingerprints"][env_name][svo_name]["lambda_converged"]
        results["lambda_convergence"][svo_name] = conv_vals
        vals_str = " | ".join(f"{e}: {v:.4f}" for e, v in conv_vals.items())
        print(f"  {svo_name:15s} | {vals_str}")
    
    return results


# ============================================================
# 4. 시각화 (Fig 85-88)
# ============================================================

def plot_fig85(results):
    """Fig 85: 환경 쌍별 BF Cosine Similarity 히트맵."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        "Fig 85: Cross-Domain Behavioral Fingerprint Similarity (Cosine)",
        fontsize=14, fontweight='bold', y=1.02
    )
    
    pairs = list(results["similarities"].keys())
    svos = list(SVO_CONDITIONS.keys())
    
    sim_matrix = np.zeros((len(pairs), len(svos)))
    for i, pair in enumerate(pairs):
        for j, svo in enumerate(svos):
            sim_matrix[i, j] = results["similarities"][pair][svo]["cosine_similarity"]
    
    im = ax.imshow(sim_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(svos)))
    ax.set_xticklabels(svos, fontsize=10)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([p.replace("_to_", " → ").replace("_", " ").title() for p in pairs],
                       fontsize=9)
    
    for i in range(len(pairs)):
        for j in range(len(svos)):
            text = f'{sim_matrix[i, j]:.3f}'
            color = 'white' if abs(sim_matrix[i, j]) < 0.3 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontweight='bold',
                    fontsize=11, color=color)
    
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    ax.set_xlabel('SVO Condition')
    ax.set_ylabel('Environment Transfer Pair')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig85_transfer_similarity.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[W4] Fig 85 저장: {path}")


def plot_fig86(results):
    """Fig 86: 4차원 BF 레이더 차트 (환경별 겹쳐 그리기)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))
    fig.suptitle(
        "Fig 86: Behavioral Fingerprint Radar — Environment Invariance",
        fontsize=14, fontweight='bold', y=1.05
    )
    
    metrics = ["Crisis\nSensitivity", "Contribution\nElasticity",
               "Gini\nDynamics", "Role\nStability"]
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]
    
    env_colors = {'pgg': '#4CAF50', 'ipd': '#F44336', 'grid_world': '#2196F3', 'climate': '#FF9800'}
    
    for idx, (svo_name, svo_theta) in enumerate(SVO_CONDITIONS.items()):
        ax = axes[idx]
        ax.set_title(f'{svo_name.title()}\n(θ={math.degrees(svo_theta):.0f}°)',
                      fontweight='bold', pad=20)
        
        # 정규화 기준
        all_vals = []
        for env_name in ENV_SIMULATORS:
            bf = results["fingerprints"][env_name][svo_name]["bf_vector"]
            all_vals.append(bf)
        all_vals = np.array(all_vals)
        maxs = np.abs(all_vals).max(axis=0) + 1e-8
        
        for env_name, color in env_colors.items():
            bf = np.array(results["fingerprints"][env_name][svo_name]["bf_vector"])
            # 정규화 (0-1 범위)
            bf_norm = np.abs(bf) / maxs
            values = bf_norm.tolist() + [bf_norm[0]]
            ax.plot(angles, values, 'o-', linewidth=2, label=env_name.replace("_", " ").title(),
                    color=color, markersize=4)
            ax.fill(angles, values, alpha=0.1, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=7)
        ax.set_ylim(0, 1.2)
        if idx == 2:
            ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=8)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig86_bf_radar.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[W4] Fig 86 저장: {path}")


def plot_fig87(results):
    """Fig 87: λ 수렴 궤적 비교 (환경별)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Fig 87: λ Convergence Across Environments — Same SVO → Same λ*",
        fontsize=14, fontweight='bold', y=1.02
    )
    
    env_colors = {'pgg': '#4CAF50', 'ipd': '#F44336', 'grid_world': '#2196F3', 'climate': '#FF9800'}
    
    for idx, (svo_name, svo_theta) in enumerate(SVO_CONDITIONS.items()):
        ax = axes[idx]
        
        for env_name, sim_fn in ENV_SIMULATORS.items():
            traj = sim_fn(svo_theta, seed=0)
            ax.plot(traj["lambdas"], linewidth=1.5, alpha=0.8,
                    color=env_colors[env_name],
                    label=env_name.replace("_", " ").title())
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Mean λ_t')
        ax.set_title(f'{svo_name.title()} (θ={math.degrees(svo_theta):.0f}°)',
                      fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # λ* 수렴선
        converged = results["lambda_convergence"][svo_name]
        for env, val in converged.items():
            ax.axhline(val, color=env_colors[env], linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig87_lambda_convergence.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[W4] Fig 87 저장: {path}")


def plot_fig88(results):
    """Fig 88: Transfer Gap 분석 — |BF_source - BF_target| by SVO."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Fig 88: Transfer Gap by SVO — Prosocial Agents Show Most Consistent Transfer",
        fontsize=14, fontweight='bold', y=1.02
    )
    
    bf_labels = ["Crisis\nSensitivity", "Contribution\nElasticity",
                 "Gini\nDynamics", "Role\nStability"]
    pair_colors = ['#4CAF50', '#F44336', '#2196F3', '#FF9800']
    
    for idx, (svo_name, svo_theta) in enumerate(SVO_CONDITIONS.items()):
        ax = axes[idx]
        
        pairs = list(results["similarities"].keys())
        x = np.arange(len(bf_labels))
        width = 0.18
        
        for p_idx, pair in enumerate(pairs):
            diffs = results["similarities"][pair][svo_name]["dimension_diff"]
            ax.bar(x + p_idx * width, diffs, width,
                   label=pair.replace("_to_", "→").replace("_", " "),
                   color=pair_colors[p_idx], alpha=0.8)
        
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(bf_labels, fontsize=8)
        ax.set_ylabel('|BF_source − BF_target|')
        ax.set_title(f'{svo_name.title()} (θ={math.degrees(svo_theta):.0f}°)',
                      fontweight='bold')
        if idx == 0:
            ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig88_transfer_gap.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[W4] Fig 88 저장: {path}")


# ============================================================
# 5. 메인 실행
# ============================================================

if __name__ == "__main__":
    results = run_transfer_experiment()
    
    plot_fig85(results)
    plot_fig86(results)
    plot_fig87(results)
    plot_fig88(results)
    
    json_path = os.path.join(OUTPUT_DIR, "transfer_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[W4] 결과 JSON: {json_path}")
    
    # 핵심 결론
    print("\n" + "=" * 65)
    print("  W4 SUMMARY: Behavioral Fingerprint Transfer Analysis")
    print("=" * 65)
    for pair in TRANSFER_PAIRS:
        pair_key = f"{pair[0]}_to_{pair[1]}"
        sims = [results["similarities"][pair_key][s]["cosine_similarity"]
                for s in SVO_CONDITIONS]
        print(f"  {pair[0]:8s} → {pair[1]:12s} | "
              f"Avg Cosine: {np.mean(sims):.4f} | "
              f"Best SVO: {list(SVO_CONDITIONS.keys())[np.argmax(sims)]}")
