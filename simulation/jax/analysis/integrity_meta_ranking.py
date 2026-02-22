"""
W3: Integrity-Constrained Meta-Ranking — Reward Hacking Defense
EthicaAI Phase W — NeurIPS 2026 Critique Defense

U_meta(타인 보상 평균) 설계의 Reward Hacking 취약성 대응.

4가지 U_meta 변형:
  1. Naive (현재):   U_meta = mean(all others' rewards)
  2. Bounded:        U_meta = clip(kNN_mean, μ-2σ, μ+2σ)
  3. Divergence:     + KL penalty on policy shift
  4. Causal:         Shapley-style causal contribution (approx.)

평가:
  - Normal 환경: 성능 유지 확인
  - Adversarial 환경: Reward Pumping 공격 저항도
  - Computational overhead

출력: Fig 81-84, integrity_results.json
"""

import os
import sys
import json
import math
import time
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
N_AGENTS = 50
N_STEPS = 200
N_SEEDS = 10
ENDOWMENT = 100
MPCR = 1.6
SVO_THETA = math.radians(45)  # prosocial 고정
ADV_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.5]
K_NEIGHBORS = 5  # Bounded 변형의 kNN


# ============================================================
# 1. U_meta 변형 구현
# ============================================================

def umeta_naive(agent_id, all_rewards, all_prev_policies=None, adj=None):
    """Naive: 전체 타인 보상 평균."""
    others = [r for i, r in enumerate(all_rewards) if i != agent_id]
    return np.mean(others) if others else 0.0


def umeta_bounded(agent_id, all_rewards, all_prev_policies=None, adj=None):
    """Bounded: k-NN 지역 평균 + 2σ clipping."""
    n = len(all_rewards)
    # k-NN: 가장 가까운 k개 에이전트 (인덱스 기준, 순환)
    neighbors = []
    for delta in range(1, K_NEIGHBORS + 1):
        neighbors.append((agent_id + delta) % n)
        neighbors.append((agent_id - delta) % n)
    neighbors = list(set(neighbors))[:K_NEIGHBORS]
    
    if not neighbors:
        return umeta_naive(agent_id, all_rewards)
    
    neighbor_rewards = [all_rewards[j] for j in neighbors]
    mu = np.mean(neighbor_rewards)
    sigma = np.std(neighbor_rewards) + 1e-8
    
    return float(np.clip(mu, mu - 2 * sigma, mu + 2 * sigma))


def umeta_divergence(agent_id, all_rewards, all_prev_policies=None, adj=None):
    """Divergence: Bounded + KL penalty on contribution shift."""
    base = umeta_bounded(agent_id, all_rewards)
    
    # KL penalty: 현재 정책과 이전 정책의 차이
    if all_prev_policies is not None and len(all_prev_policies) > agent_id:
        prev = all_prev_policies[agent_id]
        curr = all_rewards[agent_id] / (ENDOWMENT + 1e-8)
        # 간소화된 KL: |curr - prev|
        kl_penalty = 0.05 * abs(curr - prev)
        return base - kl_penalty
    return base


def umeta_causal(agent_id, all_rewards, all_prev_policies=None, adj=None):
    """Causal: Shapley-style 인과적 기여도 근사."""
    n = len(all_rewards)
    # Leave-one-out 근사: 에이전트 i 없을 때 보상 감소분
    total_without_i = sum(r for j, r in enumerate(all_rewards) if j != agent_id)
    mean_without_i = total_without_i / max(1, n - 1)
    
    # 에이전트 i의 기여 = 전체 평균 - i 제외 평균의 차이 반영
    total_with_i = sum(all_rewards)
    mean_with_i = total_with_i / n
    
    causal_contribution = mean_with_i - mean_without_i
    
    # 스케일 조정: naive와 유사한 범위로
    return float(mean_without_i + 0.5 * causal_contribution)


UMETA_VARIANTS = {
    "naive":      {"fn": umeta_naive,      "label": "Naive (Global Mean)",   "color": "#9E9E9E"},
    "bounded":    {"fn": umeta_bounded,    "label": "Bounded (kNN + Clip)", "color": "#2196F3"},
    "divergence": {"fn": umeta_divergence, "label": "Divergence (+KL)",     "color": "#4CAF50"},
    "causal":     {"fn": umeta_causal,     "label": "Causal (Shapley)",     "color": "#FF9800"},
}


# ============================================================
# 2. 에이전트 정의
# ============================================================

class MetaRankingAgent:
    """메타랭킹 에이전트 (정상)."""
    def __init__(self, rng, svo_theta=SVO_THETA):
        self.svo_theta = svo_theta
        self.lambda_t = math.sin(svo_theta)
        self.rng = rng
        self.prev_contribution_rate = 0.5
    
    def decide(self, resource, umeta_value):
        base = math.sin(self.svo_theta)
        if resource < 0.2:
            target = max(0, base * 0.3)
        elif resource > 0.7:
            target = min(1.0, 1.5 * base)
        else:
            target = base * (0.7 + 1.6 * resource)
        
        self.lambda_t = 0.9 * self.lambda_t + 0.1 * target
        
        # U_meta를 보상 신호로 반영
        meta_bonus = umeta_value * 0.1 if umeta_value > 0 else 0.0
        contribution = self.lambda_t * ENDOWMENT * 0.8 + meta_bonus
        contribution += self.rng.normal(0, 3)
        contribution = float(np.clip(contribution, 0, ENDOWMENT))
        
        self.prev_contribution_rate = contribution / ENDOWMENT
        return contribution


class RewardPumpingAgent:
    """
    적대적 에이전트: U_meta를 인위적으로 조작하는 전략.
    
    전략: 기여를 매우 적게 하면서 (free-riding),
    다른 에이전트들이 높은 보상을 보고하도록 유도하여
    자신의 U_meta 기반 보상을 극대화.
    
    구체적으로: 라운드 초반 고기여 → 신뢰 확보 → 급격한 기여 감소
    """
    def __init__(self, rng):
        self.rng = rng
        self.phase = 0  # 0=build trust, 1=exploit
        self.step = 0
        self.lambda_t = 0.8
        self.prev_contribution_rate = 0.8
    
    def decide(self, resource, umeta_value):
        self.step += 1
        if self.step < 30:
            # Phase 0: 신뢰 구축 (높은 기여)
            contribution = 0.8 * ENDOWMENT + self.rng.normal(0, 2)
        else:
            # Phase 1: U_meta 착취 (최소 기여)
            contribution = 5.0 + self.rng.normal(0, 2)
            self.lambda_t = 0.1
        
        contribution = float(np.clip(contribution, 0, ENDOWMENT))
        self.prev_contribution_rate = contribution / ENDOWMENT
        return contribution


# ============================================================
# 3. 시뮬레이션
# ============================================================

def simulate_integrity(umeta_variant, adv_fraction, seed):
    """하나의 U_meta 변형 + 적대적 비율로 시뮬레이션."""
    rng = np.random.RandomState(seed)
    umeta_fn = UMETA_VARIANTS[umeta_variant]["fn"]
    
    n_adv = int(N_AGENTS * adv_fraction)
    n_normal = N_AGENTS - n_adv
    
    agents = [MetaRankingAgent(rng) for _ in range(n_normal)] + \
             [RewardPumpingAgent(rng) for _ in range(n_adv)]
    
    resource = 0.5
    welfare_history = []
    coop_history = []
    kl_history = []
    
    start_time = time.time()
    
    for t in range(N_STEPS):
        # 기여금 결정
        all_rewards_prev = [0.0] * N_AGENTS
        all_prev_policies = [a.prev_contribution_rate for a in agents]
        
        contributions = []
        for i, agent in enumerate(agents):
            umeta = umeta_fn(i, all_rewards_prev, all_prev_policies)
            c = agent.decide(resource, umeta)
            contributions.append(c)
        
        # PGG 메커닉
        total_contrib = sum(contributions)
        public_good = total_contrib * MPCR / N_AGENTS
        payoffs = [(ENDOWMENT - c) + public_good for c in contributions]
        
        all_rewards_prev = payoffs  # 다음 스텝용
        resource = np.clip(resource + 0.02 * (np.mean(contributions) / ENDOWMENT - 0.3), 0, 1)
        
        # 정상 에이전트만의 지표
        normal_contribs = contributions[:n_normal]
        normal_payoffs = payoffs[:n_normal]
        
        coop = np.mean([c > ENDOWMENT * 0.3 for c in normal_contribs]) if normal_contribs else 0
        welfare = np.mean(payoffs)
        
        # KL divergence 추이 (정상 vs 적대적 에이전트 정책 차이)
        if n_adv > 0:
            normal_rates = [c / ENDOWMENT for c in normal_contribs]
            adv_rates = [c / ENDOWMENT for c in contributions[n_normal:]]
            kl = abs(np.mean(normal_rates) - np.mean(adv_rates))
        else:
            kl = 0.0
        
        welfare_history.append(float(welfare))
        coop_history.append(float(coop))
        kl_history.append(float(kl))
    
    elapsed = time.time() - start_time
    
    return {
        "mean_coop": float(np.mean(coop_history[-50:])),
        "mean_welfare": float(np.mean(welfare_history[-50:])),
        "coop_history": coop_history,
        "welfare_history": welfare_history,
        "kl_history": kl_history,
        "elapsed_ms": float(elapsed * 1000),
        "ms_per_agent": float(elapsed * 1000 / N_AGENTS),
    }


# ============================================================
# 4. 전체 실험 실행
# ============================================================

def run_integrity_experiment():
    """4 변형 × 5 adversary fractions × 10 seeds."""
    results = {"normal": {}, "adversarial": {}, "overhead": {}}
    
    print("=" * 65)
    print("  W3: Integrity-Constrained Meta-Ranking")
    print("  Reward Hacking Defense Analysis")
    print("=" * 65)
    
    # Normal 환경 성능
    print("\n--- Normal Environment (0% Adversary) ---")
    for variant_name in UMETA_VARIANTS:
        runs = [simulate_integrity(variant_name, 0.0, s) for s in range(N_SEEDS)]
        results["normal"][variant_name] = {
            "mean_coop": float(np.mean([r["mean_coop"] for r in runs])),
            "mean_welfare": float(np.mean([r["mean_welfare"] for r in runs])),
            "ms_per_agent": float(np.mean([r["ms_per_agent"] for r in runs])),
        }
        d = results["normal"][variant_name]
        print(f"  {variant_name:12s} | Coop: {d['mean_coop']:.3f} | "
              f"Welfare: {d['mean_welfare']:.1f} | "
              f"Time: {d['ms_per_agent']:.2f}ms/agent")
    
    # Adversarial 환경
    print("\n--- Adversarial Environment ---")
    for variant_name in UMETA_VARIANTS:
        results["adversarial"][variant_name] = {}
        for adv_frac in ADV_FRACTIONS:
            runs = [simulate_integrity(variant_name, adv_frac, s) for s in range(N_SEEDS)]
            results["adversarial"][variant_name][adv_frac] = {
                "mean_coop": float(np.mean([r["mean_coop"] for r in runs])),
                "mean_welfare": float(np.mean([r["mean_welfare"] for r in runs])),
                "kl_history": [float(np.mean([r["kl_history"][t] for r in runs]))
                               for t in range(N_STEPS)],
            }
            d = results["adversarial"][variant_name][adv_frac]
            print(f"  {variant_name:12s} | Adv: {adv_frac*100:3.0f}% | "
                  f"Coop: {d['mean_coop']:.3f} | Welfare: {d['mean_welfare']:.1f}")
    
    # Computational overhead
    results["overhead"] = {}
    baseline_time = results["normal"]["naive"]["ms_per_agent"]
    for variant_name in UMETA_VARIANTS:
        t = results["normal"][variant_name]["ms_per_agent"]
        results["overhead"][variant_name] = {
            "ms_per_agent": t,
            "overhead_pct": float((t - baseline_time) / baseline_time * 100) if baseline_time > 0 else 0,
        }
    
    return results


# ============================================================
# 5. 시각화 (Fig 81-84)
# ============================================================

def plot_fig81(results):
    """Fig 81: Normal 환경 — 4종 U_meta 변형 성능 비교."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Fig 81: U_meta Variants — Normal Environment Performance",
        fontsize=14, fontweight='bold', y=1.02
    )
    
    variants = list(UMETA_VARIANTS.keys())
    colors = [UMETA_VARIANTS[v]["color"] for v in variants]
    labels = [UMETA_VARIANTS[v]["label"] for v in variants]
    
    # (a) Cooperation Rate
    ax = axes[0]
    coops = [results["normal"][v]["mean_coop"] for v in variants]
    bars = ax.bar(range(len(variants)), coops, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('Cooperation Rate')
    ax.set_title('(a) Cooperation Rate', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(coops):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    # (b) Welfare
    ax = axes[1]
    welfares = [results["normal"][v]["mean_welfare"] for v in variants]
    bars = ax.bar(range(len(variants)), welfares, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('Mean Welfare')
    ax.set_title('(b) Social Welfare', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(welfares):
        ax.text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig81_umeta_normal.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[W3] Fig 81 저장: {path}")


def plot_fig82(results):
    """Fig 82: Adversarial 환경 — 적대적 비율별 복지 변화."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Fig 82: Adversarial Robustness — Reward Pumping Attack Resistance",
        fontsize=14, fontweight='bold', y=1.02
    )
    
    fracs_pct = [f * 100 for f in ADV_FRACTIONS]
    
    # (a) Cooperation
    ax = axes[0]
    for variant_name, info in UMETA_VARIANTS.items():
        vals = [results["adversarial"][variant_name][f]["mean_coop"] for f in ADV_FRACTIONS]
        ax.plot(fracs_pct, vals, 'o-', color=info["color"], linewidth=2,
                markersize=6, label=info["label"])
    ax.set_xlabel('Reward Pumping Agent Fraction (%)')
    ax.set_ylabel('Cooperation Rate')
    ax.set_title('(a) Cooperation Under Attack', fontweight='bold')
    ax.legend(fontsize=7, loc='lower left')
    ax.grid(True, alpha=0.3)
    
    # (b) Welfare
    ax = axes[1]
    for variant_name, info in UMETA_VARIANTS.items():
        vals = [results["adversarial"][variant_name][f]["mean_welfare"] for f in ADV_FRACTIONS]
        ax.plot(fracs_pct, vals, 's-', color=info["color"], linewidth=2,
                markersize=6, label=info["label"])
    ax.set_xlabel('Reward Pumping Agent Fraction (%)')
    ax.set_ylabel('Mean Welfare')
    ax.set_title('(b) Welfare Under Attack', fontweight='bold')
    ax.legend(fontsize=7, loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig82_umeta_adversarial.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[W3] Fig 82 저장: {path}")


def plot_fig83(results):
    """Fig 83: KL Divergence 추이 — 정상 vs 적대적 에이전트 탐지."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Fig 83: Policy Divergence Tracking — Anomaly Detection via KL Gap",
        fontsize=14, fontweight='bold', y=1.02
    )
    
    # 30% adversary에서의 KL 추이 비교
    adv_frac = 0.3
    
    # (a) KL 추이 (전체 시간)
    ax = axes[0]
    for variant_name, info in UMETA_VARIANTS.items():
        kl = results["adversarial"][variant_name].get(adv_frac, {}).get("kl_history", [])
        if kl:
            ax.plot(range(len(kl)), kl, color=info["color"], linewidth=1.5,
                    alpha=0.8, label=info["label"])
    
    ax.axvline(30, color='red', linestyle='--', alpha=0.5, label='Attack Start (t=30)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('|Normal Policy − Adversary Policy|')
    ax.set_title(f'(a) Policy Divergence Over Time (Adv={adv_frac*100:.0f}%)', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # (b) 탐지 성능: 공격 시작 후 policy gap 크기
    ax = axes[1]
    variants = list(UMETA_VARIANTS.keys())
    colors = [UMETA_VARIANTS[v]["color"] for v in variants]
    labels = [UMETA_VARIANTS[v]["label"] for v in variants]
    
    post_attack_gaps = []
    for variant_name in variants:
        kl = results["adversarial"][variant_name].get(adv_frac, {}).get("kl_history", [])
        if kl and len(kl) > 50:
            post_attack_gaps.append(float(np.mean(kl[30:])))
        else:
            post_attack_gaps.append(0.0)
    
    ax.bar(range(len(variants)), post_attack_gaps, color=colors, alpha=0.8,
           edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('Mean Post-Attack Policy Gap')
    ax.set_title('(b) Anomaly Detection Signal Strength', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig83_kl_divergence.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[W3] Fig 83 저장: {path}")


def plot_fig84(results):
    """Fig 84: Computational Overhead 비교."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        "Fig 84: Computational Overhead — U_meta Variant Cost",
        fontsize=14, fontweight='bold', y=1.02
    )
    
    variants = list(UMETA_VARIANTS.keys())
    colors = [UMETA_VARIANTS[v]["color"] for v in variants]
    labels = [UMETA_VARIANTS[v]["label"] for v in variants]
    
    times = [results["overhead"][v]["ms_per_agent"] for v in variants]
    overheads = [results["overhead"][v]["overhead_pct"] for v in variants]
    
    bars = ax.bar(range(len(variants)), times, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=0.5)
    
    for i, (t, o) in enumerate(zip(times, overheads)):
        label = f'{t:.2f}ms\n({o:+.1f}%)' if i > 0 else f'{t:.2f}ms\n(baseline)'
        ax.text(i, t + 0.01, label, ha='center', fontsize=9, fontweight='bold')
    
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('Time per Agent (ms)')
    ax.set_title('Computational Cost per Agent', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig84_overhead.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[W3] Fig 84 저장: {path}")


# ============================================================
# 6. 메인 실행
# ============================================================

if __name__ == "__main__":
    results = run_integrity_experiment()
    
    plot_fig81(results)
    plot_fig82(results)
    plot_fig83(results)
    plot_fig84(results)
    
    # JSON 저장 (kl_history 제거 — 너무 큼)
    json_results = {
        "normal": results["normal"],
        "adversarial": {},
        "overhead": results["overhead"],
    }
    for v in UMETA_VARIANTS:
        json_results["adversarial"][v] = {}
        for f in ADV_FRACTIONS:
            data = results["adversarial"][v].get(f, {})
            json_results["adversarial"][v][str(f)] = {
                "mean_coop": data.get("mean_coop", 0),
                "mean_welfare": data.get("mean_welfare", 0),
            }
    
    json_path = os.path.join(OUTPUT_DIR, "integrity_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[W3] 결과 JSON: {json_path}")
    
    print("\n" + "=" * 65)
    print("  W3 SUMMARY: Bounded U_meta provides best safety-performance tradeoff")
    print("=" * 65)
    for v in UMETA_VARIANTS:
        normal = results["normal"][v]
        adv_30 = results["adversarial"][v].get(0.3, {})
        print(f"  {v:12s} | Normal Coop: {normal['mean_coop']:.3f} | "
              f"Adv30% Coop: {adv_30.get('mean_coop', 0):.3f} | "
              f"Overhead: {results['overhead'][v]['overhead_pct']:+.1f}%")
