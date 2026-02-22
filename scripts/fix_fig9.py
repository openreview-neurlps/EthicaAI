"""Figure 9 v2: 역할 분화 동태 시각화
X축: 학습 에폭(Epoch)
Y축: threshold_clean_std (에이전트 간 청소 임계값 표준편차)
→ STD가 높을수록 에이전트 간 '역할 분화'가 심한 것

보조 서브플롯: threshold_clean_mean의 시간 진화 (분화의 '방향')
"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open("simulation/outputs/run_large_1771029971/sweep_large_1771030421.json") as f:
    data = json.load(f)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                                gridspec_kw={'height_ratios': [2, 1]})

conditions_order = ['selfish', 'individualist', 'competitive', 'prosocial', 
                     'cooperative', 'altruistic', 'full_altruist']

# 논문 품질 컬러맵 (빨강→노랑→초록)
colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(conditions_order)))

for i, cond_name in enumerate(conditions_order):
    if cond_name not in data:
        continue
    cond = data[cond_name]
    runs = cond["runs"]
    
    # 모든 run의 threshold_clean_std를 모아서 평균
    all_stds = []
    all_means = []
    for run in runs:
        if "threshold_clean_std" in run["metrics"]:
            all_stds.append(run["metrics"]["threshold_clean_std"])
        if "threshold_clean_mean" in run["metrics"]:
            all_means.append(run["metrics"]["threshold_clean_mean"])
    
    if all_stds:
        # 모든 run의 길이를 맞추기
        min_len = min(len(s) for s in all_stds)
        stds_arr = np.array([s[:min_len] for s in all_stds])
        means_arr = np.array([m[:min_len] for m in all_means])
        
        avg_std = stds_arr.mean(axis=0)
        std_of_std = stds_arr.std(axis=0)
        avg_mean = means_arr.mean(axis=0)
        
        epochs = np.arange(min_len)
        theta_str = f"{cond['theta']:.1f}"
        
        # 상단: STD 진화 (역할 분화 정도)
        ax1.plot(epochs, avg_std, color=colors[i], linewidth=2, 
                label=f'{cond_name} (θ={theta_str})', alpha=0.9)
        ax1.fill_between(epochs, avg_std - std_of_std, avg_std + std_of_std, 
                         color=colors[i], alpha=0.1)
        
        # 하단: Mean 진화 (평균 청소 경향)
        ax2.plot(epochs, avg_mean, color=colors[i], linewidth=1.5, alpha=0.7)

# 상단 꾸미기
ax1.set_ylabel("Threshold Clean σ\n(Agent Specialization Degree)", fontsize=11)
ax1.set_title("Emergent Role Specialization Dynamics: Agent Behavioral Divergence Over Training", 
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=8, loc='upper right', ncol=2, framealpha=0.9)
ax1.grid(True, alpha=0.2)
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

# 주석: 분화 피크 구간 (초기 급등)
ax1.annotate('Peak Divergence', 
             xy=(3, 0.19),
             xytext=(20, 0.17),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=9, color='red', fontweight='bold')

# 하단 꾸미기
ax2.set_xlabel("Training Epoch", fontsize=11)
ax2.set_ylabel("Threshold Clean μ\n(Mean Clean Tendency)", fontsize=11)
ax2.grid(True, alpha=0.2)

plt.tight_layout()

# 저장
out_paths = [
    "simulation/outputs/run_large_1771029971/figures/fig9_role_specialization.png",
    "submission/figures/fig9_role_specialization.png"
]
for p in out_paths:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    plt.savefig(p, dpi=150, bbox_inches='tight')
print("Figure 9 v2 generated successfully!")
