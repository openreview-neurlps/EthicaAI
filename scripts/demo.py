"""
EthicaAI Interactive Demo
NeurIPS 2026 — Streamlit Dashboard

사용법:
  streamlit run demo.py

주요 기능:
  1. SVO 슬라이더로 에이전트 성향 실시간 조절
  2. Meta-Ranking ON/OFF 토글 비교
  3. PGG / IPD 환경 선택
  4. 실시간 시뮬레이션 + Wasserstein Distance 표시
"""
import math
import numpy as np

# Streamlit 동적 임포트 (없으면 안내)
try:
    import streamlit as st
except ImportError:
    print("Streamlit이 필요합니다: pip install streamlit")
    print("설치 후: streamlit run demo.py")
    exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --- 설정 ---
SVO_LABELS = {
    "Selfish (0°)": 0.0,
    "Individualist (15°)": math.pi / 12,
    "Competitive (30°)": math.pi / 6,
    "Prosocial (45°)": math.pi / 4,
    "Cooperative (60°)": math.pi / 3,
    "Altruistic (75°)": 5 * math.pi / 12,
    "Full Altruist (90°)": math.pi / 2,
}

HUMAN_PGG_DATA = {
    "contribution_rate_round1": 0.54,
    "contribution_rate_round10": 0.22,
    "decay_rate": -0.035,
}


def compute_lambda(svo_theta, wealth, use_meta=True,
                   survival=-5.0, boost=5.0):
    """Dynamic lambda 계산."""
    lam_base = math.sin(svo_theta)
    if not use_meta:
        return lam_base
    if wealth < survival:
        return 0.0
    elif wealth > boost:
        return min(1.0, 1.5 * lam_base)
    else:
        return lam_base


def simulate_pgg(svo_theta, use_meta, n_agents=4, n_rounds=10,
                 endowment=20.0, multiplier=1.6, seed=42):
    """PGG 시뮬레이션."""
    rng = np.random.RandomState(seed)
    wealth = np.zeros(n_agents)
    contribution_rates = []
    payoffs_all = []

    group_avg_prev = None
    for t in range(n_rounds):
        contributions = []
        for i in range(n_agents):
            lam = compute_lambda(svo_theta, wealth[i], use_meta)
            base_c = lam * endowment
            if group_avg_prev is not None and use_meta:
                base_c += 0.3 * (group_avg_prev - base_c)
            noise = rng.normal(0, endowment * 0.05)
            c = np.clip(base_c + noise, 0, endowment)
            contributions.append(c)

        contributions = np.array(contributions)
        total_contrib = np.sum(contributions)
        public_good = total_contrib * multiplier / n_agents
        payoffs = (endowment - contributions) + public_good

        wealth += payoffs - endowment
        contribution_rates.append(float(np.mean(contributions) / endowment))
        payoffs_all.append(float(np.mean(payoffs)))
        group_avg_prev = float(np.mean(contributions))

    return contribution_rates, payoffs_all


def simulate_ipd(svo_theta, use_meta, n_rounds=100, seed=42):
    """IPD 시뮬레이션."""
    rng = np.random.RandomState(seed)
    T, R, P, S = 5.0, 3.0, 1.0, 0.0
    wealth = [0.0, 0.0]
    coop_history = []

    coop_count = [0, 0]
    defect_count = [0, 0]

    for t in range(n_rounds):
        actions = []
        for i in range(2):
            total = coop_count[1-i] + defect_count[1-i]
            past_coop_rate = coop_count[1-i] / max(1, total)

            r_coop_self = past_coop_rate * R + (1 - past_coop_rate) * S
            r_coop_other = past_coop_rate * R + (1 - past_coop_rate) * T
            r_defect_self = past_coop_rate * T + (1 - past_coop_rate) * P
            r_defect_other = past_coop_rate * S + (1 - past_coop_rate) * P

            lam = compute_lambda(svo_theta, wealth[i], use_meta)
            beta = 0.1

            v_coop = (1-lam)*r_coop_self + lam*(r_coop_other - beta*abs(r_coop_self - r_coop_other))
            v_defect = (1-lam)*r_defect_self + lam*(r_defect_other - beta*abs(r_defect_self - r_defect_other))

            prob_coop = 1.0 / (1.0 + np.exp(-(v_coop - v_defect)))
            action = 0 if rng.random() < prob_coop else 1
            actions.append(action)

        payoff_mat = [[R, S], [T, P]]
        r0 = payoff_mat[actions[0]][actions[1]]
        r1 = payoff_mat[actions[1]][actions[0]]
        wealth[0] += r0
        wealth[1] += r1

        for i in range(2):
            if actions[i] == 0:
                coop_count[i] += 1
            else:
                defect_count[i] += 1

        coop_history.append(float(np.mean([1 - a for a in actions])))

    # 이동 평균으로 부드럽게
    window = 10
    smoothed = []
    for i in range(len(coop_history)):
        start = max(0, i - window)
        smoothed.append(np.mean(coop_history[start:i+1]))

    return smoothed


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="EthicaAI Interactive Demo",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 EthicaAI: Meta-Ranking Interactive Demo")
st.markdown("""
**Sen's Meta-Ranking Theory** (1977)를 MARL로 구현한 결과를 실시간으로 탐색하세요.
""")

# 사이드바
st.sidebar.header("⚙️ 실험 설정")
env_choice = st.sidebar.selectbox(
    "환경 선택",
    ["Public Goods Game (PGG)", "Iterated Prisoner's Dilemma (IPD)"]
)

svo_label = st.sidebar.select_slider(
    "SVO (사회적 가치 지향)",
    options=list(SVO_LABELS.keys()),
    value="Prosocial (45°)"
)
svo_theta = SVO_LABELS[svo_label]

use_meta = st.sidebar.toggle("Meta-Ranking (Dynamic λ)", value=True)
seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=999, value=42)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**λ_base** = sin({svo_label.split('(')[1]}) = {math.sin(svo_theta):.3f}")
st.sidebar.markdown(f"**Mode**: {'Dynamic' if use_meta else 'Static'}")

# 메인 콘텐츠
col1, col2 = st.columns(2)

if "PGG" in env_choice:
    # PGG 시뮬레이션
    n_rounds = st.sidebar.slider("라운드 수", 5, 30, 10)
    n_agents = st.sidebar.slider("에이전트 수", 2, 10, 4)

    rates_meta, payoffs_meta = simulate_pgg(svo_theta, True, n_agents, n_rounds, seed=seed)
    rates_base, payoffs_base = simulate_pgg(svo_theta, False, n_agents, n_rounds, seed=seed)

    # 인간 데이터
    human_curve = [HUMAN_PGG_DATA["contribution_rate_round1"] +
                   HUMAN_PGG_DATA["decay_rate"] * t for t in range(n_rounds)]

    with col1:
        st.subheader("📊 기여율 변화 (Contribution Rate)")
        fig, ax = plt.subplots(figsize=(8, 5))
        rounds = range(1, n_rounds + 1)
        ax.plot(rounds, rates_meta, 'o-', color='#2196F3', linewidth=2, label='Meta-Ranking')
        ax.plot(rounds, rates_base, 's--', color='#9E9E9E', linewidth=2, label='Baseline')
        ax.plot(rounds, human_curve, 'k--', linewidth=2, alpha=0.5, label='Human (Avg)')
        ax.fill_between(rounds, [h-0.1 for h in human_curve],
                       [h+0.1 for h in human_curve], color='gray', alpha=0.1)
        ax.set_xlabel("Round")
        ax.set_ylabel("Contribution Rate")
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("💰 보상 변화 (Payoff)")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(rounds, payoffs_meta, 'o-', color='#4CAF50', linewidth=2, label='Meta-Ranking')
        ax.plot(rounds, payoffs_base, 's--', color='#9E9E9E', linewidth=2, label='Baseline')
        ax.set_xlabel("Round")
        ax.set_ylabel("Mean Payoff")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    # WD 계산
    from scipy.stats import wasserstein_distance
    wd_meta = wasserstein_distance(rates_meta, human_curve)
    wd_base = wasserstein_distance(rates_base, human_curve)

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Wasserstein Distance (Meta)", f"{wd_meta:.4f}",
              delta=f"{wd_base - wd_meta:.4f} vs Baseline")
    m2.metric("Avg Contribution (Meta)", f"{np.mean(rates_meta):.3f}")
    m3.metric("Final λ_t", f"{compute_lambda(svo_theta, 0, use_meta):.3f}")

else:
    # IPD 시뮬레이션
    n_rounds_ipd = st.sidebar.slider("라운드 수", 50, 500, 200)

    coop_meta = simulate_ipd(svo_theta, True, n_rounds_ipd, seed)
    coop_base = simulate_ipd(svo_theta, False, n_rounds_ipd, seed)

    with col1:
        st.subheader("🤝 협력률 변화 (Cooperation Rate)")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(coop_meta, color='#2196F3', linewidth=1.5, alpha=0.8, label='Meta-Ranking')
        ax.plot(coop_base, color='#9E9E9E', linewidth=1.5, alpha=0.8, label='Baseline')
        ax.set_xlabel("Round")
        ax.set_ylabel("Cooperation Rate (10-round MA)")
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("📈 최종 통계")
        final_meta = np.mean(coop_meta[-20:])
        final_base = np.mean(coop_base[-20:])

        st.metric("최종 협력률 (Meta)", f"{final_meta:.3f}",
                  delta=f"+{final_meta - final_base:.3f} vs Baseline")
        st.metric("최종 협력률 (Baseline)", f"{final_base:.3f}")
        st.metric("Meta-Ranking 효과", f"+{(final_meta - final_base)*100:.1f}%")

st.markdown("---")
st.markdown("""
> **논문**: *Beyond Homo Economicus: Computational Verification of Sen's Meta-Ranking Theory via MARL*
> 
> **GitHub**: [Yesol-Pilot/EthicaAI](https://github.com/Yesol-Pilot/EthicaAI)
""")
