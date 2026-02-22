# EthicaAI vs 경쟁 연구 — 차별화 분석

> NeurIPS 2026 Workshop 제출 시 Related Work 및 리뷰어 Q&A 대비용

## 가장 가까운 경쟁 연구 3편

### 1. Rodríguez-Soto et al. (IJCAI 2023)
**"Modeling Moral Choices in Social Dilemmas with Multi-Agent RL"**

| 항목 | 그들 | **우리 (EthicaAI)** |
|------|------|---------------------|
| 도덕 이론 | 결과주의 vs 규범주의 (정적 분류) | **Sen의 메타랭킹 (동적 전환)** |
| 보상 설계 | 정적 보상 성형(reward shaping) | **동적 λ_t 가중 전환** |
| 환경 | Prisoner's/Volunteer's Dilemma | **Tragedy of the Commons (연속 환경)** |
| 에이전트 수 | 2~4 | **20~100** |
| 핵심 발견 | "도덕 에이전트가 더 잘 협력" | **"상황적 도덕만 진화적으로 안정"** |
| **우리의 차별점** | 그들은 도덕을 분류(label)로 봄. 우리는 도덕을 **과정(process)**으로 모델링 |

### 2. Calvano et al. (arXiv 2024)
**"Evolutionary MARL in Group Social Dilemmas"**

| 항목 | 그들 | **우리 (EthicaAI)** |
|------|------|---------------------|
| 학습 | Q-Learning | **PPO (deep RL)** |
| 진화 | 명시적 진화 알고리즘 | **학습 과정에서 자연 발생** |
| 도덕 | 없음 (순수 게임이론) | **Sen의 Meta-Ranking** |
| 관점 | 진화 게임 이론 | **행동 경제학 + RL** |
| **우리의 차별점** | 그들은 "진화가 어떻게 일어나는지" 봄. 우리는 **"왜 특정 도덕만 살아남는지"** 봄 |

### 3. Christoffersen et al. (MIT 2025)
**"Formal Contracting for Social Dilemmas"**

| 항목 | 그들 | **우리 (EthicaAI)** |
|------|------|---------------------|
| 메커니즘 | 외부 계약(contract) | **내적 동기(meta-ranking)** |
| 정보 요구 | 에이전트 간 게임 구조 공유 필요 | **개인 내부 상태만 필요** |
| 확장성 | 2 에이전트 | **100 에이전트** |
| 실현 가능성 | 제도적 설계 필요 | **내재적 학습** |
| **우리의 차별점** | 그들은 "외부 규칙"로 해결. 우리는 **"내적 도덕"**으로 해결 |

---

## 리뷰어 예상 질문 & 답변

### Q1: "이 연구의 가장 중요한 새로운 기여는 무엇인가?"
> Sen의 Meta-Ranking을 MARL에 최초로 형식화한 것이 아니라, **"왜 절대적 이타주의가 실패하는가"를 계산적으로 증명**한 것. 이는 AI Alignment의 "corrigibility" 문제와 직결됨.

### Q2: "20-에이전트에서 100-에이전트로의 확장이 trivial하지 않은가?"
> No. 에이전트 수가 5배 증가하면 상호작용 복잡도가 $O(n^2)$로 증가. 그럼에도 ATE의 방향과 유의성이 유지되었다는 것이 핵심(Scale Invariance). 이는 이론의 일반성을 뒷받침.

### Q3: "실험 환경이 단순하지 않은가? (Cleanup만)"
> (NeurIPS 보강에서) Harvest 환경도 추가 실험 완료. 두 환경에서 일관된 결과는 환경 의존성이 아닌 **메타랭킹 자체의 효과**임을 입증.

### Q4: "Baseline과의 공정한 비교인가?"
> Baseline = 동일 SVO + PPO, 단 Meta-Ranking OFF (λ=sin(θ) 고정). 추가로 Ablation 2종 (No-ψ, Static-λ) 수행. 총 4가지 조건 비교.

### Q5: "통계적 방법론이 견고한가?"
> 3중 검증: OLS(HAC Robust SE) + LMM + Bootstrap CI. 세 방법 모두 일관된 결과. LMM이 seed별 랜덤 효과를 통제하여 가장 엄밀.
