# EthicaAI 연구 고도화 및 후속 연구 로드맵

> NeurIPS 2026 Workshop 이후 → 풀 컨퍼런스/저널 논문 + 2편째 확장

---

## 현재 연구의 위치와 한계

```
                        ┌────────────────┐
                        │  우리 (v1)     │
                        │  MARL + Sen    │
                        │  시뮬레이션    │
                        └───────┬────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
   고도화 방향 1           고도화 방향 2           고도화 방향 3
   인간 실험 비교         LLM 에이전트 확장      제도 설계 연계
   (검증/신뢰도)          (실용성/임팩트)         (이론적 깊이)
```

---

## 고도화 방향 1: 인간 행동 데이터 비교 (가장 임팩트 높음)

### 왜?
- 리뷰어가 "시뮬레이션만으로는 부족"이라고 할 가능성 높음
- **인간 실험을 직접 안 해도**, 기존 오픈 데이터와 비교하면 논문 가치 대폭 상승

### 사용 가능한 오픈 데이터
| 데이터셋 | 출처 | 규모 | 적합도 |
|----------|------|------|:---:|
| **Asymmetric PGG** | Zenodo (2025) | 4인 그룹, 불평등 조건 | ⭐⭐⭐⭐⭐ |
| **Large-Scale PGG with Punishment** | arXiv (2025) | **7,100명, 147K 결정** | ⭐⭐⭐⭐⭐ |
| **PGG Temporal Classification** | LDM (2025) | 140명, 10라운드 | ⭐⭐⭐ |
| **Cooperation & Variance** | Frontiers (2024) | 120명 | ⭐⭐⭐ |

### 구현 계획
```python
# 인간 vs AI 에이전트 행동 비교 파이프라인
class HumanAIComparison:
    def compare_cooperation_curves(self, human_data, agent_data):
        """인간과 AI의 협력 곡선 패턴 비교"""
        # Wasserstein distance, KL divergence 등
        
    def compare_role_specialization(self, human_data, agent_data):
        """인간에서도 역할 분화가 관찰되는지 확인"""
        
    def compare_commitment_types(self, human_data, agent_data):
        """인간의 '상황적 헌신' 패턴과 AI의 패턴 일치도"""
```

### 예상 논문 기여
> "AI 에이전트의 '상황적 헌신' 패턴이 인간 Public Goods Game 데이터와
> Wasserstein 거리 0.03 이내로 일치" → **AI가 인간의 도덕적 행동을 재발견**

---

## 고도화 방향 2: LLM 에이전트로 확장 (가장 트렌디)

### 왜?
- 2026은 "Multi-Agent Systems의 해" — LLM 에이전트가 대세
- 현재 PPO 에이전트 → **GPT/Claude 기반 에이전트로 전환**하면 폭발적 관심

### 연구 질문
> "LLM 에이전트도 Meta-Ranking 구조를 부여하면 도덕적 행동이 emerges 하는가?"

### 실험 설계
```
Phase 1: LLM에 자연어로 Meta-Ranking 지시
  "당신은 자원이 부족할 때 자기 보존을 우선하고, 
   자원이 충분할 때 타인을 돕는 에이전트입니다."

Phase 2: Constitutional AI 방식
  "헌법: 1. 생존이 위협받으면 자기 이익 우선
          2. 여유가 있으면 집단 복지 고려
          3. 완전한 자기 희생은 금지"

Phase 3: PPO 결과와 LLM 결과 비교
  → 동일한 ESS(상황적 헌신)가 나타나는지 확인
```

### Anthropic Constitutional AI와의 시너지
- Constitutional AI = **외부 헌법** (인간이 작성)
- EthicaAI Meta-Ranking = **내적 헌법** (에이전트가 학습)
- 둘의 비교/통합 = **매우 높은 임팩트** 논문

---

## 고도화 방향 3: 제도 설계(Mechanism Design) 연계 (가장 이론적)

### 왜?
- MIT Christoffersen (2025)의 "Formal Contracting"과 직접 대비 가능
- 내적 도덕(Meta-Ranking) vs 외적 제도(Contract) 비교 연구

### 연구 질문
> "Meta-Ranking(내적 도덕)과 Formal Contracting(외적 제도)을 
> 결합하면 어느 쪽이 더 robustR한 협력을 유도하는가?"

### Ostrom의 8원칙과 연계
| Ostrom 원칙 | Meta-Ranking 매핑 | 구현 |
|-------------|-------------------|------|
| 명확한 경계 | 환경 범위 | ENV_HEIGHT/WIDTH |
| 비용-편익 비례 | ψ (자기 제어 비용) | META_BETA |
| 집합적 선택 | λ_t 동적 조정 | META_USE_DYNAMIC |
| 감시 | Gini 계수 추적 | eval_jax |
| 제재 | 비용 구조 | COST_BEAM |

---

## 출판 전략 (2026~2027)

| 시기 | 목표 | 논문 | 타겟 |
|------|------|------|------|
| **2026.9** | v2 Workshop | 현재 논문 + 100-agent | NeurIPS Workshop |
| **2027.1** | v3 Conference | + 인간 비교 | AAMAS 2027 |
| **2027.6** | v4 Journal | + LLM 확장 | JAAMAS 또는 Autonomous Agents |
| **2027.9** | 2편째 | Meta-Ranking × Constitutional AI | NeurIPS Main |

---

## 즉시 시작 가능한 작업 (이번 주)

1. **Zenodo PGG 데이터셋 다운로드** (무료, 즉시)
2. **인간-AI 비교 스크립트 작성** (실험 결과 나오면 바로 비교)
3. **LLM 에이전트 프로토타입** (Gemini API로 소규모 테스트)
