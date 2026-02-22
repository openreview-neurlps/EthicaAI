# Critical Review Report: EthicaAI Paper

이 보고서는 "가장 비평적인" Reviewer 2의 관점에서 작성되었습니다.

## 🚨 Major Concerns (Rejection Risks)

### 1. Large Scale 실험의 수렴성 (Convergence) 문제
- **비판**: 논문에서 "Large Scale (Stage 3)"이라고 주장하지만, `MAX_STEPS=500`은 50x50 그리드에서 100명의 에이전트가 환경과 상호작용하기에 턱없이 부족한 시간일 수 있습니다.
- **논리적 구멍**: 에이전트들이 정책을 수렴시키기도 전에 시뮬레이션이 끝났을 가능성이 높습니다. 500 steps는 "학습(Learning)"이라기보다 "초기 탐색(Initial Exploration)"에 가깝습니다. 이 상태에서의 결과를 "검증되었다"고 주장하는 것은 과장(Overclaiming)입니다.
- **제안**: 최소 2000 steps 이상의 Long-run 실험이 필요하거나, 학습 곡선(Learning Curve)이 500 steps 이내에 평탄화됨을 증명해야 합니다.

### 2. 파라미터 해킹 (Parameter Hacking) 의혹
- **비판**: `META_WEALTH_BOOST=5.0`, `META_SURVIVAL_THRESHOLD=-5.0`과 같은 수치들이 어디서 왔는지 이론적 근거가 없습니다.
- **논리적 구멍**: 만약 이 수치를 4.0이나 -10.0으로 바꾸면 결과가 사라지는 것 아닙니까? 특정한 파라미터에서만 작동하는 이론은 "강건하지 않은(Brittle)" 결과입니다.
- **제안**: 민감도 분석(Sensitivity Analysis)이 빠져 있습니다. 주요 파라미터 변화에 따른 결과 안정성을 보여주지 않으면, "Cherry-picking"으로 간주될 수 있습니다.

### 3. Baseline H3(Gini) 결과의 '당연함' (Triviality)
- **비판**: Baseline(메타랭킹 OFF)에서도 SVO가 높으면 Gini가 개선됩니다(p<0.0001). 저자는 이를 "발견"이라고 하지만, 이는 수학적 필연입니다.
- **논리적 구멍**: 이타적(Altruistic) 에이전트는 정의상 "타인의 보상 평균"을 최대화하려 합니다. 당연히 불평등이 줄어들겠죠. 이것은 강화학습이나 메타랭킹의 효과가 아니라, 그냥 목적함수의 정의 때문입니다.
- **제안**: 이를 대단한 발견인 양 포장하지 말고, "예상된 베이스라인"으로 톤을 낮춰야 합니다. 진짜 발견은 **"Gini는 개선되지만(당연함), '개인 보상'은 메타랭킹 없이는 개선되지 않는다(H1 비유의)"**는 점에 집중해야 합니다.

### 4. Static-Lambda의 더 큰 효과 크기? (Interpretation Issue)
- **비판**: Full Model의 $f^2=1.50$인데, Static-Lambda는 $f^2=2.53$입니다. 효과 크기만 보면 Static이 더 강력합니다.
- **논리적 구멍**: 그런데 왜 논문은 "Dynamic Lambda가 핵심"이라고 주장합니까? $p$-value가 더 낮다는 이유만으로는 부족합니다. 오히려 "복잡한 Dynamic Lambda가 불필요하다"는 반증일 수 있습니다.
- **제안**: Dynamic Lambda가 "평균적 효과"는 작을지라도 "생존 안정성(Variance Reduction)"에 기여한다는 증거를 제시해야 합니다. 단순히 p-value로 퉁치고 넘어가면 안 됩니다.

## ⚠️ Minor Issues

1. **"GPU 가속" 강조의 과잉**: 방법론 섹션 외에 서론/결론에서도 GPU 속도를 너무 강조합니다. 이것은 시스템 성능 논문이 아니라 사회과학/AI 논문입니다. 기술적 성취는 부록이나 구현 섹션으로 미루세요.
2. **협력률(Cooperation Rate) 측정의 모호함**: 협력률이 모든 조건에서 ~12%로 비슷합니다. 측정 방식(Beam 사용 빈도 vs 사과 채집 빈도?)이 변별력이 없는 것 아닙니까?

---

## 🛡️ 방어 전략 (Rebuttal Plan)

1. **수렴성 방어**: 500 steps라도 JAX Vmap으로 인해 샘플 효율이 좋았음을 주장하거나, 추가 실험(Longer Run) 수행.
2. **트리비얼리티 방어**: H3(Gini)는 예상된 결과임을 인정하고, H1(보상)의 차별성을 핵심으로 부각.
3. **Static vs Dynamic**: 해석 수정. Static은 "무조건적 헌신"이라 효과가 클 수 있지만 위험(손해)도 큼. Dynamic은 "안전한 헌신"이라 효과는 적당하지만 신뢰도(p값)가 높음.
