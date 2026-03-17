# 🧬 EthicaAI — PROJECT SPEC (BIBLE)

> **최종 업데이트**: 2026-03-17 20:40 KST | **유형**: NeurIPS 2026 논문 프로젝트
> **상위 문서**: [PAPER PROJECT_SPEC](../PROJECT_SPEC.md) | [마스터 바이블](file:///d:/00.test/FOLDER_BIBLE.md)

---

## 개요

| 항목 | 값 |
|------|------|
| **논문 제목** | The Nash Trap: How Gradient Learners Fail Public Goods and a Mechanism to Fix It |
| **타겟 학회** | NeurIPS 2026 |
| **GitHub** | Yesol-Pilot/EthicaAI (🔒 private) |
| **Anonymous 리모트** | neogenesislab/EthicaAI (🔒 private, double-blind) |
| **도메인** | ethica.neogenesis.app |
| **Vercel Project ID** | `prj_h3GB9PzWwi1AM5pymwrmR8USrL5M` |
| **브랜치** | main |
| **커밋** | **149개** |
| **최종 커밋일** | 2026-03-17 |
| **Tracked files** | 713 |
| **크기** | ~386MB |

---

## 핵심 Contribution

1. **Nash Trap**: 표준 gradient learners(IPPO, MAPPO, QMIX, LOLA 등)이 Public Goods Game에서 수렴하는 비효율적 균형 발견
2. **Impossibility Theorem**: $\Omega(e^{cN})$ 탈출 하한 정리 — standard RL은 exponential time 필요
3. **Moral Commitment Spectrum**: commitment level $\phi_1 \in [0,1]$로 개인주의-이타주의 연속체 모델링
4. **MACCL Algorithm**: Multi-Agent Cooperative Commitment Learning — primal-dual adaptive floor로 Nash Trap 탈출

---

## 디렉토리 구조 (실제 구성, 2026-03-17)

```
EthicaAI/
├── NeurIPS2026_final_submission/  ← ⭐ SINGLE SOURCE OF TRUTH
│   ├── paper/                     ← LaTeX 논문 (48p, unified_paper.tex)
│   │   ├── unified_paper.tex      ← 메인 논문 파일
│   │   ├── new_sections.tex       ← 추가 appendix 섹션
│   │   ├── appendix_impossibility_proof.tex ← Theorem 1 증명
│   │   ├── tables/                ← SSOT 테이블 (tab_emergence.tex 등)
│   │   ├── figures/               ← Phase Diagram, convergence 등
│   │   └── references.bib
│   ├── code/
│   │   ├── scripts/               ← 61개 실험 스크립트
│   │   │   ├── reproduce_all.py   ← ⭐ 전체 재현 파이프라인 (EXIT 0)
│   │   │   ├── verify_numbers.py  ← JSON↔LaTeX 수치 검증 (EXIT 0)
│   │   │   ├── audit_submission.py← NeurIPS 제출물 감사
│   │   │   ├── cleanrl_mappo_pgg.py ← IPPO/MAPPO 실험
│   │   │   ├── cleanrl_qmix_real.py ← QMIX 실험
│   │   │   ├── lola_experiment.py ← LOLA 실험
│   │   │   ├── phi1_with_learning.py ← φ₁ commitment sweep
│   │   │   ├── phase_diagram.py   ← Phase Diagram 생성
│   │   │   └── ...
│   │   ├── outputs/               ← 15 JSON 실험 결과
│   │   │   ├── cleanrl_baselines/ ← IPPO, MAPPO, QMIX, LOLA, IQL, HP sweep
│   │   │   ├── ppo_nash_trap/     ← REINFORCE 결과
│   │   │   ├── phi1_ablation/     ← φ₁ sweep 결과
│   │   │   ├── phase_diagram/     ← 1D + 2D phase diagram
│   │   │   ├── cpr_experiment/    ← CPR cross-validation
│   │   │   ├── mn_sweep/          ← M/N ratio sweep
│   │   │   ├── maccl/             ← MACCL 결과
│   │   │   ├── maccl_multi_env/   ← MACCL 교차환경
│   │   │   └── impossibility/     ← 불가능성 수치 검증
│   │   ├── Dockerfile             ← 재현용 Docker
│   │   └── README.md
│   └── supplementary/             ← NeurIPS supplementary 패키지
├── simulation/                    ← 시뮬레이션 엔진 (JAX + PyTorch)
│   ├── jax/                       ← JAX 기반 고속 시뮬레이션
│   │   ├── environments/          ← PGG, CPR, Cleanup, Coin Game
│   │   ├── training/              ← train_pipeline.py
│   │   ├── analysis/              ← 50+ 분석 스크립트
│   │   └── config.py
│   └── genesis/                   ← critic, coordinator 등
├── site/                          ← 정적 웹사이트 (Vercel)
├── scripts/                       ← 레거시 스크립트
├── .agent/workflows/              ← /audit, /deploy, /dev, /gpu-run 등
├── .gitignore                     ← zip/tar, submissions, .env 제외
├── CITATION.cff
├── LICENSE (MIT)
├── README.md
└── requirements.txt
```

---

## 실험 현황 (2026-03-17)

### reproduce_all.py EXIT 0 — 10실험 × 20 seeds

| # | 실험 | 스크립트 | JSON 출력 | Paper 참조 | 결과 |
|:-:|------|---------|-----------|-----------|------|
| 1 | IPPO/MAPPO | cleanrl_mappo_pgg.py | cleanrl_baseline_results.json | Table 3 | TRAPPED, λ=0.415 |
| 2 | REINFORCE | reinforce_nash_trap.py | ppo_nash_trap/ippo_results.json | Table 3 | TRAPPED, λ=0.415 |
| 3 | QMIX | cleanrl_qmix_real.py | qmix_real_results.json | Table 3, App F | TRAPPED |
| 4 | LOLA | lola_experiment.py | lola_results.json | Table 3, App F | TRAPPED |
| 5 | φ₁ Sweep | phi1_with_learning.py | phi1_ablation/phi1_results.json | Table 5 | PASS |
| 6 | IQL | cleanrl_iql_pgg.py | iql_baseline_results.json | Table 3 | PASS |
| 7 | Phase Diagram | phase_diagram.py | phase_diagram/phase_diagram.json | App G | PASS |
| 8 | CPR | cpr_experiment.py | cpr_experiment/cpr_results.json | App H | PASS |
| 9 | HP Sweep | hp_sweep_ippo.py | hp_sweep_results.json | App D | PASS |
| 10 | Phase w/Learning | phase_diagram_with_learning.py | phase_diagram_learned.json | App G | PASS |

### 추가 JSON (reproduce_all.py 외)
- mn_sweep/mn_sweep_results.json (12 conditions × 20 seeds)
- phase_diagram/phase_diagram_2d.json (24 conditions)
- maccl/maccl_results.json
- maccl_multi_env/multi_env_results.json
- impossibility/impossibility_results.json

---

## 논문 구조

| 항목 | 수치 |
|------|------|
| 총 페이지 | 48 (본문 9p + appendix 39p) |
| Tables | 37 |
| Figures | ~5 |
| Theorems | 2 (Impossibility + MACCL convergence) |
| Lemmas | 2 |
| Corollaries | 1 |
| Definitions | 1 |
| Sections | 33 |
| References | 40+ |
| Word count | ~19K |

### LaTeX 빌드
```powershell
cd NeurIPS2026_final_submission\paper
pdflatex unified_paper.tex; bibtex unified_paper; pdflatex unified_paper.tex; pdflatex unified_paper.tex
```
결과: 48p, 0 multiply-defined labels, 0 undefined references

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| 시뮬레이션 | Python 3.10+, JAX, PyTorch |
| 실험 환경 | Public Goods Game, CPR, Cleanup, Coin Game |
| MARL 알고리즘 | IPPO, MAPPO, QMIX, IQL, LOLA, REINFORCE, MACCL |
| 분석 | Causal Forest, ATE, SVO, Bootstrap CI |
| 논문 | LaTeX (NeurIPS 2026 style) |
| 웹사이트 | 정적 HTML (Vercel 배포) |
| GPU | WSL2 GPU 실행 가능 (워크플로우: `/gpu-run`) |
| CI | verify_numbers.py, audit_submission.py |

---

## 배포

### 웹사이트 (site/)
```powershell
cd site; npx vercel --prod --yes
```
> ⚠️ 루트가 아닌 **site/** 에서 배포. Vercel 프로젝트 설정이 site/ 하위에 있음.

### Git (dual remote)
```powershell
# origin (Yesol-Pilot) — main development
$env:GCM_INTERACTIVE="never"; $env:GIT_TERMINAL_PROMPT=0; git -c credential.helper="" push origin main 2>&1
# anon (neogenesislab) — double-blind submission
$env:GCM_INTERACTIVE="never"; $env:GIT_TERMINAL_PROMPT=0; git -c credential.helper="" push anon main 2>&1
```

---

## 변경 이력

| 날짜 | 커밋 범위 | 주요 변경 |
|------|----------|----------|
| 2026-03-17 | 140→149 | P1 수치 교정(11곳), P2 Theorem 2 보강, P3 Phase Diagram, ACL→MACCL, 저장소 정리(311 rm), reproduce_all.py 재현성 검증 |
| 2026-03-15 | 83→140 | NeurIPS final submission 구축, unified_paper.tex 통합, 20-seed baselines, SSOT tables |
| 2026-03-08 | — | 초기 코드 + 논문 구조 수립 |

---

## 자체평가: 6.5 (Weak Accept)

| Dimension | Score |
|:---------:|:-----:|
| Novelty | 7.0 |
| Significance | 6.5 |
| Soundness | 5.5 |
| Clarity | 6.0 |
| Empirical | 6.0 |
| Reproducibility | 6.0 |
| **Overall** | **6.5** |

### Accept까지 남은 과제
1. Thm 1 Part(ii) discrete-time 증명 보강
2. Melting Pot급 벤치마크 1개 추가
3. MACCL vs CPO 직접 비교 테이블
