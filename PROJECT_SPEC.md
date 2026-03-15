# 🧬 EthicaAI — PROJECT SPEC

> **최종 업데이트**: 2026-03-15 | **유형**: NeurIPS 2026 논문 프로젝트
> **상위 문서**: [PAPER PROJECT_SPEC](../PROJECT_SPEC.md) | [마스터 바이블](file:///d:/00.test/FOLDER_BIBLE.md)

---

## 개요

| 항목 | 값 |
|------|------|
| **GitHub** | Yesol-Pilot/EthicaAI (🔒 private) |
| **도메인** | ethica.neogenesis.app |
| **Vercel Project ID** | `prj_h3GB9PzWwi1AM5pymwrmR8USrL5M` |
| **브랜치** | main |
| **커밋** | **140개** (📈 +57, 이전: 83) |
| **최종 커밋일** | 2026-03-15 |
| **크기** | ~386MB (PAPER/EthicaAI) + ~5,362MB (neo-genesis/src/sbu/ethicaai 데이터) |

---

## 설명

**"Beyond Homo Economicus: Causal Mechanisms of Emergent Cooperation in Heterogeneous Multi-Agent Systems"**

NeurIPS 2026 submission 논문. 멀티에이전트 강화학습 시뮬레이션을 통해
이기적 에이전트들 사이에서 협력이 어떻게 자발적으로 형성되는지 인과적으로 분석.

---

## 디렉토리 구조

```
EthicaAI/
├── src/                    ← Python 시뮬레이션 코드
│   ├── environments/       ← 실험 환경 (CPR 등)
│   ├── agents/             ← 에이전트 구현
│   ├── analysis/           ← 인과 분석 (Causal Forest 등)
│   └── utils/              ← 유틸리티
├── paper/                  ← LaTeX 논문
│   └── submission/         ← NeurIPS 제출용 (별도 Git)
├── site/                   ← 정적 웹사이트 (Vercel 배포)
│   ├── index.html          ← 메인 페이지
│   ├── figures/            ← 70+ 실험 그래프 (PNG/PDF)
│   ├── robots.txt
│   ├── sitemap.xml
│   ├── vercel.json         ← { "framework": null }
│   └── .vercel/project.json
├── experiments/            ← 실험 스크립트
└── results/                ← 실험 결과 데이터
```

---

## 배포

웹사이트는 `site/` 폴더에서 배포 (정적 사이트):

```bash
cd site
npx vercel --prod --yes
```

> ⚠️ 루트가 아니라 **site/** 에서 배포. Vercel 프로젝트 설정이 site/ 하위에 있음.

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| 시뮬레이션 | Python (PyTorch, JAX) |
| 분석 | Causal Forest, ATE, SVO |
| 논문 | LaTeX |
| 웹사이트 | 정적 HTML |
| GPU | WSL2 GPU 실행 가능 (워크플로우: `/gpu-run`) |
