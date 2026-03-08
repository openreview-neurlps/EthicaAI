# EthicaAI: The Moral Commitment Spectrum

> **From Situational to Unconditional: The Spectrum of Moral Commitment Required for Multi-Agent Survival in Non-linear Social Dilemmas**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](code/LICENSE)

---

## Abstract

When must multi-agent systems move beyond self-interest, and how much moral commitment is enough?
We investigate this through a systematic empirical study across **7 learning paradigms** in Public Goods Games with tipping-point dynamics, drawing from Amartya Sen's meta-ranking theory.

### Key Findings

- **Nash Trap**: All 7 paradigms (REINFORCE ×3, PPO, MAPPO, IQL, QMIX, LOLA) converge to suboptimal λ ≈ 0.37–0.58, yielding only 26–72% survival
- **Commitment Floor**: Only unconditional commitment (φ₁=1.0) achieves 100% survival
- **Phase Transition**: Clear boundary in φ₁ × β space (Theorem 1)
- **Cross-Environment**: CPR environment confirms the Moral Commitment Spectrum

---

## Quick Start

```bash
cd code

# Install dependencies
pip install -r requirements.txt

# Quick smoke test (~30 seconds)
ETHICAAI_FAST=1 python scripts/reproduce_all.py

# Full reproduction (~4 hours, 20 seeds, all 7 experiments)
python scripts/reproduce_all.py
```

### Docker

```bash
cd code
docker build -t ethicaai .
docker run ethicaai
```

---

## Repository Structure

```
EthicaAI-NeurIPS2026/
├── paper/                              # LaTeX source + compiled PDF
│   ├── unified_paper.tex
│   ├── unified_paper.pdf
│   └── unified_references.bib
├── code/                               # All experiment code
│   ├── scripts/                        # Experiment scripts
│   │   ├── envs/nonlinear_pgg_env.py   # Gymnasium-style PGG environment
│   │   ├── cleanrl_mappo_pgg.py        # IPPO/MAPPO baselines
│   │   ├── cleanrl_qmix_real.py        # QMIX (real mixing network)
│   │   ├── lola_experiment.py          # LOLA (opponent-shaping)
│   │   ├── ppo_nash_trap.py            # Ind. REINFORCE (3 architectures)
│   │   ├── phi1_with_learning.py       # φ₁ commitment floor + learning
│   │   ├── phase_diagram.py            # Phase diagram (φ₁ × β)
│   │   ├── cpr_experiment.py           # CPR cross-environment
│   │   └── reproduce_all.py           # One-click reproduction
│   ├── outputs/                        # Experiment results (JSON)
│   ├── requirements.txt
│   └── Dockerfile
└── README.md
```

---

## Experiments (7 Paradigms)

All experiments: N=20 agents, E=20.0, 30% Byzantine, 20 seeds.

| Experiment | Script | Paper Ref | Key Result |
|:---|:---|:---|:---|
| REINFORCE (3 arch.) | `ppo_nash_trap.py` | Table 3 | λ=0.37–0.49, 26–53% surv |
| IPPO / MAPPO | `cleanrl_mappo_pgg.py` | Table 3 | λ=0.39–0.41, 37–39% surv |
| IQL | `cleanrl_iql_pgg.py` | Table 3 | λ=0.58, 72% surv |
| QMIX | `cleanrl_qmix_real.py` | Table 3, App. F | λ=0.52, 67% surv |
| LOLA | `lola_experiment.py` | Table 3, App. F | λ=0.49, 51% surv |
| φ₁ Floor | `phi1_with_learning.py` | Table 5 | 39%→100% (monotonic) |
| Phase Diagram | `phase_diagram.py` | App. G | 11×11 heatmap |
| CPR Validation | `cpr_experiment.py` | App. H | Same pattern confirmed |

---

## License

MIT License — see [code/LICENSE](code/LICENSE) for details.
