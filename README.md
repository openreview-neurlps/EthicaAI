# EthicaAI: The Moral Commitment Spectrum

> **From Situational to Unconditional: The Spectrum of Moral Commitment Required for Multi-Agent Survival in Non-linear Social Dilemmas**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## Abstract

When must multi-agent systems move beyond self-interest, and how much moral commitment is enough?
We investigate this question through a systematic empirical study of cooperation dynamics in Public Goods Games (PGG) of increasing environmental severity, drawing interpretive framing from Amartya Sen's meta-ranking theory.

### Key Contributions

1. **(C1) Computational Meta-Ranking**: MARL formalization of Sen's theory with dynamic commitment λₜ ∈ [0,1] conditioned on resource state and SVO
2. **(C2) Situational Commitment in Linear Environments**: Group-level ESS (x̄=0.987) outperforming 8 baselines including M-FOS and POLA
3. **(C3) Algorithm-Invariant Cooperation Failure**: Independent RL agents (Linear, MLP, Actor-Critic) all converge to suboptimal equilibria (λ≈0.05–0.49, ≤6% survival)
4. **(C4) Unconditional Commitment + Meta-Learning Validation**: Only φ₁*=1.0 prevents collapse. Meta-learning independently recovers this optimum.

**Central finding: The Moral Commitment Spectrum** — the severity of environmental non-linearity determines the minimum commitment required for collective survival.

---

## Quick Start

```bash
# Install dependencies
pip install numpy matplotlib

# Run IPPO Nash Trap experiment (CPU, ~30s)
python scripts/ppo_nash_trap.py

# Run meta-learning validation (~3 min)
python scripts/meta_learn_g.py

# Run extended experiments (N=100, baselines, sensitivity)
python scripts/extended_experiments.py
```

---

## Repository Structure

```
EthicaAI/
├── paper/                    # LaTeX source + compiled PDF
│   ├── unified_paper.tex
│   └── unified_paper.pdf
├── scripts/                  # Experiment scripts
│   ├── ppo_nash_trap.py      # IPPO 3-level Nash Trap (NEW)
│   ├── meta_learn_g.py       # Meta-learning g(θ,R) validation
│   ├── mappo_emergence.py    # REINFORCE emergence baseline
│   ├── extended_experiments.py # Scale/baseline/sensitivity tests
│   ├── kpg_experiment.py     # K-level anticipation ablation
│   ├── spatial_dilemma.py    # Spatial social dilemma
│   ├── phase_diagram.py      # Phase diagram generation
│   └── reproduce.py          # One-click reproduction
├── outputs/                  # Experiment results (JSON)
└── simulation/               # JAX simulation core
```

---

## Reproducing Results

All paper figures and tables can be reproduced from the scripts:

| Table/Figure | Script | Output |
|:---|:---|:---|
| Table 3 (IPPO) | `ppo_nash_trap.py` | `outputs/ppo_nash_trap/ippo_results.json` |
| Table 6 (Meta-learn) | `meta_learn_g.py` | `outputs/meta_learn_g/meta_learn_results.json` |
| Fig. Phase Diagram | `phase_diagram.py` | `outputs/phase_diagram/results.json` |
| Table 4 (Scale) | `extended_experiments.py` | `outputs/extended_experiments/extended_results.json` |
| Table 5 (Baselines) | `extended_experiments.py` | `outputs/extended_experiments/extended_results.json` |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
