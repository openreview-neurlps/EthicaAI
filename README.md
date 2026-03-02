# 🧬 EthicaAI: The Moral Commitment Spectrum

> **From Situational to Unconditional: The Spectrum of Moral Commitment Required for Multi-Agent Survival in Non-linear Social Dilemmas**

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18812419-blue?style=for-the-badge&logo=zenodo)](https://doi.org/10.5281/zenodo.18812419)
[![NeurIPS 2026](https://img.shields.io/badge/Target-NeurIPS_2026-purple?style=for-the-badge)](https://neurips.cc)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## 📄 Abstract

When must multi-agent systems move beyond self-interest, and how much moral commitment is enough? We computationally instantiate Amartya Sen's **Meta-Ranking** theory — where agents' commitment level λₜ dynamically adapts to resource availability — in Public Goods Games of increasing environmental severity.

### Key Contributions

1. **(C1) Computational Meta-Ranking**: First MARL formalization of Sen's theory with dynamic commitment λₜ ∈ [0,1] conditioned on resource state and SVO
2. **(C2) Situational Commitment in Linear Environments**: Group-level ESS (x̄=0.987) outperforming 8 baselines including M-FOS and POLA across N=20–1000 agents
3. **(C3) The Nash Trap**: In non-linear environments with tipping points, pure RL (MAPPO) converges to λ≈0.5 — an equilibrium proven via Jacobian analysis to be game-theoretic, not an activation function artifact
4. **(C4) Unconditional Commitment**: Only φ₁*=1.0 prevents collapse. Decentralized baselines (Inequity Aversion, Social Influence) achieve **0% survival** — their other-regarding mechanisms cause downward drift toward adversaries' zero contributions

**Central finding: The Moral Commitment Spectrum** — the severity of environmental non-linearity determines the minimum commitment required for collective survival.

---

## 🔬 Key Results

### Nash Trap Confirmed at Scale (N=20 and N=100)

| Method | N | Welfare | λ | Survival |
|:---|:---:|:---:|:---:|:---:|
| Selfish RL (Byz=30%) | 20 | 24.2 | 0.500 | **5.3%** |
| Selfish RL (Byz=30%) | 100 | 24.2 | 0.500 | **10.4%** |
| Unconditional (Byz=30%) | 20 | 26.6 | 0.790 | **93.6%** |
| Unconditional (Byz=30%) | 100 | 26.6 | 0.790 | **93.6%** |

### Same-Class Decentralized Baselines Fail (Byz=30%)

| Method | N=20 Survival | N=100 Survival |
|:---|:---:|:---:|
| Inequity Aversion | 0.0% | 0.0% |
| Social Influence | 0.0% | 0.0% |
| **Unconditional Commitment** | **93.6%** | **93.6%** |

### SOTA Comparison (Linear PGG, N=50)

| Method | Coop | Welfare | Byz Resilience | Decentralized |
|:---|:---:|:---:|:---:|:---:|
| M-FOS | 1.000 | 156.3 | 0.143 | ✗ |
| POLA | 1.000 | 148.3 | 0.504 | ✗ |
| LOLA | 0.000 | 101.5 | 0.000 | ✗ |
| **Meta-Ranking (Ours)** | 1.000 | 148.0 | **0.500** | ✓ |

### Extended Validations (v2.1.0)

| Validation | Result | Implication |
|:---|:---|:---|
| DNN Ablation (4 architectures) | All λ≈0.500 | Nash Trap is game-theoretic, not capacity-limited |
| KPG K=0,1,2 | All λ≈0.500, 1.8× slower | SOTA opponent-shaping fails identically |
| 5×5 Spatial Grid | Selfish 0%, Unconditional 34.7% | Pattern extends to spatially-extended environments |
| tanh activation | λ≈0.500 | Not sigmoid initialization artifact |

---

## 📂 Repository Structure

```
EthicaAI/
├── paper/                    # 📄 Unified paper (LaTeX + PDF)
│   ├── unified_paper.tex     #   Main manuscript (13 pages)
│   ├── unified_paper.pdf     #   Compiled PDF
│   ├── unified_references.bib
│   └── neurips2026_main.tex  #   Paper 1 (legacy)
├── scripts/                  # 🧪 Experiment scripts
│   ├── mappo_emergence.py    #   Nash Trap RL experiment
│   ├── extended_experiments.py # N=100, IA/SI baselines, f(R_t) sensitivity
│   ├── kpg_experiment.py     #   K-Level Policy Gradient comparison
│   ├── spatial_dilemma.py    #   5×5 Grid Spatial Social Dilemma
│   ├── paper2_figures.py     #   Figure generation
│   └── zenodo_upload.py      #   Zenodo deployment
├── outputs/                  # 📊 Experiment results (JSON)
├── simulation/               # 🎮 JAX simulation core
└── experiments/              # 🧪 Legacy experiment configs
```

---

## 💻 Quick Start

```bash
# Clone
git clone https://github.com/Yesol-Pilot/EthicaAI.git
cd EthicaAI

# Run Nash Trap experiment (CPU, ~5 min)
python scripts/mappo_emergence.py

# Run extended experiments (N=100, baselines, sensitivity)
python scripts/extended_experiments.py

# Run full JAX simulations (requires NVIDIA GPU + WSL2)
bash scripts/setup_env.sh
bash scripts/run_evolution_gpu.sh
```

---

## 📝 Citation

```bibtex
@software{ethicaai2026,
  title   = {From Situational to Unconditional: The Spectrum of Moral
             Commitment Required for Multi-Agent Survival in Non-linear
             Social Dilemmas},
  author  = {Yesol Heo},
  year    = {2026},
  doi     = {10.5281/zenodo.18812419},
  url     = {https://github.com/Yesol-Pilot/EthicaAI},
  version = {v2.1.0-preprint}
}
```

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| **v2.1.0** | 2026-03-02 | Rebuttal-hardened: KPG, Spatial Dilemma, DNN ablation, 11 defense points (13p) |
| v2.0.0 | 2026-02-28 | Unified paper: Moral Commitment Spectrum (Paper 1 + Paper 2) |
| v1.2.0 | 2026-02-28 | Paper 1: Theorem 3, 8-algo SOTA, reviewer defense |
| v1.1.0 | 2026-02-27 | Paper 1: IPPO benchmark, ESS welfare reframing |
| v1.0.0 | 2026-02-26 | Paper 1: JAX LOLA, O(N³) collapse analysis |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

**Author**: [Yesol Heo](https://heoyesol.kr)
