# 🧬 EthicaAI: Computational Verification of Sen's Optimal Rationality
> *Autonomous Multi-Agent RL with Meta-Ranking for Social Choice Theory*

[![Status](https://img.shields.io/badge/Status-Research_in_Progress-blue?style=for-the-badge&logo=arxiv)](https://ethicaai.vercel.app)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18728438-blue?style=for-the-badge&logo=zenodo)](https://doi.org/10.5281/zenodo.18728438)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://python.org)
[![Engine](https://img.shields.io/badge/Engine-JAX_GPU-red?style=for-the-badge&logo=nvidia)](https://jax.readthedocs.io)
[![Brain](https://img.shields.io/badge/Brain-Gemini_2.0-orange?style=for-the-badge&logo=google-gemini)](https://ai.google.dev)

## 📄 Research Overview

**Can AI agents autonomously discover fair social contracts?**

This project computationally verifies **Amartya Sen's meta-ranking framework** — the idea that rational agents can rank their own preference orderings — through large-scale multi-agent reinforcement learning simulations. We demonstrate that meta-ranking produces **Pareto-dominant, envy-free equilibria** even under adversarial shocks, outperforming classical social welfare functions (utilitarian, Rawlsian, Nash).

### Key Results (100-Agent Full Sweep)

| Metric | Meta-Ranking | Utilitarian | Rawlsian |
|:---|:---:|:---:|:---:|
| Cooperation Rate | **0.87** | 0.71 | 0.68 |
| Gini Coefficient | **0.12** | 0.31 | 0.19 |
| Pareto Efficiency | **0.94** | 0.82 | 0.75 |
| Shock Recovery (steps) | **12** | 45 | 38 |

> **Targeting NeurIPS 2026** — [Preprint on Zenodo](https://doi.org/10.5281/zenodo.18728438)

---

## 🏛️ The Genesis Lab

**EthicaAI Genesis** is an autonomous research laboratory where AI agents live, interact, and evolve social contracts without human intervention.
Governed by a hyper-intelligent **Theorist (LLM)**, the system automatically formulates hypotheses, runs massive GPU simulations, and pivots strategies to solve the "Cooperation Dilemma".

### 🧠 The Autonomous Loop
1.  **Thinking (Theorist)**: Gemini 2.0 analyzes history and proposes a new social structure (e.g., "Let's try Inequity Aversion!").
2.  **Simulation (Engineer)**: JAX-accelerated engine runs 20 simultaneous societies (20,000+ steps) in seconds.
3.  **Judgment (Critic)**: Evaluates stability, Gini coefficient, and welfare.
4.  **Intervention (Coordinator)**: Pokes, shocks, or resets the world if stagnation is detected.

---

## 🚀 Key Features

| Feature | Description | Status |
|:---|:---|:---:|
| **GPU Revolution** | **100x Faster** simulations using JAX on RTX 4070 SUPER | ✅ |
| **Meta-Ranking** | Sen's framework as a learnable social welfare function | ✅ |
| **Self-Correction** | Automatically switches between *Adaptive*, *Inverse*, and *Institutional* modes | ✅ |
| **Inequity Aversion** | Agents feel *Envy* and *Guilt*, driving spontaneous fairness | ✅ |
| **Live Dashboard** | Real-time visualization of the evolutionary tree and metrics | ✅ |

---

## 💻 How to Run

### 1. The "Brain" (Training)
*Requires NVIDIA GPU & WSL2 (Linux)*

```bash
# Clone the repository
git clone https://github.com/Yesol-Pilot/EthicaAI.git
cd EthicaAI

# Setup Environment
bash scripts/setup_env.sh

# Start Evolution Loop
bash scripts/run_evolution_gpu.sh
```

### 2. The "Eyes" (Visualization)
*Runs on CPU (Windows/Mac/Linux)*

```bash
# Install Dashboard Dependencies
pip install -r requirements_dashboard.txt

# Launch Dashboard
streamlit run dashboard_evolution.py
```

---

## 📂 Project Structure

```
EthicaAI/
├── experiments/         # 🧪 Logs, Configs, and Results
├── simulation/          # 🎮 JAX Simulation Core
│   ├── genesis/         #   Agents (Theorist, Engineer, Critic)
│   └── jax/             #   GPU Kernels
├── paper/               # 📄 LaTeX source + compiled PDFs
├── submission_arxiv/    # 📦 arXiv submission package
├── submission_neurips/  # 📦 NeurIPS submission package
├── analysis/            # 📊 Analysis notebooks
├── dashboard/           # 📊 Streamlit dashboards
├── scripts/             # 🛠️ Automation Tools
└── original/            # 🗂️ Original prototype code
```

---

## 📝 Citation

If you use EthicaAI in your research, please cite:

```bibtex
@software{ethicaai2026,
  title={Beyond Homo Economicus: Computational Verification of
         Amartya Sen's Meta-Ranking Theory in Multi-Agent Social Dilemmas},
  author={Yesol Heo},
  year={2026},
  doi={10.5281/zenodo.18728438},
  url={https://github.com/Yesol-Pilot/EthicaAI},
  version={v5.1.1}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

**Author**: [Yesol Heo](https://heoyesol.kr)
