# EthicaAI: Commitment Floors for Tipping-Point Commons

> **Commitment Floors for Tipping-Point Commons: Escaping Nash Traps in Multi-Agent Reinforcement Learning**
>
> NeurIPS 2026 Submission

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Reproducible](https://img.shields.io/badge/Reproducible-✓-brightgreen?style=for-the-badge)]()

---

## ⚠️ Single Source of Truth

The **submission-ready paper and code** live in:

```
NeurIPS2026_final_submission/
├── paper/unified_paper.tex    ← Paper LaTeX source
└── code/                      ← All experiment scripts, outputs, Dockerfile
```

> **Note**: The root `paper/` and `scripts/` directories are legacy (pre-submission) and should NOT be used for review. See `.gitignore` for details.

---

## Quick Start (Reviewer — 5 minutes)

```bash
cd NeurIPS2026_final_submission/code

# Install dependencies (NumPy only — no GPU required)
pip install -r requirements.txt

# FAST smoke test (~5 min, 2 seeds — NOT paper numbers)
ETHICAAI_FAST=1 python scripts/reproduce_fast.py
```

> ⚠️ **FAST mode uses 2 seeds for quick validation. Paper tables report 20-seed results.** To reproduce exact paper numbers, run the full pipeline below.

## Full Reproduction (~4 hours)

```bash
cd NeurIPS2026_final_submission/code

# Full reproduction (20 seeds, all experiments)
python scripts/reproduce_all.py

# Verify tables match JSON outputs (SSOT check)
python scripts/generate_tables.py --check

# Run submission audit (0 FAIL = ready)
python scripts/audit_submission.py
```

## Docker Reproduction

```bash
docker build -t ethicaai .

# Full (20 seeds, ~4 hours)
docker run ethicaai

# FAST sanity check (~5 min)
docker run -e ETHICAAI_FAST=1 ethicaai
```

---

## Output Directory Structure

```text
code/outputs/          ← FULL 20-seed results (PAPER DATA — committed)
code/outputs_fast/     ← FAST 2-seed smoke-test results (gitignored)
```

> **Important**: `outputs/` contains the official 20-seed results used in all paper tables. `outputs_fast/` is for quick validation only and is excluded from version control. Running `reproduce_all.py` without `ETHICAAI_FAST=1` writes to `outputs/`.

## Experiment → Paper Mapping

All paper tables are **auto-generated from JSON** via `generate_tables.py` (SSOT enforced).

| Paper Reference | Script | Output JSON |
|---|---|---|
| Table 3 (RL Emergence) | `cleanrl_mappo_pgg.py` | `outputs/cleanrl_baselines/` |
| Table 3 (REINFORCE) | `reinforce_nash_trap.py` | `outputs/ppo_nash_trap/` |
| Table 3 (QMIX/LOLA) | `cleanrl_qmix_real.py`, `lola_experiment.py` | `outputs/cleanrl_baselines/` |
| Table 5 (φ₁ Sweep) | `phi1_with_learning.py` | `outputs/phi1_ablation/` |
| Table 6 (Phase Diagram) | `phase_diagram_with_learning.py` | `outputs/phase_diagram_learned/` |
| App. D (HP Sweep) | `hp_sweep_ippo.py` | `outputs/ppo_nash_trap/` |
| App. H (CPR) | `cpr_experiment.py` | `outputs/cpr_experiment/` |

---

## Key Results

- **Nash Trap**: All 7 tested RL implementations converge to λ ≈ 0.37–0.58 (subcritical commitment)
- **Commitment Floor**: φ₁=1.0 achieves 100% survival under 30% Byzantine adversaries
- **Phase Transition**: Clear boundary in φ₁ × β space confirmed with and without learning
- **Cross-Environment**: CPR environment validates the Moral Commitment Spectrum

## Requirements

- Python ≥ 3.8
- NumPy, SciPy, Matplotlib (see `requirements.txt`)
- No GPU required; all experiments run on a single CPU core
- Total compute: ~4 hours on Intel i7 for full reproduction

## License

MIT License — see [LICENSE](LICENSE) for details.
