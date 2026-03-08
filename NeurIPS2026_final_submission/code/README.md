# EthicaAI — Code & Reproduction

> From Situational to Unconditional: The Spectrum of Moral Commitment Required for
> Multi-Agent Survival in Non-linear Social Dilemmas

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Quick smoke test (~1 minute, 2 seeds, core experiments only)
ETHICAAI_FAST=1 python scripts/reproduce_fast.py

# Full reproduction (~4 hours, 20 seeds, all experiments)
python scripts/reproduce_all.py
```

## Docker

```bash
docker build -t ethicaai .
docker run ethicaai                                    # Full (20 seeds)
docker run ethicaai python scripts/reproduce_all.py    # Same as above
```

## Project Structure

```
code/
├── scripts/
│   ├── envs/
│   │   └── nonlinear_pgg_env.py       # Gymnasium-style PGG environment
│   ├── cleanrl_mappo_pgg.py           # CleanRL IPPO/MAPPO baselines
│   ├── cleanrl_iql_pgg.py             # IQL baseline (no mixing network)
│   ├── cleanrl_qmix_real.py           # QMIX (real mixing network)
│   ├── lola_experiment.py             # LOLA (opponent-shaping)
│   ├── ppo_nash_trap.py               # Ind. REINFORCE (Linear/MLP/Critic)
│   ├── phi1_with_learning.py          # phi1 commitment floor + learning
│   ├── phase_diagram.py               # Phase diagram (phi1 x beta heatmap)
│   ├── cpr_experiment.py              # CPR cross-environment validation
│   ├── hp_sweep_ippo.py               # HP sensitivity analysis
│   ├── reproduce_all.py               # One-click full reproduction
│   ├── reproduce_fast.py              # Quick smoke test (2 seeds)
│   └── audit_submission.py            # Submission integrity checker
├── outputs/                            # Experiment results (JSON)
│   ├── cleanrl_baselines/             # IPPO/MAPPO/IQL/QMIX/LOLA
│   ├── ppo_nash_trap/                 # REINFORCE (3 architectures)
│   ├── phi1_ablation/                 # phi1 floor sweep
│   ├── phase_diagram/                 # phi1 x beta heatmap
│   └── cpr_experiment/                # CPR validation
├── Dockerfile                          # Reproducible environment
├── requirements.txt                    # Python dependencies (pinned)
└── LICENSE                             # MIT License
```

## Experiments

All experiments: N=20 agents, E=20.0, 30% Byzantine, 20 seeds (unless noted).

### Core Experiments (run by `reproduce_all.py`)

| Experiment | Script | Seeds | Paper Reference |
|---|---|---|---|
| IPPO/MAPPO | `cleanrl_mappo_pgg.py` | 20 | Table 3 |
| REINFORCE (Linear/MLP/Critic) | `ppo_nash_trap.py` | 20 | Table 3 |
| QMIX (mixing network) | `cleanrl_qmix_real.py` | 20 | Table 3, App. F |
| LOLA (opponent-shaping) | `lola_experiment.py` | 20 | Table 3, App. F |
| phi1 Commitment Floor | `phi1_with_learning.py` | 20 | Table 5 |

### Extension Experiments (also run by `reproduce_all.py` if available)

| Experiment | Script | Seeds | Paper Reference |
|---|---|---|---|
| IQL | `cleanrl_iql_pgg.py` | 20 | Table 3 |
| Phase Diagram (phi1 x beta) | `phase_diagram.py` | 10 | App. G |
| CPR Cross-Validation | `cpr_experiment.py` | 20 | App. H |
| HP Sensitivity | `hp_sweep_ippo.py` | 10x20 | App. D |

## Key Results

- **Nash Trap**: All 7 paradigms converge to lambda ~ 0.37-0.58 (26-72% survival)
- **Commitment Floor**: phi1=1.0 achieves 100% survival (vs 39% at phi1=0)
- **Phase Transition**: Clear boundary in phi1 x beta space
- **Cross-Environment**: CPR confirms the Moral Commitment Spectrum

## Requirements

- Python >= 3.8
- NumPy, SciPy, Matplotlib (see `requirements.txt`)
- No GPU required; all experiments run on a single CPU core
- Total compute: ~4 hours on Intel i7 for full reproduction

## License

MIT License. See [LICENSE](LICENSE) for details.
