# EthicaAI — Code & Reproduction

> The Nash Trap: Why Gradient-Based Learners Fail in Tipping-Point Social Dilemmas
> and What Commitment Floors Can Do About It

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
│   │
│   │   # === Algorithm Baselines (5 families) ===
│   ├── reinforce_nash_trap.py         # Ind. REINFORCE (Linear/MLP/Critic)
│   ├── pytorch_reinforce_nash_trap.py # PyTorch REINFORCE (4 architectures)
│   ├── cleanrl_mappo_pgg.py           # CleanRL IPPO/MAPPO baselines
│   ├── cleanrl_iql_pgg.py             # IQL baseline
│   ├── cleanrl_qmix_real.py           # QMIX (real mixing network)
│   ├── lola_experiment.py             # LOLA (opponent-shaping)
│   │
│   │   # === Nash Trap Analysis ===
│   ├── phi1_with_learning.py          # phi1 commitment floor + learning
│   ├── phase_diagram.py               # Phase diagram (phi1 x beta)
│   ├── phase_diagram_with_learning.py # Phase diagram with IPPO learning
│   ├── hp_sweep_ippo.py               # HP sensitivity analysis
│   ├── impossibility_verification.py  # Escape complexity (N-scaling)
│   ├── scale_test_n100.py             # N=100 scale validation
│   │
│   │   # === Cross-Environment Validation ===
│   ├── cpr_experiment.py              # CPR (Common Pool Resource)
│   ├── harvest_nash_trap.py           # Harvest (abstracted SSD)
│   ├── cleanup_nash_trap.py           # Cleanup (abstracted SSD)
│   │
│   │   # === Advanced Experiments ===
│   ├── maccl.py                       # MACCL (adaptive commitment)
│   ├── mappo_team_reward.py           # Team reward shaping ablation
│   ├── kpg_experiment.py              # K-level anticipation
│   ├── cpo_lagrangian.py              # Lagrangian dual optimality
│   ├── fairness_analysis.py           # Gini coefficient analysis
│   ├── shock_sweep.py                 # Shock parameter sensitivity
│   ├── long_horizon_experiment.py     # Extended horizon (T=50,100,200)
│   │
│   │   # === Utilities ===
│   ├── reproduce_all.py               # One-click full reproduction
│   ├── reproduce_fast.py              # Quick smoke test (2 seeds)
│   ├── audit_submission.py            # Submission integrity checker
│   ├── generate_figures.py            # Figure generation
│   └── generate_tables.py            # Table generation
│
├── outputs/                            # Experiment results (JSON)
│   ├── cleanrl_baselines/             # IPPO/MAPPO/IQL/QMIX/LOLA
│   ├── ppo_nash_trap/                 # REINFORCE (3 architectures)
│   ├── pytorch_reinforce/             # PyTorch REINFORCE (4 archs)
│   ├── phi1_ablation/                 # phi1 floor sweep
│   ├── phase_diagram/                 # phi1 x beta heatmap
│   ├── phase_diagram_learned/         # With-learning companion
│   ├── cpr_experiment/                # CPR validation
│   ├── harvest/                       # Harvest validation
│   ├── cleanup/                       # Cleanup validation
│   ├── impossibility/                 # N-scaling verification
│   ├── scale_n100/                    # N=100 results
│   └── maccl/                         # MACCL results
│
├── Dockerfile                          # Reproducible environment
├── requirements.txt                    # Python dependencies (pinned)
└── LICENSE                             # MIT License
```

## Experiments

All experiments: N=20 agents, E=20.0, 30% Byzantine, 20 seeds, 300 episodes
(unless noted otherwise).

### Core Experiments (run by `reproduce_all.py`)

| Experiment | Script | Seeds | Paper Reference |
|---|---|---|---|
| IPPO/MAPPO | `cleanrl_mappo_pgg.py` | 20 | Table 3 |
| REINFORCE (Linear/MLP/Critic) | `reinforce_nash_trap.py` | 20 | Table 3 |
| QMIX (mixing network) | `cleanrl_qmix_real.py` | 20 | Table 3, App. F |
| LOLA (opponent-shaping) | `lola_experiment.py` | 20 | Table 3, App. F |
| phi1 Commitment Floor | `phi1_with_learning.py` | 20 | Table 5 |

### Extension Experiments (also run by `reproduce_all.py`)

| Experiment | Script | Seeds | Paper Reference |
|---|---|---|---|
| IQL | `cleanrl_iql_pgg.py` | 20 | Table 3 |
| Phase Diagram (phi1 x beta) | `phase_diagram.py` | 10 | App. G |
| Phase Diagram + Learning | `phase_diagram_with_learning.py` | 5 | App. G |
| CPR Cross-Validation | `cpr_experiment.py` | 20 | Table 4 |
| Harvest Cross-Validation | `harvest_nash_trap.py` | 20 | Table 4 |
| Cleanup Cross-Validation | `cleanup_nash_trap.py` | 20 | Table 4 |
| PyTorch REINFORCE (4 archs) | `pytorch_reinforce_nash_trap.py` | 20 | Sec. 4.3 |
| Impossibility (N-scaling) | `impossibility_verification.py` | 20 | Prop. 2 |
| HP Sensitivity | `hp_sweep_ippo.py` | 10×20 | App. D |

### Standalone Experiments (run individually)

| Experiment | Script | Seeds | Paper Reference |
|---|---|---|---|
| Scale N=100 | `scale_test_n100.py` | 20 | Sec. 4.5 |
| MACCL (adaptive floor) | `maccl.py` | 20 | Sec. 4.6 |
| Team Reward Shaping | `mappo_team_reward.py` | 20 | App. E |
| K-Level Anticipation | `kpg_experiment.py` | 20 | App. F |
| CPO/Lagrangian Dual | `cpo_lagrangian.py` | 20 | App. B |
| Fairness (Gini) | `fairness_analysis.py` | 20 | App. I |
| Shock Sweep | `shock_sweep.py` | 20 | App. C |
| Long Horizon | `long_horizon_experiment.py` | 10 | App. D |

## Key Results

- **Nash Trap**: All 5 algorithm families converge to λ ≈ 0.37–0.61
  (12–80% survival, 100% trap rate)
- **Commitment Floor**: φ₁=1.0 achieves 100% survival
  (vs 12–39% at φ₁=0)
- **Phase Transition**: Sharp boundary in φ₁ × β space
- **Cross-Environment**: PGG, CPR, Harvest, Cleanup all confirm the Nash Trap
- **Scale**: N=100 preserves the trap (λ=0.497, 95% survival)
- **Impossibility**: Escape complexity grows exponentially with N

## Requirements

- Python >= 3.8
- NumPy, SciPy, Matplotlib, PyTorch (see `requirements.txt`)
- No GPU required; all experiments run on a single CPU core
- Total compute: ~4 hours on Intel i7 for full reproduction

## License

MIT License. See [LICENSE](LICENSE) for details.
