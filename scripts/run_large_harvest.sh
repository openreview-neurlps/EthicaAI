#!/bin/bash
# EthicaAI: Run Large Scale Harvest Experiment
# 100 Agents, Harvest Environment

source ~/ethica_env/bin/activate
cd /mnt/d/00.test/PAPER/EthicaAI

echo "Starting Large Scale Harvest Experiment..."
echo "Config: CONFIG_LARGE_HARVEST (100 agents, 50x50 grid)"

# Run Full Pipeline with 'large_harvest' stage
# This will execute:
# 1. SVO Sweep (7 SVOs x 5 Seeds) - Note: 5 seeds for Harvest due to complexity
# 2. Causal Analysis
# 3. Figure Generation
python -m simulation.jax.run_full_pipeline large_harvest

echo "Harvest Experiment Complete."
