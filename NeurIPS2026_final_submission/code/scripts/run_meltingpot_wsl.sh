#!/bin/bash
set -e

# Use system python3 (has dmlab2d+meltingpot installed)
export PYTHONPATH="$HOME/meltingpot_src:$PYTHONPATH"

echo "=== Import test ==="
python3 -c "
import dmlab2d
print('dmlab2d OK')
from meltingpot import substrate
print('substrate OK')
"

echo "=== Running Melting Pot Deep MARL (NumPy CNN) ==="
python3 /mnt/d/00.test/PAPER/EthicaAI/NeurIPS2026_final_submission/code/scripts/meltingpot_cnn_numpy.py

echo "MELTING POT DEEP MARL EXPERIMENT COMPLETE!"
