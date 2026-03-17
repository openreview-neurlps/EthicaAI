#!/bin/bash
set -e
source ~/ethicaai_env/bin/activate
echo "=== Installing dmlab2d ==="
pip install dmlab2d -q 2>&1
echo "=== Installing dm-meltingpot ==="
pip install dm-meltingpot -q 2>&1
echo "=== Verifying ==="
python3 -c "import meltingpot; print('meltingpot:', meltingpot.__version__)"
python3 -c "import dmlab2d; print('dmlab2d OK')"
echo "=== Running experiment ==="
cd /mnt/d/00.test/PAPER/EthicaAI/NeurIPS2026_final_submission/code
python3 scripts/meltingpot_experiment.py 2>&1
echo "=== DONE ==="
