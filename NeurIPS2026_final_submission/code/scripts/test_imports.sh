#!/bin/bash
echo "=== System python check ==="
which python3
python3 --version

echo "=== System-level dmlab2d ==="
python3 -c "import dmlab2d; print('dmlab2d OK')" 2>&1

echo "=== System-level meltingpot ==="
python3 -c "from meltingpot import substrate; print('meltingpot OK')" 2>&1

echo "=== System-level torch ==="
python3 -c "import torch; print('torch', torch.__version__)" 2>&1

echo "=== pip3 packages ==="
pip3 list 2>/dev/null | grep -iE 'dmlab|meltingpot|torch|numpy'

echo "=== meltingpot_src contents ==="
ls ~/meltingpot_src/ 2>/dev/null || echo "No source dir"

echo "=== DONE ==="
