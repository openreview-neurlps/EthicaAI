#!/bin/bash
VENV=/root/meltingpot_env
echo "=== pip path ==="
$VENV/bin/pip --version
echo "=== Checking torch ==="
$VENV/bin/pip list 2>/dev/null | grep -i torch
echo "=== Installing torch (CPU, lightweight) ==="
$VENV/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -5
echo "=== Verify torch ==="
$VENV/bin/python3 -c "import torch; print('torch', torch.__version__)" 2>&1
echo "=== Full verify ==="
$VENV/bin/python3 -c "
import dmlab2d; print('dmlab2d OK')
from meltingpot import substrate; print('substrate OK')
import torch; print('torch OK:', torch.__version__)
print('ALL IMPORTS PASSED')
" 2>&1
