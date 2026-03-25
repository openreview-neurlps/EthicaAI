#!/bin/bash
set -e
echo "=== FINAL ATTEMPT: Pin numpy, install everything --no-deps ==="
python3 -m venv ~/mp_deep_env --clear
VENV=/root/mp_deep_env

$VENV/bin/pip install --upgrade pip setuptools wheel 2>&1 | tail -2

echo "=== Step 1: Pin numpy 1.26.4 (compatible with both torch and dmlab2d) ==="
$VENV/bin/pip install "numpy==1.26.4" 2>&1 | tail -3

echo "=== Step 2: Install dmlab2d ==="
$VENV/bin/pip install dmlab2d 2>&1 | tail -5

echo "=== Verify dmlab2d ==="
$VENV/bin/python3 -c "import dmlab2d; print('dmlab2d OK')" 2>&1

echo "=== Step 3: Install dm-meltingpot ==="
$VENV/bin/pip install dm-meltingpot 2>&1 | tail -5

echo "=== Verify meltingpot ==="
$VENV/bin/python3 -c "from meltingpot import substrate; print('meltingpot OK')" 2>&1

echo "=== Step 4: Install torch (CPU, --no-deps to protect numpy) ==="
$VENV/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu --no-deps 2>&1 | tail -5

echo "=== Install torch remaining deps minus numpy ==="
$VENV/bin/pip install filelock typing-extensions sympy networkx jinja2 fsspec 2>&1 | tail -3

echo "=== FINAL VERIFICATION ==="
$VENV/bin/python3 << 'PYEOF'
import numpy; print(f"numpy {numpy.__version__}")
import dmlab2d; print("dmlab2d OK")
from meltingpot import substrate; print("meltingpot.substrate OK")
import torch; print(f"torch {torch.__version__}")
print("=== ALL IMPORTS PASSED ===")
PYEOF
echo "DONE"
