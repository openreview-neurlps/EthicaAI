#!/bin/bash
set -e
source ~/ethicaai_env/bin/activate

echo "=== Attempt 1: pip install with specific version ==="
pip install "dm-meltingpot>=2.2.0" 2>&1 || echo "ATTEMPT1_FAILED"

echo "=== Attempt 2: pip install dmlab2d from PyPI ==="
pip install dmlab2d 2>&1 || echo "DMLAB2D_FAILED"

echo "=== Attempt 3: Check if meltingpot works without dmlab2d ==="
python3 -c "
try:
    from meltingpot import substrate
    configs = substrate.AVAILABLE_SUBSTRATES[:5]
    print('Available substrates:', configs)
except Exception as e:
    print('ERROR:', e)
" 2>&1

echo "=== Checking Python version ==="
python3 --version
pip --version

echo "=== Checking available meltingpot packages ==="
pip index versions dm-meltingpot 2>&1 || pip install dm-meltingpot==2.2.0 2>&1 || echo "ALL_FAILED"

echo "=== Final check ==="
python3 -c "import meltingpot; print('OK:', meltingpot.__version__)" 2>&1 || echo "MELTINGPOT_NOT_AVAILABLE"
