#!/bin/bash
set -e
echo "=== Restoring original meltingpot_env ==="
VENV=/root/meltingpot_env
$VENV/bin/pip install dmlab2d dm-meltingpot 2>&1 | tail -5
echo "=== Verify ==="
$VENV/bin/python3 -c "
import dmlab2d; print('dmlab2d OK')
from meltingpot import substrate; print('meltingpot.substrate OK')
" 2>&1
echo "=== RESTORED ==="
