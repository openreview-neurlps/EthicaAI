#!/bin/bash
# Rebuild meltingpot venv with BOTH dmlab2d and torch
VENV=/root/meltingpot_env
echo "=== Step 1: Reinstall dmlab2d (may have been removed by torch) ==="
$VENV/bin/pip install dmlab2d 2>&1 | tail -5
echo "=== Step 2: Check what we have ==="
$VENV/bin/pip list 2>/dev/null | grep -iE 'dmlab|meltingpot|torch'
echo "=== Step 3: Full import test ==="
$VENV/bin/python3 -c "
import sys
print('Python:', sys.executable)
errors = []
try:
    import dmlab2d
    print('dmlab2d OK')
except ImportError as e:
    errors.append(str(e))
    print('dmlab2d FAIL:', e)
try:
    from meltingpot import substrate
    print('meltingpot.substrate OK')
except ImportError as e:
    errors.append(str(e))
    print('meltingpot FAIL:', e)
try:
    import torch
    print('torch OK:', torch.__version__)
except ImportError as e:
    errors.append(str(e))
    print('torch FAIL:', e)
if not errors:
    print('ALL IMPORTS PASSED - READY TO TRAIN')
else:
    print(f'FAILED: {len(errors)} imports broken')
" 2>&1
