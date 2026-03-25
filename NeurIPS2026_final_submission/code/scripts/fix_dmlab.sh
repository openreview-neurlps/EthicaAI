#!/bin/bash
source ~/meltingpot_env/bin/activate
echo "Python: $(which python3)"
echo "Pip: $(which pip)"
pip install dmlab2d 2>&1 | tail -5
python3 -c "import dmlab2d; print('dmlab2d OK')"
python3 -c "from meltingpot import substrate; print('substrate OK')"
echo "=== ALL IMPORTS OK ==="
