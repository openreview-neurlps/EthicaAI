#!/bin/bash
echo "=== Nuclear fix: install all + pin numpy last ==="
# Step 1: Install everything that needs numpy
pip3 install ml_collections immutabledict dm-env chex 2>&1 | tail -3

# Step 2: FORCE numpy back to 1.26.4 (dmlab2d ABI requires this)
pip3 install "numpy==1.26.4" --force-reinstall --no-deps 2>&1 | tail -3

# Step 3: Force reinstall dmlab2d to relink against current numpy
pip3 install --force-reinstall --no-deps dmlab2d 2>&1 | tail -3

echo "=== Numpy version ==="
python3 -c "import numpy; print('numpy', numpy.__version__)"

echo "=== Full chain ==="
PYTHONPATH="$HOME/meltingpot_src:$PYTHONPATH" python3 -c "
import dmlab2d; print('1. dmlab2d OK')
from meltingpot import substrate; print('2. substrate OK')
env = substrate.build('commons_harvest__open')
print('3. env built, agents:', len(env.action_spec()))
ts = env.reset()
print('4. obs shape:', ts.observation[0]['RGB'].shape)
env.close()
print('=== ALL READY ===')
" 2>&1
