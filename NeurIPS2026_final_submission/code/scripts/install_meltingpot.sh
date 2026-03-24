#!/bin/bash
set -e

echo "=== Creating Python 3.10 venv ==="
python3.10 -m venv ~/mp_env310 2>&1 || echo "venv may exist already"
source ~/mp_env310/bin/activate
python3 --version

echo "=== Installing dmlab2d + meltingpot ==="
pip install --upgrade pip -q 2>&1
pip install dmlab2d dm-meltingpot numpy -q 2>&1

echo "=== Import test ==="
python3 -c "import dmlab2d; print('dmlab2d OK')"
python3 -c "import meltingpot; print('meltingpot OK')"
python3 -c "from meltingpot import substrate; print('substrate OK')"

echo "=== ALL IMPORTS PASSED ==="
