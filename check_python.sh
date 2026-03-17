#!/bin/bash
echo "=== Checking available Python versions ==="
ls /usr/bin/python3.* 2>/dev/null || echo "No python3.x found"
which python3.10 2>/dev/null || echo "python3.10 NOT found"
which python3.11 2>/dev/null || echo "python3.11 NOT found"
python3 --version

echo "=== Try installing python3.10 ==="
apt list --installed 2>/dev/null | grep python3.10 || echo "python3.10 not installed"

echo "=== Trying pip install with current python ==="
pip install --no-deps dm-meltingpot 2>&1 | tail -5

echo "=== Trying dmlab2d wheel search ==="
pip install dmlab2d --no-build-isolation 2>&1 | tail -5
