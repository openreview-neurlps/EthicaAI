#!/bin/bash
set -e

source ~/ethica_pytorch/bin/activate

echo "=== Installing dependencies ==="
pip install pettingzoo higher scipy matplotlib 2>&1 | tail -5

echo "=== GPU Verification ==="
python3 << 'PYEOF'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x
    print("GPU compute test:", y.shape, "OK")
else:
    print("WARNING: No GPU detected, will run on CPU")
PYEOF

echo "=== Setup Complete ==="
