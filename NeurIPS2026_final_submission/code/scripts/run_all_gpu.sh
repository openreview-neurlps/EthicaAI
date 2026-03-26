#!/bin/bash
# =============================================================
# EthicaAI Deep MARL: 전체 실험 실행 + 결과 저장
# WSL2 GPU (RTX 4070 SUPER) 로컬 실행
# =============================================================
set -e

source ~/ethica_pytorch/bin/activate

SCRIPT_DIR="/mnt/d/00.test/PAPER/EthicaAI/NeurIPS2026_final_submission/code/scripts"
OUTPUT_DIR="/mnt/d/00.test/PAPER/EthicaAI/NeurIPS2026_final_submission/code/outputs"

cd "$SCRIPT_DIR"

echo "======================================================"
echo "  EthicaAI Deep MARL - Full GPU Experiment Pipeline"
echo "  Device: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"
echo "  Start: $(date)"
echo "======================================================"

echo ""
echo "[1/4] PyTorch REINFORCE Nash Trap Verification..."
echo "------------------------------------------------------"
python3 pytorch_reinforce_nash_trap.py 2>&1 | tee "$OUTPUT_DIR/reinforce_log.txt"

echo ""
echo "[2/4] MACCL (Primal-Dual Constrained MARL)..."
echo "------------------------------------------------------"
python3 maccl_pytorch.py 2>&1 | tee "$OUTPUT_DIR/maccl_log.txt"

echo ""
echo "[3/4] LIO (Bilevel Optimization)..."
echo "------------------------------------------------------"
python3 lio_pytorch.py 2>&1 | tee "$OUTPUT_DIR/lio_log.txt"

echo ""
echo "[4/4] RND PPO Ablation..."
echo "------------------------------------------------------"
python3 rnd_ppo.py 2>&1 | tee "$OUTPUT_DIR/rnd_log.txt"

echo ""
echo "======================================================"
echo "  ALL EXPERIMENTS COMPLETE!"
echo "  End: $(date)"
echo "  Results saved to: $OUTPUT_DIR"
echo "======================================================"

ls -la "$OUTPUT_DIR/deep_models/" 2>/dev/null || echo "No deep_models dir"
ls -la "$OUTPUT_DIR/pytorch_reinforce/" 2>/dev/null || echo "No pytorch_reinforce dir"
