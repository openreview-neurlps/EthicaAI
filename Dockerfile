# ================================================================
# EthicaAI — NeurIPS 2026 Reviewer Reproduction Dockerfile
# ================================================================
# This Dockerfile builds a self-contained environment for
# reproducing the core experiments in the submitted paper.
#
# Quick Start (FAST — sanity check, ~5 min, 2 seeds):
#   docker build -t ethicaai .
#   docker run -e ETHICAAI_FAST=1 ethicaai
#
# Full Reproduction (20 seeds, ~4 hours on i7):
#   docker run ethicaai
#
# NOTE: FAST mode produces fewer seeds than reported in the paper.
#       Paper tables are generated from FULL (20-seed) runs only.
# ================================================================

FROM python:3.10-slim

WORKDIR /app

# Install dependencies (NumPy-only; no GPU required)
COPY NeurIPS2026_final_submission/code/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy submission code
COPY NeurIPS2026_final_submission/code/ ./code/

# Ensure output directories exist
RUN mkdir -p code/outputs

ENV PYTHONIOENCODING=utf-8

# Default: Full reproduction (all experiments, 20 seeds)
# Override with -e ETHICAAI_FAST=1 for quick sanity check
CMD ["python", "code/scripts/reproduce_all.py"]
