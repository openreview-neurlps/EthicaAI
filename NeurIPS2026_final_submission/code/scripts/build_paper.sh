#!/bin/bash
set -e
cd /mnt/d/00.test/PAPER/EthicaAI/NeurIPS2026_final_submission/paper
echo "=== Pass 1: pdflatex ==="
pdflatex -interaction=nonstopmode unified_paper.tex > /dev/null 2>&1 || true
echo "=== Pass 2: bibtex ==="
bibtex unified_paper > /dev/null 2>&1 || true
echo "=== Pass 3: pdflatex ==="
pdflatex -interaction=nonstopmode unified_paper.tex > /dev/null 2>&1 || true
echo "=== Pass 4: pdflatex ==="
pdflatex -interaction=nonstopmode unified_paper.tex 2>&1 | grep -E "^(Output|!|LaTeX Warning)" | head -10
echo "=== BUILD COMPLETE ==="
ls -la unified_paper.pdf
