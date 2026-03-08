#!/usr/bin/env python
"""
build_anon_zip.py - Build anonymized supplementary ZIP for NeurIPS submission
================================================================================
Creates a <100MB ZIP file containing:
- code/scripts/*.py (all experiment scripts)
- code/scripts/envs/*.py (environment modules)
- code/outputs/**/*.json (experiment results)
- code/README.md
- reproduce_all.py, reproduce_fast.py

Excludes:
- .git, __pycache__, *.pyc, .DS_Store
- Any author-identifying information
- Large binary files (plots, etc.)

Usage:
  python build_anon_zip.py
"""
import os
import zipfile
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent  # NeurIPS2026_final_submission/code
SUBMISSION_ROOT = BASE.parent  # NeurIPS2026_final_submission/
OUTPUT = SUBMISSION_ROOT / "supplementary_code.zip"

# Author names to anonymize (add all author names here)
AUTHOR_NAMES = []  # Add if needed

EXCLUDE_PATTERNS = {
    "__pycache__", ".git", ".DS_Store", "*.pyc", "*.png", "*.pdf",
    "*.egg-info", "node_modules", ".ipynb_checkpoints",
    "verify_json_tex.py",  # temp script
}

INCLUDE_DIRS = [
    "scripts",
    "outputs",
]


def should_exclude(path_str):
    """Check if path should be excluded."""
    for pattern in EXCLUDE_PATTERNS:
        if pattern in path_str:
            return True
    return False


def anonymize_content(content, filepath):
    """Remove author-identifying information from file content."""
    for name in AUTHOR_NAMES:
        content = content.replace(name, "[Anonymous]")
    return content


def build_zip():
    """Build the anonymized ZIP file."""
    print(f"Building anonymized ZIP: {OUTPUT}")
    print(f"Source: {BASE}")

    file_count = 0
    total_size = 0

    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add README
        readme = BASE / "README.md"
        if readme.exists():
            zf.write(readme, "code/README.md")
            file_count += 1
            print(f"  + code/README.md")

        # Add scripts and outputs
        for include_dir in INCLUDE_DIRS:
            dir_path = BASE / include_dir
            if not dir_path.exists():
                print(f"  SKIP {include_dir}/ (not found)")
                continue

            for root, dirs, files in os.walk(dir_path):
                # Filter out excluded directories
                dirs[:] = [d for d in dirs if d not in EXCLUDE_PATTERNS]

                for fname in sorted(files):
                    fpath = Path(root) / fname

                    if should_exclude(str(fpath)):
                        continue

                    # Only include .py, .json, .md files
                    if fpath.suffix not in {".py", ".json", ".md", ".txt"}:
                        continue

                    rel_path = fpath.relative_to(BASE)
                    arc_name = f"code/{rel_path}"

                    # Anonymize Python files
                    if fpath.suffix == ".py":
                        content = fpath.read_text(encoding="utf-8", errors="replace")
                        content = anonymize_content(content, fpath)
                        zf.writestr(arc_name, content)
                    else:
                        zf.write(fpath, arc_name)

                    size_kb = fpath.stat().st_size / 1024
                    total_size += fpath.stat().st_size
                    file_count += 1

        # Add requirements.txt
        req_content = """# EthicaAI - Requirements
# All experiments run on CPU (no GPU required)
numpy>=1.21
torch>=1.9  # For QMIX/LOLA
"""
        zf.writestr("code/requirements.txt", req_content)
        file_count += 1

    zip_size_mb = OUTPUT.stat().st_size / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"  ZIP created: {OUTPUT.name}")
    print(f"  Files: {file_count}")
    print(f"  Size: {zip_size_mb:.1f} MB (limit: 100 MB)")
    print(f"  {'PASS' if zip_size_mb < 100 else 'FAIL'}: Size check")
    print(f"{'='*60}")

    return zip_size_mb < 100


if __name__ == "__main__":
    success = build_zip()
    sys.exit(0 if success else 1)
