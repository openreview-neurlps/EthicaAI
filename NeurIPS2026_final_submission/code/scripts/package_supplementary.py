#!/usr/bin/env python3
"""
package_supplementary.py — NeurIPS Supplementary ZIP Builder
=============================================================

Builds an anonymous, self-contained supplementary ZIP for NeurIPS submission.
- Excludes .git, __pycache__, .pyc, personal paths, audit/output files
- Includes code/, paper/ (source only), README, and Dockerfile
- Anonymizes author paths in all text files
- Validates size < 100 MB

Usage:
    python package_supplementary.py [--output PATH] [--dry-run]
"""

import os
import sys
import zipfile
import re
from pathlib import Path

# === Configuration ===
# scripts/ -> code/ -> NeurIPS2026_final_submission/
SUBMISSION_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = SUBMISSION_ROOT / "supplementary_code_v2.zip"
MAX_SIZE_MB = 100

# Directories/files to EXCLUDE (relative to SUBMISSION_ROOT)
EXCLUDE_PATTERNS = [
    ".git",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.egg-info",
    ".DS_Store",
    "Thumbs.db",
    # Build artifacts
    "*.aux", "*.log", "*.out", "*.toc", "*.bbl", "*.blg",
    "*.synctex.gz", "*.fdb_latexmk", "*.fls",
    # Previous ZIPs
    "submission_anonymous.zip",
    "submission_source.zip",
    "supplementary_code.zip",
    "supplementary_code_v2.zip",
    # Audit / temp output
    "audit_*.txt",
    "deep_audit_report.txt",
    "outputs/",  # Raw outputs are large; include only JSON summaries
]

# Files to INCLUDE from outputs/ (summary JSON only)
INCLUDE_FROM_OUTPUTS = [
    "outputs/*/results.json",
    "outputs/*/*.json",
]

# Paths to anonymize (generic patterns — no hardcoded author info)
ANON_REPLACEMENTS = [
    (r"[A-Za-z]:\\[^\\]*\\PAPER\\EthicaAI\\?", ""),
    (r"[A-Za-z]:\\Users\\[^\\]+\\", "~/"),
    (r"/home/[^/]+/", "~/"),
    # GitHub org/user → anonymous (catch any non-anonymous org)
    (r"github\.com/[^/]+/EthicaAI", "github.com/anonymous/EthicaAI"),
]


def should_exclude(rel_path: str) -> bool:
    """Check if a relative path should be excluded."""
    parts = Path(rel_path).parts
    for pattern in EXCLUDE_PATTERNS:
        if pattern.endswith("/"):
            # Directory pattern
            dir_name = pattern.rstrip("/")
            if dir_name in parts:
                return True
        elif "*" in pattern:
            # Glob pattern
            import fnmatch
            if fnmatch.fnmatch(Path(rel_path).name, pattern):
                return True
        else:
            if pattern in parts or Path(rel_path).name == pattern:
                return True
    return False


def should_include_output(rel_path: str) -> bool:
    """Check if an output file should be included (JSON summaries only)."""
    p = Path(rel_path)
    if not str(p).startswith("outputs"):
        return False
    return p.suffix == ".json"


def anonymize_content(content: str) -> str:
    """Replace personal paths with anonymous placeholders."""
    for pattern, replacement in ANON_REPLACEMENTS:
        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
    return content


def is_text_file(path: Path) -> bool:
    """Check if a file is likely text (for anonymization)."""
    text_extensions = {
        ".py", ".tex", ".bib", ".md", ".txt", ".json", ".yaml", ".yml",
        ".toml", ".cfg", ".ini", ".sh", ".bat", ".ps1", ".csv", ".rules",
        ".gitignore", ".dockerignore",
    }
    return path.suffix.lower() in text_extensions or path.name in {
        "Dockerfile", "Makefile", "LICENSE", "requirements.txt",
    }


def build_zip(output_path: Path, dry_run: bool = False):
    """Build the supplementary ZIP."""
    root = SUBMISSION_ROOT
    if not root.exists():
        print(f"ERROR: Submission root not found: {root}")
        sys.exit(1)

    included_files = []
    excluded_files = []

    # Walk the submission directory
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip .git directory entirely
        dirnames[:] = [d for d in dirnames if d != ".git"]

        for fname in filenames:
            full_path = Path(dirpath) / fname
            rel_path = full_path.relative_to(root)
            rel_str = str(rel_path).replace("\\", "/")

            if should_exclude(rel_str):
                # But check if it's an output JSON we want
                if should_include_output(rel_str):
                    included_files.append((full_path, rel_str))
                else:
                    excluded_files.append(rel_str)
            else:
                included_files.append((full_path, rel_str))

    print(f"Submission root: {root}")
    print(f"Files to include: {len(included_files)}")
    print(f"Files excluded:   {len(excluded_files)}")

    if dry_run:
        print("\n--- DRY RUN: Files that would be included ---")
        for _, rel in sorted(included_files, key=lambda x: x[1]):
            sz = os.path.getsize(_)
            print(f"  {rel} ({sz:,} bytes)")
        total = sum(os.path.getsize(f) for f, _ in included_files)
        print(f"\nTotal uncompressed: {total / 1024 / 1024:.1f} MB")
        return

    # Build ZIP
    print(f"\nBuilding: {output_path}")
    anon_count = 0

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for full_path, rel_path in sorted(included_files, key=lambda x: x[1]):
            if is_text_file(full_path):
                try:
                    content = full_path.read_text(encoding="utf-8", errors="replace")
                    anon_content = anonymize_content(content)
                    if anon_content != content:
                        anon_count += 1
                    zf.writestr(rel_path, anon_content)
                except Exception as e:
                    print(f"  WARN: Could not read {rel_path}: {e}")
                    zf.write(full_path, rel_path)
            else:
                zf.write(full_path, rel_path)

    zip_size = output_path.stat().st_size
    zip_mb = zip_size / 1024 / 1024

    print(f"\n--- Supplementary ZIP Summary ---")
    print(f"  Output:      {output_path}")
    print(f"  Files:       {len(included_files)}")
    print(f"  Anonymized:  {anon_count} files")
    print(f"  Size:        {zip_mb:.2f} MB")
    print(f"  Limit:       {MAX_SIZE_MB} MB")

    if zip_mb > MAX_SIZE_MB:
        print(f"\n  WARNING: ZIP exceeds {MAX_SIZE_MB} MB limit!")
        sys.exit(1)
    else:
        print(f"  Status:      OK (under limit)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build NeurIPS supplementary ZIP")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output ZIP path")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files without creating ZIP")
    args = parser.parse_args()

    build_zip(Path(args.output), dry_run=args.dry_run)
