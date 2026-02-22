"""
EthicaAI: arXiv v2 제출 패키지 준비
- NeurIPS 2026 Main Track LaTeX + Supplementary
- 74 Figures 번들링
- .tar.gz 제출 패키지 생성
"""
import os
import sys
import shutil
import tarfile
import json
from pathlib import Path

# ── 경로 설정 ──
PROJECT_ROOT = Path(__file__).parent.resolve()
SUBMISSION_DIR = PROJECT_ROOT / "submission_arxiv"
FIGURES_DIR = PROJECT_ROOT / "simulation" / "outputs" / "reproduce"
SITE_FIGURES_DIR = PROJECT_ROOT / "site" / "figures"
TEX_DIR = PROJECT_ROOT / "paper"

# arXiv 카테고리 및 메타데이터
ARXIV_METADATA = {
    "title": "Beyond Homo Economicus: Computational Verification of "
             "Amartya Sen's Meta-Ranking Theory in Multi-Agent Social Dilemmas",
    "authors": ["Yesol Heo"],
    "primary_category": "cs.AI",
    "cross_list": ["cs.MA", "cs.GT", "econ.TH"],
    "license": "CC BY 4.0",
    "version": "v2",
    "figures": 74,
    "modules": 38,
}


def collect_figures(figures_out: Path) -> tuple[int, list[str]]:
    """site/figures → submission_arxiv/figures 수집."""
    found = 0
    missing = []

    for src_dir in [SITE_FIGURES_DIR, FIGURES_DIR]:
        if not src_dir.exists():
            continue
        for png in sorted(src_dir.glob("fig*.png")):
            dest = figures_out / png.name
            if not dest.exists():
                shutil.copy2(png, dest)
                found += 1

    return found, missing


def prepare_submission():
    """arXiv v2 제출 패키지 준비."""
    print("[R4] arXiv v2 제출 패키지 준비 시작...")

    # 1. 제출 디렉토리 생성
    if SUBMISSION_DIR.exists():
        shutil.rmtree(SUBMISSION_DIR)
    SUBMISSION_DIR.mkdir(parents=True)
    figures_out = SUBMISSION_DIR / "figures"
    figures_out.mkdir()

    # 2. Figure 수집
    found, missing = collect_figures(figures_out)
    print(f"  Figure: {found}개 수집 완료")

    # 3. LaTeX 소스 복사 (paper/ 디렉토리)
    tex_count = 0
    if TEX_DIR.exists():
        for ext in ["*.tex", "*.bib", "*.sty", "*.bst"]:
            for f in TEX_DIR.glob(ext):
                shutil.copy2(f, SUBMISSION_DIR)
                tex_count += 1
        print(f"  LaTeX: {tex_count}개 파일 복사")
    else:
        print("  ⚠ paper/ 디렉토리 없음")

    # 4. Abstract 파일 생성
    abstract = (
        "Can artificial agents develop genuine moral commitment beyond self-interest? "
        "We formalize Amartya Sen's Meta-Ranking theory—preferences over preferences "
        "implementing moral commitment—within a Multi-Agent Reinforcement Learning "
        "(MARL) framework. Our dynamic commitment mechanism λ_t, conditioned on "
        "resource availability and Social Value Orientation (SVO), is tested across "
        "8 environments with up to 1,000 agents.\n\n"
        "Three key findings: (1) Dynamic meta-ranking significantly enhances "
        "collective welfare (p<0.001 via LMM), while static SVO injection fails. "
        "(2) Non-significant cooperation rates mask emergent role specialization. "
        "(3) Situational Commitment—conditional altruism with survival instincts—is "
        "the only ESS, outperforming Utilitarian, Deontological, Virtue, and Selfish.\n\n"
        "We further demonstrate scale invariance (SII≈1.0 from 20 to 1,000 agents), "
        "Byzantine robustness (cooperation at 50% adversarial population), and "
        "policy implications for AI regulation and carbon taxation. "
        "38 reproduction modules generate all 74 figures."
    )

    with open(SUBMISSION_DIR / "abstract.txt", 'w', encoding='utf-8') as f:
        f.write(abstract)

    # 5. 메타데이터 저장
    with open(SUBMISSION_DIR / "arxiv_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(ARXIV_METADATA, f, indent=2, ensure_ascii=False)

    # 6. README
    readme = (
        "# EthicaAI — arXiv v2 Submission Package\n\n"
        f"- Primary: {ARXIV_METADATA['primary_category']}\n"
        f"- Cross-list: {', '.join(ARXIV_METADATA['cross_list'])}\n"
        f"- License: {ARXIV_METADATA['license']}\n"
        f"- Figures: {found}\n"
        f"- Modules: {ARXIV_METADATA['modules']}\n\n"
        "## Build\n"
        "```\npdflatex neurips2026_main.tex\n"
        "bibtex neurips2026_main\n"
        "pdflatex neurips2026_main.tex\n"
        "pdflatex neurips2026_main.tex\n```\n\n"
        "## Supplementary\n"
        "```\npdflatex supplementary.tex\n```\n"
    )
    with open(SUBMISSION_DIR / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme)

    # 7. .tar.gz 패키지 생성
    tar_path = PROJECT_ROOT / "ethicaai_arxiv_v2.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in SUBMISSION_DIR.rglob("*"):
            if f.is_file():
                arcname = f"ethicaai/{f.relative_to(SUBMISSION_DIR)}"
                tar.add(f, arcname=arcname)

    tar_size_mb = os.path.getsize(tar_path) / (1024 * 1024)

    # 8. 검증 요약
    print(f"\n{'='*60}")
    print("  R4 arXiv v2 SUBMISSION READINESS")
    print(f"{'='*60}")
    print(f"  Title: {ARXIV_METADATA['title'][:60]}...")
    print(f"  Authors: {', '.join(ARXIV_METADATA['authors'])}")
    print(f"  Categories: {ARXIV_METADATA['primary_category']} "
          f"[{', '.join(ARXIV_METADATA['cross_list'])}]")
    print(f"  Figures: {found}")
    print(f"  LaTeX: {tex_count} files")
    print(f"  Package: {tar_size_mb:.1f} MB")
    print(f"  Output: {tar_path}")

    if found >= 50:
        print("\n  ✅ arXiv v2 제출 준비 완료!")
    else:
        print(f"\n  ⚠ Figure {found}개 — 확인 필요")

    return str(tar_path)


if __name__ == "__main__":
    prepare_submission()
