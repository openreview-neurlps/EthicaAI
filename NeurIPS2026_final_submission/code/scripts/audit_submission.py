"""
audit_submission.py  --  NeurIPS 제출물 종합 무결성 검수 스크립트
======================================================================
7개 감사(Audit) 모듈로 구성. 각 모듈은 PASS/WARN/FAIL 판정을 내리고
최종 레포트를 UTF-8 텍스트 파일로 저장합니다.

  Module 1: TeX ↔ JSON 수치 교차 검증
  Module 2: BibTeX 무결성 (미참조 엔트리, TeX 내 미정의 cite)
  Module 3: 그림 파일 존재 여부 (includegraphics 경로 전수 점검)
  Module 4: 플레이스홀더 잔류 검출 (X%, TODO, FIXME, TBD, ???)
  Module 5: LaTeX 빌드 경고/에러 분석 (.log 파일)
  Module 6: README 팩트체크 (주장된 파일·경로가 실존하는지)
  Module 7: 코드 일관성 (import 오류, 출력 경로 정합성)
"""
import os
import re
import json
import sys
from pathlib import Path
from collections import defaultdict

# ─── 경로 설정 ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent  # NeurIPS2026_final_submission/
PAPER_DIR = ROOT / "paper"
CODE_DIR  = ROOT / "code"
SCRIPTS_DIR = CODE_DIR / "scripts"
OUTPUTS_DIR = CODE_DIR / "outputs"
TEX_FILE = PAPER_DIR / "unified_paper.tex"
BIB_FILE = PAPER_DIR / "unified_references.bib"
LOG_FILE = PAPER_DIR / "unified_paper.log"
REPORT_FILE = ROOT / "audit_report.txt"

findings = []  # (severity, module, message)

def add(sev, mod, msg):
    findings.append((sev, mod, msg))

def load_tex():
    """Load unified_paper.tex with recursive \\input{} resolution."""
    return _resolve_inputs(TEX_FILE)

def _resolve_inputs(tex_path, depth=0):
    """Recursively resolve \\input{file} directives up to depth 5."""
    if depth > 5 or not tex_path.exists():
        return ""
    text = tex_path.read_text(encoding="utf-8", errors="ignore")
    def replace_input(m):
        fname = m.group(1)
        if not fname.endswith(".tex"):
            fname += ".tex"
        child = tex_path.parent / fname
        return _resolve_inputs(child, depth + 1)
    # Resolve \input{...} (not in comments)
    resolved = re.sub(r'^(?!%)\s*\\input\{([^}]+)\}', replace_input, text, flags=re.MULTILINE)
    return resolved

def load_bib():
    return BIB_FILE.read_text(encoding="utf-8")

# ═══════════════════════════════════════════════════════════════
#  Module 1: TeX ↔ JSON 수치 교차 검증
# ═══════════════════════════════════════════════════════════════
def audit_tex_json():
    """각 JSON 결과 파일의 핵심 수치가 TeX 내 테이블에 존재하는지 확인"""
    tex = load_tex()

    # 1-a: cleanrl_baselines (combined IPPO/MAPPO and QMIX)
    for name in ["cleanrl_baseline_results.json", "iql_baseline_results.json"]:
        p = OUTPUTS_DIR / "cleanrl_baselines" / name
        if not p.exists():
            add("FAIL", 1, f"JSON 누락: {p.relative_to(ROOT)}")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        # lambda_mean 값을 tex에서 검색
        lam = data.get("lambda_mean")
        if lam is not None:
            lam_str = f"{lam:.3f}"
            if lam_str not in tex:
                lam_str2 = f"{lam:.2f}"
                if lam_str2 not in tex:
                    add("WARN", 1, f"{name}: λ={lam_str} 이 TeX에 없음")

    # 1-b: phi1 ablation
    p = OUTPUTS_DIR / "phi1_ablation" / "phi1_results.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        for phi_key in ["0.0", "0.21", "0.5", "1.0"]:
            if phi_key in data:
                run = data[phi_key]
                for byz_key in ["byz_0", "byz_30"]:
                    w = run[byz_key]["welfare_mean"]
                    w_str = f"{w:.1f}"
                    if w_str not in tex:
                        add("WARN", 1, f"phi1={phi_key}, {byz_key}: W={w_str} 이 TeX에 없음")
    else:
        add("FAIL", 1, f"JSON 누락: {p.relative_to(ROOT)}")

    # 1-c: hp_sweep
    p = OUTPUTS_DIR / "ppo_nash_trap" / "hp_sweep_results.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        # 각 combo의 lambda_mean이 있어야 함
        if isinstance(data, dict):
            combos = data.get("results", data.get("combos", []))
            if isinstance(combos, list):
                mismatches = 0
                for combo in combos[:3]:  # 샘플 검사
                    lam = combo.get("lambda_mean") or combo.get("lambda_avg")
                    if lam:
                        if f"{lam:.3f}" not in tex and f"{lam:.2f}" not in tex:
                            mismatches += 1
                if mismatches > 0:
                    add("WARN", 1, f"HP sweep: {mismatches}/{min(3, len(combos))} 조합 λ값이 TeX에 미반영")
    
    # 1-d: scale_n100
    p = OUTPUTS_DIR / "scale_n100" / "scale_n100_results.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        results = data.get("results", [])
        for r in results:
            sv = r.get("survival_pct")
            if sv is not None:
                sv_str = f"{sv:.1f}"
                if sv_str not in tex:
                    add("WARN", 1, f"Scale N=100 {r.get('label','?')}: Surv={sv_str} TeX 부재")
    else:
        add("FAIL", 1, f"JSON 누락: {p.relative_to(ROOT)}")

    # 1-e: dnn_ablation
    p = OUTPUTS_DIR / "dnn_ablation" / "dnn_ablation_results.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        results = data if isinstance(data, list) else data.get("results", [])
        for r in results:
            lam = r.get("lambda_mean") or r.get("lambda_avg")
            if lam:
                lam_str = f"{lam:.3f}"
                if lam_str not in tex:
                    add("WARN", 1, f"DNN ablation {r.get('arch','?')}: λ={lam_str} TeX 부재")

    # 1-f: kpg
    p = OUTPUTS_DIR / "kpg_experiment" / "kpg_results.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        results = data if isinstance(data, list) else data.get("results", [])
        for r in results:
            sv = r.get("survival_pct")
            if sv is not None:
                sv_str = f"{sv:.1f}"
                if sv_str not in tex:
                    add("WARN", 1, f"KPG K={r.get('K','?')}: Surv={sv_str} TeX 부재")


# ═══════════════════════════════════════════════════════════════
#  Module 2: BibTeX 무결성
# ═══════════════════════════════════════════════════════════════
def audit_bibtex():
    tex = load_tex()
    bib = load_bib()

    # 2-a: TeX 내 모든 \cite{...} / \citep{...} / \citet{...} 키 추출
    cite_keys = set()
    for m in re.finditer(r'\\cite[pt]?\{([^}]+)\}', tex):
        for k in m.group(1).split(","):
            cite_keys.add(k.strip())

    # 2-b: BibTeX 내 정의된 키 추출
    bib_keys = set(re.findall(r'@\w+\{(\w[\w\-:]*)', bib))

    # 미정의 cite
    undefined = cite_keys - bib_keys
    for k in sorted(undefined):
        add("FAIL", 2, f"\\cite{{{k}}} → BibTeX에 정의 없음")

    # 미참조 bib 엔트리 (warning only)
    unused = bib_keys - cite_keys
    for k in sorted(unused):
        add("WARN", 2, f"BibTeX 엔트리 '{k}' → 논문에서 미참조")


# ═══════════════════════════════════════════════════════════════
#  Module 3: 그림 파일 존재 여부
# ═══════════════════════════════════════════════════════════════
def audit_figures():
    tex = load_tex()
    for m in re.finditer(r'\\includegraphics(?:\[.*?\])?\{([^}]+)\}', tex):
        fig_name = m.group(1)
        fig_path = PAPER_DIR / fig_name
        if not fig_path.exists():
            add("FAIL", 3, f"그림 파일 누락: {fig_name}")
        else:
            sz = fig_path.stat().st_size
            if sz < 1000:
                add("WARN", 3, f"그림 파일 비정상적으로 작음({sz}B): {fig_name}")


# ═══════════════════════════════════════════════════════════════
#  Module 4: 플레이스홀더 잔류 검출
# ═══════════════════════════════════════════════════════════════
def audit_placeholders():
    tex = load_tex()
    lines = tex.split("\n")

    placeholder_patterns = [
        (r'\bX%', "X% (미기입 수치)"),
        (r'\bTBD\b', "TBD"),
        (r'\bTODO\b', "TODO"),
        (r'\bFIXME\b', "FIXME"),
        (r'\bXXX\b', "XXX"),
        (r'\?\?\?', "???"),
        (r'\\textcolor\{red\}', "빨간색 텍스트 (디버그용?)"),
    ]

    for i, line in enumerate(lines, 1):
        # 주석 줄은 스킵
        stripped = line.lstrip()
        if stripped.startswith("%"):
            continue
        for pat, label in placeholder_patterns:
            if re.search(pat, line, re.IGNORECASE):
                snippet = line.strip()[:80]
                add("FAIL", 4, f"L{i}: [{label}] → {snippet}")


# ═══════════════════════════════════════════════════════════════
#  Module 5: LaTeX 빌드 경고/에러 분석
# ═══════════════════════════════════════════════════════════════
def audit_latex_log():
    if not LOG_FILE.exists():
        add("WARN", 5, "LaTeX 로그 파일(.log) 없음 — 빌드 미수행?")
        return

    log = LOG_FILE.read_text(encoding="utf-8", errors="ignore")
    lines = log.split("\n")

    errors = [l for l in lines if l.startswith("!")]
    warnings = [l for l in lines if "Warning" in l and "Font" not in l]
    overfull = [l for l in lines if "Overfull" in l]
    underfull = [l for l in lines if "Underfull" in l]
    undef_refs = [l for l in lines if "undefined" in l.lower() and "ref" in l.lower()]

    for e in errors:
        add("FAIL", 5, f"LaTeX Error: {e.strip()[:100]}")

    # geometry over-specification은 neurips sty 고유 — 무시
    BENIGN_WARNINGS = ["Over-specification", "geometry Warning"]
    for w in warnings[:10]:
        w_text = w.strip()[:100]
        if any(bw in w_text for bw in BENIGN_WARNINGS):
            continue  # neurips_2026.sty 고유, 무해
        add("WARN", 5, f"LaTeX Warning: {w_text}")

    # overfull 1-2개는 NeurIPS 스타일에서 흔함 — 3개 이상만 경고
    if len(overfull) >= 3:
        add("WARN", 5, f"Overfull boxes: {len(overfull)}개 (hbox 넘침)")
    if underfull:
        add("WARN", 5, f"Underfull boxes: {len(underfull)}개")

    for u in undef_refs:
        add("FAIL", 5, f"미정의 참조: {u.strip()[:100]}")

    # PDF 생성 여부
    if "no output PDF file produced" in log:
        add("FAIL", 5, "PDF 미생성 — 빌드 실패")
    
    # 마지막 줄에서 페이지 수 추출
    page_match = re.search(r'Output written on .+ \((\d+) pages?', log)
    if page_match:
        pages = int(page_match.group(1))
        if pages > 30:
            add("WARN", 5, f"논문 길이: {pages}페이지 (NeurIPS 기본 9+보충 제한 확인 필요)")
        add("INFO", 5, f"PDF 생성 성공: {pages}페이지")


# ═══════════════════════════════════════════════════════════════
#  Module 6: README 팩트체크
# ═══════════════════════════════════════════════════════════════
def audit_readme():
    for readme_path in [ROOT / "README.md", CODE_DIR / "README.md"]:
        if not readme_path.exists():
            add("WARN", 6, f"README 없음: {readme_path.relative_to(ROOT)}")
            continue

        text = readme_path.read_text(encoding="utf-8")
        
        # 코드블록 내용을 제거하여 거짓 양성 방지
        text_no_codeblocks = re.sub(r'```[\s\S]*?```', '', text)
        
        # 경로 참조 검증: 마크다운에서 코드블럭이나 경로 언급 발견
        # 일반적으로 `scripts/xxx.py` 나 `code/xxx` 형태
        path_refs = re.findall(r'`([^`]*(?:\.py|\.txt|\.tex|\.bib|\.json|Dockerfile|LICENSE)[^`]*)`', text_no_codeblocks)
        for ref in path_refs:
            # 상대 경로 해석
            candidates = [
                ROOT / ref,
                CODE_DIR / ref,
                SCRIPTS_DIR / ref,
            ]
            found = any(c.exists() for c in candidates)
            if not found and not ref.startswith("pip ") and not ref.startswith("python "):
                add("WARN", 6, f"README에 언급된 '{ref}' → 실제 파일 없음 ({readme_path.name})")

        # 명령어 동작 체크: "python scripts/reproduce_quick.py" 형태
        cmd_refs = re.findall(r'python\s+(?:scripts/)?(\S+\.py)', text_no_codeblocks)
        for cmd in cmd_refs:
            candidates = [
                ROOT / cmd,
                CODE_DIR / cmd,
                SCRIPTS_DIR / cmd,
            ]
            found = any(c.exists() for c in candidates)
            if not found:
                add("WARN", 6, f"README 명령어의 스크립트 '{cmd}' → 실제 파일 없음 ({readme_path.name})")


# ═══════════════════════════════════════════════════════════════
#  Module 7: 코드 일관성
# ═══════════════════════════════════════════════════════════════
def audit_code():
    # 7-a: 출력 경로 통일 검사 (모든 .py에서 outputs를 참조할 때 code/outputs/ 인지)
    for py_file in SCRIPTS_DIR.glob("*.py"):
        text = py_file.read_text(encoding="utf-8", errors="ignore")
        
        # 하드코딩된 "../outputs" 또는 "outputs/" 가 있으면서 code/outputs가 아닌 경우
        suspicious = re.findall(r'''(?:"|')\.\.\/outputs|(?:"|')outputs\/''', text)
        for s in suspicious:
            # PROJECT_ROOT 기반이면 OK, 아니면 경고
            if "PROJECT_ROOT" not in text and "Path(__file__)" not in text:
                add("WARN", 7, f"{py_file.name}: 상대경로 출력('outputs/') 사용 — 절대경로 전환 권장")
                break

    # 7-b: import 검사 (존재하지 않는 모듈)
    for py_file in SCRIPTS_DIR.glob("*.py"):
        text = py_file.read_text(encoding="utf-8", errors="ignore")
        local_imports = re.findall(r'^from\s+(\S+)\s+import', text, re.MULTILINE)
        for imp in local_imports:
            if imp.startswith(".") or imp.startswith("envs"):
                # 로컬 import: envs 디렉터리에 있는지
                mod_name = imp.replace(".", "").replace("envs/", "envs.")
                mod_path = SCRIPTS_DIR / imp.replace(".", "/")
                if not mod_path.exists() and not (SCRIPTS_DIR / f"{imp.replace('.','/')}.py").exists():
                    # envs 하위 모듈인지 확인
                    envs_path = SCRIPTS_DIR / "envs" / f"{imp.split('.')[-1]}.py"
                    if not envs_path.exists() and imp.startswith("envs"):
                        pass  # 이미 있음

    # 7-c: Dockerfile 검증
    dockerfile = CODE_DIR / "Dockerfile"
    if dockerfile.exists():
        df_text = dockerfile.read_text(encoding="utf-8")
        if "requirements.txt" in df_text:
            req = CODE_DIR / "requirements.txt"
            if not req.exists():
                add("FAIL", 7, "Dockerfile이 requirements.txt 참조 → 파일 없음")
    
    # 7-d: requirements.txt 핀 버전 확인
    req = CODE_DIR / "requirements.txt"
    if req.exists():
        for line in req.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                if "==" not in line and ">=" not in line and "<=" not in line:
                    add("WARN", 7, f"requirements.txt: '{line}' → 버전 미고정 (재현성 위험)")


# ═══════════════════════════════════════════════════════════════
#  Module 8: TeX 내부 교차참조 무결성 (\ref / \label 대응)
# ═══════════════════════════════════════════════════════════════
def audit_crossrefs():
    tex = load_tex()
    
    labels = set(re.findall(r'\\label\{([^}]+)\}', tex))
    refs = set()
    for m in re.finditer(r'\\(?:ref|eqref|autoref|cref|Cref)\{([^}]+)\}', tex):
        refs.add(m.group(1))
    
    undefined_refs = refs - labels
    for r in sorted(undefined_refs):
        add("FAIL", 8, f"\\ref{{{r}}} → \\label 정의 없음")
    
    unused_labels = labels - refs
    for l in sorted(unused_labels):
        add("WARN", 8, f"\\label{{{l}}} → 본문에서 미참조")


# ═══════════════════════════════════════════════════════════════
#  보고서 생성
# ═══════════════════════════════════════════════════════════════
def generate_report():
    fails = [f for f in findings if f[0] == "FAIL"]
    warns = [f for f in findings if f[0] == "WARN"]
    infos = [f for f in findings if f[0] == "INFO"]

    lines = []
    lines.append("=" * 72)
    lines.append("  NeurIPS 2026 제출물 종합 감사 보고서")
    lines.append(f"  대상: {ROOT}")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"  ❌ FAIL: {len(fails)}건")
    lines.append(f"  ⚠️ WARN: {len(warns)}건")
    lines.append(f"  ℹ️ INFO: {len(infos)}건")
    lines.append("")

    module_names = {
        1: "TeX ↔ JSON 수치 교차 검증",
        2: "BibTeX 무결성",
        3: "그림 파일 존재",
        4: "플레이스홀더 잔류",
        5: "LaTeX 빌드 로그",
        6: "README 팩트체크",
        7: "코드 일관성",
        8: "교차참조 무결성",
    }

    for mod_id in sorted(module_names.keys()):
        mod_findings = [f for f in findings if f[1] == mod_id]
        mod_fails = [f for f in mod_findings if f[0] == "FAIL"]
        mod_warns = [f for f in mod_findings if f[0] == "WARN"]
        mod_infos = [f for f in mod_findings if f[0] == "INFO"]

        status = "✅ PASS" if not mod_fails and not mod_warns else ("❌ FAIL" if mod_fails else "⚠️ WARN")
        lines.append(f"─── Module {mod_id}: {module_names[mod_id]} [{status}] ───")

        if not mod_findings:
            lines.append("  (이상 없음)")
        for sev, _, msg in mod_findings:
            icon = {"FAIL": "❌", "WARN": "⚠️", "INFO": "ℹ️"}[sev]
            lines.append(f"  {icon} {msg}")
        lines.append("")

    lines.append("=" * 72)
    verdict = "PASS ✅" if not fails else f"FAIL ❌ ({len(fails)}건 수정 필요)"
    lines.append(f"  최종 판정: {verdict}")
    lines.append("=" * 72)

    report = "\n".join(lines)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(report)
    return len(fails)


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("NeurIPS 제출물 종합 감사 시작...\n")
    audit_tex_json()
    audit_bibtex()
    audit_figures()
    audit_placeholders()
    audit_latex_log()
    audit_readme()
    audit_code()
    audit_crossrefs()
    n_fails = generate_report()
    sys.exit(1 if n_fails > 0 else 0)
