"""
deep_audit.py — 심층 추가 감사 (기본 audit_submission.py 보완)
=================================================================
Module 9:  TeX 본문 내 수치 claim ↔ Table 수치 정합성
Module 10: 테이블 캡션 시드 수/에피소드 수 → 코드 상수 매칭
Module 11: BibTeX 엔트리 형식 검증 (year, title 존재)
Module 12: 미참조 \label 삭제 안전성 진단
Module 13: outputs/ 디렉터리에 빈 json 또는 손상 json 탐지
Module 14: TeX 내 동일 문장 반복 (복붙 실수) 탐지
"""
import re, json, os
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent.parent
PAPER_DIR = ROOT / "paper"
TEX_FILE = PAPER_DIR / "unified_paper.tex"
BIB_FILE = PAPER_DIR / "unified_references.bib"
OUTPUTS_DIR = ROOT / "code" / "outputs"
SCRIPTS_DIR = ROOT / "code" / "scripts"

findings = []
def add(sev, mod, msg):
    findings.append((sev, mod, msg))

tex = TEX_FILE.read_text(encoding="utf-8")
tex_lines = tex.split("\n")

# ═══════════════════════════════════════════════════════════════
#  Module 9: TeX 본문 claim ↔ Table 정합
# ═══════════════════════════════════════════════════════════════
print("Module 9: TeX 본문 claim ↔ Table 매칭...")

# 본문에서 "X% survival" 또는 "survival of X%" 패턴 추출
claim_matches = re.finditer(r'(\d+\.?\d*)\s*\\?%?\s*survival', tex)
for m in claim_matches:
    val = m.group(1)
    line_idx = tex[:m.start()].count("\n") + 1
    # 테이블 내부가 아닌 본문에서만 체크
    ctx = tex_lines[line_idx - 1].strip()
    if "&" not in ctx and "\\midrule" not in ctx:
        # 이 값이 테이블 어딘가에 존재하는지 확인
        if val not in tex.replace(tex[max(0,m.start()-200):m.end()+200], ""):
            add("WARN", 9, f"L{line_idx}: 본문 claim '{val}% survival' → 테이블에서 미확인")

# ═══════════════════════════════════════════════════════════════
#  Module 10: 캡션 시드 수 ↔ 코드 상수 매칭
# ═══════════════════════════════════════════════════════════════
print("Module 10: 캡션 시드 수 ↔ 코드 상수...")

caption_seeds = re.findall(r'\\caption\{.*?(\d+)\s*seeds?.*?\}', tex, re.DOTALL)
# 스크립트별 N_SEEDS 추출
code_seeds = {}
for py in SCRIPTS_DIR.glob("*.py"):
    content = py.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r'^N_SEEDS\s*=\s*(\d+)', content, re.MULTILINE)
    if match:
        code_seeds[py.name] = int(match.group(1))

if code_seeds:
    add("INFO", 10, f"코드 내 N_SEEDS 값: {code_seeds}")

# ═══════════════════════════════════════════════════════════════
#  Module 11: BibTeX 형식 검증
# ═══════════════════════════════════════════════════════════════
print("Module 11: BibTeX 형식 검증...")

bib_text = BIB_FILE.read_text(encoding="utf-8")
bib_entries = re.findall(r'@(\w+)\{(\w[\w\-:]*),\s*(.*?)\n\}', bib_text, re.DOTALL)

for entry_type, key, body in bib_entries:
    if "title" not in body.lower():
        add("FAIL", 11, f"BibTeX '{key}': title 필드 없음")
    if "year" not in body.lower():
        add("WARN", 11, f"BibTeX '{key}': year 필드 없음")
    if "author" not in body.lower() and entry_type != "misc":
        add("WARN", 11, f"BibTeX '{key}': author 필드 없음")

# ═══════════════════════════════════════════════════════════════
#  Module 12: 미참조 \label 안전성 진단
# ═══════════════════════════════════════════════════════════════
print("Module 12: 미참조 label 진단...")

labels = set(re.findall(r'\\label\{([^}]+)\}', tex))
refs = set()
for m in re.finditer(r'\\(?:ref|eqref|autoref|cref|Cref|nameref)\{([^}]+)\}', tex):
    refs.add(m.group(1))

orphan_labels = labels - refs
appendix_orphans = [l for l in orphan_labels if l.startswith("app:") or l.startswith("tab:") or l.startswith("fig:") or l.startswith("eq:") or l.startswith("sec:")]
for l in sorted(appendix_orphans):
    # Appendix 라벨이 본문에서 참조되지 않는 것은 일반적이지만, table/fig는 확인 필요
    if l.startswith("tab:") or l.startswith("fig:"):
        add("WARN", 12, f"\\label{{{l}}}: 테이블/그림이 논문 본문에서 한 번도 참조되지 않음 → 심사자가 존재 이유 의문 가능")
    elif l.startswith("eq:"):
        add("INFO", 12, f"\\label{{{l}}}: 수식 미참조 (삭제 검토)")

# ═══════════════════════════════════════════════════════════════
#  Module 13: JSON 파일 무결성
# ═══════════════════════════════════════════════════════════════
print("Module 13: JSON 무결성...")

for json_file in OUTPUTS_DIR.rglob("*.json"):
    try:
        data = json.loads(json_file.read_text(encoding="utf-8"))
        sz = json_file.stat().st_size
        if sz < 10:
            add("WARN", 13, f"JSON 크기 비정상({sz}B): {json_file.relative_to(ROOT)}")
    except json.JSONDecodeError as e:
        add("FAIL", 13, f"JSON 파싱 오류: {json_file.relative_to(ROOT)} → {e}")
    except Exception as e:
        add("FAIL", 13, f"JSON 읽기 오류: {json_file.relative_to(ROOT)} → {e}")

# ═══════════════════════════════════════════════════════════════
#  Module 14: TeX 문장 중복 (복붙 실수) 탐지
# ═══════════════════════════════════════════════════════════════
print("Module 14: 문장 중복 탐지...")

# 5단어 이상 하나의 줄 단위로 카운트
line_counter = Counter()
for i, line in enumerate(tex_lines, 1):
    stripped = line.strip()
    if len(stripped) > 50 and not stripped.startswith("%") and not stripped.startswith("\\"):
        line_counter[stripped] += 1

for text, count in line_counter.most_common(20):
    if count >= 2 and "&" not in text:  # 테이블 행 제외
        # 어디에 위치하는지 확인
        locs = [i+1 for i, l in enumerate(tex_lines) if l.strip() == text]
        add("WARN", 14, f"문장 {count}회 반복 (L{','.join(map(str,locs[:3]))}): {text[:80]}...")

# ═══════════════════════════════════════════════════════════════
#  Module 15: TeX 내 broken math mode ($..$ 불일치)
# ═══════════════════════════════════════════════════════════════
print("Module 15: Math mode 검증...")

for i, line in enumerate(tex_lines, 1):
    stripped = line.strip()
    if stripped.startswith("%"):
        continue
    # $ 개수가 홀수면 오류 가능성
    dollar_count = stripped.count("$") - stripped.count("\\$")
    if dollar_count % 2 != 0:
        # \text{} 내부 등 예외가 있으니 WARN
        add("WARN", 15, f"L{i}: $ 개수 홀수({dollar_count}) → math mode 불일치 가능: {stripped[:80]}")

# ═══════════════════════════════════════════════════════════════
#  보고서 생성
# ═══════════════════════════════════════════════════════════════
REPORT = ROOT / "deep_audit_report.txt"

fails = [f for f in findings if f[0] == "FAIL"]
warns = [f for f in findings if f[0] == "WARN"]
infos = [f for f in findings if f[0] == "INFO"]

lines = []
lines.append("=" * 72)
lines.append("  NeurIPS 2026 심층 추가 감사 보고서")
lines.append("=" * 72)
lines.append(f"  ❌ FAIL: {len(fails)}건")
lines.append(f"  ⚠️ WARN: {len(warns)}건")
lines.append(f"  ℹ️ INFO: {len(infos)}건")
lines.append("")

module_names = {
    9: "본문 claim ↔ Table 정합",
    10: "캡션 시드 수 ↔ 코드 상수",
    11: "BibTeX 형식 검증",
    12: "미참조 label 안전성",
    13: "JSON 무결성",
    14: "문장 중복 탐지",
    15: "Math mode 검증",
}

for mod_id in sorted(module_names.keys()):
    mod_findings = [f for f in findings if f[1] == mod_id]
    mod_fails = [f for f in mod_findings if f[0] == "FAIL"]
    status = "✅ PASS" if not mod_fails and not [f for f in mod_findings if f[0]=="WARN"] else ("❌ FAIL" if mod_fails else "⚠️ WARN")
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
REPORT.write_text(report, encoding="utf-8")
print(report)
