"""
deep_audit.py ???¬ì¸µ ì¶ê? ê°ì¬ (ê¸°ë³¸ audit_submission.py ë³´ì)
=================================================================
Module 9:  TeX ë³¸ë¬¸ ???ì¹ claim ??Table ?ì¹ ?í©??
Module 10: ?ì´ë¸?ìº¡ì ?ë ???í¼?ë ????ì½ë ?ì ë§¤ì¹­
Module 11: BibTeX ?í¸ë¦??ì ê²ì¦?(year, title ì¡´ì¬)
Module 12: ë¯¸ì°¸ì¡?\label ??  ?ì ??ì§ë¨
Module 13: outputs/ ?ë ?°ë¦¬??ë¹?json ?ë ?ì json ?ì?
Module 14: TeX ???ì¼ ë¬¸ì¥ ë°ë³µ (ë³µë¶ ?¤ì) ?ì?
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

# ?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â??
#  Module 9: TeX ë³¸ë¬¸ claim ??Table ?í©
# ?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â??
print("Module 9: TeX ë³¸ë¬¸ claim ??Table ë§¤ì¹­...")

# ë³¸ë¬¸?ì "X% survival" ?ë "survival of X%" ?¨í´ ì¶ì¶
claim_matches = re.finditer(r'(\d+\.?\d*)\s*\\?%?\s*survival', tex)
for m in claim_matches:
    val = m.group(1)
    line_idx = tex[:m.start()].count("\n") + 1
    # ?ì´ë¸??´ë?ê° ?ë ë³¸ë¬¸?ìë§?ì²´í¬
    ctx = tex_lines[line_idx - 1].strip()
    if "&" not in ctx and "\\midrule" not in ctx:
        # ??ê°ì´ ?ì´ë¸??´ëê°??ì¡´ì¬?ëì§ ?ì¸
        if val not in tex.replace(tex[max(0,m.start()-200):m.end()+200], ""):
            add("WARN", 9, f"L{line_idx}: ë³¸ë¬¸ claim '{val}% survival' ???ì´ë¸ì??ë¯¸í??)

# ?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â??
#  Module 10: ìº¡ì ?ë ????ì½ë ?ì ë§¤ì¹­
# ?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â??
print("Module 10: ìº¡ì ?ë ????ì½ë ?ì...")

caption_seeds = re.findall(r'\\caption\{.*?(\d+)\s*seeds?.*?\}', tex, re.DOTALL)
# ?¤í¬ë¦½í¸ë³?N_SEEDS ì¶ì¶
code_seeds = {}
for py in SCRIPTS_DIR.glob("*.py"):
    content = py.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r'^N_SEEDS\s*=\s*(\d+)', content, re.MULTILINE)
    if match:
        code_seeds[py.name] = int(match.group(1))

if code_seeds:
    add("INFO", 10, f"ì½ë ??N_SEEDS ê°? {code_seeds}")

# ?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â??
#  Module 11: BibTeX ?ì ê²ì¦?
# ?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â??
print("Module 11: BibTeX ?ì ê²ì¦?..")

bib_text = BIB_FILE.read_text(encoding="utf-8")
bib_entries = re.findall(r'@(\w+)\{(\w[\w\-:]*),\s*(.*?)\n\}', bib_text, re.DOTALL)

for entry_type, key, body in bib_entries:
    if "title" not in body.lower():
        add("FAIL", 11, f"BibTeX '{key}': title ?ë ?ì")
    if "year" not in body.lower():
        add("WARN", 11, f"BibTeX '{key}': year ?ë ?ì")
    if "author" not in body.lower() and entry_type != "misc":
        add("WARN", 11, f"BibTeX '{key}': author ?ë ?ì")

# ?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â??
#  Module 12: ë¯¸ì°¸ì¡?\label ?ì ??ì§ë¨
# ?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â??
print("Module 12: ë¯¸ì°¸ì¡?label ì§ë¨...")

labels = set(re.findall(r'\\label\{([^}]+)\}', tex))
refs = set()
for m in re.finditer(r'\\(?:ref|eqref|autoref|cref|Cref|nameref)\{([^}]+)\}', tex):
    refs.add(m.group(1))

orphan_labels = labels - refs
appendix_orphans = [l for l in orphan_labels if l.startswith("app:") or l.startswith("tab:") or l.startswith("fig:") or l.startswith("eq:") or l.startswith("sec:")]
for l in sorted(appendix_orphans):
    # Appendix ?¼ë²¨??ë³¸ë¬¸?ì ì°¸ì¡°?ì? ?ë ê²ì? ?¼ë°?ì´ì§ë§? table/fig???ì¸ ?ì
    if l.startswith("tab:") or l.startswith("fig:"):
        add("WARN", 12, f"\\label{{{l}}}: ?ì´ë¸?ê·¸ë¦¼???¼ë¬¸ ë³¸ë¬¸?ì ??ë²ë ì°¸ì¡°?ì? ?ì ???¬ì¬?ê? ì¡´ì¬ ?´ì  ?ë¬¸ ê°??)
    elif l.startswith("eq:"):
        add("INFO", 12, f"\\label{{{l}}}: ?ì ë¯¸ì°¸ì¡?(??  ê²??")

# ?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â??
#  Module 13: JSON ?ì¼ ë¬´ê²°??
# ?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â??
print("Module 13: JSON ë¬´ê²°??..")

for json_file in OUTPUTS_DIR.rglob("*.json"):
    try:
        data = json.loads(json_file.read_text(encoding="utf-8"))
        sz = json_file.stat().st_size
        if sz < 10:
            add("WARN", 13, f"JSON ?¬ê¸° ë¹ì ??{sz}B): {json_file.relative_to(ROOT)}")
    except json.JSONDecodeError as e:
        add("FAIL", 13, f"JSON ?ì± ?¤ë¥: {json_file.relative_to(ROOT)} ??{e}")
    except Exception as e:
        add("FAIL", 13, f"JSON ?½ê¸° ?¤ë¥: {json_file.relative_to(ROOT)} ??{e}")

# ?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â??
#  Module 14: TeX ë¬¸ì¥ ì¤ë³µ (ë³µë¶ ?¤ì) ?ì?
# ?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â??
print("Module 14: ë¬¸ì¥ ì¤ë³µ ?ì?...")

# 5?¨ì´ ?´ì ?ë??ì¤??¨ìë¡?ì¹´ì´??
line_counter = Counter()
for i, line in enumerate(tex_lines, 1):
    stripped = line.strip()
    if len(stripped) > 50 and not stripped.startswith("%") and not stripped.startswith("\\"):
        line_counter[stripped] += 1

for text, count in line_counter.most_common(20):
    if count >= 2 and "&" not in text:  # ?ì´ë¸????ì¸
        # ?´ë???ì¹?ëì§ ?ì¸
        locs = [i+1 for i, l in enumerate(tex_lines) if l.strip() == text]
        add("WARN", 14, f"ë¬¸ì¥ {count}??ë°ë³µ (L{','.join(map(str,locs[:3]))}): {text[:80]}...")

# ?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â??
#  Module 15: TeX ??broken math mode ($..$ ë¶ì¼ì¹?
# ?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â??
print("Module 15: Math mode ê²ì¦?..")

for i, line in enumerate(tex_lines, 1):
    stripped = line.strip()
    if stripped.startswith("%"):
        continue
    # $ ê°ìê° ??ë©´ ?¤ë¥ ê°?¥ì±
    dollar_count = stripped.count("$") - stripped.count("\\$")
    if dollar_count % 2 != 0:
        # \text{} ?´ë? ???ì¸ê° ?ì¼??WARN
        add("WARN", 15, f"L{i}: $ ê°ì ???{dollar_count}) ??math mode ë¶ì¼ì¹?ê°?? {stripped[:80]}")

# ?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â??
#  ë³´ê³ ???ì±
# ?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â?â??
REPORT = ROOT / "deep_audit_report.txt"

fails = [f for f in findings if f[0] == "FAIL"]
warns = [f for f in findings if f[0] == "WARN"]
infos = [f for f in findings if f[0] == "INFO"]

lines = []
lines.append("=" * 72)
lines.append("  NeurIPS 2026 ?¬ì¸µ ì¶ê? ê°ì¬ ë³´ê³ ??)
lines.append("=" * 72)
lines.append(f"  ??FAIL: {len(fails)}ê±?)
lines.append(f"  ? ï¸ WARN: {len(warns)}ê±?)
lines.append(f"  ?¹ï¸ INFO: {len(infos)}ê±?)
lines.append("")

module_names = {
    9: "ë³¸ë¬¸ claim ??Table ?í©",
    10: "ìº¡ì ?ë ????ì½ë ?ì",
    11: "BibTeX ?ì ê²ì¦?,
    12: "ë¯¸ì°¸ì¡?label ?ì ??,
    13: "JSON ë¬´ê²°??,
    14: "ë¬¸ì¥ ì¤ë³µ ?ì?",
    15: "Math mode ê²ì¦?,
}

for mod_id in sorted(module_names.keys()):
    mod_findings = [f for f in findings if f[1] == mod_id]
    mod_fails = [f for f in mod_findings if f[0] == "FAIL"]
    status = "??PASS" if not mod_fails and not [f for f in mod_findings if f[0]=="WARN"] else ("??FAIL" if mod_fails else "? ï¸ WARN")
    lines.append(f"??? Module {mod_id}: {module_names[mod_id]} [{status}] ???")
    if not mod_findings:
        lines.append("  (?´ì ?ì)")
    for sev, _, msg in mod_findings:
        icon = {"FAIL": "??, "WARN": "? ï¸", "INFO": "?¹ï¸"}[sev]
        lines.append(f"  {icon} {msg}")
    lines.append("")

lines.append("=" * 72)
verdict = "PASS ?? if not fails else f"FAIL ??({len(fails)}ê±??ì  ?ì)"
lines.append(f"  ìµì¢ ?ì : {verdict}")
lines.append("=" * 72)

report = "\n".join(lines)
REPORT.write_text(report, encoding="utf-8")
print(report)
