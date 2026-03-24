---
description: 논문 변경 후 커밋 전 필수 사전 검증 (audit + build + security + page check)
---

# Pre-Commit 사전 검증 워크플로우

논문이나 코드 변경 후 **커밋 전에 반드시** 실행합니다.
6단계 검증을 통과해야만 커밋합니다.

## 전제

- `NeurIPS2026_final_submission/` 디렉터리 존재
- Python 3.8+, pdflatex (MiKTeX)

## 실행

// turbo-all

### Gate 1. 보안 스캔

```powershell
cd d:\00.test\PAPER\EthicaAI
python -c "
import re, os, sys
secrets = ['api_key','API_KEY','sk-','token=','password','ACCESS_TOKEN','ZENODO_TOKEN','yesol@','010-']
found = []
for root, dirs, files in os.walk('NeurIPS2026_final_submission'):
    dirs[:] = [d for d in dirs if d not in ['.git','__pycache__','png_preview','outputs']]
    for f in files:
        if f.endswith(('.tex','.py','.bib','.md','.json','.txt')):
            path = os.path.join(root, f)
            try:
                content = open(path, 'r', encoding='utf-8', errors='ignore').read()
                for s in secrets:
                    if s.lower() in content.lower():
                        lines = [i+1 for i,l in enumerate(content.split(chr(10))) if s.lower() in l.lower()]
                        found.append(f'{path}:{lines} → {s}')
            except: pass
if found:
    print('🔴 SECURITY FAIL')
    for f in found: print(f'  {f}')
    sys.exit(1)
else:
    print('✅ Gate 1 PASS: 보안 스캔 통과')
"
```

### Gate 2. LaTeX 3-pass 빌드 + 참조 무결성

```powershell
cd d:\00.test\PAPER\EthicaAI\NeurIPS2026_final_submission\paper
pdflatex -interaction=nonstopmode unified_paper.tex 2>&1 | Out-Null
bibtex unified_paper 2>&1 | Out-Null
pdflatex -interaction=nonstopmode unified_paper.tex 2>&1 | Out-Null
pdflatex -interaction=nonstopmode unified_paper.tex 2>&1 | Out-Null
```

```powershell
cd d:\00.test\PAPER\EthicaAI\NeurIPS2026_final_submission\paper
python -c "
import sys
log = open('unified_paper.log','r',encoding='latin-1').read()
errors = [l for l in log.split(chr(10)) if 'undefined' in l.lower() or 'multiply' in l.lower()]
if errors:
    print('🔴 Gate 2 FAIL: 참조 오류')
    for e in errors: print(f'  {e.strip()}')
    sys.exit(1)
else:
    print('✅ Gate 2 PASS: undefined=0, multiply=0')
"
```

### Gate 3. 9페이지 제약 확인

```powershell
cd d:\00.test\PAPER\EthicaAI\NeurIPS2026_final_submission\paper
python -c "
import fitz, sys
doc = fitz.open('unified_paper.pdf')
total = len(doc)
p10_text = doc[9].get_text()[:200] if total > 9 else ''
doc.close()
has_ref_on_p10 = 'references' in p10_text.lower() or 'bibliography' in p10_text.lower()
p9_text = doc[8].get_text() if total > 8 else '' if not has_ref_on_p10 else p10_text
# Check: p9 should have Conclusion, p10 should start with References
if total < 10:
    print(f'🔴 Gate 3 FAIL: {total}p < 9p+refs')
    sys.exit(1)
elif has_ref_on_p10:
    print(f'✅ Gate 3 PASS: {total}p, p10=References 시작')
else:
    print(f'⚠️ Gate 3 WARN: p10에 References 미확인 (총 {total}p)')
"
```

### Gate 4. Audit 스크립트 (8모듈)

```powershell
cd d:\00.test\PAPER\EthicaAI
python NeurIPS2026_final_submission/code/scripts/audit_submission.py
```

결과에서 **FAIL 0건** 확인. WARN은 허용.

### Gate 5. 미커밋 변경 확인

```powershell
cd d:\00.test\PAPER\EthicaAI
git status --short
git diff --stat -- NeurIPS2026_final_submission/
```

변경 내용이 의도한 것인지 확인합니다.

### Gate 6. 커밋 + 듀얼 Push

Gate 1~5 모두 통과 후에만 실행합니다.

```powershell
cd d:\00.test\PAPER\EthicaAI
git add -A; git status
```

```powershell
cd d:\00.test\PAPER\EthicaAI
git commit --no-verify -m "category: description"
```

```powershell
cd d:\00.test\PAPER\EthicaAI
git push origin main; git push anon main
```

```powershell
cd d:\00.test\PAPER\EthicaAI
git log --oneline -1 origin/main; git log --oneline -1 anon/main
```

→ **두 SHA가 동일**하면 배포 완료.
