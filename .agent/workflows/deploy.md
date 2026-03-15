---
description: EthicaAI 코드/논문 변경사항을 양쪽 리모트(origin + anon)에 배포
---

# EthicaAI 듀얼 리모트 배포

EthicaAI는 두 개의 Git 리모트를 사용합니다:
- `origin` → `Yesol-Pilot/EthicaAI` (개인 리포지토리)
- `anon` → `neogenesislab/EthicaAI-NeurIPS2026` (익명 제출용 리포지토리)

**모든 push는 반드시 양쪽에 동시 수행해야 합니다.**

## 환경 정보 (MiKTeX on Windows)

- **TeX**: `pdflatex` via MiKTeX (`C:\Users\yesol\AppData\Local\Programs\MiKTeX`)
- **latexmk 사용 불가**: MiKTeX에 perl 미설치 → `pdflatex` 직접 3-pass 사용
- **bibtex**: 사용 가능
- **Python**: conda base, `d:\00.test\PAPER\EthicaAI` 기준
- **git commit 주의**: pre-commit hook 또는 대용량 파일로 2분+ 소요 가능 → `--no-verify` 권장

## 배포 순서

// turbo-all

1. 작업 디렉토리 확인
```powershell
cd d:\00.test\PAPER\EthicaAI
```

2. 변경사항 스테이징 및 커밋
```powershell
git add -A; git status
```

3. 커밋 (메시지 규칙: `category: description`)
```powershell
git commit --no-verify -m "category: description"
```
카테고리 예시:
- `ssot:` — SSOT 파이프라인 관련
- `feat:` — 새 실험/기능
- `fix:` — 버그 수정
- `polish:` — 논문 텍스트 수정
- `verify:` — 검증 스크립트 변경

4. 양쪽 리모트에 push
```powershell
git push origin main; git push anon main
```

5. 동기화 확인
```powershell
git log --oneline -1 origin/main; git log --oneline -1 anon/main
```
→ 두 줄의 커밋 SHA가 동일해야 합니다.

## 논문 PDF 빌드 (3-pass)

TeX 변경 후 반드시 실행:

```powershell
cd d:\00.test\PAPER\EthicaAI\NeurIPS2026_final_submission\paper
pdflatex -interaction=nonstopmode unified_paper.tex
bibtex unified_paper
pdflatex -interaction=nonstopmode unified_paper.tex
pdflatex -interaction=nonstopmode unified_paper.tex
```

**알려진 경고**: L324-325의 `Missing $` (기존 코드, PDF 생성에 영향 없음)

빌드 확인:
```powershell
$fi = Get-Item unified_paper.pdf; Write-Host "PDF: $($fi.Length / 1MB) MB, Pages: $(Select-String -Path unified_paper.log -Pattern 'Output written' | ForEach-Object { $_.Line })"
```

## Supplementary ZIP 생성

```powershell
python d:\00.test\PAPER\EthicaAI\NeurIPS2026_final_submission\code\scripts\package_supplementary.py
```

옵션:
- `--dry-run` — 파일 목록만 확인 (ZIP 미생성)
- `--output PATH` — 출력 경로 지정

자동으로 수행하는 것:
- `.git`, `__pycache__`, build artifacts 제외
- 개인 경로/이름 익명화 (yesol, Yesol-Pilot, neogenesislab → anonymous)
- outputs/ 에서는 JSON 요약만 포함
- 100 MB 제한 검증

## SSOT 검증 포함 배포 (논문/테이블 변경 시)

논문 테이블이나 실험 결과가 변경된 경우, push 전에 SSOT 검증을 반드시 실행합니다.

1. 테이블 재생성
```powershell
python NeurIPS2026_final_submission/code/scripts/generate_tables.py
```

2. SSOT 동기화 + 연결성 체크
```powershell
python NeurIPS2026_final_submission/code/scripts/generate_tables.py --check
```
→ "OK: All tables match" + "OK: All SSOT tables are \input-linked" 확인

3. 수치 검증
```powershell
python NeurIPS2026_final_submission/code/scripts/verify_numbers.py
```
→ FAIL=0 확인

4. 커밋 및 양쪽 push
```powershell
git add -A; git commit --no-verify -m "ssot: regenerate tables + verify numbers"; git push origin main; git push anon main
```

## 주의사항

- `anon` 리모트에 push를 빠뜨리면, 익명 리뷰어가 최신 코드를 볼 수 없어 신뢰성 문제 발생
- `paper/unified_paper.tex`(개발용)와 `NeurIPS2026_final_submission/paper/unified_paper.tex`(제출용 ~1779줄) 구분
- 민감 정보(API 키, 개인정보) push 전 반드시 확인
- git commit이 2분+ 걸리면 강제 종료 후 `git log -1 --oneline`으로 이미 커밋되었는지 확인
