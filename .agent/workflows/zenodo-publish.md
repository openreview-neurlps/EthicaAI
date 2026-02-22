---
description: Zenodo에 EthicaAI 논문 PDF 업로드 및 퍼블리시를 자동화하는 워크플로
---

# Zenodo Publish 워크플로

EthicaAI 논문을 Zenodo에 업로드하고 DOI를 발급받는 자동화 절차입니다.

## 사전 조건

- `.env` 파일에 `ZENODO_ACCESS_TOKEN`과 `ZENODO_RECORD_ID` 설정 필요
- Python 패키지: `requests`, `python-dotenv`
- PDF 변환 시: `playwright` + Chromium (`pip install playwright; playwright install chromium`)

## 단계

### 1. Zenodo 환경 확인
// turbo
```powershell
python -c "from dotenv import load_dotenv; import os; load_dotenv('.env'); print('TOKEN:', 'OK' if os.getenv('ZENODO_ACCESS_TOKEN') else 'MISSING'); print('RECORD:', os.getenv('ZENODO_RECORD_ID', 'MISSING'))"
```
작업 디렉토리: `d:\00.test\PAPER\EthicaAI`

### 2. PDF만 생성 (옵션 A: HTML → PDF 변환)
```powershell
python scripts/zenodo_upload.py --pdf-only
```
작업 디렉토리: `d:\00.test\PAPER\EthicaAI`

### 3. 기존 PDF 업로드 (퍼블리시 없이)
// turbo
```powershell
python scripts/zenodo_upload.py --upload-only
```
작업 디렉토리: `d:\00.test\PAPER\EthicaAI`

### 4. 업로드 + 퍼블리시 (전체 자동)
```powershell
python scripts/zenodo_upload.py --publish
```
작업 디렉토리: `d:\00.test\PAPER\EthicaAI`

> ⚠️ `--publish` 플래그는 DOI를 발급하며 **되돌릴 수 없습니다**. 메타데이터를 반드시 확인한 후 실행하세요.

### 5. 결과 확인
퍼블리시 후 아래 URL에서 확인:
- Zenodo: https://zenodo.org/records/18637742
- DOI: https://doi.org/10.5281/zenodo.18637742

## 관련 파일

| 파일 | 역할 |
|------|------|
| `scripts/zenodo_upload.py` | 업로드 스크립트 |
| `CITATION.cff` | 인용 정보 (DOI 포함) |
| `.env` | API 토큰 (git 추적 제외) |
| `submission/paper_english_v2.html` | PDF 변환 소스 |
| `submission/paper_english_v2.pdf` | 업로드 대상 PDF |
