@echo off
chcp 65001 > nul
echo ========================================================
echo   EthicaAI GitHub 업로드 도우미 (전체 프로젝트)
echo ========================================================
echo.

:: Git 설치 확인
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [오류] Git이 설치되어 있지 않습니다. Git을 먼저 설치해주세요.
    pause
    exit /b
)

:: .git 디렉토리 확인 (이미 초기화되었는지)
if exist ".git" (
    echo [업데이트 모드] 기존 리포지토리에 변경사항을 푸시합니다.
    echo.
    set /p COMMIT_MSG="커밋 메시지를 입력하세요 (엔터 시 'Update research progress'): "
    if "%COMMIT_MSG%"=="" set COMMIT_MSG=Update research progress
    
    git add .
    git commit -m "%COMMIT_MSG%"
    git push
) else (
    echo [초기화 모드] 새로운 리포지토리를 연결합니다.
    echo.
    echo 1. GitHub에서 'New repository'를 생성하세요.
    echo 2. 생성된 HTTPS 주소를 복사하세요.
    echo.
    set /p REPO_URL="GitHub 리포지토리 주소 붙여넣기: "
    
    git init
    git add .
    git commit -m "Initial commit: EthicaAI NeurIPS 2026 Prep"
    git branch -M main
    git remote add origin %REPO_URL%
    git push -u origin main
)

echo.
if %errorlevel% equ 0 (
    echo [성공] GitHub 동기화가 완료되었습니다!
) else (
    echo [실패] 오류가 발생했습니다. 네트워크나 권한을 확인해주세요.
)
pause
