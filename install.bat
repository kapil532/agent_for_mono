@echo off
REM MonoceptGPT installer for Windows
setlocal
cd /d "%~dp0"

echo ==========================================
echo   MonoceptGPT Installer (Windows)
echo ==========================================

REM 1. Python check
where python >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python not found. Install Python 3.9+ from https://python.org
  echo Make sure to check "Add Python to PATH" during install.
  pause
  exit /b 1
)
echo [1/4] Python found
python --version

REM 2. Virtualenv
if not exist ".venv" (
  echo [2/4] Creating virtual environment (.venv)...
  python -m venv .venv
) else (
  echo [2/4] Virtual environment already exists
)

REM 3. Dependencies
echo [3/4] Installing dependencies (this may take a few minutes)...
call .venv\Scripts\python.exe -m pip install --upgrade pip
call .venv\Scripts\pip.exe install -r requirements.txt
if errorlevel 1 (
  echo ERROR: Dependency installation failed.
  pause
  exit /b 1
)

REM 4. Config files
echo [4/4] Setting up config...

if not exist "teams.json" (
  if exist "teams.json.example" (
    copy /Y "teams.json.example" "teams.json" >nul
    echo   - Created teams.json (from example)
  )
)

if not exist ".env" (
  echo.
  echo Enter Jira credentials (press Enter to skip; you can edit .env later):
  set /p D=  JIRA_DOMAIN (e.g. your-company.atlassian.net):
  set /p E=  JIRA_EMAIL:
  set /p T=  JIRA_TOKEN:
  (
    echo JIRA_DOMAIN=%D%
    echo JIRA_EMAIL=%E%
    echo JIRA_TOKEN=%T%
  ) > .env
  echo   - Created .env
) else (
  echo   - .env already exists (skipped)
)

echo.
echo ==========================================
echo   Install complete!
echo   Run:  run.bat
echo   Open: http://localhost:8080
echo ==========================================
pause
