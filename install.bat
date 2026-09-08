@echo off
REM MonoceptGPT installer for Windows
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ==========================================
echo   MonoceptGPT Installer (Windows)
echo ==========================================
echo Working directory: %CD%
echo.

REM 1. Python check
set "PY_CMD="
where python >nul 2>&1
if not errorlevel 1 (
  set "PY_CMD=python"
) else (
  where py >nul 2>&1
  if not errorlevel 1 (
    set "PY_CMD=py"
  )
)

if "!PY_CMD!"=="" (
  echo.
  echo ERROR: Python not found in PATH.
  echo Install Python 3.9+ from https://python.org
  echo IMPORTANT: tick "Add Python to PATH" during install.
  echo If you installed from Microsoft Store, uninstall it and use python.org instead.
  echo.
  pause
  exit /b 1
)
echo [1/4] Python found: !PY_CMD!
!PY_CMD! --version
!PY_CMD! -c "import sys; print('  location:', sys.executable)"

REM 2. Virtualenv — remove broken leftover, try python, fallback to py
if exist ".venv\Scripts\python.exe" (
  echo [2/4] Virtual environment already exists — reusing
) else (
  if exist ".venv" (
    echo [2/4] Removing broken .venv from previous attempt...
    rmdir /s /q ".venv"
  )
  echo [2/4] Creating virtual environment ^(.venv^)...
  !PY_CMD! -m venv .venv
  if errorlevel 1 (
    echo.
    echo First attempt failed. Trying 'py' launcher...
    where py >nul 2>&1
    if not errorlevel 1 (
      py -m venv .venv
    )
  )
  if not exist ".venv\Scripts\python.exe" (
    echo.
    echo ERROR: venv creation failed. Common causes:
    echo   1. Path has special characters ^(dashes, ampersands^). Try moving
    echo      this folder to a simple path like C:\agent_for_mono\
    echo   2. OneDrive is syncing this folder. Move it OUT of OneDrive.
    echo   3. Antivirus is blocking .venv creation. Whitelist this folder.
    echo   4. Python 'venv' module missing. Reinstall Python from python.org
    echo      ^(NOT Microsoft Store^) and tick "Add Python to PATH".
    echo.
    pause
    exit /b 1
  )
)

REM 3. Dependencies
echo [3/4] Installing dependencies ^(this may take a few minutes^)...
call ".venv\Scripts\python.exe" -m pip install --upgrade pip
call ".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo ERROR: Dependency installation failed.
  pause
  exit /b 1
)

REM 4. Config files
echo [4/4] Setting up config...

if not exist "teams.json" (
  if exist "teams.json.example" (
    copy /Y "teams.json.example" "teams.json" >nul
    echo   - Created teams.json ^(from example^)
  )
)

if not exist ".env" (
  echo.
  echo Enter Jira credentials ^(press Enter to skip; you can edit .env later^):
  set "D="
  set "E="
  set "T="
  set /p "D=  JIRA_DOMAIN (e.g. your-company.atlassian.net): "
  set /p "E=  JIRA_EMAIL:  "
  set /p "T=  JIRA_TOKEN:  "
  (
    echo JIRA_DOMAIN=!D!
    echo JIRA_EMAIL=!E!
    echo JIRA_TOKEN=!T!
  ) > .env
  echo   - Created .env
) else (
  echo   - .env already exists ^(skipped^)
)

echo.
echo ==========================================
echo   Install complete!
echo   Run:  run.bat
echo   Open: http://localhost:8080
echo ==========================================
pause
endlocal
