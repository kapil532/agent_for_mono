@echo off
REM Launch MonoceptGPT (Windows)
cd /d "%~dp0"

if not exist ".venv" (
  echo No .venv found. Run install.bat first.
  pause
  exit /b 1
)

echo Starting MonoceptGPT on http://localhost:8080 ...
call .venv\Scripts\python.exe app.py
pause
