@echo off
setlocal
cd /d "%~dp0"
title Local Image Qwen

where python >nul 2>nul
if errorlevel 1 (
  echo Python n'est pas installe. Telechargez-le sur https://www.python.org/downloads/
  echo et cochez "Add python.exe to PATH" pendant l'installation.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/2] Creation de l'environnement Python...
  python -m venv .venv
)
echo [2/2] Installation / verification des dependances...
".venv\Scripts\python.exe" -m pip install -q --disable-pip-version-check -r requirements.txt

".venv\Scripts\python.exe" run.py %*
pause
