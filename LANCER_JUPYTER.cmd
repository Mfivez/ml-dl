@echo off
setlocal
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo Environnement absent. Lancez d'abord INSTALLER_WINDOWS.cmd.
    pause
    exit /b 1
)

.venv\Scripts\python.exe -m jupyter lab
