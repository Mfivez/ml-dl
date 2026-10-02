@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=python"
where py >nul 2>nul
if not errorlevel 1 set "PYTHON_EXE=py -3.12"

if not exist .venv\Scripts\python.exe (
    echo Creation du venv Python 3.12...
    %PYTHON_EXE% -m venv .venv
    if errorlevel 1 goto :error
)

echo Installation des dependances...
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :error
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo Enregistrement du noyau Jupyter...
.venv\Scripts\python.exe -m ipykernel install --user --name fondamentaux-ml-dl --display-name "Python 3.12 - Fondamentaux ML DL"
if errorlevel 1 goto :error

echo.
echo Environnement pret.
pause
exit /b 0

:error
echo.
echo L'installation a echoue. Consultez les messages ci-dessus.
pause
exit /b 1
