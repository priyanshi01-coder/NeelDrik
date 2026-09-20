@echo off
REM ===== NEELDRIK - diagnose why the app will not start =====
title NEELDRIK - setup check
cd /d "%~dp0"
echo.
echo   NEELDRIK setup check
echo   ====================
echo.
echo   Folder: %CD%
echo.

set PY=
py -3 --version >nul 2>&1 && set PY=py -3
if "%PY%"=="" ( python --version >nul 2>&1 && set PY=python )
if "%PY%"=="" ( python3 --version >nul 2>&1 && set PY=python3 )

if "%PY%"=="" (
  echo   [X] Python NOT FOUND.  Install from python.org and tick
  echo       "Add python.exe to PATH" during setup.
  goto end
)
for /f "delims=" %%v in ('%PY% --version 2^>^&1') do echo   [OK] Python: %%v

echo.
echo   Checking required packages:
%PY% -c "import importlib,sys; m=['numpy','scipy','PIL','cv2','starlette','uvicorn','jwt','multipart']; [print('   ',('[OK] ' if importlib.util.find_spec(x) else '[MISSING] ')+x) for x in m]" 2>nul
if errorlevel 1 echo    [X] could not run the package check

echo.
echo   Checking project files:
if exist "run.py" (echo    [OK] run.py) else (echo    [X] run.py MISSING)
if exist "backend\app.py" (echo    [OK] backend\app.py) else (echo    [X] backend\app.py MISSING)
if exist "models\unet_oilspill.npz" (echo    [OK] trained model) else (echo    [X] trained model MISSING - run: %PY% train.py)
if exist "data\samples\spill_1.png" (echo    [OK] demo samples) else (echo    [X] demo samples MISSING)

echo.
echo   Checking whether port 8000 is already in use:
netstat -ano | findstr ":8000 " >nul 2>&1
if errorlevel 1 (echo    [OK] port 8000 is free) else (echo    [!] something is already using port 8000 - start with:  %PY% run.py --port 8010)

:end
echo.
echo   ----------------------------------------------------
echo   Send me a screenshot of this window if anything is [X]
echo   ----------------------------------------------------
echo.
pause
