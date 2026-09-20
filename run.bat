@echo off
REM ===== NEELDRIK - double-click launcher (Windows) =====
title NEELDRIK - SIH26143
cd /d "%~dp0"
echo.
echo   NEELDRIK  -  oil spill detection
echo   ---------------------------------
echo.

REM --- find Python: try the py launcher first, then python ---
set PY=
py -3 --version >nul 2>&1 && set PY=py -3
if "%PY%"=="" ( python --version >nul 2>&1 && set PY=python )
if "%PY%"=="" ( python3 --version >nul 2>&1 && set PY=python3 )

if "%PY%"=="" (
  echo   [X] Python was not found on this computer.
  echo.
  echo       Install it from https://www.python.org/downloads/
  echo       IMPORTANT: on the first installer screen, tick
  echo       "Add python.exe to PATH", then run this file again.
  echo.
  pause
  exit /b 1
)

for /f "delims=" %%v in ('%PY% --version 2^>^&1') do echo   Using %%v
echo.

echo   Installing dependencies (first run only, ~1 minute)...
%PY% -m pip install --quiet --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
  echo.
  echo   [X] Dependency install failed. The error is above.
  echo       Try running this command yourself to see more detail:
  echo         %PY% -m pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)
echo   Dependencies OK.
echo.

if not exist "models\unet_oilspill.npz" (
  echo   No trained model found - training now, about 25 minutes...
  %PY% train.py
  if errorlevel 1 ( echo. & echo   [X] Training failed. & pause & exit /b 1 )
)

echo   Starting the server...
echo.
echo   ================================================
echo     Open this in your browser:  http://127.0.0.1:8000
echo     Keep THIS window open while you use the app.
echo     Press Ctrl+C here to stop it.
echo   ================================================
echo.
%PY% run.py
echo.
echo   Server stopped.
pause
