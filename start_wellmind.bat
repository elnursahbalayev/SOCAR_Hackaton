@echo off
REM ============================================================================
REM  WellMind - start the app (backend + built frontend) and open it in the browser.
REM  Close this window (or press Ctrl+C) to stop the app.
REM  Python used: %WELLMIND_PYTHON% if set, else the conda env "hackathon", else "python" on PATH.
REM ============================================================================
cd /d "%~dp0"
title WellMind server - %~dp0

REM --- refuse to start twice: an old server on port 8000 would keep serving an old version ---
netstat -ano | findstr /R /C:":8000 .*LISTENING" >nul
if not errorlevel 1 (
    echo.
    echo  Port 8000 is already in use - WellMind ^(or another program^) is already running.
    echo  Close the other WellMind window first, then start this file again.
    echo.
    pause
    exit /b 1
)

REM --- find Python ---
set "PY=%WELLMIND_PYTHON%"
if "%PY%"=="" if exist "%USERPROFILE%\anaconda3\envs\hackathon\python.exe" set "PY=%USERPROFILE%\anaconda3\envs\hackathon\python.exe"
if "%PY%"=="" if exist "%USERPROFILE%\miniconda3\envs\hackathon\python.exe" set "PY=%USERPROFILE%\miniconda3\envs\hackathon\python.exe"
if "%PY%"=="" set "PY=python"
echo  Starting WellMind from %~dp0
echo  Python: %PY%
echo  The browser opens as soon as the server is ready (http://localhost:8000).
echo.

REM --- open the browser once the server answers (checks every second, up to 90 s) ---
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "for ($i = 0; $i -lt 90; $i++) { try { Invoke-WebRequest http://127.0.0.1:8000/api/health -UseBasicParsing -TimeoutSec 2 | Out-Null; Start-Process 'http://localhost:8000'; break } catch { Start-Sleep -Seconds 1 } }"

"%PY%" -m uvicorn backend.api:app --port 8000
echo.
echo  WellMind stopped.
pause
