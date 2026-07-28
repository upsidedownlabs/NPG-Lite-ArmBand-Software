@echo off
setlocal enabledelayedexpansion
title Gesture ML Toolkit - Setup

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

set "PYTHON_VERSION=3.12.7"
set "PYTHON_INSTALLER=python-%PYTHON_VERSION%-amd64.exe"
set "PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/%PYTHON_INSTALLER%"
set "PY_EXE="

echo ================================================
echo   Gesture ML Toolkit - Setup
echo ================================================
echo.

REM ---- Step 1: look for an existing usable Python ----
for %%P in (py python python3) do (
    if not defined PY_EXE (
        %%P --version >nul 2>&1
        if !errorlevel! == 0 (
            set "PY_EXE=%%P"
        )
    )
)

REM also check the per-user install path a previous run of this script would have used
if not defined PY_EXE (
    if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
        set "PY_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"
    )
)

REM ---- Step 2: if nothing found, download and silently install Python 3.12.7 ----
if not defined PY_EXE (
    echo Python was not found on this system.
    echo Downloading Python %PYTHON_VERSION% ...
    powershell -NoProfile -Command "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%TEMP%\%PYTHON_INSTALLER%'"
    if not exist "%TEMP%\%PYTHON_INSTALLER%" (
        echo.
        echo Failed to download the Python installer. Check your internet connection
        echo and try again, or install Python manually from python.org.
        pause
        exit /b 1
    )

    echo Installing Python %PYTHON_VERSION% silently, this may take a minute...
    "%TEMP%\%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0
    timeout /t 5 /nobreak >nul

    if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
        set "PY_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"
    ) else (
        echo.
        echo Python installation could not be verified. Please install Python
        echo manually from https://www.python.org/downloads/ and re-run this script.
        pause
        exit /b 1
    )
    echo Python installed successfully.
)

echo Using Python:
"!PY_EXE!" --version

REM ---- Step 3: create the virtual environment (only once) ----
if not exist "venv\Scripts\python.exe" (
    echo.
    echo Creating virtual environment...
    "!PY_EXE!" -m venv venv
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

set "VENV_PY=%PROJECT_DIR%venv\Scripts\python.exe"

REM ---- Step 4: install/update requirements (safe to re-run, pip skips what's already installed) ----
echo.
echo Installing/updating required packages (first run can take a few minutes)...
"%VENV_PY%" -m pip install --upgrade pip >nul
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Something went wrong installing packages. See the errors above.
    pause
    exit /b 1
)

:menu
cls
echo ================================================
echo   Gesture ML Toolkit
echo ================================================
echo.
echo   1. Record gesture data     (record_gesture.py)
echo   2. Train gesture model     (train_gesture_model.py)
echo   3. Run gesture UI server   (gesture_ui_server.py)
echo   4. Exit
echo.
set /p CHOICE="Select an option [1-4]: "

if "%CHOICE%"=="1" (
    "%VENV_PY%" record_gesture.py
    echo.
    pause
    goto menu
)
if "%CHOICE%"=="2" (
    "%VENV_PY%" train_gesture_model.py
    echo.
    pause
    goto menu
)
if "%CHOICE%"=="3" (
    "%VENV_PY%" gesture_ui_server.py
    echo.
    pause
    goto menu
)
if "%CHOICE%"=="4" (
    exit /b 0
)

echo Invalid choice, try again.
pause
goto menu
