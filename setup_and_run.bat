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
echo   Turn ON your NPG Lite board before picking 1, 3 or 4.
echo.
echo   1. Record gesture data     (record_gesture.py)
echo   2. Train gesture model     (train_gesture_model.py)
echo   3. Run gesture UI server   (gesture_ui_server.py)
echo   4. Run gesture controller  (gesture_controller.py - tap / hold)
echo   5. Exit
echo.
set /p CHOICE="Type a number [1-5] and press Enter: "

if "%CHOICE%"=="1" goto run_record
if "%CHOICE%"=="2" goto run_train
if "%CHOICE%"=="3" goto run_ui
if "%CHOICE%"=="4" goto controller_menu
if "%CHOICE%"=="5" exit /b 0

echo Invalid choice - type 1, 2, 3, 4 or 5 and press Enter.
pause
goto menu

:run_record
"%VENV_PY%" record_gesture.py
echo.
pause
goto menu

:run_train
"%VENV_PY%" train_gesture_model.py
echo.
pause
goto menu

:run_ui
"%VENV_PY%" gesture_ui_server.py
echo.
pause
goto menu

REM ---- Gesture controller submenu: tap vs hold ----
:controller_menu
cls
echo ================================================
echo   Gesture Controller - keyboard control
echo ================================================
echo.
echo   Make sure your NPG Lite is TURNED ON and the ArmBand is on
echo   your forearm before you start. Calibration runs automatically.
echo.
echo   Gestures:  flexion -^> LEFT       extension -^> RIGHT
echo              arm UP   + pinch -^> UP
echo              arm DOWN + pinch -^> DOWN
echo.
echo   1. TAP mode   - one gesture = one key press
echo                   (menus, browsing, turn-based games)
echo   2. HOLD mode  - key stays held down while you hold the gesture
echo                   (driving/racing games, continuous steering)
echo   3. TAP mode, DRY RUN - shows what it detects, sends NO key
echo                   presses. Use this first to test safely.
echo   4. Back to main menu
echo.
set /p CMODE="Type a number [1-4] and press Enter: "

if "%CMODE%"=="1" goto ctrl_tap
if "%CMODE%"=="2" goto ctrl_hold
if "%CMODE%"=="3" goto ctrl_dry
if "%CMODE%"=="4" goto menu

echo Invalid choice - type 1, 2, 3 or 4 and press Enter.
pause
goto controller_menu

:ctrl_tap
call :ensure_ctrl_deps
"%VENV_PY%" gesture_controller.py --press_mode tap
echo.
pause
goto controller_menu

:ctrl_hold
call :ensure_ctrl_deps
"%VENV_PY%" gesture_controller.py --press_mode hold
echo.
pause
goto controller_menu

:ctrl_dry
REM No :ensure_ctrl_deps here on purpose: --dry_run uses NullBackend and injects
REM no keys, so the safe test mode must not depend on installing pynput.
"%VENV_PY%" gesture_controller.py --press_mode tap --dry_run
echo.
pause
goto controller_menu

REM ---- gesture_controller.py needs pynput to press keys on Windows ----
:ensure_ctrl_deps
"%VENV_PY%" -c "import pynput" >nul 2>&1
if errorlevel 1 (
    echo Installing keyboard-control package ^(pynput^) - one time only...
    "%VENV_PY%" -m pip install pynput
    if errorlevel 1 (
        echo.
        echo WARNING: pynput could not be installed, so key presses will not work.
        echo See the README troubleshooting section.
        pause
    )
)
goto :eof