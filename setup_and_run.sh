#!/usr/bin/env bash
# Gesture ML Toolkit - setup + launcher for macOS / Linux
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON_VERSION="3.12.7"
PY_BIN=""

echo "================================================"
echo "  Gesture ML Toolkit - Setup"
echo "================================================"
echo

find_python() {
    for candidate in python3.12 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            ver=$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "0.0")
            major="${ver%%.*}"
            minor="${ver##*.}"
            if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
                PY_BIN="$candidate"
                return 0
            fi
        fi
    done
    return 1
}

install_python_macos() {
    echo "Python not found. Attempting to install Python $PYTHON_VERSION..."
    if command -v brew >/dev/null 2>&1; then
        brew install python@3.12
    else
        echo "Homebrew not found - downloading the official installer from python.org instead."
        PKG_URL="https://www.python.org/ftp/python/${PYTHON_VERSION}/python-${PYTHON_VERSION}-macos11.pkg"
        curl -L -o "/tmp/python-installer.pkg" "$PKG_URL"
        echo "Installing (you may be asked for your password)..."
        sudo installer -pkg "/tmp/python-installer.pkg" -target /
    fi
}

install_python_linux() {
    echo "Python not found. Attempting to install it with your package manager..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y python3 python3-pip
    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -Sy --noconfirm python python-pip
    elif command -v zypper >/dev/null 2>&1; then
        sudo zypper install -y python3 python3-pip
    else
        echo "Could not detect a supported package manager (apt/dnf/pacman/zypper)."
        echo "Please install Python 3.10+ manually from https://www.python.org/downloads/"
        echo "and re-run this script."
        exit 1
    fi
}

if ! find_python; then
    OS_NAME="$(uname -s)"
    case "$OS_NAME" in
        Darwin) install_python_macos ;;
        Linux)  install_python_linux ;;
        *) echo "Unsupported OS: $OS_NAME"; exit 1 ;;
    esac
    if ! find_python; then
        echo
        echo "Python installation could not be verified. Please install Python"
        echo "manually and re-run this script."
        exit 1
    fi
fi

echo "Using Python: $("$PY_BIN" --version)"

if [ ! -f "venv/bin/python" ]; then
    echo
    echo "Creating virtual environment..."
    "$PY_BIN" -m venv venv
fi

VENV_PY="$PROJECT_DIR/venv/bin/python"

echo
echo "Installing/updating required packages (first run can take a few minutes)..."
"$VENV_PY" -m pip install --upgrade pip >/dev/null
"$VENV_PY" -m pip install -r requirements.txt

ensure_controller_deps() {
    # gesture_controller.py needs a keystroke-injection backend.
    if ! "$VENV_PY" -c "import pynput" >/dev/null 2>&1; then
        echo "Installing keyboard-control package (pynput) - one time only..."
        "$VENV_PY" -m pip install pynput || \
            echo "WARNING: pynput could not be installed. On Linux the uinput backend may still work."
    fi
    if [ "$(uname -s)" = "Linux" ]; then
        # Optional, but the best backend on Linux (works on Wayland + games).
        if ! "$VENV_PY" -c "import evdev" >/dev/null 2>&1; then
            echo "Installing Linux kernel-level backend (evdev) - one time only..."
            "$VENV_PY" -m pip install evdev >/dev/null 2>&1 || \
                echo "NOTE: evdev not installed (optional). Falling back to pynput."
        fi
    fi
}

controller_menu() {
    while true; do
        clear
        echo "================================================"
        echo "  Gesture Controller - keyboard control"
        echo "================================================"
        echo
        echo "  Make sure your NPG Lite is TURNED ON and the ArmBand is on"
        echo "  your forearm before you start. Calibration runs automatically."
        echo
        echo "  Gestures:  flexion -> LEFT      extension -> RIGHT"
        echo "             arm UP   + pinch -> UP"
        echo "             arm DOWN + pinch -> DOWN"
        echo
        echo "  1. TAP mode   - one gesture = one key press"
        echo "                  (menus, browsing, turn-based games)"
        echo "  2. HOLD mode  - key stays held down while you hold the gesture"
        echo "                  (driving/racing games, continuous steering)"
        echo "  3. TAP mode, DRY RUN - shows what it detects, sends NO key"
        echo "                  presses. Use this first to test safely."
        echo "  4. Back to main menu"
        echo
        read -rp "Type a number [1-4] and press Enter: " CMODE
        case "$CMODE" in
            1)  ensure_controller_deps
                "$VENV_PY" gesture_controller.py --press_mode tap
                echo; read -rp "Press Enter to continue..." _ ;;
            2)  ensure_controller_deps
                "$VENV_PY" gesture_controller.py --press_mode hold
                echo; read -rp "Press Enter to continue..." _ ;;
            3)  # No ensure_controller_deps here on purpose: --dry_run uses
                # NullBackend and injects no keys, so the safe test mode must
                # not depend on installing pynput/evdev (or on network access).
                "$VENV_PY" gesture_controller.py --press_mode tap --dry_run
                echo; read -rp "Press Enter to continue..." _ ;;
            4)  return 0 ;;
            *)  echo "Invalid choice - type 1, 2, 3 or 4 and press Enter."; sleep 1 ;;
        esac
    done
}

while true; do
    clear
    echo "================================================"
    echo "  Gesture ML Toolkit"
    echo "================================================"
    echo
    echo "  Turn ON your NPG Lite board before picking 1, 3 or 4."
    echo
    echo "  1. Record gesture data     (record_gesture.py)"
    echo "  2. Train gesture model     (train_gesture_model.py)"
    echo "  3. Run gesture UI server   (gesture_ui_server.py)"
    echo "  4. Run gesture controller  (gesture_controller.py - tap / hold)"
    echo "  5. Exit"
    echo
    read -rp "Type a number [1-5] and press Enter: " CHOICE
    case "$CHOICE" in
        1) "$VENV_PY" record_gesture.py; echo; read -rp "Press Enter to continue..." _ ;;
        2) "$VENV_PY" train_gesture_model.py; echo; read -rp "Press Enter to continue..." _ ;;
        3) "$VENV_PY" gesture_ui_server.py; echo; read -rp "Press Enter to continue..." _ ;;
        4) controller_menu ;;
        5) exit 0 ;;
        *) echo "Invalid choice - type 1, 2, 3, 4 or 5 and press Enter."; sleep 1 ;;
    esac
done