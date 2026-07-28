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

while true; do
    clear
    echo "================================================"
    echo "  Gesture ML Toolkit"
    echo "================================================"
    echo
    echo "  1. Record gesture data     (record_gesture.py)"
    echo "  2. Train gesture model     (train_gesture_model.py)"
    echo "  3. Run gesture UI server   (gesture_ui_server.py)"
    echo "  4. Exit"
    echo
    read -rp "Select an option [1-4]: " CHOICE
    case "$CHOICE" in
        1) "$VENV_PY" record_gesture.py; echo; read -rp "Press Enter to continue..." _ ;;
        2) "$VENV_PY" train_gesture_model.py; echo; read -rp "Press Enter to continue..." _ ;;
        3) "$VENV_PY" gesture_ui_server.py; echo; read -rp "Press Enter to continue..." _ ;;
        4) exit 0 ;;
        *) echo "Invalid choice."; sleep 1 ;;
    esac
done
