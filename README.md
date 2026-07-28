# Gesture ML Toolkit — Quick Start

## Get the project

Don't download these files individually — get the **whole repo** so
everything (the scripts + the setup files) lands in the same folder
automatically:

- **Clone it:**
  ```bash
  git clone <repo-url>
  cd <repo-folder>
  ```
- **Or download it as a ZIP:** on the repo's GitHub page, click
  **Code → Download ZIP**, then unzip it and open the extracted folder.

Either way, you should end up with one folder containing `record_gesture.py`,
`train_gesture_model.py`, `gesture_ui_server.py`, `model_architectures.py`,
`requirements.txt`, `setup_and_run.bat`, `setup_and_run.sh`, and
`setup_and_run.command`.

## Windows
Double-click **`setup_and_run.bat`**.
- If Python isn't installed, it silently downloads and installs Python 3.12.7
  for you (no prompts, no dialogs).
- It creates a `venv` folder and installs all required packages into it.
- It then shows a menu: **1)** Record gesture data, **2)** Train the model,
  **3)** Run the gesture UI server, **4)** Exit.

If Windows shows a SmartScreen warning (common for unsigned `.bat`/`.exe`
downloads), click "More info" → "Run anyway".

## macOS
The executable permission on `.sh`/`.command` files doesn't survive a git
clone or ZIP download, so **first time only**, open Terminal in the project
folder and run:
```bash
chmod +x setup_and_run.sh setup_and_run.command
```
Then double-click **`setup_and_run.command`**. macOS may still require
`right-click → Open` (instead of double-click) the very first time, to bypass
the "unidentified developer" warning — after that, double-click works
normally.
- If Python is missing, it installs it via Homebrew (if you have it) or
  downloads the official python.org installer.
- Same venv + requirements + menu flow as Windows.

## Linux
The executable permission doesn't survive a git clone or ZIP download either,
so **first time only**, run:
```bash
chmod +x setup_and_run.sh
```
Then run **`./setup_and_run.sh`** from a terminal (or double-click it if your
file manager is set to execute `.sh` files).

If you ever see `zsh: permission denied: ./setup_and_run.sh` or a similar
"permission denied" error, that means the `chmod +x` step above was skipped —
just run it once and the script will work fine from then on.
- If Python is missing, it installs it using your distro's package manager
  (`apt`, `dnf`, `pacman`, or `zypper`), prompting for your `sudo` password
  when needed.
- Same venv + requirements + menu flow as Windows.

## Notes
- All scripts are safe to re-run: they skip the Python install and venv
  creation if already done, and `pip install -r requirements.txt` only
  installs what's missing/outdated.
- Everything installs into a local `venv/` folder inside the project —
  nothing is installed system-wide except Python itself (if it was missing).
- `requirements.txt` covers all three scripts: `bleak` (BLE), `numpy` /
  `sounddevice` (beep cues + signal math), `pandas` / `scikit-learn` /
  `tensorflow` (training + inference).
