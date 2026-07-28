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

## Try it now with the included pretrained model

This repo already ships a trained model in `gesture_model/`, so you don't
have to record or train anything to try it out — just run the setup script
and pick **option 3 (Run gesture UI server)** straight away. It recognizes
3 gestures out of the box:

- **Pinch**
- **Flexion**
- **Extension**

<!-- TODO: add a photo/GIF for each gesture below so users can see exactly
     how to perform it -->
| Gesture | How to do it |
|---|---|
| Pinch | ![Pinch gesture](images/gesture_pinch.webp) |
| Flexion | ![Flexion gesture](images/gesture_flexion.webp) |
| Extension | ![Extension gesture](images/gesture_extension.webp) |

### Where to wear the ArmBand

<!-- TODO: add a photo showing correct placement/orientation on the forearm -->
![ArmBand placement on forearm](images/armband_placement.webp)

Place the ArmBand on your forearm, electrodes facing the skin, positioned
over the muscle belly (roughly the upper third of the forearm, below the
elbow) — this is where the pinch/flexion/extension muscle activity is
strongest. Keep the band snug but not tight, and keep the same orientation
every time you use it, since the pretrained model was trained on a specific
placement/orientation.

### If gestures aren't recognized correctly

The pretrained model was trained on a specific person's arm, hand size, and
electrode placement, so it may not generalize well to everyone. If the UI
server misreads your gestures often:

1. Run the setup script and pick **option 1 (Record gesture data)** to
   record your own pinch / flexion / extension (and rest) trials.
2. Then pick **option 2 (Train gesture model)** to train a model on your
   own data.
3. Finally go back to **option 3 (Run gesture UI server)** — it will now
   use your freshly trained model instead of the bundled one, and should
   track your gestures much more reliably.

## Manual setup (if you don't want to use the setup scripts)

The setup scripts (`setup_and_run.bat`/`.sh`/`.command`) just automate the
steps below. If you'd rather do it yourself — or the scripts don't work on
your machine for some reason — here's exactly what to do by hand.

### 1. Install Python

Install **Python 3.12.7** (or any 3.10+ version — 3.12.7 is just what the
scripts install automatically, and what this project was tested with).

- **Windows**: download from
  https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe and run
  it. During install, make sure you check **"Add python.exe to PATH"** on
  the first screen — this is unchecked by default and is the #1 reason
  `python` isn't recognized afterward.
- **macOS**: `brew install python@3.12` (if you have Homebrew), or download
  the installer from
  https://www.python.org/ftp/python/3.12.7/python-3.12.7-macos11.pkg
- **Linux**: `sudo apt-get install python3 python3-venv python3-pip`
  (Debian/Ubuntu), or the equivalent for your distro (`dnf`, `pacman`,
  `zypper`, etc.)

Check it worked:
```
python --version
```
(On macOS/Linux this may be `python3 --version` instead.) You should see
`Python 3.12.7` or similar.

### 2. Create a virtual environment

A virtual environment ("venv") is just an isolated folder that holds its own
copy of Python packages, so this project's dependencies don't clash with
anything else on your system. From inside the project folder (the one with
`record_gesture.py` in it):

**Windows:**
```
python -m venv venv
```

**macOS/Linux:**
```
python3 -m venv venv
```

This creates a `venv/` folder in the project directory. It only needs to be
done once.

### 3. Activate the virtual environment

Activating it tells your terminal "use the Python and packages inside
`venv/`, not the system ones." You need to do this **every time** you open a
new terminal to work on this project.

**Windows (Command Prompt):**
```
venv\Scripts\activate.bat
```
**Windows (PowerShell):**
```
venv\Scripts\Activate.ps1
```
**macOS/Linux:**
```
source venv/bin/activate
```

You'll know it worked because your terminal prompt will show `(venv)` at the
start of the line.

### 4. Install the required packages

With the venv activated:
```
pip install -r requirements.txt
```
This reads `requirements.txt` and installs `bleak`, `numpy`, `sounddevice`,
`pandas`, `scikit-learn`, and `tensorflow` — everything the three scripts
need.

### 5. Run whichever script you want

Still with the venv activated:
```
python record_gesture.py
python train_gesture_model.py
python gesture_ui_server.py
```
Run only the one you need — you don't have to run all three.

### 6. Next time

You don't need to redo steps 1–4 — just reopen a terminal in the project
folder, activate the venv again (step 3), and run whichever script you need
(step 5).

## What each file is

| File | What it does |
|---|---|
| `record_gesture.py` | Connects to the ArmBand over Bluetooth, plays "go/stop" beep cues, and saves your filtered EMG + accelerometer data as CSV files for training. Run this to collect your own gesture data. |
| `train_gesture_model.py` | Reads the recorded CSV data and trains a gesture-classification model from it, saving the result into `gesture_model/`. |
| `model_architectures.py` | Defines the CNN / LSTM / CNN-LSTM neural network structures used for training. You don't run this directly — the training script imports it. |
| `gesture_ui_server.py` | Connects to the ArmBand live, runs the trained model in real time, and shows the recognized gesture in a browser-based UI. |
| `gesture_model/` | The trained model files. The repo already includes a pretrained one so you can try `gesture_ui_server.py` immediately; training your own data overwrites/adds to this. |
| `training_data/` | Where `record_gesture.py` saves the CSV files (and its `dataset_index.json`) from your recording sessions. |
| `requirements.txt` | The list of Python packages this project depends on — used by both the setup scripts and the manual `pip install -r requirements.txt` step. |
| `setup_and_run.bat` / `.sh` / `.command` | One-click setup + menu launcher for Windows / Linux / macOS respectively — automates everything in the "Manual setup" section above. |

## Windows
Double-click **`setup_and_run.bat`**.
- If Python isn't installed, it silently downloads and installs Python 3.12.7
  for you (no prompts, no dialogs).
- It creates a `venv` folder and installs all required packages into it.
- It then shows a menu: **1)** Record gesture data, **2)** Train the model,
  **3)** Run the gesture UI server, **4)** Exit.

If Windows shows a SmartScreen warning (common for unsigned `.bat`/`.exe`
downloads), click "More info" → "Run anyway".

If instead you see **"Smart App Control blocked a file that may be unsafe"**
(this is stricter than SmartScreen and has no "Run anyway" button), unblock
the file manually:
1. Right-click `setup_and_run.bat` → **Properties**
2. At the bottom of the **General** tab, check **Unblock** (next to "This
   file came from another computer and might be blocked to help protect
   this computer")
3. Click **Apply** → **OK**, then run the script again

If it's still blocked after that, Smart App Control may be in "enforced"
mode. You can either run the script from an already-open terminal
(`setup_and_run.bat` typed into PowerShell/Command Prompt in that folder) or
turn off Smart App Control entirely via Settings → Privacy & security →
Windows Security → App & browser control → Smart App Control settings → Off
(note: on Windows 11 this is generally a one-way switch — you can't turn it
back on without reinstalling Windows).

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
