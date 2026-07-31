"""
gesture_controller.py
----------------------
BLE + Notch/EMG filter + sliding-window + trained-model pipeline that
detects gestures and maps them to button presses:
- Flexion -> Left button press
- Extension -> Right button press
- Pinch + Moving Up -> Up button press (requires significant movement)
- Pinch + Moving Down -> Down button press (requires significant movement)

Features:
- 300ms debounce time between button presses
- 3-step calibration (UP -> DOWN -> REST) with 2-second countdown
- Strict movement detection

Usage:
    python gesture_controller.py --model_dir gesture_model
"""

import argparse
import asyncio
import json
import os
import platform
import queue
import shutil
import subprocess
import threading
import time
from collections import deque

import numpy as np
from bleak import BleakScanner, BleakClient
import tensorflow as tf
from tensorflow import keras

# Keyboard injection is handled by the pluggable backends below. Availability is
# probed and reported at startup by build_key_backend(), not here -- an import
# warning at module load was misleading, since uinput can work perfectly well
# with pynput absent.

# ==============================
# BLE UUIDs
# ==============================
DATA_CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
CONTROL_CHAR_UUID = "0000ff01-0000-1000-8000-00805f9b34fb"
IMU_CHAR_UUID = "5a153fa9-7be0-400c-8ef8-d84502b31c4d"
DEVICE_NAME_PREFIX = "NPG-Lite-band"
SAMPLES_PER_PACKET = 20
IMU_SAMPLE_LEN = 7

ADC_BITS = "12"
NOTCH_TYPE = 1

# NotchFilter/EXGFilter only have coefficients for these rates. At any other
# rate both filters degrade to an unfiltered pass-through, so the model would be
# fed raw mains-hum-laden samples that look nothing like its training data while
# the UI still shows confident predictions. Fail loudly instead.
SUPPORTED_SAMPLING_RATES = (250, 500)

# How long the main loop waits for the first IMU sample before giving up with an
# explanation instead of printing "Waiting for accelerometer data..." forever.
ACCEL_WAIT_TIMEOUT_S = 20.0

# Each BLE teardown step gets its own budget so one unresponsive board can't
# stall the exit; BLE_SHUTDOWN_TIMEOUT_S is how long main() waits for the BLE
# thread to finish that teardown after Ctrl+C.
DISCONNECT_TIMEOUT_S = 2.0
BLE_SHUTDOWN_TIMEOUT_S = 12.0

# Per-class thresholds used when --class_thresholds is not passed. Derived from
# replay_buffer.npz with tune_thresholds.py for the shipped 4-class model; any
# class name here that the loaded model does not have is ignored (see
# parse_class_thresholds), never a startup error.
DEFAULT_CLASS_THRESHOLDS = "flexion=0.50,extension=0.91,pinch=0.66,rest=0.50"

# ==============================
# Filters
# ==============================
class NotchFilter:
    def __init__(self):
        self.z1_1 = self.z2_1 = self.z1_2 = self.z2_2 = 0.0
        self.x_1 = self.x_2 = 0.0
        self.sampling_rate = 0

    def set_sampling_rate(self, sampling_rate):
        self.sampling_rate = sampling_rate

    def process(self, input_val, notch_type):
        if not notch_type:
            return input_val
        output = input_val
        if self.sampling_rate == 500:
            if notch_type == 1:
                self.x_1 = output - (-1.56858163 * self.z1_1) - (0.96424138 * self.z2_1)
                output = 0.96508099 * self.x_1 + -1.56202714 * self.z1_1 + 0.96508099 * self.z2_1
                self.z2_1 = self.z1_1
                self.z1_1 = self.x_1
                self.x_2 = output - (-1.61100358 * self.z1_2) - (0.96592171 * self.z2_2)
                output = 1.0 * self.x_2 + -1.61854514 * self.z1_2 + 1.0 * self.z2_2
                self.z2_2 = self.z1_2
                self.z1_2 = self.x_2
            elif notch_type == 2:
                self.x_1 = output - (-1.40810535 * self.z1_1) - (0.96443153 * self.z2_1)
                output = 0.96508099 * self.x_1 + (-1.40747202 * self.z1_1) + (0.96508099 * self.z2_1)
                self.z2_1 = self.z1_1
                self.z1_1 = self.x_1
                self.x_2 = output - (-1.45687509 * self.z1_2) - (0.96573127 * self.z2_2)
                output = 1.00000000 * self.x_2 + (-1.45839783 * self.z1_2) + (1.00000000 * self.z2_2)
                self.z2_2 = self.z1_2
                self.z1_2 = self.x_2
        elif self.sampling_rate == 250:
            if notch_type == 1:
                self.x_1 = output - (-0.53127491 * self.z1_1) - (0.93061518 * self.z2_1)
                output = 0.93137886 * self.x_1 + (-0.57635175 * self.z1_1) + 0.93137886 * self.z2_1
                self.z2_1 = self.z1_1
                self.z1_1 = self.x_1
                self.x_2 = output - (-0.66243374 * self.z1_2) - (0.93214913 * self.z2_2)
                output = 1.00000000 * self.x_2 + (-0.61881558 * self.z1_2) + 1.00000000 * self.z2_2
                self.z2_2 = self.z1_2
                self.z1_2 = self.x_2
            elif notch_type == 2:
                self.x_1 = output - (-0.05269865 * self.z1_1) - (0.93123336 * self.z2_1)
                output = 0.93137886 * self.x_1 + (-0.11711144 * self.z1_1) + 0.93137886 * self.z2_1
                self.z2_1 = self.z1_1
                self.z1_1 = self.x_1
                self.x_2 = output - (-0.18985625 * self.z1_2) - (0.93153034 * self.z2_2)
                output = 1.00000000 * self.x_2 + (-0.12573985 * self.z1_2) + 1.00000000 * self.z2_2
                self.z2_2 = self.z1_2
                self.z1_2 = self.x_2
        return output


class EXGFilter:
    def __init__(self):
        self.z1 = self.z2 = 0.0
        self.x4 = 0.0
        self.bits_points = 0
        self.sampling_rate = 0

    def set_bits(self, bits, sampling_rate):
        self.sampling_rate = sampling_rate
        self.bits_points = 2 ** int(bits)

    def process(self, input_val, exg_type):
        if not exg_type:
            return input_val
        output = input_val
        if self.sampling_rate == 500 and exg_type == 4:
            self.x4 = output - (-0.82523238 * self.z1) - (0.29463653 * self.z2)
            output = 0.52996723 * self.x4 + -1.05993445 * self.z1 + 0.52996723 * self.z2
            self.z2 = self.z1
            self.z1 = self.x4
        elif self.sampling_rate == 250 and exg_type == 4:
            self.x4 = output - 0.22115344 * self.z1 - 0.18023207 * self.z2
            output = 0.23976966 * self.x4 + -0.47953932 * self.z1 + 0.23976966 * self.z2
            self.z2 = self.z1
            self.z1 = self.x4
        # Unsupported (sampling_rate, exg_type) pair: pass the sample through
        # unfiltered. This used to `return 0.0`, which handed the model an
        # all-zero EMG channel with no error and no gesture ever detected.
        # main() validates sampling_rate against SUPPORTED_SAMPLING_RATES at
        # startup, so this branch should be unreachable in practice.
        return output


# ==============================
# Keyboard backends
# ==============================
# Why not just pynput:
#   1. HOLD DURATION. The original code did press(k) then release(k) with nothing
#      in between, holding the key for <1 ms. Anything that reads key STATE once
#      per frame (SDL, Unity, most games/emulators) has a 16.7 ms window at 60 fps.
#      A sub-millisecond tap lands between polls and is never observed. This is the
# "sometimes it doesn't press" symptom, and it is independent of the library.
#   2. INJECTION LAYER. On Linux pynput uses XTest. XTest events do not reach
#      Wayland-native apps at all, and are ignored by anything reading evdev
#      directly. uinput injects at kernel level, so it is indistinguishable from a
#      real keyboard to X11, Wayland and raw-evdev readers alike.

BUTTONS = ("left", "right", "up", "down")


class NullBackend:
    name = "none (logging only)"

    def __init__(self, hold_s=0.0):
        self.hold_s = hold_s

    def key_down(self, button):
        pass

    def key_up(self, button):
        pass

    def tap(self, button):
        self.key_down(button)
        time.sleep(self.hold_s)
        self.key_up(button)

    def close(self):
        pass


class UinputBackend:
    """Kernel-level injection via /dev/uinput. Works on X11, Wayland and for
    programs that read evdev directly."""

    name = "uinput (evdev, kernel-level)"

    def __init__(self, hold_s):
        from evdev import UInput, ecodes as e
        self._e = e
        self.hold_s = hold_s
        self._map = {"left": e.KEY_LEFT, "right": e.KEY_RIGHT,
                     "up": e.KEY_UP, "down": e.KEY_DOWN}
        caps = {e.EV_KEY: list(self._map.values())}
        self._ui = UInput(caps, name="npg-lite-gesture-keyboard", version=1)
        # The compositor/X server has to enumerate the new device before it will
        # accept events from it. Writes issued during this window are silently
        # dropped, which would make the FIRST gesture after startup never register.
        time.sleep(0.5)

    def key_down(self, button):
        self._ui.write(self._e.EV_KEY, self._map[button], 1)
        self._ui.syn()

    def key_up(self, button):
        self._ui.write(self._e.EV_KEY, self._map[button], 0)
        self._ui.syn()

    def tap(self, button):
        self.key_down(button)
        time.sleep(self.hold_s)
        self.key_up(button)

    def close(self):
        try:
            self._ui.close()
        except Exception:
            pass


class PynputBackend:
    name = "pynput (XTest on Linux / Quartz on macOS)"

    def __init__(self, hold_s):
        from pynput.keyboard import Key, Controller
        self.hold_s = hold_s
        self._kb = Controller()
        self._map = {"left": Key.left, "right": Key.right,
                     "up": Key.up, "down": Key.down}

    def key_down(self, button):
        self._kb.press(self._map[button])

    def key_up(self, button):
        self._kb.release(self._map[button])

    def tap(self, button):
        self.key_down(button)
        time.sleep(self.hold_s)   # the fix that matters most, whatever the backend
        self.key_up(button)

    def close(self):
        pass


class XdotoolBackend:
    """Last resort. Spawns a process per keystroke (~10-30 ms), so it is slow and
    jittery -- but it is a useful cross-check that the problem is injection."""

    name = "xdotool (subprocess, slow)"

    def __init__(self, hold_s):
        self.hold_s = hold_s
        if not shutil.which("xdotool"):
            raise RuntimeError("xdotool not installed")

    def key_down(self, button):
        subprocess.run(["xdotool", "keydown", button], check=False)

    def key_up(self, button):
        subprocess.run(["xdotool", "keyup", button], check=False)

    def tap(self, button):
        self.key_down(button)
        time.sleep(self.hold_s)
        self.key_up(button)

    def close(self):
        pass


def release_all(backend):
    """Unconditionally lift every key we can emit.

    A stuck key in a driving game means the car steers into a wall forever, so
    this is called on shutdown regardless of what we think is held. key_up on a
    key that is already up is harmless.
    """
    for b in BUTTONS:
        try:
            backend.key_up(b)
        except Exception:
            pass


def describe_session():
    bits = [f"os={platform.system()}"]
    st = os.environ.get("XDG_SESSION_TYPE")
    if st:
        bits.append(f"session={st}")
    if os.environ.get("WAYLAND_DISPLAY"):
        bits.append("wayland=yes")
    if os.environ.get("DISPLAY"):
        bits.append(f"display={os.environ['DISPLAY']}")
    return " ".join(bits)


def build_key_backend(choice, hold_s):
    """Returns (backend, notes). 'auto' prefers uinput on Linux, then pynput."""
    notes = []
    is_linux = platform.system() == "Linux"
    wayland = bool(os.environ.get("WAYLAND_DISPLAY")) or \
        os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"

    order = {
        "auto": (["uinput", "pynput", "xdotool"] if is_linux else ["pynput"]),
        "uinput": ["uinput"],
        "pynput": ["pynput"],
        "xdotool": ["xdotool"],
        "none": ["none"],
    }[choice]

    for want in order:
        try:
            if want == "uinput":
                if not is_linux:
                    raise RuntimeError("uinput is Linux-only")
                if not os.path.exists("/dev/uinput"):
                    raise RuntimeError("/dev/uinput missing (try: sudo modprobe uinput)")
                if not os.access("/dev/uinput", os.W_OK):
                    raise RuntimeError("/dev/uinput not writable by this user")
                return UinputBackend(hold_s), notes
            if want == "pynput":
                b = PynputBackend(hold_s)
                if is_linux and wayland:
                    notes.append("[WARN] Wayland session + pynput: XTest injection does "
                                 "NOT reach Wayland-native apps. Presses will be "
                                 "logged but many programs will never see them. "
                                 "Use --key_backend uinput.")
                return b, notes
            if want == "xdotool":
                return XdotoolBackend(hold_s), notes
            if want == "none":
                return NullBackend(hold_s), notes
        except Exception as ex:
            notes.append(f" {want}: unavailable ({ex})")

    notes.append(" falling back to logging only")
    return NullBackend(hold_s), notes


def parse_class_thresholds(spec, classes, default, strict=True):
    """'flexion=0.51,extension=0.96' -> {'flexion':0.51, ...}; rest default.

    strict=False is used for the built-in defaults, which are tuned for one
    particular class set. A model trained with different class names must not
    abort startup over thresholds the user never typed - unknown names are just
    skipped and those classes fall back to --confidence_threshold."""
    out = {c: default for c in classes}
    if not spec:
        return out
    skipped = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"--class_thresholds: expected name=value, got '{part}'")
        name, val = part.split("=", 1)
        name = name.strip()
        if name not in out:
            if strict:
                raise SystemExit(f"--class_thresholds: unknown class '{name}'. "
                                 f"Known: {', '.join(classes)}")
            skipped.append(name)
            continue
        out[name] = float(val)
    if skipped:
        print(f"[NOTE] built-in class thresholds for {skipped} do not apply to this "
              f"model (classes: {', '.join(classes)}); those entries were ignored.")
    return out


def key_thread_main(state, backend, q, press_mode="tap", watchdog_s=0.4,
                    poll_s=0.005):
    """Owns the keyboard. Two modes.

    TAP mode: consumes discrete press events from the queue. One gesture = one
    keystroke of key_hold_ms. Good for menus and discrete actions.

    HOLD mode: reconciles "which key SHOULD be down right now" (state.desired_button,
    written by the model thread) against "which key IS down". This is what a
    driving game needs -- it reads key state per frame, so a 60 ms tap gives one
    frame of steering, while a held key steers continuously.

    Either way the send happens off the inference thread, so a blocking backend
    never stalls prediction.
    """
    held = None
    try:
        while state.running:
            if press_mode == "tap":
                try:
                    button = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                t0 = time.perf_counter()
                try:
                    backend.tap(button)
                except Exception as ex:
                    print(f"\r\033[K[ERROR] key send failed ({button}): {ex}")
                    continue
                state.key_ms = (time.perf_counter() - t0) * 1000.0
                print(f"\r\033[K[KEY] {button.upper():5s} tap "
                      f"[{backend.name.split()[0]}, hold {backend.hold_s*1000:.0f}ms, "
                      f"took {state.key_ms:.1f}ms]"
                      + (f" dropped:{state.dropped_keys}" if state.dropped_keys else ""))
                continue

            # ---- hold mode ----
            desired = state.desired_button

            # Watchdog. If predictions stop arriving -- BLE dropout, model thread
            # wedged, laptop suspended -- a held key would stay down forever. Any
            # stall longer than watchdog_s releases.
            last = state.last_pred_time
            if desired is not None and last and (time.time() - last) > watchdog_s:
                desired = None
                if held is not None:
                    print(f"\r\033[K[WARN] no prediction for {watchdog_s*1000:.0f}ms "
                          f"-- releasing {held.upper()} (watchdog)")

            if desired != held:
                t0 = time.perf_counter()
                try:
                    if held is not None:
                        backend.key_up(held)
                    if desired is not None:
                        backend.key_down(desired)
                except Exception as ex:
                    print(f"\r\033[K[ERROR] key state change failed: {ex}")
                    time.sleep(poll_s)
                    continue
                state.key_ms = (time.perf_counter() - t0) * 1000.0
                if desired is not None:
                    print(f"\r\033[KDOWN {desired.upper():5s} DOWN (holding)")
                else:
                    print(f"\r\033[KUP {held.upper():5s} UP")
                held = desired
                state.held_button = held
            time.sleep(poll_s)
    finally:
        if held is not None:
            try:
                backend.key_up(held)
            except Exception:
                pass
        state.held_button = None
        release_all(backend)
        backend.close()


# ==============================
# Device resolution
# ==============================
DEFAULT_CHANNELS_PER_DEVICE = 3


def device_name_matches(name):
    """True if `name` is one of our bands.

    Matched case-insensitively on purpose: the firmware advertises
    'NPG-Lite-band-3CH:...' (lowercase b) and casing has flipped between
    firmware builds, which is what silently produced "no devices found" in the
    other scripts.

    The prefix still ends at '-band' deliberately: it must NOT match other
    NPG-Lite boards such as 'NPG-Lite-6CH:...', because
    DEFAULT_CHANNELS_PER_DEVICE above is hardcoded to the band's channel count
    rather than parsed from the advertised name.
    """
    return bool(name) and name.lower().startswith(DEVICE_NAME_PREFIX.lower())

def resolve_all_known_devices(found):
    found_sorted = sorted(found, key=lambda d: d.address.upper())
    resolved = []
    for i, d in enumerate(found_sorted):
        role = f"dev{i + 1}"
        resolved.append((d, role, DEFAULT_CHANNELS_PER_DEVICE))
    return resolved


def s_up(st, exit=False):
    """Signed offset from rest_mean at which the UP zone is entered (or exited)."""
    frac = st.pos_exit_frac if exit else st.pos_enter_frac
    return (1.0 if st.up_is_positive else -1.0) * frac * st.up_span


def s_dn(st, exit=False):
    frac = st.pos_exit_frac if exit else st.pos_enter_frac
    return (-1.0 if st.up_is_positive else 1.0) * frac * st.down_span


# ==============================
# Gesture Controller State
# ==============================
class ControllerState:
    def __init__(self, num_channels, window_size, stride, classes, threshold):
        self.lock = threading.Lock()
        self.num_channels = num_channels
        self.window_size = window_size
        self.stride = stride
        self.classes = classes
        self.threshold = threshold
        self.sampling_rate = None
        
        self.sample_buffer = deque(maxlen=window_size)
        self.new_since_last_window = 0
        
        self.ready_window = None
        self.probs = None
        self.pred_class = "rest"
        self.top_prob = 0.0
        self.running = True
        self.status_msg = "connecting..."
        
        # Calibration state
        self.calibrated = False
        self.calibration_data = {"up": [], "down": [], "rest": []}
        self.up_threshold = None
        self.down_threshold = None
        self.rest_mean = None
        self.movement_threshold = None
        
        # Current accelerometer value
        self.current_accel_y = 0.0
        self.accel_lock = threading.Lock()
        
        # Gesture combination tracking with 300ms debounce
        self.last_gesture_time = 0
        self.debounce_time_ms = 300  # 300ms debounce
        self.last_button_pressed = None
        
        # Button press callback
        self.button_callback = None
        
        # Calibration control
        self.calibrating = False
        self.calibration_phase = None  # "up", "down", or "rest"
        self.samples_needed = 50
        self.collected_samples = 0
        self.calibration_step = 0  # 0=up, 1=down, 2=rest
        
        # Countdown before calibration
        self.countdown_active = False
        self.countdown_value = 2  # 2 seconds countdown
        self.countdown_start_time = 0
        
        # Movement detection with hysteresis
        self.accel_history = deque(maxlen=10)
        self.movement_start_time = 0
        self.is_moving = False
        self.movement_direction = None
        self.consecutive_movement_samples = 0
        self.min_movement_samples = 5  # Need 5 consecutive samples of movement
        self.movement_sustain_s = 0.20  # lower to 0.10 if up/down still feels slow
        self.movement_cooldown = 0  # Cooldown after movement stops

        # --- latency fixes ---
        self.infer_stride = 25          # samples between inferences (25 @500Hz = 50ms)
        self.window_ready = threading.Event()
        self.vote_history = deque(maxlen=3)
        self._movement_result = None    # written only by IMU thread
        self.raw_class = "rest"
        self.same_button_ms = 500       # same button cannot repeat inside this window
        self.require_release = True     # gesture must return to rest before re-firing
        self.last_press_by_button = {}  # button -> ms timestamp of last press
        self.blocked_button = None      # button held down, waiting for a release
        # velocity-based movement detection (replaces absolute-position check)
        self.accel_trace = deque(maxlen=256)   # (timestamp, accel_y)
        self.movement_window_s = 0.25          # look back this far to measure motion
        self.velocity_frac = 0.25              # threshold as fraction of calib range
        self.velocity_threshold = None         # set in calculate_thresholds()
        self.up_is_positive = True  # sign of "up" on this axis
        self.last_delta = 0.0

        # POSITION mode: classify where the arm IS (up / rest / down), then pinch fires.
        self.pinch_mode = "position"  # "position" or "velocity"
        self.pos_state = "rest"  # up | rest | down
        self.pos_value = 0.0               # signed, + = toward up, in raw counts
        self.accel_smooth = None           # EMA smoother
        self.accel_smooth_alpha = 0.4
        self.pos_enter_frac = 0.60         # fraction of calib span to ENTER up/down
        self.pos_exit_frac = 0.35          # fraction to fall back to rest (hysteresis)
        self.up_span = None                # |up_mean - rest_mean|
        self.down_span = None              # |down_mean - rest_mean|
        self.nonrest_since = None          # for the posture-drift warning
        self.blocked_gesture = None        # gesture held down, waiting for release
        self.infer_ms = 0.0
        self.key_ms = 0.0
        self.dropped_keys = 0
        # Per-class confidence thresholds. A single global threshold cannot serve
        # flexion and extension at once: measured on replay_buffer.npz, extension
        # reaches p=0.90 after 45 samples of gesture in the window while flexion
        # needs 80 for the same. With one threshold, flexion sits in the ambiguous
        # band far longer, the vote keeps failing, and you have to hold the gesture.
        self.class_thresholds = {}
        self.active_thr = 0.0
        # hold mode
        self.press_mode = "tap"  # "tap" or "hold"
        self.desired_button = None     # what SHOULD be down now (model thread writes)
        self.held_button = None        # what IS down (key thread writes; display only)
        self.last_pred_time = 0.0      # watchdog feed
        self.hold_votes = 3            # votes needed to KEEP holding (see below)
    
    def set_button_callback(self, callback):
        """Set callback function for button presses"""
        self.button_callback = callback
    
    def is_debounced(self, button=None):
        """Global debounce, plus a longer per-button cooldown when `button` is given.
        The global one stops two DIFFERENT buttons firing back to back; the
        per-button one stops the SAME button repeating."""
        now = time.time() * 1000
        if now - self.last_gesture_time < self.debounce_time_ms:
            return False
        if button is not None:
            last = self.last_press_by_button.get(button, 0.0)
            if now - last < self.same_button_ms:
                return False
        return True

    def update_release_gate(self, current_gesture):
        """Clear the hold-block as soon as the gesture changes.

        Keyed on the GESTURE, not the button. In position mode one pinch can map
        to different buttons as the arm moves, so blocking the button would let a
        single held pinch fire UP and then DOWN. Blocking the gesture gives one
        press per pinch. Must be fed the raw voted gesture every prediction -- a
        debounce-filtered value would read as a release and defeat the gate.
        """
        if self.blocked_gesture is not None and current_gesture != self.blocked_gesture:
            self.blocked_gesture = None
            self.blocked_button = None

    def is_blocked(self, gesture):
        return self.require_release and gesture == self.blocked_gesture
    
    def start_calibration_sequence(self):
        """Start the 3-step calibration sequence with countdown"""
        self.calibrating = True
        self.calibration_step = 0
        self.calibration_data = {"up": [], "down": [], "rest": []}
        self.countdown_active = True
        self.countdown_value = 2
        self.countdown_start_time = time.time()
        
        print("\n" + "="*60)
        print("[CAL] CALIBRATION SEQUENCE STARTING")
        print("="*60)
        print("\nYou will be guided through 3 steps:")
        print(" 1 UP position")
        print(" 2 DOWN position")
        print(" 3 REST position")
        print("\nEach step has a 2-second countdown to prepare.")
        print("Please follow the instructions carefully.\n")
        
        return True
    
    def get_calibration_phase_info(self):
        """Get current calibration phase information"""
        if self.calibration_step == 0:
            return "UP", "UP Hold your hand UP", "Raise your hand above neutral position"
        elif self.calibration_step == 1:
            return "DOWN", "DOWN Hold your hand DOWN", "Lower your hand below neutral position"
        else:
            return "REST", "REST REST your hand", "Keep your hand relaxed and still"
    
    def collect_calibration_sample(self):
        """Collect accelerometer sample during calibration"""
        if not self.calibrating:
            return False
        
        # Check if countdown is active
        if self.countdown_active:
            elapsed = time.time() - self.countdown_start_time
            remaining = max(0, 2 - elapsed)
            
            if remaining > 0:
                phase_name, _, _ = self.get_calibration_phase_info()
                # Show countdown
                countdown_dots = "." * int((2 - remaining) * 5)  # Create animation effect
                print(f"\r[TIME] Get ready for {phase_name}: {remaining:.1f}s {countdown_dots}", end="")
                return False
            else:
                self.countdown_active = False
                phase_name, instruction, detail = self.get_calibration_phase_info()
                print(f"\n[OK] {phase_name} - {instruction}")
                print(f" {detail}")
                print(f"[DATA] Recording {self.samples_needed} samples...")
                self.collected_samples = 0
                return False
        
        with self.accel_lock:
            accel_value = self.current_accel_y
        
        # Determine current phase
        if self.calibration_step == 0:
            phase = "up"
            phase_name = "UP"
        elif self.calibration_step == 1:
            phase = "down"
            phase_name = "DOWN"
        else:
            phase = "rest"
            phase_name = "REST"
        
        self.calibration_data[phase].append(accel_value)
        self.collected_samples += 1
        
        # Show progress every 5 samples
        if self.collected_samples % 5 == 0 or self.collected_samples == self.samples_needed:
            progress = int((self.collected_samples / self.samples_needed) * 100)
            bar_length = 20
            filled = int((progress / 100) * bar_length)
            bar = "#" * filled + "." * (bar_length - filled)
            print(f"\r {phase_name}: [{bar}] {progress}% ({self.collected_samples}/{self.samples_needed})", end="")
        
        if self.collected_samples >= self.samples_needed:
            print(f"\n [OK] {phase_name} calibration complete! ({self.collected_samples} samples)")
            self.calibration_step += 1
            self.collected_samples = 0
            
            if self.calibration_step == 1:
                print("\n" + "-"*40)
                print("Step 2: DOWN position")
                print("-"*40)
                self.countdown_active = True
                self.countdown_value = 2
                self.countdown_start_time = time.time()
                print("[TIME] Get ready for DOWN: 2.0s ...")
                
            elif self.calibration_step == 2:
                print("\n" + "-"*40)
                print("Step 3: REST position")
                print("-"*40)
                self.countdown_active = True
                self.countdown_value = 2
                self.countdown_start_time = time.time()
                print("[TIME] Get ready for REST: 2.0s ...")
                
            else:
                # All calibration complete
                self.calibrating = False
                self.calculate_thresholds()
                return True
        
        return False
    
    def calculate_thresholds(self):
        """Calculate thresholds from calibration data"""
        if len(self.calibration_data["up"]) < self.samples_needed or \
           len(self.calibration_data["down"]) < self.samples_needed or \
           len(self.calibration_data["rest"]) < self.samples_needed:
            print("[ERROR] Calibration incomplete! Please run calibration again.")
            return False
        
        # Median, not mean: one twitch during a 50-sample hold used to drag the
        # mean several hundred counts and shift every threshold derived from it.
        up_mean = float(np.median(self.calibration_data["up"]))
        down_mean = float(np.median(self.calibration_data["down"]))
        self.rest_mean = float(np.median(self.calibration_data["rest"]))
        up_sd = float(np.std(self.calibration_data["up"]))
        down_sd = float(np.std(self.calibration_data["down"]))
        rest_sd = float(np.std(self.calibration_data["rest"]))
        
        # Calculate thresholds (midpoint between means)
        self.up_threshold = (up_mean + self.rest_mean) / 2
        self.down_threshold = (down_mean + self.rest_mean) / 2
        
        # Calculate movement threshold based on up/down difference
        movement_range = abs(up_mean - down_mean)
        self.movement_threshold = movement_range * 0.3  # kept for display only

        # Velocity threshold: how much accel must CHANGE within movement_window_s
        # to count as a deliberate move. Derived from the calibrated sweep range,
        # so it scales with however the band is worn.
        self.velocity_threshold = movement_range * self.velocity_frac
        self.up_is_positive = up_mean > down_mean

        # POSITION mode spans, measured from rest toward each extreme
        self.up_span = abs(up_mean - self.rest_mean)
        self.down_span = abs(down_mean - self.rest_mean)
        self.pos_state = "rest"
        self.accel_smooth = None
        
        self.calibrated = True
        self.status_msg = "Calibrated successfully!"
        
        print("\n" + "="*50)
        print("[OK] CALIBRATION COMPLETE!")
        print("="*50)
        print(f" UP mean: {up_mean:.2f}")
        print(f" REST mean: {self.rest_mean:.2f}")
        print(f" DOWN mean: {down_mean:.2f}")
        print(f" Range: {movement_range:.0f} | 'up' is "
              f"{'increasing' if self.up_is_positive else 'decreasing'} accel_y")
        print(f" spread (std): up {up_sd:.0f} rest {rest_sd:.0f} down {down_sd:.0f}")
        print(f" up span {self.up_span:.0f} | down span {self.down_span:.0f}")
        if self.pinch_mode == "position":
            print(f" UP when accel passes {self.rest_mean + s_up(self):.0f} "
                  f"(back to rest below {self.rest_mean + s_up(self, exit=True):.0f})")
            print(f" DOWN when accel passes {self.rest_mean + s_dn(self):.0f} "
                  f"(back to rest above {self.rest_mean + s_dn(self, exit=True):.0f})")
        else:
            print(f" Velocity threshold: {self.velocity_threshold:.0f} "
                  f"per {self.movement_window_s*1000:.0f} ms")
        print("="*50)

        # Sanity checks. A bad calibration here is what produced the original
        # "DOWN armed while stationary" behaviour, so say so loudly now instead.
        lo, hi = min(up_mean, down_mean), max(up_mean, down_mean)
        if not (lo < self.rest_mean < hi):
            print("[WARN] WARNING: REST is NOT between UP and DOWN. Your rest pose "
                  "overlaps an extreme -- pinch will fire that direction while idle.")
            print(" Recalibrate, holding REST in the exact pose you actually "
                  "play in.")
        if min(self.up_span, self.down_span) < 3 * max(up_sd, down_sd, rest_sd):
            print("[WARN] WARNING: poses are barely separated relative to how much "
                  "you wobbled. Hold each position still, or move further.")
        if min(self.up_span, self.down_span) < 500:
            print("[WARN] WARNING: one span is tiny -- up/down will be unreliable.")
        print("\n[GAME] System ready for gesture control!")
        print(f"[TIME] {self.debounce_time_ms}ms debounce time between button presses")
        if self.pinch_mode == "position":
            print("[NOTE] Hold the arm UP or DOWN, then PINCH to fire that direction.")
            print(" Pinching at REST does nothing.")
        else:
            print("[NOTE] Remember: Up/Down buttons require PINCH + MOVEMENT together!")
        print("\nPress Ctrl+C to stop\n")
        return True
    
    def _update_movement(self, current_accel):
        """Velocity-based movement detection. MUTATES state, so it is called from
        exactly ONE place: update_accel(), i.e. the IMU notify callback.

        The previous version compared the instantaneous reading against rest_mean
        captured once at calibration. That is a POSITION test, not a movement
        test: if the arm settles anywhere other than the calibrated rest pose,
        one direction stays permanently armed and every pinch fires it. Measured
        on a real session -- calibrated rest -5167, actual resting accel ~-8600 --
        6 of 13 stationary readings satisfied "moving down".

        This version measures how much accel_y CHANGED over the last
        movement_window_s. Stationary means zero change at any pose, so posture
        drift and gravity offset no longer matter and rest_mean is not consulted.
        """
        now = time.time()
        self.accel_trace.append((now, current_accel))

        if self.velocity_threshold is None:
            self._movement_result = None
            return

        if self.movement_cooldown > 0:
            self.movement_cooldown -= 1
            self._movement_result = None
            return

        # newest sample that is at least movement_window_s old (deque is oldest-first)
        ref_val = None
        for t, v in self.accel_trace:
            if now - t >= self.movement_window_s:
                ref_val = v
            else:
                break

        if ref_val is None:          # not enough history spanning the window yet
            self._movement_result = None
            return

        delta = current_accel - ref_val
        self.last_delta = delta

        if abs(delta) < self.velocity_threshold:
            self.consecutive_movement_samples = 0
            self.is_moving = False
            self.movement_direction = None
            self._movement_result = None
            return

        moving_positive = delta > 0
        direction = "up" if (moving_positive == self.up_is_positive) else "down"

        if self.movement_direction != direction:
            self.movement_direction = direction
            self.movement_start_time = now

        self.is_moving = True
        if now - self.movement_start_time > self.movement_sustain_s:
            self._movement_result = direction
        else:
            self._movement_result = None

    def _update_position(self, current_accel):
        """Classify WHERE the arm is: up / rest / down. MUTATES state, so it is
        called from exactly ONE place: update_accel().

        Hysteresis is the whole point. A single threshold at the midpoint chatters
        while you hold near it, and every chatter looks like a fresh release to the
        gating layer. Entering a zone needs pos_enter_frac of the calibrated span;
        leaving it only happens below pos_exit_frac. Values are smoothed first so
        IMU noise cannot flip the state on one sample.
        """
        if self.up_span is None or self.down_span is None:
            return

        if self.accel_smooth is None:
            self.accel_smooth = current_accel
        else:
            self.accel_smooth += self.accel_smooth_alpha * (current_accel - self.accel_smooth)

        sign = 1.0 if self.up_is_positive else -1.0
        # v > 0 means "toward up", regardless of which way the axis points
        v = sign * (self.accel_smooth - self.rest_mean)
        self.pos_value = v

        up_enter = self.pos_enter_frac * self.up_span
        up_exit = self.pos_exit_frac * self.up_span
        down_enter = self.pos_enter_frac * self.down_span
        down_exit = self.pos_exit_frac * self.down_span

        if v >= up_enter:
            self.pos_state = "up"
        elif v <= -down_enter:
            self.pos_state = "down"
        elif self.pos_state == "up" and v < up_exit:
            self.pos_state = "rest"
        elif self.pos_state == "down" and v > -down_exit:
            self.pos_state = "rest"

        # posture-drift tracking: if we never come back to rest, the calibration
        # no longer matches how the arm is actually being held.
        if self.pos_state == "rest":
            self.nonrest_since = None
        elif self.nonrest_since is None:
            self.nonrest_since = time.time()

    def get_position(self):
        """Pure read. Safe from any thread."""
        return self.pos_state

    def get_movement(self):
        """Pure read. Safe to call from any thread, any rate."""
        return self._movement_result

    def map_gesture_to_button(self, current_gesture, accel_value):
        """PURE mapping: gesture -> button, or None. No debounce, no gating.
        Gating happens in the model thread so that the release gate can see the
        raw, unfiltered mapping every prediction."""
        if not self.calibrated:
            return None

        movement = self.get_movement()          # read-only, no side effects

        if current_gesture == "pinch":
            if self.pinch_mode == "position":
                # Where the arm IS decides the direction; the pinch is the trigger.
                # pos_state is hysteretic, so it does not chatter at the boundary.
                pos = self.get_position()
                return pos if pos in ("up", "down") else None
            # velocity mode: the pinch must coincide with actual motion
            if movement in ("up", "down"):
                return movement
            return None

        if current_gesture == "flexion":
            return "left"
        if current_gesture == "extension":
            return "right"
        return None

    def update_accel(self, accel_y):
        """Called from the IMU notify handler. Sole driver of the movement
        state machine."""
        with self.accel_lock:
            self.current_accel_y = accel_y
        self._update_position(accel_y)
        self._update_movement(accel_y)

    def record_button_press(self, button, gesture=None):
        """Record a button press for debounce tracking"""
        now = time.time() * 1000
        self.last_gesture_time = now
        self.last_press_by_button[button] = now
        self.last_button_pressed = button
        self.blocked_button = button
        self.blocked_gesture = gesture   # held: no re-fire until the gesture changes


# ==============================
# BLE handling
# ==============================
def ble_thread_main(state):
    def fail(msg):
        """Single place where a BLE problem becomes visible to the rest of the app.
        Previously an exception anywhere in run() escaped asyncio.run() inside this
        daemon thread: the thread died, state.running stayed True, status_msg stayed
        'connecting...', and the main loop printed 'Waiting for accelerometer
        data...' forever with no error and no exit."""
        state.status_msg = f"ERROR: {msg}"
        state.running = False
        print(f"\r\033[K[ERROR] BLE: {msg}")

    async def run():
        state.status_msg = f"scanning for {DEVICE_NAME_PREFIX}*..."
        print("[SCAN] Scanning for NPG-Lite devices...")
        devices = await BleakScanner.discover(timeout=6)
        found = [d for d in devices if device_name_matches(d.name)]
        if not found:
            state.status_msg = "ERROR: no NPG-Lite device found"
            state.running = False
            print("[ERROR] ERROR: No NPG-Lite device found. Make sure your device is powered on and in range.")
            return
        
        print(f"[OK] Found {len(found)} device(s): {[d.name for d in found]}")
        resolved = resolve_all_known_devices(found)
        if not resolved:
            fail("could not resolve any discovered devices")
            return

        total_emg_channels = sum(ch for _, _, ch in resolved)
        total_channels = total_emg_channels + 3 * len(resolved)
        if total_channels != state.num_channels:
            roles_found = [role for _, role, _ in resolved]
            fail(f"connected devices {roles_found} provide {total_emg_channels}ch EMG "
                 f"+ {3 * len(resolved)}ch accel = {total_channels}ch total, but the "
                 f"model expects {state.num_channels}ch. Discovery accepts every board "
                 f"whose name starts with '{DEVICE_NAME_PREFIX}', so a second powered-on "
                 f"band doubles the channel count - power the extra board off and re-run.")
            return
        
        role_order = [role for _, role, _ in resolved]
        
        # Build filters
        per_device_filters = {}
        for _, role, channels in resolved:
            notch_filters, emg_filters = [], []
            for _ in range(channels):
                nf = NotchFilter()
                nf.set_sampling_rate(state.sampling_rate)
                notch_filters.append(nf)
                ef = EXGFilter()
                ef.set_bits(ADC_BITS, state.sampling_rate)
                emg_filters.append(ef)
            per_device_filters[role] = {"notch": notch_filters, "emg": emg_filters}
        
        device_queues = {role: deque() for role in role_order}
        queue_lock = threading.Lock()
        
        # Accel state
        latest_accel = {role: np.zeros(3, dtype=np.float32) for role in role_order}
        accel_lock = threading.Lock()
        
        def make_imu_handler(role):
            def handle_imu_notify(_, data: bytearray):
                if len(data) == 0 or len(data) % IMU_SAMPLE_LEN != 0:
                    return
                n_samples = len(data) // IMU_SAMPLE_LEN
                offset = (n_samples - 1) * IMU_SAMPLE_LEN
                ax = int.from_bytes(data[offset + 1:offset + 3], "big", signed=True)
                ay = int.from_bytes(data[offset + 3:offset + 5], "big", signed=True)
                az = int.from_bytes(data[offset + 5:offset + 7], "big", signed=True)
                with accel_lock:
                    latest_accel[role][0] = ax
                    latest_accel[role][1] = ay
                    latest_accel[role][2] = az
                    # Update state with accelerometer Y value from first device
                    if role == role_order[0]:
                        state.update_accel(ay)
            return handle_imu_notify
        
        def make_handler(role, channels, notch_filters, emg_filters):
            def handle_notify(_, data: bytearray):
                expected_len = SAMPLES_PER_PACKET * (1 + channels * 2)
                if len(data) != expected_len:
                    return
                offset = 0
                for _ in range(SAMPLES_PER_PACKET):
                    sample = np.empty(channels, dtype=np.float32)
                    for ch in range(channels):
                        raw_val = (data[offset + 1 + ch * 2] << 8) | data[offset + 2 + ch * 2]
                        notched = notch_filters[ch].process(raw_val, NOTCH_TYPE)
                        filtered = emg_filters[ch].process(notched, 4)
                        sample[ch] = filtered
                    offset += 1 + channels * 2
                    with queue_lock:
                        device_queues[role].append(sample)
            return handle_notify
        
        # `clients` is appended to as soon as each connect succeeds, and the
        # whole block below sits inside one try/finally, so a failure partway
        # through a multi-board connect still tears down the boards that DID
        # come up instead of leaving them connected and streaming.
        clients = []
        try:
            for dev, role, channels in resolved:
                state.status_msg = f"connecting to {role} ({dev.address})..."
                print(f"[LINK] Connecting to {role} ({dev.address})...")
                client = BleakClient(dev.address)
                try:
                    await asyncio.wait_for(client.connect(), timeout=15.0)
                except asyncio.TimeoutError:
                    fail(f"timed out connecting to {role} ({dev.address}) after 15s")
                    return
                except Exception as ex:
                    fail(f"could not connect to {role} ({dev.address}): {ex}")
                    return
                clients.append(client)
                filters = per_device_filters[role]
                try:
                    await client.start_notify(
                        DATA_CHAR_UUID,
                        make_handler(role, channels, filters["notch"], filters["emg"]),
                    )
                except Exception as ex:
                    fail(f"could not subscribe to EMG data on {role}: {ex}")
                    return
                try:
                    await client.start_notify(IMU_CHAR_UUID, make_imu_handler(role))
                    print(f"[IMU] IMU notifications started for {role}")
                except Exception as e:
                    print(f"[WARN] (warning) IMU characteristic not available for {role} ({e})")
            state.status_msg = f"connected: {'+'.join(role_order)}"
            print(f"[OK] Connected to {len(clients)} device(s): {role_order}")

            try:
                await asyncio.gather(*(c.write_gatt_char(CONTROL_CHAR_UUID, b"STOP", response=True)
                                       for c in clients))
                await asyncio.sleep(0.1)
                await asyncio.gather(*(c.write_gatt_char(CONTROL_CHAR_UUID, b"START", response=True)
                                       for c in clients))
            except Exception as ex:
                fail(f"could not start streaming: {ex}")
                return
            state.status_msg = f"streaming: {'+'.join(role_order)}"
            print("[DATA] Streaming started. Ready for calibration and gestures!\n")

            await stream_loop(clients, resolved, role_order, device_queues,
                              queue_lock, latest_accel, accel_lock)
        finally:
            await shutdown(clients)

    async def stream_loop(clients, resolved, role_order, device_queues,
                          queue_lock, latest_accel, accel_lock):
        last_link_check = time.time()
        while state.running:
            # A dropped device leaves its queue permanently empty, so the
            # all-queues-non-empty condition below never fires and this loop
            # spins forever producing no windows. Poll the link instead.
            if time.time() - last_link_check > 1.0:
                last_link_check = time.time()
                for c, (_, role, _) in zip(clients, resolved):
                    if not c.is_connected:
                        fail(f"device {role} disconnected while streaming")
                        return

            merged_sample = None
            with queue_lock:
                if all(device_queues[r] for r in role_order):
                    parts = [device_queues[r].popleft() for r in role_order]
                    with accel_lock:
                        accel_parts = [latest_accel[r].copy() for r in role_order]
                    merged_sample = np.concatenate(parts + accel_parts)
            
            if merged_sample is None:
                await asyncio.sleep(0.001)
                continue
            
            emit = False
            with state.lock:
                state.sample_buffer.append(merged_sample)
                state.new_since_last_window += 1
                if (len(state.sample_buffer) == state.window_size
                        and state.new_since_last_window >= state.infer_stride):
                    state.ready_window = np.array(state.sample_buffer, dtype=np.float32)
                    state.new_since_last_window = 0
                    emit = True
            if emit:
                state.window_ready.set()
            # NOTE: the gesture/button check used to live here, running ~500x/sec.
            # It now runs in the model thread, once per new prediction.

    async def shutdown(clients):
        """Best-effort teardown, individually timed out per step. This used to
        sit at the end of run() outside any finally:, so every early `return`
        (dropped link, channel mismatch) skipped it entirely and left the board
        connected and streaming."""
        async def attempt(coro):
            try:
                await asyncio.wait_for(coro, timeout=DISCONNECT_TIMEOUT_S)
            except Exception:
                pass  # already gone / unresponsive

        for c in clients:
            await attempt(c.write_gatt_char(CONTROL_CHAR_UUID, b"STOP", response=True))
            await attempt(c.stop_notify(DATA_CHAR_UUID))
            await attempt(c.stop_notify(IMU_CHAR_UUID))
            await attempt(c.disconnect())

    
    try:
        asyncio.run(run())
    except Exception as ex:
        fail(f"BLE thread crashed: {ex}")


# ==============================
# Model inference thread
# ==============================
class FastPredictor:
    """Traced-graph wrapper. Measured on this model, window (1,250,6), CPU:
        model.predict(x, verbose=0)   ~63 ms mean, 204 ms max
        model(x, training=False)     ~253 ms mean   <-- worse, do not use
        this class                   ~4.2 ms mean, 5.5 ms max
    Normalisation is folded into the graph so it also runs in C++."""

    def __init__(self, model, mean, std, window_size, num_channels):
        mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
        std = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
        std = np.where(std < 1e-8, 1.0, std)   # guard zero-variance channel
        self._mean = tf.constant(mean)
        self._std = tf.constant(std)
        self._model = model

        sig = [tf.TensorSpec([1, window_size, num_channels], tf.float32)]

        @tf.function(input_signature=sig, reduce_retracing=True)
        def _graph(w):
            return self._model((w - self._mean) / self._std, training=False)

        self._graph = _graph
        # trace now, at startup, not on the first gesture
        self._graph(tf.zeros([1, window_size, num_channels], tf.float32))

    def __call__(self, window):
        x = tf.convert_to_tensor(window[np.newaxis, ...], dtype=tf.float32)
        return self._graph(x)[0].numpy()


def model_thread_main(state, predictor, threshold, votes_needed=2):
    """Blocks on an event instead of polling. Runs the vote, then does the
    button check once per new prediction."""
    consecutive_errors = 0
    while state.running:
        if not state.window_ready.wait(timeout=0.2):
            continue
        state.window_ready.clear()

        with state.lock:
            window = state.ready_window
            state.ready_window = None
        if window is None:
            continue

        t0 = time.perf_counter()
        try:
            probs = predictor(window)
        except Exception as ex:
            # Without this the daemon thread dies silently and the status line
            # freezes on the last prediction forever, with no error anywhere.
            consecutive_errors += 1
            state.status_msg = f"ERROR: inference failed ({ex})"
            print(f"\r\033[K[ERROR] inference failed ({consecutive_errors}): {ex}")
            if consecutive_errors >= 5:
                print("[ERROR] inference failed 5x in a row -- stopping.")
                state.running = False
            time.sleep(0.05)
            continue
        consecutive_errors = 0
        infer_ms = (time.perf_counter() - t0) * 1000.0

        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        label = state.classes[top_idx]
        # Threshold belongs to the class being proposed, not to the pipeline.
        thr = state.class_thresholds.get(label, threshold)
        state.active_thr = thr
        candidate = label if top_prob >= thr else "rest"

        state.vote_history.append(candidate)
        prev = state.pred_class
        voted = "rest"
        if len(state.vote_history) == state.vote_history.maxlen:
            counts = {}
            for lbl in state.vote_history:
                counts[lbl] = counts.get(lbl, 0) + 1
            best = max(counts, key=counts.get)
            # Asymmetric vote. Entering a gesture needs votes_needed; STAYING in
            # one only needs hold_votes (<= votes_needed). Without this, a single
            # dropped window mid-gesture drops the vote below threshold and the
            # held key flickers -- which in a driving game is steering stutter.
            # hold_votes is also what sets release latency: the key lifts as soon
            # as fewer than hold_votes of the last `votes` windows are the gesture.
            if prev != "rest" and counts.get(prev, 0) >= state.hold_votes:
                voted = prev
            elif counts[best] >= votes_needed:
                voted = best

        with state.lock:
            state.probs = probs
            state.top_prob = top_prob
            state.raw_class = candidate
            state.pred_class = voted
            state.infer_ms = infer_ms

        state.last_pred_time = time.time()   # feeds the hold-mode watchdog

        # button check: once per prediction (~20/sec), not per sample (~500/sec)
        if state.calibrated:
            with state.accel_lock:
                accel_val = state.current_accel_y

            button = state.map_gesture_to_button(voted, accel_val)

            if state.press_mode == "hold":
                # Publish intent only. No debounce, no release gate, no per-button
                # cooldown: those exist to stop ONE gesture producing repeated
                # discrete taps, which is the opposite of what hold mode wants.
                state.desired_button = button
            elif state.button_callback:
                # Order matters. Feed the gate the RAW voted gesture every
                # prediction, before any debounce filtering -- otherwise a
                # debounce-suppressed None reads as "released" and re-arms
                # instantly, which is exactly the auto-repeat we are killing.
                state.update_release_gate(voted)
                if button and not state.is_blocked(voted) and state.is_debounced(button):
                    state.button_callback(button)
                    state.record_button_press(button, voted)


# ==============================
# Main function
# ==============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="gesture_model")
    ap.add_argument("--confidence_threshold", type=float, default=0.80,
                   help="minimum top-class probability per window (default 0.80; a "
                        "majority vote over --votes windows provides the stability "
                        "that the old 0.95 threshold was doing by itself)")
    ap.add_argument("--debounce_ms", type=int, default=300,
                   help="debounce time in ms between button presses (default: 300ms)")
    ap.add_argument("--infer_stride", type=int, default=25,
                   help="samples between inferences. 25 @ 500Hz = 50ms = 20 predictions/sec. "
                        "This is INDEPENDENT of the training stride in meta.json.")
    ap.add_argument("--class_thresholds", default=None,
                   help="per-class confidence thresholds, e.g. "
                        "'flexion=0.51,extension=0.96'. Defaults were derived from "
                        "replay_buffer.npz with tune_thresholds.py: they equalise "
                        "onset latency across classes. tune_thresholds.py suggests "
                        "flexion=0.39; the default is 0.50 because the "
                        "false-positive headroom it measures comes from cleanly "
                        "recorded held gestures, and live rest is noisier. Raise "
                        "flexion if you see spurious LEFT presses, lower it toward "
                        "0.39 for maximum speed. RE-DERIVE AFTER RETRAINING -- these "
                        "are properties of one model. Any class not listed falls "
                        f"back to --confidence_threshold. Default: "
                        f"'{DEFAULT_CLASS_THRESHOLDS}'.")
    ap.add_argument("--votes", type=int, default=5,
                   help="window vote buffer size (default 5 = 250ms @ 20/sec)")
    ap.add_argument("--votes_needed", type=int, default=4,
                   help="votes required to accept a label (default 4 of 5; "
                        "2-of-3 let 100ms model blips through as real presses)")
    ap.add_argument("--pinch_mode", choices=["position", "velocity"], default="position",
                   help="position: pinch fires whichever zone the arm is IN "
                        "(up/down; nothing at rest). velocity: pinch must coincide "
                        "with actual motion. Default: position.")
    ap.add_argument("--pos_enter_frac", type=float, default=0.60,
                   help="fraction of the calibrated span needed to ENTER up/down")
    ap.add_argument("--pos_exit_frac", type=float, default=0.35,
                   help="fraction to fall back to rest (hysteresis; must be < enter)")
    ap.add_argument("--velocity_frac", type=float, default=0.25,
                   help="movement threshold as a fraction of the calibrated "
                        "up-down range, measured over --movement_window_ms")
    ap.add_argument("--movement_window_ms", type=int, default=250,
                   help="look-back window for measuring accel change (default 250ms)")
    ap.add_argument("--key_backend", choices=["auto","uinput","pynput","xdotool","none"],
                   default="auto",
                   help="how to inject keystrokes. auto = uinput, then pynput, then "
                        "xdotool on Linux; pynput elsewhere. uinput is kernel-level "
                        "and works on X11, Wayland and for raw-evdev readers.")
    ap.add_argument("--press_mode", choices=["tap", "hold"], default="tap",
                   help="tap: one gesture = one keystroke (menus, discrete actions). "
                        "hold: the key stays DOWN for as long as the gesture is "
                        "held, and lifts when you relax. Use hold for driving/racing "
                        "games -- they read key state per frame, so a 60ms tap gives "
                        "one frame of steering.")
    ap.add_argument("--hold_votes", type=int, default=3,
                   help="votes (out of --votes) needed to KEEP holding a gesture, "
                        "vs --votes_needed to start it. Lower = stickier hold, "
                        "tolerates dropped windows, slower release. Higher = faster "
                        "release, more prone to stutter. Only used in hold mode.")
    ap.add_argument("--key_watchdog_ms", type=int, default=400,
                   help="in hold mode, release everything if no prediction arrives "
                        "for this long (BLE dropout, wedged thread). Prevents a key "
                        "staying down forever.")
    ap.add_argument("--key_hold_ms", type=int, default=60,
                   help="how long to hold each key DOWN (default 60ms). The original "
                        "code held it <1ms, which anything polling key state at "
                        "60fps (16.7ms window) will miss entirely. Raise to 80-100 "
                        "if a game still misses presses.")
    ap.add_argument("--dry_run", action="store_true",
                   help="detect and log presses but do NOT send keystrokes. "
                        "Use this while tuning so stray presses do not hit whatever "
                        "window has focus.")
    ap.add_argument("--same_button_ms", type=int, default=500,
                   help="same button cannot repeat within this many ms (default 500)")
    ap.add_argument("--allow_hold_repeat", action="store_true",
                   help="allow a held gesture to auto-repeat at --same_button_ms. "
                        "Off by default: one gesture = one press, and you must "
                        "return to rest before that button fires again.")
    ap.add_argument("--movement_sustain_ms", type=int, default=120,
                   help="how long accel movement must be sustained for up/down (default 120ms)")
    args = ap.parse_args()
    
    with open(os.path.join(args.model_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    
    window_size = meta["window_size"]
    stride = meta["stride"]
    num_channels = meta["num_channels"]
    classes = meta["classes"]
    mean = np.array(meta["scaler_mean"], dtype=np.float32)
    std = np.array(meta["scaler_std"], dtype=np.float32)
    
    if meta["sampling_rate"] not in SUPPORTED_SAMPLING_RATES:
        raise SystemExit(
            f"[ERROR] meta.json sampling_rate={meta['sampling_rate']} Hz is not supported.\n"
            " NotchFilter/EXGFilter only have coefficients for "
            f"{SUPPORTED_SAMPLING_RATES} Hz. At any other rate the EMG filters "
            "silently emit zeros and every prediction is meaningless while still "
            "looking confident. Retrain at a supported rate, or add coefficients.")

    print(f"[LOAD] Loading model from {args.model_dir}...")
    model = keras.models.load_model(os.path.join(args.model_dir, "model.keras"))
    
    state = ControllerState(num_channels, window_size, stride, classes, args.confidence_threshold)
    state.sampling_rate = meta["sampling_rate"]
    state.debounce_time_ms = args.debounce_ms
    state.infer_stride = args.infer_stride
    state.vote_history = deque(maxlen=args.votes)
    state.movement_sustain_s = args.movement_sustain_ms / 1000.0
    state.same_button_ms = args.same_button_ms
    # Only names the USER typed are hard errors. The built-in defaults are tuned
    # for one particular class set, so a model trained with different classes
    # must not abort startup over a flag that was never passed.
    user_supplied = args.class_thresholds is not None
    state.class_thresholds = parse_class_thresholds(
        args.class_thresholds if user_supplied else DEFAULT_CLASS_THRESHOLDS,
        classes, args.confidence_threshold, strict=user_supplied)
    state.velocity_frac = args.velocity_frac
    state.pinch_mode = args.pinch_mode
    state.pos_enter_frac = args.pos_enter_frac
    state.pos_exit_frac = args.pos_exit_frac
    if state.pos_exit_frac >= state.pos_enter_frac:
        print("[WARN] pos_exit_frac >= pos_enter_frac disables hysteresis; clamping.")
        state.pos_exit_frac = state.pos_enter_frac * 0.6
    state.movement_window_s = args.movement_window_ms / 1000.0
    state.require_release = not args.allow_hold_repeat
    state.press_mode = args.press_mode
    state.hold_votes = max(1, min(args.hold_votes, args.votes_needed))
    if state.hold_votes != args.hold_votes:
        print(f"[WARN] hold_votes clamped to {state.hold_votes} "
              f"(must be between 1 and votes_needed={args.votes_needed})")

    latency_ms = 1000.0 * args.infer_stride / meta["sampling_rate"]
    print(f"[CFG] window {window_size} samples "
          f"({1000.0*window_size/meta['sampling_rate']:.0f} ms) | "
          f"inference every {args.infer_stride} samples ({latency_ms:.0f} ms) | "
          f"training stride in meta.json was {stride} -- deliberately NOT reused")
    
    # ---- keyboard backend ----
    print(f"[SYS] {describe_session()}")
    hold_s = max(0.0, args.key_hold_ms / 1000.0)
    if args.dry_run:
        key_backend, backend_notes = NullBackend(hold_s), [" (dry run: no keystrokes sent)"]
        key_backend.name = "none (dry run)"
    else:
        key_backend, backend_notes = build_key_backend(args.key_backend, hold_s)
    for n in backend_notes:
        print(n)
    print(f"[KBD] key backend: {key_backend.name} | hold {args.key_hold_ms}ms")
    if isinstance(key_backend, NullBackend) and not args.dry_run:
        print(" No working injection backend. To enable uinput on Linux:")
        print(" sudo modprobe uinput")
        print(" sudo usermod -aG input $USER # then log out and back in")
        print(" echo 'KERNEL==\"uinput\", GROUP=\"input\", MODE=\"0660\", "
              "OPTIONS+=\"static_node=uinput\"' | sudo tee /etc/udev/rules.d/99-uinput.rules")
        print(" sudo udevadm control --reload-rules && sudo udevadm trigger")

    # Bounded queue: a tap now blocks for key_hold_ms, so it must not run on the
    # model thread. Dropping a stale keystroke beats building a backlog.
    key_queue = queue.Queue(maxsize=4)

    def on_button_press(button):
        """Runs on the model thread. Must never block -- just enqueue."""
        try:
            key_queue.put_nowait(button)
        except queue.Full:
            state.dropped_keys += 1

    key_thread = threading.Thread(
        target=key_thread_main,
        args=(state, key_backend, key_queue, args.press_mode,
              args.key_watchdog_ms / 1000.0),
        daemon=True)
    key_thread.start()

    state.set_button_callback(on_button_press)
    
    # Start BLE thread
    ble_thread = threading.Thread(target=ble_thread_main, args=(state,), daemon=True)
    ble_thread.start()
    
    # Build traced predictor (compiles the graph now, so the first gesture
    # is not slowed down by tracing)
    print("[INIT] Tracing inference graph...")
    predictor = FastPredictor(model, mean, std, window_size, num_channels)
    print("[OK] Inference graph ready")

    # Start model thread
    threading.Thread(target=model_thread_main,
                    args=(state, predictor, args.confidence_threshold, args.votes_needed),
                    daemon=True).start()
    
    print("\n" + "="*60)
    print("[GAME] GESTURE CONTROLLER")
    print("="*60)
    print("\nControls:")
    if args.pinch_mode == "position":
        print(" PINCH Arm held UP + PINCH -> UP button")
        print(" PINCH Arm held DOWN + PINCH -> DOWN button")
        print(" PINCH Arm at REST + PINCH -> nothing")
    else:
        print(" PINCH PINCH + Moving UP -> UP button")
        print(" PINCH PINCH + Moving DOWN -> DOWN button")
    print(" EMG FLEXION -> LEFT button")
    print(" EMG EXTENSION -> RIGHT button")
    print()
    if args.press_mode == "hold":
        print("[CAR] " + "="*56)
        print("[CAR] PRESS MODE: HOLD -- key stays DOWN while the gesture is held")
        print(f"[CAR] enter {args.votes_needed}/{args.votes} votes | keep {state.hold_votes}/{args.votes} | "
              f"watchdog {args.key_watchdog_ms}ms")
        print("[CAR] " + "="*56)
    else:
        print("[TIME] " + "="*56)
        print(f"[TIME] PRESS MODE: TAP -- one gesture = one {args.key_hold_ms}ms keystroke")
        print(f"[TIME] debounce {state.debounce_time_ms}ms | same button {args.same_button_ms}ms")
        print("[TIME] For a driving/racing game you want: --press_mode hold")
        print("[TIME] " + "="*56)
    print(f"[KEY] vote {args.votes_needed}-of-{args.votes} | thresholds: "
          + " ".join(f"{c}={state.class_thresholds[c]:.2f}" for c in classes))
    if args.pinch_mode == "position":
        print("\n[WARN] Up/Down come from ARM POSITION. Pinch only triggers.")
    else:
        print("\n[WARN] IMPORTANT: Up/Down buttons require BOTH pinch AND movement!")
    if args.pinch_mode != "position":
        print(" Static pinch alone will NOT trigger up/down buttons.")
    print("\n" + "="*60)
    print("\nMOVING Starting calibration sequence automatically...\n")
    
    try:
        # Start calibration automatically
        calibrating = False
        drift_warned = 0.0
        
        waiting_since = time.time()
        while state.running:
            # Check if we have accelerometer data
            if state.current_accel_y == 0:
                waited = time.time() - waiting_since
                if waited > ACCEL_WAIT_TIMEOUT_S:
                    # The BLE thread treats a missing IMU characteristic as
                    # non-fatal (it warns and keeps streaming EMG), so without
                    # this timeout the loop printed "Waiting..." forever and the
                    # only way out was Ctrl+C.
                    print(f"\r\033[K[ERROR] No accelerometer data after "
                          f"{ACCEL_WAIT_TIMEOUT_S:.0f}s.")
                    print(" Up/Down need the IMU, and calibration cannot start "
                          "without it. Check that:")
                    print("  - the band is powered on and the firmware exposes the "
                          "IMU characteristic")
                    print("  - the earlier '(warning) IMU characteristic not "
                          "available' line did not appear above")
                    print(" Flexion/extension (LEFT/RIGHT) do not need the IMU - "
                          "reflash or use a build with IMU support to get UP/DOWN.")
                    state.running = False
                    break
                print(f"\r[WAIT] Waiting for accelerometer data... "
                      f"({ACCEL_WAIT_TIMEOUT_S - waited:.0f}s)", end="")
                time.sleep(1)
                continue
            waiting_since = time.time()
            
            # If not calibrated, prompt for calibration
            if not state.calibrated:
                if not calibrating:
                    state.start_calibration_sequence()
                    calibrating = True
                
                # Collect calibration samples
                if state.calibrating:
                    state.collect_calibration_sample()
                    
                    # If calibration just completed
                    if state.calibrated:
                        calibrating = False
                
                time.sleep(0.05)
            else:
                # System is calibrated - show status with movement detection
                with state.accel_lock:
                    accel_val = state.current_accel_y
                
                # Calculate time since last button press
                current_time = time.time() * 1000
                time_since_last = current_time - state.last_gesture_time
                debounce_remaining = max(0, state.debounce_time_ms - time_since_last)
                
                # Read-only. Must NOT be detect_movement(): that used to mutate
                # the state machine from this thread and cancel the sustain timer.
                movement = state.get_movement()
                is_pinch = state.pred_class == "pinch"

                # posture drift: calibrated rest no longer matches how the arm is
                # actually held, which is exactly what silently armed DOWN before.
                if (state.nonrest_since is not None
                        and time.time() - state.nonrest_since > 8.0
                        and time.time() - drift_warned > 20.0):
                    drift_warned = time.time()
                    print(f"\r\033[K[WARN] arm has read {state.pos_state.upper()} for "
                          "8s+ without returning to REST. Calibrated rest is "
                          f"{state.rest_mean:.0f}, you are at "
                          f"{state.accel_smooth:.0f}. Pinch will fire "
                          f"{state.pos_state.upper()} while idle -- recalibrate.")
                
                # Build status string
                status_parts = [
                    "HOLD" if args.press_mode == "hold" else "TAP ",
                    f"{state.pred_class.upper():9s}",
                    f"raw:{state.raw_class[:4]:4s}",
                    f"p={state.top_prob:.2f}/{state.active_thr:.2f}",
                    f"{state.infer_ms:4.0f}ms",
                    f"acc:{accel_val:8.0f}",
                    f"pos:{state.pos_state.upper():5s}",
                    f"v:{state.pos_value:+7.0f}",
                    f"key:{state.key_ms:4.0f}ms",
                ]
                if args.press_mode == "hold":
                    # Read once: the key thread can set held_button to None
                    # between the truth test and the .upper() call, which used to
                    # raise AttributeError and kill this loop.
                    held_now = state.held_button
                    status_parts.append(
                        f"DOWN HOLDING {held_now.upper()}" if held_now
                        else "---- no key")
                
                if state.blocked_button and args.press_mode == "tap":
                    status_parts.append(f"[HELD] held:{state.blocked_button} (release to re-fire)")

                if debounce_remaining > 0:
                    status_parts.append(f"[TIME] Debounce: {debounce_remaining/1000:.1f}s")
                
                if args.pinch_mode == "position":
                    if is_pinch and state.pos_state != "rest":
                        status_parts.append(f"PINCH PINCH @ {state.pos_state.upper()} -> FIRE")
                    elif is_pinch:
                        status_parts.append("PINCH PINCH @ REST (no button)")
                    elif state.pos_state != "rest":
                        status_parts.append(f"arm {state.pos_state.upper()} (pinch to fire)")
                    else:
                        status_parts.append("idle Idle")
                elif is_pinch and movement:
                    status_parts.append(f"MOVING Moving {movement.upper()} -> READY")
                elif is_pinch:
                    status_parts.append("REST PINCH (move for button)")
                elif movement:
                    status_parts.append(f"Moving {movement.upper()} (need pinch)")
                else:
                    status_parts.append("idle Idle")
                
                # \033[K clears to end of line. Without it, a shorter line leaves
                # the tail of the previous longer one behind, which is where the
                # garbled "Idleounce" / "Idle:right (release to re-fire)" came from.
                print(f"\r{' | '.join(status_parts)}\033[K", end="", flush=True)
                time.sleep(0.05)
            
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        print("\n\n[STOP] Stopping...")
        # The key thread is a daemon, so its finally: block is NOT guaranteed to
        # run at interpreter exit. Join it, then lift every key anyway -- a key
        # left down after Ctrl+C would keep driving the car.
        try:
            key_thread.join(timeout=1.5)
        except Exception:
            pass
        release_all(key_backend)
        # Same reasoning for the BLE thread: state.running = False only ASKS it
        # to stop. Its teardown (STOP write, stop_notify, GATT disconnect) is
        # async and takes hundreds of ms, and as a daemon it gets killed the
        # instant main() returns - so without this join the board never receives
        # its STOP, keeps streaming into a dead link, and stays "connected"
        # until its own supervision timeout expires.
        print("[BLE] Disconnecting device(s)...")
        ble_thread.join(timeout=BLE_SHUTDOWN_TIMEOUT_S)
        if ble_thread.is_alive():
            print(f"[WARN] device did not disconnect cleanly within "
                  f"{BLE_SHUTDOWN_TIMEOUT_S:.0f}s. If the next run cannot find "
                  f"it, power-cycle the band.")
        print("Done.")


if __name__ == "__main__":
    main()