"""
record_gesture.py
------------------
Records EMG data from ONE NPG-Lite BLE device (single band), applies a
Notch (mains hum) filter followed by an EMG bandpass filter to each
channel in real time, ALSO reads the board's onboard accelerometer (IMU
characteristic) and holds the latest accel sample alongside every EMG
sample, and saves the FILTERED EMG + accel data (not raw) as ONE merged
CSV per trial- ready to be used as training data for a gesture
classifier (pinch, flexion, extension, etc.).

Single band, multiple devices: this script still supports more than one
device connected at once- in that case every device's channels are
merged side-by-side, sample-index-aligned (all devices are told to start
streaming at the same instant), in address-sorted role order (dev1, dev2,
...; see resolve_devices() below), and each device's own
accel_x/accel_y/accel_z columns are appended after all the EMG channels.
Devices are recognized purely by their advertised name prefix
(DEVICE_NAME_PREFIX, "NPG-Lite-band", matched case-insensitively)- there's
no MAC address allow-list to
maintain, since end users generally don't know their board's MAC address
in advance. Every board of this type reports the same EMG channel count
(DEFAULT_CHANNELS_PER_DEVICE), so channel count doesn't need to be looked
up per-device either.

Accelerometer note: the onboard IMU (LIS3DH) samples at a much lower rate
(~20 notifications/sec, each internally batching a few samples) than the
EMG channels (500 Hz). Rather than resampling, this script uses simple
"sample-and-hold": every EMG sample gets tagged with whatever the most
recently received accel reading was, so wrist/arm orientation is still
available to the model as a slowly-changing feature alongside the fast
EMG signal. Values are raw signed 16-bit LIS3DH counts, not converted to
g's- that's fine for a classifier since it only cares about relative
scale (and both training and real-time inference use the same raw units).

Beep cue (repeated go/stop signal):
  For active gestures (anything other than 'rest'), one recording (up to
  MAX_RECORD_SECONDS) contains MANY short reps, not one long hold: streaming
  starts, then each rep is a single higher-pitched beep ("go" - start the
  gesture now), a hold of a RANDOM duration between GESTURE_HOLD_MIN_SEC and
  GESTURE_HOLD_MAX_SEC (drawn fresh for every rep, so the hold length isn't
  predictable), then a double lower-pitched beep ("stop" - relax now), then
  a relax gap before the next rep. Randomizing the hold duration (instead of
  a fixed length) discourages anticipating the stop cue and counting/timing
  it instead of actually listening for it. You still never have to guess how
  long to hold - the stop cue tells you. Every rep's go/stop timestamps are
  saved, and only the window between them (minus a little reaction time
  right after "go") is kept as training data for that gesture. Everything
  else in the recording (the relax time between reps, the reaction-time
  sliver right after each "go") is discarded rather than mislabeled - that's
  what was causing rest and gesture data to bleed into each other before.

  'rest' recordings are simpler: one "go" beep, then the whole rest of the
  trial is valid (no reps or "stop" cue needed - you're just holding still
  throughout).

Folder structure produced:

    training_data/
        dataset_index.json         <- the ONE index: every subject, every
                                       gesture, every session, with an
                                       include_in_training true/false flag
        <subject_name>/
            pinch/
                pinch_trial1_merged_DADA-DACE.csv
                pinch_trial1_merged_DADA-DACE.beep.json
                pinch_trial2_merged_DADA-DACE.csv
                pinch_trial2_merged_DADA-DACE.beep.json
            flexion/
                flexion_trial1_merged_DADA-DACE.csv
                flexion_trial1_merged_DADA-DACE.beep.json
                ...

The subject name is asked for interactively at startup, so different
people's trials are kept in separate folders and never mixed together.

dataset_index.json tracks, per subject and gesture, how many recording
sessions exist and how many clean/valid samples each has, plus an
"include_in_training" true/false flag per session. Sessions default to
true; use the 'sessions' command at the gesture-name prompt to review the
full breakdown and flip individual sessions to false (e.g. a trial where
an electrode slipped) without deleting the underlying CSV -
train_gesture_model.py reads this same file and skips any session flagged
false automatically, with no prompting.

Each merged CSV has columns: timestamp, counter_<role> (one per device),
ch1..chN (all FILTERED EMG channels, across all devices in role order),
then accel_x/accel_y/accel_z per device (suffixed with the role name if
more than one device is connected, e.g. accel_x_dev1, accel_x_dev2).
With the default single-device config that's just: timestamp,
counter_dev1, ch1, ch2, ch3, accel_x, accel_y, accel_z.

Each ".beep.json" sidecar records every beep's timestamp/sample index and
the resulting list of clean "segments" (sample ranges) that are actually
safe to train on. See merge_and_save() below for the exact fields.

Usage:
    python record_gesture.py
    python record_gesture.py --rep_interval 3.5 --gesture_hold_min 2.0 --gesture_hold_max 6.0   (slower pace)

Requires: bleak  (pip install bleak)
For a reliably audible beep, also install: pip install sounddevice numpy
(otherwise it falls back to system players like aplay/paplay/afplay, or the
terminal bell as a last resort, which many terminals mute by default).
"""

import argparse
import asyncio
import csv
import glob
import json
import math
import os
import platform
import queue
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave

from bleak import BleakScanner, BleakClient


# ==============================
# stdin
# ==============================
class _Nothing:
    """Sentinel: the queue was empty, as distinct from an empty input line."""


_NOTHING = _Nothing()


class StdinReader:
    """The single owner of stdin for this process.

    Recording used to do `loop.run_in_executor(None, input)` once per trial.
    When a trial hit MAX_RECORD_SECONDS instead of an early Enter, that task
    stayed blocked on stdin forever and permanently consumed a worker in the
    default ThreadPoolExecutor (bounded to min(32, cpu_count + 4)). Since
    record_gesture() runs once per trial, enough auto-stops in one session
    exhausted the pool and every later prompt queued forever, hanging the tool.
    Leftover readers also raced legitimate prompts for the same stdin fd and
    could silently swallow the line meant for the next question.

    One reader thread, one queue, reused by every prompt: nothing blocks the
    event loop, nothing accumulates, and there is never more than one consumer
    of the fd. All prompts in this file go through .input(); the async recording
    path uses .wait_for_enter().
    """

    def __init__(self):
        self._q = queue.Queue()
        threading.Thread(target=self._run, daemon=True, name="stdin-reader").start()

    def _run(self):
        try:
            for line in sys.stdin:
                self._q.put(line.rstrip("\r\n"))
        except Exception:
            pass
        self._q.put(None)  # EOF marker

    def _unwrap(self, line):
        if line is None:
            self._q.put(None)  # keep EOF sticky for any later caller
            raise EOFError("stdin closed")
        return line

    def input(self, prompt=""):
        """Drop-in replacement for the builtin input()."""
        if prompt:
            sys.stdout.write(prompt)
            sys.stdout.flush()
        return self._unwrap(self._q.get())

    def drain(self):
        """Discard anything typed before the current prompt was shown."""
        while True:
            try:
                if self._q.get_nowait() is None:
                    self._q.put(None)
                    return
            except queue.Empty:
                return

    async def wait_for_enter(self, timeout):
        """True if Enter arrived, False if `timeout` elapsed (or stdin is at
        EOF) first. Polls the queue from the event loop, so it never leaves a
        blocked worker thread behind the way the old executor task did."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                line = self._q.get_nowait()
            except queue.Empty:
                line = _NOTHING
            if line is None:
                self._q.put(None)
                return False
            if line is not _NOTHING:
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.05)


STDIN = StdinReader()

# ==============================
# BLE UUIDs (must match firmware)
# ==============================
DATA_CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
CONTROL_CHAR_UUID = "0000ff01-0000-1000-8000-00805f9b34fb"
IMU_CHAR_UUID = "5a153fa9-7be0-400c-8ef8-d84502b31c4d"  # onboard accel, notify-only
DEVICE_NAME_PREFIX = "NPG-Lite-band"
USE_ALL_DEVICES = False  # overridden by --all_devices in __main__

SAMPLES_PER_PACKET = 20  # BLOCK_COUNT in firmware
IMU_SAMPLE_LEN = 7       # firmware IMU_SAMPLE_SIZE: 1 counter byte + 3x int16 (ax,ay,az)
OUTPUT_ROOT = "training_data"
REST_LABEL = "rest"  # baseline/no-movement class, always recorded first
MAX_RECORD_SECONDS = 60  # hard cap: recording auto-stops after this many seconds

# ---- Beep cue configuration ----
# For active gestures: streaming starts, then a beep repeats for up to
# MAX_RECORD_SECONDS. Each beep means "do it now": REACTION_DELAY_SEC right
# after the beep is still reaction time (auditory cue -> brain -> muscle),
# not a clean gesture sample, and after that comes the hold window - the
# time you're expected to actually hold the gesture in, before the double
# "stop" beep. That hold window is RANDOM, redrawn independently for every
# single rep from Uniform(GESTURE_HOLD_MIN_SEC, GESTURE_HOLD_MAX_SEC), so
# you can't anticipate the stop cue by counting - you have to actually wait
# for it. After "stop", REP_INTERVAL_SEC (+/- REP_INTERVAL_JITTER_SEC) of
# relax time passes before the next "go". Only the reaction-delay-to-stop
# window is kept for training; the reaction sliver and the relax time
# between beeps are discarded, which is what stops rest/gesture data from
# mixing together. All of these can be overridden from the command line,
# e.g. `python record_gesture.py --rep_interval 3.5 --gesture_hold_min 1.5
# --gesture_hold_max 5.0` if the default pace feels too fast to comfortably
# react to.
BEEP_COUNTDOWN_SEC = 2.0        # "get ready" pause before recording/reps begin
REACTION_DELAY_SEC = 0.3        # discarded window right after each beep
GESTURE_HOLD_MIN_SEC = 1.5      # shortest possible go->stop hold duration
GESTURE_HOLD_MAX_SEC = 5.0      # longest possible go->stop hold duration
REP_INTERVAL_SEC = 3.0          # average relax time between stop and next go
REP_INTERVAL_JITTER_SEC = 0.5   # +/- random jitter on the interval above
GO_BEEP_FREQ = 1000              # single, higher-pitched beep = "start now"
STOP_BEEP_FREQ = 550              # double, lower-pitched beep = "stop/relax now"

# ---- Filter configuration ----
SAMPLING_RATE = 500      # must be 250 or 500 (see note above)
ADC_BITS = "12"          # matches ADC_BITWIDTH_12 in firmware
NOTCH_TYPE = 1            # 1 = 48-52Hz (50Hz mains, India/EU), 2 = 58-62Hz (60Hz mains, US)
EMG_TYPE = 4               # fixed: 4 = EMG bandpass in EXGFilter

# ==============================
# Device resolution - by NAME PREFIX ONLY, no MAC allow-list.
# ==============================
# Devices are recognized purely by their advertised name prefix
# (DEVICE_NAME_PREFIX). There's no MAC-address allow-list to maintain,
# since end users setting this up generally don't know their board's MAC
# address ahead of time.
#
# Two things this still needs to get right, now that identity isn't pinned
# by address:
#
#   1) Stable role ordering across runs. BleakScanner.discover() order is
#      NOT stable across runs (it depends on scan timing / RSSI at that
#      moment), so relying on "[0] = dev1" from the scan output would
#      silently swap devices between sessions whenever more than one board
#      is connected. Instead, resolve_devices() sorts discovered devices by
#      BLE address and assigns dev1, dev2, ... in that order - as long as
#      the same physical boards are used, they get the same role each time.
#
#   2) Consistent channel count. Every NPG-Lite board of this type reports
#      the same number of EMG channels, so DEFAULT_CHANNELS_PER_DEVICE below
#      is used for every prefix-matched device rather than guessing per
#      device from its advertised name (which has been observed to be
#      unreliable) or looking it up per address.
#
# IMPORTANT: gesture_ui_server.py (used at inference time) must resolve
# devices with this exact same logic (name-prefix + address-sort) so that
# the feature-column order used at training time matches the column order
# used at inference time. Both files already do this.
DEFAULT_CHANNELS_PER_DEVICE = 3  # EMG channels per NPG-Lite board


def device_name_matches(name):
    """True if `name` is one of our bands.

    Matched case-insensitively on purpose: the firmware currently advertises
    'NPG-Lite-band-3CH:...' (lowercase b) while the constant was written
    'NPG-Lite-Band' during the repo rename, and a plain startswith() rejected
    every board with a "no devices found" message. Casing has flipped between
    firmware builds before, so don't rely on it.

    The prefix still ends at '-band' deliberately: it must NOT match other
    NPG-Lite boards such as 'NPG-Lite-6CH:...', because
    DEFAULT_CHANNELS_PER_DEVICE below is hardcoded to the band's channel count
    rather than parsed from the advertised name.
    """
    return bool(name) and name.lower().startswith(DEVICE_NAME_PREFIX.lower())


# ==============================
# Filters (ported from filters.ts)
# ==============================
class NotchFilter:
    """50/60Hz mains notch filter. One instance PER CHANNEL (has internal state)."""

    def __init__(self):
        self.z1_1 = 0.0
        self.z2_1 = 0.0
        self.z1_2 = 0.0
        self.z2_2 = 0.0
        self.x_1 = 0.0
        self.x_2 = 0.0
        self.sampling_rate = 0

    def set_sampling_rate(self, sampling_rate: int):
        self.sampling_rate = sampling_rate

    def process(self, input_val: float, notch_type: int) -> float:
        if not notch_type:
            return input_val

        output = input_val

        if self.sampling_rate == 500:
            if notch_type == 1:  # 48-52 Hz
                self.x_1 = output - (-1.56858163 * self.z1_1) - (0.96424138 * self.z2_1)
                output = 0.96508099 * self.x_1 + -1.56202714 * self.z1_1 + 0.96508099 * self.z2_1
                self.z2_1 = self.z1_1
                self.z1_1 = self.x_1

                self.x_2 = output - (-1.61100358 * self.z1_2) - (0.96592171 * self.z2_2)
                output = 1.0 * self.x_2 + -1.61854514 * self.z1_2 + 1.0 * self.z2_2
                self.z2_2 = self.z1_2
                self.z1_2 = self.x_2

            elif notch_type == 2:  # 58-62 Hz
                self.x_1 = output - (-1.40810535 * self.z1_1) - (0.96443153 * self.z2_1)
                output = 0.96508099 * self.x_1 + (-1.40747202 * self.z1_1) + (0.96508099 * self.z2_1)
                self.z2_1 = self.z1_1
                self.z1_1 = self.x_1

                self.x_2 = output - (-1.45687509 * self.z1_2) - (0.96573127 * self.z2_2)
                output = 1.00000000 * self.x_2 + (-1.45839783 * self.z1_2) + (1.00000000 * self.z2_2)
                self.z2_2 = self.z1_2
                self.z1_2 = self.x_2

        elif self.sampling_rate == 250:
            if notch_type == 1:  # 48-52 Hz
                self.x_1 = output - (-0.53127491 * self.z1_1) - (0.93061518 * self.z2_1)
                output = 0.93137886 * self.x_1 + (-0.57635175 * self.z1_1) + 0.93137886 * self.z2_1
                self.z2_1 = self.z1_1
                self.z1_1 = self.x_1

                self.x_2 = output - (-0.66243374 * self.z1_2) - (0.93214913 * self.z2_2)
                output = 1.00000000 * self.x_2 + (-0.61881558 * self.z1_2) + 1.00000000 * self.z2_2
                self.z2_2 = self.z1_2
                self.z1_2 = self.x_2

            elif notch_type == 2:  # 58-62 Hz
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
    """ECG/EOG/EEG/EMG bandpass filter bank. One instance PER CHANNEL (has internal state)."""

    def __init__(self):
        self.z1 = 0.0
        self.z2 = 0.0
        self.x1 = 0.0
        self.x2 = 0.0
        self.x3 = 0.0
        self.x4 = 0.0
        self.bits = None
        self.bits_points = 0
        self.sampling_rate = 0

    def set_bits(self, bits: str, sampling_rate: int):
        self.sampling_rate = sampling_rate
        self.bits = bits
        self.bits_points = 2 ** int(bits)

    def process(self, input_val: float, exg_type: int) -> float:
        if not exg_type:
            return input_val

        output = input_val
        ch_data = 0.0

        if self.sampling_rate == 500:
            if exg_type == 4:  # EMG, 70Hz
                self.x4 = output - (-0.82523238 * self.z1) - (0.29463653 * self.z2)
                output = 0.52996723 * self.x4 + -1.05993445 * self.z1 + 0.52996723 * self.z2
                self.z2 = self.z1
                self.z1 = self.x4
                ch_data = output
            # (ECG/EOG/EEG cases omitted here- add back from filters.ts if you need them)

        elif self.sampling_rate == 250:
            if exg_type == 4:  # EMG, 70Hz
                self.x4 = output - 0.22115344 * self.z1 - 0.18023207 * self.z2
                output = 0.23976966 * self.x4 + -0.47953932 * self.z1 + 0.23976966 * self.z2
                self.z2 = self.z1
                self.z1 = self.x4
                ch_data = output

        return ch_data


_beep_wav_cache = {}         # (freq, duration_ms) -> generated wav path, reused after first use
_warned_no_audio_backend = False  # only nag about missing audio once per run


def _generate_beep_wav(freq=1000, duration_ms=250, volume=0.5, sample_rate=44100):
    """Write a short sine-wave tone to a temp .wav file (no numpy needed -
    pure stdlib) and return its path, so it can be handed to a system audio
    player like `aplay`/`paplay`/`afplay`."""
    n_samples = int(sample_rate * duration_ms / 1000)
    fade_samples = max(1, int(sample_rate * 0.01))  # 10ms fade in/out, avoids clicks
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="gesture_beep_")
    os.close(fd)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_samples):
            fade = min(1.0, i / fade_samples, (n_samples - i) / fade_samples)
            sample = volume * fade * math.sin(2 * math.pi * freq * i / sample_rate)
            frames += struct.pack("<h", int(sample * 32767))
        wf.writeframes(bytes(frames))
    return path


def _get_beep_wav(freq, duration_ms):
    key = (freq, duration_ms)
    path = _beep_wav_cache.get(key)
    if path is None or not os.path.exists(path):
        path = _generate_beep_wav(freq=freq, duration_ms=duration_ms)
        _beep_wav_cache[key] = path
    return path


def play_beep(freq=1000, duration_ms=250):
    """Play a short, clearly audible beep. Tries several backends in order
    and only falls back to the (often silent/muted) terminal bell if
    nothing else works. If you genuinely can't hear any beep, install one
    of:
        pip install sounddevice numpy      (cross-platform, most reliable)
        sudo apt install alsa-utils        (Linux -> gives you `aplay`)
        sudo apt install pulseaudio-utils  (Linux -> gives you `paplay`)
    """
    global _warned_no_audio_backend
    system = platform.system()

    # 1) Windows native beep - no extra deps needed.
    if system == "Windows":
        try:
            import winsound
            winsound.Beep(freq, duration_ms)
            return
        except Exception:
            pass

    # 2) sounddevice + numpy, if installed - cross-platform and reliable.
    try:
        import numpy as np
        import sounddevice as sd
        sr = 44100
        t = np.linspace(0, duration_ms / 1000, int(sr * duration_ms / 1000), False)
        tone = 0.5 * np.sin(2 * np.pi * freq * t)
        sd.play(tone, sr)
        sd.wait()
        return
    except Exception:
        pass

    # 3) Play a generated WAV through whatever system player is available
    # (macOS `afplay`, Linux `paplay`/`aplay`). Most Linux desktops already
    # have one of these installed. Cached per (freq, duration_ms) so the
    # go/stop cues (different pitches) don't collide on one cached file.
    player = None
    if system == "Darwin" and shutil.which("afplay"):
        player = ["afplay"]
    elif shutil.which("paplay"):
        player = ["paplay"]
    elif shutil.which("aplay"):
        player = ["aplay", "-q"]
    if player:
        try:
            wav_path = _get_beep_wav(freq, duration_ms)
            subprocess.run(player + [wav_path], check=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except Exception:
            pass

    # 4) Last resort: terminal bell. Unreliable - many terminals mute it by
    # default and it won't play at all over some SSH sessions or inside IDE
    # consoles, which is almost certainly why you're not hearing anything.
    sys.stdout.write("\a")
    sys.stdout.flush()
    if not _warned_no_audio_backend:
        _warned_no_audio_backend = True
        print(
            "\n  (warning) No working audio backend found (winsound/sounddevice/"
            "aplay/paplay/afplay) - falling back to the terminal bell, which most "
            "terminals mute by default. If you can't hear the beeps, run:\n"
            "    pip install sounddevice numpy\n"
            "  or install `alsa-utils`/`pulseaudio-utils` (Linux) so a real "
            "sound plays.\n"
        )


def play_go_cue():
    """Single, higher-pitched beep: 'do the gesture now'."""
    play_beep(freq=GO_BEEP_FREQ, duration_ms=250)


def play_stop_cue():
    """Double, lower-pitched beep: 'stop / relax now' - audibly distinct
    from the single go cue so you don't have to guess how long to hold."""
    play_beep(freq=STOP_BEEP_FREQ, duration_ms=150)
    time.sleep(0.09)
    play_beep(freq=STOP_BEEP_FREQ, duration_ms=150)


def channels_from_name(name: str) -> int:
    """Kept as a fallback for anything that still wants a name-based guess
    (e.g. display/debug code); normal device resolution now always uses
    DEFAULT_CHANNELS_PER_DEVICE instead of parsing the advertised name."""
    if "6CH" in name:
        return 6
    return 3


class DeviceRecorder:
    """Handles one BLE device: connect, filter EMG data live, buffer, save CSV."""

    def __init__(self, ble_device, out_path, channels=None, role=None):
        self.ble_device = ble_device
        self.name = ble_device.name or "UNKNOWN"
        # Prefer the explicit channel count passed in from resolve_devices()
        # (DEFAULT_CHANNELS_PER_DEVICE) over guessing from the advertised
        # name, since the name has been observed to be wrong/misleading.
        self.channels = channels if channels is not None else channels_from_name(self.name)
        self.role = role or self.name
        self.out_path = out_path
        self.client = None
        self.rows = []

        # Sample-and-hold accel state, updated asynchronously by
        # handle_imu_notify() whenever a new IMU notification arrives.
        # Starts at (0, 0, 0)- if the IMU never sends anything (e.g. no
        # IMU wired to this board), rows just get zeros for accel, which
        # is called out via self.has_imu below.
        self.latest_accel = [0, 0, 0]
        self.has_imu = False

        # One filter pair PER CHANNEL- never share instances across channels
        self.notch_filters = []
        self.emg_filters = []
        for _ in range(self.channels):
            notch = NotchFilter()
            notch.set_sampling_rate(SAMPLING_RATE)
            self.notch_filters.append(notch)

            emg = EXGFilter()
            emg.set_bits(ADC_BITS, SAMPLING_RATE)
            self.emg_filters.append(emg)

    def handle_notify(self, _, data: bytearray):
        expected_len = SAMPLES_PER_PACKET * (1 + self.channels * 2)
        if len(data) != expected_len:
            # Ignore malformed/partial packets
            return

        offset = 0
        # One packet carries SAMPLES_PER_PACKET samples acquired at a fixed rate,
        # so tagging all of them with the arrival time would quantise every
        # timestamp to a whole packet (~40 ms @ 500 Hz). merge_and_save's
        # sample_index_for() maps the beep go/stop wall-clock times onto sample
        # indices, so that quantisation blurs exactly the segment edges the beep
        # cue design exists to make sharp. Back-date the packet instead: the last
        # sample is "now", earlier ones step back one sample period each.
        t_end = time.time()
        dt = 1.0 / float(SAMPLING_RATE)
        t_start = t_end - (SAMPLES_PER_PACKET - 1) * dt
        for i in range(SAMPLES_PER_PACKET):
            t = t_start + i * dt
            counter = data[offset]
            filtered_vals = []
            for ch in range(self.channels):
                raw_val = (data[offset + 1 + ch * 2] << 8) | data[offset + 2 + ch * 2]

                notched = self.notch_filters[ch].process(raw_val, NOTCH_TYPE)
                filtered = self.emg_filters[ch].process(notched, EMG_TYPE)

                filtered_vals.append(filtered)
            offset += 1 + self.channels * 2
            # Sample-and-hold: tag this EMG sample with whatever the most
            # recent accel reading is (IMU notifies far less often than
            # EMG, so consecutive EMG samples will often carry the same
            # accel triple until the next IMU notification arrives).
            self.rows.append([t, counter] + filtered_vals + list(self.latest_accel))

    def handle_imu_notify(self, _, data: bytearray):
        """Parses the firmware's IMU characteristic: a batch of 0..32
        samples, each IMU_SAMPLE_LEN (7) bytes: 1 counter byte + big-endian
        signed int16 ax, ay, az. Only the LAST sample in the batch is kept
        (sample-and-hold just needs the most current reading, not every
        historical one)."""
        if len(data) == 0 or len(data) % IMU_SAMPLE_LEN != 0:
            return  # malformed/partial packet
        n_samples = len(data) // IMU_SAMPLE_LEN
        offset = (n_samples - 1) * IMU_SAMPLE_LEN  # most recent sample in this batch
        ax = struct.unpack(">h", data[offset + 1:offset + 3])[0]
        ay = struct.unpack(">h", data[offset + 3:offset + 5])[0]
        az = struct.unpack(">h", data[offset + 5:offset + 7])[0]
        self.latest_accel = [ax, ay, az]
        self.has_imu = True

    async def connect(self):
        """Just connect + subscribe. Must be called ONE DEVICE AT A TIME-
        BlueZ can't handle concurrent Connect() calls (raises
        org.bluez.Error.InProgress if you gather() these)."""
        self.client = BleakClient(self.ble_device.address)
        await self.client.connect()
        await self.client.start_notify(DATA_CHAR_UUID, self.handle_notify)
        try:
            await self.client.start_notify(IMU_CHAR_UUID, self.handle_imu_notify)
        except Exception as e:
            print(f"  (warning) IMU characteristic not available for {self.name} "
                  f"({e}) - accel columns will be all zeros for this device.")

    async def start_streaming(self):
        """Send START command. Safe to call concurrently once all devices
        are already connected."""
        await self.client.write_gatt_char(CONTROL_CHAR_UUID, b"STOP", response=True)
        await asyncio.sleep(0.1)
        await self.client.write_gatt_char(CONTROL_CHAR_UUID, b"START", response=True)

    async def stop_and_disconnect(self):
        try:
            await self.client.write_gatt_char(CONTROL_CHAR_UUID, b"STOP", response=True)
            await self.client.stop_notify(DATA_CHAR_UUID)
        except Exception as e:
            print(f"  (warning) clean stop failed for {self.name}: {e}")
        try:
            await self.client.stop_notify(IMU_CHAR_UUID)
        except Exception:
            pass
        try:
            await self.client.disconnect()
        except Exception:
            pass
        if not self.has_imu:
            print(f"  (warning) {self.name} never sent an IMU/accel reading this trial "
                  f"- accel columns are all zeros. Check the board has a working IMU.")

    def save_csv(self):
        """Per-device raw dump- kept available for debugging, but normal
        recording now uses merge_and_save() below instead so training data
        is one combined multi-channel file per trial."""
        header = ["timestamp", "counter"] + [f"ch{i + 1}" for i in range(self.channels)]
        with open(self.out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(self.rows)
        print(f"  Saved {len(self.rows)} filtered samples -> {self.out_path}")


def merge_and_save(gesture_dir, gesture_name, trial_num, recorders, reps=None, mode="reps"):
    """Merge every connected device's filtered samples into ONE csv, side by
    side, sample-index-aligned (dev1's channels first, then dev2's, ...).

    Why index-aligned rather than timestamp-aligned: both devices are told
    to START streaming concurrently (asyncio.gather in start_streaming, see
    record_gesture() below), so sample i of dev1 and sample i of dev2 were
    captured at essentially the same instant. This is what makes it valid
    to treat the two boards as one combined N-channel sensor (e.g. two
    3-channel boards worn on opposite sides of the same arm -> one
    6-channel gesture reading) instead of training two separate models.

    If the two streams end up with different sample counts (packet loss,
    BLE jitter), both are truncated to the shorter length so every row
    stays aligned across devices.

    If reps is given (a list of {"go": time.time(), "stop": time.time()|None}
    dicts, one per rep heard during the trial), a "<csv name>.beep.json"
    sidecar is written alongside the CSV recording every rep's go/stop
    sample index and the resulting list of clean "segments" (sample ranges)
    that are safe to train on:
      - mode="reps" (active gestures): each rep's segment starts
        REACTION_DELAY_SEC after its "go" beep and ends at its "stop" beep
        (the double, lower-pitched cue) - i.e. exactly the window the user
        was actually told to hold the gesture for. If a rep has no stop
        timestamp (trial was interrupted mid-hold), it falls back to that
        rep's own randomly-drawn hold_sec (see reps[i]["hold_sec"]) after
        "go", clipped so it never runs into the next rep's "go" cue.
      - mode="continuous" (rest): the single rep's "go" beep just marks
        "recording started for real"; everything from REACTION_DELAY_SEC
        after it to the end of the trial is one long valid segment.
    train_gesture_model.py only builds training windows from inside these
    segments, so reaction time and inter-rep relax time never get labeled
    as the gesture (or bleed into a neighboring rest recording).

    Returns a dict with csv_path/beep_path/num_samples/num_segments/
    total_valid_samples/mode for the caller to register in the subject's
    dataset_index.json, or None if nothing was captured.
    """
    recorders_sorted = sorted(recorders, key=lambda r: r.role)
    lengths = [len(r.rows) for r in recorders_sorted]
    n = min(lengths) if lengths else 0

    if len(set(lengths)) > 1:
        print(f"  (warning) devices returned different sample counts {lengths} "
              f"- truncating all to the shortest ({n}) to keep channels aligned.")
    if n == 0:
        print("  (warning) no samples captured - nothing to save for this trial.")
        return None

    header = ["timestamp"] + [f"counter_{r.role}" for r in recorders_sorted]
    ch_num = 0
    for r in recorders_sorted:
        for _ in range(r.channels):
            ch_num += 1
            header.append(f"ch{ch_num}")
    # Accel columns come after ALL devices' EMG channels (not interleaved),
    # so ch1..chN always stays a contiguous EMG block regardless of how
    # many devices are connected. With one device this is just
    # accel_x/accel_y/accel_z; with more than one, each device's triple is
    # suffixed with its role so columns stay unambiguous.
    multi_device = len(recorders_sorted) > 1
    for r in recorders_sorted:
        suffix = f"_{r.role}" if multi_device else ""
        header += [f"accel_x{suffix}", f"accel_y{suffix}", f"accel_z{suffix}"]

    rows = []
    for i in range(n):
        row = [recorders_sorted[0].rows[i][0]]          # reference timestamp
        row += [r.rows[i][1] for r in recorders_sorted]  # each device's packet counter
        for r in recorders_sorted:
            row.extend(r.rows[i][2:2 + r.channels])      # that device's filtered EMG channels
        for r in recorders_sorted:
            row.extend(r.rows[i][2 + r.channels:2 + r.channels + 3])  # that device's accel x,y,z
        rows.append(row)

    addr_tag = "-".join(r.ble_device.address.replace(":", "")[-4:] for r in recorders_sorted)
    ts_tag = time.strftime("%Y%m%d-%H%M%S")
    fname = f"{gesture_name}_trial{trial_num}_{ts_tag}_merged_{addr_tag}.csv"
    out_path = os.path.join(gesture_dir, fname)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    roles = "+".join(f"{r.role}({r.channels}ch)" for r in recorders_sorted)
    total_feature_cols = ch_num + 3 * len(recorders_sorted)
    print(f"  Saved {n} merged samples, {ch_num}ch EMG + "
          f"{3 * len(recorders_sorted)}ch accel = {total_feature_cols} feature columns total "
          f"[{roles}] -> {out_path}")

    result = {
        "csv_path": out_path,
        "beep_path": None,
        "num_samples": n,
        "num_channels": ch_num,
        "mode": mode,
        "num_segments": 0,
        "total_valid_samples": None,
    }

    if reps:
        timestamps = [row[0] for row in rows]

        def sample_index_for(ts):
            return next((i for i, s_ts in enumerate(timestamps) if s_ts >= ts), n)

        reaction_delay_samples = int(round(REACTION_DELAY_SEC * SAMPLING_RATE))

        rep_indices = []
        for rep in reps:
            go_idx = sample_index_for(rep["go"])
            stop_idx = sample_index_for(rep["stop"]) if rep.get("stop") is not None else None
            rep_indices.append({"go": go_idx, "stop": stop_idx})

        segments = []
        for rep_i, idx in enumerate(rep_indices):
            start = min(idx["go"] + reaction_delay_samples, n)
            if mode == "continuous":
                end = n  # rest: everything after the reaction delay is valid
            elif idx["stop"] is not None:
                end = min(idx["stop"], n)  # exact stop-cue timestamp, most precise
            else:
                # Trial was interrupted before the stop cue fired - fall back
                # to that specific rep's own randomly-drawn hold duration
                # (each rep gets a different one) rather than a fixed value.
                rep_hold_sec = reps[rep_i].get("hold_sec", GESTURE_HOLD_MAX_SEC)
                hold_samples = int(round(rep_hold_sec * SAMPLING_RATE))
                end = min(start + hold_samples, n)
            # Never let a segment bleed into the next rep's go cue.
            if rep_i + 1 < len(rep_indices):
                end = min(end, rep_indices[rep_i + 1]["go"])
            if end > start:
                segments.append([start, end])

        total_valid = sum(e - s for s, e in segments)
        beep_meta = {
            "mode": mode,
            "reps": [
                {
                    "go_time": rep["go"],
                    "stop_time": rep.get("stop"),
                    "hold_sec": rep.get("hold_sec") if mode == "reps" else None,
                }
                for rep in reps
            ],
            "rep_sample_indices": rep_indices,
            "reaction_delay_sec": REACTION_DELAY_SEC,
            "gesture_hold_range_sec": (
                [GESTURE_HOLD_MIN_SEC, GESTURE_HOLD_MAX_SEC] if mode == "reps" else None
            ),
            "segments": segments,
            "total_samples": n,
        }
        beep_path = out_path[:-4] + ".beep.json" if out_path.endswith(".csv") else out_path + ".beep.json"
        with open(beep_path, "w") as bf:
            json.dump(beep_meta, bf, indent=2)

        print(f"  {len(rep_indices)} rep(s) -> {len(segments)} usable "
              f"segment(s), {total_valid} clean samples total "
              f"({total_valid / SAMPLING_RATE:.2f}s) -> {beep_path}")
        if not segments:
            print(f"  (warning) no usable segments in this trial - reps may have "
                  f"been too close together or the trial stopped too early. "
                  f"Consider recording it again.")

        result["beep_path"] = beep_path
        result["num_segments"] = len(segments)
        result["total_valid_samples"] = total_valid

    return result


async def discover_devices(timeout=6):
    print(f"Scanning for NPG-Lite devices ({timeout}s)...")
    devices = await BleakScanner.discover(timeout=timeout)
    found = [d for d in devices if device_name_matches(d.name)]
    if not found:
        raise RuntimeError(
            "No NPG-Lite devices found. Make sure both are powered on and advertising."
        )
    return found


def select_device(found):
    """Pick exactly ONE device to record from.

    If a single NPG-Lite board is in range it's used automatically. If
    several are in range, list them as 1..N with their full advertised name
    + BLE address and ask the user which one to use. Devices are
    address-sorted first so the menu order is stable across runs
    (BleakScanner.discover() order is not).
    """
    found_sorted = sorted(found, key=lambda d: d.address.upper())
    if len(found_sorted) == 1:
        d = found_sorted[0]
        print(f"Found 1 device: {d.name} ({d.address}) - using it.")
        return d

    print(f"\nFound {len(found_sorted)} NPG-Lite devices:")
    for i, d in enumerate(found_sorted, 1):
        print(f"  {i}) {d.name}  ({d.address})")
    while True:
        choice = STDIN.input(f"Select device to use [1-{len(found_sorted)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(found_sorted):
            d = found_sorted[int(choice) - 1]
            print(f"Using: {d.name} ({d.address})\n")
            return d
        print("Invalid choice - enter one of the numbers shown above.")


def next_trial_number(subject_dir, gesture_name):
    """Look at what's already saved on disk for this subject+gesture and
    return the next trial number to use, so re-running the script (a new
    process, with a fresh in-memory trial_counters dict) continues
    numbering from where the LAST session left off instead of starting
    back at 1 and overwriting existing files."""
    gesture_dir = os.path.join(subject_dir, gesture_name)
    existing = glob.glob(os.path.join(gesture_dir, f"{gesture_name}_trial*_merged_*.csv"))
    max_trial = 0
    for path in existing:
        fname = os.path.basename(path)
        # fname looks like: {gesture_name}_trial{N}_{timestamp}_merged_{addrs}.csv
        try:
            after_trial = fname.split("_trial", 1)[1]
            num_str = after_trial.split("_", 1)[0]
            max_trial = max(max_trial, int(num_str))
        except (IndexError, ValueError):
            continue
    return max_trial + 1


# ==============================
# Central dataset index
# ==============================
# ONE "dataset_index.json" lives directly under training_data/ (OUTPUT_ROOT)
# and tracks every subject/gesture/session in the whole dataset:
#   {"subjects": {subject_name: {gesture_name: {"sessions": [ {...} ]}}}}
# It's the single place that answers "how much data do I have, for which
# subjects and gestures, and which of those recording sessions should
# actually be used for training" - train_gesture_model.py reads the
# "include_in_training" flag on each session and skips any session set to
# false (e.g. a trial where the user noticed halfway through that their
# electrode had slipped, but doesn't want to delete the raw CSV) -
# entirely automatically, no prompting.
DATASET_INDEX_FILENAME = "dataset_index.json"


def dataset_index_path(output_root):
    return os.path.join(output_root, DATASET_INDEX_FILENAME)


def load_dataset_index(output_root):
    """Load the single central dataset_index.json, or return a fresh empty
    one if it doesn't exist yet (first recording ever)."""
    path = dataset_index_path(output_root)
    if os.path.exists(path):
        with open(path) as f:
            index = json.load(f)
        index.setdefault("subjects", {})
        return index
    return {"subjects": {}}


def save_dataset_index(output_root, index):
    path = dataset_index_path(output_root)
    with open(path, "w") as f:
        json.dump(index, f, indent=2)
    return path


def register_session(index, subject_name, gesture_name, session_result, trial_num):
    """Add (or overwrite, if re-running the same trial number) one
    recording session's stats into the in-memory central index. Every new
    session defaults to include_in_training=True - use the 'sessions'
    command in main() to flip individual sessions to False without
    deleting them."""
    if session_result is None:
        return
    subjects = index.setdefault("subjects", {})
    gestures = subjects.setdefault(subject_name, {})
    entry = gestures.setdefault(gesture_name, {"sessions": []})
    sessions = entry["sessions"]

    session_record = {
        "trial": trial_num,
        "csv": os.path.basename(session_result["csv_path"]),
        "beep_json": (
            os.path.basename(session_result["beep_path"])
            if session_result.get("beep_path") else None
        ),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": session_result["mode"],
        "num_samples": session_result["num_samples"],
        "num_channels": session_result["num_channels"],
        "num_segments": session_result["num_segments"],
        "total_valid_samples": session_result["total_valid_samples"],
        "include_in_training": True,
    }

    # Overwrite an existing entry for the same trial number rather than
    # duplicating it (shouldn't normally happen since trial numbers are
    # monotonically re-derived from disk, but keeps re-runs idempotent).
    for i, existing in enumerate(sessions):
        if existing.get("trial") == trial_num:
            session_record["include_in_training"] = existing.get("include_in_training", True)
            sessions[i] = session_record
            return
    sessions.append(session_record)


def print_dataset_summary(index, subject_name=None):
    """Print a per-gesture table: how many sessions exist, how many are
    marked for training, and how many clean (post-beep-trim) samples are
    available in total. Scoped to one subject if given, otherwise every
    subject in the central index."""
    subjects = index.get("subjects", {})
    names = [subject_name] if subject_name else sorted(subjects)
    if not any(subjects.get(n) for n in names):
        print("  (no recordings yet)")
        return
    for name in names:
        gestures = subjects.get(name, {})
        if not gestures:
            continue
        print(f"  subject '{name}':")
        print(f"    {'gesture':<16} {'sessions':>9} {'included':>9} {'valid samples':>15}")
        for gesture_name in sorted(gestures):
            sessions = gestures[gesture_name]["sessions"]
            included = [s for s in sessions if s.get("include_in_training", True)]
            valid_total = sum(
                s.get("total_valid_samples") or s.get("num_samples") or 0 for s in included
            )
            print(f"    {gesture_name:<16} {len(sessions):>9} {len(included):>9} {valid_total:>15}")


def sessions_menu(output_root, index):
    """Interactive loop for reviewing every subject/gesture/session in the
    central index and toggling which ones are used for training. Doesn't
    touch the CSV files on disk - only the include_in_training flag in
    dataset_index.json, so nothing is ever deleted and a session can be
    flipped back on later."""
    while True:
        subjects = index.get("subjects", {})
        if not subjects:
            print("  (nothing recorded yet)")
            return
        print("\n--- All recorded sessions ---")
        for subject_name in sorted(subjects):
            gestures = subjects[subject_name]
            for gesture_name in sorted(gestures):
                sessions = sorted(gestures[gesture_name]["sessions"], key=lambda s: s["trial"])
                print(f"\n  {subject_name} / {gesture_name}:")
                for s in sessions:
                    flag = "TRUE " if s.get("include_in_training", True) else "false"
                    valid = s.get("total_valid_samples")
                    valid_str = f"{valid} valid samples" if valid is not None else f"{s['num_samples']} samples"
                    print(f"    [{flag}] trial {s['trial']:>3}  {s.get('mode', '?'):<10} "
                          f"{valid_str:<22} {s['csv']}")
        print(
            "\n  Commands: 'toggle <subject> <gesture> <trial>' flips include_in_training, "
            "'done' returns to the main menu."
        )
        cmd = STDIN.input("  > ").strip()
        if not cmd or cmd.lower() in ("done", "q", "quit", "exit"):
            return
        parts = cmd.split()
        if len(parts) == 4 and parts[0].lower() == "toggle":
            _, subj, g, t = parts
            try:
                trial_num = int(t)
            except ValueError:
                print("  Trial number must be an integer.")
                continue
            sessions = subjects.get(subj, {}).get(g, {}).get("sessions", [])
            match = next((s for s in sessions if s["trial"] == trial_num), None)
            if match is None:
                print(f"  No session found for '{subj}' / '{g}' trial {trial_num}.")
                continue
            match["include_in_training"] = not match.get("include_in_training", True)
            save_dataset_index(output_root, index)
            state = "included in" if match["include_in_training"] else "excluded from"
            print(f"  trial {trial_num} of '{subj}/{g}' is now {state} training.")
        else:
            print("  Didn't understand that. Try: toggle alice pinch 2")


def resolve_devices(found):
    """Resolve every discovered NPG-Lite-prefixed device into a stable
    (device, role, channels) triple - no MAC allow-list needed.

    Role assignment: sort discovered devices by BLE address and assign
    dev1, dev2, ... in that order. BleakScanner.discover() order is NOT
    stable across runs, so this is what keeps role assignment (and
    therefore feature-column order) consistent across sessions as long as
    the same physical boards are used - the actual addresses don't need to
    be known ahead of time, just their relative sort order, which stays
    fixed for a given set of boards.

    Channel count: every board uses DEFAULT_CHANNELS_PER_DEVICE, since this
    hardware doesn't vary channel count per unit.

    Returns a list of (device, role, channels) sorted by role (dev1, dev2, ...).
    """
    found_sorted = sorted(found, key=lambda d: d.address.upper())
    resolved = []
    for i, d in enumerate(found_sorted):
        role = f"dev{i + 1}"
        resolved.append((d, role, DEFAULT_CHANNELS_PER_DEVICE))
    return resolved


async def record_gesture(gesture_name, resolved_devices, trial_num, subject_dir):
    gesture_dir = os.path.join(subject_dir, gesture_name)
    os.makedirs(gesture_dir, exist_ok=True)

    recorders = []
    for dev, role, channels in resolved_devices:
        # No individual out_path needed anymore- merge_and_save() below
        # writes ONE combined file per trial instead of one per device.
        recorders.append(DeviceRecorder(dev, out_path=None, channels=channels, role=role))

    print(f"\nConnecting to {len(recorders)} device(s) for gesture '{gesture_name}' (trial {trial_num})...")
    fixed_order = " -> ".join(r.role for r in recorders)
    print(f"  Connect order (address-sorted, independent of scan order): {fixed_order}")
    # Connect ONE AT A TIME- BlueZ errors out (InProgress) on concurrent Connect() calls
    for r in recorders:
        print(f"  Connecting to {r.role} [{r.name}] ({r.ble_device.address}, {r.channels}ch)...")
        await r.connect()
    print("  All devices connected.")

    # Starting the stream (writing START) IS safe to do concurrently
    await asyncio.gather(*(r.start_streaming() for r in recorders))

    loop = asyncio.get_event_loop()
    is_rest = (gesture_name == REST_LABEL)

    if is_rest:
        # --- Simple flow: one beep, then hold the pose (relaxed/still) for
        # the whole rest of the trial. No reps needed since nothing is
        # supposed to change during a rest recording.
        print(f"\nGet ready for '{gesture_name}' (keep it relaxed/still)...")
        for i in range(int(BEEP_COUNTDOWN_SEC), 0, -1):
            print(f"  {i}...")
            await asyncio.sleep(1.0)
        print("  BEEP - go!")
        await loop.run_in_executor(None, play_go_cue)
        reps = [{"go": time.time(), "stop": None}]

        print(
            f"Recording (filtered)... stay relaxed. Press Enter to stop early, "
            f"or it will auto-stop after {MAX_RECORD_SECONDS}s."
        )
        STDIN.drain()
        if await STDIN.wait_for_enter(MAX_RECORD_SECONDS):
            print("Enter pressed- stopping early.")
        else:
            print(f"\nReached {MAX_RECORD_SECONDS}s limit- stopping automatically.")

    else:
        # --- Repeated-beep-reps flow: many short reps inside one recording
        # instead of one long hold. Each rep is: relax -> single high GO
        # beep ("do it now") -> hold for a random GESTURE_HOLD_MIN_SEC..MAX_SEC -> double low STOP
        # beep ("stop/relax now") -> repeat. This repeats roughly every
        # REP_INTERVAL_SEC (~20 reps over MAX_RECORD_SECONDS by default).
        # Every rep's go/stop timestamps get saved, and only the window
        # between them (skipping a bit of reaction time right after go) is
        # kept for training (see merge_and_save()); everything else -
        # reaction time and inter-rep relax time - is discarded.
        avg_hold_sec = (GESTURE_HOLD_MIN_SEC + GESTURE_HOLD_MAX_SEC) / 2
        expected_reps = max(1, round(MAX_RECORD_SECONDS / (REP_INTERVAL_SEC + avg_hold_sec)))
        print(
            f"\nRecording '{gesture_name}' - about {expected_reps} reps over up to "
            f"{MAX_RECORD_SECONDS}s. Each rep: a SINGLE beep means start the "
            f"gesture now, a DOUBLE beep (lower pitch) a RANDOM "
            f"{GESTURE_HOLD_MIN_SEC:.1f}-{GESTURE_HOLD_MAX_SEC:.1f}s later means stop "
            f"and relax until the next single beep - don't try to count/anticipate "
            f"it, just listen for the double beep. Press Enter at any time to stop early."
        )
        print("Get ready...")
        for i in range(int(BEEP_COUNTDOWN_SEC), 0, -1):
            print(f"  {i}...")
            await asyncio.sleep(1.0)

        reps = []
        stop_event = asyncio.Event()
        # Relax gap between "stop" and the next "go" - the hold duration is
        # now random per-rep (not a fixed value), so there's nothing fixed
        # to subtract here; REP_INTERVAL_SEC is just the relax time itself.
        relax_gap = max(0.3, REP_INTERVAL_SEC)

        async def beep_loop():
            while not stop_event.is_set():
                # Relax time before the next "go" cue.
                relax_wait = max(
                    0.3, relax_gap + random.uniform(-REP_INTERVAL_JITTER_SEC, REP_INTERVAL_JITTER_SEC)
                )
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=relax_wait)
                    break  # stop requested during the relax wait
                except asyncio.TimeoutError:
                    pass
                if stop_event.is_set():
                    break

                # Single high beep: start the gesture now.
                await loop.run_in_executor(None, play_go_cue)
                # Draw a fresh random hold duration for THIS rep only, so the
                # gap before the stop cue can't be anticipated/counted.
                hold_sec = random.uniform(GESTURE_HOLD_MIN_SEC, GESTURE_HOLD_MAX_SEC)
                rep = {"go": time.time(), "stop": None, "hold_sec": hold_sec}
                reps.append(rep)
                # Deliberately not printing hold_sec here - showing it live
                # would let the user count/anticipate the stop beep instead
                # of actually reacting to it. It's still saved in the sidecar.
                print(f"  rep #{len(reps)} - GO (single beep)")

                # Hold window - wait for it, but bail early if stopped mid-hold.
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=hold_sec)
                    break  # stopped mid-hold; leave rep["stop"] as None
                except asyncio.TimeoutError:
                    pass

                # Double low beep: stop and relax now.
                await loop.run_in_executor(None, play_stop_cue)
                rep["stop"] = time.time()
                print(f"  rep #{len(reps)} - STOP (double beep, relax)")

        beeper_task = asyncio.ensure_future(beep_loop())

        STDIN.drain()
        if await STDIN.wait_for_enter(MAX_RECORD_SECONDS):
            print("Enter pressed- stopping early.")
        else:
            print(f"\nReached {MAX_RECORD_SECONDS}s limit- stopping automatically.")

        stop_event.set()
        await beeper_task
        completed = sum(1 for r in reps if r["stop"] is not None)
        print(f"  Recorded {len(reps)} rep(s) this trial ({completed} completed, "
              f"{len(reps) - completed} cut short).")

    await asyncio.gather(*(r.stop_and_disconnect() for r in recorders))

    return merge_and_save(
        gesture_dir, gesture_name, trial_num, recorders,
        reps=reps, mode="continuous" if is_rest else "reps",
    )


async def main():
    found = await discover_devices()

    if USE_ALL_DEVICES:
        # Old multi-ArmBand behavior: connect to every board in range.
        print("\nFound devices (--all_devices: connecting to every one):")
        for i, d in enumerate(found):
            print(f"  [{i}] {d.name} - {d.address}")
        selected = found
    else:
        # Default: connect to exactly ONE board. If several are in range,
        # select_device() shows a 1..N menu (full name + address).
        selected = [select_device(found)]

    devices = resolve_devices(selected)
    print("\nResolved device roles (stable across runs, address-sorted):")
    for d, role, channels in devices:
        print(f"  {role}: {d.name} - {d.address} ({channels}ch)")

    # --- Subject identity ---
    # Everything below is scoped to ONE subject's folder (training_data/
    # <subject>/...), so different people's trials never mix in the same
    # gesture folder and each subject gets their own dataset_index.json.
    while True:
        subject_name = STDIN.input(
            "\nEnter subject name (used as the folder name for this person's "
            "data): "
        ).strip()
        subject_name = "".join(c for c in subject_name if c.isalnum() or c in ("-", "_")).lower()
        if subject_name:
            break
        print("Invalid/empty subject name, try again.")

    subject_dir = os.path.join(OUTPUT_ROOT, subject_name)
    os.makedirs(subject_dir, exist_ok=True)
    index = load_dataset_index(OUTPUT_ROOT)
    save_dataset_index(OUTPUT_ROOT, index)  # create the central file on first run

    print(f"\nSubject: '{subject_name}' -> {subject_dir}/")
    print("Existing data for this subject:")
    print_dataset_summary(index, subject_name)

    # --- Always offer to record 'rest' (no-movement) baseline first ---
    # Without this class the model has to force every window into one of the
    # active gestures even when the hand is just sitting still. Doing it here,
    # up front, means you don't have to remember to type "rest" manually later.
    existing_rest_trials = next_trial_number(subject_dir, REST_LABEL) - 1
    if existing_rest_trials:
        print(f"\nFound {existing_rest_trials} existing '{REST_LABEL}' trial(s) already recorded for this subject.")
    ans = STDIN.input(
        f"\nRecord '{REST_LABEL}' baseline trials now? Keep your hand relaxed/still "
        f"while recording. [Y/n]: "
    ).strip().lower()
    if ans in ("", "y", "yes"):
        try:
            n_trials = int(STDIN.input("  How many rest trials? [default 3]: ").strip() or "3")
        except ValueError:
            n_trials = 3
        for i in range(n_trials):
            trial_num = next_trial_number(subject_dir, REST_LABEL)  # re-checked disk each time, never resets
            print(f"\n--- Rest trial {trial_num} ({i + 1} of {n_trials} this session) ---")
            try:
                session_result = await record_gesture(REST_LABEL, devices, trial_num, subject_dir)
                register_session(index, subject_name, REST_LABEL, session_result, trial_num)
                save_dataset_index(OUTPUT_ROOT, index)
            except Exception as e:
                print(f"Error during rest recording: {e}")

    while True:
        gesture_name = STDIN.input(
            "\nEnter gesture name (e.g. pinch, flexion, extension), "
            "'sessions' to review/toggle which recordings are used for "
            "training, or 'q' to quit: "
        ).strip()
        if gesture_name.lower() == "q":
            break
        if not gesture_name:
            continue
        if gesture_name.lower() == "sessions":
            sessions_menu(OUTPUT_ROOT, index)
            continue

        # sanitize folder-unsafe characters
        gesture_name = "".join(c for c in gesture_name if c.isalnum() or c in ("-", "_")).lower()
        if not gesture_name:
            print("Invalid gesture name, try again.")
            continue

        trial_num = next_trial_number(subject_dir, gesture_name)  # continues from whatever's already on disk

        try:
            session_result = await record_gesture(gesture_name, devices, trial_num, subject_dir)
            register_session(index, subject_name, gesture_name, session_result, trial_num)
            save_dataset_index(OUTPUT_ROOT, index)
        except Exception as e:
            print(f"Error during recording: {e}")

    print(f"\nDone. Subject '{subject_name}'s filtered dataset is in "
          f"'{subject_dir}', one subfolder per gesture. Final tally:")
    print_dataset_summary(index, subject_name)
    print(f"\nTo use only a subset of sessions when training, re-run this "
          f"script and pick 'sessions' from the menu, or hand-edit "
          f"'{dataset_index_path(OUTPUT_ROOT)}' directly.")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Record EMG gesture trials with a repeating beep cue."
    )
    ap.add_argument("--rep_interval", type=float, default=REP_INTERVAL_SEC,
                     help=f"average seconds between beeps during a gesture recording "
                          f"(default {REP_INTERVAL_SEC}) - raise this if the pace feels "
                          f"too fast to react to")
    ap.add_argument("--rep_interval_jitter", type=float, default=REP_INTERVAL_JITTER_SEC,
                     help=f"+/- random jitter on rep_interval (default {REP_INTERVAL_JITTER_SEC})")
    ap.add_argument("--gesture_hold_min", type=float, default=GESTURE_HOLD_MIN_SEC,
                     help=f"shortest possible random hold duration after each go beep, "
                          f"in seconds (default {GESTURE_HOLD_MIN_SEC})")
    ap.add_argument("--gesture_hold_max", type=float, default=GESTURE_HOLD_MAX_SEC,
                     help=f"longest possible random hold duration after each go beep, "
                          f"in seconds (default {GESTURE_HOLD_MAX_SEC})")
    ap.add_argument("--reaction_delay", type=float, default=REACTION_DELAY_SEC,
                     help=f"seconds right after each beep to discard as reaction time "
                          f"(default {REACTION_DELAY_SEC})")
    ap.add_argument("--max_record_seconds", type=int, default=MAX_RECORD_SECONDS,
                     help=f"hard cap per trial in seconds (default {MAX_RECORD_SECONDS})")
    ap.add_argument("--all_devices", action="store_true",
                     help="connect to EVERY NPG-Lite board in range (old multi-ArmBand "
                          "behavior) instead of prompting to pick one")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Override the module-level pacing constants from the CLI. Every
    # function above reads these by name at call time, so reassigning them
    # here (before main() actually runs) takes effect everywhere.
    REP_INTERVAL_SEC = args.rep_interval
    REP_INTERVAL_JITTER_SEC = args.rep_interval_jitter
    GESTURE_HOLD_MIN_SEC = args.gesture_hold_min
    GESTURE_HOLD_MAX_SEC = args.gesture_hold_max
    if GESTURE_HOLD_MIN_SEC > GESTURE_HOLD_MAX_SEC:
        print(
            f"Error: --gesture_hold_min ({GESTURE_HOLD_MIN_SEC}) can't be greater than "
            f"--gesture_hold_max ({GESTURE_HOLD_MAX_SEC})."
        )
        sys.exit(1)
    REACTION_DELAY_SEC = args.reaction_delay
    MAX_RECORD_SECONDS = args.max_record_seconds
    USE_ALL_DEVICES = args.all_devices

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"Error: {e}")