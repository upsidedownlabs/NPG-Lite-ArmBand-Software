"""
gesture_ui_server.py
----------------------
Same BLE + Notch/EMG filter + sliding-window + trained-model pipeline as
realtime_classify.py, but instead of printing predictions to the console it
runs a tiny local web server and opens a browser dashboard that shows an
animated hand (rest / pinch / extension / flexion) matching the live
prediction, plus a confidence bar and per-class probability bars.

Works on Linux, Windows, and macOS - it's just Python's built-in
http.server + a browser tab, no extra GUI toolkit and no extra pip installs
beyond what realtime_classify.py already needs (bleak, tensorflow, numpy).

Usage:
    python gesture_ui_server.py --model_dir gesture_model
    (opens http://localhost:8765 automatically; Ctrl+C in the terminal stops it)
"""

import argparse
import asyncio
import json
import os
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from bleak import BleakScanner, BleakClient
from tensorflow import keras

# ==============================
# BLE UUIDs (must match firmware / record_gesture.py / realtime_classify.py)
# ==============================
DATA_CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
CONTROL_CHAR_UUID = "0000ff01-0000-1000-8000-00805f9b34fb"
IMU_CHAR_UUID = "5a153fa9-7be0-400c-8ef8-d84502b31c4d"  # onboard accel, notify-only
DEVICE_NAME_PREFIX = "NPG-Lite-"
SAMPLES_PER_PACKET = 20  # BLOCK_COUNT in firmware
IMU_SAMPLE_LEN = 7       # firmware IMU_SAMPLE_SIZE: 1 counter byte + 3x int16 (ax,ay,az)

ADC_BITS = "12"
NOTCH_TYPE = 1  # 1 = 50Hz mains, 2 = 60Hz mains - must match record_gesture.py setting used for training

# Gesture names the hand animation knows explicit poses for. Any predicted
# class not in this set (or "rest") just falls back to the neutral pose with
# the name still shown as text, so a re-trained model with different class
# names never breaks the UI - it just loses the specific animation for the
# new name until a pose is added for it in POSE_CSS below.
KNOWN_POSES = {"rest", "pinch", "extension", "flexion"}


# ==============================
# Filters - identical logic to record_gesture.py / realtime_classify.py
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
        ch_data = 0.0
        if self.sampling_rate == 500 and exg_type == 4:
            self.x4 = output - (-0.82523238 * self.z1) - (0.29463653 * self.z2)
            output = 0.52996723 * self.x4 + -1.05993445 * self.z1 + 0.52996723 * self.z2
            self.z2 = self.z1
            self.z1 = self.x4
            ch_data = output
        elif self.sampling_rate == 250 and exg_type == 4:
            self.x4 = output - 0.22115344 * self.z1 - 0.18023207 * self.z2
            output = 0.23976966 * self.x4 + -0.47953932 * self.z1 + 0.23976966 * self.z2
            self.z2 = self.z1
            self.z1 = self.x4
            ch_data = output
        return ch_data


# ==============================
# Fixed device identity mapping - MUST match record_gesture.py / realtime_classify.py
# Single band by default (one device). Add more entries here (and in
# record_gesture.py's KNOWN_DEVICES, identically) if you go back to
# multiple boards - the merge order below already generalizes to N devices.
# ==============================
KNOWN_DEVICES = {
    # "DC:1E:D5:80:DA:DA": {"role": "dev1", "channels": 3},
    "DC:1E:D5:80:DA:CE": {"role": "dev2", "channels": 3},  # 2nd band, disabled
}


def resolve_all_known_devices(found):
    resolved = []
    for d in found:
        info = KNOWN_DEVICES.get(d.address.upper())
        if info is not None:
            resolved.append((d, info["role"], info["channels"]))
    resolved.sort(key=lambda t: t[1])
    return resolved


# ==============================
# Shared state between BLE thread, model thread, and the HTTP server thread
# ==============================
class SharedState:
    def __init__(self, num_channels, window_size, stride, classes, threshold):
        self.lock = threading.Lock()
        self.num_channels = num_channels
        self.window_size = window_size
        self.stride = stride
        self.classes = classes
        self.threshold = threshold

        self.sample_buffer = deque(maxlen=window_size)
        self.new_since_last_window = 0

        self.ready_window = None
        self.probs = None            # list[float], aligned with self.classes
        self.pred_class = "rest"
        self.top_prob = 0.0
        self.running = True
        self.status_msg = "connecting..."

    def as_json_dict(self):
        with self.lock:
            probs = self.probs.tolist() if self.probs is not None else [0.0] * len(self.classes)
            return {
                "status": self.status_msg,
                "gesture": self.pred_class,
                "confidence": self.top_prob,
                "threshold": self.threshold,
                "classes": self.classes,
                "probs": probs,
            }


# ==============================
# BLE handling (own thread, own asyncio loop) - unchanged from realtime_classify.py
# ==============================
def ble_thread_main(state, per_device_filters):
    async def run():
        state.status_msg = f"scanning for {DEVICE_NAME_PREFIX}*..."
        devices = await BleakScanner.discover(timeout=6)
        found = [d for d in devices if d.name and d.name.startswith(DEVICE_NAME_PREFIX)]
        if not found:
            state.status_msg = "ERROR: no NPG-Lite device found"
            state.running = False
            return

        resolved = resolve_all_known_devices(found)
        if not resolved:
            state.status_msg = "ERROR: no recognized devices (check KNOWN_DEVICES addresses)"
            state.running = False
            return

        # Total feature count must match what the model was trained on:
        # EMG channels across all devices, PLUS 3 accel axes PER device
        # (see record_gesture.py's merge_and_save - accel is appended
        # after the full EMG block, once per connected device).
        total_emg_channels = sum(ch for _, _, ch in resolved)
        total_channels = total_emg_channels + 3 * len(resolved)
        if total_channels != state.num_channels:
            roles_found = [role for _, role, _ in resolved]
            state.status_msg = (f"ERROR: connected devices {roles_found} provide "
                                 f"{total_emg_channels}ch EMG + {3 * len(resolved)}ch accel = "
                                 f"{total_channels}ch total but model expects "
                                 f"{state.num_channels}ch")
            state.running = False
            return

        role_order = [role for _, role, _ in resolved]
        device_queues = {role: deque() for role in role_order}
        queue_lock = threading.Lock()

        # Sample-and-hold accel state per device, updated by the IMU
        # notification handler below at whatever (slower) rate the IMU
        # actually notifies at.
        latest_accel = {role: np.zeros(3, dtype=np.float32) for role in role_order}
        accel_lock = threading.Lock()

        def make_imu_handler(role):
            def handle_imu_notify(_, data: bytearray):
                if len(data) == 0 or len(data) % IMU_SAMPLE_LEN != 0:
                    return
                n_samples = len(data) // IMU_SAMPLE_LEN
                offset = (n_samples - 1) * IMU_SAMPLE_LEN  # most recent sample in the batch
                ax = int.from_bytes(data[offset + 1:offset + 3], "big", signed=True)
                ay = int.from_bytes(data[offset + 3:offset + 5], "big", signed=True)
                az = int.from_bytes(data[offset + 5:offset + 7], "big", signed=True)
                with accel_lock:
                    latest_accel[role][0] = ax
                    latest_accel[role][1] = ay
                    latest_accel[role][2] = az
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

        clients = []
        for dev, role, channels in resolved:
            state.status_msg = f"connecting to {role} ({dev.address})..."
            client = BleakClient(dev.address)
            await client.connect()
            filters = per_device_filters[role]
            await client.start_notify(
                DATA_CHAR_UUID,
                make_handler(role, channels, filters["notch"], filters["emg"]),
            )
            try:
                await client.start_notify(IMU_CHAR_UUID, make_imu_handler(role))
            except Exception as e:
                print(f"(warning) IMU characteristic not available for {role} "
                      f"({e}) - accel features will stay at 0 for this device.")
            clients.append(client)
        state.status_msg = f"connected: {'+'.join(role_order)}"

        await asyncio.gather(*(c.write_gatt_char(CONTROL_CHAR_UUID, b"STOP", response=True) for c in clients))
        await asyncio.sleep(0.1)
        await asyncio.gather(*(c.write_gatt_char(CONTROL_CHAR_UUID, b"START", response=True) for c in clients))
        state.status_msg = f"streaming: {'+'.join(role_order)}"

        while state.running:
            merged_sample = None
            with queue_lock:
                if all(device_queues[r] for r in role_order):
                    parts = [device_queues[r].popleft() for r in role_order]
                    with accel_lock:
                        accel_parts = [latest_accel[r].copy() for r in role_order]
                    # Order MUST match record_gesture.py's merge_and_save(): every
                    # device's EMG channels first (role order), then every
                    # device's accel_x/y/z (role order) - this is the exact
                    # column order the model was trained on.
                    merged_sample = np.concatenate(parts + accel_parts)

            if merged_sample is None:
                await asyncio.sleep(0.001)
                continue

            with state.lock:
                state.sample_buffer.append(merged_sample)
                state.new_since_last_window += 1
                if (len(state.sample_buffer) == state.window_size
                        and state.new_since_last_window >= state.stride):
                    window = np.array(state.sample_buffer, dtype=np.float32)
                    state.new_since_last_window = 0
                    state.ready_window = window

        for c in clients:
            try:
                await c.write_gatt_char(CONTROL_CHAR_UUID, b"STOP", response=True)
                await c.stop_notify(DATA_CHAR_UUID)
            except Exception:
                pass
            try:
                await c.stop_notify(IMU_CHAR_UUID)
            except Exception:
                pass
            try:
                await c.disconnect()
            except Exception:
                pass

    asyncio.run(run())


# ==============================
# Model inference thread - pulls ready windows and updates state.pred_class
# ==============================
def model_thread_main(state, model, mean, std, threshold):
    while state.running:
        with state.lock:
            window = state.ready_window
            state.ready_window = None

        if window is not None:
            x = (window[np.newaxis, :, :] - mean) / std
            probs = model.predict(x, verbose=0)[0]
            top_idx = int(np.argmax(probs))
            top_prob = float(probs[top_idx])
            with state.lock:
                state.probs = probs
                state.top_prob = top_prob
                state.pred_class = state.classes[top_idx] if top_prob >= threshold else "rest"

        time.sleep(0.02)


# ==============================
# Web dashboard (single self-contained HTML page, served inline - no extra
# static files/folders needed, works the same on any OS)
# ==============================
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Live Gesture Dashboard</title>
<style>
  :root {
    --bg: #0f1117;
    --panel: #171a23;
    --accent: #4fd1c5;
    --accent2: #f6ad55;
    --text: #e6e8ef;
    --muted: #8b90a3;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: radial-gradient(circle at 50% 0%, #1b1f2b, var(--bg));
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 24px 16px 48px;
  }
  h1 { font-weight: 600; font-size: 20px; color: var(--muted); margin-bottom: 4px; letter-spacing: 0.5px; }
  #status { font-size: 13px; color: var(--muted); margin-bottom: 18px; }
  #status.err { color: #f56565; }

  .panel {
    background: var(--panel);
    border-radius: 20px;
    padding: 28px 32px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.4);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 18px;
    width: min(440px, 92vw);
  }

  #handSvg { width: 280px; height: 243px; overflow: visible; }
  #palm, #forearm, #wristConnector, .prox rect, .dist rect {
    fill: url(#skinGrad); stroke: #8a5a3c; stroke-width: 1.5;
  }
  .nail { fill: #f7ddc0; opacity: 0.75; }
  #wristGroup, .prox, .dist, .thumbBase {
    transform-box: view-box;
    transition: transform 0.4s cubic-bezier(.4,0,.2,1);
  }
  #wristGroup { transform-origin: 100px 165px; }
  .prox, .dist { transform-origin: 0px 0px; transform: rotate(0deg); }
  .thumbBase { transform-origin: 0px 0px; transform: translate(95px,155px) rotate(45deg); }
  #handSvg.glow #palm, #handSvg.glow #wristConnector, #handSvg.glow .prox rect, #handSvg.glow .dist rect {
    stroke: var(--accent);
  }
  #handSvg.glow { filter: drop-shadow(0 0 10px rgba(79,209,197,0.35)); }

  #gestureLabel {
    font-size: 30px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--accent);
    transition: color 0.2s;
  }
  #gestureLabel.rest { color: var(--muted); }

  .confWrap { width: 100%; }
  .confLabelRow { display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); margin-bottom: 6px; }
  .confBarBg { width: 100%; height: 14px; background: #2a2e3c; border-radius: 8px; overflow: hidden; }
  .confBarFill { height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent2)); width: 0%; transition: width 0.15s linear; }

  .classBars { width: 100%; display: flex; flex-direction: column; gap: 6px; margin-top: 4px; }
  .classRow { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--muted); }
  .classRow .name { width: 68px; text-transform: capitalize; }
  .classRow .barBg { flex: 1; height: 8px; background: #2a2e3c; border-radius: 6px; overflow: hidden; }
  .classRow .barFill { height: 100%; background: #565d78; width: 0%; transition: width 0.15s linear; }
  .classRow.active .barFill { background: var(--accent); }
  .classRow .pct { width: 34px; text-align: right; }
</style>
</head>
<body>
  <h1>NPG-Lite &middot; Live Gesture Recognition</h1>
  <div id="status">connecting...</div>

  <div class="panel">
    <svg id="handSvg" viewBox="0 0 300 260">
      <defs>
        <linearGradient id="skinGrad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#f3caa2"/>
          <stop offset="100%" stop-color="#dfa679"/>
        </linearGradient>
      </defs>

      <rect id="forearm" x="-40" y="140" width="140" height="50" rx="18"/>

      <g id="wristGroup">
        <circle id="wristConnector" cx="100" cy="165" r="25"/>
        <rect id="palm" x="95" y="100" width="120" height="95" rx="28"/>

        <!-- thumb: base position+swing is itself pose-controlled (class "thumbBase")
             so it can reposition naturally - reaching up for pinch - rather than
             just curling in place like the other fingers. -->
        <g class="thumbBase">
          <g class="prox thumb"><rect x="-8.5" y="-38" width="17" height="38" rx="8.5"/>
            <g transform="translate(0,-38)">
              <g class="dist thumb"><rect x="-6.6" y="-26" width="13.2" height="26" rx="6.6"/>
                <ellipse class="nail" cx="0" cy="-20" rx="3.7" ry="5.7"/>
              </g>
            </g>
          </g>
        </g>

        <!-- index -->
        <g transform="translate(132,100) rotate(-6)">
          <g class="prox index"><rect x="-7.5" y="-44" width="15" height="44" rx="7.5"/>
            <g transform="translate(0,-44)">
              <g class="dist index"><rect x="-5.9" y="-32" width="11.7" height="32" rx="5.9"/>
                <ellipse class="nail" cx="0" cy="-26" rx="3.3" ry="7"/>
              </g>
            </g>
          </g>
        </g>

        <!-- middle -->
        <g transform="translate(157,100) rotate(0)">
          <g class="prox middle"><rect x="-8" y="-54" width="16" height="54" rx="8"/>
            <g transform="translate(0,-54)">
              <g class="dist middle"><rect x="-6.2" y="-38" width="12.5" height="38" rx="6.2"/>
                <ellipse class="nail" cx="0" cy="-31" rx="3.5" ry="8.4"/>
              </g>
            </g>
          </g>
        </g>

        <!-- ring -->
        <g transform="translate(182,100) rotate(6)">
          <g class="prox ring"><rect x="-7.5" y="-46" width="15" height="46" rx="7.5"/>
            <g transform="translate(0,-46)">
              <g class="dist ring"><rect x="-5.9" y="-34" width="11.7" height="34" rx="5.9"/>
                <ellipse class="nail" cx="0" cy="-27" rx="3.3" ry="7.4"/>
              </g>
            </g>
          </g>
        </g>

        <!-- pinky -->
        <g transform="translate(203,100) rotate(14)">
          <g class="prox pinky"><rect x="-6.5" y="-34" width="13" height="34" rx="6.5"/>
            <g transform="translate(0,-34)">
              <g class="dist pinky"><rect x="-5.1" y="-26" width="10.1" height="26" rx="5.1"/>
                <ellipse class="nail" cx="0" cy="-20" rx="2.8" ry="5.7"/>
              </g>
            </g>
          </g>
        </g>
      </g>
    </svg>

    <div id="gestureLabel" class="rest">REST</div>

    <div class="confWrap">
      <div class="confLabelRow"><span>confidence</span><span id="confPct">0%</span></div>
      <div class="confBarBg"><div class="confBarFill" id="confFill"></div></div>
    </div>

    <div class="classBars" id="classBars"></div>
  </div>

<script>
const KNOWN_POSES = new Set(["rest", "pinch", "extension", "flexion"]);
const wristGroup = document.getElementById("wristGroup");
const label = document.getElementById("gestureLabel");
const statusEl = document.getElementById("status");
const confFill = document.getElementById("confFill");
const confPct = document.getElementById("confPct");
const classBarsEl = document.getElementById("classBars");
const handSvg = document.getElementById("handSvg");

let barsBuilt = false;

function applyPose(gesture) {
  const pose = KNOWN_POSES.has(gesture) ? gesture : "rest";
  KNOWN_POSES.forEach(p => wristGroup.classList.remove("pose-" + p));
  wristGroup.classList.add("pose-" + pose);
  label.textContent = gesture;
  label.classList.toggle("rest", pose === "rest");
  handSvg.classList.toggle("glow", pose !== "rest");
}

function buildClassBars(classes) {
  classBarsEl.innerHTML = "";
  classes.forEach(c => {
    const row = document.createElement("div");
    row.className = "classRow";
    row.dataset.cls = c;
    row.innerHTML = `<span class="name">${c}</span><div class="barBg"><div class="barFill"></div></div><span class="pct">0%</span>`;
    classBarsEl.appendChild(row);
  });
  barsBuilt = true;
}

async function poll() {
  try {
    const res = await fetch("/api/state", {cache: "no-store"});
    const data = await res.json();

    statusEl.textContent = data.status;
    statusEl.classList.toggle("err", data.status.startsWith("ERROR"));

    applyPose(data.gesture);

    const pct = Math.round(data.confidence * 100);
    confFill.style.width = pct + "%";
    confPct.textContent = pct + "%";

    if (!barsBuilt) buildClassBars(data.classes);
    data.classes.forEach((c, i) => {
      const row = classBarsEl.querySelector(`.classRow[data-cls="${CSS.escape(c)}"]`);
      if (!row) return;
      const p = Math.round((data.probs[i] || 0) * 100);
      row.querySelector(".barFill").style.width = p + "%";
      row.querySelector(".pct").textContent = p + "%";
      row.classList.toggle("active", c === data.gesture);
    });
  } catch (e) {
    statusEl.textContent = "lost connection to server";
    statusEl.classList.add("err");
  }
  setTimeout(poll, 120);
}
poll();
</script>

<style>
/* ---- Pose definitions ----
   Values were tuned by rendering each pose to a PNG and checking it looks
   like an actual hand before wiring it up here. The thumb's base position
   itself (.thumbBase) is also pose-controlled - not just curl - so it can
   swing across the palm and reach up to actually meet the index fingertip
   for pinch. Fingertip coordinates for pinch were solved numerically so
   the thumb and index tips land within ~8px of each other instead of just
   guessing. */

#wristGroup.pose-extension { transform: rotate(-28deg); }
#wristGroup.pose-flexion { transform: rotate(28deg); }

/* rest: relaxed, slightly curled, thumb at its natural resting angle */
#wristGroup.pose-rest .thumbBase { transform: translate(95px,155px) rotate(45deg); }
#wristGroup.pose-rest .thumb.prox  { transform: rotate(10deg); }
#wristGroup.pose-rest .thumb.dist  { transform: rotate(8deg); }
#wristGroup.pose-rest .index.prox, #wristGroup.pose-rest .middle.prox,
#wristGroup.pose-rest .ring.prox,  #wristGroup.pose-rest .pinky.prox  { transform: rotate(12deg); }
#wristGroup.pose-rest .index.dist, #wristGroup.pose-rest .middle.dist,
#wristGroup.pose-rest .ring.dist,  #wristGroup.pose-rest .pinky.dist  { transform: rotate(15deg); }

/* extension / flexion: fingers stay straight, only the wrist tilts */
#wristGroup.pose-extension .thumbBase, #wristGroup.pose-flexion .thumbBase { transform: translate(95px,155px) rotate(40deg); }
#wristGroup.pose-extension .thumb.prox, #wristGroup.pose-flexion .thumb.prox { transform: rotate(6deg); }
#wristGroup.pose-extension .thumb.dist, #wristGroup.pose-flexion .thumb.dist { transform: rotate(4deg); }
#wristGroup.pose-extension .index.prox, #wristGroup.pose-extension .middle.prox,
#wristGroup.pose-extension .ring.prox,  #wristGroup.pose-extension .pinky.prox,
#wristGroup.pose-flexion .index.prox,   #wristGroup.pose-flexion .middle.prox,
#wristGroup.pose-flexion .ring.prox,    #wristGroup.pose-flexion .pinky.prox,
#wristGroup.pose-extension .index.dist, #wristGroup.pose-extension .middle.dist,
#wristGroup.pose-extension .ring.dist,  #wristGroup.pose-extension .pinky.dist,
#wristGroup.pose-flexion .index.dist,   #wristGroup.pose-flexion .middle.dist,
#wristGroup.pose-flexion .ring.dist,    #wristGroup.pose-flexion .pinky.dist { transform: rotate(0deg); }

/* pinch: thumb's base swings up toward the index so the fingertips actually
   meet (solved so both tips land within ~8px of each other), other three
   fingers stay loosely curled out of the way. */
#wristGroup.pose-pinch .thumbBase { transform: translate(130px,105px) rotate(40deg); }
#wristGroup.pose-pinch .thumb.prox { transform: rotate(30deg); }
#wristGroup.pose-pinch .thumb.dist { transform: rotate(0deg); }
#wristGroup.pose-pinch .index.prox { transform: rotate(55deg); }
#wristGroup.pose-pinch .index.dist { transform: rotate(55deg); }
#wristGroup.pose-pinch .middle.prox, #wristGroup.pose-pinch .ring.prox,
#wristGroup.pose-pinch .pinky.prox  { transform: rotate(16deg); }
#wristGroup.pose-pinch .middle.dist, #wristGroup.pose-pinch .ring.dist,
#wristGroup.pose-pinch .pinky.dist  { transform: rotate(18deg); }
</style>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    state = None  # set by main() before serving

    def log_message(self, fmt, *args):
        pass  # keep the terminal quiet

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/state"):
            body = json.dumps(Handler.state.as_json_dict()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="gesture_model")
    ap.add_argument("--confidence_threshold", type=float, default=0.95,
                     help="minimum top-class probability required before a gesture is "
                          "shown as predicted - below this it shows as 'rest'.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no_browser", action="store_true",
                     help="don't auto-open a browser tab on startup")
    args = ap.parse_args()

    with open(os.path.join(args.model_dir, "meta.json")) as f:
        meta = json.load(f)

    window_size = meta["window_size"]
    stride = meta["stride"]
    num_channels = meta["num_channels"]
    classes = meta["classes"]
    mean = np.array(meta["scaler_mean"], dtype=np.float32)
    std = np.array(meta["scaler_std"], dtype=np.float32)

    unknown = [c for c in classes if c not in KNOWN_POSES]
    if unknown:
        print(f"(note) model has class(es) {unknown} with no dedicated hand pose - "
              f"they'll show correctly as text/confidence bars but the hand will "
              f"just fall back to the neutral 'rest' pose for those.")

    print(f"Loading model from {args.model_dir}...")
    model = keras.models.load_model(os.path.join(args.model_dir, "model.keras"))

    state = SharedState(num_channels, window_size, stride, classes, args.confidence_threshold)

    per_device_filters = {}
    for addr, info in KNOWN_DEVICES.items():
        role = info["role"]
        channels = info["channels"]
        notch_filters, emg_filters = [], []
        for _ in range(channels):
            nf = NotchFilter()
            nf.set_sampling_rate(meta["sampling_rate"])
            notch_filters.append(nf)
            ef = EXGFilter()
            ef.set_bits(ADC_BITS, meta["sampling_rate"])
            emg_filters.append(ef)
        per_device_filters[role] = {"notch": notch_filters, "emg": emg_filters}

    threading.Thread(target=ble_thread_main, args=(state, per_device_filters), daemon=True).start()
    threading.Thread(target=model_thread_main, args=(state, model, mean, std, args.confidence_threshold),
                      daemon=True).start()

    Handler.state = state
    server = ThreadingHTTPServer(("localhost", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"\nDashboard running at {url}")
    print("Press Ctrl+C to stop.\n")

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        server.shutdown()
        print("\nDone.")


if __name__ == "__main__":
    main()
