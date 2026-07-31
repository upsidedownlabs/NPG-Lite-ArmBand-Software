"""
scan_ble.py — diagnostic only. Lists EVERY BLE device in range with its exact
advertised name, then shows which ones the toolkit's DEVICE_NAME_PREFIX would
actually accept.

Run it with the venv python, from the project folder:

    Windows:      venv\\Scripts\\python.exe scan_ble.py
    macOS/Linux:  ./venv/bin/python scan_ble.py
"""

import asyncio

from bleak import BleakScanner

# Must match the constant in record_gesture.py / gesture_ui_server.py /
# gesture_controller.py.
DEVICE_NAME_PREFIX = "NPG-Lite-Band"


async def main():
    print(f"Scanning 8s... (toolkit prefix is '{DEVICE_NAME_PREFIX}')\n")
    devices = await BleakScanner.discover(timeout=8)

    if not devices:
        print("Nothing found at all — that points at the Bluetooth adapter or")
        print("permissions, not at the device names.")
        return

    named = [d for d in devices if d.name]
    unnamed = len(devices) - len(named)

    print(f"{len(devices)} device(s) in range ({unnamed} advertising no name):\n")
    for d in sorted(named, key=lambda x: x.address.upper()):
        match = "MATCH " if d.name.startswith(DEVICE_NAME_PREFIX) else "      "
        print(f"  {match} {d.name!r:40s} {d.address}")

    matches = [d for d in named if d.name.startswith(DEVICE_NAME_PREFIX)]
    print()
    if matches:
        print(f"{len(matches)} device(s) match the prefix — the toolkit should see these.")
    else:
        print(f"NO device matches '{DEVICE_NAME_PREFIX}'.")
        close = [d for d in named if "npg" in d.name.lower()]
        if close:
            print("These look like your boards but advertise a different name:")
            for d in close:
                print(f"    {d.name!r}  ({d.address})")
            print()
            print("Fix one of the two ends so they agree:")
            print("  - set DEVICE_NAME_PREFIX to the common leading part of those")
            print("    names, in record_gesture.py, gesture_ui_server.py and")
            print("    gesture_controller.py, or")
            print("  - reflash the boards with firmware that advertises the")
            print("    'NPG-Lite-Band' name.")


if __name__ == "__main__":
    asyncio.run(main())
