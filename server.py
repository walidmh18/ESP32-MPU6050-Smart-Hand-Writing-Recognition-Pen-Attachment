"""
SmartPen ESP32 — BLE → WebSocket Bridge & Dataset Recorder
──────────────────────────────────────────────────────────
FastAPI backend that:
  1. Connects to the ESP32-C3 IMU pen over BLE (Bleak).
  2. Streams 6-axis IMU data (ax, ay, az, gx, gy, gz) in real time.
  3. Exposes a WebSocket at /ws for the React frontend.
  4. Records data on START/STOP commands and saves to CSV.

Run:  uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import csv
import json
import struct
import logging
import re
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from bleak import BleakScanner, BleakClient

# ──────────────────── Configuration ────────────────────
DEVICE_NAME = "SmartPen_ESP32C3"
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
DATASET_DIR = Path("./datasets")
DATASET_DIR.mkdir(exist_ok=True)

BLE_RECONNECT_DELAY = 3          # seconds between reconnect attempts
SAMPLE_AXES = 6                  # ax, ay, az, gx, gy, gz
SAMPLE_SIZE_BYTES = 12           # 6 × int16 = 12 bytes per sample
CAPTURE_TOTAL_SAMPLES = 500      # ESP32 captures 500 samples @ 500Hz = 1 second per button press
BLE_CHUNK_MAX_SAMPLES = 42       # ESP32 sends at most 42 samples per BLE notification

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("smartpen")

# ──────────────────── FastAPI App ──────────────────────
app = FastAPI(title="SmartPen Dataset Collector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────── Shared State ─────────────────────
class RecorderState:
    """Thread-safe-ish shared state (single event-loop, so no lock needed)."""

    def __init__(self):
        self.is_recording: bool = False
        self.current_label: str = "A"       # Default to 'A' so it's never UNKNOWN
        self.buffer: list[tuple[int, ...]] = []
        self.ble_connected: bool = False
        self.is_connecting: bool = False
        self.websocket: WebSocket | None = None
        self.sample_count: int = 0          # total samples received (lifetime)
        self.recording_sample_count: int = 0  # samples in current recording
        self.client: BleakClient | None = None
        self.connected_device_name: str = ""
        self.connected_device_address: str = ""
        self.connection_task: asyncio.Task | None = None
        # Chunk reassembly state (new chunked protocol)
        self.chunk_buffer: list[tuple[int, ...]] = []  # accumulates samples across BLE chunks
        self.capture_count: int = 0                     # how many captures saved this session

state = RecorderState()

# ──────────────────── Helpers ──────────────────────────

async def ws_send(payload: dict) -> None:
    """Send a JSON message to the connected WebSocket client (if any)."""
    if state.websocket is not None:
        try:
            await state.websocket.send_json(payload)
        except Exception:
            state.websocket = None


def get_next_dataset_info(label: str) -> tuple[Path, str, str, int, int]:
    """
    Computes directory and filename for the next capture of a given letter/label.
    Structure: datasets/<letter>/dataset_<letter>_<id>.csv
    id is sequential: 1, 2, 3, ... based on existing files in that folder.

    Returns:
        (filepath, rel_path, filename, next_id, total_existing_files)
    """
    clean_label = (label or "A").strip()
    if len(clean_label) == 1 and clean_label.isalpha():
        clean_label = clean_label.upper()
    clean_label = re.sub(r'[^\w\-]', '_', clean_label) or "A"

    letter_dir = DATASET_DIR / clean_label
    letter_dir.mkdir(parents=True, exist_ok=True)

    # Search for existing files matching dataset_<letter>_<id>.csv
    pattern = re.compile(rf"^dataset_{re.escape(clean_label)}_(\d+)\.csv$", re.IGNORECASE)
    existing_ids = []
    if letter_dir.exists():
        for entry in letter_dir.iterdir():
            if entry.is_file():
                match = pattern.match(entry.name)
                if match:
                    try:
                        existing_ids.append(int(match.group(1)))
                    except ValueError:
                        pass

    next_id = max(existing_ids, default=0) + 1
    filename = f"dataset_{clean_label}_{next_id}.csv"
    filepath = letter_dir / filename
    rel_path = f"{clean_label}/{filename}"
    return filepath, rel_path, filename, next_id, len(existing_ids)


def save_buffer_to_csv(label: str, buffer: list[tuple[int, ...]]) -> tuple[Path, str, str, int, int]:
    """
    Persist the recorded buffer to datasets/<letter>/dataset_<letter>_<id>.csv.
    Returns: (filepath, rel_path, filename, file_id, count_for_letter)
    """
    filepath, rel_path, filename, file_id, count_before = get_next_dataset_info(label)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ax", "ay", "az", "gx", "gy", "gz"])
        writer.writerows(buffer)

    log.info("Saved %s (ID: %d, %d samples) in folder '%s'", rel_path, file_id, len(buffer), filepath.parent.name)
    return filepath, rel_path, filename, file_id, count_before + 1

# ──────────────────── BLE Handler ──────────────────────

def on_ble_notification(_sender, data: bytearray) -> None:
    """
    Called by Bleak for every BLE notification.

    New chunked protocol:
    - ESP32 sends 100 samples in 3 chunks (42 + 42 + 16)
    - Each chunk is variable-length: N × 12 bytes (6 × int16 LE)
    - When chunk_buffer reaches 100 samples, a capture is complete.
    """
    n_bytes = len(data)

    # Validate: must be a whole number of 12-byte samples
    if n_bytes == 0 or n_bytes % SAMPLE_SIZE_BYTES != 0:
        log.warning("BLE notification with unexpected size: %d bytes (not a multiple of %d)",
                    n_bytes, SAMPLE_SIZE_BYTES)
        return

    n_samples = n_bytes // SAMPLE_SIZE_BYTES
    # Unpack all int16 values: n_samples × 6 values
    fmt = f"<{n_samples * SAMPLE_AXES}h"
    unpacked = struct.unpack(fmt, data)

    for i in range(n_samples):
        sample = unpacked[i * SAMPLE_AXES : (i + 1) * SAMPLE_AXES]
        state.sample_count += 1
        state.chunk_buffer.append(sample)

    log.info("BLE chunk received: %d samples (buffer now %d/%d)",
             n_samples, len(state.chunk_buffer), CAPTURE_TOTAL_SAMPLES)

    # Check if a full capture has been assembled
    if len(state.chunk_buffer) >= CAPTURE_TOTAL_SAMPLES:
        complete_capture = list(state.chunk_buffer[:CAPTURE_TOTAL_SAMPLES])
        state.chunk_buffer.clear()
        state.capture_count += 1

        # Auto-save using the current label (default 'A' if not set)
        label = (state.current_label or "A").strip()
        filepath, rel_path, filename, file_id, total_for_letter = save_buffer_to_csv(label, complete_capture)
        log.info("✅ Capture #%d complete — saved %s (%d samples)",
                 file_id, rel_path, len(complete_capture))

        # Also copy into recording buffer for the frontend
        if state.is_recording:
            state.buffer.extend(complete_capture)
            state.recording_sample_count += len(complete_capture)

        # Notify the frontend
        asyncio.get_event_loop().call_soon_threadsafe(
            asyncio.create_task,
            ws_send({
                "status": "SAVED",
                "filename": filename,
                "rel_path": rel_path,
                "folder": filepath.parent.name,
                "id": file_id,
                "total_for_letter": total_for_letter,
                "samples": len(complete_capture),
                "label": label,
                "capture_number": state.capture_count,
                "message": f"Capture #{file_id} for '{label}' saved: {rel_path}",
            })
        )


async def connect_ble(target: str = DEVICE_NAME) -> None:
    """
    Manually connects to the specified BLE device (by name or MAC address).
    Does NOT loop indefinitely: notifies progress and status to the client.
    """
    if state.ble_connected and state.client and state.client.is_connected:
        await ws_send({
            "status": "BLE_CONNECTED",
            "device_name": state.connected_device_name,
            "device_address": state.connected_device_address,
            "message": f"Already connected to {state.connected_device_name} ({state.connected_device_address})",
        })
        return

    if state.is_connecting:
        await ws_send({
            "status": "BLE_SCANNING",
            "message": "Connection attempt already in progress…",
        })
        return

    state.is_connecting = True
    target_clean = (target or DEVICE_NAME).strip()

    try:
        log.info("Manual BLE connection requested for target: %s", target_clean)
        await ws_send({
            "status": "BLE_SCANNING",
            "message": f"Searching for '{target_clean}'…",
        })

        found_device = None

        # 1. Active scan discovery (matches address, local_name or device name)
        try:
            discovered = await BleakScanner.discover(timeout=5.0, return_adv=True, service_uuids=[SERVICE_UUID])
            for addr, (dev, adv) in discovered.items():
                local_name = adv.local_name or dev.name or ""
                # Match address
                if target_clean.lower() == addr.lower():
                    found_device = dev
                    break
                # Match name
                if target_clean.lower() in local_name.lower():
                    found_device = dev
                    break
        except Exception as scan_err:
            log.warning("BleakScanner.discover error: %s", scan_err)

        # 2. Fallback to find_device_by_name
        if not found_device:
            try:
                found_device = await BleakScanner.find_device_by_name(target_clean, timeout=3.0)
            except Exception as find_err:
                log.warning("find_device_by_name error: %s", find_err)

        if not found_device:
            log.warning("Device '%s' not found.", target_clean)
            state.ble_connected = False
            state.is_connecting = False
            await ws_send({
                "status": "BLE_NOT_FOUND",
                "message": f"Device '{target_clean}' not found. Make sure ESP32 is powered on and in range.",
            })
            return

        device_name = found_device.name or target_clean
        device_addr = found_device.address
        log.info("Found %s (%s). Connecting…", device_name, device_addr)
        await ws_send({
            "status": "BLE_CONNECTING",
            "device_name": device_name,
            "device_address": device_addr,
            "message": f"Found {device_name} ({device_addr}). Connecting…",
        })

        def on_disconnect(_client: BleakClient):
            log.warning("BLE device disconnected (%s)", device_addr)
            state.ble_connected = False
            state.client = None
            asyncio.create_task(ws_send({
                "status": "BLE_DISCONNECTED",
                "message": f"BLE connection to {device_name} lost.",
            }))

        client = BleakClient(device_addr, disconnected_callback=on_disconnect)
        await client.connect(timeout=10.0)

        state.client = client
        state.ble_connected = True
        state.connected_device_name = device_name
        state.connected_device_address = device_addr
        state.is_connecting = False

        log.info("BLE connected successfully to %s ✓", device_name)
        await client.start_notify(CHARACTERISTIC_UUID, on_ble_notification)

        await ws_send({
            "status": "BLE_CONNECTED",
            "device_name": device_name,
            "device_address": device_addr,
            "message": f"Connected to {device_name} ({device_addr})",
        })

        # Keep connection alive loop to stream sample updates if recording
        while state.ble_connected and client.is_connected:
            if state.is_recording:
                await ws_send({
                    "status": "RECORDING",
                    "samples": state.recording_sample_count,
                })
            await asyncio.sleep(0.25)

    except asyncio.CancelledError:
        log.info("BLE connection task cancelled")
    except Exception as exc:
        log.error("BLE connection error: %s", exc)
        state.ble_connected = False
        state.client = None
        await ws_send({
            "status": "BLE_DISCONNECTED",
            "message": f"Connection failed: {str(exc)}",
        })
    finally:
        state.ble_connected = False if (state.client is None or not state.client.is_connected) else state.ble_connected
        state.is_connecting = False


async def disconnect_ble() -> None:
    """Manually disconnects the connected BLE device."""
    log.info("Manual BLE disconnect requested")
    if state.connection_task and not state.connection_task.done():
        state.connection_task.cancel()

    if state.client:
        try:
            if state.client.is_connected:
                await state.client.disconnect()
        except Exception as exc:
            log.warning("Error during BLE disconnect: %s", exc)

    state.client = None
    state.ble_connected = False
    state.is_connecting = False
    log.info("BLE pen disconnected ✓")
    await ws_send({
        "status": "BLE_DISCONNECTED",
        "message": "Pen disconnected manually.",
    })


async def scan_ble() -> None:
    """Scans for nearby BLE devices and returns the list to the WebSocket client."""
    log.info("Scanning for nearby BLE devices…")
    await ws_send({
        "status": "BLE_SCANNING",
        "message": "Scanning for nearby BLE devices (4s)…",
    })
    try:
        discovered = await BleakScanner.discover(timeout=4.0, return_adv=True, service_uuids=[SERVICE_UUID])
        device_list = []
        for addr, (dev, adv) in discovered.items():
            name = dev.name or adv.local_name or "Unknown Device"
            device_list.append({
                "name": name,
                "address": addr,
                "rssi": adv.rssi,
            })
        # Sort devices prioritizing SmartPen or known names, then strongest RSSI
        device_list.sort(
            key=lambda d: (1 if "SmartPen" in d["name"] else 0, d["rssi"]),
            reverse=True,
        )
        await ws_send({
            "status": "BLE_SCAN_RESULTS",
            "devices": device_list,
            "message": f"Found {len(device_list)} BLE devices nearby.",
        })
    except Exception as exc:
        log.error("Scan error: %s", exc)
        await ws_send({
            "status": "ERROR",
            "message": f"BLE Scan failed: {str(exc)}",
        })


# ──────────────────── Lifecycle ────────────────────────

@app.on_event("startup")
async def startup():
    """FastAPI startup: server is ready for manual connections."""
    log.info("SmartPen backend started — ready for manual BLE connection")


# ──────────────────── WebSocket Endpoint ───────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state.websocket = ws
    log.info("WebSocket client connected")

    # Inform client of current BLE state and active letter/folder
    filepath, rel_path, filename, next_id, count = get_next_dataset_info(state.current_label)
    welcome_payload = {
        "device_name": state.connected_device_name,
        "device_address": state.connected_device_address,
        "label": state.current_label,
        "folder": filepath.parent.name,
        "next_id": next_id,
        "next_file": rel_path,
        "total_for_letter": count,
    }
    if state.ble_connected:
        welcome_payload["status"] = "BLE_CONNECTED"
        welcome_payload["message"] = f"Connected to {state.connected_device_name} ({state.connected_device_address})"
    elif state.is_connecting:
        welcome_payload["status"] = "BLE_SCANNING"
        welcome_payload["message"] = "Connecting to pen…"
    else:
        welcome_payload["status"] = "BLE_DISCONNECTED"
        welcome_payload["message"] = "Pen disconnected. Click 'Connect SmartPen' to connect."
    await ws.send_json(welcome_payload)

    try:
        while True:
            raw = await ws.receive_text()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"status": "ERROR", "message": "Invalid JSON"})
                continue

            command = msg.get("command", "").upper()

            # ── CONNECT BLE manually ─────────────────────
            if command == "CONNECT_BLE":
                target = msg.get("target") or msg.get("device_name") or DEVICE_NAME
                if state.connection_task and not state.connection_task.done():
                    state.connection_task.cancel()
                state.connection_task = asyncio.create_task(connect_ble(target))

            # ── DISCONNECT BLE manually ──────────────────
            elif command == "DISCONNECT_BLE":
                await disconnect_ble()

            # ── SCAN BLE devices ─────────────────────────
            elif command == "SCAN_BLE":
                asyncio.create_task(scan_ble())

            # ── SET_LABEL (select active letter / folder) ──
            elif command == "SET_LABEL":
                raw_label = msg.get("label", "").strip() or "A"
                if len(raw_label) == 1 and raw_label.isalpha():
                    clean_label = raw_label.upper()
                else:
                    clean_label = re.sub(r'[^\w\-]', '_', raw_label) or "A"
                state.current_label = clean_label
                filepath, rel_path, filename, next_id, count = get_next_dataset_info(clean_label)
                log.info("Letter label set to '%s' (Folder: datasets/%s/, Next File: %s)",
                         clean_label, filepath.parent.name, rel_path)
                await ws.send_json({
                    "status": "LABEL_UPDATED",
                    "label": clean_label,
                    "folder": filepath.parent.name,
                    "next_id": next_id,
                    "next_file": rel_path,
                    "total_for_letter": count,
                    "message": f"Active letter: '{clean_label}' (Folder: datasets/{clean_label}/, Next: {filename})",
                })

            # ── START recording (arm for capture) ────────
            elif command == "START":
                raw_label = msg.get("label", state.current_label or "A").strip() or "A"
                if len(raw_label) == 1 and raw_label.isalpha():
                    clean_label = raw_label.upper()
                else:
                    clean_label = re.sub(r'[^\w\-]', '_', raw_label) or "A"
                state.current_label = clean_label
                state.buffer.clear()
                state.chunk_buffer.clear()
                state.recording_sample_count = 0
                state.is_recording = True
                filepath, rel_path, filename, next_id, count = get_next_dataset_info(clean_label)

                log.info("▶ Armed for capture [letter=%s] (next: %s)", clean_label, rel_path)
                await ws.send_json({
                    "status": "RECORDING",
                    "label": clean_label,
                    "folder": filepath.parent.name,
                    "next_id": next_id,
                    "next_file": rel_path,
                    "samples": 0,
                    "message": f"Armed for '{clean_label}' (will save to {rel_path}). Press the pen button to capture.",
                })

            # ── STOP recording (disarm) ──────────────────
            elif command == "STOP":
                state.is_recording = False
                sample_count = len(state.buffer)

                if sample_count > 0:
                    await ws.send_json({
                        "status": "SAVED",
                        "samples": sample_count,
                        "label": state.current_label,
                        "message": f"Session ended. {sample_count} samples captured across {state.capture_count} capture(s).",
                    })
                else:
                    await ws.send_json({
                        "status": "EMPTY",
                        "message": "Disarmed. No captures were made.",
                    })

                state.buffer.clear()
                state.chunk_buffer.clear()
                state.recording_sample_count = 0
                log.info("■ Disarmed  [samples=%d]", sample_count)

            # ── PING (keep-alive / health-check) ────────
            elif command == "PING":
                filepath, rel_path, filename, next_id, count = get_next_dataset_info(state.current_label)
                await ws.send_json({
                    "status": "PONG",
                    "ble_connected": state.ble_connected,
                    "is_recording": state.is_recording,
                    "total_samples": state.sample_count,
                    "device_name": state.connected_device_name,
                    "device_address": state.connected_device_address,
                    "label": state.current_label,
                    "folder": filepath.parent.name,
                    "next_id": next_id,
                    "next_file": rel_path,
                    "total_for_letter": count,
                })

            else:
                await ws.send_json({
                    "status": "ERROR",
                    "message": f"Unknown command: {command}",
                })

    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")
    except Exception as exc:
        log.error("WebSocket error: %s", exc)
    finally:
        # Disarm if in-progress recording when client drops
        if state.is_recording:
            state.is_recording = False
            state.buffer.clear()
            state.chunk_buffer.clear()
        state.websocket = None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

