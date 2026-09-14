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
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from bleak import BleakScanner, BleakClient

# ──────────────────── Configuration ────────────────────
DEVICE_NAME = "SmartPen_ESP32C3"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
DATASET_DIR = Path("./datasets")
DATASET_DIR.mkdir(exist_ok=True)

BLE_RECONNECT_DELAY = 3          # seconds between reconnect attempts
SAMPLE_AXES = 6                  # ax, ay, az, gx, gy, gz
SAMPLES_PER_PACKET = 5           # 60 bytes / 12 bytes-per-sample
PACKET_SIZE = 60                 # expected BLE notification length

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
        self.current_label: str = ""
        self.buffer: list[tuple[int, ...]] = []
        self.ble_connected: bool = False
        self.websocket: WebSocket | None = None
        self.sample_count: int = 0          # total samples received (lifetime)
        self.recording_sample_count: int = 0  # samples in current recording

state = RecorderState()

# ──────────────────── Helpers ──────────────────────────

async def ws_send(payload: dict) -> None:
    """Send a JSON message to the connected WebSocket client (if any)."""
    if state.websocket is not None:
        try:
            await state.websocket.send_json(payload)
        except Exception:
            state.websocket = None


def save_buffer_to_csv(label: str, buffer: list[tuple[int, ...]]) -> str:
    """Persist the recorded buffer to a CSV file and return the filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dataset_{label}_{timestamp}.csv"
    filepath = DATASET_DIR / filename

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ax", "ay", "az", "gx", "gy", "gz"])
        writer.writerows(buffer)

    log.info("Saved %s (%d samples)", filename, len(buffer))
    return filename

# ──────────────────── BLE Handler ──────────────────────

def on_ble_notification(_sender, data: bytearray) -> None:
    """
    Called by Bleak for every BLE notification.
    Unpacks 5 × 12-byte samples (Little-Endian int16) and optionally buffers.
    """
    if len(data) != PACKET_SIZE:
        return

    unpacked = struct.unpack("<30h", data)

    for i in range(SAMPLES_PER_PACKET):
        sample = unpacked[i * SAMPLE_AXES : (i + 1) * SAMPLE_AXES]
        state.sample_count += 1

        if state.is_recording:
            state.buffer.append(sample)
            state.recording_sample_count += 1


async def ble_connection_loop() -> None:
    """
    Endlessly scans for, connects to, and listens to the SmartPen.
    Reconnects automatically on disconnect.
    """
    while True:
        try:
            log.info("Scanning for %s …", DEVICE_NAME)
            await ws_send({"status": "BLE_SCANNING"})

            device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10)

            if device is None:
                log.warning("Device not found — retrying in %ds", BLE_RECONNECT_DELAY)
                await ws_send({"status": "BLE_NOT_FOUND"})
                await asyncio.sleep(BLE_RECONNECT_DELAY)
                continue

            log.info("Found %s (%s). Connecting …", DEVICE_NAME, device.address)

            async with BleakClient(device.address) as client:
                state.ble_connected = True
                log.info("BLE connected ✓")
                await ws_send({"status": "BLE_CONNECTED"})

                await client.start_notify(CHARACTERISTIC_UUID, on_ble_notification)

                # Keep the connection alive until it drops
                while client.is_connected:
                    # Periodically push a heartbeat with live sample count
                    if state.is_recording:
                        await ws_send({
                            "status": "RECORDING",
                            "samples": state.recording_sample_count,
                        })
                    await asyncio.sleep(0.25)

        except Exception as exc:
            log.error("BLE error: %s", exc)

        # If we reach here, the connection dropped
        state.ble_connected = False
        log.warning("BLE disconnected — reconnecting in %ds", BLE_RECONNECT_DELAY)
        await ws_send({"status": "BLE_DISCONNECTED"})
        await asyncio.sleep(BLE_RECONNECT_DELAY)


# ──────────────────── Lifecycle ────────────────────────

@app.on_event("startup")
async def startup():
    """Launch the BLE listener as a background task."""
    asyncio.create_task(ble_connection_loop())
    log.info("BLE background task started")


# ──────────────────── WebSocket Endpoint ───────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state.websocket = ws
    log.info("WebSocket client connected")

    # Immediately inform the client of the current BLE state
    if state.ble_connected:
        await ws.send_json({"status": "BLE_CONNECTED"})
    else:
        await ws.send_json({"status": "BLE_DISCONNECTED"})

    try:
        while True:
            raw = await ws.receive_text()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"status": "ERROR", "message": "Invalid JSON"})
                continue

            command = msg.get("command", "").upper()

            # ── START recording ──────────────────────────
            if command == "START":
                label = msg.get("label", "UNKNOWN").strip() or "UNKNOWN"
                state.current_label = label
                state.buffer.clear()
                state.recording_sample_count = 0
                state.is_recording = True

                log.info("▶ Recording started  [label=%s]", label)
                await ws.send_json({
                    "status": "RECORDING",
                    "label": label,
                    "samples": 0,
                })

            # ── STOP recording ───────────────────────────
            elif command == "STOP":
                state.is_recording = False
                sample_count = len(state.buffer)

                if sample_count > 0:
                    filename = save_buffer_to_csv(state.current_label, state.buffer)
                    await ws.send_json({
                        "status": "SAVED",
                        "filename": filename,
                        "samples": sample_count,
                        "label": state.current_label,
                    })
                else:
                    log.warning("STOP received but buffer is empty")
                    await ws.send_json({
                        "status": "EMPTY",
                        "message": "No samples recorded",
                    })

                state.buffer.clear()
                state.recording_sample_count = 0
                log.info("■ Recording stopped  [samples=%d]", sample_count)

            # ── PING (keep-alive / health-check) ────────
            elif command == "PING":
                await ws.send_json({
                    "status": "PONG",
                    "ble_connected": state.ble_connected,
                    "is_recording": state.is_recording,
                    "total_samples": state.sample_count,
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
        # Safety: stop any in-progress recording if the client drops
        if state.is_recording:
            state.is_recording = False
            if state.buffer:
                filename = save_buffer_to_csv(state.current_label, state.buffer)
                log.info("Auto-saved on disconnect: %s", filename)
            state.buffer.clear()
        state.websocket = None
