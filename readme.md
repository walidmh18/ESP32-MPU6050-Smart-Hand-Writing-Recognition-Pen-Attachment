# 🖊️ ESP32 SmartPen — Air-Writing Recognition System

## Complete Project Recap

> **An end-to-end IoT + Machine Learning system that recognizes handwritten characters
> drawn in the air using a 6-axis IMU sensor mounted on a pen, classified by a 1D-CNN + LSTM
> hybrid neural network, and displayed in real time on a web dashboard.**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Hardware Layer — ESP32-C3 Firmware](#3-hardware-layer--esp32-c3-firmware)
4. [Data Pipeline — Dataset Collection & Preprocessing](#4-data-pipeline--dataset-collection--preprocessing)
5. [Machine Learning — Model Architecture & Training](#5-machine-learning--model-architecture--training)
6. [Inference Engine — Live Prediction Module](#6-inference-engine--live-prediction-module)
7. [Backend Server — BLE ↔ WebSocket Bridge](#7-backend-server--ble--websocket-bridge)
8. [Frontend Dashboard — React Cockpit UI](#8-frontend-dashboard--react-cockpit-ui)
9. [Full Data Flow — Button Press to Prediction](#9-full-data-flow--button-press-to-prediction)
10. [File Structure Reference](#10-file-structure-reference)
11. [How to Run the Project](#11-how-to-run-the-project)

---

## 1. Project Overview

### What It Does
The SmartPen system lets a user **draw characters in the air** by moving a pen equipped with
an MPU6050 inertial measurement unit (IMU). The pen captures 1 second of motion data at 500 Hz
(producing a 500 × 6 matrix of accelerometer + gyroscope readings), transmits it wirelessly
over Bluetooth Low Energy (BLE) to a Python backend, and a trained neural network classifies
the gesture in real time.

### Key Technologies
| Layer | Technology |
|-------|-----------|
| **Microcontroller** | ESP32-C3 SuperMini (RISC-V, BLE 5.0) |
| **Sensor** | MPU6050 — 6-axis IMU (3-axis accelerometer + 3-axis gyroscope) |
| **Firmware** | Arduino framework via PlatformIO |
| **BLE Communication** | ESP32 BLE stack → Bleak (Python async BLE client) |
| **Backend** | FastAPI + Uvicorn (WebSocket server) |
| **ML Framework** | PyTorch (1D-CNN + LSTM hybrid) |
| **Preprocessing** | scikit-learn `StandardScaler` (Z-score normalization) |
| **Frontend** | React 19 + Vite + TypeScript |

### Current Classes
The model is currently trained on **2 classes**: digits `0` and `1`.
The dataset contains **60 samples** for class `0` and **63 samples** for class `1` (123 total).
The architecture is fully extensible — adding new characters only requires collecting more data
in new subfolders and retraining.

---

## 2. System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    SYSTEM ARCHITECTURE OVERVIEW                    │
└────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────┐        BLE Notifications         ┌────────────────────────────┐
  │   ESP32-C3 + MPU6050│ ──────────────────────────────▶  │  Python Backend (server.py)│
  │                     │  12 bytes/sample × 42 samples    │  FastAPI + Bleak           │
  │  • 500 Hz sampling  │  per BLE chunk (504 bytes)       │                            │
  │  • Button trigger   │  12 chunks per capture           │  • BLE ↔ WebSocket bridge  │
  │  • Gyro calibration │  (42+42+42+...+8 = 500 samples) │  • Mode: COLLECT / PREDICT │
  │  • Blue LED feedback│                                  │  • CSV auto-save           │
  └─────────────────────┘                                  │  • ML inference trigger    │
                                                           └──────────┬─────────────────┘
                                                                      │
                                                             WebSocket (ws://localhost:8000/ws)
                                                                      │
                                                                      ▼
  ┌─────────────────────┐                                  ┌────────────────────────────┐
  │  ML Pipeline (ML/)  │◀─── imported by server.py ────── │  React Dashboard (page.tsx)│
  │                     │                                  │                            │
  │  • dataset.py       │                                  │  • 2-column cockpit layout │
  │  • model.py         │                                  │  • Mode toggle switch      │
  │  • train.py         │                                  │  • Prediction hero display │
  │  • inference.py     │                                  │  • History feed            │
  │  • predict.py       │                                  │  • BLE control panel       │
  └─────────────────────┘                                  └────────────────────────────┘
```

---

## 3. Hardware Layer — ESP32-C3 Firmware

**File:** `firmware/src/main.cpp`  
**Config:** `firmware/platformio.ini`

### Hardware Setup
- **Board:** LOLIN C3 Mini (ESP32-C3, RISC-V single-core, 4MB flash)
- **Sensor:** MPU6050 on I2C (SDA = GPIO 4, SCL = GPIO 3, 400 kHz Fast Mode)
- **Button:** Analog read on GPIO 0 (threshold > 3000 = pressed), with 10 ms debounce
- **VCC Pin:** GPIO 1 (held HIGH to power external peripherals)
- **LED:** Built-in blue LED on GPIO 8 (active-LOW: LOW = ON, HIGH = OFF)

### MPU6050 Configuration
```
DLPF Mode: 3 → Gyro output rate = 1 kHz, bandwidth = 42 Hz
Sample Rate Divider: 1 → 1000 / (1+1) = 500 Hz
```
This gives us **500 samples per second** with 42 Hz low-pass filtering to remove high-frequency noise.

### Gyroscope Calibration
On startup, the firmware samples 500 gyroscope readings while the pen is stationary and
computes the average bias for each axis (gx, gy, gz). During recording, this bias is
subtracted from every live reading to eliminate gyro drift:
```cpp
recordedSamples[i].gx = raw_gx - gyro_bias_x;
recordedSamples[i].gy = raw_gy - gyro_bias_y;
recordedSamples[i].gz = raw_gz - gyro_bias_z;
```

### Capture Flow (Button Press → BLE Transmission)
1. User presses the button → `onButtonPress()` is called
2. A **500 ms delay** lets the user's hand stabilize after the button press
3. The **blue LED turns ON** — this tells the user to start drawing
4. A FreeRTOS task (`imuSamplingTask`) on Core 0 reads the IMU at 500 Hz
5. After **500 samples** (exactly 1 second), the LED turns OFF
6. The 500 samples are sent over BLE in chunks of **42 samples** each (504 bytes per chunk)
7. Last chunk may be smaller (e.g., 500 mod 42 = 8 samples in the final chunk)

### BLE Protocol
```
Data format per sample: 6 × int16_t (little-endian) = 12 bytes
                        [ax, ay, az, gx, gy, gz]

Chunk size:             42 samples × 12 bytes = 504 bytes (fits BLE MTU of 512)
Chunks per capture:     ceil(500 / 42) = 12 chunks
Inter-chunk delay:      10 ms
```

| BLE Property | Value |
|--------------|-------|
| Device Name | `SmartPen_ESP32C3` |
| Service UUID | `4fafc201-1fb5-459e-8fcc-c5c9c331914b` |
| Characteristic UUID | `beb5483e-36e1-4688-b7f5-ea07361b26a8` |
| MTU | 512 bytes |
| Characteristic Property | NOTIFY |

---

## 4. Data Pipeline — Dataset Collection & Preprocessing

**File:** `ML/dataset.py`

### Dataset Structure on Disk
```
datasets/
├── 0/
│   ├── dataset_0_1.csv      (500 rows × 6 columns)
│   ├── dataset_0_2.csv
│   └── ...                  (60 files total)
├── 1/
│   ├── dataset_1_1.csv
│   ├── dataset_1_2.csv
│   └── ...                  (63 files total)
├── 2/                       (additional classes ready for expansion)
├── A/
└── V/
```

Each CSV file contains exactly **500 rows** and **6 columns**: `ax, ay, az, gx, gy, gz`.
Every row represents one 2 ms sample of 6-axis IMU data.

### SmartPenDataset Class
The `SmartPenDataset` PyTorch `Dataset` class handles the entire preprocessing pipeline:

1. **Discovery:** Scans subdirectories of the datasets root, builds a deterministic `label_map`
   (digits first sorted ascending, then letters sorted ascending)
2. **Loading:** Reads every CSV, validates columns, and ensures each file has exactly 500 rows
   (truncates if too long, zero-pads if too short)
3. **Stacking:** Combines all samples into a single NumPy array of shape `(N, 500, 6)`
4. **Z-score Normalization:** Fits a `sklearn.preprocessing.StandardScaler` on the flattened
   `(N×500, 6)` matrix so each of the 6 sensor channels gets its own mean and standard deviation
5. **Tensor Output:** `__getitem__` transposes `(500, 6)` → `(6, 500)` for PyTorch `Conv1d`

### get_dataloaders() Helper
```python
train_loader, val_loader, scaler = get_dataloaders(
    root_dir="../datasets",
    batch_size=16,
    val_split=0.2,
)
```
- Creates an 80/20 train/val split with a **fixed random seed** (42) for reproducibility
- Returns the fitted `StandardScaler` — this is critical for inference to apply the
  exact same normalization

---

## 5. Machine Learning — Model Architecture & Training

### Model Architecture
**File:** `ML/model.py`

The `SmartPenModel` is a **1D-CNN + LSTM hybrid** designed for time-series classification:

```
Input: (Batch, 6, 500)  — 6 IMU channels, 500 timesteps

═══════════════════════════════════════════════════════════
  LAYER                    SHAPE TRANSFORMATION
═══════════════════════════════════════════════════════════
  Conv1d Block 1           (B, 6, 500)   → (B, 32, 125)
    Conv1d(6→32, k=7, p=3)
    BatchNorm1d(32)
    ReLU
    MaxPool1d(k=4, s=4)    500 → 125 timesteps

  Conv1d Block 2           (B, 32, 125)  → (B, 64, 31)
    Conv1d(32→64, k=5, p=2)
    BatchNorm1d(64)
    ReLU
    MaxPool1d(k=4, s=4)    125 → 31 timesteps

  Conv1d Block 3           (B, 64, 31)   → (B, 128, 7)
    Conv1d(64→128, k=3, p=1)
    BatchNorm1d(128)
    ReLU
    MaxPool1d(k=4, s=4)    31 → 7 timesteps

  ─── Dimensionality Bridge: permute(0, 2, 1) ───
                             (B, 128, 7)  → (B, 7, 128)

  LSTM                     (B, 7, 128)   → h_n: (2, B, 128)
    input_size=128
    hidden_size=128
    num_layers=2
    batch_first=True
    dropout=0.5

  Classifier Head          (B, 128)      → (B, num_classes)
    Linear(128 → 64)
    ReLU
    Dropout(0.5)
    Linear(64 → num_classes)
═══════════════════════════════════════════════════════════

Output: (Batch, num_classes)  — raw logits (unnormalized)
```

**Design rationale:**
- **CNN blocks** extract local spatial-temporal features from the raw signal and aggressively
  downsample from 500 timesteps → 7, reducing computational cost for the LSTM
- **LSTM** captures long-range temporal dependencies across the compressed feature map
- Only the **final hidden state** (`h_n[-1]`) is used for classification, representing the
  completed stroke as a fixed-length vector
- **Dropout (0.5)** before classification prevents overfitting on a small dataset

### Training Script
**File:** `ML/train.py`

#### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Batch Size | 16 |
| Epochs | 50 |
| Learning Rate | 0.001 |
| Num Classes | 2 |
| Validation Split | 20% |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |

#### Training Loop
For each epoch:
1. **Train phase:** `model.train()` → iterate through `train_loader`, forward pass → loss →
   `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`
2. **Validation phase:** `model.eval()` with `torch.no_grad()` → compute val loss and accuracy
3. **Checkpointing:** If `val_loss < best_val_loss`, save model to `checkpoints/best_model.pth`

The training script outputs a formatted table showing epoch-by-epoch progress:
```
  Epoch   Train Loss    Val Loss   Val Acc      Status
  ─────   ───────────   ─────────   ────────   ──────────
      1    0.693147    0.685432    55.00%       * saved
      2    0.651234    0.612345    70.00%       * saved
     ...
```

### Testing & Validation Scripts
- `ML/test_model.py` — Runs inference on **every sample** in the dataset and reports per-class and overall accuracy
- `ML/predict.py` — CLI tool to predict a single CSV file: `python predict.py ../datasets/0/dataset_0_1.csv`

---

## 6. Inference Engine — Live Prediction Module

**File:** `ML/inference.py`

### SmartPenPredictor Class
The `SmartPenPredictor` is designed to be **instantiated once** and reused for every prediction,
avoiding the overhead of reloading the model for each request.

#### Initialization (`__init__`)
1. **Device selection:** CUDA → MPS → CPU (automatic)
2. **Model loading:** Instantiates `SmartPenModel(num_classes=2)`, loads `best_model.pth`
   weights with `torch.load(map_location=device)`, sets `model.eval()`
3. **Scaler extraction:** Creates a dummy `SmartPenDataset(root_dir=datasets)` to re-fit the
   `StandardScaler` with the exact same training statistics — this ensures live inference
   normalization is identical to training normalization
4. **Label map:** Extracts the dataset's `label_map` and creates a reverse mapping
   (`{0: '0', 1: '1'}`)

#### Prediction Flow (`predict()`)
```python
result = predictor.predict(raw_data)  # raw_data: np.ndarray (500, 6)
# Returns: {"prediction": "0", "confidence": 0.9876, "probabilities": {"0": 0.9876, "1": 0.0124}}
```

1. **Input validation:** Checks shape is `(N, 6)`
2. **Length standardization:** Truncates or zero-pads to exactly 500 timesteps
3. **Z-score normalization:** Applies the fitted `StandardScaler.transform()`
4. **Tensor conversion:** `(500, 6)` → `(1, 500, 6)` → `permute` → `(1, 6, 500)`
5. **Forward pass:** `torch.no_grad()` → `model(tensor)` → `F.softmax(logits, dim=1)`
6. **Result extraction:** Returns predicted label, top confidence, and full probability distribution

### Path Resolution
The module uses `Path(__file__).resolve().parent` to dynamically locate:
- `checkpoints/best_model.pth` relative to `inference.py`
- `datasets/` by checking parent directories

This makes it work correctly whether called from the `ML/` directory directly or imported by
`server.py` from the project root.

---

## 7. Backend Server — BLE ↔ WebSocket Bridge

**File:** `server.py`

### Overview
The FastAPI server is the central hub connecting hardware (BLE), ML (inference), and UI (WebSocket):

```
ESP32-C3 ──BLE──▶ Bleak ──▶ server.py ──WebSocket──▶ React Dashboard
                             ├─ COLLECT mode → save CSV
                             └─ PREDICT mode → run inference → send result
```

### Operational Modes

#### Data Collection Mode (`COLLECT`)
When a BLE capture arrives (500 samples assembled from chunks):
1. Auto-saves to `datasets/<label>/dataset_<label>_<id>.csv`
2. Sequential numbering: scans existing files to find the next available ID
3. Sends a `SAVED` status to the frontend with file details

#### Live Prediction Mode (`PREDICT`)
When a BLE capture arrives:
1. Converts the 500-sample buffer to a NumPy array
2. Calls `SmartPenPredictor.predict()` to get the classification
3. Sends a `PREDICTION_RESULT` status to the frontend with:
   - Predicted label, confidence score, full probability distribution
   - Timestamp, capture number, formatted message

### BLE Chunk Reassembly
The ESP32 sends 500 samples across multiple BLE notifications (42 samples per chunk).
The server's `on_ble_notification()` callback:
1. Validates the notification is a valid multiple of 12 bytes
2. Unpacks `int16` values using `struct.unpack`
3. Appends samples to `state.chunk_buffer`
4. When `chunk_buffer` reaches 500 samples → processes the complete capture

### WebSocket Commands

| Command | Description |
|---------|-------------|
| `CONNECT_BLE` | Scan for and connect to the SmartPen device |
| `DISCONNECT_BLE` | Manually disconnect the BLE connection |
| `SCAN_BLE` | List nearby BLE devices |
| `SET_MODE` | Toggle between `COLLECT` and `PREDICT` modes |
| `SET_LABEL` | Set the active class label for data collection |
| `START` | Arm the system for the next pen button press |
| `STOP` | Disarm / stop recording |
| `TEST_PREDICTION` | Run inference on an existing dataset CSV file |
| `PING` | Health check / keep-alive |

### Lazy Model Loading
The `SmartPenPredictor` is loaded lazily on first use via `get_predictor()` to avoid
slowing down server startup. When switching to PREDICT mode, the model is pre-warmed
in a background thread.

---

## 8. Frontend Dashboard — React Cockpit UI

**File:** `page.tsx`

### Layout
The dashboard uses a **2-column cockpit layout** designed to fit entirely within
`100vh` (no page scrolling):

```
┌─────────────────────────────────────────────────────┐
│  Header: SmartPen Dashboard + Mode Toggle           │
├───────────────────────────┬─────────────────────────┤
│  LEFT COLUMN              │  RIGHT COLUMN           │
│                           │                         │
│  • BLE Connection Panel   │  DATA COLLECTION MODE:  │
│  • Device Scanner         │  • Label selector       │
│  • Connection status      │  • Quick letter/digit   │
│                           │    buttons              │
│  • Activity Log           │  • File info panel      │
│    (scrollable)           │  • Capture controls     │
│                           │                         │
│                           │  LIVE PREDICTION MODE:  │
│                           │  • Prediction Hero Orb  │
│                           │    (large glowing char) │
│                           │  • Softmax probability  │
│                           │    distribution bars    │
│                           │  • Recent Predictions   │
│                           │    History (scrollable) │
│                           │  • Test prediction btn  │
└───────────────────────────┴─────────────────────────┘
```

### Key UI Features

1. **Mode Toggle Switch:** Switches between Data Collection and Live Prediction modes.
   Sends `SET_MODE` command to the server.

2. **BLE Connection Panel:** Scan → select device → connect. Shows connection status with
   color-coded indicators (green = connected, yellow = scanning, red = error).

3. **Prediction Hero Display:** A large, animated orb showing the predicted character with
   a glow effect. Displays confidence percentage and full softmax probability bars for
   each class.

4. **Recent Predictions History:** A compact, scrollable feed showing past predictions with
   timestamps, confidence scores, and result labels.

5. **Data Collection Controls:** Label selector with quick-access letter/digit buttons,
   file path preview, and capture count per class.

### WebSocket Integration
The frontend maintains a persistent WebSocket connection to `ws://localhost:8000/ws`:
- Auto-reconnects on disconnect (3-second delay)
- Sends periodic PING heartbeats (every 5 seconds)
- Handles all status message types and updates the UI reactively

---

## 9. Full Data Flow — Button Press to Prediction

Here is the complete journey of a single air-writing gesture through the entire system:

```
STEP 1: HARDWARE CAPTURE
─────────────────────────
User presses button on pen
  → 500ms stabilization delay
  → Blue LED ON (start drawing!)
  → MPU6050 samples at 500 Hz for 1 second
  → 500 samples captured: each is [ax, ay, az, gx, gy, gz] as int16
  → Blue LED OFF (done!)
  → Gyro bias subtracted from gx, gy, gz

STEP 2: BLE TRANSMISSION
─────────────────────────
ESP32 sends 500 samples in BLE chunks:
  Chunk 1: samples[0..41]    → 42 × 12 = 504 bytes
  Chunk 2: samples[42..83]   → 42 × 12 = 504 bytes
  ...
  Chunk 12: samples[462..499] → 38 × 12 = 456 bytes
  (10ms delay between chunks)

STEP 3: SERVER REASSEMBLY
─────────────────────────
server.py receives BLE notifications via Bleak:
  → struct.unpack decodes int16 little-endian values
  → Appends to chunk_buffer
  → When chunk_buffer reaches 500 samples → capture complete!

STEP 4: ML INFERENCE (PREDICT mode)
────────────────────────────────────
  → raw_arr = np.array(buffer, dtype=float32)     # (500, 6)
  → predictor.predict(raw_arr):
      → StandardScaler.transform(raw_arr)          # Z-score normalization
      → torch.Tensor → permute to (1, 6, 500)     # Conv1d format
      → model(tensor) → logits (1, 2)
      → F.softmax(logits) → probabilities
      → argmax → predicted class + confidence

STEP 5: RESULT DELIVERY
────────────────────────
  → server sends WebSocket message:
      {"status": "PREDICTION_RESULT",
       "prediction": "0",
       "confidence": 0.9876,
       "probabilities": {"0": 0.9876, "1": 0.0124}}
  → React dashboard updates:
      → Hero orb shows "0" with glowing animation
      → Probability bars update
      → Entry added to history feed
```

---

## 10. File Structure Reference

```
stage cdta/
│
├── firmware/                          # ESP32-C3 Firmware (PlatformIO)
│   ├── platformio.ini                 # Board: lolin_c3_mini, lib: MPU6050
│   └── src/
│       └── main.cpp                   # IMU sampling, BLE, button logic
│
├── ML/                                # Machine Learning Pipeline
│   ├── dataset.py                     # SmartPenDataset, get_dataloaders()
│   ├── model.py                       # SmartPenModel (1D-CNN + LSTM)
│   ├── train.py                       # Training loop with validation
│   ├── inference.py                   # SmartPenPredictor class
│   ├── predict.py                     # CLI: single-file prediction
│   ├── test_model.py                  # Full dataset accuracy test
│   ├── verify_data_quality.py         # Dataset QA checks
│   └── checkpoints/
│       └── best_model.pth             # Trained model weights
│
├── datasets/                          # Collected IMU data
│   ├── 0/                             # 60 CSV files (digit "0")
│   │   ├── dataset_0_1.csv
│   │   └── ...
│   ├── 1/                             # 63 CSV files (digit "1")
│   │   ├── dataset_1_1.csv
│   │   └── ...
│   ├── 2/                             # (ready for future expansion)
│   ├── A/
│   └── V/
│
├── server.py                          # FastAPI + BLE + WebSocket bridge
├── page.tsx                           # React dashboard (cockpit UI)
├── index.html                         # Vite HTML entry point
├── vite.config.ts                     # Vite build configuration
├── package.json                       # Node.js dependencies
├── tsconfig.json                      # TypeScript configuration
│
├── BLE_RECIEVER.py                    # Early BLE receiver prototypes
├── BLE_RECIEVER2.py                   # (iterative development)
├── BLE_RECIEVER3.py
├── BLE_RECIEVE_THEN_DRAW.py           # BLE + matplotlib visualization
├── organize_existing_datasets.py      # Dataset organization utility
└── dataset_letter_A.csv               # Legacy single-file dataset
```

---

## 11. How to Run the Project

### Prerequisites
- Python 3.10+ with `pip`
- Node.js 18+ with `npm`
- PlatformIO (for firmware upload)
- ESP32-C3 SuperMini with MPU6050 wired and button connected

### Step 1: Flash the Firmware
```bash
cd firmware
pio run --target upload
pio device monitor        # verify serial output
```

### Step 2: Train the Model (if not already trained)
```bash
cd ML
python train.py
# → Saves best_model.pth to ML/checkpoints/
```

### Step 3: Start the Backend Server
```bash
# From project root
pip install fastapi uvicorn bleak torch numpy pandas scikit-learn
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### Step 4: Start the Frontend
```bash
# From project root
npm install
npm run dev
# → Opens at http://localhost:5173
```

### Step 5: Use the Dashboard
1. Click **"Connect SmartPen"** to discover and connect to the ESP32 over BLE
2. **Data Collection Mode:** Select a label, press the pen button to capture gestures
3. **Live Prediction Mode:** Toggle to Predict, press the pen button, see real-time results!

---

> **Built with ESP32-C3, MPU6050, PyTorch, FastAPI, and React.**
