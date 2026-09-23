import asyncio
import struct
import csv
import math
import numpy as np
from bleak import BleakScanner, BleakClient

SERVICE_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
DEVICE_NAME = "SmartPen_ESP32C3"

# --- Tuning Parameters ---
ENERGY_THRESHOLD = 1400         # Gyro threshold to wake up
IDLE_SAMPLES_TO_STOP = 20      # 20 samples (200ms) of quiet gyro to end stroke
EVALUATION_SAMPLES = 15        # Read 150ms of data before deciding if it's a valid stroke

# Variance Tuning (You will need to adjust these for your specific pen weight/speed)
PLANAR_VARIANCE_MIN = 5000    # The minimum acceleration variance required on the desk plane
VERTICAL_RATIO = 0.15          # The vertical bounce variance must be < 15% of the planar variance

# --- Global State Machine ---
state = "IDLE" # States: IDLE, EVALUATING, RECORDING, REJECTED
idle_counter = 0
stroke_buffer = []
stroke_count = 1
current_letter = "A" 

def calculate_gmag(gx, gy, gz):
    return math.sqrt(gx**2 + gy**2 + gz**2)

def is_planar_writing(buffer):
    # Extract only the accelerometer data (first 3 columns)
    accel_data = np.array([row[0:3] for row in buffer])
    
    # Calculate the variance for X, Y, and Z separately
    variances = np.var(accel_data, axis=0) 
    sorted_vars = np.sort(variances)
    
    min_var = sorted_vars[0]       # The axis with the LEAST movement (Vertical bounce)
    max_var_1 = sorted_vars[1]     # The axes with the MOST movement (Desk plane)
    max_var_2 = sorted_vars[2]
    
    # Check 1: Is there enough force to actually be writing?
    is_moving = max_var_1 > PLANAR_VARIANCE_MIN and max_var_2 > PLANAR_VARIANCE_MIN
    
    # Check 2: Is it flat? (Vertical bounce must be a tiny fraction of planar movement)
    is_flat = min_var < (max_var_1 * VERTICAL_RATIO) 
    
    return is_moving and is_flat

def save_stroke_to_csv(data_buffer, letter, count):
    filename = f"dataset_{letter}_{count:03d}.csv"
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ax", "ay", "az", "gx", "gy", "gz"])
        writer.writerows(data_buffer)
    print(f"✅ Saved stroke to {filename} ({len(data_buffer)} samples)\n")

def notification_handler(sender, data: bytearray):
    global state, idle_counter, stroke_buffer, stroke_count

    if len(data) == 60:
        unpacked_data = struct.unpack('<30h', data)
        
        for i in range(5):
            sample = unpacked_data[i*6 : (i+1)*6]
            ax, ay, az, gx, gy, gz = sample
            gmag = calculate_gmag(gx, gy, gz)

            # --- State Machine ---
            if state == "IDLE":
                if gmag > ENERGY_THRESHOLD:
                    state = "EVALUATING"
                    stroke_buffer = [sample]
                    idle_counter = 0
                    print("👀 Movement detected, evaluating plane...")
            
            elif state == "EVALUATING":
                stroke_buffer.append(sample)
                # Wait until we have enough samples to run the math
                if len(stroke_buffer) >= EVALUATION_SAMPLES:
                    if is_planar_writing(stroke_buffer):
                        state = "RECORDING"
                        print("✍️ VALID STROKE (Planar motion confirmed)")
                    else:
                        state = "REJECTED"
                        print("❌ 3D MOTION IGNORED (Pick-up or roll detected)")
                        idle_counter = 0 # Reset idle counter for the cooldown phase
            
            elif state == "RECORDING":
                stroke_buffer.append(sample)
                # Check if movement stopped
                if gmag < ENERGY_THRESHOLD:
                    idle_counter += 1
                else:
                    idle_counter = 0
                    
                if idle_counter >= IDLE_SAMPLES_TO_STOP:
                    print("🛑 STROKE ENDED.")
                    if len(stroke_buffer) > 40:
                        save_stroke_to_csv(stroke_buffer, current_letter, stroke_count)
                        stroke_count += 1
                    else:
                        print("⚠️ Stroke too short, discarded.\n")
                    
                    state = "IDLE"
                    stroke_buffer = []
                    idle_counter = 0

            elif state == "REJECTED":
                # We stay in this state, ignoring data, until the pen stops moving.
                # This prevents a pick-up motion from constantly re-triggering the IDLE state.
                if gmag < ENERGY_THRESHOLD:
                    idle_counter += 1
                else:
                    idle_counter = 0
                    
                if idle_counter >= IDLE_SAMPLES_TO_STOP:
                    state = "IDLE"
                    idle_counter = 0
                    print("🔄 Pen settled. Ready for next movement.\n")

async def main():
    print(f"Scanning for {DEVICE_NAME}...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME, service_uuids=[SERVICE_UUID])
    
    if not device:
        print("Device not found. Ensure ESP32 is on.")
        return

    print("Connecting...")
    async with BleakClient(device.address) as client:
        print("Connected! Waiting for pen movement...\n")
        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down.")

if __name__ == "__main__":
    asyncio.run(main())