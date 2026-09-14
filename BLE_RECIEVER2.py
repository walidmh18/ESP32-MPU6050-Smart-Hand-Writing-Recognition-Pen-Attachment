import asyncio
import struct
import csv
import math
import numpy as np
from bleak import BleakScanner, BleakClient

CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
DEVICE_NAME = "SmartPen_ESP32C3"

# --- Tuning Parameters ---
# The threshold energy required to trigger a "stroke started" event.
# You will need to adjust this up or down based on how hard you write.
ENERGY_THRESHOLD = 2000  
# How many consecutive samples below the threshold before we declare the stroke "ended" (20 samples = 200ms)
IDLE_SAMPLES_TO_STOP = 20  

# --- Global State ---
is_recording = False
idle_counter = 0
stroke_buffer = []
stroke_count = 1
current_letter = "A" # Change this before running the script to collect different letters

def calculate_gmag(gx, gy, gz):
    # Calculate the vector magnitude of the 3 gyroscope axes
    return math.sqrt(gx**2 + gy**2 + gz**2)

def save_stroke_to_csv(data_buffer, letter, count):
    filename = f"dataset_{letter}_{count:03d}.csv"
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ax", "ay", "az", "gx", "gy", "gz"])
        writer.writerows(data_buffer)
    print(f"✅ Saved stroke to {filename} ({len(data_buffer)} samples)")

def notification_handler(sender, data: bytearray):
    global is_recording, idle_counter, stroke_buffer, stroke_count

    if len(data) == 60:
        unpacked_data = struct.unpack('<30h', data)
        
        # Process each of the 5 samples in the batch
        for i in range(5):
            sample = unpacked_data[i*6 : (i+1)*6]
            ax, ay, az, gx, gy, gz = sample
            
            # 1. Calculate Gyro Energy
            gmag = calculate_gmag(gx, gy, gz)

            # 2. State Machine Logic
            if not is_recording:
                # Waiting for a stroke to start
                if gmag > ENERGY_THRESHOLD:
                    print("\n🖍️ STROKE STARTED!")
                    is_recording = True
                    idle_counter = 0
                    stroke_buffer = [sample] # Start a new buffer
            else:
                # Currently recording a stroke
                stroke_buffer.append(sample)
                
                if gmag < ENERGY_THRESHOLD:
                    idle_counter += 1
                else:
                    idle_counter = 0 # Reset if movement spikes again

                # 3. Check for stroke end
                if idle_counter >= IDLE_SAMPLES_TO_STOP:
                    print("🛑 STROKE ENDED.")
                    # Only save if the stroke was long enough to be a real letter (e.g., > 40 samples / 400ms)
                    if len(stroke_buffer) > 40:
                        save_stroke_to_csv(stroke_buffer, current_letter, stroke_count)
                        stroke_count += 1
                    else:
                        print("⚠️ Stroke too short, discarded.")
                    
                    is_recording = False
                    stroke_buffer = []

async def main():
    print(f"Scanning for {DEVICE_NAME}...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME)
    
    if not device:
        print("Device not found. Ensure ESP32 is on.")
        return

    print("Connecting...")
    async with BleakClient(device.address) as client:
        print("Connected! Waiting for pen movement...")
        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down.")

if __name__ == "__main__":
    asyncio.run(main())