import asyncio
import struct
import csv
from bleak import BleakScanner, BleakClient

CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
DEVICE_NAME = "SmartPen_ESP32C3"

# Create and open the CSV dataset file
csv_file = open("dataset_letter_A.csv", "w", newline='')
writer = csv.writer(csv_file)
# Write the header row
writer.writerow(["ax", "ay", "az", "gx", "gy", "gz"])

def notification_handler(sender, data: bytearray):
    if len(data) == 60:
        # '<30h' unpacks 60 bytes into 30 Little-Endian signed 16-bit integers
        unpacked_data = struct.unpack('<30h', data)
        
        for i in range(5):
            sample = unpacked_data[i*6 : (i+1)*6]
            writer.writerow(sample)
            print(f"Logged: {sample}")

async def main():
    print(f"Scanning for {DEVICE_NAME}...")
    
    # 1. Automatically scan for the ESP32 by its advertised name
    device = await BleakScanner.find_device_by_name(DEVICE_NAME)
    
    if device is None:
        print(f"Error: Could not find any BLE device named '{DEVICE_NAME}'.")
        print("Make sure the ESP32 is powered on and near the laptop.")
        return

    print(f"Found {DEVICE_NAME} at address {device.address}")
    print("Connecting...")
    
    # 2. Connect using the discovered address
    async with BleakClient(device.address) as client:
        print("Connected successfully!")
        
        # 3. Subscribe to the notifications
        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
        print("Recording data to dataset_letter_A.csv... Press Ctrl+C to stop.")
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nRecording stopped. CSV file saved.")
            
        finally:
            # Ensure the file closes cleanly when stopping
            csv_file.close()

if __name__ == "__main__":
    asyncio.run(main())