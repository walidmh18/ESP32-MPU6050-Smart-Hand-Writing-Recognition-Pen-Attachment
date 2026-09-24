#include "I2Cdev.h"
#include "MPU6050.h"
#include "Wire.h"
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>



MPU6050 mpu;


#define VCC_PIN 1
#define BUTTON_PIN 0



int16_t gyro_bias_x = 0;
int16_t gyro_bias_y = 0;
int16_t gyro_bias_z = 0;

void calibrateGyro() {
  Serial.println("Starting Gyro Calibration...");
  Serial.println("DO NOT MOVE THE PEN.");
  delay(2000); // Give the user 2 seconds to set the pen down

  int32_t sum_gx = 0;
  int32_t sum_gy = 0;
  int32_t sum_gz = 0;
  int num_samples = 500;

  // Temporary variables to hold readings during the loop
  int16_t ax, ay, az, gx, gy, gz;

  for (int i = 0; i < num_samples; i++) {
    // Read the raw values
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    // Accumulate the gyroscope readings
    sum_gx += gx;
    sum_gy += gy;
    sum_gz += gz;

    // Small delay to ensure we are actually getting new samples 
    // (matches your ~100Hz target rate)
    delay(10); 
  }

  // Calculate the average bias
  gyro_bias_x = sum_gx / num_samples;
  gyro_bias_y = sum_gy / num_samples;
  gyro_bias_z = sum_gz / num_samples;

  Serial.println("Calibration Complete!");
  // Serial.printf("Bias X: %d | Bias Y: %d | Bias Z: %d\n", gyro_bias_x, gyro_bias_y, gyro_bias_z);
  delay(1000); // Brief pause before starting the main program
}



// Use generic UUIDs or generate custom ones online
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

BLEServer* pServer = NULL;
BLECharacteristic* pCharacteristic = NULL;
bool deviceConnected = false;

struct __attribute__((packed)) ImuSample {
  int16_t ax, ay, az;
  int16_t gx, gy, gz;
};

#define CAPTURE_SAMPLES 100
#define BLE_CHUNK_SIZE 42

ImuSample recordedSamples[CAPTURE_SAMPLES];
uint16_t capturedSampleCount = 0;
bool recordingActive = false;

bool lastButtonState = false;
bool buttonState = false;

unsigned long lastDebounceTime = 0;
const unsigned long DEBOUNCE_DELAY = 10;

void onButtonPress();
void (*buttonCallback)() = onButtonPress;

void sendBleChunk(const ImuSample* data, size_t count) {
  if (pCharacteristic == nullptr || !deviceConnected) {
    return;
  }

  pCharacteristic->setValue((uint8_t*)data, count * sizeof(ImuSample));
  pCharacteristic->notify();
}

void sendRecordedSamples() {
  const size_t totalSamples = CAPTURE_SAMPLES;
  for (size_t i = 0; i < totalSamples; i += BLE_CHUNK_SIZE) {
    size_t chunkCount = totalSamples - i;
    if (chunkCount > BLE_CHUNK_SIZE) {
      chunkCount = BLE_CHUNK_SIZE;
    }

    sendBleChunk(&recordedSamples[i], chunkCount);
    delay(10);
  }
}

void checkButtons() {
    bool reading = (analogRead(BUTTON_PIN) > 3000);

    if (reading != lastButtonState) {
      lastDebounceTime = millis();
    }

    if ((millis() - lastDebounceTime) > DEBOUNCE_DELAY) {
      if (reading != buttonState) {
        buttonState = reading;

        if (buttonState) {
          Serial.println("Button pressed! Starting capture...");
          buttonCallback();
        }
      }
    }

    lastButtonState = reading;
}


class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
    }
    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
      pServer->startAdvertising(); 
    }
};


void imuSamplingTask(void * parameter) {
  TickType_t xLastWakeTime;
  const TickType_t xFrequency = pdMS_TO_TICKS(10); // Exactly 10ms = 100Hz
  xLastWakeTime = xTaskGetTickCount();

  for(;;) {
    vTaskDelayUntil(&xLastWakeTime, xFrequency);

    int16_t raw_ax, raw_ay, raw_az;
    int16_t raw_gx, raw_gy, raw_gz;
    mpu.getMotion6(&raw_ax, &raw_ay, &raw_az, &raw_gx, &raw_gy, &raw_gz);

    if (recordingActive) {
      recordedSamples[capturedSampleCount].ax = raw_ax;
      recordedSamples[capturedSampleCount].ay = raw_ay;
      recordedSamples[capturedSampleCount].az = raw_az;
      recordedSamples[capturedSampleCount].gx = raw_gx - gyro_bias_x;
      recordedSamples[capturedSampleCount].gy = raw_gy - gyro_bias_y;
      recordedSamples[capturedSampleCount].gz = raw_gz - gyro_bias_z;

      capturedSampleCount++;

      if (capturedSampleCount >= CAPTURE_SAMPLES) {
        Serial.println("Capture complete. Sending BLE chunks...");
        sendRecordedSamples();
        capturedSampleCount = 0;
        recordingActive = false;
      }
    }
  }
}

void setup() {
    Wire.begin(4, 3);
    Serial.begin(115200);
    
    Serial.println("Initializing I2C devices...");
    mpu.initialize();

    // Verify connection
    if (!mpu.testConnection()) {
      Serial.println("MPU6050 connection failed");
    } else {
      Serial.println("MPU6050 connection successful");
    }

    // Initialize the VCC pin
    pinMode(VCC_PIN, OUTPUT);
    digitalWrite(VCC_PIN, HIGH);

    // Initialize the button pin
    pinMode(BUTTON_PIN, INPUT);

    calibrateGyro();


BLEDevice::init("SmartPen_ESP32C3");
  BLEDevice::setMTU(512);
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);
    pCharacteristic = pService->createCharacteristic(
                      CHARACTERISTIC_UUID,
                      BLECharacteristic::PROPERTY_NOTIFY
                    );
                    
  // BLE2902 is required by BLE protocol to allow notifications
  pCharacteristic->addDescriptor(new BLE2902());
  pService->start();

  // 3. Start Advertising
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(false);
  pAdvertising->setMinPreferred(0x0);
  BLEDevice::startAdvertising();
  xTaskCreatePinnedToCore(
    imuSamplingTask,   
    "IMU_Task",        
    4096,              // Stack size
    NULL,              // Task parameters
    1,                 // Priority
    NULL,              // Task handle
    0                  // Core 0
  );
}


void onButtonPress() {
  if (!recordingActive) {
    capturedSampleCount = 0;
    recordingActive = true;
  }
}

void loop() {
  checkButtons();
}