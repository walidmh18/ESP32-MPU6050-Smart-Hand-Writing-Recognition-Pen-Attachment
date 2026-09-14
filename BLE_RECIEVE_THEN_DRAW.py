import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, filtfilt

# 1. Load the recorded stroke
# Replace with one of your actual CSV files
df = pd.read_csv('dataset_letter_A.csv')

# dt is your sample period. At 100Hz, dt = 0.01 seconds
dt = 0.01 

# 2. Design a High-Pass Filter to remove gravity/tilt bias
# Cutoff frequency of 0.5 Hz. Anything slower is assumed to be gravity.
nyquist = 0.5 * 100 
cutoff = 0.5
b, a = butter(2, cutoff / nyquist, btype='high')

# Apply the filter to X and Y acceleration
# (Assuming your IMU X and Y axes are aligned with the desk plane)
accel_x = filtfilt(b, a, df['ax'].values)
accel_y = filtfilt(b, a, df['ay'].values)

# 3. First Integration: Acceleration -> Velocity
vel_x = cumulative_trapezoid(accel_x, dx=dt, initial=0)
vel_y = cumulative_trapezoid(accel_y, dx=dt, initial=0)

# 4. Second Integration: Velocity -> Position
pos_x = cumulative_trapezoid(vel_x, dx=dt, initial=0)
pos_y = cumulative_trapezoid(vel_y, dx=dt, initial=0)

# 5. Plot the 2D Trajectory
plt.figure(figsize=(6, 6))
plt.plot(pos_x, pos_y, label='Pen Path', color='blue', linewidth=2)
plt.scatter(pos_x[0], pos_y[0], color='green', label='Start', zorder=5)
plt.scatter(pos_x[-1], pos_y[-1], color='red', label='End', zorder=5)

plt.title("2D Reconstructed Pen Trajectory")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.axis('equal') # Ensure X and Y scales match so the shape isn't distorted
plt.grid(True)
plt.legend()
plt.show()