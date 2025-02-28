# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:57:50 2025

@author: Mark
"""
import pandas as pd
import matplotlib.pyplot as plt

# Read in the CSV file
df = pd.read_csv('data.csv')  # Replace 'your_file.csv' with your actual file path

# Extract relevant columns for positions and velocities
x = df['x']
y = df['y']
truex = df['truex']
truey = df['truey']
measx = df['measx']
measy = df['measy']

xv = df['xv']
yv = df['yv']
truexv = df['truexv']
trueyv = df['trueyv']
measxv = df['measxv']
measyv = df['measyv']

# Extract Kalman Gain columns
gainx = df['gainx']
gainy = df['gainy']
gainxv = df['gainxv']
gainyv = df['gainyv']

# Create a figure with three subplots (1 row, 3 columns)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Plot for x vs y, truex vs truey, measx vs measy (Position plot)
ax1.scatter(x, y, label='Estimated', color='blue', alpha=0.6)
ax1.scatter(truex, truey, label='True', color='green', alpha=0.6)
ax1.scatter(measx, measy, label='Measured', color='red', alpha=0.6)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Comparison of Estimated, True, and Measured Trajectories')
ax1.legend()

# Plot for xv vs yv, truexv vs trueyv, measxv vs measyv (Velocity plot)
ax2.scatter(xv, yv, label='Estimated', color='blue', alpha=0.6)
ax2.scatter(truexv, trueyv, label='True', color='green', alpha=0.6)
ax2.scatter(measxv, measyv, label='Measured', color='red', alpha=0.6)
ax2.set_xlabel('X Velocity (xv)')
ax2.set_ylabel('Y Velocity (yv)')
ax2.set_title('Comparison of Estimated, True, and Measured Velocities')
ax2.legend()

# Plot for Kalman Gains (for positions and velocities)
ax3.plot(df['time'], gainx, label='Kalman Gain X Position', color='blue')
ax3.plot(df['time'], gainy, label='Kalman Gain Y Position', color='red')
ax3.plot(df['time'], gainxv, label='Kalman Gain X Velocity', color='green')
ax3.plot(df['time'], gainyv, label='Kalman Gain Y Velocity', color='yellow')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Kalman Gain')
ax3.set_title('Kalman Gain for X/Y Positions and Velocities')
ax3.legend()

# Adjust layout to avoid overlapping
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig('kalman_filter_plot.png')

# Show the plot
plt.show()