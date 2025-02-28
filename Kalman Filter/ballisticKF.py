# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 22:41:22 2025

@author: Mark
"""
import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81  # gravity in m/s^2

# Time step
dt = 0.1  # seconds
time = np.arange(0, 10, dt)  # simulate for 10 seconds

# Initial conditions (position and velocity in both x and y directions)
x0 = 0  # initial position in x
y0 = 0  # initial position in y
vx0 = 50  # initial velocity in x (m/s)
vy0 = 50  # initial velocity in y (m/s)

# True trajectory (ignoring air resistance)
def true_trajectory(t):
    x = x0 + vx0 * t
    y = y0 + vy0 * t - 0.5 * g * t**2
    vx = vx0
    vy = vy0 - g * t
    return x, y, vx, vy

# Simulate the true trajectory with noise
np.random.seed(42)
true_x, true_y, true_vx, true_vy = true_trajectory(time)
measurement_noise = 1.0
velocity_measurement_noise = 0.5

# Simulated measurements (both position and velocity are noisy)
measured_x = true_x + np.random.normal(0, measurement_noise, len(time))
measured_y = true_y + np.random.normal(0, measurement_noise, len(time))
measured_vx = true_vx + np.random.normal(0, velocity_measurement_noise, len(time))
measured_vy = true_vy + np.random.normal(0, velocity_measurement_noise, len(time))

# Kalman Filter setup
# State vector [x_position, y_position, x_velocity, y_velocity]
x_est = np.zeros((4, len(time)))  # Estimated state vector
P = np.eye(4)  # Covariance matrix
R = np.diag([measurement_noise, measurement_noise, velocity_measurement_noise, velocity_measurement_noise])  # Measurement noise covariance
Q = np.eye(4) * 0.005  # Process noise covariance

F = np.array([[1, 0, dt, 0],  # State transition matrix
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

H = np.array([[1, 0, 0, 0],  # Measurement matrix (position and velocity are measured)
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

I = np.eye(4)  # Identity matrix

# Initial state (assume initial guess of 0 for velocity and position)
x_est[:, 0] = [0, 0, 0, 0]

# Kalman filter loop
kalman_gain = np.zeros((4, 4, len(time)))  # Store Kalman gain for each step
for k in range(1, len(time)):
    # Prediction step
    x_pred = np.dot(F, x_est[:, k-1])
    P_pred = np.dot(F, np.dot(P, F.T)) + Q
    
    # Measurement residual
    z = np.array([measured_x[k], measured_y[k], measured_vx[k], measured_vy[k]])  # measurement
    y = z - np.dot(H, x_pred)
    
    # Kalman gain calculation
    S = np.dot(H, np.dot(P_pred, H.T)) + R
    K = np.dot(P_pred, np.dot(H.T, np.linalg.inv(S)))
    kalman_gain[:, :, k] = K  # Store the Kalman gain
    
    # Update step
    x_est[:, k] = x_pred + np.dot(K, y)
    P = np.dot(I - np.dot(K, H), P_pred)

# Plotting the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Real vs Estimated Locations
ax1.plot(true_x, true_y, label="True Trajectory", color="g")
ax1.plot(x_est[0, :], x_est[1, :], label="Estimated Trajectory", color="b", linestyle="--")
ax1.scatter(measured_x, measured_y, color="r", label="Measured Points", s=10)
ax1.set_xlabel("X Position (m)")
ax1.set_ylabel("Y Position (m)")
ax1.legend()
ax1.set_title("True vs Estimated Ballistic Trajectory")

# Plot 2: Kalman Gain (for X, Y positions, and velocities)
ax2.plot(time, kalman_gain[0, 0, :], label="Kalman Gain X Position", color="b")
ax2.plot(time, kalman_gain[1, 1, :], label="Kalman Gain Y Position", color="r")
ax2.plot(time, kalman_gain[2, 2, :], label="Kalman Gain X Velocity", color="g")
ax2.plot(time, kalman_gain[3, 3, :], label="Kalman Gain Y Velocity", color="y")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Kalman Gain")
ax2.legend()
ax2.set_title("Kalman Gain for X/Y Positions and Velocities")

plt.tight_layout()
plt.show()
