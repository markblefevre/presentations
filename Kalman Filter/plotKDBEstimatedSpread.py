# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 23:20:33 2025

@author: Mark
"""
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
file_path = 'fx_out.csv'  # Replace with your file path
df = pd.read_csv(file_path, parse_dates=['Date'])

# Ensure the Date column is datetime type
df['Date'] = pd.to_datetime(df['Date'])

# Plot EURUSD=X and USDJPY=X on different y-axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot EURUSD=X on the first y-axis
ax1.plot(df['Date'], df['EURUSD=X'], label='EURUSD', color='blue')
ax1.set_xlabel('Date')
ax1.set_ylabel('EURUSD', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a second y-axis to plot USDJPY=X
ax2 = ax1.twinx()
ax2.plot(df['Date'], df['USDJPY=X'], label='USDJPY', color='green')
ax2.set_ylabel('USDJPY', color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Add legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.title('EURUSD and USDJPY Over Time')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot the spread and velocity on the same plot with different y-axes
fig, ax5 = plt.subplots(figsize=(10, 6))

# Plot spread on the first y-axis
ax5.plot(df['Date'], df['spread'], label='Spread', color='red')
ax5.set_xlabel('Date')
ax5.set_ylabel('Spread', color='red')
ax5.tick_params(axis='y', labelcolor='red')

# Create a second y-axis to plot velocity
ax6 = ax5.twinx()
ax6.plot(df['Date'], df['velocity'], label='Velocity', color='purple')
ax6.set_ylabel('Velocity', color='purple')
ax6.tick_params(axis='y', labelcolor='purple')

# Set the y-axis limits for velocity from 0 to -1
ax6.set_ylim(-1, 0)

# Add legends
ax5.legend(loc='upper left')
ax6.legend(loc='upper right')

plt.title('Spread and Velocity Over Time')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate Actual Spread as the difference between EURUSD=X and USDJPY=X
df['Actual Spread'] = df['EURUSD=X'] - df['USDJPY=X']

# Plot Actual Spread and Estimated Spread on the same graph
fig, ax7 = plt.subplots(figsize=(10, 6))

# Plot Actual Spread on the first y-axis
ax7.plot(df['Date'], df['Actual Spread'], label='Actual Spread', color='orange')
ax7.set_xlabel('Date')
ax7.set_ylabel('Actual Spread', color='orange')
ax7.tick_params(axis='y', labelcolor='orange')

# Plot Estimated Spread on the same axis
ax7.plot(df['Date'], df['spread'], label='Estimated Spread', color='blue', linestyle='--')
ax7.set_ylabel('Spread', color='blue')
ax7.tick_params(axis='y', labelcolor='blue')

# Add legends
ax7.legend(loc='upper left')

plt.title('Actual Spread vs Estimated Spread Over Time')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Zoomed-in version: Last 30 results
df_last_30 = df.tail(365)
fig, ax7_zoom = plt.subplots(figsize=(10, 6))

# Plot Actual Spread on the first y-axis (zoomed-in)
ax7_zoom.plot(df_last_30['Date'], df_last_30['Actual Spread'], label='Actual Spread', color='orange')
ax7_zoom.set_xlabel('Date')
ax7_zoom.set_ylabel('Actual Spread', color='orange')
ax7_zoom.tick_params(axis='y', labelcolor='orange')

# Plot Estimated Spread on the same axis (zoomed-in)
ax7_zoom.plot(df_last_30['Date'], df_last_30['spread'], label='Estimated Spread', color='blue', linestyle='--')
ax7_zoom.set_ylabel('Spread', color='blue')
ax7_zoom.tick_params(axis='y', labelcolor='blue')

# Add legends
ax7_zoom.legend(loc='upper left')

plt.title('Actual Spread vs Estimated Spread (Last 365 Results)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()