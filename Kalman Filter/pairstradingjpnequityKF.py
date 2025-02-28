# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 19:14:40 2025

@author: Mark
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

# Function to fetch historical data
def fetch_data(ticker_1, ticker_2, start_date='2020-01-01', end_date='2024-01-01'):
    data_1 = yf.download(ticker_1, start=start_date, end=end_date)
    data_2 = yf.download(ticker_2, start=start_date, end=end_date)
    
    # Extract the 'Close' prices
    data_1 = data_1['Close']
    data_2 = data_2['Close']
    
    # Align the data based on common dates
    data_1, data_2 = data_1.align(data_2, join='inner', axis=0)
    
    return data_1, data_2

# Kalman Filter to estimate the spread between the two assets
def kalman_filter_spread(series_1, series_2):
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.F = np.array([[1, 1], [0, 1]])  # State transition matrix
    kf.H = np.array([[1, 0]])  # Measurement function
    kf.P *= 1000.0  # Initial uncertainty
    kf.R = 5  # Measurement noise
    kf.Q = 0.001  # Process noise
    
    x = np.array([0, 0])  # Initial state estimate
    spread = []
    
    for i in range(len(series_1)):
        z = series_1.iloc[i,0] - series_2.iloc[i,0]  # Spread between the two series
        kf.predict()
        kf.update(z)
        spread.append(kf.x[0])  # The estimated spread
    
    return np.array(spread)

# Function to generate trading signals based on the Kalman filter output
def generate_trading_signals(spread, threshold=1.5):
    mean_spread = np.mean(spread)
    std_spread = np.std(spread)
    
    # Buy when the spread is below mean - threshold * std
    # Sell when the spread is above mean + threshold * std
    buy_signal = spread < mean_spread - threshold * std_spread
    sell_signal = spread > mean_spread + threshold * std_spread
    
    return buy_signal, sell_signal

# Calculate the profit of following the pairs trading strategy
def calculate_profit(spread, buy_signal, sell_signal):
    position = 0  # 1 for long, -1 for short, 0 for no position
    entry_price = 0
    profit = 0  # Start with no profit

    for i in range(1, len(spread)):
        if position == 0:  # No position, check for buy/sell signals
            if buy_signal[i]:  # Buy signal (go long)
                position = 1
                entry_price = spread[i]
            elif sell_signal[i]:  # Sell signal (go short)
                position = -1
                entry_price = spread[i]
        elif position == 1:  # We are in a long position
            if sell_signal[i]:  # Exit long when sell signal occurs (go short)
                profit += spread[i] - entry_price
                position = -1  # Now we're short
                entry_price = spread[i]
            elif i == len(spread) - 1:  # Exit at the end of data
                profit += spread[i] - entry_price
        elif position == -1:  # We are in a short position
            if buy_signal[i]:  # Exit short when buy signal occurs (go long)
                profit += entry_price - spread[i]
                position = 1  # Now we're long
                entry_price = spread[i]
            elif i == len(spread) - 1:  # Exit at the end of data
                profit += entry_price - spread[i]

    return profit

# Main function to execute the pairs trading strategy
def pairs_trading(ticker_1, ticker_2, start_date='2020-01-01', end_date='2024-01-01'):
    # Fetch the data
    data_1, data_2 = fetch_data(ticker_1, ticker_2, start_date, end_date)
    
    # Apply Kalman filter to calculate spread
    spread = kalman_filter_spread(data_1, data_2)
    
    # Generate trading signals
    buy_signal, sell_signal = generate_trading_signals(spread)
    
    # Calculate the profit from the strategy
    profit = calculate_profit(spread, buy_signal, sell_signal)
    
    # Print out the results
    profit = np.sum(profit)
    print(f"Total Profit from the strategy: {profit:.2f}")
    
    # Plot the results
    plt.figure(figsize=(14, 8))
    
    # Plot the spread
    plt.subplot(3, 1, 1)
    plt.plot(data_1.index, spread, label="Estimated Spread", color='blue')
    plt.title(f"Estimated Spread between {ticker_1} and {ticker_2}")
    plt.axhline(y=np.mean(spread), color='green', linestyle='--', label='Mean Spread')
    plt.legend()
    
    # Plot buy signals
    plt.subplot(3, 1, 2)
    plt.plot(data_1.index, buy_signal, label="Buy Signal", color='green', alpha=0.7)
    plt.title(f"Buy signals for {ticker_1} and {ticker_2}")
    plt.legend()

    # Plot sell signals
    plt.subplot(3, 1, 3)
    plt.plot(data_1.index, sell_signal, label="Sell Signal", color='red', alpha=0.7)
    plt.title(f"Sell signals for {ticker_1} and {ticker_2}")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Run the pairs trading strategy with two Japanese equities: Toyota (7203.T) and Sony (6758.T)
pairs_trading("7203.T", "6758.T", start_date="2020-01-01", end_date="2024-01-01")
