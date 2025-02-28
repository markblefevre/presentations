# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:44:07 2025

@author: Mark
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:44:07 2025

@author: Mark
"""
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import yfinance as yf
from pykalman import KalmanFilter


def draw_date_coloured_scatterplot(etfs, prices):
    """
    Create a scatterplot of the two ETF prices, which is
    coloured by the date of the price to indicate the 
    changing relationship between the sets of prices    
    """
    # Create a yellow-to-red colourmap where yellow indicates
    # early dates and red indicates later dates
    plen = len(prices)
    colour_map = plt.cm.get_cmap('YlOrRd')    
    colours = np.linspace(0.1, 1, plen)
    
    # Create the scatterplot object
    scatterplot = plt.scatter(
        prices[etfs[0]], prices[etfs[1]], 
        s=30, c=colours, cmap=colour_map, 
        edgecolor='k', alpha=0.8
    )
    
    # Add a colour bar for the date colouring and set the 
    # corresponding axis tick labels to equal string-formatted dates
    colourbar = plt.colorbar(scatterplot)
    colourbar.ax.set_yticklabels(
        [str(p.date()) for p in prices[::plen//9].index]
    )
    plt.xlabel(prices.columns[0])
    plt.ylabel(prices.columns[1])
    plt.show()


def calc_slope_intercept_kalman(etfs, prices):
    """
    Utilize the Kalman Filter from the pyKalman package
    to calculate the slope and intercept of the regressed
    ETF prices.
    """
    delta = 1e-5
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.vstack(
        [prices[etfs[0]], np.ones(prices[etfs[0]].shape)]
    ).T[:, np.newaxis]
    
    kf = KalmanFilter(
        n_dim_obs=1, 
        n_dim_state=2,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=1.0,
        transition_covariance=trans_cov
    )
    
    state_means, state_covs = kf.filter(prices[etfs[1]].values)
    return state_means, state_covs    


def draw_slope_intercept_changes(prices, state_means):
    """
    Plot the slope and intercept changes from the 
    Kalman Filter calculated values.
    """
    pd.DataFrame(
        dict(
            slope=state_means[:, 0], 
            intercept=state_means[:, 1]
        ), index=prices.index
    ).plot(subplots=True)
    plt.show()


def plot_ols_per_year(prices, etfs):
    """
    Plot OLS regression lines for each year along with Kalman Filter regression.
    """
    years = prices.index.year.unique()  # Get the unique years
    cm = plt.get_cmap('jet')
    colors = np.linspace(0.1, 1, len(years))
    
    # Scatter plot the data
    plt.scatter(prices[etfs[0]], prices[etfs[1]], s=10, c=prices.index.year, cmap=cm, edgecolors='k', alpha=0.7)
    plt.xlabel(etfs[0])
    plt.ylabel(etfs[1])

    for i, year in enumerate(years):
        yearly_data = prices[prices.index.year == year]  # Filter data for the specific year
        x = yearly_data[etfs[0]].values
        y = yearly_data[etfs[1]].values
        
        # OLS regression using NumPy's polyfit for each year
        slope, intercept = np.polyfit(x, y, 1)  # 1 for linear fit (slope and intercept)
        
        # Generate the regression line for the current year
        xi = np.linspace(x.min(), x.max(), 100)
        yi = slope * xi + intercept
        
        # Plot the regression line with color for the year
        plt.plot(xi, yi, label=f'OLS {year}', color=cm(colors[i]), lw=2)
    
    # Add a color bar for the year coloring
    plt.colorbar(label="Year")
    plt.show()


if __name__ == "__main__":
    # Choose the ETF symbols to work with along with 
    # start and end dates for the price histories
    etfs = ['QQQ', 'AAPL']
    etfs = ['SPY', 'SPLV']
    etfs = ['IWM', 'EEM']
    start_date = "2019-01-01"
    end_date = "2024-12-31"    
    
    # Obtain the closing prices from Yahoo Finance using yfinance
    prices = yf.download(etfs, start=start_date, end=end_date)['Close']

    draw_date_coloured_scatterplot(etfs, prices)
    state_means, state_covs = calc_slope_intercept_kalman(etfs, prices)
    draw_slope_intercept_changes(prices, state_means)

    # Plot OLS regression lines for each year
    plot_ols_per_year(prices, etfs)
