import numpy as np
import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from scipy.stats import zscore
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')  # Modern equivalent

class ZScoreMeanReversion(Strategy):
    """
    Mean Reversion Strategy using Z-Score on Indian Stocks
    Entry: When price is >2 standard deviations below mean (long)
           When price is >2 standard deviations above mean (short)
    Exit: When price returns within 0.5 standard deviations of mean
    """
    
    # Strategy parameters
    lookback = 20
    entry_z = 2.0
    exit_z = 0.5
    
    def init(self):
        # Precompute rolling mean and std
        close = self.data.Close
        self.rolling_mean = self.I(SMA, close, self.lookback)
        self.rolling_std = self.I(lambda x: pd.Series(x).rolling(self.lookback).std(), close)
        
        # Calculate Z-Scores
        self.z_scores = (close - self.rolling_mean) / self.rolling_std
        
        # For visualization
        self.upper_band = self.rolling_mean + self.entry_z * self.rolling_std
        self.lower_band = self.rolling_mean - self.entry_z * self.rolling_std
        
    def next(self):
        current_z = self.z_scores[-1]
        
        # Exit conditions first
        if self.position.is_long and current_z >= -self.exit_z:
            self.position.close()
        elif self.position.is_short and current_z <= self.exit_z:
            self.position.close()
            
        # Entry conditions
        elif current_z <= -self.entry_z and not self.position.is_long:
            self.buy()
        elif current_z >= self.entry_z and not self.position.is_short:
            self.sell()

def SMA(series, window):
    """Simple Moving Average"""
    return series.rolling(window).mean()

def fetch_nse_data(ticker, start_date, end_date):
    """
    Fetch OHLCV data for NSE stocks using yfinance
    Note: For NSE stocks, append '.NS' to the ticker (e.g., 'RELIANCE.NS')
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def run_backtest():
    # Parameters
    ticker = 'RELIANCE.NS'  # Example: Reliance Industries
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    commission = 0.001  # 0.1% broker commission
    margin = 1.0  # 100% margin (no leverage)
    
    # Fetch data
    data = fetch_nse_data(ticker, start_date, end_date)
    
    # Run backtest
    bt = Backtest(data, ZScoreMeanReversion, commission=commission, margin=margin)
    results = bt.run()
    
    # Display results
    print(results)
    bt.plot()
    
    # Plot price with trading signals
    plt.figure(figsize=(12,6))
    plt.plot(data.Close, label='Price')
    plt.plot(results._strategy.upper_band, label='Upper Band', linestyle='--', color='red')
    plt.plot(results._strategy.lower_band, label='Lower Band', linestyle='--', color='green')
    plt.scatter(results._trades.index, 
                results._trades.EntryPrice.where(results._trades.Size > 0), 
                label='Buy', marker='^', color='green')
    plt.scatter(results._trades.index, 
                results._trades.EntryPrice.where(results._trades.Size < 0), 
                label='Sell', marker='v', color='red')
    plt.title(f'Z-Score Mean Reversion Strategy - {ticker}')
    plt.legend()
    plt.show()
    
    return results

if __name__ == '__main__':
    results = run_backtest()