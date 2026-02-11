"""
Data loading and preprocessing module for intraday financial data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

def load_market_data(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess market data from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Preprocessed DataFrame with datetime index and OHLCV columns
    """
    try:
        df = pd.read_csv(
            filepath,
            header=None,
            names=["Date", "Time", "Open", "High", "Low", "Close", "Volume"],
            dayfirst=False,
        )
        
        # Create datetime index
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df = df.set_index('Datetime')[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Add candlestick features
        df['Bullish'] = df['Close'] > df['Open']
        df['Bearish'] = df['Close'] < df['Open']
        df['Body'] = abs(df['Close'] - df['Open'])
        
        # Calculate ATR for volatility
        prev_close = df['Close'].shift(1)
        tr = np.maximum(df['High'] - df['Low'], 
                       np.maximum(abs(df['High'] - prev_close), 
                                 abs(df['Low'] - prev_close)))
        df['ATR'] = tr.rolling(14, min_periods=1).mean()
        
        print(f"Loaded {len(df)} bars from {df.index.min()} to {df.index.max()}")
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def resample_data(df: pd.DataFrame, frequency: str = '1H') -> pd.DataFrame:
    """
    Resample data to different frequency.
    
    Args:
        df: Original DataFrame
        frequency: Target frequency (e.g., '1H', '15T', '1D')
        
    Returns:
        Resampled DataFrame
    """
    resampled = df.resample(frequency).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    return resampled
