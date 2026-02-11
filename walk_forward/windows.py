"""
Walk-forward window creation and management.
"""

import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from config.settings import WFA_CONFIG

def create_walk_forward_windows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Create walk-forward windows with specified training/testing periods.
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        List of window dictionaries with train/test data splits
    """
    windows = []
    
    start_date = df.index.min()
    end_date = df.index.max()
    
    current_start = start_date
    window_id = 1
    
    while True:
        # Training period
        train_end = current_start + pd.DateOffset(years=WFA_CONFIG['TRAINING_YEARS'])
        
        # Testing period
        test_end = train_end + pd.DateOffset(years=WFA_CONFIG['TESTING_YEARS'])
        
        # Check if we have enough data
        if test_end > end_date:
            break
        
        windows.append({
            'window_id': window_id,
            'train_start': current_start,
            'train_end': train_end,
            'test_start': train_end,
            'test_end': test_end,
            'train_data': df[(df.index >= current_start) & (df.index < train_end)],
            'test_data': df[(df.index >= train_end) & (df.index < test_end)]
        })
        
        # Move forward
        current_start = current_start + pd.DateOffset(years=WFA_CONFIG['STEP_FORWARD_YEARS'])
        window_id += 1
    
    print(f"Created {len(windows)} walk-forward windows")
    return windows

def calculate_net_expectancy(trades_df: pd.DataFrame) -> float:
    """
    Calculate net expectancy: E = (W × Avg Win) + (L × -1)
    
    Args:
        trades_df: DataFrame of trades with 'win' and 'pnl' columns
        
    Returns:
        Net expectancy value
    """
    if len(trades_df) == 0:
        return 0.0
    
    win_rate = trades_df['win'].mean()
    avg_winning_pnl = trades_df[trades_df['win']]['pnl'].mean() if any(trades_df['win']) else 0
    
    net_expectancy = (win_rate * avg_winning_pnl) + ((1 - win_rate) * -1)
    return net_expectancy
