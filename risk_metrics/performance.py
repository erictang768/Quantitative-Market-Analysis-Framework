"""
Advanced risk and performance metrics calculation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime

def calculate_advanced_metrics(trades_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        trades_df: DataFrame of trades with 'pnl' and datetime index
        
    Returns:
        Dictionary of advanced metrics
    """
    if trades_df.empty:
        return {
            'std_r_per_trade': 0.0,
            'max_drawdown_r': 0.0,
            'avg_trades_per_day': 0.0,
            'avg_r_per_day': 0.0,
            'avg_r_per_week': 0.0,
            'sharpe_ratio_r': 0.0,
            'calmar_ratio_r': 0.0,
        }
    
    # 1. Standard Deviation of R per Trade
    std_r_per_trade = trades_df['pnl'].std()
    
    # 2. Maximum Historical Drawdown (in R)
    r_cumulative = trades_df['pnl'].cumsum()
    running_max = np.maximum.accumulate(r_cumulative)
    drawdowns = running_max - r_cumulative
    max_drawdown_r = drawdowns.max() if not drawdowns.empty else 0.0
    
    # 3. Average Number of Trades Per Day
    trades_per_day = trades_df.resample('D').size()
    avg_trades_per_day = trades_per_day.mean() if len(trades_per_day) > 0 else 0.0
    
    # 4. Average R/day and R/week
    daily_r_pnl = trades_df['pnl'].resample('D').sum()
    avg_r_per_day = daily_r_pnl.mean() if len(daily_r_pnl) > 0 else 0.0
    
    weekly_r_pnl = trades_df['pnl'].resample('W').sum()
    avg_r_per_week = weekly_r_pnl.mean() if len(weekly_r_pnl) > 0 else 0.0
    
    # 5. Risk-adjusted metrics
    if std_r_per_trade > 0 and len(trades_df) > 1:
        sharpe_ratio_r = (trades_df['pnl'].mean() / std_r_per_trade) * np.sqrt(252)
    else:
        sharpe_ratio_r = 0.0
    
    if max_drawdown_r > 0:
        total_r_return = trades_df['pnl'].sum()
        total_days = (trades_df.index[-1] - trades_df.index[0]).days
        if total_days > 0:
            annualized_r_return = (total_r_return / total_days) * 365
            calmar_ratio_r = annualized_r_return / max_drawdown_r
        else:
            calmar_ratio_r = 0.0
    else:
        calmar_ratio_r = float('inf') if trades_df['pnl'].sum() > 0 else 0.0
    
    return {
        'std_r_per_trade': std_r_per_trade,
        'max_drawdown_r': max_drawdown_r,
        'avg_trades_per_day': avg_trades_per_day,
        'avg_r_per_day': avg_r_per_day,
        'avg_r_per_week': avg_r_per_week,
        'sharpe_ratio_r': sharpe_ratio_r,
        'calmar_ratio_r': calmar_ratio_r,
        'r_cumulative': r_cumulative,
        'drawdown_curve_r': drawdowns,
        'daily_r_curve': daily_r_pnl,
        'weekly_r_curve': weekly_r_pnl
    }

def calculate_basic_metrics(trades_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate basic performance metrics.
    
    Args:
        trades_df: DataFrame with 'win' and 'dollar_pnl' columns
        
    Returns:
        Dictionary of basic metrics
    """
    if trades_df.empty:
        return {}
    
    winning_trades = trades_df[trades_df['win']]
    losing_trades = trades_df[~trades_df['win']]
    
    total_trades = len(trades_df)
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    total_pnl = trades_df['dollar_pnl'].sum()
    avg_trade_pnl = trades_df['dollar_pnl'].mean()
    
    profit_factor = abs(winning_trades['dollar_pnl'].sum() / losing_trades['dollar_pnl'].sum()) if losing_trades['dollar_pnl'].sum() != 0 else float('inf')
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_trade_pnl': avg_trade_pnl,
        'profit_factor': profit_factor
    }
