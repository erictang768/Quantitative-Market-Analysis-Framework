"""
Strategy correlation analysis and diversification metrics.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from config.settings import WFA_CONFIG

def calculate_strategy_correlation(trades_df: pd.DataFrame, stable_strategies: List[str]) -> Dict[str, Any]:
    """
    Calculate correlation analysis between strategies' trades.
    
    Args:
        trades_df: DataFrame of trades
        stable_strategies: List of strategy names to analyze
        
    Returns:
        Dictionary of correlation metrics
    """
    if trades_df.empty or len(stable_strategies) < 2:
        return {
            'correlation_matrix': None,
            'avg_abs_correlation': 0.0,
            'max_abs_correlation': 0.0,
            'diversification_score': 0.0,
            'highly_correlated_pairs': []
        }
    
    # Filter for stable strategies only
    stable_trades = trades_df[trades_df['pattern'].str.contains('|'.join(stable_strategies))]
    
    if stable_trades.empty:
        return {
            'correlation_matrix': None,
            'avg_abs_correlation': 0.0,
            'max_abs_correlation': 0.0,
            'diversification_score': 0.0,
            'highly_correlated_pairs': []
        }
    
    # Create time-aligned returns matrix
    all_exit_times = stable_trades['exit_time'].unique()
    correlation_data = []
    
    for strategy in stable_strategies:
        strategy_trades = stable_trades[stable_trades['pattern'].str.contains(strategy)]
        
        if len(strategy_trades) < WFA_CONFIG['MIN_CORRELATION_SAMPLES']:
            continue
        
        strategy_returns = strategy_trades.set_index('exit_time')['pnl']
        full_series = pd.Series(index=pd.to_datetime(all_exit_times), dtype=float)
        
        for time, ret in strategy_returns.items():
            full_series[pd.to_datetime(time)] = ret
        
        full_series = full_series.fillna(0)
        correlation_data.append(full_series)
    
    if len(correlation_data) < 2:
        return {
            'correlation_matrix': None,
            'avg_abs_correlation': 0.0,
            'max_abs_correlation': 0.0,
            'diversification_score': 0.0,
            'highly_correlated_pairs': []
        }
    
    # Create correlation matrix
    returns_df = pd.concat(correlation_data, axis=1)
    returns_df.columns = [f'Strategy_{i+1}' for i in range(len(correlation_data))]
    correlation_matrix = returns_df.corr()
    
    # Calculate correlation metrics
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    abs_correlations = upper_triangle.abs().stack().dropna()
    
    if len(abs_correlations) > 0:
        avg_abs_correlation = abs_correlations.mean()
        max_abs_correlation = abs_correlations.max()
        
        # Find highly correlated pairs
        highly_correlated_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > WFA_CONFIG['CORRELATION_THRESHOLD']:
                    highly_correlated_pairs.append({
                        'pair': (correlation_matrix.columns[i], correlation_matrix.columns[j]),
                        'correlation': corr
                    })
        
        # Calculate diversification score
        diversification_score = max(0, 1 - (
            WFA_CONFIG['DIVERSIFICATION_SCORE_WEIGHTS']['avg_correlation'] * avg_abs_correlation +
            WFA_CONFIG['DIVERSIFICATION_SCORE_WEIGHTS']['max_correlation'] * max_abs_correlation
        ))
    else:
        avg_abs_correlation = 0.0
        max_abs_correlation = 0.0
        highly_correlated_pairs = []
        diversification_score = 0.0
    
    return {
        'correlation_matrix': correlation_matrix,
        'avg_abs_correlation': avg_abs_correlation,
        'max_abs_correlation': max_abs_correlation,
        'diversification_score': diversification_score,
        'highly_correlated_pairs': highly_correlated_pairs
    }
