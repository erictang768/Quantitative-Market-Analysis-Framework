import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Callable
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import stats
warnings.filterwarnings('ignore')

# =============================================================================
# WALK-FORWARD ANALYSIS FRAMEWORK CONFIGURATION
# =============================================================================

# === WALK-FORWARD CONFIGURATION ===
WFA_CONFIG = {
    'TRAINING_YEARS': 3,
    'TESTING_YEARS': 1,
    'STEP_FORWARD_YEARS': 1,
    'MIN_OOS_EXPECTANCY': 0.02,
    'MAX_EXPECTANCY_DECAY': 0.50,
    'MAX_WINRATE_DECAY': 0.30,
    
    # Stability Selection Configuration
    'MIN_SAMPLE_SIZE': 25,
    'MIN_EXPECTANCY': 0.10,
    'STABILITY_WEIGHTS': {
        'expectancy': 0.40,
        'sample_size': 0.35,  
        'win_rate': 0.25
    },
    
    # Correlation Analysis Configuration
    'CORRELATION_THRESHOLD': 0.7,  # High correlation threshold
    'MIN_CORRELATION_SAMPLES': 10,  # Minimum trades needed for correlation analysis
    'DIVERSIFICATION_SCORE_WEIGHTS': {
        'avg_correlation': 0.4,
        'max_correlation': 0.3,
        'sign_correlation': 0.3
    }
}

# =============================================================================
# STRATEGY CONFIGURATION - TOGGLE EACH STRATEGY ON/OFF
# =============================================================================

STRATEGY_CONFIG = {
    # General Settings
    'INITIAL_CAPITAL': 100000,
    'RISK_PCT': 0.01,
    'SPREAD': 0.0000,
    'PIP_SIZE': 0.0001,
    'COMMISSION_PER_LOT': 7.0,
    'LEVERAGE': 100,
    'MARGIN_BUFFER_PCT': 0.1,
    
    # Strategy Toggles
    'STRATEGY_1_ENABLED': True,
    'STRATEGY_2_ENABLED': True, 
    'STRATEGY_3_ENABLED': True,
    'STRATEGY_4_ENABLED': True,
    'STRATEGY_5_ENABLED': True,
    
    # Strategy-specific parameters
    'STRATEGY_1': {
        'MAX_HOLD_BARS': 24,
        'MAX_TRIGGER_CANDLES': 15,
        'NET_RR_RATIO': 1.0,
        'ALLOWED_HOURS': [3, 8, 9, 12, 15]
    },
    'STRATEGY_2': {
        'MAX_HOLD_BARS': 24,
        'MAX_TRIGGER_CANDLES': 18,
        'NET_RR_RATIO': 1.25,
        'ALLOWED_HOURS': [5, 9, 16, 19, 23]
    },
    'STRATEGY_3': {
        'MAX_HOLD_BARS': 24,
        'MAX_TRIGGER_CANDLES': 24,
        'NET_RR_RATIO': 1.0,
        'ALLOWED_HOURS': [0, 10, 12, 15, 21]
    },
    'STRATEGY_4': {
        'MAX_HOLD_BARS': 24,
        'MAX_TRIGGER_CANDLES': 14,
        'NET_RR_RATIO': 1.0,
        'VALIDATION_CHECK': True,
        'ALLOWED_HOURS': [8, 15, 16, 20, 22]
    },
    'STRATEGY_5': {
        'MAX_HOLD_BARS': 24,
        'MAX_TRIGGER_CANDLES': 20,
        'NET_RR_RATIO': 1.5,
        'VALIDATION_CHECK': True,
        'ALLOWED_HOURS': [2, 6, 7, 8, 9]
    }
}

# =============================================================================
# NEW METRICS FUNCTIONS
# =============================================================================

def calculate_advanced_metrics(trades_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate advanced performance metrics including:
    - Standard Deviation of R per Trade
    - Maximum Historical Drawdown (in R)
    - Average Number of Trades Per Day
    - Average R/day and R/week
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
            'r_per_trade_distribution': None,
            'drawdown_curve_r': None,
            'daily_r_curve': None
        }
    
    # 1. Standard Deviation of R per Trade
    std_r_per_trade = trades_df['pnl'].std()
    
    # 2. Maximum Historical Drawdown (in R)
    # Create R-equity curve (starting from 0)
    r_cumulative = trades_df['pnl'].cumsum()
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(r_cumulative)
    
    # Calculate drawdowns
    drawdowns = running_max - r_cumulative
    
    # Maximum drawdown in R
    max_drawdown_r = drawdowns.max() if not drawdowns.empty else 0.0
    
    # 3. Average Number of Trades Per Day
    # Convert entry_time to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(trades_df.index):
        trades_df.index = pd.to_datetime(trades_df.index)
    
    # Count trades per day
    trades_per_day = trades_df.resample('D').size()
    
    # Calculate average trades per day (only days with trades)
    avg_trades_per_day = trades_per_day.mean() if len(trades_per_day) > 0 else 0.0
    
    # 4. Average R/day and R/week
    # Calculate daily R P&L
    daily_r_pnl = trades_df['pnl'].resample('D').sum()
    avg_r_per_day = daily_r_pnl.mean() if len(daily_r_pnl) > 0 else 0.0
    
    # Calculate weekly R P&L
    weekly_r_pnl = trades_df['pnl'].resample('W').sum()
    avg_r_per_week = weekly_r_pnl.mean() if len(weekly_r_pnl) > 0 else 0.0
    
    # 5. Risk-adjusted metrics
    # Sharpe Ratio (in R space)
    if std_r_per_trade > 0 and len(trades_df) > 1:
        sharpe_ratio_r = (trades_df['pnl'].mean() / std_r_per_trade) * np.sqrt(252)  # Annualized
    else:
        sharpe_ratio_r = 0.0
    
    # Calmar Ratio (in R space)
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
    
    # R per trade distribution statistics
    r_per_trade_distribution = {
        'mean': trades_df['pnl'].mean(),
        'median': trades_df['pnl'].median(),
        'skewness': trades_df['pnl'].skew(),
        'kurtosis': trades_df['pnl'].kurtosis(),
        'q1': trades_df['pnl'].quantile(0.25),
        'q3': trades_df['pnl'].quantile(0.75),
        'min': trades_df['pnl'].min(),
        'max': trades_df['pnl'].max()
    }
    
    return {
        'std_r_per_trade': std_r_per_trade,
        'max_drawdown_r': max_drawdown_r,
        'avg_trades_per_day': avg_trades_per_day,
        'avg_r_per_day': avg_r_per_day,
        'avg_r_per_week': avg_r_per_week,
        'sharpe_ratio_r': sharpe_ratio_r,
        'calmar_ratio_r': calmar_ratio_r,
        'r_per_trade_distribution': r_per_trade_distribution,
        'drawdown_curve_r': drawdowns,
        'daily_r_curve': daily_r_pnl,
        'weekly_r_curve': weekly_r_pnl,
        'r_cumulative': r_cumulative,
        'total_trading_days': len(daily_r_pnl),
        'trading_day_ratio': len(daily_r_pnl) / ((trades_df.index[-1] - trades_df.index[0]).days + 1) if len(trades_df) > 1 else 0.0
    }

def plot_advanced_metrics(advanced_metrics: Dict[str, Any], period_name: str = "Performance"):
    """Create visualizations for the advanced metrics"""
    
    if advanced_metrics['r_cumulative'] is None:
        return
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'{period_name} - Advanced Metrics Analysis', fontsize=14, fontweight='bold')
    
    # Plot 1: R Cumulative Curve with Drawdowns
    axes[0, 0].plot(advanced_metrics['r_cumulative'].index, advanced_metrics['r_cumulative'].values, 
                   linewidth=2, color='blue', label='Cumulative R')
    axes[0, 0].fill_between(advanced_metrics['r_cumulative'].index, 
                           advanced_metrics['r_cumulative'].values, 
                           advanced_metrics['r_cumulative'].values - advanced_metrics['drawdown_curve_r'].values,
                           color='red', alpha=0.3, label='Drawdown')
    axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Cumulative R')
    axes[0, 0].set_title(f'R Cumulative Curve (Max DD: {advanced_metrics["max_drawdown_r"]:.2f}R)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: R per Trade Distribution
    if not advanced_metrics['r_cumulative'].empty:
        axes[0, 1].hist(advanced_metrics['r_cumulative'].diff().dropna(), bins=50, alpha=0.7, 
                       color='skyblue', edgecolor='black', density=True)
        axes[0, 1].axvline(x=0, color='red', linestyle='--', label='Zero Line')
        axes[0, 1].axvline(x=advanced_metrics['r_per_trade_distribution']['mean'], 
                          color='green', linestyle='-', 
                          label=f'Mean: {advanced_metrics["r_per_trade_distribution"]["mean"]:.3f}')
        axes[0, 1].set_xlabel('R per Trade')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title(f'R per Trade Distribution (Std: {advanced_metrics["std_r_per_trade"]:.3f})')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Daily R P&L
    if advanced_metrics['daily_r_curve'] is not None and not advanced_metrics['daily_r_curve'].empty:
        colors = ['green' if x >= 0 else 'red' for x in advanced_metrics['daily_r_curve'].values]
        axes[1, 0].bar(advanced_metrics['daily_r_curve'].index, advanced_metrics['daily_r_curve'].values, 
                      color=colors, alpha=0.7, edgecolor='black')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 0].axhline(y=advanced_metrics['avg_r_per_day'], color='blue', linestyle='--',
                          label=f'Avg: {advanced_metrics["avg_r_per_day"]:.3f}R/day')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Daily R P&L')
        axes[1, 0].set_title(f'Daily R Performance (Avg {advanced_metrics["avg_trades_per_day"]:.2f} trades/day)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Weekly R P&L
    if advanced_metrics['weekly_r_curve'] is not None and not advanced_metrics['weekly_r_curve'].empty:
        colors = ['green' if x >= 0 else 'red' for x in advanced_metrics['weekly_r_curve'].values]
        axes[1, 1].bar(range(len(advanced_metrics['weekly_r_curve'])), advanced_metrics['weekly_r_curve'].values, 
                      color=colors, alpha=0.7, edgecolor='black')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 1].axhline(y=advanced_metrics['avg_r_per_week'], color='blue', linestyle='--',
                          label=f'Avg: {advanced_metrics["avg_r_per_week"]:.3f}R/week')
        axes[1, 1].set_xlabel('Week')
        axes[1, 1].set_ylabel('Weekly R P&L')
        axes[1, 1].set_title(f'Weekly R Performance')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 5: Drawdown Analysis
    if advanced_metrics['drawdown_curve_r'] is not None and not advanced_metrics['drawdown_curve_r'].empty:
        axes[2, 0].fill_between(advanced_metrics['drawdown_curve_r'].index, 
                               0, advanced_metrics['drawdown_curve_r'].values,
                               color='red', alpha=0.5, label='Drawdown (R)')
        axes[2, 0].axhline(y=advanced_metrics['max_drawdown_r'], color='darkred', linestyle='--',
                          label=f'Max DD: {advanced_metrics["max_drawdown_r"]:.2f}R')
        axes[2, 0].set_xlabel('Date')
        axes[2, 0].set_ylabel('Drawdown (R)')
        axes[2, 0].set_title('Historical Drawdown Analysis')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 6: Risk Metrics Summary
    metrics_text = f"""
    Standard Deviation (R/Trade): {advanced_metrics['std_r_per_trade']:.3f}
    Maximum Drawdown (R): {advanced_metrics['max_drawdown_r']:.2f}
    Average Trades/Day: {advanced_metrics['avg_trades_per_day']:.2f}
    Average R/Day: {advanced_metrics['avg_r_per_day']:.3f}
    Average R/Week: {advanced_metrics['avg_r_per_week']:.3f}
    Sharpe Ratio (R): {advanced_metrics['sharpe_ratio_r']:.2f}
    Calmar Ratio (R): {advanced_metrics['calmar_ratio_r']:.2f}
    Trading Days Ratio: {advanced_metrics['trading_day_ratio']:.1%}
    """
    
    axes[2, 1].text(0.1, 0.5, metrics_text, fontsize=10, fontfamily='monospace',
                   verticalalignment='center', transform=axes[2, 1].transAxes)
    axes[2, 1].set_title('Risk Metrics Summary')
    axes[2, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def print_advanced_metrics(advanced_metrics: Dict[str, Any], period_name: str = "Performance"):
    """Print the advanced metrics in a formatted way"""
    print(f"\n{period_name.upper()} - ADVANCED METRICS:")
    print("=" * 80)
    
    # Basic metrics
    print(f"\nBASIC STATISTICS:")
    print(f"  Standard Deviation (R per Trade): {advanced_metrics['std_r_per_trade']:.4f}")
    print(f"  Maximum Historical Drawdown (R): {advanced_metrics['max_drawdown_r']:.4f}")
    print(f"  Average Trades Per Day: {advanced_metrics['avg_trades_per_day']:.2f}")
    print(f"  Average R per Day: {advanced_metrics['avg_r_per_day']:.4f}")
    print(f"  Average R per Week: {advanced_metrics['avg_r_per_week']:.4f}")
    print(f"  Total Trading Days: {advanced_metrics['total_trading_days']}")
    print(f"  Trading Day Ratio: {advanced_metrics['trading_day_ratio']:.1%}")
    
    # Risk-adjusted metrics
    print(f"\nRISK-ADJUSTED METRICS:")
    print(f"  Sharpe Ratio (R, annualized): {advanced_metrics['sharpe_ratio_r']:.4f}")
    print(f"  Calmar Ratio (R): {advanced_metrics['calmar_ratio_r']:.4f}")
    
    # Distribution statistics
    if advanced_metrics['r_per_trade_distribution']:
        dist = advanced_metrics['r_per_trade_distribution']
        print(f"\nR PER TRADE DISTRIBUTION:")
        print(f"  Mean: {dist['mean']:.4f}")
        print(f"  Median: {dist['median']:.4f}")
        print(f"  Skewness: {dist['skewness']:.4f}")
        print(f"  Kurtosis: {dist['kurtosis']:.4f}")
        print(f"  25th Percentile: {dist['q1']:.4f}")
        print(f"  75th Percentile: {dist['q3']:.4f}")
        print(f"  Minimum: {dist['min']:.4f}")
        print(f"  Maximum: {dist['max']:.4f}")
    
    # Interpretations
    print(f"\nINTERPRETATIONS:")
    
    # Volatility interpretation
    if advanced_metrics['std_r_per_trade'] < 0.5:
        print("  ✅ Low volatility in R per trade")
    elif advanced_metrics['std_r_per_trade'] < 1.0:
        print("  ⚠️  Moderate volatility in R per trade")
    else:
        print("  ❌ High volatility in R per trade")
    
    # Drawdown interpretation
    if advanced_metrics['max_drawdown_r'] < 5:
        print("  ✅ Low maximum drawdown")
    elif advanced_metrics['max_drawdown_r'] < 15:
        print("  ⚠️  Moderate maximum drawdown")
    else:
        print("  ❌ High maximum drawdown")
    
    # Activity level interpretation
    if advanced_metrics['avg_trades_per_day'] > 2:
        print("  ✅ Active trading strategy")
    elif advanced_metrics['avg_trades_per_day'] > 0.5:
        print("  ⚠️  Moderate trading activity")
    else:
        print("  ❌ Low trading frequency")
    
    # Daily R performance interpretation
    if advanced_metrics['avg_r_per_day'] > 0.1:
        print(f"  ✅ Good daily R performance ({advanced_metrics['avg_r_per_day']:.3f}R/day)")
    elif advanced_metrics['avg_r_per_day'] > 0:
        print(f"  ⚠️  Marginal daily R performance ({advanced_metrics['avg_r_per_day']:.3f}R/day)")
    else:
        print(f"  ❌ Negative daily R performance ({advanced_metrics['avg_r_per_day']:.3f}R/day)")
    
    # Weekly R performance interpretation
    weekly_target = advanced_metrics['avg_r_per_day'] * 5
    if advanced_metrics['avg_r_per_week'] > weekly_target * 0.8:
        print(f"  ✅ Consistent weekly performance")
    else:
        print(f"  ⚠️  Weekly performance shows some inconsistency")

# =============================================================================
# CORRELATION ANALYSIS FUNCTIONS
# =============================================================================

def calculate_strategy_correlation(trades_df: pd.DataFrame, stable_strategies: List[str]) -> Dict[str, Any]:
    """
    Calculate correlation analysis between strategies' trades
    
    Returns correlation metrics including:
    - Pairwise correlation matrix
    - Average absolute correlation
    - Maximum correlation
    - Diversification score
    - Highly correlated pairs
    """
    
    if trades_df.empty or len(stable_strategies) < 2:
        return {
            'correlation_matrix': None,
            'avg_abs_correlation': 0.0,
            'max_abs_correlation': 0.0,
            'diversification_score': 0.0,
            'highly_correlated_pairs': [],
            'sign_correlation_matrix': None,
            'avg_sign_correlation': 0.0
        }
    
    # Filter for stable strategies only
    stable_trades = trades_df[trades_df['pattern'].str.contains('|'.join(stable_strategies))]
    
    if stable_trades.empty:
        return {
            'correlation_matrix': None,
            'avg_abs_correlation': 0.0,
            'max_abs_correlation': 0.0,
            'diversification_score': 0.0,
            'highly_correlated_pairs': [],
            'sign_correlation_matrix': None,
            'avg_sign_correlation': 0.0
        }
    
    # Create time-aligned returns matrix for correlation analysis
    # First, get all unique exit times
    all_exit_times = stable_trades['exit_time'].unique()
    
    # Create DataFrame with strategies as columns and time as index
    correlation_data = []
    
    # For each strategy, create a time series of returns
    for strategy in stable_strategies:
        # Get trades for this strategy
        strategy_trades = stable_trades[stable_trades['pattern'].str.contains(strategy)]
        
        if len(strategy_trades) < WFA_CONFIG['MIN_CORRELATION_SAMPLES']:
            # Skip strategies with insufficient data
            continue
        
        # Create series with normalized returns (R multiples)
        strategy_returns = strategy_trades.set_index('exit_time')['pnl']
        
        # Create empty series for all exit times
        full_series = pd.Series(index=pd.to_datetime(all_exit_times), dtype=float)
        
        # Fill with actual returns where trades exist
        for time, ret in strategy_returns.items():
            full_series[pd.to_datetime(time)] = ret
        
        # Forward fill for missing values (assuming no return = 0)
        full_series = full_series.fillna(0)
        
        correlation_data.append(full_series)
    
    if len(correlation_data) < 2:
        return {
            'correlation_matrix': None,
            'avg_abs_correlation': 0.0,
            'max_abs_correlation': 0.0,
            'diversification_score': 0.0,
            'highly_correlated_pairs': [],
            'sign_correlation_matrix': None,
            'avg_sign_correlation': 0.0
        }
    
    # Create correlation matrix
    returns_df = pd.concat(correlation_data, axis=1)
    returns_df.columns = [f'Strategy_{i+1}' for i in range(len(correlation_data))]
    
    # Calculate Pearson correlation
    correlation_matrix = returns_df.corr()
    
    # Calculate sign correlation (correlation of winning/losing trades)
    sign_returns_df = returns_df.applymap(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    sign_correlation_matrix = sign_returns_df.corr()
    
    # Extract upper triangle (excluding diagonal)
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    
    # Calculate correlation metrics
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
                        'correlation': corr,
                        'sign_correlation': sign_correlation_matrix.iloc[i, j]
                    })
        
        # Calculate sign correlation metrics
        sign_upper_triangle = sign_correlation_matrix.where(np.triu(np.ones(sign_correlation_matrix.shape), k=1).astype(bool))
        avg_sign_correlation = sign_upper_triangle.abs().stack().dropna().mean()
        
        # Calculate diversification score (0-1, higher is better)
        # Lower correlations = better diversification
        diversification_score = max(0, 1 - (
            WFA_CONFIG['DIVERSIFICATION_SCORE_WEIGHTS']['avg_correlation'] * avg_abs_correlation +
            WFA_CONFIG['DIVERSIFICATION_SCORE_WEIGHTS']['max_correlation'] * max_abs_correlation +
            WFA_CONFIG['DIVERSIFICATION_SCORE_WEIGHTS']['sign_correlation'] * avg_sign_correlation
        ))
        
    else:
        avg_abs_correlation = 0.0
        max_abs_correlation = 0.0
        avg_sign_correlation = 0.0
        highly_correlated_pairs = []
        diversification_score = 0.0
    
    return {
        'correlation_matrix': correlation_matrix,
        'sign_correlation_matrix': sign_correlation_matrix,
        'avg_abs_correlation': avg_abs_correlation,
        'max_abs_correlation': max_abs_correlation,
        'avg_sign_correlation': avg_sign_correlation,
        'diversification_score': diversification_score,
        'highly_correlated_pairs': highly_correlated_pairs,
        'strategy_names': returns_df.columns.tolist()
    }

def analyze_trade_overlap(trades_df: pd.DataFrame, stable_strategies: List[str]) -> Dict[str, Any]:
    """
    Analyze temporal overlap between strategies' trades
    """
    if trades_df.empty or len(stable_strategies) < 2:
        return {
            'overlap_matrix': None,
            'avg_overlap': 0.0,
            'max_overlap': 0.0,
            'concurrent_trades_distribution': None
        }
    
    # Filter for stable strategies
    stable_trades = trades_df[trades_df['pattern'].str.contains('|'.join(stable_strategies))]
    
    if stable_trades.empty:
        return {
            'overlap_matrix': None,
            'avg_overlap': 0.0,
            'max_overlap': 0.0,
            'concurrent_trades_distribution': None
        }
    
    # Create overlap matrix
    n_strategies = len(stable_strategies)
    overlap_matrix = pd.DataFrame(0, index=stable_strategies, columns=stable_strategies)
    
    # Track concurrent trades over time
    timeline = []
    
    # Create time series of active trades
    for _, trade in stable_trades.iterrows():
        timeline.append({
            'time': trade['entry_time'],
            'strategy': trade['pattern'].split('_')[1],  # Extract strategy number
            'action': 'enter',
            'trade_id': trade['trade_id']
        })
        timeline.append({
            'time': trade['exit_time'],
            'strategy': trade['pattern'].split('_')[1],
            'action': 'exit',
            'trade_id': trade['trade_id']
        })
    
    # Sort timeline by time
    timeline.sort(key=lambda x: x['time'])
    
    # Calculate concurrent trades
    active_trades = set()
    strategy_active_trades = {strategy: set() for strategy in stable_strategies}
    concurrent_counts = []
    
    for event in timeline:
        strategy_num = event['strategy']
        
        if event['action'] == 'enter':
            active_trades.add(event['trade_id'])
            strategy_active_trades[strategy_num].add(event['trade_id'])
        else:
            active_trades.discard(event['trade_id'])
            strategy_active_trades[strategy_num].discard(event['trade_id'])
        
        concurrent_counts.append(len(active_trades))
        
        # Update overlap matrix when trades overlap
        if event['action'] == 'enter':
            for other_strategy in strategy_active_trades:
                if other_strategy != strategy_num and strategy_active_trades[other_strategy]:
                    # Count overlap
                    overlap_matrix.loc[strategy_num, other_strategy] += 1
    
    # Normalize overlap matrix
    total_trades = stable_trades['pattern'].value_counts()
    for i in stable_strategies:
        for j in stable_strategies:
            if i != j and total_trades.get(f'strategy_{i}', 0) > 0:
                overlap_matrix.loc[i, j] = overlap_matrix.loc[i, j] / total_trades[f'strategy_{i}']
    
    # Calculate metrics
    if len(concurrent_counts) > 0:
        avg_concurrent = np.mean(concurrent_counts)
        max_concurrent = np.max(concurrent_counts)
        
        # Distribution of concurrent trades
        concurrent_dist = pd.Series(concurrent_counts).value_counts().sort_index()
    else:
        avg_concurrent = 0.0
        max_concurrent = 0.0
        concurrent_dist = pd.Series()
    
    return {
        'overlap_matrix': overlap_matrix,
        'avg_concurrent_trades': avg_concurrent,
        'max_concurrent_trades': max_concurrent,
        'concurrent_trades_distribution': concurrent_dist
    }

def print_correlation_analysis(correlation_results: Dict[str, Any], period_name: str):
    """Print detailed correlation analysis results"""
    print(f"\n{period_name.upper()} CORRELATION ANALYSIS:")
    print("-" * 50)
    
    if correlation_results['correlation_matrix'] is None:
        print("  Insufficient data for correlation analysis")
        return
    
    print(f"  Strategy Count: {len(correlation_results['strategy_names'])}")
    print(f"  Average Absolute Correlation: {correlation_results['avg_abs_correlation']:.3f}")
    print(f"  Maximum Absolute Correlation: {correlation_results['max_abs_correlation']:.3f}")
    print(f"  Average Sign Correlation: {correlation_results['avg_sign_correlation']:.3f}")
    print(f"  Diversification Score: {correlation_results['diversification_score']:.3f}")
    
    if correlation_results['highly_correlated_pairs']:
        print(f"\n  HIGHLY CORRELATED PAIRS (>{WFA_CONFIG['CORRELATION_THRESHOLD']}):")
        for pair_info in correlation_results['highly_correlated_pairs']:
            pair = pair_info['pair']
            corr = pair_info['correlation']
            sign_corr = pair_info['sign_correlation']
            print(f"    {pair[0]} vs {pair[1]}: {corr:.3f} (Sign: {sign_corr:.3f})")
            if abs(corr) > 0.8:
                print("      ⚠️  VERY HIGH CORRELATION - Diversification benefits may be limited")
            elif abs(corr) > 0.6:
                print("      ⚠️  Moderate-high correlation - Monitor closely")
    else:
        print(f"\n  No highly correlated pairs (all < {WFA_CONFIG['CORRELATION_THRESHOLD']})")
        print("  ✅ Good diversification potential")
    
    # Diversification recommendation
    if correlation_results['diversification_score'] > 0.7:
        print("\n  ✅ EXCELLENT DIVERSIFICATION")
    elif correlation_results['diversification_score'] > 0.5:
        print("\n  ⚠️  MODERATE DIVERSIFICATION - Consider adding uncorrelated strategies")
    else:
        print("\n  ❌ POOR DIVERSIFICATION - Strategies are highly correlated")

def plot_correlation_matrices(correlation_results: Dict[str, Any], period_name: str):
    """Plot correlation matrices for strategy analysis"""
    if correlation_results['correlation_matrix'] is None:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{period_name} Strategy Correlation Analysis', fontsize=14, fontweight='bold')
    
    # Plot 1: Return Correlation Matrix
    im1 = axes[0].imshow(correlation_results['correlation_matrix'].values, 
                        cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[0].set_title('Return Correlation Matrix')
    axes[0].set_xticks(range(len(correlation_results['strategy_names'])))
    axes[0].set_yticks(range(len(correlation_results['strategy_names'])))
    axes[0].set_xticklabels(correlation_results['strategy_names'], rotation=45, ha='right')
    axes[0].set_yticklabels(correlation_results['strategy_names'])
    
    # Add correlation values to heatmap
    for i in range(len(correlation_results['strategy_names'])):
        for j in range(len(correlation_results['strategy_names'])):
            text = axes[0].text(j, i, f'{correlation_results["correlation_matrix"].iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot 2: Sign Correlation Matrix
    if correlation_results['sign_correlation_matrix'] is not None:
        im2 = axes[1].imshow(correlation_results['sign_correlation_matrix'].values, 
                           cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        axes[1].set_title('Sign Correlation Matrix')
        axes[1].set_xticks(range(len(correlation_results['strategy_names'])))
        axes[1].set_yticks(range(len(correlation_results['strategy_names'])))
        axes[1].set_xticklabels(correlation_results['strategy_names'], rotation=45, ha='right')
        axes[1].set_yticklabels(correlation_results['strategy_names'])
        
        # Add correlation values to heatmap
        for i in range(len(correlation_results['strategy_names'])):
            for j in range(len(correlation_results['strategy_names'])):
                text = axes[1].text(j, i, f'{correlation_results["sign_correlation_matrix"].iloc[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot 3: Correlation Distribution
    if correlation_results['correlation_matrix'] is not None:
        # Extract upper triangle values
        corr_values = []
        n = len(correlation_results['correlation_matrix'])
        for i in range(n):
            for j in range(i+1, n):
                corr_values.append(correlation_results['correlation_matrix'].iloc[i, j])
        
        axes[2].hist(corr_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[2].axvline(x=WFA_CONFIG['CORRELATION_THRESHOLD'], color='red', linestyle='--',
                       label=f'High Correlation Threshold ({WFA_CONFIG["CORRELATION_THRESHOLD"]})')
        axes[2].axvline(x=-WFA_CONFIG['CORRELATION_THRESHOLD'], color='red', linestyle='--')
        axes[2].axvline(x=np.mean(corr_values), color='green', linestyle='-',
                       label=f'Mean ({np.mean(corr_values):.3f})')
        axes[2].set_xlabel('Correlation Coefficient')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Correlation Distribution')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# WALK-FORWARD ANALYSIS FRAMEWORK
# =============================================================================

def create_walk_forward_windows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Create walk-forward windows with specified training/testing periods"""
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
    for i, window in enumerate(windows):
        print(f"Window {i+1}: Train {window['train_start'].strftime('%Y-%m-%d')} to {window['train_end'].strftime('%Y-%m-%d')} "
              f"| Test {window['test_start'].strftime('%Y-%m-%d')} to {window['test_end'].strftime('%Y-%m-%d')}")
    
    return windows

def calculate_net_expectancy(trades_df: pd.DataFrame) -> float:
    """Calculate net expectancy: E = (W × Avg Win) + (L × -1)"""
    if len(trades_df) == 0:
        return 0.0
    
    win_rate = trades_df['win'].mean()
    avg_winning_pnl = trades_df[trades_df['win']]['pnl'].mean() if any(trades_df['win']) else 0
    
    net_expectancy = (win_rate * avg_winning_pnl) + ((1 - win_rate) * -1)
    return net_expectancy

def calculate_stable_strategies(trades_df: pd.DataFrame) -> List[str]:
    """
    Select top strategies based on stability criteria, not just performance
    """
    if trades_df.empty:
        return []
    
    strategy_stats = []
    
    # Analyze each strategy
    for strategy in trades_df['pattern'].str.extract(r'(strategy_\d+)')[0].unique():
        if pd.isna(strategy):
            continue
            
        strategy_trades = trades_df[trades_df['pattern'].str.contains(strategy)]
        
        # Skip strategies with insufficient data
        if len(strategy_trades) < WFA_CONFIG['MIN_SAMPLE_SIZE']:
            continue
            
        net_expectancy = calculate_net_expectancy(strategy_trades)
        
        # Skip strategies with poor performance
        if net_expectancy < WFA_CONFIG['MIN_EXPECTANCY']:
            continue
        
        win_rate = strategy_trades['win'].mean()
        total_trades = len(strategy_trades)
        
        # Calculate stability score components
        expectancy_component = net_expectancy
        sample_size_component = np.log(total_trades) / np.log(100)  # Normalized
        win_rate_component = win_rate
        
        # Calculate weighted stability score
        stability_score = (
            WFA_CONFIG['STABILITY_WEIGHTS']['expectancy'] * expectancy_component +
            WFA_CONFIG['STABILITY_WEIGHTS']['sample_size'] * sample_size_component + 
            WFA_CONFIG['STABILITY_WEIGHTS']['win_rate'] * win_rate_component
        )
        
        strategy_stats.append({
            'strategy': strategy,
            'net_expectancy': net_expectancy,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'stability_score': stability_score
        })
    
    if not strategy_stats:
        print("No strategies met stability criteria")
        return []
    
    # Sort by stability score
    strategy_df = pd.DataFrame(strategy_stats)
    strategy_df = strategy_df.sort_values('stability_score', ascending=False)
    
    # Select all stable strategies (no fixed limit)
    stable_strategies = strategy_df['strategy'].tolist()
    
    print(f"Stable strategies by stability score:")
    for _, row in strategy_df.iterrows():
        print(f"  {row['strategy']} - Stability: {row['stability_score']:.3f}, "
              f"Exp: {row['net_expectancy']:.3f}R, WR: {row['win_rate']:.1%}, "
              f"Trades: {int(row['total_trades'])}")
    
    return stable_strategies

def analyze_performance_decay(is_metrics: Dict, oos_metrics: Dict) -> Dict[str, Any]:
    """Analyze performance decay from IS to OOS"""
    
    # Calculate decay percentages
    if is_metrics['net_expectancy'] != 0:
        expectancy_decay = (is_metrics['net_expectancy'] - oos_metrics['net_expectancy']) / abs(is_metrics['net_expectancy'])
    else:
        expectancy_decay = 1.0 if oos_metrics['net_expectancy'] < 0 else 0.0
        
    win_rate_decay = (is_metrics['win_rate'] - oos_metrics['win_rate']) / is_metrics['win_rate'] if is_metrics['win_rate'] > 0 else 1.0
    
    # Check against thresholds
    passes_expectancy_decay = expectancy_decay <= WFA_CONFIG['MAX_EXPECTANCY_DECAY']
    passes_winrate_decay = win_rate_decay <= WFA_CONFIG['MAX_WINRATE_DECAY']
    passes_min_expectancy = oos_metrics['net_expectancy'] >= WFA_CONFIG['MIN_OOS_EXPECTANCY']
    passes_all = passes_expectancy_decay and passes_winrate_decay and passes_min_expectancy
    
    return {
        'expectancy_decay_pct': expectancy_decay,
        'win_rate_decay_pct': win_rate_decay,
        'passes_expectancy_decay': passes_expectancy_decay,
        'passes_winrate_decay': passes_winrate_decay,
        'passes_min_expectancy': passes_min_expectancy,
        'passes_all': passes_all
    }

# =============================================================================
# DATA LOADING FUNCTION
# =============================================================================

DATA_PATH = Path(r"C:\Users\ericj\Downloads\EURUSD60(7th of November 2025 FxPro).csv")

def load_data(path: Path) -> pd.DataFrame:
    print(f"Loading data from: {path}")
    try:
        df = pd.read_csv(
            path,
            header=None,
            names=["Date", "Time", "Open", "High", "Low", "Close", "Volume"],
            dayfirst=False,
        )
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df = df.set_index('Datetime')[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        df['Bullish'] = df['Close'] > df['Open']
        df['Bearish'] = df['Close'] < df['Open']
        df['Body'] = abs(df['Close'] - df['Open'])
        
        prev_close = df['Close'].shift(1)
        tr = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - prev_close), abs(df['Low'] - prev_close)))
        df['ATR'] = tr.rolling(14, min_periods=1).mean()
        
        print(f"Loaded {len(df)} bars from {df.index.min()} to {df.index.max()}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

# =============================================================================
# CORE UTILITY FUNCTIONS
# =============================================================================

def calculate_position_size(current_capital: float, risk_distance: float, commission_per_lot: float, risk_pct: float) -> Tuple[float, float, float]:
    """Calculate position size with commission"""
    R = current_capital * risk_pct
    D = risk_distance * 100000
    C = commission_per_lot
    
    if (D + C) <= 0:
        return 0.0, 0.0, 0.0
    
    L = R / (D + C)
    actual_risk = L * (D + C)
    commission_cost = L * C
    
    return L, actual_risk, commission_cost

def calculate_take_profit_for_net_rr(entry_price: float, stop_loss: float, side: str, 
                                   commission_per_lot: float, net_rr_ratio: float) -> float:
    """Calculate take profit for net risk-reward after commission"""
    risk_distance = abs(entry_price - stop_loss)
    D = risk_distance * 100000
    C = commission_per_lot
    
    P_dollar = net_rr_ratio * D + (net_rr_ratio + 1) * C
    P_price = P_dollar / 100000
    
    if side == 'bull':
        return entry_price + P_price
    else:
        return entry_price - P_price

def check_margin_requirement(position_size: float, entry_price: float, current_capital: float, 
                           leverage: float, margin_buffer_pct: float) -> Tuple[bool, float, float]:
    """Check margin requirements"""
    contract_size = position_size * 100000
    required_margin = (contract_size * entry_price) / leverage
    max_allowed_margin = current_capital * margin_buffer_pct
    
    if required_margin <= max_allowed_margin:
        return True, required_margin, position_size
    else:
        max_allowed_lots = (max_allowed_margin * leverage) / (entry_price * 100000)
        return False, required_margin, max_allowed_lots

# =============================================================================
# STRATEGY 1: ORIGINAL PATTERN COMPLETION
# =============================================================================

def find_patterns_strategy_1(df: pd.DataFrame, allowed_hours: List[int], net_rr_ratio: float) -> list:
    """Strategy 1: Original Pattern Completion"""
    setups = []
    
    for i in range(1, len(df) - 2):
        Bar1 = df.iloc[i]
        Bar2 = df.iloc[i+1]
        
        # === BEARISH SETUP → BUY LIMIT ===
        if (Bar1['Bullish'] and 
            Bar2['Bearish'] and 
            Bar2['Close'] < Bar1['Low'] and
            Bar2.name.hour in allowed_hours):
            
            risk_distance = max(Bar1['High'], Bar2['High']) - Bar1['Low']
            entry = Bar2['Low'] - risk_distance - STRATEGY_CONFIG['SPREAD']
            stop_loss = entry - risk_distance
            
            if risk_distance < 1e-5:
                continue
            
            take_profit = calculate_take_profit_for_net_rr(
                entry, stop_loss, 'bull', STRATEGY_CONFIG['COMMISSION_PER_LOT'], net_rr_ratio
            )
            
            setups.append({
                'bar1_index': i,
                'bar2_index': i+1,
                'trigger_index': i+2,
                'side': 'bull',
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_distance': risk_distance,
                'pattern': 'strategy_1_bearish_setup_buy_limit',
                'pattern_completion_time': Bar2.name,
            })
        
        # === BULLISH SETUP → SELL LIMIT ===
        elif (Bar1['Bearish'] and 
              Bar2['Bullish'] and 
              Bar2['Close'] > Bar1['High'] and
              Bar2.name.hour in allowed_hours):
            
            risk_distance = Bar1['High'] - min(Bar1['Low'], Bar2['Low'])
            entry = Bar2['High'] + risk_distance + STRATEGY_CONFIG['SPREAD']
            stop_loss = entry + risk_distance
            
            if risk_distance < 1e-5:
                continue
            
            take_profit = calculate_take_profit_for_net_rr(
                entry, stop_loss, 'bear', STRATEGY_CONFIG['COMMISSION_PER_LOT'], net_rr_ratio
            )
            
            setups.append({
                'bar1_index': i,
                'bar2_index': i+1,
                'trigger_index': i+2,
                'side': 'bear',
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_distance': risk_distance,
                'pattern': 'strategy_1_bullish_setup_sell_limit',
                'pattern_completion_time': Bar2.name,
            })
            
    return setups

# =============================================================================
# STRATEGY 2: DOUBLE OFFSET STRATEGY
# =============================================================================

def find_patterns_strategy_2(df: pd.DataFrame, allowed_hours: List[int], net_rr_ratio: float) -> list:
    """Strategy 2: Double Offset Strategy"""
    setups = []
    
    for i in range(1, len(df) - 2):
        Bar1 = df.iloc[i]
        Bar2 = df.iloc[i+1]
        
        if Bar2.name.hour not in allowed_hours:
            continue
        
        # === BEARISH SETUP → BUY LIMIT (DOUBLE OFFSET) ===
        if (Bar1['Bullish'] and 
            Bar2['Bearish'] and 
            Bar2['Close'] < Bar1['Low']):
            
            risk_distance = max(Bar1['High'], Bar2['High']) - Bar1['Low']
            previous_entry = Bar2['Low'] - risk_distance
            entry = previous_entry - risk_distance - STRATEGY_CONFIG['SPREAD']
            stop_loss = entry - risk_distance
            
            if risk_distance < 1e-5:
                continue
            
            take_profit = calculate_take_profit_for_net_rr(
                entry, stop_loss, 'bull', STRATEGY_CONFIG['COMMISSION_PER_LOT'], net_rr_ratio
            )
            
            setups.append({
                'bar1_index': i,
                'bar2_index': i+1,
                'trigger_index': i+2,
                'side': 'bull',
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_distance': risk_distance,
                'pattern': 'strategy_2_bearish_setup_buy_limit_offset',
                'pattern_completion_time': Bar2.name,
                'previous_entry': previous_entry,
            })
        
        # === BULLISH SETUP → SELL LIMIT (DOUBLE OFFSET) ===
        elif (Bar1['Bearish'] and 
              Bar2['Bullish'] and 
              Bar2['Close'] > Bar1['High']):
            
            risk_distance = Bar1['High'] - min(Bar1['Low'], Bar2['Low'])
            previous_entry = Bar2['High'] + risk_distance
            entry = previous_entry + risk_distance + STRATEGY_CONFIG['SPREAD']
            stop_loss = entry + risk_distance
            
            if risk_distance < 1e-5:
                continue
            
            take_profit = calculate_take_profit_for_net_rr(
                entry, stop_loss, 'bear', STRATEGY_CONFIG['COMMISSION_PER_LOT'], net_rr_ratio
            )
            
            setups.append({
                'bar1_index': i,
                'bar2_index': i+1,
                'trigger_index': i+2,
                'side': 'bear',
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_distance': risk_distance,
                'pattern': 'strategy_2_bullish_setup_sell_limit_offset',
                'pattern_completion_time': Bar2.name,
                'previous_entry': previous_entry,
            })
                
    return setups

# =============================================================================
# STRATEGY 3: TRIPLE OFFSET STRATEGY
# =============================================================================

def find_patterns_strategy_3(df: pd.DataFrame, allowed_hours: List[int], net_rr_ratio: float) -> list:
    """Strategy 3: Triple Offset Strategy"""
    setups = []
    
    for i in range(1, len(df) - 2):
        Bar1 = df.iloc[i]
        Bar2 = df.iloc[i+1]
        
        # BEARISH SETUP → BUY LIMIT (Triple Offset)
        if (Bar1['Bullish'] and 
            Bar2['Bearish'] and 
            Bar2['Close'] < Bar1['Low'] and
            Bar2.name.hour in allowed_hours):
            
            risk_distance = max(Bar1['High'], Bar2['High']) - Bar1['Low']
            entry = Bar2['Low'] - (3 * risk_distance) - STRATEGY_CONFIG['SPREAD']
            stop_loss = entry - risk_distance
            
            if risk_distance < 1e-5:
                continue
            
            take_profit = calculate_take_profit_for_net_rr(entry, stop_loss, 'bull', 
                                                         STRATEGY_CONFIG['COMMISSION_PER_LOT'], net_rr_ratio)
            
            setups.append({
                'bar1_index': i,
                'bar2_index': i+1,
                'trigger_index': i+2,
                'side': 'bull',
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_distance': risk_distance,
                'pattern': 'strategy_3_bearish_setup_buy_limit_triple_offset',
                'pattern_completion_time': Bar2.name,
            })
        
        # BULLISH SETUP → SELL LIMIT (Triple Offset)  
        elif (Bar1['Bearish'] and 
              Bar2['Bullish'] and 
              Bar2['Close'] > Bar1['High'] and
              Bar2.name.hour in allowed_hours):
            
            risk_distance = Bar1['High'] - min(Bar1['Low'], Bar2['Low'])
            entry = Bar2['High'] + (3 * risk_distance) + STRATEGY_CONFIG['SPREAD']
            stop_loss = entry + risk_distance
            
            if risk_distance < 1e-5:
                continue
            
            take_profit = calculate_take_profit_for_net_rr(entry, stop_loss, 'bear', 
                                                         STRATEGY_CONFIG['COMMISSION_PER_LOT'], net_rr_ratio)
            
            setups.append({
                'bar1_index': i,
                'bar2_index': i+1,
                'trigger_index': i+2,
                'side': 'bear',
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_distance': risk_distance,
                'pattern': 'strategy_3_bullish_setup_sell_limit_triple_offset',
                'pattern_completion_time': Bar2.name,
            })
                
    return setups

# =============================================================================
# STRATEGY 4: ENTRY AT OLD STOP LOSS LEVEL
# =============================================================================

def find_patterns_strategy_4(df: pd.DataFrame, allowed_hours: List[int], net_rr_ratio: float, validation_check: bool) -> list:
    """Strategy 4: Entry at Old Stop Loss Level - FIXED VERSION"""
    setups = []
    
    for i in range(1, len(df) - 2):
        Bar1 = df.iloc[i]
        Bar2 = df.iloc[i+1]
        
        # BEARISH SETUP PATTERN - FIXED STOP LOSS
        if (Bar1['Bullish'] and 
            Bar2['Bearish'] and 
            Bar2['Close'] < Bar1['Low'] and
            Bar2.name.hour in allowed_hours):
            
            entry = max(Bar1['High'], Bar2['High']) + STRATEGY_CONFIG['SPREAD']
            risk_distance = entry - Bar1['Low']
            # FIX: Stop loss should be ABOVE entry for bearish trades
            stop_loss = entry + risk_distance  # CORRECTED
            
            if risk_distance < 1e-5:
                continue
            
            take_profit = calculate_take_profit_for_net_rr(
                entry, stop_loss, 'bear', STRATEGY_CONFIG['COMMISSION_PER_LOT'], net_rr_ratio
            )
            
            setups.append({
                'bar1_index': i,
                'bar2_index': i+1,
                'trigger_index': i+2,
                'side': 'bear',
                'entry': entry,
                'stop_loss': stop_loss,  # Now correctly above entry
                'take_profit': take_profit,
                'risk_distance': risk_distance,
                'pattern': 'strategy_4_bearish_setup',
                'pattern_completion_time': Bar2.name,
                'original_entry': Bar1['Low'],
                'validation_required': validation_check,
            })
        
        # BULLISH SETUP PATTERN - FIXED STOP LOSS
        elif (Bar1['Bearish'] and 
              Bar2['Bullish'] and 
              Bar2['Close'] > Bar1['High'] and
              Bar2.name.hour in allowed_hours):
            
            entry = min(Bar1['Low'], Bar2['Low']) - STRATEGY_CONFIG['SPREAD']
            risk_distance = Bar1['High'] - entry
            # FIX: Stop loss should be BELOW entry for bullish trades  
            stop_loss = entry - risk_distance  # CORRECTED
            
            if risk_distance < 1e-5:
                continue
            
            take_profit = calculate_take_profit_for_net_rr(
                entry, stop_loss, 'bull', STRATEGY_CONFIG['COMMISSION_PER_LOT'], net_rr_ratio
            )
            
            setups.append({
                'bar1_index': i,
                'bar2_index': i+1,
                'trigger_index': i+2,
                'side': 'bull',
                'entry': entry,
                'stop_loss': stop_loss,  # Now correctly below entry
                'take_profit': take_profit,
                'risk_distance': risk_distance,
                'pattern': 'strategy_4_bullish_setup',
                'pattern_completion_time': Bar2.name,
                'original_entry': Bar1['High'],
                'validation_required': validation_check,
            })
            
    return setups

# =============================================================================
# STRATEGY 5: REVERSAL TRADING STRATEGY
# =============================================================================

def find_patterns_strategy_5(df: pd.DataFrame, allowed_hours: List[int], net_rr_ratio: float, validation_check: bool) -> list:
    """Strategy 5: Reversal Trading Strategy"""
    setups = []
    
    for i in range(1, len(df) - 2):
        Bar1 = df.iloc[i]
        Bar2 = df.iloc[i+1]
        
        # === BEARISH SETUP PATTERN ===
        if (Bar1['Bullish'] and 
            Bar2['Bearish'] and 
            Bar2['Close'] < Bar1['Low'] and
            Bar2.name.hour in allowed_hours):
            
            highest_high = max(Bar1['High'], Bar2['High'])
            risk_distance = highest_high - Bar1['Low']
            entry = highest_high + risk_distance + STRATEGY_CONFIG['SPREAD']
            stop_loss = entry + risk_distance
            
            if risk_distance < 1e-5:
                continue
            
            take_profit = calculate_take_profit_for_net_rr(entry, stop_loss, 'bear', 
                                                         STRATEGY_CONFIG['COMMISSION_PER_LOT'], net_rr_ratio)
            
            setups.append({
                'bar1_index': i,
                'bar2_index': i+1,
                'trigger_index': i+2,
                'side': 'bear',
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_distance': risk_distance,
                'pattern': 'strategy_5_bearish_setup',
                'pattern_completion_time': Bar2.name,
                'original_entry': Bar1['Low'],
                'highest_high': highest_high,
                'validation_required': validation_check,
            })
        
        # === BULLISH SETUP PATTERN ===
        elif (Bar1['Bearish'] and 
              Bar2['Bullish'] and 
              Bar2['Close'] > Bar1['High'] and
              Bar2.name.hour in allowed_hours):
            
            lowest_low = min(Bar1['Low'], Bar2['Low'])
            risk_distance = Bar1['High'] - lowest_low
            entry = lowest_low - risk_distance - STRATEGY_CONFIG['SPREAD']
            stop_loss = entry - risk_distance
            
            if risk_distance < 1e-5:
                continue
            
            take_profit = calculate_take_profit_for_net_rr(entry, stop_loss, 'bull', 
                                                         STRATEGY_CONFIG['COMMISSION_PER_LOT'], net_rr_ratio)
            
            setups.append({
                'bar1_index': i,
                'bar2_index': i+1,
                'trigger_index': i+2,
                'side': 'bull',
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_distance': risk_distance,
                'pattern': 'strategy_5_bullish_setup',
                'pattern_completion_time': Bar2.name,
                'original_entry': Bar1['High'],
                'lowest_low': lowest_low,
                'validation_required': validation_check,
            })
                
    return setups

# =============================================================================
# COMBINED STRATEGY EXECUTION WITH POSITION TRACKING
# =============================================================================

def validation_check(df: pd.DataFrame, setup: dict, trigger_idx: int) -> bool:
    """Validation check for strategies 4 and 5"""
    if not setup.get('validation_required', False):
        return True
    
    if trigger_idx >= len(df):
        return False
    
    bar3 = df.iloc[trigger_idx]
    
    if setup['side'] == 'bear':
        return bar3['High'] >= setup['original_entry']
    else:
        return bar3['Low'] <= setup['original_entry']

def execute_combined_strategy(df: pd.DataFrame, enabled_strategies: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    EXECUTE COMBINED TRADING STRATEGY with margin and position tracking
    Returns: (trades_df, metrics_dict)
    """
    if enabled_strategies is None:
        # Use all strategies if not specified
        enabled_strategies = ['strategy_1', 'strategy_2', 'strategy_3', 'strategy_4', 'strategy_5']
    
    print("Executing Combined Trading Strategy...")
    
    # Collect setups from all enabled strategies
    all_setups = []
    
    # Strategy 1
    if 'strategy_1' in enabled_strategies and STRATEGY_CONFIG['STRATEGY_1_ENABLED']:
        setups_1 = find_patterns_strategy_1(
            df, 
            STRATEGY_CONFIG['STRATEGY_1']['ALLOWED_HOURS'],
            STRATEGY_CONFIG['STRATEGY_1']['NET_RR_RATIO']
        )
        all_setups.extend(setups_1)
    
    # Strategy 2
    if 'strategy_2' in enabled_strategies and STRATEGY_CONFIG['STRATEGY_2_ENABLED']:
        setups_2 = find_patterns_strategy_2(
            df,
            STRATEGY_CONFIG['STRATEGY_2']['ALLOWED_HOURS'],
            STRATEGY_CONFIG['STRATEGY_2']['NET_RR_RATIO']
        )
        all_setups.extend(setups_2)
    
    # Strategy 3
    if 'strategy_3' in enabled_strategies and STRATEGY_CONFIG['STRATEGY_3_ENABLED']:
        setups_3 = find_patterns_strategy_3(
            df,
            STRATEGY_CONFIG['STRATEGY_3']['ALLOWED_HOURS'],
            STRATEGY_CONFIG['STRATEGY_3']['NET_RR_RATIO']
        )
        all_setups.extend(setups_3)
    
    # Strategy 4
    if 'strategy_4' in enabled_strategies and STRATEGY_CONFIG['STRATEGY_4_ENABLED']:
        setups_4 = find_patterns_strategy_4(
            df,
            STRATEGY_CONFIG['STRATEGY_4']['ALLOWED_HOURS'],
            STRATEGY_CONFIG['STRATEGY_4']['NET_RR_RATIO'],
            STRATEGY_CONFIG['STRATEGY_4']['VALIDATION_CHECK']
        )
        all_setups.extend(setups_4)
    
    # Strategy 5
    if 'strategy_5' in enabled_strategies and STRATEGY_CONFIG['STRATEGY_5_ENABLED']:
        setups_5 = find_patterns_strategy_5(
            df,
            STRATEGY_CONFIG['STRATEGY_5']['ALLOWED_HOURS'],
            STRATEGY_CONFIG['STRATEGY_5']['NET_RR_RATIO'],
            STRATEGY_CONFIG['STRATEGY_5']['VALIDATION_CHECK']
        )
        all_setups.extend(setups_5)
    
    print(f"Total setups found: {len(all_setups)}")
    
    # Sort setups by trigger index for sequential processing
    all_setups.sort(key=lambda x: x['trigger_index'])
    
    # Initialize tracking variables
    trades = []
    current_capital = STRATEGY_CONFIG['INITIAL_CAPITAL']
    
    # New tracking variables
    max_margin_used = 0.0
    max_margin_ratio = 0.0
    max_concurrent_positions = 0
    current_open_positions = []  # List of active trades
    
    # Create a timeline of events
    events = []
    
    for setup in all_setups:
        trigger_idx = setup['trigger_index']
        
        if trigger_idx >= len(df):
            continue
            
        # Validation check for strategies 4 and 5
        if setup.get('validation_required', False):
            if not validation_check(df, setup, trigger_idx):
                continue
        
        # Get strategy-specific parameters
        strategy_params = None
        if 'strategy_1' in setup['pattern']:
            strategy_params = STRATEGY_CONFIG['STRATEGY_1']
        elif 'strategy_2' in setup['pattern']:
            strategy_params = STRATEGY_CONFIG['STRATEGY_2']
        elif 'strategy_3' in setup['pattern']:
            strategy_params = STRATEGY_CONFIG['STRATEGY_3']
        elif 'strategy_4' in setup['pattern']:
            strategy_params = STRATEGY_CONFIG['STRATEGY_4']
        elif 'strategy_5' in setup['pattern']:
            strategy_params = STRATEGY_CONFIG['STRATEGY_5']
        
        if not strategy_params:
            continue
        
        # Position sizing with commission
        position_size, actual_risk, commission_cost = calculate_position_size(
            current_capital, setup['risk_distance'], 
            STRATEGY_CONFIG['COMMISSION_PER_LOT'], STRATEGY_CONFIG['RISK_PCT']
        )
        
        # Margin check
        sufficient_margin, required_margin, max_allowed_lots = check_margin_requirement(
            position_size, setup['entry'], current_capital,
            STRATEGY_CONFIG['LEVERAGE'], STRATEGY_CONFIG['MARGIN_BUFFER_PCT']
        )

        # Safety: define lot rules
        MIN_LOT = 0.01
        LOT_STEP = 0.01

        # If margin insufficient, cap to max_allowed_lots
        if not sufficient_margin:
            capped_lots = max_allowed_lots
            capped_lots = math.floor(capped_lots / LOT_STEP) * LOT_STEP

            if capped_lots < MIN_LOT:
                continue

            position_size = capped_lots

            # Recompute actual_risk and commission based on capped lots
            D = setup['risk_distance'] * 100000
            C = STRATEGY_CONFIG['COMMISSION_PER_LOT']
            actual_risk = position_size * (D + C)
            commission_cost = position_size * C

            # Recompute required margin
            contract_size = position_size * 100000
            required_margin = (contract_size * setup['entry']) / STRATEGY_CONFIG['LEVERAGE']
            max_allowed_margin = current_capital * STRATEGY_CONFIG['MARGIN_BUFFER_PCT']
            sufficient_margin = required_margin <= max_allowed_margin

            if not sufficient_margin:
                continue

        # Extra safety: if final position_size is below broker minimum, skip
        if position_size < MIN_LOT:
            continue
        
        # Limit order fill logic
        entry_found = False
        fill_idx = -1
        fill_bar = None
        
        max_trigger_candles = strategy_params['MAX_TRIGGER_CANDLES']
        
        for j in range(trigger_idx, min(trigger_idx + max_trigger_candles + 1, len(df))):
            bar = df.iloc[j]
            
            if setup['side'] == 'bear':
                if bar['High'] >= setup['entry']:
                    entry_found = True
                    fill_idx = j
                    fill_bar = bar
                    break
            else:
                if bar['Low'] <= setup['entry']:
                    entry_found = True
                    fill_idx = j
                    fill_bar = bar
                    break
        
        if not entry_found:
            continue
        
        # Store entry event for timeline tracking
        entry_time = fill_bar.name
        trade_id = f"{setup['pattern']}_{entry_time.strftime('%Y%m%d_%H%M')}"
        
        # Trade management
        entry, stop, tp = setup['entry'], setup['stop_loss'], setup['take_profit']
        risk_per_lot = setup['risk_distance']
        
        # Check immediate exit on fill bar
        exit_found = False
        exit_time = None
        exit_price = None
        exit_type = None
        bars_held = 0
        
        if setup['side'] == 'bull':
            stop_touched = fill_bar['Low'] <= stop
            tp_touched = fill_bar['High'] >= tp
        else:
            stop_touched = fill_bar['High'] >= stop
            tp_touched = fill_bar['Low'] <= tp
        
        if stop_touched and tp_touched:
            exit_price = stop
            pnl = -1.0
            exit_type = 'immediate_loss'
            bars_held = 0
            exit_found = True
            exit_time = entry_time
        elif stop_touched:
            exit_price = stop
            pnl = -1.0
            exit_type = 'immediate_loss'
            bars_held = 0
            exit_found = True
            exit_time = entry_time
        elif tp_touched:
            exit_price = tp
            pnl = strategy_params['NET_RR_RATIO']
            exit_type = 'immediate_win'
            bars_held = 0
            exit_found = True
            exit_time = entry_time
        
        if not exit_found:
            # Hold and monitor
            start_idx = fill_idx
            max_hold_bars = strategy_params['MAX_HOLD_BARS']
            
            for j in range(start_idx, min(start_idx + max_hold_bars + 1, len(df))):
                bar = df.iloc[j]
                
                if setup['side'] == 'bull':
                    if bar['Low'] <= stop:
                        exit_price = stop
                        exit_found = True
                    elif bar['High'] >= tp:
                        exit_price = tp
                        exit_found = True
                else:
                    if bar['High'] >= stop:
                        exit_price = stop
                        exit_found = True
                    elif bar['Low'] <= tp:
                        exit_price = tp
                        exit_found = True
                
                if exit_found:
                    if setup['side'] == 'bull':
                        gross_pnl = (exit_price - entry) / risk_per_lot
                    else:
                        gross_pnl = (entry - exit_price) / risk_per_lot
                    commission_r = commission_cost / actual_risk
                    pnl = gross_pnl - commission_r
                    bars_held = j - start_idx + 1
                    exit_type = 'tp_hit' if exit_price == tp else 'sl_hit'
                    exit_time = bar.name
                    break
            
            if not exit_found:
                # Timeout exit
                last_bar = df.iloc[min(start_idx + max_hold_bars, len(df) - 1)]
                if setup['side'] == 'bull':
                    gross_pnl = (last_bar['Close'] - entry) / risk_per_lot
                else:
                    gross_pnl = (entry - last_bar['Close']) / risk_per_lot
                commission_r = commission_cost / actual_risk
                pnl = gross_pnl - commission_r
                exit_price = last_bar['Close']
                bars_held = min(max_hold_bars, len(df) - start_idx)
                exit_type = 'timeout'
                exit_time = last_bar.name
        
        # Add to timeline events
        events.append({
            'time': entry_time,
            'type': 'entry',
            'trade_id': trade_id,
            'margin_required': required_margin,
            'position_size': position_size,
            'capital_at_entry': current_capital,
            'margin_ratio': required_margin / current_capital if current_capital > 0 else 0
        })
        
        events.append({
            'time': exit_time,
            'type': 'exit',
            'trade_id': trade_id,
            'pnl': pnl * actual_risk
        })
        
        # Calculate trade metrics
        trade_bars = df.iloc[fill_idx:fill_idx + bars_held] if bars_held > 0 else pd.DataFrame([fill_bar])
        min_low = trade_bars['Low'].min()
        max_high = trade_bars['High'].max()
        
        if setup['side'] == 'bull':
            mae_r = (entry - min_low) / risk_per_lot
        else:
            mae_r = (max_high - entry) / risk_per_lot
            
        win = pnl > 0
        
        # Update capital (compounding)
        dollar_pnl = pnl * actual_risk
        current_capital += dollar_pnl
        
        # Get pattern bars for analysis
        bar1 = df.iloc[setup['bar1_index']] 
        bar2 = df.iloc[setup['bar2_index']]
        
        trade = {
            'trade_id': trade_id,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'bar1_time': bar1.name,
            'bar2_time': bar2.name,
            'side': setup['side'],
            'entry': entry,
            'stop_loss': stop,
            'take_profit': tp,
            'exit_price': exit_price,
            'pnl': pnl,
            'dollar_pnl': dollar_pnl,
            'win': win,
            'bars_held': bars_held,
            'pattern': setup['pattern'],
            'exit_type': exit_type,
            'mae_r': mae_r,
            'actual_risk': actual_risk,
            'position_size': position_size,
            'commission_cost': commission_cost,
            'current_capital': current_capital,
            'required_margin': required_margin,
            'margin_sufficient': sufficient_margin,
            'bar1_high': bar1['High'],
            'bar1_low': bar1['Low'],
            'bar2_high': bar2['High'],
            'bar2_low': bar2['Low'],
            'hour_of_pattern': bar2.name.hour,
            'risk_distance': setup['risk_distance'],
            'trigger_candles_waited': fill_idx - trigger_idx,
        }
        
        # Add strategy-specific fields
        if 'previous_entry' in setup:
            trade['previous_entry'] = setup['previous_entry']
        if 'original_entry' in setup:
            trade['original_entry'] = setup['original_entry']
        if 'highest_high' in setup:
            trade['highest_high'] = setup['highest_high']
        if 'lowest_low' in setup:
            trade['lowest_low'] = setup['lowest_low']
            
        trades.append(trade)
    
    # Calculate the new metrics from events timeline
    if events:
        # Sort events by time
        events.sort(key=lambda x: x['time'])
        
        # Track concurrent positions and margin over time
        active_trades = set()
        current_total_margin = 0.0
        timeline_metrics = []
        
        for event in events:
            if event['type'] == 'entry':
                active_trades.add(event['trade_id'])
                current_total_margin += event['margin_required']
                
                # Update max concurrent positions
                if len(active_trades) > max_concurrent_positions:
                    max_concurrent_positions = len(active_trades)
                
                # Update max margin used
                if current_total_margin > max_margin_used:
                    max_margin_used = current_total_margin
                
                # Update max margin ratio
                margin_ratio = current_total_margin / event['capital_at_entry']
                if margin_ratio > max_margin_ratio:
                    max_margin_ratio = margin_ratio
                    
            elif event['type'] == 'exit':
                active_trades.discard(event['trade_id'])
                # Find the margin used for this trade (from entry event)
                entry_event = next((e for e in events if e['trade_id'] == event['trade_id'] and e['type'] == 'entry'), None)
                if entry_event:
                    current_total_margin -= entry_event['margin_required']
            
            timeline_metrics.append({
                'time': event['time'],
                'active_trades': len(active_trades),
                'total_margin': current_total_margin,
                'margin_ratio': current_total_margin / STRATEGY_CONFIG['INITIAL_CAPITAL']
            })
    
    # Compile metrics
    metrics = {
        'max_margin_used': max_margin_used,
        'max_margin_ratio': max_margin_ratio,
        'max_concurrent_positions': max_concurrent_positions,
        'initial_capital': STRATEGY_CONFIG['INITIAL_CAPITAL'],
        'final_capital': current_capital if trades else STRATEGY_CONFIG['INITIAL_CAPITAL'],
        'total_trades': len(trades)
    }
    
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        trades_df.set_index('entry_time', inplace=True)
        return trades_df, metrics
    
    print("No trades executed")
    return pd.DataFrame(), metrics

# =============================================================================
# WALK-FORWARD ANALYSIS WITH CORRELATION AND ADVANCED METRICS
# =============================================================================

def run_strategy_walk_forward_analysis(df: pd.DataFrame, 
                                     strategy_execution_func: Callable) -> pd.DataFrame:
    """
    Main walk-forward analysis engine for strategy evaluation with correlation analysis
    """
    print("=" * 100)
    print("STRATEGY WALK-FORWARD ANALYSIS WITH CORRELATION ANALYSIS & ADVANCED METRICS")
    print("=" * 100)
    print(f"Training Window: {WFA_CONFIG['TRAINING_YEARS']} years")
    print(f"Testing Window: {WFA_CONFIG['TESTING_YEARS']} year") 
    print(f"Step Forward: {WFA_CONFIG['STEP_FORWARD_YEARS']} year")
    print(f"Minimum OOS Expectancy: {WFA_CONFIG['MIN_OOS_EXPECTANCY']}")
    print(f"Stability Criteria: Min {WFA_CONFIG['MIN_SAMPLE_SIZE']} trades, Min {WFA_CONFIG['MIN_EXPECTANCY']}R expectancy")
    print(f"Correlation Threshold: {WFA_CONFIG['CORRELATION_THRESHOLD']}")
    print()
    
    windows = create_walk_forward_windows(df)
    
    if not windows:
        print("No valid windows created - check your data range")
        return pd.DataFrame()
    
    results = []
    all_stable_strategies = []
    last_window_correlation_data = {}
    last_window_advanced_metrics = {}
    
    for i, window in enumerate(windows):
        print(f"\n{'='*80}")
        print(f"Processing Window {i+1}/{len(windows)}")
        print(f"{'='*80}")
        print(f"Training: {window['train_start'].strftime('%Y-%m-%d')} to {window['train_end'].strftime('%Y-%m-%d')}")
        print(f"Testing:  {window['test_start'].strftime('%Y-%m-%d')} to {window['test_end'].strftime('%Y-%m-%d')}")
        
        # === IN-SAMPLE OPTIMIZATION ===
        print("\n" + "-" * 40)
        print("IN-SAMPLE OPTIMIZATION")
        print("-" * 40)
        
        # Run strategy on training data
        train_trades, train_metrics = strategy_execution_func(window['train_data'])
        
        if train_trades.empty:
            print("No trades in training period - skipping window")
            continue
        
        # Calculate STABLE strategies (not just top performing)
        stable_strategies = calculate_stable_strategies(train_trades)
        
        if not stable_strategies:
            print("No stable strategies found - skipping window")
            continue
            
        all_stable_strategies.extend(stable_strategies)
        
        # Calculate IS performance with stable strategies only
        is_stable_trades = train_trades[train_trades['pattern'].str.contains('|'.join(stable_strategies))]
        is_net_expectancy = calculate_net_expectancy(is_stable_trades)
        is_win_rate = is_stable_trades['win'].mean() if not is_stable_trades.empty else 0
        is_total_trades = len(is_stable_trades)
        
        # Calculate IS correlation analysis
        is_correlation_results = calculate_strategy_correlation(is_stable_trades, stable_strategies)
        
        # Calculate IS advanced metrics
        is_advanced_metrics = calculate_advanced_metrics(is_stable_trades)
        
        # Print IS metrics including correlation and advanced metrics
        print(f"\nIS PERFORMANCE METRICS ({len(stable_strategies)} stable strategies):")
        print(f"  Net Expectancy: {is_net_expectancy:.4f}R")
        print(f"  Win Rate: {is_win_rate:.2%}")
        print(f"  Total Trades: {is_total_trades}")
        print(f"  Max Margin Used: ${train_metrics['max_margin_used']:,.2f}")
        print(f"  Max Margin Ratio: {train_metrics['max_margin_ratio']:.2%}")
        print(f"  Max Concurrent Positions: {train_metrics['max_concurrent_positions']}")
        
        # Print IS advanced metrics
        print(f"\nIS ADVANCED METRICS:")
        print(f"  Std Dev of R per Trade: {is_advanced_metrics['std_r_per_trade']:.4f}")
        print(f"  Max Drawdown (R): {is_advanced_metrics['max_drawdown_r']:.4f}")
        print(f"  Avg Trades/Day: {is_advanced_metrics['avg_trades_per_day']:.2f}")
        print(f"  Avg R/Day: {is_advanced_metrics['avg_r_per_day']:.4f}")
        print(f"  Avg R/Week: {is_advanced_metrics['avg_r_per_week']:.4f}")
        
        # Print correlation analysis
        print_correlation_analysis(is_correlation_results, "IS Correlation")
        
        # === OUT-OF-SAMPLE VALIDATION ===
        print("\n" + "-" * 40)
        print("OUT-OF-SAMPLE VALIDATION")
        print("-" * 40)
        
        # Run strategy on test data (using all strategies for OOS validation)
        test_trades, test_metrics = strategy_execution_func(window['test_data'])
        
        if test_trades.empty:
            print("No trades in testing period - skipping window")
            continue
        
        # Filter OOS trades to only include stable strategies
        oos_stable_trades = test_trades[test_trades['pattern'].str.contains('|'.join(stable_strategies))]
        
        if oos_stable_trades.empty:
            print("No stable strategy trades in testing period - skipping window")
            continue
        
        oos_net_expectancy = calculate_net_expectancy(oos_stable_trades)
        oos_win_rate = oos_stable_trades['win'].mean()
        oos_total_trades = len(oos_stable_trades)
        
        # Calculate OOS correlation analysis
        oos_correlation_results = calculate_strategy_correlation(oos_stable_trades, stable_strategies)
        
        # Calculate OOS advanced metrics
        oos_advanced_metrics = calculate_advanced_metrics(oos_stable_trades)
        
        # Print OOS metrics including correlation and advanced metrics
        print(f"\nOOS PERFORMANCE METRICS (stable strategies only):")
        print(f"  Net Expectancy: {oos_net_expectancy:.4f}R")
        print(f"  Win Rate: {oos_win_rate:.2%}")
        print(f"  Total Trades: {oos_total_trades}")
        print(f"  Max Margin Used: ${test_metrics['max_margin_used']:,.2f}")
        print(f"  Max Margin Ratio: {test_metrics['max_margin_ratio']:.2%}")
        print(f"  Max Concurrent Positions: {test_metrics['max_concurrent_positions']}")
        
        # Print OOS advanced metrics
        print(f"\nOOS ADVANCED METRICS:")
        print(f"  Std Dev of R per Trade: {oos_advanced_metrics['std_r_per_trade']:.4f}")
        print(f"  Max Drawdown (R): {oos_advanced_metrics['max_drawdown_r']:.4f}")
        print(f"  Avg Trades/Day: {oos_advanced_metrics['avg_trades_per_day']:.2f}")
        print(f"  Avg R/Day: {oos_advanced_metrics['avg_r_per_day']:.4f}")
        print(f"  Avg R/Week: {oos_advanced_metrics['avg_r_per_week']:.4f}")
        
        # Print correlation analysis
        print_correlation_analysis(oos_correlation_results, "OOS Correlation")
        
        # Store correlation and advanced metrics data for the last window
        if i == len(windows) - 1:
            last_window_correlation_data = {
                'is_correlation': is_correlation_results,
                'oos_correlation': oos_correlation_results,
                'stable_strategies': stable_strategies
            }
            last_window_advanced_metrics = {
                'is_advanced': is_advanced_metrics,
                'oos_advanced': oos_advanced_metrics
            }
        
        # Calculate performance decay
        is_metrics_dict = {
            'net_expectancy': is_net_expectancy,
            'win_rate': is_win_rate
        }
        oos_metrics_dict = {
            'net_expectancy': oos_net_expectancy,
            'win_rate': oos_win_rate
        }
        
        decay_analysis = analyze_performance_decay(is_metrics_dict, oos_metrics_dict)
        
        print(f"\nPERFORMANCE DECAY ANALYSIS:")
        print(f"  Expectancy Decay: {decay_analysis['expectancy_decay_pct']:.2%} (Max: {WFA_CONFIG['MAX_EXPECTANCY_DECAY']:.0%}) - {'PASS' if decay_analysis['passes_expectancy_decay'] else 'FAIL'}")
        print(f"  Win Rate Decay: {decay_analysis['win_rate_decay_pct']:.2%} (Max: {WFA_CONFIG['MAX_WINRATE_DECAY']:.0%}) - {'PASS' if decay_analysis['passes_winrate_decay'] else 'FAIL'}")
        print(f"  Min OOS Expectancy: {oos_net_expectancy:.4f}R (Min: {WFA_CONFIG['MIN_OOS_EXPECTANCY']}R) - {'PASS' if decay_analysis['passes_min_expectancy'] else 'FAIL'}")
        print(f"  OVERALL: {'PASS' if decay_analysis['passes_all'] else 'FAIL'}")
        
        # Store results including new metrics
        result = {
            'window_id': i + 1,
            'train_start': window['train_start'],
            'train_end': window['train_end'],
            'test_start': window['test_start'],
            'test_end': window['test_end'],
            'stable_strategies': stable_strategies,
            
            # IS metrics
            'is_net_expectancy': is_net_expectancy,
            'is_win_rate': is_win_rate,
            'is_total_trades': is_total_trades,
            'is_max_margin_used': train_metrics['max_margin_used'],
            'is_max_margin_ratio': train_metrics['max_margin_ratio'],
            'is_max_concurrent_positions': train_metrics['max_concurrent_positions'],
            'is_final_capital': train_metrics['final_capital'],
            
            # IS advanced metrics
            'is_std_r_per_trade': is_advanced_metrics['std_r_per_trade'],
            'is_max_drawdown_r': is_advanced_metrics['max_drawdown_r'],
            'is_avg_trades_per_day': is_advanced_metrics['avg_trades_per_day'],
            'is_avg_r_per_day': is_advanced_metrics['avg_r_per_day'],
            'is_avg_r_per_week': is_advanced_metrics['avg_r_per_week'],
            'is_sharpe_ratio_r': is_advanced_metrics['sharpe_ratio_r'],
            'is_calmar_ratio_r': is_advanced_metrics['calmar_ratio_r'],
            
            # IS correlation metrics
            'is_avg_abs_correlation': is_correlation_results['avg_abs_correlation'],
            'is_max_abs_correlation': is_correlation_results['max_abs_correlation'],
            'is_diversification_score': is_correlation_results['diversification_score'],
            'is_highly_correlated_pairs': len(is_correlation_results['highly_correlated_pairs']),
            
            # OOS metrics
            'oos_net_expectancy': oos_net_expectancy,
            'oos_win_rate': oos_win_rate,
            'oos_total_trades': oos_total_trades,
            'oos_max_margin_used': test_metrics['max_margin_used'],
            'oos_max_margin_ratio': test_metrics['max_margin_ratio'],
            'oos_max_concurrent_positions': test_metrics['max_concurrent_positions'],
            'oos_final_capital': test_metrics['final_capital'],
            
            # OOS advanced metrics
            'oos_std_r_per_trade': oos_advanced_metrics['std_r_per_trade'],
            'oos_max_drawdown_r': oos_advanced_metrics['max_drawdown_r'],
            'oos_avg_trades_per_day': oos_advanced_metrics['avg_trades_per_day'],
            'oos_avg_r_per_day': oos_advanced_metrics['avg_r_per_day'],
            'oos_avg_r_per_week': oos_advanced_metrics['avg_r_per_week'],
            'oos_sharpe_ratio_r': oos_advanced_metrics['sharpe_ratio_r'],
            'oos_calmar_ratio_r': oos_advanced_metrics['calmar_ratio_r'],
            
            # OOS correlation metrics
            'oos_avg_abs_correlation': oos_correlation_results['avg_abs_correlation'],
            'oos_max_abs_correlation': oos_correlation_results['max_abs_correlation'],
            'oos_diversification_score': oos_correlation_results['diversification_score'],
            'oos_highly_correlated_pairs': len(oos_correlation_results['highly_correlated_pairs']),
            
            **decay_analysis
        }
        
        results.append(result)
    
    # Create comprehensive results DataFrame
    if results:
        results_df = pd.DataFrame(results)
        
        # Calculate overall statistics including correlation and advanced metrics
        print("\n" + "=" * 100)
        print("WALK-FORWARD ANALYSIS SUMMARY WITH CORRELATION & ADVANCED METRICS")
        print("=" * 100)
        
        total_windows = len(results_df)
        passing_windows = results_df['passes_all'].sum()
        pass_rate = passing_windows / total_windows
        
        # Performance metrics
        avg_oos_expectancy = results_df['oos_net_expectancy'].mean()
        std_oos_expectancy = results_df['oos_net_expectancy'].std()
        avg_expectancy_decay = results_df['expectancy_decay_pct'].mean()
        
        # Advanced metrics
        avg_oos_std_r = results_df['oos_std_r_per_trade'].mean()
        avg_oos_max_dd_r = results_df['oos_max_drawdown_r'].mean()
        avg_oos_trades_per_day = results_df['oos_avg_trades_per_day'].mean()
        avg_oos_r_per_day = results_df['oos_avg_r_per_day'].mean()
        avg_oos_r_per_week = results_df['oos_avg_r_per_week'].mean()
        
        # Correlation metrics
        avg_is_correlation = results_df['is_avg_abs_correlation'].mean()
        avg_oos_correlation = results_df['oos_avg_abs_correlation'].mean()
        avg_is_diversification = results_df['is_diversification_score'].mean()
        avg_oos_diversification = results_df['oos_diversification_score'].mean()
        
        print(f"Total Windows Analyzed: {total_windows}")
        print(f"Windows Passing All Criteria: {passing_windows} ({pass_rate:.1%})")
        print(f"Average OOS Net Expectancy: {avg_oos_expectancy:.4f}R")
        print(f"OOS Expectancy Std Dev: {std_oos_expectancy:.4f}R")
        print(f"Average Expectancy Decay: {avg_expectancy_decay:.1%}")
        
        print(f"\nADVANCED METRICS SUMMARY (OOS):")
        print(f"  Std Dev of R per Trade: {avg_oos_std_r:.4f}")
        print(f"  Max Drawdown (R): {avg_oos_max_dd_r:.4f}")
        print(f"  Avg Trades/Day: {avg_oos_trades_per_day:.2f}")
        print(f"  Avg R/Day: {avg_oos_r_per_day:.4f}")
        print(f"  Avg R/Week: {avg_oos_r_per_week:.4f}")
        
        print(f"\nCORRELATION ANALYSIS SUMMARY:")
        print(f"  IS Average Absolute Correlation: {avg_is_correlation:.3f}")
        print(f"  OOS Average Absolute Correlation: {avg_oos_correlation:.3f}")
        print(f"  IS Average Diversification Score: {avg_is_diversification:.3f}")
        print(f"  OOS Average Diversification Score: {avg_oos_diversification:.3f}")
        
        # Diversification assessment
        if avg_oos_diversification > 0.7:
            print(f"  ✅ Excellent diversification across windows")
        elif avg_oos_diversification > 0.5:
            print(f"  ⚠️  Moderate diversification - benefits may be limited")
        else:
            print(f"  ❌ Poor diversification - strategies are highly correlated")
        
        # Most frequent stable strategies
        if all_stable_strategies:
            strategy_counts = pd.Series(all_stable_strategies).value_counts()
            print(f"\nMost Frequently Selected STABLE Strategies:")
            for strategy, count in strategy_counts.items():
                freq = count / len(windows)
                print(f"  {strategy} - {count} windows ({freq:.1%})")
        
        # Plot correlation matrices for the last window
        if last_window_correlation_data:
            print(f"\nPlotting correlation matrices for last window...")
            plot_correlation_matrices(last_window_correlation_data['is_correlation'], "Last Window IS")
            plot_correlation_matrices(last_window_correlation_data['oos_correlation'], "Last Window OOS")
        
        # Plot advanced metrics for the last window
        if last_window_advanced_metrics:
            print(f"\nPlotting advanced metrics for last window...")
            plot_advanced_metrics(last_window_advanced_metrics['is_advanced'], "Last Window IS")
            plot_advanced_metrics(last_window_advanced_metrics['oos_advanced'], "Last Window OOS")
        
        return results_df
    else:
        print("No valid results from walk-forward analysis")
        return pd.DataFrame()

def plot_walk_forward_results(results_df: pd.DataFrame):
    """Create visualization of walk-forward results including advanced metrics"""
    if results_df.empty:
        print("No results to plot")
        return
    
    fig, axes = plt.subplots(5, 2, figsize=(15, 30))
    fig.suptitle('Strategy Walk-Forward Analysis Results', fontsize=16, fontweight='bold')
    
    # Plot 1: IS vs OOS Net Expectancy by Window
    axes[0, 0].plot(results_df['window_id'], results_df['is_net_expectancy'], 
                   marker='o', linewidth=2, label='IS Net Expectancy', color='blue')
    axes[0, 0].plot(results_df['window_id'], results_df['oos_net_expectancy'], 
                   marker='s', linewidth=2, label='OOS Net Expectancy', color='red')
    axes[0, 0].axhline(y=WFA_CONFIG['MIN_OOS_EXPECTANCY'], color='green', linestyle='--', 
                      label=f'Min OOS Threshold ({WFA_CONFIG["MIN_OOS_EXPECTANCY"]}R)')
    axes[0, 0].set_xlabel('Window ID')
    axes[0, 0].set_ylabel('Net Expectancy (R)')
    axes[0, 0].set_title('IS vs OOS Net Expectancy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Performance Decay
    axes[0, 1].bar(results_df['window_id'], results_df['expectancy_decay_pct'] * 100,
                  color=['red' if decay > WFA_CONFIG['MAX_EXPECTANCY_DECAY'] * 100 else 'green' 
                        for decay in results_df['expectancy_decay_pct'] * 100])
    axes[0, 1].axhline(y=WFA_CONFIG['MAX_EXPECTANCY_DECAY'] * 100, color='red', linestyle='--',
                      label=f'Max Decay Threshold ({WFA_CONFIG["MAX_EXPECTANCY_DECAY"]:.0%})')
    axes[0, 1].set_xlabel('Window ID')
    axes[0, 1].set_ylabel('Expectancy Decay (%)')
    axes[0, 1].set_title('Performance Decay (IS to OOS)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Standard Deviation of R per Trade
    axes[1, 0].plot(results_df['window_id'], results_df['is_std_r_per_trade'], 
                   marker='o', linewidth=2, label='IS Std Dev', color='blue')
    axes[1, 0].plot(results_df['window_id'], results_df['oos_std_r_per_trade'], 
                   marker='s', linewidth=2, label='OOS Std Dev', color='red')
    axes[1, 0].set_xlabel('Window ID')
    axes[1, 0].set_ylabel('Std Dev of R per Trade')
    axes[1, 0].set_title('Volatility of R per Trade')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Maximum Drawdown (R)
    axes[1, 1].bar(results_df['window_id'], results_df['oos_max_drawdown_r'],
                  color=['red' if dd > 10 else ('orange' if dd > 5 else 'green') 
                        for dd in results_df['oos_max_drawdown_r']])
    axes[1, 1].axhline(y=5, color='orange', linestyle='--',
                      label='Moderate Drawdown (5R)')
    axes[1, 1].axhline(y=10, color='red', linestyle='--',
                      label='High Drawdown (10R)')
    axes[1, 1].set_xlabel('Window ID')
    axes[1, 1].set_ylabel('Max Drawdown (R)')
    axes[1, 1].set_title('Maximum Historical Drawdown (OOS)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 5: Average Trades Per Day
    axes[2, 0].bar(results_df['window_id'] - 0.2, results_df['is_avg_trades_per_day'], 
                  width=0.4, label='IS Avg Trades/Day', color='blue', alpha=0.7)
    axes[2, 0].bar(results_df['window_id'] + 0.2, results_df['oos_avg_trades_per_day'], 
                  width=0.4, label='OOS Avg Trades/Day', color='red', alpha=0.7)
    axes[2, 0].set_xlabel('Window ID')
    axes[2, 0].set_ylabel('Average Trades Per Day')
    axes[2, 0].set_title('Trading Frequency')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 6: Average R per Day
    axes[2, 1].plot(results_df['window_id'], results_df['is_avg_r_per_day'], 
                   marker='o', linewidth=2, label='IS Avg R/Day', color='blue')
    axes[2, 1].plot(results_df['window_id'], results_df['oos_avg_r_per_day'], 
                   marker='s', linewidth=2, label='OOS Avg R/Day', color='red')
    axes[2, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[2, 1].set_xlabel('Window ID')
    axes[2, 1].set_ylabel('Average R per Day')
    axes[2, 1].set_title('Daily R Performance')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # Plot 7: Average R per Week
    axes[3, 0].bar(results_df['window_id'], results_df['oos_avg_r_per_week'],
                  color=['green' if r > 0 else 'red' for r in results_df['oos_avg_r_per_week']])
    axes[3, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[3, 0].set_xlabel('Window ID')
    axes[3, 0].set_ylabel('Average R per Week')
    axes[3, 0].set_title('Weekly R Performance (OOS)')
    axes[3, 0].grid(True, alpha=0.3)
    
    # Plot 8: Correlation Comparison
    axes[3, 1].plot(results_df['window_id'], results_df['is_avg_abs_correlation'], 
                   marker='o', linewidth=2, label='IS Avg Correlation', color='blue')
    axes[3, 1].plot(results_df['window_id'], results_df['oos_avg_abs_correlation'], 
                   marker='s', linewidth=2, label='OOS Avg Correlation', color='red')
    axes[3, 1].axhline(y=WFA_CONFIG['CORRELATION_THRESHOLD'], color='orange', linestyle='--',
                      label=f'High Correlation Threshold ({WFA_CONFIG["CORRELATION_THRESHOLD"]})')
    axes[3, 1].set_xlabel('Window ID')
    axes[3, 1].set_ylabel('Average Absolute Correlation')
    axes[3, 1].set_title('Strategy Correlation Over Time')
    axes[3, 1].legend()
    axes[3, 1].grid(True, alpha=0.3)
    
    # Plot 9: Diversification Score
    axes[4, 0].plot(results_df['window_id'], results_df['is_diversification_score'], 
                   marker='o', linewidth=2, label='IS Diversification', color='blue')
    axes[4, 0].plot(results_df['window_id'], results_df['oos_diversification_score'], 
                   marker='s', linewidth=2, label='OOS Diversification', color='red')
    axes[4, 0].axhline(y=0.7, color='green', linestyle='--',
                      label='Good Diversification (0.7)')
    axes[4, 0].axhline(y=0.5, color='orange', linestyle='--',
                      label='Moderate Diversification (0.5)')
    axes[4, 0].set_xlabel('Window ID')
    axes[4, 0].set_ylabel('Diversification Score')
    axes[4, 0].set_title('Diversification Score Over Time')
    axes[4, 0].legend()
    axes[4, 0].grid(True, alpha=0.3)
    
    # Plot 10: Sharpe Ratio (R)
    axes[4, 1].bar(results_df['window_id'], results_df['oos_sharpe_ratio_r'],
                  color=['green' if sharpe > 1 else ('orange' if sharpe > 0.5 else 'red') 
                        for sharpe in results_df['oos_sharpe_ratio_r']])
    axes[4, 1].axhline(y=1, color='green', linestyle='--',
                      label='Good Sharpe (1.0)')
    axes[4, 1].axhline(y=0.5, color='orange', linestyle='--',
                      label='Moderate Sharpe (0.5)')
    axes[4, 1].set_xlabel('Window ID')
    axes[4, 1].set_ylabel('Sharpe Ratio (R, annualized)')
    axes[4, 1].set_title('Risk-Adjusted Returns (OOS)')
    axes[4, 1].legend()
    axes[4, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_detailed_window_analysis(results_df: pd.DataFrame):
    """Print detailed analysis for each window including advanced metrics"""
    print("\n" + "=" * 100)
    print("DETAILED WINDOW ANALYSIS WITH ADVANCED METRICS")
    print("=" * 100)
    
    for _, row in results_df.iterrows():
        print(f"\n{'='*80}")
        print(f"Window {row['window_id']}: {row['train_start'].strftime('%Y-%m-%d')} to {row['test_end'].strftime('%Y-%m-%d')}")
        print(f"{'='*80}")
        
        print(f"\nStable Strategies: {sorted(row['stable_strategies'])}")
        
        print(f"\nIN-SAMPLE METRICS:")
        print(f"  Performance:  {row['is_net_expectancy']:.4f}R, {row['is_win_rate']:.1%} WR, {row['is_total_trades']} trades")
        print(f"  Advanced:     Std Dev: {row['is_std_r_per_trade']:.4f}, Max DD: {row['is_max_drawdown_r']:.4f}R")
        print(f"  Frequency:    {row['is_avg_trades_per_day']:.2f} trades/day, {row['is_avg_r_per_day']:.4f}R/day, {row['is_avg_r_per_week']:.4f}R/week")
        print(f"  Risk-Adjusted: Sharpe: {row['is_sharpe_ratio_r']:.4f}, Calmar: {row['is_calmar_ratio_r']:.4f}")
        print(f"  Correlation:  Avg: {row['is_avg_abs_correlation']:.3f}, Max: {row['is_max_abs_correlation']:.3f}")
        print(f"  Diversification Score: {row['is_diversification_score']:.3f}")
        print(f"  Highly Correlated Pairs: {row['is_highly_correlated_pairs']}")
        
        print(f"\nOUT-OF-SAMPLE METRICS:")
        print(f"  Performance:  {row['oos_net_expectancy']:.4f}R, {row['oos_win_rate']:.1%} WR, {row['oos_total_trades']} trades")
        print(f"  Advanced:     Std Dev: {row['oos_std_r_per_trade']:.4f}, Max DD: {row['oos_max_drawdown_r']:.4f}R")
        print(f"  Frequency:    {row['oos_avg_trades_per_day']:.2f} trades/day, {row['oos_avg_r_per_day']:.4f}R/day, {row['oos_avg_r_per_week']:.4f}R/week")
        print(f"  Risk-Adjusted: Sharpe: {row['oos_sharpe_ratio_r']:.4f}, Calmar: {row['oos_calmar_ratio_r']:.4f}")
        print(f"  Correlation:  Avg: {row['oos_avg_abs_correlation']:.3f}, Max: {row['oos_max_abs_correlation']:.3f}")
        print(f"  Diversification Score: {row['oos_diversification_score']:.3f}")
        print(f"  Highly Correlated Pairs: {row['oos_highly_correlated_pairs']}")
        
        print(f"\nPERFORMANCE DECAY:")
        print(f"  Expectancy Decay: {row['expectancy_decay_pct']:.1%}")
        print(f"  Win Rate Decay: {row['win_rate_decay_pct']:.1%}")
        print(f"  Status: {'PASS' if row['passes_all'] else 'FAIL'} All Criteria")
        
        # Highlight specific failures
        failures = []
        if not row['passes_expectancy_decay']:
            failures.append("Expectancy Decay")
        if not row['passes_winrate_decay']:
            failures.append("Win Rate Decay") 
        if not row['passes_min_expectancy']:
            failures.append("Min Expectancy")
            
        if failures:
            print(f"  Failed Criteria: {', '.join(failures)}")
        
        # Advanced metrics interpretation
        print(f"\nADVANCED METRICS INTERPRETATION:")
        
        # Volatility interpretation
        if row['oos_std_r_per_trade'] < 0.5:
            print(f"  ✅ Low volatility (Std Dev: {row['oos_std_r_per_trade']:.4f})")
        elif row['oos_std_r_per_trade'] < 1.0:
            print(f"  ⚠️  Moderate volatility (Std Dev: {row['oos_std_r_per_trade']:.4f})")
        else:
            print(f"  ❌ High volatility (Std Dev: {row['oos_std_r_per_trade']:.4f})")
        
        # Drawdown interpretation
        if row['oos_max_drawdown_r'] < 5:
            print(f"  ✅ Low max drawdown ({row['oos_max_drawdown_r']:.2f}R)")
        elif row['oos_max_drawdown_r'] < 15:
            print(f"  ⚠️  Moderate max drawdown ({row['oos_max_drawdown_r']:.2f}R)")
        else:
            print(f"  ❌ High max drawdown ({row['oos_max_drawdown_r']:.2f}R)")
        
        # Daily performance interpretation
        if row['oos_avg_r_per_day'] > 0.1:
            print(f"  ✅ Good daily R performance ({row['oos_avg_r_per_day']:.4f}R/day)")
        elif row['oos_avg_r_per_day'] > 0:
            print(f"  ⚠️  Marginal daily R performance ({row['oos_avg_r_per_day']:.4f}R/day)")
        else:
            print(f"  ❌ Negative daily R performance ({row['oos_avg_r_per_day']:.4f}R/day)")
        
        # Weekly performance interpretation
        if row['oos_avg_r_per_week'] > 0.5:
            print(f"  ✅ Good weekly R performance ({row['oos_avg_r_per_week']:.4f}R/week)")
        elif row['oos_avg_r_per_week'] > 0:
            print(f"  ⚠️  Marginal weekly R performance ({row['oos_avg_r_per_week']:.4f}R/week)")
        else:
            print(f"  ❌ Negative weekly R performance ({row['oos_avg_r_per_week']:.4f}R/week)")
        
        # Sharpe ratio interpretation
        if row['oos_sharpe_ratio_r'] > 1.0:
            print(f"  ✅ Excellent risk-adjusted returns (Sharpe: {row['oos_sharpe_ratio_r']:.4f})")
        elif row['oos_sharpe_ratio_r'] > 0.5:
            print(f"  ⚠️  Moderate risk-adjusted returns (Sharpe: {row['oos_sharpe_ratio_r']:.4f})")
        else:
            print(f"  ❌ Poor risk-adjusted returns (Sharpe: {row['oos_sharpe_ratio_r']:.4f})")
        
        # Correlation warning
        if row['oos_highly_correlated_pairs'] > 0:
            print(f"  ⚠️  WARNING: {row['oos_highly_correlated_pairs']} highly correlated strategy pairs detected")

def get_final_strategy_recommendation(results_df: pd.DataFrame) -> List[str]:
    """Determine final strategy recommendation based on walk-forward results"""
    if results_df.empty:
        print("No results available - using all strategies")
        return ['strategy_1', 'strategy_2', 'strategy_3', 'strategy_4', 'strategy_5']
    
    # Only consider windows that passed all criteria
    passing_windows = results_df[results_df['passes_all']]
    
    if passing_windows.empty:
        print("No windows passed all criteria - using all strategies")
        return ['strategy_1', 'strategy_2', 'strategy_3', 'strategy_4', 'strategy_5']
    
    # Count frequency of strategies in passing windows
    all_strategies = []
    for strategies in passing_windows['stable_strategies']:
        all_strategies.extend(strategies)
    
    strategy_counts = pd.Series(all_strategies).value_counts()
    
    # Select all strategies that appear in at least 50% of passing windows
    final_strategies = strategy_counts[strategy_counts >= len(passing_windows) * 0.5].index.tolist()
    
    # If no strategies meet the threshold, use the top 3
    if not final_strategies:
        final_strategies = strategy_counts.head(3).index.tolist()
    
    print(f"\nFINAL STRATEGY RECOMMENDATION:")
    print(f"Based on {len(passing_windows)} passing windows")
    for strategy in final_strategies:
        count = strategy_counts[strategy]
        freq = count / len(passing_windows)
        print(f"  {strategy} - {int(count)} windows ({freq:.1%})")
    
    return final_strategies

# =============================================================================
# PERFORMANCE METRICS WITH ADVANCED METRICS
# =============================================================================

def calculate_performance_metrics(trades_df: pd.DataFrame, metrics: Dict[str, Any]):
    """Calculate comprehensive performance metrics including advanced metrics"""
    if trades_df.empty:
        print("No trades to analyze")
        return
    
    print("\n" + "=" * 80)
    print("COMBINED STRATEGY PERFORMANCE SUMMARY WITH ADVANCED METRICS")
    print("=" * 80)
    
    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df['win']]
    losing_trades = trades_df[~trades_df['win']]
    
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    total_pnl = trades_df['dollar_pnl'].sum()
    avg_trade_pnl = trades_df['dollar_pnl'].mean()
    avg_winning_trade = winning_trades['dollar_pnl'].mean() if len(winning_trades) > 0 else 0
    avg_losing_trade = losing_trades['dollar_pnl'].mean() if len(losing_trades) > 0 else 0
    
    # Calculate net expectancy in R
    net_expectancy = calculate_net_expectancy(trades_df)
    
    # Risk-adjusted metrics
    profit_factor = abs(winning_trades['dollar_pnl'].sum() / losing_trades['dollar_pnl'].sum()) if losing_trades['dollar_pnl'].sum() != 0 else float('inf')
    
    # Maximum drawdown in dollars
    equity_curve = trades_df['current_capital'].values
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    max_drawdown = np.max(drawdown)
    
    # Calculate advanced metrics
    advanced_metrics = calculate_advanced_metrics(trades_df)
    
    print(f"\nBASIC PERFORMANCE METRICS:")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Net Expectancy: {net_expectancy:.4f}R")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print(f"Average Trade P&L: ${avg_trade_pnl:,.2f}")
    print(f"Average Winning Trade: ${avg_winning_trade:,.2f}")
    print(f"Average Losing Trade: ${avg_losing_trade:,.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Maximum Drawdown (Capital): {max_drawdown:.2%}")
    print(f"Final Capital: ${trades_df['current_capital'].iloc[-1]:,.2f}")
    
    print(f"\nADVANCED RISK METRICS (in R multiples):")
    print(f"Standard Deviation of R per Trade: {advanced_metrics['std_r_per_trade']:.4f}")
    print(f"Maximum Historical Drawdown (R): {advanced_metrics['max_drawdown_r']:.4f}")
    print(f"Average Trades Per Day: {advanced_metrics['avg_trades_per_day']:.2f}")
    print(f"Average R per Day: {advanced_metrics['avg_r_per_day']:.4f}")
    print(f"Average R per Week: {advanced_metrics['avg_r_per_week']:.4f}")
    print(f"Sharpe Ratio (R, annualized): {advanced_metrics['sharpe_ratio_r']:.4f}")
    print(f"Calmar Ratio (R): {advanced_metrics['calmar_ratio_r']:.4f}")
    
    print(f"\nPOSITION & MARGIN METRICS:")
    print(f"Max Margin Used: ${metrics['max_margin_used']:,.2f}")
    print(f"Max Margin Ratio: {metrics['max_margin_ratio']:.2%}")
    print(f"Max Concurrent Positions: {metrics['max_concurrent_positions']}")
    print(f"Margin Buffer: {STRATEGY_CONFIG['MARGIN_BUFFER_PCT']:.0%}")
    
    # Strategy breakdown
    strategy_counts = trades_df['pattern'].value_counts()
    
    print(f"\nStrategy Breakdown:")
    for pattern, count in strategy_counts.items():
        pattern_trades = trades_df[trades_df['pattern'] == pattern]
        pattern_pnl = pattern_trades['dollar_pnl'].sum()
        pattern_win_rate = pattern_trades['win'].mean()
        pattern_expectancy = calculate_net_expectancy(pattern_trades)
        print(f"  {pattern}: {count} trades, P&L: ${pattern_pnl:,.2f}, Win Rate: {pattern_win_rate:.2%}, Expectancy: {pattern_expectancy:.4f}R")
    
    # Hourly performance
    hourly_performance = trades_df.groupby('hour_of_pattern').agg({
        'dollar_pnl': ['count', 'sum', 'mean'],
        'win': 'mean',
        'pnl': 'mean'  # R multiple
    }).round(2)
    
    if not hourly_performance.empty:
        print(f"\nBest Performing Hours (by R expectancy):")
        for hour in hourly_performance.nlargest(3, ('pnl', 'mean')).index:
            hour_data = hourly_performance.loc[hour]
            print(f"  Hour {hour}: {hour_data[('dollar_pnl', 'count')]} trades, P&L: ${hour_data[('dollar_pnl', 'sum')]:,.2f}, R Expectancy: {hour_data[('pnl', 'mean')]:.4f}")
    
    # Print advanced metrics interpretations
    print_advanced_metrics(advanced_metrics, "Full Backtest")
    
    # Plot advanced metrics
    plot_advanced_metrics(advanced_metrics, "Full Backtest Analysis")
    
    return advanced_metrics

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("COMBINED TRADING STRATEGY WITH WALK-FORWARD, CORRELATION & ADVANCED METRICS ANALYSIS")
    print("=" * 100)
    
    # Display strategy configuration
    print("Strategy Configuration:")
    for strategy in ['STRATEGY_1', 'STRATEGY_2', 'STRATEGY_3', 'STRATEGY_4', 'STRATEGY_5']:
        enabled = STRATEGY_CONFIG[f'{strategy}_ENABLED']
        status = "ENABLED" if enabled else "DISABLED"
        hours = STRATEGY_CONFIG[strategy]['ALLOWED_HOURS']
        print(f"  {strategy}: {status} (Hours: {hours})")
    
    print(f"\nRisk Management:")
    print(f"  Margin Buffer: {STRATEGY_CONFIG['MARGIN_BUFFER_PCT']:.0%}")
    print(f"  Leverage: {STRATEGY_CONFIG['LEVERAGE']}:1")
    print(f"  Risk per Trade: {STRATEGY_CONFIG['RISK_PCT']:.1%}")
    
    print(f"\nAdvanced Metrics Tracking:")
    print(f"  Standard Deviation of R per Trade: Enabled")
    print(f"  Maximum Historical Drawdown (R): Enabled")
    print(f"  Average Trades Per Day: Enabled")
    print(f"  Average R per Day/Week: Enabled")
    
    print(f"\nCorrelation Analysis:")
    print(f"  Correlation Threshold: {WFA_CONFIG['CORRELATION_THRESHOLD']}")
    print(f"  Min Samples for Correlation: {WFA_CONFIG['MIN_CORRELATION_SAMPLES']}")
    
    # Load data
    df = load_data(DATA_PATH)
    if df.empty:
        print("Failed to load data. Exiting.")
        return
    
    # Check data range for walk-forward analysis
    data_years = (df.index.max() - df.index.min()).days / 365.25
    print(f"\nData covers {data_years:.1f} years")
    
    if data_years < (WFA_CONFIG['TRAINING_YEARS'] + WFA_CONFIG['TESTING_YEARS']):
        print(f"Warning: Need at least {WFA_CONFIG['TRAINING_YEARS'] + WFA_CONFIG['TESTING_YEARS']} years of data for walk-forward analysis")
        print("Proceeding with regular backtest instead...")
        
        # Execute combined strategy without WFA
        trades_df, metrics = execute_combined_strategy(df)
        
        if not trades_df.empty:
            # Calculate performance metrics with advanced metrics
            advanced_metrics = calculate_performance_metrics(trades_df, metrics)
            
            # Calculate correlation for the entire period
            all_strategies = ['strategy_1', 'strategy_2', 'strategy_3', 'strategy_4', 'strategy_5']
            enabled_strategies = [s for s in all_strategies if STRATEGY_CONFIG.get(f'{s.upper()}_ENABLED', False)]
            correlation_results = calculate_strategy_correlation(trades_df, enabled_strategies)
            
            print_correlation_analysis(correlation_results, "Full Period Correlation")
            
            # Plot correlation matrices
            plot_correlation_matrices(correlation_results, "Full Period")
            
            # Save results to CSV
            output_file = "combined_strategy_results.csv"
            trades_df.to_csv(output_file)
            print(f"Results saved to: {output_file}")
            
            # Save advanced metrics
            metrics_file = "advanced_metrics_summary.txt"
            with open(metrics_file, 'w') as f:
                f.write("ADVANCED METRICS SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                
                f.write("BASIC PERFORMANCE:\n")
                f.write(f"Total Trades: {len(trades_df)}\n")
                f.write(f"Win Rate: {trades_df['win'].mean():.2%}\n")
                f.write(f"Net Expectancy: {calculate_net_expectancy(trades_df):.4f}R\n")
                f.write(f"Total P&L: ${trades_df['dollar_pnl'].sum():,.2f}\n")
                f.write(f"Final Capital: ${trades_df['current_capital'].iloc[-1]:,.2f}\n\n")
                
                f.write("ADVANCED METRICS (R multiples):\n")
                f.write(f"Std Dev of R per Trade: {advanced_metrics['std_r_per_trade']:.4f}\n")
                f.write(f"Max Historical Drawdown (R): {advanced_metrics['max_drawdown_r']:.4f}\n")
                f.write(f"Avg Trades Per Day: {advanced_metrics['avg_trades_per_day']:.2f}\n")
                f.write(f"Avg R per Day: {advanced_metrics['avg_r_per_day']:.4f}\n")
                f.write(f"Avg R per Week: {advanced_metrics['avg_r_per_week']:.4f}\n")
                f.write(f"Sharpe Ratio (R): {advanced_metrics['sharpe_ratio_r']:.4f}\n")
                f.write(f"Calmar Ratio (R): {advanced_metrics['calmar_ratio_r']:.4f}\n\n")
                
                f.write("MARGIN METRICS:\n")
                f.write(f"Max Margin Used: ${metrics['max_margin_used']:,.2f}\n")
                f.write(f"Max Margin Ratio: {metrics['max_margin_ratio']:.2%}\n")
                f.write(f"Max Concurrent Positions: {metrics['max_concurrent_positions']}\n")
                
                if correlation_results['correlation_matrix'] is not None:
                    f.write("\nCORRELATION METRICS:\n")
                    f.write(f"Avg Absolute Correlation: {correlation_results['avg_abs_correlation']:.3f}\n")
                    f.write(f"Max Absolute Correlation: {correlation_results['max_abs_correlation']:.3f}\n")
                    f.write(f"Diversification Score: {correlation_results['diversification_score']:.3f}\n")
                    f.write(f"Highly Correlated Pairs: {len(correlation_results['highly_correlated_pairs'])}\n")
            
            print(f"Advanced metrics summary saved to: {metrics_file}")
        return
    
    # Run walk-forward analysis
    print("\n" + "=" * 100)
    print("STARTING WALK-FORWARD ANALYSIS WITH ADVANCED METRICS")
    print("=" * 100)
    
    results_df = run_strategy_walk_forward_analysis(df, execute_combined_strategy)
    
    if not results_df.empty:
        # Generate visualizations
        plot_walk_forward_results(results_df)
        
        # Print detailed analysis
        print_detailed_window_analysis(results_df)
        
        # Get final strategy recommendation
        final_strategies = get_final_strategy_recommendation(results_df)
        
        print("\n" + "=" * 100)
        print("FINAL TRADING RECOMMENDATION")
        print("=" * 100)
        print(f"Recommended Strategies: {sorted(final_strategies)}")
        print(f"Only trade these strategies for live trading")
        
        # Run final validation with recommended strategies
        print(f"\nFINAL VALIDATION WITH RECOMMENDED STRATEGIES:")
        final_trades, final_metrics = execute_combined_strategy(df, final_strategies)
        
        if not final_trades.empty:
            final_expectancy = calculate_net_expectancy(final_trades)
            final_win_rate = final_trades['win'].mean()
            final_total_trades = len(final_trades)
            
            # Calculate advanced metrics for final strategies
            final_advanced_metrics = calculate_advanced_metrics(final_trades)
            
            # Calculate correlation for final recommended strategies
            final_correlation_results = calculate_strategy_correlation(final_trades, final_strategies)
            
            print(f"\nOverall Performance with Recommended Strategies:")
            print(f"  Net Expectancy: {final_expectancy:.4f}R")
            print(f"  Win Rate: {final_win_rate:.2%}")
            print(f"  Total Trades: {final_total_trades}")
            print(f"  Max Margin Used: ${final_metrics['max_margin_used']:,.2f}")
            print(f"  Max Margin Ratio: {final_metrics['max_margin_ratio']:.2%}")
            print(f"  Max Concurrent Positions: {final_metrics['max_concurrent_positions']}")
            
            # Print advanced metrics for final strategies
            print(f"\nAdvanced Metrics for Final Strategies:")
            print(f"  Std Dev of R per Trade: {final_advanced_metrics['std_r_per_trade']:.4f}")
            print(f"  Max Drawdown (R): {final_advanced_metrics['max_drawdown_r']:.4f}")
            print(f"  Avg Trades/Day: {final_advanced_metrics['avg_trades_per_day']:.2f}")
            print(f"  Avg R/Day: {final_advanced_metrics['avg_r_per_day']:.4f}")
            print(f"  Avg R/Week: {final_advanced_metrics['avg_r_per_week']:.4f}")
            
            # Print correlation analysis for final strategies
            print_correlation_analysis(final_correlation_results, "Final Recommended Strategies")
            
            # Plot advanced metrics for final strategies
            plot_advanced_metrics(final_advanced_metrics, "Final Recommended Strategies")
            
            if final_expectancy >= WFA_CONFIG['MIN_OOS_EXPECTANCY']:
                print("✅ STRATEGY PORTFOLIO VALIDATED - READY FOR LIVE TRADING")
            else:
                print("❌ STRATEGY PORTFOLIO FAILED VALIDATION - DO NOT TRADE LIVE")
            
            # Diversification assessment for final strategies
            if final_correlation_results['diversification_score'] > 0.7:
                print("✅ EXCELLENT DIVERSIFICATION IN FINAL PORTFOLIO")
            elif final_correlation_results['diversification_score'] > 0.5:
                print("⚠️  MODERATE DIVERSIFICATION - Benefits may be limited")
            else:
                print("❌ POOR DIVERSIFICATION - Consider adding uncorrelated strategies")
            
            # Save results to CSV
            output_file = "walk_forward_analysis_results.csv"
            results_df.to_csv(output_file)
            print(f"Walk-forward results saved to: {output_file}")
            
            # Save final trades
            trades_file = "final_recommended_strategies_trades.csv"
            final_trades.to_csv(trades_file)
            print(f"Final recommended strategies trades saved to: {trades_file}")
            
            # Save comprehensive metrics summary
            summary_file = "comprehensive_metrics_summary.txt"
            with open(summary_file, 'w') as f:
                f.write("WALK-FORWARD ANALYSIS COMPREHENSIVE SUMMARY\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("OVERALL STATISTICS:\n")
                f.write(f"Total Windows Analyzed: {len(results_df)}\n")
                f.write(f"Windows Passing All Criteria: {results_df['passes_all'].sum()} ({results_df['passes_all'].sum()/len(results_df):.1%})\n")
                f.write(f"Average OOS Net Expectancy: {results_df['oos_net_expectancy'].mean():.4f}R\n")
                f.write(f"OOS Expectancy Std Dev: {results_df['oos_net_expectancy'].std():.4f}R\n")
                f.write(f"Average Expectancy Decay: {results_df['expectancy_decay_pct'].mean():.1%}\n\n")
                
                f.write("ADVANCED METRICS SUMMARY (OOS):\n")
                f.write(f"Std Dev of R per Trade: {results_df['oos_std_r_per_trade'].mean():.4f}\n")
                f.write(f"Max Drawdown (R): {results_df['oos_max_drawdown_r'].mean():.4f}\n")
                f.write(f"Avg Trades Per Day: {results_df['oos_avg_trades_per_day'].mean():.2f}\n")
                f.write(f"Avg R per Day: {results_df['oos_avg_r_per_day'].mean():.4f}\n")
                f.write(f"Avg R per Week: {results_df['oos_avg_r_per_week'].mean():.4f}\n")
                f.write(f"Sharpe Ratio (R): {results_df['oos_sharpe_ratio_r'].mean():.4f}\n")
                f.write(f"Calmar Ratio (R): {results_df['oos_calmar_ratio_r'].mean():.4f}\n\n")
                
                f.write("CORRELATION ANALYSIS SUMMARY:\n")
                f.write(f"IS Avg Absolute Correlation: {results_df['is_avg_abs_correlation'].mean():.3f}\n")
                f.write(f"OOS Avg Absolute Correlation: {results_df['oos_avg_abs_correlation'].mean():.3f}\n")
                f.write(f"IS Diversification Score: {results_df['is_diversification_score'].mean():.3f}\n")
                f.write(f"OOS Diversification Score: {results_df['oos_diversification_score'].mean():.3f}\n\n")
                
                f.write("FINAL RECOMMENDED STRATEGIES:\n")
                f.write(f"Strategies: {sorted(final_strategies)}\n")
                f.write(f"Net Expectancy: {final_expectancy:.4f}R\n")
                f.write(f"Win Rate: {final_win_rate:.2%}\n")
                f.write(f"Total Trades: {final_total_trades}\n")
                f.write(f"Max Margin Used: ${final_metrics['max_margin_used']:,.2f}\n")
                f.write(f"Max Margin Ratio: {final_metrics['max_margin_ratio']:.2%}\n")
                f.write(f"Max Concurrent Positions: {final_metrics['max_concurrent_positions']}\n")
                f.write(f"Std Dev of R per Trade: {final_advanced_metrics['std_r_per_trade']:.4f}\n")
                f.write(f"Max Drawdown (R): {final_advanced_metrics['max_drawdown_r']:.4f}\n")
                f.write(f"Avg Trades Per Day: {final_advanced_metrics['avg_trades_per_day']:.2f}\n")
                f.write(f"Avg R per Day: {final_advanced_metrics['avg_r_per_day']:.4f}\n")
                f.write(f"Avg R per Week: {final_advanced_metrics['avg_r_per_week']:.4f}\n")
                f.write(f"Sharpe Ratio (R): {final_advanced_metrics['sharpe_ratio_r']:.4f}\n")
                f.write(f"Calmar Ratio (R): {final_advanced_metrics['calmar_ratio_r']:.4f}\n")
                f.write(f"Diversification Score: {final_correlation_results['diversification_score']:.3f}\n")
                f.write(f"Avg Correlation: {final_correlation_results['avg_abs_correlation']:.3f}\n")
                f.write(f"Highly Correlated Pairs: {len(final_correlation_results['highly_correlated_pairs'])}\n")
            
            print(f"Comprehensive metrics summary saved to: {summary_file}")
        else:
            print("No trades with recommended strategies")
    else:
        print("Walk-forward analysis produced no results")

if __name__ == "__main__":
    main()
