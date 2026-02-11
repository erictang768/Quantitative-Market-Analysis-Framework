"""
Walk-forward analysis engine for strategy evaluation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Callable
from config.settings import WFA_CONFIG

def calculate_stable_strategies(trades_df: pd.DataFrame) -> List[str]:
    """
    Select top strategies based on stability criteria, not just performance.
    
    Args:
        trades_df: DataFrame of trades with 'pattern' and performance columns
        
    Returns:
        List of stable strategy names
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
            
        # Import here to avoid circular imports
        from walk_forward.windows import calculate_net_expectancy
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

def run_strategy_walk_forward_analysis(df: pd.DataFrame, 
                                     strategy_execution_func: Callable) -> pd.DataFrame:
    """
    Main walk-forward analysis engine for strategy evaluation with correlation analysis
    """
    print("=" * 100)
    print("STRATEGY WALK-FORWARD ANALYSIS WITH CORRELATION ANALYSIS & ADVANCED METRICS")
    print("=" * 100)
    
    # Import here to avoid circular imports
    from walk_forward.windows import create_walk_forward_windows
    from risk_metrics.performance import calculate_advanced_metrics
    from risk_metrics.correlation import calculate_strategy_correlation
    
    windows = create_walk_forward_windows(df)
    
    if not windows:
        print("No valid windows created - check your data range")
        return pd.DataFrame()
    
    results = []
    all_stable_strategies = []
    
    for i, window in enumerate(windows):
        print(f"\nProcessing Window {i+1}/{len(windows)}")
        print(f"Training: {window['train_start'].strftime('%Y-%m-%d')} to {window['train_end'].strftime('%Y-%m-%d')}")
        print(f"Testing:  {window['test_start'].strftime('%Y-%m-%d')} to {window['test_end'].strftime('%Y-%m-%d')}")
        
        # === IN-SAMPLE OPTIMIZATION ===
        print("\nIN-SAMPLE OPTIMIZATION")
        
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
        
        # Calculate IS advanced metrics
        is_advanced_metrics = calculate_advanced_metrics(is_stable_trades)
        
        # Calculate IS correlation analysis
        is_correlation_results = calculate_strategy_correlation(is_stable_trades, stable_strategies)
        
        # === OUT-OF-SAMPLE VALIDATION ===
        print("\nOUT-OF-SAMPLE VALIDATION")
        
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
        
        # Calculate OOS advanced metrics
        oos_advanced_metrics = calculate_advanced_metrics(oos_stable_trades)
        
        # Calculate OOS correlation analysis
        oos_correlation_results = calculate_strategy_correlation(oos_stable_trades, stable_strategies)
        
        # Store results
        result = {
            'window_id': i + 1,
            'train_start': window['train_start'],
            'train_end': window['train_end'],
            'test_start': window['test_start'],
            'test_end': window['test_end'],
            'stable_strategies': stable_strategies,
            
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
        }
        
        results.append(result)
    
    # Create comprehensive results DataFrame
    if results:
        results_df = pd.DataFrame(results)
        return results_df
    
    print("No valid results from walk-forward analysis")
    return pd.DataFrame()
