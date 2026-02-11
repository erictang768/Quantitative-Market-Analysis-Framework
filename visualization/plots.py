"""
Visualization functions for performance analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any

def plot_advanced_metrics(advanced_metrics: Dict[str, Any], period_name: str = "Performance"):
    """Create visualizations for advanced metrics"""
    
    if advanced_metrics['r_cumulative'] is None or advanced_metrics['r_cumulative'].empty:
        print("No data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
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
    
    # Plot 2: Daily R P&L
    if advanced_metrics['daily_r_curve'] is not None and not advanced_metrics['daily_r_curve'].empty:
        colors = ['green' if x >= 0 else 'red' for x in advanced_metrics['daily_r_curve'].values]
        axes[0, 1].bar(advanced_metrics['daily_r_curve'].index, advanced_metrics['daily_r_curve'].values, 
                      color=colors, alpha=0.7, edgecolor='black')
        axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0, 1].axhline(y=advanced_metrics['avg_r_per_day'], color='blue', linestyle='--',
                          label=f'Avg: {advanced_metrics["avg_r_per_day"]:.3f}R/day')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Daily R P&L')
        axes[0, 1].set_title(f'Daily R Performance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Drawdown Analysis
    if advanced_metrics['drawdown_curve_r'] is not None and not advanced_metrics['drawdown_curve_r'].empty:
        axes[1, 0].fill_between(advanced_metrics['drawdown_curve_r'].index, 
                               0, advanced_metrics['drawdown_curve_r'].values,
                               color='red', alpha=0.5, label='Drawdown (R)')
        axes[1, 0].axhline(y=advanced_metrics['max_drawdown_r'], color='darkred', linestyle='--',
                          label=f'Max DD: {advanced_metrics["max_drawdown_r"]:.2f}R')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Drawdown (R)')
        axes[1, 0].set_title('Historical Drawdown Analysis')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Risk Metrics Summary
    metrics_text = f"""
    Standard Deviation (R/Trade): {advanced_metrics['std_r_per_trade']:.3f}
    Maximum Drawdown (R): {advanced_metrics['max_drawdown_r']:.2f}
    Average Trades/Day: {advanced_metrics['avg_trades_per_day']:.2f}
    Average R/Day: {advanced_metrics['avg_r_per_day']:.3f}
    Average R/Week: {advanced_metrics['avg_r_per_week']:.3f}
    Sharpe Ratio (R): {advanced_metrics['sharpe_ratio_r']:.2f}
    Calmar Ratio (R): {advanced_metrics['calmar_ratio_r']:.2f}
    """
    
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=10, fontfamily='monospace',
                   verticalalignment='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Risk Metrics Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
