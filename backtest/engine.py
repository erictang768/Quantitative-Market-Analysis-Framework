"""
Backtesting engine for executing trading strategies with walk-forward validation.
Core execution logic for strategy testing and performance evaluation.
"""

import pandas as pd
import numpy as np
import math
from typing import Dict, Any, List, Tuple, Callable
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import configurations
from config.settings import STRATEGY_CONFIG, WFA_CONFIG

# Import utility functions from other modules
from strategies.base_strategy import BaseStrategy

# =============================================================================
# CORE UTILITY FUNCTIONS
# =============================================================================

def calculate_position_size(current_capital: float, risk_distance: float, 
                          commission_per_lot: float, risk_pct: float) -> Tuple[float, float, float]:
    """
    Calculate position size with commission.
    
    Args:
        current_capital: Current account capital
        risk_distance: Distance to stop loss in price units
        commission_per_lot: Commission per lot traded
        risk_pct: Risk percentage per trade
        
    Returns:
        Tuple of (position_size, actual_risk, commission_cost)
    """
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
    """
    Calculate take profit for net risk-reward after commission.
    
    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        side: 'bull' for long, 'bear' for short
        commission_per_lot: Commission per lot
        net_rr_ratio: Desired risk-reward ratio
        
    Returns:
        Take profit price
    """
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
    """
    Check margin requirements.
    
    Args:
        position_size: Size in lots
        entry_price: Entry price
        current_capital: Current account capital
        leverage: Account leverage
        margin_buffer_pct: Margin buffer percentage
        
    Returns:
        Tuple of (margin_sufficient, required_margin, max_allowed_lots)
    """
    contract_size = position_size * 100000
    required_margin = (contract_size * entry_price) / leverage
    max_allowed_margin = current_capital * margin_buffer_pct
    
    if required_margin <= max_allowed_margin:
        return True, required_margin, position_size
    else:
        max_allowed_lots = (max_allowed_margin * leverage) / (entry_price * 100000)
        return False, required_margin, max_allowed_lots

def validation_check(df: pd.DataFrame, setup: dict, trigger_idx: int) -> bool:
    """
    Validation check for strategies 4 and 5.
    
    Args:
        df: Market data
        setup: Trade setup dictionary
        trigger_idx: Trigger bar index
        
    Returns:
        Whether validation passes
    """
    if not setup.get('validation_required', False):
        return True
    
    if trigger_idx >= len(df):
        return False
    
    bar3 = df.iloc[trigger_idx]
    
    if setup['side'] == 'bear':
        return bar3['High'] >= setup['original_entry']
    else:
        return bar3['Low'] <= setup['original_entry']

# =============================================================================
# STRATEGY PATTERN FINDING FUNCTIONS
# =============================================================================

def find_patterns_strategy_1(df: pd.DataFrame, allowed_hours: List[int], net_rr_ratio: float) -> list:
    """
    Strategy 1: Original Pattern Completion.
    
    Args:
        df: Market data DataFrame
        allowed_hours: List of allowed trading hours
        net_rr_ratio: Net risk-reward ratio
        
    Returns:
        List of trade setups
    """
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

def find_patterns_strategy_2(df: pd.DataFrame, allowed_hours: List[int], net_rr_ratio: float) -> list:
    """
    Strategy 2: Double Offset Strategy.
    
    Args:
        df: Market data DataFrame
        allowed_hours: List of allowed trading hours
        net_rr_ratio: Net risk-reward ratio
        
    Returns:
        List of trade setups
    """
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

def find_patterns_strategy_3(df: pd.DataFrame, allowed_hours: List[int], net_rr_ratio: float) -> list:
    """
    Strategy 3: Triple Offset Strategy.
    
    Args:
        df: Market data DataFrame
        allowed_hours: List of allowed trading hours
        net_rr_ratio: Net risk-reward ratio
        
    Returns:
        List of trade setups
    """
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

def find_patterns_strategy_4(df: pd.DataFrame, allowed_hours: List[int], net_rr_ratio: float, validation_check_flag: bool) -> list:
    """
    Strategy 4: Entry at Old Stop Loss Level - FIXED VERSION.
    
    Args:
        df: Market data DataFrame
        allowed_hours: List of allowed trading hours
        net_rr_ratio: Net risk-reward ratio
        validation_check_flag: Whether to perform validation check
        
    Returns:
        List of trade setups
    """
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
                'validation_required': validation_check_flag,
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
                'validation_required': validation_check_flag,
            })
            
    return setups

def find_patterns_strategy_5(df: pd.DataFrame, allowed_hours: List[int], net_rr_ratio: float, validation_check_flag: bool) -> list:
    """
    Strategy 5: Reversal Trading Strategy.
    
    Args:
        df: Market data DataFrame
        allowed_hours: List of allowed trading hours
        net_rr_ratio: Net risk-reward ratio
        validation_check_flag: Whether to perform validation check
        
    Returns:
        List of trade setups
    """
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
                'validation_required': validation_check_flag,
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
                'validation_required': validation_check_flag,
            })
                
    return setups

# =============================================================================
# MAIN BACKTESTING ENGINE
# =============================================================================

def execute_combined_strategy(df: pd.DataFrame, enabled_strategies: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    EXECUTE COMBINED TRADING STRATEGY with margin and position tracking.
    
    Args:
        df: Market data DataFrame with OHLCV columns
        enabled_strategies: List of strategies to enable. If None, uses all from config.
        
    Returns:
        Tuple of (trades_df, metrics_dict)
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
        print(f"  Strategy 1: {len(setups_1)} setups")
    
    # Strategy 2
    if 'strategy_2' in enabled_strategies and STRATEGY_CONFIG['STRATEGY_2_ENABLED']:
        setups_2 = find_patterns_strategy_2(
            df,
            STRATEGY_CONFIG['STRATEGY_2']['ALLOWED_HOURS'],
            STRATEGY_CONFIG['STRATEGY_2']['NET_RR_RATIO']
        )
        all_setups.extend(setups_2)
        print(f"  Strategy 2: {len(setups_2)} setups")
    
    # Strategy 3
    if 'strategy_3' in enabled_strategies and STRATEGY_CONFIG['STRATEGY_3_ENABLED']:
        setups_3 = find_patterns_strategy_3(
            df,
            STRATEGY_CONFIG['STRATEGY_3']['ALLOWED_HOURS'],
            STRATEGY_CONFIG['STRATEGY_3']['NET_RR_RATIO']
        )
        all_setups.extend(setups_3)
        print(f"  Strategy 3: {len(setups_3)} setups")
    
    # Strategy 4
    if 'strategy_4' in enabled_strategies and STRATEGY_CONFIG['STRATEGY_4_ENABLED']:
        setups_4 = find_patterns_strategy_4(
            df,
            STRATEGY_CONFIG['STRATEGY_4']['ALLOWED_HOURS'],
            STRATEGY_CONFIG['STRATEGY_4']['NET_RR_RATIO'],
            STRATEGY_CONFIG['STRATEGY_4']['VALIDATION_CHECK']
        )
        all_setups.extend(setups_4)
        print(f"  Strategy 4: {len(setups_4)} setups")
    
    # Strategy 5
    if 'strategy_5' in enabled_strategies and STRATEGY_CONFIG['STRATEGY_5_ENABLED']:
        setups_5 = find_patterns_strategy_5(
            df,
            STRATEGY_CONFIG['STRATEGY_5']['ALLOWED_HOURS'],
            STRATEGY_CONFIG['STRATEGY_5']['NET_RR_RATIO'],
            STRATEGY_CONFIG['STRATEGY_5']['VALIDATION_CHECK']
        )
        all_setups.extend(setups_5)
        print(f"  Strategy 5: {len(setups_5)} setups")
    
    print(f"Total setups found: {len(all_setups)}")
    
    # Sort setups by trigger index for sequential processing
    all_setups.sort(key=lambda x: x['trigger_index'])
    
    # Initialize tracking variables
    trades = []
    current_capital = STRATEGY_CONFIG['INITIAL_CAPITAL']
    
    # Margin and position tracking
    max_margin_used = 0.0
    max_margin_ratio = 0.0
    max_concurrent_positions = 0
    current_open_positions = set()
    
    # Create a timeline of events for margin tracking
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
    
    # Calculate margin metrics from events timeline
    if events:
        # Sort events by time
        events.sort(key=lambda x: x['time'])
        
        # Track concurrent positions and margin over time
        active_trades = set()
        current_total_margin = 0.0
        
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
    
    # Compile metrics
    metrics = {
        'max_margin_used': max_margin_used,
        'max_margin_ratio': max_margin_ratio,
        'max_concurrent_positions': max_concurrent_positions,
        'initial_capital': STRATEGY_CONFIG['INITIAL_CAPITAL'],
        'final_capital': current_capital if trades else STRATEGY_CONFIG['INITIAL_CAPITAL'],
        'total_trades': len(trades),
        'enabled_strategies': enabled_strategies,
        'events_timeline': events
    }
    
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        trades_df.set_index('entry_time', inplace=True)
        print(f"Trades executed: {len(trades_df)}")
        print(f"Final capital: ${metrics['final_capital']:,.2f}")
        return trades_df, metrics
    
    print("No trades executed")
    return pd.DataFrame(), metrics

def run_walk_forward_analysis(df: pd.DataFrame, strategy_execution_func: Callable = execute_combined_strategy) -> pd.DataFrame:
    """
    Main walk-forward analysis engine for strategy evaluation.
    
    Args:
        df: Market data DataFrame
        strategy_execution_func: Function to execute strategy (default: execute_combined_strategy)
        
    Returns:
        DataFrame with walk-forward analysis results
    """
    print("=" * 100)
    print("STRATEGY WALK-FORWARD ANALYSIS ENGINE")
    print("=" * 100)
    
    from walk_forward.windows import create_walk_forward_windows
    from walk_forward.analysis import calculate_stable_strategies, analyze_performance_decay
    from risk_metrics.performance import calculate_advanced_metrics
    from risk_metrics.correlation import calculate_strategy_correlation
    
    windows = create_walk_forward_windows(df)
    
    if not windows:
        print("No valid windows created - check your data range")
        return pd.DataFrame()
    
    results = []
    
    for i, window in enumerate(windows):
        print(f"\nProcessing Window {i+1}/{len(windows)}")
        print(f"Training: {window['train_start'].strftime('%Y-%m-%d')} to {window['train_end'].strftime('%Y-%m-%d')}")
        print(f"Testing:  {window['test_start'].strftime('%Y-%m-%d')} to {window['test_end'].strftime('%Y-%m-%d')}")
        
        # === IN-SAMPLE OPTIMIZATION ===
        train_trades, train_metrics = strategy_execution_func(window['train_data'])
        
        if train_trades.empty:
            print("No trades in training period - skipping window")
            continue
        
        # Calculate STABLE strategies
        stable_strategies = calculate_stable_strategies(train_trades)
        
        if not stable_strategies:
            print("No stable strategies found - skipping window")
            continue
        
        # Calculate IS performance with stable strategies only
        is_stable_trades = train_trades[train_trades['pattern'].str.contains('|'.join(stable_strategies))]
        
        # Calculate IS metrics
        is_advanced_metrics = calculate_advanced_metrics(is_stable_trades)
        
        # Calculate IS correlation analysis
        is_correlation_results = calculate_strategy_correlation(is_stable_trades, stable_strategies)
        
        # === OUT-OF-SAMPLE VALIDATION ===
        test_trades, test_metrics = strategy_execution_func(window['test_data'])
        
        if test_trades.empty:
            print("No trades in testing period - skipping window")
            continue
        
        # Filter OOS trades to only include stable strategies
        oos_stable_trades = test_trades[test_trades['pattern'].str.contains('|'.join(stable_strategies))]
        
        if oos_stable_trades.empty:
            print("No stable strategy trades in testing period - skipping window")
            continue
        
        # Calculate OOS metrics
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
            
            # IS metrics
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
    
    if results:
        results_df = pd.DataFrame(results)
        print(f"\nWalk-forward analysis complete: {len(results_df)} windows analyzed")
        return results_df
    
    print("No valid results from walk-forward analysis")
    return pd.DataFrame()

# =============================================================================
# PERFORMANCE METRICS FUNCTION
# =============================================================================

def calculate_performance_metrics(trades_df: pd.DataFrame, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        trades_df: DataFrame of trades
        metrics: Basic metrics dictionary
        
    Returns:
        Dictionary of comprehensive performance metrics
    """
    if trades_df.empty:
        return {}
    
    from risk_metrics.performance import calculate_advanced_metrics, calculate_basic_metrics
    
    advanced_metrics = calculate_advanced_metrics(trades_df)
    basic_metrics = calculate_basic_metrics(trades_df)
    
    # Combine all metrics
    comprehensive_metrics = {
        **basic_metrics,
        **advanced_metrics,
        **metrics,
        'sharpe_ratio_r': advanced_metrics['sharpe_ratio_r'],
        'calmar_ratio_r': advanced_metrics['calmar_ratio_r'],
        'max_drawdown_r': advanced_metrics['max_drawdown_r'],
        'std_r_per_trade': advanced_metrics['std_r_per_trade'],
    }
    
    return comprehensive_metrics

# =============================================================================
# MAIN EXECUTION (if run as script)
# =============================================================================

if __name__ == "__main__":
    print("Backtesting Engine - Independent Module")
    print("This module provides core backtesting functionality.")
    print("Use execute_combined_strategy() to run backtests.")
    print("Use run_walk_forward_analysis() for walk-forward validation.")
