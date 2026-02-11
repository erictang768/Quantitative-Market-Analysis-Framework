"""
Configuration settings for the quantitative analysis framework.
All configuration parameters are centralized here for easy modification.
"""

# Walk-forward analysis configuration
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
    'CORRELATION_THRESHOLD': 0.7,
    'MIN_CORRELATION_SAMPLES': 10,
    'DIVERSIFICATION_SCORE_WEIGHTS': {
        'avg_correlation': 0.4,
        'max_correlation': 0.3,
        'sign_correlation': 0.3
    }
}

# Strategy configuration
STRATEGY_CONFIG = {
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
