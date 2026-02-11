"""
Base strategy class and utility functions for position sizing and margin.
"""

import numpy as np
import math
from typing import Tuple

class BaseStrategy:
    """Base class for all trading strategies."""
    
    def __init__(self, config: dict):
        self.config = config
        self.name = "base_strategy"
        
    def calculate_position_size(self, current_capital: float, risk_distance: float, 
                              commission_per_lot: float, risk_pct: float) -> Tuple[float, float, float]:
        """Calculate position size with commission."""
        R = current_capital * risk_pct
        D = risk_distance * 100000
        C = commission_per_lot
        
        if (D + C) <= 0:
            return 0.0, 0.0, 0.0
        
        L = R / (D + C)
        actual_risk = L * (D + C)
        commission_cost = L * C
        
        return L, actual_risk, commission_cost
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, side: str,
                            commission_per_lot: float, net_rr_ratio: float) -> float:
        """Calculate take profit for net risk-reward after commission."""
        risk_distance = abs(entry_price - stop_loss)
        D = risk_distance * 100000
        C = commission_per_lot
        
        P_dollar = net_rr_ratio * D + (net_rr_ratio + 1) * C
        P_price = P_dollar / 100000
        
        if side == 'bull':
            return entry_price + P_price
        else:
            return entry_price - P_price
    
    def check_margin_requirement(self, position_size: float, entry_price: float, 
                               current_capital: float, leverage: float, 
                               margin_buffer_pct: float) -> Tuple[bool, float, float]:
        """Check margin requirements."""
        contract_size = position_size * 100000
        required_margin = (contract_size * entry_price) / leverage
        max_allowed_margin = current_capital * margin_buffer_pct
        
        if required_margin <= max_allowed_margin:
            return True, required_margin, position_size
        else:
            max_allowed_lots = (max_allowed_margin * leverage) / (entry_price * 100000)
            return False, required_margin, max_allowed_lots
