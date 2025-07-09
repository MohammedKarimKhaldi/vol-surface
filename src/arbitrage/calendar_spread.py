"""
Calendar Spread Arbitrage Detection
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings


class CalendarSpreadDetector:
    """
    Detect calendar spread arbitrage violations.
    
    Calendar spread arbitrage occurs when ∂σ²/∂T ≤ 0, which violates
    the positive time value decay principle.
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize calendar spread detector.
        
        Args:
            tolerance: Numerical tolerance for violations
        """
        self.tolerance = tolerance
    
    def check_calendar_spread(self, strikes: np.ndarray, maturities: np.ndarray,
                            implied_vols: np.ndarray) -> Dict[str, Any]:
        """
        Check for calendar spread arbitrage violations.
        
        Args:
            strikes: Strike prices
            maturities: Time to expiration
            implied_vols: Implied volatilities
            
        Returns:
            Dictionary with violation details
        """
        violations = []
        total_checks = 0
        
        # Sort by maturity
        sort_idx = np.argsort(maturities)
        sorted_strikes = strikes[sort_idx]
        sorted_maturities = maturities[sort_idx]
        sorted_vols = implied_vols[sort_idx]
        
        # Check each strike across maturities
        unique_strikes = np.unique(sorted_strikes)
        
        for strike in unique_strikes:
            strike_mask = sorted_strikes == strike
            strike_maturities = sorted_maturities[strike_mask]
            strike_vols = sorted_vols[strike_mask]
            
            if len(strike_maturities) < 2:
                continue
            
            # Check calendar spread condition
            for i in range(len(strike_maturities) - 1):
                T1, T2 = strike_maturities[i], strike_maturities[i + 1]
                vol1, vol2 = strike_vols[i], strike_vols[i + 1]
                
                # Calculate total variance
                var1 = vol1 ** 2 * T1
                var2 = vol2 ** 2 * T2
                
                # Check ∂σ²/∂T > 0 condition
                if var2 <= var1 + self.tolerance:
                    violations.append({
                        'strike': strike,
                        'maturity1': T1,
                        'maturity2': T2,
                        'vol1': vol1,
                        'vol2': vol2,
                        'var1': var1,
                        'var2': var2,
                        'violation': var1 - var2
                    })
                
                total_checks += 1
        
        return {
            'violations': violations,
            'total_checks': total_checks,
            'violation_count': len(violations),
            'violation_rate': len(violations) / max(1, total_checks)
        }
    
    def check_surface_monotonicity(self, strikes: np.ndarray, maturities: np.ndarray,
                                 implied_vols: np.ndarray) -> Dict[str, Any]:
        """
        Check if the volatility surface is monotonically increasing in time.
        
        Args:
            strikes: Strike prices
            maturities: Time to expiration
            implied_vols: Implied volatilities
            
        Returns:
            Dictionary with monotonicity analysis
        """
        # Create grid
        unique_strikes = np.unique(strikes)
        unique_maturities = np.unique(maturities)
        
        # Sort maturities
        unique_maturities = np.sort(unique_maturities)
        
        monotonicity_violations = []
        
        for strike in unique_strikes:
            strike_vols = []
            strike_mats = []
            
            # Get all volatilities for this strike
            for mat in unique_maturities:
                mask = (strikes == strike) & (maturities == mat)
                if np.any(mask):
                    vol = np.mean(implied_vols[mask])
                    strike_vols.append(vol)
                    strike_mats.append(mat)
            
            if len(strike_vols) < 2:
                continue
            
            # Check if volatilities are monotonically increasing
            for i in range(len(strike_vols) - 1):
                if strike_vols[i + 1] < strike_vols[i] - self.tolerance:
                    monotonicity_violations.append({
                        'strike': strike,
                        'maturity1': strike_mats[i],
                        'maturity2': strike_mats[i + 1],
                        'vol1': strike_vols[i],
                        'vol2': strike_vols[i + 1],
                        'violation': strike_vols[i] - strike_vols[i + 1]
                    })
        
        return {
            'monotonicity_violations': monotonicity_violations,
            'total_strikes': len(unique_strikes),
            'violation_count': len(monotonicity_violations)
        }
    
    def calculate_time_decay(self, strikes: np.ndarray, maturities: np.ndarray,
                           implied_vols: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate time decay metrics for the volatility surface.
        
        Args:
            strikes: Strike prices
            maturities: Time to expiration
            implied_vols: Implied volatilities
            
        Returns:
            Dictionary with time decay metrics
        """
        unique_strikes = np.unique(strikes)
        unique_maturities = np.sort(np.unique(maturities))
        
        time_decay_rates = []
        strike_list = []
        maturity_list = []
        
        for strike in unique_strikes:
            for i, mat in enumerate(unique_maturities[:-1]):
                # Get volatilities for consecutive maturities
                mask1 = (strikes == strike) & (maturities == mat)
                mask2 = (strikes == strike) & (maturities == unique_maturities[i + 1])
                
                if np.any(mask1) and np.any(mask2):
                    vol1 = np.mean(implied_vols[mask1])
                    vol2 = np.mean(implied_vols[mask2])
                    T1, T2 = mat, unique_maturities[i + 1]
                    
                    # Calculate time decay rate
                    decay_rate = (vol2 - vol1) / (T2 - T1)
                    
                    time_decay_rates.append(decay_rate)
                    strike_list.append(strike)
                    maturity_list.append((T1 + T2) / 2)
        
        return {
            'time_decay_rates': np.array(time_decay_rates),
            'strikes': np.array(strike_list),
            'maturities': np.array(maturity_list)
        }
    
    def detect_anomalies(self, strikes: np.ndarray, maturities: np.ndarray,
                        implied_vols: np.ndarray, threshold: float = 2.0) -> Dict[str, Any]:
        """
        Detect anomalous calendar spread patterns.
        
        Args:
            strikes: Strike prices
            maturities: Time to expiration
            implied_vols: Implied volatilities
            threshold: Standard deviation threshold for anomalies
            
        Returns:
            Dictionary with anomaly detection results
        """
        # Calculate time decay metrics
        decay_metrics = self.calculate_time_decay(strikes, maturities, implied_vols)
        
        if len(decay_metrics['time_decay_rates']) == 0:
            return {'anomalies': [], 'threshold': threshold}
        
        # Calculate statistics
        mean_decay = np.mean(decay_metrics['time_decay_rates'])
        std_decay = np.std(decay_metrics['time_decay_rates'])
        
        # Find anomalies
        anomalies = []
        for i, decay_rate in enumerate(decay_metrics['time_decay_rates']):
            z_score = abs(decay_rate - mean_decay) / std_decay
            
            if z_score > threshold:
                anomalies.append({
                    'strike': decay_metrics['strikes'][i],
                    'maturity': decay_metrics['maturities'][i],
                    'decay_rate': decay_rate,
                    'z_score': z_score,
                    'mean_decay': mean_decay,
                    'std_decay': std_decay
                })
        
        return {
            'anomalies': anomalies,
            'threshold': threshold,
            'mean_decay': mean_decay,
            'std_decay': std_decay,
            'total_points': len(decay_metrics['time_decay_rates'])
        } 