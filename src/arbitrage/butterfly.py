"""
Butterfly Arbitrage Detection
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings


class ButterflyDetector:
    """
    Detect butterfly arbitrage violations.
    
    Butterfly arbitrage occurs when σ²(K₁) + σ²(K₃) - 2σ²(K₂) < 0
    for K₁ < K₂ < K₃, which violates the convexity condition.
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize butterfly detector.
        
        Args:
            tolerance: Numerical tolerance for violations
        """
        self.tolerance = tolerance
    
    def check_butterfly_arbitrage(self, strikes: np.ndarray, maturities: np.ndarray,
                                implied_vols: np.ndarray) -> Dict[str, Any]:
        """
        Check for butterfly arbitrage violations.
        
        Args:
            strikes: Strike prices
            maturities: Time to expiration
            implied_vols: Implied volatilities
            
        Returns:
            Dictionary with violation details
        """
        violations = []
        total_checks = 0
        
        # Group by maturity
        unique_maturities = np.unique(maturities)
        
        for maturity in unique_maturities:
            maturity_mask = maturities == maturity
            maturity_strikes = strikes[maturity_mask]
            maturity_vols = implied_vols[maturity_mask]
            
            # Sort by strike
            sort_idx = np.argsort(maturity_strikes)
            sorted_strikes = maturity_strikes[sort_idx]
            sorted_vols = maturity_vols[sort_idx]
            
            # Check butterfly condition for all triplets
            for i in range(len(sorted_strikes) - 2):
                K1, K2, K3 = sorted_strikes[i], sorted_strikes[i + 1], sorted_strikes[i + 2]
                vol1, vol2, vol3 = sorted_vols[i], sorted_vols[i + 1], sorted_vols[i + 2]
                
                # Calculate total variances
                var1 = vol1 ** 2 * maturity
                var2 = vol2 ** 2 * maturity
                var3 = vol3 ** 2 * maturity
                
                # Check butterfly condition: var1 + var3 - 2*var2 >= 0
                butterfly_value = var1 + var3 - 2 * var2
                
                if butterfly_value < -self.tolerance:
                    violations.append({
                        'maturity': maturity,
                        'strike1': K1,
                        'strike2': K2,
                        'strike3': K3,
                        'vol1': vol1,
                        'vol2': vol2,
                        'vol3': vol3,
                        'var1': var1,
                        'var2': var2,
                        'var3': var3,
                        'butterfly_value': butterfly_value,
                        'violation': -butterfly_value
                    })
                
                total_checks += 1
        
        return {
            'violations': violations,
            'total_checks': total_checks,
            'violation_count': len(violations),
            'violation_rate': len(violations) / max(1, total_checks)
        }
    
    def check_convexity(self, strikes: np.ndarray, maturities: np.ndarray,
                       implied_vols: np.ndarray) -> Dict[str, Any]:
        """
        Check convexity of the volatility smile.
        
        Args:
            strikes: Strike prices
            maturities: Time to expiration
            implied_vols: Implied volatilities
            
        Returns:
            Dictionary with convexity analysis
        """
        convexity_violations = []
        total_checks = 0
        
        unique_maturities = np.unique(maturities)
        
        for maturity in unique_maturities:
            maturity_mask = maturities == maturity
            maturity_strikes = strikes[maturity_mask]
            maturity_vols = implied_vols[maturity_mask]
            
            # Sort by strike
            sort_idx = np.argsort(maturity_strikes)
            sorted_strikes = maturity_strikes[sort_idx]
            sorted_vols = maturity_vols[sort_idx]
            
            # Calculate second differences
            for i in range(1, len(sorted_strikes) - 1):
                K_prev, K_curr, K_next = sorted_strikes[i-1], sorted_strikes[i], sorted_strikes[i+1]
                vol_prev, vol_curr, vol_next = sorted_vols[i-1], sorted_vols[i], sorted_vols[i+1]
                
                # Calculate second derivative approximation
                h1 = K_curr - K_prev
                h2 = K_next - K_curr
                
                # Second derivative using finite differences
                second_deriv = (2 / (h1 + h2)) * (
                    (vol_next - vol_curr) / h2 - (vol_curr - vol_prev) / h1
                )
                
                # Check convexity (second derivative should be positive)
                if second_deriv < -self.tolerance:
                    convexity_violations.append({
                        'maturity': maturity,
                        'strike': K_curr,
                        'vol_prev': vol_prev,
                        'vol_curr': vol_curr,
                        'vol_next': vol_next,
                        'second_derivative': second_deriv,
                        'violation': -second_deriv
                    })
                
                total_checks += 1
        
        return {
            'convexity_violations': convexity_violations,
            'total_checks': total_checks,
            'violation_count': len(convexity_violations),
            'violation_rate': len(convexity_violations) / max(1, total_checks)
        }
    
    def calculate_butterfly_spreads(self, strikes: np.ndarray, maturities: np.ndarray,
                                  implied_vols: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate butterfly spread metrics.
        
        Args:
            strikes: Strike prices
            maturities: Time to expiration
            implied_vols: Implied volatilities
            
        Returns:
            Dictionary with butterfly spread metrics
        """
        unique_maturities = np.unique(maturities)
        
        butterfly_values = []
        strike_list = []
        maturity_list = []
        
        for maturity in unique_maturities:
            maturity_mask = maturities == maturity
            maturity_strikes = strikes[maturity_mask]
            maturity_vols = implied_vols[maturity_mask]
            
            # Sort by strike
            sort_idx = np.argsort(maturity_strikes)
            sorted_strikes = maturity_strikes[sort_idx]
            sorted_vols = maturity_vols[sort_idx]
            
            # Calculate butterfly spreads
            for i in range(len(sorted_strikes) - 2):
                K1, K2, K3 = sorted_strikes[i], sorted_strikes[i + 1], sorted_strikes[i + 2]
                vol1, vol2, vol3 = sorted_vols[i], sorted_vols[i + 1], sorted_vols[i + 2]
                
                # Calculate total variances
                var1 = vol1 ** 2 * maturity
                var2 = vol2 ** 2 * maturity
                var3 = vol3 ** 2 * maturity
                
                # Butterfly spread value
                butterfly_value = var1 + var3 - 2 * var2
                
                butterfly_values.append(butterfly_value)
                strike_list.append(K2)  # Middle strike
                maturity_list.append(maturity)
        
        return {
            'butterfly_values': np.array(butterfly_values),
            'strikes': np.array(strike_list),
            'maturities': np.array(maturity_list)
        }
    
    def detect_smile_anomalies(self, strikes: np.ndarray, maturities: np.ndarray,
                             implied_vols: np.ndarray, threshold: float = 2.0) -> Dict[str, Any]:
        """
        Detect anomalies in the volatility smile.
        
        Args:
            strikes: Strike prices
            maturities: Time to expiration
            implied_vols: Implied volatilities
            threshold: Standard deviation threshold for anomalies
            
        Returns:
            Dictionary with smile anomaly detection results
        """
        # Calculate butterfly spread metrics
        butterfly_metrics = self.calculate_butterfly_spreads(strikes, maturities, implied_vols)
        
        if len(butterfly_metrics['butterfly_values']) == 0:
            return {'anomalies': [], 'threshold': threshold}
        
        # Calculate statistics
        mean_butterfly = np.mean(butterfly_metrics['butterfly_values'])
        std_butterfly = np.std(butterfly_metrics['butterfly_values'])
        
        # Find anomalies
        anomalies = []
        for i, butterfly_value in enumerate(butterfly_metrics['butterfly_values']):
            z_score = abs(butterfly_value - mean_butterfly) / std_butterfly
            
            if z_score > threshold:
                anomalies.append({
                    'strike': butterfly_metrics['strikes'][i],
                    'maturity': butterfly_metrics['maturities'][i],
                    'butterfly_value': butterfly_value,
                    'z_score': z_score,
                    'mean_butterfly': mean_butterfly,
                    'std_butterfly': std_butterfly
                })
        
        return {
            'anomalies': anomalies,
            'threshold': threshold,
            'mean_butterfly': mean_butterfly,
            'std_butterfly': std_butterfly,
            'total_points': len(butterfly_metrics['butterfly_values'])
        }
    
    def calculate_risk_metrics(self, strikes: np.ndarray, maturities: np.ndarray,
                             implied_vols: np.ndarray) -> Dict[str, float]:
        """
        Calculate risk metrics for butterfly arbitrage.
        
        Args:
            strikes: Strike prices
            maturities: Time to expiration
            implied_vols: Implied volatilities
            
        Returns:
            Dictionary with risk metrics
        """
        # Check butterfly arbitrage
        butterfly_results = self.check_butterfly_arbitrage(strikes, maturities, implied_vols)
        
        # Check convexity
        convexity_results = self.check_convexity(strikes, maturities, implied_vols)
        
        # Calculate butterfly spreads
        butterfly_metrics = self.calculate_butterfly_spreads(strikes, maturities, implied_vols)
        
        if len(butterfly_metrics['butterfly_values']) == 0:
            return {
                'butterfly_violation_rate': 0.0,
                'convexity_violation_rate': 0.0,
                'min_butterfly_value': 0.0,
                'max_butterfly_value': 0.0,
                'mean_butterfly_value': 0.0,
                'butterfly_volatility': 0.0
            }
        
        return {
            'butterfly_violation_rate': butterfly_results['violation_rate'],
            'convexity_violation_rate': convexity_results['violation_rate'],
            'min_butterfly_value': float(np.min(butterfly_metrics['butterfly_values'])),
            'max_butterfly_value': float(np.max(butterfly_metrics['butterfly_values'])),
            'mean_butterfly_value': float(np.mean(butterfly_metrics['butterfly_values'])),
            'butterfly_volatility': float(np.std(butterfly_metrics['butterfly_values']))
        } 