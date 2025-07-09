"""
Put-Call Parity Arbitrage Detection
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings


class PutCallParityDetector:
    """
    Detect put-call parity arbitrage violations.
    
    Put-call parity states: C - P = S₀e^(-qT) - Ke^(-rT)
    where C is call price, P is put price, S₀ is spot price,
    K is strike price, T is time to maturity, r is risk-free rate,
    and q is dividend yield.
    """
    
    def __init__(self, spot_price: float = 100.0, risk_free_rate: float = 0.02,
                 dividend_yield: float = 0.0, tolerance: float = 1e-6):
        """
        Initialize put-call parity detector.
        
        Args:
            spot_price: Current spot price
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield
            tolerance: Numerical tolerance for violations
        """
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.tolerance = tolerance
    
    def check_put_call_parity(self, strikes: np.ndarray, maturities: np.ndarray,
                             call_prices: np.ndarray, put_prices: np.ndarray) -> Dict[str, Any]:
        """
        Check for put-call parity violations.
        
        Args:
            strikes: Strike prices
            maturities: Time to expiration
            call_prices: Call option prices
            put_prices: Put option prices
            
        Returns:
            Dictionary with violation details
        """
        violations = []
        total_checks = 0
        
        for i in range(len(strikes)):
            K = strikes[i]
            T = maturities[i]
            C = call_prices[i]
            P = put_prices[i]
            
            # Calculate theoretical put-call parity value
            theoretical_diff = self._calculate_theoretical_diff(K, T)
            
            # Calculate actual difference
            actual_diff = C - P
            
            # Check for violation
            violation = abs(actual_diff - theoretical_diff)
            
            if violation > self.tolerance:
                violations.append({
                    'strike': K,
                    'maturity': T,
                    'call_price': C,
                    'put_price': P,
                    'actual_diff': actual_diff,
                    'theoretical_diff': theoretical_diff,
                    'violation': violation
                })
            
            total_checks += 1
        
        return {
            'violations': violations,
            'total_checks': total_checks,
            'violation_count': len(violations),
            'violation_rate': len(violations) / max(1, total_checks)
        }
    
    def _calculate_theoretical_diff(self, strike: float, maturity: float) -> float:
        """
        Calculate theoretical put-call parity difference.
        
        Args:
            strike: Strike price
            maturity: Time to maturity
            
        Returns:
            Theoretical difference C - P
        """
        return (self.spot_price * np.exp(-self.dividend_yield * maturity) - 
                strike * np.exp(-self.risk_free_rate * maturity))
    
    def check_implied_volatility_parity(self, strikes: np.ndarray, maturities: np.ndarray,
                                       call_vols: np.ndarray, put_vols: np.ndarray) -> Dict[str, Any]:
        """
        Check if implied volatilities from calls and puts are consistent.
        
        Args:
            strikes: Strike prices
            maturities: Time to expiration
            call_vols: Call implied volatilities
            put_vols: Put implied volatilities
            
        Returns:
            Dictionary with volatility parity analysis
        """
        violations = []
        total_checks = 0
        
        for i in range(len(strikes)):
            K = strikes[i]
            T = maturities[i]
            call_vol = call_vols[i]
            put_vol = put_vols[i]
            
            # Calculate volatility difference
            vol_diff = abs(call_vol - put_vol)
            
            if vol_diff > self.tolerance:
                violations.append({
                    'strike': K,
                    'maturity': T,
                    'call_vol': call_vol,
                    'put_vol': put_vol,
                    'vol_diff': vol_diff
                })
            
            total_checks += 1
        
        return {
            'violations': violations,
            'total_checks': total_checks,
            'violation_count': len(violations),
            'violation_rate': len(violations) / max(1, total_checks)
        }
    
    def calculate_arbitrage_opportunities(self, strikes: np.ndarray, maturities: np.ndarray,
                                        call_prices: np.ndarray, put_prices: np.ndarray) -> Dict[str, Any]:
        """
        Calculate potential arbitrage opportunities.
        
        Args:
            strikes: Strike prices
            maturities: Time to expiration
            call_prices: Call option prices
            put_prices: Put option prices
            
        Returns:
            Dictionary with arbitrage opportunities
        """
        opportunities = []
        
        for i in range(len(strikes)):
            K = strikes[i]
            T = maturities[i]
            C = call_prices[i]
            P = put_prices[i]
            
            # Calculate theoretical values
            theoretical_diff = self._calculate_theoretical_diff(K, T)
            actual_diff = C - P
            
            # Calculate potential profit
            if actual_diff > theoretical_diff + self.tolerance:
                # Sell call, buy put, buy stock
                profit = actual_diff - theoretical_diff
                strategy = "Sell Call + Buy Put + Buy Stock"
            elif actual_diff < theoretical_diff - self.tolerance:
                # Buy call, sell put, sell stock
                profit = theoretical_diff - actual_diff
                strategy = "Buy Call + Sell Put + Sell Stock"
            else:
                profit = 0.0
                strategy = "No arbitrage"
            
            if profit > 0:
                opportunities.append({
                    'strike': K,
                    'maturity': T,
                    'call_price': C,
                    'put_price': P,
                    'actual_diff': actual_diff,
                    'theoretical_diff': theoretical_diff,
                    'profit': profit,
                    'strategy': strategy
                })
        
        return {
            'opportunities': opportunities,
            'total_opportunities': len(opportunities),
            'total_profit': sum(opp['profit'] for opp in opportunities)
        }
    
    def calculate_risk_metrics(self, strikes: np.ndarray, maturities: np.ndarray,
                             call_prices: np.ndarray, put_prices: np.ndarray) -> Dict[str, float]:
        """
        Calculate risk metrics for put-call parity.
        
        Args:
            strikes: Strike prices
            maturities: Time to expiration
            call_prices: Call option prices
            put_prices: Put option prices
            
        Returns:
            Dictionary with risk metrics
        """
        # Check put-call parity
        parity_results = self.check_put_call_parity(strikes, maturities, call_prices, put_prices)
        
        # Calculate arbitrage opportunities
        arbitrage_results = self.calculate_arbitrage_opportunities(strikes, maturities, call_prices, put_prices)
        
        # Calculate price differences
        price_diffs = []
        theoretical_diffs = []
        
        for i in range(len(strikes)):
            K = strikes[i]
            T = maturities[i]
            C = call_prices[i]
            P = put_prices[i]
            
            actual_diff = C - P
            theoretical_diff = self._calculate_theoretical_diff(K, T)
            
            price_diffs.append(actual_diff)
            theoretical_diffs.append(theoretical_diff)
        
        price_diffs = np.array(price_diffs)
        theoretical_diffs = np.array(theoretical_diffs)
        
        return {
            'parity_violation_rate': parity_results['violation_rate'],
            'arbitrage_opportunities': arbitrage_results['total_opportunities'],
            'total_arbitrage_profit': arbitrage_results['total_profit'],
            'mean_price_diff': float(np.mean(price_diffs)),
            'std_price_diff': float(np.std(price_diffs)),
            'mean_theoretical_diff': float(np.mean(theoretical_diffs)),
            'std_theoretical_diff': float(np.std(theoretical_diffs)),
            'max_violation': float(np.max(np.abs(price_diffs - theoretical_diffs)))
        }
    
    def detect_anomalies(self, strikes: np.ndarray, maturities: np.ndarray,
                        call_prices: np.ndarray, put_prices: np.ndarray,
                        threshold: float = 2.0) -> Dict[str, Any]:
        """
        Detect anomalies in put-call parity relationships.
        
        Args:
            strikes: Strike prices
            maturities: Time to expiration
            call_prices: Call option prices
            put_prices: Put option prices
            threshold: Standard deviation threshold for anomalies
            
        Returns:
            Dictionary with anomaly detection results
        """
        # Calculate price differences
        price_diffs = []
        theoretical_diffs = []
        strike_list = []
        maturity_list = []
        
        for i in range(len(strikes)):
            K = strikes[i]
            T = maturities[i]
            C = call_prices[i]
            P = put_prices[i]
            
            actual_diff = C - P
            theoretical_diff = self._calculate_theoretical_diff(K, T)
            
            price_diffs.append(actual_diff)
            theoretical_diffs.append(theoretical_diff)
            strike_list.append(K)
            maturity_list.append(T)
        
        price_diffs = np.array(price_diffs)
        theoretical_diffs = np.array(theoretical_diffs)
        
        # Calculate residuals
        residuals = price_diffs - theoretical_diffs
        
        # Calculate statistics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        # Find anomalies
        anomalies = []
        for i, residual in enumerate(residuals):
            z_score = abs(residual - mean_residual) / std_residual
            
            if z_score > threshold:
                anomalies.append({
                    'strike': strike_list[i],
                    'maturity': maturity_list[i],
                    'call_price': call_prices[i],
                    'put_price': put_prices[i],
                    'actual_diff': price_diffs[i],
                    'theoretical_diff': theoretical_diffs[i],
                    'residual': residual,
                    'z_score': z_score
                })
        
        return {
            'anomalies': anomalies,
            'threshold': threshold,
            'mean_residual': mean_residual,
            'std_residual': std_residual,
            'total_points': len(residuals)
        } 