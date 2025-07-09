"""
Variance Swap Calculator Implementation
"""

import numpy as np
from scipy.integrate import quad
from typing import Tuple, Optional, List, Dict, Any
import warnings


class VarianceSwapCalculator:
    """
    Model-free variance swap calculations using the continuum of strike prices.
    
    The variance swap fair value is given by:
    σ²(T) = (2/T)∫[Q(K)/K²]dK
    
    where Q(K) is the risk-neutral density.
    """
    
    def __init__(self, spot_price: float = 100.0, risk_free_rate: float = 0.02):
        """
        Initialize variance swap calculator.
        
        Args:
            spot_price: Current spot price
            risk_free_rate: Risk-free interest rate
        """
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        
    def calculate_variance_swap_rate(self, strikes: np.ndarray, call_prices: np.ndarray,
                                   put_prices: np.ndarray, maturity: float,
                                   dividend_yield: float = 0.0) -> float:
        """
        Calculate variance swap rate using model-free approach.
        
        Args:
            strikes: Array of strike prices
            call_prices: Array of call option prices
            put_prices: Array of put option prices
            maturity: Time to maturity in years
            dividend_yield: Dividend yield
            
        Returns:
            Variance swap rate (annualized)
        """
        # Sort data by strike
        sort_idx = np.argsort(strikes)
        strikes = strikes[sort_idx]
        call_prices = call_prices[sort_idx]
        put_prices = put_prices[sort_idx]
        
        # Find ATM strike (closest to spot)
        atm_idx = np.argmin(np.abs(strikes - self.spot_price))
        atm_strike = strikes[atm_idx]
        
        # Calculate variance swap rate
        variance_rate = self._compute_variance_integral(
            strikes, call_prices, put_prices, atm_strike, maturity, dividend_yield
        )
        
        return variance_rate
    
    def _compute_variance_integral(self, strikes: np.ndarray, call_prices: np.ndarray,
                                 put_prices: np.ndarray, atm_strike: float,
                                 maturity: float, dividend_yield: float) -> float:
        """
        Compute the variance integral using numerical integration.
        
        Args:
            strikes: Strike prices
            call_prices: Call option prices
            put_prices: Put option prices
            atm_strike: At-the-money strike
            maturity: Time to maturity
            dividend_yield: Dividend yield
            
        Returns:
            Variance swap rate
        """
        # Define integrand function
        def integrand(K):
            # Find closest strike and interpolate option price
            option_price = self._interpolate_option_price(K, strikes, call_prices, put_prices)
            return option_price / (K ** 2)
        
        # Compute integral using trapezoidal rule
        # Split into OTM calls and OTM puts
        otm_calls = strikes > atm_strike
        otm_puts = strikes < atm_strike
        
        # OTM calls contribution
        call_contribution = 0.0
        if np.any(otm_calls):
            call_strikes = strikes[otm_calls]
            call_prices_otm = call_prices[otm_calls]
            
            for i in range(len(call_strikes) - 1):
                K1, K2 = call_strikes[i], call_strikes[i + 1]
                P1, P2 = call_prices_otm[i], call_prices_otm[i + 1]
                
                # Trapezoidal rule
                call_contribution += (P1 / (K1 ** 2) + P2 / (K2 ** 2)) * (K2 - K1) / 2
        
        # OTM puts contribution
        put_contribution = 0.0
        if np.any(otm_puts):
            put_strikes = strikes[otm_puts]
            put_prices_otm = put_prices[otm_puts]
            
            for i in range(len(put_strikes) - 1):
                K1, K2 = put_strikes[i], put_strikes[i + 1]
                P1, P2 = put_prices_otm[i], put_prices_otm[i + 1]
                
                # Trapezoidal rule
                put_contribution += (P1 / (K1 ** 2) + P2 / (K2 ** 2)) * (K2 - K1) / 2
        
        # ATM contribution (use average of call and put)
        atm_idx = np.where(strikes == atm_strike)[0][0]
        atm_call = float(call_prices[atm_idx])
        atm_put = float(put_prices[atm_idx])
        atm_price = (atm_call + atm_put) / 2
        
        # Total variance
        total_integral = call_contribution + put_contribution + atm_price / (atm_strike ** 2)
        
        # Variance swap rate
        variance_rate = (2 / maturity) * total_integral
        
        return variance_rate
    
    def _interpolate_option_price(self, K: float, strikes: np.ndarray,
                                call_prices: np.ndarray, put_prices: np.ndarray) -> float:
        """
        Interpolate option price for given strike.
        
        Args:
            K: Strike price
            strikes: Available strikes
            call_prices: Call prices
            put_prices: Put prices
            
        Returns:
            Interpolated option price
        """
        # Find closest strikes
        idx = np.searchsorted(strikes, K)
        
        if idx == 0:
            # K is below minimum strike, use put
            return put_prices[0]
        elif idx == len(strikes):
            # K is above maximum strike, use call
            return call_prices[-1]
        else:
            # Interpolate between strikes
            K1, K2 = strikes[idx - 1], strikes[idx]
            
            if K <= self.spot_price:
                # Use put prices
                P1, P2 = put_prices[idx - 1], put_prices[idx]
            else:
                # Use call prices
                P1, P2 = call_prices[idx - 1], call_prices[idx]
            
            # Linear interpolation
            weight = (K - K1) / (K2 - K1)
            return P1 + weight * (P2 - P1)
    
    def calculate_implied_volatility_from_variance(self, variance_rate: float,
                                                 strike: float, maturity: float,
                                                 option_type: str = 'call',
                                                 spot_price: Optional[float] = None,
                                                 risk_free_rate: Optional[float] = None,
                                                 dividend_yield: float = 0.0) -> float:
        """
        Calculate implied volatility from variance swap rate.
        
        Args:
            variance_rate: Variance swap rate
            strike: Strike price
            option_type: 'call' or 'put'
            maturity: Time to maturity
            spot_price: Current spot price
            risk_free_rate: Risk-free rate
            dividend_yield: Dividend yield
            
        Returns:
            Implied volatility
        """
        spot = spot_price or self.spot_price
        rate = risk_free_rate or self.risk_free_rate
        
        # Use variance rate as initial guess for implied vol
        implied_vol = np.sqrt(variance_rate)
        
        # Refine using Newton-Raphson method
        for _ in range(10):
            # Calculate option price using current implied vol
            option_price = self._black_scholes_price(
                spot, strike, maturity, implied_vol, rate, dividend_yield, option_type
            )
            
            # Calculate vega
            vega = self._black_scholes_vega(
                spot, strike, maturity, implied_vol, rate, dividend_yield
            )
            
            # Calculate target price (using variance rate approximation)
            target_price = self._variance_based_price(
                spot, strike, maturity, variance_rate, rate, dividend_yield, option_type
            )
            
            # Update implied vol
            price_diff = target_price - option_price
            if abs(price_diff) < 1e-6:
                break
                
            implied_vol += price_diff / vega
            implied_vol = max(0.001, implied_vol)  # Ensure positive
        
        return implied_vol
    
    def _black_scholes_price(self, S: float, K: float, T: float, sigma: float,
                           r: float, q: float, option_type: str) -> float:
        """Calculate Black-Scholes option price."""
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S * np.exp(-q * T) * self._normal_cdf(d1) - K * np.exp(-r * T) * self._normal_cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * self._normal_cdf(-d2) - S * np.exp(-q * T) * self._normal_cdf(-d1)
        
        return price
    
    def _black_scholes_vega(self, S: float, K: float, T: float, sigma: float,
                           r: float, q: float) -> float:
        """Calculate Black-Scholes vega."""
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * np.exp(-q * T) * np.sqrt(T) * self._normal_pdf(d1)
    
    def _variance_based_price(self, S: float, K: float, T: float, variance_rate: float,
                            r: float, q: float, option_type: str) -> float:
        """Calculate option price based on variance swap rate."""
        # Simple approximation using variance rate as implied vol
        implied_vol = np.sqrt(variance_rate)
        return self._black_scholes_price(S, K, T, implied_vol, r, q, option_type)
    
    def _normal_cdf(self, x: float) -> float:
        """Standard normal cumulative distribution function."""
        import math
        return 0.5 * (1 + math.erf(x / np.sqrt(2)))
    
    def _normal_pdf(self, x: float) -> float:
        """Standard normal probability density function."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def calculate_term_structure(self, maturities: np.ndarray,
                               variance_rates: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate variance swap term structure.
        
        Args:
            maturities: Array of maturities
            variance_rates: Array of variance rates
            
        Returns:
            Dictionary with term structure components
        """
        # Sort by maturity
        sort_idx = np.argsort(maturities)
        maturities = maturities[sort_idx]
        variance_rates = variance_rates[sort_idx]
        
        # Calculate forward variance rates
        forward_rates = np.zeros_like(variance_rates)
        for i in range(1, len(maturities)):
            T1, T2 = maturities[i-1], maturities[i]
            V1, V2 = variance_rates[i-1], variance_rates[i]
            
            # Forward variance rate
            forward_rates[i] = (V2 * T2 - V1 * T1) / (T2 - T1)
        
        # Calculate volatility of volatility (simplified)
        vol_of_vol = np.gradient(variance_rates, maturities)
        
        return {
            'maturities': maturities,
            'variance_rates': variance_rates,
            'forward_rates': forward_rates,
            'vol_of_vol': vol_of_vol
        } 