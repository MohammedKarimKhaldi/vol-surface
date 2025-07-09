"""
Greeks Calculation Engine
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings


class GreeksCalculator:
    """
    Calculate option Greeks: delta, gamma, vega, theta, and rho.
    
    Supports both analytical and numerical methods for Greek calculations.
    """
    
    def __init__(self, spot_price: float = 100.0, risk_free_rate: float = 0.02,
                 dividend_yield: float = 0.0):
        """
        Initialize Greeks calculator.
        
        Args:
            spot_price: Current spot price
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield
        """
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
    
    def calculate_greeks(self, strike: float, maturity: float, implied_vol: float,
                        option_type: str = 'call', method: str = 'analytical') -> Dict[str, float]:
        """
        Calculate all Greeks for an option.
        
        Args:
            strike: Strike price
            maturity: Time to maturity
            implied_vol: Implied volatility
            option_type: 'call' or 'put'
            method: 'analytical' or 'numerical'
            
        Returns:
            Dictionary with all Greeks
        """
        if method == 'analytical':
            return self._calculate_analytical_greeks(strike, maturity, implied_vol, option_type)
        else:
            return self._calculate_numerical_greeks(strike, maturity, implied_vol, option_type)
    
    def _calculate_analytical_greeks(self, strike: float, maturity: float,
                                   implied_vol: float, option_type: str) -> Dict[str, float]:
        """Calculate Greeks using analytical Black-Scholes formulas."""
        S = self.spot_price
        K = strike
        T = maturity
        sigma = implied_vol
        r = self.risk_free_rate
        q = self.dividend_yield
        
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Calculate option price
        if option_type.lower() == 'call':
            price = S * np.exp(-q * T) * self._normal_cdf(d1) - K * np.exp(-r * T) * self._normal_cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * self._normal_cdf(-d2) - S * np.exp(-q * T) * self._normal_cdf(-d1)
        
        # Calculate Greeks
        if option_type.lower() == 'call':
            delta = np.exp(-q * T) * self._normal_cdf(d1)
            theta = (-S * np.exp(-q * T) * self._normal_pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                    r * K * np.exp(-r * T) * self._normal_cdf(d2) + 
                    q * S * np.exp(-q * T) * self._normal_cdf(d1))
            rho = K * T * np.exp(-r * T) * self._normal_cdf(d2)
        else:  # put
            delta = np.exp(-q * T) * (self._normal_cdf(d1) - 1)
            theta = (-S * np.exp(-q * T) * self._normal_pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                    r * K * np.exp(-r * T) * self._normal_cdf(-d2) - 
                    q * S * np.exp(-q * T) * self._normal_cdf(-d1))
            rho = -K * T * np.exp(-r * T) * self._normal_cdf(-d2)
        
        # Gamma and Vega are the same for calls and puts
        gamma = np.exp(-q * T) * self._normal_pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * np.exp(-q * T) * self._normal_pdf(d1) * np.sqrt(T)
        
        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }
    
    def _calculate_numerical_greeks(self, strike: float, maturity: float,
                                  implied_vol: float, option_type: str) -> Dict[str, float]:
        """Calculate Greeks using finite difference methods."""
        S = self.spot_price
        K = strike
        T = maturity
        sigma = implied_vol
        
        # Small perturbation for finite differences
        eps = 1e-6
        
        # Calculate base price
        base_price = self._black_scholes_price(S, K, T, sigma, option_type)
        
        # Delta: ∂P/∂S
        delta = (self._black_scholes_price(S + eps, K, T, sigma, option_type) - 
                self._black_scholes_price(S - eps, K, T, sigma, option_type)) / (2 * eps)
        
        # Gamma: ∂²P/∂S²
        gamma = (self._black_scholes_price(S + eps, K, T, sigma, option_type) - 
                2 * base_price + 
                self._black_scholes_price(S - eps, K, T, sigma, option_type)) / (eps ** 2)
        
        # Vega: ∂P/∂σ
        vega = (self._black_scholes_price(S, K, T, sigma + eps, option_type) - 
               self._black_scholes_price(S, K, T, sigma - eps, option_type)) / (2 * eps)
        
        # Theta: ∂P/∂T
        theta = (self._black_scholes_price(S, K, T + eps, sigma, option_type) - 
                self._black_scholes_price(S, K, T - eps, sigma, option_type)) / (2 * eps)
        
        # Rho: ∂P/∂r
        rho = (self._black_scholes_price(S, K, T, sigma, option_type, r=self.risk_free_rate + eps) - 
               self._black_scholes_price(S, K, T, sigma, option_type, r=self.risk_free_rate - eps)) / (2 * eps)
        
        return {
            'price': base_price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }
    
    def _black_scholes_price(self, S: float, K: float, T: float, sigma: float,
                           option_type: str, r: Optional[float] = None) -> float:
        """Calculate Black-Scholes option price."""
        if r is None:
            r = self.risk_free_rate
        
        d1 = (np.log(S / K) + (r - self.dividend_yield + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S * np.exp(-self.dividend_yield * T) * self._normal_cdf(d1) - K * np.exp(-r * T) * self._normal_cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * self._normal_cdf(-d2) - S * np.exp(-self.dividend_yield * T) * self._normal_cdf(-d1)
        
        return price
    
    def calculate_portfolio_greeks(self, positions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate portfolio-level Greeks.
        
        Args:
            positions: List of position dictionaries with keys:
                      'strike', 'maturity', 'implied_vol', 'option_type', 'quantity'
            
        Returns:
            Dictionary with portfolio Greeks
        """
        portfolio_greeks = {
            'price': 0.0,
            'delta': 0.0,
            'gamma': 0.0,
            'vega': 0.0,
            'theta': 0.0,
            'rho': 0.0
        }
        
        for position in positions:
            greeks = self.calculate_greeks(
                position['strike'],
                position['maturity'],
                position['implied_vol'],
                position['option_type']
            )
            
            quantity = position.get('quantity', 1.0)
            
            for greek in portfolio_greeks:
                portfolio_greeks[greek] += greeks[greek] * quantity
        
        return portfolio_greeks
    
    def calculate_greek_sensitivities(self, strikes: np.ndarray, maturities: np.ndarray,
                                    implied_vols: np.ndarray, option_types: List[str]) -> Dict[str, np.ndarray]:
        """
        Calculate Greeks for multiple options.
        
        Args:
            strikes: Array of strike prices
            maturities: Array of maturities
            implied_vols: Array of implied volatilities
            option_types: List of option types
            
        Returns:
            Dictionary with arrays of Greeks
        """
        deltas = []
        gammas = []
        vegas = []
        thetas = []
        rhos = []
        prices = []
        
        for i in range(len(strikes)):
            greeks = self.calculate_greeks(
                strikes[i],
                maturities[i],
                implied_vols[i],
                option_types[i]
            )
            
            prices.append(greeks['price'])
            deltas.append(greeks['delta'])
            gammas.append(greeks['gamma'])
            vegas.append(greeks['vega'])
            thetas.append(greeks['theta'])
            rhos.append(greeks['rho'])
        
        return {
            'prices': np.array(prices),
            'deltas': np.array(deltas),
            'gammas': np.array(gammas),
            'vegas': np.array(vegas),
            'thetas': np.array(thetas),
            'rhos': np.array(rhos)
        }
    
    def calculate_risk_metrics(self, strikes: np.ndarray, maturities: np.ndarray,
                             implied_vols: np.ndarray, option_types: List[str],
                             quantities: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate risk metrics for a portfolio of options.
        
        Args:
            strikes: Array of strike prices
            maturities: Array of maturities
            implied_vols: Array of implied volatilities
            option_types: List of option types
            quantities: Array of position quantities
            
        Returns:
            Dictionary with risk metrics
        """
        if quantities is None:
            quantities = np.ones(len(strikes))
        
        # Calculate Greeks
        greeks = self.calculate_greek_sensitivities(strikes, maturities, implied_vols, option_types)
        
        # Calculate portfolio-level metrics
        portfolio_price = np.sum(greeks['prices'] * quantities)
        portfolio_delta = np.sum(greeks['deltas'] * quantities)
        portfolio_gamma = np.sum(greeks['gammas'] * quantities)
        portfolio_vega = np.sum(greeks['vegas'] * quantities)
        portfolio_theta = np.sum(greeks['thetas'] * quantities)
        portfolio_rho = np.sum(greeks['rhos'] * quantities)
        
        # Calculate risk metrics
        delta_exposure = portfolio_delta * self.spot_price
        gamma_exposure = portfolio_gamma * (self.spot_price ** 2)
        vega_exposure = portfolio_vega * 0.01  # 1% vol change
        theta_exposure = portfolio_theta * 1/365  # Daily theta
        rho_exposure = portfolio_rho * 0.01  # 1% rate change
        
        return {
            'portfolio_price': float(portfolio_price),
            'portfolio_delta': float(portfolio_delta),
            'portfolio_gamma': float(portfolio_gamma),
            'portfolio_vega': float(portfolio_vega),
            'portfolio_theta': float(portfolio_theta),
            'portfolio_rho': float(portfolio_rho),
            'delta_exposure': float(delta_exposure),
            'gamma_exposure': float(gamma_exposure),
            'vega_exposure': float(vega_exposure),
            'theta_exposure': float(theta_exposure),
            'rho_exposure': float(rho_exposure)
        }
    
    def _normal_cdf(self, x: float) -> float:
        """Standard normal cumulative distribution function."""
        import math
        return 0.5 * (1 + math.erf(x / np.sqrt(2)))
    
    def _normal_pdf(self, x: float) -> float:
        """Standard normal probability density function."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi) 