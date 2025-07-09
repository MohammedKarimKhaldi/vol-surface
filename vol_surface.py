"""
Main Volatility Surface Class
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

from src.core.svi_model import SVIModel
from src.core.variance_swap import VarianceSwapCalculator
from src.core.spline_interpolation import SplineInterpolator
from src.arbitrage.calendar_spread import CalendarSpreadDetector
from src.arbitrage.butterfly import ButterflyDetector
from src.arbitrage.put_call_parity import PutCallParityDetector
from src.greeks.sensitivities import GreeksCalculator


class VolatilitySurface:
    """
    Main volatility surface class that integrates all components.
    
    Provides a unified interface for:
    - SVI model fitting
    - Variance swap calculations
    - Spline interpolation
    - Arbitrage detection
    - Greeks calculation
    """
    
    def __init__(self, spot_price: float = 100.0, risk_free_rate: float = 0.02,
                 dividend_yield: float = 0.0):
        """
        Initialize volatility surface.
        
        Args:
            spot_price: Current spot price
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield
        """
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        
        # Initialize components
        self.svi_model = SVIModel()
        self.variance_calculator = VarianceSwapCalculator(spot_price, risk_free_rate)
        self.spline_interpolator = SplineInterpolator()
        
        # Initialize arbitrage detectors
        self.calendar_detector = CalendarSpreadDetector()
        self.butterfly_detector = ButterflyDetector()
        self.put_call_detector = PutCallParityDetector(spot_price, risk_free_rate, dividend_yield)
        
        # Initialize Greeks calculator
        self.greeks_calculator = GreeksCalculator(spot_price, risk_free_rate, dividend_yield)
        
        # Data storage
        self.data = None
        self.fitted = False
        
    def load_data(self, data: Union[pd.DataFrame, Dict[str, np.ndarray]]) -> None:
        """
        Load option data.
        
        Args:
            data: DataFrame or dictionary with option data
        """
        if isinstance(data, pd.DataFrame):
            # Convert DataFrame to arrays
            self.data = {
                'strikes': data['strike'].values,
                'maturities': data['maturity'].values,
                'implied_vols': data['implied_vol'].values,
                'call_prices': data.get('call_price', None),
                'put_prices': data.get('put_price', None),
                'option_types': data.get('option_type', None)
            }
        else:
            self.data = data
        
        self.fitted = False
    
    def fit_svi_model(self, constraints: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Fit SVI model to the data.
        
        Args:
            constraints: Optional parameter constraints
            
        Returns:
            Fitted SVI parameters
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if constraints:
            self.svi_model = SVIModel(constraints)
        
        parameters = self.svi_model.fit(
            self.data['strikes'],
            self.data['maturities'],
            self.data['implied_vols']
        )
        
        self.fitted = True
        return parameters
    
    def fit_spline_surface(self, tension: float = 0.0) -> None:
        """
        Fit spline surface to the data.
        
        Args:
            tension: Tension parameter for spline interpolation
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        self.spline_interpolator = SplineInterpolator(tension=tension)
        self.spline_interpolator.fit_2d_surface(
            self.data['strikes'],
            self.data['maturities'],
            self.data['implied_vols']
        )
        
        self.fitted = True
    
    def predict_volatility(self, strikes: np.ndarray, maturities: np.ndarray,
                          method: str = 'svi') -> Union[float, np.ndarray]:
        """
        Predict implied volatilities.
        
        Args:
            strikes: Strike prices
            maturities: Time to maturity
            method: 'svi' or 'spline'
            
        Returns:
            Predicted implied volatilities
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit_svi_model() or fit_spline_surface() first.")
        
        if method == 'svi':
            return self.svi_model.predict(strikes, maturities)
        elif method == 'spline':
            return self.spline_interpolator.predict(strikes, maturities)
        else:
            raise ValueError("Method must be 'svi' or 'spline'")
    
    def calculate_variance_swap_rate(self, maturity: float) -> float:
        """
        Calculate variance swap rate for given maturity.
        
        Args:
            maturity: Time to maturity
            
        Returns:
            Variance swap rate
        """
        if self.data is None or 'call_prices' not in self.data or 'put_prices' not in self.data:
            raise ValueError("Call and put prices required for variance swap calculation.")
        
        # Filter data for given maturity
        maturity_mask = np.abs(self.data['maturities'] - maturity) < 1e-6
        strikes = self.data['strikes'][maturity_mask]
        call_prices = self.data['call_prices'][maturity_mask]
        put_prices = self.data['put_prices'][maturity_mask]
        
        return self.variance_calculator.calculate_variance_swap_rate(
            strikes, call_prices, put_prices, maturity, self.dividend_yield
        )
    
    def detect_arbitrage(self) -> Dict[str, Any]:
        """
        Detect all types of arbitrage violations.
        
        Returns:
            Dictionary with arbitrage detection results
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        results = {}
        
        # Calendar spread arbitrage
        results['calendar_spread'] = self.calendar_detector.check_calendar_spread(
            self.data['strikes'],
            self.data['maturities'],
            self.data['implied_vols']
        )
        
        # Butterfly arbitrage
        results['butterfly'] = self.butterfly_detector.check_butterfly_arbitrage(
            self.data['strikes'],
            self.data['maturities'],
            self.data['implied_vols']
        )
        
        # Put-call parity (if call and put prices available)
        if 'call_prices' in self.data and 'put_prices' in self.data:
            results['put_call_parity'] = self.put_call_detector.check_put_call_parity(
                self.data['strikes'],
                self.data['maturities'],
                self.data['call_prices'],
                self.data['put_prices']
            )
        
        return results
    
    def calculate_greeks(self, strikes: np.ndarray, maturities: np.ndarray,
                        option_types: List[str], method: str = 'analytical') -> Dict[str, np.ndarray]:
        """
        Calculate Greeks for options.
        
        Args:
            strikes: Strike prices
            maturities: Time to maturity
            option_types: List of option types
            method: 'analytical' or 'numerical'
            
        Returns:
            Dictionary with Greeks arrays
        """
        # Predict implied volatilities
        implied_vols = self.predict_volatility(strikes, maturities)
        implied_vols = np.array(implied_vols)
        
        return self.greeks_calculator.calculate_greek_sensitivities(
            strikes, maturities, implied_vols, option_types
        )
    
    def calculate_risk_metrics(self, strikes: np.ndarray, maturities: np.ndarray,
                             option_types: List[str], quantities: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate risk metrics for a portfolio.
        
        Args:
            strikes: Strike prices
            maturities: Time to maturity
            option_types: List of option types
            quantities: Position quantities
            
        Returns:
            Dictionary with risk metrics
        """
        # Predict implied volatilities
        implied_vols = self.predict_volatility(strikes, maturities)
        implied_vols = np.array(implied_vols)
        
        return self.greeks_calculator.calculate_risk_metrics(
            strikes, maturities, implied_vols, option_types, quantities
        )
    
    def generate_surface_grid(self, strike_range: Tuple[float, float], 
                            maturity_range: Tuple[float, float],
                            n_strikes: int = 50, n_maturities: int = 20,
                            method: str = 'svi') -> Dict[str, np.ndarray]:
        """
        Generate volatility surface grid.
        
        Args:
            strike_range: (min_strike, max_strike)
            maturity_range: (min_maturity, max_maturity)
            n_strikes: Number of strike points
            n_maturities: Number of maturity points
            method: 'svi' or 'spline'
            
        Returns:
            Dictionary with surface grid data
        """
        strikes = np.linspace(strike_range[0], strike_range[1], n_strikes)
        maturities = np.linspace(maturity_range[0], maturity_range[1], n_maturities)
        
        K, T = np.meshgrid(strikes, maturities)
        
        implied_vols = self.predict_volatility(K.flatten(), T.flatten(), method)
        implied_vols = np.array(implied_vols).reshape(K.shape)
        
        return {
            'strikes': K,
            'maturities': T,
            'implied_vols': implied_vols
        }
    
    def plot_surface(self, method: str = 'svi', title: str = "Volatility Surface"):
        """Plot the volatility surface."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit_svi_model() or fit_spline_surface() first.")
        
        if method == 'svi':
            # Use SVI model's built-in plotting
            strikes = np.linspace(0.5 * self.spot_price, 1.5 * self.spot_price, 50)
            maturities = np.linspace(0.1, 2.0, 20)
            self.svi_model.plot_surface(strikes, maturities, title)
        else:
            # Generate grid and plot manually
            surface_data = self.generate_surface_grid(
                (0.5 * self.spot_price, 1.5 * self.spot_price),
                (0.1, 2.0),
                method=method
            )
            
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                surf = ax.plot_surface(
                    surface_data['strikes'],
                    surface_data['maturities'],
                    surface_data['implied_vols'],
                    cmap='viridis',
                    alpha=0.8
                )
                
                ax.set_xlabel('Strike Price')
                ax.set_ylabel('Time to Maturity')
                ax.set_zlabel('Implied Volatility')
                ax.set_title(title)
                
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
                plt.show()
                
            except ImportError:
                print("Matplotlib not available for plotting")
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics of the volatility surface.
        
        Returns:
            Dictionary with summary statistics
        """
        if self.data is None:
            return {}
        
        # Basic statistics
        stats = {
            'total_options': len(self.data['strikes']),
            'unique_strikes': len(np.unique(self.data['strikes'])),
            'unique_maturities': len(np.unique(self.data['maturities'])),
            'min_strike': float(np.min(self.data['strikes'])),
            'max_strike': float(np.max(self.data['strikes'])),
            'min_maturity': float(np.min(self.data['maturities'])),
            'max_maturity': float(np.max(self.data['maturities'])),
            'min_vol': float(np.min(self.data['implied_vols'])),
            'max_vol': float(np.max(self.data['implied_vols'])),
            'mean_vol': float(np.mean(self.data['implied_vols'])),
            'std_vol': float(np.std(self.data['implied_vols']))
        }
        
        # Arbitrage detection results
        if self.fitted:
            arbitrage_results = self.detect_arbitrage()
            stats['arbitrage_violations'] = {
                'calendar_spread': arbitrage_results.get('calendar_spread', {}).get('violation_count', 0),
                'butterfly': arbitrage_results.get('butterfly', {}).get('violation_count', 0),
                'put_call_parity': arbitrage_results.get('put_call_parity', {}).get('violation_count', 0)
            }
        
        return stats 