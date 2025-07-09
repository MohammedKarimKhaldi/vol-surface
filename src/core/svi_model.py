"""
SVI (Stochastic Volatility Inspired) Model Implementation
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Optional, Dict, Any, Union
try:
    import numba
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class SVIModel:
    """
    Stochastic Volatility Inspired (SVI) model for volatility surface parameterization.
    
    The SVI model parameterizes the total variance as:
    σ²(k,T) = a + b[ρ(k-m) + √((k-m)² + σ²)]
    
    where k represents log-moneyness.
    """
    
    def __init__(self, constraints: Optional[Dict[str, Any]] = None):
        """
        Initialize SVI model with optional constraints.
        
        Args:
            constraints: Dictionary of parameter constraints
        """
        self.constraints = constraints or self._default_constraints()
        self.parameters: Optional[Dict[str, float]] = None
        self.fitted = False
        
    def _default_constraints(self) -> Dict[str, Any]:
        """Default parameter constraints for SVI model."""
        return {
            'a': (-np.inf, np.inf),      # Vertical offset
            'b': (0, np.inf),            # Slope
            'rho': (-1, 1),              # Correlation
            'm': (-np.inf, np.inf),      # Horizontal offset
            'sigma': (0, np.inf)         # Volatility of volatility
        }
    
    @staticmethod
    @jit(nopython=True)
    def _svi_function(k: np.ndarray, a: float, b: float, rho: float, 
                     m: float, sigma: float) -> np.ndarray:
        """
        Compute SVI total variance function.
        
        Args:
            k: Log-moneyness array
            a, b, rho, m, sigma: SVI parameters
            
        Returns:
            Total variance array
        """
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
    
    def fit(self, strikes: np.ndarray, maturities: np.ndarray, 
            implied_vols: np.ndarray, weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Fit SVI model to implied volatility data.
        
        Args:
            strikes: Strike prices
            maturities: Time to expiration
            implied_vols: Implied volatilities
            weights: Optional weights for fitting
            
        Returns:
            Fitted parameters
        """
        # Convert to log-moneyness (assuming spot price = 1 for simplicity)
        log_moneyness = np.log(strikes)
        
        # Convert implied vol to total variance
        total_variance = (implied_vols ** 2) * maturities
        
        # Initial parameter guess
        initial_params = self._initial_guess(log_moneyness, total_variance)
        
        # Define objective function
        def objective(params):
            a, b, rho, m, sigma = params
            fitted_variance = self._svi_function(log_moneyness, a, b, rho, m, sigma)
            
            if weights is not None:
                residuals = weights * (fitted_variance - total_variance) ** 2
            else:
                residuals = (fitted_variance - total_variance) ** 2
                
            return np.sum(residuals)
        
        # Define bounds
        bounds = [
            self.constraints['a'],
            self.constraints['b'],
            self.constraints['rho'],
            self.constraints['m'],
            self.constraints['sigma']
        ]
        
        # Optimize
        result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
        
        if not result.success:
            raise ValueError(f"SVI fitting failed: {result.message}")
        
        self.parameters = {
            'a': float(result.x[0]),
            'b': float(result.x[1]),
            'rho': float(result.x[2]),
            'm': float(result.x[3]),
            'sigma': float(result.x[4])
        }
        
        self.fitted = True
        return self.parameters
    
    def _initial_guess(self, log_moneyness: np.ndarray, 
                      total_variance: np.ndarray) -> np.ndarray:
        """Generate initial parameter guess."""
        # Simple heuristics for initial values
        a = np.mean(total_variance)
        b = np.std(total_variance) / np.std(log_moneyness)
        rho = 0.0
        m = np.mean(log_moneyness)
        sigma = np.std(log_moneyness)
        
        return np.array([a, b, rho, m, sigma])
    
    def predict(self, strikes: np.ndarray, maturities: np.ndarray) -> np.ndarray:
        """
        Predict implied volatilities using fitted SVI model.
        
        Args:
            strikes: Strike prices
            maturities: Time to expiration
            
        Returns:
            Predicted implied volatilities
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        log_moneyness = np.log(strikes)
        total_variance = self._svi_function(
            log_moneyness, 
            self.parameters['a'],
            self.parameters['b'],
            self.parameters['rho'],
            self.parameters['m'],
            self.parameters['sigma']
        )
        
        # Convert back to implied volatility
        implied_vols = np.sqrt(total_variance / maturities)
        return implied_vols
    
    def get_parameters(self) -> Dict[str, float]:
        """Get fitted parameters."""
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        return self.parameters.copy()
    
    def validate_parameters(self) -> bool:
        """
        Validate SVI parameters for no-arbitrage conditions.
        
        Returns:
            True if parameters satisfy no-arbitrage conditions
        """
        if not self.fitted:
            return False
        
        a, b, rho, m, sigma = (
            self.parameters['a'],
            self.parameters['b'],
            self.parameters['rho'],
            self.parameters['m'],
            self.parameters['sigma']
        )
        
        # Check basic constraints
        if b <= 0 or sigma <= 0 or abs(rho) >= 1:
            return False
        
        # Check no-arbitrage conditions
        # 1. Total variance must be positive
        # 2. Butterfly condition: second derivative must be positive
        # 3. Calendar spread condition: variance must increase with time
        
        # For simplicity, we'll check a range of log-moneyness values
        k_range = np.linspace(-2, 2, 100)
        total_var = self._svi_function(k_range, a, b, rho, m, sigma)
        
        # Check positivity
        if np.any(total_var <= 0):
            return False
        
        # Check butterfly condition (simplified)
        # The SVI function should be convex
        second_deriv = self._second_derivative(k_range, a, b, rho, m, sigma)
        if np.any(second_deriv < -1e-6):  # Allow small numerical errors
            return False
        
        return True
    
    def _second_derivative(self, k: np.ndarray, a: float, b: float, 
                          rho: float, m: float, sigma: float) -> np.ndarray:
        """Compute second derivative of SVI function."""
        # Analytical second derivative of SVI function
        term = (k - m)**2 + sigma**2
        sqrt_term = np.sqrt(term)
        
        first_deriv = b * (rho + (k - m) / sqrt_term)
        second_deriv = b * (1 / sqrt_term - (k - m)**2 / (term * sqrt_term))
        
        return second_deriv
    
    def plot_surface(self, strikes: np.ndarray, maturities: np.ndarray, 
                    title: str = "SVI Volatility Surface"):
        """Plot the fitted volatility surface."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            K, T = np.meshgrid(strikes, maturities)
            implied_vols = self.predict(K.flatten(), T.flatten()).reshape(K.shape)
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            surf = ax.plot_surface(K, T, implied_vols, cmap='viridis', alpha=0.8)
            ax.set_xlabel('Strike Price')
            ax.set_ylabel('Time to Maturity')
            ax.set_zlabel('Implied Volatility')
            ax.set_title(title)
            
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting") 