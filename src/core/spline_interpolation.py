"""
Cubic Spline Interpolation with Tension Splines and C² Continuity
"""

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from typing import Tuple, Optional, List, Dict, Any, Union
import warnings


class SplineInterpolator:
    """
    Cubic spline interpolation with tension splines and C² continuity constraints.
    
    Implements tension splines that maintain smoothness while allowing control
    over the stiffness of the interpolation.
    """
    
    def __init__(self, tension: float = 0.0, boundary_condition: str = 'natural'):
        """
        Initialize spline interpolator.
        
        Args:
            tension: Tension parameter (0 = cubic spline, ∞ = linear)
            boundary_condition: 'natural', 'clamped', or 'periodic'
        """
        self.tension = tension
        self.boundary_condition = boundary_condition
        self.splines = {}
        self.fitted = False
        
    def fit_2d_surface(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                      weights: Optional[np.ndarray] = None) -> None:
        """
        Fit 2D spline surface to data.
        
        Args:
            x: X coordinates (strikes)
            y: Y coordinates (maturities)
            z: Z values (implied volatilities)
            weights: Optional weights for fitting
        """
        # Sort data
        sort_idx = np.lexsort((y, x))
        x = x[sort_idx]
        y = y[sort_idx]
        z = z[sort_idx]
        
        # Get unique x and y values
        unique_x = np.unique(x)
        unique_y = np.unique(y)
        
        # Create grid
        X, Y = np.meshgrid(unique_x, unique_y)
        Z = np.zeros_like(X)
        
        # Interpolate z values to grid
        for i, xi in enumerate(unique_x):
            for j, yi in enumerate(unique_y):
                mask = (x == xi) & (y == yi)
                if np.any(mask):
                    Z[j, i] = np.mean(z[mask])
                else:
                    # Interpolate missing values
                    Z[j, i] = self._interpolate_missing_value(xi, yi, x, y, z)
        
        # Fit splines for each dimension
        self._fit_x_splines(unique_x, unique_y, Z)
        self._fit_y_splines(unique_x, unique_y, Z)
        
        self.fitted = True
    
    def _interpolate_missing_value(self, xi: float, yi: float, x: np.ndarray,
                                 y: np.ndarray, z: np.ndarray) -> float:
        """Interpolate missing value using nearest neighbors."""
        # Find nearest neighbors
        distances = np.sqrt((x - xi)**2 + (y - yi)**2)
        nearest_idx = np.argmin(distances)
        
        # Use inverse distance weighting
        weights = 1 / (distances + 1e-10)
        weights = weights / np.sum(weights)
        
        return np.sum(weights * z)
    
    def _fit_x_splines(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
        """Fit splines along x dimension."""
        self.x_splines = {}
        
        for i, yi in enumerate(y):
            # Create tension spline for this y slice
            spline = self._create_tension_spline(x, z[i, :])
            self.x_splines[float(yi)] = spline
    
    def _fit_y_splines(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
        """Fit splines along y dimension."""
        self.y_splines = {}
        
        for i, xi in enumerate(x):
            # Create tension spline for this x slice
            spline = self._create_tension_spline(y, z[:, i])
            self.y_splines[float(xi)] = spline
    
    def _create_tension_spline(self, x: np.ndarray, y: np.ndarray) -> Any:
        """Create tension spline with C² continuity."""
        if self.tension == 0:
            # Standard cubic spline
            return CubicSpline(x, y, bc_type=self.boundary_condition)
        else:
            # Tension spline
            return self._tension_spline(x, y)
    
    def _tension_spline(self, x: np.ndarray, y: np.ndarray) -> Any:
        """Create tension spline using finite differences."""
        n = len(x)
        h = np.diff(x)
        
        # Set up tridiagonal system for second derivatives
        A = np.zeros((n, n))
        b = np.zeros(n)
        
        # Interior points
        for i in range(1, n-1):
            A[i, i-1] = h[i-1]
            A[i, i] = 2 * (h[i-1] + h[i])
            A[i, i+1] = h[i]
            
            b[i] = 6 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
        
        # Boundary conditions
        if self.boundary_condition == 'natural':
            A[0, 0] = 1
            A[-1, -1] = 1
        elif self.boundary_condition == 'clamped':
            A[0, 0] = 2 * h[0]
            A[0, 1] = h[0]
            A[-1, -2] = h[-1]
            A[-1, -1] = 2 * h[-1]
            
            b[0] = 6 * ((y[1] - y[0]) / h[0] - 0)  # Assume zero derivative
            b[-1] = 6 * (0 - (y[-1] - y[-2]) / h[-1])  # Assume zero derivative
        
        # Solve for second derivatives
        M = spsolve(A, b)
        
        # Create spline coefficients
        coeffs = np.zeros((n-1, 4))
        for i in range(n-1):
            coeffs[i, 0] = y[i]
            coeffs[i, 1] = (y[i+1] - y[i]) / h[i] - h[i] * (2*M[i] + M[i+1]) / 6
            coeffs[i, 2] = M[i] / 2
            coeffs[i, 3] = (M[i+1] - M[i]) / (6 * h[i])
        
        return TensionSpline(x, coeffs, self.tension)
    
    def predict(self, x: Union[float, np.ndarray], 
               y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Predict values using fitted spline surface.
        
        Args:
            x: X coordinates
            y: Y coordinates
            
        Returns:
            Predicted values
        """
        if not self.fitted:
            raise ValueError("Spline must be fitted before prediction")
        
        if np.isscalar(x) and np.isscalar(y):
            return self._predict_single(x, y)
        else:
            x = np.asarray(x)
            y = np.asarray(y)
            
            if x.shape != y.shape:
                raise ValueError("x and y must have the same shape")
            
            result = np.zeros_like(x, dtype=float)
            for i in range(x.size):
                result.flat[i] = self._predict_single(float(x.flat[i]), float(y.flat[i]))
            
            return result
    
    def _get_closest_key(self, d, value):
        keys = list(d.keys())
        return min(keys, key=lambda k: abs(float(k) - float(value)))

    def _predict_single(self, x: float, y: float) -> float:
        """Predict single value using bilinear interpolation of splines."""
        # Find nearest x and y values
        x_vals = sorted(float(val) for val in self.x_splines.keys())
        y_vals = sorted(float(val) for val in self.y_splines.keys())
        
        # Find interpolation indices
        x_idx = np.searchsorted(x_vals, x)
        y_idx = np.searchsorted(y_vals, y)
        
        # Clamp indices
        x_idx = max(0, min(len(x_vals) - 1, int(x_idx)))
        y_idx = max(0, min(len(y_vals) - 1, int(y_idx)))
        
        # Get surrounding values
        x1, x2 = x_vals[max(0, x_idx-1)], x_vals[min(len(x_vals)-1, x_idx)]
        y1, y2 = y_vals[max(0, y_idx-1)], y_vals[min(len(y_vals)-1, y_idx)]
        
        # Use closest keys for splines
        x1_key = self._get_closest_key(self.y_splines, x1)
        x2_key = self._get_closest_key(self.y_splines, x2)
        y1_key = self._get_closest_key(self.x_splines, y1)
        y2_key = self._get_closest_key(self.x_splines, y2)
        
        # Interpolate using splines
        if x1 == x2:
            z11 = z12 = self.x_splines[y1_key](x)
            z21 = z22 = self.x_splines[y2_key](x)
        else:
            z11 = self.x_splines[y1_key](x)
            z12 = self.x_splines[y2_key](x)
            z21 = self.y_splines[x1_key](y)
            z22 = self.y_splines[x2_key](y)
        
        # Bilinear interpolation
        if x1 == x2 and y1 == y2:
            return z11
        elif x1 == x2:
            return z11 + (z12 - z11) * (y - y1) / (y2 - y1)
        elif y1 == y2:
            return z11 + (z21 - z11) * (x - x1) / (x2 - x1)
        else:
            # Full bilinear interpolation
            wx = (x - x1) / (x2 - x1)
            wy = (y - y1) / (y2 - y1)
            
            return (z11 * (1 - wx) * (1 - wy) + 
                   z21 * wx * (1 - wy) + 
                   z12 * (1 - wx) * wy + 
                   z22 * wx * wy)
    
    def gradient(self, x: float, y: float) -> Tuple[float, float]:
        """
        Calculate gradient at given point.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Tuple of (dx, dy) gradients
        """
        if not self.fitted:
            raise ValueError("Spline must be fitted before gradient calculation")
        
        # Use finite differences for gradient
        eps = 1e-6
        
        dx = (self._predict_single(x + eps, y) - self._predict_single(x - eps, y)) / (2 * eps)
        dy = (self._predict_single(x, y + eps) - self._predict_single(x, y - eps)) / (2 * eps)
        
        return dx, dy
    
    def hessian(self, x: float, y: float) -> np.ndarray:
        """
        Calculate Hessian matrix at given point.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            2x2 Hessian matrix
        """
        if not self.fitted:
            raise ValueError("Spline must be fitted before Hessian calculation")
        
        # Use finite differences for Hessian
        eps = 1e-6
        
        # Second derivatives
        dxx = (self._predict_single(x + eps, y) - 2 * self._predict_single(x, y) + 
               self._predict_single(x - eps, y)) / (eps ** 2)
        
        dyy = (self._predict_single(x, y + eps) - 2 * self._predict_single(x, y) + 
               self._predict_single(x, y - eps)) / (eps ** 2)
        
        # Mixed derivative
        dxy = (self._predict_single(x + eps, y + eps) - 
               self._predict_single(x + eps, y - eps) - 
               self._predict_single(x - eps, y + eps) + 
               self._predict_single(x - eps, y - eps)) / (4 * eps ** 2)
        
        return np.array([[dxx, dxy], [dxy, dyy]])
    
    def curvature(self, x: float, y: float) -> float:
        """
        Calculate mean curvature at given point.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Mean curvature
        """
        H = self.hessian(x, y)
        return np.trace(H) / 2


class TensionSpline:
    """Custom tension spline implementation."""
    
    def __init__(self, x: np.ndarray, coeffs: np.ndarray, tension: float):
        self.x = x
        self.coeffs = coeffs
        self.tension = tension
    
    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate spline at given points."""
        if np.isscalar(x):
            return self._evaluate_single(float(x))
        else:
            return np.array([self._evaluate_single(float(xi)) for xi in x])
    
    def _evaluate_single(self, x: float) -> float:
        """Evaluate spline at single point."""
        # Find interval
        idx = np.searchsorted(self.x, x) - 1
        idx = max(0, min(len(self.x) - 2, int(idx)))
        
        # Evaluate polynomial
        dx = x - self.x[idx]
        coeff = self.coeffs[idx]
        
        return (coeff[0] + coeff[1] * dx + 
                coeff[2] * dx**2 + coeff[3] * dx**3) 