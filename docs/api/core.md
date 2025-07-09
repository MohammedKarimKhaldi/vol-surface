# Core API Reference

This document provides the complete API reference for the core volatility surface components.

## VolatilitySurface

The main class for volatility surface construction and management.

### Class Definition

```python
class VolatilitySurface:
    """
    Main volatility surface class that integrates SVI modeling, 
    arbitrage detection, and surface interpolation.
    """
```

### Constructor

```python
def __init__(self, 
             svi_model: Optional[SVIModel] = None,
             arbitrage_detector: Optional[ArbitrageDetector] = None,
             spline_interpolator: Optional[SplineInterpolator] = None,
             greeks_calculator: Optional[GreeksCalculator] = None):
    """
    Initialize volatility surface with optional components.
    
    Args:
        svi_model: SVI model instance for parameterization
        arbitrage_detector: Arbitrage detection instance
        spline_interpolator: Spline interpolation instance
        greeks_calculator: Greeks calculation instance
    """
```

### Methods

#### `fit_svi_model`

```python
def fit_svi_model(self, 
                  strikes: np.ndarray, 
                  maturities: np.ndarray, 
                  implied_vols: np.ndarray,
                  weights: Optional[np.ndarray] = None,
                  constraints: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, float]:
    """
    Fit SVI model to implied volatility data.
    
    Args:
        strikes: Strike prices array
        maturities: Time to expiration array
        implied_vols: Implied volatilities array (maturities x strikes)
        weights: Optional weights for fitting
        constraints: Optional parameter constraints
        
    Returns:
        Dictionary of fitted SVI parameters
        
    Raises:
        ValueError: If data dimensions don't match
        RuntimeError: If fitting fails
    """
```

#### `detect_arbitrage`

```python
def detect_arbitrage(self, 
                     tolerance: float = 1e-6,
                     check_calendar: bool = True,
                     check_butterfly: bool = True,
                     check_put_call: bool = True) -> Dict[str, Any]:
    """
    Detect arbitrage opportunities in the volatility surface.
    
    Args:
        tolerance: Numerical tolerance for arbitrage detection
        check_calendar: Whether to check calendar spread arbitrage
        check_butterfly: Whether to check butterfly arbitrage
        check_put_call: Whether to check put-call parity
        
    Returns:
        Dictionary containing arbitrage detection results
        
    Example:
        >>> results = vol_surface.detect_arbitrage()
        >>> print(f"Has arbitrage: {results['has_arbitrage']}")
        >>> print(f"Calendar spread violations: {results['calendar_spread']}")
    """
```

#### `interpolate_surface`

```python
def interpolate_surface(self, 
                       new_strikes: np.ndarray,
                       new_maturities: np.ndarray,
                       method: str = 'spline',
                       **kwargs) -> np.ndarray:
    """
    Interpolate volatility surface to new strikes and maturities.
    
    Args:
        new_strikes: New strike prices for interpolation
        new_maturities: New maturities for interpolation
        method: Interpolation method ('spline', 'svi', 'linear')
        **kwargs: Additional arguments for interpolation
        
    Returns:
        Interpolated volatility surface
        
    Example:
        >>> new_strikes = np.linspace(80, 120, 50)
        >>> new_maturities = np.linspace(0.1, 2.0, 20)
        >>> interpolated_surface = vol_surface.interpolate_surface(
        ...     new_strikes, new_maturities, method='spline'
        ... )
    """
```

#### `calculate_greeks`

```python
def calculate_greeks(self,
                    spot_price: float,
                    risk_free_rate: float,
                    dividend_yield: float = 0.0,
                    option_type: str = 'call') -> Dict[str, np.ndarray]:
    """
    Calculate option Greeks across the volatility surface.
    
    Args:
        spot_price: Current spot price
        risk_free_rate: Risk-free interest rate
        dividend_yield: Dividend yield (default: 0.0)
        option_type: Option type ('call' or 'put')
        
    Returns:
        Dictionary containing Greeks surfaces (delta, gamma, theta, vega, rho)
        
    Example:
        >>> greeks = vol_surface.calculate_greeks(
        ...     spot_price=100.0, risk_free_rate=0.05
        ... )
        >>> delta_surface = greeks['delta']
        >>> gamma_surface = greeks['gamma']
    """
```

#### `get_surface_data`

```python
def get_surface_data(self) -> Dict[str, np.ndarray]:
    """
    Get current volatility surface data.
    
    Returns:
        Dictionary containing surface data
        
    Example:
        >>> data = vol_surface.get_surface_data()
        >>> strikes = data['strikes']
        >>> maturities = data['maturities']
        >>> implied_vols = data['implied_vols']
    """
```

#### `set_surface_data`

```python
def set_surface_data(self,
                    strikes: np.ndarray,
                    maturities: np.ndarray,
                    implied_vols: np.ndarray) -> None:
    """
    Set volatility surface data.
    
    Args:
        strikes: Strike prices array
        maturities: Maturities array
        implied_vols: Implied volatilities array
        
    Raises:
        ValueError: If data dimensions don't match
    """
```

#### `export_surface`

```python
def export_surface(self,
                  filename: str,
                  format: str = 'csv',
                  include_greeks: bool = False,
                  **kwargs) -> None:
    """
    Export volatility surface to file.
    
    Args:
        filename: Output filename
        format: Export format ('csv', 'json', 'pickle')
        include_greeks: Whether to include Greeks data
        **kwargs: Additional export options
        
    Example:
        >>> vol_surface.export_surface('surface_data.csv', format='csv')
        >>> vol_surface.export_surface('surface_data.json', format='json')
    """
```

#### `import_surface`

```python
def import_surface(self,
                  filename: str,
                  format: str = 'csv',
                  **kwargs) -> None:
    """
    Import volatility surface from file.
    
    Args:
        filename: Input filename
        format: Import format ('csv', 'json', 'pickle')
        **kwargs: Additional import options
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
```

### Properties

#### `is_fitted`

```python
@property
def is_fitted(self) -> bool:
    """
    Check if SVI model is fitted.
    
    Returns:
        True if model is fitted, False otherwise
    """
```

#### `parameters`

```python
@property
def parameters(self) -> Optional[Dict[str, float]]:
    """
    Get fitted SVI parameters.
    
    Returns:
        Dictionary of SVI parameters or None if not fitted
    """
```

#### `surface_shape`

```python
@property
def surface_shape(self) -> Tuple[int, int]:
    """
    Get volatility surface dimensions.
    
    Returns:
        Tuple of (n_maturities, n_strikes)
    """
```

## SVIModel

SVI model for volatility surface parameterization.

### Class Definition

```python
class SVIModel:
    """
    Stochastic Volatility Inspired (SVI) model for volatility surface fitting.
    """
```

### Constructor

```python
def __init__(self, constraints: Optional[Dict[str, Tuple[float, float]]] = None):
    """
    Initialize SVI model.
    
    Args:
        constraints: Optional parameter constraints
    """
```

### Methods

#### `fit`

```python
def fit(self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray,
        weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Fit SVI model to implied volatility data.
    
    Args:
        strikes: Strike prices
        maturities: Time to expiration
        implied_vols: Implied volatilities
        weights: Optional weights for fitting
        
    Returns:
        Fitted parameters dictionary
    """
```

#### `predict`

```python
def predict(self,
           strikes: np.ndarray,
           maturities: np.ndarray) -> np.ndarray:
    """
    Predict implied volatilities using fitted model.
    
    Args:
        strikes: Strike prices
        maturities: Time to expiration
        
    Returns:
        Predicted implied volatilities
    """
```

#### `validate_parameters`

```python
def validate_parameters(self, params: Optional[Dict[str, float]] = None) -> bool:
    """
    Validate SVI parameters for no-arbitrage conditions.
    
    Args:
        params: Parameters to validate (uses fitted params if None)
        
    Returns:
        True if parameters are valid
    """
```

## ArbitrageDetector

Arbitrage detection algorithms.

### Class Definition

```python
class ArbitrageDetector:
    """
    Detect arbitrage opportunities in volatility surfaces.
    """
```

### Methods

#### `detect_all_arbitrage`

```python
def detect_all_arbitrage(self,
                        vol_surface: np.ndarray,
                        strikes: Optional[np.ndarray] = None,
                        maturities: Optional[np.ndarray] = None,
                        tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Detect all types of arbitrage in volatility surface.
    
    Args:
        vol_surface: Volatility surface data
        strikes: Strike prices (optional)
        maturities: Maturities (optional)
        tolerance: Numerical tolerance
        
    Returns:
        Dictionary with arbitrage detection results
    """
```

#### `detect_calendar_spread`

```python
def detect_calendar_spread(self,
                          vol_surface: np.ndarray,
                          tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Detect calendar spread arbitrage.
    
    Args:
        vol_surface: Volatility surface data
        tolerance: Numerical tolerance
        
    Returns:
        Calendar spread arbitrage results
    """
```

#### `detect_butterfly`

```python
def detect_butterfly(self,
                    vol_surface: np.ndarray,
                    tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Detect butterfly arbitrage.
    
    Args:
        vol_surface: Volatility surface data
        tolerance: Numerical tolerance
        
    Returns:
        Butterfly arbitrage results
    """
```

## SplineInterpolator

Cubic spline interpolation for volatility surfaces.

### Class Definition

```python
class SplineInterpolator:
    """
    Cubic spline interpolation for volatility surfaces.
    """
```

### Methods

#### `fit`

```python
def fit(self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray,
        tension: float = 0.1) -> None:
    """
    Fit spline interpolator to data.
    
    Args:
        strikes: Strike prices
        maturities: Maturities
        implied_vols: Implied volatilities
        tension: Spline tension parameter
    """
```

#### `interpolate`

```python
def interpolate(self,
               new_strikes: np.ndarray,
               new_maturities: np.ndarray) -> np.ndarray:
    """
    Interpolate to new strikes and maturities.
    
    Args:
        new_strikes: New strike prices
        new_maturities: New maturities
        
    Returns:
        Interpolated volatility surface
    """
```

## Error Classes

### VolatilitySurfaceError

```python
class VolatilitySurfaceError(Exception):
    """Base exception for volatility surface operations."""
    pass
```

### ArbitrageError

```python
class ArbitrageError(VolatilitySurfaceError):
    """Exception raised when arbitrage is detected."""
    pass
```

### FittingError

```python
class FittingError(VolatilitySurfaceError):
    """Exception raised when model fitting fails."""
    pass
```

## Utility Functions

### `validate_surface_data`

```python
def validate_surface_data(strikes: np.ndarray,
                         maturities: np.ndarray,
                         implied_vols: np.ndarray) -> bool:
    """
    Validate surface data dimensions and values.
    
    Args:
        strikes: Strike prices
        maturities: Maturities
        implied_vols: Implied volatilities
        
    Returns:
        True if data is valid
        
    Raises:
        ValueError: If data is invalid
    """
```

### `convert_to_log_moneyness`

```python
def convert_to_log_moneyness(strikes: np.ndarray,
                           spot_price: float) -> np.ndarray:
    """
    Convert strikes to log-moneyness.
    
    Args:
        strikes: Strike prices
        spot_price: Current spot price
        
    Returns:
        Log-moneyness array
    """
```

### `convert_to_total_variance`

```python
def convert_to_total_variance(implied_vols: np.ndarray,
                            maturities: np.ndarray) -> np.ndarray:
    """
    Convert implied volatilities to total variance.
    
    Args:
        implied_vols: Implied volatilities
        maturities: Time to expiration
        
    Returns:
        Total variance array
    """
```

## Configuration

### Default Settings

```python
DEFAULT_SETTINGS = {
    'svi': {
        'max_iterations': 1000,
        'tolerance': 1e-8,
        'method': 'L-BFGS-B'
    },
    'arbitrage': {
        'tolerance': 1e-6,
        'check_calendar': True,
        'check_butterfly': True,
        'check_put_call': True
    },
    'spline': {
        'tension': 0.1,
        'smoothing': 0.0
    }
}
```

## Examples

### Complete Workflow

```python
from src.core.volatility_surface import VolatilitySurface
import numpy as np

# Initialize
vol_surface = VolatilitySurface()

# Generate sample data
strikes = np.linspace(80, 120, 20)
maturities = np.array([0.1, 0.25, 0.5, 1.0, 2.0])
implied_vols = np.random.uniform(0.15, 0.35, (len(maturities), len(strikes)))

# Fit SVI model
params = vol_surface.fit_svi_model(strikes, maturities, implied_vols)

# Detect arbitrage
arbitrage_results = vol_surface.detect_arbitrage()

# Interpolate to finer grid
new_strikes = np.linspace(80, 120, 100)
new_maturities = np.linspace(0.1, 2.0, 50)
interpolated_surface = vol_surface.interpolate_surface(
    new_strikes, new_maturities, method='spline'
)

# Calculate Greeks
greeks = vol_surface.calculate_greeks(
    spot_price=100.0, risk_free_rate=0.05
)

# Export results
vol_surface.export_surface('results.csv', format='csv')
```

---

*Next: [SVI API](svi.md) or [Arbitrage API](arbitrage.md)* 