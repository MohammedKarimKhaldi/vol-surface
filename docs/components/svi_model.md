# SVI Model Documentation

The Stochastic Volatility Inspired (SVI) model is a powerful parameterization for volatility surfaces that ensures no-arbitrage conditions and provides excellent fit to market data.

## 📐 Mathematical Foundation

### SVI Formula

The SVI model parameterizes the total variance as:

```
σ²(k,T) = a + b[ρ(k-m) + √((k-m)² + σ²)]
```

Where:
- `k` = log-moneyness = ln(K/S)
- `T` = time to expiration
- `σ²(k,T)` = total variance at log-moneyness k and time T

### Parameters

| Parameter | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| `a` | a | ℝ | Vertical offset (level) |
| `b` | b | ℝ⁺ | Slope parameter |
| `ρ` | ρ | [-1, 1] | Correlation (skew) |
| `m` | m | ℝ | Horizontal offset (location) |
| `σ` | σ | ℝ⁺ | Volatility of volatility (curvature) |

### Parameter Interpretation

#### Parameter `a` (Level)
- Controls the overall level of variance
- Larger values → higher overall volatility
- Affects the vertical position of the smile

#### Parameter `b` (Slope)
- Controls the slope of the volatility smile
- Larger values → steeper smile
- Must be positive for no-arbitrage

#### Parameter `ρ` (Skew)
- Controls the skewness of the smile
- ρ > 0 → positive skew (higher vol for calls)
- ρ < 0 → negative skew (higher vol for puts)
- ρ = 0 → symmetric smile

#### Parameter `m` (Location)
- Controls the location of the smile center
- Shifts the smile horizontally
- m = 0 → smile centered at ATM

#### Parameter `σ` (Curvature)
- Controls the curvature of the smile
- Larger values → more curved smile
- Must be positive for no-arbitrage

## 🔧 Usage

### Basic Usage

```python
from src.core.svi_model import SVIModel
import numpy as np

# Initialize SVI model
svi = SVIModel()

# Sample data
strikes = np.array([90, 95, 100, 105, 110])
maturities = np.array([0.25, 0.5, 1.0])
implied_vols = np.random.uniform(0.15, 0.35, (len(maturities), len(strikes)))

# Fit the model
params = svi.fit(strikes, maturities, implied_vols)
print(f"Fitted parameters: {params}")

# Predict implied volatilities
predicted_vols = svi.predict(strikes, maturities)
```

### Advanced Usage

```python
# Custom constraints
constraints = {
    'a': (0.01, 0.1),      # Constrain level
    'b': (0.1, 1.0),       # Constrain slope
    'rho': (-0.5, 0.5),    # Constrain skew
    'm': (-0.5, 0.5),      # Constrain location
    'sigma': (0.05, 0.3)   # Constrain curvature
}

svi = SVIModel(constraints=constraints)
params = svi.fit(strikes, maturities, implied_vols)
```

### Parameter Validation

```python
# Check if parameters satisfy no-arbitrage conditions
is_valid = svi.validate_parameters()
print(f"Parameters are arbitrage-free: {is_valid}")

# Get parameter bounds
bounds = svi.get_parameter_bounds()
print(f"Parameter bounds: {bounds}")
```

## 📊 Visualization

### Plot SVI Surface

```python
import matplotlib.pyplot as plt

# Create surface plot
fig, ax = plt.subplots(figsize=(12, 8))
svi.plot_surface(strikes, maturities, title="SVI Volatility Surface")
plt.show()
```

### Compare with Market Data

```python
# Plot fitted vs market data
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, maturity in enumerate(maturities):
    axes[i].plot(strikes, implied_vols[i], 'o', label='Market')
    axes[i].plot(strikes, predicted_vols[i], '-', label='SVI Fit')
    axes[i].set_title(f'Maturity: {maturity:.2f}')
    axes[i].legend()
    axes[i].set_xlabel('Strike')
    axes[i].set_ylabel('Implied Volatility')

plt.tight_layout()
plt.show()
```

## ⚠️ No-Arbitrage Conditions

### Mathematical Conditions

The SVI model satisfies no-arbitrage conditions when:

1. **Butterfly Condition**: Second derivative must be positive
   ```
   ∂²σ²/∂k² ≥ 0
   ```

2. **Calendar Spread Condition**: Variance must increase with time
   ```
   ∂σ²/∂T ≥ 0
   ```

3. **Parameter Constraints**:
   - `b > 0` (slope must be positive)
   - `σ > 0` (curvature must be positive)
   - `|ρ| < 1` (correlation must be in [-1, 1])

### Validation Methods

```python
# Check all no-arbitrage conditions
validation_results = svi.validate_all_conditions()

print("No-Arbitrage Validation:")
print(f"  Butterfly condition: {validation_results['butterfly']}")
print(f"  Calendar spread: {validation_results['calendar_spread']}")
print(f"  Parameter bounds: {validation_results['parameter_bounds']}")
```

## 🔬 Advanced Features

### Custom Objective Functions

```python
# Define custom loss function
def custom_loss(params, log_moneyness, total_variance, weights=None):
    a, b, rho, m, sigma = params
    fitted_variance = svi._svi_function(log_moneyness, a, b, rho, m, sigma)
    
    # MSE with optional weights
    if weights is not None:
        residuals = weights * (fitted_variance - total_variance) ** 2
    else:
        residuals = (fitted_variance - total_variance) ** 2
    
    return np.sum(residuals)

# Use custom loss in fitting
svi.fit_with_custom_loss(strikes, maturities, implied_vols, custom_loss)
```

### Multi-Maturity Fitting

```python
# Fit SVI parameters for each maturity separately
maturity_params = {}

for i, maturity in enumerate(maturities):
    svi_temp = SVIModel()
    params = svi_temp.fit(strikes, implied_vols[i], [maturity])
    maturity_params[maturity] = params

print("Parameters by maturity:")
for maturity, params in maturity_params.items():
    print(f"  {maturity:.2f}: {params}")
```

### Parameter Smoothing

```python
# Smooth parameters across maturities
from scipy.interpolate import interp1d

# Extract parameter for smoothing
a_values = [params['a'] for params in maturity_params.values()]
maturities_array = list(maturity_params.keys())

# Create smooth interpolation
a_smooth = interp1d(maturities_array, a_values, kind='cubic')

# Use smoothed parameters
new_maturity = 0.75
smoothed_a = a_smooth(new_maturity)
```

## 📈 Performance Optimization

### Numba JIT Compilation

The SVI model uses Numba JIT compilation for optimal performance:

```python
# Check if Numba is available
if svi.NUMBA_AVAILABLE:
    print("Numba JIT compilation enabled")
else:
    print("Numba not available, using standard Python")

# Performance comparison
import time

# Time SVI function calls
start_time = time.time()
for _ in range(1000):
    svi._svi_function(log_moneyness, a, b, rho, m, sigma)
end_time = time.time()

print(f"1000 SVI evaluations: {end_time - start_time:.4f} seconds")
```

### Vectorized Operations

```python
# Vectorized parameter fitting
def vectorized_svi_fit(strikes_array, maturities_array, implied_vols_array):
    """Fit SVI model to multiple datasets efficiently."""
    results = []
    
    for strikes, maturities, implied_vols in zip(strikes_array, maturities_array, implied_vols_array):
        svi = SVIModel()
        params = svi.fit(strikes, maturities, implied_vols)
        results.append(params)
    
    return results
```

## 🧪 Testing

### Unit Tests

```python
import pytest

def test_svi_parameters():
    """Test SVI parameter validation."""
    svi = SVIModel()
    
    # Test valid parameters
    valid_params = {'a': 0.04, 'b': 0.4, 'rho': -0.1, 'm': 0.0, 'sigma': 0.1}
    assert svi.validate_parameters(valid_params) == True
    
    # Test invalid parameters
    invalid_params = {'a': 0.04, 'b': -0.4, 'rho': -0.1, 'm': 0.0, 'sigma': 0.1}
    assert svi.validate_parameters(invalid_params) == False

def test_svi_fitting():
    """Test SVI model fitting."""
    svi = SVIModel()
    
    # Generate test data
    strikes = np.array([90, 100, 110])
    maturities = np.array([0.25, 0.5])
    implied_vols = np.array([[0.2, 0.18, 0.22], [0.22, 0.2, 0.24]])
    
    # Fit model
    params = svi.fit(strikes, maturities, implied_vols)
    
    # Check that parameters are reasonable
    assert 0 < params['b'] < 2
    assert -1 < params['rho'] < 1
    assert params['sigma'] > 0
```

## 📚 References

1. **Gatheral, J. (2004)**: "A Parsimonious Arbitrage-Free Implied Volatility Parameterization"
2. **Gatheral, J. & Jacquier, A. (2014)**: "Arbitrage-Free SVI Volatility Surfaces"
3. **Zeliade Systems (2009)**: "Quasi-Explicit Calibration of Gatheral's SVI Model"

## 🔗 Related Components

- [Variance Swap Calculator](variance_swap.md) - Model-free variance calculations
- [Arbitrage Detection](arbitrage_detection.md) - No-arbitrage validation
- [Spline Interpolation](spline_interpolation.md) - Alternative surface interpolation
- [Greeks Calculator](greeks_calculator.md) - Option sensitivities

---

*Next: [Variance Swap Calculator](variance_swap.md) or [Arbitrage Detection](arbitrage_detection.md)* 