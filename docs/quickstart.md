# Quick Start Guide

Get up and running with the Volatility Surface Construction & Arbitrage Detection library in minutes.

## 🚀 5-Minute Setup

### 1. Install the Library
```bash
# Clone and install
git clone https://github.com/MohammedKarimKhaldi/vol-surface.git
cd vol-surface
pip install -r requirements.txt
pip install -e .
```

### 2. Basic Usage
```python
import numpy as np
from src.core.volatility_surface import VolatilitySurface

# Create volatility surface
vol_surface = VolatilitySurface()

# Generate sample data
strikes = np.linspace(80, 120, 20)
expiries = np.array([0.1, 0.25, 0.5, 1.0, 2.0])
implied_vols = np.random.uniform(0.15, 0.35, (len(expiries), len(strikes)))

# Fit SVI model
vol_surface.fit_svi_model(strikes, expiries, implied_vols)

# Detect arbitrage
results = vol_surface.detect_arbitrage()
print(f"Arbitrage detected: {results['has_arbitrage']}")
```

### 3. Run the Dashboard
```bash
python examples/dashboard.py
# Open http://127.0.0.1:8050 in your browser
```

## 📊 Basic Examples

### Example 1: SVI Model Fitting

```python
from src.core.svi_model import SVIModel
import numpy as np

# Initialize SVI model
svi = SVIModel()

# Sample option data
strikes = np.array([90, 95, 100, 105, 110])
maturities = np.array([0.25, 0.5, 1.0])
implied_vols = np.random.uniform(0.15, 0.35, (len(maturities), len(strikes)))

# Fit the model
params = svi.fit(strikes, maturities, implied_vols)
print("Fitted SVI parameters:")
for key, value in params.items():
    print(f"  {key}: {value:.4f}")

# Predict implied volatilities
predicted_vols = svi.predict(strikes, maturities)
print(f"Prediction RMSE: {np.sqrt(np.mean((implied_vols - predicted_vols)**2)):.4f}")
```

### Example 2: Arbitrage Detection

```python
from src.core.arbitrage_detection import ArbitrageDetector
import numpy as np

# Initialize detector
detector = ArbitrageDetector()

# Sample volatility surface data
vol_surface_data = np.random.uniform(0.1, 0.4, (5, 10))  # 5 expiries, 10 strikes

# Check for arbitrage
results = detector.detect_all_arbitrage(vol_surface_data)

print("Arbitrage Detection Results:")
print(f"  Calendar spread arbitrage: {results['calendar_spread']}")
print(f"  Butterfly arbitrage: {results['butterfly']}")
print(f"  Put-call parity violations: {results['put_call_parity']}")

if results['has_arbitrage']:
    print("⚠️  Arbitrage opportunities detected!")
else:
    print("✅ No arbitrage detected")
```

### Example 3: Greeks Calculation

```python
from src.greeks.sensitivities import GreeksCalculator
import numpy as np

# Initialize calculator
calculator = GreeksCalculator()

# Option parameters
S = 100.0  # Spot price
K = 100.0  # Strike price
T = 1.0    # Time to expiry
r = 0.05   # Risk-free rate
sigma = 0.2  # Volatility
option_type = 'call'

# Calculate Greeks
greeks = calculator.calculate_greeks(S, K, T, r, sigma, option_type)

print("Option Greeks:")
print(f"  Delta: {greeks['delta']:.4f}")
print(f"  Gamma: {greeks['gamma']:.4f}")
print(f"  Theta: {greeks['theta']:.4f}")
print(f"  Vega:  {greeks['vega']:.4f}")
print(f"  Rho:   {greeks['rho']:.4f}")
```

### Example 4: Variance Swap Pricing

```python
from src.core.variance_swap import VarianceSwapCalculator
import numpy as np

# Initialize calculator
calculator = VarianceSwapCalculator()

# Market data
spot_price = 100.0
risk_free_rate = 0.05
dividend_yield = 0.02
volatility_surface = np.random.uniform(0.15, 0.35, (5, 10))

# Calculate variance swap fair value
fair_value = calculator.calculate_fair_value(
    spot_price, risk_free_rate, dividend_yield, volatility_surface
)

print(f"Variance Swap Fair Value: {fair_value:.4f}")
```

## 🎛️ Interactive Dashboard

### Starting the Dashboard
```bash
# Run the interactive dashboard
python examples/dashboard.py
```

### Dashboard Features
- **Real-time SVI parameter adjustment** with sliders
- **3D surface plots** and **2D contour maps**
- **Live arbitrage detection** with visual alerts
- **Surface statistics** and metrics
- **Export capabilities** for plots and data

### Dashboard Controls
1. **SVI Parameters**:
   - `a` (level): Controls overall variance level
   - `b` (slope): Controls smile slope
   - `rho` (skew): Controls skewness
   - `m` (location): Controls smile location
   - `sigma` (curvature): Controls curvature

2. **View Options**:
   - Switch between 3D surface and 2D contour plots
   - Adjust camera angles and zoom levels
   - Export plots as PNG or PDF

3. **Arbitrage Monitoring**:
   - Real-time detection of calendar spread arbitrage
   - Butterfly arbitrage detection
   - Put-call parity violation alerts

## 📈 Visualization Examples

### Generate Static Plots
```bash
# Generate all visualizations
python examples/generate_visuals.py
```

This creates:
- SVI volatility surface plots
- Spline interpolation surfaces
- Greeks surfaces (Delta, Gamma, Theta, Vega, Rho)
- Animated GIF showing SVI surface evolution

### Custom Plotting
```python
import matplotlib.pyplot as plt
from src.core.svi_model import SVIModel

# Create and fit SVI model
svi = SVIModel()
# ... fit model with your data ...

# Plot the surface
fig, ax = plt.subplots(figsize=(10, 8))
svi.plot_surface(strikes, maturities, title="Custom SVI Surface")
plt.show()
```

## 🔍 Common Use Cases

### 1. Market Data Analysis
```python
# Load your market data
import pandas as pd

# Assuming you have option data in a DataFrame
data = pd.read_csv('option_data.csv')
strikes = data['strike'].values
expiries = data['expiry'].values
implied_vols = data['implied_vol'].values

# Fit SVI model
vol_surface = VolatilitySurface()
vol_surface.fit_svi_model(strikes, expiries, implied_vols)
```

### 2. Risk Management
```python
# Monitor arbitrage in real-time
from src.realtime.monitoring import RealTimeMonitor

monitor = RealTimeMonitor()
monitor.start_monitoring(
    symbols=['SPY', 'QQQ', 'IWM'],
    check_interval=60  # Check every minute
)
```

### 3. Research and Development
```python
# Test different SVI parameterizations
svi_models = []
for rho in [-0.5, -0.25, 0, 0.25, 0.5]:
    svi = SVIModel()
    # ... fit with different parameters ...
    svi_models.append(svi)
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # Make sure you're in the correct directory
   import sys
   sys.path.append('.')
   ```

2. **Dashboard Not Starting**
   ```bash
   # Check dependencies
   pip install dash plotly
   # Try different port
   python examples/dashboard.py --port 8051
   ```

3. **Memory Issues**
   ```python
   # Use smaller datasets for testing
   strikes = np.linspace(80, 120, 10)  # Reduce from 20 to 10
   ```

## 📚 Next Steps

Now that you're up and running:

1. **Explore the API**: Check the [API Reference](api/core.md)
2. **Learn about SVI**: Read [SVI Model Documentation](components/svi_model.md)
3. **Understand Arbitrage**: Study [Arbitrage Detection](components/arbitrage_detection.md)
4. **Advanced Usage**: Explore [Advanced Topics](advanced/performance.md)

## 💡 Tips

- **Start small**: Use small datasets for initial testing
- **Use the dashboard**: Great for parameter exploration
- **Check arbitrage**: Always validate your surfaces
- **Monitor performance**: Use Numba for large datasets
- **Save results**: Export your fitted models and results

---

*Next: [Core Components](components/svi_model.md) or [API Reference](api/core.md)* 