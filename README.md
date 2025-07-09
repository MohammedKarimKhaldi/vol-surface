# Volatility Surface Construction & Arbitrage Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-MohammedKarimKhaldi-green.svg)](https://github.com/MohammedKarimKhaldi)

A comprehensive Python library for volatility surface construction, SVI parameterization, and real-time arbitrage detection in options markets.

## 🎯 Features

### Core Functionality
- **SVI (Stochastic Volatility Inspired) Model** - Advanced volatility surface parameterization
- **Variance Swap Calculations** - Complete variance swap pricing and risk management
- **Cubic Spline Interpolation** - Smooth volatility surface interpolation
- **Real-time Arbitrage Detection** - Calendar spread, butterfly, and put-call parity violations
- **Greeks Calculation Engine** - Delta, Gamma, Theta, Vega, and Rho calculations
- **Interactive Dashboard** - Real-time visualization and parameter adjustment

### Arbitrage Detection Algorithms
- **Calendar Spread Arbitrage** - Detects violations in time-based spreads
- **Butterfly Arbitrage** - Identifies convexity violations in strike spreads
- **Put-Call Parity Violations** - Monitors synthetic position mispricing

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Quick Install
```bash
# Clone the repository
git clone https://github.com/MohammedKarimKhaldi/vol-surface.git
cd vol-surface

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Manual Installation
```bash
# Install core dependencies
pip install numpy scipy pandas matplotlib seaborn plotly dash

# Install optional dependencies for enhanced performance
pip install numba scikit-learn
```

## 🚀 Quick Start

### Basic Usage
```python
from src.core.volatility_surface import VolatilitySurface
import numpy as np

# Initialize volatility surface
vol_surface = VolatilitySurface()

# Generate sample data
strikes = np.linspace(80, 120, 20)
expiries = np.array([0.1, 0.25, 0.5, 1.0, 2.0])

# Fit SVI model
vol_surface.fit_svi_model(strikes, expiries, implied_vols)

# Detect arbitrage
arbitrage_results = vol_surface.detect_arbitrage()
print(f"Arbitrage detected: {arbitrage_results['has_arbitrage']}")
```

### Interactive Dashboard
```bash
# Run the interactive dashboard
python examples/dashboard.py

# Access at: http://127.0.0.1:8050
```

## 📊 Visualization

### Static Visualizations
```bash
# Generate static plots and animated GIFs
python examples/generate_visuals.py
```

### Interactive Dashboard Features
- **Real-time SVI parameter adjustment**
- **3D surface plots and 2D contour maps**
- **Live arbitrage detection alerts**
- **Surface statistics and metrics**

## 🏗️ Project Structure

```
vol-surface/
├── src/                          # Core source code
│   ├── core/                     # Core volatility surface components
│   │   ├── volatility_surface.py # Main volatility surface class
│   │   ├── svi_model.py         # SVI parameterization
│   │   ├── variance_swap.py     # Variance swap calculations
│   │   └── spline_interpolation.py # Cubic spline interpolation
│   ├── arbitrage/               # Arbitrage detection modules
│   │   ├── calendar_spread.py   # Calendar spread arbitrage
│   │   ├── butterfly.py         # Butterfly arbitrage
│   │   └── put_call_parity.py   # Put-call parity violations
│   ├── greeks/                  # Greeks calculation engine
│   │   └── greeks_calculator.py # Delta, Gamma, Theta, Vega, Rho
│   └── realtime/               # Real-time monitoring
│       └── monitoring.py       # Real-time arbitrage monitoring
├── examples/                    # Usage examples and demos
│   ├── basic_usage.py          # Basic usage examples
│   ├── generate_visuals.py     # Visualization generation
│   └── dashboard.py            # Interactive dashboard
├── tests/                      # Unit tests
├── data/                       # Sample data files
├── docs/                       # Documentation
├── visuals/                    # Generated visualizations
├── requirements.txt            # Python dependencies
├── setup.py                   # Package setup
└── README.md                  # This file
```

## 🔧 Core Components

### SVI Model
The Stochastic Volatility Inspired (SVI) model parameterizes the total variance as:
```
σ²(k,T) = a + b[ρ(k-m) + √((k-m)² + σ²)]
```

**Parameters:**
- `a`: Vertical offset (level)
- `b`: Slope parameter
- `ρ`: Correlation (skew)
- `m`: Horizontal offset (location)
- `σ`: Volatility of volatility (curvature)

### Arbitrage Detection
The library implements comprehensive arbitrage detection:

1. **Calendar Spread Arbitrage**
   - Ensures longer-term variance ≥ shorter-term variance
   - Monitors time-based arbitrage opportunities

2. **Butterfly Arbitrage**
   - Checks convexity conditions across strikes
   - Validates butterfly spread pricing

3. **Put-Call Parity**
   - Monitors synthetic position pricing
   - Detects parity violations

### Greeks Calculation
Complete Greeks calculation engine supporting:
- **Delta**: Price sensitivity to underlying
- **Gamma**: Delta sensitivity to underlying
- **Theta**: Price sensitivity to time
- **Vega**: Price sensitivity to volatility
- **Rho**: Price sensitivity to interest rates

## 📈 Usage Examples

### SVI Model Fitting
```python
from src.core.svi_model import SVIModel
import numpy as np

# Initialize SVI model
svi = SVIModel()

# Sample data
strikes = np.array([90, 95, 100, 105, 110])
maturities = np.array([0.25, 0.5, 1.0])
implied_vols = np.random.uniform(0.15, 0.35, (len(maturities), len(strikes)))

# Fit model
params = svi.fit(strikes, maturities, implied_vols)
print(f"Fitted parameters: {params}")
```

### Arbitrage Detection
```python
from src.core.arbitrage_detection import ArbitrageDetector

# Initialize detector
detector = ArbitrageDetector()

# Check for arbitrage
results = detector.detect_all_arbitrage(vol_surface_data)
print(f"Calendar spread arbitrage: {results['calendar_spread']}")
print(f"Butterfly arbitrage: {results['butterfly']}")
print(f"Put-call parity violations: {results['put_call_parity']}")
```

### Real-time Monitoring
```python
from src.realtime.monitoring import RealTimeMonitor

# Initialize monitor
monitor = RealTimeMonitor()

# Start monitoring
monitor.start_monitoring(
    symbols=['SPY', 'QQQ', 'IWM'],
    check_interval=60  # seconds
)
```

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
python -m pytest tests/

# Run specific test module
python -m pytest tests/test_svi_model.py

# Run with coverage
python -m pytest --cov=src tests/
```

## 📊 Performance

The library is optimized for performance:
- **Numba JIT compilation** for critical numerical functions
- **Vectorized operations** for large datasets
- **Efficient memory usage** for real-time applications
- **Parallel processing** support for batch operations

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Clone and setup development environment
git clone https://github.com/MohammedKarimKhaldi/vol-surface.git
cd vol-surface

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Run code formatting
black src/ tests/ examples/

# Run linting
flake8 src/ tests/ examples/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SVI Model**: Based on the work of Gatheral and Jacquier
- **Arbitrage Detection**: Inspired by industry best practices
- **Visualization**: Built with Plotly and Dash
- **Community**: Thanks to all contributors and users

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/MohammedKarimKhaldi/vol-surface/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MohammedKarimKhaldi/vol-surface/discussions)
- **Email**: [Contact via GitHub](https://github.com/MohammedKarimKhaldi)

## 🔄 Version History

- **v1.0.0**: Initial release with core functionality
- **v1.1.0**: Added interactive dashboard
- **v1.2.0**: Enhanced arbitrage detection algorithms
- **v1.3.0**: Real-time monitoring capabilities

---

**Made with ❤️ by Mohammed Karim Khaldi**

*For quantitative finance professionals and researchers* 