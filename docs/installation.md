# Installation Guide

This guide will help you install the Volatility Surface Construction & Arbitrage Detection library on your system.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 4GB RAM (8GB+ recommended for large datasets)
- **Storage**: 500MB free space

### Required Software
- **Python Package Manager**: pip or conda
- **Git**: For cloning the repository
- **C++ Compiler**: For optional Numba JIT compilation (recommended)

## 🚀 Quick Installation

### Method 1: From GitHub (Recommended)

```bash
# Clone the repository
git clone https://github.com/MohammedKarimKhaldi/vol-surface.git
cd vol-surface

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Method 2: Using pip (if published to PyPI)

```bash
# Install directly from PyPI
pip install vol-surface

# Or install with optional dependencies
pip install vol-surface[full]
```

### Method 3: Using conda

```bash
# Create a new conda environment
conda create -n vol-surface python=3.9
conda activate vol-surface

# Install dependencies
conda install numpy scipy pandas matplotlib seaborn
pip install plotly dash numba scikit-learn
```

## 📦 Dependencies

### Core Dependencies
The following packages are automatically installed:

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.20.0 | Numerical computing |
| `scipy` | ≥1.7.0 | Scientific computing |
| `pandas` | ≥1.3.0 | Data manipulation |
| `matplotlib` | ≥3.4.0 | Static plotting |
| `seaborn` | ≥0.11.0 | Statistical visualization |
| `plotly` | ≥5.0.0 | Interactive plotting |
| `dash` | ≥2.0.0 | Web dashboard |

### Optional Dependencies
For enhanced performance and additional features:

| Package | Version | Purpose |
|---------|---------|---------|
| `numba` | ≥0.56.0 | JIT compilation for speed |
| `scikit-learn` | ≥1.0.0 | Machine learning utilities |
| `jupyter` | ≥1.0.0 | Jupyter notebook support |
| `pytest` | ≥6.0.0 | Testing framework |

## 🔧 Installation Options

### Development Installation
For developers who want to modify the source code:

```bash
# Clone and install in editable mode
git clone https://github.com/MohammedKarimKhaldi/vol-surface.git
cd vol-surface
pip install -e .

# Install development dependencies
pip install pytest pytest-cov black flake8
```

### Production Installation
For production deployments:

```bash
# Install with all optimizations
pip install vol-surface[full]

# Or install specific components
pip install vol-surface[core]  # Core functionality only
pip install vol-surface[dashboard]  # Include dashboard
```

### Minimal Installation
For basic usage without optional features:

```bash
# Install core dependencies only
pip install numpy scipy pandas matplotlib
pip install vol-surface[core]
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors
**Problem**: `ModuleNotFoundError` when importing the library

**Solution**:
```bash
# Ensure you're in the correct directory
cd vol-surface

# Reinstall the package
pip uninstall vol-surface
pip install -e .
```

#### 2. Numba Compilation Issues
**Problem**: Warnings or errors related to Numba

**Solution**:
```bash
# Install LLVM (required for Numba)
# On macOS:
brew install llvm

# On Ubuntu:
sudo apt-get install llvm-dev

# On Windows:
# Download from https://github.com/llvm/llvm-project/releases
```

#### 3. Dashboard Not Starting
**Problem**: Dashboard fails to start or shows errors

**Solution**:
```bash
# Check if all dashboard dependencies are installed
pip install dash plotly

# Try running with debug mode
python examples/dashboard.py --debug
```

#### 4. Memory Issues
**Problem**: Out of memory errors with large datasets

**Solution**:
```bash
# Reduce memory usage by using smaller data chunks
# Or increase system swap space

# On Linux:
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Platform-Specific Issues

#### macOS
```bash
# Install Xcode command line tools (if not already installed)
xcode-select --install

# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Windows
```bash
# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/

# Or use conda for easier installation
conda install -c conda-forge vol-surface
```

#### Linux (Ubuntu/Debian)
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev python3-pip build-essential

# Install Python packages
pip3 install vol-surface
```

## ✅ Verification

After installation, verify that everything is working:

```python
# Test basic imports
import numpy as np
from src.core.volatility_surface import VolatilitySurface
from src.core.svi_model import SVIModel

# Test SVI model
svi = SVIModel()
print("SVI Model imported successfully!")

# Test volatility surface
vol_surface = VolatilitySurface()
print("Volatility Surface imported successfully!")

# Test dashboard (optional)
try:
    import dash
    print("Dashboard dependencies installed successfully!")
except ImportError:
    print("Dashboard dependencies not installed. Run: pip install dash plotly")
```

## 🔄 Updating

To update to the latest version:

```bash
# If installed from GitHub
cd vol-surface
git pull origin main
pip install -e . --upgrade

# If installed from PyPI
pip install vol-surface --upgrade
```

## 🗑️ Uninstallation

To remove the library:

```bash
# Remove the package
pip uninstall vol-surface

# Remove the repository (if cloned)
rm -rf vol-surface
```

## 📞 Getting Help

If you encounter issues during installation:

1. **Check the troubleshooting section** above
2. **Search existing issues** on [GitHub Issues](https://github.com/MohammedKarimKhaldi/vol-surface/issues)
3. **Create a new issue** with:
   - Your operating system and Python version
   - Complete error message
   - Steps to reproduce the issue
   - Installation method used

---

*Next: [Quick Start Guide](quickstart.md)* 