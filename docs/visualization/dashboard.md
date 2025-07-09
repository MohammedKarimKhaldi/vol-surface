# Interactive Dashboard Guide

The Interactive Dashboard provides real-time visualization and parameter adjustment for volatility surface analysis.

## 🚀 Quick Start

### Starting the Dashboard

```bash
# Navigate to the project directory
cd vol-surface

# Run the dashboard
python examples/dashboard.py
```

### Accessing the Dashboard

1. **Open your web browser**
2. **Navigate to**: `http://127.0.0.1:8050`
3. **Wait for the dashboard to load** (may take a few seconds)

## 🎛️ Dashboard Features

### Main Interface

The dashboard is divided into two main sections:

#### Left Panel: Controls
- **SVI Parameters**: Interactive sliders for all 5 SVI parameters
- **View Options**: Toggle between 3D surface and 2D contour plots
- **Arbitrage Status**: Real-time arbitrage detection alerts

#### Right Panel: Visualization
- **3D Surface Plot**: Interactive 3D volatility surface
- **2D Contour Plot**: Heatmap visualization
- **Surface Statistics**: Key metrics and statistics

## 📊 SVI Parameter Controls

### Parameter `a` (Level)
- **Range**: 0.01 - 0.1
- **Default**: 0.04
- **Effect**: Controls overall variance level
- **Visual Impact**: Moves the entire surface up/down

### Parameter `b` (Slope)
- **Range**: 0.1 - 1.0
- **Default**: 0.4
- **Effect**: Controls smile slope
- **Visual Impact**: Makes the smile steeper or flatter

### Parameter `ρ` (Skew)
- **Range**: -0.5 - 0.5
- **Default**: -0.1
- **Effect**: Controls skewness
- **Visual Impact**: 
  - ρ > 0: Higher volatility for calls (right skew)
  - ρ < 0: Higher volatility for puts (left skew)
  - ρ = 0: Symmetric smile

### Parameter `m` (Location)
- **Range**: -0.5 - 0.5
- **Default**: 0.0
- **Effect**: Controls smile center location
- **Visual Impact**: Shifts the smile horizontally

### Parameter `σ` (Curvature)
- **Range**: 0.05 - 0.3
- **Default**: 0.1
- **Effect**: Controls smile curvature
- **Visual Impact**: Makes the smile more or less curved

## 🎯 View Options

### 3D Surface Plot
- **Interactive rotation**: Click and drag to rotate
- **Zoom**: Scroll to zoom in/out
- **Pan**: Right-click and drag to pan
- **Reset view**: Double-click to reset camera

### 2D Contour Plot
- **Heatmap colors**: Viridis color scheme
- **Contour lines**: Show variance levels
- **Interactive hover**: Hover for exact values
- **Export**: Right-click to save image

## ⚠️ Arbitrage Detection

### Real-time Monitoring

The dashboard continuously monitors for arbitrage opportunities:

#### Calendar Spread Arbitrage
- **Detection**: Longer-term variance < shorter-term variance
- **Alert**: Red warning with specific strike prices
- **Impact**: Time-based arbitrage opportunities

#### Butterfly Arbitrage
- **Detection**: Convexity violations in strike spreads
- **Alert**: Red warning with expiry and strike details
- **Impact**: Strike-based arbitrage opportunities

#### Status Indicators
- **✅ Green**: No arbitrage detected
- **⚠️ Red**: Arbitrage opportunities found
- **Details**: List of specific violations

### Example Arbitrage Alerts

```
⚠️ Arbitrage Detected!
• Calendar spread arbitrage detected at strike 95.0
• Butterfly arbitrage detected at expiry 0.25, strike 100.0
• Put-call parity violations: 3 instances
```

## 📈 Surface Statistics

### Real-time Metrics

The dashboard displays key statistics that update in real-time:

#### Mean Variance
- **Calculation**: Average variance across the surface
- **Range**: Typically 0.01 - 0.25
- **Interpretation**: Overall volatility level

#### Standard Deviation
- **Calculation**: Dispersion of variance values
- **Range**: Typically 0.01 - 0.1
- **Interpretation**: Volatility of volatility

#### Minimum Variance
- **Calculation**: Lowest variance value
- **Range**: Typically 0.005 - 0.15
- **Interpretation**: Minimum volatility level

#### Maximum Variance
- **Calculation**: Highest variance value
- **Range**: Typically 0.02 - 0.4
- **Interpretation**: Maximum volatility level

## 🔧 Advanced Features

### Parameter Presets

The dashboard includes several preset parameter combinations:

#### Market Conditions
- **Normal Market**: Balanced parameters
- **High Volatility**: Elevated variance levels
- **Low Volatility**: Reduced variance levels
- **Skewed Market**: Asymmetric smile

#### Academic Examples
- **Symmetric Smile**: ρ = 0, centered at ATM
- **Steep Smile**: High curvature, high slope
- **Flat Smile**: Low curvature, low slope

### Export Capabilities

#### Save Plots
- **Format**: PNG, PDF, SVG
- **Resolution**: High-quality output
- **Metadata**: Includes parameter values

#### Export Data
- **Format**: CSV, JSON
- **Content**: Surface data, parameters, statistics
- **Use**: Further analysis in other tools

### Performance Optimization

#### Real-time Updates
- **Efficient rendering**: Optimized for smooth interaction
- **Memory management**: Automatic cleanup
- **Responsive design**: Works on different screen sizes

#### Large Datasets
- **Grid resolution**: Adjustable for performance
- **Lazy loading**: Load data on demand
- **Caching**: Cache computed results

## 🎨 Customization

### Styling Options

#### Color Schemes
- **Viridis**: Default scientific colormap
- **Plasma**: Alternative high-contrast scheme
- **Inferno**: Dark theme option
- **Custom**: User-defined colors

#### Layout Options
- **Panel arrangement**: Adjustable panel sizes
- **Control placement**: Reorganize controls
- **Theme**: Light/dark mode toggle

### Advanced Controls

#### Parameter Constraints
- **Custom ranges**: Set parameter bounds
- **Linked parameters**: Synchronize related parameters
- **Validation**: Real-time parameter validation

#### Interpolation Options
- **Grid density**: Adjust surface resolution
- **Smoothing**: Control surface smoothness
- **Extrapolation**: Handle out-of-bounds values

## 🚨 Troubleshooting

### Common Issues

#### Dashboard Won't Start
```bash
# Check dependencies
pip install dash plotly

# Try different port
python examples/dashboard.py --port 8051

# Check for port conflicts
lsof -i :8050
```

#### Slow Performance
```python
# Reduce grid resolution
# In dashboard.py, change:
strikes = np.linspace(80, 120, 10)  # Reduce from 20 to 10
expiries = np.array([0.1, 0.25, 0.5, 1.0])  # Reduce from 5 to 4
```

#### Display Issues
- **Clear browser cache**: Hard refresh (Ctrl+F5)
- **Try different browser**: Chrome, Firefox, Safari
- **Check JavaScript**: Enable JavaScript in browser

### Error Messages

#### "Module not found"
```bash
# Install missing dependencies
pip install -r requirements.txt
```

#### "Port already in use"
```bash
# Kill existing process
pkill -f dashboard.py

# Or use different port
python examples/dashboard.py --port 8051
```

#### "Memory error"
```python
# Reduce data size
# Use smaller parameter ranges
# Close other applications
```

## 📚 Usage Examples

### Educational Use

#### Parameter Exploration
1. **Start with symmetric smile**: Set ρ = 0, m = 0
2. **Add skew**: Gradually change ρ to ±0.3
3. **Adjust curvature**: Modify σ from 0.05 to 0.3
4. **Observe arbitrage**: Watch for red alerts

#### Market Scenarios
1. **Normal market**: Use default parameters
2. **Crisis mode**: Increase a and b, add negative skew
3. **Calm market**: Decrease a and b, reduce curvature

### Research Use

#### Parameter Sensitivity
```python
# Systematic parameter variation
for a in [0.02, 0.04, 0.06, 0.08]:
    for rho in [-0.3, -0.1, 0.1, 0.3]:
        # Set parameters and observe surface
        # Record arbitrage detection results
```

#### Arbitrage Analysis
```python
# Monitor arbitrage patterns
# Record parameter combinations that cause arbitrage
# Analyze transition points between arbitrage-free and arbitrage regions
```

### Professional Use

#### Risk Management
- **Monitor surface stability**: Check for parameter drift
- **Validate arbitrage-free conditions**: Ensure no violations
- **Document parameter changes**: Track surface evolution

#### Trading Support
- **Real-time surface monitoring**: Watch for market changes
- **Arbitrage opportunity detection**: Identify trading opportunities
- **Parameter validation**: Ensure surface quality

## 🔗 Integration

### API Access
```python
# Access dashboard data programmatically
from examples.dashboard import app

# Get current parameters
current_params = app.get_current_parameters()

# Set parameters programmatically
app.set_parameters({'a': 0.05, 'rho': -0.2})
```

### External Tools
- **Jupyter integration**: Embed dashboard in notebooks
- **Data export**: Save results for further analysis
- **API endpoints**: RESTful interface for automation

## 📞 Support

### Getting Help
- **Documentation**: Check this guide and API docs
- **GitHub Issues**: Report bugs and request features
- **Community**: Join discussions for tips and tricks

### Contributing
- **Feature requests**: Suggest new dashboard features
- **Bug reports**: Help improve stability
- **Code contributions**: Enhance functionality

---

*Next: [Static Visualizations](static_plots.md) or [Custom Plotting](custom_plots.md)* 