"""
Basic functionality tests for the volatility surface project
"""

import numpy as np
import pandas as pd
from vol_surface import VolatilitySurface


def test_svi_model():
    """Test SVI model functionality."""
    from src.core.svi_model import SVIModel
    
    # Create sample data
    strikes = np.array([90, 100, 110])
    maturities = np.array([0.5, 0.5, 0.5])
    implied_vols = np.array([0.25, 0.20, 0.25])
    
    # Initialize and fit SVI model
    svi = SVIModel()
    params = svi.fit(strikes, maturities, implied_vols)
    
    # Check that parameters are returned
    assert isinstance(params, dict)
    assert all(key in params for key in ['a', 'b', 'rho', 'm', 'sigma'])
    
    # Check that model is fitted
    assert svi.fitted
    
    # Test prediction
    predicted_vols = svi.predict(strikes, maturities)
    assert len(predicted_vols) == len(strikes)
    assert all(vol > 0 for vol in predicted_vols)


def test_variance_swap_calculator():
    """Test variance swap calculator."""
    from src.core.variance_swap import VarianceSwapCalculator
    
    # Create sample data
    strikes = np.array([90, 100, 110])
    call_prices = np.array([15.0, 8.0, 3.0])
    put_prices = np.array([2.0, 8.0, 15.0])
    maturity = 0.5
    
    # Initialize calculator
    calc = VarianceSwapCalculator(spot_price=100.0, risk_free_rate=0.02)
    
    # Calculate variance swap rate
    variance_rate = calc.calculate_variance_swap_rate(strikes, call_prices, put_prices, maturity)
    
    # Check that variance rate is positive
    assert variance_rate > 0


def test_arbitrage_detection():
    """Test arbitrage detection functionality."""
    from src.arbitrage.calendar_spread import CalendarSpreadDetector
    from src.arbitrage.butterfly import ButterflyDetector
    
    # Create sample data
    strikes = np.array([90, 100, 110])
    maturities = np.array([0.5, 1.0, 1.5])
    implied_vols = np.array([0.25, 0.20, 0.25])
    
    # Test calendar spread detection
    calendar_detector = CalendarSpreadDetector()
    calendar_results = calendar_detector.check_calendar_spread(strikes, maturities, implied_vols)
    
    assert isinstance(calendar_results, dict)
    assert 'violations' in calendar_results
    assert 'violation_count' in calendar_results
    
    # Test butterfly detection
    butterfly_detector = ButterflyDetector()
    butterfly_results = butterfly_detector.check_butterfly_arbitrage(strikes, maturities, implied_vols)
    
    assert isinstance(butterfly_results, dict)
    assert 'violations' in butterfly_results
    assert 'violation_count' in butterfly_results


def test_greeks_calculator():
    """Test Greeks calculation."""
    from src.greeks.sensitivities import GreeksCalculator
    
    # Initialize calculator
    calc = GreeksCalculator(spot_price=100.0, risk_free_rate=0.02)
    
    # Calculate Greeks for a call option
    greeks = calc.calculate_greeks(
        strike=100.0,
        maturity=0.5,
        implied_vol=0.2,
        option_type='call'
    )
    
    # Check that all Greeks are calculated
    assert all(key in greeks for key in ['price', 'delta', 'gamma', 'vega', 'theta', 'rho'])
    
    # Check that price is positive
    assert greeks['price'] > 0
    
    # Check that delta is between 0 and 1 for call
    assert 0 <= greeks['delta'] <= 1


def test_volatility_surface_integration():
    """Test the main volatility surface class."""
    # Create sample data
    data = pd.DataFrame({
        'strike': [90, 100, 110],
        'maturity': [0.5, 0.5, 0.5],
        'implied_vol': [0.25, 0.20, 0.25],
        'call_price': [15.0, 8.0, 3.0],
        'put_price': [2.0, 8.0, 15.0]
    })
    
    # Initialize volatility surface
    surface = VolatilitySurface(spot_price=100.0, risk_free_rate=0.02)
    surface.load_data(data)
    
    # Test SVI fitting
    svi_params = surface.fit_svi_model()
    assert isinstance(svi_params, dict)
    assert surface.fitted
    
    # Test spline fitting
    surface.fit_spline_surface()
    assert surface.fitted
    
    # Test volatility prediction
    strikes = np.array([95, 105])
    maturities = np.array([0.5, 0.5])
    predicted_vols = surface.predict_volatility(strikes, maturities, method='svi')
    predicted_vols = np.array(predicted_vols)
    assert len(predicted_vols) == len(strikes)
    
    # Test arbitrage detection
    arbitrage_results = surface.detect_arbitrage()
    assert isinstance(arbitrage_results, dict)
    
    # Test Greeks calculation
    option_types = ['call', 'call']
    greeks = surface.calculate_greeks(strikes, maturities, option_types)
    assert isinstance(greeks, dict)
    assert 'deltas' in greeks
    
    # Test risk metrics
    risk_metrics = surface.calculate_risk_metrics(strikes, maturities, option_types)
    assert isinstance(risk_metrics, dict)
    assert 'portfolio_price' in risk_metrics
    
    # Test summary statistics
    stats = surface.get_summary_statistics()
    assert isinstance(stats, dict)
    assert 'total_options' in stats


def test_data_validation():
    """Test data validation and error handling."""
    surface = VolatilitySurface()
    
    # Test with no data loaded
    try:
        surface.fit_svi_model()
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    try:
        surface.detect_arbitrage()
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Test with invalid method
    surface.load_data({'strikes': np.array([100]), 'maturities': np.array([0.5]), 'implied_vols': np.array([0.2])})
    surface.fit_svi_model()
    
    try:
        surface.predict_volatility(np.array([100]), np.array([0.5]), method='invalid')
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    # Run tests
    test_svi_model()
    test_variance_swap_calculator()
    test_arbitrage_detection()
    test_greeks_calculator()
    test_volatility_surface_integration()
    test_data_validation()
    
    print("All tests passed successfully!") 