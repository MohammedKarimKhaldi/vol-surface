"""
Basic Usage Example for Volatility Surface Construction and Arbitrage Detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from vol_surface import VolatilitySurface


def generate_sample_data():
    """Generate sample option data for demonstration."""
    # Generate strikes and maturities
    strikes = np.linspace(80, 120, 20)
    maturities = np.array([0.25, 0.5, 1.0, 1.5, 2.0])
    
    # Create grid
    K, T = np.meshgrid(strikes, maturities)
    K = K.flatten()
    T = T.flatten()
    
    # Generate realistic implied volatilities with smile
    spot_price = 100.0
    moneyness = np.log(K / spot_price)
    
    # SVI-like implied volatility surface
    implied_vols = []
    for i in range(len(K)):
        k = moneyness[i]
        t = T[i]
        
        # SVI parameters (simplified)
        a = 0.04 * t
        b = 0.1 * t
        rho = -0.1
        m = 0.0
        sigma = 0.1
        
        # SVI formula
        total_var = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
        implied_vol = np.sqrt(total_var / t)
        implied_vols.append(implied_vol)
    
    implied_vols = np.array(implied_vols)
    
    # Generate option prices using Black-Scholes
    call_prices = []
    put_prices = []
    risk_free_rate = 0.02
    dividend_yield = 0.0
    
    for i in range(len(K)):
        S = spot_price
        K_i = K[i]
        T_i = T[i]
        sigma_i = implied_vols[i]
        
        # Black-Scholes call price
        d1 = (np.log(S / K_i) + (risk_free_rate - dividend_yield + 0.5 * sigma_i**2) * T_i) / (sigma_i * np.sqrt(T_i))
        d2 = d1 - sigma_i * np.sqrt(T_i)
        
        call_price = S * np.exp(-dividend_yield * T_i) * 0.5 * (1 + math.erf(d1 / np.sqrt(2))) - \
                    K_i * np.exp(-risk_free_rate * T_i) * 0.5 * (1 + math.erf(d2 / np.sqrt(2)))
        
        put_price = K_i * np.exp(-risk_free_rate * T_i) * 0.5 * (1 + math.erf(-d2 / np.sqrt(2))) - \
                   S * np.exp(-dividend_yield * T_i) * 0.5 * (1 + math.erf(-d1 / np.sqrt(2)))
        
        call_prices.append(call_price)
        put_prices.append(put_price)
    
    # Create DataFrame
    data = pd.DataFrame({
        'strike': K,
        'maturity': T,
        'implied_vol': implied_vols,
        'call_price': call_prices,
        'put_price': put_prices,
        'option_type': ['call'] * len(K)  # For simplicity, we'll use calls
    })
    
    return data


def main():
    """Main demonstration function."""
    print("Volatility Surface Construction and Arbitrage Detection")
    print("=" * 60)
    
    # Generate sample data
    print("\n1. Generating sample option data...")
    data = generate_sample_data()
    print(f"Generated {len(data)} option contracts")
    print(f"Strike range: {data['strike'].min():.2f} - {data['strike'].max():.2f}")
    print(f"Maturity range: {data['maturity'].min():.2f} - {data['maturity'].max():.2f}")
    print(f"Volatility range: {data['implied_vol'].min():.3f} - {data['implied_vol'].max():.3f}")
    
    # Initialize volatility surface
    print("\n2. Initializing volatility surface...")
    surface = VolatilitySurface(spot_price=100.0, risk_free_rate=0.02, dividend_yield=0.0)
    surface.load_data(data)
    
    # Fit SVI model
    print("\n3. Fitting SVI model...")
    svi_params = surface.fit_svi_model()
    print("SVI Parameters:")
    for param, value in svi_params.items():
        print(f"  {param}: {value:.6f}")
    
    # Fit spline surface
    print("\n4. Fitting spline surface...")
    surface.fit_spline_surface(tension=0.1)
    print("Spline surface fitted successfully")
    
    # Detect arbitrage
    print("\n5. Detecting arbitrage violations...")
    arbitrage_results = surface.detect_arbitrage()
    
    print("Arbitrage Detection Results:")
    for arb_type, result in arbitrage_results.items():
        print(f"  {arb_type}:")
        print(f"    Violations: {result['violation_count']}")
        print(f"    Violation rate: {result['violation_rate']:.2%}")
    
    # Calculate Greeks
    print("\n6. Calculating Greeks...")
    sample_strikes = np.array([90, 100, 110])
    sample_maturities = np.array([0.5, 1.0])
    sample_types = ['call', 'call', 'call', 'call', 'call', 'call']
    
    K_sample, T_sample = np.meshgrid(sample_strikes, sample_maturities)
    K_sample = K_sample.flatten()
    T_sample = T_sample.flatten()
    
    greeks = surface.calculate_greeks(K_sample, T_sample, sample_types)
    
    print("Greeks for sample options:")
    for i in range(len(K_sample)):
        print(f"  Strike: {K_sample[i]:.0f}, Maturity: {T_sample[i]:.1f}")
        print(f"    Delta: {greeks['deltas'][i]:.4f}")
        print(f"    Gamma: {greeks['gammas'][i]:.6f}")
        print(f"    Vega: {greeks['vegas'][i]:.4f}")
        print(f"    Theta: {greeks['thetas'][i]:.4f}")
        print(f"    Rho: {greeks['rhos'][i]:.4f}")
    
    # Calculate risk metrics
    print("\n7. Calculating risk metrics...")
    risk_metrics = surface.calculate_risk_metrics(K_sample, T_sample, sample_types)
    
    print("Portfolio Risk Metrics:")
    for metric, value in risk_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Generate surface grid
    print("\n8. Generating volatility surface grid...")
    surface_grid = surface.generate_surface_grid(
        strike_range=(80, 120),
        maturity_range=(0.1, 2.0),
        n_strikes=30,
        n_maturities=15,
        method='svi'
    )
    
    print(f"Generated surface grid: {surface_grid['strikes'].shape}")
    
    # Get summary statistics
    print("\n9. Summary statistics...")
    stats = surface.get_summary_statistics()
    
    print("Summary Statistics:")
    for stat, value in stats.items():
        if isinstance(value, dict):
            print(f"  {stat}:")
            for sub_stat, sub_value in value.items():
                print(f"    {sub_stat}: {sub_value}")
        else:
            print(f"  {stat}: {value}")
    
    # Plot surface (if matplotlib is available)
    try:
        print("\n10. Plotting volatility surface...")
        surface.plot_surface(method='svi', title="SVI Volatility Surface")
        print("Surface plot generated successfully")
    except Exception as e:
        print(f"Could not generate plot: {e}")
    
    print("\n" + "=" * 60)
    print("Demonstration completed successfully!")
    print("The volatility surface system is ready for use.")


if __name__ == "__main__":
    main() 