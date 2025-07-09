"""
Generate static images and GIFs for volatility surface, arbitrage, and Greeks.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from vol_surface import VolatilitySurface

# --- Helper to generate sample data (reuse from basic_usage) ---
def generate_sample_data():
    strikes = np.linspace(80, 120, 20)
    maturities = np.array([0.25, 0.5, 1.0, 1.5, 2.0])
    K, T = np.meshgrid(strikes, maturities)
    K = K.flatten()
    T = T.flatten()
    spot_price = 100.0
    moneyness = np.log(K / spot_price)
    implied_vols = []
    for i in range(len(K)):
        k = moneyness[i]
        t = T[i]
        a = 0.04 * t
        b = 0.1 * t
        rho = -0.1
        m = 0.0
        sigma = 0.1
        total_var = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
        implied_vol = np.sqrt(total_var / t)
        implied_vols.append(implied_vol)
    implied_vols = np.array(implied_vols)
    call_prices = []
    put_prices = []
    risk_free_rate = 0.02
    dividend_yield = 0.0
    for i in range(len(K)):
        S = spot_price
        K_i = K[i]
        T_i = T[i]
        sigma_i = implied_vols[i]
        d1 = (np.log(S / K_i) + (risk_free_rate - dividend_yield + 0.5 * sigma_i**2) * T_i) / (sigma_i * np.sqrt(T_i))
        d2 = d1 - sigma_i * np.sqrt(T_i)
        call_price = S * np.exp(-dividend_yield * T_i) * 0.5 * (1 + math.erf(d1 / np.sqrt(2))) - \
                    K_i * np.exp(-risk_free_rate * T_i) * 0.5 * (1 + math.erf(d2 / np.sqrt(2)))
        put_price = K_i * np.exp(-risk_free_rate * T_i) * 0.5 * (1 + math.erf(-d2 / np.sqrt(2))) - \
                   S * np.exp(-dividend_yield * T_i) * 0.5 * (1 + math.erf(-d1 / np.sqrt(2)))
        call_prices.append(call_price)
        put_prices.append(put_price)
    data = pd.DataFrame({
        'strike': K,
        'maturity': T,
        'implied_vol': implied_vols,
        'call_price': call_prices,
        'put_price': put_prices,
        'option_type': ['call'] * len(K)
    })
    return data

# --- Main script ---
def main():
    os.makedirs("visuals", exist_ok=True)
    data = generate_sample_data()
    surface = VolatilitySurface(spot_price=100.0, risk_free_rate=0.02, dividend_yield=0.0)
    surface.load_data(data)
    surface.fit_svi_model()
    surface.fit_spline_surface(tension=0.1)

    # 1. SVI Surface Plot
    grid = surface.generate_surface_grid((80, 120), (0.25, 2.0), n_strikes=40, n_maturities=20, method='svi')
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(grid['strikes'], grid['maturities'], grid['implied_vols'], cmap='viridis', alpha=0.85)
    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity')
    ax.set_zlabel('Implied Volatility')
    ax.set_title('SVI Volatility Surface')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig('visuals/svi_surface.png')
    plt.close(fig)

    # 2. Spline Surface Plot
    grid = surface.generate_surface_grid((80, 120), (0.25, 2.0), n_strikes=40, n_maturities=20, method='spline')
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(grid['strikes'], grid['maturities'], grid['implied_vols'], cmap='plasma', alpha=0.85)
    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity')
    ax.set_zlabel('Implied Volatility')
    ax.set_title('Spline Volatility Surface')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig('visuals/spline_surface.png')
    plt.close(fig)

    # 3. Greeks Surfaces (Delta, Gamma)
    K, T = np.meshgrid(np.linspace(80, 120, 40), np.linspace(0.25, 2.0, 20))
    K_flat, T_flat = K.flatten(), T.flatten()
    option_types = ['call'] * len(K_flat)
    greeks = surface.calculate_greeks(K_flat, T_flat, option_types)
    for greek in ['deltas', 'gammas', 'vegas', 'thetas', 'rhos']:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(K, T, greeks[greek].reshape(K.shape), cmap='coolwarm', alpha=0.85)
        ax.set_xlabel('Strike')
        ax.set_ylabel('Maturity')
        ax.set_zlabel(greek.capitalize())
        ax.set_title(f'{greek.capitalize()} Surface')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        plt.tight_layout()
        plt.savefig(f'visuals/{greek}_surface.png')
        plt.close(fig)

    # 4. Arbitrage Violation Heatmaps (if any)
    arbitrage = surface.detect_arbitrage()
    for arb_type, result in arbitrage.items():
        if result['violation_count'] > 0:
            violations = result['violations']
            strikes = [v.get('strike', 0) for v in violations]
            maturities = [v.get('maturity1', v.get('maturity', 0)) for v in violations]
            plt.figure(figsize=(8, 6))
            plt.hexbin(strikes, maturities, gridsize=20, cmap='Reds', mincnt=1)
            plt.xlabel('Strike')
            plt.ylabel('Maturity')
            plt.title(f'{arb_type.capitalize()} Arbitrage Violations')
            plt.colorbar(label='Violation Count')
            plt.tight_layout()
            plt.savefig(f'visuals/{arb_type}_arbitrage_heatmap.png')
            plt.close()

    # 5. Animated GIF: SVI Surface Evolution (simulate parameter change)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    strikes = np.linspace(80, 120, 40)
    maturities = np.linspace(0.25, 2.0, 20)
    K, T = np.meshgrid(strikes, maturities)
    ims = []
    for t in np.linspace(0.25, 2.0, 20):
        svi_params = surface.svi_model.get_parameters()
        # Simulate a parameter drift for animation
        svi_params['a'] *= (0.95 + 0.1 * (t - 0.25) / 1.75)
        svi_params['b'] *= (0.95 + 0.1 * (t - 0.25) / 1.75)
        vols = surface.svi_model._svi_function(np.log(strikes), svi_params['a'], svi_params['b'], svi_params['rho'], svi_params['m'], svi_params['sigma'])
        vols = np.sqrt(vols / t)
        surf = [ax.plot(strikes, vols, zs=t, zdir='y', color='b', alpha=0.5)[0]]
        ims.append(surf)
    def update(frame):
        ax.clear()
        t = np.linspace(0.25, 2.0, 20)[frame]
        svi_params = surface.svi_model.get_parameters()
        svi_params['a'] *= (0.95 + 0.1 * (t - 0.25) / 1.75)
        svi_params['b'] *= (0.95 + 0.1 * (t - 0.25) / 1.75)
        vols = surface.svi_model._svi_function(np.log(strikes), svi_params['a'], svi_params['b'], svi_params['rho'], svi_params['m'], svi_params['sigma'])
        vols = np.sqrt(vols / t)
        ax.plot(strikes, vols, zs=t, zdir='y', color='b', alpha=0.7)
        ax.set_xlabel('Strike')
        ax.set_ylabel('Maturity')
        ax.set_zlabel('Implied Volatility')
        ax.set_title('SVI Surface Evolution')
        ax.set_xlim(80, 120)
        ax.set_ylim(0.25, 2.0)
        ax.set_zlim(0.15, 0.35)
    ani = animation.FuncAnimation(fig, update, frames=20, interval=200)
    ani.save('visuals/svi_surface_evolution.gif', writer='pillow')
    plt.close(fig)

    print("All visuals generated in the 'visuals/' directory.")

if __name__ == "__main__":
    main() 