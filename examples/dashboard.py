import dash
from dash import dcc, html, Input, Output, callback_context
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Initialize the app
app = dash.Dash(__name__)

# SVI function implementation
def svi_function(k, a, b, rho, m, sigma):
    """Compute SVI total variance function."""
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

# Generate sample data
strikes = np.linspace(80, 120, 20)
expiries = np.array([0.1, 0.25, 0.5, 1.0, 2.0])
K, T = np.meshgrid(strikes, expiries)

# Default SVI parameters
default_params = {
    'a': 0.04,
    'b': 0.4,
    'rho': -0.1,
    'm': 0.0,
    'sigma': 0.1
}

def generate_volatility_surface(params):
    """Generate volatility surface with given SVI parameters"""
    vol_surface_data = np.zeros_like(K)
    
    for i, expiry in enumerate(expiries):
        for j, strike in enumerate(strikes):
            k = np.log(strike / 100)  # log-moneyness
            vol_surface_data[i, j] = svi_function(k, params['a'], params['b'], params['rho'], params['m'], params['sigma'])
    
    return vol_surface_data

# Layout
app.layout = html.Div([
    html.H1("Volatility Surface Dashboard", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
    
    html.Div([
        # Controls Panel
        html.Div([
            html.H3("SVI Parameters", style={'color': '#34495e'}),
            
            html.Label("Parameter a (level)"),
            dcc.Slider(
                id='param-a',
                min=0.01, max=0.1, step=0.01,
                value=default_params['a'],
                marks={i/100: str(i/100) for i in range(1, 11)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            
            html.Label("Parameter b (slope)"),
            dcc.Slider(
                id='param-b',
                min=0.1, max=1.0, step=0.05,
                value=default_params['b'],
                marks={i/10: str(i/10) for i in range(1, 11)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            
            html.Label("Parameter rho (skew)"),
            dcc.Slider(
                id='param-rho',
                min=-0.5, max=0.5, step=0.05,
                value=default_params['rho'],
                marks={i/10: str(i/10) for i in range(-5, 6)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            
            html.Label("Parameter m (location)"),
            dcc.Slider(
                id='param-m',
                min=-0.5, max=0.5, step=0.05,
                value=default_params['m'],
                marks={i/10: str(i/10) for i in range(-5, 6)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            
            html.Label("Parameter sigma (curvature)"),
            dcc.Slider(
                id='param-sigma',
                min=0.05, max=0.3, step=0.01,
                value=default_params['sigma'],
                marks={i/100: str(i/100) for i in range(5, 31, 5)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            
            html.Hr(),
            
            html.H3("View Options", style={'color': '#34495e'}),
            dcc.RadioItems(
                id='plot-type',
                options=[
                    {'label': '3D Surface', 'value': '3d'},
                    {'label': '2D Contour', 'value': '2d'}
                ],
                value='3d',
                style={'marginTop': 10}
            ),
            
            html.Hr(),
            
            html.H3("Arbitrage Detection", style={'color': '#34495e'}),
            html.Div(id='arbitrage-status', style={'padding': 10, 'borderRadius': 5}),
            
        ], style={'width': '25%', 'float': 'left', 'padding': 20, 'backgroundColor': '#f8f9fa'}),
        
        # Visualization Panel
        html.Div([
            dcc.Graph(id='volatility-plot', style={'height': 600}),
            
            html.Div([
                html.H3("Surface Statistics", style={'color': '#34495e'}),
                html.Div(id='surface-stats', style={'display': 'flex', 'justifyContent': 'space-around'})
            ], style={'marginTop': 20})
            
        ], style={'width': '75%', 'float': 'right', 'padding': 20})
        
    ], style={'display': 'flex'}),
    
    # Hidden div for storing intermediate values
    html.Div(id='intermediate-value', style={'display': 'none'})
])

@app.callback(
    [Output('volatility-plot', 'figure'),
     Output('arbitrage-status', 'children'),
     Output('surface-stats', 'children')],
    [Input('param-a', 'value'),
     Input('param-b', 'value'),
     Input('param-rho', 'value'),
     Input('param-m', 'value'),
     Input('param-sigma', 'value'),
     Input('plot-type', 'value')]
)
def update_volatility_surface(a, b, rho, m, sigma, plot_type):
    # Update SVI parameters
    params = {'a': a, 'b': b, 'rho': rho, 'm': m, 'sigma': sigma}
    
    # Generate volatility surface
    vol_surface_data = generate_volatility_surface(params)
    
    # Check for arbitrage
    arbitrage_issues = []
    
    # Calendar spread arbitrage check
    for i in range(len(expiries) - 1):
        for j in range(len(strikes)):
            if vol_surface_data[i+1, j] < vol_surface_data[i, j]:
                arbitrage_issues.append(f"Calendar spread arbitrage detected at strike {strikes[j]:.1f}")
    
    # Butterfly arbitrage check (simplified)
    for i in range(len(expiries)):
        for j in range(1, len(strikes) - 1):
            # Check convexity
            vol_prev = vol_surface_data[i, j-1]
            vol_curr = vol_surface_data[i, j]
            vol_next = vol_surface_data[i, j+1]
            
            if 2 * vol_curr > vol_prev + vol_next:
                arbitrage_issues.append(f"Butterfly arbitrage detected at expiry {expiries[i]:.2f}, strike {strikes[j]:.1f}")
    
    # Create status message
    if arbitrage_issues:
        status_children = [
            html.H4("⚠️ Arbitrage Detected!", style={'color': '#e74c3c'}),
            html.Ul([html.Li(issue) for issue in arbitrage_issues[:5]])  # Show first 5 issues
        ]
    else:
        status_children = [
            html.H4("✅ No Arbitrage Detected", style={'color': '#27ae60'}),
            html.P("The volatility surface appears to be arbitrage-free.")
        ]
    
    # Create plot
    if plot_type == '3d':
        fig = go.Figure(data=[go.Surface(
            x=K, y=T, z=vol_surface_data,
            colorscale='Viridis',
            name='Volatility Surface'
        )])
        
        fig.update_layout(
            title='Volatility Surface (3D)',
            scene=dict(
                xaxis_title='Strike Price',
                yaxis_title='Time to Expiry',
                zaxis_title='Total Variance',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=800,
            height=600
        )
    else:
        fig = go.Figure(data=go.Contour(
            x=strikes,
            y=expiries,
            z=vol_surface_data,
            colorscale='Viridis',
            contours=dict(
                coloring='heatmap',
                showlabels=True
            )
        ))
        
        fig.update_layout(
            title='Volatility Surface (2D Contour)',
            xaxis_title='Strike Price',
            yaxis_title='Time to Expiry',
            width=800,
            height=600
        )
    
    # Calculate surface statistics
    stats_children = [
        html.Div([
            html.H4(f"{np.mean(vol_surface_data):.4f}", style={'color': '#3498db'}),
            html.P("Mean Variance")
        ]),
        html.Div([
            html.H4(f"{np.std(vol_surface_data):.4f}", style={'color': '#e67e22'}),
            html.P("Std Deviation")
        ]),
        html.Div([
            html.H4(f"{np.min(vol_surface_data):.4f}", style={'color': '#27ae60'}),
            html.P("Min Variance")
        ]),
        html.Div([
            html.H4(f"{np.max(vol_surface_data):.4f}", style={'color': '#e74c3c'}),
            html.P("Max Variance")
        ])
    ]
    
    return fig, status_children, stats_children

if __name__ == '__main__':
    print("Starting Volatility Surface Dashboard...")
    print("Access the dashboard at: http://127.0.0.1:8050")
    print("Press Ctrl+C to stop the server")
    app.run(debug=True, host='127.0.0.1', port=8050) 