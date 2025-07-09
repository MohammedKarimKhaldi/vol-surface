"""
Dash dashboard for Volatility Surface Construction and Arbitrage Detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from vol_surface import VolatilitySurface
import math

# Helper to generate sample data (reuse from examples)
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

# Initialize app
app = dash.Dash(__name__)
app.title = "Volatility Surface Dashboard"

# Layout
def serve_layout():
    return html.Div([
        html.H1("Volatility Surface Construction and Arbitrage Detection Dashboard"),
        html.Div([
            html.Button("Load Sample Data", id="load-sample-btn", n_clicks=0),
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Drag and Drop or ', html.A('Select CSV File')]),
                style={
                    'width': '30%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                    'textAlign': 'center', 'margin': '10px'
                },
                multiple=False
            ),
        ], style={'display': 'flex', 'gap': '20px'}),
        html.Div(id='data-table-div'),
        dcc.Tabs([
            dcc.Tab(label='SVI Surface', children=[dcc.Graph(id='svi-surface-plot')]),
            dcc.Tab(label='Spline Surface', children=[dcc.Graph(id='spline-surface-plot')]),
            dcc.Tab(label='Greeks', children=[
                dcc.Dropdown(
                    id='greek-dropdown',
                    options=[{'label': g.capitalize(), 'value': g} for g in ['deltas', 'gammas', 'vegas', 'thetas', 'rhos']],
                    value='deltas',
                    style={'width': '200px'}
                ),
                dcc.Graph(id='greek-surface-plot')
            ]),
            dcc.Tab(label='Arbitrage', children=[dcc.Graph(id='arbitrage-plot'), html.Div(id='arbitrage-table')]),
        ]),
        html.Div(id='hidden-data', style={'display': 'none'})
    ])
app.layout = serve_layout

# --- Callbacks ---
# Store data in dcc.Store or hidden div
from dash.dependencies import ClientsideFunction

def parse_contents(contents, filename):
    import base64
    import io
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return None
    except Exception as e:
        return None
    return df

@app.callback(
    Output('data-table-div', 'children'),
    Output('hidden-data', 'children'),
    Input('load-sample-btn', 'n_clicks'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def load_data(n_clicks, contents, filename):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'load-sample-btn':
        df = generate_sample_data()
    elif button_id == 'upload-data' and contents:
        df = parse_contents(contents, filename)
        if df is None:
            return html.Div("Failed to parse file."), ''
    else:
        raise dash.exceptions.PreventUpdate
    table = dash_table.DataTable(
        data=df.head(20).to_dict('records'),
        columns=[{"name": i, "id": i} for i in df.columns],
        page_size=20,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center'},
    )
    return table, df.to_json(date_format='iso', orient='split')

@app.callback(
    Output('svi-surface-plot', 'figure'),
    Input('hidden-data', 'children')
)
def update_svi_surface(json_data):
    if not json_data:
        return go.Figure()
    df = pd.read_json(json_data, orient='split')
    surface = VolatilitySurface(spot_price=100.0, risk_free_rate=0.02, dividend_yield=0.0)
    surface.load_data(df)
    surface.fit_svi_model()
    grid = surface.generate_surface_grid((80, 120), (0.25, 2.0), n_strikes=40, n_maturities=20, method='svi')
    fig = go.Figure(data=[go.Surface(z=grid['implied_vols'], x=grid['strikes'], y=grid['maturities'], colorscale='Viridis')])
    fig.update_layout(title='SVI Volatility Surface', scene={"xaxis_title": "Strike", "yaxis_title": "Maturity", "zaxis_title": "Implied Volatility"})
    return fig

@app.callback(
    Output('spline-surface-plot', 'figure'),
    Input('hidden-data', 'children')
)
def update_spline_surface(json_data):
    if not json_data:
        return go.Figure()
    df = pd.read_json(json_data, orient='split')
    surface = VolatilitySurface(spot_price=100.0, risk_free_rate=0.02, dividend_yield=0.0)
    surface.load_data(df)
    surface.fit_spline_surface(tension=0.1)
    grid = surface.generate_surface_grid((80, 120), (0.25, 2.0), n_strikes=40, n_maturities=20, method='spline')
    fig = go.Figure(data=[go.Surface(z=grid['implied_vols'], x=grid['strikes'], y=grid['maturities'], colorscale='Plasma')])
    fig.update_layout(title='Spline Volatility Surface', scene={"xaxis_title": "Strike", "yaxis_title": "Maturity", "zaxis_title": "Implied Volatility"})
    return fig

@app.callback(
    Output('greek-surface-plot', 'figure'),
    Input('hidden-data', 'children'),
    Input('greek-dropdown', 'value')
)
def update_greek_surface(json_data, greek):
    if not json_data:
        return go.Figure()
    df = pd.read_json(json_data, orient='split')
    surface = VolatilitySurface(spot_price=100.0, risk_free_rate=0.02, dividend_yield=0.0)
    surface.load_data(df)
    surface.fit_svi_model()
    K, T = np.meshgrid(np.linspace(80, 120, 40), np.linspace(0.25, 2.0, 20))
    K_flat, T_flat = K.flatten(), T.flatten()
    option_types = ['call'] * len(K_flat)
    greeks = surface.calculate_greeks(K_flat, T_flat, option_types)
    fig = go.Figure(data=[go.Surface(z=greeks[greek].reshape(K.shape), x=K, y=T, colorscale='Cividis')])
    fig.update_layout(title=f'{greek.capitalize()} Surface', scene={"xaxis_title": "Strike", "yaxis_title": "Maturity", "zaxis_title": greek.capitalize()})
    return fig

@app.callback(
    Output('arbitrage-plot', 'figure'),
    Output('arbitrage-table', 'children'),
    Input('hidden-data', 'children')
)
def update_arbitrage(json_data):
    if not json_data:
        return go.Figure(), html.Div()
    df = pd.read_json(json_data, orient='split')
    surface = VolatilitySurface(spot_price=100.0, risk_free_rate=0.02, dividend_yield=0.0)
    surface.load_data(df)
    surface.fit_svi_model()
    arbitrage = surface.detect_arbitrage()
    # Show violations for the first type with violations
    for arb_type, result in arbitrage.items():
        if result['violation_count'] > 0:
            violations = result['violations']
            strikes = [v.get('strike', 0) for v in violations]
            maturities = [v.get('maturity1', v.get('maturity', 0)) for v in violations]
            fig = go.Figure(data=[go.Histogram2d(x=strikes, y=maturities, colorscale='Reds')])
            fig.update_layout(title=f'{arb_type.capitalize()} Arbitrage Violations', xaxis_title='Strike', yaxis_title='Maturity')
            table = dash_table.DataTable(
                data=violations[:20],
                columns=[{"name": i, "id": i} for i in violations[0].keys()],
                page_size=20,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center'},
            )
            return fig, table
    # If no violations
    fig = go.Figure()
    fig.update_layout(title='No Arbitrage Violations Detected')
    return fig, html.Div("No arbitrage violations detected.")

if __name__ == "__main__":
    app.run_server(debug=True) 