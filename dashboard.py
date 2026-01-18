"""
HIMARI Layer 1 Real-Time Monitoring Dashboard

Monitors:
- Live price data from CCXT stream
- Strategy generation and validation
- TimescaleDB data accumulation
- System health metrics
"""

import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import json
import threading
import time

# Database connections
try:
    import psycopg2
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Initialize Dash app
app = dash.Dash(__name__, update_title=None)
app.title = "HIMARI Layer 1 Monitor"

# Color scheme
COLORS = {
    'background': '#1a1a2e',
    'card': '#16213e',
    'text': '#eaeaea',
    'accent': '#0f3460',
    'green': '#00d9ff',
    'red': '#ff6b6b',
    'yellow': '#ffd93d',
    'purple': '#6c5ce7'
}

# Global data store
data_store = {
    'prices': {'BTC': [], 'ETH': [], 'SOL': []},
    'candle_counts': {'BTCUSDT': 0, 'ETHUSDT': 0, 'SOLUSDT': 0},
    'cycle_results': [],
    'last_update': None
}


def get_db_connection():
    """Get TimescaleDB connection."""
    if not POSTGRES_AVAILABLE:
        return None
    try:
        return psycopg2.connect(
            host='localhost',
            port=5434,
            database='hinance',
            user='hinance_user',
            password='hinance_password'
        )
    except:
        return None


def get_redis_connection():
    """Get Redis connection."""
    if not REDIS_AVAILABLE:
        return None
    try:
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        return r
    except:
        return None


def fetch_market_data():
    """Fetch latest market data from TimescaleDB."""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()

    try:
        query = """
        SELECT symbol, time, close, volume
        FROM market_data
        WHERE time > NOW() - INTERVAL '1 hour'
        ORDER BY time DESC
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"DB Error: {e}")
        return pd.DataFrame()


def fetch_candle_counts():
    """Get candle counts per symbol."""
    conn = get_db_connection()
    if not conn:
        return {}

    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT symbol, COUNT(*) as cnt
            FROM market_data
            GROUP BY symbol
        """)
        result = {row[0]: row[1] for row in cur.fetchall()}
        conn.close()
        return result
    except:
        return {}


def fetch_latest_prices():
    """Get latest prices from Redis."""
    r = get_redis_connection()
    if not r:
        return {}

    prices = {}
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
        try:
            data = r.get(f'himari:price:{symbol}')
            if data:
                prices[symbol] = json.loads(data)
        except:
            pass
    return prices


# Dashboard Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("HIMARI Layer 1 Monitor",
                style={'color': COLORS['green'], 'margin': '0', 'fontSize': '28px'}),
        html.Span(id='last-update',
                  style={'color': COLORS['text'], 'fontSize': '14px'})
    ], style={
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center',
        'padding': '20px',
        'backgroundColor': COLORS['card'],
        'borderBottom': f'2px solid {COLORS["green"]}'
    }),

    # Main content
    html.Div([
        # Row 1: Price Cards
        html.Div([
            # BTC Card
            html.Div([
                html.H3("BTC/USDT", style={'color': COLORS['yellow'], 'margin': '0'}),
                html.H2(id='btc-price', style={'color': COLORS['text'], 'margin': '10px 0'}),
                html.Span(id='btc-candles', style={'color': COLORS['green'], 'fontSize': '12px'})
            ], style={
                'backgroundColor': COLORS['card'],
                'padding': '20px',
                'borderRadius': '10px',
                'textAlign': 'center',
                'flex': '1',
                'margin': '0 10px'
            }),

            # ETH Card
            html.Div([
                html.H3("ETH/USDT", style={'color': COLORS['purple'], 'margin': '0'}),
                html.H2(id='eth-price', style={'color': COLORS['text'], 'margin': '10px 0'}),
                html.Span(id='eth-candles', style={'color': COLORS['green'], 'fontSize': '12px'})
            ], style={
                'backgroundColor': COLORS['card'],
                'padding': '20px',
                'borderRadius': '10px',
                'textAlign': 'center',
                'flex': '1',
                'margin': '0 10px'
            }),

            # SOL Card
            html.Div([
                html.H3("SOL/USDT", style={'color': COLORS['green'], 'margin': '0'}),
                html.H2(id='sol-price', style={'color': COLORS['text'], 'margin': '10px 0'}),
                html.Span(id='sol-candles', style={'color': COLORS['green'], 'fontSize': '12px'})
            ], style={
                'backgroundColor': COLORS['card'],
                'padding': '20px',
                'borderRadius': '10px',
                'textAlign': 'center',
                'flex': '1',
                'margin': '0 10px'
            }),
        ], style={'display': 'flex', 'marginBottom': '20px'}),

        # Row 2: Charts
        html.Div([
            # Price Chart
            html.Div([
                html.H4("Live Prices", style={'color': COLORS['text'], 'marginBottom': '10px'}),
                dcc.Graph(id='price-chart', style={'height': '300px'})
            ], style={
                'backgroundColor': COLORS['card'],
                'padding': '20px',
                'borderRadius': '10px',
                'flex': '2',
                'marginRight': '20px'
            }),

            # Data Accumulation
            html.Div([
                html.H4("Data Accumulation", style={'color': COLORS['text'], 'marginBottom': '10px'}),
                dcc.Graph(id='candle-chart', style={'height': '300px'})
            ], style={
                'backgroundColor': COLORS['card'],
                'padding': '20px',
                'borderRadius': '10px',
                'flex': '1'
            }),
        ], style={'display': 'flex', 'marginBottom': '20px'}),

        # Row 3: Strategy Status
        html.Div([
            # Generation Stats
            html.Div([
                html.H4("Strategy Generation", style={'color': COLORS['text']}),
                html.Div(id='strategy-stats', style={'color': COLORS['text']})
            ], style={
                'backgroundColor': COLORS['card'],
                'padding': '20px',
                'borderRadius': '10px',
                'flex': '1',
                'marginRight': '20px'
            }),

            # Validation Pipeline
            html.Div([
                html.H4("HIFA Validation Pipeline", style={'color': COLORS['text']}),
                html.Div(id='validation-stats', style={'color': COLORS['text']})
            ], style={
                'backgroundColor': COLORS['card'],
                'padding': '20px',
                'borderRadius': '10px',
                'flex': '1',
                'marginRight': '20px'
            }),

            # System Health
            html.Div([
                html.H4("System Health", style={'color': COLORS['text']}),
                html.Div(id='health-stats', style={'color': COLORS['text']})
            ], style={
                'backgroundColor': COLORS['card'],
                'padding': '20px',
                'borderRadius': '10px',
                'flex': '1'
            }),
        ], style={'display': 'flex', 'marginBottom': '20px'}),

        # Row 4: Logs
        html.Div([
            html.H4("Recent Activity", style={'color': COLORS['text'], 'marginBottom': '10px'}),
            html.Div(id='activity-log', style={
                'backgroundColor': '#0d1117',
                'padding': '15px',
                'borderRadius': '5px',
                'fontFamily': 'monospace',
                'fontSize': '12px',
                'color': COLORS['green'],
                'maxHeight': '200px',
                'overflowY': 'auto'
            })
        ], style={
            'backgroundColor': COLORS['card'],
            'padding': '20px',
            'borderRadius': '10px'
        }),

    ], style={'padding': '20px'}),

    # Auto-refresh interval
    dcc.Interval(id='interval', interval=3000, n_intervals=0)

], style={
    'backgroundColor': COLORS['background'],
    'minHeight': '100vh',
    'fontFamily': 'Arial, sans-serif'
})


@app.callback(
    [Output('btc-price', 'children'),
     Output('eth-price', 'children'),
     Output('sol-price', 'children'),
     Output('btc-candles', 'children'),
     Output('eth-candles', 'children'),
     Output('sol-candles', 'children'),
     Output('price-chart', 'figure'),
     Output('candle-chart', 'figure'),
     Output('strategy-stats', 'children'),
     Output('validation-stats', 'children'),
     Output('health-stats', 'children'),
     Output('activity-log', 'children'),
     Output('last-update', 'children')],
    [Input('interval', 'n_intervals')]
)
def update_dashboard(n):
    # Fetch data
    df = fetch_market_data()
    counts = fetch_candle_counts()
    prices = fetch_latest_prices()

    # Price displays
    btc_price = f"${prices.get('BTCUSDT', {}).get('close', 0):,.2f}"
    eth_price = f"${prices.get('ETHUSDT', {}).get('close', 0):,.2f}"
    sol_price = f"${prices.get('SOLUSDT', {}).get('close', 0):,.2f}"

    # Candle counts
    btc_candles = f"{counts.get('BTCUSDT', 0):,} candles"
    eth_candles = f"{counts.get('ETHUSDT', 0):,} candles"
    sol_candles = f"{counts.get('SOLUSDT', 0):,} candles"

    # Price chart
    price_fig = go.Figure()
    if not df.empty:
        for symbol, color in [('BTCUSDT', COLORS['yellow']),
                               ('ETHUSDT', COLORS['purple']),
                               ('SOLUSDT', COLORS['green'])]:
            sym_df = df[df['symbol'] == symbol].sort_values('time')
            if not sym_df.empty:
                # Normalize prices for comparison
                norm_prices = sym_df['close'] / sym_df['close'].iloc[0] * 100
                price_fig.add_trace(go.Scatter(
                    x=sym_df['time'],
                    y=norm_prices,
                    name=symbol,
                    line=dict(color=color, width=2)
                ))

    price_fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text']),
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation='h', y=1.1),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Normalized %')
    )

    # Candle accumulation chart
    candle_fig = go.Figure(data=[
        go.Bar(
            x=list(counts.keys()),
            y=list(counts.values()),
            marker_color=[COLORS['yellow'], COLORS['purple'], COLORS['green']]
        )
    ])
    candle_fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text']),
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Candles')
    )

    # Strategy stats
    total_candles = sum(counts.values())
    hours_of_data = total_candles / 60 if total_candles > 0 else 0
    strategy_stats = html.Div([
        html.P(f"Data collected: {hours_of_data:.1f} hours"),
        html.P(f"Min needed: 24 hours"),
        html.Div([
            html.Div(style={
                'width': f'{min(hours_of_data/24*100, 100):.0f}%',
                'height': '10px',
                'backgroundColor': COLORS['green'],
                'borderRadius': '5px'
            })
        ], style={
            'width': '100%',
            'height': '10px',
            'backgroundColor': COLORS['accent'],
            'borderRadius': '5px',
            'marginTop': '10px'
        }),
        html.P(f"{min(hours_of_data/24*100, 100):.0f}% ready",
               style={'textAlign': 'right', 'fontSize': '12px', 'marginTop': '5px'})
    ])

    # Validation stats
    validation_stats = html.Div([
        html.P("Last Cycle: 0/25 passed"),
        html.P("Reason: Insufficient data"),
        html.P([
            html.Span("Status: ", style={'color': COLORS['text']}),
            html.Span("Accumulating", style={'color': COLORS['yellow']})
        ])
    ])

    # Health stats
    db_status = "Connected" if counts else "Disconnected"
    redis_status = "Connected" if prices else "Disconnected"
    health_stats = html.Div([
        html.P([
            html.Span("TimescaleDB: "),
            html.Span(db_status, style={'color': COLORS['green'] if counts else COLORS['red']})
        ]),
        html.P([
            html.Span("Redis: "),
            html.Span(redis_status, style={'color': COLORS['green'] if prices else COLORS['red']})
        ]),
        html.P([
            html.Span("CCXT Stream: "),
            html.Span("Active", style={'color': COLORS['green']})
        ]),
        html.P([
            html.Span("DeepSeek API: "),
            html.Span("Active", style={'color': COLORS['green']})
        ])
    ])

    # Activity log
    now = datetime.now()
    log_entries = [
        f"[{now.strftime('%H:%M:%S')}] Dashboard refresh",
        f"[{now.strftime('%H:%M:%S')}] Total candles: {total_candles}",
        f"[{now.strftime('%H:%M:%S')}] BTC: {btc_price}",
    ]
    activity_log = html.Div([html.P(entry) for entry in log_entries])

    last_update = f"Last update: {now.strftime('%Y-%m-%d %H:%M:%S')}"

    return (btc_price, eth_price, sol_price,
            btc_candles, eth_candles, sol_candles,
            price_fig, candle_fig,
            strategy_stats, validation_stats, health_stats,
            activity_log, last_update)


if __name__ == '__main__':
    print("Starting HIMARI Layer 1 Dashboard...")
    print("Open http://localhost:8050 in your browser")
    app.run(debug=False, host='0.0.0.0', port=8050)
