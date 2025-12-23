"""
Polygon.io Crypto Connector

FREE tier available for crypto:
- Real-time WebSocket (requires API key)
- REST API for historical data
- Aggregates, trades, quotes

Get FREE API key: https://polygon.io/dashboard/signup

Usage:
    poly = PolygonCryptoConnector(api_key="YOUR_KEY")
    candles = poly.get_aggregates("X:BTCUSD", "minute", 100)
"""

import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import requests
import asyncio
import json

logger = logging.getLogger(__name__)


class PolygonCryptoConnector:
    """
    Polygon.io Crypto API connector.
    
    FREE Tier Limits:
    - 5 API calls/minute
    - End of day data
    - No real-time WebSocket
    
    Basic Tier ($29/month):
    - Unlimited REST calls
    - 15-min delayed WebSocket
    
    For real-time, consider Binance (free unlimited).
    Polygon is better for historical data and stocks.
    """
    
    BASE_URL = "https://api.polygon.io"
    WS_URL = "wss://socket.polygon.io/crypto"
    
    def __init__(self, api_key: str):
        """
        Initialize Polygon connector.
        
        Args:
            api_key: Polygon.io API key (free tier available)
        """
        self.api_key = api_key
        self.session = requests.Session()
        self._last_request_time = 0
    
    def _request(self, endpoint: str, params: Dict = None) -> Dict:
        """Rate-limited request to Polygon API."""
        # Free tier: 5 calls/minute = 12 seconds between calls
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < 12:  # 5 calls/min for free tier
            time.sleep(12 - time_since_last)
        
        params = params or {}
        params['apiKey'] = self.api_key
        
        url = f"{self.BASE_URL}{endpoint}"
        response = self.session.get(url, params=params)
        self._last_request_time = time.time()
        
        if response.status_code == 429:
            logger.warning("Polygon rate limited, waiting 60s")
            time.sleep(60)
            return self._request(endpoint, params)
        
        response.raise_for_status()
        return response.json()
    
    # =========================================================================
    # AGGREGATES (OHLCV)
    # =========================================================================
    
    def get_aggregates(
        self,
        ticker: str,
        timespan: str = "minute",
        multiplier: int = 1,
        from_date: str = None,
        to_date: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get aggregated OHLCV bars.
        
        Args:
            ticker: Crypto ticker (e.g., "X:BTCUSD", "X:ETHUSD")
            timespan: minute, hour, day, week, month
            multiplier: Size of timespan (e.g., 5 for 5-minute bars)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            limit: Max results
            
        Returns:
            List of OHLCV bars
        """
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        if to_date is None:
            to_date = datetime.now().strftime("%Y-%m-%d")
        
        endpoint = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        
        data = self._request(endpoint, {'limit': limit, 'sort': 'desc'})
        
        results = data.get('results', [])
        
        return [
            {
                'timestamp': bar['t'],
                'open': bar['o'],
                'high': bar['h'],
                'low': bar['l'],
                'close': bar['c'],
                'volume': bar['v'],
                'vwap': bar.get('vw', 0),
                'trades': bar.get('n', 0),
            }
            for bar in results
        ]
    
    def get_last_trade(self, ticker: str) -> Dict:
        """
        Get last trade for a crypto pair.
        
        Args:
            ticker: e.g., "X:BTCUSD"
            
        Returns:
            Last trade data
        """
        endpoint = f"/v1/last/crypto/{ticker}"
        data = self._request(endpoint)
        
        last = data.get('last', {})
        
        return {
            'price': last.get('price', 0),
            'size': last.get('size', 0),
            'exchange': last.get('exchange', ''),
            'timestamp': last.get('timestamp', 0),
        }
    
    def get_previous_close(self, ticker: str) -> Dict:
        """
        Get previous day's OHLCV.
        
        Args:
            ticker: e.g., "X:BTCUSD"
            
        Returns:
            Previous day data
        """
        endpoint = f"/v2/aggs/ticker/{ticker}/prev"
        data = self._request(endpoint)
        
        results = data.get('results', [])
        if not results:
            return {}
        
        bar = results[0]
        return {
            'timestamp': bar['t'],
            'open': bar['o'],
            'high': bar['h'],
            'low': bar['l'],
            'close': bar['c'],
            'volume': bar['v'],
            'vwap': bar.get('vw', 0),
        }
    
    # =========================================================================
    # TICKER INFO
    # =========================================================================
    
    def get_ticker_details(self, ticker: str) -> Dict:
        """
        Get ticker details.
        
        Args:
            ticker: e.g., "X:BTCUSD"
            
        Returns:
            Ticker metadata
        """
        endpoint = f"/v3/reference/tickers/{ticker}"
        data = self._request(endpoint)
        
        return data.get('results', {})
    
    def list_crypto_tickers(self, limit: int = 100) -> List[Dict]:
        """
        List available crypto tickers.
        
        Returns:
            List of crypto tickers
        """
        endpoint = "/v3/reference/tickers"
        data = self._request(endpoint, {
            'market': 'crypto',
            'active': 'true',
            'limit': limit,
        })
        
        return data.get('results', [])
    
    # =========================================================================
    # SNAPSHOT
    # =========================================================================
    
    def get_snapshot(self, ticker: str) -> Dict:
        """
        Get current snapshot of a ticker.
        
        Args:
            ticker: e.g., "X:BTCUSD"
            
        Returns:
            Current snapshot with day stats
        """
        # Remove X: prefix for snapshot endpoint
        symbol = ticker.replace("X:", "")
        endpoint = f"/v2/snapshot/locale/global/markets/crypto/tickers/{symbol}"
        
        data = self._request(endpoint)
        
        ticker_data = data.get('ticker', {})
        
        return {
            'ticker': ticker_data.get('ticker', ''),
            'last_price': ticker_data.get('lastTrade', {}).get('p', 0),
            'last_size': ticker_data.get('lastTrade', {}).get('s', 0),
            'bid': ticker_data.get('lastQuote', {}).get('b', 0),
            'ask': ticker_data.get('lastQuote', {}).get('a', 0),
            'day_open': ticker_data.get('day', {}).get('o', 0),
            'day_high': ticker_data.get('day', {}).get('h', 0),
            'day_low': ticker_data.get('day', {}).get('l', 0),
            'day_close': ticker_data.get('day', {}).get('c', 0),
            'day_volume': ticker_data.get('day', {}).get('v', 0),
            'day_vwap': ticker_data.get('day', {}).get('vw', 0),
            'prev_close': ticker_data.get('prevDay', {}).get('c', 0),
            'change_pct': ticker_data.get('todaysChangePerc', 0),
        }
    
    def get_all_snapshots(self) -> List[Dict]:
        """
        Get snapshots for all crypto tickers.
        
        Returns:
            List of all crypto snapshots
        """
        endpoint = "/v2/snapshot/locale/global/markets/crypto/tickers"
        data = self._request(endpoint)
        
        tickers = data.get('tickers', [])
        
        return [
            {
                'ticker': t.get('ticker', ''),
                'last_price': t.get('lastTrade', {}).get('p', 0),
                'day_volume': t.get('day', {}).get('v', 0),
                'change_pct': t.get('todaysChangePerc', 0),
            }
            for t in tickers
        ]


# Symbol mapping
POLYGON_SYMBOLS = {
    'BTCUSDT': 'X:BTCUSD',
    'ETHUSDT': 'X:ETHUSD',
    'SOLUSDT': 'X:SOLUSD',
    'BNBUSDT': 'X:BNBUSD',
    'XRPUSDT': 'X:XRPUSD',
    'ADAUSDT': 'X:ADAUSD',
    'DOGEUSDT': 'X:DOGEUSD',
    'DOTUSDT': 'X:DOTUSD',
    'MATICUSDT': 'X:MATICUSD',
    'LINKUSDT': 'X:LINKUSD',
}


# Quick test
if __name__ == "__main__":
    import os
    
    api_key = os.getenv("POLYGON_API_KEY", "")
    
    if not api_key:
        print("Polygon.io Crypto Connector")
        print("=" * 40)
        print()
        print("Get FREE API key at: https://polygon.io/dashboard/signup")
        print()
        print("Free tier: 5 calls/min, end-of-day data")
        print("For real-time, use Binance (free unlimited)")
        print()
        print("Set key and test:")
        print('  $env:POLYGON_API_KEY="your-key"')
        print("  python polygon_connector.py")
    else:
        poly = PolygonCryptoConnector(api_key=api_key)
        
        print("Testing Polygon Crypto API...")
        
        # Get snapshot
        print("\nFetching BTC snapshot...")
        try:
            snapshot = poly.get_snapshot("X:BTCUSD")
            print(f"BTC Price: ${snapshot.get('last_price', 0):,.2f}")
            print(f"24h Change: {snapshot.get('change_pct', 0):+.2f}%")
            print(f"Day Volume: ${snapshot.get('day_volume', 0):,.0f}")
        except Exception as e:
            print(f"Snapshot error: {e}")
        
        # Get aggregates
        print("\nFetching 5 recent 1-hour bars...")
        try:
            bars = poly.get_aggregates("X:BTCUSD", "hour", 1, limit=5)
            for bar in bars[:5]:
                print(f"  {bar['timestamp']}: O={bar['open']:.2f} C={bar['close']:.2f}")
        except Exception as e:
            print(f"Aggregates error: {e}")
        
        print("\n✓ Polygon connector working!")
