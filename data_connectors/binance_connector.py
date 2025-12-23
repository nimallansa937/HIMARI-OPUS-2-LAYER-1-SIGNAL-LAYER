"""
Binance API Connector

FREE unlimited access to:
- REST API: OHLCV, ticker, depth, trades
- WebSocket: Real-time streams (trades, klines, depth)

No API key required for public endpoints.
"""

import json
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import requests

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

logger = logging.getLogger(__name__)


class BinanceConnector:
    """
    Binance REST API connector for market data.
    
    FREE unlimited access - no API key required.
    
    Usage:
        connector = BinanceConnector()
        ticker = connector.get_ticker("BTCUSDT")
        klines = connector.get_klines("BTCUSDT", "1m", limit=100)
    """
    
    BASE_URL = "https://api.binance.com/api/v3"
    FUTURES_URL = "https://fapi.binance.com/fapi/v1"
    
    def __init__(self, use_futures: bool = False):
        """
        Initialize Binance connector.
        
        Args:
            use_futures: Use futures API instead of spot
        """
        self.base_url = self.FUTURES_URL if use_futures else self.BASE_URL
        self.session = requests.Session()
        self._rate_limit_remaining = 1200  # Default weight limit
        self._last_request_time = 0
    
    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make rate-limited request to Binance API."""
        # Simple rate limiting (1200 weight/min)
        current_time = time.time()
        if current_time - self._last_request_time < 0.05:  # 50ms minimum between requests
            time.sleep(0.05)
        
        url = f"{self.base_url}/{endpoint}"
        response = self.session.get(url, params=params or {})
        self._last_request_time = time.time()
        
        if response.status_code == 429:
            logger.warning("Rate limited by Binance, waiting 60s")
            time.sleep(60)
            return self._request(endpoint, params)
        
        response.raise_for_status()
        return response.json()
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get 24hr ticker for symbol.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            
        Returns:
            Dict with price, volume, change data
        """
        data = self._request("ticker/24hr", {"symbol": symbol})
        return {
            'symbol': data['symbol'],
            'price': float(data['lastPrice']),
            'volume': float(data['volume']),
            'quote_volume': float(data['quoteVolume']),
            'price_change': float(data['priceChange']),
            'price_change_pct': float(data['priceChangePercent']),
            'high': float(data['highPrice']),
            'low': float(data['lowPrice']),
            'open': float(data['openPrice']),
            'close': float(data['lastPrice']),
            'timestamp': data['closeTime'],
        }
    
    def get_klines(
        self, 
        symbol: str, 
        interval: str = "1m",
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get OHLCV candlestick data.
        
        Args:
            symbol: Trading pair
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            limit: Number of candles (max 1000)
            start_time: Start timestamp in ms
            end_time: End timestamp in ms
            
        Returns:
            List of OHLCV dicts
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': min(limit, 1000),
        }
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        data = self._request("klines", params)
        
        return [
            {
                'timestamp': kline[0],
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5]),
                'close_time': kline[6],
                'quote_volume': float(kline[7]),
                'trades': int(kline[8]),
                'taker_buy_volume': float(kline[9]),
                'taker_buy_quote_volume': float(kline[10]),
            }
            for kline in data
        ]
    
    def get_depth(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """
        Get order book depth.
        
        Args:
            symbol: Trading pair
            limit: Depth limit (5, 10, 20, 50, 100, 500, 1000, 5000)
            
        Returns:
            Dict with bids and asks
        """
        data = self._request("depth", {"symbol": symbol, "limit": limit})
        return {
            'bids': [[float(p), float(q)] for p, q in data['bids']],
            'asks': [[float(p), float(q)] for p, q in data['asks']],
            'last_update_id': data['lastUpdateId'],
        }
    
    def get_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades."""
        data = self._request("trades", {"symbol": symbol, "limit": limit})
        return [
            {
                'id': t['id'],
                'price': float(t['price']),
                'qty': float(t['qty']),
                'time': t['time'],
                'is_buyer_maker': t['isBuyerMaker'],
            }
            for t in data
        ]
    
    def get_exchange_info(self) -> Dict:
        """Get exchange trading rules and symbols."""
        return self._request("exchangeInfo")
    
    def get_all_tickers(self) -> List[Dict]:
        """Get all symbol tickers."""
        data = self._request("ticker/price")
        return [{'symbol': t['symbol'], 'price': float(t['price'])} for t in data]


class BinanceWebSocket:
    """
    Binance WebSocket connector for real-time data.
    
    FREE unlimited streaming.
    
    Usage:
        async def on_trade(data):
            print(f"Trade: {data}")
        
        ws = BinanceWebSocket()
        await ws.subscribe_trades("BTCUSDT", on_trade)
        await ws.run()
    """
    
    WS_URL = "wss://stream.binance.com:9443/ws"
    FUTURES_WS_URL = "wss://fstream.binance.com/ws"
    
    def __init__(self, use_futures: bool = False):
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets package required: pip install websockets")
        
        self.ws_url = self.FUTURES_WS_URL if use_futures else self.WS_URL
        self._subscriptions: Dict[str, Callable] = {}
        self._running = False
        self._ws = None
    
    def subscribe_trades(self, symbol: str, callback: Callable):
        """Subscribe to trade stream."""
        stream = f"{symbol.lower()}@trade"
        self._subscriptions[stream] = callback
    
    def subscribe_klines(self, symbol: str, interval: str, callback: Callable):
        """Subscribe to kline/candlestick stream."""
        stream = f"{symbol.lower()}@kline_{interval}"
        self._subscriptions[stream] = callback
    
    def subscribe_depth(self, symbol: str, callback: Callable, levels: int = 20):
        """Subscribe to order book depth stream."""
        stream = f"{symbol.lower()}@depth{levels}@100ms"
        self._subscriptions[stream] = callback
    
    def subscribe_ticker(self, symbol: str, callback: Callable):
        """Subscribe to mini ticker stream."""
        stream = f"{symbol.lower()}@miniTicker"
        self._subscriptions[stream] = callback
    
    async def run(self):
        """Run WebSocket connection."""
        if not self._subscriptions:
            raise ValueError("No subscriptions configured")
        
        streams = "/".join(self._subscriptions.keys())
        url = f"{self.ws_url}/{streams}"
        
        self._running = True
        logger.info(f"Connecting to Binance WebSocket: {len(self._subscriptions)} streams")
        
        async with websockets.connect(url) as ws:
            self._ws = ws
            while self._running:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=30)
                    data = json.loads(message)
                    
                    # Find callback for this stream
                    stream_type = data.get('e', '')
                    symbol = data.get('s', '').lower()
                    
                    for stream_name, callback in self._subscriptions.items():
                        if symbol in stream_name:
                            await self._call_callback(callback, data)
                            break
                            
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    await ws.ping()
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket connection closed, reconnecting...")
                    break
    
    async def _call_callback(self, callback: Callable, data: Dict):
        """Call callback, handling both sync and async."""
        if asyncio.iscoroutinefunction(callback):
            await callback(data)
        else:
            callback(data)
    
    def stop(self):
        """Stop WebSocket connection."""
        self._running = False


# Quick test
if __name__ == "__main__":
    connector = BinanceConnector()
    
    print("Testing Binance Connector...")
    ticker = connector.get_ticker("BTCUSDT")
    print(f"BTC Price: ${ticker['price']:,.2f}")
    
    klines = connector.get_klines("BTCUSDT", "1h", limit=5)
    print(f"Last 5 hourly candles: {len(klines)}")
    
    depth = connector.get_depth("BTCUSDT", limit=5)
    print(f"Top bid: ${float(depth['bids'][0][0]):,.2f}")
    print(f"Top ask: ${float(depth['asks'][0][0]):,.2f}")
    
    print("✓ Binance connector working!")
