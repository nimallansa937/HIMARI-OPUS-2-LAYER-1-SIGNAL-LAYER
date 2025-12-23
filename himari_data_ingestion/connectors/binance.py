"""
Binance WebSocket Connector

This is the PRIMARY data source for HIMARI. Binance offers:
- Free unlimited access (no API key needed for public data)
- <5ms latency (fastest among exchanges)
- 99.9%+ uptime reliability
- All major crypto pairs

The connector subscribes to kline (candlestick) streams and converts
them to the HIMARI OHLCV format before publishing to Kafka.

Usage:
    connector = BinanceConnector(symbols=["BTCUSDT", "ETHUSDT"])
    connector.set_callback(lambda msg: kafka_producer.send(msg))
    await connector.connect()
"""

import asyncio
import json
import time
import logging
from typing import Callable, List, Dict, Any, Optional
from dataclasses import dataclass
import websockets
from websockets.exceptions import ConnectionClosed

from .base import BaseConnector, ConnectionState

logger = logging.getLogger(__name__)


@dataclass
class BinanceKline:
    """Parsed Binance kline (candlestick) message."""
    symbol: str
    interval: str
    open_time: int
    close_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades: int
    is_closed: bool  # True when candle is finalized


class BinanceConnector(BaseConnector):
    """
    Binance WebSocket connector for real-time OHLCV data.
    
    Binance WebSocket streams are formatted as:
        wss://stream.binance.com:9443/ws/btcusdt@kline_1m
    
    For multiple symbols, we use the combined stream:
        wss://stream.binance.com:9443/stream?streams=btcusdt@kline_1m/ethusdt@kline_1m
    
    The connector automatically:
    - Reconnects on disconnection (exponential backoff)
    - Handles ping/pong to maintain connection
    - Normalizes messages to HIMARI format
    """
    
    EXCHANGE_NAME = "binance"
    BASE_URL = "wss://stream.binance.com:9443"
    
    # Binance limits: 5 messages/second, 1024 connections/IP
    MAX_SYMBOLS_PER_CONNECTION = 200
    
    def __init__(
        self,
        symbols: List[str],
        interval: str = "1m",
        callback: Optional[Callable[[Dict], None]] = None,
    ):
        """
        Initialize Binance connector.
        
        Args:
            symbols: List of symbols like ["BTCUSDT", "ETHUSDT"]
            interval: Kline interval (1m, 5m, 15m, 1h, etc.)
            callback: Function to call with each normalized message
        """
        super().__init__(callback)
        
        self.symbols = [s.lower() for s in symbols]  # Binance uses lowercase
        self.interval = interval
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        
        # Build stream URL
        streams = [f"{s}@kline_{interval}" for s in self.symbols]
        stream_param = "/".join(streams)
        self._url = f"{self.BASE_URL}/stream?streams={stream_param}"
        
        logger.info(f"BinanceConnector initialized for {len(symbols)} symbols")
    
    async def connect(self) -> None:
        """
        Connect to Binance WebSocket and start processing messages.
        
        This method runs indefinitely, automatically reconnecting on failures.
        """
        self._running = True
        self._state = ConnectionState.CONNECTING
        
        while self._running:
            try:
                logger.info(f"Connecting to Binance: {self._url[:80]}...")
                
                async with websockets.connect(
                    self._url,
                    ping_interval=20,  # Binance expects pings every 30s
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._ws = ws
                    self._state = ConnectionState.CONNECTED
                    self._reconnect_delay = 1.0  # Reset backoff on success
                    
                    logger.info("Connected to Binance WebSocket")
                    
                    # Process messages until disconnection
                    await self._process_messages(ws)
                    
            except ConnectionClosed as e:
                logger.warning(f"Binance connection closed: {e.code} {e.reason}")
                self._state = ConnectionState.DISCONNECTED
                
            except Exception as e:
                logger.error(f"Binance connection error: {e}")
                self._state = ConnectionState.ERROR
            
            # Reconnect with exponential backoff
            if self._running:
                self._state = ConnectionState.RECONNECTING
                await self._backoff()
    
    async def _process_messages(self, ws: websockets.WebSocketClientProtocol) -> None:
        """Process incoming WebSocket messages."""
        async for raw_message in ws:
            try:
                # Parse JSON
                data = json.loads(raw_message)
                
                # Binance combined stream wraps data in {"stream": "...", "data": {...}}
                if "stream" in data:
                    stream_name = data["stream"]
                    payload = data["data"]
                else:
                    # Single stream format
                    payload = data
                
                # Skip non-kline messages
                if payload.get("e") != "kline":
                    continue
                
                # Parse kline data
                kline = self._parse_kline(payload)
                
                # Normalize to HIMARI format
                normalized = self._normalize(kline)
                
                # Call callback
                if self._callback:
                    await self._safe_callback(normalized)
                
                # Update metrics
                self._message_count += 1
                self._last_message_time = time.time()
                
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON from Binance: {e}")
            except Exception as e:
                logger.error(f"Error processing Binance message: {e}")
    
    def _parse_kline(self, data: Dict[str, Any]) -> BinanceKline:
        """
        Parse Binance kline message.
        
        Binance kline format:
        {
            "e": "kline",
            "E": 1638747660000,  # Event time
            "s": "BTCUSDT",      # Symbol
            "k": {
                "t": 1638747600000,  # Kline start time
                "T": 1638747659999,  # Kline close time
                "s": "BTCUSDT",
                "i": "1m",           # Interval
                "o": "48000.00",     # Open
                "c": "48050.00",     # Close
                "h": "48100.00",     # High
                "l": "47950.00",     # Low
                "v": "123.456",      # Volume
                "q": "5925678.90",   # Quote volume
                "n": 1234,           # Number of trades
                "x": false           # Is closed?
            }
        }
        """
        k = data["k"]
        
        return BinanceKline(
            symbol=k["s"],
            interval=k["i"],
            open_time=int(k["t"]),
            close_time=int(k["T"]),
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
            quote_volume=float(k["q"]),
            trades=int(k["n"]),
            is_closed=k["x"],
        )
    
    def _normalize(self, kline: BinanceKline) -> Dict[str, Any]:
        """
        Convert Binance kline to HIMARI OHLCV format.
        
        This is the format your Flink pipeline expects.
        """
        return {
            "symbol": kline.symbol.upper(),
            "exchange": self.EXCHANGE_NAME,
            "timestamp": kline.open_time,
            "open": kline.open,
            "high": kline.high,
            "low": kline.low,
            "close": kline.close,
            "volume": kline.volume,
            "quote_volume": kline.quote_volume,
            "trades": kline.trades,
            "source": "websocket",
            "received_at": int(time.time() * 1000),
            "is_closed": kline.is_closed,
        }
    
    async def disconnect(self) -> None:
        """Gracefully disconnect from Binance."""
        self._running = False
        if self._ws:
            await self._ws.close()
        self._state = ConnectionState.DISCONNECTED
        logger.info("Disconnected from Binance")


class BinanceTradeConnector(BaseConnector):
    """
    Binance trade stream connector for tick-level data.
    
    Use this instead of kline if you need:
    - Individual trades (not aggregated candles)
    - Sub-second granularity
    - Order flow analysis
    
    Note: Generates much more data (~100-1000 messages/second for BTC)
    """
    
    EXCHANGE_NAME = "binance"
    BASE_URL = "wss://stream.binance.com:9443"
    
    def __init__(
        self,
        symbols: List[str],
        callback: Optional[Callable[[Dict], None]] = None,
    ):
        super().__init__(callback)
        self.symbols = [s.lower() for s in symbols]
        
        # Use aggTrade stream (aggregated trades - less noisy)
        streams = [f"{s}@aggTrade" for s in self.symbols]
        stream_param = "/".join(streams)
        self._url = f"{self.BASE_URL}/stream?streams={stream_param}"
        
        self._ws = None
        self._running = False
    
    async def connect(self) -> None:
        """Connect and process trade messages."""
        self._running = True
        
        while self._running:
            try:
                async with websockets.connect(self._url, ping_interval=20) as ws:
                    self._ws = ws
                    self._state = ConnectionState.CONNECTED
                    self._reconnect_delay = 1.0
                    
                    async for raw_message in ws:
                        data = json.loads(raw_message)
                        
                        if "stream" in data:
                            payload = data["data"]
                        else:
                            payload = data
                        
                        # Normalize trade to OHLCV-like format
                        normalized = {
                            "symbol": payload["s"].upper(),
                            "exchange": self.EXCHANGE_NAME,
                            "timestamp": payload["T"],  # Trade time
                            "price": float(payload["p"]),
                            "quantity": float(payload["q"]),
                            "is_buyer_maker": payload["m"],  # True = sell, False = buy
                            "trade_id": payload["a"],
                            "source": "websocket_trade",
                            "received_at": int(time.time() * 1000),
                        }
                        
                        if self._callback:
                            await self._safe_callback(normalized)
                        
                        self._message_count += 1
                        self._last_message_time = time.time()
                        
            except Exception as e:
                logger.error(f"Binance trade stream error: {e}")
                if self._running:
                    await self._backoff()
    
    async def disconnect(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
