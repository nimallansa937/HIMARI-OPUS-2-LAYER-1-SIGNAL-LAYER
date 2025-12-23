"""
Kraken WebSocket Connector

Backup data source for HIMARI. Kraken offers:
- Free unlimited access
- Good for non-Binance pairs
- Slightly higher latency than Binance (~10ms)
- Different symbol format (e.g., XBT/USDT instead of BTCUSDT)

Usage:
    connector = KrakenConnector(symbols=["BTCUSDT", "ETHUSDT"])
    connector.set_callback(lambda msg: kafka_producer.send(msg))
    await connector.connect()
"""

import asyncio
import json
import time
import logging
from typing import Callable, List, Dict, Any, Optional
import websockets
from websockets.exceptions import ConnectionClosed

from .base import BaseConnector, ConnectionState

logger = logging.getLogger(__name__)


# Kraken uses different symbol formats
SYMBOL_MAP = {
    "BTCUSDT": "XBT/USDT",
    "ETHUSDT": "ETH/USDT",
    "SOLUSDT": "SOL/USDT",
    "XRPUSDT": "XRP/USDT",
    "ADAUSDT": "ADA/USDT",
    "DOGEUSDT": "DOGE/USDT",
    "DOTUSDT": "DOT/USDT",
    "LINKUSDT": "LINK/USDT",
    "MATICUSDT": "MATIC/USDT",
    "AVAXUSDT": "AVAX/USDT",
}

# Reverse mapping for normalization
REVERSE_MAP = {v: k for k, v in SYMBOL_MAP.items()}


class KrakenConnector(BaseConnector):
    """
    Kraken WebSocket connector for real-time OHLC data.
    
    Kraken WebSocket uses a subscription model:
    1. Connect to wss://ws.kraken.com
    2. Send subscription message for desired pairs
    3. Receive streaming data
    
    The connector automatically:
    - Maps HIMARI symbols to Kraken format
    - Reconnects on disconnection
    - Normalizes messages to HIMARI format
    """
    
    EXCHANGE_NAME = "kraken"
    WS_URL = "wss://ws.kraken.com"
    
    def __init__(
        self,
        symbols: List[str],
        interval: int = 1,  # 1 minute
        callback: Optional[Callable[[Dict], None]] = None,
    ):
        """
        Initialize Kraken connector.
        
        Args:
            symbols: List of HIMARI symbols like ["BTCUSDT", "ETHUSDT"]
            interval: OHLC interval in minutes (1, 5, 15, 30, 60, etc.)
            callback: Function to call with each normalized message
        """
        super().__init__(callback)
        
        # Map HIMARI symbols to Kraken format
        self.himari_symbols = symbols
        self.kraken_pairs = []
        for sym in symbols:
            kraken_sym = SYMBOL_MAP.get(sym.upper())
            if kraken_sym:
                self.kraken_pairs.append(kraken_sym)
            else:
                logger.warning(f"Symbol {sym} not available on Kraken")
        
        self.interval = interval
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._subscribed = False
        
        logger.info(f"KrakenConnector initialized for {len(self.kraken_pairs)} pairs")
    
    async def connect(self) -> None:
        """
        Connect to Kraken WebSocket and start processing messages.
        """
        if not self.kraken_pairs:
            logger.error("No valid Kraken pairs to subscribe to")
            return
        
        self._running = True
        self._state = ConnectionState.CONNECTING
        
        while self._running:
            try:
                logger.info(f"Connecting to Kraken: {self.WS_URL}")
                
                async with websockets.connect(
                    self.WS_URL,
                    ping_interval=30,
                    ping_timeout=10,
                ) as ws:
                    self._ws = ws
                    self._state = ConnectionState.CONNECTED
                    self._reconnect_delay = 1.0
                    
                    # Subscribe to OHLC stream
                    await self._subscribe(ws)
                    
                    logger.info("Connected to Kraken WebSocket")
                    
                    # Process messages
                    await self._process_messages(ws)
                    
            except ConnectionClosed as e:
                logger.warning(f"Kraken connection closed: {e.code}")
                self._state = ConnectionState.DISCONNECTED
                self._subscribed = False
                
            except Exception as e:
                logger.error(f"Kraken connection error: {e}")
                self._state = ConnectionState.ERROR
            
            if self._running:
                self._state = ConnectionState.RECONNECTING
                await self._backoff()
    
    async def _subscribe(self, ws: websockets.WebSocketClientProtocol) -> None:
        """Send subscription request for OHLC data."""
        subscription = {
            "event": "subscribe",
            "pair": self.kraken_pairs,
            "subscription": {
                "name": "ohlc",
                "interval": self.interval
            }
        }
        
        await ws.send(json.dumps(subscription))
        logger.info(f"Subscribed to Kraken OHLC for: {self.kraken_pairs}")
        self._subscribed = True
    
    async def _process_messages(self, ws: websockets.WebSocketClientProtocol) -> None:
        """Process incoming WebSocket messages."""
        async for raw_message in ws:
            try:
                data = json.loads(raw_message)
                
                # Skip system messages
                if isinstance(data, dict):
                    event = data.get("event")
                    if event in ["systemStatus", "subscriptionStatus", "heartbeat"]:
                        continue
                
                # OHLC data comes as array: [channelID, [time, etime, open, high, low, close, vwap, volume, count], channelName, pair]
                if isinstance(data, list) and len(data) >= 4:
                    channel_name = data[2] if len(data) > 2 else ""
                    
                    if "ohlc" in str(channel_name):
                        ohlc_data = data[1]
                        pair = data[3]
                        
                        normalized = self._normalize(ohlc_data, pair)
                        
                        if self._callback:
                            await self._safe_callback(normalized)
                        
                        self._message_count += 1
                        self._last_message_time = time.time()
                
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON from Kraken: {e}")
            except Exception as e:
                logger.error(f"Error processing Kraken message: {e}")
    
    def _normalize(self, ohlc: List, pair: str) -> Dict[str, Any]:
        """
        Convert Kraken OHLC to HIMARI format.
        
        Kraken OHLC array: [time, etime, open, high, low, close, vwap, volume, count]
        """
        # Map Kraken pair back to HIMARI symbol
        himari_symbol = REVERSE_MAP.get(pair, pair.replace("/", ""))
        
        return {
            "symbol": himari_symbol,
            "exchange": self.EXCHANGE_NAME,
            "timestamp": int(float(ohlc[0]) * 1000),  # Convert to ms
            "open": float(ohlc[2]),
            "high": float(ohlc[3]),
            "low": float(ohlc[4]),
            "close": float(ohlc[5]),
            "volume": float(ohlc[7]),
            "quote_volume": float(ohlc[5]) * float(ohlc[7]),  # Approximate
            "trades": int(ohlc[8]) if len(ohlc) > 8 else 0,
            "source": "websocket",
            "received_at": int(time.time() * 1000),
            "vwap": float(ohlc[6]) if len(ohlc) > 6 else None,
        }
    
    async def disconnect(self) -> None:
        """Gracefully disconnect from Kraken."""
        self._running = False
        
        if self._ws and self._subscribed:
            # Unsubscribe
            unsubscribe = {
                "event": "unsubscribe",
                "pair": self.kraken_pairs,
                "subscription": {"name": "ohlc"}
            }
            try:
                await self._ws.send(json.dumps(unsubscribe))
            except:
                pass
        
        if self._ws:
            await self._ws.close()
        
        self._state = ConnectionState.DISCONNECTED
        logger.info("Disconnected from Kraken")


# Quick test
if __name__ == "__main__":
    async def test_callback(msg):
        print(f"[{msg['exchange']}] {msg['symbol']}: ${msg['close']:.2f}")
    
    async def main():
        connector = KrakenConnector(
            symbols=["BTCUSDT", "ETHUSDT"],
            callback=test_callback
        )
        
        try:
            await asyncio.wait_for(connector.connect(), timeout=30)
        except asyncio.TimeoutError:
            print("Test completed")
        finally:
            await connector.disconnect()
    
    asyncio.run(main())
