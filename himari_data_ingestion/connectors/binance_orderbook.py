"""
Binance Order Book (Level 2) Connector

FREE unlimited access via Binance WebSocket API:
- 20 depth levels (best 20 bids + 20 asks)
- 100ms update frequency
- Real-time order book snapshots

This provides:
- Order Book Imbalance (OBI)
- Bid-Ask Spread
- Liquidity depth
- Support/Resistance levels

Usage:
    connector = BinanceOrderBookConnector(symbols=["BTCUSDT"])
    connector.set_callback(my_handler)
    await connector.connect()
"""

import asyncio
import json
import time
import logging
from typing import Callable, List, Dict, Any, Optional
from dataclasses import dataclass, field
import websockets
from websockets.exceptions import ConnectionClosed

from .base import BaseConnector, ConnectionState

logger = logging.getLogger(__name__)


@dataclass
class OrderBookSnapshot:
    """Order book snapshot with bid/ask levels."""
    symbol: str
    timestamp: int
    bids: List[List[float]]  # [[price, quantity], ...]
    asks: List[List[float]]  # [[price, quantity], ...]
    
    @property
    def best_bid(self) -> float:
        """Best bid price."""
        return self.bids[0][0] if self.bids else 0.0
    
    @property
    def best_ask(self) -> float:
        """Best ask price."""
        return self.asks[0][0] if self.asks else 0.0
    
    @property
    def mid_price(self) -> float:
        """Mid-market price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return 0.0
    
    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.best_ask - self.best_bid
    
    @property
    def spread_pct(self) -> float:
        """Spread as percentage of mid price."""
        if self.mid_price > 0:
            return (self.spread / self.mid_price) * 100
        return 0.0
    
    def get_order_book_imbalance(self, depth: int = 5) -> float:
        """
        Calculate Order Book Imbalance (OBI).
        
        OBI = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
        Returns: -1.0 (sell pressure) to +1.0 (buy pressure)
        """
        bid_vol = sum(b[1] for b in self.bids[:depth])
        ask_vol = sum(a[1] for a in self.asks[:depth])
        
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        
        return (bid_vol - ask_vol) / total
    
    def get_depth_at_price(self, price_pct: float = 1.0) -> Dict[str, float]:
        """
        Get total volume within X% of mid price.
        
        Args:
            price_pct: Percentage from mid price (e.g., 1.0 = ±1%)
            
        Returns:
            Dict with bid_depth and ask_depth in base currency
        """
        mid = self.mid_price
        if mid == 0:
            return {'bid_depth': 0, 'ask_depth': 0}
        
        threshold_up = mid * (1 + price_pct / 100)
        threshold_down = mid * (1 - price_pct / 100)
        
        bid_depth = sum(b[1] for b in self.bids if b[0] >= threshold_down)
        ask_depth = sum(a[1] for a in self.asks if a[0] <= threshold_up)
        
        return {
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
        }


class BinanceOrderBookConnector(BaseConnector):
    """
    Binance Order Book (Level 2) WebSocket connector.
    
    Subscribes to depth stream for real-time order book updates.
    
    Streams:
    - depth@100ms: Order book snapshot every 100ms
    - depth20@100ms: Top 20 levels every 100ms (lighter)
    
    This is FREE with no API key required.
    """
    
    EXCHANGE_NAME = "binance"
    BASE_URL = "wss://stream.binance.com:9443"
    
    def __init__(
        self,
        symbols: List[str],
        depth: int = 20,  # 5, 10, or 20
        update_speed: str = "100ms",  # 100ms or 1000ms
        callback: Optional[Callable[[Dict], None]] = None,
    ):
        """
        Initialize order book connector.
        
        Args:
            symbols: Trading pairs ["BTCUSDT", "ETHUSDT"]
            depth: Depth levels (5, 10, or 20)
            update_speed: Update frequency (100ms or 1000ms)
            callback: Handler for order book updates
        """
        super().__init__(callback)
        
        self.symbols = [s.lower() for s in symbols]
        self.depth = depth
        self.update_speed = update_speed
        
        self._ws = None
        self._running = False
        
        # Build stream URL
        streams = [f"{s}@depth{depth}@{update_speed}" for s in self.symbols]
        stream_param = "/".join(streams)
        self._url = f"{self.BASE_URL}/stream?streams={stream_param}"
        
        # Store latest snapshots
        self._order_books: Dict[str, OrderBookSnapshot] = {}
        
        logger.info(f"BinanceOrderBookConnector: {len(symbols)} symbols, depth={depth}")
    
    async def connect(self) -> None:
        """Connect to Binance WebSocket and stream order book data."""
        self._running = True
        self._state = ConnectionState.CONNECTING
        
        while self._running:
            try:
                logger.info(f"Connecting to Binance Order Book stream...")
                
                async with websockets.connect(
                    self._url,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    self._ws = ws
                    self._state = ConnectionState.CONNECTED
                    self._reconnect_delay = 1.0
                    
                    logger.info("Connected to Binance Order Book")
                    
                    await self._process_messages(ws)
                    
            except ConnectionClosed as e:
                logger.warning(f"Order book connection closed: {e.code}")
                self._state = ConnectionState.DISCONNECTED
                
            except Exception as e:
                logger.error(f"Order book error: {e}")
                self._state = ConnectionState.ERROR
            
            if self._running:
                self._state = ConnectionState.RECONNECTING
                await self._backoff()
    
    async def _process_messages(self, ws) -> None:
        """Process incoming order book messages."""
        async for raw_message in ws:
            try:
                data = json.loads(raw_message)
                
                # Combined stream format
                if "stream" in data:
                    stream_name = data["stream"]
                    payload = data["data"]
                else:
                    payload = data
                
                # Parse order book
                snapshot = self._parse_order_book(payload)
                
                # Store latest
                self._order_books[snapshot.symbol] = snapshot
                
                # Normalize to HIMARI format
                normalized = self._normalize(snapshot)
                
                if self._callback:
                    await self._safe_callback(normalized)
                
                self._message_count += 1
                self._last_message_time = time.time()
                
            except Exception as e:
                logger.error(f"Error processing order book: {e}")
    
    def _parse_order_book(self, data: Dict) -> OrderBookSnapshot:
        """Parse Binance depth stream message."""
        # Extract symbol from stream name or assume single symbol
        symbol = self.symbols[0].upper() if len(self.symbols) == 1 else ""
        
        return OrderBookSnapshot(
            symbol=symbol,
            timestamp=int(time.time() * 1000),
            bids=[[float(p), float(q)] for p, q in data.get("bids", [])],
            asks=[[float(p), float(q)] for p, q in data.get("asks", [])],
        )
    
    def _normalize(self, snapshot: OrderBookSnapshot) -> Dict[str, Any]:
        """Convert to HIMARI format."""
        return {
            "symbol": snapshot.symbol,
            "exchange": self.EXCHANGE_NAME,
            "timestamp": snapshot.timestamp,
            "type": "order_book",
            "best_bid": snapshot.best_bid,
            "best_ask": snapshot.best_ask,
            "mid_price": snapshot.mid_price,
            "spread": snapshot.spread,
            "spread_pct": snapshot.spread_pct,
            "order_book_imbalance": snapshot.get_order_book_imbalance(5),
            "bid_depth_1pct": snapshot.get_depth_at_price(1.0)['bid_depth'],
            "ask_depth_1pct": snapshot.get_depth_at_price(1.0)['ask_depth'],
            "bids_top5": snapshot.bids[:5],
            "asks_top5": snapshot.asks[:5],
            "received_at": int(time.time() * 1000),
        }
    
    def get_order_book(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Get latest order book for a symbol."""
        return self._order_books.get(symbol.lower())
    
    async def disconnect(self) -> None:
        """Disconnect from Binance."""
        self._running = False
        if self._ws:
            await self._ws.close()
        self._state = ConnectionState.DISCONNECTED
        logger.info("Disconnected from Binance Order Book")


# Quick test
if __name__ == "__main__":
    async def test_callback(msg):
        print(f"[{msg['symbol']}] Mid: ${msg['mid_price']:,.2f} | "
              f"Spread: {msg['spread_pct']:.4f}% | "
              f"OBI: {msg['order_book_imbalance']:+.3f}")
    
    async def main():
        connector = BinanceOrderBookConnector(
            symbols=["BTCUSDT"],
            depth=20,
            callback=test_callback
        )
        
        try:
            await asyncio.wait_for(connector.connect(), timeout=30)
        except asyncio.TimeoutError:
            print("Test completed")
        finally:
            await connector.disconnect()
    
    asyncio.run(main())
