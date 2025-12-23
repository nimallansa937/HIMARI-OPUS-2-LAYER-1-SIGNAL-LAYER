"""
Binance Tick/Trade (Level 3) Connector

FREE unlimited access via Binance WebSocket API:
- Every executed trade in real-time
- Aggregated trades (aggTrade) for reduced noise
- Buyer/seller identification

This provides:
- Tick-by-tick price movement
- Trade flow (buy vs sell volume)
- VWAP calculation
- CVD (Cumulative Volume Delta)

Usage:
    connector = BinanceTickConnector(symbols=["BTCUSDT"])
    connector.set_callback(my_handler)
    await connector.connect()
"""

import asyncio
import json
import time
import logging
from typing import Callable, List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import deque
import websockets
from websockets.exceptions import ConnectionClosed

from .base import BaseConnector, ConnectionState

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Single trade/tick data."""
    symbol: str
    trade_id: int
    timestamp: int
    price: float
    quantity: float
    is_buyer_maker: bool  # True = seller aggressive, False = buyer aggressive
    
    @property
    def side(self) -> str:
        """Trade side: 'buy' if buyer was taker, 'sell' if seller was taker."""
        return "sell" if self.is_buyer_maker else "buy"
    
    @property
    def value_usd(self) -> float:
        """Trade value in quote currency (usually USDT)."""
        return self.price * self.quantity


class TradeAggregator:
    """
    Aggregates tick data into useful metrics.
    
    Calculates:
    - VWAP (Volume Weighted Average Price)
    - CVD (Cumulative Volume Delta)
    - Trade flow imbalance
    - Tick velocity
    """
    
    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: Number of trades to keep in rolling window
        """
        self.window_size = window_size
        self.trades: deque = deque(maxlen=window_size)
        
        # Running totals
        self.total_buy_volume = 0.0
        self.total_sell_volume = 0.0
        self.total_trades = 0
        self.cvd = 0.0  # Cumulative Volume Delta
    
    def add_trade(self, trade: Trade) -> None:
        """Add a new trade to the aggregator."""
        self.trades.append(trade)
        self.total_trades += 1
        
        # Update CVD
        if trade.side == "buy":
            self.total_buy_volume += trade.quantity
            self.cvd += trade.quantity
        else:
            self.total_sell_volume += trade.quantity
            self.cvd -= trade.quantity
    
    def get_vwap(self) -> float:
        """Calculate VWAP over the window."""
        if not self.trades:
            return 0.0
        
        total_value = sum(t.price * t.quantity for t in self.trades)
        total_volume = sum(t.quantity for t in self.trades)
        
        return total_value / total_volume if total_volume > 0 else 0.0
    
    def get_trade_flow_imbalance(self) -> float:
        """
        Calculate trade flow imbalance.
        
        Returns: -1.0 (all sells) to +1.0 (all buys)
        """
        if not self.trades:
            return 0.0
        
        buy_vol = sum(t.quantity for t in self.trades if t.side == "buy")
        sell_vol = sum(t.quantity for t in self.trades if t.side == "sell")
        
        total = buy_vol + sell_vol
        if total == 0:
            return 0.0
        
        return (buy_vol - sell_vol) / total
    
    def get_tick_velocity(self) -> float:
        """Calculate trades per second over the window."""
        if len(self.trades) < 2:
            return 0.0
        
        time_span = (self.trades[-1].timestamp - self.trades[0].timestamp) / 1000
        if time_span <= 0:
            return 0.0
        
        return len(self.trades) / time_span
    
    def get_avg_trade_size(self) -> float:
        """Average trade size in window."""
        if not self.trades:
            return 0.0
        
        return sum(t.quantity for t in self.trades) / len(self.trades)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all aggregated metrics."""
        return {
            "vwap": self.get_vwap(),
            "cvd": self.cvd,
            "trade_flow_imbalance": self.get_trade_flow_imbalance(),
            "tick_velocity": self.get_tick_velocity(),
            "avg_trade_size": self.get_avg_trade_size(),
            "total_trades": self.total_trades,
            "total_buy_volume": self.total_buy_volume,
            "total_sell_volume": self.total_sell_volume,
        }


class BinanceTickConnector(BaseConnector):
    """
    Binance Tick/Trade (Level 3) WebSocket connector.
    
    Subscribes to aggTrade stream for real-time executed trades.
    
    aggTrade stream provides:
    - Trade ID
    - Price and quantity
    - Buyer/seller maker flag
    - Timestamp
    
    This is FREE with no API key required.
    """
    
    EXCHANGE_NAME = "binance"
    BASE_URL = "wss://stream.binance.com:9443"
    
    def __init__(
        self,
        symbols: List[str],
        aggregate: bool = True,  # Use aggTrade vs trade
        window_size: int = 100,  # Aggregation window
        callback: Optional[Callable[[Dict], None]] = None,
    ):
        """
        Initialize tick connector.
        
        Args:
            symbols: Trading pairs ["BTCUSDT", "ETHUSDT"]
            aggregate: Use aggTrade (True) or raw trade (False)
            window_size: Number of trades for rolling metrics
            callback: Handler for tick updates
        """
        super().__init__(callback)
        
        self.symbols = [s.lower() for s in symbols]
        self.aggregate = aggregate
        
        self._ws = None
        self._running = False
        
        # Build stream URL
        stream_type = "aggTrade" if aggregate else "trade"
        streams = [f"{s}@{stream_type}" for s in self.symbols]
        stream_param = "/".join(streams)
        self._url = f"{self.BASE_URL}/stream?streams={stream_param}"
        
        # Aggregators per symbol
        self._aggregators: Dict[str, TradeAggregator] = {
            s: TradeAggregator(window_size) for s in self.symbols
        }
        
        logger.info(f"BinanceTickConnector: {len(symbols)} symbols, aggregate={aggregate}")
    
    async def connect(self) -> None:
        """Connect to Binance WebSocket and stream trade data."""
        self._running = True
        self._state = ConnectionState.CONNECTING
        
        while self._running:
            try:
                logger.info(f"Connecting to Binance Tick stream...")
                
                async with websockets.connect(
                    self._url,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    self._ws = ws
                    self._state = ConnectionState.CONNECTED
                    self._reconnect_delay = 1.0
                    
                    logger.info("Connected to Binance Tick stream")
                    
                    await self._process_messages(ws)
                    
            except ConnectionClosed as e:
                logger.warning(f"Tick connection closed: {e.code}")
                self._state = ConnectionState.DISCONNECTED
                
            except Exception as e:
                logger.error(f"Tick stream error: {e}")
                self._state = ConnectionState.ERROR
            
            if self._running:
                self._state = ConnectionState.RECONNECTING
                await self._backoff()
    
    async def _process_messages(self, ws) -> None:
        """Process incoming trade messages."""
        async for raw_message in ws:
            try:
                data = json.loads(raw_message)
                
                # Combined stream format
                if "stream" in data:
                    payload = data["data"]
                else:
                    payload = data
                
                # Parse trade
                trade = self._parse_trade(payload)
                
                # Add to aggregator
                symbol_lower = trade.symbol.lower()
                if symbol_lower in self._aggregators:
                    self._aggregators[symbol_lower].add_trade(trade)
                
                # Normalize to HIMARI format
                normalized = self._normalize(trade, symbol_lower)
                
                if self._callback:
                    await self._safe_callback(normalized)
                
                self._message_count += 1
                self._last_message_time = time.time()
                
            except Exception as e:
                logger.error(f"Error processing tick: {e}")
    
    def _parse_trade(self, data: Dict) -> Trade:
        """Parse Binance aggTrade message."""
        return Trade(
            symbol=data.get("s", "").upper(),
            trade_id=data.get("a", data.get("t", 0)),  # aggTrade uses 'a', trade uses 't'
            timestamp=data.get("T", int(time.time() * 1000)),
            price=float(data.get("p", 0)),
            quantity=float(data.get("q", 0)),
            is_buyer_maker=data.get("m", False),
        )
    
    def _normalize(self, trade: Trade, symbol_lower: str) -> Dict[str, Any]:
        """Convert to HIMARI format with aggregated metrics."""
        agg = self._aggregators.get(symbol_lower)
        metrics = agg.get_metrics() if agg else {}
        
        return {
            "symbol": trade.symbol,
            "exchange": self.EXCHANGE_NAME,
            "timestamp": trade.timestamp,
            "type": "tick",
            "trade_id": trade.trade_id,
            "price": trade.price,
            "quantity": trade.quantity,
            "side": trade.side,
            "value_usd": trade.value_usd,
            # Aggregated metrics
            "vwap": metrics.get("vwap", 0),
            "cvd": metrics.get("cvd", 0),
            "trade_flow_imbalance": metrics.get("trade_flow_imbalance", 0),
            "tick_velocity": metrics.get("tick_velocity", 0),
            "avg_trade_size": metrics.get("avg_trade_size", 0),
            "received_at": int(time.time() * 1000),
        }
    
    def get_aggregator(self, symbol: str) -> Optional[TradeAggregator]:
        """Get trade aggregator for a symbol."""
        return self._aggregators.get(symbol.lower())
    
    async def disconnect(self) -> None:
        """Disconnect from Binance."""
        self._running = False
        if self._ws:
            await self._ws.close()
        self._state = ConnectionState.DISCONNECTED
        logger.info("Disconnected from Binance Tick stream")


# Quick test
if __name__ == "__main__":
    trade_count = 0
    
    async def test_callback(msg):
        global trade_count
        trade_count += 1
        
        if trade_count % 20 == 0:  # Print every 20th trade
            print(f"[{msg['symbol']}] {msg['side'].upper():4s} "
                  f"${msg['price']:,.2f} x {msg['quantity']:.4f} | "
                  f"CVD: {msg['cvd']:+.3f} | "
                  f"Flow: {msg['trade_flow_imbalance']:+.3f} | "
                  f"Vel: {msg['tick_velocity']:.1f}/s")
    
    async def main():
        connector = BinanceTickConnector(
            symbols=["BTCUSDT"],
            aggregate=True,
            callback=test_callback
        )
        
        try:
            await asyncio.wait_for(connector.connect(), timeout=15)
        except asyncio.TimeoutError:
            print(f"\nTest completed: {trade_count} trades received")
        finally:
            await connector.disconnect()
    
    asyncio.run(main())
