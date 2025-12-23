"""
LOB Snapshot Manager

Manages live Limit Order Book state with O(1) updates.
Integrates Binance Order Book and Tick streams with OrderFlowFeatures.

Usage:
    manager = LOBSnapshotManager(symbols=["BTCUSDT"])
    manager.set_callback(on_features_update)
    await manager.start()  # Starts WebSocket connections
"""

import asyncio
import time
import logging
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass
import json

# Import connectors
from himari_data_ingestion.connectors import BinanceOrderBookConnector, BinanceTickConnector

# Import order flow features
from primitives.order_flow import OrderFlowFeatures

logger = logging.getLogger(__name__)


@dataclass
class SymbolState:
    """Per-symbol order flow state."""
    symbol: str
    features: OrderFlowFeatures
    last_orderbook_time: int = 0
    last_trade_time: int = 0
    orderbook_updates: int = 0
    trade_updates: int = 0


class LOBSnapshotManager:
    """
    Manages live order book state and produces order flow features.
    
    Combines:
    - Level 2 order book data (top 20 bids/asks)
    - Level 3 tick data (individual trades)
    
    Produces 10D order flow feature vector per update.
    
    Architecture:
        Binance WS (depth) ─┬─→ LOBSnapshotManager ─→ 10D Feature Vector
        Binance WS (trades)─┘
    """
    
    def __init__(
        self,
        symbols: List[str],
        orderbook_depth: int = 20,
        obi_depth: int = 5,
        callback: Optional[Callable[[str, Dict], None]] = None,
    ):
        """
        Initialize LOB manager.
        
        Args:
            symbols: Trading pairs ["BTCUSDT", "ETHUSDT"]
            orderbook_depth: Order book depth to track (5, 10, 20)
            obi_depth: Levels for OBI calculation
            callback: Handler for feature updates (symbol, features_dict)
        """
        self.symbols = [s.upper() for s in symbols]
        self.orderbook_depth = orderbook_depth
        self.obi_depth = obi_depth
        self._callback = callback
        
        # Per-symbol state
        self._states: Dict[str, SymbolState] = {
            symbol: SymbolState(
                symbol=symbol,
                features=OrderFlowFeatures(obi_depth=obi_depth)
            )
            for symbol in self.symbols
        }
        
        # Connectors
        self._orderbook_connector: Optional[BinanceOrderBookConnector] = None
        self._tick_connector: Optional[BinanceTickConnector] = None
        
        # Running state
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        logger.info(f"LOBSnapshotManager initialized for {len(symbols)} symbols")
    
    def set_callback(self, callback: Callable[[str, Dict], None]) -> None:
        """Set callback for feature updates."""
        self._callback = callback
    
    async def _on_orderbook_update(self, msg: Dict) -> None:
        """Handle order book update from Binance."""
        try:
            symbol = msg.get('symbol', '').upper()
            if symbol not in self._states:
                return
            
            state = self._states[symbol]
            
            # Extract bids and asks
            bids = msg.get('bids_top5', [])
            asks = msg.get('asks_top5', [])
            timestamp = msg.get('timestamp', int(time.time() * 1000))
            
            # Update features
            state.features.update_orderbook(bids, asks, timestamp)
            state.last_orderbook_time = timestamp
            state.orderbook_updates += 1
            
            # Emit callback
            if self._callback:
                features = state.features.get_feature_dict()
                features['symbol'] = symbol
                features['type'] = 'orderbook'
                features['timestamp'] = timestamp
                await self._safe_callback(symbol, features)
                
        except Exception as e:
            logger.error(f"Error processing orderbook: {e}")
    
    async def _on_trade_update(self, msg: Dict) -> None:
        """Handle trade/tick update from Binance."""
        try:
            symbol = msg.get('symbol', '').upper()
            if symbol not in self._states:
                return
            
            state = self._states[symbol]
            
            price = msg.get('price', 0)
            quantity = msg.get('quantity', 0)
            is_buyer_maker = msg.get('side', 'buy') == 'sell'  # seller taker = buyer maker
            timestamp = msg.get('timestamp', int(time.time() * 1000))
            
            # Update features
            state.features.update_trade(price, quantity, is_buyer_maker, timestamp)
            state.last_trade_time = timestamp
            state.trade_updates += 1
            
            # Emit callback (every N trades to reduce overhead)
            if state.trade_updates % 10 == 0 and self._callback:
                features = state.features.get_feature_dict()
                features['symbol'] = symbol
                features['type'] = 'trade'
                features['timestamp'] = timestamp
                features['cvd'] = state.features._cvd
                await self._safe_callback(symbol, features)
                
        except Exception as e:
            logger.error(f"Error processing trade: {e}")
    
    async def _safe_callback(self, symbol: str, features: Dict) -> None:
        """Safely invoke callback."""
        try:
            if asyncio.iscoroutinefunction(self._callback):
                await self._callback(symbol, features)
            else:
                self._callback(symbol, features)
        except Exception as e:
            logger.error(f"Callback error: {e}")
    
    async def start(self) -> None:
        """Start order book and trade streams."""
        self._running = True
        
        logger.info("Starting LOB Snapshot Manager...")
        
        # Create connectors
        self._orderbook_connector = BinanceOrderBookConnector(
            symbols=self.symbols,
            depth=self.orderbook_depth,
            callback=self._on_orderbook_update,
        )
        
        self._tick_connector = BinanceTickConnector(
            symbols=self.symbols,
            aggregate=True,  # Use aggTrade for less noise
            callback=self._on_trade_update,
        )
        
        # Start both streams
        orderbook_task = asyncio.create_task(
            self._orderbook_connector.connect(),
            name="orderbook_stream"
        )
        
        tick_task = asyncio.create_task(
            self._tick_connector.connect(),
            name="tick_stream"
        )
        
        self._tasks = [orderbook_task, tick_task]
        
        logger.info("LOB streams started")
        
        # Wait for all tasks
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("LOB tasks cancelled")
    
    async def stop(self) -> None:
        """Stop all streams."""
        self._running = False
        
        if self._orderbook_connector:
            await self._orderbook_connector.disconnect()
        
        if self._tick_connector:
            await self._tick_connector.disconnect()
        
        for task in self._tasks:
            task.cancel()
        
        logger.info("LOB Snapshot Manager stopped")
    
    def get_features(self, symbol: str) -> Optional[Dict]:
        """Get current features for a symbol."""
        if symbol not in self._states:
            return None
        
        return self._states[symbol].features.get_feature_dict()
    
    def get_feature_vector(self, symbol: str) -> Optional[List[float]]:
        """Get current 10D feature vector for a symbol."""
        if symbol not in self._states:
            return None
        
        return self._states[symbol].features.get_feature_vector().tolist()
    
    def get_stats(self) -> Dict[str, Dict]:
        """Get statistics for all symbols."""
        return {
            symbol: {
                'orderbook_updates': state.orderbook_updates,
                'trade_updates': state.trade_updates,
                'last_orderbook': state.last_orderbook_time,
                'last_trade': state.last_trade_time,
                **state.features.get_stats(),
            }
            for symbol, state in self._states.items()
        }


# Quick test
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    update_count = 0
    
    async def on_features(symbol: str, features: Dict):
        global update_count
        update_count += 1
        
        if update_count % 50 == 0:
            print(f"\n[{symbol}] Update #{update_count}")
            print(f"  OBI: {features.get('obi_current', 0):+.4f}")
            print(f"  CVD: {features.get('cvd', 0):+.2f}")
            print(f"  VPIN: {features.get('vpin', 0):.4f}")
            print(f"  Spread Z: {features.get('spread_zscore', 0):+.2f}")
    
    async def main():
        manager = LOBSnapshotManager(
            symbols=["BTCUSDT"],
            callback=on_features
        )
        
        try:
            await asyncio.wait_for(manager.start(), timeout=30)
        except asyncio.TimeoutError:
            print(f"\nTest completed: {update_count} updates received")
        finally:
            await manager.stop()
            
            print("\nFinal Stats:")
            for symbol, stats in manager.get_stats().items():
                print(f"  {symbol}: {stats['orderbook_updates']} OB, {stats['trade_updates']} trades")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        asyncio.run(main())
    else:
        print("LOB Snapshot Manager created.")
        print("Run with --test to test live connection:")
        print("  python lob_manager.py --test")
