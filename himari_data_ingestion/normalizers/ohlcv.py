"""
OHLCV Data Normalizer

Converts exchange-specific message formats to the unified HIMARI OHLCV schema.
This ensures downstream components (Flink, Signal Layer) receive consistent data.

The HIMARI OHLCV schema is designed for:
- Low latency signal processing
- Quality validation
- Feature extraction
"""

import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class OHLCVMessage:
    """
    Unified HIMARI OHLCV message format.
    
    This is the canonical format used throughout the HIMARI pipeline.
    All exchange connectors must normalize their data to this format.
    """
    symbol: str           # Trading pair (e.g., "BTCUSDT")
    exchange: str         # Source exchange (e.g., "binance", "kraken")
    timestamp: int        # Candle open time in milliseconds
    open: float           # Opening price
    high: float           # Highest price
    low: float            # Lowest price
    close: float          # Closing price (or last trade price)
    volume: float         # Base asset volume
    quote_volume: float   # Quote asset volume (USD equivalent)
    trades: int           # Number of trades (0 if unavailable)
    source: str           # Data source type ("websocket", "rest")
    received_at: int      # Ingestion timestamp in milliseconds
    
    # Optional fields
    is_closed: Optional[bool] = None  # True when candle is finalized
    vwap: Optional[float] = None      # Volume-weighted average price
    market_cap: Optional[float] = None
    change_24h_pct: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Kafka serialization."""
        d = asdict(self)
        # Remove None values to reduce message size
        return {k: v for k, v in d.items() if v is not None}
    
    def validate(self) -> bool:
        """Validate message integrity."""
        # Basic sanity checks
        if not self.symbol or not self.exchange:
            return False
        if self.close <= 0:
            return False
        if self.volume < 0:
            return False
        if self.high < self.low:
            return False
        if self.timestamp <= 0:
            return False
        return True


class OHLCVNormalizer:
    """
    Normalizes exchange-specific messages to HIMARI OHLCV format.
    
    Handles quirks from different exchanges:
    - Binance: lowercase symbols, specific kline format
    - Kraken: XBT instead of BTC, slash separators
    - CoinGecko: REST-based, no real OHLC
    """
    
    # Symbol normalization mappings
    SYMBOL_ALIASES = {
        "XBT": "BTC",
        "XXBT": "BTC",
        "XETH": "ETH",
    }
    
    def __init__(self, default_exchange: str = "unknown"):
        self.default_exchange = default_exchange
        self._message_count = 0
        self._validation_errors = 0
    
    def normalize(self, raw_message: Dict[str, Any]) -> Optional[OHLCVMessage]:
        """
        Normalize any exchange message to OHLCV format.
        
        Args:
            raw_message: Raw message from exchange connector
            
        Returns:
            OHLCVMessage if valid, None if invalid
        """
        try:
            # Extract and normalize symbol
            symbol = self._normalize_symbol(
                raw_message.get("symbol", "UNKNOWN")
            )
            
            # Build OHLCV message
            msg = OHLCVMessage(
                symbol=symbol,
                exchange=raw_message.get("exchange", self.default_exchange),
                timestamp=int(raw_message.get("timestamp", time.time() * 1000)),
                open=float(raw_message.get("open", raw_message.get("price", 0))),
                high=float(raw_message.get("high", raw_message.get("price", 0))),
                low=float(raw_message.get("low", raw_message.get("price", 0))),
                close=float(raw_message.get("close", raw_message.get("price", 0))),
                volume=float(raw_message.get("volume", 0)),
                quote_volume=float(raw_message.get("quote_volume", 0)),
                trades=int(raw_message.get("trades", 0)),
                source=raw_message.get("source", "unknown"),
                received_at=int(raw_message.get("received_at", time.time() * 1000)),
                is_closed=raw_message.get("is_closed"),
                vwap=raw_message.get("vwap"),
                market_cap=raw_message.get("market_cap"),
                change_24h_pct=raw_message.get("change_24h_pct"),
            )
            
            # Validate
            if not msg.validate():
                self._validation_errors += 1
                logger.warning(f"Invalid OHLCV message: {symbol}")
                return None
            
            self._message_count += 1
            return msg
            
        except Exception as e:
            logger.error(f"Normalization error: {e}")
            self._validation_errors += 1
            return None
    
    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol to standard format (e.g., BTCUSDT).
        
        Handles:
        - Lowercase: btcusdt → BTCUSDT
        - Slashes: BTC/USDT → BTCUSDT
        - Aliases: XBT/USD → BTCUSD
        """
        # Uppercase and remove slashes
        symbol = symbol.upper().replace("/", "").replace("-", "")
        
        # Apply aliases
        for alias, standard in self.SYMBOL_ALIASES.items():
            if symbol.startswith(alias):
                symbol = standard + symbol[len(alias):]
        
        return symbol
    
    def get_stats(self) -> Dict[str, int]:
        """Get normalization statistics."""
        return {
            "messages_normalized": self._message_count,
            "validation_errors": self._validation_errors,
        }


def normalize_message(raw: Dict[str, Any], exchange: str = "unknown") -> Optional[Dict[str, Any]]:
    """
    Convenience function to normalize a single message.
    
    Args:
        raw: Raw message from exchange
        exchange: Exchange name
        
    Returns:
        Normalized dict or None if invalid
    """
    normalizer = OHLCVNormalizer(default_exchange=exchange)
    msg = normalizer.normalize(raw)
    return msg.to_dict() if msg else None


# Quick test
if __name__ == "__main__":
    # Test with sample messages
    test_messages = [
        {
            "symbol": "btcusdt",
            "exchange": "binance",
            "timestamp": 1703289600000,
            "open": 43250.50,
            "high": 43312.00,
            "low": 43198.25,
            "close": 43275.00,
            "volume": 1234.567,
            "quote_volume": 53456789.12,
            "trades": 15234,
            "source": "websocket",
            "received_at": 1703289600005,
        },
        {
            "symbol": "XBT/USDT",
            "exchange": "kraken",
            "timestamp": 1703289600000,
            "price": 43275.00,
            "volume": 100.5,
            "source": "websocket",
        },
    ]
    
    normalizer = OHLCVNormalizer()
    
    for raw in test_messages:
        msg = normalizer.normalize(raw)
        if msg:
            print(f"✓ {msg.exchange} {msg.symbol}: ${msg.close:,.2f}")
        else:
            print(f"✗ Failed to normalize: {raw}")
    
    print(f"\nStats: {normalizer.get_stats()}")
