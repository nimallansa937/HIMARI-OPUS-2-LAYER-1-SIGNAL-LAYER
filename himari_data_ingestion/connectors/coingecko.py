"""
CoinGecko REST Poller

Backup/supplemental data source for HIMARI. CoinGecko offers:
- Free tier: 10,000 calls/month, 30 calls/minute
- Coverage of 10M+ tokens (far more than exchanges)
- Good for tokens not on Binance/Kraken
- Market cap, volume, and sentiment data

This is a REST poller (not WebSocket) - polls at configurable interval.

Usage:
    poller = CoinGeckoPoller(symbols=["BTCUSDT", "ETHUSDT"])
    poller.set_callback(lambda msg: kafka_producer.send(msg))
    await poller.start()
"""

import asyncio
import time
import logging
from typing import Callable, List, Dict, Any, Optional
import aiohttp

from .base import BaseConnector, ConnectionState

logger = logging.getLogger(__name__)


# Map HIMARI symbols to CoinGecko IDs
COINGECKO_IDS = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "SOLUSDT": "solana",
    "BNBUSDT": "binancecoin",
    "XRPUSDT": "ripple",
    "ADAUSDT": "cardano",
    "DOGEUSDT": "dogecoin",
    "DOTUSDT": "polkadot",
    "LINKUSDT": "chainlink",
    "MATICUSDT": "matic-network",
    "AVAXUSDT": "avalanche-2",
    "UNIUSDT": "uniswap",
    "ATOMUSDT": "cosmos",
    "LTCUSDT": "litecoin",
    "SHIBUSDT": "shiba-inu",
}


class CoinGeckoPoller(BaseConnector):
    """
    CoinGecko REST API poller for market data.
    
    Unlike WebSocket connectors, this polls at intervals.
    Use as backup or for tokens not available on exchanges.
    
    Rate Limits (free tier):
    - 30 calls/minute
    - 10,000 calls/month
    
    We batch multiple symbols in one call to save quota.
    """
    
    EXCHANGE_NAME = "coingecko"
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    def __init__(
        self,
        symbols: List[str],
        poll_interval: float = 60.0,  # seconds
        api_key: Optional[str] = None,
        callback: Optional[Callable[[Dict], None]] = None,
    ):
        """
        Initialize CoinGecko poller.
        
        Args:
            symbols: List of HIMARI symbols
            poll_interval: Seconds between polls (min 60 for free tier)
            api_key: Optional API key for higher rate limits
            callback: Function to call with each message
        """
        super().__init__(callback)
        
        # Map symbols to CoinGecko IDs
        self.himari_symbols = symbols
        self.coin_ids = []
        self.id_to_symbol = {}
        
        for sym in symbols:
            cg_id = COINGECKO_IDS.get(sym.upper())
            if cg_id:
                self.coin_ids.append(cg_id)
                self.id_to_symbol[cg_id] = sym.upper()
            else:
                logger.warning(f"Symbol {sym} not mapped for CoinGecko")
        
        self.poll_interval = max(poll_interval, 30.0)  # Minimum 30s
        self.api_key = api_key
        self._running = False
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_prices: Dict[str, float] = {}
        
        logger.info(f"CoinGeckoPoller initialized for {len(self.coin_ids)} coins")
    
    async def connect(self) -> None:
        """Start polling CoinGecko API."""
        if not self.coin_ids:
            logger.error("No valid CoinGecko IDs to poll")
            return
        
        self._running = True
        self._state = ConnectionState.CONNECTING
        
        # Create aiohttp session
        headers = {}
        if self.api_key:
            headers["x-cg-demo-api-key"] = self.api_key
        
        self._session = aiohttp.ClientSession(headers=headers)
        self._state = ConnectionState.CONNECTED
        
        logger.info(f"Starting CoinGecko polling (interval: {self.poll_interval}s)")
        
        try:
            while self._running:
                try:
                    await self._poll()
                except Exception as e:
                    logger.error(f"CoinGecko poll error: {e}")
                    self._state = ConnectionState.ERROR
                
                # Wait for next poll
                await asyncio.sleep(self.poll_interval)
                
        finally:
            if self._session:
                await self._session.close()
    
    async def _poll(self) -> None:
        """Fetch latest prices from CoinGecko."""
        # Batch all coins in one request
        ids_param = ",".join(self.coin_ids)
        url = f"{self.BASE_URL}/simple/price"
        params = {
            "ids": ids_param,
            "vs_currencies": "usd",
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
            "include_market_cap": "true",
            "include_last_updated_at": "true",
        }
        
        async with self._session.get(url, params=params) as response:
            if response.status == 429:
                logger.warning("CoinGecko rate limit hit, backing off")
                await asyncio.sleep(60)
                return
            
            response.raise_for_status()
            data = await response.json()
        
        self._state = ConnectionState.CONNECTED
        now = int(time.time() * 1000)
        
        for coin_id, price_data in data.items():
            symbol = self.id_to_symbol.get(coin_id, coin_id.upper())
            price = price_data.get("usd", 0)
            
            # Get previous price for OHLC approximation
            prev_price = self._last_prices.get(coin_id, price)
            self._last_prices[coin_id] = price
            
            # Create OHLCV-like message
            normalized = {
                "symbol": symbol,
                "exchange": self.EXCHANGE_NAME,
                "timestamp": now,
                "open": prev_price,
                "high": max(prev_price, price),
                "low": min(prev_price, price),
                "close": price,
                "volume": price_data.get("usd_24h_vol", 0),
                "quote_volume": price_data.get("usd_24h_vol", 0),
                "trades": 0,  # Not available from CoinGecko
                "source": "rest",
                "received_at": now,
                "market_cap": price_data.get("usd_market_cap"),
                "change_24h_pct": price_data.get("usd_24h_change"),
            }
            
            if self._callback:
                await self._safe_callback(normalized)
            
            self._message_count += 1
            self._last_message_time = time.time()
        
        logger.debug(f"Polled {len(data)} coins from CoinGecko")
    
    async def disconnect(self) -> None:
        """Stop polling."""
        self._running = False
        
        if self._session:
            await self._session.close()
            self._session = None
        
        self._state = ConnectionState.DISCONNECTED
        logger.info("CoinGecko poller stopped")


# Quick test
if __name__ == "__main__":
    async def test_callback(msg):
        print(f"[{msg['exchange']}] {msg['symbol']}: ${msg['close']:.2f}")
    
    async def main():
        poller = CoinGeckoPoller(
            symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            poll_interval=10,  # Fast for testing
            callback=test_callback
        )
        
        try:
            await asyncio.wait_for(poller.connect(), timeout=35)
        except asyncio.TimeoutError:
            print("Test completed")
        finally:
            await poller.disconnect()
    
    asyncio.run(main())
