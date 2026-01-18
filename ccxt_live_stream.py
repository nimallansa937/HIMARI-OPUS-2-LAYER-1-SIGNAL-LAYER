"""
CCXT Live Data Stream for HIMARI

Streams real-time OHLCV data from Binance to:
1. TimescaleDB (hinance database)
2. Redis for real-time access
3. Console for monitoring

Usage:
    python ccxt_live_stream.py
"""

import ccxt
import asyncio
import time
import logging
from datetime import datetime
from typing import List, Dict, Any
import json

# Try to import database clients
try:
    import psycopg2
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("Warning: psycopg2 not installed, TimescaleDB storage disabled")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: redis not installed, Redis storage disabled")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CCXTLiveStream:
    """
    Live OHLCV data stream using CCXT.

    Fetches data from Binance and stores in TimescaleDB + Redis.
    """

    def __init__(
        self,
        symbols: List[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        timeframe: str = "1m",
        db_config: Dict[str, Any] = None,
        redis_config: Dict[str, Any] = None
    ):
        self.symbols = symbols
        self.timeframe = timeframe
        self.running = False

        # Initialize Binance exchange (no API key needed for public data)
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}  # Use futures for funding rate
        })

        # Database config
        self.db_config = db_config or {
            'host': 'localhost',
            'port': 5434,
            'database': 'hinance',
            'user': 'hinance_user',
            'password': 'hinance_password'
        }

        # Redis config
        self.redis_config = redis_config or {
            'host': 'localhost',
            'port': 6379,
            'db': 0
        }

        self.db_conn = None
        self.redis_client = None
        self.message_count = 0

    def connect_db(self) -> bool:
        """Connect to TimescaleDB."""
        if not POSTGRES_AVAILABLE:
            return False

        try:
            self.db_conn = psycopg2.connect(**self.db_config)
            self.db_conn.autocommit = True
            logger.info(f"Connected to TimescaleDB at {self.db_config['host']}:{self.db_config['port']}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to TimescaleDB: {e}")
            return False

    def connect_redis(self) -> bool:
        """Connect to Redis."""
        if not REDIS_AVAILABLE:
            return False

        try:
            self.redis_client = redis.Redis(**self.redis_config)
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.redis_config['host']}:{self.redis_config['port']}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False

    def store_ohlcv(self, symbol: str, ohlcv: List) -> bool:
        """Store OHLCV data in TimescaleDB."""
        if not self.db_conn:
            return False

        try:
            cur = self.db_conn.cursor()

            # Convert symbol format: BTC/USDT -> BTCUSDT
            db_symbol = symbol.replace("/", "")

            for candle in ohlcv:
                timestamp, open_p, high, low, close, volume = candle[:6]

                # Convert timestamp to datetime
                ts = datetime.utcfromtimestamp(timestamp / 1000)

                # Use INSERT IGNORE pattern for TimescaleDB hypertable
                try:
                    cur.execute("""
                        INSERT INTO market_data (time, symbol, open, high, low, close, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (ts, db_symbol, open_p, high, low, close, volume))
                except Exception:
                    pass  # Ignore duplicate inserts

            return True
        except Exception as e:
            logger.error(f"Failed to store OHLCV: {e}")
            return False

    def store_redis(self, symbol: str, data: Dict) -> bool:
        """Store latest data in Redis for real-time access."""
        if not self.redis_client:
            return False

        try:
            # Store latest price
            key = f"himari:price:{symbol.replace('/', '')}"
            self.redis_client.setex(key, 60, json.dumps(data))

            # Publish to channel
            self.redis_client.publish(f"himari:ohlcv", json.dumps({
                'symbol': symbol.replace('/', ''),
                **data
            }))

            return True
        except Exception as e:
            logger.error(f"Failed to store in Redis: {e}")
            return False

    async def fetch_ohlcv(self, symbol: str) -> List:
        """Fetch latest OHLCV data for a symbol."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=5)
            return ohlcv
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            return []

    async def run(self):
        """Main run loop."""
        self.running = True

        # Connect to databases
        db_ok = self.connect_db()
        redis_ok = self.connect_redis()

        logger.info(f"Starting live stream for {self.symbols}")
        logger.info(f"  TimescaleDB: {'connected' if db_ok else 'disabled'}")
        logger.info(f"  Redis: {'connected' if redis_ok else 'disabled'}")

        while self.running:
            try:
                for symbol in self.symbols:
                    # Fetch OHLCV
                    ohlcv = await self.fetch_ohlcv(symbol)

                    if ohlcv:
                        latest = ohlcv[-1]
                        timestamp, open_p, high, low, close, volume = latest[:6]

                        # Store in TimescaleDB
                        self.store_ohlcv(symbol, ohlcv)

                        # Store in Redis
                        data = {
                            'timestamp': timestamp,
                            'open': open_p,
                            'high': high,
                            'low': low,
                            'close': close,
                            'volume': volume,
                            'received_at': int(time.time() * 1000)
                        }
                        self.store_redis(symbol, data)

                        self.message_count += 1

                        # Log every 10 messages
                        if self.message_count % 10 == 0:
                            logger.info(
                                f"[{self.message_count}] {symbol}: "
                                f"${close:,.2f} | Vol: {volume:,.0f}"
                            )

                # Wait before next fetch (respect rate limits)
                await asyncio.sleep(5)

            except KeyboardInterrupt:
                logger.info("Stopping live stream...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)

        self.running = False
        logger.info(f"Stream stopped. Total messages: {self.message_count}")

    def stop(self):
        """Stop the stream."""
        self.running = False


async def main():
    """Main entry point."""
    stream = CCXTLiveStream(
        symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        timeframe="1m"
    )

    try:
        await stream.run()
    except KeyboardInterrupt:
        stream.stop()


if __name__ == "__main__":
    asyncio.run(main())
