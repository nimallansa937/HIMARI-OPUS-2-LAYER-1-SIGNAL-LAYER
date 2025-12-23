"""
HIMARI Data Ingestion Service - Main Entry Point

This is THE MISSING COMPONENT in your architecture. It:
1. Connects to Binance/Kraken WebSockets
2. Normalizes incoming data to HIMARI OHLCV format
3. Publishes to Redpanda 'raw_market_data' topic
4. Your existing Flink pipeline then picks it up for quality validation

Without this service running, your Flink pipeline has nothing to process.

Usage:
    python main.py
    
    # Or with custom symbols:
    SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT python main.py
"""

import asyncio
import signal
import logging
import sys
from typing import Dict, Any, List

from config import (
    KAFKA_BOOTSTRAP, KAFKA_TOPIC_RAW, SYMBOLS,
    DEFAULT_CONFIG, LOG_LEVEL, LOG_FORMAT,
    COINGECKO_POLL_INTERVAL
)
from connectors.base import ConnectionState
from connectors.binance import BinanceConnector
from connectors.binance_orderbook import BinanceOrderBookConnector
from connectors.kraken import KrakenConnector
from connectors.coingecko import CoinGeckoPoller
from publishers.kafka_publisher import KafkaPublisher

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


class DataIngestionService:
    """
    Main orchestrator for the data ingestion layer.
    
    Manages multiple exchange connectors and routes their data
    to the Kafka publisher. Handles graceful startup/shutdown.
    
    Architecture:
        BinanceConnector ─┐
        KrakenConnector ──┼─→ KafkaPublisher → Redpanda → Flink
        (more connectors)─┘
    """
    
    def __init__(self, symbols: List[str], config=DEFAULT_CONFIG):
        """
        Initialize the ingestion service.
        
        Args:
            symbols: List of symbols to track (e.g., ["BTCUSDT", "ETHUSDT"])
            config: IngestionConfig instance
        """
        self.symbols = symbols
        self.config = config
        
        # Components
        self._publisher: KafkaPublisher = None
        self._connectors: Dict[str, Any] = {}
        self._tasks: List[asyncio.Task] = []
        
        # State
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        logger.info(f"DataIngestionService initialized for {len(symbols)} symbols")
    
    async def start(self) -> None:
        """
        Start the ingestion service.
        
        1. Connect to Kafka
        2. Start exchange connectors
        3. Run until shutdown signal
        """
        logger.info("=" * 60)
        logger.info("HIMARI Data Ingestion Service Starting")
        logger.info("=" * 60)
        
        self._running = True
        
        # Setup signal handlers for graceful shutdown (Unix only)
        import sys
        if sys.platform != 'win32':
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(
                    sig, 
                    lambda: asyncio.create_task(self.shutdown())
                )
        
        try:
            # 1. Connect to Kafka/Redpanda
            await self._setup_publisher()
            
            # 2. Setup exchange connectors
            await self._setup_connectors()
            
            # 3. Start all connectors
            await self._start_connectors()
            
            # 4. Run health monitoring
            await self._monitor_health()
            
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            await self.shutdown()
    
    async def _setup_publisher(self) -> None:
        """Initialize Kafka publisher."""
        logger.info(f"Connecting to Kafka: {KAFKA_BOOTSTRAP}")
        
        self._publisher = KafkaPublisher(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            topic=KAFKA_TOPIC_RAW,
        )
        
        success = await self._publisher.connect()
        if not success:
            raise RuntimeError("Failed to connect to Kafka/Redpanda")
        
        logger.info(f"Publishing to topic: {KAFKA_TOPIC_RAW}")
    
    async def _setup_connectors(self) -> None:
        """Initialize exchange connectors."""
        
        # Binance - Primary source (free, fastest, most reliable)
        if self.config.enable_binance:
            logger.info("Setting up Binance OHLCV connector...")
            binance = BinanceConnector(
                symbols=self.symbols,
                interval="1m",
            )
            binance.set_callback(self._on_message)
            self._connectors["binance"] = binance

            # Binance Order Book + Trades (for order flow features)
            logger.info("Setting up Binance Order Book connector...")
            binance_orderbook = BinanceOrderBookConnector(
                symbols=self.symbols,
                depth=20,
                update_speed="100ms",
            )
            binance_orderbook.set_callback(self._on_message)
            self._connectors["binance_orderbook"] = binance_orderbook
        
        # Kraken - Backup source
        if self.config.enable_kraken:
            logger.info("Setting up Kraken connector...")
            kraken = KrakenConnector(
                symbols=self.symbols,
                interval=1,  # 1 minute
            )
            kraken.set_callback(self._on_message)
            self._connectors["kraken"] = kraken
        
        # CoinGecko - REST poller for supplemental data
        if self.config.enable_coingecko:
            logger.info("Setting up CoinGecko poller...")
            coingecko = CoinGeckoPoller(
                symbols=self.symbols,
                poll_interval=COINGECKO_POLL_INTERVAL,
            )
            coingecko.set_callback(self._on_message)
            self._connectors["coingecko"] = coingecko
        
        logger.info(f"Configured {len(self._connectors)} connector(s)")
    
    async def _start_connectors(self) -> None:
        """Start all connectors as background tasks."""
        for name, connector in self._connectors.items():
            logger.info(f"Starting {name} connector...")
            task = asyncio.create_task(
                connector.connect(),
                name=f"connector-{name}"
            )
            self._tasks.append(task)
        
        # Wait a moment for connections to establish
        await asyncio.sleep(2)
        
        # Check connection status
        for name, connector in self._connectors.items():
            if connector.is_connected:
                logger.info(f"✓ {name}: Connected")
            else:
                logger.warning(f"✗ {name}: Not connected ({connector.state.value})")
    
    async def _on_message(self, message: Dict[str, Any]) -> None:
        """
        Callback for incoming exchange messages.
        
        This is called by each connector when they receive data.
        We forward everything to the Kafka publisher.
        """
        # Add ingestion timestamp for latency tracking
        if "ingestion_time" not in message:
            import time
            message["ingestion_time"] = int(time.time() * 1000)
        
        # Publish to Kafka
        await self._publisher.publish(message)
    
    async def _monitor_health(self) -> None:
        """
        Monitor connector health and log status periodically.
        
        Runs until shutdown signal received.
        """
        while self._running:
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.config.health_check_interval
                )
                break  # Shutdown requested
            except asyncio.TimeoutError:
                # Log health status
                await self._log_health()
    
    async def _log_health(self) -> None:
        """Log current health status of all components."""
        logger.info("-" * 40)
        logger.info("Health Check:")
        
        # Connector status
        for name, connector in self._connectors.items():
            health = connector.get_health()
            status = "✓" if health["is_connected"] else "✗"
            stale = " (STALE)" if health.get("is_stale") else ""
            logger.info(
                f"  {status} {name}: {health['message_count']} msgs, "
                f"state={health['state']}{stale}"
            )
        
        # Publisher status
        pub_metrics = self._publisher.get_metrics()
        logger.info(
            f"  Publisher: {pub_metrics['messages_sent']} sent, "
            f"{pub_metrics['messages_buffered']} buffered, "
            f"latency={pub_metrics['avg_latency_ms']:.1f}ms"
        )
        logger.info("-" * 40)
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the service."""
        if not self._running:
            return
        
        logger.info("Shutting down...")
        self._running = False
        self._shutdown_event.set()
        
        # Stop connectors
        for name, connector in self._connectors.items():
            logger.info(f"Stopping {name}...")
            try:
                await connector.disconnect()
            except Exception as e:
                logger.warning(f"Error stopping {name}: {e}")
        
        # Cancel tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Disconnect publisher
        if self._publisher:
            await self._publisher.disconnect()
        
        logger.info("Shutdown complete")


async def main():
    """Entry point."""
    logger.info("=" * 60)
    logger.info("  HIMARI Data Ingestion Layer")
    logger.info("  The Missing Link: Exchanges → Redpanda")
    logger.info("=" * 60)
    logger.info(f"Symbols: {SYMBOLS}")
    logger.info(f"Kafka: {KAFKA_BOOTSTRAP}")
    logger.info(f"Topic: {KAFKA_TOPIC_RAW}")
    logger.info("")
    
    service = DataIngestionService(symbols=SYMBOLS)
    await service.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
