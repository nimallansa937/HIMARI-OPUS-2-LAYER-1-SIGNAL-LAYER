"""
HIMARI Data Ingestion Configuration

This is the MISSING configuration that connects your exchange APIs
to Redpanda. Without this, your Flink pipeline has nothing to process.
"""

import os
from typing import List, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# KAFKA/REDPANDA CONNECTION (match your existing infrastructure)
# =============================================================================

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC_RAW = os.getenv("KAFKA_TOPIC_RAW", "raw_market_data")

# Producer settings
KAFKA_PRODUCER_CONFIG = {
    'bootstrap.servers': KAFKA_BOOTSTRAP,
    'client.id': 'himari-ingestion',
    'acks': 'all',  # Wait for all replicas
    'retries': 3,
    'retry.backoff.ms': 100,
    'linger.ms': 5,  # Batch for 5ms for throughput
    'batch.size': 16384,  # 16KB batches
    'compression.type': 'none',  # lz4 requires extra lib, use none for now
}

# =============================================================================
# SYMBOLS TO TRACK
# =============================================================================

# Primary trading pairs (add more as needed)
SYMBOLS = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT").split(",")

# Symbol mapping between exchanges (they use different formats)
SYMBOL_MAPPING = {
    "binance": {
        "BTCUSDT": "btcusdt",
        "ETHUSDT": "ethusdt",
        "SOLUSDT": "solusdt",
        "BNBUSDT": "bnbusdt",
        "XRPUSDT": "xrpusdt",
    },
    "kraken": {
        "BTCUSDT": "XBT/USDT",
        "ETHUSDT": "ETH/USDT",
        "SOLUSDT": "SOL/USDT",
        "BNBUSDT": None,  # Not available on Kraken
        "XRPUSDT": "XRP/USDT",
    },
    "coingecko": {
        "BTCUSDT": "bitcoin",
        "ETHUSDT": "ethereum",
        "SOLUSDT": "solana",
        "BNBUSDT": "binancecoin",
        "XRPUSDT": "ripple",
    }
}

# =============================================================================
# EXCHANGE WEBSOCKET URLS
# =============================================================================

# Binance - Free, unlimited, best for crypto
BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"
BINANCE_STREAM_TYPES = ["kline_1m", "trade", "ticker"]  # What to subscribe to

# Kraken - Free, good backup
KRAKEN_WS_URL = "wss://ws.kraken.com"

# Bybit - Free, good for derivatives
BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/spot"

# Coinbase - Free, US-focused
COINBASE_WS_URL = "wss://ws-feed.exchange.coinbase.com"

# =============================================================================
# REST API CONFIGURATION (for backup/historical)
# =============================================================================

# CoinGecko - Free tier: 10,000 calls/month, 30 calls/min
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
COINGECKO_POLL_INTERVAL = 60  # seconds (be conservative with free tier)
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", None)  # Optional for free tier

# CoinCap - Completely free
COINCAP_API_URL = "https://api.coincap.io/v2"
COINCAP_POLL_INTERVAL = 30  # seconds

# Polygon.io - $49/month for crypto
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", None)
POLYGON_WS_URL = "wss://socket.polygon.io/crypto"

# =============================================================================
# INGESTION SETTINGS
# =============================================================================

@dataclass
class IngestionConfig:
    """Complete ingestion configuration."""
    
    # Which connectors to enable
    enable_binance: bool = True
    enable_kraken: bool = True
    enable_coingecko: bool = True
    enable_polygon: bool = False  # Requires paid API key
    
    # Reconnection settings
    reconnect_delay_initial: float = 1.0  # seconds
    reconnect_delay_max: float = 60.0  # seconds
    reconnect_delay_multiplier: float = 2.0  # exponential backoff
    
    # Health check
    health_check_interval: float = 30.0  # seconds
    stale_data_threshold: float = 60.0  # Mark as stale if no data for 60s
    
    # Buffering (when Kafka unavailable)
    buffer_max_size: int = 10000  # messages
    buffer_flush_interval: float = 1.0  # seconds
    
    # Metrics
    metrics_port: int = 8000  # Prometheus scrape endpoint


DEFAULT_CONFIG = IngestionConfig()

# =============================================================================
# MESSAGE SCHEMA (what gets published to Kafka)
# =============================================================================

OHLCV_SCHEMA = {
    "symbol": str,          # "BTCUSDT"
    "exchange": str,        # "binance", "kraken", etc.
    "timestamp": int,       # Unix milliseconds
    "open": float,          # Opening price
    "high": float,          # High price
    "low": float,           # Low price
    "close": float,         # Closing price (or last trade price)
    "volume": float,        # Base asset volume
    "quote_volume": float,  # Quote asset volume (USD value)
    "trades": int,          # Number of trades (if available)
    "source": str,          # "websocket" or "rest"
    "received_at": int,     # When we received it (for latency tracking)
}

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
