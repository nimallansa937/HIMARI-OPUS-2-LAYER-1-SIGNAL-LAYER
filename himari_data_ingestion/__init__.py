"""
HIMARI Data Ingestion Layer

The missing link between exchange WebSockets and your Redpanda pipeline.

Connects:
    Binance WebSocket ─┐
    Kraken WebSocket ──┼─→ Kafka → Flink → Redis → Signal Layer
    CoinGecko REST ────┘

Usage:
    python -m himari_data_ingestion.main
"""

__version__ = "1.0.0"
__author__ = "HIMARI Team"

from .connectors import (
    BinanceConnector,
    BinanceTradeConnector,
    KrakenConnector,
    CoinGeckoPoller,
)
from .publishers import KafkaPublisher
from .normalizers import OHLCVNormalizer, normalize_message

__all__ = [
    'BinanceConnector',
    'BinanceTradeConnector',
    'KrakenConnector',
    'CoinGeckoPoller',
    'KafkaPublisher',
    'OHLCVNormalizer',
    'normalize_message',
]
