"""
HIMARI Data Connectors Package

Free API integrations for:
- Market Data: Binance, CoinGecko, CoinCap, CryptoCompare
- Social Sentiment: Reddit, Twitter
- News: NewsData.io, Finnhub
- On-Chain: Etherscan, Alchemy
"""

from .binance_connector import BinanceConnector, BinanceWebSocket
from .coingecko_connector import CoinGeckoConnector
from .coincap_connector import CoinCapConnector
from .reddit_connector import RedditConnector
from .news_connector import NewsDataConnector, FinnhubConnector
from .etherscan_connector import EtherscanConnector

__all__ = [
    # Market Data
    'BinanceConnector',
    'BinanceWebSocket', 
    'CoinGeckoConnector',
    'CoinCapConnector',
    # Social
    'RedditConnector',
    # News
    'NewsDataConnector',
    'FinnhubConnector',
    # On-Chain
    'EtherscanConnector',
]
