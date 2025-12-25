"""
HIMARI Social Data Connectors
=============================

Real-time connectors for social media and news data.
Feeds into multi-model sentiment ensemble for alpha generation.

Usage:
    from connectors import (
        StockTwitsConnector,
        CryptoPanicConnector,
        TelegramConnector,
        create_stocktwits_connector,
        create_cryptopanic_connector,
        create_telegram_connector,
    )
    
    # StockTwits (free, no API key needed for basic access)
    stocktwits = create_stocktwits_connector()
    messages = stocktwits.get_messages("BTC.X")
    sentiment = stocktwits.analyze_sentiment("BTCUSDT")
    
    # CryptoPanic (free, requires API key)
    cryptopanic = create_cryptopanic_connector(api_key="your_key")
    news = cryptopanic.get_news(currencies="BTC", filter="hot")
    sentiment = cryptopanic.analyze_sentiment(currencies="BTC")
    
    # Telegram (free, no API key needed)
    telegram = create_telegram_connector()
    messages = telegram.get_messages("@cryptowhalesignal")
    sentiment = telegram.analyze_sentiment("@cryptowhalesignal")
"""

from .stocktwits_connector import (
    StockTwitsConnector,
    StockTwitsMessage,
    create_stocktwits_connector,
    STOCKTWITS_CRYPTO_SYMBOLS,
)

from .cryptopanic_connector import (
    CryptoPanicConnector,
    CryptoNews,
    create_cryptopanic_connector,
)

from .telegram_connector import (
    TelegramConnector,
    TelegramMessage,
    create_telegram_connector,
    CRYPTO_TELEGRAM_CHANNELS,
)

__all__ = [
    # StockTwits
    "StockTwitsConnector",
    "StockTwitsMessage",
    "create_stocktwits_connector",
    "STOCKTWITS_CRYPTO_SYMBOLS",
    # CryptoPanic
    "CryptoPanicConnector",
    "CryptoNews",
    "create_cryptopanic_connector",
    # Telegram
    "TelegramConnector",
    "TelegramMessage",
    "create_telegram_connector",
    "CRYPTO_TELEGRAM_CHANNELS",
]
