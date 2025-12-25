"""
StockTwits Real-Time Connector
==============================

Fetches real-time crypto sentiment from StockTwits API (free tier).
CryptoBERT was trained on StockTwits data - optimal alignment!

Features:
- Real-time streaming (30-60s latency)
- No API key required for basic access
- Supports BTC, ETH, SOL, and all major cryptos
- Direct integration with multi-model sentiment analyzer

Usage:
    connector = StockTwitsConnector()
    
    # Get recent messages for a symbol
    messages = connector.get_messages("BTC.X")
    
    # Analyze with multi-model ensemble
    results = connector.analyze_sentiment("BTC.X")
    
    # Stream continuously
    async for msg in connector.stream("BTC.X"):
        print(f"{msg['sentiment']}: {msg['body']}")
"""

import os
import time
import logging
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

import requests

logger = logging.getLogger(__name__)

# StockTwits crypto symbols (cashtag format)
STOCKTWITS_CRYPTO_SYMBOLS = {
    'BTCUSDT': 'BTC.X',
    'ETHUSDT': 'ETH.X',
    'SOLUSDT': 'SOL.X',
    'BNBUSDT': 'BNB.X',
    'ADAUSDT': 'ADA.X',
    'DOGEUSDT': 'DOGE.X',
    'XRPUSDT': 'XRP.X',
    'AVAXUSDT': 'AVAX.X',
    'DOTUSDT': 'DOT.X',
    'MATICUSDT': 'MATIC.X',
    'LINKUSDT': 'LINK.X',
    'LTCUSDT': 'LTC.X',
}


@dataclass
class StockTwitsMessage:
    """A single StockTwits message."""
    id: int
    body: str
    created_at: str
    user: str
    user_followers: int
    sentiment: Optional[str]  # "Bullish", "Bearish", or None
    likes: int
    symbol: str
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'body': self.body,
            'created_at': self.created_at,
            'user': self.user,
            'user_followers': self.user_followers,
            'sentiment': self.sentiment,
            'likes': self.likes,
            'symbol': self.symbol,
            'source': 'stocktwits',
        }


class StockTwitsConnector:
    """
    Real-time connector for StockTwits crypto messages.
    
    Free tier limits:
    - 200 requests per hour
    - 30 messages per request
    
    Example:
        connector = StockTwitsConnector()
        
        # Get latest messages
        messages = connector.get_messages("BTC.X", limit=30)
        for msg in messages:
            print(f"[{msg.sentiment}] {msg.body[:50]}...")
        
        # Analyze sentiment
        results = connector.analyze_sentiment("BTC.X")
        print(f"Bullish: {results['bullish_pct']:.0%}")
    """
    
    BASE_URL = "https://api.stocktwits.com/api/2"
    
    def __init__(
        self,
        access_token: Optional[str] = None,
        enable_sentiment_analysis: bool = True,
    ):
        """
        Initialize StockTwits connector.
        
        Args:
            access_token: Optional OAuth token (higher rate limits)
            enable_sentiment_analysis: Run messages through multi-model analyzer
        """
        self.access_token = access_token or os.getenv("STOCKTWITS_TOKEN")
        self.enable_sentiment_analysis = enable_sentiment_analysis
        
        # Rate limiting
        self._request_times: List[float] = []
        self._rate_limit = 200  # per hour
        
        # Sentiment analyzer (lazy load)
        self._analyzer = None
        
        # Cache
        self._last_messages: Dict[str, List[StockTwitsMessage]] = {}
        self._last_fetch: Dict[str, float] = {}
        
        logger.info("StockTwitsConnector initialized")
    
    def _get_analyzer(self):
        """Lazy load multi-model sentiment analyzer."""
        if self._analyzer is None and self.enable_sentiment_analysis:
            try:
                from primitives.multi_model_sentiment import create_phase2_analyzer
                self._analyzer = create_phase2_analyzer()
                logger.info("✓ Multi-model analyzer loaded for StockTwits")
            except Exception as e:
                logger.warning(f"Could not load multi-model analyzer: {e}")
        return self._analyzer
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = time.time()
        hour_ago = now - 3600
        
        # Clean old requests
        self._request_times = [t for t in self._request_times if t > hour_ago]
        
        if len(self._request_times) >= self._rate_limit:
            logger.warning("StockTwits rate limit reached")
            return False
        
        self._request_times.append(now)
        return True
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with rate limiting."""
        if not self._check_rate_limit():
            return {"messages": []}
        
        url = f"{self.BASE_URL}/{endpoint}"
        params = params or {}
        
        if self.access_token:
            params["access_token"] = self.access_token
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"StockTwits API error: {e}")
            return {"messages": []}
    
    def get_messages(
        self,
        symbol: str,
        limit: int = 30,
        since_id: Optional[int] = None,
    ) -> List[StockTwitsMessage]:
        """
        Get recent messages for a symbol.
        
        Args:
            symbol: StockTwits symbol (e.g., "BTC.X")
            limit: Max messages (30 default, 30 max)
            since_id: Only get messages after this ID
            
        Returns:
            List of StockTwitsMessage objects
        """
        params = {"limit": min(limit, 30)}
        if since_id:
            params["since"] = since_id
        
        data = self._make_request(f"streams/symbol/{symbol}.json", params)
        
        messages = []
        for msg in data.get("messages", []):
            try:
                # Extract user sentiment tag if present
                user_sentiment = None
                entities = msg.get("entities", {})
                if entities.get("sentiment"):
                    user_sentiment = entities["sentiment"].get("basic")
                
                messages.append(StockTwitsMessage(
                    id=msg["id"],
                    body=msg["body"],
                    created_at=msg["created_at"],
                    user=msg["user"]["username"],
                    user_followers=msg["user"].get("followers", 0),
                    sentiment=user_sentiment,
                    likes=msg.get("likes", {}).get("total", 0),
                    symbol=symbol,
                ))
            except (KeyError, TypeError) as e:
                logger.debug(f"Skipping malformed message: {e}")
        
        self._last_messages[symbol] = messages
        self._last_fetch[symbol] = time.time()
        
        return messages
    
    def get_messages_for_trading_pair(self, pair: str, limit: int = 30) -> List[StockTwitsMessage]:
        """
        Get messages for a trading pair (e.g., "BTCUSDT").
        
        Args:
            pair: Trading pair (e.g., "BTCUSDT")
            limit: Max messages
            
        Returns:
            List of messages
        """
        symbol = STOCKTWITS_CRYPTO_SYMBOLS.get(pair.upper())
        if not symbol:
            logger.warning(f"Unknown trading pair: {pair}")
            return []
        
        return self.get_messages(symbol, limit)
    
    def analyze_sentiment(
        self,
        symbol: str,
        use_cache: bool = True,
        cache_ttl: int = 60,
    ) -> Dict[str, Any]:
        """
        Analyze sentiment for a symbol using multi-model ensemble.
        
        Args:
            symbol: StockTwits symbol or trading pair
            use_cache: Use cached messages if recent
            cache_ttl: Cache TTL in seconds
            
        Returns:
            Dict with sentiment scores and breakdown
        """
        # Convert trading pair to symbol if needed
        if symbol.upper() in STOCKTWITS_CRYPTO_SYMBOLS:
            symbol = STOCKTWITS_CRYPTO_SYMBOLS[symbol.upper()]
        
        # Check cache
        if use_cache and symbol in self._last_messages:
            age = time.time() - self._last_fetch.get(symbol, 0)
            if age < cache_ttl:
                messages = self._last_messages[symbol]
            else:
                messages = self.get_messages(symbol)
        else:
            messages = self.get_messages(symbol)
        
        if not messages:
            return {
                "symbol": symbol,
                "message_count": 0,
                "sentiment_score": 0.0,
                "bullish_pct": 0.0,
                "bearish_pct": 0.0,
                "neutral_pct": 0.0,
                "error": "no_messages",
            }
        
        # Analyze with multi-model
        analyzer = self._get_analyzer()
        
        results = {
            "symbol": symbol,
            "message_count": len(messages),
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "model_predictions": [],
            "weighted_score": 0.0,
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for msg in messages:
            # Use StockTwits user sentiment if available
            if msg.sentiment == "Bullish":
                results["bullish_count"] += 1
                score = 1.0
            elif msg.sentiment == "Bearish":
                results["bearish_count"] += 1
                score = -1.0
            else:
                # Analyze with multi-model
                if analyzer:
                    try:
                        pred = analyzer.analyze(msg.body, source="stocktwits")
                        if pred:
                            score = pred.score
                            if pred.label == "bullish":
                                results["bullish_count"] += 1
                            elif pred.label == "bearish":
                                results["bearish_count"] += 1
                            else:
                                results["neutral_count"] += 1
                            
                            results["model_predictions"].append({
                                "text": msg.body[:50],
                                "score": score,
                                "label": pred.label,
                                "model": pred.model_name,
                            })
                        else:
                            results["neutral_count"] += 1
                            score = 0.0
                    except Exception as e:
                        logger.debug(f"Analysis failed: {e}")
                        results["neutral_count"] += 1
                        score = 0.0
                else:
                    results["neutral_count"] += 1
                    score = 0.0
            
            # Weight by user followers (influence)
            weight = 1 + min(msg.user_followers / 1000, 10)  # Cap at 10x
            weighted_sum += score * weight
            total_weight += weight
        
        # Calculate percentages
        n = len(messages)
        results["bullish_pct"] = results["bullish_count"] / n
        results["bearish_pct"] = results["bearish_count"] / n
        results["neutral_pct"] = results["neutral_count"] / n
        
        # Weighted sentiment score
        results["weighted_score"] = weighted_sum / total_weight if total_weight > 0 else 0
        results["sentiment_score"] = results["weighted_score"]
        
        # Signal type
        if results["weighted_score"] > 0.3:
            results["signal"] = "BULLISH"
        elif results["weighted_score"] < -0.3:
            results["signal"] = "BEARISH"
        else:
            results["signal"] = "NEUTRAL"
        
        results["timestamp"] = datetime.utcnow().isoformat()
        
        return results
    
    async def stream(
        self,
        symbol: str,
        poll_interval: int = 30,
    ) -> AsyncGenerator[StockTwitsMessage, None]:
        """
        Stream messages for a symbol.
        
        Args:
            symbol: StockTwits symbol
            poll_interval: Seconds between polls
            
        Yields:
            New StockTwitsMessage objects
        """
        last_id = None
        
        while True:
            messages = self.get_messages(symbol, since_id=last_id)
            
            for msg in reversed(messages):  # Oldest first
                if last_id is None or msg.id > last_id:
                    last_id = msg.id
                    yield msg
            
            await asyncio.sleep(poll_interval)
    
    def get_trending(self) -> List[Dict]:
        """Get trending crypto symbols on StockTwits."""
        data = self._make_request("trending/symbols.json")
        
        trending = []
        for symbol in data.get("symbols", []):
            if symbol.get("symbol", "").endswith(".X"):  # Crypto
                trending.append({
                    "symbol": symbol["symbol"],
                    "title": symbol.get("title", ""),
                    "watchlist_count": symbol.get("watchlist_count", 0),
                })
        
        return trending[:10]
    
    def health_check(self) -> Dict[str, Any]:
        """Check connector health."""
        return {
            "status": "healthy",
            "rate_limit_remaining": self._rate_limit - len(self._request_times),
            "rate_limit_total": self._rate_limit,
            "cached_symbols": list(self._last_messages.keys()),
            "analyzer_loaded": self._analyzer is not None,
        }


# =============================================================================
# FACTORY
# =============================================================================

def create_stocktwits_connector(
    with_analyzer: bool = True
) -> StockTwitsConnector:
    """Create StockTwits connector with optional analyzer."""
    return StockTwitsConnector(enable_sentiment_analysis=with_analyzer)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("STOCKTWITS CONNECTOR TEST")
    print("=" * 60)
    
    connector = StockTwitsConnector(enable_sentiment_analysis=False)
    
    # Test trending
    print("\n📈 Trending Crypto:")
    trending = connector.get_trending()
    for t in trending[:5]:
        print(f"  {t['symbol']}: {t['title']}")
    
    # Test messages
    print("\n💬 Latest BTC.X Messages:")
    messages = connector.get_messages("BTC.X", limit=5)
    for msg in messages:
        sentiment_emoji = "🟢" if msg.sentiment == "Bullish" else ("🔴" if msg.sentiment == "Bearish" else "⚪")
        print(f"  {sentiment_emoji} @{msg.user}: {msg.body[:50]}...")
    
    # Test sentiment analysis
    print("\n📊 BTC Sentiment Analysis:")
    sentiment = connector.analyze_sentiment("BTC.X")
    print(f"  Messages: {sentiment['message_count']}")
    print(f"  Bullish: {sentiment['bullish_pct']:.0%}")
    print(f"  Bearish: {sentiment['bearish_pct']:.0%}")
    print(f"  Score: {sentiment['weighted_score']:+.2f}")
    print(f"  Signal: {sentiment['signal']}")
    
    # Health
    print("\n🏥 Health:")
    print(f"  {connector.health_check()}")
