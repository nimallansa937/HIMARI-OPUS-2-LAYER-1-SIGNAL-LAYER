"""
CryptoPanic Real-Time News Connector
=====================================

Fetches real-time crypto news from CryptoPanic API (free tier).
Perfect for ModernFinBERT news sentiment analysis.

Features:
- Real-time news feed (30-60s latency)
- Free tier: 5 requests/min
- Filters: hot, rising, bullish, bearish, important
- Direct integration with multi-model sentiment analyzer

Usage:
    connector = CryptoPanicConnector(api_key="your_key")
    
    # Get recent news
    news = connector.get_news(filter="hot")
    
    # Analyze with multi-model ensemble
    results = connector.analyze_sentiment()
    
    # Get news for specific currency
    btc_news = connector.get_news(currencies="BTC")
"""

import os
import time
import logging
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
import json
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)


@dataclass
class CryptoNews:
    """A single news item from CryptoPanic."""
    id: int
    title: str
    url: str
    source: str
    published_at: str
    currencies: List[str]
    kind: str  # "news", "media", etc.
    votes: Dict[str, int]  # {"positive": N, "negative": M, ...}
    is_hot: bool
    
    @property
    def sentiment_votes(self) -> float:
        """Calculate sentiment from votes (-1 to +1)."""
        pos = self.votes.get("positive", 0) + self.votes.get("liked", 0)
        neg = self.votes.get("negative", 0) + self.votes.get("disliked", 0)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "published_at": self.published_at,
            "currencies": self.currencies,
            "kind": self.kind,
            "votes": self.votes,
            "is_hot": self.is_hot,
            "sentiment_votes": self.sentiment_votes,
            "data_source": "cryptopanic",
        }


class CryptoPanicConnector:
    """
    Real-time connector for CryptoPanic crypto news.
    
    Free tier limits:
    - 5 requests per minute
    - Requires API key (free registration)
    
    Get API key: https://cryptopanic.com/developers/api/
    
    Example:
        connector = CryptoPanicConnector(api_key="your_key")
        
        # Get hot news
        news = connector.get_news(filter="hot")
        for item in news:
            print(f"[{item.source}] {item.title}")
        
        # Analyze sentiment
        results = connector.analyze_sentiment(currencies="BTC")
        print(f"Bullish: {results['bullish_pct']:.0%}")
    """
    
    BASE_URL = "https://cryptopanic.com/api/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        enable_sentiment_analysis: bool = True,
    ):
        """
        Initialize CryptoPanic connector.
        
        Args:
            api_key: CryptoPanic API key (get free at cryptopanic.com/developers/api/)
            enable_sentiment_analysis: Run news through multi-model analyzer
        """
        self.api_key = api_key or os.getenv("CRYPTOPANIC_API_KEY")
        self.enable_sentiment_analysis = enable_sentiment_analysis
        
        if not self.api_key:
            logger.warning(
                "No CryptoPanic API key provided. "
                "Get one free at: https://cryptopanic.com/developers/api/"
            )
        
        # Rate limiting (5 per minute)
        self._request_times: List[float] = []
        self._rate_limit = 5
        self._rate_window = 60
        
        # Sentiment analyzer (lazy load)
        self._analyzer = None
        
        # Cache
        self._cache: Dict[str, List[CryptoNews]] = {}
        self._cache_time: Dict[str, float] = {}
        
        logger.info("CryptoPanicConnector initialized")
    
    def _get_analyzer(self):
        """Lazy load multi-model sentiment analyzer."""
        if self._analyzer is None and self.enable_sentiment_analysis:
            try:
                from primitives.multi_model_sentiment import create_phase2_analyzer
                self._analyzer = create_phase2_analyzer()
                logger.info("✓ Multi-model analyzer loaded for CryptoPanic")
            except Exception as e:
                logger.warning(f"Could not load multi-model analyzer: {e}")
        return self._analyzer
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = time.time()
        cutoff = now - self._rate_window
        
        # Clean old requests
        self._request_times = [t for t in self._request_times if t > cutoff]
        
        if len(self._request_times) >= self._rate_limit:
            wait_time = self._request_times[0] + self._rate_window - now
            logger.warning(f"CryptoPanic rate limit reached, wait {wait_time:.0f}s")
            return False
        
        self._request_times.append(now)
        return True
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with rate limiting."""
        if not self.api_key:
            logger.error("No API key configured")
            return {"results": []}
        
        if not self._check_rate_limit():
            return {"results": []}
        
        url = f"{self.BASE_URL}/{endpoint}"
        params = params or {}
        params["auth_token"] = self.api_key
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"CryptoPanic API error: {e}")
            return {"results": []}
    
    def get_news(
        self,
        currencies: Optional[str] = None,
        filter: str = "hot",
        kind: str = "news",
        limit: int = 20,
    ) -> List[CryptoNews]:
        """
        Get recent crypto news.
        
        Args:
            currencies: Comma-separated currencies (e.g., "BTC,ETH")
            filter: "rising", "hot", "bullish", "bearish", "important", "saved", "lol"
            kind: "news", "media", "all"
            limit: Max results
            
        Returns:
            List of CryptoNews objects
        """
        params = {
            "filter": filter,
            "kind": kind,
        }
        
        if currencies:
            params["currencies"] = currencies
        
        data = self._make_request("posts/", params)
        
        news_items = []
        for item in data.get("results", [])[:limit]:
            try:
                # Extract currencies mentioned
                currency_list = []
                for curr in item.get("currencies", []):
                    currency_list.append(curr.get("code", ""))
                
                news_items.append(CryptoNews(
                    id=item["id"],
                    title=item["title"],
                    url=item.get("url", ""),
                    source=item.get("source", {}).get("title", "Unknown"),
                    published_at=item.get("published_at", ""),
                    currencies=currency_list,
                    kind=item.get("kind", "news"),
                    votes=item.get("votes", {}),
                    is_hot=item.get("is_hot", False),
                ))
            except (KeyError, TypeError) as e:
                logger.debug(f"Skipping malformed news item: {e}")
        
        # Cache results
        cache_key = f"{currencies or 'all'}_{filter}"
        self._cache[cache_key] = news_items
        self._cache_time[cache_key] = time.time()
        
        return news_items
    
    def get_news_for_trading_pair(
        self, 
        pair: str,
        filter: str = "hot",
    ) -> List[CryptoNews]:
        """
        Get news for a trading pair (e.g., "BTCUSDT").
        
        Args:
            pair: Trading pair (e.g., "BTCUSDT")
            filter: News filter
            
        Returns:
            List of news items
        """
        # Extract base currency from pair
        currency = pair.upper().replace("USDT", "").replace("USD", "").replace("BUSD", "")
        return self.get_news(currencies=currency, filter=filter)
    
    def analyze_sentiment(
        self,
        currencies: Optional[str] = None,
        filter: str = "hot",
        use_cache: bool = True,
        cache_ttl: int = 60,
    ) -> Dict[str, Any]:
        """
        Analyze news sentiment using multi-model ensemble.
        
        Args:
            currencies: Filter by currencies
            filter: News filter
            use_cache: Use cached news if recent
            cache_ttl: Cache TTL in seconds
            
        Returns:
            Dict with sentiment scores and breakdown
        """
        cache_key = f"{currencies or 'all'}_{filter}"
        
        # Check cache
        if use_cache and cache_key in self._cache:
            age = time.time() - self._cache_time.get(cache_key, 0)
            if age < cache_ttl:
                news = self._cache[cache_key]
            else:
                news = self.get_news(currencies=currencies, filter=filter)
        else:
            news = self.get_news(currencies=currencies, filter=filter)
        
        if not news:
            return {
                "currencies": currencies,
                "news_count": 0,
                "sentiment_score": 0.0,
                "bullish_pct": 0.0,
                "bearish_pct": 0.0,
                "neutral_pct": 0.0,
                "error": "no_news",
            }
        
        # Analyze with multi-model
        analyzer = self._get_analyzer()
        
        results = {
            "currencies": currencies,
            "news_count": len(news),
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "model_predictions": [],
            "vote_sentiment": 0.0,
            "model_sentiment": 0.0,
        }
        
        vote_sum = 0
        model_sum = 0
        model_count = 0
        
        for item in news:
            # Aggregate vote sentiment
            vote_sum += item.sentiment_votes
            
            # Analyze title with multi-model
            if analyzer:
                try:
                    # Use news source name to route properly
                    source = "bloomberg" if item.source.lower() in [
                        "bloomberg", "reuters", "coindesk", "cointelegraph"
                    ] else "news"
                    
                    pred = analyzer.analyze(item.title, source=source)
                    if pred:
                        model_sum += pred.score
                        model_count += 1
                        
                        if pred.label == "bullish":
                            results["bullish_count"] += 1
                        elif pred.label == "bearish":
                            results["bearish_count"] += 1
                        else:
                            results["neutral_count"] += 1
                        
                        results["model_predictions"].append({
                            "title": item.title[:60],
                            "source": item.source,
                            "score": pred.score,
                            "label": pred.label,
                            "model": pred.model_name,
                            "is_hot": item.is_hot,
                        })
                    else:
                        results["neutral_count"] += 1
                except Exception as e:
                    logger.debug(f"Analysis failed: {e}")
                    results["neutral_count"] += 1
            else:
                # Use vote sentiment as fallback
                if item.sentiment_votes > 0.2:
                    results["bullish_count"] += 1
                elif item.sentiment_votes < -0.2:
                    results["bearish_count"] += 1
                else:
                    results["neutral_count"] += 1
        
        # Calculate percentages
        n = len(news)
        results["bullish_pct"] = results["bullish_count"] / n
        results["bearish_pct"] = results["bearish_count"] / n
        results["neutral_pct"] = results["neutral_count"] / n
        
        # Vote-based sentiment
        results["vote_sentiment"] = vote_sum / n
        
        # Model-based sentiment
        results["model_sentiment"] = model_sum / model_count if model_count > 0 else 0
        
        # Combined sentiment (weight model higher)
        results["sentiment_score"] = (
            results["model_sentiment"] * 0.7 + 
            results["vote_sentiment"] * 0.3
        ) if model_count > 0 else results["vote_sentiment"]
        
        # Signal type
        if results["sentiment_score"] > 0.3:
            results["signal"] = "BULLISH"
        elif results["sentiment_score"] < -0.3:
            results["signal"] = "BEARISH"
        else:
            results["signal"] = "NEUTRAL"
        
        results["timestamp"] = datetime.utcnow().isoformat()
        
        return results
    
    def get_bullish_news(self, currencies: Optional[str] = None) -> List[CryptoNews]:
        """Get news that community voted as bullish."""
        return self.get_news(currencies=currencies, filter="bullish")
    
    def get_bearish_news(self, currencies: Optional[str] = None) -> List[CryptoNews]:
        """Get news that community voted as bearish."""
        return self.get_news(currencies=currencies, filter="bearish")
    
    def get_important_news(self, currencies: Optional[str] = None) -> List[CryptoNews]:
        """Get news marked as important."""
        return self.get_news(currencies=currencies, filter="important")
    
    async def stream(
        self,
        currencies: Optional[str] = None,
        poll_interval: int = 60,
    ) -> AsyncGenerator[CryptoNews, None]:
        """
        Stream news continuously.
        
        Args:
            currencies: Filter by currencies
            poll_interval: Seconds between polls (min 60 due to rate limits)
            
        Yields:
            New CryptoNews objects
        """
        seen_ids = set()
        poll_interval = max(poll_interval, 60)  # Enforce rate limit
        
        while True:
            news = self.get_news(currencies=currencies, filter="rising")
            
            for item in news:
                if item.id not in seen_ids:
                    seen_ids.add(item.id)
                    yield item
            
            # Keep seen_ids manageable
            if len(seen_ids) > 1000:
                seen_ids = set(list(seen_ids)[-500:])
            
            await asyncio.sleep(poll_interval)
    
    def health_check(self) -> Dict[str, Any]:
        """Check connector health."""
        return {
            "status": "healthy" if self.api_key else "no_api_key",
            "api_key_configured": bool(self.api_key),
            "rate_limit_remaining": self._rate_limit - len(self._request_times),
            "rate_limit_total": self._rate_limit,
            "cached_queries": list(self._cache.keys()),
            "analyzer_loaded": self._analyzer is not None,
        }


# =============================================================================
# FACTORY
# =============================================================================

def create_cryptopanic_connector(
    api_key: Optional[str] = None,
    with_analyzer: bool = True,
) -> CryptoPanicConnector:
    """Create CryptoPanic connector."""
    return CryptoPanicConnector(
        api_key=api_key,
        enable_sentiment_analysis=with_analyzer,
    )


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("CRYPTOPANIC CONNECTOR TEST")
    print("=" * 60)
    
    api_key = os.getenv("CRYPTOPANIC_API_KEY")
    
    if not api_key:
        print("\n⚠️  No CRYPTOPANIC_API_KEY environment variable set")
        print("   Get free API key at: https://cryptopanic.com/developers/api/")
        print("\n   Set it with:")
        print("   export CRYPTOPANIC_API_KEY=your_key_here")
        print("   or")
        print("   $env:CRYPTOPANIC_API_KEY='your_key_here'  # PowerShell")
    else:
        connector = CryptoPanicConnector(
            api_key=api_key,
            enable_sentiment_analysis=False  # Skip model loading for test
        )
        
        # Test news fetch
        print("\n📰 Hot Crypto News:")
        news = connector.get_news(filter="hot", limit=5)
        for item in news:
            vote_emoji = "🟢" if item.sentiment_votes > 0 else ("🔴" if item.sentiment_votes < 0 else "⚪")
            hot = "🔥" if item.is_hot else "  "
            print(f"  {hot} {vote_emoji} [{item.source}] {item.title[:50]}...")
        
        # Test BTC-specific news
        print("\n₿ BTC News:")
        btc_news = connector.get_news(currencies="BTC", filter="hot", limit=3)
        for item in btc_news:
            print(f"  [{item.source}] {item.title[:50]}...")
        
        # Test sentiment (without analyzer)
        print("\n📊 News Sentiment (vote-based):")
        sentiment = connector.analyze_sentiment(currencies="BTC")
        print(f"  News count: {sentiment['news_count']}")
        print(f"  Vote sentiment: {sentiment['vote_sentiment']:+.2f}")
        print(f"  Signal: {sentiment['signal']}")
        
        # Health
        print("\n🏥 Health:")
        print(f"  {connector.health_check()}")
