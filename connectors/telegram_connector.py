"""
Telegram Public Channel Scraper
================================

Scrapes real-time crypto alpha from public Telegram channels.
No API key required - uses web scraping for public channels.

Top Crypto Alpha Channels:
- @cryptowhalesignal
- @cryptowhalepump  
- @wallstreetbetsshitcoins
- @elonmusktweets
- @binancepumpsignals

Features:
- Real-time message scraping
- No authentication required
- CryptoBERT sentiment integration
- Influence scoring by channel size

Usage:
    connector = TelegramConnector()
    
    # Get recent messages
    messages = connector.get_messages("@cryptowhalesignal", limit=10)
    
    # Analyze sentiment
    sentiment = connector.analyze_sentiment("@cryptowhalesignal")
"""

import os
import time
import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# Top crypto Telegram channels (public, no auth needed)
CRYPTO_TELEGRAM_CHANNELS = {
    'whales': '@cryptowhalesignal',
    'pumps': '@binancepumpsignals',
    'wsb': '@wallstreetbetsshitcoins',
    'alpha': '@cryptoalphacalls',
    'moonshots': '@moonshotcalls',
    'gems': '@cryptogems100x',
    'technical': '@cryptotechnicalanalysis',
    'news': '@cryptonewsofficials',
}


@dataclass
class TelegramMessage:
    """A single Telegram message."""
    channel: str
    text: str
    timestamp: str
    views: int
    message_id: int
    
    def to_dict(self) -> Dict:
        return {
            'channel': self.channel,
            'text': self.text,
            'timestamp': self.timestamp,
            'views': self.views,
            'message_id': self.message_id,
            'source': 'telegram',
        }


class TelegramConnector:
    """
    Telegram public channel scraper.
    
    Scrapes public Telegram channels via t.me preview (no API needed).
    
    Example:
        connector = TelegramConnector()
        
        # Get latest messages
        messages = connector.get_messages("@cryptowhalesignal", limit=10)
        for msg in messages:
            print(f"[{msg.views} views] {msg.text[:50]}...")
        
        # Analyze sentiment
        sentiment = connector.analyze_sentiment("@cryptowhalesignal")
        print(f"Signal: {sentiment['signal']}")
    """
    
    BASE_URL = "https://t.me/s"
    
    def __init__(
        self,
        enable_sentiment_analysis: bool = True,
        user_agent: str = None,
    ):
        """
        Initialize Telegram connector.
        
        Args:
            enable_sentiment_analysis: Run messages through multi-model analyzer
            user_agent: Custom user agent (default uses Chrome)
        """
        self.enable_sentiment_analysis = enable_sentiment_analysis
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        
        # Sentiment analyzer (lazy load)
        self._analyzer = None
        
        # Cache
        self._cache: Dict[str, List[TelegramMessage]] = {}
        self._cache_time: Dict[str, float] = {}
        
        logger.info("TelegramConnector initialized (public channels, no auth)")
    
    def _get_analyzer(self):
        """Lazy load multi-model sentiment analyzer."""
        if self._analyzer is None and self.enable_sentiment_analysis:
            try:
                from primitives.multi_model_sentiment import create_phase2_analyzer
                self._analyzer = create_phase2_analyzer()
                logger.info("✓ Multi-model analyzer loaded for Telegram")
            except Exception as e:
                logger.warning(f"Could not load multi-model analyzer: {e}")
        return self._analyzer
    
    def _clean_channel_name(self, channel: str) -> str:
        """Clean channel name (remove @ if present)."""
        return channel.lstrip('@')
    
    def _make_request(self, url: str) -> Optional[str]:
        """Make HTTP request to Telegram."""
        headers = {
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Telegram request error: {e}")
            return None
    
    def get_messages(
        self,
        channel: str,
        limit: int = 20,
        use_cache: bool = True,
        cache_ttl: int = 60,
    ) -> List[TelegramMessage]:
        """
        Get recent messages from a public Telegram channel.
        
        Args:
            channel: Channel username (with or without @)
            limit: Max messages to retrieve
            use_cache: Use cached messages if recent
            cache_ttl: Cache TTL in seconds
            
        Returns:
            List of TelegramMessage objects
        """
        channel = self._clean_channel_name(channel)
        
        # Check cache
        if use_cache and channel in self._cache:
            age = time.time() - self._cache_time.get(channel, 0)
            if age < cache_ttl:
                return self._cache[channel][:limit]
        
        # Fetch from Telegram
        url = f"{self.BASE_URL}/{channel}"
        html = self._make_request(url)
        
        if not html:
            return []
        
        # Parse HTML
        soup = BeautifulSoup(html, 'html.parser')
        messages = []
        
        # Find all message divs
        message_divs = soup.find_all('div', class_='tgme_widget_message')
        
        for div in message_divs[:limit]:
            try:
                # Extract text
                text_div = div.find('div', class_='tgme_widget_message_text')
                if not text_div:
                    continue
                text = text_div.get_text(strip=True)
                
                if not text or len(text) < 10:
                    continue
                
                # Extract message ID
                msg_id = None
                data_post = div.get('data-post')
                if data_post:
                    msg_id = int(data_post.split('/')[-1])
                
                # Extract timestamp
                time_tag = div.find('time')
                timestamp = time_tag.get('datetime') if time_tag else None
                
                # Extract views
                views = 0
                views_span = div.find('span', class_='tgme_widget_message_views')
                if views_span:
                    views_text = views_span.get_text(strip=True)
                    # Convert "1.2K" to 1200, etc.
                    if 'K' in views_text:
                        views = int(float(views_text.replace('K', '')) * 1000)
                    elif 'M' in views_text:
                        views = int(float(views_text.replace('M', '')) * 1000000)
                    else:
                        views = int(re.sub(r'[^\d]', '', views_text) or 0)
                
                messages.append(TelegramMessage(
                    channel=f"@{channel}",
                    text=text,
                    timestamp=timestamp or datetime.utcnow().isoformat(),
                    views=views,
                    message_id=msg_id or 0,
                ))
            except Exception as e:
                logger.debug(f"Skipping malformed message: {e}")
        
        # Cache results
        self._cache[channel] = messages
        self._cache_time[channel] = time.time()
        
        return messages
    
    def analyze_sentiment(
        self,
        channel: str,
        limit: int = 20,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze sentiment for a channel using multi-model ensemble.
        
        Args:
            channel: Channel username
            limit: Max messages to analyze
            use_cache: Use cached messages if recent
            
        Returns:
            Dict with sentiment scores and breakdown
        """
        messages = self.get_messages(channel, limit, use_cache)
        
        if not messages:
            return {
                "channel": channel,
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
            "channel": channel,
            "message_count": len(messages),
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "model_predictions": [],
            "weighted_score": 0.0,
            "total_views": sum(msg.views for msg in messages),
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for msg in messages:
            score = 0.0
            
            # Analyze with multi-model
            if analyzer:
                try:
                    pred = analyzer.analyze(msg.text, source="telegram")
                    if pred:
                        score = pred.score
                        
                        if pred.label == "bullish":
                            results["bullish_count"] += 1
                        elif pred.label == "bearish":
                            results["bearish_count"] += 1
                        else:
                            results["neutral_count"] += 1
                        
                        results["model_predictions"].append({
                            "text": msg.text[:50],
                            "score": score,
                            "label": pred.label,
                            "model": pred.model_name,
                            "views": msg.views,
                        })
                    else:
                        results["neutral_count"] += 1
                except Exception as e:
                    logger.debug(f"Analysis failed: {e}")
                    results["neutral_count"] += 1
            else:
                results["neutral_count"] += 1
            
            # Weight by views (virality)
            weight = 1 + min(msg.views / 1000, 50)  # Cap at 50x
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
    
    def get_popular_channels(self) -> List[Dict[str, str]]:
        """Get list of popular crypto Telegram channels."""
        return [
            {"name": name, "channel": channel}
            for name, channel in CRYPTO_TELEGRAM_CHANNELS.items()
        ]
    
    def health_check(self) -> Dict[str, Any]:
        """Check connector health."""
        return {
            "status": "healthy",
            "cached_channels": list(self._cache.keys()),
            "total_cached_messages": sum(len(msgs) for msgs in self._cache.values()),
            "analyzer_loaded": self._analyzer is not None,
        }


# =============================================================================
# FACTORY
# =============================================================================

def create_telegram_connector(
    with_analyzer: bool = True
) -> TelegramConnector:
    """Create Telegram connector with optional analyzer."""
    return TelegramConnector(enable_sentiment_analysis=with_analyzer)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("TELEGRAM CONNECTOR TEST")
    print("=" * 60)
    
    connector = TelegramConnector(enable_sentiment_analysis=False)
    
    # Test channel list
    print("\n📱 Popular Crypto Channels:")
    for ch in connector.get_popular_channels()[:5]:
        print(f"  {ch['name']}: {ch['channel']}")
    
    # Test message fetch
    print("\n💬 Latest Messages from @cryptowhalesignal:")
    messages = connector.get_messages("@cryptowhalesignal", limit=5)
    
    if messages:
        for msg in messages:
            views_k = msg.views / 1000 if msg.views > 0 else 0
            print(f"  [{views_k:.1f}K views] {msg.text[:60]}...")
    else:
        print("  ⚠️  Could not fetch messages (check internet connection)")
    
    # Test sentiment (without analyzer to save time)
    if messages:
        print("\n📊 Channel Sentiment:")
        sentiment = connector.analyze_sentiment("@cryptowhalesignal", limit=5)
        print(f"  Messages: {sentiment['message_count']}")
        print(f"  Total views: {sentiment['total_views']:,}")
        print(f"  Signal: {sentiment.get('signal', 'N/A')}")
    
    # Health
    print("\n🏥 Health:")
    print(f"  {connector.health_check()}")
