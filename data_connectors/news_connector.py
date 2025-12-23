"""
News API Connectors

NewsData.io (FREE):
- 200 requests/day
- 7+ years archive
- 86K+ sources
- Sentiment analysis included

Finnhub (FREE):
- 60 calls/minute
- Financial news with sentiment
- Company news and filings
"""

import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import requests

logger = logging.getLogger(__name__)


class NewsDataConnector:
    """
    NewsData.io FREE API connector.
    
    Rate Limits: 200 requests/day
    
    Features:
    - Crypto news from 86K+ sources
    - Sentiment analysis
    - Multi-language support
    - 7+ years of historical data
    
    Usage:
        news = NewsDataConnector(api_key="YOUR_KEY")
        articles = news.get_crypto_news()
    """
    
    BASE_URL = "https://newsdata.io/api/1"
    
    # Crypto-related keywords
    CRYPTO_KEYWORDS = [
        "bitcoin", "ethereum", "cryptocurrency", "crypto",
        "blockchain", "defi", "nft", "altcoin", "binance"
    ]
    
    def __init__(self, api_key: str):
        """
        Initialize NewsData connector.
        
        Args:
            api_key: NewsData.io API key (free tier available)
        """
        self.api_key = api_key
        self.session = requests.Session()
        self._last_request_time = 0
        self._daily_requests = 0
    
    def _request(self, endpoint: str, params: Dict) -> Dict:
        """Rate-limited request."""
        # 200/day = ~8.3/hour, be conservative
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < 60:  # 1 request per minute max
            time.sleep(60 - time_since_last)
        
        params['apikey'] = self.api_key
        url = f"{self.BASE_URL}/{endpoint}"
        
        response = self.session.get(url, params=params)
        self._last_request_time = time.time()
        self._daily_requests += 1
        
        if response.status_code == 429:
            logger.warning("Rate limited by NewsData.io")
            return {'status': 'error', 'results': []}
        
        response.raise_for_status()
        return response.json()
    
    def get_crypto_news(
        self,
        keywords: Optional[List[str]] = None,
        language: str = "en",
        size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get latest crypto news.
        
        Args:
            keywords: Additional keywords to search
            language: Language code
            size: Number of articles
            
        Returns:
            List of news articles
        """
        query = " OR ".join(keywords or self.CRYPTO_KEYWORDS[:5])
        
        params = {
            'q': query,
            'language': language,
            'category': 'business',
            'size': size,
        }
        
        data = self._request("news", params)
        
        articles = []
        for article in data.get('results', []):
            articles.append({
                'title': article.get('title'),
                'description': article.get('description'),
                'content': article.get('content', '')[:500],
                'source': article.get('source_id'),
                'url': article.get('link'),
                'published_at': article.get('pubDate'),
                'keywords': article.get('keywords', []),
                'category': article.get('category', []),
                'sentiment': article.get('sentiment'),  # If available
                'language': article.get('language'),
            })
        
        return articles
    
    def get_news_by_source(
        self,
        sources: List[str],
        size: int = 10
    ) -> List[Dict]:
        """Get news from specific sources."""
        params = {
            'domain': ','.join(sources),
            'size': size,
        }
        
        data = self._request("news", params)
        return data.get('results', [])


class FinnhubConnector:
    """
    Finnhub FREE API connector.
    
    Rate Limits: 60 calls/minute
    
    Features:
    - Financial news with sentiment scores
    - Company news
    - Market news
    - Crypto news
    
    Usage:
        fh = FinnhubConnector(api_key="YOUR_KEY")
        news = fh.get_general_news()
        sentiment = fh.get_news_sentiment("AAPL")
    """
    
    BASE_URL = "https://finnhub.io/api/v1"
    
    def __init__(self, api_key: str):
        """
        Initialize Finnhub connector.
        
        Args:
            api_key: Finnhub API key (free tier available)
        """
        self.api_key = api_key
        self.session = requests.Session()
        self._last_request_time = 0
    
    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """Rate-limited request."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < 1.0:  # 60/min = 1/sec
            time.sleep(1.0 - time_since_last)
        
        params = params or {}
        params['token'] = self.api_key
        url = f"{self.BASE_URL}/{endpoint}"
        
        response = self.session.get(url, params=params)
        self._last_request_time = time.time()
        
        if response.status_code == 429:
            logger.warning("Rate limited by Finnhub, waiting 60s")
            time.sleep(60)
            return self._request(endpoint, params)
        
        response.raise_for_status()
        return response.json()
    
    def get_general_news(self, category: str = "crypto") -> List[Dict[str, Any]]:
        """
        Get general market news.
        
        Args:
            category: general, forex, crypto, merger
            
        Returns:
            List of news articles
        """
        data = self._request("news", {'category': category})
        
        articles = []
        for article in data:
            articles.append({
                'id': article.get('id'),
                'title': article.get('headline'),
                'summary': article.get('summary'),
                'source': article.get('source'),
                'url': article.get('url'),
                'image': article.get('image'),
                'category': article.get('category'),
                'published_at': datetime.fromtimestamp(article.get('datetime', 0)),
                'related': article.get('related'),
            })
        
        return articles
    
    def get_company_news(
        self,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get news for a specific company.
        
        Args:
            symbol: Stock symbol
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            
        Returns:
            List of news articles
        """
        if not from_date:
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')
        
        params = {
            'symbol': symbol,
            'from': from_date,
            'to': to_date,
        }
        
        data = self._request("company-news", params)
        
        return [
            {
                'id': a.get('id'),
                'title': a.get('headline'),
                'summary': a.get('summary'),
                'source': a.get('source'),
                'url': a.get('url'),
                'published_at': datetime.fromtimestamp(a.get('datetime', 0)),
            }
            for a in data
        ]
    
    def get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get news sentiment for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Sentiment scores and article breakdown
        """
        data = self._request("news-sentiment", {'symbol': symbol})
        
        return {
            'symbol': symbol,
            'company_news_score': data.get('companyNewsScore'),
            'sector_avg_bullish': data.get('sectorAverageBullishPercent'),
            'sector_avg_sentiment': data.get('sectorAverageNewsScore'),
            'buzz': data.get('buzz', {}),
            'sentiment': data.get('sentiment', {}),
        }
    
    def get_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get social media sentiment (Reddit, Twitter).
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Social sentiment data
        """
        data = self._request("stock/social-sentiment", {'symbol': symbol})
        
        reddit_data = data.get('reddit', [])
        twitter_data = data.get('twitter', [])
        
        return {
            'symbol': symbol,
            'reddit': reddit_data[-1] if reddit_data else {},
            'twitter': twitter_data[-1] if twitter_data else {},
            'reddit_history': reddit_data[-7:],  # Last 7 entries
            'twitter_history': twitter_data[-7:],
        }


# Quick test
if __name__ == "__main__":
    print("News connectors require API keys.")
    print("Get free keys at:")
    print("  - NewsData.io: https://newsdata.io/register")
    print("  - Finnhub: https://finnhub.io/register")
