"""
Santiment On-Chain & Social Connector
======================================

Integrates with Santiment GraphQL API for:
- Social sentiment (Twitter, Reddit, Telegram aggregated)
- On-chain metrics (Active Addresses, NVT, etc.)
- Development activity
- Exchange flow

Features:
- Free tier: 1,000 API calls/month
- Historical data: Last 1 year
- Real-time data: 30-day lag on free tier
- GraphQL queries for flexibility

Get API key: https://app.santiment.net/account#api-keys

Usage:
    connector = SantimentConnector(api_key="your_key")
    
    # Get social sentiment
    sentiment = connector.get_social_sentiment("bitcoin", days=7)
    
    # Get on-chain metrics
    metrics = connector.get_onchain_metrics("bitcoin", ["active_addresses", "nvt"])
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

import requests

logger = logging.getLogger(__name__)


@dataclass
class SantimentSocialData:
    """Social sentiment data from Santiment."""
    slug: str
    timestamp: str
    sentiment_positive: float
    sentiment_negative: float
    sentiment_balance: float  # positive - negative
    social_volume: int
    
    def to_dict(self) -> Dict:
        return {
            'slug': self.slug,
            'timestamp': self.timestamp,
            'sentiment_positive': self.sentiment_positive,
            'sentiment_negative': self.sentiment_negative,
            'sentiment_balance': self.sentiment_balance,
            'social_volume': self.social_volume,
            'source': 'santiment',
        }


class SantimentConnector:
    """
    Santiment GraphQL API connector.
    
    Free tier limits:
    - 1,000 API calls per month
    - Historical data: Last 1 year
    - Real-time data: 30-day lag
    
    Get API key: https://app.santiment.net/account#api-keys
    
    Example:
        connector = SantimentConnector(api_key="your_key")
        
        # Get social sentiment
        sentiment = connector.get_social_sentiment("bitcoin", days=7)
        
        # Get on-chain metrics
        metrics = connector.get_onchain_metrics(
            "bitcoin",
            ["active_addresses", "nvt", "exchange_inflow"]
        )
        
        # Analyze with multi-model ensemble
        analysis = connector.analyze_sentiment("bitcoin")
    """
    
    GRAPHQL_URL = "https://api.santiment.net/graphql"
    
    # Asset slug mapping (Santiment uses slugs not symbols)
    ASSET_SLUGS = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'SOL': 'solana',
        'BNB': 'binancecoin',
        'ADA': 'cardano',
        'DOGE': 'dogecoin',
        'XRP': 'ripple',
        'AVAX': 'avalanche',
        'DOT': 'polkadot',
        'MATIC': 'polygon',
        'LINK': 'chainlink',
        'LTC': 'litecoin',
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        enable_sentiment_analysis: bool = True,
    ):
        """
        Initialize Santiment connector.
        
        Args:
            api_key: Santiment API key (get free at app.santiment.net/account#api-keys)
            enable_sentiment_analysis: Run data through multi-model analyzer
        """
        self.api_key = api_key or os.getenv("SANTIMENT_API_KEY")
        self.enable_sentiment_analysis = enable_sentiment_analysis
        
        if not self.api_key:
            logger.warning(
                "No Santiment API key provided. "
                "Get one free at: https://app.santiment.net/account#api-keys"
            )
        
        # Rate limiting (conservative for free tier)
        self._request_count = 0
        self._monthly_limit = 1000
        
        # Sentiment analyzer (lazy load)
        self._analyzer = None
        
        # Cache
        self._cache: Dict[str, Any] = {}
        self._cache_time: Dict[str, float] = {}
        
        logger.info("SantimentConnector initialized")
    
    def _get_analyzer(self):
        """Lazy load multi-model sentiment analyzer."""
        if self._analyzer is None and self.enable_sentiment_analysis:
            try:
                from primitives.multi_model_sentiment import create_phase2_analyzer
                self._analyzer = create_phase2_analyzer()
                logger.info("✓ Multi-model analyzer loaded for Santiment")
            except Exception as e:
                logger.warning(f"Could not load multi-model analyzer: {e}")
        return self._analyzer
    
    def _get_slug(self, symbol: str) -> str:
        """Convert symbol to Santiment slug."""
        return self.ASSET_SLUGS.get(symbol.upper(), symbol.lower())
    
    def _make_graphql_request(self, query: str, variables: Dict = None) -> Dict:
        """Make GraphQL request to Santiment."""
        if not self.api_key:
            logger.error("No API key configured")
            return {}
        
        if self._request_count >= self._monthly_limit:
            logger.warning(f"Monthly API limit reached ({self._monthly_limit})")
            return {}
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Apikey {self.api_key}",
        }
        
        payload = {
            "query": query,
            "variables": variables or {},
        }
        
        try:
            response = requests.post(
                self.GRAPHQL_URL,
                json=payload,
                headers=headers,
                timeout=15,
            )
            response.raise_for_status()
            self._request_count += 1
            
            data = response.json()
            if "errors" in data:
                logger.error(f"GraphQL errors: {data['errors']}")
                return {}
            
            return data.get("data", {})
        
        except requests.RequestException as e:
            logger.error(f"Santiment API error: {e}")
            return {}
    
    def get_social_sentiment(
        self,
        symbol: str,
        days: int = 7,
        use_cache: bool = True,
        cache_ttl: int = 300,
    ) -> List[SantimentSocialData]:
        """
        Get social sentiment time series.
        
        Args:
            symbol: Asset symbol (BTC, ETH, etc.) or slug (bitcoin, ethereum)
            days: Number of days of history
            use_cache: Use cached data if recent
            cache_ttl: Cache TTL in seconds
            
        Returns:
            List of SantimentSocialData objects
        """
        slug = self._get_slug(symbol)
        cache_key = f"social_{slug}_{days}"
        
        # Check cache
        if use_cache and cache_key in self._cache:
            age = time.time() - self._cache_time.get(cache_key, 0)
            if age < cache_ttl:
                return self._cache[cache_key]
        
        # Build GraphQL query
        from_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        to_date = datetime.utcnow().isoformat()
        
        query = """
        query GetSocialSentiment($slug: String!, $from: DateTime!, $to: DateTime!) {
          getMetric(metric: "sentiment_positive") {
            timeseriesData(
              slug: $slug
              from: $from
              to: $to
              interval: "1d"
            ) {
              datetime
              value
            }
          }
          getNegative: getMetric(metric: "sentiment_negative") {
            timeseriesData(
              slug: $slug
              from: $from
              to: $to
              interval: "1d"
            ) {
              datetime
              value
            }
          }
          getSocialVolume: getMetric(metric: "social_volume_total") {
            timeseriesData(
              slug: $slug
              from: $from
              to: $to
              interval: "1d"
            ) {
              datetime
              value
            }
          }
        }
        """
        
        variables = {
            "slug": slug,
            "from": from_date,
            "to": to_date,
        }
        
        data = self._make_graphql_request(query, variables)
        
        if not data:
            return []
        
        # Parse results
        positive_data = data.get("getMetric", {}).get("timeseriesData", [])
        negative_data = data.get("getNegative", {}).get("timeseriesData", [])
        volume_data = data.get("getSocialVolume", {}).get("timeseriesData", [])
        
        # Combine by timestamp
        result = []
        for i, item in enumerate(positive_data):
            try:
                pos_val = float(item.get("value", 0))
                neg_val = float(negative_data[i].get("value", 0)) if i < len(negative_data) else 0
                vol_val = int(volume_data[i].get("value", 0)) if i < len(volume_data) else 0
                
                result.append(SantimentSocialData(
                    slug=slug,
                    timestamp=item.get("datetime", ""),
                    sentiment_positive=pos_val,
                    sentiment_negative=neg_val,
                    sentiment_balance=pos_val - neg_val,
                    social_volume=vol_val,
                ))
            except (IndexError, ValueError, TypeError) as e:
                logger.debug(f"Skipping malformed data point: {e}")
        
        # Cache results
        self._cache[cache_key] = result
        self._cache_time[cache_key] = time.time()
        
        return result
    
    def get_onchain_metrics(
        self,
        symbol: str,
        metrics: List[str],
        days: int = 7,
    ) -> Dict[str, List[Dict]]:
        """
        Get on-chain metrics time series.
        
        Args:
            symbol: Asset symbol or slug
            metrics: List of metrics (e.g., ["active_addresses", "nvt", "exchange_inflow"])
            days: Number of days of history
            
        Returns:
            Dict mapping metric names to time series data
        """
        slug = self._get_slug(symbol)
        
        from_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        to_date = datetime.utcnow().isoformat()
        
        results = {}
        
        for metric in metrics:
            query = f"""
            query GetMetric($slug: String!, $from: DateTime!, $to: DateTime!) {{
              getMetric(metric: "{metric}") {{
                timeseriesData(
                  slug: $slug
                  from: $from
                  to: $to
                  interval: "1d"
                ) {{
                  datetime
                  value
                }}
              }}
            }}
            """
            
            variables = {
                "slug": slug,
                "from": from_date,
                "to": to_date,
            }
            
            data = self._make_graphql_request(query, variables)
            
            if data and "getMetric" in data:
                timeseries = data["getMetric"].get("timeseriesData", [])
                results[metric] = [
                    {
                        "datetime": item.get("datetime"),
                        "value": float(item.get("value", 0)),
                    }
                    for item in timeseries
                ]
        
        return results
    
    def analyze_sentiment(
        self,
        symbol: str,
        days: int = 7,
    ) -> Dict[str, Any]:
        """
        Analyze sentiment using Santiment data and multi-model ensemble.
        
        Args:
            symbol: Asset symbol
            days: Days of history to analyze
            
        Returns:
            Dict with sentiment analysis
        """
        social_data = self.get_social_sentiment(symbol, days)
        
        if not social_data:
            return {
                "symbol": symbol,
                "data_points": 0,
                "sentiment_score": 0.0,
                "error": "no_data",
            }
        
        # Calculate average sentiment
        avg_balance = sum(d.sentiment_balance for d in social_data) / len(social_data)
        avg_volume = sum(d.social_volume for d in social_data) / len(social_data)
        
        # Latest sentiment
        latest = social_data[-1] if social_data else None
        
        # Normalize to [-1, 1] scale
        # Santiment balance is typically in [-1, 1] already
        sentiment_score = avg_balance
        
        # Signal type
        if sentiment_score > 0.1:
            signal = "BULLISH"
        elif sentiment_score < -0.1:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"
        
        return {
            "symbol": symbol,
            "slug": self._get_slug(symbol),
            "data_points": len(social_data),
            "sentiment_score": sentiment_score,
            "latest_balance": latest.sentiment_balance if latest else 0,
            "latest_volume": latest.social_volume if latest else 0,
            "avg_volume": avg_volume,
            "signal": signal,
            "timeframe_days": days,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "santiment",
        }
    
    def get_popular_assets(self) -> List[Dict[str, str]]:
        """Get list of supported crypto assets."""
        return [
            {"symbol": symbol, "slug": slug}
            for symbol, slug in self.ASSET_SLUGS.items()
        ]
    
    def health_check(self) -> Dict[str, Any]:
        """Check connector health."""
        return {
            "status": "healthy" if self.api_key else "no_api_key",
            "api_key_configured": bool(self.api_key),
            "requests_this_session": self._request_count,
            "monthly_limit": self._monthly_limit,
            "remaining_estimate": self._monthly_limit - self._request_count,
            "cached_queries": len(self._cache),
            "analyzer_loaded": self._analyzer is not None,
        }


# =============================================================================
# FACTORY
# =============================================================================

def create_santiment_connector(
    api_key: Optional[str] = None,
    with_analyzer: bool = True,
) -> SantimentConnector:
    """Create Santiment connector."""
    return SantimentConnector(
        api_key=api_key,
        enable_sentiment_analysis=with_analyzer,
    )


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("SANTIMENT CONNECTOR TEST")
    print("=" * 60)
    
    api_key = os.getenv("SANTIMENT_API_KEY")
    
    if not api_key:
        print("\n⚠️  No SANTIMENT_API_KEY environment variable set")
        print("   Get free API key at: https://app.santiment.net/account#api-keys")
        print("\n   Set it with:")
        print("   export SANTIMENT_API_KEY=your_key_here")
        print("   or")
        print("   $env:SANTIMENT_API_KEY='your_key_here'  # PowerShell")
    else:
        connector = SantimentConnector(
            api_key=api_key,
            enable_sentiment_analysis=False
        )
        
        # Test assets
        print("\n📊 Supported Assets:")
        for asset in connector.get_popular_assets()[:5]:
            print(f"  {asset['symbol']}: {asset['slug']}")
        
        # Test social sentiment
        print("\n💬 BTC Social Sentiment (Last 7 Days):")
        sentiment_data = connector.get_social_sentiment("BTC", days=7)
        if sentiment_data:
            latest = sentiment_data[-1]
            print(f"  Latest: {latest.sentiment_balance:+.3f} balance")
            print(f"  Volume: {latest.social_volume:,} mentions")
            print(f"  Positive: {latest.sentiment_positive:.3f}")
            print(f"  Negative: {latest.sentiment_negative:.3f}")
        
        # Test sentiment analysis
        print("\n📈 BTC Sentiment Analysis:")
        analysis = connector.analyze_sentiment("BTC", days=7)
        print(f"  Signal: {analysis.get('signal', 'N/A')}")
        print(f"  Score: {analysis.get('sentiment_score', 0):+.3f}")
        print(f"  Data points: {analysis.get('data_points', 0)}")
        
        # Health
        print("\n🏥 Health:")
        health = connector.health_check()
        print(f"  API calls: {health['requests_this_session']}/{health['monthly_limit']}")
        print(f"  Remaining: {health['remaining_estimate']}")
