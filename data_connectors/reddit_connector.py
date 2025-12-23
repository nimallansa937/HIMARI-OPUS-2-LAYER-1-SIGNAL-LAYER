"""
Reddit API Connector (using PRAW or direct API)

FREE tier: 60 requests/minute
Covers: Posts, comments, sentiment from crypto subreddits

No API key required for read-only public data via old.reddit.com/JSON
"""

import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import requests

logger = logging.getLogger(__name__)


class RedditConnector:
    """
    Reddit connector for crypto sentiment analysis.
    
    Uses either:
    - PRAW (Python Reddit API Wrapper) if available
    - Direct JSON API (no auth required for public posts)
    
    Rate Limits: 60 requests/minute (free tier)
    
    Usage:
        reddit = RedditConnector()
        posts = reddit.get_subreddit_posts("cryptocurrency", limit=25)
        hot = reddit.get_hot_crypto_posts()
    """
    
    # Key crypto subreddits to monitor
    CRYPTO_SUBREDDITS = [
        'cryptocurrency',
        'bitcoin',
        'ethereum',
        'altcoin',
        'CryptoMarkets',
        'binance',
        'defi',
        'solana',
        'cardano',
    ]
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: str = "HIMARI-Sentiment-Bot/1.0"
    ):
        """
        Initialize Reddit connector.
        
        Args:
            client_id: Reddit app client ID (optional)
            client_secret: Reddit app client secret (optional)
            user_agent: User agent for requests
        """
        self._use_praw = False
        self._praw = None
        
        # Try to use PRAW if credentials provided
        if client_id and client_secret:
            try:
                import praw
                self._praw = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent
                )
                self._use_praw = True
                logger.info("Using PRAW for Reddit API")
            except ImportError:
                logger.warning("PRAW not installed, using JSON API fallback")
        
        # Fallback to JSON API
        self.session = requests.Session()
        self.session.headers['User-Agent'] = user_agent
        self._last_request_time = 0
    
    def _json_request(self, url: str) -> Dict:
        """Rate-limited JSON request."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < 1.0:  # 60 req/min = 1 req/sec
            time.sleep(1.0 - time_since_last)
        
        response = self.session.get(url)
        self._last_request_time = time.time()
        
        if response.status_code == 429:
            logger.warning("Rate limited by Reddit, waiting 60s")
            time.sleep(60)
            return self._json_request(url)
        
        response.raise_for_status()
        return response.json()
    
    def get_subreddit_posts(
        self,
        subreddit: str,
        sort: str = "hot",
        limit: int = 25,
        time_filter: str = "day"
    ) -> List[Dict[str, Any]]:
        """
        Get posts from a subreddit.
        
        Args:
            subreddit: Subreddit name
            sort: hot, new, top, rising
            limit: Number of posts (max 100)
            time_filter: hour, day, week, month, year, all (for top)
            
        Returns:
            List of post data
        """
        if self._use_praw:
            return self._get_posts_praw(subreddit, sort, limit, time_filter)
        else:
            return self._get_posts_json(subreddit, sort, limit, time_filter)
    
    def _get_posts_json(
        self,
        subreddit: str,
        sort: str,
        limit: int,
        time_filter: str
    ) -> List[Dict]:
        """Get posts using JSON API."""
        url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
        params = {'limit': limit, 't': time_filter}
        
        data = self._json_request(f"{url}?limit={limit}&t={time_filter}")
        posts = []
        
        for child in data.get('data', {}).get('children', []):
            post = child.get('data', {})
            posts.append({
                'id': post.get('id'),
                'title': post.get('title'),
                'author': post.get('author'),
                'score': post.get('score', 0),
                'upvote_ratio': post.get('upvote_ratio', 0),
                'num_comments': post.get('num_comments', 0),
                'created_utc': post.get('created_utc'),
                'created_dt': datetime.fromtimestamp(post.get('created_utc', 0)),
                'selftext': post.get('selftext', '')[:500],  # Truncate
                'url': post.get('url'),
                'permalink': f"https://reddit.com{post.get('permalink')}",
                'subreddit': post.get('subreddit'),
                'is_self': post.get('is_self'),
                'flair': post.get('link_flair_text'),
            })
        
        return posts
    
    def _get_posts_praw(
        self,
        subreddit: str,
        sort: str,
        limit: int,
        time_filter: str
    ) -> List[Dict]:
        """Get posts using PRAW."""
        sub = self._praw.subreddit(subreddit)
        
        if sort == "hot":
            submissions = sub.hot(limit=limit)
        elif sort == "new":
            submissions = sub.new(limit=limit)
        elif sort == "top":
            submissions = sub.top(time_filter=time_filter, limit=limit)
        elif sort == "rising":
            submissions = sub.rising(limit=limit)
        else:
            submissions = sub.hot(limit=limit)
        
        posts = []
        for post in submissions:
            posts.append({
                'id': post.id,
                'title': post.title,
                'author': str(post.author),
                'score': post.score,
                'upvote_ratio': post.upvote_ratio,
                'num_comments': post.num_comments,
                'created_utc': post.created_utc,
                'created_dt': datetime.fromtimestamp(post.created_utc),
                'selftext': post.selftext[:500] if post.selftext else '',
                'url': post.url,
                'permalink': f"https://reddit.com{post.permalink}",
                'subreddit': str(post.subreddit),
                'is_self': post.is_self,
                'flair': post.link_flair_text,
            })
        
        return posts
    
    def get_hot_crypto_posts(self, limit_per_sub: int = 10) -> List[Dict]:
        """
        Get hot posts from all crypto subreddits.
        
        Args:
            limit_per_sub: Posts per subreddit
            
        Returns:
            Combined list of posts
        """
        all_posts = []
        
        for subreddit in self.CRYPTO_SUBREDDITS:
            try:
                posts = self.get_subreddit_posts(
                    subreddit, 
                    sort="hot", 
                    limit=limit_per_sub
                )
                all_posts.extend(posts)
            except Exception as e:
                logger.warning(f"Failed to get posts from r/{subreddit}: {e}")
        
        # Sort by score
        all_posts.sort(key=lambda x: x['score'], reverse=True)
        return all_posts
    
    def search_posts(
        self,
        query: str,
        subreddit: Optional[str] = None,
        sort: str = "relevance",
        limit: int = 25
    ) -> List[Dict]:
        """
        Search for posts.
        
        Args:
            query: Search query
            subreddit: Limit to specific subreddit
            sort: relevance, hot, top, new, comments
            limit: Max results
            
        Returns:
            List of matching posts
        """
        if subreddit:
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
            params = f"?q={query}&restrict_sr=1&sort={sort}&limit={limit}"
        else:
            url = "https://www.reddit.com/search.json"
            params = f"?q={query}&sort={sort}&limit={limit}"
        
        data = self._json_request(url + params)
        
        posts = []
        for child in data.get('data', {}).get('children', []):
            post = child.get('data', {})
            posts.append({
                'id': post.get('id'),
                'title': post.get('title'),
                'score': post.get('score', 0),
                'num_comments': post.get('num_comments', 0),
                'subreddit': post.get('subreddit'),
                'created_utc': post.get('created_utc'),
                'permalink': f"https://reddit.com{post.get('permalink')}",
            })
        
        return posts
    
    def get_coin_mentions(
        self,
        coin_symbol: str,
        limit: int = 25
    ) -> Dict[str, Any]:
        """
        Get mentions of a specific coin across crypto subreddits.
        
        Args:
            coin_symbol: Coin symbol (e.g., "BTC", "ETH")
            limit: Posts to search
            
        Returns:
            Dict with mention stats and posts
        """
        # Search all crypto subreddits
        posts = self.search_posts(
            query=f"{coin_symbol} OR ${coin_symbol}",
            subreddit="cryptocurrency+bitcoin+ethereum+altcoin",
            sort="new",
            limit=limit
        )
        
        # Calculate basic sentiment metrics
        total_score = sum(p['score'] for p in posts)
        total_comments = sum(p['num_comments'] for p in posts)
        
        return {
            'symbol': coin_symbol,
            'mention_count': len(posts),
            'total_score': total_score,
            'total_comments': total_comments,
            'avg_score': total_score / len(posts) if posts else 0,
            'posts': posts,
        }
    
    def calculate_sentiment_score(self, posts: List[Dict]) -> float:
        """
        Calculate simple sentiment score from post engagement.
        
        Higher score = more positive/bullish sentiment
        Based on upvote ratio and engagement.
        
        Returns:
            Sentiment score from -1 to +1
        """
        if not posts:
            return 0.0
        
        weighted_scores = []
        for post in posts:
            # Weight by engagement
            engagement = post['score'] + post['num_comments']
            upvote_ratio = post.get('upvote_ratio', 0.5)
            
            # Convert upvote ratio to -1 to +1 scale
            sentiment = (upvote_ratio - 0.5) * 2
            weighted_scores.append(sentiment * (1 + engagement / 100))
        
        avg_sentiment = sum(weighted_scores) / len(weighted_scores)
        return max(-1.0, min(1.0, avg_sentiment))


# Quick test
if __name__ == "__main__":
    reddit = RedditConnector()
    
    print("Testing Reddit Connector...")
    posts = reddit.get_subreddit_posts("cryptocurrency", limit=5)
    print(f"Got {len(posts)} posts from r/cryptocurrency")
    
    for post in posts[:3]:
        print(f"  [{post['score']:+4d}] {post['title'][:60]}...")
    
    sentiment = reddit.calculate_sentiment_score(posts)
    print(f"Sentiment score: {sentiment:+.2f}")
    
    print("✓ Reddit connector working!")
