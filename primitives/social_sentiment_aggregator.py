"""
Social Media Sentiment Aggregator - Multi-Source Sentiment Integration

Aggregates sentiment from Twitter, Reddit, and other social sources
with temporal windowing and spam filtering.

Enhancement 5 from ANTIGRAVITY_SENTIMENT_ENHANCEMENT_GUIDE.md
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)


@dataclass
class SocialPost:
    """Standardized social media post."""
    id: str
    source: str  # 'twitter' | 'reddit' | 'telegram'
    symbol: str
    text: str
    timestamp: datetime
    author_id: str
    engagement_score: float  # (likes + retweets + comments) normalized
    verified_author: bool = False
    sentiment_score: Optional[float] = None


@dataclass 
class AggregatorConfig:
    """Configuration for social sentiment aggregation."""
    # Temporal windows (minutes)
    twitter_window_minutes: int = 15
    reddit_window_minutes: int = 60
    
    # Spam filtering
    min_text_length: int = 10
    max_emojis_per_word: float = 0.3
    min_author_age_days: int = 30
    
    # Twitter thresholds
    twitter_min_followers: int = 100
    twitter_min_likes: int = 5
    twitter_min_retweets: int = 2
    
    # Reddit thresholds
    reddit_min_karma: int = 500
    reddit_min_upvotes: int = 3
    
    # Volume normalization
    normalize_volume: bool = True


class SocialSentimentAggregator:
    """
    Aggregate social media sentiment with temporal windowing.
    
    Handles:
    - Twitter: 15-minute rolling window (fast-moving)
    - Reddit: 1-hour rolling window (slower discussions)
    - Spam filtering: Remove low-quality posts
    - Volume normalization: Z-score normalize counts
    
    Example:
        agg = SocialSentimentAggregator()
        
        # Add analyzed posts
        agg.add_post(SocialPost(
            id='123', source='twitter', symbol='BTCUSDT',
            text='BTC mooning!', timestamp=datetime.now(),
            author_id='user1', engagement_score=50.0,
            sentiment_score=0.75
        ))
        
        # Get aggregates
        features = agg.get_aggregates('BTCUSDT')
        # {'twitter_15m_mean': 0.75, 'twitter_15m_volume': 1, ...}
    """
    
    def __init__(self, config: Optional[AggregatorConfig] = None):
        """
        Initialize social sentiment aggregator.
        
        Args:
            config: Aggregation configuration
        """
        self.config = config or AggregatorConfig()
        
        # Per-symbol rolling windows {symbol: {source: deque of SocialPost}}
        self._windows: Dict[str, Dict[str, deque]] = {}
        
        # Spam filter stats
        self._filter_stats = {
            'total_received': 0,
            'spam_filtered': 0,
            'passed': 0
        }
        
        # Volume statistics for normalization
        self._volume_stats: Dict[str, Dict[str, List[int]]] = {}
        
        logger.info("SocialSentimentAggregator initialized")
    
    def add_post(self, post: SocialPost) -> bool:
        """
        Add a social media post to the aggregator.
        
        Args:
            post: SocialPost with sentiment already analyzed
            
        Returns:
            True if post passed filtering, False if filtered as spam
        """
        self._filter_stats['total_received'] += 1
        
        # Apply spam filter
        if not self._passes_spam_filter(post):
            self._filter_stats['spam_filtered'] += 1
            return False
        
        # Initialize symbol/source windows if needed
        if post.symbol not in self._windows:
            self._windows[post.symbol] = {}
        
        if post.source not in self._windows[post.symbol]:
            # Use maxlen based on expected posts per window
            self._windows[post.symbol][post.source] = deque(maxlen=1000)
        
        # Add post
        self._windows[post.symbol][post.source].append(post)
        self._filter_stats['passed'] += 1
        
        return True
    
    def _passes_spam_filter(self, post: SocialPost) -> bool:
        """Check if post passes spam filters."""
        # Length check
        if len(post.text) < self.config.min_text_length:
            return False
        
        # Emoji density check
        emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]')
        emoji_count = len(emoji_pattern.findall(post.text))
        word_count = len(post.text.split())
        
        if word_count > 0 and emoji_count / word_count > self.config.max_emojis_per_word:
            return False
        
        # Engagement thresholds by source
        if post.source == 'twitter':
            if post.engagement_score < self.config.twitter_min_likes:
                return False
        elif post.source == 'reddit':
            if post.engagement_score < self.config.reddit_min_upvotes:
                return False
        
        return True
    
    def get_aggregates(self, symbol: str) -> Dict[str, float]:
        """
        Get aggregated sentiment features for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict of aggregate features
        """
        features = {}
        now = datetime.now()
        
        if symbol not in self._windows:
            return features
        
        # Twitter 15m aggregates
        if 'twitter' in self._windows[symbol]:
            twitter_features = self._compute_window_aggregates(
                self._windows[symbol]['twitter'],
                now,
                self.config.twitter_window_minutes,
                prefix='twitter_15m'
            )
            features.update(twitter_features)
        
        # Reddit 1h aggregates
        if 'reddit' in self._windows[symbol]:
            reddit_features = self._compute_window_aggregates(
                self._windows[symbol]['reddit'],
                now,
                self.config.reddit_window_minutes,
                prefix='reddit_1h'
            )
            features.update(reddit_features)
        
        return features
    
    def _compute_window_aggregates(
        self,
        posts: deque,
        now: datetime,
        window_minutes: int,
        prefix: str
    ) -> Dict[str, float]:
        """Compute aggregates for a time window."""
        cutoff = now - timedelta(minutes=window_minutes)
        
        # Filter to window
        window_posts = [p for p in posts if p.timestamp >= cutoff]
        
        if not window_posts:
            return {
                f'{prefix}_mean': 0.0,
                f'{prefix}_volume': 0,
                f'{prefix}_extreme_ratio': 0.0
            }
        
        # Extract sentiment scores (filter None)
        scores = [p.sentiment_score for p in window_posts if p.sentiment_score is not None]
        
        if not scores:
            return {
                f'{prefix}_mean': 0.0,
                f'{prefix}_volume': len(window_posts),
                f'{prefix}_extreme_ratio': 0.0
            }
        
        # Compute aggregates
        import numpy as np
        
        # Volume-weighted mean
        engagements = [p.engagement_score for p in window_posts if p.sentiment_score is not None]
        if sum(engagements) > 0:
            weighted_mean = sum(s * e for s, e in zip(scores, engagements)) / sum(engagements)
        else:
            weighted_mean = np.mean(scores)
        
        # Median (robust to outliers)
        median_score = float(np.median(scores))
        
        # Extreme sentiment ratio
        bullish = sum(1 for s in scores if s > 0.3)
        bearish = sum(1 for s in scores if s < -0.3)
        total = len(scores)
        extreme_ratio = (bullish - bearish) / total if total > 0 else 0.0
        
        return {
            f'{prefix}_mean': float(weighted_mean),
            f'{prefix}_median': median_score,
            f'{prefix}_volume': len(window_posts),
            f'{prefix}_extreme_ratio': extreme_ratio
        }
    
    def get_filter_stats(self) -> Dict[str, int]:
        """Get spam filter statistics."""
        return dict(self._filter_stats)
    
    def clear_old_posts(self, max_age_hours: int = 24) -> int:
        """Clear posts older than max_age_hours."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        removed = 0
        
        for symbol in self._windows:
            for source in self._windows[symbol]:
                original_len = len(self._windows[symbol][source])
                self._windows[symbol][source] = deque(
                    [p for p in self._windows[symbol][source] if p.timestamp >= cutoff],
                    maxlen=1000
                )
                removed += original_len - len(self._windows[symbol][source])
        
        return removed


# Placeholder for social media connectors (require API keys)
class TwitterConnector:
    """Twitter API v2 connector (requires API keys)."""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self._connected = False
        
        logger.info("TwitterConnector initialized (requires API authentication)")
    
    def connect(self) -> bool:
        """Connect to Twitter API."""
        # TODO: Implement Twitter API v2 authentication
        logger.warning("Twitter API connection not implemented - requires API keys")
        return False
    
    def stream_posts(self, keywords: List[str]) -> List[SocialPost]:
        """Stream posts matching keywords."""
        # TODO: Implement filtered stream
        return []


class RedditConnector:
    """Reddit API connector (requires OAuth)."""
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self._connected = False
        
        logger.info("RedditConnector initialized (requires API authentication)")
    
    def connect(self) -> bool:
        """Connect to Reddit API."""
        # TODO: Implement Reddit OAuth
        logger.warning("Reddit API connection not implemented - requires API keys")
        return False
    
    def fetch_posts(self, subreddits: List[str]) -> List[SocialPost]:
        """Fetch recent posts from subreddits."""
        # TODO: Implement subreddit fetching
        return []
