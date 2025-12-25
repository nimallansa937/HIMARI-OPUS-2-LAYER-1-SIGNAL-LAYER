"""
Rate Limiter

Token bucket rate limiter for API calls.
Prevents hitting API rate limits which would cause data gaps.
"""

import asyncio
from collections import defaultdict
from time import time
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    
    Prevents hitting API rate limits which would cause data gaps.
    Each API has its own bucket with configurable max tokens and refill rate.
    """
    
    # Default configurations for known APIs
    DEFAULT_CONFIGS = {
        'binance': {'max_tokens': 1200, 'refill_rate': 20.0},  # 1200/min
        'coingecko': {'max_tokens': 10, 'refill_rate': 0.17},  # 10/min
        'coinglass': {'max_tokens': 30, 'refill_rate': 0.5},   # 30/min
        'yahoo': {'max_tokens': 100, 'refill_rate': 2.0},      # No hard limit
        'coinbase': {'max_tokens': 10, 'refill_rate': 0.17},   # ~10/min
        'kraken': {'max_tokens': 15, 'refill_rate': 0.25},     # 15/min
    }
    
    def __init__(self):
        self.buckets: Dict[str, dict] = defaultdict(lambda: {
            'tokens': 100,
            'last_update': time(),
            'max_tokens': 100,
            'refill_rate': 1.0  # tokens per second
        })
        
        # Initialize with default configs
        for api_name, config in self.DEFAULT_CONFIGS.items():
            self.configure(api_name, config['max_tokens'], config['refill_rate'])
    
    def _refill(self, bucket: dict) -> None:
        """Refill tokens based on time elapsed."""
        now = time()
        elapsed = now - bucket['last_update']
        bucket['tokens'] = min(
            bucket['max_tokens'],
            bucket['tokens'] + elapsed * bucket['refill_rate']
        )
        bucket['last_update'] = now
    
    async def acquire(self, api_name: str, tokens: int = 1) -> bool:
        """
        Attempt to acquire tokens for API call.
        
        Args:
            api_name: Name of the API
            tokens: Number of tokens to acquire
        
        Returns:
            True if allowed, False if rate limited
        """
        bucket = self.buckets[api_name]
        self._refill(bucket)
        
        if bucket['tokens'] >= tokens:
            bucket['tokens'] -= tokens
            return True
        
        return False
    
    async def wait_and_acquire(self, api_name: str, tokens: int = 1) -> None:
        """
        Block until tokens available.
        
        Args:
            api_name: Name of the API
            tokens: Number of tokens to acquire
        """
        while not await self.acquire(api_name, tokens):
            await asyncio.sleep(0.1)
    
    def configure(self, api_name: str, max_tokens: int, refill_rate: float) -> None:
        """
        Configure rate limit for specific API.
        
        Args:
            api_name: Name of the API
            max_tokens: Maximum tokens in bucket
            refill_rate: Tokens added per second
        """
        self.buckets[api_name] = {
            'tokens': max_tokens,
            'last_update': time(),
            'max_tokens': max_tokens,
            'refill_rate': refill_rate
        }
    
    def get_remaining(self, api_name: str) -> float:
        """
        Get remaining tokens for an API.
        
        Args:
            api_name: Name of the API
        
        Returns:
            Number of remaining tokens
        """
        bucket = self.buckets[api_name]
        self._refill(bucket)
        return bucket['tokens']
    
    def get_wait_time(self, api_name: str, tokens: int = 1) -> float:
        """
        Get estimated wait time for tokens.
        
        Args:
            api_name: Name of the API
            tokens: Number of tokens needed
        
        Returns:
            Estimated wait time in seconds
        """
        bucket = self.buckets[api_name]
        self._refill(bucket)
        
        if bucket['tokens'] >= tokens:
            return 0.0
        
        needed = tokens - bucket['tokens']
        return needed / bucket['refill_rate']
