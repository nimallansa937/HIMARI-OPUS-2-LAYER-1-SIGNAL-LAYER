"""
Lightweight Web Crawler

Rate-limited web fetcher for harvesting trading strategy ideas
from external sources like arXiv, blogs, and research sites.
"""

import asyncio
import logging
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set
from datetime import datetime, timedelta
from urllib.parse import urlparse
import re

logger = logging.getLogger(__name__)


@dataclass
class SourceConfig:
    """Configuration for a crawl source."""
    name: str
    base_url: str
    source_type: str  # "arxiv", "blog", "forum", "paper"
    rate_limit_rps: float = 0.5  # Requests per second
    enabled: bool = True

    # Selectors/patterns for content extraction
    content_selector: Optional[str] = None  # CSS selector
    link_pattern: Optional[str] = None  # Regex for links to follow

    # Filtering
    min_content_length: int = 500
    max_pages_per_crawl: int = 20


@dataclass
class CrawlResult:
    """Result from crawling a single URL."""
    url: str
    source: str
    title: str
    content: str
    published_date: Optional[datetime]
    authors: List[str]
    content_hash: str
    crawl_timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if crawl result has valid content."""
        return len(self.content) >= 100 and len(self.title) > 0


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, rate: float):
        """
        Initialize rate limiter.

        Args:
            rate: Requests per second allowed
        """
        self.rate = rate
        self.tokens = 1.0
        self.last_update = datetime.now()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = datetime.now()
            elapsed = (now - self.last_update).total_seconds()
            self.tokens = min(1.0, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens < 1.0:
                wait_time = (1.0 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0.0
            else:
                self.tokens -= 1.0


class LightweightWebCrawler:
    """
    Rate-limited web crawler for harvesting trading ideas.

    Features:
    - Configurable rate limiting per source
    - Deduplication via content hashing
    - Async operation for efficiency
    - arXiv API integration
    """

    def __init__(
        self,
        sources: Optional[List[SourceConfig]] = None,
        cache_ttl_hours: int = 24,
        max_retries: int = 3
    ):
        """
        Initialize crawler.

        Args:
            sources: List of source configurations
            cache_ttl_hours: How long to cache seen URLs
            max_retries: Max retries for failed requests
        """
        self.sources = sources or self._default_sources()
        self.cache_ttl_hours = cache_ttl_hours
        self.max_retries = max_retries

        # Rate limiters per source
        self._rate_limiters: Dict[str, RateLimiter] = {
            s.name: RateLimiter(s.rate_limit_rps)
            for s in self.sources
        }

        # Seen content hashes (deduplication)
        self._seen_hashes: Set[str] = set()
        self._seen_urls: Dict[str, datetime] = {}

    def _default_sources(self) -> List[SourceConfig]:
        """Default source configurations."""
        return [
            SourceConfig(
                name="arxiv_qfin",
                base_url="https://arxiv.org",
                source_type="arxiv",
                rate_limit_rps=0.3,
                max_pages_per_crawl=30
            ),
            SourceConfig(
                name="ssrn",
                base_url="https://papers.ssrn.com",
                source_type="paper",
                rate_limit_rps=0.2,
                max_pages_per_crawl=20
            )
        ]

    async def crawl_source(
        self,
        source_name: str,
        query: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> List[CrawlResult]:
        """
        Crawl a specific source.

        Args:
            source_name: Name of source to crawl
            query: Optional search query
            max_results: Maximum results to return

        Returns:
            List of crawl results
        """
        source = next((s for s in self.sources if s.name == source_name), None)
        if not source or not source.enabled:
            logger.warning(f"Source {source_name} not found or disabled")
            return []

        if source.source_type == "arxiv":
            return await self._crawl_arxiv(source, query, max_results)
        else:
            return await self._crawl_generic(source, query, max_results)

    async def crawl_all(
        self,
        query: Optional[str] = None,
        max_results_per_source: int = 20
    ) -> List[CrawlResult]:
        """Crawl all enabled sources."""
        all_results = []

        for source in self.sources:
            if source.enabled:
                try:
                    results = await self.crawl_source(
                        source.name,
                        query,
                        max_results_per_source
                    )
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"Error crawling {source.name}: {e}")

        return all_results

    async def _crawl_arxiv(
        self,
        source: SourceConfig,
        query: Optional[str],
        max_results: Optional[int]
    ) -> List[CrawlResult]:
        """Crawl arXiv using their API."""
        try:
            import arxiv
        except ImportError:
            logger.warning("arxiv package not installed, using mock data")
            return self._mock_arxiv_results(max_results or 10)

        results = []
        max_results = max_results or source.max_pages_per_crawl

        # Build search query
        search_query = query or "cat:q-fin.TR OR cat:q-fin.PM OR cat:q-fin.ST"

        try:
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )

            rate_limiter = self._rate_limiters[source.name]

            for paper in search.results():
                await rate_limiter.acquire()

                # Check if already seen
                content_hash = self._compute_hash(paper.summary)
                if content_hash in self._seen_hashes:
                    continue

                result = CrawlResult(
                    url=paper.entry_id,
                    source=source.name,
                    title=paper.title,
                    content=paper.summary,
                    published_date=paper.published,
                    authors=[a.name for a in paper.authors],
                    content_hash=content_hash,
                    crawl_timestamp=datetime.now(),
                    metadata={
                        'arxiv_id': paper.entry_id.split('/')[-1],
                        'categories': paper.categories,
                        'pdf_url': paper.pdf_url,
                        'primary_category': paper.primary_category
                    }
                )

                if result.is_valid:
                    self._seen_hashes.add(content_hash)
                    results.append(result)

        except Exception as e:
            logger.error(f"arXiv crawl failed: {e}")

        logger.info(f"Crawled {len(results)} papers from arXiv")
        return results

    async def _crawl_generic(
        self,
        source: SourceConfig,
        query: Optional[str],
        max_results: Optional[int]
    ) -> List[CrawlResult]:
        """Generic web page crawler."""
        try:
            import aiohttp
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning("aiohttp/bs4 not installed, using mock data")
            return self._mock_generic_results(source.name, max_results or 5)

        results = []
        max_results = max_results or source.max_pages_per_crawl
        rate_limiter = self._rate_limiters[source.name]

        try:
            async with aiohttp.ClientSession() as session:
                # Start from base URL or search URL
                url = source.base_url
                if query and '?' in source.base_url:
                    url = f"{source.base_url}&q={query}"
                elif query:
                    url = f"{source.base_url}?q={query}"

                await rate_limiter.acquire()

                async with session.get(url, timeout=30) as response:
                    if response.status != 200:
                        logger.warning(f"Got status {response.status} from {url}")
                        return results

                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    # Extract content based on selector
                    if source.content_selector:
                        elements = soup.select(source.content_selector)
                    else:
                        elements = soup.find_all('article') or soup.find_all('div', class_='content')

                    for elem in elements[:max_results]:
                        title_elem = elem.find(['h1', 'h2', 'h3', 'a'])
                        title = title_elem.get_text(strip=True) if title_elem else "Untitled"

                        content = elem.get_text(strip=True)
                        content_hash = self._compute_hash(content)

                        if content_hash in self._seen_hashes:
                            continue

                        if len(content) < source.min_content_length:
                            continue

                        result = CrawlResult(
                            url=url,
                            source=source.name,
                            title=title,
                            content=content,
                            published_date=None,
                            authors=[],
                            content_hash=content_hash,
                            crawl_timestamp=datetime.now()
                        )

                        if result.is_valid:
                            self._seen_hashes.add(content_hash)
                            results.append(result)

        except Exception as e:
            logger.error(f"Generic crawl of {source.name} failed: {e}")

        return results

    def _compute_hash(self, content: str) -> str:
        """Compute content hash for deduplication."""
        # Normalize whitespace
        normalized = ' '.join(content.split())
        return hashlib.md5(normalized.encode()).hexdigest()

    def _mock_arxiv_results(self, n: int) -> List[CrawlResult]:
        """Generate mock arXiv results for testing."""
        mock_papers = [
            {
                "title": "Deep Reinforcement Learning for Momentum Trading Strategies",
                "abstract": "We propose a novel deep reinforcement learning approach for momentum trading "
                           "in cryptocurrency markets. Our method combines technical indicators with "
                           "order flow features to detect trend continuation patterns. Backtesting on "
                           "BTC/USDT shows a Sharpe ratio of 2.1 with maximum drawdown of 15%.",
                "authors": ["Alice Chen", "Bob Smith"],
                "categories": ["q-fin.TR", "cs.LG"]
            },
            {
                "title": "Funding Rate Arbitrage in Perpetual Futures Markets",
                "abstract": "This paper analyzes funding rate dynamics in cryptocurrency perpetual futures "
                           "markets. We identify systematic patterns in funding rate mean reversion and "
                           "develop a delta-neutral arbitrage strategy. The strategy achieves consistent "
                           "returns with minimal drawdown by exploiting funding rate extremes.",
                "authors": ["Charlie Wang"],
                "categories": ["q-fin.PM", "q-fin.TR"]
            },
            {
                "title": "Order Flow Imbalance and Short-Term Price Prediction",
                "abstract": "We study the relationship between order flow imbalance and short-term price "
                           "movements in centralized cryptocurrency exchanges. Using tick-level data, we "
                           "develop features from order book snapshots that predict 1-minute returns with "
                           "high accuracy. The causal mechanism relates to liquidity provision dynamics.",
                "authors": ["Diana Lee", "Eric Johnson"],
                "categories": ["q-fin.ST", "q-fin.TR"]
            },
            {
                "title": "Volatility Regime Detection Using Hidden Markov Models",
                "abstract": "We propose a Hidden Markov Model approach for detecting volatility regimes "
                           "in cryptocurrency markets. Our model identifies distinct market states "
                           "(trending, ranging, volatile) and enables adaptive strategy selection. "
                           "Portfolio performance improves by 40% when strategies are matched to regimes.",
                "authors": ["Frank Miller"],
                "categories": ["q-fin.PM"]
            },
            {
                "title": "Cross-Exchange Latency Arbitrage Detection",
                "abstract": "This paper presents methods for detecting and exploiting price discrepancies "
                           "across cryptocurrency exchanges. We analyze the microstructure of cross-exchange "
                           "arbitrage and develop a low-latency strategy that captures small but consistent "
                           "profits from price divergences.",
                "authors": ["Grace Kim", "Henry Zhang"],
                "categories": ["q-fin.TR"]
            }
        ]

        results = []
        for i, paper in enumerate(mock_papers[:n]):
            content_hash = self._compute_hash(paper["abstract"])
            results.append(CrawlResult(
                url=f"https://arxiv.org/abs/2024.{10000 + i}",
                source="arxiv_qfin",
                title=paper["title"],
                content=paper["abstract"],
                published_date=datetime.now() - timedelta(days=i * 7),
                authors=paper["authors"],
                content_hash=content_hash,
                crawl_timestamp=datetime.now(),
                metadata={
                    'arxiv_id': f"2024.{10000 + i}",
                    'categories': paper["categories"]
                }
            ))

        return results

    def _mock_generic_results(self, source: str, n: int) -> List[CrawlResult]:
        """Generate mock generic results for testing."""
        results = []
        for i in range(n):
            content = f"Mock content from {source} article {i}. " * 50
            results.append(CrawlResult(
                url=f"https://example.com/{source}/{i}",
                source=source,
                title=f"Article {i} from {source}",
                content=content,
                published_date=datetime.now(),
                authors=[],
                content_hash=self._compute_hash(content),
                crawl_timestamp=datetime.now()
            ))
        return results

    def clear_cache(self) -> None:
        """Clear seen hashes and URLs."""
        self._seen_hashes.clear()
        self._seen_urls.clear()
        logger.info("Cleared crawler cache")

    def get_stats(self) -> Dict[str, Any]:
        """Get crawler statistics."""
        return {
            'sources_configured': len(self.sources),
            'sources_enabled': sum(1 for s in self.sources if s.enabled),
            'unique_hashes_seen': len(self._seen_hashes),
            'urls_cached': len(self._seen_urls)
        }
