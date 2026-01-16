"""
Engine 4: External Idea Harvesting

Breaks evolutionary groupthink by importing ideas from external sources:
- arXiv q-fin papers
- Academic research
- Quantitative blog posts

Think of it as competitive intelligence for alpha generation.
"""

import asyncio
import hashlib
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import re

from .llm_guided import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ExternalIdea:
    """A raw idea harvested from external source."""
    source: str           # arxiv, ssrn, blog, etc.
    url: str
    title: str
    raw_content: str
    published_date: Optional[datetime]
    relevance_score: float
    content_hash: str

    @classmethod
    def create(
        cls,
        source: str,
        url: str,
        title: str,
        content: str,
        published_date: Optional[datetime] = None
    ) -> 'ExternalIdea':
        """Create with computed hash."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return cls(
            source=source,
            url=url,
            title=title,
            raw_content=content,
            published_date=published_date,
            relevance_score=0.0,
            content_hash=content_hash
        )


@dataclass
class ParsedStrategy:
    """Structured strategy extracted from external idea."""
    idea_name: str
    mechanism: str
    entry_logic: str
    exit_logic: str
    timeframe: str
    indicators_used: List[str]
    novelty_score: float
    confidence: float
    source_url: str
    causal_hypothesis: str


class ArxivCrawler:
    """Crawler for arXiv q-fin papers."""

    CATEGORIES = [
        'q-fin.TR',   # Trading and Market Microstructure
        'q-fin.PM',   # Portfolio Management
        'q-fin.ST',   # Statistical Finance
        'q-fin.RM',   # Risk Management
        'q-fin.CP',   # Computational Finance
    ]

    RELEVANCE_KEYWORDS = {
        'high': [
            'trading strategy', 'alpha', 'sharpe ratio', 'backtest',
            'momentum', 'mean reversion', 'market microstructure',
            'order book', 'high frequency', 'crypto', 'bitcoin'
        ],
        'medium': [
            'portfolio', 'risk', 'volatility', 'prediction',
            'machine learning', 'neural network', 'reinforcement learning',
            'time series', 'forecasting'
        ]
    }

    def __init__(self, max_results_per_query: int = 20):
        self.max_results = max_results_per_query
        self.crawled_ids: Set[str] = set()

    async def harvest(
        self,
        query: Optional[str] = None,
        days_back: int = 30
    ) -> List[ExternalIdea]:
        """
        Harvest recent papers from arXiv.

        Args:
            query: Search query (optional, uses default if not provided)
            days_back: How far back to search

        Returns:
            List of ExternalIdea objects
        """
        try:
            import arxiv
        except ImportError:
            logger.error("arxiv package not installed")
            return []

        ideas = []

        # Build query
        if query is None:
            query = "trading OR strategy OR alpha OR backtest"

        # Add category filter
        category_filter = " OR ".join(f"cat:{cat}" for cat in self.CATEGORIES)
        full_query = f"({query}) AND ({category_filter})"

        try:
            search = arxiv.Search(
                query=full_query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )

            for result in search.results():
                # Skip already seen
                if result.entry_id in self.crawled_ids:
                    continue
                self.crawled_ids.add(result.entry_id)

                # Check date
                if result.published:
                    age = datetime.now(result.published.tzinfo) - result.published
                    if age.days > days_back:
                        continue

                # Compute relevance
                relevance = self._compute_relevance(
                    result.title + " " + result.summary
                )

                if relevance > 0.3:
                    idea = ExternalIdea.create(
                        source="arxiv",
                        url=result.entry_id,
                        title=result.title,
                        content=result.summary,
                        published_date=result.published
                    )
                    idea.relevance_score = relevance
                    ideas.append(idea)

        except Exception as e:
            logger.error(f"arXiv harvest failed: {e}")

        return sorted(ideas, key=lambda x: x.relevance_score, reverse=True)

    def _compute_relevance(self, text: str) -> float:
        """Compute relevance score based on keywords."""
        text_lower = text.lower()
        score = 0.0

        for keyword in self.RELEVANCE_KEYWORDS['high']:
            if keyword in text_lower:
                score += 0.15

        for keyword in self.RELEVANCE_KEYWORDS['medium']:
            if keyword in text_lower:
                score += 0.08

        return min(1.0, score)


class ExternalIdeaHarvester:
    """
    Pipeline for harvesting and processing external ideas.

    Pipeline:
    1. Web Scout: Crawl sources (arXiv, etc.)
    2. Idea Extractor: LLM parses to structured hypotheses
    3. Novelty Check: Vector similarity against existing strategies
    4. Strategy Generator: Translate to HIMARI format
    """

    EXTRACTION_PROMPT = """Extract trading strategy information from this research:

Title: {title}

Abstract/Content:
{content}

Return JSON with:
{{
    "idea_name": "short descriptive name",
    "mechanism": "how does this strategy work mechanistically",
    "entry_logic": "when to enter trades",
    "exit_logic": "when to exit trades",
    "timeframe": "expected holding period",
    "indicators_used": ["list", "of", "indicators"],
    "causal_hypothesis": "WHY this works (market inefficiency exploited)",
    "confidence": 0.0-1.0 based on how actionable this is
}}

Only extract if there's actual trading strategy content. Return {{"confidence": 0}} if not applicable."""

    def __init__(
        self,
        llm_client: LLMClient,
        existing_strategy_vectors: Optional[List] = None
    ):
        """
        Args:
            llm_client: LLM client for extraction
            existing_strategy_vectors: Vectors of existing strategies for novelty check
        """
        self.llm = llm_client
        self.arxiv_crawler = ArxivCrawler()
        self.existing_vectors = existing_strategy_vectors or []
        self.harvested_hashes: Set[str] = set()

    async def harvest_all(
        self,
        query: Optional[str] = None
    ) -> List[ExternalIdea]:
        """
        Harvest from all configured sources.

        Returns deduplicated, relevance-sorted ideas.
        """
        all_ideas = []

        # Harvest from arXiv
        arxiv_ideas = await self.arxiv_crawler.harvest(query)
        all_ideas.extend(arxiv_ideas)

        # Deduplicate by hash
        unique_ideas = []
        seen_hashes = set()
        for idea in all_ideas:
            if idea.content_hash not in seen_hashes:
                seen_hashes.add(idea.content_hash)
                unique_ideas.append(idea)

        return unique_ideas

    async def extract_strategy(
        self,
        idea: ExternalIdea
    ) -> Optional[ParsedStrategy]:
        """
        Use LLM to extract structured strategy from raw idea.

        Args:
            idea: Raw external idea

        Returns:
            ParsedStrategy or None if extraction fails
        """
        # Skip if already processed
        if idea.content_hash in self.harvested_hashes:
            return None
        self.harvested_hashes.add(idea.content_hash)

        # Truncate content for LLM
        content = idea.raw_content[:4000]

        prompt = self.EXTRACTION_PROMPT.format(
            title=idea.title,
            content=content
        )

        try:
            response = await self.llm.generate(prompt, max_tokens=1000)
            result = self._parse_json(response)

            if not result or result.get('confidence', 0) < 0.3:
                return None

            # Compute novelty
            novelty = await self._compute_novelty(result)

            return ParsedStrategy(
                idea_name=result.get('idea_name', 'Unknown'),
                mechanism=result.get('mechanism', ''),
                entry_logic=result.get('entry_logic', ''),
                exit_logic=result.get('exit_logic', ''),
                timeframe=result.get('timeframe', 'unknown'),
                indicators_used=result.get('indicators_used', []),
                novelty_score=novelty,
                confidence=result.get('confidence', 0.5),
                source_url=idea.url,
                causal_hypothesis=result.get('causal_hypothesis', '')
            )

        except Exception as e:
            logger.error(f"Strategy extraction failed: {e}")
            return None

    async def _compute_novelty(self, parsed: Dict) -> float:
        """
        Compute novelty score by comparing to existing strategies.

        Higher score = more novel (different from existing).
        """
        if not self.existing_vectors:
            return 0.8  # Default high novelty if no comparison

        # Convert parsed strategy to feature vector for comparison
        # This is simplified - real implementation would embed the strategy
        indicators = set(parsed.get('indicators_used', []))

        # Common indicators reduce novelty
        common_indicators = {'rsi', 'macd', 'sma', 'ema', 'bollinger'}
        common_count = len(indicators & common_indicators)
        novelty = 1.0 - (common_count * 0.15)

        return max(0.2, min(1.0, novelty))

    async def harvest_and_extract(
        self,
        query: Optional[str] = None,
        max_strategies: int = 10
    ) -> List[ParsedStrategy]:
        """
        Full pipeline: harvest ideas and extract strategies.

        Args:
            query: Search query
            max_strategies: Maximum strategies to return

        Returns:
            List of ParsedStrategy objects
        """
        # Harvest ideas
        ideas = await self.harvest_all(query)
        logger.info(f"Harvested {len(ideas)} ideas")

        # Extract strategies in parallel
        tasks = [self.extract_strategy(idea) for idea in ideas[:max_strategies * 2]]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful extractions
        strategies = [
            r for r in results
            if isinstance(r, ParsedStrategy) and r.confidence >= 0.3
        ]

        # Sort by novelty * confidence
        strategies.sort(
            key=lambda s: s.novelty_score * s.confidence,
            reverse=True
        )

        return strategies[:max_strategies]

    def _parse_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from LLM response."""
        import json
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        return None

    def update_existing_strategies(self, vectors: List) -> None:
        """Update the list of existing strategy vectors for novelty comparison."""
        self.existing_vectors = vectors


class BlogCrawler:
    """
    Crawler for quantitative finance blogs.

    Targets high-quality research sources:
    - Paradigm Research
    - Delphi Digital
    - Kaiko Research
    """

    SOURCES = {
        'paradigm': {
            'base_url': 'https://www.paradigm.xyz/writing',
            'selector': 'article'
        },
        'kaiko': {
            'base_url': 'https://www.kaiko.com/research',
            'selector': '.research-content'
        }
    }

    async def harvest(self, source_name: str) -> List[ExternalIdea]:
        """
        Harvest from a specific blog source.

        Note: Actual web crawling would require additional setup.
        This is a placeholder for the interface.
        """
        # In production, this would use aiohttp and BeautifulSoup
        # to crawl the actual websites
        logger.info(f"Blog crawling for {source_name} not yet implemented")
        return []


class IdeaScorer:
    """
    Score and rank harvested ideas for strategy potential.

    Criteria:
    - Relevance to crypto/trading
    - Novelty (not already known patterns)
    - Actionability (can be implemented)
    - Quality of mechanism explanation
    """

    def score(self, idea: ExternalIdea) -> float:
        """
        Compute composite score for an idea.

        Returns:
            Score between 0 and 1
        """
        text = (idea.title + " " + idea.raw_content).lower()

        # Relevance score
        relevance = idea.relevance_score

        # Actionability indicators
        actionability = 0.0
        action_keywords = [
            'backtest', 'strategy', 'trade', 'signal',
            'indicator', 'return', 'profit', 'sharpe'
        ]
        for kw in action_keywords:
            if kw in text:
                actionability += 0.1

        # Recency bonus
        recency = 0.0
        if idea.published_date:
            age_days = (datetime.now(idea.published_date.tzinfo) - idea.published_date).days
            if age_days < 7:
                recency = 0.2
            elif age_days < 30:
                recency = 0.1

        # Combine scores
        score = (
            relevance * 0.4 +
            min(1.0, actionability) * 0.4 +
            recency * 0.2
        )

        return min(1.0, score)

    def filter_and_rank(
        self,
        ideas: List[ExternalIdea],
        min_score: float = 0.4
    ) -> List[ExternalIdea]:
        """
        Filter ideas by minimum score and rank.

        Returns:
            Filtered and sorted list of ideas
        """
        scored = [(idea, self.score(idea)) for idea in ideas]
        filtered = [(idea, score) for idea, score in scored if score >= min_score]
        filtered.sort(key=lambda x: x[1], reverse=True)
        return [idea for idea, _ in filtered]
