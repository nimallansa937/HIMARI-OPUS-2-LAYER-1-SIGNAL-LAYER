"""
Crawler Orchestrator

Coordinates the full crawl-extract-store cycle and provides
a unified interface for the knowledge harvesting pipeline.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from .crawler import LightweightWebCrawler, CrawlResult, SourceConfig
from .scorer import RelevanceScorer, RelevanceResult
from .extractor import KnowledgeExtractor, ExtractedKnowledge
from .store import KnowledgeStore, StoredKnowledge
from .conditioner import CrawlerConditioner, GenerationHint

logger = logging.getLogger(__name__)


@dataclass
class CrawlCycleResult:
    """Result from a complete crawl cycle."""
    pages_crawled: int
    pages_relevant: int
    knowledge_extracted: int
    knowledge_stored: int
    cycle_duration_seconds: float
    errors: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CrawlerOrchestratorConfig:
    """Configuration for crawler orchestrator."""
    # Crawling
    enabled_sources: List[str] = field(default_factory=lambda: ['arxiv_qfin'])
    max_pages_per_cycle: int = 50
    crawl_query: Optional[str] = None

    # Filtering
    min_relevance_score: float = 0.3
    min_extraction_confidence: float = 0.5

    # Storage
    db_path: str = "data/knowledge.db"
    cleanup_days: int = 180

    # Scheduling
    cycle_interval_hours: int = 6
    max_retries: int = 3


class CrawlerOrchestrator:
    """
    Orchestrates the knowledge harvesting pipeline.

    Pipeline stages:
    1. Crawl: Fetch content from configured sources
    2. Score: Filter by relevance
    3. Extract: Use LLM to extract structured knowledge
    4. Store: Persist to SQLite database
    5. Condition: Generate hints for strategy engines

    Provides a unified interface for running cycles and
    querying harvested knowledge.
    """

    def __init__(
        self,
        config: Optional[CrawlerOrchestratorConfig] = None,
        llm_client: Optional[Any] = None
    ):
        """
        Initialize orchestrator.

        Args:
            config: Orchestrator configuration
            llm_client: Optional LLM client for extraction
        """
        self.config = config or CrawlerOrchestratorConfig()

        # Initialize components
        self.crawler = LightweightWebCrawler()
        self.scorer = RelevanceScorer(min_relevance=self.config.min_relevance_score)
        self.extractor = KnowledgeExtractor(llm_client=llm_client)
        self.store = KnowledgeStore(db_path=self.config.db_path)
        self.conditioner = CrawlerConditioner(
            store=self.store,
            min_confidence=self.config.min_extraction_confidence
        )

        # Tracking
        self.last_cycle_time: Optional[datetime] = None
        self.cycle_history: List[CrawlCycleResult] = []
        self._running = False

    async def run_cycle(
        self,
        query: Optional[str] = None
    ) -> CrawlCycleResult:
        """
        Run a complete crawl-extract-store cycle.

        Args:
            query: Optional search query override

        Returns:
            CrawlCycleResult with statistics
        """
        import time
        start_time = time.time()
        errors = []

        logger.info("Starting crawler cycle")

        # Stage 1: Crawl
        try:
            crawl_results = await self.crawler.crawl_all(
                query=query or self.config.crawl_query,
                max_results_per_source=self.config.max_pages_per_cycle
            )
            pages_crawled = len(crawl_results)
            logger.info(f"Crawled {pages_crawled} pages")
        except Exception as e:
            logger.error(f"Crawl stage failed: {e}")
            errors.append(f"Crawl: {str(e)}")
            crawl_results = []
            pages_crawled = 0

        # Stage 2: Score and filter
        try:
            relevant_results = self.scorer.filter_relevant(crawl_results)
            pages_relevant = len(relevant_results)
            logger.info(f"Filtered to {pages_relevant} relevant pages")
        except Exception as e:
            logger.error(f"Scoring stage failed: {e}")
            errors.append(f"Score: {str(e)}")
            relevant_results = []
            pages_relevant = 0

        # Stage 3: Extract knowledge
        extracted = []
        for result in relevant_results:
            try:
                # Skip if already in store
                if self.store.check_exists(result.crawl_result.url):
                    continue

                knowledge = await self.extractor.extract(result)
                if knowledge and knowledge.extraction_confidence >= self.config.min_extraction_confidence:
                    extracted.append(knowledge)
            except Exception as e:
                logger.warning(f"Extraction failed for {result.crawl_result.url}: {e}")
                errors.append(f"Extract: {str(e)[:50]}")

        logger.info(f"Extracted {len(extracted)} knowledge items")

        # Stage 4: Store
        try:
            stored = self.store.store_batch(extracted)
            logger.info(f"Stored {stored} knowledge items")
        except Exception as e:
            logger.error(f"Storage stage failed: {e}")
            errors.append(f"Store: {str(e)}")
            stored = 0

        # Record cycle
        duration = time.time() - start_time
        self.last_cycle_time = datetime.now()

        result = CrawlCycleResult(
            pages_crawled=pages_crawled,
            pages_relevant=pages_relevant,
            knowledge_extracted=len(extracted),
            knowledge_stored=stored,
            cycle_duration_seconds=duration,
            errors=errors
        )

        self.cycle_history.append(result)
        logger.info(f"Crawler cycle complete in {duration:.1f}s")

        return result

    async def run_continuous(
        self,
        max_cycles: Optional[int] = None
    ) -> None:
        """
        Run crawler continuously at configured interval.

        Args:
            max_cycles: Maximum cycles to run (None = infinite)
        """
        self._running = True
        cycles_run = 0

        logger.info(f"Starting continuous crawling (interval: {self.config.cycle_interval_hours}h)")

        while self._running:
            if max_cycles and cycles_run >= max_cycles:
                break

            try:
                await self.run_cycle()
                cycles_run += 1
            except Exception as e:
                logger.error(f"Cycle failed: {e}")

            # Wait for next cycle
            if self._running:
                await asyncio.sleep(self.config.cycle_interval_hours * 3600)

        logger.info(f"Stopped after {cycles_run} cycles")

    def stop(self) -> None:
        """Stop continuous crawling."""
        self._running = False

    def get_hints_for_generation(
        self,
        target_regime: Optional[str] = None,
        target_category: Optional[str] = None
    ) -> GenerationHint:
        """
        Get generation hints for strategy engines.

        Args:
            target_regime: Focus on specific regime
            target_category: Focus on specific category

        Returns:
            GenerationHint for conditioning
        """
        from .scorer import ContentCategory

        category = None
        if target_category:
            try:
                category = ContentCategory(target_category)
            except ValueError:
                pass

        return self.conditioner.generate_hints(
            target_regime=target_regime,
            target_category=category
        )

    def get_conditioning_for_engine(
        self,
        engine_name: str,
        target_regime: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get engine-specific conditioning."""
        return self.conditioner.get_conditioning_for_engine(
            engine_name, target_regime
        )

    def query_knowledge(
        self,
        features: Optional[List[str]] = None,
        category: Optional[str] = None,
        regime: Optional[str] = None,
        limit: int = 20
    ) -> List[StoredKnowledge]:
        """
        Query stored knowledge.

        Args:
            features: Filter by feature relevance
            category: Filter by category
            regime: Filter by market regime
            limit: Maximum results

        Returns:
            List of matching knowledge
        """
        from .scorer import ContentCategory

        if features:
            return self.store.query_by_features(features, limit=limit)
        elif category:
            try:
                cat = ContentCategory(category)
                return self.store.query_by_category(cat, limit=limit)
            except ValueError:
                return []
        elif regime:
            return self.store.query_by_regime(regime, limit=limit)
        else:
            return self.store.query_recent(limit=limit)

    def cleanup(self) -> int:
        """Clean up old knowledge."""
        return self.store.cleanup_old(self.config.cleanup_days)

    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        store_stats = self.store.get_statistics()
        crawler_stats = self.crawler.get_stats()

        return {
            'store': store_stats,
            'crawler': crawler_stats,
            'cycles_run': len(self.cycle_history),
            'last_cycle': self.last_cycle_time.isoformat() if self.last_cycle_time else None,
            'average_cycle_duration': (
                sum(c.cycle_duration_seconds for c in self.cycle_history) /
                len(self.cycle_history)
                if self.cycle_history else 0
            ),
            'total_pages_crawled': sum(c.pages_crawled for c in self.cycle_history),
            'total_knowledge_stored': sum(c.knowledge_stored for c in self.cycle_history)
        }


async def main():
    """Main entry point for standalone crawler."""
    import argparse

    parser = argparse.ArgumentParser(description='HIMARI Knowledge Crawler')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--once', action='store_true', help='Run single cycle')
    parser.add_argument('--query', type=str, help='Search query')
    args = parser.parse_args()

    # Load config if provided
    config = CrawlerOrchestratorConfig()
    if args.config:
        # Would load from YAML
        pass

    orchestrator = CrawlerOrchestrator(config)

    if args.once:
        result = await orchestrator.run_cycle(query=args.query)
        print(f"Cycle complete: {result.knowledge_stored} items stored")
    else:
        await orchestrator.run_continuous()


if __name__ == "__main__":
    asyncio.run(main())
