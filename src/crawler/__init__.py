"""
Web Crawler for HIMARI Layer 1 Explorer

Harvests trading strategy ideas from external sources:
- arXiv quantitative finance papers
- Financial blogs and research sites
- Trading forums (read-only analysis)

Components:
- LightweightWebCrawler: Rate-limited web fetching
- RelevanceScorer: Filters relevant content
- KnowledgeExtractor: LLM-based extraction
- KnowledgeStore: SQLite persistence
- CrawlerConditioner: Converts knowledge to generation hints
"""

from .crawler import LightweightWebCrawler, CrawlResult, SourceConfig
from .scorer import RelevanceScorer, RelevanceResult
from .extractor import KnowledgeExtractor, ExtractedKnowledge
from .store import KnowledgeStore, StoredKnowledge
from .conditioner import CrawlerConditioner, GenerationHint
from .orchestrator import CrawlerOrchestrator

__all__ = [
    # Crawler
    'LightweightWebCrawler', 'CrawlResult', 'SourceConfig',

    # Scoring
    'RelevanceScorer', 'RelevanceResult',

    # Extraction
    'KnowledgeExtractor', 'ExtractedKnowledge',

    # Storage
    'KnowledgeStore', 'StoredKnowledge',

    # Conditioning
    'CrawlerConditioner', 'GenerationHint',

    # Orchestration
    'CrawlerOrchestrator'
]
