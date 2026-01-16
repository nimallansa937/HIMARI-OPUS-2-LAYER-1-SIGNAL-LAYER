"""
Relevance Scorer

Filters and scores crawled content for relevance to trading
strategy generation. Uses keyword matching and feature mapping.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum

from .crawler import CrawlResult

logger = logging.getLogger(__name__)


class ContentCategory(Enum):
    """Categories of trading-related content."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    ORDER_FLOW = "order_flow"
    FUNDING = "funding"
    SENTIMENT = "sentiment"
    MICROSTRUCTURE = "microstructure"
    RISK_MANAGEMENT = "risk_management"
    PORTFOLIO = "portfolio"
    MACHINE_LEARNING = "machine_learning"
    OTHER = "other"


@dataclass
class RelevanceResult:
    """Result from relevance scoring."""
    crawl_result: CrawlResult
    relevance_score: float  # 0.0 to 1.0
    category: ContentCategory
    matched_keywords: List[str]
    feature_mapping: Dict[str, float]  # Feature name -> relevance weight
    is_actionable: bool  # Contains implementable strategy idea
    confidence: float
    details: Dict[str, any] = field(default_factory=dict)


class RelevanceScorer:
    """
    Scores crawled content for relevance to strategy generation.

    Uses keyword-based scoring with category detection and
    feature mapping for the 60-dim feature vector.
    """

    # Category keyword mappings
    CATEGORY_KEYWORDS = {
        ContentCategory.MOMENTUM: [
            'momentum', 'trend', 'trend-following', 'breakout',
            'continuation', 'relative strength', 'moving average',
            'macd', 'directional', 'trend continuation'
        ],
        ContentCategory.MEAN_REVERSION: [
            'mean reversion', 'reversal', 'oversold', 'overbought',
            'rsi', 'bollinger', 'contrarian', 'value', 'dip buying',
            'revert', 'oscillator'
        ],
        ContentCategory.VOLATILITY: [
            'volatility', 'vol', 'vix', 'atr', 'implied volatility',
            'realized volatility', 'garch', 'stochastic volatility',
            'volatility regime', 'vol clustering'
        ],
        ContentCategory.ORDER_FLOW: [
            'order flow', 'order book', 'market depth', 'liquidity',
            'bid-ask', 'spread', 'imbalance', 'toxic flow',
            'price impact', 'execution'
        ],
        ContentCategory.FUNDING: [
            'funding rate', 'funding', 'perpetual', 'basis',
            'contango', 'backwardation', 'carry', 'roll yield'
        ],
        ContentCategory.SENTIMENT: [
            'sentiment', 'fear', 'greed', 'social', 'twitter',
            'news', 'nlp', 'text mining', 'alternative data'
        ],
        ContentCategory.MICROSTRUCTURE: [
            'microstructure', 'high frequency', 'hft', 'latency',
            'market making', 'tick data', 'quote', 'order arrival'
        ],
        ContentCategory.RISK_MANAGEMENT: [
            'risk', 'drawdown', 'var', 'cvar', 'position sizing',
            'stop loss', 'risk parity', 'hedging', 'tail risk'
        ],
        ContentCategory.PORTFOLIO: [
            'portfolio', 'allocation', 'diversification', 'correlation',
            'markowitz', 'optimization', 'rebalancing', 'weight'
        ],
        ContentCategory.MACHINE_LEARNING: [
            'machine learning', 'deep learning', 'neural network',
            'reinforcement learning', 'random forest', 'xgboost',
            'lstm', 'transformer', 'prediction'
        ]
    }

    # Mechanism keywords (indicate actionable strategies)
    MECHANISM_KEYWORDS = [
        'strategy', 'trading', 'backtest', 'returns', 'sharpe',
        'alpha', 'signal', 'entry', 'exit', 'rule', 'exploit',
        'arbitrage', 'profit', 'edge', 'approach', 'method'
    ]

    # Noise keywords (indicate less useful content)
    NOISE_KEYWORDS = [
        'survey', 'review', 'overview', 'introduction',
        'tutorial', 'course', 'lecture', 'textbook',
        'disclaimer', 'advertisement', 'sponsored'
    ]

    # Feature category mappings
    FEATURE_CATEGORIES = {
        'price': ['price', 'close', 'open', 'high', 'low', 'ohlc'],
        'volume': ['volume', 'turnover', 'liquidity', 'trade count'],
        'technical': ['rsi', 'macd', 'bollinger', 'sma', 'ema', 'atr'],
        'order_flow': ['order flow', 'imbalance', 'depth', 'bid', 'ask'],
        'funding': ['funding', 'basis', 'perpetual', 'carry'],
        'sentiment': ['sentiment', 'fear', 'greed', 'social'],
        'regime': ['regime', 'state', 'volatility regime', 'trend state']
    }

    def __init__(
        self,
        min_relevance: float = 0.3,
        actionable_threshold: float = 0.5
    ):
        """
        Initialize relevance scorer.

        Args:
            min_relevance: Minimum relevance score to keep
            actionable_threshold: Score threshold for "actionable"
        """
        self.min_relevance = min_relevance
        self.actionable_threshold = actionable_threshold

        # Compile keyword patterns
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficient matching."""
        self._category_patterns = {}
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            pattern = '|'.join(r'\b' + re.escape(kw) + r'\b' for kw in keywords)
            self._category_patterns[category] = re.compile(pattern, re.IGNORECASE)

        self._mechanism_pattern = re.compile(
            '|'.join(r'\b' + re.escape(kw) + r'\b' for kw in self.MECHANISM_KEYWORDS),
            re.IGNORECASE
        )

        self._noise_pattern = re.compile(
            '|'.join(r'\b' + re.escape(kw) + r'\b' for kw in self.NOISE_KEYWORDS),
            re.IGNORECASE
        )

    def score(self, result: CrawlResult) -> RelevanceResult:
        """
        Score a crawl result for relevance.

        Args:
            result: Crawl result to score

        Returns:
            RelevanceResult with scores and mappings
        """
        text = f"{result.title} {result.content}".lower()

        # Category detection
        category_scores = self._score_categories(text)
        primary_category = max(category_scores.items(), key=lambda x: x[1])
        category = primary_category[0] if primary_category[1] > 0 else ContentCategory.OTHER

        # Mechanism score (actionability)
        mechanism_matches = self._mechanism_pattern.findall(text)
        mechanism_score = min(len(mechanism_matches) / 5.0, 1.0)

        # Noise penalty
        noise_matches = self._noise_pattern.findall(text)
        noise_penalty = min(len(noise_matches) / 3.0, 0.5)

        # Collect matched keywords
        matched_keywords = []
        for keywords in self.CATEGORY_KEYWORDS.get(category, []):
            if keywords.lower() in text:
                matched_keywords.append(keywords)
        matched_keywords.extend(mechanism_matches[:5])

        # Feature mapping
        feature_mapping = self._map_features(text)

        # Overall relevance score
        base_score = primary_category[1] if primary_category[1] > 0 else 0.1
        relevance_score = (
            0.4 * base_score +
            0.3 * mechanism_score +
            0.2 * (len(feature_mapping) / 7.0) +
            0.1 * (1.0 - noise_penalty)
        )
        relevance_score = max(0, min(1, relevance_score))

        # Confidence based on content length and keyword density
        keyword_density = len(matched_keywords) / max(len(text.split()) / 100, 1)
        confidence = min(0.5 + 0.5 * keyword_density, 0.95)

        return RelevanceResult(
            crawl_result=result,
            relevance_score=relevance_score,
            category=category,
            matched_keywords=matched_keywords[:10],
            feature_mapping=feature_mapping,
            is_actionable=mechanism_score >= self.actionable_threshold,
            confidence=confidence,
            details={
                'category_scores': {k.value: v for k, v in category_scores.items()},
                'mechanism_score': mechanism_score,
                'noise_penalty': noise_penalty
            }
        )

    def _score_categories(self, text: str) -> Dict[ContentCategory, float]:
        """Score text against each category."""
        scores = {}
        for category, pattern in self._category_patterns.items():
            matches = pattern.findall(text)
            # Normalize by text length
            score = len(matches) / max(len(text.split()) / 100, 1)
            scores[category] = min(score, 1.0)
        return scores

    def _map_features(self, text: str) -> Dict[str, float]:
        """Map content to relevant feature categories."""
        mapping = {}
        for feature_cat, keywords in self.FEATURE_CATEGORIES.items():
            relevance = 0
            for keyword in keywords:
                if keyword.lower() in text:
                    relevance += 0.3
            mapping[feature_cat] = min(relevance, 1.0)

        # Filter zero relevance
        return {k: v for k, v in mapping.items() if v > 0}

    def score_batch(
        self,
        results: List[CrawlResult]
    ) -> List[RelevanceResult]:
        """Score multiple crawl results."""
        return [self.score(r) for r in results]

    def filter_relevant(
        self,
        results: List[CrawlResult],
        min_score: Optional[float] = None
    ) -> List[RelevanceResult]:
        """
        Filter and return only relevant results.

        Args:
            results: Crawl results to filter
            min_score: Minimum relevance score (uses default if None)

        Returns:
            List of relevant results, sorted by score
        """
        min_score = min_score or self.min_relevance

        scored = self.score_batch(results)
        relevant = [r for r in scored if r.relevance_score >= min_score]

        # Sort by relevance descending
        relevant.sort(key=lambda x: x.relevance_score, reverse=True)

        return relevant

    def get_category_distribution(
        self,
        results: List[RelevanceResult]
    ) -> Dict[str, int]:
        """Get distribution of categories in results."""
        distribution = {}
        for result in results:
            cat = result.category.value
            distribution[cat] = distribution.get(cat, 0) + 1
        return distribution
