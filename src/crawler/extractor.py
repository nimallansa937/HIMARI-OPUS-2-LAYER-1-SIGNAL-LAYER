"""
Knowledge Extractor

Uses LLM to extract structured trading knowledge from
relevant crawled content. Outputs actionable strategy components.
"""

import logging
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

from .scorer import RelevanceResult, ContentCategory

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    """Direction of trading signal."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"
    BOTH = "both"  # Signal works for both directions


@dataclass
class ExtractedSignal:
    """A trading signal extracted from content."""
    name: str
    description: str
    direction: SignalDirection
    features_used: List[str]
    threshold_type: str  # "above", "below", "cross"
    suggested_threshold: Optional[float] = None


@dataclass
class ExtractedCondition:
    """A trading condition (entry/exit rule)."""
    condition_type: str  # "entry", "exit", "filter"
    description: str
    signals: List[ExtractedSignal]
    logical_operator: str = "AND"  # "AND", "OR"


@dataclass
class ExtractedRiskParams:
    """Risk management parameters extracted."""
    stop_loss_type: str  # "atr", "pct", "trailing"
    stop_loss_value: Optional[float] = None
    take_profit_type: Optional[str] = None
    take_profit_value: Optional[float] = None
    position_sizing: Optional[str] = None  # "fixed", "vol_adjusted", "kelly"


@dataclass
class ExtractedKnowledge:
    """Complete extracted knowledge from a document."""
    source_url: str
    source_title: str
    category: ContentCategory
    mechanism: str  # Why the strategy should work
    causal_hypothesis: str  # Causal explanation
    signals: List[ExtractedSignal]
    entry_conditions: List[ExtractedCondition]
    exit_conditions: List[ExtractedCondition]
    risk_params: Optional[ExtractedRiskParams]
    market_regime: str  # "trending", "ranging", "volatile", "any"
    timeframe: str  # "intraday", "daily", "weekly"
    expected_performance: Dict[str, Any]
    extraction_confidence: float
    raw_text: str
    extracted_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'source_url': self.source_url,
            'source_title': self.source_title,
            'category': self.category.value,
            'mechanism': self.mechanism,
            'causal_hypothesis': self.causal_hypothesis,
            'signals': [
                {
                    'name': s.name,
                    'description': s.description,
                    'direction': s.direction.value,
                    'features_used': s.features_used,
                    'threshold_type': s.threshold_type,
                    'suggested_threshold': s.suggested_threshold
                }
                for s in self.signals
            ],
            'market_regime': self.market_regime,
            'timeframe': self.timeframe,
            'expected_performance': self.expected_performance,
            'extraction_confidence': self.extraction_confidence,
            'extracted_at': self.extracted_at.isoformat()
        }


# Extraction prompts
EXTRACTION_PROMPT = """Analyze this trading strategy content and extract structured knowledge.

Content:
{content}

Extract the following in JSON format:
{{
    "mechanism": "Why this strategy should work (1-2 sentences)",
    "causal_hypothesis": "The causal relationship being exploited",
    "signals": [
        {{
            "name": "Signal name",
            "description": "What the signal measures",
            "direction": "long/short/both",
            "features_used": ["feature1", "feature2"],
            "threshold_type": "above/below/cross",
            "suggested_threshold": 0.0
        }}
    ],
    "entry_conditions": [
        {{
            "description": "When to enter",
            "signals_used": ["signal_name"],
            "logical_operator": "AND/OR"
        }}
    ],
    "exit_conditions": [
        {{
            "description": "When to exit",
            "signals_used": ["signal_name"],
            "logical_operator": "AND/OR"
        }}
    ],
    "risk_params": {{
        "stop_loss_type": "atr/pct/trailing",
        "stop_loss_value": 2.0,
        "take_profit_type": "atr/pct/none",
        "take_profit_value": 3.0,
        "position_sizing": "fixed/vol_adjusted/kelly"
    }},
    "market_regime": "trending/ranging/volatile/any",
    "timeframe": "intraday/daily/weekly",
    "expected_sharpe": 1.5,
    "expected_drawdown": 0.15,
    "confidence": 0.8
}}

If information is not available, use reasonable defaults or null.
Return only valid JSON, no explanation."""


class KnowledgeExtractor:
    """
    Extracts structured trading knowledge using LLM.

    Converts unstructured text into actionable strategy
    components that can inform the generation engines.
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        use_local: bool = True
    ):
        """
        Initialize knowledge extractor.

        Args:
            llm_client: LLM client for extraction
            use_local: Use local Ollama if no client provided
        """
        self.llm_client = llm_client
        self.use_local = use_local

        # Feature name normalization
        self._feature_mapping = {
            'rsi': 'rsi_14',
            'macd': 'macd_histogram',
            'bollinger': 'bb_position',
            'volume': 'volume_ratio',
            'momentum': 'price_momentum',
            'volatility': 'realized_volatility',
            'funding': 'funding_rate',
            'open interest': 'open_interest_change',
            'sentiment': 'sentiment_score',
            'trend': 'trend_strength'
        }

    async def extract(
        self,
        relevance_result: RelevanceResult
    ) -> Optional[ExtractedKnowledge]:
        """
        Extract structured knowledge from a relevance result.

        Args:
            relevance_result: Scored relevance result

        Returns:
            ExtractedKnowledge or None if extraction fails
        """
        content = relevance_result.crawl_result.content
        title = relevance_result.crawl_result.title

        # Use LLM for extraction
        if self.llm_client:
            extracted_json = await self._extract_with_llm(content)
        else:
            # Fallback to rule-based extraction
            extracted_json = self._extract_rule_based(content, relevance_result)

        if not extracted_json:
            return None

        try:
            return self._parse_extraction(
                extracted_json,
                relevance_result.crawl_result.url,
                title,
                relevance_result.category,
                content
            )
        except Exception as e:
            logger.error(f"Failed to parse extraction: {e}")
            return None

    async def _extract_with_llm(self, content: str) -> Optional[Dict]:
        """Extract using LLM."""
        prompt = EXTRACTION_PROMPT.format(content=content[:3000])

        try:
            response = await self.llm_client.generate(prompt)
            # Parse JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
            return None
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return None

    def _extract_rule_based(
        self,
        content: str,
        relevance: RelevanceResult
    ) -> Dict:
        """Fallback rule-based extraction."""
        content_lower = content.lower()

        # Detect mechanism from matched keywords
        mechanism = self._infer_mechanism(content_lower, relevance.category)

        # Extract signals from keywords
        signals = self._extract_signals(content_lower, relevance)

        # Infer market regime
        regime = self._infer_regime(content_lower)

        # Infer timeframe
        timeframe = self._infer_timeframe(content_lower)

        return {
            'mechanism': mechanism,
            'causal_hypothesis': f"{relevance.category.value} effect on returns",
            'signals': signals,
            'entry_conditions': [{
                'description': f"Enter when {signals[0]['name'] if signals else 'signal'} triggers",
                'signals_used': [s['name'] for s in signals[:2]],
                'logical_operator': 'AND'
            }] if signals else [],
            'exit_conditions': [{
                'description': "Exit on reversal or target",
                'signals_used': [],
                'logical_operator': 'OR'
            }],
            'risk_params': {
                'stop_loss_type': 'atr',
                'stop_loss_value': 2.0,
                'take_profit_type': 'atr',
                'take_profit_value': 3.0,
                'position_sizing': 'vol_adjusted'
            },
            'market_regime': regime,
            'timeframe': timeframe,
            'expected_sharpe': 1.0,
            'expected_drawdown': 0.15,
            'confidence': relevance.confidence * 0.7  # Reduce for rule-based
        }

    def _infer_mechanism(self, content: str, category: ContentCategory) -> str:
        """Infer trading mechanism from content."""
        mechanisms = {
            ContentCategory.MOMENTUM: "Trend continuation exploiting behavioral persistence",
            ContentCategory.MEAN_REVERSION: "Price reversion to fair value",
            ContentCategory.VOLATILITY: "Volatility regime exploitation",
            ContentCategory.ORDER_FLOW: "Information from order book dynamics",
            ContentCategory.FUNDING: "Funding rate mean reversion",
            ContentCategory.SENTIMENT: "Behavioral sentiment effects",
            ContentCategory.MICROSTRUCTURE: "Market microstructure patterns"
        }
        return mechanisms.get(category, "Statistical pattern exploitation")

    def _extract_signals(
        self,
        content: str,
        relevance: RelevanceResult
    ) -> List[Dict]:
        """Extract trading signals from content."""
        signals = []

        # Map features to signals
        for feature, weight in relevance.feature_mapping.items():
            if weight > 0.3:
                signal = {
                    'name': f"{feature}_signal",
                    'description': f"Signal based on {feature}",
                    'direction': self._infer_direction(content, feature),
                    'features_used': [self._normalize_feature(feature)],
                    'threshold_type': 'above' if 'buy' in content or 'long' in content else 'below',
                    'suggested_threshold': None
                }
                signals.append(signal)

        # Add category-specific signals
        if relevance.category == ContentCategory.MOMENTUM:
            signals.append({
                'name': 'momentum_signal',
                'description': 'Price momentum indicator',
                'direction': 'long',
                'features_used': ['price_momentum', 'trend_strength'],
                'threshold_type': 'above',
                'suggested_threshold': 0.5
            })
        elif relevance.category == ContentCategory.MEAN_REVERSION:
            signals.append({
                'name': 'reversion_signal',
                'description': 'Mean reversion indicator',
                'direction': 'both',
                'features_used': ['rsi_14', 'bb_position'],
                'threshold_type': 'below',
                'suggested_threshold': 30
            })

        return signals[:5]  # Limit signals

    def _infer_direction(self, content: str, feature: str) -> str:
        """Infer signal direction from context."""
        long_words = ['buy', 'long', 'bullish', 'upward']
        short_words = ['sell', 'short', 'bearish', 'downward']

        long_count = sum(1 for w in long_words if w in content)
        short_count = sum(1 for w in short_words if w in content)

        if long_count > short_count:
            return 'long'
        elif short_count > long_count:
            return 'short'
        return 'both'

    def _infer_regime(self, content: str) -> str:
        """Infer target market regime."""
        if any(w in content for w in ['trend', 'momentum', 'breakout']):
            return 'trending'
        elif any(w in content for w in ['range', 'sideways', 'consolidation']):
            return 'ranging'
        elif any(w in content for w in ['volatile', 'volatility spike']):
            return 'volatile'
        return 'any'

    def _infer_timeframe(self, content: str) -> str:
        """Infer trading timeframe."""
        if any(w in content for w in ['minute', 'intraday', 'scalp', 'hft']):
            return 'intraday'
        elif any(w in content for w in ['weekly', 'monthly', 'swing']):
            return 'weekly'
        return 'daily'

    def _normalize_feature(self, feature: str) -> str:
        """Normalize feature name to match feature schema."""
        return self._feature_mapping.get(feature.lower(), feature)

    def _parse_extraction(
        self,
        data: Dict,
        url: str,
        title: str,
        category: ContentCategory,
        raw_text: str
    ) -> ExtractedKnowledge:
        """Parse extracted JSON into ExtractedKnowledge."""
        # Parse signals
        signals = []
        for s in data.get('signals', []):
            signals.append(ExtractedSignal(
                name=s.get('name', 'unknown'),
                description=s.get('description', ''),
                direction=SignalDirection(s.get('direction', 'both')),
                features_used=[self._normalize_feature(f) for f in s.get('features_used', [])],
                threshold_type=s.get('threshold_type', 'above'),
                suggested_threshold=s.get('suggested_threshold')
            ))

        # Parse conditions
        entry_conditions = []
        for c in data.get('entry_conditions', []):
            entry_signals = [s for s in signals if s.name in c.get('signals_used', [])]
            entry_conditions.append(ExtractedCondition(
                condition_type='entry',
                description=c.get('description', ''),
                signals=entry_signals,
                logical_operator=c.get('logical_operator', 'AND')
            ))

        exit_conditions = []
        for c in data.get('exit_conditions', []):
            exit_signals = [s for s in signals if s.name in c.get('signals_used', [])]
            exit_conditions.append(ExtractedCondition(
                condition_type='exit',
                description=c.get('description', ''),
                signals=exit_signals,
                logical_operator=c.get('logical_operator', 'OR')
            ))

        # Parse risk params
        risk_data = data.get('risk_params', {})
        risk_params = ExtractedRiskParams(
            stop_loss_type=risk_data.get('stop_loss_type', 'atr'),
            stop_loss_value=risk_data.get('stop_loss_value'),
            take_profit_type=risk_data.get('take_profit_type'),
            take_profit_value=risk_data.get('take_profit_value'),
            position_sizing=risk_data.get('position_sizing')
        ) if risk_data else None

        return ExtractedKnowledge(
            source_url=url,
            source_title=title,
            category=category,
            mechanism=data.get('mechanism', ''),
            causal_hypothesis=data.get('causal_hypothesis', ''),
            signals=signals,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            risk_params=risk_params,
            market_regime=data.get('market_regime', 'any'),
            timeframe=data.get('timeframe', 'daily'),
            expected_performance={
                'sharpe': data.get('expected_sharpe', 1.0),
                'max_drawdown': data.get('expected_drawdown', 0.2)
            },
            extraction_confidence=data.get('confidence', 0.5),
            raw_text=raw_text[:1000]  # Keep truncated raw text
        )

    async def extract_batch(
        self,
        results: List[RelevanceResult]
    ) -> List[ExtractedKnowledge]:
        """Extract knowledge from multiple results."""
        extracted = []
        for result in results:
            knowledge = await self.extract(result)
            if knowledge:
                extracted.append(knowledge)
        return extracted
