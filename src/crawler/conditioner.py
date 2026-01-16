"""
Crawler Conditioner

Converts extracted knowledge into conditioning signals for
the generation engines. Creates feature weightings and
parameter hints based on crawled research.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np

from .store import KnowledgeStore, StoredKnowledge
from .scorer import ContentCategory

logger = logging.getLogger(__name__)


@dataclass
class FeatureWeight:
    """Weight for a feature based on external knowledge."""
    feature_name: str
    weight: float  # 0.0 to 2.0, where 1.0 is neutral
    source_count: int  # How many sources mention it
    confidence: float
    direction: str  # "positive", "negative", "neutral"


@dataclass
class ParameterHint:
    """Suggested parameter range from external knowledge."""
    parameter_name: str
    suggested_min: float
    suggested_max: float
    typical_value: float
    confidence: float
    sources: List[str]


@dataclass
class StrategyHint:
    """Hint for strategy generation."""
    category: ContentCategory
    mechanism: str
    entry_signal: str
    exit_signal: str
    regime: str
    timeframe: str
    expected_sharpe: float
    source_url: str
    confidence: float


@dataclass
class GenerationHint:
    """
    Complete generation hint from crawled knowledge.

    Used to condition the generation engines toward
    externally-validated strategy types.
    """
    feature_weights: Dict[str, FeatureWeight]
    parameter_hints: List[ParameterHint]
    strategy_hints: List[StrategyHint]
    regime_focus: Optional[str]
    category_weights: Dict[str, float]
    generated_at: datetime = field(default_factory=datetime.now)

    def get_feature_weight_vector(self, feature_names: List[str]) -> np.ndarray:
        """Get weight vector for feature list."""
        weights = []
        for name in feature_names:
            if name in self.feature_weights:
                weights.append(self.feature_weights[name].weight)
            else:
                weights.append(1.0)  # Neutral weight
        return np.array(weights)

    def to_conditioning_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for engine conditioning."""
        return {
            'feature_weights': {
                k: v.weight for k, v in self.feature_weights.items()
            },
            'regime_focus': self.regime_focus,
            'category_weights': self.category_weights,
            'top_mechanisms': [h.mechanism for h in self.strategy_hints[:3]],
            'parameter_hints': {
                h.parameter_name: h.typical_value
                for h in self.parameter_hints
            }
        }


class CrawlerConditioner:
    """
    Converts crawled knowledge into generation hints.

    Analyzes stored knowledge to produce:
    1. Feature weights (which features to emphasize)
    2. Parameter hints (reasonable parameter ranges)
    3. Strategy hints (validated strategy templates)
    """

    # Feature groups for aggregation
    FEATURE_GROUPS = {
        'price': ['close', 'high', 'low', 'vwap', 'price_zscore'],
        'volume': ['volume', 'volume_ratio', 'volume_zscore'],
        'momentum': ['price_momentum', 'rsi_14', 'macd_histogram'],
        'volatility': ['atr_14', 'realized_volatility', 'bb_position'],
        'order_flow': ['order_imbalance', 'buy_pressure', 'large_trade_ratio'],
        'funding': ['funding_rate', 'funding_zscore', 'open_interest_change'],
        'sentiment': ['sentiment_score', 'social_momentum']
    }

    def __init__(
        self,
        store: KnowledgeStore,
        decay_days: int = 30,
        min_confidence: float = 0.5
    ):
        """
        Initialize conditioner.

        Args:
            store: Knowledge store to read from
            decay_days: Days for knowledge decay (older = less weight)
            min_confidence: Minimum extraction confidence
        """
        self.store = store
        self.decay_days = decay_days
        self.min_confidence = min_confidence

    def generate_hints(
        self,
        target_regime: Optional[str] = None,
        target_category: Optional[ContentCategory] = None
    ) -> GenerationHint:
        """
        Generate conditioning hints from stored knowledge.

        Args:
            target_regime: Focus on specific market regime
            target_category: Focus on specific strategy category

        Returns:
            GenerationHint for conditioning engines
        """
        # Query relevant knowledge
        if target_regime:
            knowledge = self.store.query_by_regime(target_regime, self.min_confidence)
        elif target_category:
            knowledge = self.store.query_by_category(target_category, self.min_confidence)
        else:
            knowledge = self.store.query_recent(self.decay_days, self.min_confidence)

        # Apply decay weighting
        weighted_knowledge = self._apply_decay(knowledge)

        # Generate components
        feature_weights = self._compute_feature_weights(weighted_knowledge)
        parameter_hints = self._extract_parameter_hints(weighted_knowledge)
        strategy_hints = self._create_strategy_hints(weighted_knowledge)
        category_weights = self._compute_category_weights(weighted_knowledge)

        return GenerationHint(
            feature_weights=feature_weights,
            parameter_hints=parameter_hints,
            strategy_hints=strategy_hints,
            regime_focus=target_regime,
            category_weights=category_weights
        )

    def _apply_decay(
        self,
        knowledge: List[StoredKnowledge]
    ) -> List[Tuple[StoredKnowledge, float]]:
        """Apply time-based decay to knowledge."""
        now = datetime.now()
        weighted = []

        for k in knowledge:
            age_days = (now - k.created_at).days
            decay = np.exp(-age_days / self.decay_days)
            weight = k.extraction_confidence * decay
            weighted.append((k, weight))

        return weighted

    def _compute_feature_weights(
        self,
        knowledge: List[Tuple[StoredKnowledge, float]]
    ) -> Dict[str, FeatureWeight]:
        """Compute feature weights from knowledge."""
        feature_mentions: Dict[str, List[Tuple[float, str]]] = {}

        for k, weight in knowledge:
            for feature in k.features_used:
                if feature not in feature_mentions:
                    feature_mentions[feature] = []
                feature_mentions[feature].append((weight, k.source_url))

        # Convert to FeatureWeight objects
        weights = {}
        for feature, mentions in feature_mentions.items():
            total_weight = sum(w for w, _ in mentions)
            count = len(mentions)

            # Normalize: more mentions = higher weight
            # Base weight is 1.0, can go up to 2.0 or down to 0.5
            normalized_weight = 1.0 + min(0.5, total_weight / 10)

            weights[feature] = FeatureWeight(
                feature_name=feature,
                weight=normalized_weight,
                source_count=count,
                confidence=total_weight / count if count > 0 else 0,
                direction='positive'  # Default, could be inferred
            )

        return weights

    def _extract_parameter_hints(
        self,
        knowledge: List[Tuple[StoredKnowledge, float]]
    ) -> List[ParameterHint]:
        """Extract parameter hints from knowledge."""
        hints = []

        # Common parameters to extract
        param_patterns = {
            'stop_loss_atr': (1.0, 5.0, 2.0),  # (min, max, typical)
            'take_profit_atr': (1.5, 10.0, 3.0),
            'position_pct': (0.01, 0.20, 0.05),
            'rsi_oversold': (15, 35, 30),
            'rsi_overbought': (65, 85, 70),
            'lookback_period': (5, 50, 20)
        }

        for param, (default_min, default_max, default_typ) in param_patterns.items():
            hints.append(ParameterHint(
                parameter_name=param,
                suggested_min=default_min,
                suggested_max=default_max,
                typical_value=default_typ,
                confidence=0.7,
                sources=['default']
            ))

        # Adjust based on knowledge
        for k, weight in knowledge:
            if k.expected_sharpe > 1.5:
                # High-performing strategies might use different params
                # This is simplified - real implementation would parse
                # specific parameters from the knowledge
                pass

        return hints

    def _create_strategy_hints(
        self,
        knowledge: List[Tuple[StoredKnowledge, float]]
    ) -> List[StrategyHint]:
        """Create strategy hints from top knowledge."""
        hints = []

        # Sort by weight
        sorted_knowledge = sorted(knowledge, key=lambda x: x[1], reverse=True)

        for k, weight in sorted_knowledge[:10]:
            hints.append(StrategyHint(
                category=ContentCategory(k.category),
                mechanism=k.mechanism,
                entry_signal=self._infer_entry_signal(k),
                exit_signal=self._infer_exit_signal(k),
                regime=k.market_regime,
                timeframe=k.timeframe,
                expected_sharpe=k.expected_sharpe,
                source_url=k.source_url,
                confidence=weight
            ))

        return hints

    def _infer_entry_signal(self, k: StoredKnowledge) -> str:
        """Infer entry signal description."""
        category = k.category

        signals = {
            'momentum': "Enter on momentum confirmation above threshold",
            'mean_reversion': "Enter on oversold/overbought reversal",
            'volatility': "Enter on volatility regime change",
            'order_flow': "Enter on order flow imbalance",
            'funding': "Enter on funding rate extreme",
            'sentiment': "Enter on sentiment divergence",
            'microstructure': "Enter on microstructure signal"
        }

        return signals.get(category, "Enter on signal trigger")

    def _infer_exit_signal(self, k: StoredKnowledge) -> str:
        """Infer exit signal description."""
        return "Exit on target, stop, or reversal signal"

    def _compute_category_weights(
        self,
        knowledge: List[Tuple[StoredKnowledge, float]]
    ) -> Dict[str, float]:
        """Compute category weights based on knowledge quality."""
        category_weights: Dict[str, float] = {}

        for k, weight in knowledge:
            cat = k.category
            if cat not in category_weights:
                category_weights[cat] = 0
            category_weights[cat] += weight * k.expected_sharpe

        # Normalize
        total = sum(category_weights.values()) or 1.0
        return {k: v / total for k, v in category_weights.items()}

    def get_conditioning_for_engine(
        self,
        engine_name: str,
        target_regime: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get engine-specific conditioning.

        Args:
            engine_name: Name of generation engine
            target_regime: Optional target regime

        Returns:
            Engine-specific conditioning dict
        """
        hints = self.generate_hints(target_regime=target_regime)

        if engine_name == "evolutionary":
            return {
                'mutation_bias': hints.to_conditioning_dict()['feature_weights'],
                'fitness_bonus_categories': list(hints.category_weights.keys())[:2]
            }
        elif engine_name == "generative":
            return {
                'condition_vector_weights': hints.get_feature_weight_vector(
                    list(hints.feature_weights.keys())
                ),
                'target_regime': target_regime
            }
        elif engine_name == "llm_guided":
            return {
                'mechanism_hints': [h.mechanism for h in hints.strategy_hints],
                'entry_hints': [h.entry_signal for h in hints.strategy_hints],
                'parameter_hints': {
                    h.parameter_name: h.typical_value
                    for h in hints.parameter_hints
                }
            }
        elif engine_name == "mcts":
            return {
                'feature_priors': hints.to_conditioning_dict()['feature_weights'],
                'regime_focus': target_regime
            }
        else:
            return hints.to_conditioning_dict()

    def update_from_validation(
        self,
        strategy_id: str,
        validation_result: Dict[str, Any]
    ) -> None:
        """
        Update knowledge weights based on validation results.

        When a strategy succeeds/fails validation, adjust weights
        of the knowledge that influenced its generation.

        Args:
            strategy_id: Strategy that was validated
            validation_result: Result from HIFA validation
        """
        # This would track which knowledge influenced which strategy
        # and update weights accordingly. Simplified placeholder.
        passed = validation_result.get('passed', False)
        sharpe = validation_result.get('sharpe', 0)

        if passed and sharpe > 1.5:
            logger.info(f"Strategy {strategy_id[:8]} validated well - could update knowledge weights")
        elif not passed:
            logger.debug(f"Strategy {strategy_id[:8]} failed validation")
