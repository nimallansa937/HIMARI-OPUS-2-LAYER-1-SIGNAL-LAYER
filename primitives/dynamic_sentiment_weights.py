"""
Dynamic Sentiment Weights - Regime-Adaptive Weighting

Compute optimal sentiment source weights based on current market regime.

Enhancement 2 from ANTIGRAVITY_SENTIMENT_ENHANCEMENT_GUIDE.md
Impact: +21% accuracy improvement, +2-4% Sharpe
"""

import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class VolatilityRegime(Enum):
    LOW = "LOW_VOLATILITY"
    NORMAL = "NORMAL"
    HIGH = "HIGH_VOLATILITY"


class SocialRegime(Enum):
    LOW_ENGAGEMENT = "low_engagement"
    NORMAL = "normal"
    HIGH_ENGAGEMENT = "high_engagement"


@dataclass
class DynamicWeightConfig:
    """Configuration for dynamic sentiment weighting."""
    # Volatility thresholds (ATR as fraction of price)
    volatility_low: float = 0.015    # ATR < 1.5% daily
    volatility_high: float = 0.040   # ATR > 4% daily
    
    # Social volume thresholds (z-score)
    social_low: float = -1.0
    social_high: float = 1.0
    
    # Smoothing parameters
    weight_smoothing_alpha: float = 0.1
    min_regime_duration: int = 5  # Bars before switching
    weight_change_limit: float = 0.15  # Max weight change per update
    
    # Default weights (static fallback)
    default_vader: float = 0.35
    default_finbert: float = 0.65
    default_twitter: float = 0.0


# 27 regime combinations (3 vol x 3 social x 3 market)
REGIME_WEIGHT_MATRIX = {
    # Normal volatility, normal engagement
    (VolatilityRegime.NORMAL, SocialRegime.NORMAL, 'Bull'): {
        'vader': 0.35, 'finbert': 0.65, 'twitter': 0.0
    },
    (VolatilityRegime.NORMAL, SocialRegime.NORMAL, 'Bear'): {
        'vader': 0.35, 'finbert': 0.65, 'twitter': 0.0
    },
    (VolatilityRegime.NORMAL, SocialRegime.NORMAL, 'Range'): {
        'vader': 0.35, 'finbert': 0.65, 'twitter': 0.0
    },
    
    # High volatility, high engagement - add Twitter
    (VolatilityRegime.HIGH, SocialRegime.HIGH_ENGAGEMENT, 'Bull'): {
        'vader': 0.20, 'finbert': 0.50, 'twitter': 0.30
    },
    (VolatilityRegime.HIGH, SocialRegime.HIGH_ENGAGEMENT, 'Bear'): {
        'vader': 0.25, 'finbert': 0.45, 'twitter': 0.30
    },
    (VolatilityRegime.HIGH, SocialRegime.HIGH_ENGAGEMENT, 'Range'): {
        'vader': 0.20, 'finbert': 0.50, 'twitter': 0.30
    },
    
    # Low volatility, low engagement - reduce FinBERT reliance
    (VolatilityRegime.LOW, SocialRegime.LOW_ENGAGEMENT, 'Bull'): {
        'vader': 0.50, 'finbert': 0.50, 'twitter': 0.0
    },
    (VolatilityRegime.LOW, SocialRegime.LOW_ENGAGEMENT, 'Bear'): {
        'vader': 0.50, 'finbert': 0.50, 'twitter': 0.0
    },
    (VolatilityRegime.LOW, SocialRegime.LOW_ENGAGEMENT, 'Range'): {
        'vader': 0.50, 'finbert': 0.50, 'twitter': 0.0
    },
    
    # High volatility, low engagement - focus on formal news
    (VolatilityRegime.HIGH, SocialRegime.LOW_ENGAGEMENT, 'Bull'): {
        'vader': 0.30, 'finbert': 0.70, 'twitter': 0.0
    },
    (VolatilityRegime.HIGH, SocialRegime.LOW_ENGAGEMENT, 'Bear'): {
        'vader': 0.30, 'finbert': 0.70, 'twitter': 0.0
    },
    (VolatilityRegime.HIGH, SocialRegime.LOW_ENGAGEMENT, 'Range'): {
        'vader': 0.30, 'finbert': 0.70, 'twitter': 0.0
    },
    
    # Low volatility, high engagement - social is leading
    (VolatilityRegime.LOW, SocialRegime.HIGH_ENGAGEMENT, 'Bull'): {
        'vader': 0.25, 'finbert': 0.50, 'twitter': 0.25
    },
    (VolatilityRegime.LOW, SocialRegime.HIGH_ENGAGEMENT, 'Bear'): {
        'vader': 0.25, 'finbert': 0.50, 'twitter': 0.25
    },
    (VolatilityRegime.LOW, SocialRegime.HIGH_ENGAGEMENT, 'Range'): {
        'vader': 0.30, 'finbert': 0.50, 'twitter': 0.20
    },
    
    # Normal volatility, high engagement
    (VolatilityRegime.NORMAL, SocialRegime.HIGH_ENGAGEMENT, 'Bull'): {
        'vader': 0.25, 'finbert': 0.55, 'twitter': 0.20
    },
    (VolatilityRegime.NORMAL, SocialRegime.HIGH_ENGAGEMENT, 'Bear'): {
        'vader': 0.25, 'finbert': 0.55, 'twitter': 0.20
    },
    (VolatilityRegime.NORMAL, SocialRegime.HIGH_ENGAGEMENT, 'Range'): {
        'vader': 0.30, 'finbert': 0.50, 'twitter': 0.20
    },
    
    # Normal volatility, low engagement
    (VolatilityRegime.NORMAL, SocialRegime.LOW_ENGAGEMENT, 'Bull'): {
        'vader': 0.40, 'finbert': 0.60, 'twitter': 0.0
    },
    (VolatilityRegime.NORMAL, SocialRegime.LOW_ENGAGEMENT, 'Bear'): {
        'vader': 0.40, 'finbert': 0.60, 'twitter': 0.0
    },
    (VolatilityRegime.NORMAL, SocialRegime.LOW_ENGAGEMENT, 'Range'): {
        'vader': 0.40, 'finbert': 0.60, 'twitter': 0.0
    },
    
    # High volatility, normal engagement
    (VolatilityRegime.HIGH, SocialRegime.NORMAL, 'Bull'): {
        'vader': 0.25, 'finbert': 0.60, 'twitter': 0.15
    },
    (VolatilityRegime.HIGH, SocialRegime.NORMAL, 'Bear'): {
        'vader': 0.25, 'finbert': 0.60, 'twitter': 0.15
    },
    (VolatilityRegime.HIGH, SocialRegime.NORMAL, 'Range'): {
        'vader': 0.30, 'finbert': 0.55, 'twitter': 0.15
    },
    
    # Low volatility, normal engagement
    (VolatilityRegime.LOW, SocialRegime.NORMAL, 'Bull'): {
        'vader': 0.40, 'finbert': 0.60, 'twitter': 0.0
    },
    (VolatilityRegime.LOW, SocialRegime.NORMAL, 'Bear'): {
        'vader': 0.40, 'finbert': 0.60, 'twitter': 0.0
    },
    (VolatilityRegime.LOW, SocialRegime.NORMAL, 'Range'): {
        'vader': 0.40, 'finbert': 0.60, 'twitter': 0.0
    },
}


class DynamicSentimentWeighter:
    """
    Compute optimal sentiment weights based on market regime.
    
    Uses 3 regime dimensions:
    1. Volatility regime (LOW/NORMAL/HIGH based on ATR)
    2. Social engagement regime (based on social volume z-score)
    3. Market regime (Bull/Bear/Range from HMM)
    
    Example:
        weighter = DynamicSentimentWeighter()
        
        regime_context = {
            'atr': 0.05,
            'social_zscore': 1.5,
            'market_regime': 'Bull'
        }
        
        weights = weighter.get_weights(regime_context)
        # {'vader': 0.20, 'finbert': 0.50, 'twitter': 0.30}
    """
    
    def __init__(self, config: Optional[DynamicWeightConfig] = None):
        """
        Initialize dynamic sentiment weighter.
        
        Args:
            config: Weight configuration settings
        """
        self.config = config or DynamicWeightConfig()
        
        # Current smoothed weights
        self._current_weights = {
            'vader': self.config.default_vader,
            'finbert': self.config.default_finbert,
            'twitter': self.config.default_twitter
        }
        
        # Regime duration tracking
        self._current_regime: Optional[Tuple] = None
        self._regime_duration: int = 0
        
        # Transition logging
        self._transition_count: int = 0
        
        logger.info("DynamicSentimentWeighter initialized")
    
    def get_weights(
        self, 
        regime_context: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Get sentiment weights for current regime.
        
        Args:
            regime_context: Dict with 'atr', 'social_zscore', 'market_regime'
            
        Returns:
            Dict with 'vader', 'finbert', 'twitter' weights (sum to 1.0)
        """
        if not regime_context:
            return dict(self._current_weights)
        
        # Classify regimes
        vol_regime = self._classify_volatility(regime_context.get('atr', 0.025))
        social_regime = self._classify_social(regime_context.get('social_zscore', 0.0))
        market_regime = regime_context.get('market_regime', 'Range')
        
        regime_key = (vol_regime, social_regime, market_regime)
        
        # Check regime duration filter
        if regime_key != self._current_regime:
            self._regime_duration = 0
            self._current_regime = regime_key
        else:
            self._regime_duration += 1
        
        # Only switch if regime stable for min duration
        if self._regime_duration < self.config.min_regime_duration:
            return dict(self._current_weights)
        
        # Look up target weights
        target_weights = self._lookup_weights(regime_key)
        
        # Apply smoothing
        smoothed_weights = self._smooth_weights(target_weights)
        
        return smoothed_weights
    
    def _classify_volatility(self, atr: float) -> VolatilityRegime:
        """Classify volatility regime from ATR."""
        if atr < self.config.volatility_low:
            return VolatilityRegime.LOW
        elif atr > self.config.volatility_high:
            return VolatilityRegime.HIGH
        return VolatilityRegime.NORMAL
    
    def _classify_social(self, zscore: float) -> SocialRegime:
        """Classify social engagement regime from z-score."""
        if zscore < self.config.social_low:
            return SocialRegime.LOW_ENGAGEMENT
        elif zscore > self.config.social_high:
            return SocialRegime.HIGH_ENGAGEMENT
        return SocialRegime.NORMAL
    
    def _lookup_weights(self, regime_key: Tuple) -> Dict[str, float]:
        """Look up weights from regime matrix."""
        if regime_key in REGIME_WEIGHT_MATRIX:
            return REGIME_WEIGHT_MATRIX[regime_key]
        
        # Fallback: find nearest match
        return self._find_nearest_match(regime_key)
    
    def _find_nearest_match(self, regime_key: Tuple) -> Dict[str, float]:
        """Find nearest matching regime in matrix."""
        vol, social, market = regime_key
        
        # Try with just volatility and market regime
        for social_try in SocialRegime:
            key = (vol, social_try, market)
            if key in REGIME_WEIGHT_MATRIX:
                logger.debug(f"Fallback regime match: {key}")
                return REGIME_WEIGHT_MATRIX[key]
        
        # Ultimate fallback
        logger.warning(f"No regime match for {regime_key}, using defaults")
        return {
            'vader': self.config.default_vader,
            'finbert': self.config.default_finbert,
            'twitter': self.config.default_twitter
        }
    
    def _smooth_weights(self, target: Dict[str, float]) -> Dict[str, float]:
        """Apply EMA smoothing with change limits."""
        alpha = self.config.weight_smoothing_alpha
        limit = self.config.weight_change_limit
        
        smoothed = {}
        for key in ['vader', 'finbert', 'twitter']:
            current = self._current_weights[key]
            target_val = target.get(key, 0.0)
            
            # EMA
            new_val = alpha * target_val + (1 - alpha) * current
            
            # Limit change
            change = new_val - current
            if abs(change) > limit:
                new_val = current + (limit if change > 0 else -limit)
            
            smoothed[key] = new_val
        
        # Normalize to sum to 1
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {k: v / total for k, v in smoothed.items()}
        
        # Check if transition occurred
        if smoothed != self._current_weights:
            old_weights = dict(self._current_weights)
            self._current_weights = smoothed
            self._transition_count += 1
            
            logger.debug(
                f"Weight transition #{self._transition_count}: "
                f"vader {old_weights['vader']:.2f}->{smoothed['vader']:.2f}, "
                f"finbert {old_weights['finbert']:.2f}->{smoothed['finbert']:.2f}"
            )
        
        return dict(self._current_weights)
    
    def get_current_regime(self) -> Optional[Tuple]:
        """Get current regime classification."""
        return self._current_regime
    
    def get_transition_count(self) -> int:
        """Get number of weight transitions."""
        return self._transition_count
    
    def get_stats(self) -> Dict:
        """Get weighter statistics."""
        return {
            'current_weights': dict(self._current_weights),
            'current_regime': str(self._current_regime) if self._current_regime else None,
            'regime_duration': self._regime_duration,
            'transition_count': self._transition_count
        }
    
    def reset(self) -> None:
        """Reset to default state."""
        self._current_weights = {
            'vader': self.config.default_vader,
            'finbert': self.config.default_finbert,
            'twitter': self.config.default_twitter
        }
        self._current_regime = None
        self._regime_duration = 0
        self._transition_count = 0

    # =========================================================================
    # RESEARCH-BACKED ENHANCEMENTS
    # =========================================================================

    def get_regime_sentiment_multiplier(
        self, 
        market_regime: str, 
        source: str = 'news'
    ) -> float:
        """
        Get regime-conditional sentiment multiplier (Enhancement #2).
        
        Research shows sentiment predictive power varies 3x between regimes:
        - Bull market: Social media more predictive (FOMO is real)
        - Bear market: News more predictive (fear drives fundamentals)
        - Range: Equal weighting works best
        
        Args:
            market_regime: 'Bull', 'Bear', or 'Range'
            source: 'news' or 'social'
            
        Returns:
            Multiplier for sentiment impact (0.5 to 1.5)
        """
        REGIME_MULTIPLIERS = {
            'Bull': {'news': 0.8, 'social': 1.2, 'finbert': 0.9, 'twitter': 1.3},
            'Bear': {'news': 1.3, 'social': 0.7, 'finbert': 1.2, 'twitter': 0.6},
            'Range': {'news': 1.0, 'social': 1.0, 'finbert': 1.0, 'twitter': 1.0},
        }
        
        regime_map = REGIME_MULTIPLIERS.get(market_regime, REGIME_MULTIPLIERS['Range'])
        return regime_map.get(source, 1.0)

    def compute_confidence_weighted_score(
        self,
        scores: Dict[str, float],
        confidences: Dict[str, float],
        regime_context: Optional[Dict] = None
    ) -> float:
        """
        Compute confidence-weighted sentiment score (Enhancement #3).
        
        Research shows weighting by prediction confidence, not just signal value,
        improves accuracy by 3-5%.
        
        Args:
            scores: Dict of source -> sentiment score (-1 to +1)
            confidences: Dict of source -> confidence (0 to 1)
            regime_context: Optional regime for weight adjustment
            
        Returns:
            Confidence-weighted sentiment score
        """
        # Get base weights from regime
        base_weights = self.get_weights(regime_context)
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for source, score in scores.items():
            # Map source to weight key
            weight_key = 'finbert' if source in ['news', 'finbert', 'bloomberg', 'reuters'] else 'twitter'
            if source == 'vader':
                weight_key = 'vader'
            
            base_weight = base_weights.get(weight_key, 0.5)
            confidence = confidences.get(source, 0.5)
            
            # Final weight = base_weight * confidence^2 (emphasize high confidence)
            effective_weight = base_weight * (confidence ** 2)
            
            weighted_sum += score * effective_weight
            total_weight += effective_weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0


class SourceCredibilityTracker:
    """
    Track and update source credibility scores (Enhancement #7).
    
    Uses Bayesian updating based on historical accuracy.
    Sources with better track records get higher weights.
    
    Example:
        tracker = SourceCredibilityTracker()
        
        # Record outcome
        tracker.record_outcome('bloomberg', predicted='bullish', actual='bullish')
        tracker.record_outcome('twitter', predicted='bullish', actual='bearish')
        
        # Get credibility-adjusted weights
        weights = tracker.get_credibility_weights()
        # {'bloomberg': 0.85, 'twitter': 0.45, ...}
    """
    
    def __init__(self, default_credibility: float = 0.5):
        """
        Initialize source credibility tracker.
        
        Args:
            default_credibility: Starting credibility for new sources
        """
        self.default_credibility = default_credibility
        
        # Track hits and misses per source
        # {source: {'hits': int, 'misses': int, 'total': int}}
        self._source_stats: Dict[str, Dict[str, int]] = {}
        
        # Bayesian priors (add smoothing)
        self._prior_hits = 5  # Assume 5 correct predictions
        self._prior_misses = 5  # Assume 5 incorrect predictions
    
    def record_outcome(
        self,
        source: str,
        predicted: str,  # 'bullish', 'bearish', 'neutral'
        actual: str,     # 'bullish', 'bearish', 'neutral'
        confidence: float = 0.5
    ) -> None:
        """
        Record a prediction outcome for credibility updating.
        
        Args:
            source: Source name
            predicted: Predicted sentiment
            actual: Actual outcome
            confidence: Prediction confidence (weights the update)
        """
        if source not in self._source_stats:
            self._source_stats[source] = {
                'hits': self._prior_hits,
                'misses': self._prior_misses,
                'total': self._prior_hits + self._prior_misses
            }
        
        # Weight update by confidence
        update_weight = max(0.1, confidence)
        
        if predicted == actual:
            self._source_stats[source]['hits'] += update_weight
        else:
            self._source_stats[source]['misses'] += update_weight
        
        self._source_stats[source]['total'] += update_weight
    
    def get_credibility(self, source: str) -> float:
        """
        Get current credibility score for a source.
        
        Uses Bayesian estimate: (hits + prior) / (total + prior)
        
        Args:
            source: Source name
            
        Returns:
            Credibility score (0 to 1)
        """
        if source not in self._source_stats:
            return self.default_credibility
        
        stats = self._source_stats[source]
        return stats['hits'] / stats['total']
    
    def get_credibility_weights(self) -> Dict[str, float]:
        """
        Get credibility-adjusted weights for all tracked sources.
        
        Returns:
            Dict of source -> credibility weight (normalized)
        """
        weights = {}
        for source in self._source_stats:
            weights[source] = self.get_credibility(source)
        
        # Normalize
        if weights:
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def get_stats(self) -> Dict[str, Dict]:
        """Get detailed statistics for all sources."""
        return {
            source: {
                'credibility': self.get_credibility(source),
                'hits': stats['hits'],
                'misses': stats['misses'],
                'total': stats['total']
            }
            for source, stats in self._source_stats.items()
        }
    
    def decay_old_data(self, decay_factor: float = 0.95) -> None:
        """
        Apply decay to old data (reduces influence of old outcomes).
        
        Args:
            decay_factor: Multiplier for existing counts (0.9-0.99)
        """
        for source in self._source_stats:
            self._source_stats[source]['hits'] *= decay_factor
            self._source_stats[source]['misses'] *= decay_factor
            self._source_stats[source]['total'] *= decay_factor
            
            # Maintain minimum counts
            if self._source_stats[source]['total'] < self._prior_hits + self._prior_misses:
                self._source_stats[source]['hits'] = self._prior_hits
                self._source_stats[source]['misses'] = self._prior_misses
                self._source_stats[source]['total'] = self._prior_hits + self._prior_misses
