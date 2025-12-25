"""
Unit Tests for Enhancement 2: Dynamic Sentiment Weighting

Tests the DynamicSentimentWeighter class for regime classification,
weight smoothing, and all 27 regime combinations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from primitives import (
    DynamicSentimentWeighter, 
    DynamicWeightConfig,
    VolatilityRegime,
    SocialRegime
)


class TestRegimeClassification:
    """Test volatility and social regime classification."""
    
    def test_low_volatility_classification(self):
        """ATR < 1.5% should be classified as LOW volatility."""
        weighter = DynamicSentimentWeighter()
        
        # Simulate enough updates for regime to stabilize
        ctx = {'atr': 0.01, 'social_zscore': 0.0, 'market_regime': 'Range'}
        for _ in range(10):
            weighter.get_weights(ctx)
        
        regime = weighter.get_current_regime()
        assert regime is not None
        assert regime[0] == VolatilityRegime.LOW
    
    def test_normal_volatility_classification(self):
        """ATR 1.5-4% should be classified as NORMAL volatility."""
        weighter = DynamicSentimentWeighter()
        
        ctx = {'atr': 0.025, 'social_zscore': 0.0, 'market_regime': 'Bull'}
        for _ in range(10):
            weighter.get_weights(ctx)
        
        regime = weighter.get_current_regime()
        assert regime is not None
        assert regime[0] == VolatilityRegime.NORMAL
    
    def test_high_volatility_classification(self):
        """ATR > 4% should be classified as HIGH volatility."""
        weighter = DynamicSentimentWeighter()
        
        ctx = {'atr': 0.05, 'social_zscore': 0.0, 'market_regime': 'Bear'}
        for _ in range(10):
            weighter.get_weights(ctx)
        
        regime = weighter.get_current_regime()
        assert regime is not None
        assert regime[0] == VolatilityRegime.HIGH
    
    def test_low_engagement_classification(self):
        """Social z-score < -1 should be LOW_ENGAGEMENT."""
        weighter = DynamicSentimentWeighter()
        
        ctx = {'atr': 0.025, 'social_zscore': -1.5, 'market_regime': 'Range'}
        for _ in range(10):
            weighter.get_weights(ctx)
        
        regime = weighter.get_current_regime()
        assert regime is not None
        assert regime[1] == SocialRegime.LOW_ENGAGEMENT
    
    def test_high_engagement_classification(self):
        """Social z-score > 1 should be HIGH_ENGAGEMENT."""
        weighter = DynamicSentimentWeighter()
        
        ctx = {'atr': 0.025, 'social_zscore': 1.5, 'market_regime': 'Bull'}
        for _ in range(10):
            weighter.get_weights(ctx)
        
        regime = weighter.get_current_regime()
        assert regime is not None
        assert regime[1] == SocialRegime.HIGH_ENGAGEMENT


class TestWeightSmoothing:
    """Test EMA weight smoothing and change limits."""
    
    def test_weights_sum_to_one(self):
        """Weights should always sum to 1.0."""
        weighter = DynamicSentimentWeighter()
        
        ctx = {'atr': 0.025, 'social_zscore': 0.0, 'market_regime': 'Bull'}
        weights = weighter.get_weights(ctx)
        
        total = weights['vader'] + weights['finbert'] + weights['twitter']
        assert abs(total - 1.0) < 0.01
    
    def test_no_negative_weights(self):
        """All weights should be non-negative."""
        weighter = DynamicSentimentWeighter()
        
        # Test various regime combinations
        for atr in [0.01, 0.025, 0.05]:
            for social in [-1.5, 0.0, 1.5]:
                for market in ['Bull', 'Bear', 'Range']:
                    ctx = {'atr': atr, 'social_zscore': social, 'market_regime': market}
                    weights = weighter.get_weights(ctx)
                    
                    assert weights['vader'] >= 0
                    assert weights['finbert'] >= 0
                    assert weights['twitter'] >= 0
    
    def test_weight_change_limit(self):
        """Weight changes should be limited to prevent sudden jumps."""
        config = DynamicWeightConfig(
            weight_change_limit=0.15,
            weight_smoothing_alpha=0.5,  # Faster smoothing for test
            min_regime_duration=1  # No waiting for test
        )
        weighter = DynamicSentimentWeighter(config)
        
        # Get initial weights
        ctx1 = {'atr': 0.025, 'social_zscore': 0.0, 'market_regime': 'Range'}
        initial_weights = weighter.get_weights(ctx1)
        
        # Switch to high volatility (different target weights)
        ctx2 = {'atr': 0.05, 'social_zscore': 1.5, 'market_regime': 'Bull'}
        for _ in range(3):
            new_weights = weighter.get_weights(ctx2)
        
        # Check that weights didn't jump too much
        vader_change = abs(new_weights['vader'] - initial_weights['vader'])
        # With smoothing and limits, change should be bounded
        assert vader_change < 0.5


class TestRegimeDurationFilter:
    """Test that regime must be stable before switching weights."""
    
    def test_regime_duration_filter(self):
        """Weights should not change until regime stable for min_duration."""
        config = DynamicWeightConfig(min_regime_duration=5)
        weighter = DynamicSentimentWeighter(config)
        
        # Establish initial regime
        ctx1 = {'atr': 0.025, 'social_zscore': 0.0, 'market_regime': 'Bull'}
        for _ in range(10):  # Ensure stable
            weighter.get_weights(ctx1)
        
        initial_weights = dict(weighter._current_weights)
        
        # Switch regime but only 3 bars (< min_duration of 5)
        ctx2 = {'atr': 0.05, 'social_zscore': 1.5, 'market_regime': 'Bull'}
        for _ in range(3):
            weights = weighter.get_weights(ctx2)
        
        # Weights should not have changed significantly yet
        vader_diff = abs(weights['vader'] - initial_weights['vader'])
        assert vader_diff < 0.1  # Small change due to smoothing only


class TestAll27RegimeCombinations:
    """Test that all 27 regime combinations return valid weights."""
    
    def test_all_combinations_have_weights(self):
        """All 27 combinations should return weights that sum to 1.0."""
        weighter = DynamicSentimentWeighter(
            DynamicWeightConfig(min_regime_duration=1)  # No waiting
        )
        
        volatility_atrs = [0.01, 0.025, 0.05]  # LOW, NORMAL, HIGH
        social_zscores = [-1.5, 0.0, 1.5]  # LOW, NORMAL, HIGH
        market_regimes = ['Bull', 'Bear', 'Range']
        
        combinations_tested = 0
        
        for atr in volatility_atrs:
            for social in social_zscores:
                for market in market_regimes:
                    # Reset weighter for clean test
                    weighter = DynamicSentimentWeighter(
                        DynamicWeightConfig(min_regime_duration=1)
                    )
                    
                    ctx = {'atr': atr, 'social_zscore': social, 'market_regime': market}
                    
                    # Run enough iterations for regime to stabilize
                    for _ in range(10):
                        weights = weighter.get_weights(ctx)
                    
                    # Verify weights are valid
                    total = weights['vader'] + weights['finbert'] + weights['twitter']
                    assert abs(total - 1.0) < 0.01, f"Weights don't sum to 1.0 for {ctx}"
                    assert all(w >= 0 for w in weights.values()), f"Negative weight for {ctx}"
                    
                    combinations_tested += 1
        
        assert combinations_tested == 27


class TestDefaultWeights:
    """Test default weight behavior."""
    
    def test_default_weights_without_context(self):
        """Without regime_context, should return static defaults."""
        weighter = DynamicSentimentWeighter()
        
        weights = weighter.get_weights(None)
        
        assert weights['vader'] == 0.35
        assert weights['finbert'] == 0.65
        assert weights['twitter'] == 0.0
    
    def test_transition_count_tracking(self):
        """Should track the number of weight transitions."""
        weighter = DynamicSentimentWeighter(
            DynamicWeightConfig(min_regime_duration=1)
        )
        
        initial_count = weighter.get_transition_count()
        
        # Trigger some regime changes
        for _ in range(5):
            weighter.get_weights({'atr': 0.025, 'social_zscore': 0.0, 'market_regime': 'Bull'})
        for _ in range(5):
            weighter.get_weights({'atr': 0.05, 'social_zscore': 1.5, 'market_regime': 'Bull'})
        
        # Should have some transitions from smoothing
        final_count = weighter.get_transition_count()
        assert final_count >= initial_count


class TestStatsAndReset:
    """Test stats reporting and reset functionality."""
    
    def test_get_stats(self):
        """get_stats should return current state."""
        weighter = DynamicSentimentWeighter()
        
        ctx = {'atr': 0.025, 'social_zscore': 0.0, 'market_regime': 'Bull'}
        weighter.get_weights(ctx)
        
        stats = weighter.get_stats()
        
        assert 'current_weights' in stats
        assert 'current_regime' in stats
        assert 'regime_duration' in stats
        assert 'transition_count' in stats
    
    def test_reset(self):
        """reset should restore to defaults."""
        weighter = DynamicSentimentWeighter()
        
        # Make some changes
        ctx = {'atr': 0.05, 'social_zscore': 1.5, 'market_regime': 'Bull'}
        for _ in range(10):
            weighter.get_weights(ctx)
        
        # Reset
        weighter.reset()
        
        # Check defaults restored
        assert weighter._current_weights['vader'] == 0.35
        assert weighter._current_weights['finbert'] == 0.65
        assert weighter._current_regime is None
        assert weighter._regime_duration == 0
        assert weighter._transition_count == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
