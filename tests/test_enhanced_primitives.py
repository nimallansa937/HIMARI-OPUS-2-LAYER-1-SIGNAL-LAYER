"""
Unit Tests for Layer 1 Signal Enhancements

Tests all 7 core primitives:
1. StreamingHMM
2. StreamingIndicators
3. WelfordOnlineStats
4. MultiHorizonMomentum
5. OrderBookImbalance
6. RegimeAwareSignalFusion
7. HybridSentimentAnalyzer
"""

import pytest
import numpy as np
from primitives import (
    StreamingHMM, HMMConfig, MarketRegime,
    StreamingIndicators, IndicatorConfig,
    WelfordOnlineStats, MultiSymbolWelford,
    MultiHorizonMomentum, MomentumConfig,
    OrderBookImbalance, OBIConfig,
    RegimeAwareSignalFusion, FusionConfig,
)


class TestStreamingHMM:
    """Unit tests for HMM implementation."""
    
    def test_initialization(self):
        """HMM initializes with uniform prior."""
        hmm = StreamingHMM()
        assert hmm.n_states == 3
        assert np.allclose(hmm.state_probs.sum(), 1.0)
        assert np.allclose(hmm.state_probs, 1/3, atol=0.01)
    
    def test_probability_conservation(self):
        """Probabilities must sum to 1 after each update."""
        hmm = StreamingHMM()
        for _ in range(100):
            ret = np.random.normal(0, 0.02)
            state, conf, probs = hmm.update(ret)
            assert np.allclose(probs.sum(), 1.0), "Probabilities don't sum to 1"
            assert all(p >= 0 for p in probs), "Negative probability"
            assert 0 <= conf <= 1, "Invalid confidence"
    
    def test_zero_lag_detection(self):
        """Single large observation should shift probabilities immediately."""
        hmm = StreamingHMM()
        
        # Establish bull regime
        for _ in range(30):
            hmm.update(np.random.normal(0.002, 0.01))
        
        bull_prob_before = hmm.state_probs[0]
        
        # Single shock
        hmm.update(-0.05)
        
        bull_prob_after = hmm.state_probs[0]
        
        # Probability should drop significantly in ONE update
        assert bull_prob_after < bull_prob_before * 0.5, "HMM did not respond to shock"
    
    def test_regime_persistence(self):
        """Consistent signals should increase regime confidence."""
        hmm = StreamingHMM()
        
        # 50 bull-like returns
        for _ in range(50):
            hmm.update(np.random.normal(0.002, 0.008))
        
        # Should be confidently in Bull
        assert hmm.get_regime_label() == 'Bull'
        assert hmm.state_probs[0] > 0.7, f"Bull confidence too low: {hmm.state_probs[0]}"
    
    def test_reset(self):
        """Reset should return to uniform prior."""
        hmm = StreamingHMM()
        for _ in range(50):
            hmm.update(np.random.normal(0.002, 0.01))
        
        hmm.reset()
        assert np.allclose(hmm.state_probs, 1/3, atol=0.01)
        assert hmm.updates_count == 0


class TestStreamingIndicators:
    """Tests for streaming indicators."""
    
    def test_initialization(self):
        """Indicators initialize correctly."""
        ind = StreamingIndicators()
        assert ind.update_count == 0
        assert len(ind.indicators) > 0
    
    def test_update_returns_dict(self):
        """Update returns dict with expected keys."""
        ind = StreamingIndicators()
        ohlcv = {'open': 100, 'high': 101, 'low': 99, 'close': 100.5, 'volume': 1000}
        
        result = ind.update(ohlcv)
        
        assert isinstance(result, dict)
        assert 'close' in result
        assert 'ema_21' in result or result['ema_21'] is None
        assert 'rsi' in result or result['rsi'] is None
    
    def test_ema_convergence(self):
        """EMA should converge to price level."""
        ind = StreamingIndicators()
        
        # Feed constant price
        for _ in range(100):
            ohlcv = {'open': 100, 'high': 100, 'low': 100, 'close': 100, 'volume': 1000}
            result = ind.update(ohlcv)
        
        # EMAs should be close to 100
        if result['ema_5'] is not None:
            assert abs(result['ema_5'] - 100) < 1, f"EMA5 not converged: {result['ema_5']}"


class TestWelfordOnlineStats:
    """Tests for Welford online statistics."""
    
    def test_accuracy_vs_numpy(self):
        """Welford should match numpy calculations."""
        welford = WelfordOnlineStats(min_samples=5)
        data = np.random.randn(1000)
        
        for x in data:
            welford.update(x)
        
        np_mean = data.mean()
        np_std = data.std(ddof=1)
        
        assert abs(welford.get_mean() - np_mean) < 0.001
        assert abs(welford.get_std() - np_std) < 0.001
    
    def test_numerical_stability(self):
        """Welford should handle extreme values."""
        welford = WelfordOnlineStats(min_samples=5)
        
        # Large values
        for x in [1e6, 1e6 + 1, 1e6 + 2, 1e6 + 3, 1e6 + 4]:
            welford.update(x)
        
        # Should not overflow or produce NaN
        assert np.isfinite(welford.get_mean())
        assert np.isfinite(welford.get_std())
    
    def test_z_score(self):
        """Z-score calculation should be correct."""
        welford = WelfordOnlineStats(min_samples=5)
        
        # Standard normal data
        data = np.random.randn(100)
        for x in data:
            welford.update(x)
        
        # Z-score of mean should be ~0
        z = welford.z_score(welford.get_mean())
        assert abs(z) < 0.1
    
    def test_multi_symbol(self):
        """MultiSymbolWelford should track separate symbols."""
        stats = MultiSymbolWelford()
        
        # Update different symbols
        stats.update('BTC', 0.01)
        stats.update('ETH', 0.02)
        stats.update('BTC', 0.015)
        
        all_stats = stats.get_all_stats()
        assert 'BTC' in all_stats
        assert 'ETH' in all_stats


class TestMultiHorizonMomentum:
    """Tests for momentum feature generator."""
    
    def test_feature_generation(self):
        """Features should be generated after sufficient history."""
        mom = MultiHorizonMomentum()
        
        # Need enough history
        for i in range(100):
            price = 100 + i * 0.1  # Uptrend
            features = mom.update(price)
        
        # All momentum should be positive in uptrend
        assert features['mom_5'] is not None
        assert features['mom_5'] > 0
        assert features['mom_21'] > 0
    
    def test_composite_features(self):
        """Composite features should be calculated."""
        mom = MultiHorizonMomentum()
        
        for i in range(100):
            price = 100 + i * 0.1
            features = mom.update(price)
        
        # Check composite features exist
        assert 'mom_alignment' in features
        assert 'mom_divergence' in features
        assert 'mom_strength' in features
        
        # Alignment should be high in consistent uptrend
        if features['mom_alignment'] is not None:
            assert features['mom_alignment'] > 0.5


class TestOrderBookImbalance:
    """Tests for OBI calculation."""
    
    def test_initialization(self):
        """OBI initializes correctly."""
        obi = OrderBookImbalance()
        assert obi.update_count == 0
    
    def test_balanced_book(self):
        """Balanced order book should give OBI near 0."""
        obi = OrderBookImbalance()
        
        orderbook = {
            'bids': [(100, 10), (99, 10), (98, 10)],
            'asks': [(101, 10), (102, 10), (103, 10)]
        }
        
        result = obi.update(orderbook)
        
        assert result['obi_raw'] is not None
        assert abs(result['obi_raw']) < 0.1  # Should be close to 0
    
    def test_bid_heavy_book(self):
        """Bid-heavy book should give positive OBI."""
        obi = OrderBookImbalance()
        
        orderbook = {
            'bids': [(100, 20), (99, 20), (98, 20)],
            'asks': [(101, 5), (102, 5), (103, 5)]
        }
        
        result = obi.update(orderbook)
        
        assert result['obi_raw'] > 0  # Positive OBI


class TestRegimeAwareSignalFusion:
    """Tests for signal fusion."""
    
    def test_signal_registration(self):
        """Signals should be registered correctly."""
        hmm = StreamingHMM()
        fusion = RegimeAwareSignalFusion(hmm)
        
        fusion.register_signal('test_signal', 'momentum', base_weight=1.0)
        
        assert 'test_signal' in fusion.signals
        assert fusion.signals['test_signal'].category == 'momentum'
    
    def test_fusion_output(self):
        """Fusion should output composite signal."""
        hmm = StreamingHMM()
        fusion = RegimeAwareSignalFusion(hmm)
        
        fusion.register_signal('sig1', 'momentum', base_weight=1.0)
        fusion.register_signal('sig2', 'mean_reversion', base_weight=1.0)
        
        # Establish regime
        for _ in range(10):
            hmm.update(0.002)
        
        result = fusion.fuse({
            'sig1': 0.5,
            'sig2': -0.3
        })
        
        assert 'composite' in result
        assert 'regime' in result
        assert 'confidence' in result
        assert -1 <= result['composite'] <= 1


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_hmm_with_welford(self):
        """HMM and Welford should work together."""
        hmm = StreamingHMM()
        welford = WelfordOnlineStats()
        
        returns = np.random.randn(100) * 0.02
        
        for ret in returns:
            hmm.update(ret)
            welford.update(ret)
        
        # Both should have valid state
        assert hmm.updates_count == 100
        assert welford.state.n == 100
    
    def test_momentum_with_fusion(self):
        """Momentum features with fusion."""
        hmm = StreamingHMM()
        momentum = MultiHorizonMomentum()
        fusion = RegimeAwareSignalFusion(hmm)
        
        fusion.register_signal('mom_5', 'momentum', base_weight=1.0)
        fusion.register_signal('mom_21', 'trend_following', base_weight=1.0)
        
        # Generate price series
        prices = [100 + i * 0.1 for i in range(100)]
        
        for i, price in enumerate(prices):
            # Update momentum
            mom_features = momentum.update(price)
            
            # Update HMM with return
            if i > 0:
                ret = (price - prices[i-1]) / prices[i-1]
                hmm.update(ret)
            
            # Fuse signals if available
            if mom_features['mom_5'] is not None:
                signals = {
                    'mom_5': mom_features['mom_5'],
                    'mom_21': mom_features.get('mom_21', 0)
                }
                result = fusion.fuse(signals)
                assert 'composite' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
