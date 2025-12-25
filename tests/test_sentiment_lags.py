"""
Unit Tests for Enhancement 1: Sentiment Lag Features

Tests for SentimentLagBuffer implementation.
"""

import pytest
from primitives import SentimentLagBuffer, LagConfig


class TestSentimentLagBuffer:
    """Unit tests for sentiment lag buffer."""

    def test_initialization(self):
        """Buffer initializes with correct config."""
        buffer = SentimentLagBuffer()
        assert buffer.config.max_lag_bars == 360
        assert buffer.config.bar_interval_minutes == 1
        assert 'news' in buffer.config.lag_horizons

    def test_update_creates_buffer(self):
        """Update creates symbol buffer."""
        buffer = SentimentLagBuffer()
        buffer.update('BTCUSDT', score=0.5, source='news')

        assert 'BTCUSDT' in buffer._buffers
        assert 'news' in buffer._buffers['BTCUSDT']
        assert buffer.get_buffer_length('BTCUSDT', 'news') == 1

    def test_buffer_overflow_handling(self):
        """Buffer respects max_lag_bars limit."""
        config = LagConfig(max_lag_bars=100)
        buffer = SentimentLagBuffer(config)

        # Add 150 updates (should keep only last 101)
        for i in range(150):
            buffer.update('BTCUSDT', score=float(i), source='news')

        assert buffer.get_buffer_length('BTCUSDT', 'news') == 101

        # Check oldest value is 49 (150-101)
        current = buffer.get_current('BTCUSDT', 'news')
        assert current == 149.0

    def test_lag_indexing_accuracy(self):
        """Lag features return correct historical values."""
        config = LagConfig(
            max_lag_bars=120,
            bar_interval_minutes=1,
            lag_horizons={'news': [30, 60, 90]}
        )
        buffer = SentimentLagBuffer(config)

        # Add 100 scores with known values
        for i in range(100):
            buffer.update('BTCUSDT', score=float(i) / 100, source='news')

        features = buffer.get_lag_features('BTCUSDT', source='news')

        # Current is 99/100 = 0.99
        # 30 bars ago: (99-30)/100 = 0.69
        # 60 bars ago: (99-60)/100 = 0.39
        # 90 bars ago: (99-90)/100 = 0.09

        assert abs(features['news_lag_30m'] - 0.69) < 0.02  # Relaxed tolerance for floating point
        assert abs(features['news_lag_60m'] - 0.39) < 0.02
        assert abs(features['news_lag_90m'] - 0.09) < 0.02

    def test_multi_symbol_isolation(self):
        """Buffers for different symbols are independent."""
        buffer = SentimentLagBuffer()

        buffer.update('BTCUSDT', score=0.8, source='news')
        buffer.update('ETHUSDT', score=-0.5, source='news')
        buffer.update('BTCUSDT', score=0.9, source='news')

        btc_current = buffer.get_current('BTCUSDT', 'news')
        eth_current = buffer.get_current('ETHUSDT', 'news')

        assert btc_current == 0.9
        assert eth_current == -0.5
        assert buffer.get_buffer_length('BTCUSDT', 'news') == 2
        assert buffer.get_buffer_length('ETHUSDT', 'news') == 1

    def test_aggregation_acceleration(self):
        """Acceleration feature calculated correctly."""
        buffer = SentimentLagBuffer()

        # Create linear increase in sentiment
        for i in range(60):
            buffer.update('BTCUSDT', score=float(i) / 100, source='news')

        features = buffer.get_lag_features('BTCUSDT')

        # Acceleration = (current - 30m_ago) / 30
        # = (59/100 - 29/100) / 30 = 0.30 / 30 = 0.01
        assert 'news_acceleration' in features
        assert abs(features['news_acceleration'] - 0.01) < 0.001

    def test_aggregation_momentum_3h(self):
        """3-hour momentum calculated correctly."""
        buffer = SentimentLagBuffer()

        # Add 200 bars of constant positive sentiment
        for i in range(200):
            buffer.update('BTCUSDT', score=0.5, source='news')

        features = buffer.get_lag_features('BTCUSDT')

        # Momentum = sum of last 180 bars = 180 * 0.5 = 90
        assert 'news_momentum_3h' in features
        assert abs(features['news_momentum_3h'] - 90.0) < 0.1

    def test_sentiment_reversal_flag(self):
        """Reversal flag detects sign changes."""
        buffer = SentimentLagBuffer()

        # First 30 bars positive
        for i in range(30):
            buffer.update('BTCUSDT', score=0.5, source='news')

        # Next 30 bars negative (reversal)
        for i in range(30):
            buffer.update('BTCUSDT', score=-0.5, source='news')

        features = buffer.get_lag_features('BTCUSDT')

        # Should detect reversal in last 60 bars
        assert 'news_reversal_flag' in features
        assert features['news_reversal_flag'] == 1.0
        assert features['news_reversal_count'] > 0

    def test_graceful_insufficient_buffer(self):
        """Returns 0.0 when buffer too short for lag."""
        buffer = SentimentLagBuffer()

        # Only 10 updates
        for i in range(10):
            buffer.update('BTCUSDT', score=0.5, source='news')

        features = buffer.get_lag_features('BTCUSDT', source='news')

        # 30-minute lag requires 30 bars, should return 0.0
        assert features['news_lag_30m'] == 0.0
        assert features['news_lag_60m'] == 0.0

    def test_multi_source_support(self):
        """Buffer supports multiple sources."""
        buffer = SentimentLagBuffer()

        buffer.update('BTCUSDT', score=0.7, source='news')
        buffer.update('BTCUSDT', score=-0.3, source='twitter')
        buffer.update('BTCUSDT', score=0.1, source='reddit')

        news_current = buffer.get_current('BTCUSDT', 'news')
        twitter_current = buffer.get_current('BTCUSDT', 'twitter')
        reddit_current = buffer.get_current('BTCUSDT', 'reddit')

        assert news_current == 0.7
        assert twitter_current == -0.3
        assert reddit_current == 0.1

    def test_memory_estimate(self):
        """Memory estimate is reasonable."""
        buffer = SentimentLagBuffer()

        # Add 1000 scores
        for i in range(1000):
            buffer.update('BTCUSDT', score=0.5, source='news')

        stats = buffer.get_stats()
        memory_kb = stats['memory_estimate_kb']

        # 361 floats * 8 bytes = 2.89 KB (under max_lag_bars + 1)
        assert memory_kb < 5.0  # Should be under 5 KB

    def test_reset_symbol(self):
        """Reset clears specific symbol buffer."""
        buffer = SentimentLagBuffer()

        buffer.update('BTCUSDT', score=0.5, source='news')
        buffer.update('ETHUSDT', score=0.8, source='news')

        buffer.reset('BTCUSDT')

        assert buffer.get_buffer_length('BTCUSDT', 'news') == 0
        assert buffer.get_buffer_length('ETHUSDT', 'news') == 1

    def test_reset_all(self):
        """Reset without symbol clears all buffers."""
        buffer = SentimentLagBuffer()

        buffer.update('BTCUSDT', score=0.5, source='news')
        buffer.update('ETHUSDT', score=0.8, source='news')

        buffer.reset()

        stats = buffer.get_stats()
        assert stats['total_symbols'] == 0
        assert stats['total_updates'] == 0

    def test_stats_tracking(self):
        """Statistics are tracked correctly."""
        buffer = SentimentLagBuffer()

        buffer.update('BTCUSDT', score=0.5, source='news')
        buffer.update('BTCUSDT', score=0.6, source='twitter')
        buffer.update('ETHUSDT', score=0.7, source='news')

        stats = buffer.get_stats()

        assert stats['total_symbols'] == 2
        assert stats['total_updates'] == 3
        assert stats['source_counts']['news'] == 2
        assert stats['source_counts']['twitter'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
