"""
Sentiment Lag Buffer - Temporal Lag Feature Engineering

Maintains rolling buffer of sentiment scores across multiple time horizons.

Enhancement 1 from ANTIGRAVITY_SENTIMENT_ENHANCEMENT_GUIDE.md
Evidence: 0.806 Pearson correlation with 3-hour lagged sentiment
Impact: +2-5% Sharpe improvement
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class LagConfig:
    """Configuration for sentiment lag features."""
    max_lag_bars: int = 360  # 6 hours at 1-minute bars
    bar_interval_minutes: int = 1
    lag_horizons: Dict[str, List[int]] = field(default_factory=lambda: {
        'news': [30, 60, 90, 120, 180],      # 30min to 3h
        'twitter': [15, 30, 45, 60, 120],    # 15min to 2h  
        'reddit': [180, 240, 270, 300, 360]  # 3h to 6h
    })


class SentimentLagBuffer:
    """
    Maintain rolling buffer of sentiment scores for lag feature engineering.
    
    Research shows optimal sentiment lags:
    - News: 0-3 hour lag (strongest at 1.5 hours)
    - Twitter: 15min-2 hour lag (strongest at 45 minutes)
    - Reddit: 3-6 hour lag (strongest at 4.5 hours)
    
    Example:
        buffer = SentimentLagBuffer()
        
        # Update with new sentiment
        buffer.update('BTCUSDT', score=0.65, source='news')
        
        # Get lag features
        features = buffer.get_lag_features('BTCUSDT')
        # {'news_lag_30m': 0.42, 'news_lag_60m': 0.38, ...}
    """
    
    def __init__(self, config: Optional[LagConfig] = None):
        """
        Initialize sentiment lag buffer.
        
        Args:
            config: Lag configuration settings
        """
        self.config = config or LagConfig()
        
        # Per-symbol, per-source buffers
        # {symbol: {source: deque}}
        self._buffers: Dict[str, Dict[str, deque]] = {}
        
        # Track update counts for monitoring
        self._update_counts: Dict[str, int] = {}
        
        logger.info(
            f"SentimentLagBuffer initialized: "
            f"max_lag={self.config.max_lag_bars} bars, "
            f"interval={self.config.bar_interval_minutes}min"
        )
    
    def update(
        self, 
        symbol: str, 
        score: float, 
        source: str = 'news'
    ) -> None:
        """
        Add new sentiment score to buffer.
        
        Args:
            symbol: Trading symbol
            score: Sentiment score (-1 to +1)
            source: Source type ('news', 'twitter', 'reddit')
        """
        # Initialize symbol buffers if needed
        if symbol not in self._buffers:
            self._buffers[symbol] = {}
            self._update_counts[symbol] = 0
        
        # Initialize source buffer if needed
        if source not in self._buffers[symbol]:
            self._buffers[symbol][source] = deque(maxlen=self.config.max_lag_bars + 1)
        
        # Add score
        self._buffers[symbol][source].append(score)
        self._update_counts[symbol] += 1
    
    def get_lag_features(
        self, 
        symbol: str,
        source: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Get lag features for a symbol.
        
        Args:
            symbol: Trading symbol
            source: Specific source, or None for all sources
            
        Returns:
            Dict of lag feature names to values
        """
        features = {}
        
        if symbol not in self._buffers:
            return features
        
        sources = [source] if source else self.config.lag_horizons.keys()
        
        for src in sources:
            if src not in self._buffers.get(symbol, {}):
                continue
            
            horizons = self.config.lag_horizons.get(src, [])
            buffer = self._buffers[symbol][src]
            
            for lag_minutes in horizons:
                feature_name = f"{src}_lag_{lag_minutes}m"
                lag_index = -(lag_minutes // self.config.bar_interval_minutes)
                
                # Get lagged value if buffer is long enough
                if abs(lag_index) < len(buffer):
                    features[feature_name] = buffer[lag_index]
                else:
                    features[feature_name] = 0.0
        
        # Add aggregation features
        features.update(self._compute_aggregation_features(symbol))
        
        return features
    
    def _compute_aggregation_features(self, symbol: str) -> Dict[str, float]:
        """Compute derived aggregation features."""
        features = {}
        
        if symbol not in self._buffers:
            return features
        
        # Sentiment acceleration (current - 30m ago) / 30
        for source, buffer in self._buffers[symbol].items():
            if len(buffer) >= 31:
                current = buffer[-1]
                lag_30 = buffer[-31]
                features[f'{source}_acceleration'] = (current - lag_30) / 30
        
        # Sentiment momentum (sum of last 180 bars if available)
        for source, buffer in self._buffers[symbol].items():
            if len(buffer) >= 180:
                momentum = sum(list(buffer)[-180:])
                features[f'{source}_momentum_3h'] = momentum
        
        # Sentiment reversal flag (sign flip in last hour)
        for source, buffer in self._buffers[symbol].items():
            if len(buffer) >= 60:
                recent = list(buffer)[-60:]
                sign_changes = sum(
                    1 for i in range(1, len(recent)) 
                    if (recent[i] > 0) != (recent[i-1] > 0)
                )
                features[f'{source}_reversal_count'] = sign_changes
                features[f'{source}_reversal_flag'] = 1.0 if sign_changes > 0 else 0.0
        
        return features
    
    def get_current(self, symbol: str, source: str = 'news') -> Optional[float]:
        """Get current (most recent) sentiment score."""
        if symbol not in self._buffers:
            return None
        if source not in self._buffers[symbol]:
            return None
        
        buffer = self._buffers[symbol][source]
        return buffer[-1] if buffer else None
    
    def get_buffer_length(self, symbol: str, source: str = 'news') -> int:
        """Get current buffer length for a symbol/source."""
        if symbol not in self._buffers:
            return 0
        if source not in self._buffers[symbol]:
            return 0
        return len(self._buffers[symbol][source])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        total_symbols = len(self._buffers)
        total_updates = sum(self._update_counts.values())
        
        source_counts = {}
        for symbol_buffers in self._buffers.values():
            for source in symbol_buffers:
                source_counts[source] = source_counts.get(source, 0) + 1
        
        return {
            'total_symbols': total_symbols,
            'total_updates': total_updates,
            'source_counts': source_counts,
            'max_lag_bars': self.config.max_lag_bars,
            'memory_estimate_kb': self._estimate_memory_kb()
        }
    
    def _estimate_memory_kb(self) -> float:
        """Estimate memory usage in KB."""
        # Each float is ~8 bytes, plus deque overhead
        total_floats = 0
        for symbol_buffers in self._buffers.values():
            for buffer in symbol_buffers.values():
                total_floats += len(buffer)
        
        return (total_floats * 8) / 1024  # Convert to KB
    
    def reset(self, symbol: Optional[str] = None) -> None:
        """Reset buffers for a symbol or all symbols."""
        if symbol:
            if symbol in self._buffers:
                self._buffers[symbol].clear()
                self._update_counts[symbol] = 0
        else:
            self._buffers.clear()
            self._update_counts.clear()

    # =========================================================================
    # RESEARCH-BACKED ENHANCEMENTS
    # =========================================================================
    
    def get_decayed_sentiment(
        self, 
        symbol: str, 
        source: str = 'news',
        decay_half_life_minutes: float = 120.0
    ) -> float:
        """
        Apply exponential decay to historical sentiment (Enhancement #1).
        
        Research shows older news decays exponentially with ~2-4h half-life for crypto.
        This gives recent sentiment more weight than older sentiment.
        
        Args:
            symbol: Trading symbol
            source: Sentiment source
            decay_half_life_minutes: Half-life in minutes (default 2h)
        
        Returns:
            Decay-weighted sentiment score (-1 to +1)
        """
        if symbol not in self._buffers or source not in self._buffers[symbol]:
            return 0.0
        
        buffer = list(self._buffers[symbol][source])
        if not buffer:
            return 0.0
        
        weighted_sum = 0.0
        weight_sum = 0.0
        
        # Most recent is at end of buffer, oldest at start
        for i, sentiment in enumerate(reversed(buffer)):
            age_minutes = i * self.config.bar_interval_minutes
            weight = 2 ** (-age_minutes / decay_half_life_minutes)
            weighted_sum += sentiment * weight
            weight_sum += weight
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    def get_sentiment_momentum(
        self, 
        symbol: str, 
        source: str = 'news',
        lookback_minutes: int = 60
    ) -> Dict[str, float]:
        """
        Calculate sentiment momentum (rate of change) (Enhancement #6).
        
        Research shows sentiment change rate predicts reversals.
        
        Args:
            symbol: Trading symbol
            source: Sentiment source
            lookback_minutes: Lookback window
        
        Returns:
            Dict with momentum metrics: rate, acceleration, velocity
        """
        if symbol not in self._buffers or source not in self._buffers[symbol]:
            return {'rate': 0.0, 'acceleration': 0.0, 'velocity': 0.0}
        
        buffer = list(self._buffers[symbol][source])
        lookback_bars = lookback_minutes // self.config.bar_interval_minutes
        
        if len(buffer) < lookback_bars:
            return {'rate': 0.0, 'acceleration': 0.0, 'velocity': 0.0}
        
        recent = buffer[-lookback_bars:]
        
        # Velocity: Current - Start
        velocity = recent[-1] - recent[0]
        
        # Rate: Average change per bar
        rate = velocity / lookback_bars if lookback_bars > 0 else 0.0
        
        # Acceleration: Change in velocity (2nd derivative)
        mid_point = len(recent) // 2
        first_half_vel = recent[mid_point] - recent[0]
        second_half_vel = recent[-1] - recent[mid_point]
        acceleration = (second_half_vel - first_half_vel) / mid_point if mid_point > 0 else 0.0
        
        return {
            'rate': rate,
            'acceleration': acceleration,
            'velocity': velocity
        }
    
    def get_time_of_day_weight(self, hour_utc: int) -> float:
        """
        Get time-of-day weight for sentiment impact (Enhancement #10).
        
        Research shows news impact varies 2-3x by market session:
        - Asian session (00:00-08:00 UTC): Lower impact
        - London open (08:00-12:00 UTC): High impact
        - US open (12:00-16:00 UTC): Highest impact  
        - US afternoon (16:00-21:00 UTC): Medium impact
        - Overnight (21:00-00:00 UTC): Lower impact
        
        Args:
            hour_utc: Hour of day in UTC (0-23)
            
        Returns:
            Weight multiplier (0.5 to 1.5)
        """
        TOD_WEIGHTS = {
            0: 0.6, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.6, 5: 0.7, 6: 0.8, 7: 0.9,
            8: 1.1, 9: 1.2, 10: 1.3, 11: 1.3,  # London session
            12: 1.5, 13: 1.5, 14: 1.4, 15: 1.4, 16: 1.3,  # US session
            17: 1.2, 18: 1.1, 19: 1.0, 20: 0.9,
            21: 0.8, 22: 0.7, 23: 0.6
        }
        return TOD_WEIGHTS.get(hour_utc, 1.0)
    
    def get_cross_source_weighted_sentiment(
        self, 
        symbol: str,
        source_weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Get weighted sentiment across all sources (Enhancement #7 prep).
        
        Args:
            symbol: Trading symbol
            source_weights: Custom weights per source (default: equal)
            
        Returns:
            Weighted average sentiment
        """
        if symbol not in self._buffers:
            return 0.0
        
        default_weights = {'news': 1.0, 'twitter': 0.7, 'reddit': 0.5}
        weights = source_weights or default_weights
        
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for source, buffer in self._buffers[symbol].items():
            if buffer:
                current = buffer[-1]
                weight = weights.get(source, 0.5)
                weighted_sum += current * weight
                weight_sum += weight
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0

    def get_all_enhanced_features(
        self,
        symbol: str,
        hour_utc: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Get all enhanced sentiment features for signal generation.
        
        Combines all research-backed enhancements into single feature dict.
        
        Args:
            symbol: Trading symbol
            hour_utc: Current hour for time-of-day weighting
            
        Returns:
            Dict with all enhanced features
        """
        import time
        from datetime import datetime
        
        if hour_utc is None:
            hour_utc = datetime.utcnow().hour
        
        features = {}
        
        # Standard lag features
        features.update(self.get_lag_features(symbol))
        
        # Time-of-day weight
        features['time_of_day_weight'] = self.get_time_of_day_weight(hour_utc)
        
        # Per-source enhanced features
        for source in ['news', 'twitter', 'reddit']:
            prefix = source
            
            # Decayed sentiment (half-life varies by source)
            half_lives = {'news': 120.0, 'twitter': 60.0, 'reddit': 240.0}
            features[f'{prefix}_decayed'] = self.get_decayed_sentiment(
                symbol, source, half_lives.get(source, 120.0)
            )
            
            # Sentiment momentum
            momentum = self.get_sentiment_momentum(symbol, source)
            features[f'{prefix}_momentum_rate'] = momentum['rate']
            features[f'{prefix}_momentum_accel'] = momentum['acceleration']
            features[f'{prefix}_momentum_velocity'] = momentum['velocity']
        
        # Cross-source weighted
        features['weighted_sentiment'] = self.get_cross_source_weighted_sentiment(symbol)
        
        # Apply time-of-day weight to weighted sentiment
        features['weighted_sentiment_tod_adj'] = (
            features['weighted_sentiment'] * features['time_of_day_weight']
        )
        
        return features
