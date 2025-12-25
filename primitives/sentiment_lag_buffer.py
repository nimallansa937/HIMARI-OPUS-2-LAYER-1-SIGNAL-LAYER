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
