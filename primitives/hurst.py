"""
Moving Hurst Exponent for Regime Detection

Replaces the academically-unvalidated Choppiness Index with
the Hurst Exponent, which has demonstrated 3-4× better returns
than Buy & Hold (JRFM 2024, ICEIS 2018).

Interpretation:
    H > 0.5: Trending market (momentum strategies work)
    H = 0.5: Random walk (no predictable pattern)
    H < 0.5: Mean-reverting (contrarian strategies work)

Memory: ~8KB for 100-period window
Latency: <0.5ms per update (computed every 10 bars for efficiency)

Usage:
    hurst = MovingHurst(window=100)
    for price in prices:
        h, regime = hurst.update(price)
        if regime == 'trending':
            # Use momentum strategy
        elif regime == 'mean_reverting':
            # Use mean-reversion strategy
"""

import math
import json
from typing import Dict, Any, Tuple, List, Optional
from collections import deque
import numpy as np


class MovingHurst:
    """
    Moving Hurst Exponent using Rescaled Range (R/S) analysis.
    
    The Hurst exponent H is estimated by fitting:
    log(R/S) = H * log(n) + c
    
    Where R/S is the rescaled range statistic.
    """
    
    __slots__ = (
        '_window', '_update_freq',
        '_prices', '_count',
        '_current_hurst', '_last_update'
    )
    
    def __init__(
        self,
        window: int = 100,
        update_frequency: int = 10
    ):
        """
        Initialize Hurst estimator.
        
        Args:
            window: Lookback period for R/S analysis
            update_frequency: Recompute Hurst every N bars (efficiency)
        """
        self._window = window
        self._update_freq = update_frequency
        self._prices = deque(maxlen=window)
        self._count = 0
        self._current_hurst = 0.5  # Start with random walk assumption
        self._last_update = 0
    
    def update(self, price: float) -> Tuple[float, str]:
        """
        Add new price and update Hurst estimate.
        
        Args:
            price: New price observation
            
        Returns:
            (hurst_value, regime_label)
        """
        self._prices.append(price)
        self._count += 1
        
        # Only recompute periodically for efficiency
        if (
            len(self._prices) >= self._window and
            self._count - self._last_update >= self._update_freq
        ):
            self._current_hurst = self._compute_hurst()
            self._last_update = self._count
        
        return self._current_hurst, self.regime
    
    def _compute_hurst(self) -> float:
        """
        Compute Hurst exponent using R/S analysis.
        
        Uses multiple sub-periods to fit the log-log relationship.
        """
        prices = np.array(self._prices)
        returns = np.diff(np.log(prices))
        n = len(returns)
        
        if n < 20:
            return 0.5
        
        # Compute R/S for different period lengths
        period_lengths = []
        rs_values = []
        
        for period in [10, 15, 20, 25, 30, 40, 50]:
            if period > n // 2:
                continue
            
            # Average R/S over non-overlapping periods
            num_periods = n // period
            if num_periods < 2:
                continue
            
            rs_sum = 0.0
            for i in range(num_periods):
                start = i * period
                end = start + period
                segment = returns[start:end]
                
                rs = self._rescaled_range(segment)
                if rs > 0:
                    rs_sum += rs
            
            avg_rs = rs_sum / num_periods
            if avg_rs > 0:
                period_lengths.append(period)
                rs_values.append(avg_rs)
        
        if len(period_lengths) < 3:
            return 0.5
        
        # Linear regression: log(R/S) = H * log(n) + c
        log_n = np.log(period_lengths)
        log_rs = np.log(rs_values)
        
        # Least squares fit
        n_pts = len(log_n)
        sum_x = np.sum(log_n)
        sum_y = np.sum(log_rs)
        sum_xy = np.sum(log_n * log_rs)
        sum_x2 = np.sum(log_n ** 2)
        
        denom = n_pts * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-10:
            return 0.5
        
        hurst = (n_pts * sum_xy - sum_x * sum_y) / denom
        
        # Clamp to valid range [0, 1]
        return max(0.0, min(1.0, hurst))
    
    def _rescaled_range(self, series: np.ndarray) -> float:
        """
        Compute R/S statistic for a series.
        
        R/S = (max(cumsum) - min(cumsum)) / std(series)
        """
        n = len(series)
        if n < 2:
            return 0.0
        
        # Mean-centered cumulative sum
        mean = np.mean(series)
        centered = series - mean
        cumsum = np.cumsum(centered)
        
        # Range
        r = np.max(cumsum) - np.min(cumsum)
        
        # Standard deviation
        s = np.std(series, ddof=1)
        
        if s < 1e-10:
            return 0.0
        
        return r / s
    
    @property
    def hurst(self) -> float:
        """Current Hurst exponent estimate."""
        return self._current_hurst
    
    @property
    def regime(self) -> str:
        """
        Current regime classification.
        
        Returns:
            'trending', 'random_walk', or 'mean_reverting'
        """
        h = self._current_hurst
        
        if h > 0.55:
            return 'trending'
        elif h < 0.45:
            return 'mean_reverting'
        else:
            return 'random_walk'
    
    def get_momentum_weight(self) -> float:
        """
        Weight for momentum strategies [0, 1].
        
        High Hurst = high momentum weight
        """
        # Map H from [0.5, 1.0] to [0, 1]
        return max(0, min(1, (self._current_hurst - 0.5) * 2))
    
    def get_mean_reversion_weight(self) -> float:
        """
        Weight for mean-reversion strategies [0, 1].
        
        Low Hurst = high mean-reversion weight
        """
        # Map H from [0.0, 0.5] to [1, 0]
        return max(0, min(1, (0.5 - self._current_hurst) * 2))
    
    @property
    def is_ready(self) -> bool:
        """True when window is fully populated."""
        return len(self._prices) >= self._window
    
    @property
    def count(self) -> int:
        """Number of observations processed."""
        return self._count
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for persistence."""
        return {
            'window': self._window,
            'update_freq': self._update_freq,
            'prices': list(self._prices),
            'count': self._count,
            'hurst': self._current_hurst,
            'last_update': self._last_update,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MovingHurst':
        """Restore from serialized state."""
        instance = cls(
            window=data['window'],
            update_frequency=data['update_freq']
        )
        for p in data['prices']:
            instance._prices.append(p)
        instance._count = data['count']
        instance._current_hurst = data['hurst']
        instance._last_update = data['last_update']
        return instance
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MovingHurst':
        """Restore from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def reset(self) -> None:
        """Clear all state."""
        self._prices.clear()
        self._count = 0
        self._current_hurst = 0.5
        self._last_update = 0
    
    def __repr__(self) -> str:
        return (
            f"MovingHurst(H={self._current_hurst:.3f}, "
            f"regime={self.regime}, n={self._count})"
        )


class SampleEntropy:
    """
    Sample Entropy (SampEn) for complexity-based regime detection.
    
    Replaces Approximate Entropy (ApEn) which has self-matching bias.
    Lower entropy = more predictable market (ML performs better).
    
    COVID finding: Crisis periods showed DECREASED entropy (more predictable).
    """
    
    __slots__ = (
        '_m', '_r_fraction',
        '_buffer', '_window',
        '_current_entropy', '_count'
    )
    
    def __init__(
        self,
        embedding_dim: int = 2,
        tolerance_fraction: float = 0.2,
        window: int = 100
    ):
        """
        Args:
            embedding_dim: m parameter (pattern length)
            tolerance_fraction: r parameter as fraction of std dev
            window: Rolling window size
        """
        self._m = embedding_dim
        self._r_fraction = tolerance_fraction
        self._buffer = deque(maxlen=window)
        self._window = window
        self._current_entropy = 0.0
        self._count = 0
    
    def update(self, value: float) -> float:
        """
        Add observation and update entropy estimate.
        
        Returns:
            Current sample entropy estimate
        """
        self._buffer.append(value)
        self._count += 1
        
        # Only compute when buffer is full
        if len(self._buffer) >= self._window:
            self._current_entropy = self._compute_sample_entropy()
        
        return self._current_entropy
    
    def _compute_sample_entropy(self) -> float:
        """Compute sample entropy using template matching."""
        data = np.array(self._buffer)
        N = len(data)
        m = self._m
        r = self._r_fraction * np.std(data)
        
        if N < m + 2 or r <= 0:
            return 0.0
        
        def count_matches(template_length: int) -> int:
            """Count pairs of matching templates."""
            templates = np.array([
                data[i:i+template_length]
                for i in range(N - template_length)
            ])
            count = 0
            n_templates = len(templates)
            
            for i in range(n_templates):
                for j in range(i + 1, n_templates):
                    if np.max(np.abs(templates[i] - templates[j])) < r:
                        count += 1
            
            return count
        
        A = count_matches(m + 1)  # Matches of length m+1
        B = count_matches(m)      # Matches of length m
        
        if B == 0 or A == 0:
            return 0.0
        
        return -math.log(A / B)
    
    @property
    def entropy(self) -> float:
        """Current sample entropy estimate."""
        return self._current_entropy
    
    @property
    def complexity_regime(self) -> str:
        """
        Classification based on entropy level.
        
        Low entropy = more predictable (ML works better)
        """
        if self._current_entropy < 0.5:
            return 'low_complexity'  # More predictable
        elif self._current_entropy > 1.5:
            return 'high_complexity'  # Less predictable
        else:
            return 'normal_complexity'
    
    def get_confidence_scalar(self) -> float:
        """
        Confidence scalar based on entropy.
        
        Low entropy -> higher confidence in predictions
        """
        # Map entropy [0, 2] to confidence [1, 0.3]
        return max(0.3, min(1.0, 1.0 - self._current_entropy * 0.35))
    
    def reset(self) -> None:
        """Clear all state."""
        self._buffer.clear()
        self._current_entropy = 0.0
        self._count = 0
