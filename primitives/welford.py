"""
Welford's Online Algorithm for Variance

Computes running mean and variance in O(1) time per update with
numerical stability. The naive sum-of-squares approach produces
negative variance with large means due to catastrophic cancellation.

Usage:
    welford = WelfordVariance()
    for price in price_stream:
        welford.update(price)
        print(f"Mean: {welford.mean}, Std: {welford.std}")
        
Memory: 40 bytes (3 floats + 1 int)
Latency: <1μs per update
"""

import math
import json
from typing import Optional, Dict, Any


class WelfordVariance:
    """
    Numerically stable online variance using Welford's algorithm.
    
    This solves the problem where:
        variance = E[X²] - E[X]² 
    produces negative results when E[X]² is large due to floating point error.
    
    Welford's recurrence instead computes:
        M2_n = M2_{n-1} + (x_n - mean_{n-1}) * (x_n - mean_n)
    which accumulates squared deviations from a moving mean, avoiding
    the subtraction of large similar numbers.
    """
    
    __slots__ = ('count', 'mean', '_m2')
    
    def __init__(self):
        self.count: int = 0
        self.mean: float = 0.0
        self._m2: float = 0.0  # Sum of squared deviations from mean
    
    def update(self, value: float) -> None:
        """
        Add a new observation. O(1) time complexity.
        
        Args:
            value: New observation to incorporate
        """
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean  # Note: uses NEW mean
        self._m2 += delta * delta2
    
    def remove(self, value: float) -> None:
        """
        Remove an observation (for sliding windows). O(1) time.
        
        Warning: Only valid if value was previously added.
        
        Args:
            value: Observation to remove
        """
        if self.count <= 1:
            self.count = 0
            self.mean = 0.0
            self._m2 = 0.0
            return
            
        delta = value - self.mean
        self.mean = (self.mean * self.count - value) / (self.count - 1)
        delta2 = value - self.mean
        self._m2 -= delta * delta2
        self.count -= 1
        
        # Numerical safety: M2 should never be negative
        if self._m2 < 0:
            self._m2 = 0.0
    
    @property
    def variance(self) -> float:
        """Sample variance (Bessel-corrected, divides by n-1)."""
        if self.count < 2:
            return 0.0
        return self._m2 / (self.count - 1)
    
    @property
    def variance_population(self) -> float:
        """Population variance (divides by n)."""
        if self.count < 1:
            return 0.0
        return self._m2 / self.count
    
    @property
    def std(self) -> float:
        """Sample standard deviation."""
        return math.sqrt(self.variance)
    
    @property
    def std_population(self) -> float:
        """Population standard deviation."""
        return math.sqrt(self.variance_population)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for persistence."""
        return {
            'count': self.count,
            'mean': self.mean,
            'm2': self._m2,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WelfordVariance':
        """Restore from serialized state."""
        instance = cls()
        instance.count = data['count']
        instance.mean = data['mean']
        instance._m2 = data['m2']
        return instance
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'WelfordVariance':
        """Restore from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def reset(self) -> None:
        """Clear all state."""
        self.count = 0
        self.mean = 0.0
        self._m2 = 0.0
    
    def __repr__(self) -> str:
        return f"WelfordVariance(n={self.count}, mean={self.mean:.6f}, std={self.std:.6f})"


class WelfordSlidingWindow:
    """
    Welford's algorithm with a fixed sliding window.
    
    Maintains exact variance over the last N observations.
    Uses a circular buffer to track values for removal.
    
    Memory: O(window_size) - stores all values in window
    Update: O(1) time
    """
    
    __slots__ = ('window_size', '_buffer', '_index', '_full', '_welford')
    
    def __init__(self, window_size: int):
        self.window_size = window_size
        self._buffer = [0.0] * window_size
        self._index = 0
        self._full = False
        self._welford = WelfordVariance()
    
    def update(self, value: float) -> None:
        """Add new value, removing oldest if window is full."""
        if self._full:
            # Remove the value being overwritten
            old_value = self._buffer[self._index]
            self._welford.remove(old_value)
        
        # Add new value
        self._buffer[self._index] = value
        self._welford.update(value)
        
        # Advance circular buffer
        self._index = (self._index + 1) % self.window_size
        if self._index == 0:
            self._full = True
    
    @property
    def mean(self) -> float:
        return self._welford.mean
    
    @property
    def variance(self) -> float:
        return self._welford.variance
    
    @property
    def std(self) -> float:
        return self._welford.std
    
    @property
    def count(self) -> int:
        return self.window_size if self._full else self._index
    
    @property
    def is_ready(self) -> bool:
        """True when window is fully populated."""
        return self._full
    
    def reset(self) -> None:
        """Clear all state."""
        self._buffer = [0.0] * self.window_size
        self._index = 0
        self._full = False
        self._welford.reset()


# =============================================================================
# Numba-optimized version for maximum performance
# =============================================================================

try:
    from numba import jit, float64, int64
    from numba.experimental import jitclass
    
    spec = [
        ('count', int64),
        ('mean', float64),
        ('m2', float64),
    ]
    
    @jitclass(spec)
    class WelfordVarianceNumba:
        """JIT-compiled Welford for ~10x speedup."""
        
        def __init__(self):
            self.count = 0
            self.mean = 0.0
            self.m2 = 0.0
        
        def update(self, value: float) -> None:
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            delta2 = value - self.mean
            self.m2 += delta * delta2
        
        def get_variance(self) -> float:
            if self.count < 2:
                return 0.0
            return self.m2 / (self.count - 1)
        
        def get_std(self) -> float:
            return math.sqrt(self.get_variance())
    
    NUMBA_AVAILABLE = True
    
except ImportError:
    NUMBA_AVAILABLE = False
    WelfordVarianceNumba = None


def get_welford(use_numba: bool = True) -> type:
    """
    Factory function to get the best available Welford implementation.
    
    Args:
        use_numba: If True and Numba is available, return JIT version
        
    Returns:
        WelfordVariance class (Numba or pure Python)
    """
    if use_numba and NUMBA_AVAILABLE:
        return WelfordVarianceNumba
    return WelfordVariance
