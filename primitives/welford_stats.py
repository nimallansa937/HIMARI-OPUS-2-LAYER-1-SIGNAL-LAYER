"""
Welford's Algorithm for Numerically Stable Online Statistics

Computes running mean, variance, and standard deviation in O(1) time per update
with O(1) memory, compared to O(n) for naive rolling windows.

Memory savings:
- 1000-bar rolling window: 8KB per symbol
- Welford state: 24 bytes per symbol
- For 100 symbols: 800KB vs 2.4KB (333x reduction)

Reference: B.P. Welford (1962), "Note on a method for calculating
corrected sums of squares and products"
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class WelfordState:
    """State container for Welford's algorithm."""
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0  # Sum of squared differences from mean


class WelfordOnlineStats:
    """
    Welford's algorithm for numerically stable online statistics.
    
    The algorithm is stable for arbitrarily large n and doesn't suffer from
    catastrophic cancellation that affects naive variance computation.
    
    Example:
        stats = WelfordOnlineStats()
        for price in price_stream:
            ret = (price - prev_price) / prev_price
            stats.update(ret)
            z_score = stats.z_score(ret)
            print(f"Return: {ret:.4f}, Z-Score: {z_score:.2f}")
    """
    
    def __init__(self, min_samples: int = 20):
        """
        Args:
            min_samples: Minimum observations before returning valid statistics
        """
        self.state = WelfordState()
        self.min_samples = min_samples
    
    def update(self, value: float) -> None:
        """
        Incorporate new observation using Welford's update equations.
        
        The key insight is that we can update mean and variance incrementally:
        
        new_mean = old_mean + (x - old_mean) / n
        new_M2 = old_M2 + (x - old_mean) * (x - new_mean)
        
        Args:
            value: New observation to incorporate
        
        Complexity: O(1) time, O(1) space
        """
        self.state.n += 1
        delta = value - self.state.mean
        self.state.mean += delta / self.state.n
        delta2 = value - self.state.mean  # Use updated mean
        self.state.M2 += delta * delta2
    
    def get_mean(self) -> Optional[float]:
        """Get current mean estimate."""
        if self.state.n < self.min_samples:
            return None
        return self.state.mean
    
    def get_variance(self) -> Optional[float]:
        """
        Get current variance estimate (sample variance with n-1 denominator).
        
        Returns None if insufficient samples.
        """
        if self.state.n < self.min_samples:
            return None
        if self.state.n < 2:
            return 0.0
        return self.state.M2 / (self.state.n - 1)
    
    def get_std(self) -> Optional[float]:
        """Get current standard deviation estimate."""
        var = self.get_variance()
        if var is None:
            return None
        return math.sqrt(max(var, 0))  # Protect against tiny negative from float errors
    
    def z_score(self, value: float) -> Optional[float]:
        """
        Compute z-score for given value using current statistics.
        
        Z-score = (value - mean) / std
        
        Args:
            value: Value to normalize
        
        Returns:
            Z-score or None if statistics not yet available
        """
        mean = self.get_mean()
        std = self.get_std()
        
        if mean is None or std is None:
            return None
        
        if std < 1e-10:
            return 0.0  # Constant series, no deviation
        
        return (value - mean) / std
    
    def get_stats(self) -> dict:
        """Return current statistics."""
        return {
            'n': self.state.n,
            'mean': self.get_mean(),
            'variance': self.get_variance(),
            'std': self.get_std(),
            'sufficient_data': self.state.n >= self.min_samples
        }
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.state = WelfordState()


class MultiSymbolWelford:
    """
    Welford statistics manager for multiple symbols.
    
    Maintains independent Welford state for each symbol, enabling
    efficient z-score normalization across a large symbol universe.
    
    Example:
        stats = MultiSymbolWelford()
        for symbol, ret in returns_stream:
            z = stats.update_and_zscore(symbol, ret)
            if z is not None and abs(z) > 2:
                print(f"{symbol} extreme move: z={z:.2f}")
    """
    
    def __init__(self, min_samples: int = 20):
        self.min_samples = min_samples
        self.symbol_stats: Dict[str, WelfordOnlineStats] = {}
    
    def update(self, symbol: str, value: float) -> None:
        """Update statistics for symbol."""
        if symbol not in self.symbol_stats:
            self.symbol_stats[symbol] = WelfordOnlineStats(self.min_samples)
        self.symbol_stats[symbol].update(value)
    
    def get_zscore(self, symbol: str, value: float) -> Optional[float]:
        """Get z-score for value using symbol's statistics."""
        if symbol not in self.symbol_stats:
            return None
        return self.symbol_stats[symbol].z_score(value)
    
    def update_and_zscore(self, symbol: str, value: float) -> Optional[float]:
        """Update statistics and return z-score in one call."""
        self.update(symbol, value)
        return self.get_zscore(symbol, value)
    
    def get_all_stats(self) -> Dict[str, dict]:
        """Get statistics for all symbols."""
        return {sym: stats.get_stats() for sym, stats in self.symbol_stats.items()}
