"""
Volume Microstructure Analysis for HIMARI L1

Implements:
- Synthetic Volume Delta (intrabar buy/sell estimation)
- Cumulative Volume Delta (CVD)
- Relative Volume at Time (RVOL)
- Order Book Imbalance approximation

Memory: ~200 bytes per indicator
Latency: <0.1ms per update

Usage:
    vol_delta = SyntheticVolumeDelta()
    rvol = RelativeVolume()
    
    for candle in candles:
        delta = vol_delta.update(candle.open, candle.high, 
                                 candle.low, candle.close, candle.volume)
        rvol_zscore = rvol.update(candle.volume, candle.timestamp)
"""

import math
import json
import numpy as np
from typing import Dict, Any, Tuple, Optional
from collections import defaultdict


class SyntheticVolumeDelta:
    """
    Estimate buying vs selling volume from OHLCV data.
    
    Theory: Close position relative to the bar range indicates
    whether buyers or sellers dominated:
    - Close near high = more buying
    - Close near low = more selling
    
    Formula:
        buying_ratio = [2 * (close - low) - (high - low)] / (high - low)
        volume_delta = volume * buying_ratio
    
    Accuracy: 75-85% vs true tick data (per HIMARI L1 spec)
    """
    
    __slots__ = (
        '_cumulative_delta', '_last_delta',
        '_delta_ema', '_count',
        '_ema_alpha'
    )
    
    def __init__(self, ema_period: int = 20):
        """
        Initialize volume delta tracker.
        
        Args:
            ema_period: Period for smoothing delta trend
        """
        self._cumulative_delta = 0.0
        self._last_delta = 0.0
        self._delta_ema = 0.0
        self._count = 0
        self._ema_alpha = 2.0 / (ema_period + 1)
    
    def update(
        self,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float
    ) -> float:
        """
        Compute volume delta for new candle.
        
        Args:
            open_price, high, low, close: OHLC prices
            volume: Total bar volume
            
        Returns:
            volume_delta (positive = more buying, negative = more selling)
        """
        range_size = high - low
        
        if range_size <= 0:
            delta = 0.0
        else:
            # Close position relative to range [-1, 1]
            buying_ratio = (2 * (close - low) - range_size) / range_size
            delta = volume * buying_ratio
        
        # Update cumulative
        self._cumulative_delta += delta
        self._last_delta = delta
        
        # Update EMA
        self._count += 1
        if self._count == 1:
            self._delta_ema = delta
        else:
            self._delta_ema = (
                self._ema_alpha * delta +
                (1 - self._ema_alpha) * self._delta_ema
            )
        
        return delta
    
    @property
    def delta(self) -> float:
        """Last computed volume delta."""
        return self._last_delta
    
    @property
    def cumulative_delta(self) -> float:
        """Cumulative volume delta (CVD)."""
        return self._cumulative_delta
    
    @property
    def delta_trend(self) -> float:
        """Smoothed delta trend (EMA)."""
        return self._delta_ema
    
    def cvd_price_divergence(
        self,
        price_change: float
    ) -> float:
        """
        Detect divergence between CVD and price.
        
        Divergence indicates potential exhaustion:
        - Price up + CVD down = bearish divergence
        - Price down + CVD up = bullish divergence
        
        Args:
            price_change: Recent price change direction (+/-)
            
        Returns:
            Divergence score [-1, 1]
            - Positive = bullish divergence
            - Negative = bearish divergence
        """
        if abs(price_change) < 1e-10:
            return 0.0
        
        # Normalize delta trend to [-1, 1] roughly
        # Using sign * log scale for robustness
        if abs(self._delta_ema) < 1:
            delta_signal = self._delta_ema
        else:
            delta_signal = math.copysign(
                math.log(1 + abs(self._delta_ema)),
                self._delta_ema
            )
        
        price_signal = math.copysign(1, price_change)
        
        # Divergence when signs differ
        if price_signal * delta_signal < 0:
            return -price_signal  # Return divergence direction
        else:
            return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            'cumulative_delta': self._cumulative_delta,
            'last_delta': self._last_delta,
            'delta_ema': self._delta_ema,
            'count': self._count,
            'ema_alpha': self._ema_alpha,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SyntheticVolumeDelta':
        """Restore from serialized state."""
        # Calculate period from alpha
        period = int((2 / data['ema_alpha']) - 1)
        instance = cls(ema_period=period)
        instance._cumulative_delta = data['cumulative_delta']
        instance._last_delta = data['last_delta']
        instance._delta_ema = data['delta_ema']
        instance._count = data['count']
        return instance
    
    def reset(self) -> None:
        """Clear all state."""
        self._cumulative_delta = 0.0
        self._last_delta = 0.0
        self._delta_ema = 0.0
        self._count = 0
    
    def __repr__(self) -> str:
        return (
            f"SyntheticVolumeDelta(delta={self._last_delta:.2f}, "
            f"CVD={self._cumulative_delta:.2f})"
        )


class RelativeVolume:
    """
    Relative Volume at Time (RVOL) - normalized volume by time-of-day.
    
    Filters out the natural "U-shape" intraday volume pattern.
    RVOL > 2σ indicates institutional activity.
    
    Uses hash map to track volume statistics per minute-of-day.
    Memory: O(minutes in day) = ~1440 * 24 bytes = 35KB
    """
    
    __slots__ = (
        '_volume_stats',  # Dict of minute -> WelfordVariance
        '_current_zscore', '_count'
    )
    
    def __init__(self):
        """Initialize RVOL tracker."""
        self._volume_stats: Dict[int, Dict[str, float]] = defaultdict(
            lambda: {'count': 0, 'mean': 0.0, 'm2': 0.0}
        )
        self._current_zscore = 0.0
        self._count = 0
    
    def update(
        self,
        volume: float,
        timestamp: int
    ) -> float:
        """
        Update RVOL with new volume observation.
        
        Args:
            volume: Bar volume
            timestamp: Unix timestamp in milliseconds
            
        Returns:
            RVOL z-score (how many std devs from normal for this time)
        """
        # Get minute of day (0-1439)
        minute_of_day = self._get_minute_of_day(timestamp)
        
        stats = self._volume_stats[minute_of_day]
        
        # Welford update for this minute
        stats['count'] += 1
        delta = volume - stats['mean']
        stats['mean'] += delta / stats['count']
        delta2 = volume - stats['mean']
        stats['m2'] += delta * delta2
        
        # Compute z-score
        if stats['count'] > 1:
            variance = stats['m2'] / (stats['count'] - 1)
            std = math.sqrt(max(variance, 0))
            
            if std > 0:
                self._current_zscore = (volume - stats['mean']) / std
            else:
                self._current_zscore = 0.0
        else:
            self._current_zscore = 0.0
        
        self._count += 1
        return self._current_zscore
    
    def _get_minute_of_day(self, timestamp_ms: int) -> int:
        """Convert timestamp to minute of day (0-1439)."""
        # Convert to seconds, then to minute of day
        seconds_since_epoch = timestamp_ms // 1000
        seconds_of_day = seconds_since_epoch % 86400
        minute_of_day = seconds_of_day // 60
        return minute_of_day
    
    @property
    def zscore(self) -> float:
        """Current RVOL z-score."""
        return self._current_zscore
    
    @property
    def percentile(self) -> float:
        """
        Current RVOL as percentile (0-100).
        
        Uses normal CDF approximation.
        """
        from math import erfc
        # Standard normal CDF
        cdf = 0.5 * erfc(-self._current_zscore / math.sqrt(2))
        return cdf * 100
    
    def is_unusual_volume(self, threshold: float = 2.0) -> bool:
        """Check if current volume is unusually high."""
        return abs(self._current_zscore) > threshold
    
    def get_interpretation(self) -> str:
        """Human-readable interpretation of current RVOL."""
        z = self._current_zscore
        
        if z > 3:
            return 'extreme_high'
        elif z > 2:
            return 'high'
        elif z > 1:
            return 'above_average'
        elif z > -1:
            return 'normal'
        elif z > -2:
            return 'below_average'
        else:
            return 'low'
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        # Convert defaultdict to regular dict
        stats_dict = {k: v for k, v in self._volume_stats.items()}
        return {
            'volume_stats': stats_dict,
            'current_zscore': self._current_zscore,
            'count': self._count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RelativeVolume':
        """Restore from serialized state."""
        instance = cls()
        for k, v in data['volume_stats'].items():
            instance._volume_stats[int(k)] = v
        instance._current_zscore = data['current_zscore']
        instance._count = data['count']
        return instance
    
    def reset(self) -> None:
        """Clear all state."""
        self._volume_stats.clear()
        self._current_zscore = 0.0
        self._count = 0


class VolumeOrderBookImbalance:
    """
    LEGACY: Synthetic Order Book Imbalance from OHLCV data.
    
    NOTE: This is the OLD implementation. Use primitives.order_book_imbalance.OrderBookImbalance
    for the NEW academic-validated version with true Level 2 data.
    
    Approximates OBI without Level 2 data by using close position
    relative to the bar range as a proxy for order flow direction.
    
    Academically validated: Gould & Bonart (2015), Cont et al. (2014)
    
    Note: OHLCV approximation has unknown error rates.
    Use with caution and validate against true OBI when possible.
    """
    
    __slots__ = (
        '_last_obi', '_obi_ema',
        '_count', '_ema_alpha'
    )
    
    def __init__(self, ema_period: int = 10):
        """
        Initialize OBI tracker.
        
        Args:
            ema_period: Period for smoothing OBI
        """
        self._last_obi = 0.0
        self._obi_ema = 0.0
        self._count = 0
        self._ema_alpha = 2.0 / (ema_period + 1)
    
    def update(
        self,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: Optional[float] = None
    ) -> float:
        """
        Compute synthetic OBI from OHLCV.
        
        Args:
            open_price, high, low, close: OHLC prices
            volume: Optional volume (not used in basic calculation)
            
        Returns:
            OBI in [-1, 1] where positive = buying pressure
        """
        range_size = high - low
        
        if range_size <= 0:
            obi = 0.0
        else:
            # Close position relative to range: 0 = at low, 1 = at high
            close_position = (close - low) / range_size
            # Transform to [-1, 1]
            obi = 2 * close_position - 1
        
        self._last_obi = obi
        self._count += 1
        
        # Update EMA
        if self._count == 1:
            self._obi_ema = obi
        else:
            self._obi_ema = (
                self._ema_alpha * obi +
                (1 - self._ema_alpha) * self._obi_ema
            )
        
        return obi
    
    @property
    def imbalance(self) -> float:
        """Current OBI value [-1, 1]."""
        return self._last_obi
    
    @property
    def imbalance_trend(self) -> float:
        """Smoothed OBI trend."""
        return self._obi_ema
    
    def get_pressure_direction(self) -> str:
        """Get buying/selling pressure direction."""
        if self._obi_ema > 0.2:
            return 'buying'
        elif self._obi_ema < -0.2:
            return 'selling'
        else:
            return 'neutral'
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            'last_obi': self._last_obi,
            'obi_ema': self._obi_ema,
            'count': self._count,
            'ema_alpha': self._ema_alpha,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VolumeOrderBookImbalance':
        """Restore from serialized state."""
        period = int((2 / data['ema_alpha']) - 1)
        instance = cls(ema_period=period)
        instance._last_obi = data['last_obi']
        instance._obi_ema = data['obi_ema']
        instance._count = data['count']
        return instance
    
    def reset(self) -> None:
        """Clear all state."""
        self._last_obi = 0.0
        self._obi_ema = 0.0
        self._count = 0


# Backwards compatibility alias
OrderBookImbalance = VolumeOrderBookImbalance
