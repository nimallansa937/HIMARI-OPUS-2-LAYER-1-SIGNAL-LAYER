"""
HIMARI L1 Primitives - O(1) Streaming Algorithms

All algorithms in this module operate in constant time per update,
making them suitable for <10ms latency requirements.

Tier 5 Components (Foundation):
- WelfordVariance: Numerically stable online variance
- KalmanFilter: Adaptive trend estimation
- UltimateSmoother: Zero-lag trend filter (Ehlers 2024)
- RecursiveLeastSquares: O(1) regression channels
- OnlineGARCH: Volatility forecasting
- StreamingQuantiles: T-Digest quantile estimation
- OnlineCovariance: Cross-asset correlation tracking
- MovingHurst: Regime detection (replaces Choppiness Index)
- SyntheticVolumeDelta: Volume microstructure

Tier 4 Components (DSP):
- SampleEntropy: Complexity-based regime detection
"""

from .welford import WelfordVariance, WelfordSlidingWindow
from .kalman import KalmanFilter, AdaptiveKalmanFilter, KalmanWithVelocity
from .ultimate_smoother import UltimateSmoother, SuperSmoother
from .rls import RecursiveLeastSquares, RegressionChannel
from .garch import OnlineGARCH, AdaptiveGARCH, OnlineEGARCH
from .tdigest_quantiles import StreamingQuantiles, VolumeProfile
from .covariance import OnlineCovariance, ExponentialCovariance, CorrelationMatrix
from .hurst import MovingHurst, SampleEntropy
from .volume import SyntheticVolumeDelta, RelativeVolume, OrderBookImbalance

__all__ = [
    # Tier 5: Core primitives
    'WelfordVariance',
    'WelfordSlidingWindow',
    'KalmanFilter',
    'AdaptiveKalmanFilter',
    'KalmanWithVelocity',
    'UltimateSmoother',
    'SuperSmoother',
    'RecursiveLeastSquares',
    'RegressionChannel',
    'OnlineGARCH',
    'AdaptiveGARCH',
    'OnlineEGARCH',
    'StreamingQuantiles',
    'VolumeProfile',
    'OnlineCovariance',
    'ExponentialCovariance',
    'CorrelationMatrix',
    # Tier 4: DSP/Regime
    'MovingHurst',
    'SampleEntropy',
    # Tier 2: Volume microstructure
    'SyntheticVolumeDelta',
    'RelativeVolume',
    'OrderBookImbalance',
]
