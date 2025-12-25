"""
HIMARI Layer 1 Signal Layer

Production-ready signal processing for cryptocurrency trading.
Integrates with existing HIMARI data infrastructure.

Quick Start:
    from himari_l1 import SignalProcessor
    processor = SignalProcessor()
    processor.run()

Components:
    - primitives: O(1) streaming algorithms (Welford, Kalman, RLS)
    - regime: Regime detection (HMM, GARCH, Hurst, Entropy)
    - signals: Signal generators (Momentum, Mean Reversion, Volume)
    - validation: Strategy validation (DSR, CPCV, SPA tests)
"""

# Conditional imports to support both package and script modes
try:
    from .signal_processor import SignalProcessor
    from .config import L1Config, RedisKeys
    __all__ = ['SignalProcessor', 'L1Config', 'RedisKeys']
except ImportError:
    # Running as script, not package - imports will fail
    SignalProcessor = None
    L1Config = None
    RedisKeys = None
    __all__ = []

__version__ = "1.0.0"
