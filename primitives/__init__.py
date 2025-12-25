"""
Primitives Module - Enhanced Layer 1 Signal Components

Backwards-compatible imports for the enhanced signal generation system.
"""

from .streaming_hmm import StreamingHMM, MarketRegime, HMMConfig
from .streaming_indicators import StreamingIndicators, IndicatorConfig
from .welford_stats import WelfordOnlineStats, MultiSymbolWelford, WelfordState
from .multi_horizon_momentum import MultiHorizonMomentum, MomentumConfig
from .order_book_imbalance import OrderBookImbalance, OBIConfig
from .regime_fusion import RegimeAwareSignalFusion, FusionConfig, SignalDefinition, SignalCategory

# Lazy import for sentiment (optional - requires torch/transformers)
# Use get_hybrid_sentiment() to access when needed
try:
    from .hybrid_sentiment import HybridSentimentAnalyzer, HybridSentimentConfig, CRYPTO_LEXICON
    _SENTIMENT_AVAILABLE = True
except (ImportError, OSError) as e:
    import logging
    logging.getLogger(__name__).debug(f"Sentiment module not available: {e}")
    _SENTIMENT_AVAILABLE = False
    HybridSentimentAnalyzer = None
    HybridSentimentConfig = None
    CRYPTO_LEXICON = {}

from .integrated_signal_layer import IntegratedSignalLayer, IntegratedSignalOutput


def is_sentiment_available() -> bool:
    """Check if sentiment analysis is available."""
    return _SENTIMENT_AVAILABLE


__all__ = [
    # HMM Regime Detection
    'StreamingHMM',
    'MarketRegime',
    'HMMConfig',
    
    # Streaming Indicators
    'StreamingIndicators',
    'IndicatorConfig',
    
    # Online Statistics
    'WelfordOnlineStats',
    'MultiSymbolWelford',
    'WelfordState',
    
    # Momentum Features
    'MultiHorizonMomentum',
    'MomentumConfig',
    
    # Order Book Imbalance
    'OrderBookImbalance',
    'OBIConfig',
    
    # Signal Fusion
    'RegimeAwareSignalFusion',
    'FusionConfig',
    'SignalDefinition',
    'SignalCategory',
    
    # Hybrid Sentiment (optional - may be None if deps unavailable)
    'HybridSentimentAnalyzer',
    'HybridSentimentConfig',
    'CRYPTO_LEXICON',
    'is_sentiment_available',
    
    # Integrated Layer (Complete System)
    'IntegratedSignalLayer',
    'IntegratedSignalOutput',
]
