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

# Enhancement 1: Sentiment Lag Features
from .sentiment_lag_buffer import SentimentLagBuffer, LagConfig

# Enhancement 2: Dynamic Regime-Based Weighting
from .dynamic_sentiment_weights import (
    DynamicSentimentWeighter, 
    DynamicWeightConfig,
    VolatilityRegime,
    SocialRegime
)


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
    
    # Enhancement 1: Sentiment Lag Features
    'SentimentLagBuffer',
    'LagConfig',
    
    # Enhancement 2: Dynamic Weighting
    'DynamicSentimentWeighter',
    'DynamicWeightConfig',
    'VolatilityRegime',
    'SocialRegime',
    
    # Enhancement 5: Social Media
    'SocialSentimentAggregator',
    'SocialPost',
    'AggregatorConfig',
    
    # Enhancement 7: Multi-Asset
    'MultiAssetSentimentManager',
    'AssetConfig',
    'ModelVariant',
]

# Import new modules (lazy)
try:
    from .social_sentiment_aggregator import (
        SocialSentimentAggregator,
        SocialPost,
        AggregatorConfig
    )
except ImportError:
    SocialSentimentAggregator = None
    SocialPost = None
    AggregatorConfig = None

try:
    from .multi_asset_sentiment_manager import (
        MultiAssetSentimentManager,
        AssetConfig,
        ModelVariant
    )
except ImportError:
    MultiAssetSentimentManager = None
    AssetConfig = None
    ModelVariant = None

# Dual Path Sentiment (Fast + Accurate routing)
try:
    from .dual_path_sentiment import (
        DualPathSentimentAnalyzer,
        DualPathConfig,
        SentimentResult,
        SignalType,
        SourceType,
        create_dual_path_analyzer
    )
    _DUAL_PATH_AVAILABLE = True
except ImportError:
    _DUAL_PATH_AVAILABLE = False
    DualPathSentimentAnalyzer = None
    DualPathConfig = None
    SentimentResult = None
    SignalType = None
    SourceType = None
    create_dual_path_analyzer = None


def is_dual_path_available() -> bool:
    """Check if dual-path sentiment analyzer is available."""
    return _DUAL_PATH_AVAILABLE


# Add to __all__
__all__.extend([
    # Dual Path Sentiment
    'DualPathSentimentAnalyzer',
    'DualPathConfig',
    'SentimentResult',
    'SignalType',
    'SourceType',
    'create_dual_path_analyzer',
    'is_dual_path_available',
    
    # Research-Backed Enhancements
    'SourceCredibilityTracker',
    'EnhancedDualPathAnalyzer',
    'DisagreementSignal',
    'CrossAssetSentimentContagion',
    'SentimentVolumeDivergence',
    'compute_disagreement',
    'compute_sentiment_volume_divergence',
])

# Import additional enhancement classes
try:
    from .dual_path_sentiment import (
        DisagreementSignal,
        CrossAssetSentimentContagion,
        SentimentVolumeDivergence,
        EnhancedDualPathAnalyzer,
        compute_disagreement,
        compute_sentiment_volume_divergence,
    )
except ImportError:
    DisagreementSignal = None
    CrossAssetSentimentContagion = None
    SentimentVolumeDivergence = None
    EnhancedDualPathAnalyzer = None
    compute_disagreement = None
    compute_sentiment_volume_divergence = None

try:
    from .dynamic_sentiment_weights import SourceCredibilityTracker
except ImportError:
    SourceCredibilityTracker = None

# Multi-Model Sentiment Ensemble (Phase 1-3)
try:
    from .multi_model_sentiment import (
        MultiModelSentimentAnalyzer,
        EnsembleConfig,
        EnsembleResult,
        ModelPrediction,
        ModelType,
        ModelConfig,
        MODEL_REGISTRY,
        LABEL_MAPS,
        create_phase1_analyzer,
        create_phase2_analyzer,
        create_phase3_analyzer,
    )
    _MULTI_MODEL_AVAILABLE = True
except ImportError:
    _MULTI_MODEL_AVAILABLE = False
    MultiModelSentimentAnalyzer = None
    EnsembleConfig = None
    EnsembleResult = None
    ModelPrediction = None
    ModelType = None
    ModelConfig = None
    MODEL_REGISTRY = None
    LABEL_MAPS = None
    create_phase1_analyzer = None
    create_phase2_analyzer = None
    create_phase3_analyzer = None


def is_multi_model_available() -> bool:
    """Check if multi-model sentiment analyzer is available."""
    return _MULTI_MODEL_AVAILABLE


# Extended exports
__all__.extend([
    # Multi-Model Sentiment
    'MultiModelSentimentAnalyzer',
    'EnsembleConfig',
    'EnsembleResult',
    'ModelPrediction',
    'ModelType',
    'ModelConfig',
    'MODEL_REGISTRY',
    'LABEL_MAPS',
    'create_phase1_analyzer',
    'create_phase2_analyzer',
    'create_phase3_analyzer',
    'is_multi_model_available',
])

# On-Chain Analytics Primitives (NEW)
try:
    from .onchain_whale_tracker import OnChainWhaleTracker, create_whale_tracker
    from .onchain_network_health import OnChainNetworkHealth, create_network_health
    from .enhanced_cascade_detector import EnhancedCascadeDetector
    _ONCHAIN_AVAILABLE = True
except ImportError:
    _ONCHAIN_AVAILABLE = False
    OnChainWhaleTracker = None
    create_whale_tracker = None
    OnChainNetworkHealth = None
    create_network_health = None
    EnhancedCascadeDetector = None


def is_onchain_available() -> bool:
    """Check if on-chain analytics primitives are available."""
    return _ONCHAIN_AVAILABLE


# Extended exports for on-chain analytics
__all__.extend([
    # On-Chain Whale Tracking
    'OnChainWhaleTracker',
    'create_whale_tracker',
    
    # On-Chain Network Health
    'OnChainNetworkHealth',
    'create_network_health',
    
    # Cascade Detection
    'EnhancedCascadeDetector',
    
    # Availability check
    'is_onchain_available',
])
