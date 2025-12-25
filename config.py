"""
HIMARI L1 Signal Layer Configuration

Integrates with existing HIMARI data infrastructure.
Adjust REDIS_* and KAFKA_* settings to match your deployment.
"""

import os
from dataclasses import dataclass
from typing import Dict, List

# =============================================================================
# INFRASTRUCTURE CONNECTIONS (match your existing setup)
# =============================================================================

# Redis - from your existing feature store
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_SSL = os.getenv("REDIS_SSL", "false").lower() == "true"
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# Kafka/Redpanda - from your existing ingestion
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "himari-l1-signals")
KAFKA_INPUT_TOPIC = os.getenv("KAFKA_INPUT_TOPIC", "quality_scores")  # Output from your Flink pipeline
KAFKA_OUTPUT_TOPIC = os.getenv("KAFKA_OUTPUT_TOPIC", "l1_signals")

# =============================================================================
# REDIS KEY SCHEMAS
# =============================================================================

class RedisKeys:
    """
    Redis key patterns for L1 signals.
    All keys use symbol as primary partition.
    """
    
    # Signal outputs (consumed by Layer 2)
    SIGNALS_LATEST = "signals:{symbol}:latest"          # Hash with all signals
    SIGNAL_MOMENTUM = "signals:{symbol}:momentum"       # Float [-1, 1]
    SIGNAL_MEAN_REV = "signals:{symbol}:mean_reversion" # Float [-1, 1]
    SIGNAL_VOLATILITY = "signals:{symbol}:volatility"   # Float [0, 1]
    SIGNAL_OBI = "signals:{symbol}:obi"                 # Float [-1, 1]
    
    # Regime detection
    REGIME_STATE = "signals:{symbol}:regime"            # BULL|BEAR|RANGE
    REGIME_CONFIDENCE = "signals:{symbol}:regime_conf"  # Float [0, 1]
    REGIME_HURST = "signals:{symbol}:hurst"             # Float [0, 1]
    REGIME_ENTROPY = "signals:{symbol}:entropy"         # Float [0, 2]
    
    # Volatility estimates
    VOL_GARCH = "signals:{symbol}:vol_garch"            # Float (annualized)
    VOL_REALIZED = "signals:{symbol}:vol_realized"      # Float (annualized)
    
    # Internal state (for warm restarts)
    STATE_KALMAN = "state:{symbol}:kalman"              # JSON blob
    STATE_HMM = "state:{symbol}:hmm"                    # JSON blob
    STATE_GARCH = "state:{symbol}:garch"                # JSON blob
    STATE_WELFORD = "state:{symbol}:welford"            # JSON blob
    
    # Metadata
    SIGNAL_TIMESTAMP = "signals:{symbol}:timestamp"     # Unix ms
    SIGNAL_VERSION = "signals:{symbol}:version"         # Schema version
    
    @classmethod
    def for_symbol(cls, key_pattern: str, symbol: str) -> str:
        """Generate actual Redis key for a symbol."""
        return key_pattern.format(symbol=symbol.upper())


# =============================================================================
# SIGNAL LAYER PARAMETERS
# =============================================================================

@dataclass
class KalmanConfig:
    """Kalman filter parameters."""
    process_noise: float = 0.01      # Q: how much price can change per step
    measurement_noise: float = 0.1   # R: how noisy are price observations
    
@dataclass
class UltimateSmootherConfig:
    """Ehlers Ultimate Smoother (2024) parameters."""
    period: int = 20                 # Smoothing period
    
@dataclass
class GARCHConfig:
    """Online GARCH(1,1) parameters."""
    omega: float = 0.00001           # Long-run variance weight
    alpha: float = 0.1               # Recent shock weight
    beta: float = 0.85               # Persistence weight
    # Note: alpha + beta should be < 1 for stationarity

@dataclass
class HMMConfig:
    """Hidden Markov Model parameters."""
    n_states: int = 3                # BULL, BEAR, RANGE
    # Transition matrix (rows sum to 1)
    # High diagonal = regimes are persistent
    transition_matrix: List[List[float]] = None
    
    def __post_init__(self):
        if self.transition_matrix is None:
            self.transition_matrix = [
                [0.95, 0.03, 0.02],  # BULL → BULL, BEAR, RANGE
                [0.03, 0.95, 0.02],  # BEAR → ...
                [0.10, 0.10, 0.80],  # RANGE → ...
            ]

@dataclass
class HurstConfig:
    """Moving Hurst exponent parameters."""
    window: int = 100                # Lookback window
    min_window: int = 20             # Minimum for calculation
    
@dataclass
class EntropyConfig:
    """Sample entropy parameters."""
    embedding_dim: int = 2           # m parameter
    tolerance_ratio: float = 0.2     # r as fraction of std dev

@dataclass  
class OBIConfig:
    """Order Book Imbalance parameters."""
    # Filtration thresholds
    min_persistence_ms: int = 500    # Order must exist >500ms
    min_size_percentile: float = 0.5 # Order must be >median size

@dataclass
class ValidationConfig:
    """Validation framework parameters."""
    sharpe_hurdle: float = 3.0       # Minimum Sharpe for acceptance
    dsr_confidence: float = 0.95     # DSR probability threshold
    cpcv_folds: int = 6              # Number of CV folds
    cpcv_test_folds: int = 2         # Test folds per combination
    purge_bars: int = 10             # Purge period (prevent leakage)
    embargo_pct: float = 0.02        # Embargo as % of data

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

@dataclass
class L1Config:
    """Complete L1 Signal Layer configuration."""
    
    # Symbols to process
    symbols: List[str] = None
    
    # Component configs
    kalman: KalmanConfig = None
    ultimate_smoother: UltimateSmootherConfig = None
    garch: GARCHConfig = None
    hmm: HMMConfig = None
    hurst: HurstConfig = None
    entropy: EntropyConfig = None
    obi: OBIConfig = None
    validation: ValidationConfig = None
    
    # Processing settings
    batch_size: int = 100            # Messages per batch
    flush_interval_ms: int = 100     # Max time before Redis flush
    
    # Feature flags
    enable_hmm: bool = True
    enable_garch: bool = True
    enable_hurst: bool = True
    enable_entropy: bool = True
    enable_obi: bool = True
    
    # SRM Integration
    enable_srm: bool = True          # Enable Systemic Risk Monitor integration
    srm_redis_url: str = None        # SRM Redis URL (uses same Redis if None)
    srm_reduce_threshold: float = 0.5   # Score above this = reduce position size
    srm_close_only_threshold: float = 0.7  # Score above this = close-only mode
    srm_halt_threshold: float = 0.9  # Score above this = halt trading
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["BTCUSDT", "ETHUSDT"]
        if self.kalman is None:
            self.kalman = KalmanConfig()
        if self.ultimate_smoother is None:
            self.ultimate_smoother = UltimateSmootherConfig()
        if self.garch is None:
            self.garch = GARCHConfig()
        if self.hmm is None:
            self.hmm = HMMConfig()
        if self.hurst is None:
            self.hurst = HurstConfig()
        if self.entropy is None:
            self.entropy = EntropyConfig()
        if self.obi is None:
            self.obi = OBIConfig()
        if self.validation is None:
            self.validation = ValidationConfig()
        # SRM Redis defaults to main Redis if not specified
        if self.srm_redis_url is None:
            self.srm_redis_url = os.getenv(
                "SRM_REDIS_URL", 
                f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
            )


# =============================================================================
# ENHANCED LAYER 1 CONFIGURATION (NEW)
# =============================================================================

@dataclass
class EnhancedSignalConfig:
    """
    Layer 1 Enhancement Configuration.
    
    Controls the new algorithmic enhancements:
    - StreamingHMM for zero-lag regime detection
    - Streaming indicators (talipp)
    - Multi-horizon momentum
    - Order Book Imbalance
    - Regime-aware signal fusion
    - Hybrid sentiment analysis
    """
    
    # Feature Flag: Enable entire enhanced system
    enabled: bool = False  # Set to True to activate
    
    # ===== HMM Configuration =====
    hmm_enabled: bool = True
    hmm_n_states: int = 3  # Bull, Bear, Range
    hmm_transition_persistence: float = 0.95
    hmm_range_persistence: float = 0.80
    hmm_adaptive_enabled: bool = True
    hmm_adaptive_lookback: int = 200
    hmm_adaptive_frequency: int = 50
    
    # Emission parameters (calibrated from BTC 1h)
    hmm_bull_mean: float = 0.001
    hmm_bull_std: float = 0.010
    hmm_bear_mean: float = -0.001
    hmm_bear_std: float = 0.020
    hmm_range_mean: float = 0.0
    hmm_range_std: float = 0.005
    
    # ===== Streaming Indicators =====
    indicators_enabled: bool = True
    ema_periods: tuple = (5, 10, 21, 50, 200)
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    
    # ===== Welford Statistics =====
    welford_min_samples: int = 20
    
    # ===== Multi-Horizon Momentum =====
    momentum_enabled: bool = True
    momentum_horizons: tuple = (5, 10, 21, 63)
    momentum_normalization: str = 'zscore'
    
    # ===== Order Book Imbalance =====
    obi_enabled: bool = True
    obi_levels: int = 5
    obi_depth_percentage: float = 0.01  # 1%
    obi_ema_period: int = 20
    
    # ===== Regime-Aware Signal Fusion =====
    fusion_enabled: bool = True
    fusion_confidence_threshold: float = 0.70
    fusion_min_regime_duration: int = 5
    
    # Regime weight multipliers
    fusion_bull_momentum: float = 1.5
    fusion_bull_mean_rev: float = 0.4
    fusion_bull_trend: float = 1.3
    
    fusion_bear_momentum: float = 1.2
    fusion_bear_mean_rev: float = 0.6
    fusion_bear_trend: float = 1.1
    
    fusion_range_momentum: float = 0.3
    fusion_range_mean_rev: float = 1.8
    fusion_range_trend: float = 0.2
    
    # ===== Hybrid Sentiment =====
    sentiment_enabled: bool = False  # Optional, requires API keys
    sentiment_vader_weight: float = 0.35
    sentiment_finbert_weight: float = 0.65
    sentiment_model: str = "ProsusAI/finbert"
    sentiment_batch_size: int = 32
    
    # ===== Enhancement 1: Sentiment Lag Features =====
    sentiment_enable_lag_features: bool = True
    sentiment_max_lag_bars: int = 360  # 6 hours at 1-minute bars
    sentiment_bar_interval_minutes: int = 1
    
    # ===== Enhancement 2: Dynamic Weighting =====
    sentiment_enable_dynamic_weighting: bool = True
    sentiment_weight_smoothing_alpha: float = 0.1
    sentiment_min_regime_duration: int = 5
    sentiment_weight_change_limit: float = 0.15
    volatility_threshold_low: float = 0.015
    volatility_threshold_high: float = 0.040
    
    # ===== Enhancement 3: Latency Validation =====
    enable_latency_gating: bool = True
    latency_check_interval: int = 100
    latency_p99_sentiment: float = 100.0  # ms
    latency_p99_total: float = 50.0  # ms
    
    # ===== Enhancement 4: Fine-Tuning =====
    sentiment_use_fine_tuned: bool = False
    sentiment_fine_tuned_model_path: str = "./models/finbert-crypto-finetuned"
    
    # ===== Enhancement 5: Social Media =====
    enable_social_sentiment: bool = False
    twitter_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    social_spam_filter_enabled: bool = True
    social_min_engagement: int = 5
    
    # ===== Enhancement 6: Monitoring =====
    enable_prometheus_metrics: bool = False
    prometheus_port: int = 8000
    metrics_update_interval: int = 100
    
    # ===== Enhancement 7: Multi-Asset =====
    enable_multi_asset: bool = False
    max_concurrent_models: int = 3
    
    # ===== SRM Integration (Enhanced) =====
    srm_reduce_threshold: float = 0.5
    srm_close_only_threshold: float = 0.7
    srm_halt_threshold: float = 0.9
    
    def get_regime_weights(self) -> dict:
        """Get regime-specific weight multipliers."""
        return {
            'Bull': {
                'momentum': self.fusion_bull_momentum,
                'mean_reversion': self.fusion_bull_mean_rev,
                'trend_following': self.fusion_bull_trend,
            },
            'Bear': {
                'momentum': self.fusion_bear_momentum,
                'mean_reversion': self.fusion_bear_mean_rev,
                'trend_following': self.fusion_bear_trend,
            },
            'Range': {
                'momentum': self.fusion_range_momentum,
                'mean_reversion': self.fusion_range_mean_rev,
                'trend_following': self.fusion_range_trend,
            }
        }


# Environment variable overrides for EnhancedSignalConfig
def load_enhanced_config() -> EnhancedSignalConfig:
    """Load enhanced signal config from environment variables."""
    return EnhancedSignalConfig(
        enabled=os.getenv("HIMARI_ENHANCED_LAYER1_ENABLED", "false").lower() == "true",
        hmm_enabled=os.getenv("HIMARI_HMM_ENABLED", "true").lower() == "true",
        obi_enabled=os.getenv("HIMARI_OBI_ENABLED", "true").lower() == "true",
        momentum_enabled=os.getenv("HIMARI_MOMENTUM_ENABLED", "true").lower() == "true",
        fusion_enabled=os.getenv("HIMARI_FUSION_ENABLED", "true").lower() == "true",
        sentiment_enabled=os.getenv("HIMARI_SENTIMENT_ENABLED", "false").lower() == "true",
    )


# Default config instance
DEFAULT_CONFIG = L1Config()

# Enhanced config instance (NEW)
ENHANCED_CONFIG = load_enhanced_config()

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
