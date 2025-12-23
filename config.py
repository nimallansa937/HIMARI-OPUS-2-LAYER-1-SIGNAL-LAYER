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


# Default config instance
DEFAULT_CONFIG = L1Config()
