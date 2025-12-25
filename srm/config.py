"""
HIMARI SRM Configuration

Contains all configuration dataclasses for the Systemic Risk Monitor.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


# =============================================================================
# API CONFIGURATION
# =============================================================================

@dataclass
class APIConfig:
    """Configuration for external API access."""
    
    # Binance Futures (free)
    binance_futures_base: str = "https://fapi.binance.com/fapi/v1"
    binance_spot_base: str = "https://api.binance.com/api/v3"
    
    # CoinGecko (free, limited)
    coingecko_base: str = "https://api.coingecko.com/api/v3"
    coingecko_api_key: Optional[str] = None  # Optional for higher limits
    
    # CoinGlass (paid, optional for LCI)
    coinglass_base: str = "https://open-api.coinglass.com/api/pro/v1"
    coinglass_api_key: Optional[str] = None
    
    # Yahoo Finance (free)
    yahoo_finance_base: str = "https://query1.finance.yahoo.com/v8/finance/chart"
    
    # Coinbase (free)
    coinbase_base: str = "https://api.coinbase.com/v2"
    
    # Kraken (free)
    kraken_base: str = "https://api.kraken.com/0/public"
    
    # Redis
    redis_url: str = field(default_factory=lambda: os.getenv("SRM_REDIS_URL", "redis://localhost:6379"))
    
    # Polling intervals (seconds)
    fsi_interval: int = 30   # Funding updates every 8h, frequent checks for freshness
    lei_interval: int = 5    # Order book changes rapidly
    ods_interval: int = 15   # Cross-venue comparison
    scsi_interval: int = 30  # Stablecoin stress
    lci_interval: int = 60   # OI distribution changes slowly
    caci_interval: int = 60  # TradFi data
    
    @classmethod
    def from_env(cls) -> "APIConfig":
        """Load configuration from environment variables."""
        return cls(
            coingecko_api_key=os.getenv("COINGECKO_API_KEY"),
            coinglass_api_key=os.getenv("COINGLASS_API_KEY"),
            redis_url=os.getenv("SRM_REDIS_URL", "redis://localhost:6379"),
        )


# =============================================================================
# SIGNAL CONFIGURATIONS
# =============================================================================

@dataclass
class FSIConfig:
    """Configuration for Funding Saturation Index calculation."""
    ema_span: int = 48  # Hours for EMA smoothing
    low_vol_threshold: float = 0.10  # Funding threshold in low volatility regime (as %)
    high_vol_threshold: float = 0.15  # Funding threshold in high volatility regime (as %)
    velocity_lookback: int = 24  # Hours for velocity calculation
    velocity_penalty_threshold: float = 0.01  # %/8h slope triggering penalty
    velocity_penalty_magnitude: float = 0.20  # Added to score when velocity exceeds threshold
    # API endpoint (configurable for testnet/proxy support)
    binance_funding_endpoint: str = "https://fapi.binance.com/fapi/v1/fundingRate"


@dataclass
class LEIConfig:
    """Configuration for Liquidity Evaporation Index calculation."""
    depth_percentage: float = 0.01  # Measure depth within 1% of mid price
    baseline_window_days: int = 7  # Rolling baseline period
    weekend_adjustment: float = 0.70  # Weekends have 30% less liquidity structurally
    velocity_lookback_hours: int = 1  # Window for depth change velocity
    velocity_penalty_threshold: float = -0.05  # 5% depth decline per hour
    velocity_penalty_magnitude: float = 0.30
    # API endpoint (configurable for testnet/proxy support)
    binance_orderbook_endpoint: str = "https://fapi.binance.com/fapi/v1/depth"


@dataclass
class ODSConfig:
    """Configuration for Oracle Divergence Score calculation."""
    critical_divergence_threshold: float = 0.05  # 5% divergence = score 1.0
    alert_divergence_threshold: float = 0.10  # Log critical warning above this
    venues: List[str] = field(default_factory=lambda: ['binance', 'coinbase', 'kraken', 'coingecko'])


@dataclass
class SCSIConfig:
    """Configuration for Stablecoin Stress Index calculation."""
    stablecoins: List[str] = field(default_factory=lambda: ['USDT', 'USDC', 'DAI', 'FDUSD'])
    critical_deviation_threshold: float = 0.05  # 5% = score 1.0
    venue_spread_threshold: float = 0.02  # 2% cross-venue spread = concern


@dataclass
class LCIConfig:
    """Configuration for Leverage Concentration Index calculation."""
    hhi_high_threshold: float = 2500  # HHI above this = elevated risk
    hhi_critical_threshold: float = 5000  # HHI above this = critical
    venues: List[str] = field(default_factory=lambda: ['binance', 'bybit', 'okx', 'deribit', 'bitget'])
    use_coinglass: bool = False  # Set to True if you have CoinGlass subscription


@dataclass
class CACIConfig:
    """Configuration for Cross-Asset Contagion Index calculation."""
    vix_elevated_threshold: float = 25  # VIX above 25 = elevated fear
    vix_crisis_threshold: float = 40  # VIX above 40 = market panic
    usdjpy_move_threshold: float = 0.02  # 2% daily move = carry trade stress
    spx_drawdown_threshold: float = 0.03  # 3% drawdown from recent high
    lookback_days: int = 5  # Window for calculating SPX drawdown


# =============================================================================
# GUARDIAN CONFIGURATION
# =============================================================================

@dataclass
class GuardianConfig:
    """Configuration for SystemicRiskGuardian."""
    # Tier thresholds
    reduce_threshold: float = 0.5   # Score above this = reduce exposure
    close_only_threshold: float = 0.7  # Score above this = close only
    halt_threshold: float = 0.9   # Score above this = emergency halt
    
    # Confirmation requirements
    confirmation_count: int = 3  # Consecutive critical readings before halt
    velocity_threshold: float = 0.001  # Score change per second triggering emergency
    
    # Execution settings
    dry_run: bool = True  # If True, log but don't execute trades
    

# =============================================================================
# MASTER CONFIGURATION
# =============================================================================

@dataclass
class SRMConfig:
    """Complete SRM configuration."""
    
    # Symbols to monitor
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    
    # Component configs
    api: APIConfig = field(default_factory=APIConfig)
    fsi: FSIConfig = field(default_factory=FSIConfig)
    lei: LEIConfig = field(default_factory=LEIConfig)
    ods: ODSConfig = field(default_factory=ODSConfig)
    scsi: SCSIConfig = field(default_factory=SCSIConfig)
    lci: LCIConfig = field(default_factory=LCIConfig)
    caci: CACIConfig = field(default_factory=CACIConfig)
    guardian: GuardianConfig = field(default_factory=GuardianConfig)
    
    # Processing settings
    main_loop_interval: float = 5.0  # Seconds between main loop iterations
    regime_lookback_minutes: int = 60  # Lookback for regime detection
    
    # Feature flags
    enable_fsi: bool = True
    enable_lei: bool = True
    enable_ods: bool = True
    enable_scsi: bool = True
    enable_lci: bool = True
    enable_caci: bool = True
    
    # Alerting
    telegram_bot_token: Optional[str] = field(
        default_factory=lambda: os.getenv("SRM_TELEGRAM_BOT_TOKEN")
    )
    telegram_chat_id: Optional[str] = field(
        default_factory=lambda: os.getenv("SRM_TELEGRAM_CHAT_ID")
    )
    
    @classmethod
    def from_env(cls) -> "SRMConfig":
        """Load configuration from environment variables."""
        config = cls()
        config.api = APIConfig.from_env()
        config.guardian.dry_run = os.getenv("SRM_DRY_RUN", "true").lower() == "true"
        config.lci.use_coinglass = os.getenv("SRM_USE_COINGLASS", "false").lower() == "true"
        
        symbols_env = os.getenv("SRM_SYMBOLS")
        if symbols_env:
            config.symbols = [s.strip() for s in symbols_env.split(",")]
        
        return config


# Default config instance
DEFAULT_SRM_CONFIG = SRMConfig()
