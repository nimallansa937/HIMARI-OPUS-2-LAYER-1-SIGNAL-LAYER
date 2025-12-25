"""
HIMARI Systemic Risk Monitor (SRM)

A 6-signal cascade detection system that operates as an asynchronous sidecar
to HIMARI's existing 10ms fast-path signal layer. Publishes risk scores to
Redis at 1-5 second intervals without impacting core trading latency.

Signals:
    - FSI: Funding Saturation Index
    - LEI: Liquidity Evaporation Index
    - ODS: Oracle Divergence Score
    - SCSI: Stablecoin Stress Index
    - LCI: Leverage Concentration Index
    - CACI: Cross-Asset Contagion Index
"""

from .config import SRMConfig, APIConfig
from .signals import (
    FundingSaturationIndex, FSIConfig,
    LiquidityEvaporationIndex, LEIConfig,
    OracleDivergenceScore, ODSConfig,
    StablecoinStressIndex, SCSIConfig,
    LeverageConcentrationIndex, LCIConfig,
    CrossAssetContagionIndex, CACIConfig,
)
from .regime import RegimeDetector, MarketRegime, RegimeWeights
from .composite import CompositeRiskCalculator, CompositeRiskResult
from .guardian import SystemicRiskGuardian, RiskAction, RiskDecision
from .services import SRMRedisClient, RateLimiter

__version__ = "1.0.0"
__all__ = [
    # Config
    "SRMConfig",
    "APIConfig",
    # Signals
    "FundingSaturationIndex",
    "FSIConfig",
    "LiquidityEvaporationIndex",
    "LEIConfig",
    "OracleDivergenceScore",
    "ODSConfig",
    "StablecoinStressIndex",
    "SCSIConfig",
    "LeverageConcentrationIndex",
    "LCIConfig",
    "CrossAssetContagionIndex",
    "CACIConfig",
    # Regime
    "RegimeDetector",
    "MarketRegime",
    "RegimeWeights",
    # Composite
    "CompositeRiskCalculator",
    "CompositeRiskResult",
    # Guardian
    "SystemicRiskGuardian",
    "RiskAction",
    "RiskDecision",
    # Services
    "SRMRedisClient",
    "RateLimiter",
]
