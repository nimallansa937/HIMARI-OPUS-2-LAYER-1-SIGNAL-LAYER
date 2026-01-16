"""
Deployment Pipeline for HIMARI Layer 1 Explorer

Handles the journey from validated strategy to live trading:
- Shadow Environment (paper trading)
- Epistemic Uncertainty Gating
- Transfer Ratio Confidence
- Deployment Management

Gap enhancement:
- Automatic Domain Randomization (ADR)
"""

from .shadow import ShadowEnvironment, ShadowTradeRecord, ShadowPerformance
from .uncertainty import EpistemicUncertaintyGate
from .transfer import TransferRatioConfidence
from .deployment import DeploymentManager, DeploymentDecision

# Gap enhancement
from .adr import (
    AutomaticDomainRandomization, ADRConfig, ADRResult,
    DomainType, DomainRange, DomainSample
)

__all__ = [
    # Core deployment
    'ShadowEnvironment', 'ShadowTradeRecord', 'ShadowPerformance',
    'EpistemicUncertaintyGate',
    'TransferRatioConfidence',
    'DeploymentManager', 'DeploymentDecision',

    # ADR
    'AutomaticDomainRandomization', 'ADRConfig', 'ADRResult',
    'DomainType', 'DomainRange', 'DomainSample'
]
