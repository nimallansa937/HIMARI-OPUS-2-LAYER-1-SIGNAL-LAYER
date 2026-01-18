"""
HIMARI ML Package - Layer 3 Prediction Components

Contains:
- Lorentzian K-NN Classification
- Ensemble Fusion (Meta-learner)
- KRLS (Kernel Recursive Least Squares)
- TINs (Technical Indicator Networks)
- TINs V2 (DSP-based for A/B testing)
"""

from .lorentzian_knn import LorentzianKNN, StreamingCentroids
from .ensemble import EnsembleFusion, VotingEnsemble, StackingEnsemble
from .krls import KernelRLS, AdaptiveKRLS
from .tins import (
    # Original TINs
    TIN_EMA, TIN_MACD, TIN_RSI, TIN_BollingerBands, TIN_ATR, TINStack,
    # DSP-based TINs for A/B testing
    TIN_JMA, TIN_FisherRSI, TIN_MESA_MACD, TIN_Keltner, TIN_SmoothedATR, TINStackV2,
)

__all__ = [
    'LorentzianKNN',
    'StreamingCentroids',
    'EnsembleFusion',
    'VotingEnsemble',
    'StackingEnsemble',
    'KernelRLS',
    'AdaptiveKRLS',
    # Original TINs
    'TIN_EMA',
    'TIN_MACD',
    'TIN_RSI',
    'TIN_BollingerBands',
    'TIN_ATR',
    'TINStack',
    # DSP-based TINs for A/B testing
    'TIN_JMA',
    'TIN_FisherRSI',
    'TIN_MESA_MACD',
    'TIN_Keltner',
    'TIN_SmoothedATR',
    'TINStackV2',
]

