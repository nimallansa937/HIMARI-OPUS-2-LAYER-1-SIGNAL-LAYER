"""
HIMARI Validation Package - Statistical Testing Framework

Contains:
- Deflated Sharpe Ratio (DSR)
- Transaction Cost Model
- Combinatorial Purged Cross-Validation (CPCV)
- White/SPA Statistical Tests
"""

from .dsr import DeflatedSharpeRatio, SharpeRatioCalculator
from .transaction_costs import TransactionCostModel, TransactionCostOptimizer
from .cpcv import CPCVValidator, WalkForwardValidator, DataLeakageDetector
from .statistical_tests import WhiteSPATest, FamilyWiseErrorControl

__all__ = [
    'DeflatedSharpeRatio',
    'SharpeRatioCalculator',
    'TransactionCostModel',
    'TransactionCostOptimizer',
    'CPCVValidator',
    'WalkForwardValidator',
    'DataLeakageDetector',
    'WhiteSPATest',
    'FamilyWiseErrorControl',
]
