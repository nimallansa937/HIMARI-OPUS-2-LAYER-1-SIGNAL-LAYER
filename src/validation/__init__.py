"""
Validation Pipeline (HIFA) for HIMARI Layer 1 Explorer

Hierarchical Intelligent Filtering Architecture:
- 7-stage progressive filtering
- Cheap tests first, expensive tests for survivors
- Reduces compute by 10-50x vs full backtest on all candidates

Gap enhancements:
- CPCV: Combinatorial Purged Cross-Validation (Stage 4)
- Permutation testing for statistical significance
- Causal validation gate
- Multi-fidelity Bayesian optimization
"""

from .hifa import (
    HIFAResult, ValidationReport, HIFAPipeline,
    VALIDATION_THRESHOLDS
)
from .surrogate import SurrogateModel, SurrogateTrainer
from .batch_hifa import BatchHIFAProcessor

# CPCV and permutation testing (Stage 4 enhancement)
from .cpcv import (
    CPCVValidator, CPCVConfig, CPCVResult,
    CPCVSplitter, FoldMetrics
)
from .permutation_test import (
    PermutationTester, PermutationConfig, PermutationResult,
    BlockPermutationTester, StationaryBootstrapTester
)

# Gap enhancements
from .causal import (
    CausalValidationGate, CausalValidationResult, CausalHypothesis,
    CausalMechanism, RefutationResult
)
from .bayesian import (
    MultiFidelityBayesianOptimizer, GaussianProcessSurrogate,
    MultiFidelityGP, FidelityLevel, BayesianOptResult
)

__all__ = [
    # Core HIFA
    'HIFAResult', 'ValidationReport', 'HIFAPipeline',
    'VALIDATION_THRESHOLDS',
    'SurrogateModel', 'SurrogateTrainer',
    'BatchHIFAProcessor',

    # CPCV validation (Stage 4)
    'CPCVValidator', 'CPCVConfig', 'CPCVResult',
    'CPCVSplitter', 'FoldMetrics',

    # Permutation testing
    'PermutationTester', 'PermutationConfig', 'PermutationResult',
    'BlockPermutationTester', 'StationaryBootstrapTester',

    # Causal validation
    'CausalValidationGate', 'CausalValidationResult', 'CausalHypothesis',
    'CausalMechanism', 'RefutationResult',

    # Bayesian optimization
    'MultiFidelityBayesianOptimizer', 'GaussianProcessSurrogate',
    'MultiFidelityGP', 'FidelityLevel', 'BayesianOptResult'
]
