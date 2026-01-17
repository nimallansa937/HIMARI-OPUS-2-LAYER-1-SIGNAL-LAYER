"""
HIMARI Layer 1 Explorer: Strategy Representation Layer

This module provides multiple strategy representations for the Explorer Agent:
- FSM Position Management: Stateful position tracking with transitions
- Fuzzy Logic Preprocessing: Gradual membership functions for indicators
- STGP Formula Discovery: Strongly-typed genetic programming for alpha formulas
- Bayesian Regime Gates: Probabilistic regime detection and gating
- Linear Benchmark: Simple models for comparison and fallback
- Combined Strategy: Orchestration of all components
"""

from .fsm_position_manager import (
    FSMState,
    FSMPositionManager,
    GuardCondition,
    Signal,
    Action,
    ActionResult,
    Transition,
    time_in_state_guard,
    pnl_guard,
    volatility_guard,
    cooldown_guard,
    FSMGeneticOperators,
)

from .fuzzy_preprocessor import (
    MembershipFunction,
    TriangularMF,
    TrapezoidalMF,
    GaussianMF,
    SigmoidMF,
    FuzzyVariable,
    FuzzyRule,
    FuzzyInferenceSystem,
    FuzzyPreprocessor,
)

from .stgp_formula import (
    DimensionType,
    STGPFormula,
    create_typed_primitive_set,
    evaluate_expression,
)

from .bayesian_regime_gate import (
    VolatilityRegime,
    TrendRegime,
    MarketRegime,
    GateAction,
    BayesianRegimeGate,
)

from .linear_benchmark import (
    LinearBenchmark,
    BenchmarkResult,
)

from .combined_strategy import (
    CombinedStrategy,
    StrategyGenome,
    SignalType,
    PositionManagerType,
)

__all__ = [
    # FSM
    "FSMState",
    "FSMPositionManager",
    "GuardCondition",
    "Signal",
    "Action",
    "ActionResult",
    "Transition",
    "time_in_state_guard",
    "pnl_guard",
    "volatility_guard",
    "cooldown_guard",
    "FSMGeneticOperators",
    # Fuzzy
    "MembershipFunction",
    "TriangularMF",
    "TrapezoidalMF",
    "GaussianMF",
    "SigmoidMF",
    "FuzzyVariable",
    "FuzzyRule",
    "FuzzyInferenceSystem",
    "FuzzyPreprocessor",
    # STGP
    "DimensionType",
    "STGPFormula",
    "create_typed_primitive_set",
    "evaluate_expression",
    # Bayesian
    "VolatilityRegime",
    "TrendRegime",
    "MarketRegime",
    "GateAction",
    "BayesianRegimeGate",
    # Linear
    "LinearBenchmark",
    "BenchmarkResult",
    # Combined
    "CombinedStrategy",
    "StrategyGenome",
    "SignalType",
    "PositionManagerType",
]
