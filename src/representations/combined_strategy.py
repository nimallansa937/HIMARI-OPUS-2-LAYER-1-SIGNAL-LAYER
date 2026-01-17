"""
Combined Strategy - Integration Layer

Orchestrates all strategy representation components:
- Fuzzy Preprocessing (optional)
- Signal Generation (Decision Tree / STGP / Fuzzy Rules)
- Bayesian Regime Gating (optional)
- FSM Position Management

Execution Flow:
1. Raw Features (60-dim) arrive
2. [Optional] Fuzzy Preprocessing → 180-dim fuzzy vector
3. Signal Generation → signal_strength [-1, +1], confidence [0, 1]
4. [Optional] Bayesian Regime Gate → gated_signal, gate_action
5. FSM Position Manager → action (ENTER, EXIT, SCALE, HOLD)
6. Forward to Layer 2 Execution
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum, auto
import json
import uuid
from datetime import datetime

from .fsm_position_manager import (
    FSMPositionManager, FSMState, Signal, Action, ActionResult
)
from .fuzzy_preprocessor import (
    FuzzyPreprocessor, FuzzyInferenceSystem, create_signal_fis
)
from .stgp_formula import (
    STGPFormula, STGPGenerator, create_typed_primitive_set, DimensionType
)
from .bayesian_regime_gate import (
    BayesianRegimeGate, GateAction, MarketRegime
)
from .linear_benchmark import (
    LinearBenchmark, BenchmarkResult
)


class SignalType(Enum):
    """Types of signal generators."""
    DECISION_TREE = auto()
    STGP = auto()
    FUZZY_RULES = auto()


class PositionManagerType(Enum):
    """Types of position managers."""
    SIMPLE = auto()
    FSM = auto()


@dataclass
class SimpleDecisionTree:
    """
    Simple decision tree placeholder.
    In production, this would be a full tree implementation.
    """
    rules: List[Dict[str, Any]] = field(default_factory=list)
    default_signal: float = 0.0

    def evaluate(self, features: Dict[str, float]) -> Tuple[float, float]:
        """Evaluate the tree and return (signal, confidence)."""
        for rule in self.rules:
            feature = rule.get("feature", "")
            threshold = rule.get("threshold", 0)
            operator = rule.get("operator", ">")
            signal = rule.get("signal", 0)
            confidence = rule.get("confidence", 0.5)

            value = features.get(feature, 0)

            if operator == ">" and value > threshold:
                return signal, confidence
            elif operator == "<" and value < threshold:
                return signal, confidence
            elif operator == ">=" and value >= threshold:
                return signal, confidence
            elif operator == "<=" and value <= threshold:
                return signal, confidence
            elif operator == "==" and abs(value - threshold) < 1e-10:
                return signal, confidence

        return self.default_signal, 0.3

    def to_dict(self) -> Dict[str, Any]:
        return {"rules": self.rules, "default_signal": self.default_signal}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimpleDecisionTree':
        return cls(
            rules=data.get("rules", []),
            default_signal=data.get("default_signal", 0.0)
        )


@dataclass
class StrategyGenome:
    """
    Complete genome representation for a combined strategy.

    Contains all configuration needed to instantiate and run a strategy.
    """
    # Identification
    strategy_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    created_by_engine: int = 1  # 1, 2, 3, or 4
    generation: int = 0
    lineage: List[str] = field(default_factory=list)

    # Signal Generation Configuration
    signal_type: SignalType = SignalType.STGP
    decision_tree_config: Optional[Dict[str, Any]] = None
    stgp_expression: Optional[str] = None
    fuzzy_rules_config: Optional[Dict[str, Any]] = None

    # Preprocessing Configuration
    fuzzy_preprocessing: bool = False
    fuzzy_config: Optional[Dict[str, Any]] = None

    # Position Management Configuration
    position_manager: PositionManagerType = PositionManagerType.FSM
    fsm_config: Optional[Dict[str, Any]] = None

    # Regime Gating Configuration
    regime_gate: bool = True
    bayesian_config: Optional[Dict[str, Any]] = None

    # Performance metadata
    fitness: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize genome to dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "created_at": self.created_at,
            "created_by_engine": self.created_by_engine,
            "generation": self.generation,
            "lineage": self.lineage,
            "signal_type": self.signal_type.name,
            "decision_tree_config": self.decision_tree_config,
            "stgp_expression": self.stgp_expression,
            "fuzzy_rules_config": self.fuzzy_rules_config,
            "fuzzy_preprocessing": self.fuzzy_preprocessing,
            "fuzzy_config": self.fuzzy_config,
            "position_manager": self.position_manager.name,
            "fsm_config": self.fsm_config,
            "regime_gate": self.regime_gate,
            "bayesian_config": self.bayesian_config,
            "fitness": self.fitness,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyGenome':
        """Deserialize genome from dictionary."""
        return cls(
            strategy_id=data.get("strategy_id", str(uuid.uuid4())[:8]),
            created_at=data.get("created_at", datetime.now().isoformat()),
            created_by_engine=data.get("created_by_engine", 1),
            generation=data.get("generation", 0),
            lineage=data.get("lineage", []),
            signal_type=SignalType[data.get("signal_type", "STGP")],
            decision_tree_config=data.get("decision_tree_config"),
            stgp_expression=data.get("stgp_expression"),
            fuzzy_rules_config=data.get("fuzzy_rules_config"),
            fuzzy_preprocessing=data.get("fuzzy_preprocessing", False),
            fuzzy_config=data.get("fuzzy_config"),
            position_manager=PositionManagerType[data.get("position_manager", "FSM")],
            fsm_config=data.get("fsm_config"),
            regime_gate=data.get("regime_gate", True),
            bayesian_config=data.get("bayesian_config"),
            fitness=data.get("fitness", 0.0),
            sharpe_ratio=data.get("sharpe_ratio", 0.0),
            win_rate=data.get("win_rate", 0.0),
            max_drawdown=data.get("max_drawdown", 0.0),
        )


@dataclass
class ExecutionResult:
    """Result from strategy execution."""
    action: Action
    size: float
    urgency: float
    signal_strength: float
    signal_confidence: float
    gate_action: Optional[GateAction] = None
    regime_probs: Optional[Dict[str, float]] = None
    fsm_state: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CombinedStrategy:
    """
    Combined strategy that orchestrates all representation components.

    Execution pipeline:
    1. Fuzzy preprocessing (optional)
    2. Signal generation
    3. Regime gating (optional)
    4. Position management (FSM)
    """

    def __init__(self, genome: Optional[StrategyGenome] = None):
        """
        Initialize combined strategy from genome.

        Args:
            genome: Strategy genome configuration
        """
        self.genome = genome or StrategyGenome()

        # Initialize components
        self.preprocessor: Optional[FuzzyPreprocessor] = None
        self.signal_generator: Union[SimpleDecisionTree, STGPFormula, FuzzyInferenceSystem, None] = None
        self.regime_gate: Optional[BayesianRegimeGate] = None
        self.position_manager: Optional[FSMPositionManager] = None
        self.linear_benchmark: Optional[LinearBenchmark] = None

        self._build_from_genome()

    def _build_from_genome(self) -> None:
        """Build strategy components from genome configuration."""
        # Fuzzy preprocessor
        if self.genome.fuzzy_preprocessing:
            self.preprocessor = FuzzyPreprocessor()
            if self.genome.fuzzy_config:
                # Would restore from config
                pass

        # Signal generator
        if self.genome.signal_type == SignalType.DECISION_TREE:
            if self.genome.decision_tree_config:
                self.signal_generator = SimpleDecisionTree.from_dict(
                    self.genome.decision_tree_config
                )
            else:
                self.signal_generator = SimpleDecisionTree()

        elif self.genome.signal_type == SignalType.STGP:
            pset = create_typed_primitive_set()
            if self.genome.stgp_expression:
                # Would parse expression and rebuild tree
                generator = STGPGenerator(pset)
                self.signal_generator = generator.generate_formula()
            else:
                generator = STGPGenerator(pset)
                self.signal_generator = generator.generate_formula()

        elif self.genome.signal_type == SignalType.FUZZY_RULES:
            if self.genome.fuzzy_rules_config:
                # Would restore from config
                self.signal_generator = create_signal_fis()
            else:
                self.signal_generator = create_signal_fis()

        # Regime gate
        if self.genome.regime_gate:
            self.regime_gate = BayesianRegimeGate()
            if self.genome.bayesian_config:
                # Would restore thresholds etc
                pass

        # Position manager
        if self.genome.position_manager == PositionManagerType.FSM:
            self.position_manager = FSMPositionManager()
            if self.genome.fsm_config:
                # Would restore transitions etc
                pass
        else:
            # Simple position manager (just passes through)
            self.position_manager = None

        # Linear benchmark for fallback
        self.linear_benchmark = LinearBenchmark()

    def execute(
        self,
        features: Dict[str, float],
        current_state: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute the full strategy pipeline.

        Args:
            features: Raw feature dictionary (60 features from Layer 0)
            current_state: Optional current state information

        Returns:
            ExecutionResult with action, size, and metadata
        """
        metadata = {
            "strategy_id": self.genome.strategy_id,
            "timestamp": datetime.now().isoformat()
        }

        # Step 1: Fuzzy preprocessing (optional)
        if self.preprocessor is not None:
            processed_features = self.preprocessor.fuzzify_feature_vector(features)
            metadata["preprocessing"] = "fuzzy"
            metadata["expanded_dim"] = len(processed_features)
        else:
            processed_features = features
            metadata["preprocessing"] = "none"

        # Step 2: Signal generation
        signal_strength, signal_confidence = self._generate_signal(processed_features)
        metadata["raw_signal"] = signal_strength
        metadata["raw_confidence"] = signal_confidence

        # Step 3: Regime gating (optional)
        gate_action = None
        regime_probs = None
        gated_signal = signal_strength

        if self.regime_gate is not None:
            gate_action, size_mult, gate_meta = self.regime_gate.process_features(features)
            regime_probs = gate_meta.get("regime_probs", {})

            # Apply gating
            if gate_action == GateAction.BLOCK:
                gated_signal = 0.0
                signal_confidence *= 0.1
            elif gate_action == GateAction.REDUCE_SIZE:
                gated_signal *= size_mult
                signal_confidence *= size_mult

            metadata["gate_action"] = gate_action.name
            metadata["regime_probs"] = regime_probs
            metadata["size_multiplier"] = size_mult

        # Step 4: Convert signal to FSM signal enum
        fsm_signal = self._signal_to_fsm(gated_signal, signal_confidence)
        metadata["fsm_signal"] = fsm_signal.name

        # Step 5: Position management
        if self.position_manager is not None:
            action_result = self.position_manager.process_signal(
                fsm_signal,
                features,
                confidence=signal_confidence
            )
            action = action_result.action
            size = action_result.size
            urgency = action_result.urgency
            fsm_state = self.position_manager.current_state.name
            metadata["fsm_metadata"] = action_result.metadata
        else:
            # Simple pass-through
            action = self._signal_to_action(gated_signal)
            size = abs(gated_signal)
            urgency = signal_confidence
            fsm_state = None

        return ExecutionResult(
            action=action,
            size=size,
            urgency=urgency,
            signal_strength=gated_signal,
            signal_confidence=signal_confidence,
            gate_action=gate_action,
            regime_probs=regime_probs,
            fsm_state=fsm_state,
            metadata=metadata
        )

    def _generate_signal(
        self,
        features: Dict[str, float]
    ) -> Tuple[float, float]:
        """Generate signal from the configured signal generator."""
        if self.signal_generator is None:
            return 0.0, 0.0

        if isinstance(self.signal_generator, SimpleDecisionTree):
            return self.signal_generator.evaluate(features)

        elif isinstance(self.signal_generator, STGPFormula):
            return self.signal_generator.get_signal(features)

        elif isinstance(self.signal_generator, FuzzyInferenceSystem):
            output, _ = self.signal_generator.infer(features)
            # Normalize output to [-1, 1] and compute confidence
            signal = max(-1, min(1, output))
            confidence = min(1.0, abs(output))
            return signal, confidence

        return 0.0, 0.0

    def _signal_to_fsm(
        self,
        signal: float,
        confidence: float
    ) -> Signal:
        """Convert signal strength to FSM signal enum."""
        if abs(signal) < 0.1:
            return Signal.HOLD

        if signal > 0:
            if confidence > 0.7 and signal > 0.5:
                return Signal.STRONG_BUY
            else:
                return Signal.BUY
        else:
            if confidence > 0.7 and signal < -0.5:
                return Signal.STRONG_SELL
            else:
                return Signal.SELL

    def _signal_to_action(self, signal: float) -> Action:
        """Convert signal to action for simple position manager."""
        if signal > 0.3:
            return Action.ENTER_LONG
        elif signal < -0.3:
            return Action.ENTER_SHORT
        else:
            return Action.HOLD

    def reset(self) -> None:
        """Reset strategy state."""
        if self.position_manager is not None:
            self.position_manager.reset()

    def get_state_info(self) -> Dict[str, Any]:
        """Get current strategy state information."""
        info = {
            "strategy_id": self.genome.strategy_id,
            "signal_type": self.genome.signal_type.name,
            "has_preprocessing": self.preprocessor is not None,
            "has_regime_gate": self.regime_gate is not None,
            "position_manager_type": self.genome.position_manager.name,
        }

        if self.position_manager is not None:
            info["fsm_state"] = self.position_manager.get_state_info()

        return info

    def to_genome(self) -> StrategyGenome:
        """Export current configuration as genome."""
        genome = StrategyGenome(
            strategy_id=self.genome.strategy_id,
            created_at=self.genome.created_at,
            created_by_engine=self.genome.created_by_engine,
            generation=self.genome.generation,
            lineage=self.genome.lineage,
            signal_type=self.genome.signal_type,
            fuzzy_preprocessing=self.preprocessor is not None,
            position_manager=self.genome.position_manager,
            regime_gate=self.regime_gate is not None,
            fitness=self.genome.fitness,
            sharpe_ratio=self.genome.sharpe_ratio,
            win_rate=self.genome.win_rate,
            max_drawdown=self.genome.max_drawdown,
        )

        # Export component configurations
        if isinstance(self.signal_generator, SimpleDecisionTree):
            genome.decision_tree_config = self.signal_generator.to_dict()
        elif isinstance(self.signal_generator, STGPFormula):
            genome.stgp_expression = self.signal_generator.to_string()
        elif isinstance(self.signal_generator, FuzzyInferenceSystem):
            genome.fuzzy_rules_config = self.signal_generator.to_dict()

        if self.position_manager is not None:
            genome.fsm_config = self.position_manager.to_dict()

        if self.regime_gate is not None:
            genome.bayesian_config = self.regime_gate.to_dict()

        return genome

    @classmethod
    def from_genome(cls, genome: StrategyGenome) -> 'CombinedStrategy':
        """Create strategy from genome."""
        return cls(genome=genome)


class StrategyFactory:
    """Factory for creating and evolving combined strategies."""

    @staticmethod
    def create_random(
        signal_type: SignalType = SignalType.STGP,
        with_preprocessing: bool = False,
        with_regime_gate: bool = True,
        engine_id: int = 1
    ) -> CombinedStrategy:
        """Create a random strategy with specified configuration."""
        genome = StrategyGenome(
            signal_type=signal_type,
            fuzzy_preprocessing=with_preprocessing,
            position_manager=PositionManagerType.FSM,
            regime_gate=with_regime_gate,
            created_by_engine=engine_id,
        )
        return CombinedStrategy(genome)

    @staticmethod
    def create_from_expression(
        expression: str,
        with_preprocessing: bool = False,
        with_regime_gate: bool = True
    ) -> CombinedStrategy:
        """Create a strategy from an STGP expression string."""
        genome = StrategyGenome(
            signal_type=SignalType.STGP,
            stgp_expression=expression,
            fuzzy_preprocessing=with_preprocessing,
            regime_gate=with_regime_gate,
        )
        return CombinedStrategy(genome)

    @staticmethod
    def create_decision_tree_strategy(
        rules: List[Dict[str, Any]],
        with_regime_gate: bool = True
    ) -> CombinedStrategy:
        """Create a strategy from decision tree rules."""
        genome = StrategyGenome(
            signal_type=SignalType.DECISION_TREE,
            decision_tree_config={"rules": rules, "default_signal": 0.0},
            regime_gate=with_regime_gate,
        )
        return CombinedStrategy(genome)


class StrategyStatistics:
    """Track statistics across different strategy representations."""

    def __init__(self):
        self.stats_by_signal_type: Dict[str, Dict[str, float]] = {}
        self.stats_by_position_manager: Dict[str, Dict[str, float]] = {}
        self.linear_benchmark_comparisons: List[Dict[str, Any]] = []

    def record_evaluation(
        self,
        strategy: CombinedStrategy,
        sharpe: float,
        win_rate: float,
        drawdown: float,
        vs_benchmark: BenchmarkResult
    ) -> None:
        """Record evaluation results for a strategy."""
        signal_type = strategy.genome.signal_type.name
        pm_type = strategy.genome.position_manager.name

        # Update signal type stats
        if signal_type not in self.stats_by_signal_type:
            self.stats_by_signal_type[signal_type] = {
                "count": 0,
                "total_sharpe": 0,
                "total_win_rate": 0,
                "approved": 0,
                "beat_benchmark": 0
            }

        self.stats_by_signal_type[signal_type]["count"] += 1
        self.stats_by_signal_type[signal_type]["total_sharpe"] += sharpe
        self.stats_by_signal_type[signal_type]["total_win_rate"] += win_rate

        if sharpe > 0.5:  # Example approval threshold
            self.stats_by_signal_type[signal_type]["approved"] += 1

        if vs_benchmark == BenchmarkResult.BEATS_BENCHMARK:
            self.stats_by_signal_type[signal_type]["beat_benchmark"] += 1

        # Update position manager stats
        if pm_type not in self.stats_by_position_manager:
            self.stats_by_position_manager[pm_type] = {
                "count": 0,
                "total_sharpe": 0,
                "approved": 0
            }

        self.stats_by_position_manager[pm_type]["count"] += 1
        self.stats_by_position_manager[pm_type]["total_sharpe"] += sharpe

        if sharpe > 0.5:
            self.stats_by_position_manager[pm_type]["approved"] += 1

        # Record benchmark comparison
        self.linear_benchmark_comparisons.append({
            "strategy_id": strategy.genome.strategy_id,
            "signal_type": signal_type,
            "sharpe": sharpe,
            "vs_benchmark": vs_benchmark.name
        })

    def get_approval_rate(self, signal_type: str) -> float:
        """Get approval rate for a signal type."""
        stats = self.stats_by_signal_type.get(signal_type, {})
        count = stats.get("count", 0)
        if count == 0:
            return 0.0
        return stats.get("approved", 0) / count

    def get_benchmark_beat_rate(self, signal_type: str) -> float:
        """Get rate of beating linear benchmark."""
        stats = self.stats_by_signal_type.get(signal_type, {})
        count = stats.get("count", 0)
        if count == 0:
            return 0.0
        return stats.get("beat_benchmark", 0) / count

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        summary = {}

        for signal_type, stats in self.stats_by_signal_type.items():
            count = stats["count"]
            if count > 0:
                summary[signal_type] = {
                    "count": count,
                    "avg_sharpe": stats["total_sharpe"] / count,
                    "avg_win_rate": stats["total_win_rate"] / count,
                    "approval_rate": stats["approved"] / count,
                    "benchmark_beat_rate": stats["beat_benchmark"] / count
                }

        return summary
