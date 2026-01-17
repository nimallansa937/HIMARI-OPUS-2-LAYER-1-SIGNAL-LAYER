"""
Strategy Genome Encoding

Provides genetic representation of trading strategies with:
- Decision tree structure for trading logic
- 127-dimensional vector encoding for ML models
- Genetic operators (copy, mutate, crossover)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid
import random
import numpy as np


class SignalType(Enum):
    """Types of trading signals the strategy can use."""
    # Momentum signals
    MOMENTUM_RSI = "momentum_rsi"           # RSI-based momentum
    MOMENTUM_EMA = "momentum_ema"           # EMA crossover momentum
    MOMENTUM_MACD = "momentum_macd"         # MACD-based momentum

    # Mean reversion signals
    REVERSION_BB = "reversion_bb"           # Bollinger Band reversion
    REVERSION_RSI = "reversion_rsi"         # RSI oversold/overbought
    REVERSION_ZSCORE = "reversion_zscore"   # Z-score mean reversion

    # Volatility signals
    VOLATILITY_ATR = "volatility_atr"       # ATR-based volatility
    VOLATILITY_BB_WIDTH = "volatility_bbw"  # Bollinger Band width

    # Order flow signals
    ORDERFLOW_IMBALANCE = "orderflow_imb"   # Order book imbalance
    ORDERFLOW_CVD = "orderflow_cvd"         # Cumulative volume delta

    # Funding/carry signals
    FUNDING_RATE = "funding_rate"           # Funding rate carry
    FUNDING_OI = "funding_oi"               # Open interest divergence

    # Regime signals
    REGIME_TREND = "regime_trend"           # Trend strength
    REGIME_VOL = "regime_vol"               # Volatility regime

    # ===== ADVANCED NON-LAG INDICATORS (10 new signals) =====
    # Adaptive moving averages (lower lag than EMA)
    MOMENTUM_JMA = "momentum_jma"                    # Jurik Moving Average
    MOMENTUM_KAMA = "momentum_kama"                  # Kaufman Adaptive MA
    MOMENTUM_HMA = "momentum_hma"                    # Hull Moving Average

    # Advanced mean reversion
    MOMENTUM_FISHER = "momentum_fisher"              # Fisher Transform
    REVERSION_KELTNER = "reversion_keltner"          # Keltner Channel position

    # Advanced volatility (OHLC-based, more accurate than ATR)
    VOLATILITY_GARMAN_KLASS = "volatility_gk"        # Garman-Klass estimator

    # Market microstructure
    ORDERFLOW_VPIN = "orderflow_vpin"                # Volume-Synchronized PIN
    MICROSTRUCTURE_VWAP_DIST = "microstructure_vwap" # Distance from VWAP

    # Ehlers cycle analysis
    TREND_INSTANTANEOUS = "trend_instantaneous"      # Instantaneous trendline
    CYCLE_DOMINANT = "cycle_dominant"                # Dominant cycle period


# Signal type to feature index mapping
SIGNAL_FEATURE_MAP: Dict[SignalType, int] = {
    SignalType.MOMENTUM_RSI: 25,        # rsi_14
    SignalType.MOMENTUM_EMA: 6,         # ema_12 (vs ema_26)
    SignalType.MOMENTUM_MACD: 27,       # macd
    SignalType.REVERSION_BB: 13,        # price_zscore
    SignalType.REVERSION_RSI: 25,       # rsi_14
    SignalType.REVERSION_ZSCORE: 13,    # price_zscore
    SignalType.VOLATILITY_ATR: 12,      # atr_14
    SignalType.VOLATILITY_BB_WIDTH: 8,  # bb_upper (calculate width)
    SignalType.ORDERFLOW_IMBALANCE: 36, # order_book_imbalance
    SignalType.ORDERFLOW_CVD: 21,       # cvd_slope
    SignalType.FUNDING_RATE: 46,        # funding_rate_zscore
    SignalType.FUNDING_OI: 48,          # oi_change_1h
    SignalType.REGIME_TREND: 58,        # trend_strength
    SignalType.REGIME_VOL: 57,          # volatility_regime
    # Advanced indicators (indices 60-69)
    SignalType.MOMENTUM_JMA: 60,        # jma_14 (Jurik MA)
    SignalType.MOMENTUM_KAMA: 61,       # kama_14 (Kaufman Adaptive)
    SignalType.MOMENTUM_HMA: 62,        # hma_14 (Hull MA)
    SignalType.MOMENTUM_FISHER: 63,     # fisher_transform
    SignalType.REVERSION_KELTNER: 64,   # keltner_position
    SignalType.VOLATILITY_GARMAN_KLASS: 65,  # garman_klass_vol
    SignalType.ORDERFLOW_VPIN: 66,      # vpin (Volume-sync PIN)
    SignalType.MICROSTRUCTURE_VWAP_DIST: 67,  # vwap_distance
    SignalType.TREND_INSTANTANEOUS: 68, # instantaneous_trend
    SignalType.CYCLE_DOMINANT: 69,      # dominant_cycle_period
}

# Default threshold ranges for each signal type
SIGNAL_THRESHOLD_RANGES: Dict[SignalType, tuple] = {
    SignalType.MOMENTUM_RSI: (20, 80),
    SignalType.MOMENTUM_EMA: (-0.05, 0.05),
    SignalType.MOMENTUM_MACD: (-2, 2),
    SignalType.REVERSION_BB: (-2, 2),
    SignalType.REVERSION_RSI: (20, 80),
    SignalType.REVERSION_ZSCORE: (-2, 2),
    SignalType.VOLATILITY_ATR: (0.5, 2.0),
    SignalType.VOLATILITY_BB_WIDTH: (0.02, 0.10),
    SignalType.ORDERFLOW_IMBALANCE: (-0.5, 0.5),
    SignalType.ORDERFLOW_CVD: (-0.5, 0.5),
    SignalType.FUNDING_RATE: (-2, 2),
    SignalType.FUNDING_OI: (-0.2, 0.2),
    SignalType.REGIME_TREND: (0.3, 0.8),
    SignalType.REGIME_VOL: (0, 2),
    # Advanced indicator thresholds
    SignalType.MOMENTUM_JMA: (-0.05, 0.05),          # Normalized price difference
    SignalType.MOMENTUM_KAMA: (-0.05, 0.05),         # Normalized price difference
    SignalType.MOMENTUM_HMA: (-0.05, 0.05),          # Normalized price difference
    SignalType.MOMENTUM_FISHER: (-3, 3),             # Fisher transform output
    SignalType.REVERSION_KELTNER: (-2, 2),           # Z-score position in channel
    SignalType.VOLATILITY_GARMAN_KLASS: (0.01, 0.10), # Volatility percentage
    SignalType.ORDERFLOW_VPIN: (0, 1),               # VPIN in [0, 1]
    SignalType.MICROSTRUCTURE_VWAP_DIST: (-0.02, 0.02), # Distance as percentage
    SignalType.TREND_INSTANTANEOUS: (-0.1, 0.1),     # Trend slope
    SignalType.CYCLE_DOMINANT: (10, 50),             # Cycle period in bars
}


@dataclass
class Condition:
    """A single condition in the decision tree."""
    signal: SignalType
    operator: str  # '>' or '<'
    threshold: float

    def evaluate(self, features: np.ndarray) -> bool:
        """Evaluate condition against feature vector."""
        feature_idx = SIGNAL_FEATURE_MAP.get(self.signal)
        if feature_idx is None:
            return False

        value = features[feature_idx]

        if self.operator == '>':
            return value > self.threshold
        elif self.operator == '<':
            return value < self.threshold
        elif self.operator == '>=':
            return value >= self.threshold
        elif self.operator == '<=':
            return value <= self.threshold
        return False

    def evaluate_dict(self, signals: Dict[str, float]) -> bool:
        """Evaluate condition against signal dictionary."""
        value = signals.get(self.signal.value, 0)
        if self.operator == '>':
            return value > self.threshold
        return value < self.threshold

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'signal': self.signal.value,
            'operator': self.operator,
            'threshold': self.threshold
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Condition':
        """Deserialize from dictionary."""
        return cls(
            signal=SignalType(data['signal']),
            operator=data['operator'],
            threshold=data['threshold']
        )

    def mutate(self) -> 'Condition':
        """Return mutated copy of this condition."""
        new_condition = Condition(
            signal=self.signal,
            operator=self.operator,
            threshold=self.threshold
        )

        mutation_type = random.choice(['threshold', 'operator', 'signal'])

        if mutation_type == 'threshold':
            # Perturb threshold
            min_t, max_t = SIGNAL_THRESHOLD_RANGES[self.signal]
            delta = (max_t - min_t) * random.gauss(0, 0.1)
            new_condition.threshold = np.clip(self.threshold + delta, min_t, max_t)

        elif mutation_type == 'operator':
            # Flip operator
            new_condition.operator = '<' if self.operator == '>' else '>'

        elif mutation_type == 'signal':
            # Change signal type
            new_condition.signal = random.choice(list(SignalType))
            min_t, max_t = SIGNAL_THRESHOLD_RANGES[new_condition.signal]
            new_condition.threshold = random.uniform(min_t, max_t)

        return new_condition


@dataclass
class DecisionNode:
    """A node in the strategy decision tree."""
    condition: Optional[Condition] = None
    true_branch: Optional['DecisionNode'] = None
    false_branch: Optional['DecisionNode'] = None
    action: Optional[int] = None  # -1=sell, 0=hold, 1=buy

    def evaluate(self, features: np.ndarray) -> int:
        """Evaluate tree and return action."""
        # Leaf node - return action
        if self.action is not None:
            return self.action

        # Internal node - evaluate condition
        if self.condition is None:
            return 0  # Hold if no condition

        if self.condition.evaluate(features):
            if self.true_branch is not None:
                return self.true_branch.evaluate(features)
            return 0
        else:
            if self.false_branch is not None:
                return self.false_branch.evaluate(features)
            return 0

    def evaluate_dict(self, signals: Dict[str, float]) -> int:
        """Evaluate using signal dictionary."""
        if self.action is not None:
            return self.action

        if self.condition is None:
            return 0

        if self.condition.evaluate_dict(signals):
            return self.true_branch.evaluate_dict(signals) if self.true_branch else 0
        return self.false_branch.evaluate_dict(signals) if self.false_branch else 0

    def depth(self) -> int:
        """Calculate tree depth."""
        if self.action is not None:
            return 1

        left_depth = self.true_branch.depth() if self.true_branch else 0
        right_depth = self.false_branch.depth() if self.false_branch else 0
        return 1 + max(left_depth, right_depth)

    def node_count(self) -> int:
        """Count total nodes in tree."""
        count = 1
        if self.true_branch:
            count += self.true_branch.node_count()
        if self.false_branch:
            count += self.false_branch.node_count()
        return count

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {}
        if self.action is not None:
            result['action'] = self.action
        else:
            result['condition'] = self.condition.to_dict() if self.condition else None
            result['true_branch'] = self.true_branch.to_dict() if self.true_branch else None
            result['false_branch'] = self.false_branch.to_dict() if self.false_branch else None
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DecisionNode':
        """Deserialize from dictionary."""
        if 'action' in data:
            return cls(action=data['action'])

        return cls(
            condition=Condition.from_dict(data['condition']) if data.get('condition') else None,
            true_branch=cls.from_dict(data['true_branch']) if data.get('true_branch') else None,
            false_branch=cls.from_dict(data['false_branch']) if data.get('false_branch') else None
        )


@dataclass
class StrategyGenome:
    """
    Genetic representation of a trading strategy.

    Encodes:
    - Decision tree for entry/exit logic
    - Position sizing parameters
    - Risk management parameters

    Can be encoded as 127-dimensional vector for ML models.
    """
    id: str
    decision_tree: DecisionNode
    base_position_pct: float = 0.05       # Base position as % of portfolio
    stop_loss_atr_mult: float = 2.0       # Stop loss in ATR multiples
    take_profit_atr_mult: float = 3.0     # Take profit in ATR multiples
    max_positions: int = 5                 # Maximum concurrent positions
    fitness: float = 0.0                   # Calculated fitness score
    lineage: List[str] = field(default_factory=list)
    generation: int = 0
    source_engine: str = "unknown"         # Which engine created this

    # Additional metadata
    creation_timestamp: Optional[float] = None
    backtest_metrics: Dict[str, float] = field(default_factory=dict)

    def evaluate(self, features: np.ndarray) -> int:
        """Get trading signal from feature vector."""
        return self.decision_tree.evaluate(features)

    def to_vector(self) -> np.ndarray:
        """
        Encode strategy as 127-dimensional vector for surrogate model.

        Layout:
        - [0-123]: Decision tree encoding (31 nodes * 4 values each)
        - [124]: base_position_pct
        - [125]: stop_loss_atr_mult
        - [126]: take_profit_atr_mult
        """
        vector = np.zeros(127, dtype=np.float32)

        # Encode decision tree (BFS order)
        self._encode_node(self.decision_tree, vector, 0)

        # Encode parameters
        vector[124] = self.base_position_pct
        vector[125] = self.stop_loss_atr_mult / 5.0  # Normalize
        vector[126] = self.take_profit_atr_mult / 10.0  # Normalize

        return vector

    def _encode_node(self, node: Optional[DecisionNode], vector: np.ndarray, index: int) -> None:
        """Recursively encode decision tree node."""
        if node is None or index >= 31:
            return

        base = index * 4

        if node.action is not None:
            # Leaf node: [0, action, 0, 0]
            vector[base] = 0  # Node type: leaf
            vector[base + 1] = (node.action + 1) / 2  # Normalize -1,0,1 to 0,0.5,1
            vector[base + 2] = 0
            vector[base + 3] = 0
        else:
            # Internal node: [1, signal_idx, operator, threshold]
            vector[base] = 1  # Node type: internal
            if node.condition:
                signal_idx = list(SignalType).index(node.condition.signal)
                vector[base + 1] = signal_idx / len(SignalType)
                vector[base + 2] = 1 if node.condition.operator == '>' else 0
                # Normalize threshold
                min_t, max_t = SIGNAL_THRESHOLD_RANGES[node.condition.signal]
                if max_t > min_t:
                    vector[base + 3] = (node.condition.threshold - min_t) / (max_t - min_t)

            # Recurse to children
            self._encode_node(node.true_branch, vector, 2 * index + 1)
            self._encode_node(node.false_branch, vector, 2 * index + 2)

    @classmethod
    def from_vector(cls, vector: np.ndarray, strategy_id: Optional[str] = None) -> 'StrategyGenome':
        """Decode strategy from 127-dimensional vector."""
        tree = cls._decode_node(vector, 0)

        return cls(
            id=strategy_id or str(uuid.uuid4()),
            decision_tree=tree,
            base_position_pct=float(vector[124]),
            stop_loss_atr_mult=float(vector[125]) * 5.0,
            take_profit_atr_mult=float(vector[126]) * 10.0
        )

    @classmethod
    def _decode_node(cls, vector: np.ndarray, index: int) -> DecisionNode:
        """Recursively decode decision tree node."""
        if index >= 31:
            return DecisionNode(action=0)  # Default to hold

        base = index * 4
        node_type = vector[base]

        if node_type < 0.5:  # Leaf node
            action_normalized = vector[base + 1]
            action = int(round(action_normalized * 2 - 1))
            action = max(-1, min(1, action))
            return DecisionNode(action=action)
        else:  # Internal node
            signal_idx = int(round(vector[base + 1] * len(SignalType)))
            signal_idx = max(0, min(len(SignalType) - 1, signal_idx))
            signal = list(SignalType)[signal_idx]

            operator = '>' if vector[base + 2] > 0.5 else '<'

            min_t, max_t = SIGNAL_THRESHOLD_RANGES[signal]
            threshold = vector[base + 3] * (max_t - min_t) + min_t

            condition = Condition(signal=signal, operator=operator, threshold=threshold)

            return DecisionNode(
                condition=condition,
                true_branch=cls._decode_node(vector, 2 * index + 1),
                false_branch=cls._decode_node(vector, 2 * index + 2)
            )

    def copy(self) -> 'StrategyGenome':
        """Create a deep copy for genetic operations."""
        return StrategyGenome(
            id=str(uuid.uuid4()),
            decision_tree=self._deep_copy_tree(self.decision_tree),
            base_position_pct=self.base_position_pct,
            stop_loss_atr_mult=self.stop_loss_atr_mult,
            take_profit_atr_mult=self.take_profit_atr_mult,
            max_positions=self.max_positions,
            fitness=0.0,  # Reset fitness for new genome
            lineage=[self.id],
            generation=self.generation,
            source_engine=self.source_engine
        )

    def _deep_copy_tree(self, node: Optional[DecisionNode]) -> Optional[DecisionNode]:
        """Deep copy a decision tree."""
        if node is None:
            return None

        if node.action is not None:
            return DecisionNode(action=node.action)

        return DecisionNode(
            condition=Condition(
                signal=node.condition.signal,
                operator=node.condition.operator,
                threshold=node.condition.threshold
            ) if node.condition else None,
            true_branch=self._deep_copy_tree(node.true_branch),
            false_branch=self._deep_copy_tree(node.false_branch)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            'id': self.id,
            'decision_tree': self.decision_tree.to_dict(),
            'base_position_pct': self.base_position_pct,
            'stop_loss_atr_mult': self.stop_loss_atr_mult,
            'take_profit_atr_mult': self.take_profit_atr_mult,
            'max_positions': self.max_positions,
            'fitness': self.fitness,
            'lineage': self.lineage,
            'generation': self.generation,
            'source_engine': self.source_engine,
            'backtest_metrics': self.backtest_metrics
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyGenome':
        """Deserialize from dictionary."""
        return cls(
            id=data['id'],
            decision_tree=DecisionNode.from_dict(data['decision_tree']),
            base_position_pct=data['base_position_pct'],
            stop_loss_atr_mult=data['stop_loss_atr_mult'],
            take_profit_atr_mult=data['take_profit_atr_mult'],
            max_positions=data.get('max_positions', 5),
            fitness=data.get('fitness', 0.0),
            lineage=data.get('lineage', []),
            generation=data.get('generation', 0),
            source_engine=data.get('source_engine', 'unknown'),
            backtest_metrics=data.get('backtest_metrics', {})
        )

    def to_python_code(self) -> str:
        """Generate Python code representation of strategy."""
        lines = [
            "def evaluate(features):",
            "    '''Auto-generated strategy'''",
        ]

        def _generate_node_code(node: DecisionNode, indent: int) -> List[str]:
            prefix = "    " * indent
            if node.action is not None:
                action_map = {-1: "SELL", 0: "HOLD", 1: "BUY"}
                return [f"{prefix}return {node.action}  # {action_map[node.action]}"]

            if node.condition:
                feature_idx = SIGNAL_FEATURE_MAP[node.condition.signal]
                condition_str = f"features[{feature_idx}] {node.condition.operator} {node.condition.threshold:.4f}"
                code = [f"{prefix}if {condition_str}:  # {node.condition.signal.value}"]
                if node.true_branch:
                    code.extend(_generate_node_code(node.true_branch, indent + 1))
                else:
                    code.append(f"{prefix}    return 0")
                code.append(f"{prefix}else:")
                if node.false_branch:
                    code.extend(_generate_node_code(node.false_branch, indent + 1))
                else:
                    code.append(f"{prefix}    return 0")
                return code

            return [f"{prefix}return 0"]

        lines.extend(_generate_node_code(self.decision_tree, 1))
        return "\n".join(lines)

    def complexity_score(self) -> float:
        """Calculate complexity score (lower is simpler)."""
        depth = self.decision_tree.depth()
        nodes = self.decision_tree.node_count()
        return (depth * 2 + nodes) / 20.0  # Normalize roughly to 0-1

    def __repr__(self) -> str:
        return (f"StrategyGenome(id={self.id[:8]}..., gen={self.generation}, "
                f"fitness={self.fitness:.3f}, depth={self.decision_tree.depth()})")


def generate_random_strategy(max_depth: int = 4) -> StrategyGenome:
    """Generate a random valid strategy."""

    def _generate_random_tree(depth: int = 0) -> DecisionNode:
        # Higher probability of leaf at deeper levels
        leaf_prob = 0.3 + (depth * 0.15)

        if depth >= max_depth or random.random() < leaf_prob:
            return DecisionNode(action=random.choice([-1, 0, 1]))

        # Create internal node
        signal = random.choice(list(SignalType))
        operator = random.choice(['>', '<'])
        min_t, max_t = SIGNAL_THRESHOLD_RANGES[signal]
        threshold = random.uniform(min_t, max_t)

        return DecisionNode(
            condition=Condition(signal=signal, operator=operator, threshold=threshold),
            true_branch=_generate_random_tree(depth + 1),
            false_branch=_generate_random_tree(depth + 1)
        )

    return StrategyGenome(
        id=str(uuid.uuid4()),
        decision_tree=_generate_random_tree(),
        base_position_pct=random.uniform(0.02, 0.10),
        stop_loss_atr_mult=random.uniform(1.0, 3.0),
        take_profit_atr_mult=random.uniform(2.0, 5.0),
        max_positions=random.randint(3, 10),
        generation=0
    )
