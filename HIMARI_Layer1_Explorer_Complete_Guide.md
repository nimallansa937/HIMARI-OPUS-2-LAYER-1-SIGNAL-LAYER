# HIMARI Layer 1 Explorer Agent: Complete Implementation Guide

**Version:** 1.0  
**Target:** Claude Code Implementation  
**Budget Constraint:** $50–100 USD/month  
**Latency Target:** 10–50ms decision latency  

---

# Part 0: Architectural Foundation

## 0.1 The Alpha Factory Principle

HIMARI is a reproductive system that continuously births, validates, and evolves strategies. Layer 1 Explorer is the fast generative organ; Layer 6 Explorer is the slow research brain. Both feed the same validation pipeline but operate at fundamentally different timescales.

## 0.2 Layer 1 vs Layer 6 Explorer

| Dimension | Layer 1 Explorer | Layer 6 Explorer |
|-----------|------------------|------------------|
| **Purpose** | Generate strategy candidates from existing patterns | Discover new causal mechanisms |
| **Timescale** | 1–4 hours per generation cycle | Days to weeks |
| **Latency** | Decisions in 10–50ms | No constraint |
| **Input** | 60-dim feature vector + regime signal + Shadow feedback | Papers, on-chain data, news |
| **Output** | 5–15 executable strategy candidates per cycle | Causal hypotheses, strategy templates |
| **Compute Budget** | $50–100/month | $100–150/month |

---

# Part 1: Strategy Representation

## 1.1 The 60-Dimensional Feature Vector

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import numpy as np


class FeatureType(Enum):
    """Dimensional typing prevents nonsense comparisons."""
    PRICE = "price"
    VOLUME = "volume"
    RATIO = "ratio"
    ZSCORE = "zscore"
    RATE = "rate"
    BOOLEAN = "boolean"
    COUNT = "count"


@dataclass
class FeatureSpec:
    """Specification for a single feature."""
    index: int
    name: str
    type: FeatureType
    min_val: float
    max_val: float
    update_freq_ms: int
    description: str


# Complete 60-Feature Vector Schema
FEATURE_SCHEMA = [
    # PRICE-DERIVED (0-14)
    FeatureSpec(0, "close", FeatureType.PRICE, 0, float('inf'), 100, "Current close"),
    FeatureSpec(1, "open", FeatureType.PRICE, 0, float('inf'), 100, "Current open"),
    FeatureSpec(2, "high", FeatureType.PRICE, 0, float('inf'), 100, "Current high"),
    FeatureSpec(3, "low", FeatureType.PRICE, 0, float('inf'), 100, "Current low"),
    FeatureSpec(4, "sma_20", FeatureType.PRICE, 0, float('inf'), 1000, "20-period SMA"),
    FeatureSpec(5, "sma_50", FeatureType.PRICE, 0, float('inf'), 1000, "50-period SMA"),
    FeatureSpec(6, "ema_12", FeatureType.PRICE, 0, float('inf'), 1000, "12-period EMA"),
    FeatureSpec(7, "ema_26", FeatureType.PRICE, 0, float('inf'), 1000, "26-period EMA"),
    FeatureSpec(8, "bb_upper", FeatureType.PRICE, 0, float('inf'), 1000, "Bollinger upper"),
    FeatureSpec(9, "bb_lower", FeatureType.PRICE, 0, float('inf'), 1000, "Bollinger lower"),
    FeatureSpec(10, "bb_mid", FeatureType.PRICE, 0, float('inf'), 1000, "Bollinger middle"),
    FeatureSpec(11, "vwap", FeatureType.PRICE, 0, float('inf'), 1000, "VWAP"),
    FeatureSpec(12, "atr_14", FeatureType.PRICE, 0, float('inf'), 1000, "14-period ATR"),
    FeatureSpec(13, "price_zscore", FeatureType.ZSCORE, -5, 5, 1000, "Price z-score"),
    FeatureSpec(14, "price_pct_change_1h", FeatureType.RATE, -0.5, 0.5, 1000, "1h price change"),
    
    # VOLUME-DERIVED (15-24)
    FeatureSpec(15, "volume", FeatureType.VOLUME, 0, float('inf'), 100, "Current volume"),
    FeatureSpec(16, "volume_sma_20", FeatureType.VOLUME, 0, float('inf'), 1000, "20-period vol SMA"),
    FeatureSpec(17, "volume_ratio", FeatureType.RATIO, 0, 10, 1000, "Volume / 20-period avg"),
    FeatureSpec(18, "obv", FeatureType.VOLUME, float('-inf'), float('inf'), 1000, "On-balance volume"),
    FeatureSpec(19, "obv_slope", FeatureType.RATE, -1, 1, 1000, "OBV slope"),
    FeatureSpec(20, "cvd", FeatureType.VOLUME, float('-inf'), float('inf'), 100, "Cumulative volume delta"),
    FeatureSpec(21, "cvd_slope", FeatureType.RATE, -1, 1, 1000, "CVD slope"),
    FeatureSpec(22, "buy_volume_ratio", FeatureType.RATIO, 0, 1, 100, "Buy volume ratio"),
    FeatureSpec(23, "large_trade_count", FeatureType.COUNT, 0, 1000, 1000, "Large trades count"),
    FeatureSpec(24, "volume_zscore", FeatureType.ZSCORE, -5, 5, 1000, "Volume z-score"),
    
    # TECHNICAL INDICATORS (25-34)
    FeatureSpec(25, "rsi_14", FeatureType.RATIO, 0, 100, 1000, "14-period RSI"),
    FeatureSpec(26, "rsi_30", FeatureType.RATIO, 0, 100, 1000, "30-period RSI"),
    FeatureSpec(27, "macd", FeatureType.ZSCORE, -10, 10, 1000, "MACD line"),
    FeatureSpec(28, "macd_signal", FeatureType.ZSCORE, -10, 10, 1000, "MACD signal"),
    FeatureSpec(29, "macd_hist", FeatureType.ZSCORE, -5, 5, 1000, "MACD histogram"),
    FeatureSpec(30, "stoch_k", FeatureType.RATIO, 0, 100, 1000, "Stochastic %K"),
    FeatureSpec(31, "stoch_d", FeatureType.RATIO, 0, 100, 1000, "Stochastic %D"),
    FeatureSpec(32, "adx_14", FeatureType.RATIO, 0, 100, 1000, "14-period ADX"),
    FeatureSpec(33, "cci_20", FeatureType.ZSCORE, -300, 300, 1000, "20-period CCI"),
    FeatureSpec(34, "mfi_14", FeatureType.RATIO, 0, 100, 1000, "14-period MFI"),
    
    # ORDER FLOW (35-44)
    FeatureSpec(35, "bid_ask_spread", FeatureType.RATE, 0, 0.1, 100, "Spread ratio"),
    FeatureSpec(36, "order_book_imbalance", FeatureType.RATIO, -1, 1, 100, "OB imbalance"),
    FeatureSpec(37, "bid_depth_5", FeatureType.VOLUME, 0, float('inf'), 100, "Bid depth 5 levels"),
    FeatureSpec(38, "ask_depth_5", FeatureType.VOLUME, 0, float('inf'), 100, "Ask depth 5 levels"),
    FeatureSpec(39, "depth_imbalance", FeatureType.RATIO, -1, 1, 100, "Depth imbalance"),
    FeatureSpec(40, "microprice", FeatureType.PRICE, 0, float('inf'), 100, "Volume-weighted mid"),
    FeatureSpec(41, "trade_flow_imbalance", FeatureType.RATIO, -1, 1, 100, "Trade flow direction"),
    FeatureSpec(42, "large_order_pressure", FeatureType.RATIO, -1, 1, 1000, "Large order direction"),
    FeatureSpec(43, "spread_zscore", FeatureType.ZSCORE, -5, 5, 1000, "Spread vs historical"),
    FeatureSpec(44, "liquidity_score", FeatureType.RATIO, 0, 1, 1000, "Composite liquidity"),
    
    # FUNDING & CARRY (45-49)
    FeatureSpec(45, "funding_rate", FeatureType.RATE, -0.01, 0.01, 8*3600*1000, "Perpetual funding"),
    FeatureSpec(46, "funding_rate_zscore", FeatureType.ZSCORE, -3, 3, 8*3600*1000, "Funding z-score"),
    FeatureSpec(47, "open_interest", FeatureType.VOLUME, 0, float('inf'), 60000, "Open interest"),
    FeatureSpec(48, "oi_change_1h", FeatureType.RATE, -0.5, 0.5, 60000, "OI 1h change"),
    FeatureSpec(49, "long_short_ratio", FeatureType.RATIO, 0.1, 10, 60000, "Long/short ratio"),
    
    # SENTIMENT & CROSS-ASSET (50-54)
    FeatureSpec(50, "fear_greed_index", FeatureType.RATIO, 0, 100, 24*3600*1000, "Fear & Greed"),
    FeatureSpec(51, "social_sentiment", FeatureType.RATIO, -1, 1, 3600*1000, "Social sentiment"),
    FeatureSpec(52, "btc_dominance", FeatureType.RATIO, 0, 1, 3600*1000, "BTC dominance"),
    FeatureSpec(53, "eth_btc_ratio", FeatureType.RATIO, 0, 1, 60000, "ETH/BTC ratio"),
    FeatureSpec(54, "usdt_dominance", FeatureType.RATIO, 0, 0.2, 3600*1000, "USDT share"),
    
    # REGIME INDICATORS (55-59)
    FeatureSpec(55, "regime_label", FeatureType.COUNT, 0, 4, 60000, "HMM regime"),
    FeatureSpec(56, "regime_confidence", FeatureType.RATIO, 0, 1, 60000, "Regime confidence"),
    FeatureSpec(57, "volatility_regime", FeatureType.COUNT, 0, 2, 60000, "Vol regime"),
    FeatureSpec(58, "trend_strength", FeatureType.RATIO, 0, 1, 60000, "Trend strength"),
    FeatureSpec(59, "regime_transition_prob", FeatureType.RATIO, 0, 1, 60000, "Transition prob"),
]


class FeatureVector:
    """60-dimensional feature vector with type safety."""
    
    def __init__(self):
        self.values = np.zeros(60, dtype=np.float32)
        self.timestamps = np.zeros(60, dtype=np.int64)
        self.schema = {spec.name: spec for spec in FEATURE_SCHEMA}
        self.index_map = {spec.name: spec.index for spec in FEATURE_SCHEMA}
    
    def set(self, name: str, value: float, timestamp_ms: int):
        """Set feature value with validation."""
        spec = self.schema[name]
        clamped = np.clip(value, spec.min_val, spec.max_val)
        self.values[spec.index] = clamped
        self.timestamps[spec.index] = timestamp_ms
    
    def get(self, name: str) -> float:
        return self.values[self.index_map[name]]
    
    def normalize(self) -> np.ndarray:
        """Return min-max normalized vector."""
        normalized = np.zeros(60, dtype=np.float32)
        for spec in FEATURE_SCHEMA:
            if spec.max_val == float('inf') or spec.min_val == float('-inf'):
                normalized[spec.index] = self.values[spec.index]
            else:
                range_val = spec.max_val - spec.min_val
                if range_val > 0:
                    normalized[spec.index] = (self.values[spec.index] - spec.min_val) / range_val
        return normalized
```

## 1.2 Strategy Grammar (AlphaCFG)

```python
"""
AlphaCFG: Context-Free Grammar for Trading Strategies

BNF Specification:
<strategy>      ::= <entry_rule> ";" <exit_rule> ";" <risk_control>
<condition>     ::= <comparison> | <condition> <logical_op> <comparison>
<comparison>    ::= <price_feature> <comp_op> <price_value>
                  | <ratio_feature> <comp_op> <ratio_value>
                  | <zscore_feature> <comp_op> <zscore_value>
<comp_op>       ::= ">" | "<" | ">=" | "<="
<logical_op>    ::= "AND" | "OR"
"""

FEATURE_TYPES = {
    # Price features
    "close": FeatureType.PRICE, "sma_20": FeatureType.PRICE,
    "ema_12": FeatureType.PRICE, "bb_upper": FeatureType.PRICE,
    # Ratio features
    "rsi_14": FeatureType.RATIO, "stoch_k": FeatureType.RATIO,
    "adx_14": FeatureType.RATIO, "order_book_imbalance": FeatureType.RATIO,
    # Z-score features
    "macd": FeatureType.ZSCORE, "price_zscore": FeatureType.ZSCORE,
    "funding_rate_zscore": FeatureType.ZSCORE,
}

VALID_COMPARISONS = {
    (FeatureType.PRICE, FeatureType.PRICE): True,
    (FeatureType.RATIO, FeatureType.RATIO): True,
    (FeatureType.ZSCORE, FeatureType.ZSCORE): True,
    (FeatureType.RATIO, None): True,  # Compare to literal
    (FeatureType.ZSCORE, None): True,
}


class GrammarValidator:
    """Validates strategy expressions against AlphaCFG grammar."""
    
    def validate(self, strategy_text: str) -> tuple[bool, list[str]]:
        """Returns (is_valid, error_messages)."""
        errors = []
        # Implementation: tokenize, parse, type-check
        # Full implementation ~200 lines
        return len(errors) == 0, errors
```

## 1.3 Strategy Genome Encoding

```python
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
import uuid


class SignalType(Enum):
    MOMENTUM_RSI = "momentum_rsi"
    MOMENTUM_EMA = "momentum_ema"
    REVERSION_BB = "reversion_bb"
    VOLATILITY_ATR = "volatility_atr"


@dataclass
class Condition:
    signal: SignalType
    operator: str  # '>' or '<'
    threshold: float
    
    def evaluate(self, signals: dict) -> bool:
        value = signals.get(self.signal.value, 0)
        return value > self.threshold if self.operator == '>' else value < self.threshold


@dataclass
class DecisionNode:
    condition: Optional[Condition] = None
    true_branch: Optional['DecisionNode'] = None
    false_branch: Optional['DecisionNode'] = None
    action: Optional[int] = None  # -1=sell, 0=hold, 1=buy
    
    def evaluate(self, signals: dict) -> int:
        if self.action is not None:
            return self.action
        if self.condition.evaluate(signals):
            return self.true_branch.evaluate(signals)
        return self.false_branch.evaluate(signals)


@dataclass
class StrategyGenome:
    """Genetic representation of a trading strategy."""
    id: str
    decision_tree: DecisionNode
    base_position_pct: float = 0.05
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 3.0
    fitness: float = 0.0
    lineage: List[str] = None
    generation: int = 0
    
    def to_vector(self) -> np.ndarray:
        """Encode as 127-dim vector for surrogate model."""
        vector = np.zeros(127, dtype=np.float32)
        self._encode_node(self.decision_tree, vector, 0)
        vector[124] = self.base_position_pct
        vector[125] = self.stop_loss_atr_mult
        vector[126] = self.take_profit_atr_mult
        return vector
    
    def _encode_node(self, node, vector, index):
        if node is None or index >= 31:
            return
        base = index * 4
        if node.action is not None:
            vector[base] = 0
            vector[base + 1] = node.action
        else:
            vector[base] = 1
            # Encode condition...
            self._encode_node(node.true_branch, vector, 2*index + 1)
            self._encode_node(node.false_branch, vector, 2*index + 2)
    
    @classmethod
    def from_vector(cls, vector: np.ndarray, strategy_id: str) -> 'StrategyGenome':
        """Decode vector back to genome."""
        # Implementation mirrors to_vector
        pass
    
    def copy(self) -> 'StrategyGenome':
        """Deep copy for genetic operations."""
        return StrategyGenome(
            id=str(uuid.uuid4()),
            decision_tree=self._deep_copy_tree(self.decision_tree),
            base_position_pct=self.base_position_pct,
            stop_loss_atr_mult=self.stop_loss_atr_mult,
            take_profit_atr_mult=self.take_profit_atr_mult,
            lineage=[self.id],
            generation=self.generation
        )
    
    def _deep_copy_tree(self, node):
        if node is None:
            return None
        return DecisionNode(
            condition=node.condition,
            true_branch=self._deep_copy_tree(node.true_branch),
            false_branch=self._deep_copy_tree(node.false_branch),
            action=node.action
        )
```
# Part 2: Generation Engines

The Layer 1 Explorer uses four complementary generation engines, each with different exploration-exploitation profiles. This diversity-by-design ensures the system doesn't converge to a fragile monoculture of similar strategies.

## 2.1 Engine 1: Evolutionary Search

The evolutionary engine maintains a population of strategy candidates and evolves them through genetic operators—mutation, crossover, and selection. Think of it as breeding strategies: fit parents produce offspring that inherit successful traits while random mutations introduce novelty.

```python
import random
import uuid
from typing import List, Callable
import numpy as np


class EvolutionaryExplorer:
    """
    Population-based evolutionary search over strategy space.
    
    Operator allocation (per HIMARI spec):
    - Mutation: 15%
    - Crossover: 40%
    - Novel generation: 10%
    - Recombination: 35%
    """
    
    def __init__(
        self,
        population_size: int = 100,
        elite_size: int = 10,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.40,
        tournament_size: int = 5
    ):
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.population: List[StrategyGenome] = []
        self.generation = 0
        self.stagnation_counter = 0
        self.last_best_fitness = float('-inf')
    
    def initialize_population(self, seed_strategies: List[StrategyGenome] = None):
        """Initialize with seeds or random valid strategies."""
        self.population = []
        if seed_strategies:
            self.population.extend(seed_strategies[:self.elite_size])
        
        while len(self.population) < self.population_size:
            strategy = self._generate_random_strategy()
            if strategy:
                self.population.append(strategy)
    
    def evolve_generation(self, evaluator: Callable, grammar_validator) -> StrategyGenome:
        """Evolve one generation, return best strategy."""
        # Evaluate fitness
        for strategy in self.population:
            if strategy.fitness == 0.0:
                result = evaluator(strategy)
                strategy.fitness = self._compute_fitness(result)
        
        # Sort by fitness (descending)
        self.population.sort(key=lambda s: s.fitness, reverse=True)
        
        # Track stagnation for adaptive mutation
        current_best = self.population[0].fitness
        if current_best <= self.last_best_fitness:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            self.last_best_fitness = current_best
        
        # Adaptive mutation: increase rate when stuck
        effective_mutation = self.mutation_rate
        if self.stagnation_counter > 5:
            effective_mutation = min(0.5, self.mutation_rate * (1 + 0.1 * self.stagnation_counter))
        
        # Create new population
        new_pop = [s.copy() for s in self.population[:self.elite_size]]  # Elites
        
        while len(new_pop) < self.population_size:
            rand = random.random()
            
            if rand < self.crossover_rate:
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()
                child = self._crossover(parent1, parent2)
            elif rand < self.crossover_rate + 0.10:  # Novel generation
                child = self._generate_random_strategy()
            else:
                parent = self._tournament_select()
                child = parent.copy()
            
            if random.random() < effective_mutation:
                child = self._mutate(child)
            
            # Grammar validation
            is_valid, _ = grammar_validator.validate(child)
            if is_valid:
                child.generation = self.generation + 1
                child.fitness = 0.0
                new_pop.append(child)
        
        self.population = new_pop
        self.generation += 1
        return self.population[0]
    
    def _tournament_select(self) -> StrategyGenome:
        """Select via tournament selection."""
        tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
        return max(tournament, key=lambda s: s.fitness)
    
    def _crossover(self, p1: StrategyGenome, p2: StrategyGenome) -> StrategyGenome:
        """Subtree crossover between two parents."""
        child_tree = self._subtree_crossover(p1.decision_tree, p2.decision_tree)
        alpha = random.random()
        
        return StrategyGenome(
            id=str(uuid.uuid4()),
            decision_tree=child_tree,
            base_position_pct=alpha * p1.base_position_pct + (1-alpha) * p2.base_position_pct,
            stop_loss_atr_mult=alpha * p1.stop_loss_atr_mult + (1-alpha) * p2.stop_loss_atr_mult,
            take_profit_atr_mult=alpha * p1.take_profit_atr_mult + (1-alpha) * p2.take_profit_atr_mult,
            lineage=[p1.id, p2.id],
            generation=max(p1.generation, p2.generation) + 1
        )
    
    def _mutate(self, strategy: StrategyGenome) -> StrategyGenome:
        """Apply random mutation."""
        child = strategy.copy()
        mutation_type = random.choice(['param', 'operator', 'subtree', 'indicator'])
        
        if mutation_type == 'param':
            child.stop_loss_atr_mult *= (1 + random.gauss(0, 0.1))
            child.take_profit_atr_mult *= (1 + random.gauss(0, 0.1))
            child.base_position_pct *= (1 + random.gauss(0, 0.1))
            # Clamp to valid ranges
            child.stop_loss_atr_mult = np.clip(child.stop_loss_atr_mult, 0.5, 5.0)
            child.take_profit_atr_mult = np.clip(child.take_profit_atr_mult, 1.0, 10.0)
            child.base_position_pct = np.clip(child.base_position_pct, 0.01, 0.5)
        
        child.id = str(uuid.uuid4())
        child.lineage = [strategy.id]
        return child
    
    def _compute_fitness(self, result: dict) -> float:
        """Multi-objective fitness: Sharpe, drawdown, profit factor."""
        sharpe = result.get('sharpe', 0)
        max_dd = result.get('max_drawdown', 1)
        pf = result.get('profit_factor', 0)
        trades = result.get('trade_count', 0)
        
        trade_penalty = 0 if trades >= 100 else (100 - trades) * 0.01
        dd_penalty = max(0, max_dd - 0.15) * 10
        
        return sharpe * 1.0 + np.log1p(pf) * 0.3 - dd_penalty - trade_penalty
    
    def _generate_random_strategy(self) -> StrategyGenome:
        """Generate random valid strategy."""
        tree = self._generate_random_subtree(max_depth=4)
        return StrategyGenome(
            id=str(uuid.uuid4()),
            decision_tree=tree,
            base_position_pct=random.uniform(0.02, 0.10),
            stop_loss_atr_mult=random.uniform(1.0, 3.0),
            take_profit_atr_mult=random.uniform(2.0, 5.0),
            generation=0
        )
    
    def _generate_random_subtree(self, max_depth: int, depth: int = 0) -> DecisionNode:
        """Generate random decision subtree."""
        if depth >= max_depth or random.random() < 0.3:
            return DecisionNode(action=random.choice([-1, 0, 1]))
        
        signal = random.choice(list(SignalType))
        operator = random.choice(['>', '<'])
        threshold = self._random_threshold(signal)
        
        return DecisionNode(
            condition=Condition(signal=signal, operator=operator, threshold=threshold),
            true_branch=self._generate_random_subtree(max_depth, depth + 1),
            false_branch=self._generate_random_subtree(max_depth, depth + 1)
        )
    
    def _random_threshold(self, signal: SignalType) -> float:
        thresholds = {
            SignalType.MOMENTUM_RSI: random.uniform(20, 80),
            SignalType.MOMENTUM_EMA: random.uniform(-0.05, 0.05),
            SignalType.REVERSION_BB: random.uniform(-2, 2),
            SignalType.VOLATILITY_ATR: random.uniform(0.5, 2.0),
        }
        return thresholds.get(signal, random.uniform(-1, 1))
    
    def _subtree_crossover(self, tree1, tree2):
        """Perform subtree crossover."""
        # Deep copy tree1, replace random subtree with subtree from tree2
        return self._deep_copy_tree(tree1)  # Simplified
    
    def _deep_copy_tree(self, node):
        if node is None:
            return None
        return DecisionNode(
            condition=node.condition,
            true_branch=self._deep_copy_tree(node.true_branch),
            false_branch=self._deep_copy_tree(node.false_branch),
            action=node.action
        )
```

### Neutral Drift Enhancement

When fitness stagnates, neutral drift allows exploration without selection pressure—escaping local optima:

```python
class NeutralDriftManager:
    """Accept neutral mutations to explore fitness landscape."""
    
    def __init__(self, tolerance: float = 0.05, drift_duration: int = 10):
        self.tolerance = tolerance
        self.drift_duration = drift_duration
        self.drift_counter = 0
        self.in_drift_mode = False
    
    def should_accept(self, old_fitness: float, new_fitness: float) -> bool:
        if self.in_drift_mode:
            relative_change = abs(new_fitness - old_fitness) / max(abs(old_fitness), 1e-6)
            return relative_change < self.tolerance
        return new_fitness >= old_fitness
    
    def update(self, best_fitness: float, stagnation_count: int):
        if stagnation_count > 5 and not self.in_drift_mode:
            self.in_drift_mode = True
            self.drift_counter = 0
        
        if self.in_drift_mode:
            self.drift_counter += 1
            if self.drift_counter >= self.drift_duration:
                self.in_drift_mode = False
```

## 2.2 Engine 2: Flow Matching Generation

Flow matching generates strategies by learning to transform noise into valid strategy parameters. The key advantage over diffusion models: 10–100x faster sampling (15 steps vs 100–1000) while achieving comparable quality.

```python
import torch
import torch.nn as nn


class ConditionalFlowMatching(nn.Module):
    """
    Flow matching for trading strategy generation.
    
    Performance:
    - Sampling steps: 15 (vs 100 for diffusion)
    - Time per sample: 1.5ms (vs 25ms for diffusion)
    - Daily capacity: 56M samples
    """
    
    def __init__(self, strategy_dim: int = 127, condition_dim: int = 16, hidden_dim: int = 512):
        super().__init__()
        self.strategy_dim = strategy_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Condition embedding
        self.cond_embed = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Velocity network
        self.velocity_net = nn.Sequential(
            nn.Linear(strategy_dim + hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, strategy_dim)
        )
        
        self.cfg_dropout = 0.1
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Predict velocity field at (x_t, t) given condition."""
        t_embed = self.time_embed(t.unsqueeze(-1))
        c_embed = self.cond_embed(condition)
        inp = torch.cat([x_t, t_embed, c_embed], dim=-1)
        return self.velocity_net(inp)
    
    def training_loss(self, x_1: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        OT conditional flow matching loss.
        
        x_t = (1-t)*x_0 + t*x_1 where x_0 ~ N(0, I)
        Target velocity: v* = x_1 - x_0
        Loss: MSE(v_predicted, v_target)
        """
        batch_size = x_1.shape[0]
        x_0 = torch.randn_like(x_1)
        t = torch.rand(batch_size, device=x_1.device)
        
        t_expand = t.unsqueeze(-1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        v_target = x_1 - x_0
        
        # Classifier-free guidance dropout
        if self.training:
            mask = torch.rand(batch_size, device=x_1.device) < self.cfg_dropout
            condition = condition.clone()
            condition[mask] = 0
        
        v_pred = self.forward(x_t, t, condition)
        return ((v_pred - v_target) ** 2).mean()
    
    @torch.no_grad()
    def sample(self, condition: torch.Tensor, num_steps: int = 15, cfg_scale: float = 7.5) -> torch.Tensor:
        """Generate strategies via Euler integration with classifier-free guidance."""
        batch_size = condition.shape[0]
        x = torch.randn(batch_size, self.strategy_dim, device=condition.device)
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=condition.device)
            
            if cfg_scale > 1.0:
                v_cond = self.forward(x, t, condition)
                v_uncond = self.forward(x, t, torch.zeros_like(condition))
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = self.forward(x, t, condition)
            
            x = x + v * dt
        
        return x


@dataclass
class GenerationCondition:
    """Target properties for generated strategies."""
    target_sharpe: float = 2.0
    target_max_drawdown: float = 0.10
    target_trades_per_month: int = 50
    regime_label: int = 0  # 0=bull, 1=bear, 2=range
    risk_tolerance: float = 0.5
    min_orthogonality: float = 0.3
    
    def to_tensor(self, device: str = 'cpu') -> torch.Tensor:
        return torch.tensor([
            self.target_sharpe / 5.0,
            self.target_max_drawdown,
            self.target_trades_per_month / 200,
            self.regime_label / 3.0,
            self.risk_tolerance,
            self.min_orthogonality,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0  # Padding to 16
        ], dtype=torch.float32, device=device)
```

## 2.3 Engine 3: LLM-Guided Generation

The LLM engine performs semantic mutations—understanding the purpose of strategy components rather than randomly swapping subtrees. Critical constraint: LLMs operate offline, generating artifacts that execute deterministically at runtime.

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMGeneratedStrategy:
    code: str
    explanation: str
    causal_hypothesis: str
    confidence: float


class LLMGuidedGenerator:
    """
    LLM-augmented strategy generation via Chain-of-Alpha.
    
    NO LLM IN REAL-TIME LOOP:
    - LLM generates offline artifacts
    - Artifacts execute deterministically
    - Latency: seconds for generation, microseconds for execution
    """
    
    MUTATION_PROMPT = """You are a quantitative researcher.

Strategy code:
```python
{strategy_code}
```

Backtest results: {backtest_summary}
Regime: {regime_context}
Modification request: {mutation_intent}

Generate improved version. Return ONLY the modified Python code."""

    NOVEL_PROMPT = """Generate a trading strategy with:
- Target Sharpe: {target_sharpe}
- Max drawdown: {max_drawdown}%
- Regime: {regime}

Portfolio gaps: {portfolio_gaps}
Available indicators: {indicators}

Return JSON with: strategy_name, entry_logic, exit_logic, stop_loss_atr, 
take_profit_atr, causal_hypothesis, python_code"""

    def __init__(self, api_client, model: str = "claude-sonnet-4-20250514"):
        self.api_client = api_client
        self.model = model
    
    async def mutate_strategy(
        self,
        strategy: StrategyGenome,
        backtest_result: dict,
        mutation_intent: str,
        regime: str = "normal"
    ) -> Optional[LLMGeneratedStrategy]:
        """LLM-guided targeted mutation."""
        prompt = self.MUTATION_PROMPT.format(
            strategy_code=strategy.to_python_code(),
            backtest_summary=self._format_backtest(backtest_result),
            mutation_intent=mutation_intent,
            regime_context=regime
        )
        
        try:
            response = await self.api_client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            code = self._extract_code(response.content[0].text)
            if not self._validate_syntax(code):
                return None
            
            return LLMGeneratedStrategy(
                code=code,
                explanation=f"LLM mutation: {mutation_intent}",
                causal_hypothesis="See mutation intent",
                confidence=0.6
            )
        except Exception as e:
            print(f"LLM mutation failed: {e}")
            return None
    
    async def generate_novel(
        self,
        condition: GenerationCondition,
        portfolio_gaps: list,
        indicators: list
    ) -> Optional[LLMGeneratedStrategy]:
        """Generate completely novel strategy from specs."""
        prompt = self.NOVEL_PROMPT.format(
            target_sharpe=condition.target_sharpe,
            max_drawdown=condition.target_max_drawdown * 100,
            regime=["bull", "bear", "range"][condition.regime_label],
            portfolio_gaps="\n".join(f"- {g}" for g in portfolio_gaps),
            indicators=", ".join(indicators)
        )
        
        try:
            response = await self.api_client.messages.create(
                model=self.model,
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = self._parse_json(response.content[0].text)
            if not result:
                return None
            
            return LLMGeneratedStrategy(
                code=result.get("python_code", ""),
                explanation=result.get("strategy_name", ""),
                causal_hypothesis=result.get("causal_hypothesis", ""),
                confidence=0.5
            )
        except Exception as e:
            print(f"LLM generation failed: {e}")
            return None
    
    def _format_backtest(self, result: dict) -> str:
        return f"Sharpe: {result.get('sharpe', 0):.2f}, DD: {result.get('max_drawdown', 0):.1%}"
    
    def _extract_code(self, text: str) -> str:
        if "```python" in text:
            start = text.find("```python") + 9
            end = text.find("```", start)
            return text[start:end].strip()
        return text.strip()
    
    def _validate_syntax(self, code: str) -> bool:
        import ast
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def _parse_json(self, text: str) -> Optional[dict]:
        import json
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        return None
```

## 2.4 Engine 4: External Idea Harvesting

The external harvester breaks evolutionary groupthink by importing ideas from arXiv, TradingView, Reddit, and YouTube. Think of it as competitive intelligence for alpha generation.

```python
@dataclass
class ExternalIdea:
    source: str
    url: str
    title: str
    raw_content: str
    relevance_score: float


@dataclass
class ParsedStrategy:
    idea_name: str
    mechanism: str
    entry_logic: str
    exit_logic: str
    indicators_used: list
    novelty_score: float


class ExternalIdeaHarvester:
    """
    Pipeline:
    1. Web Scout: Crawl sources
    2. Idea Extractor: LLM parses to structured hypotheses
    3. Strategy Generator: Translate to HIMARI format
    4. Novelty Check: Vector similarity against existing strategies
    """
    
    EXTRACTION_PROMPT = """Extract trading strategy from:
{source_text}

Return JSON: idea_name, mechanism, entry_logic, exit_logic, 
timeframe, indicators_used, confidence (0-1)"""

    def __init__(self, api_client, vector_db):
        self.api_client = api_client
        self.vector_db = vector_db
        self.crawled_urls = set()
    
    async def harvest_arxiv(self, query: str = "trading strategy", max_papers: int = 10) -> list:
        """Harvest from arXiv q-fin."""
        import arxiv
        
        ideas = []
        search = arxiv.Search(
            query=f"cat:q-fin.* AND ({query})",
            max_results=max_papers,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        for result in search.results():
            if result.entry_id in self.crawled_urls:
                continue
            self.crawled_urls.add(result.entry_id)
            
            relevance = self._compute_relevance(result.title + " " + result.summary)
            if relevance > 0.5:
                ideas.append(ExternalIdea(
                    source="arxiv",
                    url=result.entry_id,
                    title=result.title,
                    raw_content=result.summary,
                    relevance_score=relevance
                ))
        return ideas
    
    async def extract_strategy(self, idea: ExternalIdea) -> Optional[ParsedStrategy]:
        """Use LLM to extract structured strategy."""
        prompt = self.EXTRACTION_PROMPT.format(source_text=idea.raw_content[:4000])
        
        try:
            response = await self.api_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = self._parse_json(response.content[0].text)
            if not result or result.get("confidence", 0) < 0.3:
                return None
            
            novelty = await self._compute_novelty(result)
            
            return ParsedStrategy(
                idea_name=result.get("idea_name", "Unknown"),
                mechanism=result.get("mechanism", ""),
                entry_logic=result.get("entry_logic", ""),
                exit_logic=result.get("exit_logic", ""),
                indicators_used=result.get("indicators_used", []),
                novelty_score=novelty
            )
        except Exception:
            return None
    
    def _compute_relevance(self, text: str) -> float:
        keywords = ["strategy", "trading", "backtest", "sharpe", "alpha", "momentum"]
        matches = sum(1 for kw in keywords if kw in text.lower())
        return min(1.0, matches / 5)
    
    async def _compute_novelty(self, parsed: dict) -> float:
        # Query vector DB for similar strategies
        # novelty = 1 - max_similarity
        return 0.7  # Placeholder
    
    def _parse_json(self, text: str) -> Optional[dict]:
        import json
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end]) if start >= 0 else None
        except json.JSONDecodeError:
            return None
```

## 2.5 Engine Orchestration

The orchestrator coordinates all four engines, allocating compute budget and ensuring diversity:

```python
@dataclass
class GenerationBudget:
    evolutionary_pct: float = 0.40
    generative_pct: float = 0.25
    llm_guided_pct: float = 0.20
    external_pct: float = 0.15
    total_candidates_per_cycle: int = 100
    max_llm_calls_per_cycle: int = 20


class EngineOrchestrator:
    """Coordinates generation engines and enforces diversity."""
    
    def __init__(self, evolutionary, generative, llm, external, budget: GenerationBudget):
        self.evolutionary = evolutionary
        self.generative = generative
        self.llm = llm
        self.external = external
        self.budget = budget
    
    async def generate_candidates(
        self,
        condition: GenerationCondition,
        grammar_validator,
        existing_portfolio: list
    ) -> list:
        """Generate diverse candidates from all engines."""
        import asyncio
        
        # Calculate targets
        evo_target = int(self.budget.total_candidates_per_cycle * self.budget.evolutionary_pct)
        gen_target = int(self.budget.total_candidates_per_cycle * self.budget.generative_pct)
        llm_target = min(
            int(self.budget.total_candidates_per_cycle * self.budget.llm_guided_pct),
            self.budget.max_llm_calls_per_cycle
        )
        ext_target = int(self.budget.total_candidates_per_cycle * self.budget.external_pct)
        
        # Run engines in parallel
        tasks = [
            self._run_evolutionary(evo_target, grammar_validator),
            self._run_generative(gen_target, condition, grammar_validator),
            self._run_llm_guided(llm_target, condition, existing_portfolio),
            self._run_external(ext_target)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        candidates = []
        for result, source in zip(results, ["evolutionary", "generative", "llm", "external"]):
            if not isinstance(result, Exception):
                for s in result:
                    s.source_engine = source
                    candidates.append(s)
        
        # Diversity filtering
        return self._enforce_diversity(candidates, existing_portfolio)
    
    def _enforce_diversity(self, candidates: list, existing: list) -> list:
        """Filter to maintain population diversity."""
        diverse = []
        for candidate in candidates:
            is_diverse = True
            for other in existing + diverse:
                if self._compute_similarity(candidate, other) > 0.85:
                    is_diverse = False
                    break
            if is_diverse:
                diverse.append(candidate)
        return diverse
    
    def _compute_similarity(self, s1, s2) -> float:
        v1, v2 = s1.to_vector(), s2.to_vector()
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return dot / norm if norm > 0 else 0
```
# Part 3: Validation Pipeline (HIFA)

The Hierarchical Intelligent Filtering Architecture (HIFA) addresses a fundamental challenge: how do you validate thousands of strategy candidates when each full backtest costs $15 and 60 seconds? The answer is progressive filtering—cheap tests first, expensive tests only for survivors. This pyramid structure reduces compute costs by 10–50x while maintaining institutional-grade validation standards.

## 3.1 The Funnel Principle

```
Generation: 100 candidates
         │
         ▼
┌─────────────────────────────────────┐
│ Stage 0: Grammar Validation         │ Cost: <1ms    Pass: 95%
└─────────────────┬───────────────────┘
                  │ 95 valid
                  ▼
┌─────────────────────────────────────┐
│ Stage 1: DSR Gate                   │ Cost: <10ms   Pass: 60%
└─────────────────┬───────────────────┘
                  │ 57 pass
                  ▼
┌─────────────────────────────────────┐
│ Stage 2: Surrogate Ranking          │ Cost: ~10ms   Select: Top 20
└─────────────────┬───────────────────┘
                  │ 20 selected
                  ▼
┌─────────────────────────────────────┐
│ Stage 3: Fast Backtest              │ Cost: ~10s    Pass: 50%
└─────────────────┬───────────────────┘
                  │ 10 pass
                  ▼
┌─────────────────────────────────────┐
│ Stage 4: Full Backtest              │ Cost: ~60s    Pass: 50%
└─────────────────┬───────────────────┘
                  │ 5 pass
                  ▼
┌─────────────────────────────────────┐
│ Stage 5: True Contribution          │ Cost: ~5s     Pass: 40%
└─────────────────┬───────────────────┘
                  │ 2 pass
                  ▼
┌─────────────────────────────────────┐
│ Stage 6: Feature Neutralization     │ Cost: ~5s     Pass: 80%
└─────────────────┬───────────────────┘
                  │ 1-2 approved
                  ▼
         Paper Trading Queue
```

## 3.2 Stage Implementations

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from scipy import stats
import time


@dataclass
class HIFAResult:
    """Result from a single HIFA stage."""
    passed: bool
    score: float
    metrics: Dict[str, float]
    reason: str
    latency_ms: float


@dataclass
class ValidationReport:
    """Complete validation report for a strategy."""
    strategy_id: str
    stages_passed: List[str]
    final_stage: str
    final_result: HIFAResult
    total_latency_ms: float
    approved: bool
    approval_confidence: float


class HIFAPipeline:
    """
    Hierarchical Intelligent Filtering Architecture.
    
    7-stage validation with progressive compute cost.
    Total cost per approved strategy: ~$20-30
    vs ~$1500 if running full backtest on all candidates
    """
    
    def __init__(self, grammar_validator, surrogate_model, backtester, portfolio: list):
        self.grammar = grammar_validator
        self.surrogate = surrogate_model
        self.backtester = backtester
        self.portfolio = portfolio
        self.total_trials = 0  # For DSR calculation
    
    def validate(self, strategy: StrategyGenome) -> ValidationReport:
        """Run strategy through complete HIFA pipeline."""
        start_time = time.time()
        stages_passed = []
        
        # Stage 0: Grammar
        result = self._stage0_grammar(strategy)
        if not result.passed:
            return self._build_report(strategy, stages_passed, "grammar", result, start_time)
        stages_passed.append("grammar")
        
        # Stage 1: DSR
        result = self._stage1_dsr(strategy)
        if not result.passed:
            return self._build_report(strategy, stages_passed, "dsr", result, start_time)
        stages_passed.append("dsr")
        
        # Stage 2: Surrogate (ranking only)
        result = self._stage2_surrogate(strategy)
        stages_passed.append("surrogate")
        
        # Stage 3: Fast Backtest
        result = self._stage3_fast_backtest(strategy)
        if not result.passed:
            return self._build_report(strategy, stages_passed, "fast_backtest", result, start_time)
        stages_passed.append("fast_backtest")
        
        # Stage 4: Full Backtest
        result = self._stage4_full_backtest(strategy)
        if not result.passed:
            return self._build_report(strategy, stages_passed, "full_backtest", result, start_time)
        stages_passed.append("full_backtest")
        
        # Stage 5: True Contribution
        result = self._stage5_true_contribution(strategy)
        if not result.passed:
            return self._build_report(strategy, stages_passed, "true_contribution", result, start_time)
        stages_passed.append("true_contribution")
        
        # Stage 6: Neutralization
        result = self._stage6_neutralization(strategy)
        if not result.passed:
            return self._build_report(strategy, stages_passed, "neutralization", result, start_time)
        stages_passed.append("neutralization")
        
        return self._build_report(strategy, stages_passed, "approved", result, start_time)
    
    def _stage0_grammar(self, strategy: StrategyGenome) -> HIFAResult:
        """
        Stage 0: Grammar Validation
        Cost: <1ms | Checks: Syntax, dimensional consistency
        """
        start = time.time()
        is_valid, errors = self.grammar.validate(strategy)
        
        return HIFAResult(
            passed=is_valid,
            score=1.0 if is_valid else 0.0,
            metrics={"error_count": len(errors)},
            reason="; ".join(errors) if errors else "Valid",
            latency_ms=(time.time() - start) * 1000
        )
    
    def _stage1_dsr(self, strategy: StrategyGenome) -> HIFAResult:
        """
        Stage 1: Deflated Sharpe Ratio Gate
        Cost: <10ms | Purpose: Reject multiple-testing artifacts
        
        Key insight: When you test 1000 strategies, the best one will
        have SR ~2.5 purely by chance. DSR corrects for this.
        
        Formula:
        E[max(SR)] ≈ (1 - γ) × Φ⁻¹(1 - 1/N) + γ × Φ⁻¹(1 - 1/(N×e))
        DSR = (Observed_SR - E[max(SR)]) / σ[max(SR)]
        """
        start = time.time()
        self.total_trials += 1
        
        # Quick evaluation
        quick_result = self.backtester.quick_eval(strategy)
        observed_sharpe = quick_result.get('sharpe', 0)
        
        # DSR calculation
        gamma = 0.5772  # Euler-Mascheroni constant
        n = self.total_trials
        
        # Expected max under null hypothesis
        e_max = (1 - gamma) * stats.norm.ppf(1 - 1/n) + \
                gamma * stats.norm.ppf(1 - 1/(n * np.e))
        
        # Standard deviation of max
        sigma_max = np.sqrt(2 * np.log(n) - np.log(4 * np.pi))
        
        # DSR and p-value
        dsr = (observed_sharpe - e_max) / sigma_max
        p_value = 1 - stats.norm.cdf(dsr)
        
        passed = p_value < 0.05  # 95% significance
        threshold = e_max + stats.norm.ppf(0.95) * sigma_max
        
        return HIFAResult(
            passed=passed,
            score=observed_sharpe,
            metrics={
                "observed_sharpe": observed_sharpe,
                "dsr_threshold": threshold,
                "total_trials": self.total_trials,
                "p_value": p_value
            },
            reason=f"SR {observed_sharpe:.2f} {'>' if passed else '<='} threshold {threshold:.2f}",
            latency_ms=(time.time() - start) * 1000
        )
    
    def _stage2_surrogate(self, strategy: StrategyGenome) -> HIFAResult:
        """
        Stage 2: Surrogate Model Ranking
        Cost: ~10ms | Purpose: Cheap performance prediction
        
        A neural network trained on (strategy_vector, backtest_sharpe) pairs
        predicts performance without running expensive simulations.
        """
        start = time.time()
        import torch
        
        vector = torch.tensor(strategy.to_vector(), dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            prediction = self.surrogate(vector)
        
        predicted_sharpe = prediction[0, 0].item()
        uncertainty = prediction[0, 1].item() if prediction.shape[1] > 1 else 0.5
        
        return HIFAResult(
            passed=True,  # Ranking stage, doesn't reject
            score=predicted_sharpe,
            metrics={
                "predicted_sharpe": predicted_sharpe,
                "uncertainty": uncertainty
            },
            reason=f"Predicted SR: {predicted_sharpe:.2f} ± {uncertainty:.2f}",
            latency_ms=(time.time() - start) * 1000
        )
    
    def _stage3_fast_backtest(self, strategy: StrategyGenome) -> HIFAResult:
        """
        Stage 3: Fast Backtest
        Cost: ~10s | Simulation: Top 20 assets, 2-year window
        """
        start = time.time()
        
        result = self.backtester.run(
            strategy=strategy,
            assets="top20",
            start_date="2022-01-01",
            end_date="2024-01-01",
            execution_model="instant"
        )
        
        passed = (
            result.sharpe >= 1.5 and
            result.max_drawdown <= 0.20 and
            result.trade_count >= 50 and
            result.profit_factor >= 1.2
        )
        
        return HIFAResult(
            passed=passed,
            score=result.sharpe,
            metrics={
                "sharpe": result.sharpe,
                "max_drawdown": result.max_drawdown,
                "trade_count": result.trade_count,
                "profit_factor": result.profit_factor
            },
            reason=f"Fast BT: SR={result.sharpe:.2f}, DD={result.max_drawdown:.1%}",
            latency_ms=(time.time() - start) * 1000
        )
    
    def _stage4_full_backtest(self, strategy: StrategyGenome) -> HIFAResult:
        """
        Stage 4: Full Backtest
        Cost: ~60s | Simulation: All assets, 5-year, realistic execution
        """
        start = time.time()
        
        result = self.backtester.run(
            strategy=strategy,
            assets="all",
            start_date="2019-01-01",
            end_date="2024-01-01",
            execution_model="realistic",
            regime_splits=True
        )
        
        passed = (
            result.sharpe >= 2.0 and
            result.max_drawdown <= 0.15 and
            result.trade_count >= 200 and
            result.profit_factor >= 1.5 and
            result.regime_consistency >= 0.6
        )
        
        return HIFAResult(
            passed=passed,
            score=result.sharpe,
            metrics={
                "sharpe": result.sharpe,
                "max_drawdown": result.max_drawdown,
                "trade_count": result.trade_count,
                "profit_factor": result.profit_factor,
                "calmar_ratio": result.calmar_ratio,
                "regime_consistency": result.regime_consistency,
                "bull_sharpe": result.regime_sharpes.get("bull", 0),
                "bear_sharpe": result.regime_sharpes.get("bear", 0)
            },
            reason=f"Full BT: SR={result.sharpe:.2f}, Regimes={result.regime_consistency:.1%}",
            latency_ms=(time.time() - start) * 1000
        )
    
    def _stage5_true_contribution(self, strategy: StrategyGenome) -> HIFAResult:
        """
        Stage 5: True Contribution Check
        Cost: ~5s | Purpose: Portfolio orthogonality
        
        Problem: A strategy with SR=3.0 might be 95% correlated with
        your existing portfolio—it adds almost nothing new.
        
        True Contribution measures marginal value: what does this
        strategy add that existing strategies don't capture?
        """
        start = time.time()
        
        strategy_returns = self.backtester.get_returns(strategy)
        ensemble_returns = self._get_ensemble_returns()
        existing_returns = [self.backtester.get_returns(s) for s in self.portfolio]
        
        # 1. Marginal Sharpe Contribution
        def sharpe_ratio(returns):
            return np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        current_sharpe = sharpe_ratio(ensemble_returns)
        combined = 0.9 * ensemble_returns + 0.1 * strategy_returns
        new_sharpe = sharpe_ratio(combined)
        marginal_sharpe = new_sharpe - current_sharpe
        
        # 2. Orthogonality (1 - max correlation)
        if existing_returns:
            correlations = [np.corrcoef(strategy_returns, s)[0, 1] 
                          for s in existing_returns if len(s) == len(strategy_returns)]
            max_corr = max(abs(c) for c in correlations) if correlations else 0
        else:
            max_corr = 0
        orthogonality = 1 - max_corr
        
        # 3. Residual IC (predictive power after neutralizing existing signals)
        if existing_returns and len(existing_returns) > 0:
            X = np.column_stack([s for s in existing_returns if len(s) == len(strategy_returns)])
            if X.shape[1] > 0:
                beta = np.linalg.lstsq(X, strategy_returns, rcond=None)[0]
                residual = strategy_returns - X @ beta
                residual_ic = np.corrcoef(residual[:-1], strategy_returns[1:])[0, 1]
            else:
                residual_ic = 0.5
        else:
            residual_ic = 0.5
        
        passed = (
            marginal_sharpe > 0.05 and
            orthogonality > 0.3 and
            residual_ic > 0.02
        )
        
        return HIFAResult(
            passed=passed,
            score=marginal_sharpe,
            metrics={
                "marginal_sharpe": marginal_sharpe,
                "orthogonality": orthogonality,
                "residual_ic": residual_ic,
                "max_correlation": max_corr
            },
            reason=f"TC: Marginal SR={marginal_sharpe:.3f}, Orth={orthogonality:.2f}",
            latency_ms=(time.time() - start) * 1000
        )
    
    def _stage6_neutralization(self, strategy: StrategyGenome) -> HIFAResult:
        """
        Stage 6: Feature Neutralization Audit
        Cost: ~5s | Purpose: Alpha vs beta separation
        
        Problem: A "high alpha" strategy might just be long beta—
        it goes up when the market goes up.
        
        Solution: Remove factor exposure and check if residual has value.
        """
        start = time.time()
        from sklearn.linear_model import LinearRegression
        
        strategy_returns = self.backtester.get_returns(strategy)
        factor_returns = self._get_factor_returns()  # market, momentum, size, funding
        
        # Fit factor model
        model = LinearRegression()
        model.fit(factor_returns, strategy_returns)
        
        # Residual (pure alpha)
        predicted_beta = model.predict(factor_returns)
        residual_returns = strategy_returns - predicted_beta
        
        # Residual Sharpe
        residual_sharpe = np.mean(residual_returns) / (np.std(residual_returns) + 1e-8) * np.sqrt(252)
        
        # Factor exposures
        exposures = dict(zip(["market", "momentum", "size", "funding"], model.coef_))
        
        # IC retention
        original_ic = np.corrcoef(strategy_returns[:-1], strategy_returns[1:])[0, 1]
        residual_ic = np.corrcoef(residual_returns[:-1], residual_returns[1:])[0, 1]
        ic_retention = residual_ic / (original_ic + 1e-8) if original_ic > 0 else 0
        
        passed = (
            residual_sharpe >= 1.0 and
            ic_retention >= 0.5 and
            all(abs(b) < 0.5 for b in exposures.values())
        )
        
        return HIFAResult(
            passed=passed,
            score=residual_sharpe,
            metrics={
                "residual_sharpe": residual_sharpe,
                "ic_retention": ic_retention,
                "r_squared": model.score(factor_returns, strategy_returns),
                **{f"{k}_beta": v for k, v in exposures.items()}
            },
            reason=f"Neutralization: Residual SR={residual_sharpe:.2f}, IC Retention={ic_retention:.1%}",
            latency_ms=(time.time() - start) * 1000
        )
    
    def _get_ensemble_returns(self) -> np.ndarray:
        if not self.portfolio:
            return np.zeros(252 * 5)
        returns = [self.backtester.get_returns(s) for s in self.portfolio]
        return np.mean(returns, axis=0)
    
    def _get_factor_returns(self) -> np.ndarray:
        # Would fetch from data store
        n_days = 252 * 5
        return np.random.randn(n_days, 4) * 0.01  # Placeholder
    
    def _build_report(self, strategy, stages_passed, final_stage, final_result, start_time):
        return ValidationReport(
            strategy_id=strategy.id,
            stages_passed=stages_passed,
            final_stage=final_stage,
            final_result=final_result,
            total_latency_ms=(time.time() - start_time) * 1000,
            approved=final_stage == "approved",
            approval_confidence=final_result.score / 3.0 if final_stage == "approved" else 0
        )
```

## 3.3 Batch Processing Optimization

For high-throughput validation, batch candidates through early stages:

```python
class BatchHIFAProcessor:
    """
    Batch processing for HIFA validation.
    
    Optimization: Run stages 0-2 on all candidates in batch,
    then selectively run expensive stages 3-6 on survivors.
    """
    
    def __init__(self, pipeline: HIFAPipeline, batch_size: int = 100):
        self.pipeline = pipeline
        self.batch_size = batch_size
    
    def process_batch(self, candidates: List[StrategyGenome]) -> List[ValidationReport]:
        """Process batch through HIFA with early filtering."""
        results = []
        
        # Stage 0-1: Grammar + DSR (fast, run on all)
        stage1_survivors = []
        for candidate in candidates:
            grammar_result = self.pipeline._stage0_grammar(candidate)
            if not grammar_result.passed:
                results.append(self.pipeline._build_report(
                    candidate, [], "grammar", grammar_result, time.time()))
                continue
            
            dsr_result = self.pipeline._stage1_dsr(candidate)
            if not dsr_result.passed:
                results.append(self.pipeline._build_report(
                    candidate, ["grammar"], "dsr", dsr_result, time.time()))
                continue
            
            stage1_survivors.append(candidate)
        
        # Stage 2: Surrogate ranking (batch inference)
        if stage1_survivors:
            import torch
            vectors = torch.stack([
                torch.tensor(s.to_vector(), dtype=torch.float32)
                for s in stage1_survivors
            ])
            
            with torch.no_grad():
                predictions = self.pipeline.surrogate(vectors)
            
            # Sort by predicted Sharpe, take top 20
            scores = predictions[:, 0].numpy()
            top_indices = np.argsort(scores)[-20:]
            stage2_survivors = [stage1_survivors[i] for i in top_indices]
        else:
            stage2_survivors = []
        
        # Stages 3-6: Run sequentially on survivors
        for candidate in stage2_survivors:
            report = self.pipeline.validate(candidate)
            results.append(report)
        
        return results
```
# Part 4: Deployment Pipeline

A strategy that passes HIFA validation has proven itself against historical data. But historical performance doesn't guarantee future results—the market is non-stationary. Before live deployment, strategies must prove themselves in shadow mode: real market data, simulated execution. Think of it as a probationary period where the strategy demonstrates it can handle live market conditions without risking actual capital.

## 4.1 Shadow Environment (Paper Trading)

```python
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, timedelta
import numpy as np


@dataclass
class ShadowTradeRecord:
    """Record of a trade executed in shadow mode."""
    timestamp: datetime
    symbol: str
    direction: int
    size: float
    entry_price: float
    exit_price: Optional[float]
    pnl: Optional[float]
    strategy_id: str


@dataclass
class ShadowPerformance:
    """Performance metrics from shadow trading."""
    strategy_id: str
    start_date: datetime
    end_date: datetime
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    trade_count: int
    win_rate: float
    transfer_ratio: float  # KEY METRIC: shadow_sharpe / backtest_sharpe
    execution_residuals: List[float]  # actual_fill - predicted_fill


class ShadowEnvironment:
    """
    Paper trading environment for strategy validation.
    
    Key metrics:
    - Transfer ratio: shadow_sharpe / backtest_sharpe (target: >0.7)
    - Execution residuals: prediction error of fill prices
    - Regime behavior: consistency across market conditions
    
    Duration: Minimum 2-3 weeks before live deployment
    """
    
    def __init__(self, market_data_feed, exchange_simulator, backtest_sharpe: float):
        self.market_feed = market_data_feed
        self.exchange = exchange_simulator
        self.backtest_sharpe = backtest_sharpe
        self.trades: List[ShadowTradeRecord] = []
        self.daily_pnl: List[float] = []
        self.execution_residuals: List[float] = []
    
    async def run_shadow(
        self,
        strategy: StrategyGenome,
        duration_days: int = 21,
        capital: float = 10000
    ) -> ShadowPerformance:
        """Run strategy in shadow mode."""
        start_date = datetime.now()
        position = 0
        equity = capital
        peak_equity = capital
        max_drawdown = 0
        
        for day in range(duration_days):
            market_data = await self.market_feed.get_day_data(day)
            
            for candle in market_data:
                features = self._extract_features(candle)
                signal = strategy.evaluate(features)
                
                # Execute on signal change
                if signal != np.sign(position):
                    if position != 0:  # Close existing
                        exit_price = candle['close']
                        pnl = position * (exit_price - self.trades[-1].entry_price)
                        self.trades[-1].exit_price = exit_price
                        self.trades[-1].pnl = pnl
                        equity += pnl
                    
                    if signal != 0:  # Open new
                        size = capital * strategy.base_position_pct / candle['close']
                        actual_fill, residual = self.exchange.simulate_fill(
                            symbol=candle['symbol'],
                            side='buy' if signal > 0 else 'sell',
                            size=size,
                            expected_price=candle['close']
                        )
                        self.execution_residuals.append(residual)
                        
                        self.trades.append(ShadowTradeRecord(
                            timestamp=candle['timestamp'],
                            symbol=candle['symbol'],
                            direction=signal,
                            size=size,
                            entry_price=actual_fill,
                            exit_price=None,
                            pnl=None,
                            strategy_id=strategy.id
                        ))
                        position = signal
                
                peak_equity = max(peak_equity, equity)
                drawdown = (peak_equity - equity) / peak_equity
                max_drawdown = max(max_drawdown, drawdown)
            
            self.daily_pnl.append(equity - capital)
        
        # Calculate metrics
        returns = np.diff([capital] + [capital + p for p in self.daily_pnl]) / capital
        shadow_sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        winning_trades = [t for t in self.trades if t.pnl and t.pnl > 0]
        win_rate = len(winning_trades) / max(len(self.trades), 1)
        
        return ShadowPerformance(
            strategy_id=strategy.id,
            start_date=start_date,
            end_date=start_date + timedelta(days=duration_days),
            total_pnl=sum(t.pnl or 0 for t in self.trades),
            sharpe_ratio=shadow_sharpe,
            max_drawdown=max_drawdown,
            trade_count=len(self.trades),
            win_rate=win_rate,
            transfer_ratio=shadow_sharpe / self.backtest_sharpe if self.backtest_sharpe > 0 else 0,
            execution_residuals=self.execution_residuals
        )
    
    def _extract_features(self, candle: dict) -> dict:
        return candle  # Would use FeatureVector class
```

## 4.2 Epistemic Uncertainty Gating

Before deployment, we need confidence that the strategy's predictions are reliable—not just accurate on average, but consistently trustworthy. Epistemic uncertainty measures how much the model "knows it doesn't know."

```python
class EpistemicUncertaintyGate:
    """
    Gate deployment based on ensemble disagreement.
    
    Method: Train 5 policy variants with different random seeds/dropout.
    If they disagree significantly, the model is uncertain about its predictions.
    
    Deploy only when:
    1. Uncertainty < threshold (predictions are confident)
    2. Uncertainty is decreasing (model is converging)
    """
    
    def __init__(self, n_ensemble: int = 5, uncertainty_threshold: float = 0.10):
        self.n_ensemble = n_ensemble
        self.threshold = uncertainty_threshold
        self.uncertainty_history = []
    
    def should_deploy(self, strategy: StrategyGenome, market_data: np.ndarray) -> tuple:
        """Returns (should_deploy, uncertainty_score)."""
        predictions = []
        
        for i in range(self.n_ensemble):
            pred = self._get_ensemble_prediction(strategy, market_data, member=i)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        uncertainty = np.std(predictions, axis=0).mean()
        
        self.uncertainty_history.append(uncertainty)
        is_converging = (len(self.uncertainty_history) >= 3 and 
                        self.uncertainty_history[-1] < self.uncertainty_history[-3])
        
        should_deploy = uncertainty < self.threshold and is_converging
        return should_deploy, uncertainty
    
    def _get_ensemble_prediction(self, strategy, market_data, member: int) -> float:
        # Each ensemble member trained with different seed/dropout
        return np.random.randn() * 0.1  # Placeholder
```

## 4.3 Transfer Ratio Confidence

Historical data suggests that strategies typically achieve 60–80% of their backtested performance in live trading. The transfer ratio model predicts this degradation and recommends position sizing accordingly.

```python
class TransferRatioConfidence:
    """
    Compute confidence in strategy transfer from backtest to live.
    
    P(Sharpe_live > threshold | Sharpe_backtest, historical_data)
    
    Position sizing recommendation based on confidence:
    - 70% confidence → 50% of target size
    - 90% confidence → 100% of target size
    """
    
    def __init__(self, history: List[tuple]):
        """
        Args:
            history: List of (backtest_sharpe, live_sharpe) pairs from past strategies
        """
        self.history = history
        self._fit_model()
    
    def _fit_model(self):
        """Fit linear regression: live = alpha * backtest + beta."""
        if len(self.history) < 10:
            self.alpha, self.beta, self.residual_std = 0.7, 0.3, 0.5
            return
        
        from sklearn.linear_model import LinearRegression
        backtest = np.array([h[0] for h in self.history]).reshape(-1, 1)
        live = np.array([h[1] for h in self.history])
        
        model = LinearRegression().fit(backtest, live)
        self.alpha = model.coef_[0]
        self.beta = model.intercept_
        self.residual_std = np.std(live - model.predict(backtest))
    
    def get_confidence(self, backtest_sharpe: float, threshold: float = 1.5) -> tuple:
        """Returns (confidence, expected_live_sharpe)."""
        expected_live = self.alpha * backtest_sharpe + self.beta
        from scipy import stats
        z = (threshold - expected_live) / (self.residual_std + 1e-8)
        confidence = 1 - stats.norm.cdf(z)
        return confidence, expected_live
    
    def recommend_position_size(self, backtest_sharpe: float, base_size: float = 1.0) -> float:
        """Scale position size by confidence."""
        confidence, _ = self.get_confidence(backtest_sharpe)
        
        if confidence < 0.7:
            return base_size * 0.25
        elif confidence < 0.8:
            return base_size * 0.5
        elif confidence < 0.9:
            return base_size * 0.75
        return base_size
```

## 4.4 Live Deployment Protocol

```python
@dataclass
class DeploymentDecision:
    strategy_id: str
    approved: bool
    initial_position_pct: float
    confidence: float
    conditions: List[str]  # What must remain true for continued deployment
    monitoring_alerts: List[str]  # What triggers review


class DeploymentManager:
    """
    Manages strategy progression from shadow to live.
    
    Protocol:
    1. Shadow trading for 21+ days
    2. Transfer ratio > 0.7
    3. Epistemic uncertainty < 0.10 and converging
    4. Initial sizing: 25-50% of target
    5. Scale up if TR > 0.75 sustained for 30 days
    """
    
    def __init__(self, shadow_env, uncertainty_gate, transfer_model):
        self.shadow = shadow_env
        self.uncertainty = uncertainty_gate
        self.transfer = transfer_model
    
    async def evaluate_for_deployment(
        self,
        strategy: StrategyGenome,
        backtest_sharpe: float
    ) -> DeploymentDecision:
        """Complete deployment evaluation."""
        
        # Run shadow trading
        shadow_perf = await self.shadow.run_shadow(strategy, duration_days=21)
        
        # Check transfer ratio
        if shadow_perf.transfer_ratio < 0.7:
            return DeploymentDecision(
                strategy_id=strategy.id,
                approved=False,
                initial_position_pct=0,
                confidence=0,
                conditions=[],
                monitoring_alerts=[f"Transfer ratio {shadow_perf.transfer_ratio:.2f} < 0.7"]
            )
        
        # Check uncertainty
        should_deploy, uncertainty = self.uncertainty.should_deploy(
            strategy, np.array([])  # Would pass recent market data
        )
        
        if not should_deploy:
            return DeploymentDecision(
                strategy_id=strategy.id,
                approved=False,
                initial_position_pct=0,
                confidence=0,
                conditions=[],
                monitoring_alerts=[f"Epistemic uncertainty {uncertainty:.3f} too high"]
            )
        
        # Calculate position sizing
        confidence, expected_live = self.transfer.get_confidence(backtest_sharpe)
        position_size = self.transfer.recommend_position_size(backtest_sharpe, base_size=0.05)
        
        return DeploymentDecision(
            strategy_id=strategy.id,
            approved=True,
            initial_position_pct=position_size,
            confidence=confidence,
            conditions=[
                f"Transfer ratio > 0.6",
                f"Drawdown < 15%",
                f"Sharpe > {expected_live * 0.7:.2f}"
            ],
            monitoring_alerts=[
                "Daily TR check",
                "Weekly regime consistency check",
                "Monthly full review"
            ]
        )
```

---

# Part 5: Adaptation Systems

Markets are non-stationary—the statistical properties that made a strategy profitable last month may not exist today. The adaptation layer detects regime shifts and responds appropriately, from subtle parameter adjustments to full strategy retirement.

## 5.1 Concept Drift Detection

Concept drift occurs when the relationship between features and returns changes. HIMARI uses an ensemble of drift detectors because no single detector catches all drift types. The ensemble votes: alert only when 2+ detectors agree, reducing false positives.

```python
from river import drift
from typing import Dict, List


class DriftDetectionEnsemble:
    """
    Ensemble of drift detectors for robust regime shift detection.
    
    Monitors:
    - ADWIN: Adaptive windowing for gradual drift
    - Page-Hinkley: Sequential change detection  
    - KSWIN: Kolmogorov-Smirnov windowed detection
    
    Voting: Alert if 2+ detectors agree
    """
    
    def __init__(self):
        self.detectors = {
            "adwin": drift.ADWIN(delta=0.002),
            "page_hinkley": drift.PageHinkley(
                min_instances=30,
                delta=0.005,
                threshold=50
            ),
            "kswin": drift.KSWIN(
                alpha=0.005,
                window_size=100
            ),
        }
        self.votes_for_drift = 0
        self.alert_history: List[Dict] = []
    
    def update(self, value: float) -> Dict:
        """
        Update detectors with new observation.
        
        Args:
            value: Performance metric (rolling Sharpe, returns, etc.)
        
        Returns:
            Dict with drift status and detector votes
        """
        alerts = {}
        for name, detector in self.detectors.items():
            detector.update(value)
            alerts[name] = detector.drift_detected
        
        self.votes_for_drift = sum(alerts.values())
        confirmed_drift = self.votes_for_drift >= 2
        
        result = {
            "drift_detected": confirmed_drift,
            "detector_alerts": alerts,
            "confidence": self.votes_for_drift / len(self.detectors)
        }
        
        if confirmed_drift:
            self.alert_history.append(result)
        
        return result
    
    def reset(self):
        """Reset after confirmed regime change."""
        for detector in self.detectors.values():
            detector.reset()
        self.votes_for_drift = 0


class AdaptiveResponseManager:
    """
    Graduated response to detected drift.
    
    Response hierarchy:
    - Green (no drift): Normal operation
    - Yellow (emerging): Increase buffers, reduce positions 20%
    - Orange (confirmed): Reweight agents, reduce 50%, alert Layer 4-5
    - Red (critical): Halt trading, trigger full review
    """
    
    THRESHOLDS = {"green": 0.5, "yellow": 0.75, "orange": 0.95, "red": 1.0}
    
    def __init__(self, portfolio_manager, alerter):
        self.portfolio = portfolio_manager
        self.alerter = alerter
        self.current_level = "green"
    
    def respond(self, drift_result: Dict) -> Dict:
        """Execute response based on drift level."""
        confidence = drift_result["confidence"]
        
        if confidence < self.THRESHOLDS["green"]:
            level = "green"
        elif confidence < self.THRESHOLDS["yellow"]:
            level = "yellow"
        elif confidence < self.THRESHOLDS["orange"]:
            level = "orange"
        else:
            level = "red"
        
        actions = self._execute_response(level)
        self.current_level = level
        
        return {"level": level, "confidence": confidence, "actions_taken": actions}
    
    def _execute_response(self, level: str) -> List[str]:
        actions = []
        
        if level == "green":
            actions.append("Normal operation")
        elif level == "yellow":
            self.portfolio.increase_volatility_buffer(1.2)
            self.portfolio.reduce_positions(0.8)
            actions.extend(["Volatility buffer +20%", "Positions reduced to 80%"])
        elif level == "orange":
            self.portfolio.reduce_positions(0.5)
            self.alerter.notify_layer4_5("Drift confirmed")
            actions.extend(["Positions reduced to 50%", "Layer 4-5 alerted"])
        elif level == "red":
            self.portfolio.halt_trading()
            self.alerter.notify_human("Critical drift - trading halted")
            actions.extend(["TRADING HALTED", "Human review required"])
        
        return actions
```

## 5.2 MAML Adaptation

When drift is detected, the strategy needs to adapt quickly—but not so quickly that it forgets what worked before. Model-Agnostic Meta-Learning (MAML) provides a solution: learn an initialization that adapts well to new tasks with just a few gradient steps.

```python
import torch
import torch.nn as nn
from typing import Tuple
import copy


class MAMLAdapter:
    """
    Rapid strategy adaptation using MAML.
    
    Key insight: Instead of training from scratch, MAML learns an
    initialization that's close to optimal for many related tasks.
    When drift occurs, 3-5 gradient steps adapt to the new regime.
    
    Constraint: Only adapter layers update (base policy frozen).
    This prevents catastrophic forgetting of core trading logic.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        adapter_layers: List[str],
        inner_lr: float = 0.01,
        inner_steps: int = 5
    ):
        self.base_model = base_model
        self.adapter_layers = adapter_layers
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        
        # Freeze base, keep only adapters trainable
        for name, param in self.base_model.named_parameters():
            param.requires_grad = any(adapter in name for adapter in adapter_layers)
    
    def adapt(
        self,
        recent_data: torch.Tensor,
        recent_labels: torch.Tensor,
        validation_data: torch.Tensor,
        validation_labels: torch.Tensor
    ) -> Tuple[nn.Module, float]:
        """
        Adapt model to recent regime using few gradient steps.
        
        Returns (adapted_model, validation_score)
        """
        adapted_model = copy.deepcopy(self.base_model)
        
        # Inner loop: adapt to recent data
        inner_opt = torch.optim.SGD(
            [p for p in adapted_model.parameters() if p.requires_grad],
            lr=self.inner_lr
        )
        
        for _ in range(self.inner_steps):
            predictions = adapted_model(recent_data)
            loss = nn.MSELoss()(predictions, recent_labels)
            inner_opt.zero_grad()
            loss.backward()
            inner_opt.step()
        
        # Validate
        adapted_model.eval()
        with torch.no_grad():
            val_pred = adapted_model(validation_data)
            returns = val_pred * validation_labels
            val_score = (returns.mean() / (returns.std() + 1e-8)).item()
        
        return adapted_model, val_score
```

## 5.3 Strategy Retirement

Not all strategies can be adapted—some should be retired gracefully. Retirement triggers include sustained poor transfer ratio, negative true contribution (strategy hurts portfolio), or invalidation of the underlying causal mechanism.

```python
@dataclass
class RetirementDecision:
    strategy_id: str
    should_retire: bool
    reason: str
    recommended_action: str
    wind_down_days: int


class StrategyRetirementManager:
    """
    Manages graceful strategy retirement.
    
    Retirement triggers:
    - Transfer ratio < 0.5 sustained 30 days
    - True contribution becomes negative
    - Causal mechanism invalidated (Layer 6 research)
    - Repeated drift without successful adaptation
    
    Process: Gradual position reduction, not abrupt halt
    """
    
    def __init__(self, min_tr: float = 0.5, min_tc: float = 0.0, 
                 sustained_days: int = 30):
        self.min_tr = min_tr
        self.min_tc = min_tc
        self.sustained_days = sustained_days
        self.performance_history: Dict[str, List] = {}
    
    def evaluate_retirement(
        self,
        strategy_id: str,
        current_tr: float,
        current_tc: float,
        causal_valid: bool,
        adaptation_failures: int
    ) -> RetirementDecision:
        """Evaluate if strategy should be retired."""
        
        # Track history
        if strategy_id not in self.performance_history:
            self.performance_history[strategy_id] = []
        self.performance_history[strategy_id].append({
            "tr": current_tr, "tc": current_tc, "timestamp": time.time()
        })
        
        history = self.performance_history[strategy_id]
        
        # Check sustained poor TR
        if len(history) >= self.sustained_days:
            recent_tr = [h["tr"] for h in history[-self.sustained_days:]]
            if all(tr < self.min_tr for tr in recent_tr):
                return RetirementDecision(
                    strategy_id=strategy_id,
                    should_retire=True,
                    reason=f"Transfer ratio < {self.min_tr} for {self.sustained_days} days",
                    recommended_action="Gradual wind-down",
                    wind_down_days=14
                )
        
        # Check negative TC
        if current_tc < self.min_tc:
            return RetirementDecision(
                strategy_id=strategy_id,
                should_retire=True,
                reason=f"True contribution negative ({current_tc:.3f})",
                recommended_action="Immediate removal (hurting portfolio)",
                wind_down_days=3
            )
        
        # Check causal invalidation
        if not causal_valid:
            return RetirementDecision(
                strategy_id=strategy_id,
                should_retire=True,
                reason="Causal mechanism invalidated by Layer 6 research",
                recommended_action="Research-driven retirement",
                wind_down_days=7
            )
        
        # Check adaptation failures
        if adaptation_failures >= 3:
            return RetirementDecision(
                strategy_id=strategy_id,
                should_retire=True,
                reason=f"Failed {adaptation_failures} consecutive adaptations",
                recommended_action="Strategy fundamentally broken",
                wind_down_days=7
            )
        
        return RetirementDecision(
            strategy_id=strategy_id,
            should_retire=False,
            reason="Strategy performing adequately",
            recommended_action="Continue monitoring",
            wind_down_days=0
        )
```
# Part 6: Infrastructure Integration

The Layer 1 Explorer doesn't operate in isolation—it consumes real-time data from Layer 0, receives regime signals from Layer 4, publishes candidates to Layer 2, and queries the Knowledge Graph in Layer 5/6. This section details the data contracts, message formats, and integration patterns that connect these components.

## 6.1 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW DIAGRAM                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LAYER 0 (Data Foundation)                                                  │
│  ┌──────────────┐                                                          │
│  │ Redis        │ ──────60-dim features────▶ ┌─────────────────────┐      │
│  │ Feature Store│                            │                     │      │
│  └──────────────┘                            │   LAYER 1 EXPLORER  │      │
│                                              │                     │      │
│  LAYER 4 (Meta-Controller)                   │   • Generation      │      │
│  ┌──────────────┐                            │   • Validation      │      │
│  │ Kafka        │ ──────regime signals──────▶│   • Adaptation      │      │
│  │ Event Bus    │                            │                     │      │
│  └──────────────┘                            └──────────┬──────────┘      │
│                                                         │                  │
│  LAYER 5/6 (Knowledge/Meta-Learning)                    │                  │
│  ┌──────────────┐                                       │                  │
│  │ Neo4j        │ ◀─────archetype queries───────────────┤                  │
│  │ Knowledge    │ ──────strategy templates─────────────▶│                  │
│  │ Graph        │                                       │                  │
│  └──────────────┘                                       │                  │
│                                                         ▼                  │
│  LAYER 2 (Tactical Decisions)              ┌────────────────────────┐     │
│  ┌──────────────┐                          │ Strategy Candidates    │     │
│  │ Shadow       │◀─────candidates──────────│ (Kafka topic)          │     │
│  │ Environment  │                          └────────────────────────┘     │
│  └──────────────┘                                                          │
│         │                                                                  │
│         └─────────feedback (sharpe, TR)──────▶ Layer 1 Explorer           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 6.2 Data Interface Implementation

```python
from dataclasses import dataclass
from typing import Dict, Optional
import redis
import json
from kafka import KafkaConsumer, KafkaProducer
from neo4j import GraphDatabase


@dataclass
class Layer1Config:
    """Configuration for Layer 1 Explorer infrastructure."""
    # Redis (feature store)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Kafka (event bus)
    kafka_bootstrap: str = "localhost:9092"
    kafka_consumer_group: str = "layer1-explorer"
    
    # Topics
    feature_topic: str = "market-features"
    regime_topic: str = "regime-signals"
    shadow_feedback_topic: str = "shadow-feedback"
    candidate_output_topic: str = "strategy-candidates"
    
    # Neo4j (knowledge graph)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"


class Layer1DataInterface:
    """
    Data interface for Layer 1 Explorer.
    
    Consumes:
    - Real-time features from Layer 0 via Redis
    - Regime signals from Layer 4 via Kafka
    - Shadow feedback from Layer 2 via Kafka
    
    Produces:
    - Strategy candidates to validation pipeline via Kafka
    
    Queries:
    - Strategy archetypes and failure patterns from Neo4j
    """
    
    def __init__(self, config: Layer1Config):
        self.config = config
        
        # Redis connection
        self.redis = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db
        )
        
        # Kafka consumer
        self.consumer = KafkaConsumer(
            config.regime_topic,
            config.shadow_feedback_topic,
            bootstrap_servers=config.kafka_bootstrap,
            group_id=config.kafka_consumer_group,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        # Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=config.kafka_bootstrap,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        # Neo4j connection
        self.neo4j_driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password)
        )
    
    def get_current_features(self, symbol: str = "BTCUSDT") -> Optional[FeatureVector]:
        """Fetch 60-dimensional feature vector from Redis."""
        key = f"features:{symbol}"
        data = self.redis.hgetall(key)
        
        if not data:
            return None
        
        fv = FeatureVector()
        for name, value in data.items():
            name = name.decode() if isinstance(name, bytes) else name
            value = float(value.decode() if isinstance(value, bytes) else value)
            fv.set(name, value, int(self.redis.time()[0] * 1000))
        
        return fv
    
    def get_regime_signal(self) -> Dict:
        """Get latest regime signal from Layer 4."""
        for message in self.consumer:
            if message.topic == self.config.regime_topic:
                return message.value
        return {"regime_label": 0, "regime_confidence": 0.5}
    
    def get_shadow_feedback(self) -> Optional[Dict]:
        """Get feedback from shadow trading environment."""
        for message in self.consumer:
            if message.topic == self.config.shadow_feedback_topic:
                return message.value
        return None
    
    def publish_candidate(self, strategy: StrategyGenome):
        """Publish strategy candidate to validation pipeline."""
        payload = {
            "strategy_id": strategy.id,
            "genome_vector": strategy.to_vector().tolist(),
            "generation": strategy.generation,
            "lineage": strategy.lineage or [],
            "source_engine": getattr(strategy, 'source_engine', 'unknown'),
            "timestamp": int(time.time() * 1000)
        }
        self.producer.send(self.config.candidate_output_topic, payload)
        self.producer.flush()
    
    def query_archetypes(self, regime: str, limit: int = 10) -> list:
        """Query strategy archetypes from Knowledge Graph."""
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (s:StrategyArchetype)-[:PERFORMS_IN]->(r:Regime {name: $regime})
                WHERE s.sharpe > 1.5 AND s.max_drawdown < 0.15
                RETURN s.id, s.name, s.entry_logic, s.exit_logic, s.sharpe
                ORDER BY s.sharpe DESC
                LIMIT $limit
            """, regime=regime, limit=limit)
            
            return [dict(record) for record in result]
    
    def query_failure_patterns(self, strategy_features: list) -> list:
        """Query failure patterns similar to strategy features."""
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (f:FailurePattern)
                WHERE f.features IN $features
                RETURN f.id, f.description, f.conditions, f.severity
                ORDER BY f.severity DESC
                LIMIT 5
            """, features=strategy_features)
            
            return [dict(record) for record in result]
    
    def close(self):
        """Clean up connections."""
        self.redis.close()
        self.consumer.close()
        self.producer.close()
        self.neo4j_driver.close()
```

## 6.3 Monitoring Dashboard

Prometheus metrics for observability:

```python
from prometheus_client import Counter, Gauge, Histogram, start_http_server


class ExplorerMetrics:
    """Prometheus metrics for Layer 1 Explorer monitoring."""
    
    def __init__(self):
        # Generation metrics
        self.candidates_generated = Counter(
            'explorer_candidates_generated_total',
            'Total strategy candidates generated',
            ['engine']
        )
        
        self.generation_latency = Histogram(
            'explorer_generation_latency_seconds',
            'Time to generate a candidate',
            ['engine'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1, 5, 10]
        )
        
        # Validation metrics
        self.validation_stage_passed = Counter(
            'explorer_validation_stage_passed_total',
            'Candidates passing each stage',
            ['stage']
        )
        
        self.validation_stage_failed = Counter(
            'explorer_validation_stage_failed_total',
            'Candidates failing each stage',
            ['stage']
        )
        
        self.dsr_threshold = Gauge(
            'explorer_dsr_threshold',
            'Current DSR threshold based on trial count'
        )
        
        # Diversity metrics
        self.population_diversity = Gauge(
            'explorer_population_diversity',
            'Average pairwise distance in population'
        )
        
        self.engine_contribution = Gauge(
            'explorer_engine_contribution_ratio',
            'Fraction of approved strategies by engine',
            ['engine']
        )
        
        # Drift metrics
        self.drift_confidence = Gauge(
            'explorer_drift_confidence',
            'Current drift detection confidence'
        )
        
        self.drift_alerts = Counter(
            'explorer_drift_alerts_total',
            'Total drift alerts',
            ['level']
        )
        
        # Deployment metrics
        self.shadow_transfer_ratio = Gauge(
            'explorer_shadow_transfer_ratio',
            'Transfer ratio from shadow trading',
            ['strategy_id']
        )
        
        self.strategies_deployed = Counter(
            'explorer_strategies_deployed_total',
            'Total strategies deployed to live'
        )
    
    def start_server(self, port: int = 8000):
        """Start Prometheus metrics server."""
        start_http_server(port)


# Grafana dashboard queries (for reference)
GRAFANA_QUERIES = {
    "generation_rate": 'rate(explorer_candidates_generated_total[5m])',
    "validation_funnel": '''
        sum by (stage) (
            explorer_validation_stage_passed_total / 
            (explorer_validation_stage_passed_total + explorer_validation_stage_failed_total)
        )
    ''',
    "drift_status": 'explorer_drift_confidence',
    "diversity_trend": 'explorer_population_diversity',
}
```

---

# Part 7: Configuration and Testing

## 7.1 Configuration Template

```yaml
# config/layer1_explorer.yaml

# ═══════════════════════════════════════════════════════════════════════════
# GENERATION ENGINE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
generation:
  population_size: 100
  elite_size: 10
  candidates_per_cycle: 100
  
  evolutionary:
    mutation_rate: 0.15
    crossover_rate: 0.40
    novel_generation_rate: 0.10
    recombination_rate: 0.35
    tournament_size: 5
    neutral_drift_tolerance: 0.05
    neutral_drift_duration: 10
  
  generative:
    latent_dim: 256
    hidden_dim: 512
    num_steps: 15  # Flow matching steps (10-20 optimal)
    cfg_scale: 7.5  # Classifier-free guidance scale
    model_checkpoint: "models/flow_matching.pt"
  
  llm:
    model: "claude-sonnet-4-20250514"
    max_calls_per_cycle: 20
    temperature: 0.7
    max_tokens: 2000
  
  external:
    max_crawls_per_day: 50
    arxiv_enabled: true
    tradingview_enabled: false
    reddit_enabled: false
    novelty_threshold: 0.5

# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION PIPELINE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
validation:
  dsr:
    significance_level: 0.95
    
  surrogate:
    model_checkpoint: "models/surrogate.pt"
    top_k: 20
    
  fast_backtest:
    assets: "top20"
    window_years: 2
    min_sharpe: 1.5
    max_drawdown: 0.20
    min_trades: 50
    min_profit_factor: 1.2
    
  full_backtest:
    assets: "all"
    window_years: 5
    min_sharpe: 2.0
    max_drawdown: 0.15
    min_trades: 200
    min_profit_factor: 1.5
    min_regime_consistency: 0.6
    
  true_contribution:
    min_marginal_sharpe: 0.05
    min_orthogonality: 0.3
    min_residual_ic: 0.02
    max_correlation: 0.7
    
  neutralization:
    min_residual_sharpe: 1.0
    min_ic_retention: 0.5
    max_factor_beta: 0.5
    factors: ["market", "momentum", "size", "funding"]

# ═══════════════════════════════════════════════════════════════════════════
# DEPLOYMENT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
deployment:
  shadow:
    min_duration_days: 21
    min_transfer_ratio: 0.7
    
  uncertainty:
    n_ensemble: 5
    max_uncertainty: 0.10
    require_converging: true
    
  transfer_ratio:
    threshold: 1.5  # Live Sharpe threshold for confidence
    
  position_sizing:
    initial_pct: 0.25  # Start at 25% of target
    max_pct: 1.0
    scale_up_days: 30  # Days of TR > 0.75 before scaling

# ═══════════════════════════════════════════════════════════════════════════
# ADAPTATION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
adaptation:
  drift_detection:
    adwin_delta: 0.002
    page_hinkley_threshold: 50
    kswin_alpha: 0.005
    vote_threshold: 2  # Require 2+ detectors to agree
    
  response:
    green_threshold: 0.5
    yellow_threshold: 0.75
    orange_threshold: 0.95
    yellow_position_reduction: 0.8
    orange_position_reduction: 0.5
    
  maml:
    inner_lr: 0.01
    inner_steps: 5
    adapter_layers: ["fc3", "fc4"]  # Only adapt these layers
    
  retirement:
    min_transfer_ratio: 0.5
    min_true_contribution: 0.0
    sustained_days: 30
    max_adaptation_failures: 3
    wind_down_days: 14

# ═══════════════════════════════════════════════════════════════════════════
# INFRASTRUCTURE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
infrastructure:
  redis:
    host: "localhost"
    port: 6379
    db: 0
    
  kafka:
    bootstrap_servers: "localhost:9092"
    consumer_group: "layer1-explorer"
    topics:
      features: "market-features"
      regime: "regime-signals"
      feedback: "shadow-feedback"
      candidates: "strategy-candidates"
      
  neo4j:
    uri: "bolt://localhost:7687"
    user: "neo4j"
    password: "${NEO4J_PASSWORD}"  # From environment
    
  monitoring:
    prometheus_port: 8000
    log_level: "INFO"

# ═══════════════════════════════════════════════════════════════════════════
# BUDGET CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════
budget:
  monthly_total_usd: 100
  compute:
    gpu_hours_per_month: 50
    cpu_hours_per_month: 200
  api:
    llm_calls_per_month: 600
    max_per_call_usd: 0.10
  backtest:
    max_full_backtests_per_day: 20
    cost_per_full_backtest_usd: 0.50
```

## 7.2 Testing Framework

```python
import pytest
from unittest.mock import Mock, AsyncMock
import numpy as np


class TestGrammarValidator:
    """Unit tests for grammar validation."""
    
    def test_valid_strategy(self, grammar_validator):
        strategy = create_test_strategy()
        is_valid, errors = grammar_validator.validate(strategy)
        assert is_valid
        assert len(errors) == 0
    
    def test_type_mismatch_rejected(self, grammar_validator):
        """Cannot compare RSI (ratio) with volume (count)."""
        strategy = create_invalid_type_strategy()
        is_valid, errors = grammar_validator.validate(strategy)
        assert not is_valid
        assert "type mismatch" in errors[0].lower()
    
    def test_exceeds_depth_limit(self, grammar_validator):
        """Strategy tree depth > 5 rejected."""
        strategy = create_deep_strategy(depth=6)
        is_valid, errors = grammar_validator.validate(strategy)
        assert not is_valid


class TestDSRGate:
    """Unit tests for Deflated Sharpe Ratio calculation."""
    
    def test_high_sharpe_low_trials_passes(self):
        """SR=2.0 with 10 trials should pass."""
        result = dsr_gate(observed_sharpe=2.0, n_strategies=10)
        assert result.passed
    
    def test_high_sharpe_many_trials_fails(self):
        """SR=2.0 with 1000 trials might fail (multiple testing)."""
        result = dsr_gate(observed_sharpe=2.0, n_strategies=1000)
        # Threshold at 1000 trials is ~2.5
        assert not result.passed
    
    def test_very_high_sharpe_always_passes(self):
        """SR=4.0 should pass even with many trials."""
        result = dsr_gate(observed_sharpe=4.0, n_strategies=5000)
        assert result.passed
    
    def test_threshold_increases_with_trials(self):
        """More trials = higher threshold."""
        t1 = dsr_threshold(n=100)
        t2 = dsr_threshold(n=1000)
        t3 = dsr_threshold(n=10000)
        assert t1 < t2 < t3


class TestTrueContribution:
    """Unit tests for portfolio orthogonality check."""
    
    def test_orthogonal_strategy_accepted(self):
        """Strategy uncorrelated with portfolio should pass."""
        candidate = np.random.randn(252)
        portfolio = [np.random.randn(252) for _ in range(5)]
        
        result = calculate_true_contribution(candidate, portfolio)
        assert result["orthogonality"] > 0.5
    
    def test_duplicate_strategy_rejected(self):
        """Strategy highly correlated with existing should fail."""
        existing = np.random.randn(252)
        candidate = existing + np.random.randn(252) * 0.1  # 90%+ correlated
        
        result = calculate_true_contribution(candidate, [existing])
        assert result["orthogonality"] < 0.2


class TestEvolutionaryExplorer:
    """Integration tests for evolutionary engine."""
    
    def test_population_improves_over_generations(self):
        """Average fitness should increase over 10 generations."""
        explorer = EvolutionaryExplorer(population_size=50)
        explorer.initialize_population()
        
        initial_fitness = np.mean([s.fitness for s in explorer.population])
        
        for _ in range(10):
            explorer.evolve_generation(
                evaluator=mock_evaluator,
                grammar_validator=mock_validator
            )
        
        final_fitness = np.mean([s.fitness for s in explorer.population])
        assert final_fitness > initial_fitness
    
    def test_diversity_maintained(self):
        """Population should maintain diversity (not converge to clones)."""
        explorer = EvolutionaryExplorer(population_size=50)
        explorer.initialize_population()
        
        for _ in range(20):
            explorer.evolve_generation(mock_evaluator, mock_validator)
        
        diversity = calculate_population_diversity(explorer.population)
        assert diversity > 0.3  # At least 30% average pairwise distance


class TestFlowMatching:
    """Unit tests for flow matching generation."""
    
    def test_sample_produces_valid_vectors(self):
        """Generated vectors should be in valid range."""
        model = ConditionalFlowMatching()
        condition = GenerationCondition().to_tensor().unsqueeze(0)
        
        samples = model.sample(condition.repeat(10, 1), num_steps=15)
        
        assert samples.shape == (10, 127)
        assert not torch.isnan(samples).any()
    
    def test_conditioning_affects_output(self):
        """Different conditions should produce different distributions."""
        model = ConditionalFlowMatching()
        
        aggressive = GenerationCondition(risk_tolerance=0.9).to_tensor()
        conservative = GenerationCondition(risk_tolerance=0.1).to_tensor()
        
        agg_samples = model.sample(aggressive.unsqueeze(0).repeat(100, 1))
        con_samples = model.sample(conservative.unsqueeze(0).repeat(100, 1))
        
        # Risk parameters should differ
        assert agg_samples[:, 124].mean() != con_samples[:, 124].mean()


class TestDriftDetection:
    """Unit tests for drift detection ensemble."""
    
    def test_stable_returns_no_drift(self):
        """Stable performance should not trigger drift."""
        detector = DriftDetectionEnsemble()
        
        for _ in range(100):
            result = detector.update(np.random.randn() * 0.01 + 1.0)
        
        assert not result["drift_detected"]
    
    def test_sudden_shift_triggers_drift(self):
        """Sudden performance change should trigger drift."""
        detector = DriftDetectionEnsemble()
        
        # Stable period
        for _ in range(50):
            detector.update(1.0)
        
        # Sudden drop
        for _ in range(20):
            result = detector.update(-0.5)
        
        assert result["drift_detected"]
        assert result["confidence"] >= 0.66  # At least 2/3 detectors agree


class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_generation_validation_cycle(self):
        """Test complete generation → validation → deployment flow."""
        # Setup
        config = Layer1Config()
        orchestrator = create_test_orchestrator(config)
        pipeline = create_test_pipeline()
        
        # Generate candidates
        candidates = await orchestrator.generate_candidates(
            condition=GenerationCondition(),
            grammar_validator=GrammarValidator(),
            existing_portfolio=[]
        )
        
        assert len(candidates) > 0
        
        # Validate
        reports = []
        for candidate in candidates:
            report = pipeline.validate(candidate)
            reports.append(report)
        
        # At least some should pass grammar/DSR
        grammar_passed = sum(1 for r in reports if "grammar" in r.stages_passed)
        assert grammar_passed > len(candidates) * 0.8
        
        # Few should be fully approved
        approved = sum(1 for r in reports if r.approved)
        assert approved <= len(candidates) * 0.3  # Strict filtering
```

## 7.3 Deployment Checklist

```markdown
# Layer 1 Explorer Deployment Checklist

## Pre-Deployment
- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] Configuration validated (YAML syntax, value ranges)
- [ ] Model checkpoints available (surrogate, flow matching)
- [ ] Infrastructure connections verified (Redis, Kafka, Neo4j)
- [ ] Budget constraints configured

## Infrastructure
- [ ] Redis running and accessible
- [ ] Kafka topics created (features, regime, feedback, candidates)
- [ ] Neo4j populated with initial archetypes
- [ ] Prometheus metrics endpoint exposed
- [ ] Grafana dashboards configured

## Initial Run
- [ ] Start with reduced population (50 instead of 100)
- [ ] Disable LLM calls initially (verify evolutionary works)
- [ ] Enable verbose logging
- [ ] Monitor memory usage
- [ ] Verify Kafka message flow

## Validation
- [ ] Generate 100 candidates manually
- [ ] Run through HIFA pipeline
- [ ] Verify DSR threshold calculation
- [ ] Check surrogate model predictions
- [ ] Confirm TC calculation with known strategies

## Shadow Deployment
- [ ] Deploy first approved strategy to shadow
- [ ] Monitor for 21+ days
- [ ] Calculate transfer ratio
- [ ] Check epistemic uncertainty

## Go-Live
- [ ] Transfer ratio > 0.7 confirmed
- [ ] Position sizing calculated
- [ ] Monitoring alerts configured
- [ ] Escalation procedures documented
- [ ] Rollback plan ready
```

---

# Appendix A: Dependencies

```bash
# Core dependencies
pip install numpy scipy pandas torch scikit-learn

# Drift detection
pip install river  # For ADWIN, Page-Hinkley, KSWIN

# Genetic programming
pip install deap  # For strongly-typed GP

# Infrastructure
pip install redis kafka-python neo4j prometheus-client

# LLM integration
pip install anthropic  # For Claude API

# Backtesting
pip install vectorbt backtrader

# Monitoring
pip install grafana-api

# Testing
pip install pytest pytest-asyncio pytest-mock
```

# Appendix B: Cost Model

| Component | Cost per Unit | Monthly Usage | Monthly Cost |
|-----------|---------------|---------------|--------------|
| GPU (A10) | $0.60/hour | 50 hours | $30 |
| CPU (backtest) | $0.10/hour | 100 hours | $10 |
| LLM (Claude Sonnet) | $0.003/1K tokens | 200K tokens | $20 |
| Full Backtest | $0.50 each | 400/month | $20 |
| External APIs | varies | - | $10 |
| **Total** | | | **~$90/month** |

# Appendix C: Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Candidates/week | 100-200 | Count |
| Grammar pass rate | >95% | Stage 0 survivors / total |
| DSR pass rate | >50% | Stage 1 survivors / Stage 0 |
| Approval rate | 15-25% | Approved / total candidates |
| Transfer ratio | >0.7 | Shadow Sharpe / Backtest Sharpe |
| Time to approval | <4 hours | Generation to HIFA pass |
| Adaptation latency | <5 steps | MAML gradient steps |
