# HIMARI Layer 1 Explorer: Strategy Representation Layer

## Implementation Summary

**Date:** January 17, 2026
**Version:** 1.0
**Status:** Complete - All Tests Passing

---

## Overview

This implementation adds a new **Strategy Representation Layer** to HIMARI's Layer 1 Explorer Agent. The layer provides multiple strategy representations that extend the existing four-engine architecture without modifying the core engines.

```
BEFORE: Engines → Decision Trees only
AFTER:  Engines → Decision Trees + FSM + Fuzzy + STGP + Bayesian Gates + Linear Benchmark
```

---

## Files Created

### Core Implementation (`src/representations/`)

| File | Lines | Description |
|------|-------|-------------|
| `__init__.py` | ~100 | Module exports and package initialization |
| `fsm_position_manager.py` | ~380 | Finite State Machine for stateful position management |
| `fuzzy_preprocessor.py` | ~520 | Fuzzy logic preprocessing with membership functions |
| `stgp_formula.py` | ~520 | Strongly-typed genetic programming for formula discovery |
| `bayesian_regime_gate.py` | ~480 | Bayesian network for regime detection and gating |
| `linear_benchmark.py` | ~400 | Linear models for benchmarking and fast fallback |
| `combined_strategy.py` | ~480 | Integration layer orchestrating all components |

### Tests (`tests/`)

| File | Description |
|------|-------------|
| `test_representations.py` | Comprehensive test suite for all 6 components |

---

## Component Details

### 1. FSM Position Manager (`fsm_position_manager.py`)

**Purpose:** Add position state memory to enable workflows like "scale in gradually" or "hold through noise."

**States:**
- `FLAT` - No position, waiting for entry
- `LONG` - Holding long position
- `SHORT` - Holding short position
- `SCALING_IN` - Building position incrementally
- `SCALING_OUT` - Reducing position incrementally
- `STOPPED` - Cooldown after stop-out

**Features:**
- Transition table with guard conditions
- Time-in-state, P&L, and volatility guards
- Genetic operators for evolution (mutate_transition, mutate_guard, crossover_fsm)
- Full serialization support

---

### 2. Fuzzy Preprocessor (`fuzzy_preprocessor.py`)

**Purpose:** Replace hard thresholds with gradual membership functions to reduce overfitting.

**Membership Functions:**
- `TriangularMF` - Triangle shape (left, center, right)
- `TrapezoidalMF` - Trapezoid shape (a, b, c, d)
- `GaussianMF` - Bell curve (mean, std)
- `SigmoidMF` - S-curve (center, slope)

**Features:**
- FuzzyVariable class with multiple membership functions
- FuzzyRule for IF-THEN fuzzy logic
- Sugeno-style inference system
- Feature vector expansion (60 features × 3 sets = 180 fuzzy values)
- Genetic operators (mutate_mf_params, mutate_rule_weight, crossover_rules)

---

### 3. STGP Formula (`stgp_formula.py`)

**Purpose:** Evolve variable-structure expression trees with type checking for WorldQuant-style alpha formulas.

**Type System:**
- `PRICE` - Close, Open, High, Low
- `VOLUME` - Volume, OBV
- `RATIO` - RSI, Returns, normalized values
- `RANK` - Output of Rank() operator
- `BOOLEAN` - Comparison results
- `TIMESERIES` - Any numeric series
- `INTEGER` - Constants for window sizes

**Primitives (27 total):**
- Arithmetic: +, -, *, /, max, min
- Transform: rank, abs, sign, log, sqrt, neg
- Temporal: delay, rolling_mean, rolling_std, rolling_max, rolling_min
- Comparison: >, <, ==
- Conditional: if_then_else

**Features:**
- Type-safe primitive set with 19 terminals
- Random formula generation with depth/size limits
- Genetic operators (crossover, mutate_uniform, mutate_shrink, mutate_hoist)
- Bloat control via lexicographic parsimony

---

### 4. Bayesian Regime Gate (`bayesian_regime_gate.py`)

**Purpose:** Probabilistic regime detection to gate strategy execution based on market conditions.

**Network Structure:**
```
Observable Nodes          Hidden Nodes              Output
─────────────────         ────────────              ──────
VIX Level         ─┬─►  Volatility Regime  ─┐
Volume Profile    ─┘                         ├─►  Market Regime  ─►  Gate Action
Trend Strength    ─┬─►  Trend Regime      ─┤      (RISK_ON/OFF/    (ALLOW/REDUCE/
Funding Sentiment ─┘                         │       TRANSITIONING)   BLOCK)
Correlation       ─────────────────────────┘
```

**Features:**
- 8-variable Bayesian network
- Variable elimination for exact inference
- Automatic feature discretization
- Configurable gate thresholds
- Genetic operators (mutate_cpd, mutate_threshold, crossover)

---

### 5. Linear Benchmark (`linear_benchmark.py`)

**Purpose:** Simple linear models for baseline comparison and fast production fallback.

**Model Types:**
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 - sparse feature selection)
- Elastic Net (L1 + L2 combined)
- Logistic Regression (for classification)

**Features:**
- Gradient descent training with regularization
- Sub-microsecond inference latency target
- Feature importance analysis
- Sharpe ratio computation
- Strategy vs benchmark comparison
- Online learning (partial_fit)
- Coefficient drift detection

---

### 6. Combined Strategy (`combined_strategy.py`)

**Purpose:** Orchestrate all components into a unified execution pipeline.

**Execution Flow:**
```
1. Raw Features (60-dim)
        │
        ▼
2. [Optional] Fuzzy Preprocessing → 180-dim fuzzy vector
        │
        ▼
3. Signal Generation (Decision Tree / STGP / Fuzzy Rules)
   Output: signal_strength [-1, +1], confidence [0, 1]
        │
        ▼
4. [Optional] Bayesian Regime Gate
   Output: gated_signal, gate_action
        │
        ▼
5. FSM Position Manager
   Output: action (ENTER, EXIT, SCALE, HOLD)
        │
        ▼
6. Forward to Layer 2 Execution
```

**Features:**
- StrategyGenome for complete strategy serialization
- StrategyFactory for creating strategies
- StrategyStatistics for tracking performance by representation type
- Support for 3 signal types: DECISION_TREE, STGP, FUZZY_RULES
- Support for 2 position manager types: SIMPLE, FSM

---

## Test Results

All 7 test suites pass:

```
============================================================
TEST SUMMARY
============================================================
  FSM Position Manager: [PASS]
  Fuzzy Preprocessor: [PASS]
  STGP Formula: [PASS]
  Bayesian Regime Gate: [PASS]
  Linear Benchmark: [PASS]
  Combined Strategy: [PASS]
  Integration Pipeline: [PASS]
------------------------------------------------------------
  Total: 7/7 tests passed

*** ALL TESTS PASSED! ***
```

---

## Usage Example

```python
from src.representations import (
    CombinedStrategy, StrategyGenome, SignalType, PositionManagerType
)

# Create a strategy genome
genome = StrategyGenome(
    signal_type=SignalType.STGP,
    fuzzy_preprocessing=True,
    position_manager=PositionManagerType.FSM,
    regime_gate=True,
)

# Instantiate the strategy
strategy = CombinedStrategy(genome)

# Execute with market features
features = {
    'close': 50000,
    'rsi': 35,
    'macd_histogram': 20,
    'vix': 18,
    'funding_rate': 0.0001,
    'volume_zscore': 1.2,
    # ... other features
}

result = strategy.execute(features)
print(f"Action: {result.action.name}")
print(f"Signal: {result.signal_strength}")
print(f"Gate: {result.gate_action.name}")
print(f"FSM State: {result.fsm_state}")
```

---

## Dependencies

No additional external dependencies required. All implementations use Python standard library only.

Optional dependencies for enhanced functionality (not required):
```
transitions>=0.9.0      # Alternative FSM library
scikit-fuzzy>=0.4.2     # Alternative fuzzy logic
deap>=1.4.1             # Alternative genetic programming
pgmpy>=0.1.24           # Alternative Bayesian networks
```

---

## Integration with Existing Engines

The representation layer is designed to be used BY the existing engines:

- **Engine 1 (Evolutionary):** Uses genetic operators to evolve FSM configs, fuzzy rules, STGP formulas
- **Engine 2 (Generative AI):** Generates STGP expressions, FSM configurations, fuzzy rule sets
- **Engine 3 (Pattern Discovery):** Feeds patterns into Bayesian regime detection
- **Engine 4 (Component Recombination):** Recombines strategy components across representations

---

## Next Steps

1. Integrate with existing Engine 1 genetic algorithm
2. Add HIFA validation pipeline support for new representations
3. Implement representation-specific crossover between engines
4. Add performance benchmarking against linear baseline
5. Create genetic operators submodule for advanced evolution

---

*Implementation by Claude Code CLI - January 17, 2026*
