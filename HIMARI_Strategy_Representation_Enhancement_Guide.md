# HIMARI Layer 1 Explorer: Strategy Representation Enhancement Guide

**Version:** 1.0  
**Date:** January 17, 2026  
**Purpose:** Implementation guide for adding new strategy representations to existing engines  
**For Use With:** Claude Code CLI

---

## Overview

This guide adds **new strategy representations** to HIMARI's existing four-engine architecture. The existing engines remain unchanged—we're expanding what they can generate and evolve.

```
BEFORE: Engines → Decision Trees only
AFTER:  Engines → Decision Trees + FSM + Fuzzy + STGP + Bayesian Gates
```

---

## Architecture Principle

**Do NOT replace engines. Extend their output formats.**

```
┌────────────────────────────────────────────────────────────────┐
│                     EXISTING ENGINES (UNCHANGED)               │
├────────────────────────────────────────────────────────────────┤
│  Engine 1: Evolutionary    │  Engine 2: Generative AI         │
│  Engine 3: Pattern Discovery│  Engine 4: Component Recombination│
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                  NEW: STRATEGY REPRESENTATION LAYER            │
├────────────────────────────────────────────────────────────────┤
│  • FSM Position Management                                     │
│  • Fuzzy Logic Preprocessing                                   │
│  • STGP Formula Discovery                                      │
│  • Bayesian Regime Gates                                       │
│  • Linear Model Benchmark                                      │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                  EXISTING: VALIDATION PIPELINE (HIFA)          │
└────────────────────────────────────────────────────────────────┘
```

---

## Enhancement 1: Finite State Machine Position Management

### Problem Solved
Decision trees are stateless. Same input → same output, regardless of current position. Cannot implement "scale in gradually" or "hold through noise" workflows.

### Principle
Add position state memory. Strategy output depends on BOTH market signals AND current position state.

### States to Implement
```
FLAT        → No position, waiting for entry signal
LONG        → Holding long position
SHORT       → Holding short position  
SCALING_IN  → Building position incrementally
SCALING_OUT → Reducing position incrementally
STOPPED     → Recently stopped out, cooldown period
```

### Transition Logic
Each state has allowed transitions based on:
- Market signals (from existing engines)
- Time in state
- P&L since entry
- Volatility conditions

### Integration Point
- **Input:** Signal from Engine 1/2/3/4 (BUY/SELL/HOLD + confidence)
- **Output:** Action (ENTER_LONG, ADD_TO_LONG, EXIT_PARTIAL, EXIT_FULL, etc.)
- **Memory:** Current state persists between ticks

### Claude Code Instructions
```
1. Create file: src/representations/fsm_position_manager.py

2. Define FSMState enum with states: FLAT, LONG, SHORT, SCALING_IN, SCALING_OUT, STOPPED

3. Create FSMPositionManager class with:
   - current_state: FSMState
   - state_entry_time: timestamp
   - position_size: float
   - entry_price: float
   
4. Implement transition_table as dict mapping:
   (current_state, signal) → (next_state, action)
   
5. Add guard conditions as callable functions:
   - time_in_state_guard(min_seconds)
   - pnl_guard(min_pnl_percent)
   - volatility_guard(max_atr_multiple)
   
6. Implement process_signal(signal, market_data) method:
   - Look up transition from table
   - Check all guard conditions
   - Execute transition if guards pass
   - Return action tuple (action_type, size, urgency)

7. Add to Engine 1's genetic operators:
   - mutate_transition: randomly change one transition's target state
   - mutate_guard: adjust guard threshold by ±10%
   - crossover_fsm: swap transition subsets between two FSMs

8. Library: Use 'transitions' package for hierarchical FSM support
```

### Genetic Operators for FSM
- **State mutation:** Change action or next-state for one transition
- **Guard mutation:** Adjust threshold values (±10-20%)
- **Transition addition:** Add new edge between states
- **Crossover:** Exchange transition subsets at state boundaries

---

## Enhancement 2: Fuzzy Logic Preprocessing

### Problem Solved
Hard thresholds cause overfitting. RSI=69 vs RSI=71 shouldn't produce completely different signals.

### Principle
Replace crisp thresholds with gradual membership functions. Output is degree of membership (0.0-1.0), not binary.

### Membership Functions to Implement
For each indicator, define fuzzy sets:
```
RSI:
  - OVERSOLD:    Triangle(0, 0, 30)
  - NEUTRAL:     Triangle(20, 50, 80)
  - OVERBOUGHT:  Triangle(70, 100, 100)

MACD_Histogram:
  - BEARISH:     Sigmoid(center=0, slope=-5)
  - NEUTRAL:     Gaussian(mean=0, std=0.5)
  - BULLISH:     Sigmoid(center=0, slope=5)

Volume_ZScore:
  - LOW:         Triangle(-3, -3, 0)
  - NORMAL:      Triangle(-1, 0, 1)
  - HIGH:        Triangle(0, 3, 3)
```

### Fuzzy Rules Format
```
IF RSI is OVERSOLD AND Volume is HIGH THEN Signal is STRONG_BUY (weight=0.8)
IF RSI is OVERBOUGHT AND MACD is BEARISH THEN Signal is SELL (weight=0.9)
```

### Integration Point
- **Input:** Raw 60-feature vector
- **Output:** Fuzzy membership degrees for each feature + aggregated signal strength
- **Feeds into:** All four engines as preprocessed input

### Claude Code Instructions
```
1. Create file: src/representations/fuzzy_preprocessor.py

2. Define MembershipFunction base class with:
   - evaluate(crisp_value) → float [0,1]
   - get_parameters() → dict
   - set_parameters(dict)

3. Implement subclasses:
   - TriangularMF(left, center, right)
   - TrapezoidalMF(a, b, c, d)
   - GaussianMF(mean, std)
   - SigmoidMF(center, slope)

4. Create FuzzyVariable class:
   - name: str
   - universe: (min, max) range
   - membership_functions: dict[str, MembershipFunction]
   
5. Create FuzzyRule class:
   - antecedents: list of (variable, mf_name, operator)
   - consequent: (output_variable, mf_name)
   - weight: float [0,1]
   
6. Create FuzzyInferenceSystem class (Sugeno-style):
   - variables: list[FuzzyVariable]
   - rules: list[FuzzyRule]
   - defuzzify(rule_activations) → crisp output
   
7. Implement fuzzify_feature_vector(raw_features) method:
   - For each of 60 features, compute membership in each fuzzy set
   - Return expanded vector: 60 features × 3 sets = 180 fuzzy values

8. Add to Engine 1's genetic operators:
   - mutate_mf_params: shift triangle vertices by ±5-10%
   - mutate_rule_weight: adjust rule weight by ±0.1
   - crossover_rules: exchange rule subsets

9. Library: Use 'scikit-fuzzy' for Mamdani/Sugeno inference
```

### Genetic Operators for Fuzzy
- **Parameter mutation:** Shift MF vertices/centers by ±5-10%
- **Rule weight mutation:** Adjust weight by ±0.1
- **Rule addition/deletion:** Add new rule or remove low-weight rule
- **MF type mutation:** Change Triangle → Gaussian (same coverage)

---

## Enhancement 3: Strongly-Typed Genetic Programming (STGP)

### Problem Solved
Fixed decision tree structure cannot discover mathematical formulas. WorldQuant-style alphas like `Rank(Close) - Rank(MA(Open,10))` are impossible.

### Principle
Evolve variable-structure expression trees with type checking. Grammar constraints prevent invalid operations (Volume + Price rejected).

### Type System
```
Types:
  - PRICE:      Close, Open, High, Low
  - VOLUME:     Volume, OBV
  - RATIO:      RSI, Returns, Correlation
  - RANK:       Output of Rank() operator
  - BOOLEAN:    Comparison results
  - TIMESERIES: Any numeric series

Type Rules:
  - Add/Sub: same types only (PRICE+PRICE ok, PRICE+VOLUME error)
  - Mul/Div: any numeric types
  - Rank(): any numeric → RANK
  - Delay(n): preserves type
  - Correlation(x,y,n): any numeric pair → RATIO
```

### Primitive Set
```
Terminals (leaves):
  - Features: funding_rate, whale_pressure, rsi, volume_zscore, ...
  - Constants: 0.0, 0.5, 1.0, 2.0, 5, 10, 20

Functions (nodes):
  - Arithmetic: Add, Sub, Mul, Div, Max, Min
  - Transform: Rank, Abs, Sign, Log, Sqrt
  - Temporal: Delay(n), RollingMean(n), RollingStd(n), RollingMax(n)
  - Statistical: Correlation(x, y, n), ZScore(x, n)
  - Comparison: Greater, Less, Equal (return BOOLEAN)
  - Conditional: IfThenElse(bool, x, y)
```

### Integration Point
- **Output:** Executable expression tree that computes signal strength
- **Replaces:** Fixed decision tree structure in strategy genome
- **Feeds into:** FSM as signal input, validation pipeline

### Claude Code Instructions
```
1. Create file: src/representations/stgp_formula.py

2. Define DimensionType enum:
   PRICE, VOLUME, RATIO, RANK, BOOLEAN, TIMESERIES, ANY

3. Create typed primitive set using DEAP:
   - pset = gp.PrimitiveSetTyped("ALPHA", [input_types], output_type)
   
4. Register terminals with types:
   - pset.addTerminal(name="close", terminal=get_close, ret_type=PRICE)
   - pset.addTerminal(name="volume", terminal=get_volume, ret_type=VOLUME)
   - pset.addEphemeralConstant("rand_const", lambda: random.uniform(-1,1), RATIO)

5. Register primitives with type signatures:
   - pset.addPrimitive(add, [PRICE, PRICE], PRICE)
   - pset.addPrimitive(rank, [ANY], RANK)
   - pset.addPrimitive(delay, [ANY, int], ANY)
   - pset.addPrimitive(correlation, [ANY, ANY, int], RATIO)

6. Create BNF grammar validator:
   - Parses expression tree
   - Validates dimensional consistency
   - Rejects invalid combinations before evaluation

7. Implement evaluate_expression(tree, market_data) method:
   - Compile tree to callable function
   - Cache compilation for repeated evaluation
   - Return signal value

8. Add bloat control:
   - Lexicographic parsimony: (fitness, -size) as selection key
   - Max depth limit: 10 levels
   - Subtree size limit: 50 nodes

9. Configure genetic operators:
   - gp.cxOnePointLeafBiased with type checking
   - gp.mutUniform with type-compatible replacement
   - gp.mutShrink for bloat control

10. Library: DEAP with PrimitiveSetTyped
```

### Genetic Operators for STGP
- **Subtree crossover:** Type-compatible subtree exchange
- **Point mutation:** Replace node with same-signature alternative
- **Shrink mutation:** Replace subtree with terminal (bloat control)
- **Hoist mutation:** Replace tree with one of its subtrees

---

## Enhancement 4: Bayesian Network Regime Gates

### Problem Solved
Strategies lack uncertainty quantification. Cannot express "75% confident this is bullish regime" or disable trading during uncertain periods.

### Principle
Build probabilistic graphical model for regime detection. Gate strategy execution based on regime confidence.

### Network Structure
```
        ┌─────────┐     ┌─────────┐
        │Volatility│     │  Trend  │
        │ Regime  │     │ Regime  │
        └────┬────┘     └────┬────┘
             │               │
             ▼               ▼
        ┌─────────────────────────┐
        │     Market Regime       │
        │  (RISK_ON / RISK_OFF /  │
        │   TRANSITIONING)        │
        └───────────┬─────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │   Strategy Gate         │
        │  (ALLOW / REDUCE / BLOCK)│
        └─────────────────────────┘
```

### Observable Nodes (Evidence)
```
- VIX_level: LOW / MEDIUM / HIGH
- Trend_strength: WEAK / MODERATE / STRONG  
- Volume_profile: DECLINING / STABLE / INCREASING
- Funding_rate: NEGATIVE / NEUTRAL / POSITIVE
- Correlation_regime: DECORRELATED / NORMAL / HIGHLY_CORRELATED
```

### Integration Point
- **Input:** Current market observations (discretized from 60 features)
- **Output:** P(regime=X) for each regime + recommended gate action
- **Gates:** Block trades when P(RISK_OFF) > 0.7 or P(TRANSITIONING) > 0.5

### Claude Code Instructions
```
1. Create file: src/representations/bayesian_regime_gate.py

2. Define regime states as enums:
   - VolatilityRegime: LOW, MEDIUM, HIGH
   - TrendRegime: BEAR, NEUTRAL, BULL
   - MarketRegime: RISK_ON, RISK_OFF, TRANSITIONING
   - GateAction: ALLOW, REDUCE_SIZE, BLOCK

3. Create BayesianRegimeGate class:
   - model: pgmpy.models.BayesianNetwork
   - discretizers: dict mapping feature → bins
   - gate_thresholds: dict mapping regime → action threshold

4. Define network structure:
   - Add nodes for each observable and hidden state
   - Add edges encoding causal relationships
   - Initialize CPDs (Conditional Probability Distributions)

5. Implement discretize_features(raw_features) method:
   - Convert continuous features to discrete bins
   - Return evidence dict for inference

6. Implement infer_regime(evidence) method:
   - Use variable elimination for exact inference
   - Return dict of P(regime=X) for each regime

7. Implement get_gate_action(regime_probs) method:
   - If P(RISK_OFF) > 0.7: return BLOCK
   - If P(TRANSITIONING) > 0.5: return REDUCE_SIZE
   - Else: return ALLOW

8. Add CPD learning from historical data:
   - Maximum likelihood estimation from labeled regime periods
   - Bayesian estimation with Dirichlet priors for sparse data

9. Add structure learning (optional):
   - Hill climbing with BIC scoring
   - Constraint: max 3-4 parents per node (prevents CPD explosion)

10. Library: pgmpy for BN construction and inference
```

### Genetic Operators for Bayesian Networks
- **CPD mutation:** Adjust probability values by ±0.05
- **Edge mutation:** Add/remove/reverse edge (maintain DAG)
- **Threshold mutation:** Adjust gate thresholds
- **Structure crossover:** Exchange subgraph between networks

---

## Enhancement 5: Linear Model Benchmark

### Problem Solved
No baseline to measure whether complex strategies actually beat simple weighted indicators. Need fast fallback for production.

### Principle
Maintain simple linear model trained on same features. Serves as both benchmark AND production fallback if complex models fail.

### Model Types
```
1. Ridge Regression (L2): Shrinks all coefficients, handles multicollinearity
2. Lasso Regression (L1): Sparse feature selection, zeroes irrelevant features
3. Elastic Net: Combines L1+L2, best for correlated features
4. Logistic Regression: For classification (BUY/SELL/HOLD)
```

### Integration Point
- **Training:** Fit on same labeled data as GP/NN models
- **Benchmark:** Compare strategy Sharpe vs linear model Sharpe
- **Fallback:** If main strategy confidence < threshold, use linear signal
- **Latency:** 100-500ns (orders of magnitude faster than trees)

### Claude Code Instructions
```
1. Create file: src/representations/linear_benchmark.py

2. Create LinearBenchmark class:
   - models: dict[str, sklearn model]  # ridge, lasso, elastic, logistic
   - feature_selector: SelectKBest or RFECV
   - scaler: StandardScaler
   - coefficients_history: list for tracking stability

3. Implement fit(X, y, model_type='elastic') method:
   - Scale features
   - Fit selected model with cross-validation for hyperparameters
   - Store coefficients for interpretability

4. Implement predict(features) method:
   - Scale input features
   - Return prediction + confidence interval
   - Latency target: <1 microsecond

5. Implement get_feature_importance() method:
   - Return sorted list of (feature_name, coefficient, abs_coefficient)
   - Highlight statistically significant features

6. Implement compare_to_strategy(strategy_returns, linear_returns) method:
   - Compute Sharpe ratio for both
   - Compute information ratio of strategy vs linear
   - Return "BEATS_BENCHMARK" / "MATCHES" / "UNDERPERFORMS"

7. Add online learning capability:
   - Implement partial_fit() for SGD-based updates
   - Track coefficient drift over time
   - Alert if coefficients change significantly

8. Add regularization path analysis:
   - Plot coefficient values vs lambda
   - Identify stable features (non-zero across lambda range)

9. Library: scikit-learn (Ridge, Lasso, ElasticNet, SGDRegressor)
```

### Usage in Validation Pipeline
```
# In HIFA validation, add benchmark comparison step
linear_sharpe = linear_benchmark.evaluate(strategy_period)
strategy_sharpe = candidate_strategy.evaluate(strategy_period)

# Require strategy to beat benchmark by margin
if strategy_sharpe < linear_sharpe + MIN_IMPROVEMENT:
    reject("Does not beat linear benchmark")
```

---

## Integration: Combined Strategy Representation

### New Strategy Genome Structure
```python
StrategyGenome:
  # Signal Generation (choose one or combine)
  signal_type: "tree" | "stgp" | "fuzzy_rules"
  signal_spec: DecisionTree | ExpressionTree | FuzzyRuleSet
  
  # Preprocessing (optional)
  fuzzy_preprocessing: bool
  fuzzy_config: FuzzyPreprocessorConfig | None
  
  # Position Management (required)
  position_manager: "simple" | "fsm"
  fsm_config: FSMConfig | None
  
  # Regime Gating (optional)
  regime_gate: bool
  bayesian_config: BayesianGateConfig | None
  
  # Metadata
  created_by_engine: 1 | 2 | 3 | 4
  generation: int
  lineage: list[str]  # parent strategy IDs
```

### Execution Flow
```
1. Raw Features (60-dim) arrive from Layer 0
                │
                ▼
2. [Optional] Fuzzy Preprocessing
   - Expand to fuzzy memberships
   - Output: 180-dim fuzzy vector OR original 60-dim
                │
                ▼
3. Signal Generation
   - Decision Tree: threshold-based rules
   - STGP Formula: evaluate expression tree
   - Fuzzy Rules: fuzzy inference
   - Output: signal_strength [-1, +1], confidence [0, 1]
                │
                ▼
4. [Optional] Bayesian Regime Gate
   - Infer current regime probabilities
   - Gate decision: ALLOW / REDUCE / BLOCK
   - Output: gated_signal, gate_action
                │
                ▼
5. FSM Position Manager
   - Input: signal + current_state
   - Output: action (ENTER, EXIT, SCALE, HOLD)
                │
                ▼
6. Forward to Layer 2 Execution
```

### Claude Code Instructions for Integration
```
1. Create file: src/representations/combined_strategy.py

2. Create CombinedStrategy class that orchestrates all components:
   - preprocessor: FuzzyPreprocessor | None
   - signal_generator: DecisionTree | STGPFormula | FuzzyRuleSet
   - regime_gate: BayesianRegimeGate | None
   - position_manager: SimpleManager | FSMPositionManager

3. Implement execute(features, current_state) method:
   - Step through execution flow above
   - Return (action, size, metadata)

4. Implement to_genome() and from_genome() for serialization

5. Update Engine 1 (Evolutionary) to:
   - Handle new genome structure
   - Apply appropriate genetic operators per component
   - Maintain type consistency across components

6. Update Engine 2 (Generative) to:
   - Generate STGP formulas via constrained sampling
   - Generate FSM configurations
   - Generate fuzzy rule sets

7. Update validation pipeline (HIFA) to:
   - Accept new strategy format
   - Compare against linear benchmark
   - Track which representation types succeed

8. Add representation statistics tracking:
   - Approval rate by signal_type
   - Transfer ratio by position_manager type
   - Sharpe improvement vs linear benchmark
```

---

## File Structure

```
src/representations/
├── __init__.py
├── fsm_position_manager.py      # Enhancement 1
├── fuzzy_preprocessor.py        # Enhancement 2
├── stgp_formula.py              # Enhancement 3
├── bayesian_regime_gate.py      # Enhancement 4
├── linear_benchmark.py          # Enhancement 5
├── combined_strategy.py         # Integration layer
└── genetic_operators/
    ├── __init__.py
    ├── fsm_operators.py         # FSM-specific mutation/crossover
    ├── fuzzy_operators.py       # Fuzzy MF/rule operators
    ├── stgp_operators.py        # GP tree operators
    └── bayesian_operators.py    # BN structure/parameter operators
```

---

## Testing Checklist

### Unit Tests
- [ ] FSM transitions correctly based on signal + state
- [ ] Fuzzy membership functions compute correct degrees
- [ ] STGP type system rejects invalid expressions
- [ ] Bayesian inference returns valid probabilities
- [ ] Linear model achieves sub-microsecond inference

### Integration Tests
- [ ] Combined strategy executes full pipeline
- [ ] Genetic operators produce valid offspring
- [ ] Serialization/deserialization preserves strategy
- [ ] Benchmark comparison works correctly

### Performance Tests
- [ ] End-to-end latency < 20ms for combined strategy
- [ ] FSM transition < 1μs
- [ ] Fuzzy preprocessing < 1ms
- [ ] STGP evaluation < 5ms
- [ ] Bayesian inference < 10ms
- [ ] Linear prediction < 1μs

---

## Dependencies to Add

```
# requirements.txt additions
transitions>=0.9.0      # FSM library
scikit-fuzzy>=0.4.2     # Fuzzy logic
deap>=1.4.1             # Genetic programming
pgmpy>=0.1.24           # Bayesian networks
```

---

## Success Metrics

| Metric | Current | Target | How Enhancement Helps |
|--------|---------|--------|----------------------|
| Approval Rate | 5-15% | 20-30% | Fuzzy reduces overfitting, STGP finds better formulas |
| Transfer Ratio | 0.7 | 0.85+ | FSM enables robust position management |
| Strategy Diversity | Low | High | Multiple representation types prevent monoculture |
| Latency | 10-50ms | 10-20ms | Linear benchmark provides fast fallback |
| Interpretability | 7/10 | 8/10 | All representations maintain explainability |

---

## Implementation Order

```
Week 1:
  Day 1-2: FSM Position Manager + unit tests
  Day 3-4: Fuzzy Preprocessor + unit tests
  Day 5: Integration of FSM + Fuzzy with existing Engine 1

Week 2:
  Day 1-3: STGP Formula with type system + genetic operators
  Day 4: Linear Benchmark + comparison logic
  Day 5: Integration of STGP + Linear with validation pipeline

Week 3:
  Day 1-2: Bayesian Regime Gate + inference
  Day 3: Combined Strategy orchestration
  Day 4-5: End-to-end testing + performance optimization
```

---

## Notes for Claude Code

When implementing, remember:
1. **Keep existing engines untouched** - only add new representation handlers
2. **Type safety is critical** for STGP - invalid expressions waste compute
3. **Latency budget:** Total must stay under 50ms, target 20ms
4. **Test each component independently** before integration
5. **Track statistics** on which representations perform best - this informs future evolution

---

*End of Guide*
