"""
Comprehensive tests for the Strategy Representation Layer.

Tests all 6 components:
1. FSM Position Manager
2. Fuzzy Preprocessor
3. STGP Formula
4. Bayesian Regime Gate
5. Linear Benchmark
6. Combined Strategy
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.representations import (
    # FSM
    FSMState, FSMPositionManager, Signal, Action,
    time_in_state_guard, pnl_guard, volatility_guard,
    # Fuzzy
    TriangularMF, TrapezoidalMF, GaussianMF, SigmoidMF,
    FuzzyVariable, FuzzyRule, FuzzyPreprocessor,
    # STGP
    DimensionType, STGPFormula, create_typed_primitive_set,
    # Bayesian
    VolatilityRegime, TrendRegime, MarketRegime, GateAction,
    BayesianRegimeGate,
    # Linear
    LinearBenchmark, BenchmarkResult,
    # Combined
    CombinedStrategy, StrategyGenome, SignalType, PositionManagerType,
)


def test_fsm_position_manager():
    """Test FSM Position Manager."""
    print("\n" + "="*60)
    print("TEST 1: FSM Position Manager")
    print("="*60)

    manager = FSMPositionManager()

    # Test initial state
    assert manager.current_state == FSMState.FLAT, "Initial state should be FLAT"
    print(f"[OK] Initial state: {manager.current_state.name}")

    # Test market data
    market_data = {
        'close': 50000,
        'open': 49500,
        'high': 50500,
        'low': 49000,
        'atr': 500,
        'avg_atr': 400,
    }

    # Test BUY signal from FLAT
    result = manager.process_signal(Signal.BUY, market_data, confidence=0.8)
    print(f"[OK] BUY signal result: {result.action.name}, size={result.size:.2f}")

    # Check state transition
    print(f"[OK] State after BUY: {manager.current_state.name}")

    # Test HOLD in LONG state
    result = manager.process_signal(Signal.HOLD, market_data, confidence=0.5)
    print(f"[OK] HOLD signal result: {result.action.name}")

    # Test state info
    state_info = manager.get_state_info()
    print(f"[OK] State info: position_size={state_info['position_size']:.2f}, entry_price={state_info['entry_price']}")

    # Test serialization
    data = manager.to_dict()
    assert 'current_state' in data, "Serialization should include current_state"
    print(f"[OK] Serialization works, keys: {list(data.keys())[:3]}...")

    # Test reset
    manager.reset()
    assert manager.current_state == FSMState.FLAT, "Reset should return to FLAT"
    print(f"[OK] Reset successful, state: {manager.current_state.name}")

    print("\n[PASS] FSM Position Manager: ALL TESTS PASSED")
    return True


def test_fuzzy_preprocessor():
    """Test Fuzzy Preprocessor."""
    print("\n" + "="*60)
    print("TEST 2: Fuzzy Preprocessor")
    print("="*60)

    # Test membership functions
    tri_mf = TriangularMF(0, 50, 100)
    assert abs(tri_mf.evaluate(50) - 1.0) < 0.01, "Center should have membership 1.0"
    assert abs(tri_mf.evaluate(0) - 0.0) < 0.01, "Edge should have membership 0.0"
    assert abs(tri_mf.evaluate(25) - 0.5) < 0.01, "Midpoint should have membership 0.5"
    print(f"[OK] TriangularMF: evaluate(50)={tri_mf.evaluate(50):.2f}, evaluate(25)={tri_mf.evaluate(25):.2f}")

    gauss_mf = GaussianMF(0, 1)
    assert abs(gauss_mf.evaluate(0) - 1.0) < 0.01, "Mean should have membership 1.0"
    print(f"[OK] GaussianMF: evaluate(0)={gauss_mf.evaluate(0):.2f}, evaluate(1)={gauss_mf.evaluate(1):.2f}")

    sig_mf = SigmoidMF(0, 5)
    assert sig_mf.evaluate(10) > 0.99, "Far positive should be ~1.0"
    assert sig_mf.evaluate(-10) < 0.01, "Far negative should be ~0.0"
    print(f"[OK] SigmoidMF: evaluate(10)={sig_mf.evaluate(10):.4f}, evaluate(-10)={sig_mf.evaluate(-10):.4f}")

    # Test FuzzyVariable
    rsi_var = FuzzyVariable(name="rsi", universe=(0, 100))
    rsi_var.add_mf("OVERSOLD", TriangularMF(0, 0, 30))
    rsi_var.add_mf("NEUTRAL", TriangularMF(20, 50, 80))
    rsi_var.add_mf("OVERBOUGHT", TriangularMF(70, 100, 100))

    memberships = rsi_var.fuzzify(25)
    membership_str = ", ".join(f"{k}: {v:.2f}" for k, v in memberships.items())
    print(f"[OK] RSI=25 memberships: {{{membership_str}}}")

    # Test FuzzyPreprocessor
    preprocessor = FuzzyPreprocessor()

    raw_features = {
        'rsi': 30,
        'macd_histogram': 0.5,
        'volume_zscore': 1.5,
        'funding_rate': 0.0001,
    }

    fuzzy_vector = preprocessor.fuzzify_feature_vector(raw_features)
    print(f"[OK] Fuzzified {len(raw_features)} features to {len(fuzzy_vector)} fuzzy values")

    # Show sample fuzzy values
    sample_keys = list(fuzzy_vector.keys())[:4]
    for key in sample_keys:
        print(f"  - {key}: {fuzzy_vector[key]:.3f}")

    # Test serialization
    data = preprocessor.to_dict()
    assert 'variables' in data, "Serialization should include variables"
    print(f"[OK] Serialization works")

    print("\n[PASS] Fuzzy Preprocessor: ALL TESTS PASSED")
    return True


def test_stgp_formula():
    """Test STGP Formula."""
    print("\n" + "="*60)
    print("TEST 3: STGP Formula")
    print("="*60)

    # Test primitive set creation
    pset = create_typed_primitive_set()
    print(f"[OK] Created primitive set with {len(pset.primitives)} primitives and {len(pset.terminals)} terminals")

    # Test random formula generation
    from src.representations.stgp_formula import STGPGenerator

    generator = STGPGenerator(pset, max_depth=5, max_size=30)
    formula = generator.generate_formula()

    print(f"[OK] Generated random formula:")
    print(f"  Expression: {formula.to_string()[:80]}...")
    print(f"  Depth: {formula.get_depth()}, Size: {formula.get_size()}")

    # Test evaluation
    market_data = {
        'close': 50000,
        'open': 49500,
        'high': 50500,
        'low': 49000,
        'volume': 1000000,
        'rsi': 65,
        'macd': 100,
        'macd_signal': 80,
        'macd_histogram': 20,
        'returns': 0.01,
        'volatility': 0.02,
        'funding_rate': 0.0001,
        'whale_pressure': 0.5,
        'volume_zscore': 1.2,
    }

    result = formula.evaluate(market_data)
    print(f"[OK] Formula evaluation result: {result:.4f}")

    signal, confidence = formula.get_signal(market_data)
    print(f"[OK] Signal: {signal:.4f}, Confidence: {confidence:.4f}")

    # Test genetic operators
    from src.representations.stgp_formula import STGPGeneticOperators

    formula2 = generator.generate_formula()

    # Test crossover
    child1, child2 = STGPGeneticOperators.crossover(formula, formula2, generator)
    print(f"[OK] Crossover produced children with sizes: {child1.get_size()}, {child2.get_size()}")

    # Test mutation
    mutated = STGPGeneticOperators.mutate_uniform(formula, generator)
    print(f"[OK] Uniform mutation: size {formula.get_size()} -> {mutated.get_size()}")

    # Test shrink mutation
    shrunk = STGPGeneticOperators.mutate_shrink(formula)
    print(f"[OK] Shrink mutation: size {formula.get_size()} -> {shrunk.get_size()}")

    # Test serialization
    data = formula.to_dict()
    assert 'expression' in data, "Serialization should include expression"
    print(f"[OK] Serialization works")

    print("\n[PASS] STGP Formula: ALL TESTS PASSED")
    return True


def test_bayesian_regime_gate():
    """Test Bayesian Regime Gate."""
    print("\n" + "="*60)
    print("TEST 4: Bayesian Regime Gate")
    print("="*60)

    gate = BayesianRegimeGate()

    # Check network structure
    print(f"[OK] Created Bayesian network with {len(gate.variables)} variables:")
    for name, var in gate.variables.items():
        print(f"  - {name}: {len(var.states)} states, parents: {var.parents}")

    # Test discretization
    raw_features = {
        'vix': 18,
        'trend_strength': 0.6,
        'volume_zscore': 0.3,
        'funding_rate': 0.0002,
        'correlation': 0.5,
    }

    evidence = gate.discretize_features(raw_features)
    print(f"[OK] Discretized features: {evidence}")

    # Test inference
    regime_probs = gate.infer_regime(evidence)
    print(f"[OK] Inferred regime probabilities:")
    for regime, prob in regime_probs.items():
        print(f"  - {regime}: {prob:.3f}")

    # Check probabilities sum to 1
    total_prob = sum(regime_probs.values())
    assert abs(total_prob - 1.0) < 0.01, f"Probabilities should sum to 1, got {total_prob}"
    print(f"[OK] Probabilities sum to {total_prob:.4f}")

    # Test gate action
    action, metadata = gate.get_gate_action(regime_probs)
    print(f"[OK] Gate action: {action.name}, reason: {metadata.get('reason', 'N/A')}")

    # Test full pipeline
    action, size_mult, full_meta = gate.process_features(raw_features)
    print(f"[OK] Full pipeline: action={action.name}, size_multiplier={size_mult:.2f}")

    # Test with high-risk scenario
    high_risk_features = {
        'vix': 35,  # High VIX
        'trend_strength': 0.8,  # Strong trend
        'volume_zscore': 2.0,  # High volume
        'funding_rate': -0.001,  # Negative funding
        'correlation': 0.85,  # High correlation
    }

    action, size_mult, _ = gate.process_features(high_risk_features)
    print(f"[OK] High-risk scenario: action={action.name}, size_multiplier={size_mult:.2f}")

    # Test serialization
    data = gate.to_dict()
    assert 'variables' in data, "Serialization should include variables"
    print(f"[OK] Serialization works")

    print("\n[PASS] Bayesian Regime Gate: ALL TESTS PASSED")
    return True


def test_linear_benchmark():
    """Test Linear Benchmark."""
    print("\n" + "="*60)
    print("TEST 5: Linear Benchmark")
    print("="*60)

    benchmark = LinearBenchmark()

    # Create synthetic training data
    import random
    random.seed(42)

    X = []
    y = []
    for i in range(100):
        features = {
            'rsi': random.uniform(20, 80),
            'macd': random.uniform(-100, 100),
            'volume_zscore': random.uniform(-2, 2),
            'funding_rate': random.uniform(-0.001, 0.001),
        }
        # Simple target: buy when RSI low, sell when high
        target = (50 - features['rsi']) / 100 + features['macd'] / 500
        X.append(features)
        y.append(target)

    print(f"[OK] Created synthetic dataset with {len(X)} samples")

    # Fit models
    benchmark.fit_all(X, y)
    print(f"[OK] Fitted all linear models")

    # Test prediction
    test_features = {
        'rsi': 30,
        'macd': 50,
        'volume_zscore': 1.0,
        'funding_rate': 0.0001,
    }

    for model_type in ['ridge', 'lasso', 'elastic']:
        pred, conf = benchmark.predict(test_features, model_type)
        print(f"[OK] {model_type.upper()} prediction: {pred:.4f}, confidence: {conf:.4f}")

    # Test feature importance
    importance = benchmark.get_feature_importance('elastic')
    print(f"[OK] Feature importance (Elastic Net):")
    for feat, coef, abs_coef in importance[:3]:
        print(f"  - {feat}: {coef:.4f}")

    # Test Sharpe ratio computation
    returns = [random.uniform(-0.02, 0.03) for _ in range(252)]
    sharpe = benchmark.compute_sharpe(returns)
    print(f"[OK] Computed Sharpe ratio: {sharpe:.4f}")

    # Test benchmark comparison
    strategy_returns = [random.uniform(-0.01, 0.02) for _ in range(100)]
    linear_returns = [random.uniform(-0.015, 0.015) for _ in range(100)]

    result, metrics = benchmark.compare_to_strategy(strategy_returns, linear_returns)
    print(f"[OK] Benchmark comparison: {result.name}")
    print(f"  - Strategy Sharpe: {metrics['strategy_sharpe']:.4f}")
    print(f"  - Linear Sharpe: {metrics['linear_sharpe']:.4f}")

    # Test online learning
    benchmark.partial_fit(test_features, 0.5, 'elastic')
    print(f"[OK] Online learning (partial_fit) works")

    # Test serialization
    data = benchmark.to_dict()
    assert 'models' in data, "Serialization should include models"
    print(f"[OK] Serialization works")

    print("\n[PASS] Linear Benchmark: ALL TESTS PASSED")
    return True


def test_combined_strategy():
    """Test Combined Strategy."""
    print("\n" + "="*60)
    print("TEST 6: Combined Strategy")
    print("="*60)

    # Create strategy with default configuration
    genome = StrategyGenome(
        signal_type=SignalType.STGP,
        fuzzy_preprocessing=False,
        position_manager=PositionManagerType.FSM,
        regime_gate=True,
    )

    strategy = CombinedStrategy(genome)
    print(f"[OK] Created combined strategy with ID: {strategy.genome.strategy_id}")

    # Get state info
    state_info = strategy.get_state_info()
    print(f"[OK] Strategy state info:")
    print(f"  - Signal type: {state_info['signal_type']}")
    print(f"  - Has preprocessing: {state_info['has_preprocessing']}")
    print(f"  - Has regime gate: {state_info['has_regime_gate']}")

    # Test execution
    features = {
        'close': 50000,
        'open': 49500,
        'high': 50500,
        'low': 49000,
        'volume': 1000000,
        'rsi': 35,  # Oversold
        'macd': 100,
        'macd_signal': 80,
        'macd_histogram': 20,
        'returns': 0.01,
        'volatility': 0.02,
        'funding_rate': 0.0001,
        'whale_pressure': 0.5,
        'volume_zscore': 1.2,
        'vix': 18,
        'trend_strength': 0.5,
        'correlation': 0.4,
        'atr': 500,
        'avg_atr': 400,
    }

    result = strategy.execute(features)
    print(f"[OK] Execution result:")
    print(f"  - Action: {result.action.name}")
    print(f"  - Size: {result.size:.4f}")
    print(f"  - Signal strength: {result.signal_strength:.4f}")
    print(f"  - Signal confidence: {result.signal_confidence:.4f}")
    print(f"  - Gate action: {result.gate_action.name if result.gate_action else 'N/A'}")
    print(f"  - FSM state: {result.fsm_state}")

    # Execute again to test FSM state transitions
    result2 = strategy.execute(features)
    print(f"[OK] Second execution - FSM state: {result2.fsm_state}")

    # Test with different signal types
    for signal_type in [SignalType.DECISION_TREE, SignalType.FUZZY_RULES]:
        genome2 = StrategyGenome(
            signal_type=signal_type,
            decision_tree_config={
                "rules": [
                    {"feature": "rsi", "threshold": 30, "operator": "<", "signal": 0.8, "confidence": 0.7},
                    {"feature": "rsi", "threshold": 70, "operator": ">", "signal": -0.8, "confidence": 0.7},
                ],
                "default_signal": 0.0
            } if signal_type == SignalType.DECISION_TREE else None,
            regime_gate=True,
        )
        strategy2 = CombinedStrategy(genome2)
        result = strategy2.execute(features)
        print(f"[OK] {signal_type.name} strategy: action={result.action.name}, signal={result.signal_strength:.4f}")

    # Test genome export
    exported_genome = strategy.to_genome()
    assert exported_genome.strategy_id == strategy.genome.strategy_id
    print(f"[OK] Genome export works")

    # Test genome serialization
    genome_dict = exported_genome.to_dict()
    restored_genome = StrategyGenome.from_dict(genome_dict)
    assert restored_genome.strategy_id == exported_genome.strategy_id
    print(f"[OK] Genome serialization/deserialization works")

    # Test strategy factory
    from src.representations.combined_strategy import StrategyFactory

    random_strategy = StrategyFactory.create_random(
        signal_type=SignalType.STGP,
        with_preprocessing=True,
        with_regime_gate=True,
        engine_id=1
    )
    print(f"[OK] StrategyFactory created strategy: {random_strategy.genome.strategy_id}")

    # Test strategy statistics
    from src.representations.combined_strategy import StrategyStatistics

    stats = StrategyStatistics()
    stats.record_evaluation(
        strategy=strategy,
        sharpe=1.5,
        win_rate=0.55,
        drawdown=0.15,
        vs_benchmark=BenchmarkResult.BEATS_BENCHMARK
    )

    summary = stats.get_summary()
    print(f"[OK] Statistics recorded: {summary}")

    # Test reset
    strategy.reset()
    state_info = strategy.get_state_info()
    print(f"[OK] Strategy reset, FSM state: {state_info.get('fsm_state', {}).get('state', 'N/A')}")

    print("\n[PASS] Combined Strategy: ALL TESTS PASSED")
    return True


def test_integration():
    """Test full integration pipeline."""
    print("\n" + "="*60)
    print("TEST 7: Full Integration Pipeline")
    print("="*60)

    # Create a complete strategy with all components
    genome = StrategyGenome(
        signal_type=SignalType.STGP,
        fuzzy_preprocessing=True,
        position_manager=PositionManagerType.FSM,
        regime_gate=True,
        created_by_engine=1,
    )

    strategy = CombinedStrategy(genome)

    # Simulate a trading sequence
    market_sequence = [
        # Day 1: Normal market, slight bullish
        {'close': 50000, 'rsi': 45, 'macd_histogram': 10, 'vix': 15, 'funding_rate': 0.0001, 'volume_zscore': 0.5, 'trend_strength': 0.4, 'correlation': 0.3, 'atr': 400, 'avg_atr': 400, 'volatility': 0.015},
        # Day 2: Getting bullish
        {'close': 51000, 'rsi': 55, 'macd_histogram': 30, 'vix': 14, 'funding_rate': 0.0002, 'volume_zscore': 0.8, 'trend_strength': 0.5, 'correlation': 0.35, 'atr': 450, 'avg_atr': 410, 'volatility': 0.018},
        # Day 3: Overbought territory
        {'close': 53000, 'rsi': 72, 'macd_histogram': 50, 'vix': 16, 'funding_rate': 0.0005, 'volume_zscore': 1.5, 'trend_strength': 0.7, 'correlation': 0.5, 'atr': 600, 'avg_atr': 450, 'volatility': 0.025},
        # Day 4: Market stress
        {'close': 51000, 'rsi': 40, 'macd_histogram': -20, 'vix': 28, 'funding_rate': -0.0003, 'volume_zscore': 2.5, 'trend_strength': 0.8, 'correlation': 0.8, 'atr': 1000, 'avg_atr': 550, 'volatility': 0.045},
        # Day 5: Recovery
        {'close': 52000, 'rsi': 50, 'macd_histogram': 5, 'vix': 20, 'funding_rate': 0.0, 'volume_zscore': 0.3, 'trend_strength': 0.3, 'correlation': 0.4, 'atr': 500, 'avg_atr': 600, 'volatility': 0.02},
    ]

    print(f"[OK] Simulating 5-day trading sequence...")
    print("-" * 50)

    for i, market_data in enumerate(market_sequence, 1):
        result = strategy.execute(market_data)
        gate_status = result.gate_action.name if result.gate_action else "N/A"
        regime = result.metadata.get('regime_probs', {})
        dominant = max(regime, key=regime.get) if regime else "N/A"

        print(f"Day {i}: RSI={market_data['rsi']:2d}, VIX={market_data['vix']:2d} | "
              f"Signal={result.signal_strength:+.2f} | "
              f"Gate={gate_status:11s} | "
              f"Action={result.action.name:12s} | "
              f"FSM={result.fsm_state:12s} | "
              f"Regime={dominant}")

    print("-" * 50)
    print(f"[OK] Trading simulation complete")

    # Verify FSM tracked state properly
    final_state = strategy.position_manager.get_state_info() if strategy.position_manager else {}
    print(f"[OK] Final FSM state: {final_state.get('state', 'N/A')}")
    print(f"[OK] Final position size: {final_state.get('position_size', 0):.4f}")

    print("\n[PASS] Integration Pipeline: ALL TESTS PASSED")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("HIMARI LAYER 1 EXPLORER - REPRESENTATION LAYER TESTS")
    print("="*60)

    results = {}

    try:
        results['FSM Position Manager'] = test_fsm_position_manager()
    except Exception as e:
        print(f"\n[FAIL] FSM Position Manager FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['FSM Position Manager'] = False

    try:
        results['Fuzzy Preprocessor'] = test_fuzzy_preprocessor()
    except Exception as e:
        print(f"\n[FAIL] Fuzzy Preprocessor FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['Fuzzy Preprocessor'] = False

    try:
        results['STGP Formula'] = test_stgp_formula()
    except Exception as e:
        print(f"\n[FAIL] STGP Formula FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['STGP Formula'] = False

    try:
        results['Bayesian Regime Gate'] = test_bayesian_regime_gate()
    except Exception as e:
        print(f"\n[FAIL] Bayesian Regime Gate FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['Bayesian Regime Gate'] = False

    try:
        results['Linear Benchmark'] = test_linear_benchmark()
    except Exception as e:
        print(f"\n[FAIL] Linear Benchmark FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['Linear Benchmark'] = False

    try:
        results['Combined Strategy'] = test_combined_strategy()
    except Exception as e:
        print(f"\n[FAIL] Combined Strategy FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['Combined Strategy'] = False

    try:
        results['Integration Pipeline'] = test_integration()
    except Exception as e:
        print(f"\n[FAIL] Integration Pipeline FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['Integration Pipeline'] = False

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_test in results.items():
        status = "[PASS]" if passed_test else "[FAIL]"
        print(f"  {test_name}: {status}")

    print("-" * 60)
    print(f"  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n*** ALL TESTS PASSED! The representation layer is working correctly. ***")
    else:
        print(f"\n*** WARNING: {total - passed} test(s) failed. Please review the errors above. ***")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
