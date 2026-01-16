"""
HIMARI Layer 1 Explorer - Functional Test Suite
"""

import sys
import traceback

def run_tests():
    print('=' * 60)
    print('HIMARI Layer 1 Explorer - Functional Test')
    print('=' * 60)
    print()

    results = []

    # Test 1: Feature Vector System
    print('Test 1: Feature Vector System')
    try:
        from src.core.features import FeatureVector, FEATURE_SCHEMA
        import time
        ts = int(time.time() * 1000)
        fv = FeatureVector()
        fv.set('rsi_14', 0.35, ts)
        fv.set('close', 50000.0, ts)  # Use valid feature name
        vec = fv.normalize()
        assert len(vec) > 0, 'Vector should not be empty'
        assert len(FEATURE_SCHEMA) == 60, f'Schema should have 60 features, got {len(FEATURE_SCHEMA)}'
        print(f'  [OK] Feature vector created ({len(FEATURE_SCHEMA)} features in schema)')
        results.append(('Feature Vector', True))
    except Exception as e:
        print(f'  [FAIL] {e}')
        results.append(('Feature Vector', False))

    # Test 2: Strategy Genome
    print('Test 2: Strategy Genome Generation')
    try:
        from src.core.genome import generate_random_strategy, StrategyGenome
        strategy = generate_random_strategy()
        assert strategy is not None, 'Strategy should be generated'
        assert hasattr(strategy, 'id'), 'Strategy should have id'
        vec = strategy.to_vector()
        assert len(vec) == 127, f'Vector should be 127-dim, got {len(vec)}'
        print('  [OK] Strategy genome generated with 127-dim vector')
        results.append(('Strategy Genome', True))
    except Exception as e:
        print(f'  [FAIL] {e}')
        results.append(('Strategy Genome', False))

    # Test 3: Grammar Validation
    print('Test 3: Grammar Validation')
    try:
        from src.core.grammar import GrammarValidator
        grammar = GrammarValidator()
        strategy = generate_random_strategy()
        is_valid = grammar.validate(strategy)
        status = "valid" if is_valid else "invalid"
        print(f'  [OK] Grammar validation: {status}')
        results.append(('Grammar Validation', True))
    except Exception as e:
        print(f'  [FAIL] {e}')
        results.append(('Grammar Validation', False))

    # Test 4: Evolutionary Explorer
    print('Test 4: Evolutionary Explorer')
    try:
        from src.engines.evolutionary import EvolutionaryExplorer
        evo = EvolutionaryExplorer(population_size=20, elite_size=3)
        evo.initialize_population()  # No args needed
        assert len(evo.population) == 20, f'Population should be 20, got {len(evo.population)}'
        print(f'  [OK] Initialized population of {len(evo.population)} strategies')
        results.append(('Evolutionary Explorer', True))
    except Exception as e:
        print(f'  [FAIL] {e}')
        traceback.print_exc()
        results.append(('Evolutionary Explorer', False))

    # Test 5: Flow Matching Generator
    print('Test 5: Flow Matching Generator')
    try:
        from src.engines.flow_matching import FlowMatchingGenerator, GenerationCondition
        gen = FlowMatchingGenerator()
        cond = GenerationCondition(target_sharpe=1.5, regime_label=0)
        strategies = gen.generate(condition=cond, num_samples=3)
        assert len(strategies) == 3, f'Should generate 3 strategies, got {len(strategies)}'
        print(f'  [OK] Generated {len(strategies)} strategies via flow matching')
        results.append(('Flow Matching', True))
    except Exception as e:
        print(f'  [FAIL] {e}')
        results.append(('Flow Matching', False))

    # Test 6: HIFA Pipeline
    print('Test 6: HIFA Pipeline')
    try:
        from src.validation.hifa import HIFAPipeline
        from src.core.grammar import GrammarValidator
        grammar = GrammarValidator()
        hifa = HIFAPipeline(grammar_validator=grammar, surrogate_model=None)
        assert hifa is not None, 'HIFA pipeline should be created'
        print('  [OK] HIFA validation pipeline created')
        results.append(('HIFA Pipeline', True))
    except Exception as e:
        print(f'  [FAIL] {e}')
        results.append(('HIFA Pipeline', False))

    # Test 7: Shadow Environment
    print('Test 7: Shadow Environment')
    try:
        from src.deployment.shadow import ShadowEnvironment
        shadow = ShadowEnvironment()
        assert shadow is not None, 'Shadow environment should be created'
        print('  [OK] Shadow environment created')
        results.append(('Shadow Environment', True))
    except Exception as e:
        print(f'  [FAIL] {e}')
        results.append(('Shadow Environment', False))

    # Test 8: Drift Detection
    print('Test 8: Drift Detection Ensemble')
    try:
        from src.adaptation.drift import DriftDetectionEnsemble
        drift = DriftDetectionEnsemble()
        # Update expects a single float, not an array
        result = drift.update(0.5, 'sharpe')
        assert isinstance(result, dict), 'Should return dict'
        votes = result.get('votes', 0)
        print(f'  [OK] Drift detection: votes={votes}')
        results.append(('Drift Detection', True))
    except Exception as e:
        print(f'  [FAIL] {e}')
        results.append(('Drift Detection', False))

    # Test 9: MCTS Generator
    print('Test 9: MCTS Strategy Generator')
    try:
        from src.engines.mcts import MCTSStrategyGenerator, MCTSConfig
        mcts = MCTSStrategyGenerator(MCTSConfig(num_simulations=10))
        strategies = mcts.generate_batch(num_strategies=2)
        assert len(strategies) == 2, f'Should generate 2 strategies, got {len(strategies)}'
        print(f'  [OK] MCTS generated {len(strategies)} strategies')
        results.append(('MCTS Generator', True))
    except Exception as e:
        print(f'  [FAIL] {e}')
        results.append(('MCTS Generator', False))

    # Test 10: PSRO Diversity
    print('Test 10: PSRO Diversity Manager')
    try:
        from src.engines.psro import PSRODiversityManager, PSROConfig
        psro = PSRODiversityManager(PSROConfig())
        # Initialize population with seed strategies
        psro.initialize_population(seed_strategies=evo.population[:5], size=5)
        diverse = psro.get_diverse_sample(n=3)
        print(f'  [OK] PSRO manages {len(psro.population)} strategies, sampled {len(diverse)} diverse')
        results.append(('PSRO Diversity', True))
    except Exception as e:
        print(f'  [FAIL] {e}')
        results.append(('PSRO Diversity', False))

    # Test 11: Causal Validation
    print('Test 11: Causal Validation Gate')
    try:
        from src.validation.causal import CausalValidationGate
        causal = CausalValidationGate()
        strategy = generate_random_strategy()
        hypothesis = causal.infer_hypothesis(strategy, ['rsi_14', 'close'])
        assert hypothesis is not None, 'Should generate hypothesis'
        print(f'  [OK] Causal hypothesis: {hypothesis.cause_feature} -> {hypothesis.effect}')
        results.append(('Causal Validation', True))
    except Exception as e:
        print(f'  [FAIL] {e}')
        results.append(('Causal Validation', False))

    # Test 12: Knowledge Store
    print('Test 12: Knowledge Store')
    try:
        from src.crawler.store import KnowledgeStore
        import tempfile
        import os
        # Create temp file and close it so we can use the path
        fd, db_path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        store = KnowledgeStore(db_path=db_path)
        stats = store.get_statistics()
        total = stats.get('total', stats.get('total_items', 0))
        print(f'  [OK] Knowledge store initialized (total items: {total})')
        # Clean up - use try/except for Windows file locking
        try:
            os.unlink(db_path)
        except PermissionError:
            pass  # File may still be in use on Windows
        results.append(('Knowledge Store', True))
    except Exception as e:
        print(f'  [FAIL] {e}')
        results.append(('Knowledge Store', False))

    # Summary
    print()
    print('=' * 60)
    print('SUMMARY')
    print('=' * 60)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    for name, result in results:
        status = '[OK]' if result else '[FAIL]'
        print(f'  {status} {name}')
    print()
    print(f'Passed: {passed}/{total}')
    if passed == total:
        print('All tests passed!')
    else:
        print(f'Failed: {total - passed} tests')

    return passed == total


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
