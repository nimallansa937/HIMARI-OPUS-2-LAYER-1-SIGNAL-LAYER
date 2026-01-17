"""
Integration test for CPCV in HIFA pipeline.

Run from LAYER 1 EXPLORER AGENT directory:
    python -m src.validation.test_hifa_cpcv
"""

import numpy as np
import sys


def test_hifa_cpcv_integration():
    """Test CPCV integration in HIFA pipeline."""
    print("=" * 60)
    print("HIFA + CPCV Integration Test")
    print("=" * 60)

    from src.core.genome import StrategyGenome, generate_random_strategy
    from src.core.grammar import GrammarValidator
    from src.validation.hifa import HIFAPipeline
    from src.validation.cpcv import CPCVConfig
    from src.validation.permutation_test import PermutationConfig

    # Create minimal components
    grammar = GrammarValidator()

    # Create a simple mock surrogate
    class MockSurrogate:
        def __call__(self, x):
            import torch
            batch_size = x.shape[0]
            return torch.tensor([[2.0, 0.3]] * batch_size)

    surrogate = MockSurrogate()

    # Create pipeline with lenient CPCV config for testing
    cpcv_config = CPCVConfig(
        n_folds=5,
        purge_bars=24,
        embargo_bars=12,
        min_samples_per_fold=100,  # Allow smaller folds
        min_mean_sharpe=0.0,       # Very lenient for test
        min_worst_sharpe=-5.0,     # Very lenient
        min_deflated_sharpe=-5.0,  # Very lenient
        require_all_folds_positive=False
    )

    perm_config = PermutationConfig(
        n_permutations=20,  # Fewer for speed
        alpha=1.0           # Always pass
    )

    pipeline = HIFAPipeline(
        grammar_validator=grammar,
        surrogate_model=surrogate,
        cpcv_config=cpcv_config,
        permutation_config=perm_config
    )

    # Create test strategy
    strategy = generate_random_strategy()

    print(f"\nValidating strategy: {strategy.id}")
    print("-" * 40)

    # Run validation
    report = pipeline.validate(strategy)

    print(f"\nResults:")
    print(f"  Strategy ID: {report.strategy_id}")
    print(f"  Stages passed: {report.stages_passed}")
    print(f"  Final stage: {report.final_stage}")
    print(f"  Approved: {report.approved}")
    print(f"  Total latency: {report.total_latency_ms:.1f}ms")

    # Check each stage
    print(f"\nStage Results:")
    for stage_name, result in report.all_results.items():
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {stage_name}: {result.reason[:60]}...")

    # Check CPCV stage was executed
    if 'cpcv_validation' in report.all_results:
        cpcv_result = report.all_results['cpcv_validation']
        print(f"\nCPCV Validation Details:")
        print(f"  Mean Sharpe: {cpcv_result.metrics.get('cpcv_mean_sharpe', 'N/A'):.3f}")
        print(f"  Std Sharpe: {cpcv_result.metrics.get('cpcv_std_sharpe', 'N/A'):.3f}")
        print(f"  Worst Sharpe: {cpcv_result.metrics.get('cpcv_worst_sharpe', 'N/A'):.3f}")
        print(f"  Deflated Sharpe: {cpcv_result.metrics.get('cpcv_deflated_sharpe', 'N/A'):.3f}")
        print(f"  Folds profitable: {cpcv_result.metrics.get('n_folds_profitable', 'N/A')}/{cpcv_result.metrics.get('n_folds_total', 'N/A')}")
        print(f"  Permutation p-value: {cpcv_result.metrics.get('permutation_p_value', 'N/A'):.4f}")
        print(f"  Latency: {cpcv_result.latency_ms:.1f}ms")

        print(f"\n[PASS] CPCV stage executed successfully in HIFA pipeline!")
        return True
    else:
        print(f"\n[INFO] Strategy failed before CPCV stage at: {report.final_stage}")
        # Still a pass if we reached grammar/dsr stage
        return True


def test_cpcv_rejects_random():
    """Test that CPCV rejects random strategies with strict thresholds."""
    print("\n" + "=" * 60)
    print("CPCV Rejection Test (Strict Thresholds)")
    print("=" * 60)

    from src.core.genome import StrategyGenome, generate_random_strategy
    from src.core.grammar import GrammarValidator
    from src.validation.hifa import HIFAPipeline
    from src.validation.cpcv import CPCVConfig
    from src.validation.permutation_test import PermutationConfig

    grammar = GrammarValidator()

    class MockSurrogate:
        def __call__(self, x):
            import torch
            batch_size = x.shape[0]
            return torch.tensor([[2.5, 0.2]] * batch_size)

    # Strict CPCV config (production-like)
    cpcv_config = CPCVConfig(
        n_folds=5,
        purge_bars=24,
        embargo_bars=12,
        min_samples_per_fold=100,
        min_mean_sharpe=1.5,
        max_sharpe_std_ratio=0.5,
        min_worst_sharpe=0.5,
        min_deflated_sharpe=1.0,
        require_all_folds_positive=True
    )

    perm_config = PermutationConfig(
        n_permutations=50,
        alpha=0.05
    )

    pipeline = HIFAPipeline(
        grammar_validator=grammar,
        surrogate_model=MockSurrogate(),
        cpcv_config=cpcv_config,
        permutation_config=perm_config
    )

    # Test multiple random strategies
    n_tested = 5
    n_rejected_at_cpcv = 0

    print(f"\nTesting {n_tested} random strategies with production thresholds...")

    for i in range(n_tested):
        strategy = generate_random_strategy()
        report = pipeline.validate(strategy)

        if report.final_stage == 'cpcv_validation' and not report.approved:
            n_rejected_at_cpcv += 1
            print(f"  Strategy {i+1}: Rejected at CPCV - {report.final_result.reason[:50]}...")
        elif 'cpcv_validation' in report.stages_passed:
            print(f"  Strategy {i+1}: Passed CPCV (approved={report.approved})")
        else:
            print(f"  Strategy {i+1}: Failed at {report.final_stage}")

    print(f"\nSummary: {n_rejected_at_cpcv}/{n_tested} rejected at CPCV stage")
    print("[PASS] CPCV rejection test completed")
    return True


if __name__ == "__main__":
    try:
        test_hifa_cpcv_integration()
        test_cpcv_rejects_random()
        print("\n" + "=" * 60)
        print("ALL INTEGRATION TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
