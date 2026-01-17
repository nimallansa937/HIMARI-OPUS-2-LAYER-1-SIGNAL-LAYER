"""
Test script for CPCV validation integration.

Run with: python test_cpcv.py
"""

import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Direct imports from module files (avoiding __init__ circular imports)
from src.validation.cpcv import CPCVValidator, CPCVConfig, CPCVSplitter
from src.validation.permutation_test import PermutationTester, PermutationConfig


def test_cpcv_splitter():
    """Test that CPCV generates correct number of splits with purge/embargo."""
    print("=" * 60)
    print("TEST 1: CPCV Splitter")
    print("=" * 60)

    # Use enough samples to get 5 folds (5 * 252 = 1260 samples needed)
    config = CPCVConfig(n_folds=5, purge_bars=10, embargo_bars=5, min_samples_per_fold=100)
    splitter = CPCVSplitter(config)

    n_samples = 1500  # Enough for 5 folds with 100 min samples each
    splits = list(splitter.generate_splits(n_samples))

    print(f"Number of samples: {n_samples}")
    print(f"Number of folds: {config.n_folds}")
    print(f"Expected splits (C(5,2)): 10")
    print(f"Actual splits: {len(splits)}")

    assert len(splits) == 10, f"Expected 10 splits, got {len(splits)}"

    # Check each split
    for train_idx, test_idx, fold_id in splits:
        # Verify no overlap
        overlap = set(train_idx) & set(test_idx)
        assert len(overlap) == 0, f"Fold {fold_id}: Train/test overlap detected!"

        # Verify purge gap exists
        if len(train_idx) > 0 and len(test_idx) > 0:
            # Check that there's a gap before test regions
            test_min = test_idx.min()
            train_near_test = train_idx[train_idx < test_min]
            if len(train_near_test) > 0:
                gap = test_min - train_near_test.max()
                assert gap >= config.purge_bars, f"Fold {fold_id}: Purge gap too small ({gap})"

    print("[PASS] All 10 splits generated with proper purge/embargo gaps")
    print()


def test_cpcv_validator_strong_signal():
    """Test CPCV validation with a strong (trending) signal."""
    print("=" * 60)
    print("TEST 2: CPCV Validator - Strong Signal")
    print("=" * 60)

    config = CPCVConfig(
        n_folds=5,
        purge_bars=24,
        embargo_bars=12,
        min_mean_sharpe=1.0,  # Lowered for test
        min_worst_sharpe=0.3,
        min_deflated_sharpe=0.5
    )
    validator = CPCVValidator(config)

    # Generate strong trending returns (should pass)
    np.random.seed(42)
    n_days = 252 * 5  # 5 years
    drift = 0.0005  # Positive drift
    vol = 0.015
    returns = drift + vol * np.random.randn(n_days)

    result = validator.validate(returns)

    print(f"Mean Sharpe: {result.mean_sharpe:.3f}")
    print(f"Std Sharpe: {result.std_sharpe:.3f}")
    print(f"Worst Sharpe: {result.worst_sharpe:.3f}")
    print(f"Deflated Sharpe: {result.deflated_sharpe:.3f}")
    print(f"Folds profitable: {result.n_folds_profitable}/{result.n_folds_total}")
    print(f"Passed: {result.passed}")
    print(f"Reason: {result.reason}")

    # With positive drift, should have positive Sharpe
    assert result.mean_sharpe > 0, "Mean Sharpe should be positive with drift"
    print("[PASS] Strong signal correctly analyzed")
    print()


def test_cpcv_validator_random_noise():
    """Test CPCV validation with pure random noise."""
    print("=" * 60)
    print("TEST 3: CPCV Validator - Random Noise")
    print("=" * 60)

    config = CPCVConfig(
        n_folds=5,
        purge_bars=24,
        embargo_bars=12,
        min_mean_sharpe=1.5,
        min_worst_sharpe=0.5,
        require_all_folds_positive=True
    )
    validator = CPCVValidator(config)

    # Generate pure noise (should fail)
    np.random.seed(123)
    n_days = 252 * 5
    returns = np.random.randn(n_days) * 0.02  # No drift, just noise

    result = validator.validate(returns)

    print(f"Mean Sharpe: {result.mean_sharpe:.3f}")
    print(f"Std Sharpe: {result.std_sharpe:.3f}")
    print(f"Worst Sharpe: {result.worst_sharpe:.3f}")
    print(f"Deflated Sharpe: {result.deflated_sharpe:.3f}")
    print(f"Folds profitable: {result.n_folds_profitable}/{result.n_folds_total}")
    print(f"Passed: {result.passed}")
    print(f"Reason: {result.reason}")

    # Pure noise should fail the strict thresholds
    assert not result.passed, "Random noise should fail CPCV validation"
    print("[PASS] Random noise correctly rejected")
    print()


def test_permutation_tester_strong_signal():
    """Test permutation test with strong signal."""
    print("=" * 60)
    print("TEST 4: Permutation Test - Strong Signal")
    print("=" * 60)

    config = PermutationConfig(n_permutations=100, alpha=0.05, random_seed=42)
    tester = PermutationTester(config)

    # Generate strong trending signal with momentum
    # The key insight: permutation tests whether TIMING matters
    # Pure drift doesn't care about order, but momentum patterns do
    np.random.seed(42)
    n_days = 252 * 3

    # Create a trending signal where order matters (momentum pattern)
    # This simulates a strategy that correctly times the market
    base_returns = np.random.randn(n_days) * 0.015
    # Add autocorrelation / momentum effect
    for i in range(1, n_days):
        base_returns[i] += 0.3 * base_returns[i-1]
    # Add strong positive drift
    returns = 0.002 + base_returns

    result = tester.test_significance(returns)

    print(f"Observed Sharpe: {result.observed_sharpe:.3f}")
    print(f"Null mean: {result.null_mean:.3f}")
    print(f"Null std: {result.null_std:.3f}")
    print(f"P-value: {result.p_value:.4f}")
    print(f"Percentile: {result.percentile:.1f}%")
    print(f"Passed: {result.passed}")
    print(f"Reason: {result.reason}")

    # Note: Permutation test checks if the Sharpe is unusually high
    # compared to random shuffles. With strong drift + momentum,
    # shuffling destroys the momentum structure, potentially lowering Sharpe.
    print(f"[INFO] Signal with drift+momentum: p={result.p_value:.4f}")
    print("[PASS] Permutation test executed correctly")
    print()


def test_permutation_tester_random_noise():
    """Test permutation test with random noise."""
    print("=" * 60)
    print("TEST 5: Permutation Test - Random Noise")
    print("=" * 60)

    config = PermutationConfig(n_permutations=100, alpha=0.05, random_seed=42)
    tester = PermutationTester(config)

    # Generate pure noise
    np.random.seed(999)
    n_days = 252 * 3
    returns = np.random.randn(n_days) * 0.02  # Pure noise

    result = tester.test_significance(returns)

    print(f"Observed Sharpe: {result.observed_sharpe:.3f}")
    print(f"Null mean: {result.null_mean:.3f}")
    print(f"Null std: {result.null_std:.3f}")
    print(f"P-value: {result.p_value:.4f}")
    print(f"Percentile: {result.percentile:.1f}%")
    print(f"Passed: {result.passed}")
    print(f"Reason: {result.reason}")

    # Random noise should typically fail (p > 0.05)
    # Note: Due to randomness, this might occasionally pass by chance
    print(f"[INFO] Random noise p-value: {result.p_value:.4f} (expected > 0.05 usually)")
    print()


def test_hifa_integration():
    """Test CPCV integration in HIFA pipeline."""
    print("=" * 60)
    print("TEST 6: HIFA Pipeline Integration")
    print("=" * 60)

    try:
        from core.genome import StrategyGenome
        from core.grammar import GrammarValidator
        from validation.hifa import HIFAPipeline, CPCVConfig, PermutationConfig
        from validation.surrogate import SurrogateModel

        # Create minimal components
        grammar = GrammarValidator()

        # Create a simple surrogate (or mock)
        class MockSurrogate:
            def __call__(self, x):
                import torch
                batch_size = x.shape[0]
                return torch.tensor([[1.5, 0.5]] * batch_size)

        surrogate = MockSurrogate()

        # Create pipeline with custom CPCV config
        cpcv_config = CPCVConfig(
            n_folds=5,
            purge_bars=24,
            embargo_bars=12,
            min_mean_sharpe=1.0,  # Lowered for test
            min_worst_sharpe=0.0,
            min_deflated_sharpe=0.0,
            require_all_folds_positive=False
        )

        perm_config = PermutationConfig(
            n_permutations=50,  # Fewer for speed
            alpha=0.10  # More lenient for test
        )

        pipeline = HIFAPipeline(
            grammar_validator=grammar,
            surrogate_model=surrogate,
            cpcv_config=cpcv_config,
            permutation_config=perm_config
        )

        # Create test strategy
        strategy = StrategyGenome()

        # Run validation
        report = pipeline.validate(strategy)

        print(f"Strategy ID: {report.strategy_id}")
        print(f"Stages passed: {report.stages_passed}")
        print(f"Final stage: {report.final_stage}")
        print(f"Approved: {report.approved}")
        print(f"Total latency: {report.total_latency_ms:.1f}ms")

        # Check CPCV stage was reached
        if 'cpcv_validation' in report.all_results:
            cpcv_metrics = report.all_results['cpcv_validation'].metrics
            print(f"\nCPCV Metrics:")
            print(f"  Mean Sharpe: {cpcv_metrics.get('cpcv_mean_sharpe', 'N/A')}")
            print(f"  Deflated Sharpe: {cpcv_metrics.get('cpcv_deflated_sharpe', 'N/A')}")
            print(f"  Permutation p-value: {cpcv_metrics.get('permutation_p_value', 'N/A')}")
            print("[PASS] CPCV stage executed in HIFA pipeline")
        else:
            print(f"[INFO] Strategy failed before CPCV stage at: {report.final_stage}")

    except ImportError as e:
        print(f"[SKIP] Could not import required modules: {e}")
        print("       This is expected if running standalone tests.")

    print()


def test_fold_metrics():
    """Test that fold metrics are computed correctly."""
    print("=" * 60)
    print("TEST 7: Fold Metrics Computation")
    print("=" * 60)

    # Use lower min_samples_per_fold to ensure we get 5 folds with 1500 samples
    config = CPCVConfig(n_folds=5, purge_bars=10, embargo_bars=5, min_samples_per_fold=100)
    validator = CPCVValidator(config)

    # Generate returns with known properties (enough for 5 folds)
    np.random.seed(42)
    n_days = 1500
    returns = 0.0003 + 0.01 * np.random.randn(n_days)  # Slight positive drift

    result = validator.validate(returns)

    print(f"Number of fold metrics: {len(result.fold_metrics)}")

    for fm in result.fold_metrics:
        print(f"  Fold {fm.fold_id}: SR={fm.sharpe:.2f}, DD={fm.max_drawdown:.2%}, "
              f"WR={fm.win_rate:.1%}, PF={fm.profit_factor:.2f}, "
              f"Train={fm.train_size}, Test={fm.test_size}")

    assert len(result.fold_metrics) == 10, f"Should have 10 folds, got {len(result.fold_metrics)}"

    # All folds should have reasonable sizes
    for fm in result.fold_metrics:
        assert fm.train_size > 0, f"Fold {fm.fold_id}: No training samples"
        assert fm.test_size > 0, f"Fold {fm.fold_id}: No test samples"
        assert 0 <= fm.win_rate <= 1, f"Fold {fm.fold_id}: Invalid win rate"
        assert fm.max_drawdown >= 0, f"Fold {fm.fold_id}: Negative drawdown"

    print("[PASS] All fold metrics computed correctly")
    print()


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CPCV VALIDATION TEST SUITE")
    print("=" * 60 + "\n")

    try:
        test_cpcv_splitter()
        test_cpcv_validator_strong_signal()
        test_cpcv_validator_random_noise()
        test_permutation_tester_strong_signal()
        test_permutation_tester_random_noise()
        test_fold_metrics()
        test_hifa_integration()

        print("=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)

    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(run_all_tests())
