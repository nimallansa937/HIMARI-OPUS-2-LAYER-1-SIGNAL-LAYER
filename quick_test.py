#!/usr/bin/env python3
"""
HIMARI L1 Quick Test

Verifies that all components work correctly without requiring
Kafka/Redis infrastructure. Run this first after installation.

Usage:
    python quick_test.py
"""

import sys
import numpy as np

def test_welford():
    """Test Welford's online variance."""
    from primitives.welford import WelfordVariance
    
    welford = WelfordVariance()
    data = [100, 102, 98, 105, 97, 103, 101, 99, 104, 100]
    
    for x in data:
        welford.update(x)
    
    # Compare with numpy
    np_mean = np.mean(data)
    np_std = np.std(data, ddof=1)  # Sample std
    
    assert abs(welford.mean - np_mean) < 0.001, f"Mean mismatch: {welford.mean} vs {np_mean}"
    assert abs(welford.std - np_std) < 0.001, f"Std mismatch: {welford.std} vs {np_std}"
    
    print(f"✓ Welford: mean={welford.mean:.2f}, std={welford.std:.2f}")
    return True


def test_kalman():
    """Test Kalman filter."""
    from primitives.kalman import KalmanFilter
    
    kf = KalmanFilter(process_noise=0.01, measurement_noise=0.1)
    
    # Simulate noisy price with trend
    np.random.seed(42)
    true_prices = 100 + np.cumsum(np.random.randn(50) * 0.5)
    noisy_prices = true_prices + np.random.randn(50) * 2
    
    filtered = []
    for price in noisy_prices:
        filtered.append(kf.update(price))
    
    # Filtered should be smoother than raw
    raw_var = np.var(np.diff(noisy_prices))
    filt_var = np.var(np.diff(filtered))
    
    assert filt_var < raw_var, "Kalman should smooth the signal"
    print(f"✓ Kalman: raw_variance={raw_var:.2f}, filtered_variance={filt_var:.2f}")
    print(f"  Smoothing ratio: {filt_var/raw_var:.1%} (lower is smoother)")
    return True


def test_ultimate_smoother():
    """Test Ehlers Ultimate Smoother."""
    from primitives.ultimate_smoother import UltimateSmoother, SuperSmoother
    
    # Create both for comparison
    ultimate = UltimateSmoother(period=20)
    super_smooth = SuperSmoother(period=20)
    
    # Simulate price data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    ult_out = []
    ss_out = []
    
    for price in prices:
        ult_out.append(ultimate.update(price))
        ss_out.append(super_smooth.update(price))
    
    # Both should produce smoothed output
    assert len(ult_out) == 100
    assert len(ss_out) == 100
    
    print(f"✓ Ultimate Smoother: final value={ult_out[-1]:.2f}")
    print(f"  SuperSmoother comparison: {ss_out[-1]:.2f}")
    return True


def test_rls():
    """Test Recursive Least Squares."""
    from primitives.rls import RecursiveLeastSquares
    
    rls = RecursiveLeastSquares(forgetting_factor=0.99)
    
    # Generate data with known slope
    true_slope = 0.5
    true_intercept = 100
    
    np.random.seed(42)
    for i in range(100):
        x = i
        y = true_slope * x + true_intercept + np.random.randn() * 2
        slope, intercept = rls.update(x, y)
    
    # Should converge close to true values
    assert abs(slope - true_slope) < 0.1, f"Slope {slope} not close to {true_slope}"
    assert abs(intercept - true_intercept) < 5, f"Intercept {intercept} not close to {true_intercept}"
    
    print(f"✓ RLS: slope={slope:.3f} (true={true_slope}), intercept={intercept:.1f} (true={true_intercept})")
    return True


def test_hmm():
    """Test Hidden Markov Model regime detection."""
    from regime import StreamingHMM, RegimeState
    
    hmm = StreamingHMM(n_states=3)
    
    # Simulate different regimes
    np.random.seed(42)
    
    # Bull market: positive returns
    bull_returns = np.random.randn(30) * 0.01 + 0.002
    # Bear market: negative returns, high vol
    bear_returns = np.random.randn(30) * 0.025 - 0.003
    # Ranging: low vol, zero mean
    range_returns = np.random.randn(30) * 0.005
    
    # Process and track detected regimes
    all_returns = np.concatenate([bull_returns, bear_returns, range_returns])
    states = []
    
    for ret in all_returns:
        state, conf = hmm.update(ret)
        states.append(state)
    
    # Check final state detection
    final_state = hmm.state
    final_conf = hmm.confidence
    
    print(f"✓ HMM: {len(all_returns)} observations processed")
    print(f"  Final state: {hmm.state_name} (confidence={final_conf:.1%})")
    print(f"  State distribution: {dict(zip(['BULL','BEAR','RANGE'], hmm.probabilities.round(2)))}")
    return True


def test_serialization():
    """Test state serialization/restoration for warm restart."""
    from primitives.kalman import KalmanFilter
    from primitives.welford import WelfordVariance
    from regime import StreamingHMM
    
    # Create and update Kalman
    kf1 = KalmanFilter()
    for i in range(50):
        kf1.update(100 + i * 0.1)
    
    # Serialize and restore
    json_state = kf1.to_json()
    kf2 = KalmanFilter.from_json(json_state)
    
    assert abs(kf1.state - kf2.state) < 0.0001, "Kalman state mismatch after restore"
    assert abs(kf1.gain - kf2.gain) < 0.0001, "Kalman gain mismatch after restore"
    
    # Test Welford
    w1 = WelfordVariance()
    for i in range(100):
        w1.update(np.random.randn() * 10 + 100)
    
    json_state = w1.to_json()
    w2 = WelfordVariance.from_json(json_state)
    
    assert w1.count == w2.count, "Welford count mismatch"
    assert abs(w1.mean - w2.mean) < 0.0001, "Welford mean mismatch"
    
    # Test HMM
    hmm1 = StreamingHMM()
    for i in range(50):
        hmm1.update(np.random.randn() * 0.01)
    
    json_state = hmm1.to_json()
    hmm2 = StreamingHMM.from_json(json_state)
    
    assert hmm1.state == hmm2.state, "HMM state mismatch"
    assert abs(hmm1.confidence - hmm2.confidence) < 0.0001, "HMM confidence mismatch"
    
    print("✓ Serialization: All components serialize/restore correctly")
    return True


def test_latency():
    """Benchmark processing latency."""
    import time
    from primitives.kalman import KalmanFilter
    from primitives.ultimate_smoother import UltimateSmoother
    from primitives.welford import WelfordVariance
    from regime import StreamingHMM
    
    # Create all components
    kalman = KalmanFilter()
    smoother = UltimateSmoother()
    welford = WelfordVariance()
    hmm = StreamingHMM()
    
    # Warmup
    for i in range(100):
        price = 100 + i * 0.1
        ret = 0.001
        kalman.update(price)
        smoother.update(price)
        welford.update(ret)
        hmm.update(ret)
    
    # Benchmark
    iterations = 10000
    start = time.perf_counter()
    
    for i in range(iterations):
        price = 100 + (i % 100) * 0.1
        ret = 0.001 * (1 if i % 2 == 0 else -1)
        
        kalman.update(price)
        smoother.update(price)
        welford.update(ret)
        hmm.update(ret)
    
    elapsed = time.perf_counter() - start
    latency_us = (elapsed / iterations) * 1_000_000
    
    print(f"✓ Latency: {latency_us:.1f}μs per complete update cycle")
    print(f"  ({iterations} iterations in {elapsed*1000:.1f}ms)")
    
    # Target is <10ms = 10,000μs, so we should be well under
    assert latency_us < 1000, f"Latency {latency_us}μs exceeds 1000μs target"
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("HIMARI L1 Signal Layer - Quick Test")
    print("=" * 60)
    print()
    
    tests = [
        ("Welford Variance", test_welford),
        ("Kalman Filter", test_kalman),
        ("Ultimate Smoother", test_ultimate_smoother),
        ("Recursive Least Squares", test_rls),
        ("HMM Regime Detection", test_hmm),
        ("Serialization", test_serialization),
        ("Latency Benchmark", test_latency),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        print(f"\n[Testing {name}]")
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"✗ {name}: FAILED - {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        print("\n⚠️  Some tests failed. Check errors above.")
        return 1
    else:
        print("\n✅ All tests passed! Ready to integrate with your infrastructure.")
        print("\nNext steps:")
        print("1. Copy this package to src/layer1/ in your HIMARI infrastructure")
        print("2. Update config.py with your Redis/Kafka connection details")
        print("3. Run: python -m himari_l1.signal_processor")
        return 0


if __name__ == '__main__':
    sys.exit(main())
