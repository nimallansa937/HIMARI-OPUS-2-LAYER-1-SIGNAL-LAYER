"""
Latency Benchmark Script - Performance Validation

Measures p50/p95/p99 latency for all components and validates SLAs.

Usage:
    python scripts/benchmark_latency.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import time
import random
import numpy as np
from validation.latency_validator import LatencyBenchmark, validate_quantization_latency
from primitives import (
    StreamingHMM, 
    StreamingIndicators,
    MultiHorizonMomentum,
    RegimeAwareSignalFusion,
    SentimentLagBuffer,
    DynamicSentimentWeighter
)
from config import load_enhanced_config


def generate_synthetic_ohlcv():
    """Generate synthetic OHLCV data."""
    base_price = 100 + random.uniform(-5, 5)
    high = base_price + random.uniform(0, 2)
    low = base_price - random.uniform(0, 2)
    close = random.uniform(low, high)
    volume = random.uniform(1000, 10000)
    
    return {
        'open': base_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }


def benchmark_component(bench: LatencyBenchmark, name: str, func, iterations: int = 1000):
    """Benchmark a single component."""
    print(f"\nBenchmarking {name}...")
    
    for _ in range(iterations):
        with bench.measure(name):
            func()
    
    stats = bench.get_latency_stats(name)
    if name in stats:
        s = stats[name]
        print(f"  p50: {s.p50:.3f}ms  p95: {s.p95:.3f}ms  p99: {s.p99:.3f}ms  mean: {s.mean:.3f}ms")
        return s
    return None


def main():
    print("=" * 70)
    print("LATENCY BENCHMARK - Enhanced Layer 1 Components")
    print("=" * 70)
    
    config = load_enhanced_config()
    bench = LatencyBenchmark()
    
    # Initialize components
    print("\nInitializing components...")
    hmm = StreamingHMM()
    indicators = StreamingIndicators()
    momentum = MultiHorizonMomentum()
    lag_buffer = SentimentLagBuffer()
    weighter = DynamicSentimentWeighter()
    
    # Pre-warm with some data
    for _ in range(100):
        ohlcv = generate_synthetic_ohlcv()
        price_return = random.gauss(0, 0.02)
        hmm.update(price_return)
        indicators.update(ohlcv)
        momentum.update(ohlcv['close'])
    
    print("\nRunning benchmarks (10,000 iterations each)...")
    
    # Benchmark HMM
    benchmark_component(
        bench, 'hmm_update',
        lambda: hmm.update(random.gauss(0, 0.02)),
        iterations=10000
    )
    
    # Benchmark Indicators
    benchmark_component(
        bench, 'indicators_update',
        lambda: indicators.update(generate_synthetic_ohlcv()),
        iterations=10000
    )
    
    # Benchmark Momentum
    benchmark_component(
        bench, 'momentum_update',
        lambda: momentum.update(100 + random.uniform(-5, 5)),
        iterations=10000
    )
    
    # Benchmark Lag Buffer
    benchmark_component(
        bench, 'lag_buffer_update',
        lambda: lag_buffer.update('BTCUSDT', random.uniform(-1, 1), 'news'),
        iterations=10000
    )
    
    # Benchmark Dynamic Weighter
    regime_context = {
        'atr': 0.025,
        'social_zscore': 0.5,
        'market_regime': 'Bull'
    }
    benchmark_component(
        bench, 'weighter_get_weights',
        lambda: weighter.get_weights(regime_context),
        iterations=10000
    )
    
    # Print final report
    print("\n" + bench.get_report())
    
    # Check for SLA breaches
    breaches = bench.check_sla_breaches()
    if breaches:
        print("\n[!] SLA BREACHES DETECTED:")
        for b in breaches:
            print(f"  {b['component']}: {b['breach_type']} = {b['actual_value']:.2f}ms > {b['sla_value']}ms")
    else:
        print("\n[+] All components within SLA!")
    
    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"\nPerformance Summary:")
    print(f"  Total components tested: {len(bench.get_latency_stats())}")
    print(f"  SLA breaches: {len(breaches)}")
    
    # Calculate combined latency estimate
    total_stats = bench.get_latency_stats()
    combined_p99 = sum(s.p99 for s in total_stats.values())
    print(f"  Combined p99 estimate: {combined_p99:.3f}ms")
    print(f"  Target total p99: <50ms")
    
    if combined_p99 < 50:
        print("\n[PASS] System meets latency requirements!")
    else:
        print("\n[WARN] Combined latency exceeds target")
    
    return 0 if not breaches else 1


if __name__ == '__main__':
    sys.exit(main())
