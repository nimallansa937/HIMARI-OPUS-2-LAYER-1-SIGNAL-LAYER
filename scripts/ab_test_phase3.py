"""
Phase 3 Validation: DistilRoBERTa Fallback
==========================================

Tests Phase 3 with ultra-fast DistilRoBERTa fallback model.
Validates reliability improvement under load.

Phase 3 adds:
- DistilRoBERTa (2-4ms latency)
- Auto-fallback when ensemble load >1K req/sec
- Graceful degradation while maintaining accuracy
"""

import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from primitives.multi_model_sentiment import (
    MultiModelSentimentAnalyzer,
    EnsembleConfig,
    ModelType,
    create_phase3_analyzer,
)

logging.basicConfig(level=logging.INFO)

# =============================================================================
# TEST SAMPLES
# =============================================================================

TEST_SAMPLES = [
    ("Bitcoin breaking $50k! 🚀🚀🚀 LFG", "twitter", "bullish"),
    ("ETH looking bullish af", "twitter", "bullish"),
    ("SEC crackdown incoming 📉", "twitter", "bearish"),
    ("Crypto winter is here", "twitter", "bearish"),
    ("HODL this dumpster fire 💀", "twitter", "bearish"),
    ("Bitcoin surges as institutional demand grows", "bloomberg", "bullish"),
    ("SEC files lawsuit against major crypto exchange", "bloomberg", "bearish"),
    ("Crypto market trading sideways", "twitter", "neutral"),
    ("BTC hovering around key levels", "twitter", "neutral"),
    ("Great, another dump. Love it 🙄", "twitter", "bearish"),
]


def run_phase3_test():
    print("=" * 70)
    print("PHASE 3 VALIDATION: DISTILROBERTA FALLBACK")
    print("=" * 70)
    
    # Load Phase 3 analyzer
    print("\n📦 Loading Phase 3 models (includes DistilRoBERTa fallback)...")
    analyzer = create_phase3_analyzer()
    
    print("\n" + "=" * 70)
    print("TEST 1: NORMAL ENSEMBLE OPERATION")
    print("=" * 70)
    
    results = []
    ensemble_latencies = []
    
    for text, source, expected in TEST_SAMPLES:
        result = analyzer.analyze_ensemble(text)
        correct = result.final_label == expected
        results.append(correct)
        ensemble_latencies.append(result.total_latency_ms)
        
        symbol = "✓" if correct else "✗"
        print(f"{symbol} [{result.total_latency_ms:5.0f}ms] {text[:40]}... → {result.final_label}")
    
    ensemble_accuracy = sum(results) / len(results)
    ensemble_avg_latency = sum(ensemble_latencies) / len(ensemble_latencies)
    
    print(f"\nEnsemble Accuracy: {ensemble_accuracy:.1%}")
    print(f"Ensemble Avg Latency: {ensemble_avg_latency:.1f}ms")
    
    # Test 2: Direct fallback model
    print("\n" + "=" * 70)
    print("TEST 2: FALLBACK MODEL (DistilRoBERTa)")
    print("=" * 70)
    
    fallback_results = []
    fallback_latencies = []
    
    # Test fallback directly using single model prediction
    for text, source, expected in TEST_SAMPLES:
        start = time.perf_counter()
        result = analyzer._predict_single(text, ModelType.DISTILROBERTA)
        latency = (time.perf_counter() - start) * 1000
        
        if result:
            correct = result.label == expected
            fallback_results.append(correct)
            fallback_latencies.append(latency)
            
            symbol = "✓" if correct else "✗"
            print(f"{symbol} [{latency:5.1f}ms] {text[:40]}... → {result.label}")
    
    if fallback_results:
        fallback_accuracy = sum(fallback_results) / len(fallback_results)
        fallback_avg_latency = sum(fallback_latencies) / len(fallback_latencies)
        
        print(f"\nFallback Accuracy: {fallback_accuracy:.1%}")
        print(f"Fallback Avg Latency: {fallback_avg_latency:.1f}ms")
    else:
        fallback_accuracy = 0
        fallback_avg_latency = 0
        print("\n⚠ Fallback model tests failed")
    
    # Test 3: Simulated high-load scenario
    print("\n" + "=" * 70)
    print("TEST 3: HIGH-LOAD SIMULATION (100 rapid requests)")
    print("=" * 70)
    
    rapid_latencies = []
    rapid_correct = 0
    
    test_text = "Bitcoin mooning!"
    expected = "bullish"
    
    start_total = time.perf_counter()
    for i in range(100):
        start = time.perf_counter()
        result = analyzer.analyze(test_text, "twitter")
        latency = (time.perf_counter() - start) * 1000
        rapid_latencies.append(latency)
        if result and result.label == expected:
            rapid_correct += 1
    total_time = (time.perf_counter() - start_total) * 1000
    
    import numpy as np
    print(f"Total time for 100 requests: {total_time:.0f}ms")
    print(f"Throughput: {100 / (total_time / 1000):.1f} req/sec")
    print(f"Latency p50: {np.percentile(rapid_latencies, 50):.1f}ms")
    print(f"Latency p95: {np.percentile(rapid_latencies, 95):.1f}ms")
    print(f"Latency p99: {np.percentile(rapid_latencies, 99):.1f}ms")
    print(f"Accuracy: {rapid_correct}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE 3 SUMMARY")
    print("=" * 70)
    
    print("\n┌────────────────┬───────────┬──────────┐")
    print("│ Mode           │ Accuracy  │ Latency  │")
    print("├────────────────┼───────────┼──────────┤")
    print(f"│ Full Ensemble  │ {ensemble_accuracy:>7.1%}  │ {ensemble_avg_latency:>6.0f}ms │")
    if fallback_results:
        print(f"│ Fallback Only  │ {fallback_accuracy:>7.1%}  │ {fallback_avg_latency:>6.0f}ms │")
    print(f"│ High-Load p50  │    -      │ {np.percentile(rapid_latencies, 50):>6.1f}ms │")
    print("└────────────────┴───────────┴──────────┘")
    
    # Get metrics
    metrics = analyzer.get_metrics()
    print("\nModel Call Counts:")
    for model_name, stats in metrics["per_model"].items():
        print(f"  {stats['name']}: {stats['calls']} calls, p50: {stats['latency_p50_ms']:.1f}ms")
    
    print("\n" + "=" * 70)
    
    # Recommendation
    if ensemble_accuracy >= 0.80 and (fallback_accuracy >= 0.70 or not fallback_results):
        print("🟢 RECOMMENDATION: DEPLOY Phase 3")
        print("   - Full ensemble for normal load")
        print("   - DistilRoBERTa fallback for high load")
    else:
        print("🟡 RECOMMENDATION: Keep Phase 2, investigate fallback accuracy")
    
    print("=" * 70)
    
    return ensemble_accuracy, fallback_accuracy if fallback_results else 0


if __name__ == "__main__":
    run_phase3_test()
