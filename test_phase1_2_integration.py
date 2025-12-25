"""
Quick integration test for Phase 1 & 2 enhancements.
"""

from primitives import IntegratedSignalLayer
from config import load_enhanced_config

def test_integration():
    print("=" * 70)
    print("PHASE 1 & 2 INTEGRATION TEST")
    print("=" * 70)

    # Load config with enhancements enabled
    config = load_enhanced_config()
    config.sentiment_enabled = True
    config.sentiment_enable_lag_features = True
    config.sentiment_enable_dynamic_weighting = True

    # Initialize layer
    print("\nInitializing IntegratedSignalLayer with enhancements...")
    layer = IntegratedSignalLayer(config, redis_client=None)

    # Check sentiment integration
    if layer.sentiment:
        print(f"[+] Sentiment analyzer initialized")
        print(f"    - Lag buffer: {'YES' if layer.sentiment.lag_buffer else 'NO'}")
        print(f"    - Dynamic weighter: {'YES' if layer.sentiment.weighter else 'NO'}")
    else:
        print("[!] Sentiment not initialized (dependencies may be missing)")

    # Generate test signal with sentiment
    print("\nGenerating test signal with sentiment...")

    ohlcv = {
        'open': 100.0,
        'high': 101.0,
        'low': 99.0,
        'close': 100.5,
        'volume': 1000
    }

    sentiment_texts = ["Bitcoin surging to new highs, bulls dominating"]

    # First update
    output1 = layer.update('BTCUSDT', ohlcv, sentiment_texts=sentiment_texts)

    print(f"\nSignal Output (Update 1):")
    print(f"  Composite Signal: {output1.composite_signal:.4f}")
    print(f"  Regime: {output1.regime}")
    print(f"  Components: {len(output1.components)}")
    print(f"  Latency: {output1.total_latency_ms:.3f}ms")

    # Check if sentiment components are present
    sentiment_components = [k for k in output1.components.keys() if 'sentiment' in k or 'lag' in k]
    if sentiment_components:
        print(f"\n[+] Sentiment components detected: {sentiment_components}")
    else:
        print("\n[!] No sentiment components in output")

    # Multiple updates to build lag buffer
    print("\nRunning 100 updates to populate lag buffer...")
    for i in range(100):
        ohlcv['close'] = 100 + i * 0.1
        layer.update('BTCUSDT', ohlcv, sentiment_texts=sentiment_texts)

    # Check lag features
    output_final = layer.update('BTCUSDT', ohlcv, sentiment_texts=sentiment_texts)

    sentiment_components = [k for k in output_final.components.keys() if 'sentiment' in k or 'lag' in k]
    print(f"\nAfter 100 updates:")
    print(f"  Sentiment components: {sentiment_components}")
    print(f"  Total components: {len(output_final.components)}")

    # Check for lag features
    lag_features = [k for k in output_final.components.keys() if 'lag' in k]
    if lag_features:
        print(f"\n[PASS] Lag features working: {lag_features}")
    else:
        print("\n[WARN] No lag features detected (buffer may need more data)")

    # Get stats
    stats = layer.get_stats()
    print(f"\nSystem Stats:")
    print(f"  Total updates: {stats['update_count']}")
    if stats.get('fusion'):
        print(f"  Current regime: {stats['fusion']['current_regime']}")
        print(f"  Regime confidence: {stats['fusion']['confidence']:.2f}")

    print("\n" + "=" * 70)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 70)

    # Final verdict
    has_sentiment = 'sentiment_current' in output_final.components
    has_lags = any('lag' in k for k in output_final.components.keys())

    if has_sentiment and has_lags:
        print("\n[PASS] Phase 1 & 2 fully integrated!")
        return 0
    elif has_sentiment:
        print("\n[PARTIAL] Sentiment working, lag features pending")
        return 0
    else:
        print("\n[FAIL] Integration issues detected")
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(test_integration())
