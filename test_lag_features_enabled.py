"""
Test lag features with proper config enabled.
"""

from primitives import HybridSentimentAnalyzer, HybridSentimentConfig

# Test 1: Check if lag buffer initializes
print("=" * 70)
print("TEST 1: Lag Buffer Initialization")
print("=" * 70)

config = HybridSentimentConfig(
    enable_lag_features=True,
    max_lag_bars=360
)

analyzer = HybridSentimentAnalyzer(config)

if analyzer.lag_buffer:
    print("[PASS] Lag buffer initialized")
    print(f"  Max lag bars: {analyzer.lag_buffer.config.max_lag_bars}")
else:
    print("[FAIL] Lag buffer NOT initialized")

# Test 2: Analyze with lag features
print("\n" + "=" * 70)
print("TEST 2: Lag Features in Output")
print("=" * 70)

# First analysis - buffer will be empty
result1 = analyzer.analyze(
    "Bitcoin surging to all-time highs!",
    symbol='BTCUSDT',
    source='news'
)

print(f"\nFirst analysis:")
print(f"  Score: {result1['score']:.3f}")
print(f"  Lag features: {list(result1.get('lag_features', {}).keys())}")

# Add 100 more analyses to populate buffer
for i in range(100):
    analyzer.analyze(
        f"Test sentiment {i}",
        symbol='BTCUSDT',
        source='news'
    )

# Final analysis - buffer should have lag features
result_final = analyzer.analyze(
    "Bitcoin continues strong momentum",
    symbol='BTCUSDT',
    source='news'
)

print(f"\nAfter 101 analyses:")
print(f"  Score: {result_final['score']:.3f}")
lag_features = result_final.get('lag_features', {})
print(f"  Lag features available: {len(lag_features)}")

if lag_features:
    print("\n  Lag feature values:")
    for key, value in sorted(lag_features.items()):
        if value != 0.0:
            print(f"    {key}: {value:.3f}")

# Test 3: Check buffer stats
print("\n" + "=" * 70)
print("TEST 3: Buffer Stats")
print("=" * 70)

if analyzer.lag_buffer:
    stats = analyzer.lag_buffer.get_stats()
    print(f"  Total symbols: {stats['total_symbols']}")
    print(f"  Total updates: {stats['total_updates']}")
    print(f"  Buffer length: {analyzer.lag_buffer.get_buffer_length('BTCUSDT', 'news')}")
    print(f"  Memory (KB): {stats['memory_estimate_kb']:.2f}")

    if stats['total_updates'] > 100:
        print("\n[PASS] Lag features are working!")
    else:
        print("\n[WARN] Need more updates for full lag features")

print("\n" + "=" * 70)
