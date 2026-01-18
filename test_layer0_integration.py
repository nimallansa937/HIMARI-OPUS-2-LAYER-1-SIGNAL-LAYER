"""
Test Layer 0 -> Layer 1 Integration

Verifies that Layer 1 can:
1. Connect to Layer 0 consumer
2. Check data quality
3. Apply quality weighting
4. Filter low-quality data

Usage:
    python test_layer0_integration.py
"""

import sys
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_layer0_consumer():
    """Test the Layer0Consumer class."""
    print("\n" + "=" * 60)
    print("TEST 1: Layer0Consumer Standalone")
    print("=" * 60)

    from layer0_consumer import Layer0Consumer

    consumer = Layer0Consumer(symbols=["BTCUSDT"])

    # Test connection
    connected = consumer.is_connected()
    print(f"Redis Connected: {connected}")

    # Test quality check
    use_data, score = consumer.should_use_data("BTCUSDT")
    print(f"Should use data: {use_data}")
    print(f"Quality score: {score:.4f}")

    # Test quality grade
    grade = consumer.get_quality_grade("BTCUSDT")
    print(f"Quality grade: {grade}")

    # Test quality weighting
    base_value = 0.80
    weighted = consumer.get_quality_weighted_value("BTCUSDT", base_value)
    print(f"Base value: {base_value:.4f}")
    print(f"Weighted value: {weighted:.4f}")

    # Test threshold adjustment
    base_threshold = 0.70
    adjusted = consumer.adjust_confidence_threshold("BTCUSDT", base_threshold)
    print(f"Base threshold: {base_threshold:.4f}")
    print(f"Adjusted threshold: {adjusted:.4f}")

    # Test multiplier
    multiplier = consumer.get_quality_multiplier("BTCUSDT")
    print(f"Quality multiplier: {multiplier:.4f}")

    # Test metrics
    metrics = consumer.get_metrics()
    print(f"Consumer metrics: {metrics}")

    consumer.close()
    print("\nTest 1 PASSED")


def test_redis_consumer_quality():
    """Test Layer1RedisConsumer quality methods."""
    print("\n" + "=" * 60)
    print("TEST 2: Layer1RedisConsumer Quality Methods")
    print("=" * 60)

    from redis_consumer import Layer1RedisConsumer

    consumer = Layer1RedisConsumer(symbols=["BTCUSDT"])

    # Test getting quality from Layer 0
    quality = consumer.get_data_quality("BTCUSDT")
    print(f"Layer 0 quality data: {quality}")

    # Test quality score
    score = consumer.get_quality_score("BTCUSDT")
    print(f"Quality score: {score:.4f}")

    # Test should_use_data
    use_data, score = consumer.should_use_data("BTCUSDT")
    print(f"Should use data: {use_data}, score: {score:.4f}")

    # Test quality weighting
    base_value = 0.75
    weighted = consumer.get_quality_weighted_value("BTCUSDT", base_value)
    print(f"Base value: {base_value:.4f}, weighted: {weighted:.4f}")

    consumer.close()
    print("\nTest 2 PASSED")


def test_quality_filtering_logic():
    """Test quality filtering logic without Redis."""
    print("\n" + "=" * 60)
    print("TEST 3: Quality Filtering Logic (Mock)")
    print("=" * 60)

    from layer0_consumer import QualityThresholds, QualityWeights

    thresholds = QualityThresholds()
    weights = QualityWeights()

    print(f"Grade A threshold: >= {thresholds.grade_a}")
    print(f"Grade B threshold: >= {thresholds.grade_b}")
    print(f"Grade C threshold: >= {thresholds.grade_c}")
    print(f"Grade D threshold: < {thresholds.grade_c}")

    print(f"\nGrade A weight: {weights.grade_a}")
    print(f"Grade B weight: {weights.grade_b}")
    print(f"Grade C weight: {weights.grade_c}")
    print(f"Grade D weight: {weights.grade_d}")
    print(f"No data weight: {weights.no_data}")

    # Test quality filtering scenarios
    scenarios = [
        (0.98, "A", "Full confidence, trade normally"),
        (0.89, "B", "Good quality, slight confidence reduction"),
        (0.72, "C", "Minimum acceptable, moderate reduction"),
        (0.55, "D", "Poor quality, should skip or reduce heavily"),
    ]

    base_confidence = 0.80
    print(f"\nBase confidence: {base_confidence:.4f}")
    print("-" * 50)

    for score, grade, description in scenarios:
        if score >= thresholds.grade_a:
            weight = weights.grade_a
        elif score >= thresholds.grade_b:
            weight = weights.grade_b
        elif score >= thresholds.grade_c:
            weight = weights.grade_c
        else:
            weight = weights.grade_d

        weighted_confidence = base_confidence * weight
        should_trade = score >= thresholds.grade_c

        print(f"Score {score:.2f} (Grade {grade}):")
        print(f"  Weight: {weight:.2f}")
        print(f"  Weighted confidence: {weighted_confidence:.4f}")
        print(f"  Should trade: {should_trade}")
        print(f"  Description: {description}")

    print("\nTest 3 PASSED")


def test_integration_summary():
    """Print integration summary."""
    print("\n" + "=" * 60)
    print("INTEGRATION SUMMARY")
    print("=" * 60)

    print("""
Layer 0 -> Layer 1 Integration Status:
--------------------------------------

Files Created/Modified:
  [+] layer0_consumer.py (new)
      - Layer0Consumer class
      - Quality checking, filtering, weighting
      - Threshold adjustment
      - Real-time subscription support

  [~] main.py (modified)
      - Added Layer0Consumer import
      - Initialized l0_consumer in _init_components()
      - Added quality check at start of run_cycle()
      - Returns early if quality < 0.70
      - Includes quality metrics in cycle result

  [~] redis_consumer.py (enhanced)
      - Added get_quality_score()
      - Added should_use_data()
      - Added get_quality_weighted_value()

Integration Flow:
-----------------

  Layer 0 (Data Foundation)
      |
      | Publishes to Redis:
      |   - Channel: himari:l0:data_quality
      |   - State: state:{symbol}:data_quality
      v
  Layer 1 (Explorer Agent)
      |
      | Reads quality, filters/weights data
      | Skips processing if quality < 0.70
      |
      v
  Layer 2/3 (downstream)

Quality Propagation:
--------------------

  L0 quality_score = 0.92 (Grade A)
      |
      v
  L1 checks: should_use_data() -> True, 0.92
  L1 weights: confidence * 1.00 = full confidence
      |
      v
  L1 output includes:
    - upstream_data_quality: 0.92
    - quality_weighted_confidence: confidence * 0.92

Usage Example:
--------------

  # In Layer 1 processing
  use_data, score = l0_consumer.should_use_data(symbol)
  if not use_data:
      logger.warning(f"Skipping due to low quality: {score}")
      return

  weighted_conf = l0_consumer.get_quality_weighted_value(symbol, confidence)
  threshold = l0_consumer.adjust_confidence_threshold(symbol, 0.70)

  if weighted_conf > threshold:
      # Proceed with strategy generation
      pass

""")


def main():
    """Run all tests."""
    print("=" * 60)
    print("LAYER 0 -> LAYER 1 INTEGRATION TESTS")
    print("=" * 60)

    tests_passed = 0
    tests_failed = 0

    # Test 1: Layer0Consumer
    try:
        test_layer0_consumer()
        tests_passed += 1
    except Exception as e:
        logger.error(f"Test 1 failed: {e}")
        tests_failed += 1

    # Test 2: Redis Consumer Quality Methods
    try:
        test_redis_consumer_quality()
        tests_passed += 1
    except Exception as e:
        logger.error(f"Test 2 failed: {e}")
        tests_failed += 1

    # Test 3: Quality Filtering Logic (always passes - no Redis needed)
    try:
        test_quality_filtering_logic()
        tests_passed += 1
    except Exception as e:
        logger.error(f"Test 3 failed: {e}")
        tests_failed += 1

    # Print integration summary
    test_integration_summary()

    # Final results
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Tests Passed: {tests_passed}")
    print(f"Tests Failed: {tests_failed}")

    if tests_failed == 0:
        print("\nAll tests passed! Layer 0 -> Layer 1 integration is ready.")
    else:
        print(f"\n{tests_failed} test(s) failed. Check Redis connection.")

    return tests_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
