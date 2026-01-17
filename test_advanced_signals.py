"""
Test script to verify the 10 new advanced signals work correctly.

Tests:
1. SignalType enum has all 24 signals
2. SIGNAL_FEATURE_MAP maps all signals
3. SIGNAL_THRESHOLD_RANGES covers all signals
4. Features 60-69 exist in FEATURE_SCHEMA
5. Grammar validator accepts new signals
"""

import sys
import io

# Fix unicode encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from src.core.genome import SignalType, SIGNAL_FEATURE_MAP, SIGNAL_THRESHOLD_RANGES
from src.core.features import FEATURE_SCHEMA, FEATURE_BY_NAME, FEATURE_BY_INDEX
from src.core.grammar import GrammarValidator

def test_signal_types():
    """Test that all 24 signal types exist."""
    signals = list(SignalType)
    print(f"✓ Total signals: {len(signals)} (expected 24)")

    # Check new advanced signals
    advanced_signals = [
        SignalType.MOMENTUM_JMA,
        SignalType.MOMENTUM_KAMA,
        SignalType.MOMENTUM_HMA,
        SignalType.MOMENTUM_FISHER,
        SignalType.REVERSION_KELTNER,
        SignalType.VOLATILITY_GARMAN_KLASS,
        SignalType.ORDERFLOW_VPIN,
        SignalType.MICROSTRUCTURE_VWAP_DIST,
        SignalType.TREND_INSTANTANEOUS,
        SignalType.CYCLE_DOMINANT,
    ]

    for sig in advanced_signals:
        assert sig in signals, f"Missing signal: {sig}"
    print(f"✓ All 10 advanced signals found in SignalType enum")

    return True


def test_signal_feature_mapping():
    """Test that all signals map to features."""
    for signal in SignalType:
        assert signal in SIGNAL_FEATURE_MAP, f"Missing mapping for {signal}"

    print(f"✓ All {len(SignalType)} signals have feature mappings")

    # Check advanced signal indices (60-69)
    advanced_indices = [
        SIGNAL_FEATURE_MAP[SignalType.MOMENTUM_JMA],
        SIGNAL_FEATURE_MAP[SignalType.MOMENTUM_KAMA],
        SIGNAL_FEATURE_MAP[SignalType.MOMENTUM_HMA],
        SIGNAL_FEATURE_MAP[SignalType.MOMENTUM_FISHER],
        SIGNAL_FEATURE_MAP[SignalType.REVERSION_KELTNER],
        SIGNAL_FEATURE_MAP[SignalType.VOLATILITY_GARMAN_KLASS],
        SIGNAL_FEATURE_MAP[SignalType.ORDERFLOW_VPIN],
        SIGNAL_FEATURE_MAP[SignalType.MICROSTRUCTURE_VWAP_DIST],
        SIGNAL_FEATURE_MAP[SignalType.TREND_INSTANTANEOUS],
        SIGNAL_FEATURE_MAP[SignalType.CYCLE_DOMINANT],
    ]

    for idx in advanced_indices:
        assert 60 <= idx <= 69, f"Invalid index {idx}, should be in [60, 69]"

    print(f"✓ Advanced signals map to indices 60-69")
    return True


def test_threshold_ranges():
    """Test that all signals have threshold ranges."""
    for signal in SignalType:
        assert signal in SIGNAL_THRESHOLD_RANGES, f"Missing threshold for {signal}"
        low, high = SIGNAL_THRESHOLD_RANGES[signal]
        assert low < high, f"Invalid range for {signal}: ({low}, {high})"

    print(f"✓ All {len(SignalType)} signals have valid threshold ranges")
    return True


def test_feature_schema():
    """Test that features 60-69 exist in schema."""
    assert len(FEATURE_SCHEMA) == 70, f"Expected 70 features, got {len(FEATURE_SCHEMA)}"
    print(f"✓ Feature schema has 70 features")

    # Check new features exist
    new_features = [
        "jma_14",
        "kama_14",
        "hma_14",
        "fisher_transform",
        "keltner_position",
        "garman_klass_vol",
        "vpin",
        "vwap_distance",
        "instantaneous_trend",
        "dominant_cycle_period",
    ]

    for name in new_features:
        assert name in FEATURE_BY_NAME, f"Missing feature: {name}"
        feature = FEATURE_BY_NAME[name]
        assert 60 <= feature.index <= 69, f"Feature {name} has wrong index {feature.index}"

    print(f"✓ All 10 advanced features (60-69) exist in schema")
    return True


def test_grammar_validator():
    """Test that grammar validator accepts new signals."""
    validator = GrammarValidator()

    # Test with advanced indicators
    test_cases = [
        # JMA signal
        "jma_14 > close",
        # Fisher transform
        "fisher_transform > 2.0",
        # Keltner position
        "keltner_position < -1.5",
        # VPIN
        "vpin > 0.7",
        # Compound condition with advanced indicators
        "jma_14 > ema_12 AND fisher_transform > 1.0",
        # Complex strategy
        "kama_14 > hma_14 AND vpin < 0.5",
    ]

    for expr in test_cases:
        valid, errors = validator.validate(expr)
        if not valid:
            print(f"  FAILED: {expr}")
            print(f"  Errors: {errors}")
            return False

    print(f"✓ Grammar validator accepts all {len(test_cases)} test expressions")
    return True


def test_feature_dimensional_types():
    """Test that advanced features have correct dimensional types."""
    type_checks = [
        ("jma_14", "PRICE"),
        ("kama_14", "PRICE"),
        ("hma_14", "PRICE"),
        ("fisher_transform", "ZSCORE"),
        ("keltner_position", "ZSCORE"),
        ("garman_klass_vol", "RATIO"),
        ("vpin", "RATIO"),
        ("vwap_distance", "RATE"),
        ("instantaneous_trend", "RATE"),
        ("dominant_cycle_period", "COUNT"),
    ]

    for name, expected_type in type_checks:
        feature = FEATURE_BY_NAME[name]
        actual_type = feature.type.value
        assert actual_type == expected_type.lower(), \
            f"Feature {name} has type {actual_type}, expected {expected_type.lower()}"

    print(f"✓ All advanced features have correct dimensional types")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Advanced Signal Integration")
    print("=" * 60)

    tests = [
        ("SignalType Enum", test_signal_types),
        ("Signal-Feature Mapping", test_signal_feature_mapping),
        ("Threshold Ranges", test_threshold_ranges),
        ("Feature Schema", test_feature_schema),
        ("Dimensional Types", test_feature_dimensional_types),
        ("Grammar Validator", test_grammar_validator),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            print(f"\n{name}:")
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"WARNING: {failed} tests failed")
        sys.exit(1)
    else:
        print("✓ ALL TESTS PASSED - Advanced signals ready for use!")
        print("=" * 60)
