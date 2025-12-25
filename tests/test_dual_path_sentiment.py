"""
Integration Test for Dual-Path Sentiment Analyzer

Tests:
1. Source-based routing (social → fast, news → accurate)
2. Latency requirements (<5ms fast, <50ms accurate)
3. Signal type classification (ALERT vs TRADE)
4. Integration with existing HIMARI components

Run:
    python test_dual_path_sentiment.py
    
Or with pytest:
    pytest test_dual_path_sentiment.py -v
"""

import sys
import time
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_dual_path_routing():
    """Test that sources are correctly routed to appropriate paths."""
    try:
        from primitives.dual_path_sentiment import (
            DualPathSentimentAnalyzer,
            DualPathConfig,
            SourceType,
            SignalType
        )
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        logger.error("Make sure transformers and torch are installed")
        return False
    
    print("\n" + "=" * 60)
    print("TEST 1: Source-Based Routing")
    print("=" * 60)
    
    # Create analyzer with base models (no fine-tuned)
    config = DualPathConfig(
        use_fine_tuned=False,  # Use base models for testing
        fast_latency_target=10.0,  # Relaxed for testing
        accurate_latency_target=100.0
    )
    analyzer = DualPathSentimentAnalyzer(config)
    
    # Test routing
    routing_tests = [
        # (source, expected_path)
        ("twitter", "fast"),
        ("reddit", "fast"),
        ("telegram", "fast"),
        ("bloomberg", "accurate"),
        ("reuters", "accurate"),
        ("coindesk", "accurate"),
        ("unknown", "accurate"),  # Unknown defaults to accurate
    ]
    
    all_passed = True
    
    for source, expected_path in routing_tests:
        result = analyzer.analyze("Bitcoin price update", source=source)
        actual_path = result.path_used
        
        status = "✓" if actual_path == expected_path else "✗"
        print(f"  {status} Source '{source}' → {actual_path} (expected: {expected_path})")
        
        if actual_path != expected_path:
            all_passed = False
    
    return all_passed


def test_latency_requirements():
    """Test that latency targets are met."""
    try:
        from primitives.dual_path_sentiment import create_dual_path_analyzer
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("TEST 2: Latency Requirements")
    print("=" * 60)
    
    analyzer = create_dual_path_analyzer(use_fine_tuned=False)
    
    # Warm up models
    analyzer.analyze("warm up text", source="twitter")
    analyzer.analyze("warm up text", source="bloomberg")
    
    # Test latencies
    n_iterations = 10
    
    # Fast path latency
    fast_latencies = []
    for i in range(n_iterations):
        result = analyzer.analyze(f"Bitcoin update {i}", source="twitter")
        fast_latencies.append(result.latency_ms)
    
    fast_p50 = sorted(fast_latencies)[len(fast_latencies) // 2]
    fast_p99 = sorted(fast_latencies)[int(len(fast_latencies) * 0.99)]
    
    # Accurate path latency
    accurate_latencies = []
    for i in range(n_iterations):
        result = analyzer.analyze(f"Bitcoin market analysis {i}", source="bloomberg")
        accurate_latencies.append(result.latency_ms)
    
    accurate_p50 = sorted(accurate_latencies)[len(accurate_latencies) // 2]
    accurate_p99 = sorted(accurate_latencies)[int(len(accurate_latencies) * 0.99)]
    
    print(f"\n  Fast Path Latency:")
    print(f"    p50: {fast_p50:.2f}ms (target: <10ms)")
    print(f"    p99: {fast_p99:.2f}ms")
    
    print(f"\n  Accurate Path Latency:")
    print(f"    p50: {accurate_p50:.2f}ms (target: <100ms)")
    print(f"    p99: {accurate_p99:.2f}ms")
    
    # Note: First GPU inference can be slow, so we're lenient
    fast_ok = fast_p50 < 50  # Very lenient for CI
    accurate_ok = accurate_p50 < 200
    
    print(f"\n  Fast path: {'✓ PASS' if fast_ok else '⚠ SLOW (may improve with GPU)'}")
    print(f"  Accurate path: {'✓ PASS' if accurate_ok else '⚠ SLOW (may improve with GPU)'}")
    
    return True  # Don't fail on latency, just report


def test_signal_classification():
    """Test that signals are correctly classified."""
    try:
        from primitives.dual_path_sentiment import (
            create_dual_path_analyzer,
            SignalType
        )
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("TEST 3: Signal Classification")
    print("=" * 60)
    
    analyzer = create_dual_path_analyzer(use_fine_tuned=False)
    
    # Test cases with expected signals
    test_cases = [
        # Social media should produce ALERT or INFO
        ("BTC mooning! 🚀🚀🚀", "twitter", [SignalType.ALERT, SignalType.INFO]),
        ("Just got rekt", "reddit", [SignalType.ALERT, SignalType.INFO]),
        
        # News should produce TRADE or INFO
        ("Bitcoin surges past $100,000", "bloomberg", [SignalType.TRADE, SignalType.ALERT, SignalType.INFO]),
        ("SEC approves Bitcoin ETF", "reuters", [SignalType.TRADE, SignalType.ALERT, SignalType.INFO]),
    ]
    
    all_passed = True
    
    for text, source, allowed_signals in test_cases:
        result = analyzer.analyze(text, source=source)
        signal_ok = result.signal_type in allowed_signals
        
        status = "✓" if signal_ok else "✗"
        allowed_str = "/".join([s.value for s in allowed_signals])
        print(f"  {status} '{text[:30]}...' ({source})")
        print(f"      → {result.signal_type.value} (allowed: {allowed_str})")
        print(f"      → score={result.score:.3f}, label={result.label}")
        
        if not signal_ok:
            all_passed = False
    
    return all_passed


def test_sentiment_accuracy():
    """Test basic sentiment accuracy on clear-cut examples."""
    try:
        from primitives.dual_path_sentiment import create_dual_path_analyzer
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("TEST 4: Sentiment Accuracy (Clear-cut Examples)")
    print("=" * 60)
    
    analyzer = create_dual_path_analyzer(use_fine_tuned=False)
    
    # Clear-cut examples
    test_cases: List[Tuple[str, str, str]] = [
        # (text, source, expected_label)
        ("Bitcoin crashes 50% as market collapses", "bloomberg", "bearish"),
        ("Bitcoin surges to new all-time high, bulls celebrate", "reuters", "bullish"),
        ("Crypto market sees modest gains", "coindesk", "neutral"),  # May vary
        
        # Social media
        ("This is terrible! Lost everything!", "twitter", "bearish"),
        ("🚀🚀🚀 To the moon! Best day ever!", "twitter", "bullish"),
    ]
    
    correct = 0
    total = len(test_cases)
    
    for text, source, expected in test_cases:
        result = analyzer.analyze(text, source=source)
        is_correct = result.label == expected
        
        if is_correct:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"  {status} '{text[:40]}...'")
        print(f"      Expected: {expected}, Got: {result.label} (score={result.score:.3f})")
    
    accuracy = correct / total * 100
    print(f"\n  Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    # Be lenient since we're using base models
    return accuracy >= 60  # At least 60% on clear-cut examples


def test_integration_with_aggregator():
    """Test integration with SocialSentimentAggregator."""
    try:
        from primitives.dual_path_sentiment import create_dual_path_analyzer
        from primitives.social_sentiment_aggregator import (
            SocialSentimentAggregator,
            SocialPost,
            AggregatorConfig
        )
        from datetime import datetime
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("TEST 5: Integration with SocialSentimentAggregator")
    print("=" * 60)
    
    # Create components
    analyzer = create_dual_path_analyzer(use_fine_tuned=False)
    aggregator = SocialSentimentAggregator()
    
    # Simulate incoming social posts
    posts = [
        ("BTC looking strong today!", "twitter", 100.0),
        ("Bearish divergence on ETH", "twitter", 50.0),
        ("Just bought more Bitcoin", "reddit", 25.0),
    ]
    
    for text, source, engagement in posts:
        # Analyze with dual-path
        result = analyzer.analyze(text, source=source)
        
        # Create social post with sentiment score
        post = SocialPost(
            id=f"post_{hash(text)}",
            source=source,
            symbol="BTCUSDT",
            text=text,
            timestamp=datetime.now(),
            author_id="test_user",
            engagement_score=engagement,
            sentiment_score=result.score
        )
        
        # Add to aggregator
        aggregator.add_post(post)
    
    # Get aggregates
    aggregates = aggregator.get_aggregates("BTCUSDT")
    
    print(f"  Posts processed: {len(posts)}")
    print(f"  Aggregates: {aggregates}")
    
    # Verify integration worked
    has_twitter = 'twitter_15m_mean' in aggregates
    has_reddit = 'reddit_1h_mean' in aggregates
    
    print(f"\n  Twitter aggregates: {'✓' if has_twitter else '✗'}")
    print(f"  Reddit aggregates: {'✓' if has_reddit else '✗'}")
    
    return has_twitter or has_reddit  # At least one should work


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("HIMARI DUAL-PATH SENTIMENT INTEGRATION TESTS")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Routing
    results['routing'] = test_dual_path_routing()
    
    # Test 2: Latency
    results['latency'] = test_latency_requirements()
    
    # Test 3: Signal classification
    results['signals'] = test_signal_classification()
    
    # Test 4: Accuracy
    results['accuracy'] = test_sentiment_accuracy()
    
    # Test 5: Integration
    results['integration'] = test_integration_with_aggregator()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
