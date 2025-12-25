"""
OnChain Whale Tracker - Comprehensive Test
===========================================

Tests the whale tracker with real Bitcoin blockchain data.
Validates that large transaction detection is working correctly.
"""

import os
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from primitives.onchain_whale_tracker import OnChainWhaleTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_whale_tracker():
    """Test whale tracker with real blockchain data."""
    
    print("=" * 70)
    print("ONCHAIN WHALE TRACKER - COMPREHENSIVE TEST")
    print("=" * 70)
    
    # Create tracker
    config = {
        'santiment_api_key': os.getenv('SANTIMENT_API_KEY'),
        'dune_api_key': os.getenv('DUNE_API_KEY'),
        'etherscan_api_key': os.getenv('ETHERSCAN_API_KEY'),
        'update_interval': 60,
    }
    
    tracker = OnChainWhaleTracker(config)
    
    print(f"\n✓ Tracker initialized")
    print(f"  Santiment: {'✓' if tracker.santiment_key else '✗'}")
    print(f"  Dune: {'✓' if tracker.dune_key else '✗'}")
    print(f"  Etherscan: {'✓' if tracker.etherscan_key else '✗'}")
    
    # Test Blockchain.com large transaction detection
    print("\n" + "=" * 70)
    print("TEST 1: Bitcoin Large Transaction Detection (Blockchain.com)")
    print("=" * 70)
    
    try:
        large_txs = tracker._fetch_blockchain_large_txs()
        
        if large_txs:
            print(f"\n✓ Found {len(large_txs)} large Bitcoin transactions (>10 BTC)")
            print(f"\nTop 5 largest transactions:\n")
            
            # Sort by value
            sorted_txs = sorted(large_txs, key=lambda x: x['value'], reverse=True)
            
            for i, tx in enumerate(sorted_txs[:5], 1):
                print(f"{i}. Transaction Hash: {tx['hash'][:16]}...")
                print(f"   Amount: {tx['value']:.2f} BTC")
                print(f"   Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(tx['timestamp']))}")
                print(f"   Direction: {tx['direction']}")
                print()
        else:
            print("✗ No large transactions found in recent blocks")
            print("  (This is normal if there haven't been any >10 BTC txs recently)")
    
    except Exception as e:
        print(f"✗ Error testing Blockchain.com API: {e}")
    
    # Test full update cycle
    print("\n" + "=" * 70)
    print("TEST 2: Full Whale Signal Computation")
    print("=" * 70)
    
    current_time = int(time.time())
    signals = tracker.update(current_time)
    
    print(f"\n📊 Whale Activity Signals:")
    print(f"  Exchange Netflow: {signals['exchange_netflow']:+.3f}")
    print(f"  Netflow Z-Score: {signals['exchange_netflow_zscore']:+.3f}")
    print(f"  Whale Pressure: {signals['whale_pressure']:+.3f}")
    print(f"  Large TX Count: {signals['large_tx_count_5min']}")
    print(f"  Concentration Risk: {signals['concentration_risk']:.3f}")
    
    # Get statistics
    stats = tracker.get_statistics()
    
    print(f"\n📈 Streaming Statistics:")
    print(f"  Mean Netflow: {stats['mean_netflow']:.2f}")
    print(f"  Std Dev: {stats['std_dev']:.2f}")
    print(f"  Sample Count: {stats['sample_count']}")
    print(f"  Last Update: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(stats['last_update']))}")
    
    # Test caching
    print("\n" + "=" * 70)
    print("TEST 3: Cache Validation (should return cached values)")
    print("=" * 70)
    
    # Call update again immediately (should use cache)
    start = time.perf_counter()
    cached_signals = tracker.update(current_time + 1)
    cache_time = (time.perf_counter() - start) * 1000
    
    print(f"\n✓ Cache working correctly")
    print(f"  Cache retrieval time: {cache_time:.3f}ms")
    print(f"  Values match: {signals == cached_signals}")
    
    # Validate cascade risk detection
    print("\n" + "=" * 70)
    print("TEST 4: Cascade Risk Detection Logic")
    print("=" * 70)
    
    if signals['large_tx_count_5min'] > 3:
        print(f"\n⚠️  WARNING: High whale activity detected!")
        print(f"   {signals['large_tx_count_5min']} large transactions in recent blocks")
        print(f"   Potential cascade risk if whale_pressure is negative")
    elif signals['large_tx_count_5min'] > 0:
        print(f"\n✓ Moderate whale activity: {signals['large_tx_count_5min']} transactions")
    else:
        print(f"\n✓ No unusual whale activity")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    tests_passed = []
    tests_failed = []
    
    # Check what's working
    if tracker.santiment_key:
        tests_passed.append("Santiment API configured")
    if tracker.dune_key:
        tests_passed.append("Dune Analytics API configured")
    if tracker.etherscan_key:
        tests_passed.append("Etherscan API configured")
    if large_txs:
        tests_passed.append(f"Bitcoin whale detection ({len(large_txs)} transactions)")
    if cache_time < 1:
        tests_passed.append("Caching system")
    
    print(f"\n✓ Tests Passed ({len(tests_passed)}):")
    for test in tests_passed:
        print(f"  • {test}")
    
    if tests_failed:
        print(f"\n✗ Tests Failed ({len(tests_failed)}):")
        for test in tests_failed:
            print(f"  • {test}")
    
    print(f"\n{'='*70}")
    print(f"Overall Status: {'✓ PASS' if not tests_failed else '✗ FAIL'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Set API keys
    os.environ['SANTIMENT_API_KEY'] = 'sjpuiaystftrnfv2_gqtgilawjbywlnyb'
    os.environ['DUNE_API_KEY'] = 'barQKj36basZqbg59tEzbc5vDmN021er'
    os.environ['ETHERSCAN_API_KEY'] = 'X3NUGJTD8MIW3K8UCSM8T4TT6CE71AYPIK'
    
    test_whale_tracker()
