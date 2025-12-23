#!/usr/bin/env python3
"""
Quick Test for HIMARI Data Ingestion Layer

This script tests the Binance WebSocket connection WITHOUT requiring
Kafka/Redpanda. Use this to verify the connector works before full deployment.

Usage:
    python quick_test.py

Expected output:
    - Connection to Binance
    - 5-10 OHLCV messages printed
    - Clean disconnection
"""

import asyncio
import json
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Message counter
message_count = 0
max_messages = 10  # Stop after this many messages


async def on_message(message):
    """Callback to print received messages."""
    global message_count
    message_count += 1
    
    # Pretty print
    print(f"\n--- Message {message_count} ---")
    print(f"Symbol: {message.get('symbol')}")
    print(f"Exchange: {message.get('exchange')}")
    print(f"Time: {message.get('timestamp')}")
    print(f"OHLCV: O={message.get('open'):.2f} H={message.get('high'):.2f} "
          f"L={message.get('low'):.2f} C={message.get('close'):.2f}")
    print(f"Volume: {message.get('volume'):.4f}")
    print(f"Is Closed: {message.get('is_closed')}")
    
    if message_count >= max_messages:
        print(f"\n✓ Received {max_messages} messages successfully!")
        print("Stopping test...")
        raise KeyboardInterrupt


async def test_binance_connector():
    """Test Binance WebSocket connection."""
    print("=" * 60)
    print("HIMARI Data Ingestion - Quick Test")
    print("=" * 60)
    print()
    print("Testing Binance WebSocket connector...")
    print("Will receive 10 messages then stop.")
    print()
    
    try:
        # Import connector
        from connectors.binance import BinanceConnector
        
        # Create connector for just BTC
        connector = BinanceConnector(
            symbols=["BTCUSDT"],
            interval="1m",
            callback=on_message,
        )
        
        # Connect with timeout
        try:
            await asyncio.wait_for(
                connector.connect(),
                timeout=30  # 30 second timeout
            )
        except asyncio.TimeoutError:
            print("Test timed out after 30 seconds")
        except KeyboardInterrupt:
            pass
        
        # Disconnect
        await connector.disconnect()
        
        print()
        print("=" * 60)
        print(f"Test Results:")
        print(f"  Messages received: {message_count}")
        print(f"  Status: {'✓ PASSED' if message_count > 0 else '✗ FAILED'}")
        print("=" * 60)
        
        return message_count > 0
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Make sure you're in the himari_data_ingestion directory")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


async def test_kafka_publisher_mock():
    """Test Kafka publisher without actual Kafka connection."""
    print()
    print("Testing Kafka publisher (mock mode)...")
    
    try:
        from publishers.kafka_publisher import KafkaPublisher, PublisherMetrics
        
        # Create publisher (won't actually connect)
        publisher = KafkaPublisher(
            bootstrap_servers="localhost:9092",
            topic="raw_market_data"
        )
        
        # Check metrics work
        metrics = publisher.get_metrics()
        print(f"  Initial metrics: {metrics}")
        
        print("  ✓ Publisher module loads correctly")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


async def test_config():
    """Test configuration loading."""
    print()
    print("Testing configuration...")
    
    try:
        from config import (
            KAFKA_BOOTSTRAP, KAFKA_TOPIC_RAW, SYMBOLS,
            SYMBOL_MAPPING, DEFAULT_CONFIG
        )
        
        print(f"  Kafka: {KAFKA_BOOTSTRAP}")
        print(f"  Topic: {KAFKA_TOPIC_RAW}")
        print(f"  Symbols: {SYMBOLS}")
        print(f"  Config: {DEFAULT_CONFIG}")
        print("  ✓ Configuration loads correctly")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


async def main():
    """Run all tests."""
    results = {}
    
    # Test 1: Config
    results['config'] = await test_config()
    
    # Test 2: Publisher (mock)
    results['publisher'] = await test_kafka_publisher_mock()
    
    # Test 3: Binance connector (real connection)
    results['binance'] = await test_binance_connector()
    
    # Summary
    print()
    print("=" * 60)
    print("Test Summary:")
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(results.values())
    print()
    if all_passed:
        print("✓ All tests passed!")
        print()
        print("Next steps:")
        print("1. Start your Redpanda cluster")
        print("2. Run: python main.py")
        print("3. Verify messages appear in Redpanda")
        print("4. Your Flink pipeline will process them automatically")
    else:
        print("✗ Some tests failed. Check errors above.")
    
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(0)
