#!/usr/bin/env python
"""
HIMARI L1 Signal Layer - Launcher Script

This script properly initializes the package and starts the signal processor.
It handles the relative imports issue when running from the project directory.

Usage:
    python run.py
"""

import sys
import os

# Add the parent directory to path so relative imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Note: python-dotenv not installed, using environment variables directly")

# Now we can import with absolute imports
# First, let's patch the signal_processor to use absolute imports temporarily
import importlib.util

def run_signal_processor():
    """Start the HIMARI L1 Signal Processor."""
    print("=" * 60)
    print("HIMARI L1 Signal Layer")
    print("=" * 60)
    print()
    
    # Check if infrastructure is running
    print("[1/4] Checking Redis connection...")
    try:
        import redis
        from config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD
        r = redis.Redis(
            host=REDIS_HOST, 
            port=REDIS_PORT, 
            password=REDIS_PASSWORD or None,
            socket_timeout=5
        )
        r.ping()
        print(f"  ✓ Redis connected at {REDIS_HOST}:{REDIS_PORT}")
    except Exception as e:
        print(f"  ✗ Redis connection failed: {e}")
        print()
        print("Make sure your HIMARI OPUS infrastructure is running:")
        print("  cd 'HIMARI OPUS'")
        print("  docker-compose -f docker-compose.prod.yml up -d")
        return False
    
    print()
    print("[2/4] Checking Kafka/Redpanda connection...")
    try:
        from kafka import KafkaConsumer
        from config import KAFKA_BOOTSTRAP, KAFKA_INPUT_TOPIC
        
        consumer = KafkaConsumer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            api_version_auto_timeout_ms=5000,
        )
        topics = consumer.topics()
        consumer.close()
        print(f"  ✓ Kafka connected at {KAFKA_BOOTSTRAP}")
        print(f"  ✓ Available topics: {list(topics)[:5]}...")
    except Exception as e:
        print(f"  ✗ Kafka connection failed: {e}")
        print()
        print("Make sure Redpanda is running in your HIMARI OPUS:")
        print("  docker-compose -f docker-compose.prod.yml up -d redpanda")
        return False
    
    print()
    print("[3/4] Loading HIMARI L1 components...")
    try:
        from primitives import (
            WelfordVariance, KalmanFilter, UltimateSmoother, 
            OnlineGARCH, MovingHurst, SyntheticVolumeDelta
        )
        from ml import LorentzianKNN, EnsembleFusion
        from fusion import DempsterShafer
        print("  ✓ All primitives loaded")
        print("  ✓ ML components loaded")
        print("  ✓ Fusion components loaded")
    except Exception as e:
        print(f"  ✗ Component loading failed: {e}")
        return False
    
    print()
    print("[4/4] Starting Signal Processor...")
    print("-" * 60)
    print()
    
    # Now run a standalone version that doesn't use relative imports
    from standalone_processor import StandaloneSignalProcessor
    processor = StandaloneSignalProcessor()
    processor.run()
    
    return True

if __name__ == "__main__":
    try:
        run_signal_processor()
    except KeyboardInterrupt:
        print("\n\nShutdown requested. Goodbye!")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
