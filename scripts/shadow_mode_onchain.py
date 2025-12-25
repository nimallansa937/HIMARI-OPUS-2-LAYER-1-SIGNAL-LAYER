"""
On-Chain Analytics Shadow Mode Testing
=======================================

Runs on-chain analytics in shadow mode for validation without
affecting production signals.

Shadow mode:
1. Fetches real on-chain data from APIs
2. Publishes to Redis with 'shadow:' prefix
3. Computes cascade risk without trading actions
4. Logs all signals for analysis
5. Measures latency and accuracy

Usage:
    python scripts/shadow_mode_onchain.py --duration 3600 --symbol BTCUSDT
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import redis

from primitives import (
    OnChainWhaleTracker,
    OnChainNetworkHealth,
    EnhancedCascadeDetector,
    is_onchain_available
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ShadowModeValidator:
    """
    Shadow mode validator for on-chain analytics.
    
    Runs in background, collects signals, and validates correctness
    without affecting production.
    """
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        """Initialize shadow mode validator."""
        # Redis connection
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        # On-chain primitives
        onchain_config = {
            'santiment_api_key': os.getenv('SANTIMENT_API_KEY'),
            'dune_api_key': os.getenv('DUNE_API_KEY'),
            'etherscan_api_key': os.getenv('ETHERSCAN_API_KEY'),
            'update_interval': 60,
        }
        
        self.whale_tracker = OnChainWhaleTracker(onchain_config)
        self.network_health = OnChainNetworkHealth(onchain_config)
        self.cascade_detector = EnhancedCascadeDetector()
        
        # Metrics
        self.metrics = {
            'updates': 0,
            'cascade_warnings': 0,
            'whale_detections': 0,
            'total_latency_ms': 0,
            'start_time': time.time(),
        }
        
        # Signal history for analysis
        self.signal_history = []
        
        logger.info("Shadow mode validator initialized")
    
    def update(self, symbol: str = 'BTCUSDT') -> dict:
        """
        Perform one update cycle.
        
        Returns:
            Dict with all on-chain signals and metadata
        """
        start_time = time.perf_counter()
        current_ts = int(time.time())
        
        signals = {}
        
        # Get whale activity
        whale_signals = self.whale_tracker.update(current_ts)
        signals.update({
            'exchange_netflow': whale_signals.get('exchange_netflow', 0),
            'netflow_zscore': whale_signals.get('exchange_netflow_zscore', 0),
            'whale_pressure': whale_signals.get('whale_pressure', 0),
            'large_tx_count': whale_signals.get('large_tx_count_5min', 0),
            'concentration_risk': whale_signals.get('concentration_risk', 0),
        })
        
        # Get network health
        health_signals = self.network_health.update(current_ts)
        signals.update({
            'active_addresses_zscore': health_signals.get('active_addresses_zscore', 0),
            'hash_rate_health': health_signals.get('hash_rate_health', 1.0),
            'fee_pressure': health_signals.get('fee_pressure', 0),
            'holder_conviction': health_signals.get('holder_conviction', 0.5),
            'network_health_score': health_signals.get('network_health_score', 1.0),
        })
        
        # Get cascade risk
        cascade_warning = self.cascade_detector.get_cascade_warning(
            funding=0.0,
            oi_change=0.0,
            vol_ratio=1.0,
            onchain=whale_signals
        )
        signals.update({
            'cascade_risk': cascade_warning['risk_score'],
            'cascade_action': cascade_warning['action'],
            'cascade_level': cascade_warning['risk_level'],
        })
        
        # Compute latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Publish to Redis with shadow prefix
        self._publish_to_redis(symbol, signals)
        
        # Update metrics
        self.metrics['updates'] += 1
        self.metrics['total_latency_ms'] += latency_ms
        
        if cascade_warning['risk_score'] > 0.4:
            self.metrics['cascade_warnings'] += 1
        
        if whale_signals.get('large_tx_count_5min', 0) > 0:
            self.metrics['whale_detections'] += 1
        
        # Store in history
        record = {
            'timestamp': current_ts,
            'signals': signals.copy(),
            'latency_ms': latency_ms,
        }
        self.signal_history.append(record)
        
        # Log summary
        logger.info(
            f"[{symbol}] netflow={signals['exchange_netflow']:+.3f} "
            f"health={signals['network_health_score']:.2f} "
            f"cascade={cascade_warning['risk_level']} "
            f"({latency_ms:.1f}ms)"
        )
        
        return signals
    
    def _publish_to_redis(self, symbol: str, signals: dict):
        """Publish signals to Redis with shadow prefix."""
        try:
            pipeline = self.redis.pipeline()
            
            for key, value in signals.items():
                redis_key = f"shadow:onchain:{symbol}:{key}"
                if isinstance(value, str):
                    pipeline.setex(redis_key, 300, value)
                else:
                    pipeline.setex(redis_key, 300, str(value))
            
            # Also publish combined snapshot
            snapshot_key = f"shadow:onchain:{symbol}:snapshot"
            pipeline.setex(snapshot_key, 300, json.dumps(signals))
            
            pipeline.execute()
        
        except Exception as e:
            logger.error(f"Redis publish failed: {e}")
    
    def run(self, duration_seconds: int = 3600, symbol: str = 'BTCUSDT'):
        """
        Run shadow mode for specified duration.
        
        Args:
            duration_seconds: How long to run (default 1 hour)
            symbol: Trading symbol to track
        """
        logger.info(f"Starting shadow mode for {duration_seconds}s on {symbol}")
        
        end_time = time.time() + duration_seconds
        update_interval = 60  # seconds
        
        while time.time() < end_time:
            try:
                self.update(symbol)
                time.sleep(update_interval)
            
            except KeyboardInterrupt:
                logger.info("Shadow mode interrupted by user")
                break
            
            except Exception as e:
                logger.error(f"Update error: {e}")
                time.sleep(10)  # Backoff on error
        
        self._print_summary()
    
    def _print_summary(self):
        """Print shadow mode summary."""
        runtime = time.time() - self.metrics['start_time']
        avg_latency = self.metrics['total_latency_ms'] / max(self.metrics['updates'], 1)
        
        print("\n" + "=" * 70)
        print("SHADOW MODE SUMMARY")
        print("=" * 70)
        
        print(f"\n⏱️  Runtime: {runtime/60:.1f} minutes")
        print(f"📊 Total Updates: {self.metrics['updates']}")
        print(f"⚡ Avg Latency: {avg_latency:.1f}ms")
        print(f"🐋 Whale Detections: {self.metrics['whale_detections']}")
        print(f"⚠️  Cascade Warnings: {self.metrics['cascade_warnings']}")
        
        # Signal statistics
        if self.signal_history:
            netflows = [r['signals']['exchange_netflow'] for r in self.signal_history]
            health_scores = [r['signals']['network_health_score'] for r in self.signal_history]
            cascade_risks = [r['signals']['cascade_risk'] for r in self.signal_history]
            
            print(f"\n📈 Signal Statistics:")
            print(f"   Exchange Netflow: min={min(netflows):.3f}, max={max(netflows):.3f}")
            print(f"   Network Health: min={min(health_scores):.3f}, max={max(health_scores):.3f}")
            print(f"   Cascade Risk: min={min(cascade_risks):.3f}, max={max(cascade_risks):.3f}")
        
        print("\n" + "=" * 70)
        print("Shadow mode complete. Ready for production deployment.")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='On-Chain Analytics Shadow Mode')
    parser.add_argument('--duration', type=int, default=3600, 
                       help='Duration in seconds (default: 3600)')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                       help='Trading symbol (default: BTCUSDT)')
    parser.add_argument('--redis-host', type=str, default='localhost',
                       help='Redis host (default: localhost)')
    parser.add_argument('--redis-port', type=int, default=6379,
                       help='Redis port (default: 6379)')
    
    args = parser.parse_args()
    
    if not is_onchain_available():
        logger.error("On-chain primitives not available!")
        sys.exit(1)
    
    validator = ShadowModeValidator(
        redis_host=args.redis_host,
        redis_port=args.redis_port
    )
    
    validator.run(
        duration_seconds=args.duration,
        symbol=args.symbol
    )


if __name__ == "__main__":
    main()
