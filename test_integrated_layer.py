"""
Test IntegratedSignalLayer - Complete System Integration

Validates all 7 primitives working together with SRM gating.
"""

from dotenv import load_dotenv
load_dotenv()  # Load .env file first

from primitives import IntegratedSignalLayer, IntegratedSignalOutput
from config import load_enhanced_config
import time

def test_integrated_layer():
    """Test complete integrated signal layer."""
    
    print("=" * 60)
    print("Integrated Signal Layer Test")
    print("=" * 60)
    print()
    
    # Load configuration
    config = load_enhanced_config()
    
    if not config.enabled:
        print("❌ Enhanced Layer 1 is DISABLED")
        print("   Set HIMARI_ENHANCED_LAYER1_ENABLED=true to test")
        return
    
    # Initialize (without Redis for now)
    print("Initializing IntegratedSignalLayer...")
    layer = IntegratedSignalLayer(config, redis_client=None)
    print()
    
    # Simulate price data
    import numpy as np
    
    print("Testing with simulated market data...")
    print()
    
    prices = []
    signals = []
   
    for i in range(100):
        # Generate realistic price
        if i == 0:
            price = 100.0
        else:
            # Random walk with slight upward drift
            price = prices[-1] * (1 + np.random.normal(0.001, 0.01))
        
        prices.append(price)
        
        # Create OHLCV
        ohlcv = {
            'open': price * 0.999,
            'high': price * 1.001,
            'low': price * 0.998,
            'close': price,
            'volume': 1000 + np.random.randint(-200, 200)
        }
        
        # Create orderbook
        orderbook = {
            'bids': [
                (price * 0.999, 10 + np.random.randint(-2, 2)),
                (price * 0.998, 15 + np.random.randint(-3, 3)),
                (price * 0.997, 20 + np.random.randint(-4, 4)),
            ],
            'asks': [
                (price * 1.001, 12 + np.random.randint(-2, 2)),
                (price * 1.002, 18 + np.random.randint(-3, 3)),
                (price * 1.003, 22 + np.random.randint(-4, 4)),
            ]
        }
        
        # Update layer
        signal = layer.update('BTCUSDT', ohlcv, orderbook)
        signals.append(signal)
        
        # Print status every 20 bars
        if (i + 1) % 20 == 0:
            print(f"Bar {i+1}:")
            print(f"  Price: ${price:.2f}")
            print(f"  Signal: {signal.composite_signal:+.3f}")
            print(f"  Regime: {signal.regime} ({signal.regime_confidence:.1%})")
            print(f"  Components: {len(signal.components)}")
            print(f"  SRM Action: {signal.srm_action}")
            print(f"  Position Mult: {signal.position_multiplier:.1f}")
            print(f"  Latency: {signal.total_latency_ms:.2f}ms")
            print()
    
    print("=" * 60)
    print("Performance Summary")
    print("=" * 60)
    
    # Calculate statistics
    avg_latency = sum(s.total_latency_ms for s in signals) / len(signals)
    max_latency = max(s.total_latency_ms for s in signals)
    
    # Count regime changes
    regime_changes = sum(
        1 for i in range(1, len(signals))
        if signals[i].regime != signals[i-1].regime
    )
    
    print(f"Updates: {len(signals)}")
    print(f"Avg Latency: {avg_latency:.2f}ms")
    print(f"Max Latency: {max_latency:.2f}ms")
    print(f"Regime Changes: {regime_changes}")
    print(f"Final Regime: {signals[-1].regime}")
    print(f"Final Confidence: {signals[-1].regime_confidence:.1%}")
    print()
    
    # Check performance targets
    print("Performance Targets:")
    print(f"  Latency < 10ms: {'[PASS]' if avg_latency < 10 else '[FAIL]'}")
    print(f"  Max latency < 25ms: {'[PASS]' if max_latency < 25 else '[FAIL]'}")
    print()

    print("=" * 60)
    print("[+] IntegratedSignalLayer Integration Test Complete")
    print("=" * 60)

if __name__ == '__main__':
    test_integrated_layer()
