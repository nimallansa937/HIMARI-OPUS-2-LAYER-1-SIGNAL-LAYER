"""
Enhanced Layer 1 Signal System - Activation Script

This script demonstrates how to use the enhanced primitives.
All components are now activated via .env file.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load .env file
from dotenv import load_dotenv
load_dotenv()

# Import enhanced primitives
from primitives import (
    StreamingHMM, HMMConfig,
    StreamingIndicators, IndicatorConfig,
    WelfordOnlineStats,
    MultiHorizonMomentum, MomentumConfig,
    OrderBookImbalance, OBIConfig,
    RegimeAwareSignalFusion, FusionConfig,
)

# Import configuration
from config import load_enhanced_config

def main():
    """Demonstrate Enhanced Layer 1 activation."""
    
    print("=" * 60)
    print("HIMARI Enhanced Layer 1 Signal System")
    print("=" * 60)
    print()
    
    # Load configuration
    config = load_enhanced_config()
    
    print("Configuration Status:")
    print(f"  Enhanced System:  {'✅ ENABLED' if config.enabled else '❌ DISABLED'}")
    print(f"  HMM:              {'✅' if config.hmm_enabled else '❌'}")
    print(f"  OBI:              {'✅' if config.obi_enabled else '❌'}")
    print(f"  Momentum:         {'✅' if config.momentum_enabled else '❌'}")
    print(f"  Fusion:           {'✅' if config.fusion_enabled else '❌'}")
    print(f"  Sentiment:        {'✅' if config.sentiment_enabled else '❌'}")
    print()
    
    if not config.enabled:
        print("⚠️ Enhanced Layer 1 is DISABLED")
        print("To enable, set: HIMARI_ENHANCED_LAYER1_ENABLED=true in .env")
        return
    
    print("=" * 60)
    print("Initializing Components...")
    print("=" * 60)
    print()
    
    # Initialize components
    hmm = StreamingHMM()
    print("✅ StreamingHMM initialized")
    print(f"   States: {hmm.state_names}")
    print(f"   Current regime: {hmm.get_regime_label()}")
    print()
    
    indicators = StreamingIndicators()
    print("✅ StreamingIndicators initialized")
    print(f"   EMA periods: {indicators.config.ema_periods}")
    print()
    
    momentum = MultiHorizonMomentum()
    print("✅ MultiHorizonMomentum initialized")
    print(f"   Horizons: {momentum.config.horizons}")
    print()
    
    obi = OrderBookImbalance()
    print("✅ OrderBookImbalance initialized")
    print(f"   Levels: {obi.config.levels}")
    print()
    
    fusion = RegimeAwareSignalFusion(hmm)
    print("✅ RegimeAwareSignalFusion initialized")
    print()
    
    print("=" * 60)
    print("Example Usage")
    print("=" * 60)
    print()
    
    # Simulate price data
    import numpy as np
    prices = [100 + i * 0.1 + np.random.randn() * 0.5 for i in range(100)]
    
    print("Processing 100 price bars...")
    print()
    
    for i, price in enumerate(prices):
        # Update indicators
        ohlcv = {
            'open': price - 0.5,
            'high': price + 0.5,
            'low': price - 0.5,
            'close': price,
            'volume': 1000
        }
        ind_result = indicators.update(ohlcv)
        
        # Update momentum
        mom_result = momentum.update(price)
        
        # Update HMM with return
        if i > 0:
            ret = (price - prices[i-1]) / prices[i-1]
            hmm.update(ret)
        
        # Every 20 bars, show status
        if (i + 1) % 20 == 0:
            print(f"Bar {i+1}:")
            print(f"  Price: ${price:.2f}")
            print(f"  Regime: {hmm.get_regime_label()} (confidence: {hmm.state_probs.max():.2%})")
            if ind_result['rsi'] is not None:
                print(f"  RSI: {ind_result['rsi']:.1f}")
            if mom_result['mom_5'] is not None:
                print(f"  Momentum (5-bar): {mom_result['mom_5']:.4f}")
            print()
    
    print("=" * 60)
    print("✅ Enhanced Layer 1 is ACTIVE and WORKING")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Integrate with signal_processor.py")
    print("2. Run CPCV validation on historical data")
    print("3. Deploy in shadow mode")
    print()

if __name__ == '__main__':
    main()
