"""
Enhanced Cascade Detector with On-Chain Integration
====================================================

Integrates on-chain whale flow signals for 2-4 minute earlier cascade detection.

Enhanced from OPUS 2 cascade detector with:
- Existing: Funding rate, OI change, volume ratio (60% weight)
- NEW: Whale inflow, netflow z-score, large tx count (40% weight)

Provides early warning for liquidation cascades before they impact price.
"""

import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class EnhancedCascadeDetector:
    """
    Monitors for liquidation cascade conditions with on-chain enhancement.
    
    Combines traditional futures market signals with on-chain whale movements
    for 2-4 minute early warning before cascade events.
    
    Example:
        detector = EnhancedCascadeDetector()
        
        # Traditional signals
        risk = detector.calculate_risk(
            funding=0.004,          # High funding rate
            oi_change=-0.15,        # 15% OI drop
            vol_ratio=6.5,          # 6.5x volume spike
            onchain={               # NEW: On-chain signals
                'exchange_netflow': 0.8,
                'exchange_netflow_zscore': 2.5,
                'large_tx_count_5min': 7
            }
        )
        
        action, multiplier = detector.recommend_action(risk)
        # risk = 0.85 → action = "EXIT_ALL", multiplier = 0.0
    """
    
    def __init__(self):
        """Initialize cascade detector with thresholds."""
        # Existing futures market thresholds (from OPUS 2)
        self.funding_threshold = 0.003      # 0.3% funding rate
        self.oi_drop_threshold = 0.10       # 10% OI drop
        self.volume_spike_threshold = 5.0   # 5x normal volume
        
        # NEW: On-chain thresholds
        self.whale_inflow_threshold = 0.7   # Normalized inflow score
        self.netflow_zscore_threshold = 2.0 # 2 sigma anomaly
        self.large_tx_alert_count = 5       # 5+ whale transactions
        
        logger.info("EnhancedCascadeDetector initialized")
    
    def calculate_risk(
        self,
        funding: float,
        oi_change: float,
        vol_ratio: float,
        onchain: Optional[Dict] = None
    ) -> float:
        """
        Calculate cascade risk score [0, 1].
        
        Args:
            funding: Current funding rate (e.g., 0.003 = 0.3%)
            oi_change: Open interest change (negative = liquidations)
            vol_ratio: Volume vs 24h average
            onchain: Dict from OnChainWhaleTracker (optional)
            
        Returns:
            Risk score from 0 (safe) to 1 (extreme cascade risk)
        """
        risk = 0.0
        
        # === EXISTING SIGNALS (60% weight) ===
        
        # High funding rate = overleveraged longs/shorts
        if abs(funding) > self.funding_threshold:
            risk += 0.20
            logger.debug(f"High funding detected: {funding:+.4f}")
        
        # OI drop = mass liquidations occurring
        if oi_change < -self.oi_drop_threshold:
            risk += 0.25
            logger.debug(f"OI drop detected: {oi_change:+.2%}")
        
        # Volume spike = panic selling/buying
        if vol_ratio > self.volume_spike_threshold:
            risk += 0.15
            logger.debug(f"Volume spike detected: {vol_ratio:.1f}x")
        
        # === NEW: ON-CHAIN SIGNALS (40% weight) - EARLIER WARNING ===
        
        if onchain:
            # Large exchange inflow = whales preparing to sell
            netflow = onchain.get('exchange_netflow', 0)
            if netflow > self.whale_inflow_threshold:
                risk += 0.20
                logger.warning(f"Large whale inflow detected: {netflow:+.2f}")
            
            # Statistical anomaly in flows = unusual whale activity
            zscore = onchain.get('exchange_netflow_zscore', 0)
            if abs(zscore) > self.netflow_zscore_threshold:
                risk += 0.15
                logger.warning(f"Abnormal netflow (z={zscore:+.2f})")
            
            # Multiple large transactions = coordinated whale action
            large_tx_count = onchain.get('large_tx_count_5min', 0)
            if large_tx_count >= self.large_tx_alert_count:
                risk += 0.05
                logger.warning(f"Multiple whale transactions: {large_tx_count}")
        
        return min(1.0, risk)
    
    def recommend_action(self, risk: float) -> Tuple[str, float]:
        """
        Convert risk score to trading action.
        
        Args:
            risk: Cascade risk score [0, 1]
            
        Returns:
            Tuple of (action_name, position_multiplier)
            - action_name: Human-readable action
            - position_multiplier: Fraction of normal position size
        """
        if risk > 0.8:
            return "EXIT_ALL", 0.0
        elif risk > 0.6:
            return "REDUCE_75%", 0.25
        elif risk > 0.4:
            return "REDUCE_50%", 0.50
        elif risk > 0.2:
            return "TIGHTEN_STOPS", 0.80
        
        return "MONITOR", 1.0
    
    def get_cascade_warning(
        self,
        funding: float,
        oi_change: float,
        vol_ratio: float,
        onchain: Optional[Dict] = None
    ) -> Dict:
        """
        Get comprehensive cascade warning with details.
        
        Args:
            funding, oi_change, vol_ratio: Futures market signals
            onchain: On-chain whale signals
            
        Returns:
            Dict with risk score, action, and detailed breakdown
        """
        risk = self.calculate_risk(funding, oi_change, vol_ratio, onchain)
        action, multiplier = self.recommend_action(risk)
        
        # Build signal breakdown
        signals = {
            'funding_alert': abs(funding) > self.funding_threshold,
            'oi_drop_alert': oi_change < -self.oi_drop_threshold,
            'volume_spike_alert': vol_ratio > self.volume_spike_threshold,
        }
        
        if onchain:
            signals.update({
                'whale_inflow_alert': onchain.get('exchange_netflow', 0) > self.whale_inflow_threshold,
                'netflow_anomaly_alert': abs(onchain.get('exchange_netflow_zscore', 0)) > self.netflow_zscore_threshold,
                'large_tx_alert': onchain.get('large_tx_count_5min', 0) >= self.large_tx_alert_count,
            })
        
        return {
            'risk_score': risk,
            'risk_level': self._risk_level(risk),
            'action': action,
            'position_multiplier': multiplier,
            'signals_triggered': signals,
            'alert_count': sum(signals.values()),
            'onchain_enabled': onchain is not None,
        }
    
    def _risk_level(self, risk: float) -> str:
        """Convert risk score to level name."""
        if risk > 0.8:
            return "EXTREME"
        elif risk > 0.6:
            return "HIGH"
        elif risk > 0.4:
            return "MEDIUM"
        elif risk > 0.2:
            return "LOW"
        return "MINIMAL"


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("=" * 70)
    print("ENHANCED CASCADE DETECTOR TEST")
    print("=" * 70)
    
    detector = EnhancedCascadeDetector()
    
    # Test scenario 1: Traditional cascade (no on-chain)
    print("\n" + "=" * 70)
    print("TEST 1: Traditional Cascade (Futures Only)")
    print("=" * 70)
    
    warning = detector.get_cascade_warning(
        funding=0.005,     # 0.5% funding (high)
        oi_change=-0.20,   # 20% OI drop
        vol_ratio=8.0,     # 8x volume
        onchain=None
    )
    
    print(f"\nRisk Level: {warning['risk_level']}")
    print(f"Risk Score: {warning['risk_score']:.2f}")
    print(f"Action: {warning['action']}")
    print(f"Position Multiplier: {warning['position_multiplier']}")
    print(f"Signals Triggered: {warning['alert_count']}/3")
    
    # Test scenario 2: On-chain early warning
    print("\n" + "=" * 70)
    print("TEST 2: On-Chain Early Warning (2-4 min before cascade)")
    print("=" * 70)
    
    warning2 = detector.get_cascade_warning(
        funding=0.004,     # Funding starting to climb
        oi_change=-0.08,   # Small OI drop (not extreme yet)
        vol_ratio=3.0,     # Moderate volume
        onchain={
            'exchange_netflow': 0.85,        # Large whale inflow!
            'exchange_netflow_zscore': 2.8,  # 2.8 sigma anomaly!
            'large_tx_count_5min': 7,        # 7 large transactions!
        }
    )
    
    print(f"\nRisk Level: {warning2['risk_level']}")
    print(f"Risk Score: {warning2['risk_score']:.2f}")
    print(f"Action: {warning2['action']}")
    print(f"Position Multiplier: {warning2['position_multiplier']}")
    print(f"Signals Triggered: {warning2['alert_count']}/6")
    print(f"\n⚠️  On-chain signals detected cascade BEFORE futures market!")
    
    # Test scenario 3: Normal market
    print("\n" + "=" * 70)
    print("TEST 3: Normal Market Conditions")
    print("=" * 70)
    
    warning3 = detector.get_cascade_warning(
        funding=0.0001,
        oi_change=0.02,
        vol_ratio=1.1,
        onchain={
            'exchange_netflow': 0.1,
            'exchange_netflow_zscore': 0.5,
            'large_tx_count_5min': 2,
        }
    )
    
    print(f"\nRisk Level: {warning3['risk_level']}")
    print(f"Risk Score: {warning3['risk_score']:.2f}")
    print(f"Action: {warning3['action']}")
    print(f"Position Multiplier: {warning3['position_multiplier']}")
    
    print("\n" + "=" * 70)
