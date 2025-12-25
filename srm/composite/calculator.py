"""
Composite Risk Calculator

Synthesizes all six signals with regime-aware weights and optional
multi-signal amplification.
"""

from dataclasses import dataclass
from typing import Dict
from datetime import datetime
import logging

from ..regime import RegimeDetector, MarketRegime, RegimeWeights

logger = logging.getLogger(__name__)


@dataclass
class CompositeRiskResult:
    """Result of composite risk calculation."""
    score: float  # 0.0-1.0, higher = more risk
    regime: MarketRegime
    weights_used: RegimeWeights
    signal_values: Dict[str, float]
    amplification_applied: bool
    metadata: dict
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'score': self.score,
            'regime': self.regime.value,
            'weights': self.weights_used.to_dict(),
            'signals': self.signal_values,
            'amplification': self.amplification_applied,
            'metadata': self.metadata
        }


class CompositeRiskCalculator:
    """
    Calculates final systemic risk score from individual signals.
    
    The composite score combines six signals with adaptive weights based
    on detected market regime. An amplification factor triggers when
    multiple signals simultaneously exceed threshold, indicating
    compound stress conditions.
    
    Response Tiers:
    - Score < 0.5: Normal operation
    - Score 0.5-0.7: Reduced exposure
    - Score 0.7-0.9: Close-only mode
    - Score > 0.9: Emergency halt
    """
    
    def __init__(self, regime_lookback_minutes: int = 60):
        self.regime_detector = RegimeDetector(lookback_minutes=regime_lookback_minutes)
        self._last_score = 0.0
    
    def calculate(
        self, 
        signals: Dict[str, float], 
        timestamp: datetime
    ) -> CompositeRiskResult:
        """
        Calculate composite risk score with regime-aware weighting.
        
        Args:
            signals: Dict with keys 'fsi', 'lei', 'ods', 'scsi', 'lci', 'caci'
                     Each value in range 0.0-1.0
            timestamp: Current observation timestamp
        
        Returns:
            CompositeRiskResult with score and metadata
        """
        # Validate signals
        for key in ['fsi', 'lei', 'ods', 'scsi', 'lci', 'caci']:
            if key not in signals:
                signals[key] = 0.0
            # Clamp to valid range
            signals[key] = max(0.0, min(1.0, signals[key]))
        
        # Update regime detector with current signals
        self.regime_detector.update(signals, timestamp)
        
        # Detect regime from LAGGED history (not current values)
        regime, regime_meta = self.regime_detector.detect_regime(timestamp)
        
        # Get weights for detected regime
        weights = self.regime_detector.get_weights(regime)
        
        # Calculate weighted composite
        composite = (
            weights.fsi * signals['fsi'] +
            weights.lei * signals['lei'] +
            weights.ods * signals['ods'] +
            weights.scsi * signals['scsi'] +
            weights.lci * signals['lci'] +
            weights.caci * signals['caci']
        )
        
        # Multi-signal amplification
        # When 4+ signals exceed 0.5, compound stress is likely
        elevated_count = sum(1 for v in signals.values() if v > 0.5)
        high_count = sum(1 for v in signals.values() if v > 0.7)
        amplification_applied = elevated_count >= 4 or high_count >= 3
        
        if amplification_applied:
            amplification_factor = 1.3 if elevated_count >= 4 else 1.2
            composite = composite * amplification_factor
        
        # Ensure score is in valid range
        composite = max(0.0, min(1.0, composite))
        
        # Calculate individual contributions for debugging
        contributions = {
            'fsi': weights.fsi * signals['fsi'],
            'lei': weights.lei * signals['lei'],
            'ods': weights.ods * signals['ods'],
            'scsi': weights.scsi * signals['scsi'],
            'lci': weights.lci * signals['lci'],
            'caci': weights.caci * signals['caci'],
        }
        
        # Determine risk level
        if composite >= 0.9:
            risk_level = 'CRITICAL'
        elif composite >= 0.7:
            risk_level = 'HIGH'
        elif composite >= 0.5:
            risk_level = 'ELEVATED'
        else:
            risk_level = 'LOW'
        
        # Log significant changes
        if abs(composite - self._last_score) > 0.1:
            logger.info(
                f"RISK SCORE CHANGE: {self._last_score:.3f} → {composite:.3f} "
                f"(regime={regime.value}, level={risk_level})"
            )
        
        self._last_score = composite
        
        metadata = {
            'regime_detection': regime_meta,
            'elevated_signals': elevated_count,
            'high_signals': high_count,
            'contributions': contributions,
            'dominant_signal': max(contributions, key=contributions.get),
            'risk_level': risk_level,
            'timestamp': timestamp.isoformat()
        }
        
        return CompositeRiskResult(
            score=composite,
            regime=regime,
            weights_used=weights,
            signal_values=signals.copy(),
            amplification_applied=amplification_applied,
            metadata=metadata
        )
    
    @property
    def current_regime(self) -> MarketRegime:
        """Get current detected regime."""
        return self.regime_detector.current_regime
    
    @property
    def last_score(self) -> float:
        """Get last calculated score."""
        return self._last_score
    
    def get_state(self) -> dict:
        """Get current state for persistence."""
        return {
            'regime_detector': self.regime_detector.get_state(),
            'last_score': self._last_score
        }
    
    def restore_state(self, state: dict) -> None:
        """Restore state from persistence."""
        if 'regime_detector' in state:
            self.regime_detector.restore_state(state['regime_detector'])
        self._last_score = state.get('last_score', 0.0)
