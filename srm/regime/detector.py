"""
Adaptive Regime-Based Weighting System

Uses 1-hour lagged lookback to classify the current market regime,
then applies regime-specific weights to current signal values.
The lag prevents circular feedback.

Regimes:
- NORMAL: Balanced weights across all signals
- LEVERAGE_SATURATION: FSI and LEI dominate (May 19, 2021 pattern)
- ORACLE_FAILURE: ODS and SCSI dominate (October 10, 2025 pattern)
- TRADFI_CONTAGION: CACI dominates (August 5, 2024 pattern)
"""

from enum import Enum
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications based on dominant risk factors."""
    NORMAL = "normal"
    LEVERAGE_SATURATION = "leverage_saturation"
    ORACLE_FAILURE = "oracle_failure"
    TRADFI_CONTAGION = "tradfi_contagion"


@dataclass
class RegimeWeights:
    """Weight configuration for each regime."""
    fsi: float
    lei: float
    ods: float
    scsi: float
    lci: float
    caci: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'fsi': self.fsi,
            'lei': self.lei,
            'ods': self.ods,
            'scsi': self.scsi,
            'lci': self.lci,
            'caci': self.caci,
        }
    
    def validate(self) -> bool:
        """Verify weights sum to approximately 1.0."""
        total = self.fsi + self.lei + self.ods + self.scsi + self.lci + self.caci
        return 0.99 <= total <= 1.01


# Forensically-validated weight configurations per regime
REGIME_WEIGHTS = {
    MarketRegime.NORMAL: RegimeWeights(
        fsi=0.25, lei=0.30, ods=0.15, scsi=0.15, lci=0.10, caci=0.05
    ),
    MarketRegime.LEVERAGE_SATURATION: RegimeWeights(
        fsi=0.35, lei=0.30, ods=0.10, scsi=0.10, lci=0.10, caci=0.05
    ),
    MarketRegime.ORACLE_FAILURE: RegimeWeights(
        fsi=0.10, lei=0.20, ods=0.35, scsi=0.25, lci=0.05, caci=0.05
    ),
    MarketRegime.TRADFI_CONTAGION: RegimeWeights(
        fsi=0.25, lei=0.20, ods=0.10, scsi=0.10, lci=0.05, caci=0.30
    ),
}


class RegimeDetector:
    """
    Classifies current market regime based on recent signal history.
    
    Uses 1-hour lagged lookback to determine regime, avoiding circular
    feedback between regime detection and signal weighting.
    
    Classification Logic:
    - If ODS > 0.3 or SCSI > 0.4 in the last hour → ORACLE_FAILURE
    - If CACI > 0.5 in the last hour → TRADFI_CONTAGION
    - If FSI > 0.6 or LCI > 0.6 in the last hour → LEVERAGE_SATURATION
    - Otherwise → NORMAL
    """
    
    def __init__(self, lookback_minutes: int = 60):
        self.lookback_minutes = lookback_minutes
        # Store (timestamp, signals_dict) tuples
        self.signal_history: deque = deque(maxlen=lookback_minutes * 2)
        self._current_regime = MarketRegime.NORMAL
    
    def update(self, signals: Dict[str, float], timestamp: datetime) -> None:
        """
        Record new signal observation.
        
        Args:
            signals: Dict with keys 'fsi', 'lei', 'ods', 'scsi', 'lci', 'caci'
            timestamp: Observation timestamp
        """
        self.signal_history.append((timestamp, signals.copy()))
    
    def detect_regime(self, current_timestamp: datetime) -> Tuple[MarketRegime, dict]:
        """
        Determine current regime based on 1-hour historical max.
        
        Args:
            current_timestamp: Current time for lookback calculation
        
        Returns:
            Tuple of (regime, metadata_dict)
        """
        if len(self.signal_history) < 10:  # Need minimum history
            return MarketRegime.NORMAL, {
                'status': 'insufficient_history',
                'observations': len(self.signal_history)
            }
        
        # Filter to last hour
        cutoff = current_timestamp - timedelta(minutes=self.lookback_minutes)
        recent_signals = [
            signals for ts, signals in self.signal_history 
            if ts >= cutoff
        ]
        
        if not recent_signals:
            return MarketRegime.NORMAL, {'status': 'no_recent_data'}
        
        # Calculate max values over lookback period
        maxes = {}
        for key in ['fsi', 'lei', 'ods', 'scsi', 'lci', 'caci']:
            values = [s.get(key, 0) for s in recent_signals]
            maxes[key] = max(values) if values else 0
        
        # Regime classification logic
        # Thresholds derived from forensic analysis
        if maxes['ods'] > 0.3 or maxes['scsi'] > 0.4:
            regime = MarketRegime.ORACLE_FAILURE
        elif maxes['caci'] > 0.5:
            regime = MarketRegime.TRADFI_CONTAGION
        elif maxes['fsi'] > 0.6 or maxes['lci'] > 0.6:
            regime = MarketRegime.LEVERAGE_SATURATION
        else:
            regime = MarketRegime.NORMAL
        
        # Log regime transitions
        if regime != self._current_regime:
            logger.info(f"REGIME TRANSITION: {self._current_regime.value} → {regime.value}")
            self._current_regime = regime
        
        metadata = {
            'lookback_minutes': self.lookback_minutes,
            'observations': len(recent_signals),
            'max_values': maxes,
            'regime': regime.value,
            'trigger_signals': self._get_trigger_signals(regime, maxes)
        }
        
        return regime, metadata
    
    def _get_trigger_signals(self, regime: MarketRegime, maxes: Dict[str, float]) -> list:
        """Identify which signals triggered the regime classification."""
        triggers = []
        
        if regime == MarketRegime.ORACLE_FAILURE:
            if maxes.get('ods', 0) > 0.3:
                triggers.append(f"ODS={maxes['ods']:.2f}")
            if maxes.get('scsi', 0) > 0.4:
                triggers.append(f"SCSI={maxes['scsi']:.2f}")
        elif regime == MarketRegime.TRADFI_CONTAGION:
            if maxes.get('caci', 0) > 0.5:
                triggers.append(f"CACI={maxes['caci']:.2f}")
        elif regime == MarketRegime.LEVERAGE_SATURATION:
            if maxes.get('fsi', 0) > 0.6:
                triggers.append(f"FSI={maxes['fsi']:.2f}")
            if maxes.get('lci', 0) > 0.6:
                triggers.append(f"LCI={maxes['lci']:.2f}")
        
        return triggers
    
    def get_weights(self, regime: Optional[MarketRegime] = None) -> RegimeWeights:
        """
        Return weight configuration for given regime.
        
        Args:
            regime: Market regime (uses current regime if None)
        
        Returns:
            RegimeWeights for the specified regime
        """
        if regime is None:
            regime = self._current_regime
        return REGIME_WEIGHTS[regime]
    
    @property
    def current_regime(self) -> MarketRegime:
        """Get current detected regime."""
        return self._current_regime
    
    def get_state(self) -> dict:
        """Get current state for persistence."""
        return {
            'signal_history': [
                (ts.isoformat(), signals) 
                for ts, signals in self.signal_history
            ],
            'current_regime': self._current_regime.value
        }
    
    def restore_state(self, state: dict) -> None:
        """Restore state from persistence."""
        self.signal_history.clear()
        for ts_str, signals in state.get('signal_history', []):
            self.signal_history.append((datetime.fromisoformat(ts_str), signals))
        
        regime_str = state.get('current_regime', 'normal')
        self._current_regime = MarketRegime(regime_str)
