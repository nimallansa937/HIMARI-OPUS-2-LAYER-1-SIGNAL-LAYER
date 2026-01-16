"""
Adaptive Response Manager

Graduated response to detected drift:
- Green (no drift): Normal operation
- Yellow (emerging): Increase buffers, reduce positions 20%
- Orange (confirmed): Reweight agents, reduce 50%, alert Layer 4-5
- Red (critical): Halt trading, trigger full review
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ResponseLevel(Enum):
    """Response levels for drift handling."""
    GREEN = "green"      # Normal operation
    YELLOW = "yellow"    # Emerging concern
    ORANGE = "orange"    # Confirmed drift
    RED = "red"          # Critical - halt


@dataclass
class ResponseAction:
    """An action taken in response to drift."""
    timestamp: datetime
    level: ResponseLevel
    action_type: str  # 'position_reduction', 'buffer_increase', 'alert', 'halt'
    details: str
    parameters: Dict[str, Any]


class MockPortfolioManager:
    """Mock portfolio manager for testing."""

    def __init__(self):
        self.volatility_buffer = 1.0
        self.position_scale = 1.0
        self.trading_halted = False

    def increase_volatility_buffer(self, factor: float) -> None:
        """Increase volatility buffer."""
        self.volatility_buffer *= factor
        logger.info(f"Volatility buffer increased to {self.volatility_buffer:.2f}")

    def reduce_positions(self, scale: float) -> None:
        """Scale down all positions."""
        self.position_scale *= scale
        logger.info(f"Positions scaled to {self.position_scale:.2f}")

    def halt_trading(self) -> None:
        """Halt all trading."""
        self.trading_halted = True
        logger.warning("Trading HALTED")

    def resume_trading(self) -> None:
        """Resume trading."""
        self.trading_halted = False
        logger.info("Trading resumed")

    def reset_buffers(self) -> None:
        """Reset to normal operation."""
        self.volatility_buffer = 1.0
        self.position_scale = 1.0
        self.trading_halted = False


class MockAlerter:
    """Mock alerter for testing."""

    def __init__(self):
        self.alerts: List[Dict] = []

    def notify_layer4_5(self, message: str) -> None:
        """Send alert to Layer 4-5."""
        alert = {
            'timestamp': datetime.now(),
            'destination': 'layer4_5',
            'message': message
        }
        self.alerts.append(alert)
        logger.info(f"Alert to Layer 4-5: {message}")

    def notify_human(self, message: str) -> None:
        """Send alert requiring human attention."""
        alert = {
            'timestamp': datetime.now(),
            'destination': 'human',
            'message': message,
            'priority': 'high'
        }
        self.alerts.append(alert)
        logger.warning(f"Human alert: {message}")


class AdaptiveResponseManager:
    """
    Graduated response to detected drift.

    Response hierarchy:
    - Green (confidence < 0.33): Normal operation
    - Yellow (0.33-0.66): Increase buffers, reduce positions 20%
    - Orange (0.66-0.90): Reweight agents, reduce 50%, alert Layer 4-5
    - Red (> 0.90): Halt trading, trigger full review
    """

    THRESHOLDS = {
        ResponseLevel.GREEN: 0.33,
        ResponseLevel.YELLOW: 0.66,
        ResponseLevel.ORANGE: 0.90,
        ResponseLevel.RED: 1.0
    }

    def __init__(
        self,
        portfolio_manager=None,
        alerter=None
    ):
        self.portfolio = portfolio_manager or MockPortfolioManager()
        self.alerter = alerter or MockAlerter()
        self.current_level = ResponseLevel.GREEN
        self.action_history: List[ResponseAction] = []
        self.level_entry_time: Dict[ResponseLevel, datetime] = {}

    def respond(self, drift_result: Dict) -> Dict:
        """
        Execute response based on drift detection result.

        Args:
            drift_result: Dict from DriftDetectionEnsemble.update()

        Returns:
            Dict with level, confidence, and actions taken
        """
        confidence = drift_result.get("confidence", 0)
        severity = drift_result.get("severity", "low")

        # Determine response level
        level = self._determine_level(confidence, severity)

        # Track level changes
        if level != self.current_level:
            logger.info(f"Response level change: {self.current_level.value} -> {level.value}")
            self.level_entry_time[level] = datetime.now()

        # Execute appropriate response
        actions = self._execute_response(level, confidence, drift_result)

        self.current_level = level

        return {
            "level": level.value,
            "confidence": confidence,
            "actions_taken": actions,
            "previous_level": self.current_level.value
        }

    def _determine_level(self, confidence: float, severity: str) -> ResponseLevel:
        """Determine response level from confidence and severity."""
        # Severity can override confidence
        if severity == 'critical':
            return ResponseLevel.RED

        if confidence >= self.THRESHOLDS[ResponseLevel.ORANGE]:
            return ResponseLevel.RED if severity == 'high' else ResponseLevel.ORANGE
        elif confidence >= self.THRESHOLDS[ResponseLevel.YELLOW]:
            return ResponseLevel.ORANGE if severity == 'high' else ResponseLevel.YELLOW
        elif confidence >= self.THRESHOLDS[ResponseLevel.GREEN]:
            return ResponseLevel.YELLOW
        else:
            return ResponseLevel.GREEN

    def _execute_response(
        self,
        level: ResponseLevel,
        confidence: float,
        drift_result: Dict
    ) -> List[str]:
        """Execute response actions for given level."""
        actions = []

        if level == ResponseLevel.GREEN:
            # Normal operation - potentially recover from previous level
            if self.current_level in [ResponseLevel.YELLOW, ResponseLevel.ORANGE]:
                self._gradual_recovery()
                actions.append("Initiating gradual recovery")
            actions.append("Normal operation")

        elif level == ResponseLevel.YELLOW:
            # Emerging concern
            self.portfolio.increase_volatility_buffer(1.2)
            self.portfolio.reduce_positions(0.8)

            self._record_action(level, "buffer_increase", "Volatility buffer +20%", {})
            self._record_action(level, "position_reduction", "Positions to 80%", {'scale': 0.8})

            actions.extend([
                "Volatility buffer increased +20%",
                "Positions reduced to 80%"
            ])

        elif level == ResponseLevel.ORANGE:
            # Confirmed drift
            self.portfolio.reduce_positions(0.5)
            self.alerter.notify_layer4_5(
                f"Drift confirmed (confidence={confidence:.2f}). "
                f"Detectors: {drift_result.get('detector_alerts', {})}"
            )

            self._record_action(level, "position_reduction", "Positions to 50%", {'scale': 0.5})
            self._record_action(level, "alert", "Layer 4-5 notified", {})

            actions.extend([
                "Positions reduced to 50%",
                "Layer 4-5 alerted for regime analysis"
            ])

        elif level == ResponseLevel.RED:
            # Critical - halt trading
            self.portfolio.halt_trading()
            self.alerter.notify_human(
                f"CRITICAL DRIFT - Trading halted. "
                f"Confidence: {confidence:.2f}, "
                f"Manual review required."
            )

            self._record_action(level, "halt", "Trading halted", {})
            self._record_action(level, "alert", "Human review requested", {'priority': 'critical'})

            actions.extend([
                "TRADING HALTED",
                "Human review required"
            ])

        return actions

    def _gradual_recovery(self) -> None:
        """Gradually recover from elevated response level."""
        # Don't immediately restore - do it incrementally
        if self.portfolio.position_scale < 1.0:
            new_scale = min(1.0, self.portfolio.position_scale * 1.1)
            self.portfolio.position_scale = new_scale
            logger.info(f"Gradual recovery: positions at {new_scale:.2f}")

        if self.portfolio.volatility_buffer > 1.0:
            new_buffer = max(1.0, self.portfolio.volatility_buffer * 0.95)
            self.portfolio.volatility_buffer = new_buffer
            logger.info(f"Gradual recovery: buffer at {new_buffer:.2f}")

    def _record_action(
        self,
        level: ResponseLevel,
        action_type: str,
        details: str,
        parameters: Dict
    ) -> None:
        """Record an action in history."""
        action = ResponseAction(
            timestamp=datetime.now(),
            level=level,
            action_type=action_type,
            details=details,
            parameters=parameters
        )
        self.action_history.append(action)

    def force_level(self, level: ResponseLevel) -> Dict:
        """
        Force a specific response level (for testing or manual override).

        Args:
            level: Target response level

        Returns:
            Actions taken
        """
        actions = self._execute_response(level, 1.0, {})
        self.current_level = level
        return {
            "level": level.value,
            "actions_taken": actions,
            "forced": True
        }

    def reset(self) -> None:
        """Reset to normal operation."""
        self.portfolio.reset_buffers()
        self.current_level = ResponseLevel.GREEN
        logger.info("Response manager reset to GREEN")

    def get_status(self) -> Dict:
        """Get current status."""
        return {
            "current_level": self.current_level.value,
            "position_scale": self.portfolio.position_scale,
            "volatility_buffer": self.portfolio.volatility_buffer,
            "trading_halted": self.portfolio.trading_halted,
            "recent_actions": [
                {
                    "timestamp": a.timestamp.isoformat(),
                    "level": a.level.value,
                    "type": a.action_type,
                    "details": a.details
                }
                for a in self.action_history[-10:]
            ]
        }


class RegimeAwareResponseManager(AdaptiveResponseManager):
    """
    Response manager that considers market regime in decisions.

    Different regimes may warrant different response thresholds.
    """

    REGIME_ADJUSTMENTS = {
        'bull': {'threshold_mult': 1.2, 'recovery_rate': 1.2},  # More tolerant
        'bear': {'threshold_mult': 0.8, 'recovery_rate': 0.8},  # More cautious
        'range': {'threshold_mult': 1.0, 'recovery_rate': 1.0},
        'volatile': {'threshold_mult': 0.7, 'recovery_rate': 0.5}  # Very cautious
    }

    def __init__(
        self,
        portfolio_manager=None,
        alerter=None,
        initial_regime: str = 'range'
    ):
        super().__init__(portfolio_manager, alerter)
        self.current_regime = initial_regime

    def set_regime(self, regime: str) -> None:
        """Update current market regime."""
        if regime in self.REGIME_ADJUSTMENTS:
            self.current_regime = regime
            logger.info(f"Regime updated to: {regime}")

    def _determine_level(self, confidence: float, severity: str) -> ResponseLevel:
        """Determine level with regime adjustment."""
        adjustment = self.REGIME_ADJUSTMENTS.get(
            self.current_regime,
            {'threshold_mult': 1.0}
        )

        # Adjust confidence based on regime
        adjusted_confidence = confidence / adjustment['threshold_mult']

        return super()._determine_level(adjusted_confidence, severity)
