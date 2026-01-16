"""
Drift Detection Ensemble

Concept drift occurs when the relationship between features and returns changes.
HIMARI uses an ensemble of drift detectors because no single detector catches
all drift types. The ensemble votes: alert only when 2+ detectors agree.

Monitors:
- ADWIN: Adaptive windowing for gradual drift
- Page-Hinkley: Sequential change detection
- KSWIN: Kolmogorov-Smirnov windowed detection
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """Alert from drift detection."""
    timestamp: datetime
    confidence: float
    detector_alerts: Dict[str, bool]
    metric_value: float
    metric_name: str
    severity: str  # 'low', 'medium', 'high', 'critical'


class ADWINDetector:
    """
    ADWIN (Adaptive Windowing) drift detector.

    Maintains a variable-size window of recent data. When statistical
    properties of two subwindows differ significantly, drift is detected.

    Good for: Gradual drift, concept drift.
    """

    def __init__(self, delta: float = 0.002):
        """
        Args:
            delta: Significance level (lower = more sensitive)
        """
        self.delta = delta
        self.window: List[float] = []
        self.total = 0.0
        self.variance = 0.0
        self.width = 0
        self._drift_detected = False

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected

    def update(self, value: float) -> bool:
        """
        Update with new observation.

        Returns:
            True if drift detected
        """
        self._drift_detected = False
        self.window.append(value)
        self.total += value
        self.width += 1

        if self.width < 10:
            return False

        # Check for drift by comparing subwindows
        for split in range(5, self.width - 5):
            left = self.window[:split]
            right = self.window[split:]

            left_mean = np.mean(left)
            right_mean = np.mean(right)

            n1, n2 = len(left), len(right)
            m = 1 / (1/n1 + 1/n2)

            # Hoeffding bound
            eps = np.sqrt(2 * m * np.log(2 / self.delta) / self.width)

            if abs(left_mean - right_mean) > eps:
                # Drift detected, shrink window
                self.window = right
                self.width = len(right)
                self.total = sum(right)
                self._drift_detected = True
                return True

        return False

    def reset(self) -> None:
        """Reset detector state."""
        self.window = []
        self.total = 0.0
        self.width = 0
        self._drift_detected = False


class PageHinkleyDetector:
    """
    Page-Hinkley sequential change detector.

    Tracks cumulative sum of differences from running mean.
    Detects abrupt changes in mean.

    Good for: Sudden drift, mean shifts.
    """

    def __init__(
        self,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50,
        alpha: float = 0.9999
    ):
        """
        Args:
            min_instances: Minimum samples before detection
            delta: Allowance for natural variance
            threshold: Cumulative sum threshold for drift
            alpha: Forgetting factor
        """
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha

        self.n = 0
        self.x_mean = 0.0
        self.sum = 0.0
        self.min_sum = float('inf')
        self._drift_detected = False

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected

    def update(self, value: float) -> bool:
        """Update with new observation."""
        self._drift_detected = False

        self.n += 1

        # Update running mean with forgetting
        self.x_mean = self.alpha * self.x_mean + (1 - self.alpha) * value

        # Update cumulative sum
        self.sum += value - self.x_mean - self.delta
        self.min_sum = min(self.min_sum, self.sum)

        # Check for drift
        if self.n >= self.min_instances:
            if self.sum - self.min_sum > self.threshold:
                self._drift_detected = True
                return True

        return False

    def reset(self) -> None:
        """Reset detector state."""
        self.n = 0
        self.x_mean = 0.0
        self.sum = 0.0
        self.min_sum = float('inf')
        self._drift_detected = False


class KSWINDetector:
    """
    KSWIN (Kolmogorov-Smirnov Windowed) drift detector.

    Uses KS test to compare distributions in sliding windows.

    Good for: Distribution shift, variance changes.
    """

    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30
    ):
        """
        Args:
            alpha: Significance level for KS test
            window_size: Size of sliding window
            stat_size: Size of window for KS comparison
        """
        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size

        self.window: List[float] = []
        self._drift_detected = False

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected

    def update(self, value: float) -> bool:
        """Update with new observation."""
        self._drift_detected = False

        self.window.append(value)

        # Keep window bounded
        if len(self.window) > self.window_size:
            self.window.pop(0)

        if len(self.window) < self.window_size:
            return False

        # Compare first and last portions
        old_window = self.window[:self.stat_size]
        new_window = self.window[-self.stat_size:]

        # KS test
        from scipy import stats
        ks_stat, p_value = stats.ks_2samp(old_window, new_window)

        if p_value < self.alpha:
            self._drift_detected = True
            # Reset window on drift
            self.window = self.window[-self.stat_size:]
            return True

        return False

    def reset(self) -> None:
        """Reset detector state."""
        self.window = []
        self._drift_detected = False


class DriftDetectionEnsemble:
    """
    Ensemble of drift detectors for robust regime shift detection.

    Uses voting: Alert only when 2+ detectors agree, reducing false positives.
    """

    def __init__(
        self,
        adwin_delta: float = 0.002,
        ph_delta: float = 0.005,
        ph_threshold: float = 50,
        ks_alpha: float = 0.005,
        ks_window: int = 100
    ):
        """Initialize ensemble with configured detectors."""
        self.detectors = {
            "adwin": ADWINDetector(delta=adwin_delta),
            "page_hinkley": PageHinkleyDetector(
                delta=ph_delta,
                threshold=ph_threshold
            ),
            "kswin": KSWINDetector(
                alpha=ks_alpha,
                window_size=ks_window
            )
        }

        self.votes_for_drift = 0
        self.alert_history: List[DriftAlert] = []
        self.last_metric_value = 0.0
        self.metric_name = "sharpe"

    def update(
        self,
        value: float,
        metric_name: str = "sharpe"
    ) -> Dict:
        """
        Update all detectors with new observation.

        Args:
            value: Performance metric (rolling Sharpe, returns, etc.)
            metric_name: Name of metric being monitored

        Returns:
            Dict with drift status and detector votes
        """
        self.last_metric_value = value
        self.metric_name = metric_name

        alerts = {}
        for name, detector in self.detectors.items():
            detector.update(value)
            alerts[name] = detector.drift_detected

        self.votes_for_drift = sum(alerts.values())
        confirmed_drift = self.votes_for_drift >= 2

        # Determine severity
        if self.votes_for_drift == 3:
            severity = 'critical'
        elif self.votes_for_drift == 2:
            severity = 'high'
        elif self.votes_for_drift == 1:
            severity = 'medium'
        else:
            severity = 'low'

        confidence = self.votes_for_drift / len(self.detectors)

        result = {
            "drift_detected": confirmed_drift,
            "detector_alerts": alerts,
            "confidence": confidence,
            "severity": severity,
            "votes": self.votes_for_drift,
            "total_detectors": len(self.detectors)
        }

        if confirmed_drift:
            alert = DriftAlert(
                timestamp=datetime.now(),
                confidence=confidence,
                detector_alerts=alerts.copy(),
                metric_value=value,
                metric_name=metric_name,
                severity=severity
            )
            self.alert_history.append(alert)
            logger.warning(f"Drift detected! Confidence: {confidence:.2f}, Severity: {severity}")

        return result

    def reset(self) -> None:
        """Reset all detectors after confirmed regime change."""
        for detector in self.detectors.values():
            detector.reset()
        self.votes_for_drift = 0

    def get_alert_history(self, limit: int = 10) -> List[DriftAlert]:
        """Get recent drift alerts."""
        return self.alert_history[-limit:]

    def get_status(self) -> Dict:
        """Get current ensemble status."""
        return {
            "detectors": list(self.detectors.keys()),
            "current_votes": self.votes_for_drift,
            "alert_count": len(self.alert_history),
            "last_metric_value": self.last_metric_value,
            "metric_name": self.metric_name
        }


class MultiMetricDriftEnsemble:
    """
    Monitor drift across multiple metrics simultaneously.

    Different metrics may indicate different types of regime change.
    """

    def __init__(self, metrics: List[str] = None):
        """
        Args:
            metrics: List of metric names to monitor
        """
        default_metrics = ['sharpe', 'volatility', 'correlation', 'spread']
        self.metrics = metrics or default_metrics

        self.ensembles = {
            metric: DriftDetectionEnsemble()
            for metric in self.metrics
        }

    def update(self, metric_values: Dict[str, float]) -> Dict:
        """
        Update all metric ensembles.

        Args:
            metric_values: Dict mapping metric name to value

        Returns:
            Combined drift status
        """
        results = {}
        any_drift = False

        for metric, value in metric_values.items():
            if metric in self.ensembles:
                result = self.ensembles[metric].update(value, metric)
                results[metric] = result
                if result['drift_detected']:
                    any_drift = True

        # Overall severity
        max_votes = max(r.get('votes', 0) for r in results.values())
        overall_severity = 'critical' if max_votes >= 3 else (
            'high' if any_drift else 'normal'
        )

        return {
            'any_drift': any_drift,
            'overall_severity': overall_severity,
            'metric_results': results
        }

    def reset_all(self) -> None:
        """Reset all ensembles."""
        for ensemble in self.ensembles.values():
            ensemble.reset()
