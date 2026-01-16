"""
Prometheus Metrics for HIMARI Layer 1 Explorer

Exposes metrics for monitoring:
- Strategy generation rates and quality
- Validation pipeline performance
- Drift detection and alerts
- Deployment status
"""

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
import threading

logger = logging.getLogger(__name__)


# Try to import prometheus_client, fall back to mock if not available
try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        CollectorRegistry, start_http_server, REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed, using mock metrics")


@dataclass
class MetricValue:
    """Simple metric value for mock implementation."""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class MockCounter:
    """Mock Prometheus Counter."""

    def __init__(self, name: str, description: str, labelnames: List[str] = None):
        self.name = name
        self._value = 0
        self._labels = {}
        self._labelnames = labelnames or []

    def labels(self, **kwargs):
        key = tuple(sorted(kwargs.items()))
        if key not in self._labels:
            self._labels[key] = MockCounter(self.name, "", [])
        return self._labels[key]

    def inc(self, amount: float = 1):
        self._value += amount

    def _get_value(self):
        return self._value


class MockGauge:
    """Mock Prometheus Gauge."""

    def __init__(self, name: str, description: str, labelnames: List[str] = None):
        self.name = name
        self._value = 0
        self._labels = {}
        self._labelnames = labelnames or []

    def labels(self, **kwargs):
        key = tuple(sorted(kwargs.items()))
        if key not in self._labels:
            self._labels[key] = MockGauge(self.name, "", [])
        return self._labels[key]

    def set(self, value: float):
        self._value = value

    def inc(self, amount: float = 1):
        self._value += amount

    def dec(self, amount: float = 1):
        self._value -= amount

    def _get_value(self):
        return self._value


class MockHistogram:
    """Mock Prometheus Histogram."""

    def __init__(self, name: str, description: str, labelnames: List[str] = None, buckets=None):
        self.name = name
        self._values = []
        self._labels = {}
        self._labelnames = labelnames or []

    def labels(self, **kwargs):
        key = tuple(sorted(kwargs.items()))
        if key not in self._labels:
            self._labels[key] = MockHistogram(self.name, "", [])
        return self._labels[key]

    def observe(self, value: float):
        self._values.append(value)

    def time(self):
        return MockTimer(self)


class MockTimer:
    """Mock timer context manager."""

    def __init__(self, histogram):
        self._histogram = histogram
        self._start = None

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self._histogram.observe(time.time() - self._start)


class MockSummary:
    """Mock Prometheus Summary."""

    def __init__(self, name: str, description: str, labelnames: List[str] = None):
        self.name = name
        self._values = []
        self._labels = {}
        self._labelnames = labelnames or []

    def labels(self, **kwargs):
        key = tuple(sorted(kwargs.items()))
        if key not in self._labels:
            self._labels[key] = MockSummary(self.name, "", [])
        return self._labels[key]

    def observe(self, value: float):
        self._values.append(value)


class ExplorerMetrics:
    """
    Prometheus metrics for Layer 1 Explorer.

    Metrics categories:
    1. Generation metrics (counts, rates, diversity)
    2. Validation metrics (pass rates, latencies)
    3. Drift detection metrics
    4. Deployment metrics
    """

    def __init__(self, registry=None):
        """Initialize metrics."""
        if PROMETHEUS_AVAILABLE:
            self._registry = registry or REGISTRY
            self._init_prometheus_metrics()
        else:
            self._init_mock_metrics()

        self._server_started = False

    def _init_prometheus_metrics(self):
        """Initialize real Prometheus metrics."""
        # Generation metrics
        self.strategies_generated = Counter(
            'layer1_strategies_generated_total',
            'Total strategies generated',
            ['engine']
        )

        self.generation_cycle_duration = Histogram(
            'layer1_generation_cycle_duration_seconds',
            'Duration of generation cycles',
            buckets=[1, 5, 10, 30, 60, 120, 300]
        )

        self.diversity_score = Gauge(
            'layer1_diversity_score',
            'Current population diversity score'
        )

        self.engine_success_rate = Gauge(
            'layer1_engine_success_rate',
            'Success rate by generation engine',
            ['engine']
        )

        # Validation metrics
        self.validation_stage_passed = Counter(
            'layer1_validation_stage_passed_total',
            'Strategies passing each validation stage',
            ['stage']
        )

        self.validation_stage_failed = Counter(
            'layer1_validation_stage_failed_total',
            'Strategies failing each validation stage',
            ['stage']
        )

        self.validation_stage_duration = Histogram(
            'layer1_validation_stage_duration_seconds',
            'Duration of each validation stage',
            ['stage'],
            buckets=[0.001, 0.01, 0.1, 1, 10, 60, 300]
        )

        self.hifa_pass_rate = Gauge(
            'layer1_hifa_pass_rate',
            'Overall HIFA pipeline pass rate'
        )

        self.surrogate_accuracy = Gauge(
            'layer1_surrogate_accuracy',
            'Surrogate model prediction accuracy'
        )

        # Strategy quality metrics
        self.strategy_sharpe = Histogram(
            'layer1_strategy_sharpe_ratio',
            'Distribution of strategy Sharpe ratios',
            buckets=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        )

        self.strategy_max_drawdown = Histogram(
            'layer1_strategy_max_drawdown',
            'Distribution of strategy max drawdowns',
            buckets=[0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
        )

        self.strategy_transfer_ratio = Histogram(
            'layer1_strategy_transfer_ratio',
            'Distribution of transfer ratios',
            buckets=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )

        # Drift detection metrics
        self.drift_alerts = Counter(
            'layer1_drift_alerts_total',
            'Drift alerts triggered',
            ['detector', 'severity']
        )

        self.drift_detection_rate = Gauge(
            'layer1_drift_detection_rate',
            'Rate of drift detections per hour'
        )

        self.active_strategies = Gauge(
            'layer1_active_strategies',
            'Number of active deployed strategies'
        )

        # Deployment metrics
        self.deployment_decisions = Counter(
            'layer1_deployment_decisions_total',
            'Deployment decisions made',
            ['decision']  # approved, rejected
        )

        self.shadow_trading_days = Histogram(
            'layer1_shadow_trading_days',
            'Shadow trading duration before deployment',
            buckets=[7, 14, 21, 30, 45, 60]
        )

        self.position_size_pct = Gauge(
            'layer1_deployed_position_size_pct',
            'Total deployed position size percentage'
        )

        # Response metrics
        self.response_level = Gauge(
            'layer1_response_level',
            'Current adaptive response level (0=green, 3=red)'
        )

        self.strategies_retired = Counter(
            'layer1_strategies_retired_total',
            'Strategies retired',
            ['reason']
        )

        # Infrastructure metrics
        self.redis_operations = Counter(
            'layer1_redis_operations_total',
            'Redis operations',
            ['operation', 'status']
        )

        self.kafka_messages = Counter(
            'layer1_kafka_messages_total',
            'Kafka messages',
            ['topic', 'direction']  # direction: in/out
        )

        self.llm_calls = Counter(
            'layer1_llm_calls_total',
            'LLM API calls',
            ['provider', 'status']
        )

        self.llm_latency = Histogram(
            'layer1_llm_latency_seconds',
            'LLM API call latency',
            ['provider'],
            buckets=[0.1, 0.5, 1, 2, 5, 10, 30]
        )

    def _init_mock_metrics(self):
        """Initialize mock metrics for testing."""
        self.strategies_generated = MockCounter('strategies_generated', '', ['engine'])
        self.generation_cycle_duration = MockHistogram('generation_cycle_duration', '')
        self.diversity_score = MockGauge('diversity_score', '')
        self.engine_success_rate = MockGauge('engine_success_rate', '', ['engine'])

        self.validation_stage_passed = MockCounter('validation_stage_passed', '', ['stage'])
        self.validation_stage_failed = MockCounter('validation_stage_failed', '', ['stage'])
        self.validation_stage_duration = MockHistogram('validation_stage_duration', '', ['stage'])
        self.hifa_pass_rate = MockGauge('hifa_pass_rate', '')
        self.surrogate_accuracy = MockGauge('surrogate_accuracy', '')

        self.strategy_sharpe = MockHistogram('strategy_sharpe', '')
        self.strategy_max_drawdown = MockHistogram('strategy_max_drawdown', '')
        self.strategy_transfer_ratio = MockHistogram('strategy_transfer_ratio', '')

        self.drift_alerts = MockCounter('drift_alerts', '', ['detector', 'severity'])
        self.drift_detection_rate = MockGauge('drift_detection_rate', '')
        self.active_strategies = MockGauge('active_strategies', '')

        self.deployment_decisions = MockCounter('deployment_decisions', '', ['decision'])
        self.shadow_trading_days = MockHistogram('shadow_trading_days', '')
        self.position_size_pct = MockGauge('position_size_pct', '')

        self.response_level = MockGauge('response_level', '')
        self.strategies_retired = MockCounter('strategies_retired', '', ['reason'])

        self.redis_operations = MockCounter('redis_operations', '', ['operation', 'status'])
        self.kafka_messages = MockCounter('kafka_messages', '', ['topic', 'direction'])
        self.llm_calls = MockCounter('llm_calls', '', ['provider', 'status'])
        self.llm_latency = MockHistogram('llm_latency', '', ['provider'])

    def start_server(self, port: int = 8000) -> bool:
        """Start Prometheus metrics HTTP server."""
        if self._server_started:
            logger.warning("Metrics server already started")
            return True

        try:
            if PROMETHEUS_AVAILABLE:
                start_http_server(port)
                logger.info(f"Prometheus metrics server started on port {port}")
            else:
                logger.info(f"Mock metrics server (no actual server) on port {port}")

            self._server_started = True
            return True

        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            return False

    # Generation tracking methods

    def record_generation_cycle(
        self,
        by_engine: Dict[str, int],
        diversity: float,
        duration_seconds: float
    ):
        """Record results from a generation cycle."""
        for engine, count in by_engine.items():
            self.strategies_generated.labels(engine=engine).inc(count)

        self.diversity_score.set(diversity)
        self.generation_cycle_duration.observe(duration_seconds)

    def record_engine_success(self, engine: str, rate: float):
        """Record engine success rate."""
        self.engine_success_rate.labels(engine=engine).set(rate)

    # Validation tracking methods

    def record_validation_result(
        self,
        stage: str,
        passed: bool,
        duration_seconds: float
    ):
        """Record validation stage result."""
        if passed:
            self.validation_stage_passed.labels(stage=stage).inc()
        else:
            self.validation_stage_failed.labels(stage=stage).inc()

        self.validation_stage_duration.labels(stage=stage).observe(duration_seconds)

    def record_hifa_pass_rate(self, rate: float):
        """Record overall HIFA pass rate."""
        self.hifa_pass_rate.set(rate)

    def record_surrogate_accuracy(self, accuracy: float):
        """Record surrogate model accuracy."""
        self.surrogate_accuracy.set(accuracy)

    # Strategy quality methods

    def record_strategy_metrics(
        self,
        sharpe: float,
        max_drawdown: float,
        transfer_ratio: Optional[float] = None
    ):
        """Record strategy quality metrics."""
        self.strategy_sharpe.observe(sharpe)
        self.strategy_max_drawdown.observe(max_drawdown)

        if transfer_ratio is not None:
            self.strategy_transfer_ratio.observe(transfer_ratio)

    # Drift tracking methods

    def record_drift_alert(self, detector: str, severity: str):
        """Record a drift detection alert."""
        self.drift_alerts.labels(detector=detector, severity=severity).inc()

    def set_drift_detection_rate(self, rate: float):
        """Set current drift detection rate."""
        self.drift_detection_rate.set(rate)

    def set_active_strategies(self, count: int):
        """Set number of active strategies."""
        self.active_strategies.set(count)

    # Deployment tracking methods

    def record_deployment_decision(self, approved: bool):
        """Record a deployment decision."""
        decision = 'approved' if approved else 'rejected'
        self.deployment_decisions.labels(decision=decision).inc()

    def record_shadow_duration(self, days: int):
        """Record shadow trading duration."""
        self.shadow_trading_days.observe(days)

    def set_position_size(self, pct: float):
        """Set total deployed position size."""
        self.position_size_pct.set(pct)

    # Response tracking methods

    def set_response_level(self, level: int):
        """Set current adaptive response level (0-3)."""
        self.response_level.set(level)

    def record_retirement(self, reason: str):
        """Record strategy retirement."""
        self.strategies_retired.labels(reason=reason).inc()

    # Infrastructure tracking methods

    def record_redis_operation(self, operation: str, success: bool):
        """Record Redis operation."""
        status = 'success' if success else 'failure'
        self.redis_operations.labels(operation=operation, status=status).inc()

    def record_kafka_message(self, topic: str, direction: str):
        """Record Kafka message (direction: 'in' or 'out')."""
        self.kafka_messages.labels(topic=topic, direction=direction).inc()

    def record_llm_call(
        self,
        provider: str,
        success: bool,
        latency_seconds: float
    ):
        """Record LLM API call."""
        status = 'success' if success else 'failure'
        self.llm_calls.labels(provider=provider, status=status).inc()
        self.llm_latency.labels(provider=provider).observe(latency_seconds)

    # Utility methods

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of current metric values."""
        if not PROMETHEUS_AVAILABLE:
            return {
                'prometheus_available': False,
                'note': 'Using mock metrics'
            }

        return {
            'prometheus_available': True,
            'server_started': self._server_started,
            'diversity_score': self.diversity_score._value._value if hasattr(self.diversity_score, '_value') else 0,
            'hifa_pass_rate': self.hifa_pass_rate._value._value if hasattr(self.hifa_pass_rate, '_value') else 0,
            'active_strategies': self.active_strategies._value._value if hasattr(self.active_strategies, '_value') else 0,
            'response_level': self.response_level._value._value if hasattr(self.response_level, '_value') else 0
        }


def timed_metric(histogram):
    """Decorator to time function execution and record to histogram."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                histogram.observe(time.time() - start)
        return wrapper
    return decorator


def async_timed_metric(histogram):
    """Async decorator to time function execution."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                histogram.observe(time.time() - start)
        return wrapper
    return decorator


# Global metrics instance
_metrics_instance: Optional[ExplorerMetrics] = None
_metrics_lock = threading.Lock()


def get_metrics() -> ExplorerMetrics:
    """Get global metrics instance (singleton)."""
    global _metrics_instance

    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = ExplorerMetrics()

    return _metrics_instance


def init_metrics(port: int = 8000) -> ExplorerMetrics:
    """Initialize and start metrics server."""
    metrics = get_metrics()
    metrics.start_server(port)
    return metrics
