"""
Prometheus Metrics Collector - Production Monitoring Infrastructure

Exposes system metrics in Prometheus format for scraping.

Enhancement 6 from ANTIGRAVITY_SENTIMENT_ENHANCEMENT_GUIDE.md
"""

import time
import logging
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from collections import deque
import threading

logger = logging.getLogger(__name__)

# Try to import prometheus_client, provide fallback if not available
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed - metrics will be in-memory only")


@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""
    port: int = 8000
    update_interval: int = 100  # Update gauges every N updates
    enable_prometheus: bool = True
    histogram_buckets: tuple = (1, 5, 10, 25, 50, 100, 250, 500, 1000)


class PrometheusMetricsCollector:
    """
    Collect and expose metrics for production monitoring.
    
    Metrics tracked:
    1. Latency histograms (p50/p95/p99)
    2. Throughput counters
    3. Accuracy gauges
    4. Error counters
    5. Data quality gauges
    6. Model drift indicators
    
    Example:
        collector = PrometheusMetricsCollector()
        
        # Record latency
        collector.record_latency('sentiment_analysis', 25.3)
        
        # Increment counter
        collector.increment_counter('signals_generated', labels={'symbol': 'BTCUSDT'})
        
        # Set gauge
        collector.set_gauge('sentiment_accuracy', 0.87, labels={'source': 'news'})
    """
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        """
        Initialize metrics collector.
        
        Args:
            config: Metrics configuration
        """
        self.config = config or MetricsConfig()
        self._started = False
        self._update_count = 0
        
        # In-memory buffers for non-Prometheus mode
        self._latency_buffers: Dict[str, deque] = {}
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        
        # Prometheus metrics (if available)
        self._prom_histograms: Dict[str, Any] = {}
        self._prom_counters: Dict[str, Any] = {}
        self._prom_gauges: Dict[str, Any] = {}
        
        if PROMETHEUS_AVAILABLE and self.config.enable_prometheus:
            self._init_prometheus_metrics()
        
        logger.info(
            f"PrometheusMetricsCollector initialized "
            f"(prometheus: {PROMETHEUS_AVAILABLE and self.config.enable_prometheus})"
        )
    
    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus metric objects."""
        # Latency histograms
        self._prom_histograms['sentiment_analysis_latency_ms'] = Histogram(
            'sentiment_analysis_latency_ms',
            'Sentiment analysis latency in milliseconds',
            ['source'],
            buckets=self.config.histogram_buckets
        )
        
        self._prom_histograms['total_signal_latency_ms'] = Histogram(
            'total_signal_latency_ms',
            'End-to-end signal generation latency',
            ['symbol'],
            buckets=self.config.histogram_buckets
        )
        
        self._prom_histograms['hmm_update_latency_ms'] = Histogram(
            'hmm_update_latency_ms',
            'HMM regime update latency',
            buckets=self.config.histogram_buckets
        )
        
        # Throughput counters
        self._prom_counters['signals_generated_total'] = Counter(
            'signals_generated_total',
            'Total signals generated',
            ['symbol']
        )
        
        self._prom_counters['sentiment_analyses_total'] = Counter(
            'sentiment_analyses_total',
            'Total sentiment analyses performed',
            ['source']
        )
        
        self._prom_counters['cache_hits_total'] = Counter(
            'cache_hits_total',
            'Total cache hits'
        )
        
        self._prom_counters['cache_misses_total'] = Counter(
            'cache_misses_total',
            'Total cache misses'
        )
        
        # Accuracy gauges
        self._prom_gauges['sentiment_accuracy_percent'] = Gauge(
            'sentiment_accuracy_percent',
            'Sentiment prediction accuracy',
            ['source', 'window']
        )
        
        self._prom_gauges['signal_sharpe_ratio'] = Gauge(
            'signal_sharpe_ratio',
            'Rolling Sharpe ratio',
            ['symbol', 'window']
        )
        
        # Error counters
        self._prom_counters['sentiment_errors_total'] = Counter(
            'sentiment_errors_total',
            'Total sentiment errors',
            ['error_type']
        )
        
        self._prom_counters['redis_connection_errors_total'] = Counter(
            'redis_connection_errors_total',
            'Total Redis connection errors'
        )
        
        self._prom_counters['srm_gate_activations_total'] = Counter(
            'srm_gate_activations_total',
            'SRM gate activations',
            ['action']
        )
        
        # Data quality gauges
        self._prom_gauges['sentiment_confidence_mean'] = Gauge(
            'sentiment_confidence_mean',
            'Mean sentiment confidence',
            ['source']
        )
        
        self._prom_gauges['data_freshness_seconds'] = Gauge(
            'data_freshness_seconds',
            'Seconds since last data update',
            ['source']
        )
        
        self._prom_gauges['social_posts_filtered_total'] = Gauge(
            'social_posts_filtered_total',
            'Social posts filtered by spam filter',
            ['source', 'reason']
        )
        
        # Model drift gauges
        self._prom_gauges['sentiment_distribution_skew'] = Gauge(
            'sentiment_distribution_skew',
            'Sentiment score distribution skewness',
            ['source']
        )
        
        self._prom_gauges['regime_transition_rate'] = Gauge(
            'regime_transition_rate',
            'Regime transitions per hour'
        )
    
    def start_server(self) -> None:
        """Start Prometheus HTTP server."""
        if not PROMETHEUS_AVAILABLE or not self.config.enable_prometheus:
            logger.warning("Prometheus not available - server not started")
            return
        
        if self._started:
            return
        
        try:
            start_http_server(self.config.port)
            self._started = True
            logger.info(f"Prometheus metrics server started on port {self.config.port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
    
    def record_latency(
        self, 
        metric_name: str, 
        latency_ms: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a latency measurement.
        
        Args:
            metric_name: Metric name (e.g., 'sentiment_analysis')
            latency_ms: Latency in milliseconds
            labels: Optional labels
        """
        # In-memory buffer
        key = f"{metric_name}:{labels}" if labels else metric_name
        if key not in self._latency_buffers:
            self._latency_buffers[key] = deque(maxlen=1000)
        self._latency_buffers[key].append(latency_ms)
        
        # Prometheus
        if PROMETHEUS_AVAILABLE and self.config.enable_prometheus:
            hist_key = f"{metric_name}_latency_ms"
            if hist_key in self._prom_histograms:
                hist = self._prom_histograms[hist_key]
                if labels:
                    hist.labels(**labels).observe(latency_ms)
                else:
                    hist.observe(latency_ms)
    
    def increment_counter(
        self, 
        metric_name: str,
        amount: int = 1,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Increment a counter metric.
        
        Args:
            metric_name: Counter name
            amount: Amount to increment
            labels: Optional labels
        """
        # In-memory
        key = f"{metric_name}:{labels}" if labels else metric_name
        self._counters[key] = self._counters.get(key, 0) + amount
        
        # Prometheus
        if PROMETHEUS_AVAILABLE and self.config.enable_prometheus:
            counter_key = f"{metric_name}_total"
            if counter_key in self._prom_counters:
                counter = self._prom_counters[counter_key]
                if labels:
                    counter.labels(**labels).inc(amount)
                else:
                    counter.inc(amount)
    
    def set_gauge(
        self, 
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Set a gauge metric value.
        
        Args:
            metric_name: Gauge name
            value: Value to set
            labels: Optional labels
        """
        # In-memory
        key = f"{metric_name}:{labels}" if labels else metric_name
        self._gauges[key] = value
        
        # Prometheus
        if PROMETHEUS_AVAILABLE and self.config.enable_prometheus:
            if metric_name in self._prom_gauges:
                gauge = self._prom_gauges[metric_name]
                if labels:
                    gauge.labels(**labels).set(value)
                else:
                    gauge.set(value)
    
    def record_error(self, error_type: str, component: str = 'sentiment') -> None:
        """Record an error occurrence."""
        self.increment_counter(f'{component}_errors', labels={'error_type': error_type})
    
    def record_srm_activation(self, action: str) -> None:
        """Record SRM gate activation."""
        self.increment_counter('srm_gate_activations', labels={'action': action})
    
    def get_latency_percentiles(
        self, 
        metric_name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """Get latency percentiles from in-memory buffer."""
        import numpy as np
        
        key = f"{metric_name}:{labels}" if labels else metric_name
        if key not in self._latency_buffers:
            return {}
        
        data = list(self._latency_buffers[key])
        if len(data) < 10:
            return {}
        
        return {
            'p50': float(np.percentile(data, 50)),
            'p95': float(np.percentile(data, 95)),
            'p99': float(np.percentile(data, 99)),
            'mean': float(np.mean(data)),
            'count': len(data)
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as dictionary."""
        return {
            'counters': dict(self._counters),
            'gauges': dict(self._gauges),
            'latency_summary': {
                k: self.get_latency_percentiles(k.split(':')[0])
                for k in self._latency_buffers.keys()
            }
        }
    
    def reset(self) -> None:
        """Reset all in-memory metrics."""
        self._latency_buffers.clear()
        self._counters.clear()
        self._gauges.clear()
        self._update_count = 0


class AccuracyTracker:
    """
    Track accuracy metrics using proxy signals.
    
    Since we don't have labels in production, use:
    1. Signal-return correlation as accuracy proxy
    2. Regime stability as HMM health indicator
    3. Sentiment distribution skew as drift indicator
    """
    
    def __init__(self, metrics: PrometheusMetricsCollector):
        """
        Initialize accuracy tracker.
        
        Args:
            metrics: Metrics collector to report to
        """
        self.metrics = metrics
        
        # Rolling buffers
        self._signals: deque = deque(maxlen=1000)
        self._returns: deque = deque(maxlen=1000)
        self._sentiments: deque = deque(maxlen=1000)
        self._regime_history: deque = deque(maxlen=1000)
        
        self._last_regime: Optional[str] = None
        self._regime_changes: int = 0
    
    def update(
        self,
        signal: float,
        actual_return: Optional[float] = None,
        sentiment: Optional[float] = None,
        regime: Optional[str] = None
    ) -> None:
        """
        Update accuracy tracking with new data.
        
        Args:
            signal: Generated signal value
            actual_return: Actual return (for correlation)
            sentiment: Sentiment score
            regime: Current regime
        """
        self._signals.append(signal)
        
        if actual_return is not None:
            self._returns.append(actual_return)
        
        if sentiment is not None:
            self._sentiments.append(sentiment)
        
        if regime is not None:
            self._regime_history.append(regime)
            if self._last_regime and regime != self._last_regime:
                self._regime_changes += 1
            self._last_regime = regime
    
    def compute_signal_return_correlation(self) -> Optional[float]:
        """Compute correlation between signals and returns."""
        import numpy as np
        
        if len(self._signals) < 100 or len(self._returns) < 100:
            return None
        
        # Align lengths
        min_len = min(len(self._signals), len(self._returns))
        signals = list(self._signals)[-min_len:]
        returns = list(self._returns)[-min_len:]
        
        if np.std(signals) == 0 or np.std(returns) == 0:
            return 0.0
        
        correlation = float(np.corrcoef(signals, returns)[0, 1])
        
        # Update metrics
        self.metrics.set_gauge('signal_accuracy_proxy', correlation)
        
        return correlation
    
    def compute_sentiment_skew(self) -> Optional[float]:
        """Compute sentiment distribution skewness."""
        from scipy import stats
        
        if len(self._sentiments) < 100:
            return None
        
        skew = float(stats.skew(list(self._sentiments)))
        
        self.metrics.set_gauge(
            'sentiment_distribution_skew', 
            skew, 
            labels={'source': 'combined'}
        )
        
        return skew
    
    def compute_regime_transition_rate(self, hours: float = 1.0) -> float:
        """Compute regime transitions per hour."""
        # Approximate from regime change count
        rate = self._regime_changes / hours if hours > 0 else 0
        
        self.metrics.set_gauge('regime_transition_rate', rate)
        
        return rate
    
    def get_stats(self) -> Dict[str, Any]:
        """Get accuracy tracking statistics."""
        return {
            'signal_count': len(self._signals),
            'return_count': len(self._returns),
            'sentiment_count': len(self._sentiments),
            'regime_changes': self._regime_changes,
            'current_regime': self._last_regime,
            'correlation': self.compute_signal_return_correlation(),
            'skew': self.compute_sentiment_skew()
        }
