"""
Prometheus Metrics for HIMARI Data Ingestion

Exposes metrics for monitoring:
- Message throughput per exchange/symbol
- Latency histograms
- Connection status
- Error rates

Scrape endpoint: http://localhost:8000/metrics
"""

import time
import logging
from typing import Dict, Any, Optional
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)

# Try to import prometheus_client, fall back to stub if not installed
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed, metrics disabled")


class IngestionMetrics:
    """
    Prometheus metrics for data ingestion pipeline.
    
    Metrics:
    - ingestion_messages_total: Counter of messages by exchange/symbol
    - ingestion_latency_seconds: Histogram of message latency
    - ingestion_connection_status: Gauge of connection status (1=connected)
    - ingestion_errors_total: Counter of errors by type
    - ingestion_buffer_size: Gauge of local buffer size (when Kafka unavailable)
    
    Usage:
        metrics = IngestionMetrics(port=8000)
        metrics.start()
        
        # Record a message
        metrics.record_message("binance", "BTCUSDT", latency_ms=5.2)
        
        # Record connection status
        metrics.set_connection_status("binance", True)
    """
    
    def __init__(self, port: int = 8000, enable: bool = True):
        """
        Initialize metrics.
        
        Args:
            port: Port for Prometheus scrape endpoint
            enable: Whether to enable metrics (False for testing)
        """
        self.port = port
        self.enabled = enable and PROMETHEUS_AVAILABLE
        self._started = False
        
        if self.enabled:
            self._setup_metrics()
        else:
            # Stub counters for when Prometheus not available
            self._stub_counters: Dict[str, int] = defaultdict(int)
    
    def _setup_metrics(self):
        """Create Prometheus metric objects."""
        # Message counter
        self.messages_total = Counter(
            'ingestion_messages_total',
            'Total messages ingested',
            ['exchange', 'symbol']
        )
        
        # Latency histogram (buckets in seconds)
        self.latency_histogram = Histogram(
            'ingestion_latency_seconds',
            'Message ingestion latency',
            ['exchange'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0]
        )
        
        # Connection status gauge (1 = connected, 0 = disconnected)
        self.connection_status = Gauge(
            'ingestion_connection_status',
            'Exchange connection status',
            ['exchange']
        )
        
        # Error counter
        self.errors_total = Counter(
            'ingestion_errors_total',
            'Total ingestion errors',
            ['exchange', 'error_type']
        )
        
        # Buffer size (messages waiting for Kafka)
        self.buffer_size = Gauge(
            'ingestion_buffer_size',
            'Messages buffered locally'
        )
        
        # Message rate (derived from counter, but useful)
        self.messages_per_second = Gauge(
            'ingestion_messages_per_second',
            'Current messages per second',
            ['exchange']
        )
    
    def start(self) -> None:
        """Start Prometheus HTTP server."""
        if not self.enabled or self._started:
            return
        
        try:
            start_http_server(self.port)
            self._started = True
            logger.info(f"Prometheus metrics available at http://localhost:{self.port}/metrics")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
    
    def record_message(
        self,
        exchange: str,
        symbol: str,
        latency_ms: Optional[float] = None
    ) -> None:
        """
        Record an ingested message.
        
        Args:
            exchange: Source exchange
            symbol: Trading pair
            latency_ms: Message latency in milliseconds
        """
        if self.enabled:
            self.messages_total.labels(exchange=exchange, symbol=symbol).inc()
            
            if latency_ms is not None:
                # Convert to seconds for Prometheus
                self.latency_histogram.labels(exchange=exchange).observe(latency_ms / 1000)
        else:
            self._stub_counters[f"messages:{exchange}:{symbol}"] += 1
    
    def set_connection_status(self, exchange: str, connected: bool) -> None:
        """
        Update connection status for an exchange.
        
        Args:
            exchange: Exchange name
            connected: True if connected
        """
        if self.enabled:
            self.connection_status.labels(exchange=exchange).set(1 if connected else 0)
        else:
            self._stub_counters[f"connected:{exchange}"] = 1 if connected else 0
    
    def record_error(self, exchange: str, error_type: str) -> None:
        """
        Record an error.
        
        Args:
            exchange: Source exchange
            error_type: Type of error (e.g., "connection", "parse", "timeout")
        """
        if self.enabled:
            self.errors_total.labels(exchange=exchange, error_type=error_type).inc()
        else:
            self._stub_counters[f"errors:{exchange}:{error_type}"] += 1
    
    def set_buffer_size(self, size: int) -> None:
        """Update the local buffer size."""
        if self.enabled:
            self.buffer_size.set(size)
        else:
            self._stub_counters["buffer_size"] = size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current metric values (for logging/debugging)."""
        if not self.enabled:
            return dict(self._stub_counters)
        
        # Collect current values
        stats = {
            "messages_total": {},
            "connection_status": {},
            "buffer_size": 0,
        }
        
        # This is a simplified view - full metrics in /metrics endpoint
        return stats


# Simple test
if __name__ == "__main__":
    import random
    
    metrics = IngestionMetrics(port=8099)
    metrics.start()
    
    print("Metrics server running at http://localhost:8099/metrics")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            # Simulate messages
            for exchange in ["binance", "kraken"]:
                for symbol in ["BTCUSDT", "ETHUSDT"]:
                    metrics.record_message(exchange, symbol, latency_ms=random.uniform(1, 20))
                
                metrics.set_connection_status(exchange, True)
            
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped")
