"""
Latency Validation - Production SLA Enforcement

Measures, validates, and enforces p50/p95/p99 latency SLAs across all components.

Enhancement 3 from ANTIGRAVITY_SENTIMENT_ENHANCEMENT_GUIDE.md
"""

import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class LatencySLA:
    """SLA definition for a component."""
    p50: float  # milliseconds
    p95: float
    p99: float
    action_on_breach: str = 'LOG_WARNING'  # LOG_WARNING, DISABLE_COMPONENT, RAISE_ALERT, EMERGENCY_HALT


@dataclass 
class LatencyStats:
    """Current latency statistics for a component."""
    p50: float
    p95: float
    p99: float
    mean: float
    min_latency: float
    max_latency: float
    sample_count: int
    sla_breached: bool = False
    breach_type: Optional[str] = None


DEFAULT_LATENCY_SLAS = {
    'sentiment_analysis': LatencySLA(
        p50=30.0,
        p95=80.0,
        p99=100.0,
        action_on_breach='LOG_WARNING'
    ),
    'hmm_update': LatencySLA(
        p50=0.5,
        p95=1.0,
        p99=2.0,
        action_on_breach='LOG_WARNING'
    ),
    'fusion': LatencySLA(
        p50=1.0,
        p95=3.0,
        p99=5.0,
        action_on_breach='LOG_WARNING'
    ),
    'total_signal': LatencySLA(
        p50=5.0,
        p95=20.0,
        p99=50.0,
        action_on_breach='DISABLE_COMPONENT'
    )
}


class LatencyBenchmark:
    """
    Measure, validate, and enforce p50/p95/p99 latency SLAs.
    
    Features:
    - Track latency distributions per component
    - Compute percentiles efficiently
    - Enforce SLA with configurable actions
    - Support circuit breaker pattern
    
    Example:
        bench = LatencyBenchmark()
        
        with bench.measure('sentiment_analysis'):
            result = sentiment.analyze(text)
        
        stats = bench.get_latency_stats('sentiment_analysis')
        breaches = bench.check_sla_breaches()
    """
    
    def __init__(
        self, 
        slas: Optional[Dict[str, LatencySLA]] = None,
        window_size: int = 10000,
        percentile_update_interval: int = 100
    ):
        """
        Initialize latency benchmark.
        
        Args:
            slas: Component SLA definitions
            window_size: Rolling window size for measurements
            percentile_update_interval: Recompute percentiles every N updates
        """
        self.slas = slas or DEFAULT_LATENCY_SLAS
        self.window_size = window_size
        self.percentile_update_interval = percentile_update_interval
        
        # Per-component latency buffers
        self._buffers: Dict[str, deque] = {}
        self._update_counts: Dict[str, int] = {}
        self._cached_stats: Dict[str, LatencyStats] = {}
        
        # Circuit breaker state
        self._disabled_components: set = set()
        self._breach_counts: Dict[str, int] = {}
        
        logger.info(f"LatencyBenchmark initialized with SLAs for: {list(self.slas.keys())}")
    
    @contextmanager
    def measure(self, component: str):
        """
        Context manager to measure latency of a code block.
        
        Args:
            component: Component name to track
            
        Yields:
            None - use as context manager
            
        Example:
            with bench.measure('sentiment_analysis'):
                result = do_work()
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._record_latency(component, elapsed_ms)
    
    def record_latency(self, component: str, latency_ms: float) -> None:
        """
        Manually record a latency measurement.
        
        Args:
            component: Component name
            latency_ms: Latency in milliseconds
        """
        self._record_latency(component, latency_ms)
    
    def _record_latency(self, component: str, latency_ms: float) -> None:
        """Internal method to record latency."""
        # Initialize buffer if needed
        if component not in self._buffers:
            self._buffers[component] = deque(maxlen=self.window_size)
            self._update_counts[component] = 0
            self._breach_counts[component] = 0
        
        # Add measurement
        self._buffers[component].append(latency_ms)
        self._update_counts[component] += 1
        
        # Recompute percentiles periodically
        if self._update_counts[component] % self.percentile_update_interval == 0:
            self._compute_percentiles(component)
    
    def _compute_percentiles(self, component: str) -> None:
        """Compute and cache percentile statistics."""
        if component not in self._buffers or len(self._buffers[component]) < 10:
            return
        
        data = np.array(self._buffers[component])
        
        p50 = float(np.percentile(data, 50))
        p95 = float(np.percentile(data, 95))
        p99 = float(np.percentile(data, 99))
        
        # Check for SLA breach
        sla_breached = False
        breach_type = None
        
        if component in self.slas:
            sla = self.slas[component]
            if p99 > sla.p99:
                sla_breached = True
                breach_type = 'p99'
            elif p95 > sla.p95:
                sla_breached = True
                breach_type = 'p95'
        
        self._cached_stats[component] = LatencyStats(
            p50=p50,
            p95=p95,
            p99=p99,
            mean=float(np.mean(data)),
            min_latency=float(np.min(data)),
            max_latency=float(np.max(data)),
            sample_count=len(data),
            sla_breached=sla_breached,
            breach_type=breach_type
        )
        
        if sla_breached:
            self._handle_breach(component, breach_type)
    
    def _handle_breach(self, component: str, breach_type: str) -> None:
        """Handle SLA breach based on configured action."""
        self._breach_counts[component] = self._breach_counts.get(component, 0) + 1
        
        if component not in self.slas:
            return
        
        sla = self.slas[component]
        stats = self._cached_stats.get(component)
        
        if sla.action_on_breach == 'LOG_WARNING':
            logger.warning(
                f"Latency SLA breach [{component}]: {breach_type} = "
                f"{getattr(stats, breach_type, 0):.2f}ms > {getattr(sla, breach_type)}ms"
            )
        
        elif sla.action_on_breach == 'DISABLE_COMPONENT':
            if self._breach_counts[component] >= 3:  # 3 consecutive breaches
                self._disabled_components.add(component)
                logger.critical(
                    f"Component DISABLED [{component}]: {self._breach_counts[component]} "
                    f"consecutive SLA breaches"
                )
        
        elif sla.action_on_breach == 'RAISE_ALERT':
            logger.error(f"ALERT: Latency SLA breach [{component}]: {breach_type}")
            # TODO: Send webhook/PagerDuty alert
        
        elif sla.action_on_breach == 'EMERGENCY_HALT':
            logger.critical(f"EMERGENCY HALT triggered by [{component}] latency breach")
            # TODO: Trigger SRM emergency halt
    
    def get_latency_stats(self, component: Optional[str] = None) -> Dict[str, LatencyStats]:
        """
        Get latency statistics for component(s).
        
        Args:
            component: Specific component, or None for all
            
        Returns:
            Dict of component -> LatencyStats
        """
        if component:
            if component in self._cached_stats:
                return {component: self._cached_stats[component]}
            return {}
        return dict(self._cached_stats)
    
    def check_sla_breaches(self) -> List[Dict[str, Any]]:
        """
        Check for SLA breaches across all components.
        
        Returns:
            List of breach details
        """
        breaches = []
        
        for component, stats in self._cached_stats.items():
            if stats.sla_breached:
                sla = self.slas.get(component)
                breaches.append({
                    'component': component,
                    'breach_type': stats.breach_type,
                    'actual_value': getattr(stats, stats.breach_type, 0),
                    'sla_value': getattr(sla, stats.breach_type, 0) if sla else 0,
                    'action': sla.action_on_breach if sla else 'NONE'
                })
        
        return breaches
    
    def is_component_disabled(self, component: str) -> bool:
        """Check if component has been disabled by circuit breaker."""
        return component in self._disabled_components
    
    def enable_component(self, component: str) -> None:
        """Re-enable a disabled component."""
        self._disabled_components.discard(component)
        self._breach_counts[component] = 0
        logger.info(f"Component re-enabled: {component}")
    
    def get_report(self) -> str:
        """Generate human-readable latency report."""
        lines = [
            "=" * 70,
            "LATENCY BENCHMARK REPORT",
            "=" * 70,
            "",
            f"{'Component':<25} {'p50':>8} {'p95':>8} {'p99':>8} {'SLA':>8} {'Status':>10}",
            "-" * 70
        ]
        
        for component in sorted(self._cached_stats.keys()):
            stats = self._cached_stats[component]
            sla = self.slas.get(component)
            sla_p99 = sla.p99 if sla else 0
            
            status = "[PASS]" if not stats.sla_breached else "[FAIL]"
            if component in self._disabled_components:
                status = "[DISABLED]"
            
            lines.append(
                f"{component:<25} {stats.p50:>7.2f} {stats.p95:>7.2f} "
                f"{stats.p99:>7.2f} {sla_p99:>7.2f} {status:>10}"
            )
        
        lines.extend(["", "=" * 70])
        return "\n".join(lines)
    
    def reset(self) -> None:
        """Reset all measurements and state."""
        self._buffers.clear()
        self._update_counts.clear()
        self._cached_stats.clear()
        self._disabled_components.clear()
        self._breach_counts.clear()


def validate_quantization_latency(
    baseline_p99: float,
    quantized_p99: float,
    tolerance: float = 1.2
) -> bool:
    """
    Validate that quantization doesn't increase latency beyond tolerance.
    
    CRITICAL: INT8 quantization can INCREASE CPU latency despite smaller model.
    
    Args:
        baseline_p99: p99 latency before quantization (ms)
        quantized_p99: p99 latency after quantization (ms)
        tolerance: Maximum acceptable slowdown ratio (1.2 = 20% slower OK)
        
    Returns:
        True if quantization acceptable, False if should be rejected
    """
    ratio = quantized_p99 / baseline_p99 if baseline_p99 > 0 else float('inf')
    
    if ratio > tolerance:
        logger.warning(
            f"Quantization REJECTED: {quantized_p99:.1f}ms > {baseline_p99:.1f}ms * {tolerance} = "
            f"{baseline_p99 * tolerance:.1f}ms (ratio: {ratio:.2f}x)"
        )
        return False
    
    logger.info(
        f"Quantization ACCEPTED: {quantized_p99:.1f}ms vs baseline {baseline_p99:.1f}ms "
        f"(ratio: {ratio:.2f}x, tolerance: {tolerance}x)"
    )
    return True
