"""
Latency Profiler for HIMARI L1 Pipeline

Ensures the full signal processing pipeline meets the <12ms target.

Profiles:
- Individual primitive latencies
- Tier-by-tier breakdown
- End-to-end pipeline latency
- Identifies bottlenecks

Usage:
    profiler = LatencyProfiler()
    profiler.profile_pipeline()
    profiler.print_report()
"""

import time
import numpy as np
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from functools import wraps
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class LatencyMeasurement:
    """Single latency measurement."""
    name: str
    duration_ms: float
    tier: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class LatencyStats:
    """Statistics for a component."""
    name: str
    tier: str
    count: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'tier': self.tier,
            'count': self.count,
            'mean_ms': self.mean_ms,
            'std_ms': self.std_ms,
            'min_ms': self.min_ms,
            'max_ms': self.max_ms,
            'p50_ms': self.p50_ms,
            'p95_ms': self.p95_ms,
            'p99_ms': self.p99_ms,
        }


class LatencyProfiler:
    """
    Profile latency of HIMARI L1 signal processing pipeline.
    
    Target: <12ms end-to-end per update
    
    Tier Budgets:
    - Tier 6 (LOB ingestion): 2ms
    - Tier 5 (Primitives): 2ms  
    - Tier 4 (DSP/Filters): 2ms
    - Tier 3 (ML): 3ms
    - Tier 2 (Volatility/Regime): 2ms
    - Tier 1 (Fusion): 1ms
    """
    
    TARGET_LATENCY_MS = 12.0
    
    TIER_BUDGETS = {
        'tier6': 2.0,  # LOB ingestion
        'tier5': 2.0,  # Primitives
        'tier4': 2.0,  # DSP/Filters
        'tier3': 3.0,  # ML
        'tier2': 2.0,  # Volatility/Regime
        'tier1': 1.0,  # Fusion
    }
    
    def __init__(self):
        self.measurements: Dict[str, List[float]] = {}
        self.tier_mapping: Dict[str, str] = {}
        self._start_times: Dict[str, float] = {}
    
    @contextmanager
    def measure(self, name: str, tier: str = ""):
        """
        Context manager for measuring latency.
        
        Usage:
            with profiler.measure("kalman_update", tier="tier5"):
                kalman.update(price)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            duration_ms = (end - start) * 1000
            
            if name not in self.measurements:
                self.measurements[name] = []
                self.tier_mapping[name] = tier
            
            self.measurements[name].append(duration_ms)
    
    def start(self, name: str, tier: str = "") -> None:
        """Start timing a component."""
        self._start_times[name] = time.perf_counter()
        if name not in self.tier_mapping:
            self.tier_mapping[name] = tier
    
    def stop(self, name: str) -> float:
        """Stop timing and record measurement."""
        if name not in self._start_times:
            return 0.0
        
        end = time.perf_counter()
        duration_ms = (end - self._start_times[name]) * 1000
        del self._start_times[name]
        
        if name not in self.measurements:
            self.measurements[name] = []
        
        self.measurements[name].append(duration_ms)
        return duration_ms
    
    def record(self, name: str, duration_ms: float, tier: str = "") -> None:
        """Directly record a measurement."""
        if name not in self.measurements:
            self.measurements[name] = []
            self.tier_mapping[name] = tier
        
        self.measurements[name].append(duration_ms)
    
    def get_stats(self, name: str) -> Optional[LatencyStats]:
        """Get statistics for a component."""
        if name not in self.measurements:
            return None
        
        durations = np.array(self.measurements[name])
        
        return LatencyStats(
            name=name,
            tier=self.tier_mapping.get(name, ""),
            count=len(durations),
            mean_ms=float(np.mean(durations)),
            std_ms=float(np.std(durations)),
            min_ms=float(np.min(durations)),
            max_ms=float(np.max(durations)),
            p50_ms=float(np.percentile(durations, 50)),
            p95_ms=float(np.percentile(durations, 95)),
            p99_ms=float(np.percentile(durations, 99)),
        )
    
    def get_tier_stats(self) -> Dict[str, LatencyStats]:
        """Get aggregate statistics by tier."""
        tier_durations: Dict[str, List[float]] = {}
        
        # Aggregate by tier
        for name, durations in self.measurements.items():
            tier = self.tier_mapping.get(name, "other")
            if tier not in tier_durations:
                tier_durations[tier] = []
            # Sum durations within same update (approximate)
            tier_durations[tier].extend(durations)
        
        # Calculate stats
        result = {}
        for tier, durations in tier_durations.items():
            durations = np.array(durations)
            result[tier] = LatencyStats(
                name=tier,
                tier=tier,
                count=len(durations),
                mean_ms=float(np.mean(durations)),
                std_ms=float(np.std(durations)),
                min_ms=float(np.min(durations)),
                max_ms=float(np.max(durations)),
                p50_ms=float(np.percentile(durations, 50)),
                p95_ms=float(np.percentile(durations, 95)),
                p99_ms=float(np.percentile(durations, 99)),
            )
        
        return result
    
    def get_total_latency(self) -> LatencyStats:
        """Get total pipeline latency."""
        if 'total' in self.measurements:
            return self.get_stats('total')
        
        # Estimate from tier sums
        tier_stats = self.get_tier_stats()
        total_mean = sum(s.mean_ms for s in tier_stats.values())
        
        return LatencyStats(
            name='total_estimated',
            tier='all',
            count=1,
            mean_ms=total_mean,
            std_ms=0,
            min_ms=total_mean,
            max_ms=total_mean,
            p50_ms=total_mean,
            p95_ms=total_mean,
            p99_ms=total_mean,
        )
    
    def print_report(self) -> None:
        """Print formatted latency report."""
        print("\n" + "=" * 70)
        print("HIMARI L1 LATENCY PROFILING REPORT")
        print("=" * 70)
        print(f"Target: <{self.TARGET_LATENCY_MS}ms end-to-end")
        print()
        
        # Per-component
        print("Component Latencies:")
        print("-" * 70)
        print(f"{'Component':<25} {'Tier':<8} {'Mean':>8} {'P95':>8} {'P99':>8} {'Max':>8}")
        print("-" * 70)
        
        for name in sorted(self.measurements.keys()):
            stats = self.get_stats(name)
            print(f"{name:<25} {stats.tier:<8} "
                  f"{stats.mean_ms:>7.3f}ms {stats.p95_ms:>7.3f}ms "
                  f"{stats.p99_ms:>7.3f}ms {stats.max_ms:>7.3f}ms")
        
        # Per-tier
        print("\n" + "-" * 70)
        print("Tier Summary:")
        print("-" * 70)
        
        tier_stats = self.get_tier_stats()
        total_mean = 0
        
        for tier in sorted(tier_stats.keys()):
            stats = tier_stats[tier]
            budget = self.TIER_BUDGETS.get(tier, 2.0)
            status = "✓" if stats.mean_ms <= budget else "✗"
            print(f"{status} {tier:<10} Mean: {stats.mean_ms:>6.3f}ms / {budget}ms budget")
            total_mean += stats.mean_ms
        
        # Total
        print("-" * 70)
        status = "✓ PASS" if total_mean <= self.TARGET_LATENCY_MS else "✗ FAIL"
        print(f"Total Pipeline: {total_mean:.3f}ms ({status})")
        print("=" * 70)
    
    def identify_bottlenecks(self, threshold_pct: float = 20) -> List[str]:
        """Identify components using >threshold% of budget."""
        bottlenecks = []
        total = sum(np.mean(d) for d in self.measurements.values())
        
        for name, durations in self.measurements.items():
            mean = np.mean(durations)
            pct = (mean / total) * 100 if total > 0 else 0
            
            if pct >= threshold_pct:
                bottlenecks.append(f"{name}: {mean:.3f}ms ({pct:.1f}%)")
        
        return bottlenecks
    
    def export_json(self) -> dict:
        """Export all stats as JSON-serializable dict."""
        return {
            'target_ms': self.TARGET_LATENCY_MS,
            'components': {
                name: self.get_stats(name).to_dict()
                for name in self.measurements
            },
            'tiers': {
                tier: stats.to_dict()
                for tier, stats in self.get_tier_stats().items()
            },
            'bottlenecks': self.identify_bottlenecks(),
        }
    
    def reset(self) -> None:
        """Reset all measurements."""
        self.measurements.clear()
        self._start_times.clear()


def profile_decorator(profiler: LatencyProfiler, name: str, tier: str = ""):
    """Decorator to automatically profile a function."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with profiler.measure(name, tier):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Simulate full pipeline profiling
def profile_simulated_pipeline(n_iterations: int = 1000) -> LatencyProfiler:
    """
    Simulate and profile the full HIMARI L1 pipeline.
    
    Uses estimated latencies for each component.
    """
    profiler = LatencyProfiler()
    
    for _ in range(n_iterations):
        # Tier 6: LOB Ingestion
        with profiler.measure("lob_orderbook_parse", "tier6"):
            time.sleep(0.0002)  # 0.2ms
        with profiler.measure("lob_trade_parse", "tier6"):
            time.sleep(0.0001)  # 0.1ms
        
        # Tier 5: Primitives
        with profiler.measure("welford_update", "tier5"):
            time.sleep(0.00005)  # 0.05ms
        with profiler.measure("kalman_update", "tier5"):
            time.sleep(0.0001)  # 0.1ms
        with profiler.measure("garch_update", "tier5"):
            time.sleep(0.0002)  # 0.2ms
        with profiler.measure("order_flow_features", "tier5"):
            time.sleep(0.0003)  # 0.3ms
        
        # Tier 4: DSP/Filters
        with profiler.measure("ultimate_smoother", "tier4"):
            time.sleep(0.0001)  # 0.1ms
        with profiler.measure("hmm_forward", "tier4"):
            time.sleep(0.0003)  # 0.3ms
        with profiler.measure("hurst_estimator", "tier4"):
            time.sleep(0.0002)  # 0.2ms
        
        # Tier 3: ML
        with profiler.measure("lorentzian_knn", "tier3"):
            time.sleep(0.0008)  # 0.8ms
        with profiler.measure("ensemble_predict", "tier3"):
            time.sleep(0.0005)  # 0.5ms
        
        # Tier 2: Volatility/Regime
        with profiler.measure("regime_detection", "tier2"):
            time.sleep(0.0003)  # 0.3ms
        with profiler.measure("rvol_calculation", "tier2"):
            time.sleep(0.0001)  # 0.1ms
        
        # Tier 1: Fusion
        with profiler.measure("dempster_shafer", "tier1"):
            time.sleep(0.0002)  # 0.2ms
        with profiler.measure("feature_assembly", "tier1"):
            time.sleep(0.0001)  # 0.1ms
    
    return profiler


# Quick test
if __name__ == "__main__":
    print("Profiling HIMARI L1 Pipeline (simulated)...")
    print("Running 1000 iterations...\n")
    
    profiler = profile_simulated_pipeline(1000)
    profiler.print_report()
    
    print("\nBottlenecks (>15% of budget):")
    for b in profiler.identify_bottlenecks(15):
        print(f"  ⚠️ {b}")
    
    print("\n✓ Latency profiling complete!")
