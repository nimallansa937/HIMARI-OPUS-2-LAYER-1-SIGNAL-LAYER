"""
Shadow Mode Runner - Production Deployment Phase 1

Runs enhanced Layer 1 in parallel with legacy system for 72-hour comparison.
Logs all signal discrepancies for analysis before production cutover.

Usage:
    python -m deployment.shadow_mode_runner --duration 72h --symbols BTC,ETH
"""

import time
import json
import logging
import threading
import redis
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SignalComparison:
    """Single comparison between legacy and enhanced signals."""
    timestamp: int
    symbol: str
    legacy_signal: float
    enhanced_signal: float
    signal_diff: float
    legacy_regime: str
    enhanced_regime: str
    regime_match: bool
    legacy_latency_ms: float
    enhanced_latency_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShadowModeConfig:
    """Configuration for shadow mode execution."""
    duration_hours: float = 72.0
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    signal_diff_threshold: float = 0.1  # Alert if diff > 10%
    regime_mismatch_alert_rate: float = 0.05  # Alert if > 5% mismatch
    latency_degradation_threshold: float = 2.0  # Alert if enhanced > 2x legacy
    log_interval_seconds: int = 60
    checkpoint_interval_minutes: int = 30
    redis_key_prefix: str = "himari:shadow:"
    output_dir: str = "shadow_mode_results"


@dataclass
class ShadowModeMetrics:
    """Aggregate metrics from shadow mode execution."""
    start_time: datetime = field(default_factory=datetime.utcnow)
    total_comparisons: int = 0
    signal_diff_sum: float = 0.0
    signal_diff_squared_sum: float = 0.0
    regime_matches: int = 0
    latency_legacy_sum: float = 0.0
    latency_enhanced_sum: float = 0.0
    large_discrepancies: int = 0  # Count of signals exceeding threshold
    
    @property
    def avg_signal_diff(self) -> float:
        if self.total_comparisons == 0:
            return 0.0
        return self.signal_diff_sum / self.total_comparisons
    
    @property
    def signal_diff_std(self) -> float:
        if self.total_comparisons < 2:
            return 0.0
        variance = (self.signal_diff_squared_sum / self.total_comparisons) - (self.avg_signal_diff ** 2)
        return np.sqrt(max(0, variance))
    
    @property
    def regime_match_rate(self) -> float:
        if self.total_comparisons == 0:
            return 0.0
        return self.regime_matches / self.total_comparisons
    
    @property
    def avg_latency_legacy(self) -> float:
        if self.total_comparisons == 0:
            return 0.0
        return self.latency_legacy_sum / self.total_comparisons
    
    @property
    def avg_latency_enhanced(self) -> float:
        if self.total_comparisons == 0:
            return 0.0
        return self.latency_enhanced_sum / self.total_comparisons


class ShadowModeRunner:
    """
    Execute enhanced Layer 1 in shadow mode alongside legacy system.
    
    Shadow mode enables safe production validation by:
    1. Running both systems on live data
    2. Comparing signals in real-time
    3. Logging discrepancies for analysis
    4. NOT affecting actual trading decisions
    
    After 72 hours with acceptable metrics, proceed to production cutover.
    """
    
    def __init__(
        self,
        config: Optional[ShadowModeConfig] = None,
        redis_client: Optional[redis.Redis] = None
    ):
        self.config = config or ShadowModeConfig()
        self.redis = redis_client
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Per-symbol metrics
        self._metrics: Dict[str, ShadowModeMetrics] = defaultdict(ShadowModeMetrics)
        
        # Recent comparisons buffer (for real-time dashboard)
        self._recent_comparisons: List[SignalComparison] = []
        self._max_recent = 1000
        
        # Alerts
        self._alerts: List[Dict[str, Any]] = []
        
        logger.info(f"ShadowModeRunner initialized: {self.config.duration_hours}h for {self.config.symbols}")
    
    def start(self) -> None:
        """Start shadow mode execution in background thread."""
        if self._running:
            logger.warning("Shadow mode already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"Shadow mode STARTED at {datetime.utcnow()}")
    
    def stop(self) -> None:
        """Stop shadow mode execution."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Shadow mode STOPPED")
    
    def compare_signals(
        self,
        symbol: str,
        legacy_result: Dict[str, Any],
        enhanced_result: Dict[str, Any]
    ) -> SignalComparison:
        """
        Compare legacy and enhanced signal outputs.
        
        Args:
            symbol: Trading symbol
            legacy_result: Output from legacy signal processor
            enhanced_result: Output from IntegratedSignalLayer
        
        Returns:
            SignalComparison with full comparison data
        """
        legacy_signal = legacy_result.get('signal', legacy_result.get('momentum', 0))
        enhanced_signal = enhanced_result.get('composite_signal', 0)
        
        comparison = SignalComparison(
            timestamp=int(time.time() * 1000),
            symbol=symbol,
            legacy_signal=float(legacy_signal),
            enhanced_signal=float(enhanced_signal),
            signal_diff=abs(float(enhanced_signal) - float(legacy_signal)),
            legacy_regime=legacy_result.get('regime', 'UNKNOWN'),
            enhanced_regime=enhanced_result.get('regime', 'UNKNOWN'),
            regime_match=legacy_result.get('regime') == enhanced_result.get('regime'),
            legacy_latency_ms=float(legacy_result.get('latency_ms', 0)),
            enhanced_latency_ms=float(enhanced_result.get('latency_ms', 0))
        )
        
        # Update metrics
        self._update_metrics(symbol, comparison)
        
        # Store comparison
        self._recent_comparisons.append(comparison)
        if len(self._recent_comparisons) > self._max_recent:
            self._recent_comparisons.pop(0)
        
        # Check for alerts
        self._check_alerts(symbol, comparison)
        
        # Persist to Redis if available
        if self.redis:
            self._persist_comparison(comparison)
        
        return comparison
    
    def _update_metrics(self, symbol: str, comparison: SignalComparison) -> None:
        """Update aggregate metrics for symbol."""
        metrics = self._metrics[symbol]
        metrics.total_comparisons += 1
        metrics.signal_diff_sum += comparison.signal_diff
        metrics.signal_diff_squared_sum += comparison.signal_diff ** 2
        metrics.latency_legacy_sum += comparison.legacy_latency_ms
        metrics.latency_enhanced_sum += comparison.enhanced_latency_ms
        
        if comparison.regime_match:
            metrics.regime_matches += 1
        
        if comparison.signal_diff > self.config.signal_diff_threshold:
            metrics.large_discrepancies += 1
    
    def _check_alerts(self, symbol: str, comparison: SignalComparison) -> None:
        """Check if comparison triggers any alerts."""
        alerts = []
        
        # Large signal discrepancy
        if comparison.signal_diff > self.config.signal_diff_threshold:
            alerts.append({
                'type': 'SIGNAL_DISCREPANCY',
                'severity': 'WARNING',
                'symbol': symbol,
                'message': f"Signal diff {comparison.signal_diff:.3f} > threshold {self.config.signal_diff_threshold}",
                'legacy': comparison.legacy_signal,
                'enhanced': comparison.enhanced_signal
            })
        
        # Latency degradation
        if comparison.legacy_latency_ms > 0:
            latency_ratio = comparison.enhanced_latency_ms / comparison.legacy_latency_ms
            if latency_ratio > self.config.latency_degradation_threshold:
                alerts.append({
                    'type': 'LATENCY_DEGRADATION',
                    'severity': 'WARNING',
                    'symbol': symbol,
                    'message': f"Enhanced latency {latency_ratio:.1f}x legacy",
                    'legacy_ms': comparison.legacy_latency_ms,
                    'enhanced_ms': comparison.enhanced_latency_ms
                })
        
        # Regime mismatch tracking
        metrics = self._metrics[symbol]
        if metrics.total_comparisons >= 100:  # Only alert after warmup
            mismatch_rate = 1.0 - metrics.regime_match_rate
            if mismatch_rate > self.config.regime_mismatch_alert_rate:
                alerts.append({
                    'type': 'REGIME_MISMATCH_RATE',
                    'severity': 'INFO',
                    'symbol': symbol,
                    'message': f"Regime mismatch rate {mismatch_rate:.1%} > {self.config.regime_mismatch_alert_rate:.1%}",
                    'match_rate': metrics.regime_match_rate
                })
        
        for alert in alerts:
            alert['timestamp'] = datetime.utcnow().isoformat()
            self._alerts.append(alert)
            logger.warning(f"SHADOW ALERT: {alert['type']} - {alert['message']}")
    
    def _persist_comparison(self, comparison: SignalComparison) -> None:
        """Persist comparison to Redis for dashboard access."""
        if not self.redis:
            return
        
        try:
            key = f"{self.config.redis_key_prefix}comparison:{comparison.symbol}"
            self.redis.lpush(key, json.dumps(comparison.to_dict()))
            self.redis.ltrim(key, 0, 999)  # Keep last 1000
            
            # Update metrics hash
            metrics_key = f"{self.config.redis_key_prefix}metrics:{comparison.symbol}"
            metrics = self._metrics[comparison.symbol]
            self.redis.hset(metrics_key, mapping={
                'total_comparisons': metrics.total_comparisons,
                'avg_signal_diff': metrics.avg_signal_diff,
                'regime_match_rate': metrics.regime_match_rate,
                'avg_latency_legacy': metrics.avg_latency_legacy,
                'avg_latency_enhanced': metrics.avg_latency_enhanced,
                'large_discrepancies': metrics.large_discrepancies,
                'last_update': datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.error(f"Failed to persist comparison: {e}")
    
    def _run_loop(self) -> None:
        """Main shadow mode execution loop."""
        end_time = datetime.utcnow() + timedelta(hours=self.config.duration_hours)
        last_checkpoint = datetime.utcnow()
        
        logger.info(f"Shadow mode will run until {end_time}")
        
        while self._running and datetime.utcnow() < end_time:
            try:
                # Checkpoint metrics periodically
                now = datetime.utcnow()
                if (now - last_checkpoint).total_seconds() >= self.config.checkpoint_interval_minutes * 60:
                    self._save_checkpoint()
                    last_checkpoint = now
                
                time.sleep(1)  # Main loop tick
                
            except Exception as e:
                logger.error(f"Shadow mode error: {e}")
                time.sleep(5)
        
        # Final save
        self._save_checkpoint()
        self._save_final_report()
        logger.info("Shadow mode completed")
    
    def _save_checkpoint(self) -> None:
        """Save checkpoint of current metrics."""
        import os
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        checkpoint = {
            'timestamp': datetime.utcnow().isoformat(),
            'duration_hours': self.config.duration_hours,
            'symbols': self.config.symbols,
            'metrics': {
                symbol: {
                    'total_comparisons': m.total_comparisons,
                    'avg_signal_diff': m.avg_signal_diff,
                    'signal_diff_std': m.signal_diff_std,
                    'regime_match_rate': m.regime_match_rate,
                    'avg_latency_legacy': m.avg_latency_legacy,
                    'avg_latency_enhanced': m.avg_latency_enhanced,
                    'large_discrepancies': m.large_discrepancies
                }
                for symbol, m in self._metrics.items()
            },
            'alerts_count': len(self._alerts)
        }
        
        filepath = f"{self.config.output_dir}/checkpoint_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        logger.info(f"Saved checkpoint: {filepath}")
    
    def _save_final_report(self) -> None:
        """Save final shadow mode report."""
        import os
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        report = {
            'report_type': 'SHADOW_MODE_FINAL',
            'generated_at': datetime.utcnow().isoformat(),
            'config': asdict(self.config),
            'summary': {},
            'per_symbol_metrics': {},
            'alerts': self._alerts[-100:],  # Last 100 alerts
            'recommendation': None
        }
        
        # Aggregate summary
        total_comparisons = sum(m.total_comparisons for m in self._metrics.values())
        total_discrepancies = sum(m.large_discrepancies for m in self._metrics.values())
        
        # Compute go/no-go recommendation
        all_pass = True
        issues = []
        
        for symbol, m in self._metrics.items():
            report['per_symbol_metrics'][symbol] = {
                'total_comparisons': m.total_comparisons,
                'avg_signal_diff': round(m.avg_signal_diff, 4),
                'signal_diff_std': round(m.signal_diff_std, 4),
                'regime_match_rate': round(m.regime_match_rate, 4),
                'avg_latency_legacy_ms': round(m.avg_latency_legacy, 2),
                'avg_latency_enhanced_ms': round(m.avg_latency_enhanced, 2),
                'large_discrepancy_rate': round(m.large_discrepancies / max(1, m.total_comparisons), 4)
            }
            
            # Check pass criteria
            if m.regime_match_rate < 0.9:
                all_pass = False
                issues.append(f"{symbol}: Regime match rate {m.regime_match_rate:.1%} < 90%")
            
            if m.avg_signal_diff > 0.15:
                all_pass = False
                issues.append(f"{symbol}: Avg signal diff {m.avg_signal_diff:.3f} > 0.15")
            
            if m.avg_latency_enhanced > m.avg_latency_legacy * 3:
                all_pass = False
                issues.append(f"{symbol}: Enhanced latency {m.avg_latency_enhanced:.1f}ms > 3x legacy")
        
        report['summary'] = {
            'total_comparisons': total_comparisons,
            'total_discrepancies': total_discrepancies,
            'discrepancy_rate': round(total_discrepancies / max(1, total_comparisons), 4),
            'alerts_triggered': len(self._alerts),
            'all_pass': all_pass,
            'issues': issues
        }
        
        if all_pass:
            report['recommendation'] = 'PROCEED_TO_PRODUCTION'
        else:
            report['recommendation'] = 'HOLD_FOR_REVIEW'
        
        filepath = f"{self.config.output_dir}/shadow_mode_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved final report: {filepath}")
        
        # Log summary
        logger.info("=" * 60)
        logger.info("SHADOW MODE COMPLETE")
        logger.info(f"Total comparisons: {total_comparisons}")
        logger.info(f"Discrepancy rate: {report['summary']['discrepancy_rate']:.1%}")
        logger.info(f"Recommendation: {report['recommendation']}")
        if issues:
            logger.warning(f"Issues found: {issues}")
        logger.info("=" * 60)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current shadow mode status."""
        return {
            'running': self._running,
            'start_time': min((m.start_time for m in self._metrics.values()), default=None),
            'symbols': list(self._metrics.keys()),
            'metrics': {
                symbol: {
                    'comparisons': m.total_comparisons,
                    'avg_diff': round(m.avg_signal_diff, 4),
                    'regime_match': round(m.regime_match_rate, 4)
                }
                for symbol, m in self._metrics.items()
            },
            'alerts_count': len(self._alerts),
            'recent_alerts': self._alerts[-5:]
        }


# CLI entry point
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='HIMARI Shadow Mode Runner')
    parser.add_argument('--duration', default='72h', help='Duration (e.g., 72h, 24h)')
    parser.add_argument('--symbols', default='BTCUSDT,ETHUSDT', help='Comma-separated symbols')
    parser.add_argument('--output', default='shadow_mode_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Parse duration
    duration_str = args.duration.lower()
    if duration_str.endswith('h'):
        duration_hours = float(duration_str[:-1])
    elif duration_str.endswith('d'):
        duration_hours = float(duration_str[:-1]) * 24
    else:
        duration_hours = float(duration_str)
    
    config = ShadowModeConfig(
        duration_hours=duration_hours,
        symbols=args.symbols.split(','),
        output_dir=args.output
    )
    
    runner = ShadowModeRunner(config=config)
    runner.start()
    
    try:
        while runner._running:
            time.sleep(10)
            status = runner.get_status()
            print(f"Shadow mode: {status['metrics']}")
    except KeyboardInterrupt:
        runner.stop()
