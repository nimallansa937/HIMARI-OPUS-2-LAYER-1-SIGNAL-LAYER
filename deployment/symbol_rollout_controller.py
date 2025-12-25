"""
Symbol Rollout Controller - Gradual Production Cutover

Manages phased rollout of enhanced Layer 1 by symbol:
1. BTC (highest volume, best test case)
2. ETH (second tier)
3. Remaining symbols (gradual expansion)

Each phase requires green metrics before proceeding.
"""

import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import redis

logger = logging.getLogger(__name__)


class RolloutPhase(Enum):
    """Rollout phases for production cutover."""
    NOT_STARTED = "not_started"
    PHASE_1_BTC = "phase_1_btc"       # BTC only
    PHASE_2_ETH = "phase_2_eth"       # + ETH
    PHASE_3_MAJORS = "phase_3_majors" # + SOL, BNB, XRP
    PHASE_4_FULL = "phase_4_full"     # All symbols
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"


@dataclass
class RolloutCriteria:
    """Criteria that must be met to proceed to next phase."""
    min_duration_hours: float = 24.0
    min_comparisons: int = 1000
    max_signal_diff: float = 0.15
    min_regime_match_rate: float = 0.85
    max_latency_ms: float = 10.0
    max_error_rate: float = 0.01
    max_discrepancy_rate: float = 0.05


@dataclass
class PhaseMetrics:
    """Metrics for a rollout phase."""
    phase: RolloutPhase
    start_time: datetime = field(default_factory=datetime.utcnow)
    symbols: List[str] = field(default_factory=list)
    comparisons: int = 0
    errors: int = 0
    avg_signal_diff: float = 0.0
    regime_match_rate: float = 1.0
    avg_latency_ms: float = 0.0
    discrepancy_count: int = 0
    
    @property
    def duration_hours(self) -> float:
        return (datetime.utcnow() - self.start_time).total_seconds() / 3600
    
    @property
    def error_rate(self) -> float:
        if self.comparisons == 0:
            return 0.0
        return self.errors / self.comparisons
    
    @property
    def discrepancy_rate(self) -> float:
        if self.comparisons == 0:
            return 0.0
        return self.discrepancy_count / self.comparisons


@dataclass
class RolloutConfig:
    """Configuration for symbol rollout."""
    phase_1_symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])
    phase_2_symbols: List[str] = field(default_factory=lambda: ["ETHUSDT"])
    phase_3_symbols: List[str] = field(default_factory=lambda: ["SOLUSDT", "BNBUSDT", "XRPUSDT"])
    phase_4_symbols: List[str] = field(default_factory=lambda: ["ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT"])
    criteria: RolloutCriteria = field(default_factory=RolloutCriteria)
    auto_advance: bool = False  # If True, auto-advance when criteria met
    redis_key_prefix: str = "himari:rollout:"


class SymbolRolloutController:
    """
    Control gradual production rollout by symbol.
    
    Rollout sequence:
    1. Enable enhanced layer for BTC only
    2. Monitor for 24h, verify metrics
    3. If green, add ETH
    4. Continue expanding until full rollout
    
    At any point, can rollback to legacy for specific symbols.
    """
    
    def __init__(
        self,
        config: Optional[RolloutConfig] = None,
        redis_client: Optional[redis.Redis] = None
    ):
        self.config = config or RolloutConfig()
        self.redis = redis_client
        
        self._current_phase = RolloutPhase.NOT_STARTED
        self._phase_metrics: Dict[RolloutPhase, PhaseMetrics] = {}
        self._enabled_symbols: set = set()
        self._rollback_reasons: List[Dict[str, Any]] = []
        
        # Callbacks for rollout events
        self._on_phase_change: Optional[Callable[[RolloutPhase, RolloutPhase], None]] = None
        self._on_symbol_enabled: Optional[Callable[[str], None]] = None
        self._on_rollback: Optional[Callable[[str, str], None]] = None
        
        logger.info("SymbolRolloutController initialized")
    
    @property
    def current_phase(self) -> RolloutPhase:
        return self._current_phase
    
    @property
    def enabled_symbols(self) -> List[str]:
        return list(self._enabled_symbols)
    
    def is_symbol_enabled(self, symbol: str) -> bool:
        """Check if enhanced layer is enabled for symbol."""
        return symbol in self._enabled_symbols
    
    def start_rollout(self) -> bool:
        """Start the rollout process (Phase 1: BTC)."""
        if self._current_phase != RolloutPhase.NOT_STARTED:
            logger.warning(f"Rollout already in progress: {self._current_phase}")
            return False
        
        self._advance_to_phase(RolloutPhase.PHASE_1_BTC)
        return True
    
    def advance_phase(self, force: bool = False) -> bool:
        """
        Advance to next rollout phase.
        
        Args:
            force: If True, skip criteria check
        
        Returns:
            True if advanced, False if criteria not met
        """
        if not force:
            passed, issues = self._check_criteria()
            if not passed:
                logger.warning(f"Cannot advance - criteria not met: {issues}")
                return False
        
        next_phase = self._get_next_phase()
        if next_phase is None:
            logger.info("Rollout already at final phase")
            return False
        
        self._advance_to_phase(next_phase)
        return True
    
    def rollback_symbol(self, symbol: str, reason: str) -> bool:
        """
        Rollback specific symbol to legacy processing.
        
        Args:
            symbol: Symbol to rollback
            reason: Reason for rollback
        
        Returns:
            True if rolled back
        """
        if symbol not in self._enabled_symbols:
            logger.warning(f"Symbol {symbol} not enabled, cannot rollback")
            return False
        
        self._enabled_symbols.remove(symbol)
        
        rollback_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': symbol,
            'reason': reason,
            'phase': self._current_phase.value
        }
        self._rollback_reasons.append(rollback_record)
        
        logger.warning(f"ROLLBACK: {symbol} - {reason}")
        
        if self._on_rollback:
            self._on_rollback(symbol, reason)
        
        self._persist_state()
        return True
    
    def rollback_all(self, reason: str) -> None:
        """Rollback all symbols to legacy processing."""
        logger.error(f"FULL ROLLBACK: {reason}")
        
        for symbol in list(self._enabled_symbols):
            self.rollback_symbol(symbol, reason)
        
        self._current_phase = RolloutPhase.ROLLED_BACK
        self._persist_state()
    
    def update_metrics(
        self,
        symbol: str,
        signal_diff: float,
        regime_match: bool,
        latency_ms: float,
        is_error: bool = False,
        is_discrepancy: bool = False
    ) -> None:
        """
        Update metrics for current phase.
        
        Called by shadow mode or production monitoring.
        """
        if self._current_phase not in self._phase_metrics:
            return
        
        metrics = self._phase_metrics[self._current_phase]
        metrics.comparisons += 1
        
        if is_error:
            metrics.errors += 1
        
        if is_discrepancy:
            metrics.discrepancy_count += 1
        
        # Running averages
        n = metrics.comparisons
        metrics.avg_signal_diff = ((n - 1) * metrics.avg_signal_diff + signal_diff) / n
        metrics.avg_latency_ms = ((n - 1) * metrics.avg_latency_ms + latency_ms) / n
        
        # Regime match rate
        match_weight = 1.0 if regime_match else 0.0
        metrics.regime_match_rate = ((n - 1) * metrics.regime_match_rate + match_weight) / n
        
        # Auto-advance check
        if self.config.auto_advance and n % 100 == 0:
            passed, _ = self._check_criteria()
            if passed:
                self.advance_phase()
    
    def _advance_to_phase(self, phase: RolloutPhase) -> None:
        """Advance to specified phase."""
        old_phase = self._current_phase
        self._current_phase = phase
        
        # Initialize metrics for new phase
        symbols = self._get_phase_symbols(phase)
        self._phase_metrics[phase] = PhaseMetrics(
            phase=phase,
            symbols=symbols
        )
        
        # Enable symbols for this phase
        for symbol in symbols:
            self._enabled_symbols.add(symbol)
            logger.info(f"ENABLED: {symbol} (Phase: {phase.value})")
            if self._on_symbol_enabled:
                self._on_symbol_enabled(symbol)
        
        if self._on_phase_change:
            self._on_phase_change(old_phase, phase)
        
        self._persist_state()
        logger.info(f"Advanced to {phase.value}: symbols={symbols}")
    
    def _get_phase_symbols(self, phase: RolloutPhase) -> List[str]:
        """Get symbols for a given phase."""
        if phase == RolloutPhase.PHASE_1_BTC:
            return self.config.phase_1_symbols
        elif phase == RolloutPhase.PHASE_2_ETH:
            return self.config.phase_2_symbols
        elif phase == RolloutPhase.PHASE_3_MAJORS:
            return self.config.phase_3_symbols
        elif phase == RolloutPhase.PHASE_4_FULL:
            return self.config.phase_4_symbols
        return []
    
    def _get_next_phase(self) -> Optional[RolloutPhase]:
        """Get the next phase in sequence."""
        sequence = [
            RolloutPhase.NOT_STARTED,
            RolloutPhase.PHASE_1_BTC,
            RolloutPhase.PHASE_2_ETH,
            RolloutPhase.PHASE_3_MAJORS,
            RolloutPhase.PHASE_4_FULL,
            RolloutPhase.COMPLETED
        ]
        
        try:
            idx = sequence.index(self._current_phase)
            if idx < len(sequence) - 1:
                return sequence[idx + 1]
        except ValueError:
            pass
        
        return None
    
    def _check_criteria(self) -> tuple:
        """
        Check if current phase meets advancement criteria.
        
        Returns:
            (passed: bool, issues: List[str])
        """
        if self._current_phase not in self._phase_metrics:
            return False, ["No metrics for current phase"]
        
        metrics = self._phase_metrics[self._current_phase]
        criteria = self.config.criteria
        issues = []
        
        # Duration check
        if metrics.duration_hours < criteria.min_duration_hours:
            issues.append(f"Duration {metrics.duration_hours:.1f}h < {criteria.min_duration_hours}h")
        
        # Comparison count
        if metrics.comparisons < criteria.min_comparisons:
            issues.append(f"Comparisons {metrics.comparisons} < {criteria.min_comparisons}")
        
        # Signal diff
        if metrics.avg_signal_diff > criteria.max_signal_diff:
            issues.append(f"Signal diff {metrics.avg_signal_diff:.3f} > {criteria.max_signal_diff}")
        
        # Regime match
        if metrics.regime_match_rate < criteria.min_regime_match_rate:
            issues.append(f"Regime match {metrics.regime_match_rate:.1%} < {criteria.min_regime_match_rate:.1%}")
        
        # Latency
        if metrics.avg_latency_ms > criteria.max_latency_ms:
            issues.append(f"Latency {metrics.avg_latency_ms:.1f}ms > {criteria.max_latency_ms}ms")
        
        # Error rate
        if metrics.error_rate > criteria.max_error_rate:
            issues.append(f"Error rate {metrics.error_rate:.1%} > {criteria.max_error_rate:.1%}")
        
        # Discrepancy rate
        if metrics.discrepancy_rate > criteria.max_discrepancy_rate:
            issues.append(f"Discrepancy rate {metrics.discrepancy_rate:.1%} > {criteria.max_discrepancy_rate:.1%}")
        
        return len(issues) == 0, issues
    
    def _persist_state(self) -> None:
        """Persist rollout state to Redis."""
        if not self.redis:
            return
        
        try:
            state = {
                'current_phase': self._current_phase.value,
                'enabled_symbols': list(self._enabled_symbols),
                'last_update': datetime.utcnow().isoformat(),
                'rollback_count': len(self._rollback_reasons)
            }
            
            key = f"{self.config.redis_key_prefix}state"
            self.redis.set(key, json.dumps(state))
            
            # Persist phase metrics
            for phase, metrics in self._phase_metrics.items():
                metrics_key = f"{self.config.redis_key_prefix}metrics:{phase.value}"
                self.redis.hset(metrics_key, mapping={
                    'comparisons': metrics.comparisons,
                    'errors': metrics.errors,
                    'avg_signal_diff': metrics.avg_signal_diff,
                    'regime_match_rate': metrics.regime_match_rate,
                    'avg_latency_ms': metrics.avg_latency_ms,
                    'duration_hours': metrics.duration_hours
                })
                
        except Exception as e:
            logger.error(f"Failed to persist rollout state: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rollout status."""
        current_metrics = None
        if self._current_phase in self._phase_metrics:
            m = self._phase_metrics[self._current_phase]
            passed, issues = self._check_criteria()
            current_metrics = {
                'comparisons': m.comparisons,
                'duration_hours': round(m.duration_hours, 2),
                'avg_signal_diff': round(m.avg_signal_diff, 4),
                'regime_match_rate': round(m.regime_match_rate, 4),
                'avg_latency_ms': round(m.avg_latency_ms, 2),
                'error_rate': round(m.error_rate, 4),
                'discrepancy_rate': round(m.discrepancy_rate, 4),
                'criteria_met': passed,
                'issues': issues
            }
        
        return {
            'phase': self._current_phase.value,
            'enabled_symbols': list(self._enabled_symbols),
            'next_phase': self._get_next_phase().value if self._get_next_phase() else None,
            'metrics': current_metrics,
            'rollback_count': len(self._rollback_reasons),
            'recent_rollbacks': self._rollback_reasons[-5:]
        }


# Rollback criteria helper
class RollbackCriteria:
    """
    Define when automatic rollback should occur.
    
    CRITICAL (Immediate Rollback):
    - Error rate > 5%
    - Latency > 50ms sustained
    - Signal divergence > 50%
    
    WARNING (Alert + Manual Review):
    - Error rate > 1%
    - Latency > 25ms
    - Signal divergence > 20%
    """
    
    CRITICAL_ERROR_RATE = 0.05
    CRITICAL_LATENCY_MS = 50.0
    CRITICAL_SIGNAL_DIVERGENCE = 0.50
    
    WARNING_ERROR_RATE = 0.01
    WARNING_LATENCY_MS = 25.0
    WARNING_SIGNAL_DIVERGENCE = 0.20
    
    @classmethod
    def should_rollback(
        cls,
        error_rate: float,
        latency_ms: float,
        signal_divergence: float
    ) -> tuple:
        """
        Check if rollback is needed.
        
        Returns:
            (should_rollback: bool, severity: str, reason: str)
        """
        if error_rate > cls.CRITICAL_ERROR_RATE:
            return True, 'CRITICAL', f'Error rate {error_rate:.1%} > {cls.CRITICAL_ERROR_RATE:.1%}'
        
        if latency_ms > cls.CRITICAL_LATENCY_MS:
            return True, 'CRITICAL', f'Latency {latency_ms:.0f}ms > {cls.CRITICAL_LATENCY_MS}ms'
        
        if signal_divergence > cls.CRITICAL_SIGNAL_DIVERGENCE:
            return True, 'CRITICAL', f'Signal divergence {signal_divergence:.1%} > {cls.CRITICAL_SIGNAL_DIVERGENCE:.1%}'
        
        # Warnings (don't auto-rollback, but alert)
        if error_rate > cls.WARNING_ERROR_RATE:
            return False, 'WARNING', f'Error rate {error_rate:.1%} elevated'
        
        if latency_ms > cls.WARNING_LATENCY_MS:
            return False, 'WARNING', f'Latency {latency_ms:.0f}ms elevated'
        
        if signal_divergence > cls.WARNING_SIGNAL_DIVERGENCE:
            return False, 'WARNING', f'Signal divergence {signal_divergence:.1%} elevated'
        
        return False, 'OK', 'All metrics within bounds'
