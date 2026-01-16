"""
Strategy Retirement Manager

Not all strategies can be adapted—some should be retired gracefully.

Retirement triggers:
- Transfer ratio < 0.5 sustained 30 days
- True contribution becomes negative
- Causal mechanism invalidated (Layer 6 research)
- Repeated drift without successful adaptation

Process: Gradual position reduction, not abrupt halt.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetirementDecision:
    """Decision about whether to retire a strategy."""
    strategy_id: str
    should_retire: bool
    reason: str
    recommended_action: str
    wind_down_days: int
    urgency: str  # 'immediate', 'gradual', 'monitor'
    details: Dict = field(default_factory=dict)


@dataclass
class StrategyPerformanceRecord:
    """Record of strategy performance over time."""
    timestamp: float
    transfer_ratio: float
    true_contribution: float
    sharpe_ratio: float
    drawdown: float


class StrategyRetirementManager:
    """
    Manages graceful strategy retirement.

    Monitors performance and decides when to retire strategies.
    Implements gradual wind-down to avoid market impact.
    """

    # Default thresholds
    MIN_TRANSFER_RATIO = 0.50
    MIN_TRUE_CONTRIBUTION = 0.0
    SUSTAINED_DAYS = 30
    MAX_ADAPTATION_FAILURES = 3

    def __init__(
        self,
        min_tr: float = 0.50,
        min_tc: float = 0.0,
        sustained_days: int = 30,
        max_adaptation_failures: int = 3
    ):
        """
        Args:
            min_tr: Minimum acceptable transfer ratio
            min_tc: Minimum acceptable true contribution
            sustained_days: Days of poor performance before retirement
            max_adaptation_failures: Failed adaptations before retirement
        """
        self.min_tr = min_tr
        self.min_tc = min_tc
        self.sustained_days = sustained_days
        self.max_adaptation_failures = max_adaptation_failures

        # Strategy ID -> performance history
        self.performance_history: Dict[str, List[StrategyPerformanceRecord]] = {}

        # Strategy ID -> adaptation failure count
        self.adaptation_failures: Dict[str, int] = {}

        # Strategies currently winding down
        self.wind_down_queue: Dict[str, RetirementDecision] = {}

        # Fully retired strategies
        self.retired_strategies: Dict[str, RetirementDecision] = {}

    def record_performance(
        self,
        strategy_id: str,
        transfer_ratio: float,
        true_contribution: float,
        sharpe_ratio: float = 0.0,
        drawdown: float = 0.0
    ) -> None:
        """
        Record strategy performance for retirement analysis.

        Args:
            strategy_id: Strategy identifier
            transfer_ratio: Current transfer ratio
            true_contribution: Marginal contribution to portfolio
            sharpe_ratio: Current Sharpe ratio
            drawdown: Current drawdown
        """
        if strategy_id not in self.performance_history:
            self.performance_history[strategy_id] = []

        record = StrategyPerformanceRecord(
            timestamp=time.time(),
            transfer_ratio=transfer_ratio,
            true_contribution=true_contribution,
            sharpe_ratio=sharpe_ratio,
            drawdown=drawdown
        )

        self.performance_history[strategy_id].append(record)

        # Keep only recent history
        max_records = self.sustained_days * 24  # Hourly records for sustained_days
        if len(self.performance_history[strategy_id]) > max_records:
            self.performance_history[strategy_id] = self.performance_history[strategy_id][-max_records:]

    def record_adaptation_failure(self, strategy_id: str) -> None:
        """Record a failed adaptation attempt."""
        if strategy_id not in self.adaptation_failures:
            self.adaptation_failures[strategy_id] = 0
        self.adaptation_failures[strategy_id] += 1
        logger.warning(f"Strategy {strategy_id[:8]} adaptation failure #{self.adaptation_failures[strategy_id]}")

    def record_adaptation_success(self, strategy_id: str) -> None:
        """Record a successful adaptation (resets failure count)."""
        self.adaptation_failures[strategy_id] = 0

    def evaluate_retirement(
        self,
        strategy_id: str,
        current_tr: float,
        current_tc: float,
        causal_valid: bool = True
    ) -> RetirementDecision:
        """
        Evaluate if strategy should be retired.

        Args:
            strategy_id: Strategy to evaluate
            current_tr: Current transfer ratio
            current_tc: Current true contribution
            causal_valid: Whether causal mechanism is still valid

        Returns:
            RetirementDecision
        """
        # Record current performance
        self.record_performance(strategy_id, current_tr, current_tc)

        adaptation_failures = self.adaptation_failures.get(strategy_id, 0)

        # Check immediate triggers
        # 1. Negative true contribution (hurting portfolio)
        if current_tc < self.min_tc:
            return RetirementDecision(
                strategy_id=strategy_id,
                should_retire=True,
                reason=f"True contribution negative ({current_tc:.3f})",
                recommended_action="Immediate removal (hurting portfolio)",
                wind_down_days=3,
                urgency='immediate',
                details={'true_contribution': current_tc}
            )

        # 2. Causal mechanism invalidated
        if not causal_valid:
            return RetirementDecision(
                strategy_id=strategy_id,
                should_retire=True,
                reason="Causal mechanism invalidated by Layer 6 research",
                recommended_action="Research-driven retirement",
                wind_down_days=7,
                urgency='gradual',
                details={'causal_valid': False}
            )

        # 3. Too many adaptation failures
        if adaptation_failures >= self.max_adaptation_failures:
            return RetirementDecision(
                strategy_id=strategy_id,
                should_retire=True,
                reason=f"Failed {adaptation_failures} consecutive adaptations",
                recommended_action="Strategy fundamentally broken for current regime",
                wind_down_days=7,
                urgency='gradual',
                details={'adaptation_failures': adaptation_failures}
            )

        # 4. Sustained poor transfer ratio
        history = self.performance_history.get(strategy_id, [])
        if len(history) >= self.sustained_days:
            recent_tr = [h.transfer_ratio for h in history[-self.sustained_days:]]
            if all(tr < self.min_tr for tr in recent_tr):
                return RetirementDecision(
                    strategy_id=strategy_id,
                    should_retire=True,
                    reason=f"Transfer ratio < {self.min_tr} for {self.sustained_days} days",
                    recommended_action="Gradual wind-down",
                    wind_down_days=14,
                    urgency='gradual',
                    details={
                        'avg_recent_tr': sum(recent_tr) / len(recent_tr),
                        'min_recent_tr': min(recent_tr)
                    }
                )

        # No retirement needed
        return RetirementDecision(
            strategy_id=strategy_id,
            should_retire=False,
            reason="Strategy performing adequately",
            recommended_action="Continue monitoring",
            wind_down_days=0,
            urgency='monitor',
            details={
                'current_tr': current_tr,
                'current_tc': current_tc,
                'history_length': len(history)
            }
        )

    def start_wind_down(
        self,
        strategy_id: str,
        decision: RetirementDecision
    ) -> Dict:
        """
        Start wind-down process for a strategy.

        Args:
            strategy_id: Strategy to wind down
            decision: Retirement decision

        Returns:
            Wind-down schedule
        """
        self.wind_down_queue[strategy_id] = decision

        # Create wind-down schedule
        # Gradual position reduction over wind_down_days
        schedule = []
        total_days = decision.wind_down_days
        for day in range(total_days):
            # Linear reduction
            remaining_pct = 1.0 - ((day + 1) / total_days)
            schedule.append({
                'day': day + 1,
                'position_pct': max(0, remaining_pct),
                'action': 'reduce' if remaining_pct > 0 else 'close'
            })

        logger.info(f"Started wind-down for {strategy_id[:8]}: {total_days} days")

        return {
            'strategy_id': strategy_id,
            'reason': decision.reason,
            'schedule': schedule,
            'urgency': decision.urgency
        }

    def complete_retirement(self, strategy_id: str) -> bool:
        """
        Complete retirement of a strategy.

        Args:
            strategy_id: Strategy to retire

        Returns:
            True if successfully retired
        """
        if strategy_id in self.wind_down_queue:
            decision = self.wind_down_queue.pop(strategy_id)
            self.retired_strategies[strategy_id] = decision
            logger.info(f"Retired strategy {strategy_id[:8]}: {decision.reason}")
            return True
        return False

    def get_wind_down_status(self, strategy_id: str) -> Optional[Dict]:
        """Get wind-down status for a strategy."""
        if strategy_id in self.wind_down_queue:
            decision = self.wind_down_queue[strategy_id]
            return {
                'strategy_id': strategy_id,
                'reason': decision.reason,
                'urgency': decision.urgency,
                'wind_down_days': decision.wind_down_days,
                'status': 'winding_down'
            }
        elif strategy_id in self.retired_strategies:
            return {'status': 'retired'}
        return None

    def get_all_active_wind_downs(self) -> List[Dict]:
        """Get all strategies currently winding down."""
        return [
            {
                'strategy_id': sid,
                'reason': decision.reason,
                'urgency': decision.urgency
            }
            for sid, decision in self.wind_down_queue.items()
        ]

    def get_retirement_statistics(self) -> Dict:
        """Get retirement statistics."""
        retirement_reasons = {}
        for decision in self.retired_strategies.values():
            reason_key = decision.reason.split()[0]  # First word
            retirement_reasons[reason_key] = retirement_reasons.get(reason_key, 0) + 1

        return {
            'total_retired': len(self.retired_strategies),
            'currently_winding_down': len(self.wind_down_queue),
            'monitored_strategies': len(self.performance_history),
            'retirement_reasons': retirement_reasons
        }


class RetirementAnalyzer:
    """
    Analyzes retirement patterns to improve strategy selection.

    Tracks why strategies fail to inform future generation.
    """

    def __init__(self, retirement_manager: StrategyRetirementManager):
        self.manager = retirement_manager
        self.failure_patterns: List[Dict] = []

    def analyze_retirement(self, strategy_id: str) -> Dict:
        """
        Analyze why a strategy was retired.

        Returns:
            Analysis with patterns and recommendations
        """
        decision = self.manager.retired_strategies.get(strategy_id)
        if not decision:
            return {'error': 'Strategy not found in retired list'}

        history = self.manager.performance_history.get(strategy_id, [])

        # Analyze performance trajectory
        if history:
            trs = [h.transfer_ratio for h in history]
            tcs = [h.true_contribution for h in history]

            trajectory = {
                'tr_trend': 'declining' if len(trs) > 1 and trs[-1] < trs[0] else 'stable',
                'tc_trend': 'declining' if len(tcs) > 1 and tcs[-1] < tcs[0] else 'stable',
                'avg_tr': sum(trs) / len(trs),
                'avg_tc': sum(tcs) / len(tcs),
                'volatility_tr': (max(trs) - min(trs)) if trs else 0
            }
        else:
            trajectory = {}

        analysis = {
            'strategy_id': strategy_id,
            'reason': decision.reason,
            'urgency': decision.urgency,
            'trajectory': trajectory,
            'adaptation_failures': self.manager.adaptation_failures.get(strategy_id, 0)
        }

        self.failure_patterns.append(analysis)
        return analysis

    def get_common_failure_patterns(self) -> List[Dict]:
        """Get most common failure patterns."""
        if not self.failure_patterns:
            return []

        # Group by reason
        pattern_counts = {}
        for pattern in self.failure_patterns:
            reason = pattern['reason']
            if reason not in pattern_counts:
                pattern_counts[reason] = {'count': 0, 'examples': []}
            pattern_counts[reason]['count'] += 1
            pattern_counts[reason]['examples'].append(pattern['strategy_id'][:8])

        return sorted(
            [{'reason': k, **v} for k, v in pattern_counts.items()],
            key=lambda x: x['count'],
            reverse=True
        )
