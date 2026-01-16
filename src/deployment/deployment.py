"""
Deployment Manager

Manages strategy progression from shadow to live trading.

Protocol:
1. Shadow trading for 21+ days
2. Transfer ratio > 0.7
3. Epistemic uncertainty < 0.10 and converging
4. Initial sizing: 25-50% of target
5. Scale up if TR > 0.75 sustained for 30 days
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import numpy as np
import logging

from ..core.genome import StrategyGenome
from .shadow import ShadowEnvironment, ShadowPerformance, ShadowPerformanceAnalyzer
from .uncertainty import EpistemicUncertaintyGate, UncertaintyResult
from .transfer import TransferRatioConfidence, TransferPrediction

logger = logging.getLogger(__name__)


@dataclass
class DeploymentDecision:
    """Result of deployment evaluation."""
    strategy_id: str
    approved: bool
    initial_position_pct: float
    confidence: float
    expected_sharpe: float
    conditions: List[str]  # What must remain true for continued deployment
    monitoring_alerts: List[str]  # What triggers review
    rejection_reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'strategy_id': self.strategy_id,
            'approved': self.approved,
            'initial_position_pct': self.initial_position_pct,
            'confidence': self.confidence,
            'expected_sharpe': self.expected_sharpe,
            'conditions': self.conditions,
            'monitoring_alerts': self.monitoring_alerts,
            'rejection_reason': self.rejection_reason
        }


@dataclass
class DeployedStrategy:
    """Tracking record for a deployed strategy."""
    strategy: StrategyGenome
    deployment_date: datetime
    initial_position_pct: float
    current_position_pct: float
    backtest_sharpe: float
    expected_live_sharpe: float
    shadow_performance: ShadowPerformance
    status: str  # 'active', 'scaling_up', 'scaling_down', 'suspended', 'retired'
    performance_history: List[Dict] = field(default_factory=list)


class DeploymentManager:
    """
    Manages strategy progression from shadow to live.

    Responsibilities:
    1. Run shadow trading evaluation
    2. Check uncertainty and transfer confidence
    3. Make deployment decisions
    4. Track deployed strategies
    5. Scale positions up/down based on performance
    """

    # Deployment thresholds
    MIN_SHADOW_DAYS = 21
    MIN_TRANSFER_RATIO = 0.70
    MAX_UNCERTAINTY = 0.10
    TARGET_SHARPE = 1.5
    SCALE_UP_THRESHOLD = 0.75  # TR threshold for scaling up
    SCALE_DOWN_THRESHOLD = 0.60  # TR threshold for scaling down

    def __init__(
        self,
        shadow_env: Optional[ShadowEnvironment] = None,
        uncertainty_gate: Optional[EpistemicUncertaintyGate] = None,
        transfer_model: Optional[TransferRatioConfidence] = None
    ):
        self.shadow = shadow_env or ShadowEnvironment()
        self.uncertainty = uncertainty_gate or EpistemicUncertaintyGate()
        self.transfer = transfer_model or TransferRatioConfidence()
        self.analyzer = ShadowPerformanceAnalyzer(
            min_transfer_ratio=self.MIN_TRANSFER_RATIO
        )

        self.deployed_strategies: Dict[str, DeployedStrategy] = {}
        self.deployment_history: List[DeploymentDecision] = []

    async def evaluate_for_deployment(
        self,
        strategy: StrategyGenome,
        backtest_sharpe: float,
        shadow_days: int = 21
    ) -> DeploymentDecision:
        """
        Complete deployment evaluation.

        Steps:
        1. Run shadow trading
        2. Check transfer ratio
        3. Check epistemic uncertainty
        4. Calculate position sizing
        5. Make deployment decision

        Args:
            strategy: Strategy to evaluate
            backtest_sharpe: Strategy's backtest Sharpe ratio
            shadow_days: Days to run shadow trading

        Returns:
            DeploymentDecision
        """
        logger.info(f"Evaluating strategy {strategy.id[:8]} for deployment")

        # Step 1: Run shadow trading
        self.shadow.backtest_sharpe = backtest_sharpe
        self.shadow.reset()

        shadow_perf = await self.shadow.run_shadow(
            strategy=strategy,
            duration_days=shadow_days
        )

        # Analyze shadow performance
        shadow_analysis = self.analyzer.analyze(shadow_perf)

        # Step 2: Check transfer ratio
        if shadow_perf.transfer_ratio < self.MIN_TRANSFER_RATIO:
            return DeploymentDecision(
                strategy_id=strategy.id,
                approved=False,
                initial_position_pct=0,
                confidence=0,
                expected_sharpe=0,
                conditions=[],
                monitoring_alerts=[],
                rejection_reason=f"Transfer ratio {shadow_perf.transfer_ratio:.2f} < {self.MIN_TRANSFER_RATIO}",
                details={'shadow_performance': shadow_perf.to_dict()}
            )

        # Step 3: Check uncertainty
        market_data = np.random.randn(100, 60)  # Mock recent market data
        uncertainty_result = self.uncertainty.should_deploy(strategy, market_data)

        if not uncertainty_result.should_deploy:
            return DeploymentDecision(
                strategy_id=strategy.id,
                approved=False,
                initial_position_pct=0,
                confidence=0,
                expected_sharpe=0,
                conditions=[],
                monitoring_alerts=[],
                rejection_reason=f"Epistemic uncertainty {uncertainty_result.uncertainty_score:.3f} too high or not converging",
                details={
                    'shadow_performance': shadow_perf.to_dict(),
                    'uncertainty': uncertainty_result.details
                }
            )

        # Step 4: Calculate position sizing
        transfer_prediction = self.transfer.get_prediction(
            backtest_sharpe=backtest_sharpe,
            threshold=self.TARGET_SHARPE,
            target_position_pct=0.05  # 5% base position
        )

        confidence = transfer_prediction.probability_above_threshold
        position_size = transfer_prediction.recommended_position_pct

        # Step 5: Build deployment decision
        decision = DeploymentDecision(
            strategy_id=strategy.id,
            approved=True,
            initial_position_pct=position_size,
            confidence=confidence,
            expected_sharpe=transfer_prediction.expected_live_sharpe,
            conditions=[
                f"Transfer ratio > {self.SCALE_DOWN_THRESHOLD}",
                f"Drawdown < 15%",
                f"Sharpe > {transfer_prediction.expected_live_sharpe * 0.7:.2f}"
            ],
            monitoring_alerts=[
                "Daily TR check",
                "Weekly regime consistency check",
                "Monthly full review"
            ],
            details={
                'shadow_performance': shadow_perf.to_dict(),
                'uncertainty': uncertainty_result.details,
                'transfer_prediction': {
                    'expected_sharpe': transfer_prediction.expected_live_sharpe,
                    'confidence_interval': transfer_prediction.confidence_interval,
                    'risk_adjusted_size': transfer_prediction.risk_adjusted_size
                }
            }
        )

        # Record deployment
        self.deployment_history.append(decision)

        if decision.approved:
            self._register_deployment(
                strategy=strategy,
                backtest_sharpe=backtest_sharpe,
                shadow_perf=shadow_perf,
                decision=decision
            )

        return decision

    def _register_deployment(
        self,
        strategy: StrategyGenome,
        backtest_sharpe: float,
        shadow_perf: ShadowPerformance,
        decision: DeploymentDecision
    ) -> None:
        """Register a newly deployed strategy."""
        deployed = DeployedStrategy(
            strategy=strategy,
            deployment_date=datetime.now(),
            initial_position_pct=decision.initial_position_pct,
            current_position_pct=decision.initial_position_pct,
            backtest_sharpe=backtest_sharpe,
            expected_live_sharpe=decision.expected_sharpe,
            shadow_performance=shadow_perf,
            status='active'
        )
        self.deployed_strategies[strategy.id] = deployed
        logger.info(f"Deployed strategy {strategy.id[:8]} with {decision.initial_position_pct:.1%} position")

    def update_deployed_strategy(
        self,
        strategy_id: str,
        live_sharpe: float,
        live_drawdown: float,
        transfer_ratio: float
    ) -> Dict[str, Any]:
        """
        Update a deployed strategy based on live performance.

        Args:
            strategy_id: ID of deployed strategy
            live_sharpe: Current live Sharpe ratio
            live_drawdown: Current drawdown
            transfer_ratio: Current transfer ratio

        Returns:
            Action taken (scale_up, scale_down, suspend, none)
        """
        if strategy_id not in self.deployed_strategies:
            return {'action': 'not_found'}

        deployed = self.deployed_strategies[strategy_id]

        # Record performance
        deployed.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'live_sharpe': live_sharpe,
            'live_drawdown': live_drawdown,
            'transfer_ratio': transfer_ratio
        })

        action = {'action': 'none'}

        # Check for scaling/suspension conditions
        if transfer_ratio >= self.SCALE_UP_THRESHOLD and deployed.status == 'active':
            # Consider scaling up
            if len(deployed.performance_history) >= 30:  # 30 days of good performance
                recent_tr = [h['transfer_ratio'] for h in deployed.performance_history[-30:]]
                if min(recent_tr) >= self.SCALE_UP_THRESHOLD:
                    deployed.current_position_pct = min(
                        deployed.current_position_pct * 1.5,
                        deployed.initial_position_pct * 2  # Max 2x initial
                    )
                    deployed.status = 'scaling_up'
                    action = {
                        'action': 'scale_up',
                        'new_position': deployed.current_position_pct
                    }

        elif transfer_ratio < self.SCALE_DOWN_THRESHOLD:
            # Scale down
            deployed.current_position_pct = max(
                deployed.current_position_pct * 0.5,
                deployed.initial_position_pct * 0.25  # Min 25% of initial
            )
            deployed.status = 'scaling_down'
            action = {
                'action': 'scale_down',
                'new_position': deployed.current_position_pct,
                'reason': f'TR {transfer_ratio:.2f} < {self.SCALE_DOWN_THRESHOLD}'
            }

        elif transfer_ratio < 0.5 or live_drawdown > 0.20:
            # Suspend
            deployed.status = 'suspended'
            deployed.current_position_pct = 0
            action = {
                'action': 'suspend',
                'reason': f'TR {transfer_ratio:.2f} critical' if transfer_ratio < 0.5
                         else f'Drawdown {live_drawdown:.1%} exceeded limit'
            }

        return action

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get summary of all deployed strategies."""
        active = [s for s in self.deployed_strategies.values() if s.status == 'active']
        suspended = [s for s in self.deployed_strategies.values() if s.status == 'suspended']

        total_allocation = sum(s.current_position_pct for s in active)

        return {
            'total_strategies': len(self.deployed_strategies),
            'active_strategies': len(active),
            'suspended_strategies': len(suspended),
            'total_allocation': total_allocation,
            'strategies': [
                {
                    'id': s.strategy.id[:8],
                    'status': s.status,
                    'position_pct': s.current_position_pct,
                    'expected_sharpe': s.expected_live_sharpe,
                    'days_deployed': (datetime.now() - s.deployment_date).days
                }
                for s in self.deployed_strategies.values()
            ]
        }

    def retire_strategy(
        self,
        strategy_id: str,
        reason: str = "Manual retirement"
    ) -> bool:
        """Retire a deployed strategy."""
        if strategy_id not in self.deployed_strategies:
            return False

        deployed = self.deployed_strategies[strategy_id]
        deployed.status = 'retired'
        deployed.current_position_pct = 0

        logger.info(f"Retired strategy {strategy_id[:8]}: {reason}")
        return True


class DeploymentScheduler:
    """
    Schedules and manages deployment pipeline execution.

    Handles:
    - Queuing strategies for shadow testing
    - Parallel shadow execution
    - Deployment decision batching
    """

    def __init__(
        self,
        manager: DeploymentManager,
        max_concurrent_shadow: int = 5
    ):
        self.manager = manager
        self.max_concurrent = max_concurrent_shadow

        self.shadow_queue: List[Tuple[StrategyGenome, float]] = []  # (strategy, backtest_sharpe)
        self.in_shadow: Dict[str, datetime] = {}  # strategy_id -> start_time

    def queue_for_shadow(
        self,
        strategy: StrategyGenome,
        backtest_sharpe: float
    ) -> int:
        """
        Queue a strategy for shadow testing.

        Returns:
            Queue position
        """
        self.shadow_queue.append((strategy, backtest_sharpe))
        return len(self.shadow_queue)

    async def process_queue(self) -> List[DeploymentDecision]:
        """
        Process queued strategies through shadow testing.

        Returns:
            List of deployment decisions for completed shadow tests
        """
        import asyncio

        decisions = []

        # Start new shadow tests up to max concurrent
        while self.shadow_queue and len(self.in_shadow) < self.max_concurrent:
            strategy, backtest_sharpe = self.shadow_queue.pop(0)
            self.in_shadow[strategy.id] = datetime.now()

            # Run shadow evaluation
            decision = await self.manager.evaluate_for_deployment(
                strategy=strategy,
                backtest_sharpe=backtest_sharpe,
                shadow_days=21
            )

            decisions.append(decision)
            del self.in_shadow[strategy.id]

        return decisions

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            'queued': len(self.shadow_queue),
            'in_shadow': len(self.in_shadow),
            'max_concurrent': self.max_concurrent,
            'shadow_strategy_ids': list(self.in_shadow.keys())
        }
