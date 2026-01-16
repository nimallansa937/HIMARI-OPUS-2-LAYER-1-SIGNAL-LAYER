"""
Automatic Domain Randomization (ADR)

Automatically expands domain randomization during shadow trading
to improve strategy robustness. Progressively increases noise
in slippage, latency, and volatility until strategy fails.

This addresses Gap #5 from the gap analysis: Add adaptive
domain randomization for more robust deployment.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DomainType(Enum):
    """Types of domains to randomize."""
    SLIPPAGE = "slippage"       # Order execution slippage
    LATENCY = "latency"         # Network/execution latency
    VOLATILITY = "volatility"   # Market volatility scaling
    SPREAD = "spread"           # Bid-ask spread
    LIQUIDITY = "liquidity"     # Available liquidity
    FEES = "fees"               # Trading fees


@dataclass
class DomainRange:
    """Defines the range for a domain parameter."""
    domain_type: DomainType
    min_value: float
    max_value: float
    current_value: float
    step_size: float = 0.1  # How much to expand per iteration

    # Tracking
    expansion_count: int = 0
    failure_count: int = 0

    def expand(self) -> bool:
        """
        Expand the domain range by step_size.

        Returns:
            True if expansion was possible
        """
        new_value = self.current_value + self.step_size
        if new_value <= self.max_value:
            self.current_value = new_value
            self.expansion_count += 1
            return True
        return False

    def contract(self) -> bool:
        """
        Contract the domain range.

        Returns:
            True if contraction was possible
        """
        new_value = self.current_value - self.step_size
        if new_value >= self.min_value:
            self.current_value = new_value
            return True
        return False

    def sample(self) -> float:
        """Sample a value from the current range."""
        if self.current_value == 0:
            return 0.0
        return np.random.uniform(0, self.current_value)


@dataclass
class ADRConfig:
    """Configuration for Automatic Domain Randomization."""
    # Initial ranges
    initial_slippage: float = 0.0001  # 1 bps
    initial_latency_ms: float = 10.0
    initial_volatility_scale: float = 1.0
    initial_spread: float = 0.0001
    initial_liquidity_factor: float = 1.0
    initial_fee_multiplier: float = 1.0

    # Maximum ranges
    max_slippage: float = 0.005  # 50 bps
    max_latency_ms: float = 500.0
    max_volatility_scale: float = 3.0
    max_spread: float = 0.002
    max_liquidity_factor: float = 0.1  # 10% of normal
    max_fee_multiplier: float = 3.0

    # Expansion parameters
    expansion_threshold: float = 0.8  # Performance threshold to expand
    contraction_threshold: float = 0.5  # Performance threshold to contract
    step_size_pct: float = 0.1  # 10% step per iteration

    # Training parameters
    episodes_per_iteration: int = 10
    min_iterations: int = 10
    max_iterations: int = 100


@dataclass
class ADRState:
    """Current state of ADR training."""
    domains: Dict[DomainType, DomainRange]
    iteration: int = 0
    total_episodes: int = 0
    performance_history: List[float] = field(default_factory=list)
    expansion_history: List[Tuple[datetime, DomainType, float]] = field(default_factory=list)

    def get_current_ranges(self) -> Dict[str, float]:
        """Get current domain ranges as dict."""
        return {
            d.value: self.domains[d].current_value
            for d in self.domains
        }


@dataclass
class DomainSample:
    """A sample from the domain distribution."""
    slippage: float
    latency_ms: float
    volatility_scale: float
    spread: float
    liquidity_factor: float
    fee_multiplier: float

    def to_dict(self) -> Dict[str, float]:
        return {
            'slippage': self.slippage,
            'latency_ms': self.latency_ms,
            'volatility_scale': self.volatility_scale,
            'spread': self.spread,
            'liquidity_factor': self.liquidity_factor,
            'fee_multiplier': self.fee_multiplier
        }


@dataclass
class ADRResult:
    """Result from ADR training."""
    final_ranges: Dict[str, float]
    robustness_score: float
    iterations_completed: int
    performance_at_final: float
    domains_at_max: List[str]
    recommended_for_deployment: bool


class AutomaticDomainRandomization:
    """
    Implements Automatic Domain Randomization for strategy robustness.

    Progressively expands domain randomization during shadow trading:
    1. Start with minimal randomization
    2. If strategy performs well, expand ranges
    3. If strategy fails, contract or stop
    4. Final ranges indicate robustness requirements

    A strategy that survives wide domain ranges is more robust
    and likely to transfer well to live trading.
    """

    def __init__(
        self,
        config: Optional[ADRConfig] = None,
        evaluator: Optional[callable] = None
    ):
        """
        Initialize ADR.

        Args:
            config: ADR configuration
            evaluator: Function(strategy, domain_sample) -> performance
        """
        self.config = config or ADRConfig()
        self.evaluator = evaluator or self._mock_evaluator
        self.state: Optional[ADRState] = None

    def initialize(self) -> ADRState:
        """Initialize ADR state with starting ranges."""
        config = self.config

        domains = {
            DomainType.SLIPPAGE: DomainRange(
                domain_type=DomainType.SLIPPAGE,
                min_value=0,
                max_value=config.max_slippage,
                current_value=config.initial_slippage,
                step_size=config.max_slippage * config.step_size_pct
            ),
            DomainType.LATENCY: DomainRange(
                domain_type=DomainType.LATENCY,
                min_value=0,
                max_value=config.max_latency_ms,
                current_value=config.initial_latency_ms,
                step_size=config.max_latency_ms * config.step_size_pct
            ),
            DomainType.VOLATILITY: DomainRange(
                domain_type=DomainType.VOLATILITY,
                min_value=0.5,
                max_value=config.max_volatility_scale,
                current_value=config.initial_volatility_scale,
                step_size=(config.max_volatility_scale - 0.5) * config.step_size_pct
            ),
            DomainType.SPREAD: DomainRange(
                domain_type=DomainType.SPREAD,
                min_value=0,
                max_value=config.max_spread,
                current_value=config.initial_spread,
                step_size=config.max_spread * config.step_size_pct
            ),
            DomainType.LIQUIDITY: DomainRange(
                domain_type=DomainType.LIQUIDITY,
                min_value=config.max_liquidity_factor,
                max_value=1.0,
                current_value=config.initial_liquidity_factor,
                step_size=(1.0 - config.max_liquidity_factor) * config.step_size_pct
            ),
            DomainType.FEES: DomainRange(
                domain_type=DomainType.FEES,
                min_value=1.0,
                max_value=config.max_fee_multiplier,
                current_value=config.initial_fee_multiplier,
                step_size=(config.max_fee_multiplier - 1.0) * config.step_size_pct
            )
        }

        self.state = ADRState(domains=domains)
        logger.info("Initialized ADR with starting ranges")
        return self.state

    def sample_domain(self) -> DomainSample:
        """Sample from current domain distribution."""
        if self.state is None:
            self.initialize()

        return DomainSample(
            slippage=self.state.domains[DomainType.SLIPPAGE].sample(),
            latency_ms=self.state.domains[DomainType.LATENCY].sample(),
            volatility_scale=self._sample_volatility(),
            spread=self.state.domains[DomainType.SPREAD].sample(),
            liquidity_factor=self._sample_liquidity(),
            fee_multiplier=self._sample_fees()
        )

    def _sample_volatility(self) -> float:
        """Sample volatility scale around 1.0."""
        domain = self.state.domains[DomainType.VOLATILITY]
        # Sample from range centered at 1.0
        deviation = (domain.current_value - 1.0) * np.random.uniform(-1, 1)
        return 1.0 + deviation

    def _sample_liquidity(self) -> float:
        """Sample liquidity factor (lower is worse)."""
        domain = self.state.domains[DomainType.LIQUIDITY]
        return np.random.uniform(domain.current_value, 1.0)

    def _sample_fees(self) -> float:
        """Sample fee multiplier."""
        domain = self.state.domains[DomainType.FEES]
        return np.random.uniform(1.0, domain.current_value)

    def run_iteration(self, strategy) -> Tuple[float, bool]:
        """
        Run one ADR iteration.

        Args:
            strategy: Strategy to evaluate

        Returns:
            (average_performance, should_continue)
        """
        if self.state is None:
            self.initialize()

        self.state.iteration += 1
        performances = []

        # Run episodes with current domain
        for _ in range(self.config.episodes_per_iteration):
            domain_sample = self.sample_domain()
            performance = self.evaluator(strategy, domain_sample)
            performances.append(performance)
            self.state.total_episodes += 1

        avg_performance = np.mean(performances)
        self.state.performance_history.append(avg_performance)

        # Decide expansion/contraction
        if avg_performance >= self.config.expansion_threshold:
            # Strategy is robust - expand domains
            self._expand_domains()
        elif avg_performance < self.config.contraction_threshold:
            # Strategy is struggling - contract or stop
            if not self._contract_domains():
                return avg_performance, False

        # Check termination
        should_continue = (
            self.state.iteration < self.config.max_iterations and
            not self._all_domains_at_max()
        )

        return avg_performance, should_continue

    def _expand_domains(self) -> None:
        """Expand all domain ranges."""
        for domain_type, domain in self.state.domains.items():
            if domain.expand():
                self.state.expansion_history.append(
                    (datetime.now(), domain_type, domain.current_value)
                )
                logger.debug(f"Expanded {domain_type.value} to {domain.current_value:.4f}")

    def _contract_domains(self) -> bool:
        """
        Contract domain ranges.

        Returns:
            True if any domain was contracted
        """
        any_contracted = False
        for domain in self.state.domains.values():
            if domain.contract():
                any_contracted = True
                domain.failure_count += 1

        return any_contracted

    def _all_domains_at_max(self) -> bool:
        """Check if all domains are at maximum."""
        for domain in self.state.domains.values():
            if domain.current_value < domain.max_value:
                return False
        return True

    def train(self, strategy) -> ADRResult:
        """
        Run full ADR training.

        Args:
            strategy: Strategy to train

        Returns:
            ADRResult with final robustness assessment
        """
        self.initialize()

        logger.info(f"Starting ADR training for strategy {strategy.id[:8]}")

        while True:
            performance, should_continue = self.run_iteration(strategy)

            if self.state.iteration % 10 == 0:
                logger.info(
                    f"ADR iteration {self.state.iteration}: "
                    f"performance={performance:.3f}, "
                    f"ranges={self.state.get_current_ranges()}"
                )

            if not should_continue:
                break

            if self.state.iteration >= self.config.max_iterations:
                break

        return self._compute_result()

    def _compute_result(self) -> ADRResult:
        """Compute final ADR result."""
        final_ranges = self.state.get_current_ranges()

        # Compute robustness score (how much of max range was achieved)
        range_ratios = []
        for domain_type, domain in self.state.domains.items():
            if domain.max_value > domain.min_value:
                ratio = (domain.current_value - domain.min_value) / (domain.max_value - domain.min_value)
                range_ratios.append(ratio)

        robustness_score = np.mean(range_ratios) if range_ratios else 0.0

        # Check which domains reached max
        domains_at_max = [
            d.value for d, domain in self.state.domains.items()
            if domain.current_value >= domain.max_value * 0.95
        ]

        # Final performance
        final_performance = (
            self.state.performance_history[-1]
            if self.state.performance_history else 0.0
        )

        # Recommendation
        recommended = (
            robustness_score >= 0.5 and
            final_performance >= self.config.contraction_threshold and
            self.state.iteration >= self.config.min_iterations
        )

        return ADRResult(
            final_ranges=final_ranges,
            robustness_score=robustness_score,
            iterations_completed=self.state.iteration,
            performance_at_final=final_performance,
            domains_at_max=domains_at_max,
            recommended_for_deployment=recommended
        )

    def _mock_evaluator(self, strategy, domain_sample: DomainSample) -> float:
        """Mock evaluator for testing."""
        # Simulate performance degradation with more extreme domains
        base_performance = 0.9

        # Penalties for each domain
        slippage_penalty = domain_sample.slippage * 10
        latency_penalty = domain_sample.latency_ms / 500
        volatility_penalty = abs(domain_sample.volatility_scale - 1.0) * 0.3
        spread_penalty = domain_sample.spread * 10
        liquidity_penalty = (1 - domain_sample.liquidity_factor) * 0.5
        fee_penalty = (domain_sample.fee_multiplier - 1.0) * 0.2

        total_penalty = (
            slippage_penalty + latency_penalty + volatility_penalty +
            spread_penalty + liquidity_penalty + fee_penalty
        )

        # Add noise
        noise = np.random.normal(0, 0.05)

        performance = base_performance - total_penalty + noise
        return np.clip(performance, 0, 1)

    def apply_domain_to_simulation(
        self,
        simulation_params: Dict[str, Any],
        domain_sample: DomainSample
    ) -> Dict[str, Any]:
        """
        Apply domain sample to simulation parameters.

        Args:
            simulation_params: Base simulation parameters
            domain_sample: Domain randomization sample

        Returns:
            Modified simulation parameters
        """
        modified = simulation_params.copy()

        # Apply slippage
        modified['slippage_bps'] = modified.get('slippage_bps', 0) + domain_sample.slippage * 10000

        # Apply latency
        modified['latency_ms'] = max(
            modified.get('latency_ms', 0),
            domain_sample.latency_ms
        )

        # Apply volatility scaling
        if 'volatility' in modified:
            modified['volatility'] *= domain_sample.volatility_scale

        # Apply spread
        modified['spread_bps'] = modified.get('spread_bps', 0) + domain_sample.spread * 10000

        # Apply liquidity factor
        if 'available_liquidity' in modified:
            modified['available_liquidity'] *= domain_sample.liquidity_factor

        # Apply fee multiplier
        if 'fee_rate' in modified:
            modified['fee_rate'] *= domain_sample.fee_multiplier

        return modified

    def get_recommended_live_params(self) -> Dict[str, float]:
        """
        Get recommended parameters for live trading.

        Based on final domain ranges, suggests conservative
        settings that the strategy has proven robust to.
        """
        if self.state is None:
            return {}

        ranges = self.state.get_current_ranges()

        # Use 80% of proven range as safety margin
        safety_factor = 0.8

        return {
            'expected_slippage': ranges['slippage'] * safety_factor,
            'max_latency_tolerance_ms': ranges['latency'] * safety_factor,
            'volatility_range': (
                1.0 - (ranges['volatility'] - 1.0) * safety_factor,
                1.0 + (ranges['volatility'] - 1.0) * safety_factor
            ),
            'expected_spread': ranges['spread'] * safety_factor,
            'min_liquidity_factor': ranges['liquidity'] / safety_factor,
            'max_fee_multiplier': 1.0 + (ranges['fees'] - 1.0) * safety_factor
        }
