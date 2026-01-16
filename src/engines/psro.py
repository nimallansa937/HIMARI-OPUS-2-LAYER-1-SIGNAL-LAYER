"""
Policy Space Response Oracles (PSRO) for Strategy Diversity

Maintains a diverse strategy population by computing meta-Nash equilibrium
and generating best-response strategies. Prevents population collapse
to a single dominant strategy type.

This addresses Gap #2 from the gap analysis: Replace simple cosine
similarity diversity with game-theoretic diversity maintenance.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Callable
import numpy as np
from datetime import datetime

from ..core.genome import StrategyGenome, generate_random_strategy

logger = logging.getLogger(__name__)


@dataclass
class StrategyProfile:
    """A strategy with its performance data."""
    strategy: StrategyGenome
    performance_vector: np.ndarray  # Performance against all strategies
    weight: float = 1.0  # Weight in meta-Nash distribution
    generation: int = 0


@dataclass
class PSROConfig:
    """Configuration for PSRO diversity manager."""
    # Population parameters
    max_population_size: int = 50
    min_population_size: int = 10

    # Nash computation
    nash_iterations: int = 100
    nash_tolerance: float = 1e-6

    # Best response
    br_samples: int = 20  # Strategies to sample when computing BR
    br_evaluation_depth: int = 5  # How many opponents to evaluate against

    # Diversity parameters
    diversity_weight: float = 0.3  # Weight for diversity vs performance
    min_strategy_weight: float = 0.01  # Minimum weight to keep strategy

    # Pruning
    prune_threshold: float = 0.005  # Remove strategies below this weight
    prune_interval: int = 10  # Generations between pruning


@dataclass
class PSROResult:
    """Result from PSRO iteration."""
    new_strategies: List[StrategyGenome]
    nash_weights: Dict[str, float]
    exploitability: float
    population_diversity: float
    generation: int


class PayoffMatrix:
    """
    Manages the payoff matrix between strategies.

    Tracks pairwise performance (strategy A vs strategy B)
    for meta-Nash computation.
    """

    def __init__(self, strategies: List[StrategyProfile]):
        """Initialize payoff matrix."""
        self.strategies = strategies
        self.n = len(strategies)
        self._matrix: Optional[np.ndarray] = None

    @property
    def matrix(self) -> np.ndarray:
        """Get or compute payoff matrix."""
        if self._matrix is None:
            self._compute_matrix()
        return self._matrix

    def _compute_matrix(self) -> None:
        """Compute payoff matrix from strategy performance vectors."""
        self._matrix = np.zeros((self.n, self.n))

        for i, profile_i in enumerate(self.strategies):
            for j, profile_j in enumerate(self.strategies):
                if i == j:
                    self._matrix[i, j] = 0  # Self-play is neutral
                else:
                    # Use performance vectors to estimate payoff
                    # Higher correlation means similar performance patterns
                    perf_i = profile_i.performance_vector
                    perf_j = profile_j.performance_vector

                    if len(perf_i) > 0 and len(perf_j) > 0:
                        # Strategy i's advantage over j
                        # Positive means i is better in diverse conditions
                        min_len = min(len(perf_i), len(perf_j))
                        diff = perf_i[:min_len] - perf_j[:min_len]
                        self._matrix[i, j] = np.mean(diff)
                    else:
                        self._matrix[i, j] = 0

    def update_payoff(self, i: int, j: int, value: float) -> None:
        """Update a specific payoff value."""
        if self._matrix is None:
            self._compute_matrix()
        self._matrix[i, j] = value

    def add_strategy(self, profile: StrategyProfile) -> None:
        """Add a new strategy to the matrix."""
        self.strategies.append(profile)
        self.n += 1
        self._matrix = None  # Invalidate cached matrix


class NashSolver:
    """
    Computes Nash equilibrium for the strategy population.

    Uses fictitious play / iterative best response to find
    approximate Nash equilibrium mixture.
    """

    def __init__(self, config: PSROConfig):
        self.config = config

    def solve(self, payoff_matrix: np.ndarray) -> np.ndarray:
        """
        Find Nash equilibrium weights.

        Args:
            payoff_matrix: n x n matrix of payoffs

        Returns:
            n-dimensional probability distribution (Nash weights)
        """
        n = payoff_matrix.shape[0]

        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([1.0])

        # Initialize uniform weights
        weights = np.ones(n) / n

        # Fictitious play iteration
        for iteration in range(self.config.nash_iterations):
            # Compute expected payoff for each strategy against current mixture
            expected_payoffs = payoff_matrix @ weights

            # Update weights toward best response
            best_response = np.zeros(n)
            best_idx = np.argmax(expected_payoffs)
            best_response[best_idx] = 1.0

            # Smooth update
            alpha = 2.0 / (iteration + 2)  # Decreasing step size
            new_weights = (1 - alpha) * weights + alpha * best_response

            # Check convergence
            if np.max(np.abs(new_weights - weights)) < self.config.nash_tolerance:
                break

            weights = new_weights

        # Normalize
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)

        return weights

    def compute_exploitability(
        self,
        payoff_matrix: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """
        Compute exploitability of the Nash mixture.

        Exploitability = max payoff any pure strategy can get
        against the mixture - value of the game.
        """
        if len(weights) == 0:
            return 0.0

        # Value of the game (expected payoff of mixture vs mixture)
        game_value = weights @ payoff_matrix @ weights

        # Best response payoff
        expected_vs_mixture = payoff_matrix @ weights
        best_response_payoff = np.max(expected_vs_mixture)

        return best_response_payoff - game_value


class PSRODiversityManager:
    """
    PSRO-based diversity manager for strategy populations.

    Maintains diverse strategy population by:
    1. Computing meta-Nash equilibrium over strategies
    2. Generating best-response strategies to current mixture
    3. Pruning low-weight strategies

    This ensures the population contains complementary strategies
    rather than converging to a monoculture.
    """

    def __init__(
        self,
        config: PSROConfig,
        strategy_generator: Optional[Callable[[], StrategyGenome]] = None,
        evaluator: Optional[Callable[[StrategyGenome, StrategyGenome], float]] = None
    ):
        """
        Initialize PSRO diversity manager.

        Args:
            config: PSRO configuration
            strategy_generator: Function to generate new candidate strategies
            evaluator: Function to evaluate strategy A vs strategy B (returns A's payoff)
        """
        self.config = config
        self.generate_strategy = strategy_generator or (lambda: generate_random_strategy(4))
        self.evaluator = evaluator or self._default_evaluator

        self.nash_solver = NashSolver(config)

        self.population: List[StrategyProfile] = []
        self.generation = 0
        self.payoff_cache: Dict[Tuple[str, str], float] = {}

    def initialize_population(
        self,
        seed_strategies: Optional[List[StrategyGenome]] = None,
        size: int = 10
    ) -> None:
        """Initialize population with seed strategies or random."""
        self.population = []

        if seed_strategies:
            for strategy in seed_strategies[:size]:
                self._add_strategy(strategy)

        # Fill remaining with random
        while len(self.population) < size:
            strategy = self.generate_strategy()
            self._add_strategy(strategy)

        logger.info(f"Initialized PSRO population with {len(self.population)} strategies")

    def _add_strategy(self, strategy: StrategyGenome) -> None:
        """Add a strategy to the population."""
        # Compute performance vector
        perf_vector = self._compute_performance_vector(strategy)

        profile = StrategyProfile(
            strategy=strategy,
            performance_vector=perf_vector,
            generation=self.generation
        )
        self.population.append(profile)

    def _compute_performance_vector(
        self,
        strategy: StrategyGenome,
        num_samples: int = 10
    ) -> np.ndarray:
        """
        Compute performance vector for a strategy.

        Evaluates against sample of existing population.
        """
        if not self.population:
            # Bootstrap with random performance
            return np.random.randn(num_samples)

        # Sample opponents from population
        num_opponents = min(num_samples, len(self.population))
        opponents = np.random.choice(
            len(self.population),
            size=num_opponents,
            replace=False
        )

        performances = []
        for idx in opponents:
            opponent = self.population[idx]
            payoff = self._get_payoff(strategy, opponent.strategy)
            performances.append(payoff)

        return np.array(performances)

    def _get_payoff(
        self,
        strategy_a: StrategyGenome,
        strategy_b: StrategyGenome
    ) -> float:
        """Get or compute payoff for A vs B."""
        key = (strategy_a.id, strategy_b.id)

        if key not in self.payoff_cache:
            payoff = self.evaluator(strategy_a, strategy_b)
            self.payoff_cache[key] = payoff

        return self.payoff_cache[key]

    def _default_evaluator(
        self,
        strategy_a: StrategyGenome,
        strategy_b: StrategyGenome
    ) -> float:
        """Default evaluator based on strategy vectors."""
        vec_a = strategy_a.to_vector()
        vec_b = strategy_b.to_vector()

        # Compute difference in characteristics
        diff = vec_a - vec_b

        # Strategy A's advantage: higher variance in different dimensions
        a_diversity = np.std(vec_a)
        b_diversity = np.std(vec_b)

        # More diverse strategy has advantage
        diversity_advantage = a_diversity - b_diversity

        # Random market conditions favor different strategies
        market_conditions = np.random.randn(len(vec_a))
        a_fit = np.dot(vec_a, market_conditions) / (np.linalg.norm(vec_a) + 1e-8)
        b_fit = np.dot(vec_b, market_conditions) / (np.linalg.norm(vec_b) + 1e-8)
        condition_advantage = a_fit - b_fit

        return 0.3 * diversity_advantage + 0.7 * condition_advantage

    def compute_nash_equilibrium(self) -> np.ndarray:
        """Compute Nash equilibrium weights for current population."""
        if not self.population:
            return np.array([])

        # Build payoff matrix
        n = len(self.population)
        payoff_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    payoff = self._get_payoff(
                        self.population[i].strategy,
                        self.population[j].strategy
                    )
                    payoff_matrix[i, j] = payoff

        # Solve for Nash
        weights = self.nash_solver.solve(payoff_matrix)

        # Update population weights
        for i, profile in enumerate(self.population):
            profile.weight = weights[i]

        return weights

    def generate_best_response(self) -> List[StrategyGenome]:
        """
        Generate best-response strategies to current Nash mixture.

        Samples candidate strategies and selects those with highest
        expected payoff against the Nash mixture.
        """
        if not self.population:
            return [self.generate_strategy()]

        # Current Nash weights
        weights = np.array([p.weight for p in self.population])

        # Generate candidates
        candidates = []
        for _ in range(self.config.br_samples):
            candidates.append(self.generate_strategy())

        # Evaluate each candidate against Nash mixture
        candidate_payoffs = []
        for candidate in candidates:
            expected_payoff = 0
            for i, profile in enumerate(self.population):
                payoff = self._get_payoff(candidate, profile.strategy)
                expected_payoff += weights[i] * payoff
            candidate_payoffs.append(expected_payoff)

        # Select best candidates
        indices = np.argsort(candidate_payoffs)[::-1]
        best_candidates = [candidates[i] for i in indices[:3]]

        return best_candidates

    def iterate(self) -> PSROResult:
        """
        Run one PSRO iteration.

        1. Compute Nash equilibrium
        2. Generate best-response strategies
        3. Add to population
        4. Prune low-weight strategies

        Returns:
            PSROResult with iteration statistics
        """
        self.generation += 1

        # Compute Nash
        weights = self.compute_nash_equilibrium()

        # Compute exploitability
        if len(self.population) > 1:
            n = len(self.population)
            payoff_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        payoff_matrix[i, j] = self._get_payoff(
                            self.population[i].strategy,
                            self.population[j].strategy
                        )
            exploitability = self.nash_solver.compute_exploitability(payoff_matrix, weights)
        else:
            exploitability = 0.0

        # Generate best responses
        best_responses = self.generate_best_response()

        # Add to population
        for strategy in best_responses:
            if len(self.population) < self.config.max_population_size:
                self._add_strategy(strategy)

        # Prune if needed
        if self.generation % self.config.prune_interval == 0:
            self._prune_population()

        # Calculate diversity
        diversity = self._calculate_diversity()

        # Prepare result
        nash_weights = {
            self.population[i].strategy.id[:8]: weights[i]
            for i in range(len(self.population))
            if i < len(weights)
        }

        return PSROResult(
            new_strategies=best_responses,
            nash_weights=nash_weights,
            exploitability=exploitability,
            population_diversity=diversity,
            generation=self.generation
        )

    def _prune_population(self) -> None:
        """Remove low-weight strategies from population."""
        if len(self.population) <= self.config.min_population_size:
            return

        # Keep strategies above threshold
        surviving = [
            p for p in self.population
            if p.weight >= self.config.prune_threshold
        ]

        # Ensure minimum population
        if len(surviving) < self.config.min_population_size:
            # Sort by weight and keep top
            self.population.sort(key=lambda p: p.weight, reverse=True)
            surviving = self.population[:self.config.min_population_size]

        removed_count = len(self.population) - len(surviving)
        self.population = surviving

        if removed_count > 0:
            logger.info(f"Pruned {removed_count} strategies, {len(self.population)} remaining")

    def _calculate_diversity(self) -> float:
        """Calculate population diversity score."""
        if len(self.population) < 2:
            return 1.0

        # Compute average pairwise distance
        vectors = [p.strategy.to_vector() for p in self.population]
        total_dist = 0
        count = 0

        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                # Cosine distance
                dot = np.dot(vectors[i], vectors[j])
                norm = np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j])
                similarity = dot / (norm + 1e-8)
                total_dist += 1 - similarity
                count += 1

        return total_dist / count if count > 0 else 1.0

    def get_diverse_sample(self, n: int) -> List[StrategyGenome]:
        """
        Get diverse sample of strategies weighted by Nash distribution.

        Args:
            n: Number of strategies to sample

        Returns:
            List of sampled strategies
        """
        if not self.population:
            return []

        weights = np.array([p.weight for p in self.population])
        weights = weights / np.sum(weights)

        # Sample with replacement according to Nash weights
        indices = np.random.choice(
            len(self.population),
            size=min(n, len(self.population)),
            replace=False,
            p=weights
        )

        return [self.population[i].strategy for i in indices]

    def get_top_strategies(self, n: int) -> List[Tuple[StrategyGenome, float]]:
        """Get top strategies by Nash weight."""
        sorted_pop = sorted(self.population, key=lambda p: p.weight, reverse=True)
        return [(p.strategy, p.weight) for p in sorted_pop[:n]]

    def inject_strategy(self, strategy: StrategyGenome) -> None:
        """Inject an external strategy into the population."""
        self._add_strategy(strategy)
        logger.info(f"Injected strategy {strategy.id[:8]} into PSRO population")

    def get_status(self) -> Dict[str, Any]:
        """Get current status of PSRO manager."""
        return {
            'population_size': len(self.population),
            'generation': self.generation,
            'total_payoffs_computed': len(self.payoff_cache),
            'top_5_weights': [
                (p.strategy.id[:8], p.weight)
                for p in sorted(self.population, key=lambda x: x.weight, reverse=True)[:5]
            ],
            'diversity': self._calculate_diversity()
        }
