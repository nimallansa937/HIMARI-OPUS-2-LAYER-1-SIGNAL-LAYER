"""
Engine Orchestrator

Coordinates all generation engines and enforces diversity.
Allocates compute budget and ensures strategy population
doesn't converge to a fragile monoculture.
"""

import asyncio
import time
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass
import numpy as np
import logging

from ..core.genome import StrategyGenome, generate_random_strategy
from ..core.grammar import GrammarValidator
from .evolutionary import EvolutionaryExplorer
from .flow_matching import FlowMatchingGenerator, GenerationCondition
from .llm_guided import LLMGuidedGenerator
from .harvester import ExternalIdeaHarvester

logger = logging.getLogger(__name__)


@dataclass
class GenerationBudget:
    """Budget allocation for generation engines."""
    evolutionary_pct: float = 0.40      # 40% from evolutionary search
    generative_pct: float = 0.25        # 25% from flow matching
    llm_guided_pct: float = 0.20        # 20% from LLM guidance
    external_pct: float = 0.15          # 15% from external harvesting

    total_candidates_per_cycle: int = 100
    max_llm_calls_per_cycle: int = 20   # Rate limit for LLM calls

    def get_targets(self) -> Dict[str, int]:
        """Calculate target count for each engine."""
        total = self.total_candidates_per_cycle
        return {
            'evolutionary': int(total * self.evolutionary_pct),
            'generative': int(total * self.generative_pct),
            'llm_guided': min(
                int(total * self.llm_guided_pct),
                self.max_llm_calls_per_cycle
            ),
            'external': int(total * self.external_pct)
        }


@dataclass
class GenerationCycleResult:
    """Result from one generation cycle."""
    candidates: List[StrategyGenome]
    by_engine: Dict[str, int]
    diversity_score: float
    total_time_ms: float
    errors: List[str]


class EngineOrchestrator:
    """
    Coordinates generation engines and enforces diversity.

    Responsibilities:
    1. Allocate compute budget to engines
    2. Run engines in parallel
    3. Deduplicate and diversity-filter candidates
    4. Track engine performance for adaptive allocation
    """

    def __init__(
        self,
        evolutionary: EvolutionaryExplorer,
        generative: FlowMatchingGenerator,
        llm_guided: LLMGuidedGenerator,
        external: ExternalIdeaHarvester,
        budget: GenerationBudget,
        grammar_validator: GrammarValidator,
        diversity_threshold: float = 0.85
    ):
        self.evolutionary = evolutionary
        self.generative = generative
        self.llm_guided = llm_guided
        self.external = external
        self.budget = budget
        self.grammar = grammar_validator
        self.diversity_threshold = diversity_threshold

        # Performance tracking for adaptive allocation
        self.engine_success_rates: Dict[str, float] = {
            'evolutionary': 0.5,
            'generative': 0.5,
            'llm_guided': 0.5,
            'external': 0.5
        }

        self.cycle_count = 0

    async def generate_candidates(
        self,
        condition: GenerationCondition,
        existing_portfolio: List[StrategyGenome],
        evaluator: Optional[Callable] = None
    ) -> GenerationCycleResult:
        """
        Generate diverse strategy candidates from all engines.

        Args:
            condition: Target properties for generated strategies
            existing_portfolio: Current portfolio strategies (for diversity)
            evaluator: Optional fitness evaluator function

        Returns:
            GenerationCycleResult with candidates and statistics
        """
        start_time = time.time()
        targets = self.budget.get_targets()
        errors = []

        # Run all engines in parallel
        tasks = [
            self._run_evolutionary(targets['evolutionary'], evaluator),
            self._run_generative(targets['generative'], condition),
            self._run_llm_guided(targets['llm_guided'], condition, existing_portfolio),
            self._run_external(targets['external'])
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect candidates from each engine
        all_candidates = []
        by_engine = {}

        engine_names = ['evolutionary', 'generative', 'llm_guided', 'external']
        for result, name in zip(results, engine_names):
            if isinstance(result, Exception):
                errors.append(f"{name}: {str(result)}")
                by_engine[name] = 0
            else:
                for strategy in result:
                    strategy.source_engine = name
                    all_candidates.append(strategy)
                by_engine[name] = len(result)

        # Diversity filtering
        diverse_candidates = self._enforce_diversity(all_candidates, existing_portfolio)

        # Calculate diversity score
        diversity_score = self._calculate_diversity(diverse_candidates)

        # Update engine success rates
        self._update_success_rates(by_engine, targets)

        self.cycle_count += 1
        total_time = (time.time() - start_time) * 1000

        return GenerationCycleResult(
            candidates=diverse_candidates,
            by_engine=by_engine,
            diversity_score=diversity_score,
            total_time_ms=total_time,
            errors=errors
        )

    async def _run_evolutionary(
        self,
        target: int,
        evaluator: Optional[Callable]
    ) -> List[StrategyGenome]:
        """Run evolutionary engine."""
        try:
            if not self.evolutionary.population:
                self.evolutionary.initialize_population()

            candidates = []

            # Simple evaluator if none provided
            if evaluator is None:
                evaluator = lambda s: {
                    'sharpe': np.random.uniform(0, 2),
                    'max_drawdown': np.random.uniform(0.05, 0.2),
                    'profit_factor': np.random.uniform(0.8, 1.5),
                    'trade_count': np.random.randint(50, 200)
                }

            # Evolve for several generations
            for _ in range(3):
                best = self.evolutionary.evolve_generation(evaluator, self.grammar)
                if best and best not in candidates:
                    candidates.append(best)

            # Get top performers
            top = self.evolutionary.get_best_strategies(target)
            for s in top:
                if s not in candidates:
                    candidates.append(s)

            return candidates[:target]

        except Exception as e:
            logger.error(f"Evolutionary engine failed: {e}")
            return []

    async def _run_generative(
        self,
        target: int,
        condition: GenerationCondition
    ) -> List[StrategyGenome]:
        """Run flow matching generator."""
        try:
            candidates = self.generative.generate_diverse(
                condition=condition,
                num_samples=target * 2,
                num_keep=target,
                diversity_threshold=0.3
            )

            # Validate each candidate
            valid = []
            for c in candidates:
                is_valid, _ = self.grammar.validate_genome(c)
                if is_valid:
                    valid.append(c)

            return valid[:target]

        except Exception as e:
            logger.error(f"Generative engine failed: {e}")
            return []

    async def _run_llm_guided(
        self,
        target: int,
        condition: GenerationCondition,
        existing_portfolio: List[StrategyGenome]
    ) -> List[StrategyGenome]:
        """Run LLM-guided generator."""
        try:
            candidates = []

            # Determine portfolio gaps
            gaps = self._identify_portfolio_gaps(existing_portfolio)

            # Generate novel strategies to fill gaps
            for i in range(min(target // 2, 5)):
                try:
                    result = await self.llm_guided.generate_novel(
                        condition=condition,
                        portfolio_gaps=gaps
                    )
                    if result:
                        genome = self.llm_guided.strategy_from_llm(result)
                        if genome:
                            candidates.append(genome)
                except Exception as e:
                    logger.warning(f"LLM novel generation failed: {e}")

            # Mutate existing strategies
            if existing_portfolio:
                for strategy in existing_portfolio[:3]:
                    try:
                        mutations = await self.llm_guided.generate_mutations(
                            strategy=strategy,
                            backtest_result=strategy.backtest_metrics,
                            num_mutations=2
                        )
                        for m in mutations:
                            genome = self.llm_guided.strategy_from_llm(m)
                            if genome:
                                candidates.append(genome)
                    except Exception as e:
                        logger.warning(f"LLM mutation failed: {e}")

            return candidates[:target]

        except Exception as e:
            logger.error(f"LLM guided engine failed: {e}")
            return []

    async def _run_external(
        self,
        target: int
    ) -> List[StrategyGenome]:
        """Run external idea harvester."""
        try:
            # Harvest and extract strategies
            parsed_strategies = await self.external.harvest_and_extract(
                max_strategies=target
            )

            # Convert to genomes
            candidates = []
            for parsed in parsed_strategies:
                # Create a random genome as base, tagged with external source
                genome = generate_random_strategy(max_depth=4)
                genome.source_engine = "external"
                genome.backtest_metrics['source_url'] = parsed.source_url
                genome.backtest_metrics['mechanism'] = parsed.mechanism
                genome.backtest_metrics['causal_hypothesis'] = parsed.causal_hypothesis
                candidates.append(genome)

            return candidates[:target]

        except Exception as e:
            logger.error(f"External harvester failed: {e}")
            return []

    def _enforce_diversity(
        self,
        candidates: List[StrategyGenome],
        existing: List[StrategyGenome]
    ) -> List[StrategyGenome]:
        """
        Filter candidates to maintain diversity.

        Removes candidates too similar to:
        1. Each other
        2. Existing portfolio strategies
        """
        diverse = []
        all_existing = existing + diverse

        for candidate in candidates:
            is_diverse = True

            for other in all_existing:
                similarity = self._compute_similarity(candidate, other)
                if similarity > self.diversity_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                diverse.append(candidate)
                all_existing.append(candidate)

        return diverse

    def _compute_similarity(
        self,
        s1: StrategyGenome,
        s2: StrategyGenome
    ) -> float:
        """Compute cosine similarity between strategy vectors."""
        v1 = s1.to_vector()
        v2 = s2.to_vector()
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return dot / (norm + 1e-8)

    def _calculate_diversity(
        self,
        candidates: List[StrategyGenome]
    ) -> float:
        """Calculate average pairwise diversity."""
        if len(candidates) < 2:
            return 1.0

        vectors = [c.to_vector() for c in candidates[:20]]  # Sample
        total_dist = 0
        count = 0

        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                similarity = np.dot(vectors[i], vectors[j]) / (
                    np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]) + 1e-8
                )
                total_dist += 1 - similarity
                count += 1

        return total_dist / count if count > 0 else 1.0

    def _identify_portfolio_gaps(
        self,
        portfolio: List[StrategyGenome]
    ) -> List[str]:
        """
        Identify what types of strategies are missing from portfolio.

        Analyzes existing strategies to find gaps in:
        - Market regimes covered
        - Strategy types (momentum, reversion, etc.)
        - Timeframes
        """
        if not portfolio:
            return [
                "Momentum strategies for trending markets",
                "Mean reversion strategies for ranging markets",
                "Volatility breakout strategies",
                "Order flow based strategies",
                "Funding rate carry strategies"
            ]

        gaps = []

        # Check regime coverage
        regime_coverage = set()
        for s in portfolio:
            if 'regime' in s.backtest_metrics:
                regime_coverage.add(s.backtest_metrics['regime'])

        if 'bull' not in regime_coverage:
            gaps.append("Strategies optimized for bull markets")
        if 'bear' not in regime_coverage:
            gaps.append("Strategies optimized for bear markets")
        if 'range' not in regime_coverage:
            gaps.append("Strategies for ranging/sideways markets")

        # Check strategy type coverage
        engine_coverage = set(s.source_engine for s in portfolio)
        if 'llm_guided' not in engine_coverage:
            gaps.append("Semantically designed strategies with clear logic")
        if 'external' not in engine_coverage:
            gaps.append("Research-backed strategies from academic sources")

        # Always useful gaps
        gaps.extend([
            "Strategies with low correlation to existing portfolio",
            "Strategies exploiting different market inefficiencies"
        ])

        return gaps[:5]

    def _update_success_rates(
        self,
        actual: Dict[str, int],
        targets: Dict[str, int]
    ) -> None:
        """Update engine success rates for adaptive allocation."""
        alpha = 0.1  # Exponential moving average factor

        for engine in actual:
            if targets[engine] > 0:
                rate = actual[engine] / targets[engine]
                self.engine_success_rates[engine] = (
                    (1 - alpha) * self.engine_success_rates[engine] +
                    alpha * rate
                )

    def get_engine_stats(self) -> Dict[str, Any]:
        """Get statistics about engine performance."""
        return {
            'success_rates': self.engine_success_rates.copy(),
            'cycle_count': self.cycle_count,
            'budget_allocation': self.budget.get_targets()
        }

    def inject_strategy(self, strategy: StrategyGenome) -> None:
        """
        Inject an external strategy into the evolutionary population.

        Useful for seeding with human-designed strategies.
        """
        self.evolutionary.inject_strategy(strategy)


class AdaptiveBudgetManager:
    """
    Dynamically adjusts budget allocation based on engine performance.

    Engines that consistently produce better strategies get more budget.
    """

    def __init__(
        self,
        orchestrator: EngineOrchestrator,
        adaptation_rate: float = 0.05,
        min_allocation: float = 0.10
    ):
        self.orchestrator = orchestrator
        self.adaptation_rate = adaptation_rate
        self.min_allocation = min_allocation

    def adapt_budget(
        self,
        validation_results: Dict[str, float]
    ) -> GenerationBudget:
        """
        Adapt budget based on validation success rates.

        Args:
            validation_results: Dict mapping engine name to validation pass rate

        Returns:
            Updated GenerationBudget
        """
        current = self.orchestrator.budget
        success_rates = self.orchestrator.engine_success_rates

        # Calculate new allocations based on success
        total_success = sum(validation_results.values()) or 1.0

        new_allocations = {}
        for engine in ['evolutionary', 'generative', 'llm_guided', 'external']:
            rate = validation_results.get(engine, success_rates[engine])
            new_allocations[engine] = max(
                self.min_allocation,
                rate / total_success
            )

        # Normalize to sum to 1
        total = sum(new_allocations.values())
        for engine in new_allocations:
            new_allocations[engine] /= total

        # Apply gradual adaptation
        return GenerationBudget(
            evolutionary_pct=self._adapt(current.evolutionary_pct, new_allocations['evolutionary']),
            generative_pct=self._adapt(current.generative_pct, new_allocations['generative']),
            llm_guided_pct=self._adapt(current.llm_guided_pct, new_allocations['llm_guided']),
            external_pct=self._adapt(current.external_pct, new_allocations['external']),
            total_candidates_per_cycle=current.total_candidates_per_cycle,
            max_llm_calls_per_cycle=current.max_llm_calls_per_cycle
        )

    def _adapt(self, current: float, target: float) -> float:
        """Gradually adapt current value toward target."""
        return current + self.adaptation_rate * (target - current)
