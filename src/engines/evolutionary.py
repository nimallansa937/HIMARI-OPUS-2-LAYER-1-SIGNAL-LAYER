"""
Engine 1: Evolutionary Search

Population-based evolutionary search over strategy space.
Maintains genetic diversity while selecting for fitness.

Operator allocation:
- Mutation: 15%
- Crossover: 40%
- Novel generation: 10%
- Recombination: 35%
"""

import random
import uuid
import time
from typing import List, Callable, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

from ..core.genome import (
    StrategyGenome, DecisionNode, Condition, SignalType,
    SIGNAL_THRESHOLD_RANGES, generate_random_strategy
)


@dataclass
class EvolutionStats:
    """Statistics from one generation of evolution."""
    generation: int
    population_size: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    diversity_score: float
    stagnation_count: int
    mutations: int
    crossovers: int
    novel: int


class NeutralDriftManager:
    """
    Accept neutral mutations to explore fitness landscape.

    When fitness stagnates, neutral drift allows exploration
    without selection pressure—escaping local optima.

    Key insight from protein engineering: neutral networks
    in fitness landscape connect distant optimal regions.
    """

    def __init__(
        self,
        tolerance: float = 0.05,
        drift_duration: int = 10,
        trigger_stagnation: int = 5
    ):
        """
        Args:
            tolerance: Accept mutations within this % of parent fitness
            drift_duration: Number of generations to drift
            trigger_stagnation: Generations of stagnation before drift mode
        """
        self.tolerance = tolerance
        self.drift_duration = drift_duration
        self.trigger_stagnation = trigger_stagnation

        self.drift_counter = 0
        self.in_drift_mode = False
        self.drift_start_fitness = 0.0
        self.drift_generations = 0

    def should_accept(self, old_fitness: float, new_fitness: float) -> bool:
        """
        Determine if a mutation should be accepted.

        In drift mode: accept if within tolerance band.
        Normal mode: accept only if improved.
        """
        if self.in_drift_mode:
            relative_change = abs(new_fitness - old_fitness) / max(abs(old_fitness), 1e-6)
            return relative_change < self.tolerance
        return new_fitness >= old_fitness

    def update(self, best_fitness: float, stagnation_count: int) -> Dict[str, Any]:
        """
        Update drift mode based on evolution progress.

        Returns:
            Status dict with drift mode info
        """
        status = {
            'in_drift_mode': self.in_drift_mode,
            'drift_counter': self.drift_counter,
            'action': None
        }

        # Check if we should enter drift mode
        if stagnation_count >= self.trigger_stagnation and not self.in_drift_mode:
            self.in_drift_mode = True
            self.drift_counter = 0
            self.drift_start_fitness = best_fitness
            status['action'] = 'entered_drift_mode'
            status['in_drift_mode'] = True

        # Update drift counter if in drift mode
        if self.in_drift_mode:
            self.drift_counter += 1
            self.drift_generations += 1

            # Exit drift mode after duration
            if self.drift_counter >= self.drift_duration:
                self.in_drift_mode = False
                self.drift_counter = 0
                status['action'] = 'exited_drift_mode'
                status['in_drift_mode'] = False

                # Check if drift found improvement
                if best_fitness > self.drift_start_fitness * 1.05:
                    status['drift_success'] = True

        return status

    def reset(self):
        """Reset drift manager state."""
        self.drift_counter = 0
        self.in_drift_mode = False
        self.drift_generations = 0


class EvolutionaryExplorer:
    """
    Population-based evolutionary search over strategy space.

    Features:
    - Tournament selection for parent choice
    - Subtree crossover for genetic recombination
    - Adaptive mutation rate based on stagnation
    - Neutral drift for escaping local optima
    - Elite preservation for stability
    """

    def __init__(
        self,
        population_size: int = 100,
        elite_size: int = 10,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.40,
        tournament_size: int = 5,
        max_depth: int = 6
    ):
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.max_depth = max_depth

        self.population: List[StrategyGenome] = []
        self.generation = 0
        self.stagnation_counter = 0
        self.last_best_fitness = float('-inf')

        self.neutral_drift = NeutralDriftManager()
        self.stats_history: List[EvolutionStats] = []

    def initialize_population(
        self,
        seed_strategies: Optional[List[StrategyGenome]] = None
    ) -> None:
        """
        Initialize population with seeds or random strategies.

        Args:
            seed_strategies: Optional list of seed strategies to include
        """
        self.population = []

        # Add seed strategies as elites
        if seed_strategies:
            for seed in seed_strategies[:self.elite_size]:
                seed_copy = seed.copy()
                seed_copy.source_engine = "seed"
                self.population.append(seed_copy)

        # Fill rest with random strategies
        while len(self.population) < self.population_size:
            strategy = generate_random_strategy(max_depth=self.max_depth)
            strategy.source_engine = "evolutionary"
            self.population.append(strategy)

        self.generation = 0
        self.stagnation_counter = 0
        self.last_best_fitness = float('-inf')

    def evolve_generation(
        self,
        evaluator: Callable[[StrategyGenome], Dict],
        grammar_validator
    ) -> StrategyGenome:
        """
        Evolve one generation and return best strategy.

        Args:
            evaluator: Function that takes a strategy and returns metrics dict
            grammar_validator: Validates strategy grammar

        Returns:
            Best strategy from this generation
        """
        start_time = time.time()
        stats = {
            'mutations': 0,
            'crossovers': 0,
            'novel': 0
        }

        # Evaluate fitness for strategies that haven't been evaluated
        for strategy in self.population:
            if strategy.fitness == 0.0:
                result = evaluator(strategy)
                strategy.fitness = self._compute_fitness(result)
                strategy.backtest_metrics = result

        # Sort by fitness (descending)
        self.population.sort(key=lambda s: s.fitness, reverse=True)

        # Track stagnation
        current_best = self.population[0].fitness
        if current_best <= self.last_best_fitness:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            self.last_best_fitness = current_best

        # Update neutral drift
        drift_status = self.neutral_drift.update(current_best, self.stagnation_counter)

        # Adaptive mutation rate
        effective_mutation = self.mutation_rate
        if self.stagnation_counter > 5:
            effective_mutation = min(0.5, self.mutation_rate * (1 + 0.1 * self.stagnation_counter))

        # Create new population
        new_pop = []

        # Keep elites
        for elite in self.population[:self.elite_size]:
            elite_copy = elite.copy()
            elite_copy.fitness = elite.fitness  # Preserve fitness
            new_pop.append(elite_copy)

        # Generate offspring
        while len(new_pop) < self.population_size:
            rand = random.random()

            if rand < self.crossover_rate:
                # Crossover
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()
                child = self._crossover(parent1, parent2)
                stats['crossovers'] += 1
            elif rand < self.crossover_rate + 0.10:
                # Novel generation (10%)
                child = generate_random_strategy(max_depth=self.max_depth)
                stats['novel'] += 1
            else:
                # Clone + mutation
                parent = self._tournament_select()
                child = parent.copy()

            # Apply mutation
            if random.random() < effective_mutation:
                child = self._mutate(child)
                stats['mutations'] += 1

            # Validate grammar
            is_valid, _ = grammar_validator.validate_genome(child)
            if is_valid:
                child.generation = self.generation + 1
                child.fitness = 0.0  # Mark for evaluation
                child.source_engine = "evolutionary"
                new_pop.append(child)

        self.population = new_pop
        self.generation += 1

        # Calculate diversity
        diversity = self._calculate_diversity()

        # Record stats
        fitnesses = [s.fitness for s in self.population if s.fitness > 0]
        self.stats_history.append(EvolutionStats(
            generation=self.generation,
            population_size=len(self.population),
            best_fitness=current_best,
            mean_fitness=np.mean(fitnesses) if fitnesses else 0,
            std_fitness=np.std(fitnesses) if fitnesses else 0,
            diversity_score=diversity,
            stagnation_count=self.stagnation_counter,
            mutations=stats['mutations'],
            crossovers=stats['crossovers'],
            novel=stats['novel']
        ))

        return self.population[0]

    def _tournament_select(self) -> StrategyGenome:
        """Select parent using tournament selection."""
        tournament = random.sample(
            self.population,
            min(self.tournament_size, len(self.population))
        )
        return max(tournament, key=lambda s: s.fitness)

    def _crossover(self, p1: StrategyGenome, p2: StrategyGenome) -> StrategyGenome:
        """
        Create child through subtree crossover.

        Randomly select a subtree from each parent and swap.
        """
        # Start with copy of parent 1
        child_tree = self._deep_copy_tree(p1.decision_tree)

        # Get random node from p2 as donor
        p2_nodes = self._get_all_nodes(p2.decision_tree)
        if p2_nodes:
            donor = random.choice(p2_nodes)

            # Find insertion point in child
            child_nodes = self._get_all_nodes(child_tree)
            if child_nodes and len(child_nodes) > 1:
                # Don't replace root
                insertion_point = random.choice(child_nodes[1:])
                if insertion_point is not None:
                    # Copy donor subtree to insertion point
                    self._replace_subtree(child_tree, insertion_point, donor)

        # Interpolate parameters
        alpha = random.random()
        return StrategyGenome(
            id=str(uuid.uuid4()),
            decision_tree=child_tree,
            base_position_pct=alpha * p1.base_position_pct + (1-alpha) * p2.base_position_pct,
            stop_loss_atr_mult=alpha * p1.stop_loss_atr_mult + (1-alpha) * p2.stop_loss_atr_mult,
            take_profit_atr_mult=alpha * p1.take_profit_atr_mult + (1-alpha) * p2.take_profit_atr_mult,
            lineage=[p1.id, p2.id],
            generation=max(p1.generation, p2.generation) + 1
        )

    def _mutate(self, strategy: StrategyGenome) -> StrategyGenome:
        """Apply random mutation to strategy."""
        child = strategy.copy()
        mutation_type = random.choice(['param', 'operator', 'threshold', 'subtree', 'signal'])

        if mutation_type == 'param':
            # Mutate risk parameters
            child.base_position_pct *= (1 + random.gauss(0, 0.1))
            child.stop_loss_atr_mult *= (1 + random.gauss(0, 0.1))
            child.take_profit_atr_mult *= (1 + random.gauss(0, 0.1))

            # Clamp to valid ranges
            child.base_position_pct = np.clip(child.base_position_pct, 0.01, 0.5)
            child.stop_loss_atr_mult = np.clip(child.stop_loss_atr_mult, 0.5, 5.0)
            child.take_profit_atr_mult = np.clip(child.take_profit_atr_mult, 1.0, 10.0)

        elif mutation_type in ['operator', 'threshold', 'signal']:
            # Mutate a random condition in the tree
            nodes = self._get_internal_nodes(child.decision_tree)
            if nodes:
                node = random.choice(nodes)
                if node.condition:
                    node.condition = node.condition.mutate()

        elif mutation_type == 'subtree':
            # Replace a random subtree
            nodes = self._get_all_nodes(child.decision_tree)
            if nodes and len(nodes) > 1:
                target = random.choice(nodes[1:])  # Don't replace root
                new_subtree = self._generate_random_subtree(max_depth=3)
                self._replace_subtree(child.decision_tree, target, new_subtree)

        child.id = str(uuid.uuid4())
        child.lineage = [strategy.id]
        return child

    def _compute_fitness(self, result: Dict) -> float:
        """
        Compute multi-objective fitness score.

        Components:
        - Sharpe ratio (primary)
        - Max drawdown penalty
        - Trade count requirement
        - Profit factor bonus
        """
        sharpe = result.get('sharpe', 0)
        max_dd = result.get('max_drawdown', 1)
        profit_factor = result.get('profit_factor', 0)
        trades = result.get('trade_count', 0)

        # Penalties
        trade_penalty = 0 if trades >= 100 else (100 - trades) * 0.01
        dd_penalty = max(0, max_dd - 0.15) * 10

        # Bonuses
        pf_bonus = np.log1p(max(profit_factor, 0)) * 0.3

        fitness = sharpe * 1.0 + pf_bonus - dd_penalty - trade_penalty
        return max(0, fitness)

    def _calculate_diversity(self) -> float:
        """Calculate population diversity using average pairwise distance."""
        if len(self.population) < 2:
            return 1.0

        vectors = [s.to_vector() for s in self.population[:20]]  # Sample for efficiency
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

    def _get_all_nodes(self, node: DecisionNode) -> List[DecisionNode]:
        """Get all nodes in tree (BFS order)."""
        nodes = [node]
        queue = [node]

        while queue:
            current = queue.pop(0)
            if current.true_branch:
                nodes.append(current.true_branch)
                queue.append(current.true_branch)
            if current.false_branch:
                nodes.append(current.false_branch)
                queue.append(current.false_branch)

        return nodes

    def _get_internal_nodes(self, node: DecisionNode) -> List[DecisionNode]:
        """Get only internal (non-leaf) nodes."""
        return [n for n in self._get_all_nodes(node) if n.action is None]

    def _deep_copy_tree(self, node: Optional[DecisionNode]) -> Optional[DecisionNode]:
        """Create deep copy of decision tree."""
        if node is None:
            return None

        if node.action is not None:
            return DecisionNode(action=node.action)

        return DecisionNode(
            condition=Condition(
                signal=node.condition.signal,
                operator=node.condition.operator,
                threshold=node.condition.threshold
            ) if node.condition else None,
            true_branch=self._deep_copy_tree(node.true_branch),
            false_branch=self._deep_copy_tree(node.false_branch)
        )

    def _generate_random_subtree(self, max_depth: int = 3, depth: int = 0) -> DecisionNode:
        """Generate a random subtree."""
        if depth >= max_depth or random.random() < 0.3:
            return DecisionNode(action=random.choice([-1, 0, 1]))

        signal = random.choice(list(SignalType))
        operator = random.choice(['>', '<'])
        min_t, max_t = SIGNAL_THRESHOLD_RANGES[signal]
        threshold = random.uniform(min_t, max_t)

        return DecisionNode(
            condition=Condition(signal=signal, operator=operator, threshold=threshold),
            true_branch=self._generate_random_subtree(max_depth, depth + 1),
            false_branch=self._generate_random_subtree(max_depth, depth + 1)
        )

    def _replace_subtree(
        self,
        root: DecisionNode,
        target: DecisionNode,
        replacement: DecisionNode
    ) -> None:
        """Replace target node with replacement in tree rooted at root."""
        queue = [(root, None, None)]  # (node, parent, is_true_branch)

        while queue:
            node, parent, is_true = queue.pop(0)

            if node is target and parent is not None:
                if is_true:
                    parent.true_branch = self._deep_copy_tree(replacement)
                else:
                    parent.false_branch = self._deep_copy_tree(replacement)
                return

            if node.true_branch:
                queue.append((node.true_branch, node, True))
            if node.false_branch:
                queue.append((node.false_branch, node, False))

    def get_best_strategies(self, n: int = 10) -> List[StrategyGenome]:
        """Get top N strategies by fitness."""
        sorted_pop = sorted(self.population, key=lambda s: s.fitness, reverse=True)
        return sorted_pop[:n]

    def inject_strategy(self, strategy: StrategyGenome) -> None:
        """Inject an external strategy into the population."""
        strategy_copy = strategy.copy()
        strategy_copy.fitness = 0.0  # Mark for evaluation

        # Replace worst individual
        if self.population:
            self.population[-1] = strategy_copy
        else:
            self.population.append(strategy_copy)
