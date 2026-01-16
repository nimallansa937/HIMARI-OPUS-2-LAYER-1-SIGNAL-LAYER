"""
Monte Carlo Tree Search (MCTS) Engine

An additional generation engine that uses tree search to explore
the strategy space more systematically. Uses UCT formula with
progressive widening for effective exploration-exploitation tradeoff.

This addresses Gap #1 from the gap analysis: Adding principled
tree search to complement random mutation and gradient-based methods.
"""

import math
import random
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Tuple
import numpy as np
from datetime import datetime

from ..core.genome import StrategyGenome, generate_random_strategy
from ..core.grammar import GrammarValidator

logger = logging.getLogger(__name__)


@dataclass
class MCTSNode:
    """
    A node in the MCTS search tree.

    Each node represents a strategy (state) and tracks
    visit counts and value estimates for UCT selection.
    """
    strategy: StrategyGenome
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)

    # MCTS statistics
    visits: int = 0
    total_value: float = 0.0  # Sum of all backpropagated values
    prior: float = 1.0  # Prior probability from policy network

    # Metadata
    depth: int = 0
    action_from_parent: Optional[str] = None  # Mutation type that created this

    @property
    def value(self) -> float:
        """Average value estimate."""
        return self.total_value / max(self.visits, 1)

    @property
    def ucb_score(self) -> float:
        """Upper Confidence Bound score for selection."""
        if self.visits == 0:
            return float('inf')

        if self.parent is None or self.parent.visits == 0:
            return self.value

        exploration = math.sqrt(2 * math.log(self.parent.visits) / self.visits)
        return self.value + exploration

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0

    def is_fully_expanded(self, max_children: int) -> bool:
        """Check if all possible children have been created."""
        return len(self.children) >= max_children


@dataclass
class MCTSConfig:
    """Configuration for MCTS search."""
    # Search parameters
    num_simulations: int = 100  # Simulations per search
    max_depth: int = 10  # Maximum tree depth
    max_children: int = 5  # Maximum children per node

    # UCT parameters
    exploration_constant: float = 1.414  # sqrt(2) is standard
    prior_weight: float = 1.0  # Weight for prior in PUCT

    # Progressive widening
    progressive_widening_c: float = 1.0  # Controls widening rate
    progressive_widening_alpha: float = 0.5  # Exponent for widening

    # Value estimation
    use_value_network: bool = False
    rollout_depth: int = 3  # Steps for random rollout


@dataclass
class MCTSResult:
    """Result from MCTS search."""
    best_strategy: StrategyGenome
    root_value: float
    simulations_run: int
    tree_depth: int
    total_nodes: int
    search_time_ms: float
    top_strategies: List[Tuple[StrategyGenome, float]]


class MCTSStrategyGenerator:
    """
    Monte Carlo Tree Search for strategy generation.

    Uses tree search to systematically explore the strategy space:
    1. Selection: Use UCT to select promising nodes
    2. Expansion: Create new child strategies via mutation
    3. Simulation: Evaluate strategy quality
    4. Backpropagation: Update statistics along path

    Integrates with evolutionary engine for mutations.
    """

    # Mutation actions available
    MUTATION_ACTIONS = [
        'mutate_threshold',      # Change condition thresholds
        'mutate_operator',       # Change comparison operators
        'mutate_indicator',      # Change signal indicators
        'add_condition',         # Add new condition to tree
        'remove_condition',      # Simplify tree
        'swap_subtree'           # Structural change
    ]

    def __init__(
        self,
        config: MCTSConfig,
        evaluator: Optional[Callable[[StrategyGenome], float]] = None,
        grammar: Optional[GrammarValidator] = None,
        value_network: Optional[Any] = None,
        policy_network: Optional[Any] = None
    ):
        """
        Initialize MCTS generator.

        Args:
            config: MCTS configuration
            evaluator: Function to evaluate strategy quality (returns value)
            grammar: Grammar validator for checking valid strategies
            value_network: Optional neural network for value estimation
            policy_network: Optional neural network for action priors
        """
        self.config = config
        self.evaluator = evaluator or self._default_evaluator
        self.grammar = grammar or GrammarValidator()
        self.value_network = value_network
        self.policy_network = policy_network

        self._total_nodes = 0

    def search(
        self,
        root_strategy: Optional[StrategyGenome] = None,
        num_simulations: Optional[int] = None
    ) -> MCTSResult:
        """
        Run MCTS search to find improved strategies.

        Args:
            root_strategy: Starting strategy (random if None)
            num_simulations: Override config simulation count

        Returns:
            MCTSResult with best strategy and statistics
        """
        import time
        start_time = time.time()

        # Initialize root
        if root_strategy is None:
            root_strategy = generate_random_strategy(max_depth=4)

        root = MCTSNode(strategy=root_strategy, depth=0)
        self._total_nodes = 1

        simulations = num_simulations or self.config.num_simulations

        # Run simulations
        for i in range(simulations):
            self._run_simulation(root)

        # Collect results
        best_child = self._select_best_child(root)
        best_strategy = best_child.strategy if best_child else root_strategy

        # Get top strategies from tree
        top_strategies = self._collect_top_strategies(root, n=10)

        search_time = (time.time() - start_time) * 1000
        tree_depth = self._get_tree_depth(root)

        return MCTSResult(
            best_strategy=best_strategy,
            root_value=root.value,
            simulations_run=simulations,
            tree_depth=tree_depth,
            total_nodes=self._total_nodes,
            search_time_ms=search_time,
            top_strategies=top_strategies
        )

    def _run_simulation(self, root: MCTSNode) -> None:
        """Run a single MCTS simulation."""
        # Selection
        node = self._select(root)

        # Expansion
        if node.visits > 0 and not node.is_fully_expanded(self._max_children_for_node(node)):
            node = self._expand(node)

        # Simulation/Evaluation
        value = self._simulate(node)

        # Backpropagation
        self._backpropagate(node, value)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a leaf node using UCT."""
        while not node.is_leaf():
            if not node.is_fully_expanded(self._max_children_for_node(node)):
                return node

            # Select child with highest UCT score
            node = self._select_child(node)

            if node.depth >= self.config.max_depth:
                break

        return node

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest UCT score."""
        best_score = float('-inf')
        best_child = None

        for child in node.children:
            score = self._uct_score(child, node.visits)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child or node.children[0]

    def _uct_score(self, node: MCTSNode, parent_visits: int) -> float:
        """
        Calculate UCT score with optional PUCT enhancement.

        PUCT formula: Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        if node.visits == 0:
            return float('inf')

        exploitation = node.value
        exploration = self.config.exploration_constant * math.sqrt(
            math.log(parent_visits + 1) / node.visits
        )

        # Add prior bonus (PUCT)
        prior_bonus = (
            self.config.prior_weight * node.prior *
            math.sqrt(parent_visits) / (1 + node.visits)
        )

        return exploitation + exploration + prior_bonus

    def _max_children_for_node(self, node: MCTSNode) -> int:
        """
        Calculate max children using progressive widening.

        Formula: max_children = c * N^alpha
        where N is visit count, c and alpha are parameters.
        """
        if node.visits == 0:
            return 1

        max_children = int(
            self.config.progressive_widening_c *
            (node.visits ** self.config.progressive_widening_alpha)
        )

        return min(max_children, self.config.max_children)

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by creating a new child."""
        # Select mutation action
        action = self._select_action(node)

        # Apply mutation to create child strategy
        child_strategy = self._apply_mutation(node.strategy, action)

        # Validate child
        is_valid, _ = self.grammar.validate_genome(child_strategy)
        if not is_valid:
            # Try another action
            for fallback_action in self.MUTATION_ACTIONS:
                if fallback_action != action:
                    child_strategy = self._apply_mutation(node.strategy, fallback_action)
                    is_valid, _ = self.grammar.validate_genome(child_strategy)
                    if is_valid:
                        action = fallback_action
                        break

        # Get prior from policy network if available
        prior = self._get_action_prior(node.strategy, action)

        # Create child node
        child = MCTSNode(
            strategy=child_strategy,
            parent=node,
            depth=node.depth + 1,
            action_from_parent=action,
            prior=prior
        )

        node.children.append(child)
        self._total_nodes += 1

        return child

    def _select_action(self, node: MCTSNode) -> str:
        """Select mutation action, avoiding already-used actions."""
        used_actions = {child.action_from_parent for child in node.children}
        available = [a for a in self.MUTATION_ACTIONS if a not in used_actions]

        if not available:
            available = self.MUTATION_ACTIONS

        # Use policy network if available
        if self.policy_network:
            priors = self._get_all_priors(node.strategy)
            available_priors = {a: priors.get(a, 1.0) for a in available}
            total = sum(available_priors.values())
            probs = {a: p / total for a, p in available_priors.items()}

            # Sample from distribution
            r = random.random()
            cumsum = 0
            for action, prob in probs.items():
                cumsum += prob
                if r <= cumsum:
                    return action

        return random.choice(available)

    def _apply_mutation(
        self,
        strategy: StrategyGenome,
        action: str
    ) -> StrategyGenome:
        """Apply a mutation action to create new strategy."""
        child = strategy.copy()

        if action == 'mutate_threshold':
            self._mutate_threshold(child)
        elif action == 'mutate_operator':
            self._mutate_operator(child)
        elif action == 'mutate_indicator':
            self._mutate_indicator(child)
        elif action == 'add_condition':
            self._add_condition(child)
        elif action == 'remove_condition':
            self._remove_condition(child)
        elif action == 'swap_subtree':
            self._swap_subtree(child)

        return child

    def _mutate_threshold(self, strategy: StrategyGenome) -> None:
        """Mutate a condition threshold."""
        def mutate_node(node):
            if node is None:
                return
            if node.condition:
                # Adjust threshold by small amount
                delta = np.random.normal(0, 0.1)
                node.condition.threshold = node.condition.threshold * (1 + delta)
            mutate_node(node.true_branch)
            mutate_node(node.false_branch)

        mutate_node(strategy.decision_tree)

    def _mutate_operator(self, strategy: StrategyGenome) -> None:
        """Mutate a comparison operator."""
        from ..core.genome import SignalType
        operators = ['>', '<', '>=', '<=']

        def mutate_node(node):
            if node is None:
                return
            if node.condition and random.random() < 0.3:
                node.condition.operator = random.choice(operators)
            mutate_node(node.true_branch)
            mutate_node(node.false_branch)

        mutate_node(strategy.decision_tree)

    def _mutate_indicator(self, strategy: StrategyGenome) -> None:
        """Mutate a signal indicator."""
        from ..core.genome import SignalType

        def mutate_node(node):
            if node is None:
                return
            if node.condition and random.random() < 0.3:
                node.condition.signal = random.choice(list(SignalType))
            mutate_node(node.true_branch)
            mutate_node(node.false_branch)

        mutate_node(strategy.decision_tree)

    def _add_condition(self, strategy: StrategyGenome) -> None:
        """Add a new condition to the tree."""
        from ..core.genome import SignalType, Condition, DecisionNode

        def find_leaf(node):
            if node is None:
                return None
            if node.action is not None:  # Leaf
                return node
            result = find_leaf(node.true_branch)
            if result:
                return result
            return find_leaf(node.false_branch)

        leaf = find_leaf(strategy.decision_tree)
        if leaf and random.random() < 0.5:
            # Replace leaf with new condition
            old_action = leaf.action
            leaf.action = None
            leaf.condition = Condition(
                signal=random.choice(list(SignalType)),
                operator=random.choice(['>', '<']),
                threshold=random.uniform(-2, 2)
            )
            leaf.true_branch = DecisionNode(action=old_action)
            leaf.false_branch = DecisionNode(action=0)  # Hold

    def _remove_condition(self, strategy: StrategyGenome) -> None:
        """Simplify tree by removing a condition."""

        def simplify_node(node, parent, is_true_branch):
            if node is None:
                return
            if node.condition and random.random() < 0.2:
                # Replace with one of its children
                replacement = node.true_branch if random.random() < 0.5 else node.false_branch
                if parent:
                    if is_true_branch:
                        parent.true_branch = replacement
                    else:
                        parent.false_branch = replacement
            else:
                simplify_node(node.true_branch, node, True)
                simplify_node(node.false_branch, node, False)

        simplify_node(strategy.decision_tree, None, True)

    def _swap_subtree(self, strategy: StrategyGenome) -> None:
        """Swap two subtrees in the decision tree."""
        if strategy.decision_tree and strategy.decision_tree.condition:
            # Simply swap true and false branches
            if random.random() < 0.3:
                strategy.decision_tree.true_branch, strategy.decision_tree.false_branch = \
                    strategy.decision_tree.false_branch, strategy.decision_tree.true_branch

    def _simulate(self, node: MCTSNode) -> float:
        """
        Evaluate a node via simulation/rollout or value network.

        Returns estimated value for the strategy.
        """
        # Use value network if available
        if self.config.use_value_network and self.value_network:
            return self._estimate_with_network(node.strategy)

        # Otherwise use direct evaluation
        return self.evaluator(node.strategy)

    def _estimate_with_network(self, strategy: StrategyGenome) -> float:
        """Estimate value using neural network."""
        if self.value_network is None:
            return 0.0

        try:
            vector = strategy.to_vector()
            # Assume value network takes numpy array and returns scalar
            value = self.value_network(vector)
            return float(value)
        except Exception as e:
            logger.warning(f"Value network estimation failed: {e}")
            return 0.0

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagate value up the tree."""
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent

    def _select_best_child(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Select best child based on visit count (most robust)."""
        if not node.children:
            return None

        return max(node.children, key=lambda c: c.visits)

    def _collect_top_strategies(
        self,
        root: MCTSNode,
        n: int = 10
    ) -> List[Tuple[StrategyGenome, float]]:
        """Collect top strategies from the tree by value."""
        strategies = []

        def collect(node):
            if node.visits > 0:
                strategies.append((node.strategy, node.value))
            for child in node.children:
                collect(child)

        collect(root)

        # Sort by value descending
        strategies.sort(key=lambda x: x[1], reverse=True)
        return strategies[:n]

    def _get_tree_depth(self, root: MCTSNode) -> int:
        """Get maximum depth of tree."""

        def depth(node):
            if not node.children:
                return node.depth
            return max(depth(child) for child in node.children)

        return depth(root)

    def _get_action_prior(self, strategy: StrategyGenome, action: str) -> float:
        """Get prior probability for action from policy network."""
        if self.policy_network is None:
            return 1.0 / len(self.MUTATION_ACTIONS)

        priors = self._get_all_priors(strategy)
        return priors.get(action, 1.0 / len(self.MUTATION_ACTIONS))

    def _get_all_priors(self, strategy: StrategyGenome) -> Dict[str, float]:
        """Get all action priors from policy network."""
        if self.policy_network is None:
            return {a: 1.0 / len(self.MUTATION_ACTIONS) for a in self.MUTATION_ACTIONS}

        try:
            vector = strategy.to_vector()
            priors = self.policy_network(vector)
            return dict(zip(self.MUTATION_ACTIONS, priors))
        except Exception as e:
            logger.warning(f"Policy network failed: {e}")
            return {a: 1.0 / len(self.MUTATION_ACTIONS) for a in self.MUTATION_ACTIONS}

    def _default_evaluator(self, strategy: StrategyGenome) -> float:
        """Default strategy evaluator (mock)."""
        # In practice, this would run a fast backtest
        # Here we use genome vector properties as proxy
        vector = strategy.to_vector()

        # Favor diverse vectors (high variance)
        diversity = np.std(vector)

        # Favor valid parameter ranges
        validity = 1.0 if all(-10 < v < 10 for v in vector[:50]) else 0.5

        # Random component for exploration
        noise = np.random.uniform(0, 0.2)

        return diversity * validity + noise

    def generate_batch(
        self,
        num_strategies: int,
        seed_strategies: Optional[List[StrategyGenome]] = None
    ) -> List[StrategyGenome]:
        """
        Generate multiple strategies using MCTS.

        Args:
            num_strategies: Number of strategies to generate
            seed_strategies: Optional seed strategies to start from

        Returns:
            List of generated strategies
        """
        results = []

        # Use seeds or generate random roots
        roots = seed_strategies[:num_strategies] if seed_strategies else []
        while len(roots) < num_strategies:
            roots.append(None)

        for root in roots:
            # Run search from each root
            result = self.search(
                root_strategy=root,
                num_simulations=self.config.num_simulations // num_strategies
            )
            results.append(result.best_strategy)

        return results
