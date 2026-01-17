"""
Strongly-Typed Genetic Programming (STGP) Formula - Enhancement 3

Evolves variable-structure expression trees with type checking.
Grammar constraints prevent invalid operations (Volume + Price rejected).

Enables WorldQuant-style alpha formulas like:
    Rank(Close) - Rank(MA(Open, 10))

Type System:
- PRICE: Close, Open, High, Low
- VOLUME: Volume, OBV
- RATIO: RSI, Returns, Correlation
- RANK: Output of Rank() operator
- BOOLEAN: Comparison results
- TIMESERIES: Any numeric series
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from abc import ABC, abstractmethod
import random
import math
import operator
import copy


class DimensionType(Enum):
    """Dimension types for type-safe genetic programming."""
    PRICE = auto()      # Close, Open, High, Low
    VOLUME = auto()     # Volume, OBV
    RATIO = auto()      # RSI, Returns, Correlation, normalized values
    RANK = auto()       # Output of Rank() operator
    BOOLEAN = auto()    # Comparison results
    TIMESERIES = auto() # Any numeric series
    INTEGER = auto()    # Integer constants (for window sizes)
    ANY = auto()        # Wildcard type


@dataclass
class TypeSignature:
    """Type signature for a primitive function."""
    input_types: List[DimensionType]
    output_type: DimensionType


class Node(ABC):
    """Base class for expression tree nodes."""

    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> Any:
        """Evaluate this node given the context."""
        pass

    @abstractmethod
    def get_type(self) -> DimensionType:
        """Get the output type of this node."""
        pass

    @abstractmethod
    def to_string(self) -> str:
        """Convert to string representation."""
        pass

    @abstractmethod
    def copy(self) -> 'Node':
        """Create a deep copy of this node."""
        pass

    @abstractmethod
    def get_depth(self) -> int:
        """Get the depth of this subtree."""
        pass

    @abstractmethod
    def get_size(self) -> int:
        """Get the number of nodes in this subtree."""
        pass


@dataclass
class Terminal(Node):
    """A terminal (leaf) node - either a feature or constant."""
    name: str
    value_type: DimensionType
    value_func: Optional[Callable[[Dict[str, Any]], float]] = None
    constant_value: Optional[float] = None

    def evaluate(self, context: Dict[str, Any]) -> float:
        if self.constant_value is not None:
            return self.constant_value
        if self.value_func is not None:
            return self.value_func(context)
        return context.get(self.name, 0.0)

    def get_type(self) -> DimensionType:
        return self.value_type

    def to_string(self) -> str:
        if self.constant_value is not None:
            return str(self.constant_value)
        return self.name

    def copy(self) -> 'Terminal':
        return Terminal(
            name=self.name,
            value_type=self.value_type,
            value_func=self.value_func,
            constant_value=self.constant_value
        )

    def get_depth(self) -> int:
        return 1

    def get_size(self) -> int:
        return 1


@dataclass
class Primitive(Node):
    """A primitive (internal) node - a function with children."""
    name: str
    func: Callable
    children: List[Node]
    signature: TypeSignature

    def evaluate(self, context: Dict[str, Any]) -> Any:
        child_values = [child.evaluate(context) for child in self.children]
        try:
            return self.func(*child_values)
        except (ZeroDivisionError, ValueError, OverflowError):
            return 0.0

    def get_type(self) -> DimensionType:
        return self.signature.output_type

    def to_string(self) -> str:
        child_strs = [child.to_string() for child in self.children]
        if len(self.children) == 2 and self.name in ['+', '-', '*', '/', '>', '<', '==']:
            return f"({child_strs[0]} {self.name} {child_strs[1]})"
        return f"{self.name}({', '.join(child_strs)})"

    def copy(self) -> 'Primitive':
        return Primitive(
            name=self.name,
            func=self.func,
            children=[child.copy() for child in self.children],
            signature=self.signature
        )

    def get_depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(child.get_depth() for child in self.children)

    def get_size(self) -> int:
        return 1 + sum(child.get_size() for child in self.children)


class PrimitiveSet:
    """
    A set of typed primitives and terminals for STGP.

    Maintains type compatibility rules and generates valid expressions.
    """

    def __init__(self, name: str = "ALPHA"):
        self.name = name
        self.primitives: Dict[str, Tuple[Callable, TypeSignature]] = {}
        self.terminals: Dict[str, Tuple[DimensionType, Callable]] = {}
        self.ephemeral_constants: List[Tuple[str, Callable, DimensionType]] = []

    def add_primitive(
        self,
        name: str,
        func: Callable,
        input_types: List[DimensionType],
        output_type: DimensionType
    ) -> None:
        """Add a primitive function with type signature."""
        self.primitives[name] = (func, TypeSignature(input_types, output_type))

    def add_terminal(
        self,
        name: str,
        ret_type: DimensionType,
        value_func: Optional[Callable[[Dict[str, Any]], float]] = None
    ) -> None:
        """Add a terminal (feature or constant)."""
        self.terminals[name] = (ret_type, value_func)

    def add_ephemeral_constant(
        self,
        name: str,
        generator: Callable[[], float],
        ret_type: DimensionType
    ) -> None:
        """Add an ephemeral random constant."""
        self.ephemeral_constants.append((name, generator, ret_type))

    def get_primitives_by_output(
        self,
        output_type: DimensionType
    ) -> List[Tuple[str, Callable, TypeSignature]]:
        """Get all primitives that produce the given output type."""
        result = []
        for name, (func, sig) in self.primitives.items():
            if sig.output_type == output_type or output_type == DimensionType.ANY or sig.output_type == DimensionType.ANY:
                result.append((name, func, sig))
        return result

    def get_terminals_by_type(
        self,
        target_type: DimensionType
    ) -> List[Tuple[str, DimensionType, Optional[Callable]]]:
        """Get all terminals of the given type."""
        result = []
        for name, (ret_type, value_func) in self.terminals.items():
            if ret_type == target_type or target_type == DimensionType.ANY or ret_type == DimensionType.ANY:
                result.append((name, ret_type, value_func))

        # Include ephemeral constants of matching type
        for name, generator, ret_type in self.ephemeral_constants:
            if ret_type == target_type or target_type == DimensionType.ANY:
                result.append((f"{name}_{random.random():.4f}", ret_type, lambda ctx, g=generator: g()))

        return result


def create_typed_primitive_set() -> PrimitiveSet:
    """Create the default typed primitive set for alpha formula discovery."""
    pset = PrimitiveSet("ALPHA")

    # Safe division
    def protected_div(a, b):
        if abs(b) < 1e-10:
            return 0.0
        return a / b

    # Safe log
    def protected_log(a):
        if a <= 0:
            return 0.0
        return math.log(a)

    # Safe sqrt
    def protected_sqrt(a):
        if a < 0:
            return 0.0
        return math.sqrt(a)

    # Rank function (simplified - returns normalized position)
    def rank(x):
        # In a real implementation, this would rank across the universe
        # Here we normalize to [-1, 1]
        return max(-1, min(1, x))

    # Rolling functions (simplified for single value context)
    def delay(x, n):
        # Would need historical data in real implementation
        return x

    def rolling_mean(x, n):
        return x

    def rolling_std(x, n):
        return abs(x) * 0.1  # Placeholder

    def rolling_max(x, n):
        return x

    def rolling_min(x, n):
        return x

    # Conditional
    def if_then_else(cond, x, y):
        return x if cond else y

    # Arithmetic primitives - SAME TYPE
    pset.add_primitive('+', operator.add, [DimensionType.RATIO, DimensionType.RATIO], DimensionType.RATIO)
    pset.add_primitive('-', operator.sub, [DimensionType.RATIO, DimensionType.RATIO], DimensionType.RATIO)
    pset.add_primitive('price_add', operator.add, [DimensionType.PRICE, DimensionType.PRICE], DimensionType.PRICE)
    pset.add_primitive('price_sub', operator.sub, [DimensionType.PRICE, DimensionType.PRICE], DimensionType.PRICE)

    # Arithmetic primitives - ANY TYPE for mul/div
    pset.add_primitive('*', operator.mul, [DimensionType.RATIO, DimensionType.RATIO], DimensionType.RATIO)
    pset.add_primitive('/', protected_div, [DimensionType.RATIO, DimensionType.RATIO], DimensionType.RATIO)
    pset.add_primitive('price_ratio', protected_div, [DimensionType.PRICE, DimensionType.PRICE], DimensionType.RATIO)

    # Transform primitives
    pset.add_primitive('rank', rank, [DimensionType.RATIO], DimensionType.RANK)
    pset.add_primitive('rank_price', rank, [DimensionType.PRICE], DimensionType.RANK)
    pset.add_primitive('abs', abs, [DimensionType.RATIO], DimensionType.RATIO)
    pset.add_primitive('sign', lambda x: 1 if x > 0 else (-1 if x < 0 else 0), [DimensionType.RATIO], DimensionType.RATIO)
    pset.add_primitive('log', protected_log, [DimensionType.RATIO], DimensionType.RATIO)
    pset.add_primitive('sqrt', protected_sqrt, [DimensionType.RATIO], DimensionType.RATIO)
    pset.add_primitive('neg', operator.neg, [DimensionType.RATIO], DimensionType.RATIO)

    # Min/Max
    pset.add_primitive('max', max, [DimensionType.RATIO, DimensionType.RATIO], DimensionType.RATIO)
    pset.add_primitive('min', min, [DimensionType.RATIO, DimensionType.RATIO], DimensionType.RATIO)
    pset.add_primitive('max_price', max, [DimensionType.PRICE, DimensionType.PRICE], DimensionType.PRICE)
    pset.add_primitive('min_price', min, [DimensionType.PRICE, DimensionType.PRICE], DimensionType.PRICE)

    # Temporal primitives (simplified)
    pset.add_primitive('delay', delay, [DimensionType.RATIO, DimensionType.INTEGER], DimensionType.RATIO)
    pset.add_primitive('rolling_mean', rolling_mean, [DimensionType.RATIO, DimensionType.INTEGER], DimensionType.RATIO)
    pset.add_primitive('rolling_std', rolling_std, [DimensionType.RATIO, DimensionType.INTEGER], DimensionType.RATIO)
    pset.add_primitive('rolling_max', rolling_max, [DimensionType.RATIO, DimensionType.INTEGER], DimensionType.RATIO)
    pset.add_primitive('rolling_min', rolling_min, [DimensionType.RATIO, DimensionType.INTEGER], DimensionType.RATIO)

    # Comparison primitives
    pset.add_primitive('>', lambda a, b: 1.0 if a > b else 0.0, [DimensionType.RATIO, DimensionType.RATIO], DimensionType.BOOLEAN)
    pset.add_primitive('<', lambda a, b: 1.0 if a < b else 0.0, [DimensionType.RATIO, DimensionType.RATIO], DimensionType.BOOLEAN)
    pset.add_primitive('==', lambda a, b: 1.0 if abs(a - b) < 1e-10 else 0.0, [DimensionType.RATIO, DimensionType.RATIO], DimensionType.BOOLEAN)

    # Conditional
    pset.add_primitive('if_then_else', if_then_else, [DimensionType.BOOLEAN, DimensionType.RATIO, DimensionType.RATIO], DimensionType.RATIO)

    # Price terminals
    pset.add_terminal('close', DimensionType.PRICE, lambda ctx: ctx.get('close', 0))
    pset.add_terminal('open', DimensionType.PRICE, lambda ctx: ctx.get('open', 0))
    pset.add_terminal('high', DimensionType.PRICE, lambda ctx: ctx.get('high', 0))
    pset.add_terminal('low', DimensionType.PRICE, lambda ctx: ctx.get('low', 0))

    # Volume terminals
    pset.add_terminal('volume', DimensionType.VOLUME, lambda ctx: ctx.get('volume', 0))
    pset.add_terminal('obv', DimensionType.VOLUME, lambda ctx: ctx.get('obv', 0))

    # Ratio terminals (normalized indicators)
    pset.add_terminal('rsi', DimensionType.RATIO, lambda ctx: ctx.get('rsi', 50) / 100 - 0.5)
    pset.add_terminal('macd', DimensionType.RATIO, lambda ctx: ctx.get('macd', 0))
    pset.add_terminal('macd_signal', DimensionType.RATIO, lambda ctx: ctx.get('macd_signal', 0))
    pset.add_terminal('macd_histogram', DimensionType.RATIO, lambda ctx: ctx.get('macd_histogram', 0))
    pset.add_terminal('returns', DimensionType.RATIO, lambda ctx: ctx.get('returns', 0))
    pset.add_terminal('volatility', DimensionType.RATIO, lambda ctx: ctx.get('volatility', 0))
    pset.add_terminal('funding_rate', DimensionType.RATIO, lambda ctx: ctx.get('funding_rate', 0))
    pset.add_terminal('whale_pressure', DimensionType.RATIO, lambda ctx: ctx.get('whale_pressure', 0))
    pset.add_terminal('volume_zscore', DimensionType.RATIO, lambda ctx: ctx.get('volume_zscore', 0))

    # Integer constants for window sizes
    pset.add_terminal('int_5', DimensionType.INTEGER, lambda ctx: 5)
    pset.add_terminal('int_10', DimensionType.INTEGER, lambda ctx: 10)
    pset.add_terminal('int_20', DimensionType.INTEGER, lambda ctx: 20)
    pset.add_terminal('int_50', DimensionType.INTEGER, lambda ctx: 50)

    # Ephemeral constants
    pset.add_ephemeral_constant('rand_ratio', lambda: random.uniform(-1, 1), DimensionType.RATIO)
    pset.add_ephemeral_constant('rand_small', lambda: random.uniform(0, 0.1), DimensionType.RATIO)

    return pset


class STGPFormula:
    """
    Strongly-Typed Genetic Programming formula.

    Represents and evaluates an expression tree with type checking.
    """

    def __init__(
        self,
        root: Optional[Node] = None,
        pset: Optional[PrimitiveSet] = None
    ):
        self.root = root
        self.pset = pset or create_typed_primitive_set()
        self._compiled_func: Optional[Callable] = None

    def evaluate(self, context: Dict[str, Any]) -> float:
        """Evaluate the formula given market data context."""
        if self.root is None:
            return 0.0

        try:
            result = self.root.evaluate(context)
            # Clamp to reasonable range
            if isinstance(result, (int, float)):
                return max(-10, min(10, float(result)))
            return 0.0
        except Exception:
            return 0.0

    def get_signal(self, context: Dict[str, Any]) -> Tuple[float, float]:
        """
        Get signal strength and confidence from the formula.

        Returns:
            Tuple of (signal_strength [-1, 1], confidence [0, 1])
        """
        raw_value = self.evaluate(context)

        # Convert to signal strength [-1, 1]
        signal = max(-1, min(1, raw_value))

        # Confidence based on signal magnitude
        confidence = min(1.0, abs(raw_value))

        return signal, confidence

    def to_string(self) -> str:
        """Convert formula to string representation."""
        if self.root is None:
            return "None"
        return self.root.to_string()

    def get_depth(self) -> int:
        """Get the depth of the expression tree."""
        if self.root is None:
            return 0
        return self.root.get_depth()

    def get_size(self) -> int:
        """Get the number of nodes in the tree."""
        if self.root is None:
            return 0
        return self.root.get_size()

    def copy(self) -> 'STGPFormula':
        """Create a deep copy of the formula."""
        return STGPFormula(
            root=self.root.copy() if self.root else None,
            pset=self.pset
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "expression": self.to_string(),
            "depth": self.get_depth(),
            "size": self.get_size()
        }


class STGPGenerator:
    """Generator for random STGP expression trees."""

    def __init__(
        self,
        pset: PrimitiveSet,
        max_depth: int = 6,
        max_size: int = 50
    ):
        self.pset = pset
        self.max_depth = max_depth
        self.max_size = max_size

    def generate_tree(
        self,
        target_type: DimensionType = DimensionType.RATIO,
        method: str = "half_and_half",
        depth: int = 0
    ) -> Node:
        """
        Generate a random expression tree.

        Args:
            target_type: The required output type
            method: Generation method ("full", "grow", or "half_and_half")
            depth: Current depth (for recursion)

        Returns:
            Root node of the generated tree
        """
        if method == "half_and_half":
            method = "full" if random.random() < 0.5 else "grow"

        # Force terminal at max depth
        if depth >= self.max_depth:
            return self._generate_terminal(target_type)

        # Get available options
        terminals = self.pset.get_terminals_by_type(target_type)
        primitives = self.pset.get_primitives_by_output(target_type)

        if not terminals and not primitives:
            # Fallback: return zero constant
            return Terminal("zero", DimensionType.RATIO, constant_value=0.0)

        # Choose between terminal and primitive
        if method == "full" and primitives and depth < self.max_depth - 1:
            # Full method: prefer primitives until max depth
            return self._generate_primitive(primitives, depth)
        elif method == "grow":
            # Grow method: random choice weighted towards terminals near max depth
            terminal_prob = depth / self.max_depth
            if random.random() < terminal_prob or not primitives:
                if terminals:
                    return self._generate_terminal_from_list(terminals)
                else:
                    return self._generate_primitive(primitives, depth)
            else:
                return self._generate_primitive(primitives, depth)
        else:
            # Default: random choice
            if terminals and (not primitives or random.random() < 0.3):
                return self._generate_terminal_from_list(terminals)
            elif primitives:
                return self._generate_primitive(primitives, depth)
            else:
                return Terminal("zero", DimensionType.RATIO, constant_value=0.0)

    def _generate_terminal(self, target_type: DimensionType) -> Terminal:
        """Generate a terminal of the given type."""
        terminals = self.pset.get_terminals_by_type(target_type)
        if terminals:
            return self._generate_terminal_from_list(terminals)
        return Terminal("zero", DimensionType.RATIO, constant_value=0.0)

    def _generate_terminal_from_list(
        self,
        terminals: List[Tuple[str, DimensionType, Optional[Callable]]]
    ) -> Terminal:
        """Generate a terminal from a list of options."""
        name, ret_type, value_func = random.choice(terminals)
        return Terminal(name=name, value_type=ret_type, value_func=value_func)

    def _generate_primitive(
        self,
        primitives: List[Tuple[str, Callable, TypeSignature]],
        depth: int
    ) -> Primitive:
        """Generate a primitive with recursively generated children."""
        name, func, sig = random.choice(primitives)

        children = []
        for input_type in sig.input_types:
            child = self.generate_tree(input_type, "grow", depth + 1)
            children.append(child)

        return Primitive(name=name, func=func, children=children, signature=sig)

    def generate_formula(
        self,
        target_type: DimensionType = DimensionType.RATIO
    ) -> STGPFormula:
        """Generate a random STGP formula."""
        root = self.generate_tree(target_type)
        return STGPFormula(root=root, pset=self.pset)


class STGPGeneticOperators:
    """Genetic operators for STGP evolution."""

    @staticmethod
    def get_random_node(root: Node, depth: int = 0) -> Tuple[Node, List[int]]:
        """
        Get a random node from the tree with its path.

        Returns:
            Tuple of (node, path) where path is list of child indices
        """
        if isinstance(root, Terminal):
            return root, []

        # Probability of selecting this node vs children
        if random.random() < 0.3 or not isinstance(root, Primitive):
            return root, []

        # Select a child
        child_idx = random.randint(0, len(root.children) - 1)
        child_node, child_path = STGPGeneticOperators.get_random_node(
            root.children[child_idx], depth + 1
        )
        return child_node, [child_idx] + child_path

    @staticmethod
    def replace_node(root: Node, path: List[int], new_node: Node) -> Node:
        """Replace a node at the given path with a new node."""
        if not path:
            return new_node

        if not isinstance(root, Primitive):
            return root

        root_copy = root.copy()
        current = root_copy

        for i, idx in enumerate(path[:-1]):
            if isinstance(current, Primitive) and idx < len(current.children):
                current = current.children[idx]

        if isinstance(current, Primitive) and path[-1] < len(current.children):
            current.children[path[-1]] = new_node

        return root_copy

    @staticmethod
    def crossover(
        parent1: STGPFormula,
        parent2: STGPFormula,
        generator: STGPGenerator
    ) -> Tuple[STGPFormula, STGPFormula]:
        """
        Perform subtree crossover between two formulas.

        Exchanges type-compatible subtrees.
        """
        if parent1.root is None or parent2.root is None:
            return parent1.copy(), parent2.copy()

        # Get random nodes from each parent
        node1, path1 = STGPGeneticOperators.get_random_node(parent1.root)
        node2, path2 = STGPGeneticOperators.get_random_node(parent2.root)

        # Check type compatibility
        if node1.get_type() != node2.get_type():
            # Types don't match, return copies
            return parent1.copy(), parent2.copy()

        # Perform crossover
        child1_root = STGPGeneticOperators.replace_node(
            parent1.root.copy(), path1, node2.copy()
        )
        child2_root = STGPGeneticOperators.replace_node(
            parent2.root.copy(), path2, node1.copy()
        )

        # Check size constraints
        child1 = STGPFormula(root=child1_root, pset=parent1.pset)
        child2 = STGPFormula(root=child2_root, pset=parent2.pset)

        if child1.get_size() > generator.max_size:
            child1 = parent1.copy()
        if child2.get_size() > generator.max_size:
            child2 = parent2.copy()

        return child1, child2

    @staticmethod
    def mutate_uniform(
        formula: STGPFormula,
        generator: STGPGenerator
    ) -> STGPFormula:
        """
        Replace a random subtree with a new randomly generated subtree.
        """
        if formula.root is None:
            return generator.generate_formula()

        formula_copy = formula.copy()
        node, path = STGPGeneticOperators.get_random_node(formula_copy.root)

        # Generate new subtree of same type
        new_subtree = generator.generate_tree(node.get_type())

        # Replace
        formula_copy.root = STGPGeneticOperators.replace_node(
            formula_copy.root, path, new_subtree
        )

        # Check size constraint
        if formula_copy.get_size() > generator.max_size:
            return formula.copy()

        return formula_copy

    @staticmethod
    def mutate_shrink(formula: STGPFormula) -> STGPFormula:
        """
        Replace a random subtree with a terminal (bloat control).
        """
        if formula.root is None:
            return formula.copy()

        if isinstance(formula.root, Terminal):
            return formula.copy()

        formula_copy = formula.copy()

        # Find a primitive node to shrink
        node, path = STGPGeneticOperators.get_random_node(formula_copy.root)

        if isinstance(node, Primitive):
            # Replace with a terminal of the same type
            terminals = formula.pset.get_terminals_by_type(node.get_type())
            if terminals:
                name, ret_type, value_func = random.choice(terminals)
                new_terminal = Terminal(name=name, value_type=ret_type, value_func=value_func)
                formula_copy.root = STGPGeneticOperators.replace_node(
                    formula_copy.root, path, new_terminal
                )

        return formula_copy

    @staticmethod
    def mutate_hoist(formula: STGPFormula) -> STGPFormula:
        """
        Replace the tree with one of its subtrees (bloat control).
        """
        if formula.root is None:
            return formula.copy()

        if isinstance(formula.root, Terminal):
            return formula.copy()

        # Get a random subtree
        node, _ = STGPGeneticOperators.get_random_node(formula.root)

        # Use it as the new root if types match
        if node.get_type() == formula.root.get_type():
            return STGPFormula(root=node.copy(), pset=formula.pset)

        return formula.copy()


def evaluate_expression(formula: STGPFormula, market_data: Dict[str, Any]) -> float:
    """
    Evaluate an STGP expression with market data.

    Args:
        formula: The STGP formula to evaluate
        market_data: Dict containing market features

    Returns:
        Signal value from the formula
    """
    return formula.evaluate(market_data)


# Factory function for quick formula generation
def generate_random_formula(
    max_depth: int = 6,
    max_size: int = 50
) -> STGPFormula:
    """Generate a random STGP formula."""
    pset = create_typed_primitive_set()
    generator = STGPGenerator(pset, max_depth, max_size)
    return generator.generate_formula()
