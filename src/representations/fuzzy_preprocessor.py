"""
Fuzzy Logic Preprocessor - Enhancement 2

Replaces hard thresholds with gradual membership functions.
RSI=69 vs RSI=71 shouldn't produce completely different signals.
Output is degree of membership (0.0-1.0), not binary.

Supports:
- Triangular membership functions
- Trapezoidal membership functions
- Gaussian membership functions
- Sigmoid membership functions
- Sugeno-style fuzzy inference
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from enum import Enum
import math
import random


class FuzzyOperator(Enum):
    """Fuzzy logic operators for combining antecedents."""
    AND = "and"  # min(a, b)
    OR = "or"    # max(a, b)
    NOT = "not"  # 1 - a


class MembershipFunction(ABC):
    """Base class for fuzzy membership functions."""

    @abstractmethod
    def evaluate(self, crisp_value: float) -> float:
        """
        Evaluate membership degree for a crisp value.

        Args:
            crisp_value: The crisp input value

        Returns:
            Membership degree in [0, 1]
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, float]:
        """Get the parameters of this membership function."""
        pass

    @abstractmethod
    def set_parameters(self, params: Dict[str, float]) -> None:
        """Set the parameters of this membership function."""
        pass

    @abstractmethod
    def get_type(self) -> str:
        """Get the type name of this membership function."""
        pass


class TriangularMF(MembershipFunction):
    """
    Triangular membership function.

    Shape: /\
    Parameters: left, center, right
    """

    def __init__(self, left: float, center: float, right: float):
        self.left = left
        self.center = center
        self.right = right

    def evaluate(self, crisp_value: float) -> float:
        if crisp_value <= self.left or crisp_value >= self.right:
            return 0.0
        elif crisp_value == self.center:
            return 1.0
        elif crisp_value < self.center:
            return (crisp_value - self.left) / (self.center - self.left)
        else:
            return (self.right - crisp_value) / (self.right - self.center)

    def get_parameters(self) -> Dict[str, float]:
        return {"left": self.left, "center": self.center, "right": self.right}

    def set_parameters(self, params: Dict[str, float]) -> None:
        self.left = params.get("left", self.left)
        self.center = params.get("center", self.center)
        self.right = params.get("right", self.right)

    def get_type(self) -> str:
        return "triangular"


class TrapezoidalMF(MembershipFunction):
    """
    Trapezoidal membership function.

    Shape: /‾‾\
    Parameters: a, b, c, d (rising edge, plateau start, plateau end, falling edge)
    """

    def __init__(self, a: float, b: float, c: float, d: float):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def evaluate(self, crisp_value: float) -> float:
        if crisp_value <= self.a or crisp_value >= self.d:
            return 0.0
        elif self.b <= crisp_value <= self.c:
            return 1.0
        elif crisp_value < self.b:
            return (crisp_value - self.a) / (self.b - self.a)
        else:
            return (self.d - crisp_value) / (self.d - self.c)

    def get_parameters(self) -> Dict[str, float]:
        return {"a": self.a, "b": self.b, "c": self.c, "d": self.d}

    def set_parameters(self, params: Dict[str, float]) -> None:
        self.a = params.get("a", self.a)
        self.b = params.get("b", self.b)
        self.c = params.get("c", self.c)
        self.d = params.get("d", self.d)

    def get_type(self) -> str:
        return "trapezoidal"


class GaussianMF(MembershipFunction):
    """
    Gaussian membership function.

    Shape: Bell curve
    Parameters: mean, std
    """

    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = max(std, 1e-10)  # Prevent division by zero

    def evaluate(self, crisp_value: float) -> float:
        return math.exp(-0.5 * ((crisp_value - self.mean) / self.std) ** 2)

    def get_parameters(self) -> Dict[str, float]:
        return {"mean": self.mean, "std": self.std}

    def set_parameters(self, params: Dict[str, float]) -> None:
        self.mean = params.get("mean", self.mean)
        self.std = max(params.get("std", self.std), 1e-10)

    def get_type(self) -> str:
        return "gaussian"


class SigmoidMF(MembershipFunction):
    """
    Sigmoid membership function.

    Shape: S-curve (or reversed)
    Parameters: center, slope (positive = rising, negative = falling)
    """

    def __init__(self, center: float, slope: float):
        self.center = center
        self.slope = slope

    def evaluate(self, crisp_value: float) -> float:
        return 1.0 / (1.0 + math.exp(-self.slope * (crisp_value - self.center)))

    def get_parameters(self) -> Dict[str, float]:
        return {"center": self.center, "slope": self.slope}

    def set_parameters(self, params: Dict[str, float]) -> None:
        self.center = params.get("center", self.center)
        self.slope = params.get("slope", self.slope)

    def get_type(self) -> str:
        return "sigmoid"


@dataclass
class FuzzyVariable:
    """
    A fuzzy variable with multiple membership functions.

    Example: RSI variable with OVERSOLD, NEUTRAL, OVERBOUGHT sets
    """
    name: str
    universe: Tuple[float, float]  # (min, max) range
    membership_functions: Dict[str, MembershipFunction] = field(default_factory=dict)

    def add_mf(self, name: str, mf: MembershipFunction) -> None:
        """Add a membership function to this variable."""
        self.membership_functions[name] = mf

    def fuzzify(self, crisp_value: float) -> Dict[str, float]:
        """
        Compute membership degree in each fuzzy set.

        Args:
            crisp_value: The crisp input value

        Returns:
            Dict mapping set name to membership degree
        """
        return {
            name: mf.evaluate(crisp_value)
            for name, mf in self.membership_functions.items()
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "universe": self.universe,
            "membership_functions": {
                name: {
                    "type": mf.get_type(),
                    "params": mf.get_parameters()
                }
                for name, mf in self.membership_functions.items()
            }
        }


@dataclass
class FuzzyAntecedent:
    """An antecedent (IF part) of a fuzzy rule."""
    variable_name: str
    mf_name: str
    negated: bool = False


@dataclass
class FuzzyConsequent:
    """A consequent (THEN part) of a fuzzy rule."""
    variable_name: str
    mf_name: str
    # For Sugeno-style, can also be a constant or function
    sugeno_value: Optional[float] = None


@dataclass
class FuzzyRule:
    """
    A fuzzy rule of the form:
    IF antecedent1 AND/OR antecedent2 ... THEN consequent (weight)
    """
    antecedents: List[FuzzyAntecedent]
    consequent: FuzzyConsequent
    operator: FuzzyOperator = FuzzyOperator.AND
    weight: float = 1.0

    def evaluate_antecedents(
        self,
        fuzzified_values: Dict[str, Dict[str, float]]
    ) -> float:
        """
        Evaluate the firing strength of this rule.

        Args:
            fuzzified_values: Dict mapping variable_name -> {mf_name: degree}

        Returns:
            Firing strength in [0, 1]
        """
        if not self.antecedents:
            return 0.0

        strengths = []
        for ant in self.antecedents:
            if ant.variable_name not in fuzzified_values:
                strengths.append(0.0)
                continue

            degree = fuzzified_values[ant.variable_name].get(ant.mf_name, 0.0)
            if ant.negated:
                degree = 1.0 - degree
            strengths.append(degree)

        if self.operator == FuzzyOperator.AND:
            return min(strengths) * self.weight
        elif self.operator == FuzzyOperator.OR:
            return max(strengths) * self.weight
        else:
            return strengths[0] * self.weight if strengths else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "antecedents": [
                {"variable": a.variable_name, "mf": a.mf_name, "negated": a.negated}
                for a in self.antecedents
            ],
            "consequent": {
                "variable": self.consequent.variable_name,
                "mf": self.consequent.mf_name,
                "sugeno_value": self.consequent.sugeno_value
            },
            "operator": self.operator.value,
            "weight": self.weight
        }


class FuzzyInferenceSystem:
    """
    Sugeno-style fuzzy inference system.

    Computes crisp output from fuzzy rules using weighted average
    of Sugeno consequents.
    """

    def __init__(self, name: str = "FIS"):
        self.name = name
        self.input_variables: Dict[str, FuzzyVariable] = {}
        self.output_variable: Optional[FuzzyVariable] = None
        self.rules: List[FuzzyRule] = []

    def add_input_variable(self, variable: FuzzyVariable) -> None:
        """Add an input variable."""
        self.input_variables[variable.name] = variable

    def set_output_variable(self, variable: FuzzyVariable) -> None:
        """Set the output variable."""
        self.output_variable = variable

    def add_rule(self, rule: FuzzyRule) -> None:
        """Add a fuzzy rule."""
        self.rules.append(rule)

    def fuzzify_inputs(
        self,
        crisp_inputs: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Fuzzify all input values.

        Args:
            crisp_inputs: Dict mapping variable_name -> crisp_value

        Returns:
            Dict mapping variable_name -> {mf_name: degree}
        """
        result = {}
        for var_name, var in self.input_variables.items():
            if var_name in crisp_inputs:
                result[var_name] = var.fuzzify(crisp_inputs[var_name])
            else:
                result[var_name] = {mf: 0.0 for mf in var.membership_functions}
        return result

    def infer(self, crisp_inputs: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        Perform fuzzy inference.

        Args:
            crisp_inputs: Dict mapping variable_name -> crisp_value

        Returns:
            Tuple of (defuzzified_output, rule_activations)
        """
        # Fuzzify inputs
        fuzzified = self.fuzzify_inputs(crisp_inputs)

        # Evaluate rules
        rule_activations = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for i, rule in enumerate(self.rules):
            firing_strength = rule.evaluate_antecedents(fuzzified)
            rule_activations[f"rule_{i}"] = firing_strength

            if firing_strength > 0 and rule.consequent.sugeno_value is not None:
                weighted_sum += firing_strength * rule.consequent.sugeno_value
                total_weight += firing_strength

        # Defuzzify (weighted average for Sugeno)
        if total_weight > 0:
            output = weighted_sum / total_weight
        else:
            output = 0.0

        return output, rule_activations

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "input_variables": {
                name: var.to_dict()
                for name, var in self.input_variables.items()
            },
            "output_variable": self.output_variable.to_dict() if self.output_variable else None,
            "rules": [rule.to_dict() for rule in self.rules]
        }


class FuzzyPreprocessor:
    """
    Preprocessor that converts raw features to fuzzy membership degrees.

    Expands 60 features × 3 sets = 180 fuzzy values
    """

    # Default fuzzy sets for common indicators
    DEFAULT_FUZZY_SETS = {
        "rsi": [
            ("OVERSOLD", TriangularMF(0, 0, 30)),
            ("NEUTRAL", TriangularMF(20, 50, 80)),
            ("OVERBOUGHT", TriangularMF(70, 100, 100)),
        ],
        "macd_histogram": [
            ("BEARISH", SigmoidMF(0, -5)),
            ("NEUTRAL", GaussianMF(0, 0.5)),
            ("BULLISH", SigmoidMF(0, 5)),
        ],
        "volume_zscore": [
            ("LOW", TriangularMF(-3, -3, 0)),
            ("NORMAL", TriangularMF(-1, 0, 1)),
            ("HIGH", TriangularMF(0, 3, 3)),
        ],
        "funding_rate": [
            ("NEGATIVE", SigmoidMF(0, -10)),
            ("NEUTRAL", GaussianMF(0, 0.001)),
            ("POSITIVE", SigmoidMF(0, 10)),
        ],
        "returns": [
            ("NEGATIVE", SigmoidMF(0, -50)),
            ("NEUTRAL", GaussianMF(0, 0.01)),
            ("POSITIVE", SigmoidMF(0, 50)),
        ],
        "volatility": [
            ("LOW", TriangularMF(0, 0, 0.02)),
            ("MEDIUM", TriangularMF(0.01, 0.03, 0.05)),
            ("HIGH", TriangularMF(0.04, 0.1, 0.1)),
        ],
    }

    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Initialize the fuzzy preprocessor.

        Args:
            feature_names: List of feature names to preprocess
        """
        self.variables: Dict[str, FuzzyVariable] = {}
        self.feature_names = feature_names or []
        self._build_default_variables()

    def _build_default_variables(self) -> None:
        """Build default fuzzy variables for common indicators."""
        for feature_name, fuzzy_sets in self.DEFAULT_FUZZY_SETS.items():
            var = FuzzyVariable(
                name=feature_name,
                universe=(-100, 100)  # Default range
            )
            for mf_name, mf in fuzzy_sets:
                var.add_mf(mf_name, mf)
            self.variables[feature_name] = var

    def add_variable(self, variable: FuzzyVariable) -> None:
        """Add a custom fuzzy variable."""
        self.variables[variable.name] = variable

    def _get_variable_for_feature(self, feature_name: str) -> Optional[FuzzyVariable]:
        """Get the fuzzy variable for a feature, using defaults for common patterns."""
        if feature_name in self.variables:
            return self.variables[feature_name]

        # Try to match common patterns
        feature_lower = feature_name.lower()
        if "rsi" in feature_lower:
            return self.variables.get("rsi")
        elif "macd" in feature_lower or "histogram" in feature_lower:
            return self.variables.get("macd_histogram")
        elif "volume" in feature_lower or "zscore" in feature_lower:
            return self.variables.get("volume_zscore")
        elif "funding" in feature_lower:
            return self.variables.get("funding_rate")
        elif "return" in feature_lower or "pnl" in feature_lower:
            return self.variables.get("returns")
        elif "volatility" in feature_lower or "atr" in feature_lower:
            return self.variables.get("volatility")

        return None

    def fuzzify_feature_vector(
        self,
        raw_features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Convert raw feature vector to fuzzy membership degrees.

        Args:
            raw_features: Dict mapping feature_name -> raw_value

        Returns:
            Expanded dict with fuzzy membership values:
            {feature_name_MF_NAME: degree, ...}
        """
        fuzzy_vector = {}

        for feature_name, value in raw_features.items():
            variable = self._get_variable_for_feature(feature_name)

            if variable is not None:
                memberships = variable.fuzzify(value)
                for mf_name, degree in memberships.items():
                    key = f"{feature_name}_{mf_name}"
                    fuzzy_vector[key] = degree
            else:
                # No fuzzy sets defined, use raw value normalized
                fuzzy_vector[f"{feature_name}_RAW"] = value

        return fuzzy_vector

    def fuzzify_array(
        self,
        raw_features: List[float],
        feature_names: Optional[List[str]] = None
    ) -> List[float]:
        """
        Convert raw feature array to fuzzy membership array.

        Args:
            raw_features: List of raw feature values
            feature_names: Corresponding feature names

        Returns:
            Expanded list of fuzzy membership values
        """
        names = feature_names or self.feature_names
        if len(names) != len(raw_features):
            names = [f"feature_{i}" for i in range(len(raw_features))]

        feature_dict = dict(zip(names, raw_features))
        fuzzy_dict = self.fuzzify_feature_vector(feature_dict)

        return list(fuzzy_dict.values())

    def get_fuzzy_feature_names(
        self,
        raw_feature_names: List[str]
    ) -> List[str]:
        """
        Get the names of fuzzy features for given raw feature names.

        Args:
            raw_feature_names: List of raw feature names

        Returns:
            List of fuzzy feature names (expanded)
        """
        fuzzy_names = []

        for feature_name in raw_feature_names:
            variable = self._get_variable_for_feature(feature_name)

            if variable is not None:
                for mf_name in variable.membership_functions:
                    fuzzy_names.append(f"{feature_name}_{mf_name}")
            else:
                fuzzy_names.append(f"{feature_name}_RAW")

        return fuzzy_names

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "variables": {
                name: var.to_dict()
                for name, var in self.variables.items()
            },
            "feature_names": self.feature_names
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FuzzyPreprocessor':
        """Deserialize from dictionary."""
        preprocessor = cls(feature_names=data.get("feature_names", []))
        # Would need to reconstruct variables from dict
        return preprocessor


# Genetic operators for fuzzy logic evolution
class FuzzyGeneticOperators:
    """Genetic operators for evolving fuzzy configurations."""

    @staticmethod
    def mutate_mf_params(
        preprocessor: FuzzyPreprocessor,
        change_pct: float = 0.1
    ) -> FuzzyPreprocessor:
        """Shift membership function parameters by ±5-10%."""
        if not preprocessor.variables:
            return preprocessor

        var_name = random.choice(list(preprocessor.variables.keys()))
        variable = preprocessor.variables[var_name]

        if not variable.membership_functions:
            return preprocessor

        mf_name = random.choice(list(variable.membership_functions.keys()))
        mf = variable.membership_functions[mf_name]

        params = mf.get_parameters()
        for key in params:
            factor = 1.0 + random.uniform(-change_pct, change_pct)
            params[key] *= factor

        mf.set_parameters(params)
        return preprocessor

    @staticmethod
    def mutate_rule_weight(
        fis: FuzzyInferenceSystem,
        change: float = 0.1
    ) -> FuzzyInferenceSystem:
        """Adjust rule weight by ±0.1."""
        if not fis.rules:
            return fis

        rule_idx = random.randint(0, len(fis.rules) - 1)
        rule = fis.rules[rule_idx]
        rule.weight = max(0.0, min(1.0, rule.weight + random.uniform(-change, change)))

        return fis

    @staticmethod
    def crossover_rules(
        fis1: FuzzyInferenceSystem,
        fis2: FuzzyInferenceSystem
    ) -> Tuple[FuzzyInferenceSystem, FuzzyInferenceSystem]:
        """Exchange rule subsets between two FIS."""
        child1 = FuzzyInferenceSystem(fis1.name)
        child2 = FuzzyInferenceSystem(fis2.name)

        # Copy input variables
        child1.input_variables = dict(fis1.input_variables)
        child2.input_variables = dict(fis2.input_variables)
        child1.output_variable = fis1.output_variable
        child2.output_variable = fis2.output_variable

        # Crossover rules
        all_rules = fis1.rules + fis2.rules
        random.shuffle(all_rules)

        split = len(all_rules) // 2
        child1.rules = all_rules[:split]
        child2.rules = all_rules[split:]

        return child1, child2

    @staticmethod
    def add_rule(fis: FuzzyInferenceSystem) -> FuzzyInferenceSystem:
        """Add a new random rule."""
        if not fis.input_variables:
            return fis

        # Pick 1-3 random antecedents
        num_antecedents = random.randint(1, min(3, len(fis.input_variables)))
        var_names = random.sample(list(fis.input_variables.keys()), num_antecedents)

        antecedents = []
        for var_name in var_names:
            var = fis.input_variables[var_name]
            if var.membership_functions:
                mf_name = random.choice(list(var.membership_functions.keys()))
                antecedents.append(FuzzyAntecedent(var_name, mf_name, negated=random.random() < 0.2))

        # Create consequent
        consequent = FuzzyConsequent(
            variable_name="output",
            mf_name="signal",
            sugeno_value=random.uniform(-1, 1)
        )

        rule = FuzzyRule(
            antecedents=antecedents,
            consequent=consequent,
            operator=random.choice([FuzzyOperator.AND, FuzzyOperator.OR]),
            weight=random.uniform(0.5, 1.0)
        )

        fis.rules.append(rule)
        return fis

    @staticmethod
    def remove_low_weight_rule(
        fis: FuzzyInferenceSystem,
        threshold: float = 0.2
    ) -> FuzzyInferenceSystem:
        """Remove rules with weight below threshold."""
        fis.rules = [r for r in fis.rules if r.weight >= threshold]
        return fis


# Factory functions for common fuzzy variable configurations
def create_rsi_variable() -> FuzzyVariable:
    """Create standard RSI fuzzy variable."""
    var = FuzzyVariable(name="rsi", universe=(0, 100))
    var.add_mf("OVERSOLD", TriangularMF(0, 0, 30))
    var.add_mf("NEUTRAL", TriangularMF(20, 50, 80))
    var.add_mf("OVERBOUGHT", TriangularMF(70, 100, 100))
    return var


def create_momentum_variable() -> FuzzyVariable:
    """Create standard momentum/returns fuzzy variable."""
    var = FuzzyVariable(name="momentum", universe=(-0.1, 0.1))
    var.add_mf("BEARISH", SigmoidMF(0, -50))
    var.add_mf("NEUTRAL", GaussianMF(0, 0.01))
    var.add_mf("BULLISH", SigmoidMF(0, 50))
    return var


def create_signal_fis() -> FuzzyInferenceSystem:
    """Create a basic signal generation FIS."""
    fis = FuzzyInferenceSystem(name="SignalFIS")

    # Add input variables
    fis.add_input_variable(create_rsi_variable())
    fis.add_input_variable(create_momentum_variable())

    # Add output variable
    signal_var = FuzzyVariable(name="signal", universe=(-1, 1))
    signal_var.add_mf("STRONG_SELL", TriangularMF(-1, -1, -0.5))
    signal_var.add_mf("SELL", TriangularMF(-0.8, -0.5, 0))
    signal_var.add_mf("HOLD", GaussianMF(0, 0.2))
    signal_var.add_mf("BUY", TriangularMF(0, 0.5, 0.8))
    signal_var.add_mf("STRONG_BUY", TriangularMF(0.5, 1, 1))
    fis.set_output_variable(signal_var)

    # Add default rules
    fis.add_rule(FuzzyRule(
        antecedents=[
            FuzzyAntecedent("rsi", "OVERSOLD"),
            FuzzyAntecedent("momentum", "BULLISH")
        ],
        consequent=FuzzyConsequent("signal", "STRONG_BUY", sugeno_value=0.9),
        operator=FuzzyOperator.AND,
        weight=0.9
    ))

    fis.add_rule(FuzzyRule(
        antecedents=[
            FuzzyAntecedent("rsi", "OVERBOUGHT"),
            FuzzyAntecedent("momentum", "BEARISH")
        ],
        consequent=FuzzyConsequent("signal", "STRONG_SELL", sugeno_value=-0.9),
        operator=FuzzyOperator.AND,
        weight=0.9
    ))

    fis.add_rule(FuzzyRule(
        antecedents=[FuzzyAntecedent("rsi", "NEUTRAL")],
        consequent=FuzzyConsequent("signal", "HOLD", sugeno_value=0.0),
        operator=FuzzyOperator.AND,
        weight=0.5
    ))

    return fis
