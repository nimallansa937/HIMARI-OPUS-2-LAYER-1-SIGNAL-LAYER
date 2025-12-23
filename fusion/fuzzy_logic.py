"""
Fuzzy Logic Signal Mapping for HIMARI L1

Converts hard indicator thresholds into soft membership functions,
allowing smooth signal generation that avoids whipsaws at boundaries.

Example:
    Instead of: RSI > 70 = SELL (binary)
    Use: μ(RSI, "overbought") = smooth gradient from 0 to 1

Benefits:
    - Smooth, interpretable signal generation
    - Reduces false signals at threshold boundaries
    - Allows combination of multiple fuzzy rules

Latency: <0.1ms per update

Usage:
    fuzzy = FuzzySignalMapper()
    signal = fuzzy.evaluate(rsi=75, volatility=0.02, momentum=1.2)
"""

import math
import json
from typing import Dict, Any, Tuple, List, Callable


class FuzzySet:
    """
    A fuzzy set with membership function.
    
    Supports triangular, trapezoidal, and Gaussian membership.
    """
    
    def __init__(
        self,
        name: str,
        shape: str = 'triangular',
        params: List[float] = None
    ):
        """
        Initialize fuzzy set.
        
        Args:
            name: Name of the fuzzy set (e.g., 'high', 'low')
            shape: 'triangular', 'trapezoidal', 'gaussian', 'sigmoid'
            params: Shape-specific parameters
        """
        self.name = name
        self.shape = shape
        self.params = params or [0, 1, 2]
    
    def membership(self, x: float) -> float:
        """Compute membership degree for value x."""
        if self.shape == 'triangular':
            return self._triangular(x)
        elif self.shape == 'trapezoidal':
            return self._trapezoidal(x)
        elif self.shape == 'gaussian':
            return self._gaussian(x)
        elif self.shape == 'sigmoid':
            return self._sigmoid(x)
        else:
            return 0.0
    
    def _triangular(self, x: float) -> float:
        """Triangular membership: params = [left, peak, right]"""
        left, peak, right = self.params
        
        if x <= left or x >= right:
            return 0.0
        elif x <= peak:
            return (x - left) / (peak - left) if peak != left else 1.0
        else:
            return (right - x) / (right - peak) if right != peak else 1.0
    
    def _trapezoidal(self, x: float) -> float:
        """Trapezoidal membership: params = [left, left_peak, right_peak, right]"""
        a, b, c, d = self.params
        
        if x <= a or x >= d:
            return 0.0
        elif x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        elif x <= c:
            return 1.0
        else:
            return (d - x) / (d - c) if d != c else 1.0
    
    def _gaussian(self, x: float) -> float:
        """Gaussian membership: params = [mean, std]"""
        mean, std = self.params
        return math.exp(-0.5 * ((x - mean) / std) ** 2)
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid membership: params = [center, slope]"""
        center, slope = self.params
        return 1.0 / (1.0 + math.exp(-slope * (x - center)))


class FuzzyVariable:
    """
    A fuzzy variable with multiple fuzzy sets.
    
    Example: RSI variable with sets [oversold, neutral, overbought]
    """
    
    def __init__(self, name: str):
        """
        Initialize fuzzy variable.
        
        Args:
            name: Variable name (e.g., 'rsi', 'momentum')
        """
        self.name = name
        self.sets: Dict[str, FuzzySet] = {}
    
    def add_set(self, fuzzy_set: FuzzySet) -> 'FuzzyVariable':
        """Add a fuzzy set to this variable."""
        self.sets[fuzzy_set.name] = fuzzy_set
        return self
    
    def fuzzify(self, x: float) -> Dict[str, float]:
        """
        Fuzzify a crisp value to membership degrees.
        
        Args:
            x: Input value
            
        Returns:
            Dict mapping set name to membership degree
        """
        return {name: fs.membership(x) for name, fs in self.sets.items()}


class FuzzyRule:
    """
    A fuzzy if-then rule.
    
    Example: IF momentum IS high AND volatility IS low THEN signal IS strong_buy
    """
    
    def __init__(
        self,
        antecedents: List[Tuple[str, str]],  # [(var_name, set_name), ...]
        consequent: Tuple[str, str],  # (output_var, output_set)
        weight: float = 1.0
    ):
        """
        Initialize fuzzy rule.
        
        Args:
            antecedents: List of (variable, set) tuples for conditions
            consequent: (output_variable, output_set) tuple
            weight: Rule importance weight
        """
        self.antecedents = antecedents
        self.consequent = consequent
        self.weight = weight
    
    def evaluate(
        self,
        fuzzified_inputs: Dict[str, Dict[str, float]]
    ) -> float:
        """
        Evaluate rule given fuzzified inputs.
        
        Uses min (AND) to combine antecedent membership degrees.
        
        Returns:
            Firing strength of this rule
        """
        memberships = []
        
        for var_name, set_name in self.antecedents:
            if var_name in fuzzified_inputs:
                degree = fuzzified_inputs[var_name].get(set_name, 0.0)
                memberships.append(degree)
            else:
                return 0.0
        
        if not memberships:
            return 0.0
        
        # AND = min (Gödel t-norm)
        return min(memberships) * self.weight


class FuzzySignalMapper:
    """
    Complete fuzzy logic signal mapper for trading.
    
    Pre-configured with trading-relevant variables and rules.
    """
    
    def __init__(self):
        """Initialize with default trading variables and rules."""
        self.variables: Dict[str, FuzzyVariable] = {}
        self.output_variable: FuzzyVariable = None
        self.rules: List[FuzzyRule] = []
        
        self._setup_default_system()
    
    def _setup_default_system(self):
        """Set up default trading fuzzy system."""
        # RSI variable
        rsi = FuzzyVariable('rsi')
        rsi.add_set(FuzzySet('oversold', 'trapezoidal', [0, 0, 25, 35]))
        rsi.add_set(FuzzySet('neutral', 'triangular', [25, 50, 75]))
        rsi.add_set(FuzzySet('overbought', 'trapezoidal', [65, 75, 100, 100]))
        self.variables['rsi'] = rsi
        
        # Momentum variable (normalized -2 to 2)
        momentum = FuzzyVariable('momentum')
        momentum.add_set(FuzzySet('negative', 'trapezoidal', [-2, -2, -1, 0]))
        momentum.add_set(FuzzySet('neutral', 'triangular', [-0.5, 0, 0.5]))
        momentum.add_set(FuzzySet('positive', 'trapezoidal', [0, 1, 2, 2]))
        self.variables['momentum'] = momentum
        
        # Volatility variable (normalized 0 to 0.1)
        volatility = FuzzyVariable('volatility')
        volatility.add_set(FuzzySet('low', 'trapezoidal', [0, 0, 0.01, 0.02]))
        volatility.add_set(FuzzySet('medium', 'triangular', [0.01, 0.03, 0.05]))
        volatility.add_set(FuzzySet('high', 'trapezoidal', [0.04, 0.06, 0.1, 0.1]))
        self.variables['volatility'] = volatility
        
        # Trend variable (-1 to 1)
        trend = FuzzyVariable('trend')
        trend.add_set(FuzzySet('down', 'trapezoidal', [-1, -1, -0.5, 0]))
        trend.add_set(FuzzySet('flat', 'triangular', [-0.3, 0, 0.3]))
        trend.add_set(FuzzySet('up', 'trapezoidal', [0, 0.5, 1, 1]))
        self.variables['trend'] = trend
        
        # Output signal variable
        signal = FuzzyVariable('signal')
        signal.add_set(FuzzySet('strong_sell', 'triangular', [-1, -1, -0.5]))
        signal.add_set(FuzzySet('sell', 'triangular', [-0.75, -0.5, -0.25]))
        signal.add_set(FuzzySet('neutral', 'triangular', [-0.25, 0, 0.25]))
        signal.add_set(FuzzySet('buy', 'triangular', [0.25, 0.5, 0.75]))
        signal.add_set(FuzzySet('strong_buy', 'triangular', [0.5, 1, 1]))
        self.output_variable = signal
        
        # Define rules
        # Strong buy conditions
        self.rules.append(FuzzyRule(
            [('rsi', 'oversold'), ('momentum', 'positive')],
            ('signal', 'strong_buy'), weight=1.0
        ))
        self.rules.append(FuzzyRule(
            [('trend', 'up'), ('momentum', 'positive'), ('volatility', 'low')],
            ('signal', 'strong_buy'), weight=0.8
        ))
        
        # Buy conditions
        self.rules.append(FuzzyRule(
            [('momentum', 'positive'), ('volatility', 'medium')],
            ('signal', 'buy'), weight=0.6
        ))
        self.rules.append(FuzzyRule(
            [('rsi', 'neutral'), ('trend', 'up')],
            ('signal', 'buy'), weight=0.5
        ))
        
        # Neutral conditions
        self.rules.append(FuzzyRule(
            [('volatility', 'high')],
            ('signal', 'neutral'), weight=0.7
        ))
        self.rules.append(FuzzyRule(
            [('trend', 'flat'), ('momentum', 'neutral')],
            ('signal', 'neutral'), weight=0.8
        ))
        
        # Sell conditions
        self.rules.append(FuzzyRule(
            [('momentum', 'negative'), ('volatility', 'medium')],
            ('signal', 'sell'), weight=0.6
        ))
        self.rules.append(FuzzyRule(
            [('rsi', 'neutral'), ('trend', 'down')],
            ('signal', 'sell'), weight=0.5
        ))
        
        # Strong sell conditions
        self.rules.append(FuzzyRule(
            [('rsi', 'overbought'), ('momentum', 'negative')],
            ('signal', 'strong_sell'), weight=1.0
        ))
        self.rules.append(FuzzyRule(
            [('trend', 'down'), ('momentum', 'negative'), ('volatility', 'low')],
            ('signal', 'strong_sell'), weight=0.8
        ))
    
    def evaluate(self, **inputs) -> float:
        """
        Evaluate fuzzy system with given inputs.
        
        Args:
            **inputs: Variable values (e.g., rsi=75, momentum=1.2)
            
        Returns:
            Defuzzified output signal in [-1, 1]
        """
        # Step 1: Fuzzify inputs
        fuzzified = {}
        for var_name, value in inputs.items():
            if var_name in self.variables:
                fuzzified[var_name] = self.variables[var_name].fuzzify(value)
        
        # Step 2: Evaluate rules
        output_aggregation = {}
        for rule in self.rules:
            firing_strength = rule.evaluate(fuzzified)
            if firing_strength > 0:
                output_set = rule.consequent[1]
                # MAX aggregation for same output set
                output_aggregation[output_set] = max(
                    output_aggregation.get(output_set, 0),
                    firing_strength
                )
        
        # Step 3: Defuzzify using centroid method
        return self._defuzzify_centroid(output_aggregation)
    
    def _defuzzify_centroid(
        self,
        aggregation: Dict[str, float]
    ) -> float:
        """
        Defuzzify using centroid method.
        
        Computes weighted average of output set centers.
        """
        if not aggregation:
            return 0.0
        
        # Output set centers (approximate)
        centers = {
            'strong_sell': -0.875,
            'sell': -0.5,
            'neutral': 0.0,
            'buy': 0.5,
            'strong_buy': 0.875,
        }
        
        numerator = sum(
            centers.get(name, 0) * strength
            for name, strength in aggregation.items()
        )
        denominator = sum(aggregation.values())
        
        if denominator < 1e-10:
            return 0.0
        
        return numerator / denominator
    
    def get_memberships(self, **inputs) -> Dict[str, Dict[str, float]]:
        """
        Get all membership degrees for debugging.
        
        Returns:
            Nested dict of variable -> set -> membership degree
        """
        result = {}
        for var_name, value in inputs.items():
            if var_name in self.variables:
                result[var_name] = self.variables[var_name].fuzzify(value)
        return result
    
    def add_variable(self, variable: FuzzyVariable) -> None:
        """Add a custom fuzzy variable."""
        self.variables[variable.name] = variable
    
    def add_rule(self, rule: FuzzyRule) -> None:
        """Add a custom fuzzy rule."""
        self.rules.append(rule)
    
    def __repr__(self) -> str:
        return (
            f"FuzzySignalMapper(variables={len(self.variables)}, "
            f"rules={len(self.rules)})"
        )
