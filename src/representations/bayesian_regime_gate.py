"""
Bayesian Network Regime Gates - Enhancement 4

Probabilistic graphical model for regime detection.
Gates strategy execution based on regime confidence.

Network Structure:
- Observable nodes: VIX level, Trend strength, Volume profile, etc.
- Hidden nodes: Volatility regime, Trend regime
- Output: Market regime (RISK_ON / RISK_OFF / TRANSITIONING)
- Gate action: ALLOW / REDUCE_SIZE / BLOCK
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import random
import math


class VolatilityRegime(Enum):
    """Volatility regime states."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()


class TrendRegime(Enum):
    """Trend regime states."""
    BEAR = auto()
    NEUTRAL = auto()
    BULL = auto()


class MarketRegime(Enum):
    """Overall market regime states."""
    RISK_ON = auto()
    RISK_OFF = auto()
    TRANSITIONING = auto()


class GateAction(Enum):
    """Gate actions for strategy execution."""
    ALLOW = auto()        # Full position sizing
    REDUCE_SIZE = auto()  # Reduce position size
    BLOCK = auto()        # Block trading


@dataclass
class DiscreteVariable:
    """A discrete random variable in the Bayesian network."""
    name: str
    states: List[str]
    parents: List[str] = field(default_factory=list)
    # CPD: Conditional Probability Distribution
    # For root nodes: {state: probability}
    # For child nodes: {(parent_states...): {state: probability}}
    cpd: Dict[Any, Dict[str, float]] = field(default_factory=dict)

    def get_probability(
        self,
        state: str,
        parent_values: Optional[Dict[str, str]] = None
    ) -> float:
        """Get P(state | parent_values)."""
        if not self.parents:
            # Root node
            return self.cpd.get((), {}).get(state, 1.0 / len(self.states))

        if parent_values is None:
            return 1.0 / len(self.states)

        # Build parent state tuple
        parent_tuple = tuple(parent_values.get(p, self.states[0]) for p in self.parents)

        if parent_tuple in self.cpd:
            return self.cpd[parent_tuple].get(state, 1.0 / len(self.states))

        return 1.0 / len(self.states)


class BayesianRegimeGate:
    """
    Bayesian Network for regime detection and strategy gating.

    Uses variable elimination for exact inference on the regime
    given observed market conditions.
    """

    def __init__(self):
        self.variables: Dict[str, DiscreteVariable] = {}
        self.discretizers: Dict[str, Callable] = {}
        self.gate_thresholds = {
            MarketRegime.RISK_OFF: 0.7,     # Block if P(RISK_OFF) > 0.7
            MarketRegime.TRANSITIONING: 0.5  # Reduce if P(TRANSITIONING) > 0.5
        }
        self._build_default_network()

    def _build_default_network(self):
        """Build the default Bayesian network structure."""
        # Observable variables (evidence nodes)
        self.variables["vix_level"] = DiscreteVariable(
            name="vix_level",
            states=["LOW", "MEDIUM", "HIGH"],
            parents=[],
            cpd={(): {"LOW": 0.4, "MEDIUM": 0.4, "HIGH": 0.2}}
        )

        self.variables["trend_strength"] = DiscreteVariable(
            name="trend_strength",
            states=["WEAK", "MODERATE", "STRONG"],
            parents=[],
            cpd={(): {"WEAK": 0.3, "MODERATE": 0.5, "STRONG": 0.2}}
        )

        self.variables["volume_profile"] = DiscreteVariable(
            name="volume_profile",
            states=["DECLINING", "STABLE", "INCREASING"],
            parents=[],
            cpd={(): {"DECLINING": 0.25, "STABLE": 0.5, "INCREASING": 0.25}}
        )

        self.variables["funding_sentiment"] = DiscreteVariable(
            name="funding_sentiment",
            states=["NEGATIVE", "NEUTRAL", "POSITIVE"],
            parents=[],
            cpd={(): {"NEGATIVE": 0.3, "NEUTRAL": 0.4, "POSITIVE": 0.3}}
        )

        self.variables["correlation_regime"] = DiscreteVariable(
            name="correlation_regime",
            states=["DECORRELATED", "NORMAL", "HIGHLY_CORRELATED"],
            parents=[],
            cpd={(): {"DECORRELATED": 0.2, "NORMAL": 0.6, "HIGHLY_CORRELATED": 0.2}}
        )

        # Hidden intermediate variables
        self.variables["volatility_regime"] = DiscreteVariable(
            name="volatility_regime",
            states=["LOW", "MEDIUM", "HIGH"],
            parents=["vix_level", "volume_profile"],
            cpd=self._build_volatility_cpd()
        )

        self.variables["trend_regime"] = DiscreteVariable(
            name="trend_regime",
            states=["BEAR", "NEUTRAL", "BULL"],
            parents=["trend_strength", "funding_sentiment"],
            cpd=self._build_trend_cpd()
        )

        # Market regime (target variable)
        self.variables["market_regime"] = DiscreteVariable(
            name="market_regime",
            states=["RISK_ON", "RISK_OFF", "TRANSITIONING"],
            parents=["volatility_regime", "trend_regime", "correlation_regime"],
            cpd=self._build_market_regime_cpd()
        )

        # Set up discretizers
        self._setup_default_discretizers()

    def _build_volatility_cpd(self) -> Dict[Tuple, Dict[str, float]]:
        """Build CPD for volatility regime."""
        cpd = {}

        # VIX levels × Volume profiles → Volatility regime
        combinations = [
            (("LOW", "DECLINING"), {"LOW": 0.7, "MEDIUM": 0.2, "HIGH": 0.1}),
            (("LOW", "STABLE"), {"LOW": 0.6, "MEDIUM": 0.3, "HIGH": 0.1}),
            (("LOW", "INCREASING"), {"LOW": 0.4, "MEDIUM": 0.4, "HIGH": 0.2}),
            (("MEDIUM", "DECLINING"), {"LOW": 0.3, "MEDIUM": 0.5, "HIGH": 0.2}),
            (("MEDIUM", "STABLE"), {"LOW": 0.2, "MEDIUM": 0.6, "HIGH": 0.2}),
            (("MEDIUM", "INCREASING"), {"LOW": 0.1, "MEDIUM": 0.5, "HIGH": 0.4}),
            (("HIGH", "DECLINING"), {"LOW": 0.1, "MEDIUM": 0.3, "HIGH": 0.6}),
            (("HIGH", "STABLE"), {"LOW": 0.1, "MEDIUM": 0.2, "HIGH": 0.7}),
            (("HIGH", "INCREASING"), {"LOW": 0.05, "MEDIUM": 0.15, "HIGH": 0.8}),
        ]

        for parent_vals, probs in combinations:
            cpd[parent_vals] = probs

        return cpd

    def _build_trend_cpd(self) -> Dict[Tuple, Dict[str, float]]:
        """Build CPD for trend regime."""
        cpd = {}

        # Trend strength × Funding sentiment → Trend regime
        combinations = [
            (("WEAK", "NEGATIVE"), {"BEAR": 0.5, "NEUTRAL": 0.4, "BULL": 0.1}),
            (("WEAK", "NEUTRAL"), {"BEAR": 0.3, "NEUTRAL": 0.5, "BULL": 0.2}),
            (("WEAK", "POSITIVE"), {"BEAR": 0.2, "NEUTRAL": 0.4, "BULL": 0.4}),
            (("MODERATE", "NEGATIVE"), {"BEAR": 0.6, "NEUTRAL": 0.3, "BULL": 0.1}),
            (("MODERATE", "NEUTRAL"), {"BEAR": 0.25, "NEUTRAL": 0.5, "BULL": 0.25}),
            (("MODERATE", "POSITIVE"), {"BEAR": 0.1, "NEUTRAL": 0.3, "BULL": 0.6}),
            (("STRONG", "NEGATIVE"), {"BEAR": 0.7, "NEUTRAL": 0.2, "BULL": 0.1}),
            (("STRONG", "NEUTRAL"), {"BEAR": 0.3, "NEUTRAL": 0.4, "BULL": 0.3}),
            (("STRONG", "POSITIVE"), {"BEAR": 0.1, "NEUTRAL": 0.2, "BULL": 0.7}),
        ]

        for parent_vals, probs in combinations:
            cpd[parent_vals] = probs

        return cpd

    def _build_market_regime_cpd(self) -> Dict[Tuple, Dict[str, float]]:
        """Build CPD for market regime."""
        cpd = {}

        # Build all combinations of parent states
        vol_states = ["LOW", "MEDIUM", "HIGH"]
        trend_states = ["BEAR", "NEUTRAL", "BULL"]
        corr_states = ["DECORRELATED", "NORMAL", "HIGHLY_CORRELATED"]

        for vol in vol_states:
            for trend in trend_states:
                for corr in corr_states:
                    parent_vals = (vol, trend, corr)

                    # Heuristic probability assignment
                    if vol == "HIGH" or corr == "HIGHLY_CORRELATED":
                        if trend == "BEAR":
                            probs = {"RISK_ON": 0.05, "RISK_OFF": 0.8, "TRANSITIONING": 0.15}
                        else:
                            probs = {"RISK_ON": 0.1, "RISK_OFF": 0.6, "TRANSITIONING": 0.3}
                    elif vol == "LOW" and trend == "BULL":
                        probs = {"RISK_ON": 0.7, "RISK_OFF": 0.1, "TRANSITIONING": 0.2}
                    elif vol == "MEDIUM" and trend == "NEUTRAL":
                        probs = {"RISK_ON": 0.3, "RISK_OFF": 0.2, "TRANSITIONING": 0.5}
                    elif trend == "BEAR":
                        probs = {"RISK_ON": 0.2, "RISK_OFF": 0.5, "TRANSITIONING": 0.3}
                    elif trend == "BULL":
                        probs = {"RISK_ON": 0.6, "RISK_OFF": 0.15, "TRANSITIONING": 0.25}
                    else:
                        probs = {"RISK_ON": 0.35, "RISK_OFF": 0.25, "TRANSITIONING": 0.4}

                    cpd[parent_vals] = probs

        return cpd

    def _setup_default_discretizers(self):
        """Set up default discretization functions."""

        def discretize_vix(value: float) -> str:
            if value < 15:
                return "LOW"
            elif value < 25:
                return "MEDIUM"
            else:
                return "HIGH"

        def discretize_trend_strength(value: float) -> str:
            # Assume value is a normalized strength [0, 1]
            if value < 0.33:
                return "WEAK"
            elif value < 0.67:
                return "MODERATE"
            else:
                return "STRONG"

        def discretize_volume(value: float) -> str:
            # Assume value is z-score
            if value < -0.5:
                return "DECLINING"
            elif value < 0.5:
                return "STABLE"
            else:
                return "INCREASING"

        def discretize_funding(value: float) -> str:
            if value < -0.0001:
                return "NEGATIVE"
            elif value < 0.0001:
                return "NEUTRAL"
            else:
                return "POSITIVE"

        def discretize_correlation(value: float) -> str:
            if value < 0.3:
                return "DECORRELATED"
            elif value < 0.7:
                return "NORMAL"
            else:
                return "HIGHLY_CORRELATED"

        self.discretizers = {
            "vix": discretize_vix,
            "volatility": lambda v: discretize_vix(v * 100),  # Convert to VIX-like scale
            "trend_strength": discretize_trend_strength,
            "adx": lambda v: discretize_trend_strength(v / 100),  # ADX is 0-100
            "volume_zscore": discretize_volume,
            "volume": discretize_volume,
            "funding_rate": discretize_funding,
            "correlation": discretize_correlation,
        }

    def discretize_features(
        self,
        raw_features: Dict[str, float]
    ) -> Dict[str, str]:
        """
        Convert continuous features to discrete evidence.

        Args:
            raw_features: Dict of feature_name -> continuous value

        Returns:
            Dict of variable_name -> discrete state
        """
        evidence = {}

        # Map features to observable variables
        feature_to_var = {
            "vix": "vix_level",
            "volatility": "vix_level",
            "atr": "vix_level",
            "trend_strength": "trend_strength",
            "adx": "trend_strength",
            "volume_zscore": "volume_profile",
            "volume": "volume_profile",
            "funding_rate": "funding_sentiment",
            "correlation": "correlation_regime",
        }

        for feature_name, value in raw_features.items():
            feature_lower = feature_name.lower()

            # Find matching discretizer
            for key, discretizer in self.discretizers.items():
                if key in feature_lower:
                    var_name = feature_to_var.get(key)
                    if var_name and var_name not in evidence:
                        try:
                            evidence[var_name] = discretizer(value)
                        except (ValueError, TypeError):
                            pass
                    break

        return evidence

    def _variable_elimination(
        self,
        query_var: str,
        evidence: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Perform variable elimination to compute P(query_var | evidence).

        Simplified implementation for the specific network structure.
        """
        # For our network structure, we can compute exactly

        # Get probabilities for observable variables
        obs_probs = {}
        for var_name, var in self.variables.items():
            if not var.parents:
                if var_name in evidence:
                    # Hard evidence
                    obs_probs[var_name] = {evidence[var_name]: 1.0}
                else:
                    # Use prior
                    obs_probs[var_name] = var.cpd.get((), {})

        # Compute volatility regime distribution
        vol_dist = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        vix_probs = obs_probs.get("vix_level", {"LOW": 0.4, "MEDIUM": 0.4, "HIGH": 0.2})
        vol_probs = obs_probs.get("volume_profile", {"DECLINING": 0.25, "STABLE": 0.5, "INCREASING": 0.25})

        for vix_state, vix_p in vix_probs.items():
            for vol_state, vol_p in vol_probs.items():
                cpd = self.variables["volatility_regime"].cpd.get(
                    (vix_state, vol_state),
                    {"LOW": 0.33, "MEDIUM": 0.34, "HIGH": 0.33}
                )
                for state, p in cpd.items():
                    vol_dist[state] += vix_p * vol_p * p

        # Compute trend regime distribution
        trend_dist = {"BEAR": 0, "NEUTRAL": 0, "BULL": 0}
        str_probs = obs_probs.get("trend_strength", {"WEAK": 0.3, "MODERATE": 0.5, "STRONG": 0.2})
        fund_probs = obs_probs.get("funding_sentiment", {"NEGATIVE": 0.3, "NEUTRAL": 0.4, "POSITIVE": 0.3})

        for str_state, str_p in str_probs.items():
            for fund_state, fund_p in fund_probs.items():
                cpd = self.variables["trend_regime"].cpd.get(
                    (str_state, fund_state),
                    {"BEAR": 0.33, "NEUTRAL": 0.34, "BULL": 0.33}
                )
                for state, p in cpd.items():
                    trend_dist[state] += str_p * fund_p * p

        # Get correlation distribution
        corr_dist = obs_probs.get(
            "correlation_regime",
            {"DECORRELATED": 0.2, "NORMAL": 0.6, "HIGHLY_CORRELATED": 0.2}
        )

        # Compute market regime distribution
        market_dist = {"RISK_ON": 0, "RISK_OFF": 0, "TRANSITIONING": 0}

        for vol_state, vol_p in vol_dist.items():
            for trend_state, trend_p in trend_dist.items():
                for corr_state, corr_p in corr_dist.items():
                    cpd = self.variables["market_regime"].cpd.get(
                        (vol_state, trend_state, corr_state),
                        {"RISK_ON": 0.33, "RISK_OFF": 0.34, "TRANSITIONING": 0.33}
                    )
                    for state, p in cpd.items():
                        market_dist[state] += vol_p * trend_p * corr_p * p

        # Normalize
        total = sum(market_dist.values())
        if total > 0:
            market_dist = {k: v / total for k, v in market_dist.items()}

        return market_dist

    def infer_regime(
        self,
        evidence: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Infer the market regime given observed evidence.

        Args:
            evidence: Dict mapping variable_name -> observed state

        Returns:
            Dict mapping regime state -> probability
        """
        return self._variable_elimination("market_regime", evidence)

    def get_gate_action(
        self,
        regime_probs: Dict[str, float]
    ) -> Tuple[GateAction, Dict[str, Any]]:
        """
        Determine gate action based on regime probabilities.

        Args:
            regime_probs: Dict mapping regime state -> probability

        Returns:
            Tuple of (gate_action, metadata)
        """
        risk_off_prob = regime_probs.get("RISK_OFF", 0)
        transitioning_prob = regime_probs.get("TRANSITIONING", 0)
        risk_on_prob = regime_probs.get("RISK_ON", 0)

        metadata = {
            "regime_probs": regime_probs,
            "dominant_regime": max(regime_probs, key=regime_probs.get) if regime_probs else "UNKNOWN"
        }

        if risk_off_prob > self.gate_thresholds[MarketRegime.RISK_OFF]:
            return GateAction.BLOCK, {**metadata, "reason": "high_risk_off_probability"}

        if transitioning_prob > self.gate_thresholds[MarketRegime.TRANSITIONING]:
            return GateAction.REDUCE_SIZE, {**metadata, "reason": "high_transitioning_probability"}

        return GateAction.ALLOW, {**metadata, "reason": "risk_on_dominant"}

    def process_features(
        self,
        raw_features: Dict[str, float]
    ) -> Tuple[GateAction, float, Dict[str, Any]]:
        """
        Full pipeline: discretize features, infer regime, determine gate action.

        Args:
            raw_features: Dict of feature_name -> continuous value

        Returns:
            Tuple of (gate_action, size_multiplier, metadata)
        """
        # Discretize features
        evidence = self.discretize_features(raw_features)

        # Infer regime
        regime_probs = self.infer_regime(evidence)

        # Get gate action
        action, metadata = self.get_gate_action(regime_probs)

        # Determine size multiplier
        if action == GateAction.BLOCK:
            size_multiplier = 0.0
        elif action == GateAction.REDUCE_SIZE:
            # Reduce proportionally to confidence
            size_multiplier = 0.25 + 0.5 * regime_probs.get("RISK_ON", 0)
        else:
            size_multiplier = 1.0

        metadata["evidence"] = evidence
        metadata["size_multiplier"] = size_multiplier

        return action, size_multiplier, metadata

    def set_gate_threshold(
        self,
        regime: MarketRegime,
        threshold: float
    ) -> None:
        """Set the probability threshold for a regime gate."""
        self.gate_thresholds[regime] = max(0.0, min(1.0, threshold))

    def update_cpd(
        self,
        variable_name: str,
        parent_values: Tuple[str, ...],
        new_probs: Dict[str, float]
    ) -> None:
        """Update a CPD entry."""
        if variable_name in self.variables:
            # Normalize probabilities
            total = sum(new_probs.values())
            if total > 0:
                normalized = {k: v / total for k, v in new_probs.items()}
                self.variables[variable_name].cpd[parent_values] = normalized

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "variables": {
                name: {
                    "states": var.states,
                    "parents": var.parents,
                    "cpd": {str(k): v for k, v in var.cpd.items()}
                }
                for name, var in self.variables.items()
            },
            "gate_thresholds": {r.name: t for r, t in self.gate_thresholds.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BayesianRegimeGate':
        """Deserialize from dictionary."""
        gate = cls()
        # Would need to reconstruct variables from dict
        if "gate_thresholds" in data:
            for regime_name, threshold in data["gate_thresholds"].items():
                regime = MarketRegime[regime_name]
                gate.gate_thresholds[regime] = threshold
        return gate


# Genetic operators for Bayesian network evolution
class BayesianGeneticOperators:
    """Genetic operators for evolving Bayesian network configurations."""

    @staticmethod
    def mutate_cpd(
        gate: BayesianRegimeGate,
        change: float = 0.05
    ) -> BayesianRegimeGate:
        """Adjust CPD probability values by ±0.05."""
        # Pick a random variable with CPD entries
        var_names = [
            name for name, var in gate.variables.items()
            if var.cpd
        ]

        if not var_names:
            return gate

        var_name = random.choice(var_names)
        var = gate.variables[var_name]

        if not var.cpd:
            return gate

        # Pick a random CPD entry
        parent_vals = random.choice(list(var.cpd.keys()))
        probs = dict(var.cpd[parent_vals])

        # Mutate probabilities
        state = random.choice(list(probs.keys()))
        delta = random.uniform(-change, change)
        probs[state] = max(0.01, min(0.99, probs[state] + delta))

        # Normalize
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}

        gate.variables[var_name].cpd[parent_vals] = probs

        return gate

    @staticmethod
    def mutate_threshold(
        gate: BayesianRegimeGate,
        change: float = 0.05
    ) -> BayesianRegimeGate:
        """Adjust gate thresholds."""
        regime = random.choice(list(gate.gate_thresholds.keys()))
        delta = random.uniform(-change, change)
        gate.gate_thresholds[regime] = max(0.3, min(0.9, gate.gate_thresholds[regime] + delta))
        return gate

    @staticmethod
    def crossover(
        gate1: BayesianRegimeGate,
        gate2: BayesianRegimeGate
    ) -> Tuple[BayesianRegimeGate, BayesianRegimeGate]:
        """Exchange CPD subsets between two gates."""
        child1 = BayesianRegimeGate()
        child2 = BayesianRegimeGate()

        # Exchange thresholds
        for regime in gate1.gate_thresholds:
            if random.random() < 0.5:
                child1.gate_thresholds[regime] = gate1.gate_thresholds.get(regime, 0.5)
                child2.gate_thresholds[regime] = gate2.gate_thresholds.get(regime, 0.5)
            else:
                child1.gate_thresholds[regime] = gate2.gate_thresholds.get(regime, 0.5)
                child2.gate_thresholds[regime] = gate1.gate_thresholds.get(regime, 0.5)

        return child1, child2


# Type alias for external use
Callable = type(lambda: None)
