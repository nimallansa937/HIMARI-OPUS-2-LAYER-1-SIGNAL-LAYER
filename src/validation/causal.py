"""
Causal Validation Gate

Validates that strategy performance has a causal basis rather than
being spurious correlation. Uses DoWhy-style causal inference with
refutation tests.

This addresses Gap #3 from the gap analysis: Add causal validation
to detect strategies exploiting spurious correlations.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class CausalMechanism(Enum):
    """Types of causal mechanisms a strategy might exploit."""
    MOMENTUM = "momentum"           # Trend continuation
    MEAN_REVERSION = "mean_reversion"  # Price returning to mean
    VOLATILITY = "volatility"       # Volatility clustering/regime
    LIQUIDITY = "liquidity"         # Order flow / liquidity effects
    FUNDING = "funding"             # Funding rate arbitrage
    SENTIMENT = "sentiment"         # Market sentiment dynamics
    MICROSTRUCTURE = "microstructure"  # Order book dynamics
    UNKNOWN = "unknown"


@dataclass
class CausalHypothesis:
    """
    A causal hypothesis about why a strategy works.

    Specifies the presumed cause (signal/feature), effect (returns),
    and mechanism through which the cause produces the effect.
    """
    cause_feature: str              # e.g., "rsi_14"
    effect: str                     # Usually "returns"
    mechanism: CausalMechanism      # Why this should work
    confounders: List[str]          # Potential confounding variables
    mediators: List[str]            # Variables on causal path
    description: str                # Human-readable explanation

    expected_effect_direction: int = 1  # 1 for positive, -1 for negative
    confidence: float = 0.5         # Prior confidence in hypothesis


@dataclass
class RefutationResult:
    """Result from a single refutation test."""
    test_name: str
    passed: bool
    p_value: float
    effect_size_original: float
    effect_size_refuted: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalValidationResult:
    """Result from causal validation."""
    strategy_id: str
    hypothesis: CausalHypothesis
    is_causal: bool
    causal_confidence: float
    estimated_effect: float
    refutation_results: List[RefutationResult]
    recommendation: str
    details: Dict[str, Any] = field(default_factory=dict)


class CausalModel:
    """
    Simplified causal model for strategy validation.

    Estimates causal effect of strategy signals on returns
    and tests robustness via refutation.
    """

    def __init__(
        self,
        treatment: str,
        outcome: str,
        confounders: List[str],
        data: Optional[np.ndarray] = None
    ):
        """
        Initialize causal model.

        Args:
            treatment: Treatment variable (strategy signal)
            outcome: Outcome variable (returns)
            confounders: Potential confounding variables
            data: Optional data matrix (features x time)
        """
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = confounders
        self.data = data

        self._estimated_effect: Optional[float] = None
        self._effect_variance: Optional[float] = None

    def estimate_effect(
        self,
        data: Optional[np.ndarray] = None,
        method: str = "regression"
    ) -> Tuple[float, float]:
        """
        Estimate causal effect of treatment on outcome.

        Args:
            data: Data matrix (if not provided at init)
            method: Estimation method ('regression', 'ipw', 'matching')

        Returns:
            (effect_estimate, standard_error)
        """
        if data is not None:
            self.data = data

        if self.data is None:
            raise ValueError("No data provided for causal estimation")

        if method == "regression":
            return self._estimate_regression()
        elif method == "ipw":
            return self._estimate_ipw()
        elif method == "matching":
            return self._estimate_matching()
        else:
            return self._estimate_regression()

    def _estimate_regression(self) -> Tuple[float, float]:
        """Estimate effect via linear regression with controls."""
        # Simplified: assume data columns are [treatment, outcome, confounders...]
        n_samples = self.data.shape[0]

        if n_samples < 10:
            return 0.0, 1.0

        # Treatment (first column or specified)
        treatment_idx = 0
        outcome_idx = 1

        X = self.data[:, treatment_idx]
        y = self.data[:, outcome_idx]

        # Control for confounders if available
        if self.data.shape[1] > 2:
            # Add confounders as controls
            controls = self.data[:, 2:]
            X_full = np.column_stack([X, controls])
        else:
            X_full = X.reshape(-1, 1)

        # Add intercept
        X_full = np.column_stack([np.ones(n_samples), X_full])

        # OLS estimation
        try:
            beta = np.linalg.lstsq(X_full, y, rcond=None)[0]
            self._estimated_effect = beta[1]  # Treatment coefficient

            # Estimate standard error
            residuals = y - X_full @ beta
            mse = np.mean(residuals ** 2)
            var_beta = mse * np.linalg.inv(X_full.T @ X_full)[1, 1]
            self._effect_variance = var_beta

            return self._estimated_effect, np.sqrt(var_beta)

        except Exception as e:
            logger.warning(f"Regression estimation failed: {e}")
            return 0.0, 1.0

    def _estimate_ipw(self) -> Tuple[float, float]:
        """Estimate effect via inverse propensity weighting."""
        # Simplified IPW implementation
        n_samples = self.data.shape[0]

        if n_samples < 10:
            return 0.0, 1.0

        treatment = self.data[:, 0]
        outcome = self.data[:, 1]

        # Binarize treatment for propensity
        treatment_binary = (treatment > np.median(treatment)).astype(float)

        # Estimate propensity scores (simplified)
        if self.data.shape[1] > 2:
            confounders = self.data[:, 2:]
            # Logistic regression approximation
            propensity = 0.5 + 0.1 * np.mean(confounders, axis=1)
            propensity = np.clip(propensity, 0.1, 0.9)
        else:
            propensity = np.full(n_samples, 0.5)

        # IPW estimator
        weights_treated = treatment_binary / propensity
        weights_control = (1 - treatment_binary) / (1 - propensity)

        effect_treated = np.sum(weights_treated * outcome) / np.sum(weights_treated)
        effect_control = np.sum(weights_control * outcome) / np.sum(weights_control)

        effect = effect_treated - effect_control
        se = np.std(outcome) / np.sqrt(n_samples)  # Simplified SE

        self._estimated_effect = effect
        return effect, se

    def _estimate_matching(self) -> Tuple[float, float]:
        """Estimate effect via nearest-neighbor matching."""
        n_samples = self.data.shape[0]

        if n_samples < 10:
            return 0.0, 1.0

        treatment = self.data[:, 0]
        outcome = self.data[:, 1]

        # Binarize treatment
        median_treatment = np.median(treatment)
        treated_mask = treatment > median_treatment

        treated_outcomes = outcome[treated_mask]
        control_outcomes = outcome[~treated_mask]

        # Simple difference in means (matching on nothing)
        if len(treated_outcomes) > 0 and len(control_outcomes) > 0:
            effect = np.mean(treated_outcomes) - np.mean(control_outcomes)
            se = np.sqrt(
                np.var(treated_outcomes) / len(treated_outcomes) +
                np.var(control_outcomes) / len(control_outcomes)
            )
        else:
            effect, se = 0.0, 1.0

        self._estimated_effect = effect
        return effect, se


class RefutationTest:
    """Base class for refutation tests."""

    def __init__(self, name: str):
        self.name = name

    def run(
        self,
        model: CausalModel,
        original_effect: float
    ) -> RefutationResult:
        """Run the refutation test."""
        raise NotImplementedError


class PlaceboTreatmentTest(RefutationTest):
    """
    Placebo treatment refutation test.

    Replaces real treatment with random placebo and checks
    if estimated effect goes to zero.
    """

    def __init__(self, num_simulations: int = 100):
        super().__init__("placebo_treatment")
        self.num_simulations = num_simulations

    def run(
        self,
        model: CausalModel,
        original_effect: float
    ) -> RefutationResult:
        """Run placebo treatment test."""
        if model.data is None:
            return RefutationResult(
                test_name=self.name,
                passed=False,
                p_value=1.0,
                effect_size_original=original_effect,
                effect_size_refuted=0.0,
                details={"error": "No data available"}
            )

        placebo_effects = []

        for _ in range(self.num_simulations):
            # Create placebo data (random treatment)
            placebo_data = model.data.copy()
            placebo_data[:, 0] = np.random.permutation(placebo_data[:, 0])

            # Estimate effect with placebo
            placebo_model = CausalModel(
                treatment=model.treatment,
                outcome=model.outcome,
                confounders=model.confounders,
                data=placebo_data
            )
            placebo_effect, _ = placebo_model.estimate_effect()
            placebo_effects.append(placebo_effect)

        # Check if original effect is significantly different from placebo
        placebo_mean = np.mean(placebo_effects)
        placebo_std = np.std(placebo_effects) + 1e-8

        # Z-score of original effect
        z_score = (original_effect - placebo_mean) / placebo_std
        p_value = 2 * (1 - self._normal_cdf(abs(z_score)))

        passed = p_value < 0.05  # Original effect survives placebo test

        return RefutationResult(
            test_name=self.name,
            passed=passed,
            p_value=p_value,
            effect_size_original=original_effect,
            effect_size_refuted=placebo_mean,
            details={
                "z_score": z_score,
                "placebo_std": placebo_std,
                "num_simulations": self.num_simulations
            }
        )

    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF."""
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class RandomCauseTest(RefutationTest):
    """
    Random common cause refutation test.

    Adds a random confounder and checks if it significantly
    changes the estimated effect.
    """

    def __init__(self, num_confounders: int = 5):
        super().__init__("random_common_cause")
        self.num_confounders = num_confounders

    def run(
        self,
        model: CausalModel,
        original_effect: float
    ) -> RefutationResult:
        """Run random common cause test."""
        if model.data is None:
            return RefutationResult(
                test_name=self.name,
                passed=False,
                p_value=1.0,
                effect_size_original=original_effect,
                effect_size_refuted=0.0,
                details={"error": "No data available"}
            )

        adjusted_effects = []
        n_samples = model.data.shape[0]

        for _ in range(self.num_confounders):
            # Add random confounder
            random_confounder = np.random.randn(n_samples)
            augmented_data = np.column_stack([model.data, random_confounder])

            # Re-estimate with additional confounder
            augmented_model = CausalModel(
                treatment=model.treatment,
                outcome=model.outcome,
                confounders=model.confounders + ["random_confounder"],
                data=augmented_data
            )
            effect, _ = augmented_model.estimate_effect()
            adjusted_effects.append(effect)

        # Effect should be stable (not change much)
        mean_adjusted = np.mean(adjusted_effects)
        effect_change = abs(original_effect - mean_adjusted) / (abs(original_effect) + 1e-8)

        # Pass if effect doesn't change by more than 20%
        passed = effect_change < 0.20

        # Approximate p-value
        p_value = min(1.0, effect_change * 5)

        return RefutationResult(
            test_name=self.name,
            passed=passed,
            p_value=p_value,
            effect_size_original=original_effect,
            effect_size_refuted=mean_adjusted,
            details={
                "effect_change_pct": effect_change * 100,
                "num_confounders_tested": self.num_confounders
            }
        )


class DataSubsetTest(RefutationTest):
    """
    Data subset refutation test.

    Checks if effect is consistent across different
    subsets of the data (time periods, regimes).
    """

    def __init__(self, num_subsets: int = 5):
        super().__init__("data_subset")
        self.num_subsets = num_subsets

    def run(
        self,
        model: CausalModel,
        original_effect: float
    ) -> RefutationResult:
        """Run data subset test."""
        if model.data is None or len(model.data) < 20:
            return RefutationResult(
                test_name=self.name,
                passed=False,
                p_value=1.0,
                effect_size_original=original_effect,
                effect_size_refuted=0.0,
                details={"error": "Insufficient data"}
            )

        subset_effects = []
        n_samples = model.data.shape[0]
        subset_size = n_samples // self.num_subsets

        for i in range(self.num_subsets):
            start_idx = i * subset_size
            end_idx = start_idx + subset_size
            subset_data = model.data[start_idx:end_idx]

            if len(subset_data) < 5:
                continue

            subset_model = CausalModel(
                treatment=model.treatment,
                outcome=model.outcome,
                confounders=model.confounders,
                data=subset_data
            )
            effect, _ = subset_model.estimate_effect()
            subset_effects.append(effect)

        if not subset_effects:
            return RefutationResult(
                test_name=self.name,
                passed=False,
                p_value=1.0,
                effect_size_original=original_effect,
                effect_size_refuted=0.0,
                details={"error": "Could not create valid subsets"}
            )

        # Check consistency across subsets
        effect_std = np.std(subset_effects)
        mean_effect = np.mean(subset_effects)

        # Coefficient of variation
        cv = effect_std / (abs(mean_effect) + 1e-8)

        # Pass if effect is consistent (low CV)
        passed = cv < 0.5  # Less than 50% variation

        # Check sign consistency
        sign_consistency = np.mean(np.sign(subset_effects) == np.sign(original_effect))
        passed = passed and sign_consistency > 0.6

        p_value = min(1.0, cv)

        return RefutationResult(
            test_name=self.name,
            passed=passed,
            p_value=p_value,
            effect_size_original=original_effect,
            effect_size_refuted=mean_effect,
            details={
                "coefficient_of_variation": cv,
                "sign_consistency": sign_consistency,
                "subset_effects": subset_effects
            }
        )


class CausalValidationGate:
    """
    Causal validation gate for the HIFA pipeline.

    Validates that strategies have genuine causal mechanisms
    rather than exploiting spurious correlations.
    """

    def __init__(
        self,
        min_effect_size: float = 0.01,
        min_refutation_pass_rate: float = 0.66,
        confidence_threshold: float = 0.6
    ):
        """
        Initialize causal validation gate.

        Args:
            min_effect_size: Minimum causal effect size to consider
            min_refutation_pass_rate: Minimum fraction of refutation tests to pass
            confidence_threshold: Minimum confidence in causal relationship
        """
        self.min_effect_size = min_effect_size
        self.min_refutation_pass_rate = min_refutation_pass_rate
        self.confidence_threshold = confidence_threshold

        # Initialize refutation tests
        self.refutation_tests = [
            PlaceboTreatmentTest(num_simulations=50),
            RandomCauseTest(num_confounders=3),
            DataSubsetTest(num_subsets=5)
        ]

    def validate(
        self,
        strategy_id: str,
        hypothesis: CausalHypothesis,
        data: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> CausalValidationResult:
        """
        Validate causal hypothesis for a strategy.

        Args:
            strategy_id: Strategy identifier
            hypothesis: Causal hypothesis to validate
            data: Data matrix [treatment, outcome, confounders...]
            feature_names: Names of features in data

        Returns:
            CausalValidationResult
        """
        logger.info(f"Validating causal hypothesis for strategy {strategy_id[:8]}")

        # Create causal model
        model = CausalModel(
            treatment=hypothesis.cause_feature,
            outcome=hypothesis.effect,
            confounders=hypothesis.confounders,
            data=data
        )

        # Estimate causal effect
        estimated_effect, effect_se = model.estimate_effect()

        # Run refutation tests
        refutation_results = []
        for test in self.refutation_tests:
            result = test.run(model, estimated_effect)
            refutation_results.append(result)

        # Calculate overall causal confidence
        tests_passed = sum(1 for r in refutation_results if r.passed)
        pass_rate = tests_passed / len(refutation_results)

        # Effect size check
        effect_significant = abs(estimated_effect) >= self.min_effect_size

        # Direction check
        direction_correct = (
            np.sign(estimated_effect) == hypothesis.expected_effect_direction
            or hypothesis.expected_effect_direction == 0
        )

        # Overall causal confidence
        causal_confidence = (
            0.4 * pass_rate +
            0.3 * (1.0 if effect_significant else 0.5) +
            0.2 * (1.0 if direction_correct else 0.3) +
            0.1 * hypothesis.confidence
        )

        # Decision
        is_causal = (
            causal_confidence >= self.confidence_threshold and
            pass_rate >= self.min_refutation_pass_rate and
            effect_significant
        )

        # Recommendation
        if is_causal:
            recommendation = "PASS - Causal relationship validated"
        elif pass_rate < self.min_refutation_pass_rate:
            recommendation = f"FAIL - Only passed {tests_passed}/{len(refutation_results)} refutation tests"
        elif not effect_significant:
            recommendation = f"FAIL - Effect size {estimated_effect:.4f} below threshold {self.min_effect_size}"
        else:
            recommendation = f"FAIL - Causal confidence {causal_confidence:.2f} below threshold"

        return CausalValidationResult(
            strategy_id=strategy_id,
            hypothesis=hypothesis,
            is_causal=is_causal,
            causal_confidence=causal_confidence,
            estimated_effect=estimated_effect,
            refutation_results=refutation_results,
            recommendation=recommendation,
            details={
                "effect_se": effect_se,
                "pass_rate": pass_rate,
                "effect_significant": effect_significant,
                "direction_correct": direction_correct
            }
        )

    def infer_hypothesis(
        self,
        strategy,  # StrategyGenome
        feature_names: List[str]
    ) -> CausalHypothesis:
        """
        Infer a causal hypothesis from strategy structure.

        Analyzes the strategy's decision tree to identify
        the primary signal and likely mechanism.
        """
        from ..core.genome import SignalType

        # Find primary signal from decision tree
        primary_signal = None
        if strategy.decision_tree and strategy.decision_tree.condition:
            primary_signal = strategy.decision_tree.condition.signal

        # Map signal to mechanism
        signal_to_mechanism = {
            SignalType.RSI: CausalMechanism.MEAN_REVERSION,
            SignalType.MACD: CausalMechanism.MOMENTUM,
            SignalType.BOLLINGER: CausalMechanism.VOLATILITY,
            SignalType.ATR: CausalMechanism.VOLATILITY,
            SignalType.VOLUME: CausalMechanism.LIQUIDITY,
            SignalType.OBV: CausalMechanism.LIQUIDITY,
            SignalType.FUNDING_RATE: CausalMechanism.FUNDING,
            SignalType.OPEN_INTEREST: CausalMechanism.SENTIMENT,
            SignalType.PRICE_MOMENTUM: CausalMechanism.MOMENTUM,
            SignalType.VWAP: CausalMechanism.MICROSTRUCTURE,
        }

        mechanism = signal_to_mechanism.get(
            primary_signal,
            CausalMechanism.UNKNOWN
        )

        # Determine cause feature
        cause_feature = primary_signal.value if primary_signal else "unknown"

        # Identify potential confounders
        confounders = ["volatility", "volume", "trend"]

        # Generate description
        descriptions = {
            CausalMechanism.MOMENTUM: f"Price momentum measured by {cause_feature} causes future returns through trend continuation",
            CausalMechanism.MEAN_REVERSION: f"Extreme {cause_feature} values cause mean reversion in returns",
            CausalMechanism.VOLATILITY: f"Volatility regime indicated by {cause_feature} affects return distribution",
            CausalMechanism.LIQUIDITY: f"Liquidity measured by {cause_feature} predicts short-term price movements",
            CausalMechanism.FUNDING: f"Funding rate imbalances cause arbitrage opportunities",
            CausalMechanism.SENTIMENT: f"Market sentiment indicated by {cause_feature} drives price dynamics",
            CausalMechanism.MICROSTRUCTURE: f"Microstructure signal {cause_feature} predicts immediate price impact",
            CausalMechanism.UNKNOWN: f"Unknown mechanism relating {cause_feature} to returns"
        }

        return CausalHypothesis(
            cause_feature=cause_feature,
            effect="returns",
            mechanism=mechanism,
            confounders=confounders,
            mediators=[],
            description=descriptions[mechanism],
            expected_effect_direction=1,
            confidence=0.5
        )
