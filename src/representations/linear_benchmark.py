"""
Linear Model Benchmark - Enhancement 5

Simple linear models for baseline comparison and fast fallback.
Measures whether complex strategies actually beat simple weighted indicators.

Model Types:
- Ridge Regression (L2): Handles multicollinearity
- Lasso Regression (L1): Sparse feature selection
- Elastic Net: Combines L1+L2
- Logistic Regression: For classification

Performance Target: <1 microsecond inference
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum, auto
import math
import random


class ModelType(Enum):
    """Types of linear models."""
    RIDGE = auto()      # L2 regularization
    LASSO = auto()      # L1 regularization
    ELASTIC_NET = auto() # L1 + L2
    LOGISTIC = auto()   # For classification


class BenchmarkResult(Enum):
    """Result of benchmark comparison."""
    BEATS_BENCHMARK = auto()
    MATCHES = auto()
    UNDERPERFORMS = auto()


@dataclass
class ModelCoefficients:
    """Coefficients for a linear model."""
    weights: Dict[str, float]  # Feature name -> coefficient
    bias: float = 0.0
    regularization: float = 0.01  # Lambda parameter


class LinearModel:
    """
    A simple linear model for fast inference.

    y = sum(w_i * x_i) + bias
    """

    def __init__(
        self,
        model_type: ModelType = ModelType.RIDGE,
        regularization: float = 0.01
    ):
        self.model_type = model_type
        self.regularization = regularization
        self.weights: Dict[str, float] = {}
        self.bias: float = 0.0
        self.feature_names: List[str] = []
        self.feature_mean: Dict[str, float] = {}
        self.feature_std: Dict[str, float] = {}
        self._is_fitted = False

    def fit(
        self,
        X: List[Dict[str, float]],
        y: List[float],
        max_iterations: int = 100,
        learning_rate: float = 0.01
    ) -> 'LinearModel':
        """
        Fit the linear model using gradient descent.

        Args:
            X: List of feature dictionaries
            y: List of target values
            max_iterations: Maximum training iterations
            learning_rate: Learning rate for gradient descent

        Returns:
            self
        """
        if not X or not y:
            return self

        # Extract feature names from first sample
        self.feature_names = list(X[0].keys())

        # Compute feature statistics for standardization
        for feature in self.feature_names:
            values = [sample.get(feature, 0) for sample in X]
            self.feature_mean[feature] = sum(values) / len(values)
            variance = sum((v - self.feature_mean[feature]) ** 2 for v in values) / len(values)
            self.feature_std[feature] = math.sqrt(variance) if variance > 0 else 1.0

        # Initialize weights
        for feature in self.feature_names:
            self.weights[feature] = random.uniform(-0.1, 0.1)
        self.bias = 0.0

        # Standardize features
        X_scaled = []
        for sample in X:
            scaled = {}
            for feature in self.feature_names:
                value = sample.get(feature, 0)
                scaled[feature] = (value - self.feature_mean[feature]) / self.feature_std[feature]
            X_scaled.append(scaled)

        # Gradient descent
        n_samples = len(X_scaled)

        for iteration in range(max_iterations):
            # Compute predictions
            predictions = [self._predict_scaled(sample) for sample in X_scaled]

            # Compute gradients
            gradient_weights = {f: 0.0 for f in self.feature_names}
            gradient_bias = 0.0

            for i, (sample, pred, target) in enumerate(zip(X_scaled, predictions, y)):
                error = pred - target

                for feature in self.feature_names:
                    gradient_weights[feature] += error * sample.get(feature, 0)

                gradient_bias += error

            # Average gradients
            for feature in self.feature_names:
                gradient_weights[feature] /= n_samples

            gradient_bias /= n_samples

            # Add regularization gradient
            if self.model_type == ModelType.RIDGE:
                for feature in self.feature_names:
                    gradient_weights[feature] += self.regularization * self.weights[feature]
            elif self.model_type == ModelType.LASSO:
                for feature in self.feature_names:
                    if self.weights[feature] > 0:
                        gradient_weights[feature] += self.regularization
                    elif self.weights[feature] < 0:
                        gradient_weights[feature] -= self.regularization
            elif self.model_type == ModelType.ELASTIC_NET:
                alpha = 0.5  # Mix ratio
                for feature in self.feature_names:
                    l2_grad = self.regularization * (1 - alpha) * self.weights[feature]
                    l1_grad = self.regularization * alpha * (1 if self.weights[feature] > 0 else -1)
                    gradient_weights[feature] += l2_grad + l1_grad

            # Update weights
            for feature in self.feature_names:
                self.weights[feature] -= learning_rate * gradient_weights[feature]

            self.bias -= learning_rate * gradient_bias

            # Apply L1 soft thresholding for Lasso
            if self.model_type in (ModelType.LASSO, ModelType.ELASTIC_NET):
                threshold = self.regularization * learning_rate
                for feature in self.feature_names:
                    if abs(self.weights[feature]) < threshold:
                        self.weights[feature] = 0.0

        self._is_fitted = True
        return self

    def _predict_scaled(self, scaled_sample: Dict[str, float]) -> float:
        """Predict from scaled features."""
        result = self.bias
        for feature, weight in self.weights.items():
            result += weight * scaled_sample.get(feature, 0)

        if self.model_type == ModelType.LOGISTIC:
            # Apply sigmoid for classification
            result = 1.0 / (1.0 + math.exp(-max(-700, min(700, result))))

        return result

    def predict(self, features: Dict[str, float]) -> float:
        """
        Make a prediction from raw features.

        Args:
            features: Dict of feature_name -> value

        Returns:
            Predicted value
        """
        if not self._is_fitted:
            return 0.0

        # Scale features
        scaled = {}
        for feature in self.feature_names:
            value = features.get(feature, 0)
            mean = self.feature_mean.get(feature, 0)
            std = self.feature_std.get(feature, 1)
            scaled[feature] = (value - mean) / std

        return self._predict_scaled(scaled)

    def predict_with_confidence(
        self,
        features: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Make a prediction with confidence estimate.

        Args:
            features: Dict of feature_name -> value

        Returns:
            Tuple of (prediction, confidence)
        """
        prediction = self.predict(features)

        # Simple confidence based on prediction magnitude
        if self.model_type == ModelType.LOGISTIC:
            confidence = abs(prediction - 0.5) * 2  # Distance from decision boundary
        else:
            # Confidence based on how extreme the prediction is
            confidence = min(1.0, abs(prediction))

        return prediction, confidence

    def get_feature_importance(self) -> List[Tuple[str, float, float]]:
        """
        Get feature importance based on coefficient magnitudes.

        Returns:
            List of (feature_name, coefficient, abs_coefficient) sorted by importance
        """
        importance = [
            (name, coef, abs(coef))
            for name, coef in self.weights.items()
        ]
        return sorted(importance, key=lambda x: x[2], reverse=True)

    def get_non_zero_features(self) -> List[str]:
        """Get features with non-zero coefficients (for sparse models)."""
        return [name for name, coef in self.weights.items() if abs(coef) > 1e-10]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_type": self.model_type.name,
            "regularization": self.regularization,
            "weights": self.weights,
            "bias": self.bias,
            "feature_names": self.feature_names,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "is_fitted": self._is_fitted
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LinearModel':
        """Deserialize from dictionary."""
        model = cls(
            model_type=ModelType[data.get("model_type", "RIDGE")],
            regularization=data.get("regularization", 0.01)
        )
        model.weights = data.get("weights", {})
        model.bias = data.get("bias", 0.0)
        model.feature_names = data.get("feature_names", [])
        model.feature_mean = data.get("feature_mean", {})
        model.feature_std = data.get("feature_std", {})
        model._is_fitted = data.get("is_fitted", False)
        return model


class LinearBenchmark:
    """
    Benchmark manager for comparing strategies against linear models.

    Maintains multiple linear model types and tracks performance.
    """

    def __init__(self):
        self.models: Dict[str, LinearModel] = {
            "ridge": LinearModel(ModelType.RIDGE, regularization=0.01),
            "lasso": LinearModel(ModelType.LASSO, regularization=0.01),
            "elastic": LinearModel(ModelType.ELASTIC_NET, regularization=0.01),
            "logistic": LinearModel(ModelType.LOGISTIC, regularization=0.01),
        }
        self.coefficients_history: List[Dict[str, Dict[str, float]]] = []
        self.performance_history: List[Dict[str, float]] = []

    def fit(
        self,
        X: List[Dict[str, float]],
        y: List[float],
        model_type: str = "elastic"
    ) -> None:
        """
        Fit a specific linear model.

        Args:
            X: List of feature dictionaries
            y: List of target values
            model_type: Which model to fit ('ridge', 'lasso', 'elastic', 'logistic')
        """
        if model_type in self.models:
            self.models[model_type].fit(X, y)

            # Record coefficients for stability tracking
            self.coefficients_history.append({
                model_type: dict(self.models[model_type].weights)
            })

    def fit_all(
        self,
        X: List[Dict[str, float]],
        y: List[float]
    ) -> None:
        """Fit all linear models."""
        for model_type in self.models:
            self.fit(X, y, model_type)

    def predict(
        self,
        features: Dict[str, float],
        model_type: str = "elastic"
    ) -> Tuple[float, float]:
        """
        Make a prediction with the specified model.

        Args:
            features: Dict of feature_name -> value
            model_type: Which model to use

        Returns:
            Tuple of (prediction, confidence)
        """
        if model_type not in self.models:
            return 0.0, 0.0

        return self.models[model_type].predict_with_confidence(features)

    def get_feature_importance(
        self,
        model_type: str = "elastic"
    ) -> List[Tuple[str, float, float]]:
        """Get feature importance from the specified model."""
        if model_type not in self.models:
            return []
        return self.models[model_type].get_feature_importance()

    def compute_sharpe(
        self,
        returns: List[float],
        risk_free_rate: float = 0.0
    ) -> float:
        """
        Compute Sharpe ratio for a series of returns.

        Args:
            returns: List of period returns
            risk_free_rate: Risk-free rate per period

        Returns:
            Annualized Sharpe ratio
        """
        if not returns or len(returns) < 2:
            return 0.0

        mean_return = sum(returns) / len(returns)
        excess_return = mean_return - risk_free_rate

        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        std_return = math.sqrt(variance) if variance > 0 else 1e-10

        # Annualize (assuming daily returns, 252 trading days)
        sharpe = (excess_return / std_return) * math.sqrt(252)

        return sharpe

    def compare_to_strategy(
        self,
        strategy_returns: List[float],
        linear_returns: List[float],
        min_improvement: float = 0.1
    ) -> Tuple[BenchmarkResult, Dict[str, float]]:
        """
        Compare strategy returns to linear benchmark returns.

        Args:
            strategy_returns: Returns from the strategy
            linear_returns: Returns from the linear benchmark
            min_improvement: Minimum Sharpe improvement to beat benchmark

        Returns:
            Tuple of (result, metrics)
        """
        strategy_sharpe = self.compute_sharpe(strategy_returns)
        linear_sharpe = self.compute_sharpe(linear_returns)

        # Information ratio
        if linear_returns and len(linear_returns) == len(strategy_returns):
            tracking_diff = [s - l for s, l in zip(strategy_returns, linear_returns)]
            tracking_mean = sum(tracking_diff) / len(tracking_diff)
            tracking_var = sum((d - tracking_mean) ** 2 for d in tracking_diff) / len(tracking_diff)
            tracking_std = math.sqrt(tracking_var) if tracking_var > 0 else 1e-10
            information_ratio = (tracking_mean / tracking_std) * math.sqrt(252)
        else:
            information_ratio = 0.0

        metrics = {
            "strategy_sharpe": strategy_sharpe,
            "linear_sharpe": linear_sharpe,
            "sharpe_difference": strategy_sharpe - linear_sharpe,
            "information_ratio": information_ratio
        }

        if strategy_sharpe >= linear_sharpe + min_improvement:
            return BenchmarkResult.BEATS_BENCHMARK, metrics
        elif abs(strategy_sharpe - linear_sharpe) < min_improvement:
            return BenchmarkResult.MATCHES, metrics
        else:
            return BenchmarkResult.UNDERPERFORMS, metrics

    def partial_fit(
        self,
        features: Dict[str, float],
        target: float,
        model_type: str = "elastic",
        learning_rate: float = 0.001
    ) -> None:
        """
        Online update of the model with a single sample.

        Args:
            features: Dict of feature_name -> value
            target: Target value
            model_type: Which model to update
            learning_rate: Learning rate for update
        """
        if model_type not in self.models:
            return

        model = self.models[model_type]
        if not model._is_fitted:
            return

        # Scale features
        scaled = {}
        for feature in model.feature_names:
            value = features.get(feature, 0)
            mean = model.feature_mean.get(feature, 0)
            std = model.feature_std.get(feature, 1)
            scaled[feature] = (value - mean) / std

        # Compute prediction and error
        prediction = model._predict_scaled(scaled)
        error = prediction - target

        # Update weights (SGD step)
        for feature in model.feature_names:
            gradient = error * scaled.get(feature, 0)

            # Add regularization
            if model.model_type == ModelType.RIDGE:
                gradient += model.regularization * model.weights[feature]
            elif model.model_type == ModelType.LASSO:
                if model.weights[feature] > 0:
                    gradient += model.regularization
                elif model.weights[feature] < 0:
                    gradient -= model.regularization

            model.weights[feature] -= learning_rate * gradient

        model.bias -= learning_rate * error

    def check_coefficient_drift(
        self,
        model_type: str = "elastic",
        threshold: float = 0.5
    ) -> Tuple[bool, List[str]]:
        """
        Check if coefficients have drifted significantly.

        Args:
            model_type: Which model to check
            threshold: Maximum allowed drift ratio

        Returns:
            Tuple of (has_drift, drifted_features)
        """
        if len(self.coefficients_history) < 2:
            return False, []

        # Get recent and older coefficients
        recent = self.coefficients_history[-1].get(model_type, {})
        older = self.coefficients_history[-2].get(model_type, {})

        if not recent or not older:
            return False, []

        drifted = []
        for feature in recent:
            if feature in older:
                old_val = older[feature]
                new_val = recent[feature]

                if abs(old_val) > 1e-10:
                    drift_ratio = abs((new_val - old_val) / old_val)
                    if drift_ratio > threshold:
                        drifted.append(feature)

        return len(drifted) > 0, drifted

    def get_stable_features(
        self,
        model_type: str = "lasso"
    ) -> List[str]:
        """
        Get features that are consistently non-zero across history.

        Uses Lasso model which tends to zero out irrelevant features.
        """
        if model_type not in self.models:
            return []

        # Get current non-zero features
        return self.models[model_type].get_non_zero_features()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "models": {
                name: model.to_dict()
                for name, model in self.models.items()
            },
            "coefficients_history": self.coefficients_history[-10:],  # Keep last 10
            "performance_history": self.performance_history[-100:]  # Keep last 100
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LinearBenchmark':
        """Deserialize from dictionary."""
        benchmark = cls()
        if "models" in data:
            for name, model_data in data["models"].items():
                if name in benchmark.models:
                    benchmark.models[name] = LinearModel.from_dict(model_data)
        benchmark.coefficients_history = data.get("coefficients_history", [])
        benchmark.performance_history = data.get("performance_history", [])
        return benchmark


# Utility functions for benchmark integration
def evaluate_linear_benchmark(
    benchmark: LinearBenchmark,
    features: Dict[str, float],
    model_type: str = "elastic"
) -> Tuple[float, float]:
    """
    Evaluate the linear benchmark on given features.

    Args:
        benchmark: The LinearBenchmark instance
        features: Feature dictionary
        model_type: Which model to use

    Returns:
        Tuple of (signal, confidence)
    """
    prediction, confidence = benchmark.predict(features, model_type)

    # Convert prediction to signal [-1, 1]
    signal = max(-1, min(1, prediction))

    return signal, confidence


def should_use_linear_fallback(
    strategy_confidence: float,
    threshold: float = 0.3
) -> bool:
    """
    Determine if linear fallback should be used.

    Args:
        strategy_confidence: Confidence from main strategy
        threshold: Minimum confidence to use main strategy

    Returns:
        True if fallback should be used
    """
    return strategy_confidence < threshold
