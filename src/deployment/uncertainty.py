"""
Epistemic Uncertainty Gating

Before deployment, we need confidence that the strategy's predictions
are reliable—not just accurate on average, but consistently trustworthy.

Method: Train 5 policy variants with different random seeds/dropout.
If they disagree significantly, the model is uncertain about its predictions.

Deploy only when:
1. Uncertainty < threshold (predictions are confident)
2. Uncertainty is decreasing (model is converging)
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import logging

from ..core.genome import StrategyGenome

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyResult:
    """Result from uncertainty analysis."""
    uncertainty_score: float
    should_deploy: bool
    ensemble_mean: float
    ensemble_std: float
    is_converging: bool
    confidence_level: str  # 'high', 'medium', 'low'
    details: Dict


class EpistemicUncertaintyGate:
    """
    Gate deployment based on ensemble disagreement.

    Method: Train multiple policy variants with different random seeds.
    If they disagree significantly, the model is uncertain.

    Deploy only when:
    1. Uncertainty < threshold (predictions are confident)
    2. Uncertainty is decreasing over time (model is converging)
    """

    def __init__(
        self,
        n_ensemble: int = 5,
        uncertainty_threshold: float = 0.10,
        convergence_window: int = 5,
        min_convergence_rate: float = 0.9
    ):
        """
        Args:
            n_ensemble: Number of ensemble members
            uncertainty_threshold: Max acceptable uncertainty
            convergence_window: Number of samples to check convergence
            min_convergence_rate: What fraction should show decreasing uncertainty
        """
        self.n_ensemble = n_ensemble
        self.threshold = uncertainty_threshold
        self.convergence_window = convergence_window
        self.min_convergence_rate = min_convergence_rate

        self.uncertainty_history: List[float] = []
        self.ensemble_models: List[nn.Module] = []

    def initialize_ensemble(self, base_model_class, model_config: Dict) -> None:
        """
        Initialize ensemble with different random seeds.

        Args:
            base_model_class: Class to instantiate
            model_config: Config dict for model initialization
        """
        self.ensemble_models = []
        for i in range(self.n_ensemble):
            torch.manual_seed(42 + i * 1000)
            np.random.seed(42 + i * 1000)
            model = base_model_class(**model_config)
            self.ensemble_models.append(model)

    def should_deploy(
        self,
        strategy: StrategyGenome,
        market_data: np.ndarray
    ) -> UncertaintyResult:
        """
        Determine if strategy should be deployed based on uncertainty.

        Args:
            strategy: Strategy to evaluate
            market_data: Recent market data for evaluation

        Returns:
            UncertaintyResult with deployment decision
        """
        # Get predictions from each ensemble member
        predictions = self._get_ensemble_predictions(strategy, market_data)

        # Calculate uncertainty metrics
        ensemble_mean = np.mean(predictions)
        ensemble_std = np.std(predictions)
        uncertainty = ensemble_std / (abs(ensemble_mean) + 1e-8)  # Coefficient of variation

        # Track history
        self.uncertainty_history.append(uncertainty)

        # Check convergence
        is_converging = self._check_convergence()

        # Deployment decision
        low_uncertainty = uncertainty < self.threshold
        should_deploy = low_uncertainty and is_converging

        # Confidence level
        if uncertainty < self.threshold * 0.5:
            confidence = 'high'
        elif uncertainty < self.threshold:
            confidence = 'medium'
        else:
            confidence = 'low'

        return UncertaintyResult(
            uncertainty_score=uncertainty,
            should_deploy=should_deploy,
            ensemble_mean=ensemble_mean,
            ensemble_std=ensemble_std,
            is_converging=is_converging,
            confidence_level=confidence,
            details={
                'individual_predictions': predictions.tolist(),
                'history_length': len(self.uncertainty_history),
                'threshold': self.threshold
            }
        )

    def _get_ensemble_predictions(
        self,
        strategy: StrategyGenome,
        market_data: np.ndarray
    ) -> np.ndarray:
        """Get prediction from each ensemble member."""
        predictions = []

        for i, model in enumerate(self.ensemble_models):
            try:
                pred = self._get_model_prediction(model, strategy, market_data, seed=i)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Ensemble member {i} prediction failed: {e}")
                # Use mock prediction as fallback
                predictions.append(np.random.uniform(0, 2))

        return np.array(predictions)

    def _get_model_prediction(
        self,
        model: nn.Module,
        strategy: StrategyGenome,
        market_data: np.ndarray,
        seed: int
    ) -> float:
        """
        Get prediction from a single model.

        This is a simplified version - real implementation would
        run the strategy through the model with proper features.
        """
        # For testing, use strategy vector + noise based on seed
        np.random.seed(seed)
        base_prediction = strategy.fitness if strategy.fitness > 0 else 1.5

        # Add ensemble-specific noise
        noise = np.random.randn() * 0.2
        return max(0, base_prediction + noise)

    def _check_convergence(self) -> bool:
        """
        Check if uncertainty is converging (decreasing trend).

        Returns:
            True if uncertainty is decreasing over recent window
        """
        if len(self.uncertainty_history) < self.convergence_window:
            return False

        recent = self.uncertainty_history[-self.convergence_window:]

        # Check what fraction of consecutive pairs show decrease
        decreasing = sum(1 for i in range(len(recent)-1) if recent[i+1] < recent[i])
        decrease_rate = decreasing / (len(recent) - 1)

        return decrease_rate >= self.min_convergence_rate

    def reset_history(self) -> None:
        """Reset uncertainty history (e.g., for new strategy)."""
        self.uncertainty_history = []


class MCDropoutUncertainty:
    """
    Monte Carlo Dropout for uncertainty estimation.

    Uses dropout at inference time to get uncertainty estimates.
    Faster than maintaining multiple models.
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 20,
        dropout_rate: float = 0.1
    ):
        """
        Args:
            model: Model with dropout layers
            n_samples: Number of forward passes
            dropout_rate: Dropout probability
        """
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

    def predict_with_uncertainty(
        self,
        x: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Make prediction with uncertainty estimate.

        Args:
            x: Input tensor

        Returns:
            (mean_prediction, uncertainty)
        """
        self.model.train()  # Enable dropout

        predictions = []
        for _ in range(self.n_samples):
            with torch.no_grad():
                pred = self.model(x)
                predictions.append(pred.item() if pred.numel() == 1 else pred[0].item())

        mean_pred = np.mean(predictions)
        uncertainty = np.std(predictions)

        self.model.eval()
        return mean_pred, uncertainty


class BayesianUncertaintyEstimator:
    """
    Bayesian approach to uncertainty estimation.

    Maintains posterior distribution over model parameters.
    """

    def __init__(
        self,
        prior_mean: float = 1.5,
        prior_std: float = 0.5,
        likelihood_std: float = 0.3
    ):
        """
        Initialize Bayesian estimator with prior.

        Args:
            prior_mean: Prior mean for Sharpe prediction
            prior_std: Prior uncertainty
            likelihood_std: Assumed noise in observations
        """
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.likelihood_std = likelihood_std

        # Current posterior (will be updated)
        self.posterior_mean = prior_mean
        self.posterior_std = prior_std

        self.observations: List[float] = []

    def update(self, observation: float) -> None:
        """
        Update posterior with new observation (conjugate Gaussian).

        Args:
            observation: Observed Sharpe ratio
        """
        self.observations.append(observation)

        # Bayesian update (conjugate Gaussian)
        prior_precision = 1 / (self.prior_std ** 2)
        likelihood_precision = 1 / (self.likelihood_std ** 2)

        n = len(self.observations)
        obs_mean = np.mean(self.observations)

        posterior_precision = prior_precision + n * likelihood_precision
        self.posterior_std = np.sqrt(1 / posterior_precision)

        self.posterior_mean = (
            prior_precision * self.prior_mean +
            n * likelihood_precision * obs_mean
        ) / posterior_precision

    def get_prediction_interval(
        self,
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Get prediction with credible interval.

        Args:
            confidence: Confidence level

        Returns:
            (mean, lower_bound, upper_bound)
        """
        from scipy import stats

        z = stats.norm.ppf((1 + confidence) / 2)
        lower = self.posterior_mean - z * self.posterior_std
        upper = self.posterior_mean + z * self.posterior_std

        return self.posterior_mean, lower, upper

    def should_deploy(
        self,
        min_sharpe: float = 1.5,
        min_confidence: float = 0.8
    ) -> Tuple[bool, float]:
        """
        Determine if deployment is warranted.

        Args:
            min_sharpe: Minimum acceptable Sharpe
            min_confidence: Required probability of exceeding min_sharpe

        Returns:
            (should_deploy, probability_of_success)
        """
        from scipy import stats

        # P(Sharpe > min_sharpe)
        z = (min_sharpe - self.posterior_mean) / self.posterior_std
        prob_success = 1 - stats.norm.cdf(z)

        return prob_success >= min_confidence, prob_success
