"""
Ensemble Fusion Meta-Learner for HIMARI L1

Combines predictions from multiple base models:
- Kalman Filter trend direction
- Lorentzian KNN probability
- HMM regime probabilities

Uses logistic regression to optimally weight base predictions.

Performance (from HIMARI L1 spec):
    - Sharpe: 1.5-1.9 vs single models
    - Sharpe contribution: +0.15
    - Latency: <2ms (4 multiplications + sigmoid)

Usage:
    ensemble = EnsembleFusion()
    
    # Process with all models
    kalman_pred = kalman_filter.update(price)
    knn_pred, _ = lorentzian.predict(features)
    hmm_state, hmm_conf = hmm.update(ret)
    
    # Fuse signals
    final_signal = ensemble.predict([kalman_pred, knn_pred, hmm_conf])
"""

import math
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


class EnsembleFusion:
    """
    Meta-learner for combining multiple model predictions.
    
    Uses logistic regression:
    p_bullish = σ(w₁·p₁ + w₂·p₂ + w₃·p₃ + ... + bias)
    
    Weights can be set manually or learned from feedback.
    """
    
    __slots__ = (
        '_num_models', '_weights', '_bias',
        '_learning_rate', '_last_output',
        '_model_names', '_component_predictions'
    )
    
    def __init__(
        self,
        num_models: int = 4,
        initial_weights: Optional[List[float]] = None,
        model_names: Optional[List[str]] = None
    ):
        """
        Initialize ensemble fusion.
        
        Args:
            num_models: Number of base models to combine
            initial_weights: Starting weights (equal if None)
            model_names: Names for interpretability
        """
        self._num_models = num_models
        
        if initial_weights is not None:
            self._weights = np.array(initial_weights, dtype=np.float64)
        else:
            # Equal weights initially
            self._weights = np.ones(num_models, dtype=np.float64)
        
        self._bias = 0.0
        self._learning_rate = 0.01
        self._last_output = 0.5
        
        if model_names is not None:
            self._model_names = list(model_names)
        else:
            self._model_names = [f'model_{i}' for i in range(num_models)]
        
        self._component_predictions = np.zeros(num_models)
    
    def predict(
        self,
        predictions: List[float]
    ) -> Tuple[float, float]:
        """
        Combine base model predictions into final signal.
        
        Args:
            predictions: List of probabilities from base models [0, 1]
            
        Returns:
            (ensemble_signal, component_agreement)
            - ensemble_signal: Combined probability [0, 1]
            - component_agreement: Agreement score [0, 1]
        """
        preds = np.array(predictions[:self._num_models], dtype=np.float64)
        self._component_predictions = preds
        
        # Pad if fewer predictions than expected
        if len(preds) < self._num_models:
            preds = np.pad(preds, (0, self._num_models - len(preds)), 
                          constant_values=0.5)
        
        # Logistic regression combination
        # Transform predictions to logits for better combination
        # Clip to avoid log(0) or log(1)
        preds_clipped = np.clip(preds, 0.01, 0.99)
        logits = np.log(preds_clipped / (1 - preds_clipped))
        
        # Weighted combination
        combined_logit = np.dot(self._weights, logits) + self._bias
        
        # Apply sigmoid
        self._last_output = sigmoid(combined_logit)
        
        # Compute agreement (low std = high agreement)
        agreement = self._compute_agreement(preds)
        
        return self._last_output, agreement
    
    def predict_detailed(
        self,
        predictions: List[float]
    ) -> Dict[str, float]:
        """
        Return detailed ensemble prediction with breakdown.
        
        Returns:
            Dict with signal, agreement, component contributions
        """
        signal, agreement = self.predict(predictions)
        
        # Compute individual contributions
        contributions = {}
        for i, (name, pred, weight) in enumerate(
            zip(self._model_names, predictions, self._weights)
        ):
            contributions[f'{name}_prediction'] = pred
            contributions[f'{name}_weight'] = weight
        
        return {
            'ensemble_signal': signal,
            'component_agreement': agreement,
            'signal_direction': 'bullish' if signal > 0.5 else 'bearish',
            'signal_strength': abs(signal - 0.5) * 2,
            'confidence': agreement * abs(signal - 0.5) * 2,
            **contributions
        }
    
    def _compute_agreement(self, predictions: np.ndarray) -> float:
        """
        Compute agreement between component predictions.
        
        Returns:
            Score [0, 1] where 1 = unanimous agreement
        """
        if len(predictions) < 2:
            return 1.0
        
        # Standard deviation of predictions
        std = np.std(predictions)
        
        # Map std to agreement: std=0 -> 1.0, std=0.25 -> 0.5, std=0.5 -> 0
        return max(0.0, 1.0 - 2 * std)
    
    def update_weights(
        self,
        actual_label: int,
        learning_rate: Optional[float] = None
    ) -> None:
        """
        Update weights based on feedback (online learning).
        
        Uses gradient descent on log loss:
        gradient = (predicted - actual) * prediction
        
        Args:
            actual_label: Ground truth (1=bullish, 0=bearish)
            learning_rate: Step size (uses default if None)
        """
        lr = learning_rate or self._learning_rate
        
        # Compute error
        error = self._last_output - actual_label
        
        # Gradient for logistic regression
        gradient = error * self._component_predictions
        
        # Update weights
        self._weights -= lr * gradient
        self._bias -= lr * error
        
        # Regularization: prevent weights from exploding
        self._weights = np.clip(self._weights, -5, 5)
        self._bias = np.clip(self._bias, -2, 2)
    
    def set_weights(
        self,
        weights: List[float],
        bias: float = 0.0
    ) -> None:
        """
        Manually set fusion weights.
        
        Args:
            weights: Weight for each base model
            bias: Bias term
        """
        self._weights = np.array(weights, dtype=np.float64)
        self._bias = bias
    
    def normalize_weights(self) -> None:
        """Normalize weights to sum to 1."""
        total = np.sum(np.abs(self._weights))
        if total > 0:
            self._weights = self._weights / total
    
    @property
    def weights(self) -> np.ndarray:
        """Current weights."""
        return self._weights.copy()
    
    @property
    def dominant_model(self) -> str:
        """Name of model with highest weight."""
        idx = np.argmax(np.abs(self._weights))
        return self._model_names[idx]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get relative importance of each model."""
        abs_weights = np.abs(self._weights)
        total = np.sum(abs_weights)
        
        if total < 1e-10:
            return {name: 1/len(self._model_names) 
                   for name in self._model_names}
        
        importances = abs_weights / total
        return dict(zip(self._model_names, importances))
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            'num_models': self._num_models,
            'weights': self._weights.tolist(),
            'bias': self._bias,
            'learning_rate': self._learning_rate,
            'model_names': self._model_names,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnsembleFusion':
        """Restore from serialized state."""
        instance = cls(
            num_models=data['num_models'],
            initial_weights=data['weights'],
            model_names=data['model_names']
        )
        instance._bias = data['bias']
        instance._learning_rate = data['learning_rate']
        return instance
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'EnsembleFusion':
        """Restore from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def reset(self) -> None:
        """Reset to equal weights."""
        self._weights = np.ones(self._num_models, dtype=np.float64)
        self._bias = 0.0
        self._last_output = 0.5
    
    def __repr__(self) -> str:
        return (
            f"EnsembleFusion(models={self._num_models}, "
            f"dominant={self.dominant_model})"
        )


class VotingEnsemble:
    """
    Simple voting ensemble (majority vote).
    
    More interpretable alternative when weights are uncertain.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        min_votes: int = 2
    ):
        """
        Args:
            threshold: Prediction > threshold counts as bullish
            min_votes: Minimum bullish votes for final bullish signal
        """
        self.threshold = threshold
        self.min_votes = min_votes
    
    def predict(
        self,
        predictions: List[float]
    ) -> Tuple[float, float]:
        """
        Simple majority voting.
        
        Returns:
            (p_bullish, agreement)
        """
        bullish_votes = sum(1 for p in predictions if p > self.threshold)
        total = len(predictions)
        
        if total == 0:
            return 0.5, 0.0
        
        p_bullish = bullish_votes / total
        
        # Agreement = how unanimous
        agreement = abs(p_bullish - 0.5) * 2
        
        return p_bullish, agreement


class StackingEnsemble:
    """
    Stacking ensemble with multiple meta-learners.
    
    Uses first-level predictions + raw features as input
    to second-level model.
    """
    
    def __init__(
        self,
        num_base_models: int = 4,
        num_raw_features: int = 10
    ):
        """
        Args:
            num_base_models: Number of base model predictions
            num_raw_features: Number of raw features to include
        """
        self.num_base = num_base_models
        self.num_raw = num_raw_features
        
        # Total features for meta-learner
        total_features = num_base_models + num_raw_features
        
        # Simple linear meta-learner
        self._weights = np.zeros(total_features)
        self._bias = 0.0
    
    def predict(
        self,
        base_predictions: List[float],
        raw_features: Optional[List[float]] = None
    ) -> Tuple[float, float]:
        """
        Stacking prediction using base outputs + raw features.
        """
        # Combine base predictions and raw features
        if raw_features is None:
            raw_features = [0.5] * self.num_raw
        
        combined = np.concatenate([
            base_predictions[:self.num_base],
            raw_features[:self.num_raw]
        ])
        
        # Pad if needed
        if len(combined) < len(self._weights):
            combined = np.pad(
                combined,
                (0, len(self._weights) - len(combined)),
                constant_values=0.5
            )
        
        # Linear combination + sigmoid
        logit = np.dot(self._weights[:len(combined)], combined) + self._bias
        p_bullish = sigmoid(logit)
        
        # Agreement from base predictions only
        agreement = 1.0 - 2 * np.std(base_predictions)
        agreement = max(0.0, agreement)
        
        return p_bullish, agreement
