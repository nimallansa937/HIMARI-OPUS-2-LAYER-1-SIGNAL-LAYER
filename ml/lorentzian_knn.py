"""
Lorentzian K-NN Classification for Market Direction Prediction

Uses Lorentzian distance metric which is robust to fat-tailed distributions
common in financial returns. Unlike Euclidean distance, Lorentzian handles
outliers gracefully by using log transformation.

Lorentzian Distance:
    D(x, y) = Σ log(1 + |x_i - y_i|)

Performance (from HIMARI L1 spec):
    - Win rate: 52-56% vs 48-51% baseline
    - Sharpe contribution: +0.15
    - Latency: <0.5ms (K=20, O(K) = effectively O(1))

Usage:
    knn = LorentzianKNN(centroids=pretrained_centroids, labels=labels)
    for features in feature_stream:
        prob_bullish, confidence = knn.predict(features)
"""

import math
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from collections import deque


class LorentzianKNN:
    """
    Lorentzian distance K-Nearest Neighbors classifier.
    
    Uses pre-trained centroids from K-means clustering on historical data.
    At inference, computes Lorentzian distance to each centroid and
    returns the label of the nearest one with confidence score.
    """
    
    __slots__ = (
        '_centroids', '_labels', '_k',
        '_median_distance', '_last_prediction',
        '_feature_dim', '_prediction_history'
    )
    
    def __init__(
        self,
        centroids: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        k: int = 20,
        feature_dim: int = 15
    ):
        """
        Initialize Lorentzian KNN classifier.
        
        Args:
            centroids: Pre-trained centroid vectors (K x D array)
            labels: Labels for each centroid (K array, 1=bullish, 0=bearish)
            k: Number of centroids
            feature_dim: Dimensionality of feature vectors
        """
        self._k = k
        self._feature_dim = feature_dim
        
        # Initialize with default centroids if not provided
        if centroids is not None:
            self._centroids = np.array(centroids, dtype=np.float64)
            self._labels = np.array(labels, dtype=np.int32)
        else:
            # Create default centroids spread in feature space
            self._centroids = self._create_default_centroids(k, feature_dim)
            # Alternate labels (will need training to be useful)
            self._labels = np.array([i % 2 for i in range(k)], dtype=np.int32)
        
        self._median_distance = 1.0
        self._last_prediction = 0.5
        self._prediction_history = deque(maxlen=100)
    
    def _create_default_centroids(
        self,
        k: int,
        dim: int
    ) -> np.ndarray:
        """Create evenly spaced default centroids."""
        # Random but reproducible initialization
        np.random.seed(42)
        return np.random.randn(k, dim) * 0.5
    
    def lorentzian_distance(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> float:
        """
        Compute Lorentzian distance between two vectors.
        
        D(x, y) = Σ log(1 + |x_i - y_i|)
        
        This is robust to outliers because:
        - Small differences: ≈ linear (log(1+ε) ≈ ε)
        - Large differences: ≈ logarithmic (dampened)
        """
        return np.sum(np.log(1 + np.abs(x - y)))
    
    def predict(
        self,
        features: np.ndarray
    ) -> Tuple[float, float]:
        """
        Predict direction probability and confidence.
        
        Args:
            features: Feature vector (normalized, D dimensions)
            
        Returns:
            (p_bullish, confidence)
            - p_bullish: Probability of bullish direction [0, 1]
            - confidence: Confidence in prediction [0, 1]
        """
        features = np.array(features, dtype=np.float64)
        
        # Compute distances to all centroids
        distances = np.array([
            self.lorentzian_distance(features, centroid)
            for centroid in self._centroids
        ])
        
        # Find k nearest neighbors (or all if k > len)
        k = min(self._k, len(distances))
        nearest_indices = np.argpartition(distances, k-1)[:k]
        nearest_distances = distances[nearest_indices]
        nearest_labels = self._labels[nearest_indices]
        
        # Distance-weighted voting
        weights = 1.0 / (1.0 + nearest_distances)
        total_weight = np.sum(weights)
        
        if total_weight < 1e-10:
            p_bullish = 0.5
        else:
            bullish_weight = np.sum(weights[nearest_labels == 1])
            p_bullish = bullish_weight / total_weight
        
        # Confidence based on distance relative to median
        min_distance = np.min(nearest_distances)
        self._update_median_distance(min_distance)
        
        confidence = 1.0 / (1.0 + min_distance / self._median_distance)
        
        # Store prediction
        self._last_prediction = p_bullish
        self._prediction_history.append(p_bullish)
        
        return p_bullish, confidence
    
    def predict_multi(
        self,
        features: np.ndarray
    ) -> Dict[str, float]:
        """
        Return detailed prediction with regime information.
        
        Returns:
            Dict with p_bullish, p_bearish, confidence, regime_flag
        """
        p_bullish, confidence = self.predict(features)
        
        return {
            'p_bullish': p_bullish,
            'p_bearish': 1.0 - p_bullish,
            'confidence': confidence,
            'regime_flag': self._get_regime_flag(p_bullish, confidence),
            'signal_strength': abs(p_bullish - 0.5) * 2,
        }
    
    def _get_regime_flag(
        self,
        p_bullish: float,
        confidence: float
    ) -> str:
        """Classify market regime from prediction."""
        if confidence < 0.3:
            return 'uncertain'
        
        if p_bullish > 0.6:
            return 'bullish'
        elif p_bullish < 0.4:
            return 'bearish'
        else:
            return 'neutral'
    
    def _update_median_distance(self, distance: float) -> None:
        """Update running median distance estimate via EMA."""
        alpha = 0.1
        self._median_distance = (
            alpha * distance +
            (1 - alpha) * self._median_distance
        )
    
    def update_centroids(
        self,
        new_point: np.ndarray,
        label: int,
        learning_rate: float = 0.01
    ) -> None:
        """
        Online centroid update for adaptive learning.
        
        Moves the nearest same-label centroid slightly toward new point.
        
        Args:
            new_point: New feature observation
            label: Observed label (1=bullish, 0=bearish)
            learning_rate: How much to adjust centroid
        """
        new_point = np.array(new_point, dtype=np.float64)
        
        # Find nearest centroid with same label
        same_label_mask = self._labels == label
        same_label_indices = np.where(same_label_mask)[0]
        
        if len(same_label_indices) == 0:
            return
        
        distances = np.array([
            self.lorentzian_distance(new_point, self._centroids[i])
            for i in same_label_indices
        ])
        
        nearest_idx = same_label_indices[np.argmin(distances)]
        
        # Move centroid toward new point
        self._centroids[nearest_idx] += learning_rate * (
            new_point - self._centroids[nearest_idx]
        )
    
    @property
    def prediction_volatility(self) -> float:
        """Volatility of recent predictions (std dev)."""
        if len(self._prediction_history) < 2:
            return 0.0
        return float(np.std(list(self._prediction_history)))
    
    def get_agreement_score(
        self,
        external_predictions: List[float]
    ) -> float:
        """
        Compute agreement between KNN and external predictions.
        
        Args:
            external_predictions: List of other model predictions [0, 1]
            
        Returns:
            Agreement score [0, 1] where 1 = unanimous
        """
        all_preds = [self._last_prediction] + list(external_predictions)
        
        if not all_preds:
            return 1.0
        
        # Standard deviation of predictions (low = agreement)
        std = np.std(all_preds)
        
        # Map std to agreement: std=0 -> 1.0, std=0.5 -> 0.0
        return max(0.0, 1.0 - 2 * std)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            'centroids': self._centroids.tolist(),
            'labels': self._labels.tolist(),
            'k': self._k,
            'feature_dim': self._feature_dim,
            'median_distance': self._median_distance,
            'last_prediction': self._last_prediction,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LorentzianKNN':
        """Restore from serialized state."""
        instance = cls(
            centroids=np.array(data['centroids']),
            labels=np.array(data['labels']),
            k=data['k'],
            feature_dim=data['feature_dim']
        )
        instance._median_distance = data['median_distance']
        instance._last_prediction = data['last_prediction']
        return instance
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'LorentzianKNN':
        """Restore from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def reset(self) -> None:
        """Reset to default state (keeps centroids)."""
        self._median_distance = 1.0
        self._last_prediction = 0.5
        self._prediction_history.clear()
    
    def __repr__(self) -> str:
        return (
            f"LorentzianKNN(K={self._k}, dim={self._feature_dim}, "
            f"last_pred={self._last_prediction:.2f})"
        )


class StreamingCentroids:
    """
    Streaming K-Means for online centroid updates.
    
    Maintains centroids that adapt to incoming data without
    storing historical observations. Useful for HIMARI to
    adapt to changing market regimes.
    """
    
    def __init__(
        self,
        k: int = 20,
        feature_dim: int = 15,
        decay_rate: float = 0.01
    ):
        """
        Args:
            k: Number of centroids
            feature_dim: Feature dimensionality
            decay_rate: Learning rate for centroid updates
        """
        self.k = k
        self.feature_dim = feature_dim
        self.decay_rate = decay_rate
        
        # Initialize centroids randomly
        np.random.seed(42)
        self.centroids = np.random.randn(k, feature_dim) * 0.5
        self.counts = np.ones(k)  # Per-centroid observation count
        self.labels = np.zeros(k, dtype=np.int32)  # Will be set by labeling
    
    def update(
        self,
        features: np.ndarray,
        label: Optional[int] = None
    ) -> int:
        """
        Update nearest centroid with new observation.
        
        Args:
            features: New feature vector
            label: Optional label for centroid labeling
            
        Returns:
            Index of nearest centroid
        """
        features = np.array(features, dtype=np.float64)
        
        # Find nearest centroid (Euclidean for speed)
        distances = np.sum((self.centroids - features) ** 2, axis=1)
        nearest_idx = np.argmin(distances)
        
        # Update centroid (weighted average)
        count = self.counts[nearest_idx]
        rate = 1.0 / (1.0 + count * self.decay_rate)
        
        self.centroids[nearest_idx] = (
            (1 - rate) * self.centroids[nearest_idx] +
            rate * features
        )
        
        self.counts[nearest_idx] += 1
        
        # Update label if provided
        if label is not None:
            # Majority voting via EMA
            current_label = self.labels[nearest_idx]
            self.labels[nearest_idx] = int(
                round(0.95 * current_label + 0.05 * label)
            )
        
        return nearest_idx
    
    def get_knn_classifier(self) -> LorentzianKNN:
        """Create LorentzianKNN from current centroids."""
        return LorentzianKNN(
            centroids=self.centroids,
            labels=self.labels,
            k=self.k,
            feature_dim=self.feature_dim
        )
