"""
Kernel Recursive Least Squares (KRLS) for Non-Linear Regression

Extends standard RLS to non-linear relationships via kernel functions,
while maintaining O(1) updates through dictionary sparsification.

Key Finding: KRLS achieves 153× lower MSE than deep learning methods
on Indian equity data at 1-minute timeframes (Computational Intelligence
& Neuroscience, 2021).

Usage:
    krls = KernelRLS(kernel='rbf', sigma=1.0)
    for x, y in data:
        prediction = krls.predict(x)
        krls.update(x, y)
        
Memory: O(dictionary_size × feature_dim)
Latency: <1ms per update with sparse dictionary
"""

import math
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable


class KernelRLS:
    """
    Kernel Recursive Least Squares with sparse dictionary.
    
    Approximates non-linear functions in streaming fashion using
    kernel methods. Dictionary sparsification keeps memory bounded.
    """
    
    __slots__ = (
        '_kernel_type', '_sigma', '_lambda', '_nu',
        '_dictionary', '_alpha', '_K_inv',
        '_max_dict_size', '_count'
    )
    
    def __init__(
        self,
        kernel: str = 'rbf',
        sigma: float = 1.0,
        forgetting_factor: float = 0.99,
        novelty_threshold: float = 0.1,
        max_dictionary_size: int = 200
    ):
        """
        Initialize KRLS.
        
        Args:
            kernel: Kernel type ('rbf', 'polynomial', 'linear')
            sigma: RBF kernel bandwidth (larger = smoother)
            forgetting_factor: λ for exponential weighting (0.95-0.999)
            novelty_threshold: ν threshold for adding to dictionary
            max_dictionary_size: Maximum number of support vectors
        """
        self._kernel_type = kernel
        self._sigma = sigma
        self._lambda = forgetting_factor
        self._nu = novelty_threshold
        self._max_dict_size = max_dictionary_size
        
        # Dictionary of support vectors
        self._dictionary: List[np.ndarray] = []
        # Alpha coefficients
        self._alpha: np.ndarray = np.array([])
        # Inverse kernel matrix
        self._K_inv: np.ndarray = np.array([[]])
        
        self._count = 0
    
    def kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute kernel between two vectors.
        
        Args:
            x, y: Feature vectors
            
        Returns:
            Kernel value k(x, y)
        """
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        
        if self._kernel_type == 'rbf':
            # Gaussian RBF: k(x,y) = exp(-||x-y||² / (2σ²))
            diff = x - y
            return math.exp(-np.dot(diff, diff) / (2 * self._sigma ** 2))
        
        elif self._kernel_type == 'polynomial':
            # Polynomial: k(x,y) = (1 + x·y)^d
            degree = 3
            return (1 + np.dot(x, y)) ** degree
        
        elif self._kernel_type == 'linear':
            return np.dot(x, y)
        
        else:
            return np.dot(x, y)
    
    def predict(self, x: np.ndarray) -> float:
        """
        Predict output for input x.
        
        Args:
            x: Input feature vector
            
        Returns:
            Predicted value
        """
        if len(self._dictionary) == 0:
            return 0.0
        
        x = np.asarray(x).flatten()
        
        # Compute kernel with all dictionary elements
        k_vec = np.array([
            self.kernel(x, d) for d in self._dictionary
        ])
        
        return float(np.dot(self._alpha, k_vec))
    
    def update(self, x: np.ndarray, y: float) -> float:
        """
        Update model with new observation.
        
        Uses approximate linear dependency (ALD) test to decide
        whether to add x to dictionary or just update coefficients.
        
        Args:
            x: Input feature vector
            y: Target value
            
        Returns:
            Prediction error
        """
        x = np.asarray(x).flatten()
        self._count += 1
        
        if len(self._dictionary) == 0:
            # First observation - initialize dictionary
            self._dictionary.append(x.copy())
            self._alpha = np.array([y])
            self._K_inv = np.array([[1.0]])
            return y
        
        # Compute kernel vector with dictionary
        k_vec = np.array([self.kernel(x, d) for d in self._dictionary])
        k_xx = self.kernel(x, x)
        
        # Prediction error
        prediction = np.dot(self._alpha, k_vec)
        error = y - prediction
        
        # ALD test: check if x is linearly dependent in feature space
        # Compute: δ = k(x,x) - k_vec' @ K_inv @ k_vec
        a = self._K_inv @ k_vec
        delta = k_xx - np.dot(k_vec, a)
        
        # Apply forgetting factor
        self._K_inv = self._K_inv / self._lambda
        
        if delta > self._nu and len(self._dictionary) < self._max_dict_size:
            # x is novel - add to dictionary
            self._add_to_dictionary(x, k_vec, a, delta, error)
        else:
            # x is approximately dependent - just update coefficients
            self._update_coefficients(a, error, delta)
        
        return error
    
    def _add_to_dictionary(
        self,
        x: np.ndarray,
        k_vec: np.ndarray,
        a: np.ndarray,
        delta: float,
        error: float
    ) -> None:
        """Add new support vector to dictionary."""
        self._dictionary.append(x.copy())
        
        # Expand K_inv using block matrix formula
        n = len(self._K_inv)
        new_K_inv = np.zeros((n + 1, n + 1))
        
        delta_inv = 1.0 / delta
        new_K_inv[:n, :n] = self._K_inv + delta_inv * np.outer(a, a)
        new_K_inv[:n, n] = -delta_inv * a
        new_K_inv[n, :n] = -delta_inv * a
        new_K_inv[n, n] = delta_inv
        
        self._K_inv = new_K_inv
        
        # Expand alpha
        self._alpha = np.append(self._alpha, 0)
        self._alpha += delta_inv * error * np.append(-a, 1)
    
    def _update_coefficients(
        self,
        a: np.ndarray,
        error: float,
        delta: float
    ) -> None:
        """Update coefficients without adding to dictionary."""
        # Regularized update
        q = a / (1 + np.dot(a, a))
        self._alpha += error * q
    
    def prune_dictionary(self, keep_size: int = None) -> int:
        """
        Prune dictionary by removing least important vectors.
        
        Args:
            keep_size: Target dictionary size (default: max_size * 0.8)
            
        Returns:
            Number of vectors removed
        """
        if keep_size is None:
            keep_size = int(self._max_dict_size * 0.8)
        
        if len(self._dictionary) <= keep_size:
            return 0
        
        # Score by |alpha| - importance
        importance = np.abs(self._alpha)
        keep_indices = np.argsort(importance)[-keep_size:]
        keep_indices = np.sort(keep_indices)
        
        removed = len(self._dictionary) - keep_size
        
        # Rebuild with kept vectors
        self._dictionary = [self._dictionary[i] for i in keep_indices]
        self._alpha = self._alpha[keep_indices]
        
        # Rebuild K_inv (expensive but infrequent)
        if len(self._dictionary) > 0:
            K = np.array([
                [self.kernel(d1, d2) for d2 in self._dictionary]
                for d1 in self._dictionary
            ])
            self._K_inv = np.linalg.inv(K + 1e-6 * np.eye(len(self._dictionary)))
        
        return removed
    
    @property
    def dictionary_size(self) -> int:
        """Current dictionary size."""
        return len(self._dictionary)
    
    @property
    def count(self) -> int:
        """Number of observations processed."""
        return self._count
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            'kernel_type': self._kernel_type,
            'sigma': self._sigma,
            'lambda': self._lambda,
            'nu': self._nu,
            'max_dict_size': self._max_dict_size,
            'dictionary': [d.tolist() for d in self._dictionary],
            'alpha': self._alpha.tolist(),
            'K_inv': self._K_inv.tolist(),
            'count': self._count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KernelRLS':
        """Restore from serialized state."""
        instance = cls(
            kernel=data['kernel_type'],
            sigma=data['sigma'],
            forgetting_factor=data['lambda'],
            novelty_threshold=data['nu'],
            max_dictionary_size=data['max_dict_size']
        )
        instance._dictionary = [np.array(d) for d in data['dictionary']]
        instance._alpha = np.array(data['alpha'])
        instance._K_inv = np.array(data['K_inv'])
        instance._count = data['count']
        return instance
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'KernelRLS':
        """Restore from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def reset(self) -> None:
        """Clear all state."""
        self._dictionary = []
        self._alpha = np.array([])
        self._K_inv = np.array([[]])
        self._count = 0
    
    def __repr__(self) -> str:
        return (
            f"KernelRLS(kernel={self._kernel_type}, "
            f"dict_size={len(self._dictionary)}, count={self._count})"
        )


class AdaptiveKRLS(KernelRLS):
    """
    KRLS with adaptive kernel parameters.
    
    Adjusts sigma based on prediction error to adapt to
    changing market regimes.
    """
    
    def __init__(
        self,
        min_sigma: float = 0.1,
        max_sigma: float = 10.0,
        adaptation_rate: float = 0.01,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._min_sigma = min_sigma
        self._max_sigma = max_sigma
        self._adaptation_rate = adaptation_rate
        self._error_ema = 0.0
    
    def update(self, x: np.ndarray, y: float) -> float:
        """Update with adaptive sigma adjustment."""
        error = super().update(x, y)
        
        # Track error EMA
        self._error_ema = (
            0.1 * abs(error) +
            0.9 * self._error_ema
        )
        
        # Adjust sigma based on error
        if self._error_ema > 0.1:  # High error - need smoother
            self._sigma = min(
                self._sigma * (1 + self._adaptation_rate),
                self._max_sigma
            )
        elif self._error_ema < 0.01:  # Low error - can be more local
            self._sigma = max(
                self._sigma * (1 - self._adaptation_rate),
                self._min_sigma
            )
        
        return error
