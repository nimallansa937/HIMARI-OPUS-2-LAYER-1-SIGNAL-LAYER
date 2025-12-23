"""
Recursive Least Squares (RLS) for O(1) Linear Regression

Standard linear regression requires O(N) computation to recalculate
slope and intercept when new data arrives. RLS maintains these estimates
incrementally, updating in O(p²) time where p is the number of parameters
(typically 2 for slope/intercept).

With forgetting factor λ, the effective memory window is 1/(1-λ):
- λ = 0.99 → ~100 sample effective window
- λ = 0.95 → ~20 sample effective window

Usage:
    rls = RecursiveLeastSquares(forgetting_factor=0.99)
    for i, price in enumerate(prices):
        slope, intercept = rls.update(i, price)
        predicted = rls.predict(i + 1)  # Next step prediction

Memory: ~200 bytes
Latency: ~100ns per update
"""

import math
import json
from typing import Tuple, Dict, Any, Optional
import numpy as np


class RecursiveLeastSquares:
    """
    RLS for online linear regression: y = slope*x + intercept
    
    The algorithm maintains:
    - θ: parameter vector [slope, intercept]
    - P: inverse correlation matrix (2x2)
    
    Update equations (simplified for 2D):
        gain = P @ x / (λ + x.T @ P @ x)
        error = y - θ.T @ x
        θ = θ + gain * error
        P = (P - gain @ x.T @ P) / λ
        
    Parameters:
        forgetting_factor (λ): Controls memory. Values close to 1 give
            longer memory. Range: 0.9 to 0.9999. Default 0.99 gives
            effective window of ~100 samples.
            
        initial_variance: Initial diagonal values of P matrix.
            Higher values = faster initial adaptation.
    """
    
    __slots__ = (
        'forgetting_factor', '_theta', '_P', '_count',
        '_last_x', '_last_y', '_initialized'
    )
    
    def __init__(
        self,
        forgetting_factor: float = 0.99,
        initial_variance: float = 100.0
    ):
        """
        Initialize RLS estimator.
        
        Args:
            forgetting_factor: λ in (0, 1]. Effective window = 1/(1-λ)
            initial_variance: Initial P matrix diagonal (higher = faster adaptation)
        """
        if not 0 < forgetting_factor <= 1:
            raise ValueError("forgetting_factor must be in (0, 1]")
            
        self.forgetting_factor = forgetting_factor
        
        # Parameter vector: [slope, intercept]
        self._theta = np.array([0.0, 0.0])
        
        # Inverse correlation matrix (2x2)
        # Initialize to large diagonal for fast initial learning
        self._P = np.array([
            [initial_variance, 0.0],
            [0.0, initial_variance]
        ])
        
        self._count = 0
        self._last_x = 0.0
        self._last_y = 0.0
        self._initialized = False
    
    def update(self, x: float, y: float) -> Tuple[float, float]:
        """
        Update regression with new (x, y) observation.
        
        For time series: x is often the index/timestamp, y is the price.
        
        Args:
            x: Independent variable (e.g., time index)
            y: Dependent variable (e.g., price)
            
        Returns:
            (slope, intercept) of current regression line
        """
        self._count += 1
        self._last_x = x
        self._last_y = y
        
        # Feature vector: [x, 1] (for slope*x + intercept*1)
        phi = np.array([x, 1.0])
        
        # Prediction error
        y_pred = np.dot(self._theta, phi)
        error = y - y_pred
        
        # Intermediate calculation
        P_phi = np.dot(self._P, phi)  # P @ φ
        denom = self.forgetting_factor + np.dot(phi, P_phi)  # λ + φᵀPφ
        
        # Kalman-like gain
        gain = P_phi / denom
        
        # Update parameter estimates
        self._theta = self._theta + gain * error
        
        # Update inverse correlation matrix
        self._P = (self._P - np.outer(gain, np.dot(phi, self._P))) / self.forgetting_factor
        
        self._initialized = True
        
        return (self._theta[0], self._theta[1])  # (slope, intercept)
    
    def predict(self, x: float) -> float:
        """
        Predict y for given x using current regression.
        
        Args:
            x: Value to predict at
            
        Returns:
            Predicted y value
        """
        return self._theta[0] * x + self._theta[1]
    
    @property
    def slope(self) -> float:
        """Current slope estimate."""
        return self._theta[0]
    
    @property
    def intercept(self) -> float:
        """Current intercept estimate."""
        return self._theta[1]
    
    @property
    def r_squared(self) -> float:
        """
        Approximate R² based on prediction variance.
        
        Note: This is an approximation since we don't store all data.
        """
        # Use trace of P as proxy for uncertainty
        uncertainty = np.trace(self._P)
        # Lower uncertainty = higher R²
        return 1.0 / (1.0 + uncertainty / 100)
    
    def get_trend_direction(self) -> str:
        """
        Get qualitative trend direction.
        
        Returns:
            'UP', 'DOWN', or 'FLAT' based on slope magnitude
        """
        if self.slope > 0.0001:
            return 'UP'
        elif self.slope < -0.0001:
            return 'DOWN'
        return 'FLAT'
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for Redis persistence."""
        return {
            'forgetting_factor': self.forgetting_factor,
            'theta': self._theta.tolist(),
            'P': self._P.tolist(),
            'count': self._count,
            'last_x': self._last_x,
            'last_y': self._last_y,
            'initialized': self._initialized,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RecursiveLeastSquares':
        """Restore from serialized state."""
        instance = cls(forgetting_factor=data['forgetting_factor'])
        instance._theta = np.array(data['theta'])
        instance._P = np.array(data['P'])
        instance._count = data['count']
        instance._last_x = data['last_x']
        instance._last_y = data['last_y']
        instance._initialized = data['initialized']
        return instance
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'RecursiveLeastSquares':
        return cls.from_dict(json.loads(json_str))
    
    def reset(self) -> None:
        """Clear all state."""
        self._theta = np.array([0.0, 0.0])
        self._P = np.array([[100.0, 0.0], [0.0, 100.0]])
        self._count = 0
        self._initialized = False
    
    def __repr__(self) -> str:
        return (
            f"RLS(slope={self.slope:.6f}, intercept={self.intercept:.4f}, "
            f"trend={self.get_trend_direction()})"
        )


class RegressionChannel:
    """
    Regression channel with dynamic bands.
    
    Combines RLS for center line with Welford for channel width.
    Useful for mean-reversion signals: price at upper band = sell,
    price at lower band = buy.
    """
    
    def __init__(
        self,
        forgetting_factor: float = 0.99,
        band_width_std: float = 2.0
    ):
        from .welford import WelfordVariance
        
        self._rls = RecursiveLeastSquares(forgetting_factor)
        self._residual_std = WelfordVariance()
        self.band_width_std = band_width_std
        self._count = 0
    
    def update(self, x: float, y: float) -> Dict[str, float]:
        """
        Update channel with new observation.
        
        Returns:
            dict with center, upper, lower, z_score, position
        """
        # Update regression
        slope, intercept = self._rls.update(x, y)
        center = slope * x + intercept
        
        # Track residuals for band width
        residual = y - center
        self._residual_std.update(residual)
        self._count += 1
        
        # Compute bands
        std = self._residual_std.std if self._count > 2 else 0.01
        band_offset = self.band_width_std * std
        
        upper = center + band_offset
        lower = center - band_offset
        
        # Z-score: how many std from center
        z_score = residual / std if std > 0 else 0.0
        
        # Position in channel: 0 = at lower, 1 = at upper
        channel_width = upper - lower
        position = (y - lower) / channel_width if channel_width > 0 else 0.5
        position = max(0, min(1, position))  # Clamp to [0, 1]
        
        return {
            'slope': slope,
            'intercept': intercept,
            'center': center,
            'upper': upper,
            'lower': lower,
            'z_score': z_score,
            'position': position,
            'residual_std': std,
        }
    
    def get_signal(self, x: float, y: float) -> float:
        """
        Get mean-reversion signal in [-1, 1].
        
        +1 = strongly oversold (at lower band)
        -1 = strongly overbought (at upper band)
        0 = at center
        """
        result = self.update(x, y)
        # Convert position [0,1] to signal [-1,+1]
        # At lower (0) → +1 (buy), at upper (1) → -1 (sell)
        return 1 - 2 * result['position']
