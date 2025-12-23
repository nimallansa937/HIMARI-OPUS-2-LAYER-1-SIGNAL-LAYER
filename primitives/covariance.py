"""
Online Bivariate Covariance and Correlation

Extends Welford's algorithm to track covariance between two variables
in O(1) time. Used for:
- Cross-asset correlations (BTC/ETH, SPY/QQQ pairs)
- Regime detection via correlation breakdown
- Portfolio weighting based on correlation structure

Memory: 56 bytes (7 floats)
Latency: <0.1ms per update

Usage:
    cov = OnlineCovariance()
    for ret_btc, ret_eth in returns:
        cov.update(ret_btc, ret_eth)
    print(f"Correlation: {cov.correlation:.3f}")
"""

import math
import json
from typing import Dict, Any, Tuple


class OnlineCovariance:
    """
    Online covariance using extended Welford algorithm.
    
    Computes running covariance and correlation between two
    time series without storing history.
    
    Based on the parallel algorithm from Chan et al. (1979).
    """
    
    __slots__ = (
        '_count',
        '_mean_x', '_mean_y',
        '_m2_x', '_m2_y',  # Sum of squared deviations
        '_c'  # Sum of products of deviations (co-moment)
    )
    
    def __init__(self):
        self._count = 0
        self._mean_x = 0.0
        self._mean_y = 0.0
        self._m2_x = 0.0
        self._m2_y = 0.0
        self._c = 0.0  # Co-moment
    
    def update(self, x: float, y: float) -> Tuple[float, float]:
        """
        Add paired observation. O(1) time.
        
        Args:
            x: Value from first series
            y: Value from second series
            
        Returns:
            (covariance, correlation)
        """
        self._count += 1
        n = self._count
        
        # Update means
        delta_x = x - self._mean_x
        self._mean_x += delta_x / n
        delta_x_new = x - self._mean_x
        
        delta_y = y - self._mean_y
        self._mean_y += delta_y / n
        delta_y_new = y - self._mean_y
        
        # Update sum of squared deviations
        self._m2_x += delta_x * delta_x_new
        self._m2_y += delta_y * delta_y_new
        
        # Update co-moment (key step for covariance)
        self._c += delta_x * delta_y_new
        
        return self.covariance, self.correlation
    
    @property
    def covariance(self) -> float:
        """Sample covariance (Bessel-corrected)."""
        if self._count < 2:
            return 0.0
        return self._c / (self._count - 1)
    
    @property
    def covariance_population(self) -> float:
        """Population covariance."""
        if self._count < 1:
            return 0.0
        return self._c / self._count
    
    @property
    def variance_x(self) -> float:
        """Sample variance of X."""
        if self._count < 2:
            return 0.0
        return self._m2_x / (self._count - 1)
    
    @property
    def variance_y(self) -> float:
        """Sample variance of Y."""
        if self._count < 2:
            return 0.0
        return self._m2_y / (self._count - 1)
    
    @property
    def std_x(self) -> float:
        """Sample standard deviation of X."""
        return math.sqrt(max(self.variance_x, 0))
    
    @property
    def std_y(self) -> float:
        """Sample standard deviation of Y."""
        return math.sqrt(max(self.variance_y, 0))
    
    @property
    def correlation(self) -> float:
        """
        Pearson correlation coefficient [-1, 1].
        
        correlation = covariance / (std_x * std_y)
        """
        if self._count < 2:
            return 0.0
        
        denom = math.sqrt(self._m2_x * self._m2_y)
        if denom < 1e-10:
            return 0.0
        
        corr = self._c / denom
        # Clamp to [-1, 1] for numerical stability
        return max(-1.0, min(1.0, corr))
    
    @property
    def mean_x(self) -> float:
        """Running mean of X."""
        return self._mean_x
    
    @property
    def mean_y(self) -> float:
        """Running mean of Y."""
        return self._mean_y
    
    @property
    def count(self) -> int:
        """Number of observations."""
        return self._count
    
    def beta(self) -> float:
        """
        Regression coefficient: Y = α + β*X
        
        β = Cov(X,Y) / Var(X)
        """
        if self.variance_x < 1e-10:
            return 0.0
        return self.covariance / self.variance_x
    
    def alpha(self) -> float:
        """
        Regression intercept: Y = α + β*X
        
        α = mean_y - β * mean_x
        """
        return self._mean_y - self.beta() * self._mean_x
    
    def r_squared(self) -> float:
        """
        Coefficient of determination R².
        
        R² = correlation²
        """
        return self.correlation ** 2
    
    def hedge_ratio(self) -> float:
        """
        Minimum variance hedge ratio.
        
        H = Cov(X,Y) / Var(X) = β
        
        To hedge Y with X, hold -H units of X per unit of Y.
        """
        return self.beta()
    
    def correlation_regime(self) -> str:
        """
        Classify correlation regime.
        
        Returns:
            'high_positive', 'positive', 'uncorrelated',
            'negative', 'high_negative'
        """
        corr = self.correlation
        
        if corr >= 0.7:
            return 'high_positive'
        elif corr >= 0.3:
            return 'positive'
        elif corr > -0.3:
            return 'uncorrelated'
        elif corr > -0.7:
            return 'negative'
        else:
            return 'high_negative'
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for persistence."""
        return {
            'count': self._count,
            'mean_x': self._mean_x,
            'mean_y': self._mean_y,
            'm2_x': self._m2_x,
            'm2_y': self._m2_y,
            'c': self._c,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OnlineCovariance':
        """Restore from serialized state."""
        instance = cls()
        instance._count = data['count']
        instance._mean_x = data['mean_x']
        instance._mean_y = data['mean_y']
        instance._m2_x = data['m2_x']
        instance._m2_y = data['m2_y']
        instance._c = data['c']
        return instance
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'OnlineCovariance':
        """Restore from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def reset(self) -> None:
        """Clear all state."""
        self._count = 0
        self._mean_x = 0.0
        self._mean_y = 0.0
        self._m2_x = 0.0
        self._m2_y = 0.0
        self._c = 0.0
    
    def __repr__(self) -> str:
        return (
            f"OnlineCovariance(n={self._count}, "
            f"corr={self.correlation:.3f}, "
            f"cov={self.covariance:.6f})"
        )


class ExponentialCovariance:
    """
    Exponentially weighted covariance with decay.
    
    Gives more weight to recent observations, useful for
    detecting correlation regime changes quickly.
    """
    
    __slots__ = (
        '_alpha',  # Decay factor
        '_mean_x', '_mean_y',
        '_var_x', '_var_y',
        '_cov',
        '_count'
    )
    
    def __init__(self, halflife: int = 20):
        """
        Args:
            halflife: Number of periods for weight to decay by half
        """
        self._alpha = 1 - math.exp(-math.log(2) / halflife)
        self._mean_x = 0.0
        self._mean_y = 0.0
        self._var_x = 0.0
        self._var_y = 0.0
        self._cov = 0.0
        self._count = 0
    
    def update(self, x: float, y: float) -> Tuple[float, float]:
        """
        Add observation with exponential weighting.
        
        Returns:
            (covariance, correlation)
        """
        self._count += 1
        
        if self._count == 1:
            self._mean_x = x
            self._mean_y = y
            return 0.0, 0.0
        
        # Update means (EMA style)
        delta_x = x - self._mean_x
        delta_y = y - self._mean_y
        
        self._mean_x += self._alpha * delta_x
        self._mean_y += self._alpha * delta_y
        
        delta_x_new = x - self._mean_x
        delta_y_new = y - self._mean_y
        
        # Update variances and covariance (exponential)
        self._var_x = (
            (1 - self._alpha) * self._var_x +
            self._alpha * delta_x * delta_x_new
        )
        self._var_y = (
            (1 - self._alpha) * self._var_y +
            self._alpha * delta_y * delta_y_new
        )
        self._cov = (
            (1 - self._alpha) * self._cov +
            self._alpha * delta_x * delta_y_new
        )
        
        return self.covariance, self.correlation
    
    @property
    def covariance(self) -> float:
        """Exponentially weighted covariance."""
        return self._cov
    
    @property
    def correlation(self) -> float:
        """Exponentially weighted correlation."""
        denom = math.sqrt(self._var_x * self._var_y)
        if denom < 1e-10:
            return 0.0
        return max(-1.0, min(1.0, self._cov / denom))
    
    @property
    def std_x(self) -> float:
        return math.sqrt(max(self._var_x, 0))
    
    @property
    def std_y(self) -> float:
        return math.sqrt(max(self._var_y, 0))


class CorrelationMatrix:
    """
    Track pairwise correlations for multiple assets.
    
    Maintains N*(N-1)/2 OnlineCovariance instances for N assets.
    """
    
    def __init__(self, asset_names: list):
        """
        Args:
            asset_names: List of asset identifiers
        """
        self.assets = list(asset_names)
        self.n = len(self.assets)
        self._covariances = {}
        
        # Create covariance trackers for each pair
        for i in range(self.n):
            for j in range(i + 1, self.n):
                key = (self.assets[i], self.assets[j])
                self._covariances[key] = OnlineCovariance()
    
    def update(self, returns: Dict[str, float]) -> None:
        """
        Update with new returns for each asset.
        
        Args:
            returns: Dict mapping asset name to return
        """
        for i in range(self.n):
            for j in range(i + 1, self.n):
                key = (self.assets[i], self.assets[j])
                if self.assets[i] in returns and self.assets[j] in returns:
                    self._covariances[key].update(
                        returns[self.assets[i]],
                        returns[self.assets[j]]
                    )
    
    def get_correlation(self, asset1: str, asset2: str) -> float:
        """Get correlation between two assets."""
        if asset1 == asset2:
            return 1.0
        
        # Order pair consistently
        if asset1 > asset2:
            asset1, asset2 = asset2, asset1
        
        key = (asset1, asset2)
        if key in self._covariances:
            return self._covariances[key].correlation
        return 0.0
    
    def correlation_matrix(self) -> Dict[Tuple[str, str], float]:
        """Get full correlation matrix as dict."""
        result = {}
        for key, cov in self._covariances.items():
            result[key] = cov.correlation
            result[(key[1], key[0])] = cov.correlation
        
        # Add diagonal
        for asset in self.assets:
            result[(asset, asset)] = 1.0
        
        return result
