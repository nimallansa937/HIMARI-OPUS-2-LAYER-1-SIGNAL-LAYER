"""
Online GARCH(1,1) Volatility Model

Computes volatility forecasts in O(1) time per update using the
GARCH(1,1) specification from the HIMARI L1 addendum.

Model:
    σ²_t = ω + α·(r_{t-1})² + β·σ²_{t-1}
    
Where:
    ω (omega) = long-run variance baseline
    α (alpha) = reaction to recent shock (innovation weight)
    β (beta)  = persistence of past volatility
    
Default parameters (ω=0.00001, α=0.05, β=0.94) from Tier 2 spec.
Note: α + β < 1 ensures stationarity.

Usage:
    garch = OnlineGARCH()
    for ret in returns:
        vol = garch.update(ret)
        position_size = base_size / vol  # Volatility scaling
        
Memory: 24 bytes (3 floats)
Latency: <0.05ms per update
"""

import math
import json
from typing import Dict, Any, Optional, Tuple


class OnlineGARCH:
    """
    Online GARCH(1,1) volatility estimator.
    
    Updates volatility forecast with each new return observation.
    Provides volatility-scaled position sizing guidance.
    """
    
    __slots__ = (
        'omega', 'alpha', 'beta',
        '_sigma2', '_last_return', '_count',
        '_long_run_var'
    )
    
    def __init__(
        self,
        omega: float = 0.00001,
        alpha: float = 0.05,
        beta: float = 0.94,
        initial_variance: Optional[float] = None
    ):
        """
        Initialize GARCH(1,1) model.
        
        Args:
            omega: Long-run variance constant
            alpha: Weight on squared innovation (recent shock)
            beta: Weight on lagged variance (persistence)
            initial_variance: Starting variance estimate. If None,
                             uses unconditional variance ω/(1-α-β)
        """
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        
        # Validate stationarity condition
        if alpha + beta >= 1.0:
            raise ValueError(
                f"GARCH not stationary: α + β = {alpha + beta} >= 1. "
                "Must have α + β < 1 for mean-reverting volatility."
            )
        
        # Unconditional (long-run) variance
        self._long_run_var = omega / (1 - alpha - beta)
        
        # Initialize variance
        if initial_variance is not None:
            self._sigma2 = initial_variance
        else:
            self._sigma2 = self._long_run_var
        
        self._last_return = 0.0
        self._count = 0
    
    def update(self, log_return: float) -> float:
        """
        Update volatility forecast with new log return. O(1) time.
        
        The GARCH update computes:
        σ²_t = ω + α·(r_{t-1})² + β·σ²_{t-1}
        
        Args:
            log_return: Log return ln(P_t / P_{t-1})
            
        Returns:
            Updated volatility forecast (σ, standard deviation)
        """
        if self._count > 0:
            # GARCH(1,1) recursion
            self._sigma2 = (
                self.omega +
                self.alpha * (self._last_return ** 2) +
                self.beta * self._sigma2
            )
        
        self._last_return = log_return
        self._count += 1
        
        return self.volatility
    
    @property
    def volatility(self) -> float:
        """Current volatility forecast (σ = sqrt(σ²))."""
        return math.sqrt(max(self._sigma2, 0))
    
    @property
    def variance(self) -> float:
        """Current variance forecast (σ²)."""
        return self._sigma2
    
    @property
    def long_run_volatility(self) -> float:
        """Unconditional long-run volatility."""
        return math.sqrt(self._long_run_var)
    
    def get_volatility_regime(self) -> str:
        """
        Classify current volatility regime.
        
        Returns:
            'low', 'normal', or 'high' based on deviation from long-run
        """
        ratio = self._sigma2 / self._long_run_var
        
        if ratio < 0.7:
            return 'low'
        elif ratio > 1.5:
            return 'high'
        else:
            return 'normal'
    
    def get_position_scalar(self, target_volatility: float = 0.01) -> float:
        """
        Compute position size scalar for volatility targeting.
        
        Args:
            target_volatility: Target portfolio volatility (default 1% daily)
            
        Returns:
            Scalar to multiply base position size by
        """
        if self.volatility <= 0:
            return 1.0
        return min(target_volatility / self.volatility, 3.0)  # Cap at 3x
    
    def get_var_estimate(self, confidence: float = 0.99) -> float:
        """
        Compute Value at Risk estimate.
        
        Args:
            confidence: Confidence level (0.95, 0.99, etc.)
            
        Returns:
            VaR as positive percentage loss (multiply by position value)
        """
        from scipy.stats import norm
        z_score = norm.ppf(1 - confidence)
        return abs(z_score * self.volatility)
    
    def forecast(self, steps: int = 1) -> float:
        """
        Forecast volatility N steps ahead.
        
        For GARCH(1,1):
        σ²_{t+h} = ω·Σ(α+β)^i + (α+β)^h·σ²_t
        
        Args:
            steps: Number of periods ahead
            
        Returns:
            Forecasted volatility at horizon
        """
        persistence = self.alpha + self.beta
        
        if abs(persistence - 1) < 1e-10:
            # Near unit root, variance grows linearly
            forecast_var = self._sigma2 + steps * self.omega
        else:
            # Mean-reverting
            forecast_var = (
                self._long_run_var +
                (persistence ** steps) * (self._sigma2 - self._long_run_var)
            )
        
        return math.sqrt(max(forecast_var, 0))
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for Redis persistence."""
        return {
            'omega': self.omega,
            'alpha': self.alpha,
            'beta': self.beta,
            'sigma2': self._sigma2,
            'last_return': self._last_return,
            'count': self._count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OnlineGARCH':
        """Restore from serialized state."""
        instance = cls(
            omega=data['omega'],
            alpha=data['alpha'],
            beta=data['beta'],
            initial_variance=data['sigma2']
        )
        instance._last_return = data['last_return']
        instance._count = data['count']
        return instance
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'OnlineGARCH':
        """Restore from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def reset(self) -> None:
        """Reset to initial state."""
        self._sigma2 = self._long_run_var
        self._last_return = 0.0
        self._count = 0
    
    def __repr__(self) -> str:
        regime = self.get_volatility_regime()
        return (
            f"OnlineGARCH(vol={self.volatility:.4f}, regime={regime}, "
            f"count={self._count})"
        )


class AdaptiveGARCH(OnlineGARCH):
    """
    GARCH with adaptive parameter estimation.
    
    Uses exponentially weighted MLE-like updates to adapt
    parameters to changing market conditions.
    """
    
    def __init__(
        self,
        adaptation_rate: float = 0.01,
        **kwargs
    ):
        """
        Args:
            adaptation_rate: How quickly parameters adapt (0.001-0.1)
        """
        super().__init__(**kwargs)
        self._adaptation_rate = adaptation_rate
        self._return_sq_ema = self._long_run_var
        self._sigma2_ema = self._long_run_var
    
    def update(self, log_return: float) -> float:
        """Update with adaptive parameter adjustment."""
        # Standard GARCH update first
        vol = super().update(log_return)
        
        # Adaptive EMA of squared returns and variance
        rate = self._adaptation_rate
        self._return_sq_ema = (
            rate * (log_return ** 2) +
            (1 - rate) * self._return_sq_ema
        )
        self._sigma2_ema = (
            rate * self._sigma2 +
            (1 - rate) * self._sigma2_ema
        )
        
        # Adapt alpha based on recent shock relative to variance
        shock_ratio = (log_return ** 2) / max(self._sigma2, 1e-10)
        if shock_ratio > 4:  # Unexpected large shock
            # Increase alpha temporarily (more reactive)
            self.alpha = min(self.alpha * 1.05, 0.2)
        elif shock_ratio < 0.25:  # Calm period
            # Decrease alpha (less reactive)
            self.alpha = max(self.alpha * 0.99, 0.02)
        
        # Ensure stationarity maintained
        self.beta = min(self.beta, 0.98 - self.alpha)
        
        return vol


# =============================================================================
# EGARCH variant (handles asymmetric effects)
# =============================================================================

class OnlineEGARCH:
    """
    Exponential GARCH for asymmetric volatility response.
    
    Log volatility model:
    log(σ²_t) = ω + α·g(z_{t-1}) + β·log(σ²_{t-1})
    
    Where g(z) = θ·z + γ·(|z| - E[|z|])
    
    This captures the "leverage effect" where negative returns
    typically increase volatility more than positive returns.
    """
    
    __slots__ = (
        'omega', 'alpha', 'beta', 'gamma',
        '_log_sigma2', '_last_z', '_count'
    )
    
    def __init__(
        self,
        omega: float = -0.5,
        alpha: float = 0.1,
        beta: float = 0.9,
        gamma: float = -0.1  # Negative = leverage effect
    ):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Log variance (ensures σ² always positive)
        self._log_sigma2 = omega / (1 - beta)
        self._last_z = 0.0
        self._count = 0
    
    def update(self, log_return: float) -> float:
        """Update EGARCH volatility estimate."""
        sigma = math.exp(0.5 * self._log_sigma2)
        
        if self._count > 0 and sigma > 0:
            z = log_return / sigma  # Standardized residual
            
            # g(z) = α·z + γ·(|z| - E[|z|])
            # E[|z|] ≈ sqrt(2/π) for standard normal
            e_abs_z = math.sqrt(2 / math.pi)
            gz = self.alpha * z + self.gamma * (abs(z) - e_abs_z)
            
            # EGARCH recursion
            self._log_sigma2 = (
                self.omega +
                gz +
                self.beta * self._log_sigma2
            )
            
            self._last_z = z
        
        self._count += 1
        return self.volatility
    
    @property
    def volatility(self) -> float:
        """Current volatility estimate."""
        return math.exp(0.5 * self._log_sigma2)
    
    @property
    def variance(self) -> float:
        """Current variance estimate."""
        return math.exp(self._log_sigma2)
