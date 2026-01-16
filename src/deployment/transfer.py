"""
Transfer Ratio Confidence

Historical data suggests that strategies typically achieve 60-80% of their
backtested performance in live trading. The transfer ratio model predicts
this degradation and recommends position sizing accordingly.

P(Sharpe_live > threshold | Sharpe_backtest, historical_data)

Position sizing recommendation based on confidence:
- 70% confidence → 50% of target size
- 90% confidence → 100% of target size
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class TransferMetrics:
    """Historical transfer ratio data point."""
    strategy_id: str
    backtest_sharpe: float
    live_sharpe: float
    transfer_ratio: float
    strategy_type: str  # momentum, reversion, etc.
    market_regime: str  # bull, bear, range


@dataclass
class TransferPrediction:
    """Prediction of live performance from backtest."""
    expected_live_sharpe: float
    confidence_interval: Tuple[float, float]
    probability_above_threshold: float
    recommended_position_pct: float
    risk_adjusted_size: float


class TransferRatioConfidence:
    """
    Compute confidence in strategy transfer from backtest to live.

    Model: live_sharpe = alpha * backtest_sharpe + beta + noise
    """

    # Default historical statistics (based on industry research)
    DEFAULT_ALPHA = 0.65  # Typical degradation factor
    DEFAULT_BETA = 0.30   # Base offset
    DEFAULT_RESIDUAL_STD = 0.40  # Typical prediction error

    def __init__(
        self,
        history: Optional[List[Tuple[float, float]]] = None,
        min_history: int = 10
    ):
        """
        Args:
            history: List of (backtest_sharpe, live_sharpe) pairs
            min_history: Minimum history before fitting custom model
        """
        self.history = history or []
        self.min_history = min_history

        # Model parameters
        self.alpha = self.DEFAULT_ALPHA
        self.beta = self.DEFAULT_BETA
        self.residual_std = self.DEFAULT_RESIDUAL_STD

        if len(self.history) >= min_history:
            self._fit_model()

    def _fit_model(self) -> None:
        """Fit linear regression: live = alpha * backtest + beta."""
        if len(self.history) < self.min_history:
            return

        from sklearn.linear_model import LinearRegression

        backtest = np.array([h[0] for h in self.history]).reshape(-1, 1)
        live = np.array([h[1] for h in self.history])

        model = LinearRegression().fit(backtest, live)

        self.alpha = float(model.coef_[0])
        self.beta = float(model.intercept_)
        self.residual_std = float(np.std(live - model.predict(backtest)))

        logger.info(f"Fitted transfer model: live = {self.alpha:.3f} * backtest + {self.beta:.3f}")

    def add_observation(
        self,
        backtest_sharpe: float,
        live_sharpe: float
    ) -> None:
        """Add new backtest-live pair and refit model."""
        self.history.append((backtest_sharpe, live_sharpe))
        if len(self.history) >= self.min_history:
            self._fit_model()

    def get_confidence(
        self,
        backtest_sharpe: float,
        threshold: float = 1.5
    ) -> Tuple[float, float]:
        """
        Compute confidence that live Sharpe exceeds threshold.

        Args:
            backtest_sharpe: Backtest Sharpe ratio
            threshold: Minimum acceptable live Sharpe

        Returns:
            (confidence, expected_live_sharpe)
        """
        # Predict expected live Sharpe
        expected_live = self.alpha * backtest_sharpe + self.beta

        # P(live > threshold) = P(Z > (threshold - expected) / std)
        z = (threshold - expected_live) / (self.residual_std + 1e-8)
        confidence = 1 - stats.norm.cdf(z)

        return float(confidence), float(expected_live)

    def get_prediction(
        self,
        backtest_sharpe: float,
        threshold: float = 1.5,
        target_position_pct: float = 0.10
    ) -> TransferPrediction:
        """
        Get full transfer prediction with position sizing.

        Args:
            backtest_sharpe: Backtest Sharpe ratio
            threshold: Minimum acceptable live Sharpe
            target_position_pct: Target position size if fully confident

        Returns:
            TransferPrediction with all metrics
        """
        confidence, expected_live = self.get_confidence(backtest_sharpe, threshold)

        # Confidence interval (95%)
        ci_lower = expected_live - 1.96 * self.residual_std
        ci_upper = expected_live + 1.96 * self.residual_std

        # Position sizing based on confidence
        recommended_size = self._compute_position_size(confidence, target_position_pct)

        # Risk-adjusted size (Kelly-like)
        risk_adjusted = self._kelly_fraction(expected_live, self.residual_std) * target_position_pct

        return TransferPrediction(
            expected_live_sharpe=expected_live,
            confidence_interval=(ci_lower, ci_upper),
            probability_above_threshold=confidence,
            recommended_position_pct=recommended_size,
            risk_adjusted_size=risk_adjusted
        )

    def _compute_position_size(
        self,
        confidence: float,
        target: float
    ) -> float:
        """
        Compute position size based on confidence level.

        Scaling:
        - <50% confidence: 0% (don't deploy)
        - 50-70%: 25% of target
        - 70-80%: 50% of target
        - 80-90%: 75% of target
        - >90%: 100% of target
        """
        if confidence < 0.50:
            return 0.0
        elif confidence < 0.70:
            return target * 0.25
        elif confidence < 0.80:
            return target * 0.50
        elif confidence < 0.90:
            return target * 0.75
        else:
            return target

    def _kelly_fraction(
        self,
        expected_sharpe: float,
        sharpe_std: float
    ) -> float:
        """
        Compute Kelly-like optimal fraction.

        f* = (expected_return - rf) / variance
        Simplified using Sharpe approximation.
        """
        if sharpe_std <= 0:
            return 0.5

        # Half-Kelly for safety
        kelly = 0.5 * (expected_sharpe / (sharpe_std ** 2 + 1))
        return max(0.0, min(1.0, kelly))

    def recommend_position_size(
        self,
        backtest_sharpe: float,
        base_size: float = 1.0
    ) -> float:
        """
        Simple position sizing recommendation.

        Args:
            backtest_sharpe: Strategy's backtest Sharpe
            base_size: Base position size (will be scaled)

        Returns:
            Recommended position size
        """
        confidence, _ = self.get_confidence(backtest_sharpe)

        if confidence < 0.70:
            return base_size * 0.25
        elif confidence < 0.80:
            return base_size * 0.50
        elif confidence < 0.90:
            return base_size * 0.75
        return base_size

    def get_statistics(self) -> Dict:
        """Get model statistics."""
        return {
            'n_observations': len(self.history),
            'alpha': self.alpha,
            'beta': self.beta,
            'residual_std': self.residual_std,
            'expected_transfer_ratio': self.alpha,  # For SR=1 strategy
            'model_fitted': len(self.history) >= self.min_history
        }


class AdaptiveTransferModel:
    """
    Transfer model that adapts to different strategy types and regimes.

    Different strategy types (momentum, reversion) may have different
    transfer characteristics in different market regimes.
    """

    def __init__(self):
        # Separate models per (strategy_type, regime)
        self.models: Dict[Tuple[str, str], TransferRatioConfidence] = {}
        self.default_model = TransferRatioConfidence()

    def add_observation(
        self,
        backtest_sharpe: float,
        live_sharpe: float,
        strategy_type: str = "general",
        regime: str = "normal"
    ) -> None:
        """Add observation for specific strategy type and regime."""
        key = (strategy_type, regime)

        if key not in self.models:
            self.models[key] = TransferRatioConfidence()

        self.models[key].add_observation(backtest_sharpe, live_sharpe)
        self.default_model.add_observation(backtest_sharpe, live_sharpe)

    def get_confidence(
        self,
        backtest_sharpe: float,
        threshold: float = 1.5,
        strategy_type: str = "general",
        regime: str = "normal"
    ) -> Tuple[float, float]:
        """Get confidence using appropriate model."""
        key = (strategy_type, regime)

        if key in self.models and len(self.models[key].history) >= 5:
            return self.models[key].get_confidence(backtest_sharpe, threshold)
        else:
            return self.default_model.get_confidence(backtest_sharpe, threshold)

    def get_all_statistics(self) -> Dict:
        """Get statistics for all models."""
        stats = {'default': self.default_model.get_statistics()}
        for key, model in self.models.items():
            stats[f"{key[0]}_{key[1]}"] = model.get_statistics()
        return stats


class TransferRatioTracker:
    """
    Track transfer ratios over time for deployed strategies.

    Monitors whether strategies maintain expected transfer.
    """

    def __init__(
        self,
        alert_threshold: float = 0.5,
        window_days: int = 30
    ):
        """
        Args:
            alert_threshold: Alert if TR drops below this
            window_days: Rolling window for TR calculation
        """
        self.alert_threshold = alert_threshold
        self.window_days = window_days

        # Strategy ID -> list of (timestamp, backtest_sharpe, live_sharpe)
        self.tracking: Dict[str, List[Tuple[float, float, float]]] = {}

    def record(
        self,
        strategy_id: str,
        timestamp: float,
        backtest_sharpe: float,
        live_sharpe: float
    ) -> None:
        """Record a transfer observation."""
        if strategy_id not in self.tracking:
            self.tracking[strategy_id] = []

        self.tracking[strategy_id].append((timestamp, backtest_sharpe, live_sharpe))

    def get_rolling_tr(self, strategy_id: str) -> Optional[float]:
        """Get rolling transfer ratio for strategy."""
        if strategy_id not in self.tracking:
            return None

        records = self.tracking[strategy_id]
        if not records:
            return None

        # Use recent records
        recent = records[-self.window_days:]

        avg_backtest = np.mean([r[1] for r in recent])
        avg_live = np.mean([r[2] for r in recent])

        if avg_backtest > 0:
            return avg_live / avg_backtest
        return None

    def get_alerts(self) -> List[Dict]:
        """Get list of strategies with concerning transfer ratios."""
        alerts = []

        for strategy_id in self.tracking:
            tr = self.get_rolling_tr(strategy_id)
            if tr is not None and tr < self.alert_threshold:
                alerts.append({
                    'strategy_id': strategy_id,
                    'transfer_ratio': tr,
                    'threshold': self.alert_threshold,
                    'message': f"TR {tr:.2f} below threshold {self.alert_threshold}"
                })

        return alerts
