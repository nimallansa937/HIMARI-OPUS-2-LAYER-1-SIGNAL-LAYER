"""
Combinatorial Purged Cross-Validation (CPCV) for Strategy Validation

Prevents overfitting and temporal leakage in strategy validation:
- Purging: Removes observations near test boundaries to prevent look-ahead bias
- Embargo: Excludes data after test period to break autocorrelation
- Combinatorial: Tests all C(n,2) train/test combinations for robustness

Based on: "Advances in Financial Machine Learning" by Marcos Lopez de Prado
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Generator, Optional
from itertools import combinations
import logging

logger = logging.getLogger(__name__)


@dataclass
class CPCVConfig:
    """Configuration for CPCV validation."""
    n_folds: int = 5                    # Number of folds (C(5,2) = 10 splits)
    purge_bars: int = 24                # Bars to remove before test (2 hours at 5-min)
    embargo_bars: int = 12              # Bars to remove after test (1 hour at 5-min)
    min_samples_per_fold: int = 252     # Minimum samples per fold (1 year daily)

    # Pass/fail thresholds
    min_mean_sharpe: float = 1.5        # Average Sharpe across folds
    max_sharpe_std_ratio: float = 0.5   # Max std/mean ratio for consistency
    min_worst_sharpe: float = 0.5       # Minimum worst-fold Sharpe
    min_deflated_sharpe: float = 1.0    # Multiple-testing adjusted Sharpe
    require_all_folds_positive: bool = True  # All folds must have positive Sharpe


@dataclass
class FoldMetrics:
    """Metrics computed for a single fold."""
    fold_id: int
    train_size: int
    test_size: int
    sharpe: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    n_trades: int


@dataclass
class CPCVResult:
    """Result from CPCV validation."""
    passed: bool
    mean_sharpe: float                  # Average across folds
    std_sharpe: float                   # Consistency metric
    worst_sharpe: float                 # Worst fold performance
    deflated_sharpe: float              # Multiple-testing adjusted
    n_folds_profitable: int             # Count of positive Sharpe folds
    n_folds_total: int                  # Total number of folds
    fold_metrics: List[FoldMetrics]     # Individual fold results
    reason: str

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "passed": self.passed,
            "mean_sharpe": self.mean_sharpe,
            "std_sharpe": self.std_sharpe,
            "worst_sharpe": self.worst_sharpe,
            "deflated_sharpe": self.deflated_sharpe,
            "n_folds_profitable": self.n_folds_profitable,
            "n_folds_total": self.n_folds_total,
            "reason": self.reason,
            "fold_metrics": [
                {
                    "fold_id": fm.fold_id,
                    "train_size": fm.train_size,
                    "test_size": fm.test_size,
                    "sharpe": fm.sharpe,
                    "max_drawdown": fm.max_drawdown,
                    "win_rate": fm.win_rate,
                    "profit_factor": fm.profit_factor,
                    "n_trades": fm.n_trades
                }
                for fm in self.fold_metrics
            ]
        }


class CPCVSplitter:
    """
    Generates train/test splits for Combinatorial Purged Cross-Validation.

    Key features:
    1. Combinatorial: All C(n,2) test fold combinations
    2. Purged: Remove `purge_bars` before test period
    3. Embargo: Remove `embargo_bars` after test period
    """

    def __init__(self, config: CPCVConfig):
        self.config = config

    def generate_splits(
        self,
        n_samples: int
    ) -> Generator[Tuple[np.ndarray, np.ndarray, int], None, None]:
        """
        Generate all train/test splits with purge and embargo applied.

        Args:
            n_samples: Total number of samples in the dataset

        Yields:
            Tuple of (train_indices, test_indices, fold_id)
        """
        n_folds = self.config.n_folds
        fold_size = n_samples // n_folds

        if fold_size < self.config.min_samples_per_fold:
            logger.warning(
                f"Fold size {fold_size} < min {self.config.min_samples_per_fold}. "
                f"Reducing n_folds from {n_folds}."
            )
            n_folds = max(2, n_samples // self.config.min_samples_per_fold)
            fold_size = n_samples // n_folds

        # Create fold boundaries
        fold_bounds = []
        for i in range(n_folds):
            start = i * fold_size
            end = (i + 1) * fold_size if i < n_folds - 1 else n_samples
            fold_bounds.append((start, end))

        # Generate all C(n,2) test combinations
        fold_id = 0
        for test_combo in combinations(range(n_folds), 2):
            train_mask = np.ones(n_samples, dtype=bool)
            test_mask = np.zeros(n_samples, dtype=bool)

            # Mark test folds
            for fold_idx in test_combo:
                start, end = fold_bounds[fold_idx]
                test_mask[start:end] = True
                train_mask[start:end] = False

            # Apply purge: remove samples before each test segment
            train_mask = self._apply_purge(train_mask, test_mask, fold_bounds, test_combo)

            # Apply embargo: remove samples after each test segment
            train_mask = self._apply_embargo(train_mask, test_mask, fold_bounds, test_combo)

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            yield train_indices, test_indices, fold_id
            fold_id += 1

    def _apply_purge(
        self,
        train_mask: np.ndarray,
        test_mask: np.ndarray,
        fold_bounds: List[Tuple[int, int]],
        test_combo: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Remove samples within purge_bars before each test fold start.

        This prevents information leakage from features that might look ahead.
        """
        purge_bars = self.config.purge_bars

        for fold_idx in test_combo:
            test_start, _ = fold_bounds[fold_idx]
            purge_start = max(0, test_start - purge_bars)
            train_mask[purge_start:test_start] = False

        return train_mask

    def _apply_embargo(
        self,
        train_mask: np.ndarray,
        test_mask: np.ndarray,
        fold_bounds: List[Tuple[int, int]],
        test_combo: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Remove samples within embargo_bars after each test fold end.

        This breaks autocorrelation between test and subsequent training data.
        """
        embargo_bars = self.config.embargo_bars
        n_samples = len(train_mask)

        for fold_idx in test_combo:
            _, test_end = fold_bounds[fold_idx]
            embargo_end = min(n_samples, test_end + embargo_bars)
            train_mask[test_end:embargo_end] = False

        return train_mask

    def get_n_splits(self) -> int:
        """Return total number of splits: C(n_folds, 2)."""
        from math import comb
        return comb(self.config.n_folds, 2)


class CPCVValidator:
    """
    Validates strategies using Combinatorial Purged Cross-Validation.

    Usage:
        validator = CPCVValidator(CPCVConfig(n_folds=5, purge_bars=24))
        result = validator.validate(strategy, returns_series)

        if result.passed:
            print(f"Strategy approved with deflated SR: {result.deflated_sharpe:.2f}")
    """

    def __init__(self, config: Optional[CPCVConfig] = None):
        self.config = config or CPCVConfig()
        self.splitter = CPCVSplitter(self.config)

    def validate(
        self,
        returns: np.ndarray,
        strategy=None  # Optional: for logging/identification only
    ) -> CPCVResult:
        """
        Run CPCV validation on strategy returns.

        Args:
            returns: 1D array of strategy returns (daily or bar-level)
            strategy: Optional strategy object for identification

        Returns:
            CPCVResult with pass/fail decision and detailed metrics
        """
        if len(returns) < self.config.min_samples_per_fold * 2:
            return CPCVResult(
                passed=False,
                mean_sharpe=0.0,
                std_sharpe=0.0,
                worst_sharpe=0.0,
                deflated_sharpe=0.0,
                n_folds_profitable=0,
                n_folds_total=0,
                fold_metrics=[],
                reason=f"Insufficient data: {len(returns)} < {self.config.min_samples_per_fold * 2}"
            )

        # Run validation on each fold
        fold_metrics = []
        for train_idx, test_idx, fold_id in self.splitter.generate_splits(len(returns)):
            metrics = self._compute_fold_metrics(
                returns=returns,
                train_indices=train_idx,
                test_indices=test_idx,
                fold_id=fold_id
            )
            fold_metrics.append(metrics)

        # Aggregate results
        return self._aggregate_results(fold_metrics)

    def _compute_fold_metrics(
        self,
        returns: np.ndarray,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        fold_id: int
    ) -> FoldMetrics:
        """
        Compute performance metrics for a single fold.

        In production, this would apply the strategy logic trained on train_indices
        to generate predictions on test_indices. For now, we compute metrics
        directly on the test returns.
        """
        test_returns = returns[test_indices]

        # Sharpe ratio (annualized, assuming 252 trading days)
        sharpe = self._compute_sharpe(test_returns)

        # Maximum drawdown
        max_drawdown = self._compute_max_drawdown(test_returns)

        # Win rate (proportion of positive returns)
        win_rate = np.mean(test_returns > 0) if len(test_returns) > 0 else 0.0

        # Profit factor (sum of gains / sum of losses)
        profit_factor = self._compute_profit_factor(test_returns)

        # Number of "trades" (non-zero returns as proxy)
        n_trades = np.sum(np.abs(test_returns) > 1e-8)

        return FoldMetrics(
            fold_id=fold_id,
            train_size=len(train_indices),
            test_size=len(test_indices),
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            n_trades=int(n_trades)
        )

    def _compute_sharpe(self, returns: np.ndarray, annualization: int = 252) -> float:
        """Compute annualized Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) < 1e-8:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(annualization))

    def _compute_max_drawdown(self, returns: np.ndarray) -> float:
        """Compute maximum drawdown from returns series."""
        if len(returns) == 0:
            return 0.0

        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        return float(np.max(drawdowns))

    def _compute_profit_factor(self, returns: np.ndarray) -> float:
        """Compute profit factor (gross profits / gross losses)."""
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())

        if losses < 1e-8:
            return 10.0 if gains > 0 else 1.0  # Cap at 10
        return float(min(10.0, gains / losses))

    def _aggregate_results(self, fold_metrics: List[FoldMetrics]) -> CPCVResult:
        """
        Aggregate fold metrics into final CPCV result.

        Pass criteria:
        1. mean_sharpe >= min_mean_sharpe
        2. std_sharpe <= max_sharpe_std_ratio * mean_sharpe
        3. worst_sharpe >= min_worst_sharpe
        4. All folds profitable (if require_all_folds_positive)
        5. deflated_sharpe >= min_deflated_sharpe
        """
        if not fold_metrics:
            return CPCVResult(
                passed=False,
                mean_sharpe=0.0,
                std_sharpe=0.0,
                worst_sharpe=0.0,
                deflated_sharpe=0.0,
                n_folds_profitable=0,
                n_folds_total=0,
                fold_metrics=[],
                reason="No fold metrics computed"
            )

        sharpes = np.array([fm.sharpe for fm in fold_metrics])

        mean_sharpe = float(np.mean(sharpes))
        std_sharpe = float(np.std(sharpes))
        worst_sharpe = float(np.min(sharpes))
        n_folds_profitable = int(np.sum(sharpes > 0))
        n_folds_total = len(fold_metrics)

        # Compute deflated Sharpe ratio
        deflated_sharpe = self._compute_deflated_sharpe(mean_sharpe, std_sharpe, n_folds_total)

        # Check pass criteria
        config = self.config
        reasons = []

        if mean_sharpe < config.min_mean_sharpe:
            reasons.append(f"Mean SR {mean_sharpe:.2f} < {config.min_mean_sharpe}")

        if mean_sharpe > 0 and std_sharpe > config.max_sharpe_std_ratio * mean_sharpe:
            reasons.append(f"SR std ratio {std_sharpe/mean_sharpe:.2f} > {config.max_sharpe_std_ratio}")

        if worst_sharpe < config.min_worst_sharpe:
            reasons.append(f"Worst SR {worst_sharpe:.2f} < {config.min_worst_sharpe}")

        if config.require_all_folds_positive and n_folds_profitable < n_folds_total:
            reasons.append(f"Only {n_folds_profitable}/{n_folds_total} folds profitable")

        if deflated_sharpe < config.min_deflated_sharpe:
            reasons.append(f"Deflated SR {deflated_sharpe:.2f} < {config.min_deflated_sharpe}")

        passed = len(reasons) == 0

        if passed:
            reason = (
                f"CPCV passed: Mean SR={mean_sharpe:.2f}+/-{std_sharpe:.2f}, "
                f"Worst={worst_sharpe:.2f}, Deflated={deflated_sharpe:.2f}"
            )
        else:
            reason = "; ".join(reasons)

        return CPCVResult(
            passed=passed,
            mean_sharpe=mean_sharpe,
            std_sharpe=std_sharpe,
            worst_sharpe=worst_sharpe,
            deflated_sharpe=deflated_sharpe,
            n_folds_profitable=n_folds_profitable,
            n_folds_total=n_folds_total,
            fold_metrics=fold_metrics,
            reason=reason
        )

    def _compute_deflated_sharpe(
        self,
        mean_sharpe: float,
        std_sharpe: float,
        n_folds: int
    ) -> float:
        """
        Compute deflated Sharpe ratio adjusted for multiple testing.

        Based on Bailey & Lopez de Prado (2014):
        "The Deflated Sharpe Ratio: Correcting for Selection Bias"

        DSR = (observed - expected_max) / std_dev(max)

        Applies a haircut based on the number of trials (folds).
        """
        if n_folds <= 1 or mean_sharpe <= 0:
            return mean_sharpe

        # Haircut adjustment (simplified from full DSR formula)
        # Penalize for testing multiple folds
        haircut = std_sharpe * np.sqrt(n_folds / (n_folds - 1))
        deflated = mean_sharpe - haircut

        return float(max(0.0, deflated))
