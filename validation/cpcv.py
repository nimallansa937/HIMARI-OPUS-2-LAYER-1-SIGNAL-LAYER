"""
Combinatorial Purged Cross-Validation (CPCV) for Strategy Validation

CPCV outperforms walk-forward analysis (ScienceDirect 2024) by:
- Generating hundreds of train/test combinations
- Preventing temporal leakage with purge/embargo periods
- Testing strategy robustness across many market orderings

Key Parameters:
- N = 6-10 folds
- k = 2-3 test folds per combination
- ≥100 paths (combinations)
- Purge period = max label horizon
- Embargo = 1-5% of total period

Usage:
    cpcv = CPCVValidator(n_folds=6, n_test_folds=2)
    
    for train_idx, test_idx in cpcv.split(data, labels):
        model.fit(data[train_idx], labels[train_idx])
        pred = model.predict(data[test_idx])
        score = evaluate(pred, labels[test_idx])
    
    results = cpcv.get_results()
    if results['passed']:
        print("Strategy is robust!")
"""

import math
import numpy as np
from typing import Dict, Any, List, Tuple, Iterator, Optional
from itertools import combinations


class CPCVValidator:
    """
    Combinatorial Purged Cross-Validation.
    
    Generates all C(N, k) combinations of test folds,
    with purge and embargo periods to prevent leakage.
    """
    
    def __init__(
        self,
        n_folds: int = 6,
        n_test_folds: int = 2,
        purge_length: int = 5,
        embargo_pct: float = 0.02
    ):
        """
        Initialize CPCV.
        
        Args:
            n_folds: Total number of folds (N)
            n_test_folds: Number of folds in test set (k)
            purge_length: Samples to purge around test boundary
            embargo_pct: Percentage of data to embargo after test
        """
        self.n_folds = n_folds
        self.n_test_folds = n_test_folds
        self.purge_length = purge_length
        self.embargo_pct = embargo_pct
        
        self._results: List[Dict[str, float]] = []
        self._n_paths = self._compute_n_paths()
    
    def _compute_n_paths(self) -> int:
        """Compute number of paths C(N, k)."""
        return math.comb(self.n_folds, self.n_test_folds)
    
    @property
    def n_paths(self) -> int:
        """Number of train/test combinations."""
        return self._n_paths
    
    def split(
        self,
        n_samples: int,
        return_indices: bool = True
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits.
        
        Args:
            n_samples: Total number of samples
            return_indices: If True, return indices. If False, return masks.
            
        Yields:
            (train_indices, test_indices) or (train_mask, test_mask)
        """
        # Compute fold boundaries
        fold_size = n_samples // self.n_folds
        fold_bounds = [(i * fold_size, (i + 1) * fold_size) 
                       for i in range(self.n_folds)]
        fold_bounds[-1] = (fold_bounds[-1][0], n_samples)  # Last fold to end
        
        # Embargo length
        embargo_len = int(n_samples * self.embargo_pct)
        
        # Generate all combinations of test folds
        for test_folds in combinations(range(self.n_folds), self.n_test_folds):
            train_mask = np.ones(n_samples, dtype=bool)
            test_mask = np.zeros(n_samples, dtype=bool)
            
            for fold_idx in test_folds:
                start, end = fold_bounds[fold_idx]
                
                # Mark test region
                test_mask[start:end] = True
                train_mask[start:end] = False
                
                # Purge: remove samples before test
                purge_start = max(0, start - self.purge_length)
                train_mask[purge_start:start] = False
                
                # Embargo: remove samples after test
                embargo_end = min(n_samples, end + embargo_len)
                train_mask[end:embargo_end] = False
            
            if return_indices:
                yield np.where(train_mask)[0], np.where(test_mask)[0]
            else:
                yield train_mask, test_mask
    
    def validate_strategy(
        self,
        returns: np.ndarray,
        signals: np.ndarray,
        min_sharpe: float = 0.5,
        max_sharpe_std: float = 0.5
    ) -> Dict[str, Any]:
        """
        Validate strategy across all CPCV paths.
        
        Args:
            returns: Array of actual returns
            signals: Array of strategy signals (-1, 0, 1)
            min_sharpe: Minimum average Sharpe across paths
            max_sharpe_std: Maximum std dev of Sharpe (consistency)
            
        Returns:
            Validation results
        """
        n_samples = len(returns)
        sharpes = []
        
        for train_idx, test_idx in self.split(n_samples):
            if len(test_idx) < 10:
                continue
            
            # Compute test period returns
            test_returns = returns[test_idx]
            test_signals = signals[test_idx]
            
            # Strategy returns
            strategy_returns = test_returns * test_signals
            
            # Sharpe ratio (annualized, assuming daily)
            if len(strategy_returns) > 1:
                mean_ret = np.mean(strategy_returns)
                std_ret = np.std(strategy_returns, ddof=1)
                
                if std_ret > 1e-10:
                    sharpe = (mean_ret / std_ret) * np.sqrt(252)
                else:
                    sharpe = 0.0
                
                sharpes.append(sharpe)
        
        if len(sharpes) == 0:
            return {
                'passed': False,
                'reason': 'no_valid_paths',
                'n_paths': 0,
            }
        
        sharpes = np.array(sharpes)
        mean_sharpe = np.mean(sharpes)
        std_sharpe = np.std(sharpes, ddof=1)
        sharpe_consistency = std_sharpe / abs(mean_sharpe) if abs(mean_sharpe) > 0.01 else float('inf')
        
        passed = (
            mean_sharpe >= min_sharpe and
            sharpe_consistency <= max_sharpe_std
        )
        
        return {
            'passed': passed,
            'mean_sharpe': mean_sharpe,
            'std_sharpe': std_sharpe,
            'sharpe_consistency': sharpe_consistency,
            'min_sharpe': np.min(sharpes),
            'max_sharpe': np.max(sharpes),
            'n_paths': len(sharpes),
            'pct_positive': np.mean(sharpes > 0) * 100,
            'reason': 'passed' if passed else 'failed_criteria',
        }
    
    def probability_of_backtest_overfitting(
        self,
        in_sample_sharpes: np.ndarray,
        out_of_sample_sharpes: np.ndarray
    ) -> float:
        """
        Compute Probability of Backtest Overfitting (PBO).
        
        PBO = proportion of paths where OOS_rank > IS_rank
        
        Lower PBO = less likely to be overfit.
        PBO > 0.5 = likely overfit
        
        Args:
            in_sample_sharpes: Sharpes on training data
            out_of_sample_sharpes: Sharpes on test data
            
        Returns:
            PBO value [0, 1]
        """
        if len(in_sample_sharpes) != len(out_of_sample_sharpes):
            raise ValueError("Arrays must have same length")
        
        n = len(in_sample_sharpes)
        is_ranks = np.argsort(np.argsort(-in_sample_sharpes))  # Descending rank
        oos_ranks = np.argsort(np.argsort(-out_of_sample_sharpes))
        
        # PBO = proportion where OOS rank is worse than IS rank
        pbo = np.mean(oos_ranks > is_ranks)
        
        return pbo
    
    def __repr__(self) -> str:
        return (
            f"CPCVValidator(n_folds={self.n_folds}, "
            f"n_test_folds={self.n_test_folds}, "
            f"n_paths={self._n_paths})"
        )


class WalkForwardValidator:
    """
    Walk-Forward validation for comparison with CPCV.
    
    Note: CPCV is preferred per HIMARI L1 spec, but walk-forward
    is included for comparison and legacy compatibility.
    """
    
    def __init__(
        self,
        n_splits: int = 10,
        train_ratio: float = 0.5,
        gap: int = 0
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            n_splits: Number of walk-forward periods
            train_ratio: Fraction of each period for training
            gap: Gap between train and test (prevents leakage)
        """
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.gap = gap
    
    def split(
        self,
        n_samples: int
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate walk-forward splits.
        
        Yields:
            (train_indices, test_indices)
        """
        period_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            start = i * period_size
            end = start + period_size if i < self.n_splits - 1 else n_samples
            
            split_point = start + int((end - start) * self.train_ratio)
            
            train_idx = np.arange(start, split_point)
            test_start = split_point + self.gap
            test_idx = np.arange(test_start, end)
            
            if len(test_idx) > 0:
                yield train_idx, test_idx


class DataLeakageDetector:
    """
    Utilities for detecting data leakage in backtests.
    
    Warning Signs:
    - Annual returns > 12% unleveraged
    - Sharpe > 1.5 on simple strategies
    - Exponentially smooth equity curves
    """
    
    @staticmethod
    def shuffle_test(
        features: np.ndarray,
        labels: np.ndarray,
        model_factory: callable,
        n_trials: int = 10
    ) -> Dict[str, float]:
        """
        Shuffle test: Randomize labels, retrain, should get ~50% accuracy.
        
        If shuffled accuracy is much better than 50%, there's leakage
        from features containing future information.
        
        Args:
            features: Feature array
            labels: Label array
            model_factory: Function that returns untrained model
            n_trials: Number of shuffle trials
            
        Returns:
            Dict with real vs shuffled accuracy
        """
        from sklearn.model_selection import train_test_split
        
        # Real accuracy
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.3, shuffle=False
        )
        
        model = model_factory()
        model.fit(X_train, y_train)
        real_accuracy = model.score(X_test, y_test)
        
        # Shuffled accuracy (should be ~0.5)
        shuffled_accuracies = []
        for _ in range(n_trials):
            y_shuffled = np.random.permutation(labels)
            X_train, X_test, y_train, y_test = train_test_split(
                features, y_shuffled, test_size=0.3, shuffle=False
            )
            model = model_factory()
            model.fit(X_train, y_train)
            shuffled_accuracies.append(model.score(X_test, y_test))
        
        mean_shuffled = np.mean(shuffled_accuracies)
        
        # Leakage suspected if shuffled >> 0.5
        leakage_suspected = mean_shuffled > 0.55
        
        return {
            'real_accuracy': real_accuracy,
            'shuffled_accuracy': mean_shuffled,
            'leakage_suspected': leakage_suspected,
            'leakage_score': mean_shuffled - 0.5,  # 0 = no leakage
        }
    
    @staticmethod
    def check_equity_curve(returns: np.ndarray) -> Dict[str, Any]:
        """
        Check equity curve for suspicious patterns.
        
        Args:
            returns: Strategy returns
            
        Returns:
            Dict with warning flags
        """
        cumulative = np.cumprod(1 + returns)
        
        # Compute drawdowns
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        max_dd = np.min(drawdown)
        
        # Annualized return
        n_years = len(returns) / 252
        total_return = cumulative[-1] - 1
        ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Sharpe (simplified)
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        warnings = []
        
        if ann_return > 0.12:
            warnings.append('high_annual_return')
        
        if sharpe > 1.5:
            warnings.append('high_sharpe')
        
        if max_dd > -0.05:  # Less than 5% drawdown is suspicious
            warnings.append('suspiciously_low_drawdown')
        
        # Check for exponential smoothness
        returns_std = np.std(returns)
        equity_std = np.std(np.diff(cumulative))
        if equity_std < returns_std * 0.3:
            warnings.append('exponentially_smooth_equity')
        
        return {
            'annualized_return': ann_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'warnings': warnings,
            'leakage_likely': len(warnings) >= 2,
        }
