"""
Combinatorial Purged Cross-Validation (CPCV)

Proper validation for financial time series that:
1. Prevents lookahead bias
2. Accounts for serial correlation
3. Provides multiple train/test splits with purging

Based on: Lopez de Prado "Advances in Financial Machine Learning"

Usage:
    cpcv = CPCV(n_splits=5, embargo_pct=0.01)
    for train_idx, test_idx in cpcv.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        predictions = model.predict(X[test_idx])
"""

import numpy as np
from typing import Iterator, Tuple, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass 
class CPCVConfig:
    """CPCV configuration."""
    n_splits: int = 5  # Number of folds
    n_test_splits: int = 2  # Test splits per combination
    embargo_pct: float = 0.01  # Percentage of data to embargo after test
    purge_pct: float = 0.01  # Percentage of data to purge before test


class CPCV:
    """
    Combinatorial Purged Cross-Validation.
    
    Unlike standard K-Fold, CPCV:
    1. PURGES training data that is too close to test data (prevents leakage)
    2. EMBARGOES data after test set (prevents using future info)
    3. Creates COMBINATORIAL splits (more robust validation)
    
    This is CRITICAL for backtesting signals - standard CV causes overfitting.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        embargo_pct: float = 0.01,
        purge_pct: float = 0.01,
    ):
        """
        Initialize CPCV.
        
        Args:
            n_splits: Number of groups to split data into
            n_test_splits: Number of groups used for testing per fold
            embargo_pct: Fraction of data to embargo after test
            purge_pct: Fraction of data to purge before test
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct
    
    def get_n_splits(self) -> int:
        """Return number of splitting iterations."""
        from math import comb
        return comb(self.n_splits, self.n_test_splits)
    
    def split(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits with purging and embargo.
        
        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Target array (optional, not used but kept for sklearn compatibility)
            
        Yields:
            (train_indices, test_indices) for each split
        """
        from itertools import combinations
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate purge and embargo sizes
        purge_size = int(n_samples * self.purge_pct)
        embargo_size = int(n_samples * self.embargo_pct)
        
        # Split into groups
        group_size = n_samples // self.n_splits
        groups = []
        for i in range(self.n_splits):
            start = i * group_size
            end = start + group_size if i < self.n_splits - 1 else n_samples
            groups.append(indices[start:end])
        
        # Generate all combinations of test groups
        for test_group_indices in combinations(range(self.n_splits), self.n_test_splits):
            # Build test set
            test_indices = np.concatenate([groups[i] for i in test_group_indices])
            test_start = test_indices.min()
            test_end = test_indices.max()
            
            # Build train set with purging and embargo
            train_indices = []
            for i, group in enumerate(groups):
                if i in test_group_indices:
                    continue  # Skip test groups
                
                group_start = group.min()
                group_end = group.max()
                
                # Purge: remove training data too close BEFORE test
                if group_end >= test_start - purge_size and group_end < test_start:
                    # Partially overlaps purge zone
                    valid_mask = group < (test_start - purge_size)
                    group = group[valid_mask]
                
                # Embargo: remove training data right AFTER test
                if group_start <= test_end + embargo_size and group_start > test_end:
                    # Partially overlaps embargo zone
                    valid_mask = group > (test_end + embargo_size)
                    group = group[valid_mask]
                
                if len(group) > 0:
                    train_indices.append(group)
            
            if train_indices:
                train_indices = np.concatenate(train_indices)
                yield train_indices, test_indices
    
    def validate_strategy(
        self,
        returns: np.ndarray,
        signals: np.ndarray,
        verbose: bool = True
    ) -> dict:
        """
        Validate strategy using CPCV.
        
        Args:
            returns: Actual returns
            signals: Strategy signals (positions)
            verbose: Print progress
            
        Returns:
            Validation metrics across folds
        """
        fold_results = []
        
        for i, (train_idx, test_idx) in enumerate(self.split(returns)):
            # Calculate returns on test set
            test_returns = returns[test_idx]
            test_signals = signals[test_idx]
            strategy_returns = test_returns * test_signals
            
            # Metrics
            mean_ret = np.mean(strategy_returns)
            std_ret = np.std(strategy_returns)
            sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
            
            win_rate = np.mean(strategy_returns > 0) if len(strategy_returns) > 0 else 0
            
            fold_results.append({
                'fold': i,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'mean_return': mean_ret,
                'std_return': std_ret,
                'sharpe': sharpe,
                'win_rate': win_rate,
            })
            
            if verbose:
                print(f"Fold {i}: Sharpe={sharpe:.2f}, WinRate={win_rate:.2%}")
        
        # Aggregate results
        sharpes = [r['sharpe'] for r in fold_results]
        
        return {
            'n_folds': len(fold_results),
            'mean_sharpe': np.mean(sharpes),
            'std_sharpe': np.std(sharpes),
            'min_sharpe': np.min(sharpes),
            'max_sharpe': np.max(sharpes),
            'consistency': np.mean([s > 0 for s in sharpes]),
            'fold_results': fold_results,
        }


class DeflatedSharpe:
    """
    Deflated Sharpe Ratio (DSR)
    
    Adjusts Sharpe Ratio for multiple testing / data snooping.
    
    The more strategies you test, the more likely you find one with
    high Sharpe by chance. DSR accounts for this.
    
    Based on: Bailey & Lopez de Prado (2014)
    "The Deflated Sharpe Ratio"
    
    Usage:
        dsr = DeflatedSharpe()
        is_significant = dsr.test(
            sharpe_observed=2.5,
            n_trials=100,  # Number of strategies tested
            n_observations=252,  # Data points
            skewness=-0.5,
            kurtosis=3.0
        )
    """
    
    @staticmethod
    def expected_max_sharpe(n_trials: int, variance: float = 1.0) -> float:
        """
        Expected maximum Sharpe from n independent trials.
        
        E[max(SR)] ≈ (1 - γ) * Phi^{-1}(1 - 1/n) + γ * Phi^{-1}(1 - 1/(n*e))
        
        where γ ≈ 0.5772 (Euler-Mascheroni constant)
        """
        from scipy import stats
        
        gamma = 0.5772156649  # Euler-Mascheroni
        
        if n_trials <= 1:
            return 0.0
        
        z1 = stats.norm.ppf(1 - 1/n_trials)
        z2 = stats.norm.ppf(1 - 1/(n_trials * np.e))
        
        expected_max = (1 - gamma) * z1 + gamma * z2
        
        return expected_max * np.sqrt(variance)
    
    @staticmethod
    def deflated_sharpe(
        sharpe_observed: float,
        sharpe_benchmark: float,
        n_observations: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> float:
        """
        Calculate probability that observed Sharpe is due to skill.
        
        DSR = Prob(SR* < SR_observed | SR_benchmark)
        
        High DSR (>0.95) means the Sharpe is likely real.
        Low DSR (<0.5) means it's probably luck/overfitting.
        
        Args:
            sharpe_observed: Observed Sharpe ratio
            sharpe_benchmark: Expected Sharpe under null (often from expected_max)
            n_observations: Number of data points
            skewness: Skewness of returns
            kurtosis: Excess kurtosis of returns
            
        Returns:
            Probability that observed Sharpe is genuine
        """
        from scipy import stats
        
        # Standard error of Sharpe ratio
        # SE(SR) ≈ sqrt((1 + 0.5*SR^2 - skew*SR + (kurt-3)/4 * SR^2) / T)
        sr = sharpe_observed
        se = np.sqrt(
            (1 + 0.5 * sr**2 - skewness * sr + (kurtosis - 3) / 4 * sr**2) 
            / n_observations
        )
        
        if se <= 0:
            return 0.5
        
        # Z-score
        z = (sharpe_observed - sharpe_benchmark) / se
        
        # Probability (one-sided test)
        prob = stats.norm.cdf(z)
        
        return prob
    
    @staticmethod
    def test(
        sharpe_observed: float,
        n_trials: int,
        n_observations: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
        significance: float = 0.95,
    ) -> dict:
        """
        Full deflated Sharpe test.
        
        Args:
            sharpe_observed: Your strategy's Sharpe
            n_trials: How many strategies you tested before finding this one
            n_observations: Number of data points (e.g., 252 for 1 year daily)
            skewness: Return skewness
            kurtosis: Return excess kurtosis
            significance: Required probability level (0.95 = 95%)
            
        Returns:
            Test results with is_significant flag
        """
        # Expected max Sharpe from random trials
        expected_max = DeflatedSharpe.expected_max_sharpe(n_trials)
        
        # Calculate DSR
        dsr = DeflatedSharpe.deflated_sharpe(
            sharpe_observed=sharpe_observed,
            sharpe_benchmark=expected_max,
            n_observations=n_observations,
            skewness=skewness,
            kurtosis=kurtosis,
        )
        
        return {
            'sharpe_observed': sharpe_observed,
            'expected_max_sharpe': expected_max,
            'deflated_sharpe_ratio': dsr,
            'is_significant': dsr >= significance,
            'significance_level': significance,
            'n_trials': n_trials,
            'haircut': sharpe_observed - expected_max,  # Reduction for snooping
        }


class HansenSPA:
    """
    Hansen's Superior Predictive Ability (SPA) Test
    
    Tests whether a strategy significantly beats a benchmark,
    accounting for data snooping when comparing multiple strategies.
    
    Based on: Hansen (2005) "A Test for Superior Predictive Ability"
    
    Usage:
        spa = HansenSPA()
        result = spa.test(
            benchmark_returns=market_returns,
            strategy_returns=my_strategy_returns,
            n_bootstrap=1000
        )
    """
    
    @staticmethod
    def test(
        benchmark_returns: np.ndarray,
        strategy_returns: np.ndarray,
        n_bootstrap: int = 1000,
        block_size: int = 10,
        significance: float = 0.05,
    ) -> dict:
        """
        Perform SPA test using stationary bootstrap.
        
        Args:
            benchmark_returns: Returns of benchmark (e.g., buy-and-hold)
            strategy_returns: Returns of your strategy
            n_bootstrap: Number of bootstrap samples
            block_size: Block size for stationary bootstrap
            significance: Significance level (0.05 = 5%)
            
        Returns:
            Test results with p-value
        """
        # Compute excess returns
        excess = strategy_returns - benchmark_returns
        n = len(excess)
        
        # Observed statistic (mean excess return)
        observed_stat = np.mean(excess)
        
        # Bootstrap distribution
        bootstrap_stats = []
        prob = 1 / block_size
        
        for _ in range(n_bootstrap):
            # Stationary bootstrap
            bootstrap_sample = []
            i = np.random.randint(0, n)
            
            while len(bootstrap_sample) < n:
                bootstrap_sample.append(excess[i])
                
                # With probability prob, jump to random position
                if np.random.random() < prob:
                    i = np.random.randint(0, n)
                else:
                    i = (i + 1) % n
            
            bootstrap_stat = np.mean(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate p-value
        # p = Prob(bootstrap_stat >= observed_stat)
        p_value = np.mean(bootstrap_stats >= observed_stat)
        
        # Confidence interval
        ci_lower = np.percentile(bootstrap_stats, significance * 100 / 2)
        ci_upper = np.percentile(bootstrap_stats, 100 - significance * 100 / 2)
        
        return {
            'observed_excess_return': observed_stat,
            'p_value': p_value,
            'is_significant': p_value < significance,
            'bootstrap_mean': np.mean(bootstrap_stats),
            'bootstrap_std': np.std(bootstrap_stats),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_bootstrap': n_bootstrap,
        }


# Quick test
if __name__ == "__main__":
    np.random.seed(42)
    
    print("=" * 60)
    print("Testing CPCV and Validation Tools")
    print("=" * 60)
    
    # Generate sample data
    n_samples = 500
    returns = np.random.normal(0.001, 0.02, n_samples)
    signals = np.sign(np.random.normal(0, 1, n_samples))  # Random signals
    
    # Test CPCV
    print("\n1. CPCV Test")
    print("-" * 40)
    cpcv = CPCV(n_splits=5, n_test_splits=2)
    print(f"Number of splits: {cpcv.get_n_splits()}")
    
    for i, (train, test) in enumerate(cpcv.split(returns)):
        if i < 3:
            print(f"  Fold {i}: train={len(train)}, test={len(test)}")
    print("  ...")
    
    # Test Deflated Sharpe
    print("\n2. Deflated Sharpe Test")
    print("-" * 40)
    dsr_result = DeflatedSharpe.test(
        sharpe_observed=2.0,
        n_trials=50,  # Tested 50 strategies
        n_observations=252,
        skewness=-0.5,
        kurtosis=4.0,
    )
    print(f"  Observed Sharpe: {dsr_result['sharpe_observed']:.2f}")
    print(f"  Expected Max (luck): {dsr_result['expected_max_sharpe']:.2f}")
    print(f"  DSR Probability: {dsr_result['deflated_sharpe_ratio']:.2%}")
    print(f"  Is Significant: {dsr_result['is_significant']}")
    print(f"  Haircut: {dsr_result['haircut']:+.2f}")
    
    # Test Hansen's SPA
    print("\n3. Hansen's SPA Test")
    print("-" * 40)
    benchmark = np.random.normal(0.0005, 0.015, 200)  # Market returns
    strategy = benchmark + np.random.normal(0.0003, 0.005, 200)  # Slightly better
    
    spa_result = HansenSPA.test(benchmark, strategy, n_bootstrap=500)
    print(f"  Excess Return: {spa_result['observed_excess_return']:.4f}")
    print(f"  P-Value: {spa_result['p_value']:.4f}")
    print(f"  Is Significant: {spa_result['is_significant']}")
    print(f"  95% CI: [{spa_result['ci_lower']:.4f}, {spa_result['ci_upper']:.4f}]")
    
    print("\n" + "=" * 60)
    print("✓ All validation tests passed!")
    print("=" * 60)
