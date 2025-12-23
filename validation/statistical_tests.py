"""
White's Reality Check and Hansen's SPA Test

Statistical tests to validate strategy outperformance is real,
not just data mining luck.

White's Reality Check (2000):
- Tests if best strategy is significantly better than benchmark
- Bootstrap-based p-values
- Controls for multiple testing

Hansen's Superior Predictive Ability (SPA) Test (2005):
- Improvement over White's test
- Uses studentized statistics
- More power with better size control

Usage:
    test = WhiteSPATest()
    result = test.test(
        benchmark_returns=market_returns,
        strategy_returns=strategy_returns,
        n_bootstrap=1000
    )
    
    if result['p_value'] < 0.05:
        print("Strategy significantly outperforms!")
"""

import numpy as np
from typing import Dict, Any, List, Optional


class WhiteSPATest:
    """
    Combined White's Reality Check and Hansen's SPA Test.
    
    Tests whether the best strategy outperforms a benchmark
    beyond what would be expected by chance.
    """
    
    def __init__(self, block_size: int = None):
        """
        Initialize test.
        
        Args:
            block_size: Block size for block bootstrap.
                       If None, uses sqrt(n).
        """
        self._block_size = block_size
    
    def test(
        self,
        benchmark_returns: np.ndarray,
        strategy_returns: np.ndarray,
        n_bootstrap: int = 1000,
        test_type: str = 'spa'
    ) -> Dict[str, Any]:
        """
        Perform White/SPA test.
        
        Args:
            benchmark_returns: Returns of benchmark (e.g., buy-and-hold)
            strategy_returns: Returns of strategy being tested
            n_bootstrap: Number of bootstrap samples
            test_type: 'white' for Reality Check, 'spa' for Hansen's SPA
            
        Returns:
            Dict with p_value, test_statistic, and interpretation
        """
        # Excess returns over benchmark
        excess = np.array(strategy_returns) - np.array(benchmark_returns)
        n = len(excess)
        
        if n < 10:
            return {
                'p_value': 1.0,
                'passed': False,
                'reason': 'insufficient_data',
            }
        
        # Observed test statistic: standardized mean excess return
        observed_mean = np.mean(excess)
        observed_std = np.std(excess, ddof=1) / np.sqrt(n)
        observed_stat = observed_mean / observed_std if observed_std > 0 else 0
        
        # Block bootstrap
        block_size = self._block_size or max(1, int(np.sqrt(n)))
        n_blocks = n // block_size
        
        # Generate bootstrap distribution
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            # Block bootstrap: sample blocks with replacement
            block_starts = np.random.randint(0, n - block_size + 1, n_blocks)
            
            bootstrap_sample = []
            for start in block_starts:
                bootstrap_sample.extend(excess[start:start + block_size])
            
            bootstrap_sample = np.array(bootstrap_sample[:n])
            
            if test_type == 'spa':
                # Hansen's SPA: recenter under null
                centered_sample = bootstrap_sample - np.mean(bootstrap_sample)
                boot_mean = np.mean(centered_sample)
                boot_std = np.std(centered_sample, ddof=1) / np.sqrt(len(centered_sample))
                boot_stat = boot_mean / boot_std if boot_std > 0 else 0
            else:
                # White's Reality Check: just compute statistic
                boot_mean = np.mean(bootstrap_sample)
                boot_std = np.std(bootstrap_sample, ddof=1) / np.sqrt(len(bootstrap_sample))
                boot_stat = boot_mean / boot_std if boot_std > 0 else 0
            
            bootstrap_stats.append(boot_stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # P-value: proportion of bootstrap stats >= observed
        p_value = np.mean(bootstrap_stats >= observed_stat)
        
        # Interpretation
        if p_value < 0.01:
            interpretation = 'strong_outperformance'
        elif p_value < 0.05:
            interpretation = 'significant_outperformance'
        elif p_value < 0.10:
            interpretation = 'marginal_outperformance'
        else:
            interpretation = 'no_significant_outperformance'
        
        return {
            'p_value': p_value,
            'test_statistic': observed_stat,
            'mean_excess_return': observed_mean,
            'n_bootstrap': n_bootstrap,
            'test_type': test_type,
            'interpretation': interpretation,
            'passed': p_value < 0.05,
            'bootstrap_mean': np.mean(bootstrap_stats),
            'bootstrap_std': np.std(bootstrap_stats),
        }
    
    def test_multiple_strategies(
        self,
        benchmark_returns: np.ndarray,
        strategies: Dict[str, np.ndarray],
        n_bootstrap: int = 1000
    ) -> Dict[str, Any]:
        """
        Test multiple strategies simultaneously.
        
        Controls for multiple testing by using the supremum of
        test statistics across all strategies.
        
        Args:
            benchmark_returns: Benchmark returns
            strategies: Dict mapping strategy name to returns
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Combined test results
        """
        n = len(benchmark_returns)
        
        # Compute excess returns for all strategies
        excess_returns = {}
        individual_stats = {}
        
        for name, strat_returns in strategies.items():
            excess = np.array(strat_returns) - np.array(benchmark_returns)
            excess_returns[name] = excess
            
            mean_excess = np.mean(excess)
            std_excess = np.std(excess, ddof=1) / np.sqrt(n)
            stat = mean_excess / std_excess if std_excess > 0 else 0
            individual_stats[name] = {
                'statistic': stat,
                'mean_excess': mean_excess,
            }
        
        # Best observed statistic
        best_name = max(individual_stats.keys(), 
                       key=lambda k: individual_stats[k]['statistic'])
        best_stat = individual_stats[best_name]['statistic']
        
        # Bootstrap under null (all strategies same as benchmark)
        block_size = max(1, int(np.sqrt(n)))
        n_blocks = n // block_size
        
        bootstrap_max_stats = []
        
        for _ in range(n_bootstrap):
            # Generate bootstrap indices
            block_starts = np.random.randint(0, n - block_size + 1, n_blocks)
            indices = []
            for start in block_starts:
                indices.extend(range(start, start + block_size))
            indices = np.array(indices[:n])
            
            # Compute max statistic across strategies
            max_stat = -np.inf
            for name, excess in excess_returns.items():
                boot_excess = excess[indices] - np.mean(excess[indices])
                boot_mean = np.mean(boot_excess)
                boot_std = np.std(boot_excess, ddof=1) / np.sqrt(n)
                boot_stat = boot_mean / boot_std if boot_std > 0 else 0
                max_stat = max(max_stat, boot_stat)
            
            bootstrap_max_stats.append(max_stat)
        
        bootstrap_max_stats = np.array(bootstrap_max_stats)
        
        # P-value for supremum test
        p_value = np.mean(bootstrap_max_stats >= best_stat)
        
        return {
            'p_value': p_value,
            'best_strategy': best_name,
            'best_statistic': best_stat,
            'individual_stats': individual_stats,
            'passed': p_value < 0.05,
            'n_strategies': len(strategies),
            'interpretation': 'significant' if p_value < 0.05 else 'not_significant',
        }


class FamilyWiseErrorControl:
    """
    Methods for controlling family-wise error rate with multiple strategies.
    """
    
    @staticmethod
    def bonferroni_correction(
        p_values: List[float],
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Bonferroni correction for multiple testing.
        
        Conservative: adjusted_alpha = alpha / n_tests
        
        Args:
            p_values: List of raw p-values
            alpha: Desired family-wise error rate
            
        Returns:
            Adjusted significance thresholds
        """
        n = len(p_values)
        adjusted_alpha = alpha / n
        
        significant = [p < adjusted_alpha for p in p_values]
        
        return {
            'adjusted_alpha': adjusted_alpha,
            'significant': significant,
            'n_significant': sum(significant),
            'method': 'bonferroni',
        }
    
    @staticmethod
    def holm_stepdown(
        p_values: List[float],
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Holm's step-down procedure (more powerful than Bonferroni).
        
        Args:
            p_values: List of raw p-values
            alpha: Desired family-wise error rate
            
        Returns:
            Which tests are significant
        """
        n = len(p_values)
        
        # Sort p-values
        sorted_indices = np.argsort(p_values)
        sorted_p = [p_values[i] for i in sorted_indices]
        
        # Holm thresholds
        thresholds = [alpha / (n - i) for i in range(n)]
        
        # Find rejection cutoff
        significant_mask = [False] * n
        for i, (p, thresh) in enumerate(zip(sorted_p, thresholds)):
            if p < thresh:
                significant_mask[sorted_indices[i]] = True
            else:
                break  # Stop at first non-rejection
        
        return {
            'significant': significant_mask,
            'n_significant': sum(significant_mask),
            'thresholds': thresholds,
            'method': 'holm',
        }
    
    @staticmethod
    def benjamini_hochberg(
        p_values: List[float],
        fdr: float = 0.05
    ) -> Dict[str, Any]:
        """
        Benjamini-Hochberg FDR control.
        
        Controls false discovery rate instead of family-wise error.
        More powerful when many true positives expected.
        
        Args:
            p_values: List of raw p-values
            fdr: Desired false discovery rate
            
        Returns:
            Which tests are significant
        """
        n = len(p_values)
        
        # Sort p-values
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array([p_values[i] for i in sorted_indices])
        
        # BH thresholds
        thresholds = np.array([(i + 1) * fdr / n for i in range(n)])
        
        # Find largest k where p_k <= threshold_k
        below_threshold = sorted_p <= thresholds
        if np.any(below_threshold):
            max_k = np.max(np.where(below_threshold)[0])
            significant_sorted = list(range(max_k + 1))
        else:
            significant_sorted = []
        
        significant_mask = [False] * n
        for i in significant_sorted:
            significant_mask[sorted_indices[i]] = True
        
        return {
            'significant': significant_mask,
            'n_significant': sum(significant_mask),
            'method': 'benjamini_hochberg',
            'fdr': fdr,
        }
