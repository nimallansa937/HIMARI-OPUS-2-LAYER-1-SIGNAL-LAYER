"""
Monte Carlo Permutation Test for Strategy Significance

Tests whether strategy returns are statistically different from random noise
by comparing observed Sharpe ratio against a null distribution of shuffled returns.

Key insight: If shuffling the return order produces similar Sharpe ratios,
the strategy's performance might be due to chance rather than genuine alpha.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class PermutationConfig:
    """Configuration for permutation testing."""
    n_permutations: int = 100           # Number of shuffles for null distribution
    alpha: float = 0.05                 # Significance level (5% = 95% confidence)
    random_seed: Optional[int] = 42     # For reproducibility (None for random)
    min_samples: int = 100              # Minimum samples required


@dataclass
class PermutationResult:
    """Result from permutation significance test."""
    passed: bool                        # p < alpha
    observed_sharpe: float              # Sharpe from actual returns
    p_value: float                      # Fraction of null >= observed
    null_mean: float                    # Mean of shuffled Sharpes
    null_std: float                     # Std of shuffled Sharpes
    percentile: float                   # Where observed falls (0-100)
    n_permutations: int                 # Number of permutations run
    reason: str

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "passed": self.passed,
            "observed_sharpe": self.observed_sharpe,
            "p_value": self.p_value,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "percentile": self.percentile,
            "n_permutations": self.n_permutations,
            "reason": self.reason
        }


class PermutationTester:
    """
    Monte Carlo permutation test for strategy significance.

    How it works:
    1. Compute the observed Sharpe ratio from actual returns
    2. Shuffle returns n_permutations times
    3. Compute Sharpe for each shuffled series (null distribution)
    4. p-value = fraction of null Sharpes >= observed Sharpe
    5. Strategy is significant if p < alpha (default 0.05)

    Usage:
        tester = PermutationTester(PermutationConfig(n_permutations=100))
        result = tester.test_significance(strategy_returns)

        if result.passed:
            print(f"Strategy is significant with p={result.p_value:.4f}")
    """

    def __init__(self, config: Optional[PermutationConfig] = None):
        self.config = config or PermutationConfig()

    def test_significance(self, returns: np.ndarray) -> PermutationResult:
        """
        Test if strategy returns are statistically significant.

        Args:
            returns: 1D array of strategy returns

        Returns:
            PermutationResult with pass/fail and detailed statistics
        """
        if len(returns) < self.config.min_samples:
            return PermutationResult(
                passed=False,
                observed_sharpe=0.0,
                p_value=1.0,
                null_mean=0.0,
                null_std=0.0,
                percentile=0.0,
                n_permutations=0,
                reason=f"Insufficient data: {len(returns)} < {self.config.min_samples}"
            )

        # Set random seed for reproducibility
        rng = np.random.default_rng(self.config.random_seed)

        # Compute observed Sharpe ratio
        observed_sharpe = self._compute_sharpe(returns)

        # Build null distribution via permutation
        null_distribution = self._build_null_distribution(returns, rng)

        # Compute p-value and statistics
        p_value = self._compute_p_value(observed_sharpe, null_distribution)
        null_mean = float(np.mean(null_distribution))
        null_std = float(np.std(null_distribution))

        # Percentile rank (0-100)
        percentile = float(np.sum(null_distribution < observed_sharpe) / len(null_distribution) * 100)

        # Determine pass/fail
        passed = p_value < self.config.alpha

        if passed:
            reason = (
                f"Significant: Observed SR={observed_sharpe:.2f} > "
                f"{100*(1-self.config.alpha):.0f}% of null (p={p_value:.4f})"
            )
        else:
            reason = (
                f"Not significant: p={p_value:.4f} >= {self.config.alpha} "
                f"(observed={observed_sharpe:.2f}, null mean={null_mean:.2f})"
            )

        return PermutationResult(
            passed=passed,
            observed_sharpe=observed_sharpe,
            p_value=p_value,
            null_mean=null_mean,
            null_std=null_std,
            percentile=percentile,
            n_permutations=self.config.n_permutations,
            reason=reason
        )

    def _compute_sharpe(self, returns: np.ndarray, annualization: int = 252) -> float:
        """Compute annualized Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) < 1e-8:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(annualization))

    def _build_null_distribution(
        self,
        returns: np.ndarray,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Build null distribution by shuffling returns n_permutations times.

        Shuffling destroys any temporal structure while preserving the
        marginal distribution. If the strategy's Sharpe comes from
        timing, shuffled returns should have lower Sharpe.
        """
        null_sharpes = np.zeros(self.config.n_permutations)

        for i in range(self.config.n_permutations):
            shuffled = rng.permutation(returns)
            null_sharpes[i] = self._compute_sharpe(shuffled)

        return null_sharpes

    def _compute_p_value(
        self,
        observed: float,
        null_distribution: np.ndarray
    ) -> float:
        """
        Compute p-value: fraction of null distribution >= observed.

        This is a one-tailed test (we only care if observed is unusually high).
        """
        if len(null_distribution) == 0:
            return 1.0

        # Count how many null values are >= observed
        n_extreme = np.sum(null_distribution >= observed)

        # Add 1 to numerator and denominator for more conservative estimate
        # (Phipson & Smyth, 2010)
        p_value = (n_extreme + 1) / (len(null_distribution) + 1)

        return float(p_value)


class BlockPermutationTester(PermutationTester):
    """
    Block permutation test that preserves local autocorrelation structure.

    Standard permutation completely destroys temporal structure. Block
    permutation shuffles contiguous blocks, preserving within-block
    autocorrelation while testing for cross-block predictability.

    Use this variant when:
    - Returns have strong intraday patterns that should be preserved
    - You want to test longer-horizon signal strength
    """

    def __init__(
        self,
        config: Optional[PermutationConfig] = None,
        block_size: int = 20  # Default: ~1 month of daily bars
    ):
        super().__init__(config)
        self.block_size = block_size

    def _build_null_distribution(
        self,
        returns: np.ndarray,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Build null distribution by shuffling blocks of returns.
        """
        n = len(returns)
        block_size = self.block_size

        # Number of complete blocks
        n_blocks = n // block_size

        if n_blocks < 2:
            # Fall back to regular permutation if too few blocks
            return super()._build_null_distribution(returns, rng)

        null_sharpes = np.zeros(self.config.n_permutations)

        for i in range(self.config.n_permutations):
            # Create block indices
            block_order = rng.permutation(n_blocks)

            # Reconstruct shuffled returns
            shuffled = np.zeros(n_blocks * block_size)
            for j, block_idx in enumerate(block_order):
                start_src = block_idx * block_size
                end_src = start_src + block_size
                start_dst = j * block_size
                end_dst = start_dst + block_size
                shuffled[start_dst:end_dst] = returns[start_src:end_src]

            null_sharpes[i] = self._compute_sharpe(shuffled)

        return null_sharpes


class StationaryBootstrapTester(PermutationTester):
    """
    Stationary bootstrap test for time series with autocorrelation.

    Uses Politis & Romano (1994) stationary bootstrap:
    - Random block lengths (geometric distribution)
    - Preserves stationarity properties of the original series

    Use this variant when:
    - Returns are highly autocorrelated
    - Standard permutation test might be anti-conservative
    """

    def __init__(
        self,
        config: Optional[PermutationConfig] = None,
        expected_block_length: int = 20  # Expected block length
    ):
        super().__init__(config)
        self.expected_block_length = expected_block_length
        # Probability of starting new block
        self.p_new_block = 1.0 / expected_block_length

    def _build_null_distribution(
        self,
        returns: np.ndarray,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Build null distribution using stationary bootstrap.
        """
        n = len(returns)
        null_sharpes = np.zeros(self.config.n_permutations)

        for i in range(self.config.n_permutations):
            bootstrapped = self._stationary_bootstrap(returns, n, rng)
            null_sharpes[i] = self._compute_sharpe(bootstrapped)

        return null_sharpes

    def _stationary_bootstrap(
        self,
        returns: np.ndarray,
        length: int,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Generate a stationary bootstrap sample.
        """
        n = len(returns)
        result = np.zeros(length)

        # Start at random position
        pos = rng.integers(0, n)

        for i in range(length):
            result[i] = returns[pos]

            # With probability p, start a new block
            if rng.random() < self.p_new_block:
                pos = rng.integers(0, n)
            else:
                pos = (pos + 1) % n

        return result
