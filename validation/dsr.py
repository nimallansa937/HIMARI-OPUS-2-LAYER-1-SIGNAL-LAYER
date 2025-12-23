"""
Deflated Sharpe Ratio (DSR) for Multiple Testing Correction

When testing N strategies, the observed "best" Sharpe is inflated by
selection bias. DSR computes the probability that the true Sharpe
is positive after correcting for:
1. Number of trials (multiple testing)
2. Non-normal returns (skewness, kurtosis)
3. Track record length

CRITICAL: Minimum Sharpe hurdle should be 3.0 (not 2.0) when
multiple testing is involved (Bailey & López de Prado, 2014).

Usage:
    dsr = DeflatedSharpeRatio()
    
    # After testing 10 strategies
    prob = dsr.compute(
        observed_sharpe=2.5,
        n_trials=10,
        track_record_years=3,
        skewness=-0.5,
        kurtosis=4.0
    )
    
    if prob < 0.95:
        print("Strategy may be luck, not skill")
"""

import math
import numpy as np
from typing import Tuple, Dict, Any


def normal_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def normal_ppf(p: float) -> float:
    """Standard normal inverse CDF (approximation)."""
    # Rational approximation
    if p <= 0:
        return -10
    if p >= 1:
        return 10
    
    if p < 0.5:
        return -_rational_approx(math.sqrt(-2 * math.log(p)))
    else:
        return _rational_approx(math.sqrt(-2 * math.log(1 - p)))


def _rational_approx(t: float) -> float:
    """Rational approximation for normal inverse CDF."""
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    return t - (c[0] + c[1]*t + c[2]*t**2) / (1 + d[0]*t + d[1]*t**2 + d[2]*t**3)


class DeflatedSharpeRatio:
    """
    Deflated Sharpe Ratio implementation.
    
    Corrects for multiple testing bias when selecting strategies.
    """
    
    def __init__(self, min_sharpe_hurdle: float = 3.0):
        """
        Args:
            min_sharpe_hurdle: Minimum Sharpe to consider (default 3.0 per spec)
        """
        self.min_sharpe_hurdle = min_sharpe_hurdle
    
    def compute(
        self,
        observed_sharpe: float,
        n_trials: int,
        track_record_years: float,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
        annualization_factor: float = 252
    ) -> float:
        """
        Compute probability that true Sharpe > 0.
        
        Args:
            observed_sharpe: Observed annualized Sharpe ratio
            n_trials: Number of strategies/configurations tested
            track_record_years: Length of backtest in years
            skewness: Return skewness (negative = left tail)
            kurtosis: Return kurtosis (3.0 = normal)
            annualization_factor: Trading days per year
            
        Returns:
            Probability that true Sharpe > 0
        """
        # Number of observations
        T = track_record_years * annualization_factor
        
        if T < 10 or n_trials < 1:
            return 0.0
        
        # Expected maximum Sharpe under null hypothesis
        # (all strategies have true SR = 0)
        expected_max_sr = self._expected_max_sharpe(n_trials, T)
        
        # Standard error of Sharpe ratio (corrected for non-normality)
        sr_std = self._sharpe_std(observed_sharpe, T, skewness, kurtosis)
        
        if sr_std < 1e-10:
            return 1.0 if observed_sharpe > expected_max_sr else 0.0
        
        # Probability that true Sharpe > 0
        prob = normal_cdf((observed_sharpe - expected_max_sr) / sr_std)
        
        return prob
    
    def _expected_max_sharpe(
        self,
        n_trials: int,
        T: float
    ) -> float:
        """
        Expected maximum Sharpe ratio under null hypothesis.
        
        When testing N strategies with true SR=0, the expected
        maximum observed SR is approximately:
        E[max(SR)] = ppf(1 - 1/N) * sqrt(1/T)
        """
        if n_trials <= 1:
            return 0.0
        
        # Quantile corresponding to 1 - 1/N
        q = 1 - 1 / n_trials
        z = normal_ppf(min(q, 0.9999))
        
        return z * math.sqrt(1 / T)
    
    def _sharpe_std(
        self,
        sr: float,
        T: float,
        skewness: float,
        kurtosis: float
    ) -> float:
        """
        Standard error of Sharpe ratio with non-normality correction.
        
        Formula from Bailey & López de Prado (2014).
        """
        # Excess kurtosis
        excess_kurt = kurtosis - 3
        
        # Variance of SR estimator
        sr_var = (
            1 +
            0.5 * skewness * sr -
            (excess_kurt / 4) * (sr ** 2)
        ) / T
        
        return math.sqrt(max(sr_var, 0))
    
    def validate_strategy(
        self,
        observed_sharpe: float,
        n_trials: int,
        track_record_years: float,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
        confidence_threshold: float = 0.95
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate whether strategy passes DSR test.
        
        Args:
            observed_sharpe: Observed Sharpe
            n_trials: Number of strategies tested
            track_record_years: Backtest length
            skewness: Return skewness
            kurtosis: Return kurtosis
            confidence_threshold: Required probability (default 95%)
            
        Returns:
            (passed, details)
        """
        # Check minimum hurdle first
        if observed_sharpe < self.min_sharpe_hurdle:
            return False, {
                'reason': 'below_minimum_hurdle',
                'observed_sharpe': observed_sharpe,
                'min_hurdle': self.min_sharpe_hurdle,
                'dsr_probability': 0.0,
            }
        
        # Compute DSR probability
        prob = self.compute(
            observed_sharpe, n_trials, track_record_years,
            skewness, kurtosis
        )
        
        passed = prob >= confidence_threshold
        
        return passed, {
            'reason': 'passed' if passed else 'insufficient_confidence',
            'observed_sharpe': observed_sharpe,
            'dsr_probability': prob,
            'confidence_threshold': confidence_threshold,
            'n_trials': n_trials,
            'track_record_years': track_record_years,
        }
    
    def required_sharpe(
        self,
        n_trials: int,
        track_record_years: float,
        confidence: float = 0.95
    ) -> float:
        """
        Compute minimum Sharpe required to pass DSR test.
        
        Args:
            n_trials: Number of strategies to test
            track_record_years: Expected track record length
            confidence: Desired confidence level
            
        Returns:
            Minimum required observed Sharpe ratio
        """
        T = track_record_years * 252
        
        # Expected max under null
        expected_max = self._expected_max_sharpe(n_trials, T)
        
        # Required z-score for confidence
        z = normal_ppf(confidence)
        
        # Approximate std (assuming normal returns)
        std_approx = math.sqrt(1 / T)
        
        return expected_max + z * std_approx
    
    def __repr__(self) -> str:
        return f"DeflatedSharpeRatio(min_hurdle={self.min_sharpe_hurdle})"


class SharpeRatioCalculator:
    """
    Utility class for computing Sharpe ratios from returns.
    """
    
    @staticmethod
    def compute(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        annualization_factor: float = 252
    ) -> Dict[str, float]:
        """
        Compute Sharpe ratio and related statistics.
        
        Args:
            returns: Array of periodic returns
            risk_free_rate: Annualized risk-free rate
            annualization_factor: Periods per year
            
        Returns:
            Dict with sharpe, skewness, kurtosis, etc.
        """
        if len(returns) < 2:
            return {
                'sharpe': 0.0,
                'skewness': 0.0,
                'kurtosis': 3.0,
                'mean_return': 0.0,
                'volatility': 0.0,
                'n_observations': len(returns),
            }
        
        excess_returns = returns - risk_free_rate / annualization_factor
        
        mean_ret = np.mean(excess_returns)
        std_ret = np.std(excess_returns, ddof=1)
        
        if std_ret < 1e-10:
            sharpe = 0.0
        else:
            sharpe = (mean_ret / std_ret) * math.sqrt(annualization_factor)
        
        # Skewness
        n = len(returns)
        centered = returns - np.mean(returns)
        skewness = (np.sum(centered ** 3) / n) / (std_ret ** 3)
        
        # Kurtosis
        kurtosis = (np.sum(centered ** 4) / n) / (std_ret ** 4)
        
        return {
            'sharpe': sharpe,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'mean_return': mean_ret * annualization_factor,
            'volatility': std_ret * math.sqrt(annualization_factor),
            'n_observations': n,
            'track_record_years': n / annualization_factor,
        }
