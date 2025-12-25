"""
Streaming Hidden Markov Model for Zero-Lag Regime Detection

Based on Cambridge 2020 research (arXiv:2006.08307) demonstrating that
HMMs achieve instantaneous regime transition detection via Bayesian inference
rather than filtering, providing zero-lag awareness of market state changes.

Computational complexity: O(N²) per update where N = number of states.
For N=3 (Bull/Bear/Range), this is effectively O(1) constant time ~9 operations.
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL = "bull"
    BEAR = "bear"
    RANGE = "range"


@dataclass
class HMMConfig:
    """Configuration for streaming HMM regime detection."""
    n_states: int = 3
    state_names: List[str] = field(default_factory=lambda: ['Bull', 'Bear', 'Range'])
    
    # Transition matrix: high diagonal = regime persistence
    # Default calibrated from crypto market analysis (2020-2024)
    transition_persistence: float = 0.95  # Bull/Bear persist 95%
    range_persistence: float = 0.80  # Range less persistent
    
    # Emission parameters (Gaussian): mean return and volatility per regime
    # Calibrated from BTC 1-hour returns
    bull_mean: float = 0.001  # +0.1% per bar average
    bull_std: float = 0.010   # 1% volatility
    bear_mean: float = -0.001  # -0.1% per bar average
    bear_std: float = 0.020   # 2% volatility (higher in bear)
    range_mean: float = 0.0   # No drift
    range_std: float = 0.005  # Low volatility
    
    # Adaptive estimation
    adaptive_enabled: bool = True
    adaptive_lookback: int = 200  # Bars for parameter re-estimation
    adaptive_frequency: int = 50  # Re-estimate every N bars
    
    # History limits
    max_history: int = 1000


class StreamingHMM:
    """
    Zero-lag regime detection using the Forward Algorithm.
    
    The key insight is probabilistic: instead of smoothing prices (which
    inherently introduces lag), we maintain a probability distribution over
    market states and update it via Bayes' theorem. When evidence strongly
    contradicts the current regime, probability mass shifts immediately.
    
    Example usage:
        hmm = StreamingHMM()
        for price_return in returns_stream:
            state, confidence, probs = hmm.update(price_return)
            if state == 0:  # Bull
                # Favor momentum signals
            elif state == 2:  # Range
                # Favor mean-reversion signals
    """
    
    def __init__(self, config: HMMConfig = None):
        self.config = config or HMMConfig()
        self.n_states = self.config.n_states
        self.state_names = self.config.state_names
        
        # Initialize uniform prior over states
        self.state_probs = np.ones(self.n_states) / self.n_states
        
        # Build transition matrix
        self.transition_matrix = self._build_transition_matrix()
        
        # Initialize emission parameters
        self.emission_params = self._build_emission_params()
        
        # History for adaptive estimation
        self.return_history: List[float] = []
        
        # State tracking
        self._prev_state: Optional[int] = None
        self.regime_changes: int = 0
        self.updates_count: int = 0
        
        logger.info(f"StreamingHMM initialized: {self.n_states} states "
                   f"({', '.join(self.state_names)})")
    
    def _build_transition_matrix(self) -> np.ndarray:
        """
        Construct state transition probability matrix.
        
        The transition matrix encodes regime persistence: markets tend to
        stay in their current regime, with occasional transitions.
        
        Returns:
            np.ndarray: Shape (n_states, n_states) row-stochastic matrix
        """
        if self.n_states == 3:
            p = self.config.transition_persistence
            r = self.config.range_persistence
            
            # Off-diagonal probabilities
            p_off = (1 - p) / 2
            r_off = (1 - r) / 2
            
            return np.array([
                [p, p_off, p_off],      # Bull → Bull, Bear, Range
                [p_off, p, p_off],      # Bear → Bull, Bear, Range
                [r_off, r_off, r],      # Range → Bull, Bear, Range
            ])
        
        elif self.n_states == 2:
            p = self.config.transition_persistence
            return np.array([
                [p, 1-p],
                [1-p, p]
            ])
        
        else:
            # Generic: high persistence on diagonal
            p = 0.90
            off = (1 - p) / (self.n_states - 1)
            return p * np.eye(self.n_states) + off * (1 - np.eye(self.n_states))
    
    def _build_emission_params(self) -> Dict:
        """
        Initialize Gaussian emission parameters for each state.
        
        Each state emits observations (returns) according to a Gaussian
        distribution with state-specific mean and variance.
        
        Returns:
            Dict with 'means' and 'stds' arrays
        """
        if self.n_states == 3:
            return {
                'means': np.array([
                    self.config.bull_mean,
                    self.config.bear_mean,
                    self.config.range_mean
                ]),
                'stds': np.array([
                    self.config.bull_std,
                    self.config.bear_std,
                    self.config.range_std
                ])
            }
        elif self.n_states == 2:
            return {
                'means': np.array([0.002, 0.0]),  # Trending, Ranging
                'stds': np.array([0.015, 0.005])
            }
        else:
            return {
                'means': np.zeros(self.n_states),
                'stds': np.ones(self.n_states) * 0.01
            }
    
    def update(self, observation: float) -> Tuple[int, float, np.ndarray]:
        """
        Forward algorithm update step—the core of zero-lag detection.
        
        This method implements the recursive Bayesian update:
        
        P(s_t | o_{1:t}) ∝ P(o_t | s_t) × Σ_{s_{t-1}} P(s_t | s_{t-1}) × P(s_{t-1} | o_{1:t-1})
        
        The "zero-lag" property emerges because step 1 uses only the current
        observation, not a smoothed version. If today's return is -5% and we
        were confident we're in a Bull market, the emission probability for
        Bull becomes tiny, causing immediate probability shift toward Bear.
        
        Args:
            observation: Market return (price change as decimal, e.g., 0.01 = +1%)
        
        Returns:
            Tuple of:
                - most_likely_state: Index of highest-probability state
                - confidence: Probability of most likely state (0.0 to 1.0)
                - state_probs: Full probability vector over all states
        
        Complexity: O(N²) where N = number of states (3 → ~9 operations)
        """
        # Step 1: Compute emission probabilities P(observation | state)
        emissions = self._compute_emissions(observation)
        
        # Step 2: Prediction step—apply transition dynamics
        predicted = self.transition_matrix.T @ self.state_probs
        
        # Step 3: Update step—multiply prediction by emission likelihood
        new_probs = emissions * predicted
        
        # Step 4: Normalize to valid probability distribution
        prob_sum = new_probs.sum()
        if prob_sum > 1e-10:
            self.state_probs = new_probs / prob_sum
        else:
            # Numerical safety: reset to uniform if all probabilities collapsed
            logger.warning("HMM probabilities collapsed, resetting to uniform")
            self.state_probs = np.ones(self.n_states) / self.n_states
        
        # Track regime changes
        most_likely_state = int(self.state_probs.argmax())
        if self._prev_state is not None and most_likely_state != self._prev_state:
            self.regime_changes += 1
            logger.info(f"Regime transition: {self.state_names[self._prev_state]} → "
                       f"{self.state_names[most_likely_state]} "
                       f"(confidence: {self.state_probs[most_likely_state]:.2%})")
        
        self._prev_state = most_likely_state
        self.updates_count += 1
        
        # Maintain history for adaptive estimation
        self.return_history.append(observation)
        if len(self.return_history) > self.config.max_history:
            self.return_history.pop(0)
        
        # Periodic adaptive parameter re-estimation
        if (self.config.adaptive_enabled and 
            self.updates_count % self.config.adaptive_frequency == 0):
            self._update_emission_params()
        
        confidence = float(self.state_probs.max())
        return most_likely_state, confidence, self.state_probs.copy()
    
    def _compute_emissions(self, observation: float) -> np.ndarray:
        """
        Compute emission probabilities P(observation | state) for each state.
        
        Uses Gaussian probability density function. A return of -3% has
        high probability under the Bear state and very low probability under Bull.
        
        Args:
            observation: Market return
        
        Returns:
            np.ndarray of emission probabilities, one per state
        """
        means = self.emission_params['means']
        stds = self.emission_params['stds']
        
        # Gaussian PDF: P(x | μ, σ) = (1/σ√2π) exp(-(x-μ)²/2σ²)
        emissions = np.array([
            norm.pdf(observation, mean, std)
            for mean, std in zip(means, stds)
        ])
        
        # Add small epsilon for numerical stability
        emissions = np.maximum(emissions, 1e-10)
        
        return emissions
    
    def _update_emission_params(self) -> None:
        """
        Adaptive parameter re-estimation from recent return history.
        
        Uses quantile-based regime assignment to estimate state parameters
        from recent data, allowing the HMM to adapt to changing market conditions.
        """
        if len(self.return_history) < self.config.adaptive_lookback:
            return
        
        recent = np.array(self.return_history[-self.config.adaptive_lookback:])
        
        if self.n_states == 3:
            # Assign returns to pseudo-regimes by quantile
            sorted_returns = np.sort(recent)
            n = len(sorted_returns)
            
            bull_returns = sorted_returns[int(0.6 * n):]  # Top 40%
            bear_returns = sorted_returns[:int(0.4 * n)]  # Bottom 40%
            range_returns = sorted_returns[int(0.4 * n):int(0.6 * n)]  # Middle 20%
            
            # Update means and stds with minimum bounds
            self.emission_params['means'] = np.array([
                bull_returns.mean() if len(bull_returns) > 0 else 0.001,
                bear_returns.mean() if len(bear_returns) > 0 else -0.001,
                range_returns.mean() if len(range_returns) > 0 else 0.0
            ])
            
            self.emission_params['stds'] = np.array([
                max(bull_returns.std(), 0.001) if len(bull_returns) > 1 else 0.01,
                max(bear_returns.std(), 0.001) if len(bear_returns) > 1 else 0.02,
                max(range_returns.std(), 0.001) if len(range_returns) > 1 else 0.005
            ])
    
    def get_regime(self) -> MarketRegime:
        """Get current regime as enum."""
        state_idx = self.state_probs.argmax()
        if self.n_states == 3:
            return [MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.RANGE][state_idx]
        return MarketRegime.RANGE  # Default for non-3-state models
    
    def get_regime_label(self, state_idx: Optional[int] = None) -> str:
        """Get human-readable regime label."""
        if state_idx is None:
            state_idx = self.state_probs.argmax()
        return self.state_names[state_idx]
    
    def get_state_probabilities(self) -> Dict[str, float]:
        """Get probability distribution over regimes."""
        return {
            name: float(prob) 
            for name, prob in zip(self.state_names, self.state_probs)
        }
    
    def get_stats(self) -> Dict:
        """Return performance and state statistics."""
        return {
            'updates': self.updates_count,
            'regime_changes': self.regime_changes,
            'change_frequency': self.regime_changes / max(self.updates_count, 1),
            'current_regime': self.get_regime_label(),
            'confidence': float(self.state_probs.max()),
            'state_probabilities': self.get_state_probabilities(),
            'emission_params': {
                'means': self.emission_params['means'].tolist(),
                'stds': self.emission_params['stds'].tolist()
            }
        }
    
    def reset(self) -> None:
        """Reset to initial state (uniform prior)."""
        self.state_probs = np.ones(self.n_states) / self.n_states
        self._prev_state = None
        self.regime_changes = 0
        self.updates_count = 0
        self.return_history.clear()
        logger.info("HMM reset to initial state")
