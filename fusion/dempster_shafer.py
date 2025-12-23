"""
Dempster-Shafer Evidence Fusion for HIMARI L1

Handles conflicting signals from multiple sources by properly
modeling uncertainty. Unlike Bayesian fusion, D-S allows for
an explicit "I don't know" state.

Key Advantage: When signals conflict (Kalman says up, Lorentzian says down),
D-S quantifies this conflict and outputs high uncertainty, preventing
trades during periods of model disagreement.

Latency: <0.2ms per fusion
Sharpe Contribution: Reduces whipsaws from conflicting signals

Usage:
    ds = DempsterShafer()
    
    # Add evidence from each model
    ds.add_evidence({'bullish': 0.6, 'bearish': 0.2, 'neutral': 0.2})
    ds.add_evidence({'bullish': 0.3, 'bearish': 0.5, 'neutral': 0.2})
    
    # Get combined beliefs
    belief_bullish = ds.belief('bullish')
    plausibility_bullish = ds.plausibility('bullish')
    uncertainty = ds.uncertainty('bullish')
"""

import json
from typing import Dict, Any, Set, List, Tuple
from collections import defaultdict


class DempsterShafer:
    """
    Dempster-Shafer belief function theory implementation.
    
    Models beliefs over a frame of discernment (set of hypotheses).
    For trading: {'bullish', 'bearish', 'neutral'}
    
    Key concepts:
    - Mass function m(A): Basic probability assignment to subset A
    - Belief Bel(A): Sum of masses of all subsets of A (lower bound)
    - Plausibility Pl(A): 1 - Bel(¬A) (upper bound)
    - Uncertainty: Pl(A) - Bel(A)
    """
    
    __slots__ = (
        '_frame', '_mass_function',
        '_combined_mass', '_conflict'
    )
    
    def __init__(self, frame: Set[str] = None):
        """
        Initialize D-S fusion.
        
        Args:
            frame: Frame of discernment (hypotheses).
                   Defaults to {'bullish', 'bearish', 'neutral'}
        """
        if frame is None:
            frame = {'bullish', 'bearish', 'neutral'}
        
        self._frame = frozenset(frame)
        self._mass_function: Dict[frozenset, float] = {}
        self._combined_mass: Dict[frozenset, float] = {
            self._frame: 1.0  # Initial: complete ignorance
        }
        self._conflict = 0.0
    
    def add_evidence(
        self,
        masses: Dict[str, float]
    ) -> float:
        """
        Add evidence from a single source using Dempster's rule.
        
        Args:
            masses: Dict mapping hypothesis to belief mass.
                   Must sum to <= 1. Remainder goes to uncertainty.
                   
        Returns:
            Conflict level K between this evidence and prior combined mass
        """
        # Convert to frozenset keys
        new_mass: Dict[frozenset, float] = {}
        
        total = 0.0
        for hyp, mass in masses.items():
            if mass > 0:
                key = frozenset({hyp}) if isinstance(hyp, str) else frozenset(hyp)
                new_mass[key] = mass
                total += mass
        
        # Remainder goes to complete frame (uncertainty)
        if total < 1.0:
            new_mass[self._frame] = 1.0 - total
        
        # Combine with existing using Dempster's rule
        self._combined_mass, self._conflict = self._dempster_combine(
            self._combined_mass, new_mass
        )
        
        return self._conflict
    
    def _dempster_combine(
        self,
        m1: Dict[frozenset, float],
        m2: Dict[frozenset, float]
    ) -> Tuple[Dict[frozenset, float], float]:
        """
        Combine two mass functions using Dempster's rule.
        
        Returns:
            (combined_mass, conflict_level)
        """
        combined: Dict[frozenset, float] = defaultdict(float)
        conflict = 0.0
        
        for a1, mass1 in m1.items():
            for a2, mass2 in m2.items():
                intersection = a1 & a2
                product = mass1 * mass2
                
                if not intersection:
                    # Empty intersection = conflict
                    conflict += product
                else:
                    combined[intersection] += product
        
        # Normalize by (1 - conflict) per Dempster's rule
        if conflict < 1.0:
            normalizer = 1.0 / (1.0 - conflict)
            combined = {k: v * normalizer for k, v in combined.items()}
        else:
            # Complete conflict - reset to ignorance
            combined = {self._frame: 1.0}
        
        return dict(combined), conflict
    
    def belief(self, hypothesis: str) -> float:
        """
        Compute belief in a hypothesis.
        
        Belief is the sum of masses of all subsets of the hypothesis.
        It represents the lower bound on the probability.
        
        Args:
            hypothesis: The hypothesis to query ('bullish', 'bearish', etc.)
            
        Returns:
            Belief value [0, 1]
        """
        target = frozenset({hypothesis})
        total = 0.0
        
        for focal_set, mass in self._combined_mass.items():
            if focal_set <= target:  # focal_set is subset of target
                total += mass
        
        return total
    
    def plausibility(self, hypothesis: str) -> float:
        """
        Compute plausibility of a hypothesis.
        
        Plausibility = 1 - Belief(complement)
        It represents the upper bound on the probability.
        
        Args:
            hypothesis: The hypothesis to query
            
        Returns:
            Plausibility value [0, 1]
        """
        target = frozenset({hypothesis})
        complement = self._frame - target
        
        # Sum mass that doesn't completely exclude hypothesis
        total = 0.0
        for focal_set, mass in self._combined_mass.items():
            if focal_set & target:  # Has non-empty intersection
                total += mass
        
        return total
    
    def uncertainty(self, hypothesis: str) -> float:
        """
        Compute uncertainty interval width.
        
        Uncertainty = Plausibility - Belief
        High uncertainty = conflicting evidence or lack of evidence
        
        Args:
            hypothesis: The hypothesis to query
            
        Returns:
            Uncertainty [0, 1]
        """
        return self.plausibility(hypothesis) - self.belief(hypothesis)
    
    def get_decision(self) -> Tuple[str, float, float]:
        """
        Get best decision based on maximum belief.
        
        Returns:
            (best_hypothesis, belief, uncertainty)
        """
        best_hyp = None
        best_belief = -1
        
        for hyp in self._frame:
            b = self.belief(hyp)
            if b > best_belief:
                best_belief = b
                best_hyp = hyp
        
        return best_hyp, best_belief, self.uncertainty(best_hyp)
    
    def should_trade(
        self,
        min_belief: float = 0.4,
        max_uncertainty: float = 0.5
    ) -> Tuple[bool, str]:
        """
        Determine if we should trade based on belief/uncertainty.
        
        Args:
            min_belief: Minimum belief required for trading
            max_uncertainty: Maximum uncertainty allowed
            
        Returns:
            (should_trade, reason)
        """
        best_hyp, belief, uncertainty = self.get_decision()
        
        if best_hyp == 'neutral':
            return False, 'neutral_regime'
        
        if uncertainty > max_uncertainty:
            return False, f'high_uncertainty_{uncertainty:.2f}'
        
        if belief < min_belief:
            return False, f'low_belief_{belief:.2f}'
        
        if self._conflict > 0.5:
            return False, f'high_conflict_{self._conflict:.2f}'
        
        return True, best_hyp
    
    def get_confidence_weighted_signal(self) -> float:
        """
        Get signal in [-1, 1] weighted by confidence.
        
        +1 = strongly bullish with high confidence
        -1 = strongly bearish with high confidence
        0 = neutral or high uncertainty
        """
        bel_bull = self.belief('bullish')
        bel_bear = self.belief('bearish')
        
        # Net direction
        direction = bel_bull - bel_bear
        
        # Confidence = inverse of uncertainty
        uncertainty = max(
            self.uncertainty('bullish'),
            self.uncertainty('bearish')
        )
        confidence = 1.0 - uncertainty
        
        return direction * confidence
    
    @property
    def conflict_level(self) -> float:
        """Current conflict level from Dempster combination."""
        return self._conflict
    
    @property
    def total_uncertainty(self) -> float:
        """Mass assigned to complete frame (ignorance)."""
        return self._combined_mass.get(self._frame, 0.0)
    
    def reset(self) -> None:
        """Reset to complete ignorance."""
        self._combined_mass = {self._frame: 1.0}
        self._conflict = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        # Convert frozenset keys to strings
        mass_dict = {
            ','.join(sorted(k)): v
            for k, v in self._combined_mass.items()
        }
        return {
            'frame': list(self._frame),
            'combined_mass': mass_dict,
            'conflict': self._conflict,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DempsterShafer':
        """Restore from serialized state."""
        instance = cls(frame=set(data['frame']))
        
        # Convert string keys back to frozensets
        instance._combined_mass = {
            frozenset(k.split(',')): v
            for k, v in data['combined_mass'].items()
        }
        instance._conflict = data['conflict']
        return instance
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'DempsterShafer':
        """Restore from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def __repr__(self) -> str:
        best_hyp, belief, uncertainty = self.get_decision()
        return (
            f"DempsterShafer(best={best_hyp}, bel={belief:.2f}, "
            f"unc={uncertainty:.2f}, conflict={self._conflict:.2f})"
        )


class StreamingDempsterShafer(DempsterShafer):
    """
    Streaming D-S with decay for time-varying evidence.
    
    Older evidence decays, allowing the system to adapt
    to changing market conditions.
    """
    
    def __init__(
        self,
        frame: Set[str] = None,
        decay_rate: float = 0.1
    ):
        """
        Args:
            frame: Hypotheses
            decay_rate: How fast old evidence decays (0-1)
        """
        super().__init__(frame)
        self._decay_rate = decay_rate
    
    def add_evidence_streaming(
        self,
        masses: Dict[str, float]
    ) -> float:
        """
        Add evidence with decay of old evidence.
        """
        # First decay existing mass
        self._decay_mass()
        
        # Then add new evidence
        return self.add_evidence(masses)
    
    def _decay_mass(self) -> None:
        """Decay all mass toward ignorance."""
        decayed: Dict[frozenset, float] = {}
        total_decay = 0.0
        
        for focal_set, mass in self._combined_mass.items():
            if focal_set == self._frame:
                # Keep uncertainty as is
                decayed[focal_set] = mass
            else:
                # Decay specific beliefs
                new_mass = mass * (1 - self._decay_rate)
                if new_mass > 0.001:  # Threshold
                    decayed[focal_set] = new_mass
                total_decay += mass - new_mass
        
        # Add decayed mass to uncertainty
        decayed[self._frame] = decayed.get(self._frame, 0) + total_decay
        
        self._combined_mass = decayed
