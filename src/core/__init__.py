"""
Core Data Structures for HIMARI Layer 1 Explorer

- features.py: 60-dimensional feature vector with type safety
- grammar.py: AlphaCFG context-free grammar for strategy validation
- genome.py: Strategy genome encoding and genetic operators
"""

from .features import FeatureType, FeatureSpec, FeatureVector, FEATURE_SCHEMA
from .grammar import GrammarValidator, VALID_COMPARISONS
from .genome import SignalType, Condition, DecisionNode, StrategyGenome

__all__ = [
    'FeatureType', 'FeatureSpec', 'FeatureVector', 'FEATURE_SCHEMA',
    'GrammarValidator', 'VALID_COMPARISONS',
    'SignalType', 'Condition', 'DecisionNode', 'StrategyGenome'
]
