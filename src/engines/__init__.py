"""
Generation Engines for HIMARI Layer 1 Explorer

Four complementary engines with different exploration-exploitation profiles:
- Engine 1: Evolutionary Search (40% budget)
- Engine 2: Flow Matching Generation (25% budget)
- Engine 3: LLM-Guided Generation (20% budget)
- Engine 4: External Idea Harvesting (15% budget)

Plus gap enhancements:
- Engine 5: MCTS Strategy Search
- PSRO Diversity Manager
"""

from .evolutionary import EvolutionaryExplorer, NeutralDriftManager
from .flow_matching import ConditionalFlowMatching, GenerationCondition
from .llm_guided import LLMGuidedGenerator, LLMClient
from .harvester import ExternalIdeaHarvester
from .orchestrator import EngineOrchestrator, GenerationBudget

# Gap enhancements
from .mcts import MCTSStrategyGenerator, MCTSConfig, MCTSNode, MCTSResult
from .psro import PSRODiversityManager, PSROConfig, PSROResult, StrategyProfile

__all__ = [
    # Core engines
    'EvolutionaryExplorer', 'NeutralDriftManager',
    'ConditionalFlowMatching', 'GenerationCondition',
    'LLMGuidedGenerator', 'LLMClient',
    'ExternalIdeaHarvester',
    'EngineOrchestrator', 'GenerationBudget',

    # Gap enhancements
    'MCTSStrategyGenerator', 'MCTSConfig', 'MCTSNode', 'MCTSResult',
    'PSRODiversityManager', 'PSROConfig', 'PSROResult', 'StrategyProfile'
]
