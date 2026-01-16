"""
Infrastructure Integration for HIMARI Layer 1 Explorer

Connects Layer 1 to the broader HIMARI ecosystem:
- Redis: Feature store (Layer 0)
- Kafka: Event bus (regime signals, feedback)
- Neo4j: Knowledge graph (Layer 5/6)
- Prometheus: Metrics and monitoring
"""

from .data_interface import (
    Layer1DataInterface, Layer1Config as DataConfig,
    RedisFeatureStore, KafkaEventBus, Neo4jKnowledgeGraph,
    RegimeSignal, StrategyFeedback
)
from .metrics import ExplorerMetrics, get_metrics, init_metrics
from .config import (
    Layer1ExplorerConfig, GenerationConfig, ValidationConfig,
    DeploymentConfig, AdaptationConfig, LLMConfig,
    InfrastructureConfig, CrawlerConfig,
    load_config, save_config, get_config, init_config,
    create_default_config_file
)

# Re-export Layer1Config from data_interface as the primary config name
Layer1Config = DataConfig

__all__ = [
    # Data interface
    'Layer1DataInterface', 'Layer1Config', 'DataConfig',
    'RedisFeatureStore', 'KafkaEventBus', 'Neo4jKnowledgeGraph',
    'RegimeSignal', 'StrategyFeedback',

    # Metrics
    'ExplorerMetrics', 'get_metrics', 'init_metrics',

    # Configuration
    'Layer1ExplorerConfig', 'GenerationConfig', 'ValidationConfig',
    'DeploymentConfig', 'AdaptationConfig', 'LLMConfig',
    'InfrastructureConfig', 'CrawlerConfig',
    'load_config', 'save_config', 'get_config', 'init_config',
    'create_default_config_file'
]
