"""
Configuration Management for HIMARI Layer 1 Explorer

Handles:
- YAML configuration loading
- Environment variable substitution
- Configuration validation
- Runtime configuration access
"""

import os
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import re

logger = logging.getLogger(__name__)


# Try to import yaml, fall back to simple parsing
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not installed, using simple config loading")


@dataclass
class GenerationConfig:
    """Configuration for strategy generation."""
    # Budget allocation
    evolutionary_pct: float = 0.40
    generative_pct: float = 0.25
    llm_guided_pct: float = 0.20
    external_pct: float = 0.15

    # Generation parameters
    total_candidates_per_cycle: int = 100
    max_llm_calls_per_cycle: int = 20
    diversity_threshold: float = 0.85

    # Evolutionary parameters
    population_size: int = 100
    elite_count: int = 10
    tournament_size: int = 5
    mutation_rate: float = 0.15
    crossover_rate: float = 0.7

    # Flow matching parameters
    flow_steps: int = 15
    cfg_scale: float = 7.5


@dataclass
class ValidationConfig:
    """Configuration for HIFA validation pipeline."""
    # Stage thresholds
    min_sharpe: float = 0.5
    max_drawdown: float = 0.20
    min_profit_factor: float = 1.1
    min_trade_count: int = 50

    # DSR parameters
    dsr_multiple_testing_penalty: float = 0.1
    min_dsr: float = 1.5

    # True contribution thresholds
    min_marginal_sharpe: float = 0.2
    min_orthogonality: float = 0.3
    min_residual_ic: float = 0.01

    # Neutralization thresholds
    min_ic_retention: float = 0.50

    # Batch processing
    batch_size: int = 50
    stage2_top_k: int = 30
    stage3_top_k: int = 10


@dataclass
class DeploymentConfig:
    """Configuration for deployment pipeline."""
    # Shadow trading
    min_shadow_days: int = 21
    min_transfer_ratio: float = 0.70
    max_epistemic_uncertainty: float = 0.10

    # Position sizing
    base_position_pct: float = 0.05
    max_position_pct: float = 0.20
    scale_up_threshold: float = 0.75
    scale_down_threshold: float = 0.60

    # Monitoring
    daily_tr_check: bool = True
    weekly_regime_check: bool = True
    monthly_full_review: bool = True


@dataclass
class AdaptationConfig:
    """Configuration for adaptation systems."""
    # Drift detection
    adwin_delta: float = 0.002
    page_hinkley_delta: float = 0.005
    page_hinkley_threshold: float = 50.0
    kswin_window_size: int = 100
    kswin_stat_size: int = 30
    kswin_alpha: float = 0.005
    drift_vote_threshold: int = 2

    # Response levels
    yellow_threshold: float = 0.65
    orange_threshold: float = 0.55
    red_threshold: float = 0.45

    # MAML parameters
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    adaptation_steps: int = 5

    # Retirement
    min_tr_for_survival: float = 0.50
    max_consecutive_failures: int = 3
    winddown_days: int = 7


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    # Primary provider (Gemini)
    primary_provider: str = "gemini"
    primary_model: str = "gemini-1.5-pro"
    primary_api_key: Optional[str] = None

    # Local provider (Ollama)
    local_provider: str = "ollama"
    local_base_url: str = "http://localhost:11434"
    local_model: str = "deepseek-coder:33b"

    # Fallback provider (DeepSeek API)
    fallback_provider: str = "deepseek"
    fallback_base_url: str = "https://api.deepseek.com"
    fallback_model: str = "deepseek-chat"
    fallback_api_key: Optional[str] = None

    # Usage allocation
    strategy_generation_provider: str = "primary"
    knowledge_extraction_provider: str = "local"
    idea_harvesting_provider: str = "local"
    mutation_guidance_provider: str = "primary"

    # Rate limiting
    max_requests_per_minute: int = 30
    timeout_seconds: int = 60


@dataclass
class InfrastructureConfig:
    """Configuration for external infrastructure."""
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_group_id: str = "layer1-explorer"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Prometheus
    metrics_port: int = 8000
    metrics_enabled: bool = True


@dataclass
class CrawlerConfig:
    """Configuration for web crawler."""
    # Sources
    arxiv_enabled: bool = True
    arxiv_categories: List[str] = field(default_factory=lambda: ["q-fin.TR", "q-fin.PM", "q-fin.ST"])
    max_papers_per_crawl: int = 20

    # Rate limiting
    requests_per_second: float = 0.5
    max_retries: int = 3

    # Storage
    sqlite_path: str = "data/knowledge.db"
    cache_ttl_hours: int = 24

    # Extraction
    min_relevance_score: float = 0.3


@dataclass
class Layer1ExplorerConfig:
    """
    Complete configuration for Layer 1 Explorer.

    Aggregates all component configurations.
    """
    # Component configurations
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    infrastructure: InfrastructureConfig = field(default_factory=InfrastructureConfig)
    crawler: CrawlerConfig = field(default_factory=CrawlerConfig)

    # Global settings
    log_level: str = "INFO"
    cycle_interval_seconds: int = 3600  # 1 hour
    max_cycles_per_day: int = 24

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Layer1ExplorerConfig':
        """Create configuration from dictionary."""
        config = cls()

        if 'generation' in data:
            config.generation = GenerationConfig(**data['generation'])
        if 'validation' in data:
            config.validation = ValidationConfig(**data['validation'])
        if 'deployment' in data:
            config.deployment = DeploymentConfig(**data['deployment'])
        if 'adaptation' in data:
            config.adaptation = AdaptationConfig(**data['adaptation'])
        if 'llm' in data:
            config.llm = LLMConfig(**data['llm'])
        if 'infrastructure' in data:
            config.infrastructure = InfrastructureConfig(**data['infrastructure'])
        if 'crawler' in data:
            config.crawler = CrawlerConfig(**data['crawler'])

        # Global settings
        config.log_level = data.get('log_level', config.log_level)
        config.cycle_interval_seconds = data.get('cycle_interval_seconds', config.cycle_interval_seconds)
        config.max_cycles_per_day = data.get('max_cycles_per_day', config.max_cycles_per_day)

        return config


def _substitute_env_vars(value: str) -> str:
    """Substitute environment variables in string values."""
    if not isinstance(value, str):
        return value

    # Match ${VAR_NAME} or $VAR_NAME patterns
    pattern = r'\$\{([^}]+)\}|\$([A-Z_][A-Z0-9_]*)'

    def replace(match):
        var_name = match.group(1) or match.group(2)
        env_value = os.environ.get(var_name)
        if env_value is None:
            logger.warning(f"Environment variable {var_name} not set")
            return match.group(0)  # Keep original if not found
        return env_value

    return re.sub(pattern, replace, value)


def _process_dict(data: Dict) -> Dict:
    """Recursively process dictionary, substituting environment variables."""
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = _process_dict(value)
        elif isinstance(value, list):
            result[key] = [_substitute_env_vars(v) if isinstance(v, str) else v for v in value]
        elif isinstance(value, str):
            result[key] = _substitute_env_vars(value)
        else:
            result[key] = value
    return result


def load_config(config_path: str) -> Layer1ExplorerConfig:
    """
    Load configuration from YAML file.

    Supports environment variable substitution using ${VAR_NAME} syntax.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Layer1ExplorerConfig instance
    """
    path = Path(config_path)

    if not path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return Layer1ExplorerConfig()

    try:
        with open(path, 'r') as f:
            if YAML_AVAILABLE:
                data = yaml.safe_load(f)
            else:
                # Simple fallback - won't handle complex YAML
                import json
                # Try JSON first (valid YAML is often valid JSON for simple configs)
                try:
                    data = json.load(f)
                except:
                    logger.error("Cannot parse config without PyYAML")
                    return Layer1ExplorerConfig()

        if data is None:
            data = {}

        # Substitute environment variables
        data = _process_dict(data)

        return Layer1ExplorerConfig.from_dict(data)

    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        return Layer1ExplorerConfig()


def save_config(config: Layer1ExplorerConfig, config_path: str) -> bool:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration to save
        config_path: Path to save to

    Returns:
        True if successful
    """
    path = Path(config_path)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            if YAML_AVAILABLE:
                yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
            else:
                import json
                json.dump(config.to_dict(), f, indent=2)

        logger.info(f"Configuration saved to {config_path}")
        return True

    except Exception as e:
        logger.error(f"Error saving config to {config_path}: {e}")
        return False


# Global configuration instance
_config_instance: Optional[Layer1ExplorerConfig] = None


def get_config() -> Layer1ExplorerConfig:
    """
    Get global configuration instance.

    Returns default configuration if not initialized.
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = Layer1ExplorerConfig()

    return _config_instance


def init_config(config_path: Optional[str] = None) -> Layer1ExplorerConfig:
    """
    Initialize global configuration.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Initialized configuration
    """
    global _config_instance

    if config_path:
        _config_instance = load_config(config_path)
    else:
        # Try default paths
        default_paths = [
            "config/layer1.yaml",
            "layer1.yaml",
            "../config/layer1.yaml"
        ]

        for path in default_paths:
            if Path(path).exists():
                _config_instance = load_config(path)
                break
        else:
            _config_instance = Layer1ExplorerConfig()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, _config_instance.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    return _config_instance


def create_default_config_file(config_path: str = "config/layer1.yaml") -> bool:
    """
    Create a default configuration file with comments.

    Args:
        config_path: Path to create configuration file

    Returns:
        True if successful
    """
    default_yaml = '''# HIMARI Layer 1 Explorer Configuration

# Logging
log_level: INFO

# Generation cycle settings
cycle_interval_seconds: 3600  # 1 hour between cycles
max_cycles_per_day: 24

# Strategy Generation
generation:
  # Budget allocation (must sum to 1.0)
  evolutionary_pct: 0.40
  generative_pct: 0.25
  llm_guided_pct: 0.20
  external_pct: 0.15

  # Generation parameters
  total_candidates_per_cycle: 100
  max_llm_calls_per_cycle: 20
  diversity_threshold: 0.85

  # Evolutionary parameters
  population_size: 100
  elite_count: 10
  tournament_size: 5
  mutation_rate: 0.15
  crossover_rate: 0.7

  # Flow matching parameters
  flow_steps: 15
  cfg_scale: 7.5

# HIFA Validation Pipeline
validation:
  # Stage thresholds
  min_sharpe: 0.5
  max_drawdown: 0.20
  min_profit_factor: 1.1
  min_trade_count: 50

  # DSR parameters
  dsr_multiple_testing_penalty: 0.1
  min_dsr: 1.5

  # True contribution thresholds
  min_marginal_sharpe: 0.2
  min_orthogonality: 0.3
  min_residual_ic: 0.01

  # Neutralization thresholds
  min_ic_retention: 0.50

  # Batch processing
  batch_size: 50
  stage2_top_k: 30
  stage3_top_k: 10

# Deployment Pipeline
deployment:
  # Shadow trading
  min_shadow_days: 21
  min_transfer_ratio: 0.70
  max_epistemic_uncertainty: 0.10

  # Position sizing
  base_position_pct: 0.05
  max_position_pct: 0.20
  scale_up_threshold: 0.75
  scale_down_threshold: 0.60

  # Monitoring
  daily_tr_check: true
  weekly_regime_check: true
  monthly_full_review: true

# Adaptation Systems
adaptation:
  # Drift detection
  adwin_delta: 0.002
  page_hinkley_delta: 0.005
  page_hinkley_threshold: 50.0
  kswin_window_size: 100
  kswin_stat_size: 30
  kswin_alpha: 0.005
  drift_vote_threshold: 2

  # Response levels (transfer ratio thresholds)
  yellow_threshold: 0.65
  orange_threshold: 0.55
  red_threshold: 0.45

  # MAML parameters
  inner_lr: 0.01
  outer_lr: 0.001
  adaptation_steps: 5

  # Retirement
  min_tr_for_survival: 0.50
  max_consecutive_failures: 3
  winddown_days: 7

# LLM Configuration
llm:
  # Primary: Gemini
  primary_provider: gemini
  primary_model: gemini-1.5-pro
  primary_api_key: ${GEMINI_API_KEY}

  # Local: Ollama with DeepSeek
  local_provider: ollama
  local_base_url: http://localhost:11434
  local_model: deepseek-coder:33b

  # Fallback: DeepSeek API
  fallback_provider: deepseek
  fallback_base_url: https://api.deepseek.com
  fallback_model: deepseek-chat
  fallback_api_key: ${DEEPSEEK_API_KEY}

  # Usage allocation
  strategy_generation_provider: primary
  knowledge_extraction_provider: local
  idea_harvesting_provider: local
  mutation_guidance_provider: primary

  # Rate limiting
  max_requests_per_minute: 30
  timeout_seconds: 60

# Infrastructure
infrastructure:
  # Redis
  redis_host: localhost
  redis_port: 6379
  redis_db: 0
  redis_password: ${REDIS_PASSWORD}

  # Kafka
  kafka_bootstrap_servers: localhost:9092
  kafka_group_id: layer1-explorer

  # Neo4j
  neo4j_uri: bolt://localhost:7687
  neo4j_user: neo4j
  neo4j_password: ${NEO4J_PASSWORD}

  # Prometheus
  metrics_port: 8000
  metrics_enabled: true

# Web Crawler
crawler:
  # Sources
  arxiv_enabled: true
  arxiv_categories:
    - q-fin.TR
    - q-fin.PM
    - q-fin.ST
  max_papers_per_crawl: 20

  # Rate limiting
  requests_per_second: 0.5
  max_retries: 3

  # Storage
  sqlite_path: data/knowledge.db
  cache_ttl_hours: 24

  # Extraction
  min_relevance_score: 0.3
'''

    try:
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            f.write(default_yaml)

        logger.info(f"Created default configuration at {config_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating default config: {e}")
        return False
