"""
Multi-Asset Sentiment Manager - Scalable Model Management

Manages sentiment models across multiple assets with model pooling
and transfer learning support.

Enhancement 7 from ANTIGRAVITY_SENTIMENT_ENHANCEMENT_GUIDE.md
"""

import logging
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelVariant(Enum):
    """Model variant types for transfer learning."""
    BTC_BASE = "btc_base"           # Use BTC model directly
    LIGHT_FINETUNED = "light_finetuned"  # BTC model + light fine-tuning
    DEDICATED = "dedicated"          # Separate dedicated model


@dataclass
class AssetConfig:
    """Configuration for a specific asset."""
    symbol: str
    model_variant: ModelVariant = ModelVariant.BTC_BASE
    sentiment_sources: List[str] = field(default_factory=lambda: ['news'])
    model_path: Optional[str] = None
    
    # Asset-specific lag horizons (optional override)
    lag_horizons: Optional[Dict[str, List[int]]] = None
    
    # Expected correlation with BTC (for monitoring)
    expected_btc_correlation: float = 0.9


# Pre-configured asset configs based on research
DEFAULT_ASSET_CONFIGS = {
    'BTCUSDT': AssetConfig(
        symbol='BTCUSDT',
        model_variant=ModelVariant.BTC_BASE,
        sentiment_sources=['news', 'twitter', 'reddit'],
        expected_btc_correlation=1.0
    ),
    'ETHUSDT': AssetConfig(
        symbol='ETHUSDT',
        model_variant=ModelVariant.BTC_BASE,  # High correlation (0.94)
        sentiment_sources=['news', 'twitter'],
        expected_btc_correlation=0.94
    ),
    'SOLUSDT': AssetConfig(
        symbol='SOLUSDT',
        model_variant=ModelVariant.LIGHT_FINETUNED,  # Medium correlation (0.82)
        sentiment_sources=['news', 'twitter'],
        expected_btc_correlation=0.82
    ),
    'BNBUSDT': AssetConfig(
        symbol='BNBUSDT',
        model_variant=ModelVariant.BTC_BASE,  # Exchange token, high correlation
        sentiment_sources=['news', 'twitter'],
        expected_btc_correlation=0.88
    ),
    'ADAUSDT': AssetConfig(
        symbol='ADAUSDT',
        model_variant=ModelVariant.LIGHT_FINETUNED,
        sentiment_sources=['news', 'twitter'],
        expected_btc_correlation=0.78
    ),
}


class MultiAssetSentimentManager:
    """
    Manage sentiment models across multiple assets efficiently.
    
    Features:
    - Model pooling (share BTC model across correlated assets)
    - Transfer learning support (light fine-tuning)
    - Memory budget enforcement
    - Correlation monitoring
    
    Example:
        manager = MultiAssetSentimentManager()
        
        # Get analyzer for asset
        analyzer = manager.get_analyzer('ETHUSDT')
        result = analyzer.analyze('ETH breaking out!')
        
        # Check memory usage
        print(manager.get_memory_usage())
    """
    
    def __init__(
        self,
        asset_configs: Optional[Dict[str, AssetConfig]] = None,
        max_concurrent_models: int = 3,
        btc_model_path: str = "./models/finbert-crypto-finetuned"
    ):
        """
        Initialize multi-asset sentiment manager.
        
        Args:
            asset_configs: Per-asset configurations
            max_concurrent_models: Maximum model instances in memory
            btc_model_path: Path to BTC fine-tuned model
        """
        self.asset_configs = asset_configs or DEFAULT_ASSET_CONFIGS
        self.max_concurrent_models = max_concurrent_models
        self.btc_model_path = btc_model_path
        
        # Model instances
        self._btc_base_analyzer = None  # Shared across BTC_BASE variants
        self._light_models: Dict[str, Any] = {}  # Per-asset light fine-tuned
        self._dedicated_models: Dict[str, Any] = {}  # Per-asset dedicated
        
        # Usage tracking for LRU eviction
        self._model_usage: Dict[str, int] = {}
        self._total_usage: int = 0
        
        logger.info(f"MultiAssetSentimentManager initialized with {len(self.asset_configs)} assets")
    
    def get_analyzer(self, symbol: str):
        """
        Get sentiment analyzer for an asset.
        
        Uses model pooling - correlated assets share the BTC base model.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            HybridSentimentAnalyzer instance (or None if unavailable)
        """
        config = self.asset_configs.get(symbol)
        
        if not config:
            logger.warning(f"No config for {symbol}, using BTC base model")
            return self._get_btc_base_analyzer()
        
        # Update usage
        self._total_usage += 1
        self._model_usage[symbol] = self._total_usage
        
        if config.model_variant == ModelVariant.BTC_BASE:
            return self._get_btc_base_analyzer()
        
        elif config.model_variant == ModelVariant.LIGHT_FINETUNED:
            return self._get_light_finetuned(symbol, config)
        
        else:  # DEDICATED
            return self._get_dedicated(symbol, config)
    
    def _get_btc_base_analyzer(self):
        """Get or create BTC base analyzer (shared)."""
        if self._btc_base_analyzer is not None:
            return self._btc_base_analyzer
        
        try:
            from .hybrid_sentiment import HybridSentimentAnalyzer, HybridSentimentConfig
            
            # Check for fine-tuned model
            use_fine_tuned = Path(self.btc_model_path).exists()
            
            config = HybridSentimentConfig(
                enable_lag_features=True,
                enable_dynamic_weighting=True
            )
            
            if use_fine_tuned:
                # TODO: Load fine-tuned model once supported
                logger.info("Loading BTC fine-tuned model (if available)")
            
            self._btc_base_analyzer = HybridSentimentAnalyzer(config)
            logger.info("BTC base analyzer initialized")
            
            return self._btc_base_analyzer
            
        except Exception as e:
            logger.error(f"Failed to initialize BTC base analyzer: {e}")
            return None
    
    def _get_light_finetuned(self, symbol: str, config: AssetConfig):
        """Get or create light fine-tuned analyzer for asset."""
        if symbol in self._light_models:
            return self._light_models[symbol]
        
        # Check memory budget
        self._enforce_memory_budget()
        
        # For now, fall back to BTC base (light fine-tuning requires training)
        logger.info(f"Light fine-tuned model for {symbol} not available, using BTC base")
        return self._get_btc_base_analyzer()
    
    def _get_dedicated(self, symbol: str, config: AssetConfig):
        """Get or create dedicated analyzer for asset."""
        if symbol in self._dedicated_models:
            return self._dedicated_models[symbol]
        
        # Check memory budget
        self._enforce_memory_budget()
        
        # For now, fall back to BTC base
        logger.info(f"Dedicated model for {symbol} not available, using BTC base")
        return self._get_btc_base_analyzer()
    
    def _enforce_memory_budget(self):
        """Evict least recently used models if over budget."""
        total_models = len(self._light_models) + len(self._dedicated_models)
        
        # BTC base is always loaded (doesn't count toward limit)
        if total_models >= self.max_concurrent_models:
            # Find LRU symbol
            all_symbols = list(self._light_models.keys()) + list(self._dedicated_models.keys())
            if not all_symbols:
                return
            
            lru_symbol = min(all_symbols, key=lambda s: self._model_usage.get(s, 0))
            
            # Evict
            if lru_symbol in self._light_models:
                del self._light_models[lru_symbol]
                logger.info(f"Evicted light model: {lru_symbol}")
            elif lru_symbol in self._dedicated_models:
                del self._dedicated_models[lru_symbol]
                logger.info(f"Evicted dedicated model: {lru_symbol}")
    
    def get_asset_config(self, symbol: str) -> Optional[AssetConfig]:
        """Get configuration for an asset."""
        return self.asset_configs.get(symbol)
    
    def register_asset(self, config: AssetConfig) -> None:
        """Register a new asset configuration."""
        self.asset_configs[config.symbol] = config
        logger.info(f"Registered asset: {config.symbol} ({config.model_variant.value})")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            'btc_base_loaded': self._btc_base_analyzer is not None,
            'light_models_count': len(self._light_models),
            'dedicated_models_count': len(self._dedicated_models),
            'total_models': 1 + len(self._light_models) + len(self._dedicated_models),
            'max_concurrent': self.max_concurrent_models,
            'light_model_symbols': list(self._light_models.keys()),
            'dedicated_model_symbols': list(self._dedicated_models.keys())
        }
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of configured symbols."""
        return list(self.asset_configs.keys())
    
    def analyze(self, symbol: str, text: str, **kwargs) -> Dict[str, float]:
        """
        Analyze sentiment for an asset.
        
        Convenience method that gets the appropriate analyzer and runs analysis.
        
        Args:
            symbol: Trading symbol
            text: Text to analyze
            **kwargs: Additional args for analyze()
            
        Returns:
            Sentiment analysis result
        """
        analyzer = self.get_analyzer(symbol)
        
        if analyzer is None:
            return {'score': 0.0, 'label': 'neutral', 'error': 'no_analyzer'}
        
        return analyzer.analyze(text, symbol=symbol, **kwargs)
